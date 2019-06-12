/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn_graph.h"
#include "quantize.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_inlines.h"
#include "hvx_funnel_reduce.h"

static inline HVX_Vector __attribute__((always_inline))
find_minmax_in_rows_outfunc( HVX_Vector vin, void const * unused)
{
	return Q6_V_vand_VV( vin, Q6_V_vsplat_R(0xFFFFFF));
}

//
// given an array of 'rows' by 'cols' uint8;
// - find the col index of the largest # (or smallest) in each row. If the max (or min) occurs more than once,
//   the result is the smallest index of all these.
// - must have rows >= 1, cols >=1, cols <= 2^24
// - the results are stored at outp[0..rows-1]
// - no vector-alignment constraints. The 'row stride' is a separate parameter.
//
// strategy is:
//    - convert each byte x[i] to  ((x[i]^127)<<24) | i    as an i32 value
//    - find the min of all these in each row
//    - lower 24 bits of that are the result.
//  'extra' lanes in the last vector (when cols%128!=0) are forced to 0 on reading, and combined with their
//   out-of-range column index; this prevents them  from surviving the min reduction.
//
//

void
hvx_argmin_or_max_in_rows( uint8_t const * data, int rows, int cols, int row_stride, int32_t * outp, int find_argmax )
{
	if(0){		// scalar reference
		uint8_t inv = find_argmax? 0:0xFF;
		uint8_t const * data0 = data;
		for(int i = 0; i < rows; i++){
			uint8_t best = data[0]^inv;
			int besti= 0;
			for(int j = 1; j < cols; j++){
				int v = data[j]^inv;
				if( v > best){  besti =j; best =v; }
			}
			outp[i] = besti;
			data += row_stride;
		}
		data = data0;
		//for( int i = 0; i < rows; i++) printf("  %d", (int)outp[i]); printf("\n");
		return;
	}

	HVX_VectorPred lastmask = Q6_Q_vsetq_R( cols);
	int vecs_across = (cols+127)/128u;
	// if 'cols' is not a multiple of 128, this will be vecs_across-1, and it triggers right-masking
	// if cols is a multiple of 128, it will be the same as vecs_across (no masking needed)
	int ilast = cols/128u;
	// if we are finding min instead of max, invert all the  values as we read them.
	HVX_Vector invert_input = Q6_V_vsplat_R( find_argmax?0:-1);

	HVX_Vector indices = *(HVX_Vector const*)const_Count128;		// bytes { 0, ... 127 }
	indices = Q6_V_vand_VV( indices,Q6_V_vsplat_R(0xFF));			//  { 0, 4, 8, .. 124} in w lanes.

    // Funnel reduction: reduce using vmax_Vw, in 8 byte units (even lane = even row, odd lanes = odd row)
	// .. and all outputs to be 'anded' with 0xFFFFFF before being stored.
	struct funnelreduce_state frstate;
	funnelreduce_init( &frstate, outp, 8, funnelreduce_min_Vw, find_minmax_in_rows_outfunc , NULL);

	for( int rowno = 0; rowno < rows; rowno+=2){
		int only_one = rowno==(rows-1);
		//printf("row %d; only_one = %d cols = %d vecs_across = %d ilast = %d\n", rowno, only_one, cols,vecs_across , ilast);
		HVX_Vector const *inrow0 = (HVX_Vector const *)(data + rowno*row_stride);
		HVX_Vector const *inrow1 = (HVX_Vector const *)( (uint8_t const*)inrow0 + (only_one? 0: row_stride));
		// prepare for a pass across
		HVX_Vector allmin = Q6_V_vsplat_R( 0x7FFFFFFF);
		HVX_Vector curr_index = indices;
		for( int i =0; i < vecs_across; i++){
			HVX_Vector vin0 = q6op_V_vldu_A( inrow0+i);
			HVX_Vector vin1 = q6op_V_vldu_A( inrow1+i);
			vin0 = Q6_V_vxor_VV(vin0, invert_input);
			vin1 = Q6_V_vxor_VV(vin1, invert_input);
#if __HEXAGON_ARCH__ >= 62
			// eval mask on each loop, avoid the conditional. *big* improvement in the inner loop
			lastmask = Q6_Q_vsetq2_R(min_i32(128,cols-i*128));
			ilast = i;
#endif
			if( i == ilast){	// blank out the extra bytes
				vin0 = q6op_V_vand_QV( lastmask, vin0);
				vin1 = q6op_V_vand_QV( lastmask, vin1);
			}

			// before we do anything else, interleave 2x2 and decide which lane wins the odd/even
			HVX_VectorPair vshuf = Q6_Wb_vshuffoe_VbVb( vin1, vin0);
			HVX_Vector vmax = Q6_Vub_vmax_VubVub( Q6_V_lo_W(vshuf), Q6_V_hi_W(vshuf));
			// the 'odd' lane wins iff lane 1 > lane 0, 3>2, 5>4 etc
			HVX_VectorPred oddcol = Q6_Q_vcmp_gt_VubVub(Q6_V_hi_W(vshuf), Q6_V_lo_W(vshuf) );
			// at this point, even b lanes belong to problem 0 and odd to problem 1.
			// do it again, but with groups of 2
			HVX_Vector vmax1 = Q6_Vh_vshuffo_VhVh( vmax,vmax);	// bring lanes 2 to 0, 3 to 1 etc
			HVX_VectorPred odd2col = Q6_Q_vcmp_gt_VubVub(vmax1, vmax );
			vmax = Q6_Vub_vmax_VubVub( vmax, vmax1);
			// ok, now:
			// vmax lanes 0,4, .. 124 are the best value in each group of 4 lanes for problem 0
			//            1,5 ..  125 are for problem 1
			// the 'odd2col' value in those lanes tells us whether the byte came from first or 2nd group of 2.
			//  but oddcol is trickier; it has bit 0 of the lane index, but we need to select odd or even
			// lanes based on odd2col.
			HVX_Vector lane_ind = Q6_V_vand_QR( oddcol, 0x01010101);  // get odd/even lane select
			HVX_Vector lane_ind2 = Q6_V_vand_QR( odd2col, 0x0202); // bit 1 of lane select
			lane_ind = Q6_V_vdelta_VV( lane_ind, lane_ind2 );		// select odd/even from lane+0 or +2
			lane_ind = Q6_V_vor_VV( lane_ind, lane_ind2);			// combine

			// now each 'w' lane contains bytes { ind0, ind1, X,X } where 'ind0' is 0..3 for job0
			// and ind1 is 0..3 for job1 .
			// we need to extract these into two 'w' words (and add the 'curr_index' in the process)
			HVX_Vector indices_0 = Q6_Vuw_vrmpyacc_VuwVubRub( curr_index, lane_ind, 0x000000001 );
			HVX_Vector indices_1 = Q6_Vuw_vrmpyacc_VuwVubRub( curr_index, lane_ind, 0x000000100 );
			// now we have, in each job, 32 'best indices' each describing a group of 4 lanes.
			// need to combine these with the 'x' values now (xor them with 0x7f first)
			// each w lane of vmax contains { max_0, max_1, xx, xx }
			vmax = Q6_V_vxor_VV( vmax, q6op_Vb_vsplat_R(0x7F));

			// insert those bytes in bits 31..24 of each index word.
			// apply the even slots to indices_0 (shifting out 24 bits of garbage in the process)
			indices_0 = Q6_Vw_vaslacc_VwVwR( indices_0, vmax, 24);
			indices_1 = Q6_Vw_vaslacc_VwVwR( indices_1, Q6_Vb_vshuffo_VbVb( vmax, vmax), 24);

			// now we have two groups of 32 which need min-reducing. transpose them together and reduce across.
			// Then, even w lanes contain the results for problem 0, odd lanes for problem 1.
			HVX_VectorPair sh = Q6_W_vshuff_VVR( indices_1, indices_0, 4);
			allmin = Q6_Vw_vmin_VwVw( allmin, Q6_Vw_vmin_VwVw( Q6_V_lo_W(sh), Q6_V_hi_W(sh)));
			curr_index  = Q6_Vw_vadd_VwVw( curr_index,Q6_V_vsplat_R(128) );	// move to next set of indices
		}
		// ok .. we now have 16 min-reduced values for each of 2 jobs. Funnel-reduce
		funnelreduce_insert_value( &frstate, allmin);
	}
	funnelreduce_flush_pipe( &frstate, outp + rows);
}

//The range of the indices is taken care of by the caller. indices are always valid
//rowno is the width idx among the total number of width including back paddings(not including front paddings)
//real_data_h is the pointer to the current height block
//the widths in the same height block are connected (no paddings), so read the block based on the width index (w_idx)
void get_4_width_block(int *rowno, int * h_idx, int * b_idx, int dim_w, int dim_h, struct tensor const * data_tensor,
                       int dim_w_aligned, int h_stride, uint8_t **real_data_h,
                       uint8_t ** out_row_p, int * out_num_valid_rows) {

    //real_data_h points to the current h block
    //figure out if should move to the next h block
    int w_idx;
    if(dim_w < 4) { //every 4w block is in a new h block
        w_idx = 0;
    }
    else {
        w_idx = *rowno % dim_w_aligned;// rowno can only be 0, 4, 8, 12,...; dim_w_aligned: round up dim_w to multiple of 4
    }
    *out_row_p = (uint8_t*)*real_data_h + w_idx*32;

    //check if this is the last 4w of an h block
    *out_num_valid_rows = 4;
    if(w_idx + 4 >= dim_w) { //last 4w in h, or dim_w < 4
        //update the pointer to the h block for the next function call
        ++(*h_idx);
        *real_data_h += h_stride;   //next height
        if( (*h_idx) == dim_h) {
            (*h_idx) = 0;
            ++(*b_idx);
            *real_data_h = tensor_location_bhw_d32(data_tensor,*b_idx,0,0); //next batch
        }

        if(w_idx + 4 > dim_w) {
            *out_num_valid_rows = dim_w - w_idx;
        }
    }
}

//argmin/max along depth for d32 format
void
hvx_argmin_or_max_d_8_d32(struct tensor const * data_tensor, int32_t * outp, int find_argmax) {

    HVX_Vector indices = *(HVX_Vector const *) const_Count32;        // bytes {0..31, 0..31, 0..31, 0..31}
    indices = Q6_V_vand_VV(indices, Q6_V_vsplat_R(0xFF));            // { 0,4,8,..28, 0..28, 0..28, 0..28 } in w lanes.

    // if we are finding min instead of max, invert all the  values as we read them.
    HVX_Vector invert_input = Q6_V_vsplat_R(find_argmax ? 0 : -1);

    int dim_b = data_tensor->shape.batches;
    int dim_w = data_tensor->shape.width;
    int dim_h = data_tensor->shape.height;
    int dim_d = data_tensor->shape.depth;

    int h_stride = tensor_row_stride_d32(data_tensor);

    int rows;
    int dim_w_aligned = (dim_w+3)& ~0x03;//round up dim_w to multiple of 4
    rows = dim_b * dim_h * dim_w_aligned; //number of rows with paddings

    int dim_d32blocks = (dim_d + 31) / 32;
    int d32_block_stride =  tensor_d32_stride_d32(data_tensor);
    uint8_t * real_data_h = tensor_location_bhw_d32(data_tensor,0,0,0); //points to each height block

    int b_idx = 0;
    int h_idx = 0;
    int only_one = 0;

    static const unsigned char vrdelta_controls[] __attribute__((aligned(128))) = {
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,
            0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,
            0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
            0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
            0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
            0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
    };

    HVX_Vector depth_pattern =*(HVX_Vector*) vrdelta_controls;

    HVX_VectorPred depth_mask_ub_last_d32 = Q6_Q_vsetq_R(dim_d%32u);
    depth_mask_ub_last_d32 = Q6_V_vrdelta_VV(depth_mask_ub_last_d32, depth_pattern);  //mask of the last depth block
    HVX_Vector depth_mask_ub = Q6_V_vand_QR(depth_mask_ub_last_d32, 0xFFFFFFFF);

    uint8_t * new_4w0;
    uint8_t * new_4w1;

    int output_offset = 0;
    int output_increment0 = 0;//dim_w % 4 == 0 ? 16 : (dim_w % 4) * 4;//num of bytes to output per 4 widths
    int output_increment1 = 0;

    for (int rowno = 0; rowno < rows; rowno += 4) {

        // Read data block of every 4 widths. There are 32 quantized 8 values in one width. An HVX Vector can contain 4 widths
        get_4_width_block(&rowno, &h_idx, &b_idx, dim_w, dim_h, data_tensor,
                          dim_w_aligned, h_stride, &real_data_h, &new_4w0, &output_increment0);
        only_one = (rowno + 4) >= rows; //only one rwo (4 widths) to be processed

        if(only_one) {
            new_4w1 = new_4w0;
            output_increment1 = 0;
        }
        else {
            rowno += 4;
            get_4_width_block(&rowno, &h_idx, &b_idx, dim_w, dim_h, data_tensor,
                              dim_w_aligned, h_stride, &real_data_h, &new_4w1, &output_increment1);
        }

        output_increment0 *= 4;  //num_valid_rows * 4-byte output per row
        output_increment1 *= 4;

        // prepare for a pass across
        HVX_Vector allmin = Q6_V_vsplat_R(0x7FFFFFFF);

        HVX_Vector curr_index = indices;
        for (int i = 0; i < dim_d32blocks; i++) {

            HVX_Vector vin0 = q6op_V_vldu_A((HVX_Vector *)&new_4w0[i * d32_block_stride]);
            HVX_Vector vin1 = q6op_V_vldu_A((HVX_Vector *)&new_4w1[i * d32_block_stride]);

            vin0 = Q6_V_vxor_VV(vin0, invert_input);
            vin1 = Q6_V_vxor_VV(vin1, invert_input);

            //update depth mask for the last d32 block
            if((i+1) == dim_d32blocks && (dim_d % 32 != 0)) { //if dim_d %32 == 0, no garbage to mask out

                vin0 = Q6_V_vand_VV(depth_mask_ub, vin0);   //keep valid depth values
                vin1 = Q6_V_vand_VV(depth_mask_ub, vin1);
            }

            // Have loaded 4 widths of d32 data in an HVX vector
            // Find min/max in every 4 bytes, and keep the min/max in the first byte. The rest is for keeping its index
            HVX_VectorPair vshuf = Q6_Wb_vshuffoe_VbVb(vin1,
                                                       vin0); //gather odd lanes in lo_W vshuf, even lanes in hi

            HVX_Vector vmax = Q6_Vub_vmax_VubVub(Q6_V_lo_W(vshuf),
                                                 Q6_V_hi_W(vshuf)); //reduce between odds and evens within vin_block

            HVX_VectorPred oddcol = Q6_Q_vcmp_gt_VubVub(Q6_V_hi_W(vshuf), Q6_V_lo_W(
                    vshuf)); //keep the indices of max between every two elems

            // at this point, even b lanes belong to problem 0 and odd to problem 1.
            // do it again, but with groups of 2
            HVX_Vector vmax1 = Q6_Vh_vshuffo_VhVh(vmax, vmax);    //duplicate the odds
            HVX_VectorPred odd2col = Q6_Q_vcmp_gt_VubVub(vmax1, vmax); //keep the indices of the groups of 2
            vmax = Q6_Vub_vmax_VubVub(vmax, vmax1); //compare max of 0,1 and max of 2,3 to reduce every 4 elements

            // ok, now:
            // vmax lanes 0,4, .. 124 are the best value in each group of 4 lanes for problem 0
            //            1,5 ..  125 are for problem 1
            // the 'odd2col' value in those lanes tells us whether the byte came from first or 2nd group of 2.
            //  but oddcol is trickier; it has bit 0 of the lane index, but we need to select odd or even
            // lanes based on odd2col.
            HVX_Vector lane_ind = Q6_V_vand_QR(oddcol, 0x01010101);  // get odd/even lane select
            HVX_Vector lane_ind2 = Q6_V_vand_QR(odd2col, 0x0202); // bit 1 of lane select
            lane_ind = Q6_V_vdelta_VV(lane_ind, lane_ind2);        // select odd/even from lane+0 or +2
            lane_ind = Q6_V_vor_VV(lane_ind, lane_ind2);            // combine-indices within the 4 elememts group

            // now each 'w' lane contains bytes { ind0, ind1, X,X } where 'ind0' is 0..3 for job0
            // and ind1 is 0..3 for job1 .
            // we need to extract these into two 'w' words (and add the 'curr_index' in the process)
            HVX_Vector indices_0 = Q6_Vuw_vrmpyacc_VuwVubRub(curr_index, lane_ind, 0x000000001);
            HVX_Vector indices_1 = Q6_Vuw_vrmpyacc_VuwVubRub(curr_index, lane_ind, 0x000000100);

            // now we have, in each job, 32 'best indices' each describing a group of 4 lanes.
            // need to combine these with the 'x' values now (xor them with 0x7f first)
            // each w lane of vmax contains { max_0, max_1, xx, xx } - xx is garbage data
            vmax = Q6_V_vxor_VV(vmax, q6op_Vb_vsplat_R(0x7F));

            // insert those bytes in bits 31..24 of each index word.
            // apply the even slots to indices_0 (shifting out 24 bits of garbage in the process)
            indices_0 = Q6_Vw_vaslacc_VwVwR(indices_0, vmax, 24);
            indices_1 = Q6_Vw_vaslacc_VwVwR(indices_1, Q6_Vb_vshuffo_VbVb(vmax, vmax), 24);

            // now we have two groups of 32 which need min-reducing. transpose them together and reduce across.
            // Then, even w lanes contain the results for problem 0, odd lanes for problem 1.
            HVX_VectorPair sh = Q6_W_vshuff_VVR(indices_1, indices_0, 4);
            allmin = Q6_Vw_vmin_VwVw(allmin, Q6_Vw_vmin_VwVw(Q6_V_lo_W(sh), Q6_V_hi_W(sh)));//get the minimum index if the values are the same
            curr_index = Q6_Vw_vadd_VwVw(curr_index, Q6_V_vsplat_R(32));    // move to next d32 block
        }
        //allmin contains 2 x 4w x 4(each width) x i32(each value)
        //horizontally reduce every 4
        //get 2 x 4w x 1(each width) x i32(each value) -> 8 x 4Bytes results
        HVX_Vector allmin1 = Q6_V_vror_VR(allmin, 16);//16
        allmin = Q6_Vw_vmin_VwVw(allmin, allmin1);

        allmin1 = Q6_V_vror_VR(allmin, 8);
        allmin = Q6_Vw_vmin_VwVw(allmin, allmin1);

        //mask out value. Keep indices only
        allmin = Q6_V_vand_VV( allmin, Q6_V_vsplat_R(0xFFFFFF));

        HVX_Vector tmp = Q6_V_lo_W( Q6_W_vdeal_VVR( allmin,allmin, 0x48));
        // now in w lanes { 0,1, 8, 9, 2,3, 10, 11 }
        HVX_VectorPair result  = Q6_W_vdeal_VVR( tmp,tmp, 0x24);
        // now even values in lanes 0,1,2,3 of result.lo, odd in lanes 0,1,2,3 of result.hi

        //outp + rows
        q6op_vstu_variable_ARV((HVX_Vector *) ((uint8_t*)outp+output_offset), output_increment0, Q6_V_lo_W(result));
        output_offset+= output_increment0;

        if(!only_one) {
            q6op_vstu_variable_ARV((HVX_Vector *) ((uint8_t*)outp+output_offset), output_increment1, Q6_V_hi_W(result));
            output_offset+= output_increment1;
        }
    }
}

//
// this is called with a dest pointer, two vectors, and 'bytes' in range 1..256.
// The first 'bytes' bytes from the vectors (v0 followed by v1) will be stored at the address, using
// unaligned and masked stores as needed. If bytes <=0, nothing is stored; if bytes > 256
// the effect is the same as bytes == 256 (all stored).
//
void __attribute__((noinline))
hvx_store_vec_x2_unaligned ( void * addr, HVX_Vector v0, HVX_Vector v1, int bytes)
{
	HVX_Vector * outp = (HVX_Vector *)addr;

	if( bytes >= 128){
		q6op_vstu_AV( outp, v0);
		outp ++;
		bytes -= 128;
		v0 = v1;
	}
	if( bytes >= 128){
		q6op_vstu_AV( outp, v0);
	}else if (bytes >= 1 ){
		q6op_vstu_variable_ARV( outp, bytes,  v0 );
	}
}
//
// given an array of 'rows' by 'cols' uint8;
// - find the row index of the largest # (or smallest) in each column. If the max (or min) occurs more than once,
// the result is the smallest index of all these.
// - must have rows >= 1, cols >=1
// - the results are stored at outp[ 0..cols-1]
// - no vector-alignment constraints. The 'row stride' is a separate parameter.
//
// Strategy:
//   - work on slices of up to 128 cols at once
//   within each slice: process 256 rows at once, entirely in byte lanes;
//    at the end of each block, update the 32-bit indices in any of the columns which are improved in the block.
//
//
void
hvx_argmin_or_max_in_cols( uint8_t const * data, int rows, int cols, int row_stride, int32_t * outp , int find_argmax)
{
	int slices = (cols+127)/128u;		// number of col slices to do
	// if we are finding min instead of max, invert all the  values as we read them.
	HVX_Vector invert_input = Q6_V_vsplat_R( find_argmax?0:-1);
	HVX_Vector k_1 = q6op_Vb_vsplat_R(1);

	// column-slice loop : up to 128 columns

	for( int islc = 0; islc < slices; islc++){
		uint8_t const * datap = data + 128*islc;			// point to start of slice
		int slc_cols = min_i32( cols-128*islc, 128);	// # of cols in this slice

		// initialize the 'best index' values for all 128 cols
		HVX_Vector best_index_0 = Q6_V_vzero();		// cols 0,2, ...
		HVX_Vector best_index_1 = Q6_V_vzero();		// cols 1,3, ...
		HVX_Vector best_index_64 = Q6_V_vzero();		// cols 64,66, ...
		HVX_Vector best_index_65 = Q6_V_vzero();		// cols 66,67 ...
		HVX_Vector all_max_outer = Q6_V_vzero();

		// outer row loop (groups of up to 256 rows)
		for( int rowbase = 0; rowbase < rows; rowbase += 256){
			HVX_Vector all_max_inner = all_max_outer;

			int inner_rows = rows-rowbase;
			if( inner_rows > 256) inner_rows = 256;		// inner row count (1...256)

			HVX_Vector cur_index = Q6_V_vzero();		// same as 'irow' in all byte lanes
			HVX_Vector inrow_index = Q6_V_vzero();		// best 'cur_index' in block

			// inner row loop - find any values which improve 'all_max_inner'; keep their
			// 8-bit row indices within the block.
			for(int irow = 0; irow < inner_rows; irow++){
				HVX_Vector vin = q6op_V_vldu_A( (HVX_Vector const*) datap);
				datap += row_stride;
				vin = Q6_V_vxor_VV( vin,invert_input );
				HVX_Vector new_better = Q6_Q_vcmp_gt_VubVub(vin,all_max_inner );
				all_max_inner = Q6_Vub_vmax_VubVub( all_max_inner, vin);
				inrow_index = Q6_V_vmux_QVV( new_better, cur_index, inrow_index);	// replace where better.
				cur_index = Q6_Vb_vadd_VbVb( cur_index, k_1);			// +1 for next time
			}
			// now: find the lanes which improved on the previous all_max_outer
			// these are the lanes where we need to update the best_index_XX values
			HVX_Vector is_better = Q6_Q_vcmp_gt_VubVub( all_max_inner, all_max_outer);
			all_max_outer = all_max_inner;
			// mask is 0xFF in lanes where there was no change in this batch.
			HVX_Vector mask = q6op_V_vand_QnR( is_better, -1);
			// shuffle the inrow_index with the mask.
			HVX_VectorPair newindh = Q6_W_vshuff_VVR( mask, inrow_index, -1);
			// results are <0 in lanes which don't need update, and 0..255 in lanes which do.
			// sign-extend these to w size...
			HVX_VectorPair newindw_0 = Q6_Ww_vsxt_Vh( Q6_V_lo_W(newindh)); // 0,2, ... and 1,3 ...
			HVX_VectorPair newindw_64 = Q6_Ww_vsxt_Vh( Q6_V_hi_W(newindh)); // 64,66, ... and 65, 67, ...
			// now 'or' in the current 'rowbase' in all rows; the result will stay <0 where is_better =0,
			// and will be the full row index in other cases. then 'vmax' to the running vals.
			HVX_Vector rowbase_vec = Q6_V_vsplat_R(rowbase);
			best_index_0 = Q6_Vw_vmax_VwVw( best_index_0, Q6_V_vor_VV( Q6_V_lo_W(newindw_0), rowbase_vec));
			best_index_1 = Q6_Vw_vmax_VwVw( best_index_1, Q6_V_vor_VV( Q6_V_hi_W(newindw_0), rowbase_vec));
			best_index_64 = Q6_Vw_vmax_VwVw( best_index_64, Q6_V_vor_VV( Q6_V_lo_W(newindw_64), rowbase_vec));
			best_index_65 = Q6_Vw_vmax_VwVw( best_index_65, Q6_V_vor_VV( Q6_V_hi_W(newindw_64), rowbase_vec));
		}
		// now we just need to store out the values (up to 128 of them).
		// Shuffle the first two vectors together first.
		HVX_VectorPair shuffled = Q6_W_vshuff_VVR( best_index_1, best_index_0,-4);

		if( slc_cols >= 64){		// need to store at least 64
			q6op_vstu_AV( (HVX_Vector *)outp, Q6_V_lo_W(shuffled));
			outp += 32;
			q6op_vstu_AV( (HVX_Vector *)outp, Q6_V_hi_W(shuffled));
			outp += 32;
			slc_cols -= 64;
			// shuffle the rest
			shuffled = Q6_W_vshuff_VVR( best_index_65, best_index_64,-4);
		}
		// slc_cols = 0..64 now
		if( slc_cols == 64){
			q6op_vstu_AV( (HVX_Vector *)outp, Q6_V_lo_W(shuffled));
			outp += 32;
			q6op_vstu_AV( (HVX_Vector *)outp, Q6_V_hi_W(shuffled));
			outp += 32;
		}else {
			// must be the last outer loop - tail call
			if( slc_cols >0 )
				hvx_store_vec_x2_unaligned( outp, Q6_V_lo_W(shuffled), Q6_V_hi_W(shuffled), 4*slc_cols );
			return;
		}
	} // for islc
}

//argmin/max along batch, height or width for D32 format
void
hvx_argmin_or_max_whb_8_d32( struct tensor const * data_tensor, int32_t * outp, int32_t  axis, int find_argmax) {

    int dim_b = data_tensor->shape.batches;
    int dim_w = data_tensor->shape.width;
    int dim_h = data_tensor->shape.height;
    int dim_d = data_tensor->shape.depth;

    int outer_dim = 0;
    int inner_dim = 0;
    int outer = 0;
    int inner = 0;

    int dim_axis = 0;

    int32_t data_stride = 0;
    int32_t inner_stride = 0;
    int32_t outer_stride = 0;
    switch(axis) {
        case 0: //batch
            data_stride = tensor_batch_stride_d32(data_tensor);    //to the next batch
            outer_dim = dim_h;  //h
            inner_dim = dim_w;  //w
            dim_axis = dim_b;
            outer_stride = tensor_row_stride_d32(data_tensor);
            inner_stride = 32;
            break;
        case 1: //height
            data_stride = tensor_row_stride_d32(data_tensor);   //to the next height
            outer_dim = dim_b;  //b
            inner_dim = dim_w;  //w
            dim_axis = dim_h;
            outer_stride = tensor_batch_stride_d32(data_tensor);
            inner_stride = 32;
            break;
        case 2: //width
            data_stride = 32;                                   //to the next width
            outer_dim = dim_b;  //b
            inner_dim = dim_h;  //h
            dim_axis = dim_w;
            outer_stride = tensor_batch_stride_d32(data_tensor);
            inner_stride = tensor_row_stride_d32(data_tensor);
            break;
        default:
            break;
    }

    int32_t out_offset = 0;
    int32_t output_increment = 0;
    int num_d32blocks = (dim_d + 31) / 32;
    int d32_block_stride = tensor_d32_stride_d32(data_tensor);  //to the next d32 block
    uint8_t* real_data_h = tensor_location_bhw_d32(data_tensor,0,0,0);
    uint8_t* real_data_p = real_data_h;
    int num_cols = 32;  //process a d32 slice per loop

    for(outer = 0; outer < outer_dim; ++outer) {
        real_data_p = real_data_h + outer_stride * outer;
        for(inner = 0; inner < inner_dim; ++inner){

            //process a d32 slice per loop
            for(int32_t d = 0; d < num_d32blocks; ++d) {

                hvx_argmin_or_max_in_cols(real_data_p+d*d32_block_stride, dim_axis, num_cols, data_stride, outp+out_offset, find_argmax);

                if(d == (num_d32blocks -1)) {
                    output_increment = dim_d - (num_d32blocks - 1) * 32;
                }
                else {
                    output_increment = 32;
                }
                out_offset += output_increment ;       //get rid of garbage by not outputing it
            }

            real_data_p += inner_stride;
        }
    }
}
