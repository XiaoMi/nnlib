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

