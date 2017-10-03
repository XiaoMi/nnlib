/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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
/*
 * utilities for tensor shapes
 */
#include <stdint.h>
#include <nn_graph.h>
#include "nn_graph_types.h"
#include "quantize.h"

//
// 'actual function' version of this inline operation.
struct tensor_addressing
tensor_addressing_d32_func( struct tensor const * src)
{
	return tensor_addressing_d32(src);
}

//
// check to see if tensorA and tensorB
// (both d32) are compatible for an elementwise operation.
// This includes broadcasting B dims to A.
// It also checks for misalignments in W & D dimensions.
//
// Normally it will a value >= 0, an 'or' of the tensor_compat_flags.
// if a problem is found, it will report an error and then return -1;
// 'ernm' is a name for the error messages.
//
// If there are compatibility flags and the corresponding bits are
// *not* in allowed_compat, this is considered an error.
//  e.g. A.height = 20, B.height=1 is considered a mismatch if compat_broadcast_H
// is not in the width dimension.
//
// if compat_AtoB is in allowed_compat, situations where A broadcasts to B are
// accepted, and allowed_compat will be set in the return value when this is encountered
// (it will only be set if the A tensor is smaller than B).
// 'mixed' cases (e.g. (1,1,3,32), (1,5,3,1) are not accepted in any case.
//
// NOTE:
// - if dimensions are both 1, this tagged as 'broadcast' but only if the
//    broadcast is enabled in 'allowed_compat' ; otherwise it will not be tagged.
//
// - if you allow 'broadcast_W', then misalign_W is considered acceptable when B.width =1,
//    even if it's not in allowed_compat. It will be reported in the return result. This
//    may occur when both widths are 1
//
// - if you allow 'broadcast_D', then misalign_D is considered acceptable when B.depth=1,
//    even if it's not in allowed_compat.This will be reported in the return result. This
//    may occur when both depths are 1.
//
//  - skewed_D is a subset of misalign_D.
//    misaligned_D     skewed_D
//           0             0              both have the same depth_before padding
//           1             0              depth-before paddings different, but both fit in one d32
//           1             1              other cases.
//     skewed_D will never be flagged in the return value when tensB.depth =1.

//
int
check_compatible_elementwise_d32(
		struct nn_graph *nn,
		char const * errnm,
		struct tensor const * tensA,
		struct tensor const * tensB,
		int allowed_compat)
{
	// check formats
	if( tensA->format.layout != NN_LAYOUT_D32){
		return errlog(nn, "%s: 1st tensor not d32", errnm);
	}
	if( tensB->format.layout != NN_LAYOUT_D32){
		return errlog(nn, "%s: 2nd tensor not d32", errnm);
	}
	int BtoA_ok = 1;		// ok to broadcast B to A
	int AtoB_ok = (allowed_compat & compat_AtoB)? 1: 0;

	// check all the dims
	int compat_tags = 0;
	for( int i =0; i < 4; i++){
		int dimtag = compat_broadcast_B << i;
		int dima = tensA->shape.dimension[i];
		int dimb = tensB->shape.dimension[i];
		if( dima == dimb){
			if(dimb == 1)  compat_tags  |= ( allowed_compat & dimtag);
		}else{
			if( dimb == 1 &&  BtoA_ok &&  ( allowed_compat & dimtag)!=0 ){	// it's ok to broadcast B->A
				compat_tags |= dimtag;
				AtoB_ok = 0;	// block the other option
			}else if( dima == 1 &&  AtoB_ok &&  ( allowed_compat & dimtag)!=0 ){		// A->B
				compat_tags |= dimtag|compat_AtoB;
				BtoA_ok = 0;	// block the other option
			}else{
				return errlog(nn,"%s: incompatible %c: dims: %d vs %d", errnm, "BHWD"[i], dima, dimb );
			}
		}
	}
	// check width alignment
	if(  ((tensA->format.width_pad[0]^tensB->format.width_pad[0])&3)!=0 ){
		compat_tags |= compat_misalign_W;
		if( (allowed_compat & compat_misalign_W )!= 0   && (compat_tags & compat_broadcast_W)==0){
			return errlog(nn,"%s: width misalignment not supported", errnm);
		}
	}

	// check depth alignment
	int dpA = tensA->format.depth_pad[0];
	int dpB = tensB->format.depth_pad[0];

	if( dpA != dpB){
		compat_tags |= compat_misalign_D;
		if( (compat_tags & compat_broadcast_D)== 0 ){
			if(	max_i32(dpA,dpB)+ tensA->shape.depth > 32 ){
				compat_tags |= compat_skewed_D;
			}
			if(  (compat_tags & ~allowed_compat & (compat_misalign_D|compat_skewed_D)) != 0 ){
				return errlog(nn,"%s: depth misalignment not supported", errnm);
			}
		}
	}
	return compat_tags;
}


//
// This function looks at the shapes in an array of 1 or more 'struct_tensor'
// and, assuming they will be concatenated on dimension 'concat_dim', finds
// the overall shape.
// It also range-checks 'concat_dim' (must be 0..3) and
// ensures that all shapes match in all dims *other* than concat_dim.
//
// The output shape is the sum of the input shapes (on concat_dim) and
// matches them on all others.
//
// returns:
//   0:  ok
//   -1: concat_dim out of range
//   -2..-5: mismatch on dimension 0,1,2,3
//
// - No change is made to *allshape unless the function returns 0
// - caller to ensure that n_input >= 1.
//
//
int
find_concat_shape(
		const struct tensor **input_tensors,
		int n_input,			// >= 1
		int concat_dim,			// 0..3
		struct shape *allshape )
{
	// the first input tensor provides 'ref' dims

	uint32_t ref_batches = input_tensors[0]->shape.batches;
	uint32_t ref_height = input_tensors[0]->shape.height;
	uint32_t ref_width = input_tensors[0]->shape.width;
	uint32_t ref_depth = input_tensors[0]->shape.depth;

	uint32_t anydel_batches = 0, anydel_height = 0;
	uint32_t anydel_width = 0, anydel_depth = 0;
	uint32_t sum_del = 0;
	if( !(concat_dim >= 0 && concat_dim <=3)) return -1;

	// for all the others:
	//  find  del_XX = XX - ref_XX (mod uint32)
	//    - 'or' them all so that we can tell if any are different from ref
	//    - sum them all (across all dims) so we can figure the size of the
	//     concat dimension later.

	int i;
	for( i = 1 ; i < n_input; i++){
		uint32_t del_batches =  input_tensors[i]->shape.batches - ref_batches;
		anydel_batches |= del_batches;
		sum_del += del_batches;
		uint32_t del_height =  input_tensors[i]->shape.height - ref_height;
		anydel_height |= del_height;
		sum_del += del_height;
		uint32_t del_width =  input_tensors[i]->shape.width - ref_width;
		anydel_width |= del_width;
		sum_del += del_width;
		uint32_t del_depth =  input_tensors[i]->shape.depth - ref_depth;
		anydel_depth |= del_depth;
		sum_del += del_depth;
	}
	// now:
	//  -all anydel_XX (except in the current dim) must be zero.
	//  -sum_del is the sum of the deltas XX - ref_XX in all dimensions on all inputs.
	//     This contains no contributions from the non-selected
	//     dimensions. So if we add n*ref_XX to this, the result is the sum
	//     of the selected dim across all inputs.


	if( anydel_batches != 0 && concat_dim != 0 ) return -2;	// mismatch on batches
	if( anydel_height != 0 && concat_dim != 1 ) return -3;	// mismatch on height
	if( anydel_width != 0 && concat_dim != 2 ) return -4;	// mismatch on width
	if( anydel_depth != 0 && concat_dim != 3 ) return -5;	// mismatch on depth

	// fill out the result...
	// one of these needs to be corrected later

	allshape->batches = ref_batches;
	allshape->height = ref_height;
	allshape->width = ref_width;
	allshape->depth = ref_depth;

	switch( concat_dim){
	 case 0:
		allshape->batches = n_input * ref_batches + sum_del;
		break;
	 case 1:
		allshape->height = n_input * ref_height + sum_del;
		break;
	 case 2:
		allshape->width = n_input * ref_width + sum_del;
		break;
	 case 3:
		allshape->depth = n_input * ref_depth + sum_del;
		break;
	}
	return 0;
}



//
// This function is given an input d32 tensor, and a reference d32 tensor
// (only used for its width,depth dimensions, and padding info), and optionally
// a work area; and it constructs a memory array formatted as if it were
// a (1,1,w,d) tensor (with w,d taken from tens_ref); this data also has
// the same depth and width padding as tens_ref.
//
// - the function provides a d32_stride for the constructed data. If broadcasting
//   on depth is being done (tens_in->depth = 1), this pitch is always 0, regardless
//   of the output depth (i.e. only one depth slice is constructed, all values are the
//   same across depth anyway.
//
// CALLER MUST ENSURE:
//   - tens_in and test_ref are both valid d32 tensors. b & h dimensions and padding are ignored.
//   - in the w and d dimensions, tens_in must either be 1 or must match tens_ref.
//   - if workbuf is not NULL, it must point to a vec-aligned work area of at least workbuf_len bytes
//   .. otherwise function may return NULL, or undefined behaviour may occur.
//
//
// The function returns the (vector-aligned) pointer to the data.
// it will return NULL if allocation fails or if a parameter problem is detected.
//
// Memory allocation:
//	 - In a fews cases the function may avoid a copy, and returns a pointer to the data referenced
// 	   by tens_in. So d32_stride_out will be from tens_in.
//   - Caller can supply a workbuf (and workbuf_len); this will be used if it is large enough,
//     otherwise memory will be allocated. (use workbuf = NULL, or workbuf_len = 0  if not supplying a buf)
// 'allocbuf_out' must be a pointer to a void * variable.
//    - this will be set to NULL if the routine allocated no memory.
//    - if not NULL, the value must be passed to nn_free to free the allocated memory.
//
// This function doesn't use any HVX ops (but it uses some hexagon 64-bit ops)
//
//
uint8_t const *
construct_broadcasted_data_d32(
		struct tensor const * tens_in,		// tensor containing data
		struct tensor const * tens_ref,		// tensor used as shape/alignment ref
		int32_t *d32_stride_out,			// used to return the d32 stride of the result
		void * workbuf,						// optional work area
		uint32_t workbuf_len,				// len in bytes of work area
		void **allocbuf_out )
{
	int d_in = tens_in->shape.depth;
	int w_in = tens_in->shape.width;
	int wpad0_in = tens_in->format.width_pad[0];
	int dpad0_in = tens_in->format.depth_pad[0] & 31;

	int d_out = tens_ref->shape.depth;
	int w_out = tens_ref->shape.width;
	int wpad0_out = tens_ref->format.width_pad[0];
	int wpad1_out = tens_ref->format.width_pad[1];
	int dpad0_out = tens_ref->format.depth_pad[0] & 31;

	// get input data pointer. This is the start of the first depth group.
	uint8_t const *data_in = tensor_location_bhw_d32( tens_in, 0,0,0);
	int d32_stride_in = tensor_d32_stride_d32( tens_in);


	// determine output size
	// nd32_out is the nunber of deph slices, based on depth and output padding
	// we only need one when d_in = 1 (and when d_in != 1, d_in == d_out).

	int nd32_out = (unsigned)( dpad0_out + d_in + 31)/32;
	// output row bytes
	int wtotal_out = (wpad0_out + w_out + wpad1_out + 3)&~3;	// it should already be multiple of 4.

	*allocbuf_out = NULL;

	// (check for case where we can alias the input)
	// only possible when:
	//   d & w dimensions match
	//  dpad0 matches
	//   wpad0 matches mod 2, and output padding <= input padding.
	//  I.e. we are adjusting the pointer for (possibly) less left padding, nothing else.
	if(  w_in==w_out && d_in==d_out  && dpad0_in == dpad0_out && ((wpad0_in^wpad0_out)&3)==0
		&& wpad0_in >= wpad0_out  && tens_in->format.width_pad[1] >= wpad1_out ){
		*d32_stride_out = d32_stride_in;
		return data_in - wpad0_out*32;
	}

	int d32row_bytes = wtotal_out* 32;
	int allocsize = d32row_bytes * nd32_out;
	if( allocsize <= 0){
		return NULL;
	}
	void *mbuf;
	if( workbuf != NULL && workbuf_len >= (uint32_t)allocsize){
		mbuf = workbuf;
	}else{
		mbuf = nn_memalign(128,allocsize);
		*allocbuf_out = mbuf;
		if( mbuf == NULL) return NULL;
	}
	uint8_t * mbuf_u8 = (uint8_t *)mbuf;

	//
	if( d_in == 1){	// broadcast along depth
		uint8_t const * rp =  & data_in[dpad0_in];
		uint64_t * wp;
		int wcount;
		int dcount;		// this is in units if 16 bytes
		int i,j;

		if( w_in == 1){ // broadcast along width too
			wp = (uint64_t*)mbuf;
			wcount = 1;			// just fill the whole thing...
			dcount = wtotal_out*2;
		}else{
			if( w_in != w_out) goto do_error;
			wp = (uint64_t*)mbuf + wpad0_out * 4;	// start after output width padding
			wcount = w_out;
			dcount = 2;
		}
		for( i =0; i < wcount; i++ ){
			uint32_t d32 = Q6_R_vsplatb_R(rp[i*32]);	// get one byte from input
			uint64_t data = Q6_P_combine_RR( d32,d32);	// splat to 8 bytes
			for( j = 0; j < dcount; j++){
				*wp++ = data;
				*wp++ = data;
			}
		}
		// broadcast along depth, d32_stride is zero.
		*d32_stride_out = 0;
		return mbuf_u8;
	}else{
		if( d_in != d_out) goto do_error;
		if(w_in ==1){
			// broadcast along width only; and align on depth to output.
			// Start by constructing the depth slice (with dpad0_out) in the first
			// 32*nd32_out bytes of the buffer; and then copy to the rows.
			uint8_t *wp = mbuf_u8 + dpad0_out;
			int di_in = dpad0_in;
			int depth_remain = d_out;
			int i,j;
			while(1){
				int dcopy = min_i32( depth_remain, 32-di_in);	// most we can copy
				memcpy( wp, data_in+di_in, dcopy);
				depth_remain -= dcopy;
				if( depth_remain <= 0) break;
				wp += dcopy;		// writes are contiguous
				di_in = 0;			// move to next depth slice in input
				data_in += d32_stride_in;
			}
			// for each d32 slice, copy it to its allocated row. Start with the last
			// one (so we don't overwrite)
			//
			for( i = nd32_out-1; i >= 0; --i ){
				uint64_t const * rp = (uint64_t const *)( mbuf_u8 + 32 * i);
				uint64_t *wp = (uint64_t *)( mbuf_u8 + d32row_bytes * i);
				uint64_t x0 = rp[0];
				uint64_t x1 = rp[1];
				uint64_t x2 = rp[2];
				uint64_t x3 = rp[3];
				for( j = 0; j < wtotal_out; j++){
					wp[0] = x0;
					wp[1] = x1;
					wp[2] = x2;
					wp[3] = x3;
					wp += 4;
				}
			}
		}else{
			if( w_in != w_out) goto do_error;
			// broadcasting neither on width or depth, but there is some misalignment to take care of.
			// This can be done with one memcpy per d32 slice if
			//  (1) both source and dest fit in a single depth unit; *or*
			//  (2) if they have the same depth padding (i.e. only the w is misaligned).
			if( ( nd32_out == 1 && dpad0_in + d_in <=32)
			  || dpad0_in == dpad0_out ){
					int dx_in = dpad0_in;
			  	  	int dx_out = dpad0_out;
			  	  	int dremain = d_in;
			  	  	uint8_t *wp = mbuf_u8+ 32*wpad0_out;
			  	  	uint8_t const * rp = data_in;
			  	  	while(1){
			  	  		int dcopy = min_i32( dremain, 32-dx_out);	// number to copy (per width)
						memcpy( wp + dx_out,	// output address
								rp + dx_in,					// input address
								(w_in-1)*32 + dcopy);
						dremain -= dcopy;
						if(dremain <=0)
							break;
						// we only get here if there are multiple depth slices with
						// the same dpad0 on in & out.
						wp += d32row_bytes;
						rp += d32_stride_in;
						dx_in = dx_out = 0;
			  	  	}
			}else{
				// the ugly case...
				// do this within each width slot, using 64-bit aligned reads and writes.
				// There are 4 u64 elements within each slice.
				//  precalculated:
				//   - xin0 = offset for initial read (0..3)
				//   - xout0 = offset for initial write (0..3)
				//   - skew = byte skew for align
				//   - count = number of u64's to write
				// rd_first is a flag indicating we need a pre-read at the start.
				// rd_last is a flag which is true if the last output word needs its second input read.
				//
				int skewn = (dpad0_in&7)-(dpad0_out&7);
				int xin0 = dpad0_in >> 3;	// offset containing first input byte; 0..3
				int xout0 = dpad0_out >>3;	// offset containing first output byte; 0..3
				int xcount =  ((dpad0_out + d_out+7)>>3) - (xout0+1);	// total # of u64s to write (-1)
				int rd_first = skewn >= 0;
				// only need to read extra for last, if the last output word is less full than
				// the last input word (avoid possible 'wild read').
				int rd_last =  (( dpad0_out + d_out-1)&7)  >  (( dpad0_in + d_out-1)&7);
				int i,j;
				uint64_t * wp = (uint64_t *)( mbuf_u8+ 32*wpad0_out );
				uint64_t const * rp = (uint64_t const *) data_in;

				uint64_t dL = 0;
				uint8_t skew = (uint8_t)skewn;

				for( i = 0; i < w_in; i++){
					uint64_t const * rpx = rp + i*4;
					uint64_t  * wpx = wp + i*4;
					int xin = xin0;
					int xout = xout0;
					if( rd_first){		// preload dL (if rd_first){
						dL = rpx[xin];
						if(++xin>=4){
							rpx = (uint64_t const*)( (char const*)rpx + d32_stride_in);
							xin = 0;
						}
					}
					for( j = 0; j < xcount; j++){
						uint64_t dR = rpx[xin];
						wpx[xout] =  Q6_P_valignb_PPp( dR, dL, skew);
						dL = dR;
						if(++xin>=4){
							rpx  = (uint64_t const*)( (char const*)rpx + d32_stride_in);
							xin = 0;
						}
						if(++xout>=4){
							wpx  = (uint64_t*)( (char*)wpx + d32row_bytes);
							xout = 0;
						}
					}
					// we have one more to do .. but only do read if rd_last != 0.
					uint64_t dLast = dL;
					if( rd_last) dLast = rpx[xin];
					wpx[xout] =  Q6_P_valignb_PPp( dLast, dL, skew);

				}
			}
		}
		*d32_stride_out = d32row_bytes;
		return mbuf_u8;
	}
 /* unreachable*/

 do_error:
 	 if( *allocbuf_out!=NULL){
 		 nn_free(*allocbuf_out);
 		 *allocbuf_out = NULL;
 	 }
 	 return NULL;
}


//
// This utility examines a d32 tensor, which is assumed to have a shape (1,1,w,d),
// and finds the actual range of u8 values stored within.
// The range is returned as  (maxval<<8) | minval.
// No hvx instructions (it uses 64-bit hexagon vector ops).
//
//
// it works in depth chunks of size 8, finding the range in each chunk,
// and then masking out unused lanes in the chunk before proceeding.
//
//  all_min,all_max = {all_ff},{all_00}
//  for each depth chunk containing data:
//      dc_min,dc_max = all_min, all_max
//      for iw = 0 .. width-1:
//          update dc_min,dc_max with data at iwid
//      all_min, all_max = dc_min,dc_max (but only in 'active' byte lanes)
//  reduce all_min, all_max laterally
//

uint32_t find_range_in_wd_tensor_d32( struct tensor const * tens_in )
{
	int width = tens_in->shape.width;
	int depth = tens_in->shape.depth;

	int dpos = tens_in->format.depth_pad[0];		// where depth data starts
	int dend = dpos + depth;						// where it ends
	int dpos0  = dpos & ~7;							// round down to boundary

	uint8_t const *data_in = tensor_location_bhw_d32( tens_in, 0,0,0);
	int d32_stride_in = tensor_d32_stride_d32( tens_in);

	int xind = dpos >> 3;			// get chunk index (0..3) within d32 slice

	uint64_t const * chunkp = (uint64_t const *)data_in + xind;	// point to first containing data.

	uint64_t all_min = (uint64_t)-1;
	uint64_t all_max = 0;

	while(1){		// while dpos < dend
		uint64_t dc_min = all_min;
		uint64_t dc_max = all_max;
		for( int i = 0; i < width; i++){
			uint64_t d = chunkp[i*4];
			dc_min = Q6_P_vminub_PP( dc_min, d);
			dc_max = Q6_P_vmaxub_PP( dc_max, d);
		}
		//
		// is chunk full?
		//
		int dpnext = dpos0 + 8;				// start of next chunk
		int msk= 0xFF;
		if( dpos0 < dpos || dpnext > dend ){	// chunk is not full
			msk = msk << (dpos&7);			// lo end mask
			if( dpnext > dend){
				int m2 = (1 << (dend&7))-1;		// hi end mask
				msk &= m2;
			}
		}
		// where msk=1, replace all_min, all_max with 'dc_min', 'dc_max'.
		all_min = Q6_P_vmux_pPP( msk, dc_min, all_min);
		all_max = Q6_P_vmux_pPP( msk, dc_max, all_max);
		if( dpnext >= dend)
			break;
		dpos =  dpos0 = dpnext;
		// advance to next chunk
		chunkp ++;
		if( ++ xind >= 4){
			xind = 0;
			chunkp = (uint64_t const *)( (uint8_t const *) chunkp - 32 + d32_stride_in);
		}
	}
	// now reduce the 8 max and 8 min..

	all_max = Q6_P_vmaxub_PP(
	 Q6_P_shuffeb_PP( all_max, ~all_min),
	 Q6_P_shuffob_PP( all_max, ~all_min)
	);
	// now we have 4 of { ~min, max}
	// 2 more reductions.
	all_max = Q6_P_vmaxub_PP( all_max, all_max >> 32 );
	uint32_t x = Q6_P_vmaxub_PP( all_max, all_max >> 16 );

	// truncate to 16 bits and ~min -> min
	return (uint16_t)x ^ 0xFF;
}

//
// When reducing selected dimensions on a linearly-addressed tensor,
// it is possible to 'munge' adjacent dims together if they are both
// reduced, or if both are not reduced. (where the input dim is 1,
// the dimension may be combined with either).
// for instance [2,3,5,6] -> [2,3,1,1] can be done as 6 1-d reductions [2*3,5*6]->[2*3,1]

// Thus with 4 dimensions, the worst case is to have 2 loops
// of reduction  (e.g [2,3,4,12] -> [2,1,4,1] we need reductions on 3->1, 12->1
// but  [2,3,1,12] -> [2,1,1,1] can be done as [2,3*1*12]->[2,1]
// Similarly, you never need more than 2 loops if iteration.
//
// General case can be done by mapping to a 5-dimensional case where
// the reduction is always done on the 2nd and 4th dims:
//    [p,q,r,s,t]->[p,1,r,1,t]
// (where at least one of p,r,t is 1; and s>1 (unless there is no reduction
// at all).
//
// Examples:
//  [2,3,5,6]->[1,1,1,1]     [1,1,1,2*3*5*2,1] -> [1,1,1,1,1]
//  [2,3,5,6]->[2,3,1,1]	 [1,1,2*3,5*6,1]	->[1,1,2*3,1,1]
//  [2,3,5,6]->[2,1,5,6]     [1,1,2,3,5*6] -> [1,1,2,1,5*6]
//
// This function finds the p,q,r,s,t parms to map a given
// reduction problem to [p,q,r,s,t]-> [p,1,r,1,t]
// It promises that:
//     -all are >= 1
//     -at least one of p,t is 1
//     -if s=1, then p=q=r=1 as well (this is a 'no reduction' case)
//     -if q=1, then p = 1  (1d reduction needs only r,s,t)
//     -if q >1, then r != 1  (2d reduction is only spec when needed)
//
// So there are four possible outcomes:
//     [1,1,1,1,t] :         no reduction, copy 't'
//     [1,1,r,s,t], s>1:     'r' of 1d-reduction(s) of t-vector
//                            r=t=1 is a full reduction.
//     [p,q,r,s,1], q,r,s>1  2-d reduction.
//     [1,q,r,s,t], q,r,s,t>1 :  2-d reduction
//
// IMPORTANT:
//  caller needs to ensure all dims >=1, and all dims
// match across in/out except where shape_out dim is 1.
//
void
nn_find_generic_reduction_dims(
		struct shape const *shape_in,
		struct shape const *shape_out,
		int generic_reduction_dims[5])
{
	int i;
	int red_dims = 0;
	int optred_dims = 0;		// 1->1 dims
	// make a tally of the dimensions that need reduction;
	// also find the input * output sizes
	int red_total = 1;
	int out_total = 1;
	for( i = 0; i < 4; i++ ){
		int dout = shape_out->dimension[i];
		int din = shape_in->dimension[i];
		out_total *= dout;
		if( dout == 1){
			int m = 1<<i;
			if( din > 1){
				red_dims |= m;
				red_total *= din;
			}else{
				optred_dims |= m;
			}
		}
	}
	// red_dims : a->1  (a > 1)
	// optred_dims: 1->1
	// nored_dims : a->a
	int nored_dims = (~(red_dims | optred_dims)) & 0xF;

	// note: out_total * red_total = in_total
	// default the first 4 outputs to 1...
	for( i = 0; i < 4; i++) generic_reduction_dims[i] = 1;

	// check special cases:
	// red_total = 1:  no reduction at all
	// out_total = 1: full reduction
	if( red_total==1 || out_total==1){
		generic_reduction_dims[3] = red_total;	// 's'
		generic_reduction_dims[4] = out_total;	// 't'
		return;
	}
	// now there is at least one dimension which is non-trivially
	// reduced, and at least one which is non-trivially maintained.
	// Starting with depth, and working to 'batch',
	// first, collect 't' from dims which are a->a (and including 1->1)
	// then 's' from dims which are a->1 (and including 1->1)
	// (the condition alternates, but 1->1 cases never force a change
	// of partition)
	//
	int idim = 3;
	int opos;
	for( opos = 4; opos >= 1;opos -- ){	// t,s,r,q
		int k = 1;
		int exclude_set = ((opos&1)==0)? red_dims : nored_dims;
		while(idim >=0 ){
			if( ((exclude_set>>idim)&1) != 0 )
				break;	// dim isn't in this partition.
			k *= shape_in->dimension[idim];
			idim--;
		}
		generic_reduction_dims[opos] = k;
		if(idim <0)
			break;	// all done
	}
	// we only finish that loop, with idim = 0,
	// in cases where we need p = in_batches = out_batches.
	//  e.g. [ 2, 4, 5, 16] -> [2, 1, 5, 1] needs pqrst = {2,4,5,16,1}
	if( idim >= 0){
		// it should be only 0
		generic_reduction_dims[0] = shape_in->batches;
	}
}
//
// This function finds the output shape for reductions
// The 'generic_reduction_dims' is obtained by by passing
// the input and output shape to find_generic_reduction_dims,
// but in this case the 'output shape' is the shape *before* squeezing
// reduced dims (which can be different from the output shape
// when padding == NN_PAD_VALID).
//
void
nn_find_reduction_shape(	struct nn_node *self,struct nn_graph *nn,
		struct shape *out_shape_p, int generic_reduction_dims[5])
{
	int i;
	if (self->n_inputs >= 2) {
		struct shape out_shape = self->inputs[0]->shape;
		const struct tensor *reduction_dims_tensor = self->inputs[1];
		const int32_t *dims = (const int32_t *)reduction_dims_tensor->data;
		int32_t dim;
		int repl = (self->padding == NN_PAD_VALID)? 0: 1;
		int32_t true_rank = 4;
		if( self->n_inputs >= 3) true_rank = tensor_get_int32(self->inputs[2],0);
		int reduce_all = 0;

		for (i = 0; i < reduction_dims_tensor->shape.depth; i++) {
			dim = 4 -true_rank + dims[i];	// 0,1,2,3 -> b,h,w,d
			if (dims[i] < 0){
				reduce_all = 1;
				break;
			}else if( 0 <= dim && dim <=3){
				out_shape.dimension[dim]= repl;
			}
		}
		if( !reduce_all){
			if ( self->padding == NN_PAD_VALID) {
				/* Dimensions to be reduced have been set to 0 */
				// copy the dims to out_shape_p, starting at the D, but skipping
				// zero dims; then pad 1's after.
				// As we do this we also need to replace the 0's with 1 in out_shape.
				int ir,iw = 3;
				for( ir = 3; ir >=0; ir--){
					int dn = out_shape.dimension[ir];
					if( dn != 0){
						out_shape_p->dimension[iw--] = dn;
					}else{
						// skip, and replace in out_shape.
						out_shape.dimension[ir] = 1;
					}
				}
				while( iw >= 0)	// fill vacated dims with 1
					out_shape_p->dimension[iw--]=1;
			}else{
				*out_shape_p = out_shape;
			}
			// find the 'generic reduction dims'
			nn_find_generic_reduction_dims( &self->inputs[0]->shape, &out_shape, generic_reduction_dims);
			/*
			{
				struct shape const *insh = &self->inputs[0]->shape;
				printf("[%d:%d:%d:%d]->[%d:%d:%d:%d] : [%d:%d:%d:%d:%d]\n",
						(int)insh->batches, (int)insh->height, (int)insh->width, (int)insh->depth,
						(int)out_shape_p->batches,(int)out_shape_p->height,(int)out_shape_p->width,(int)out_shape_p->depth,
						generic_reduction_dims[0],generic_reduction_dims[1],generic_reduction_dims[2],
						generic_reduction_dims[3],generic_reduction_dims[4]);
			}*/

			return;
		}
	}
	// set all to 1 (full reduction)
	//
	out_shape_p->batches = 1;
	out_shape_p->height = 1;
	out_shape_p->width = 1;
	out_shape_p->depth = 1;
	{
		struct shape const *insh = &self->inputs[0]->shape;
		generic_reduction_dims[0] = 1;
		generic_reduction_dims[1] = 1;
		generic_reduction_dims[2] = 1;
		generic_reduction_dims[3] = insh->batches * insh->height * insh->width* insh->depth;
		generic_reduction_dims[4] = 1;
	}

}

// for running a generic unary float op.
// This could be enhanced to use threads,
// currently does not (and need_hvx is ignored)
//
//
int nn_generic_unary_float_op( struct nn_node *self, struct nn_graph *nn,
		void (*func)( float *, float const *, int n, void *info),
		void * info, int need_hvx)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int elements = tensor_element_count(in_tensor);
	const float *in_data = in_tensor->data;
	float *out_data = out_tensor->data;

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT)!= 0){
		return errlog(nn,"%s: output too small", hexagon_nn_op_names[self->node_type]);
	}
	if(elements > 0 ){
		(*func)( out_data, in_data, elements, info);
	}
	return 0;
}



