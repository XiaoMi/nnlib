/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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

// Code from nn_integral_buffer.h
// which is shared across uses (avgpool_d32, l2pool_d32)
// and doesn't need to be inlined.
//
#include "nn_integral_buffer.h"

static void integ_buf_find_padding_scales( int16_t *outp, int n, int winsize, int infeas );
//
// set up the plan
//  return value:
//   <0:   error (and errlog has been called)
//    0:   OK (and reused old plan)
//    1:   OK (and built new plan)

int
setup_integral_buffer_plan( struct nn_node *self, struct nn_graph * nn,
		struct integral_buffer_plan * ibp,
		struct tensor const * in_tensor,
		struct tensor const * window_tensor,
		struct tensor const * stride_tensor)
{
	int in_ht = in_tensor->shape.height;
	int in_wid = in_tensor->shape.width;
	int input_wpad_left = in_tensor->format.width_pad[0];
	ibp->tin = tensor_addressing_d32(in_tensor);
	ibp->input_wpad_left = input_wpad_left;
	ibp->inshape.batches = ibp->outshape.batches = in_tensor->shape.batches;
	ibp->inshape.depth = ibp->outshape.depth = in_tensor->shape.depth;

	int win_ht = window_tensor->shape.height;
	int win_wid = window_tensor->shape.width;
	int stride_ht = stride_tensor->shape.height;
	int stride_wid = stride_tensor->shape.width;

	// 'fast path'
	if(  in_ht == ibp->prev_in_ht &&  in_wid == ibp->prev_in_wid
		&& win_ht == ibp->prev_win_ht && win_wid == ibp->prev_win_wid
		&& stride_ht == ibp->prev_str_ht && stride_wid == ibp->prev_str_wid ){
		return 0;
	}
	ibp->prev_in_ht = in_ht;
	ibp->prev_in_wid = in_wid;
	ibp->prev_win_ht = win_ht;
	ibp->prev_win_wid = win_wid;
	ibp->prev_str_ht = stride_ht;
	ibp->prev_str_wid = stride_wid;

	// find the output dims and padding

	int32_t left_pad, right_pad, top_pad, bottom_pad;

	int32_t out_wid = nn_pad_compute_outsize_and_pad( in_wid, win_wid, stride_wid, self->padding, & left_pad, &right_pad);
	int32_t out_ht = nn_pad_compute_outsize_and_pad( in_ht, win_ht, stride_ht, self->padding, & top_pad, &bottom_pad);

	if( out_wid < 1 || out_ht < 1) return errlog(nn,"input too small for filter");

	// Checks to simplify the operation.
	// The order of these is important.
	//  Example: win = 5, stride = 3  in_wid = 7 => out=1, lpad = rpad = 0
	//    (1) trim input to 5 wide
	//    (2) change stride to 1, leave win = 5

	// can we trim the input ? may be possible if right_pad = 0 or bot_pad = 0, and strides >= 2
	if( bottom_pad == 0 && stride_ht > 1) {
		in_ht =  min_i32(in_ht, win_ht + (out_ht-1)*stride_ht);		// may be less than before
	}
	if( right_pad == 0 && stride_wid > 1) {
		in_wid =  min_i32( in_wid,win_wid + (out_wid-1)*stride_wid);	// may be less than before
	}
	// if an output dimension is 1, we can trim the window to match the input; also set stride to 1
	// otherwise, find out the # of 'infeasible' padding rows...
	// This is only >0 when the window size is > input_size + 1.
	//
	int infeas_pad_ht=0, infeas_pad_wid = 0;

	if( out_ht == 1 ){
		win_ht = in_ht;
		top_pad = bottom_pad = 0;
		stride_ht = 1;
	}else{
		int infeas = win_ht - 1 - in_ht;
		if( infeas >0) {
			infeas_pad_ht = infeas;
			// we can't have infeas+1 > win/2, otherwise we'll need scales > 1.0
			// to do the right/bottom padding. That happens when the window is > 2*input_len;
			// in such cases all the outputs are the same. these may be simplified by setting
			// win = inlen*2-1, or inlen*2 (whichever matches its original parity)
			// and reducing the padding evenly.
			//
			int shrink = win_ht - max_i32( top_pad, bottom_pad) - in_ht;
			// 'shrink' is the amount by which window can be reduced on both
			// sides without changing the result.
			if( shrink > 0){
				win_ht -= 2*shrink;
				infeas_pad_ht -= 2*shrink;
				top_pad -= shrink;
				bottom_pad -= shrink;
			}
		}
	}
	if( out_wid == 1 ){
		win_wid = in_wid;
		left_pad = right_pad = 0;
		stride_wid = 1;
	}else{
		int infeas = win_wid - 1 - in_wid;
		if( infeas >0) {
			infeas_pad_wid = infeas;
			int shrink = win_wid - max_i32( left_pad, right_pad) - in_wid;
			if( shrink > 0){
				win_wid -= 2*shrink;
				infeas_pad_wid -= 2*shrink;
				left_pad -= shrink;
				right_pad -= shrink;
			}
		}
	}

	ibp->inshape.height = in_ht;
	ibp->inshape.width = in_wid;
	ibp->outshape.height = out_ht;
	ibp->outshape.width = out_wid;
	ibp->window_ht = win_ht;
	ibp->window_wid = win_wid;
	ibp->stride_ht = stride_ht;
	ibp->stride_wid = stride_wid;
	ibp->infeas_pad_ht = infeas_pad_ht;
	ibp->infeas_pad_wid = infeas_pad_wid;
	ibp->wpad_top = top_pad;
	ibp->wpad_bottom = bottom_pad;
	ibp->wpad_left = left_pad;
	ibp->wpad_right = right_pad;
	ibp->wpad_flags = ((top_pad > 0) ? intbuf_PAD_T : 0)
			       |((bottom_pad > 0) ? intbuf_PAD_B : 0)
			       |(( left_pad > 0) ? intbuf_PAD_L : 0)
			       |(( right_pad > 0) ? intbuf_PAD_R : 0);

	//
	// scale factors for left/right padding, if any
	//
	if ( ibp->wpad_flags != 0 ) {
		// figure out how many we need.

		int npad_h = max_i32( top_pad, bottom_pad );
		int npad_w = max_i32( left_pad, right_pad );
		int tpad = npad_h + npad_w;
		int16_t * scales_h;
		if( tpad <= (int)( sizeof(ibp->edgepad_scale_storage)/sizeof(ibp->edgepad_scale_storage[0]))){
			// use the storage there
			scales_h = &ibp->edgepad_scale_storage[0];
		}else{
			void * p  = ibp->edgepad_scale_alloc;
			if( tpad > ibp->edgepad_scale_size ){	// need to allocate...
				void * newbuf = nn_realloc( p, tpad *sizeof(int16_t));
				if( newbuf == NULL ) return errlog(nn,"can't alloc for %d x i16", tpad );
				ibp->edgepad_scale_alloc = newbuf;
				ibp->edgepad_scale_size = tpad;
				p = newbuf;
			}
			scales_h = (int16_t *)p;
		}
		ibp->edgepad_scales_h = scales_h;
		ibp->edgepad_scales_w = scales_h + npad_h;
		// now fill in the scales
		if( npad_h > 0 ){
			integ_buf_find_padding_scales( scales_h, npad_h, ibp->window_ht, ibp->infeas_pad_ht);
		}
		if( npad_w > 0 ){
			integ_buf_find_padding_scales( scales_h + npad_h, npad_w, ibp->window_wid, ibp->infeas_pad_wid);
		}
	}

	// design the integral buffer.
	int ibuf_lpad = max_i32( 3, left_pad);
	int ibuf_rpad = max_i32( 3, right_pad);

	// figure out how many input rows need to be loaded first; to
	// have enough context to generate the first 4 output rows (assuming stride_h=1)
	// and also to have enough rows to generate the top-padding rows (if any).
	// This is the 'initial row load' and it will be rounded up to an even #, and
	// then limited to be not more than in_ht.
	//
	int initial_row_load = win_ht + 3 - top_pad; 	// needed for 4 outputs
	if( top_pad >= 1){
		// ensure enough to generate padding
		initial_row_load = max_i32( win_ht-1, initial_row_load);
	}
	initial_row_load = (initial_row_load+1)&~1;		// round up to even

	ibp->ibuf_initial_load = min_i32(in_ht, initial_row_load);
	// (intentionally keeping the unclipped, even, value for below)

	// How many rows in the buffer? our constraints are:
	// (1) we load input rows in pairs, and the two loaded rows can't be 'wrapped' across.
	//    so the height must be even.
	// (2) The top padding zone must be an odd number (so that buffer rows 1,2, the first rows loaded,
	// 		are in an even,odd pair).
	// (3) the height must be >=win_ht+4 in order to allow 4 rows to be loaded and then
	//    4 outputs to extracted (assuming stride_h = 1)
	// (4) height must also be enough to support the 'initial row load', following the top
	//    padding and 'row 0'
	//
	// We also consider the full height of the integral buffer.
	// If the rolled height, based on the above, is within 'win_h' of the full
	// height, then the buffer is unrolled and requirements (1),(2) are then waived, since
	// no 'wrapped' loads will occur. Requirement (4) becomes moot since we have enough
	// buffer to load the whole input.

	// ibuf_fullrows is how large it would be if fully unrolled; don't exceed that
	//
	int ibuf_fullrows = top_pad + 1 + in_ht + bottom_pad;
	int ibuf_top_pad = top_pad | 1;		// round up padding to odd #
	int ibuf_rows = (win_ht + 4+1) & ~1;	// buffer height, rounded up to even #

	// it must be enough to support the initial row load (an even #)
	// (when initial_row_load > in_ht, this is overcharging; but in that
	// case we're likely going to be unrolled anyway, so it will make no
	// difference).

	ibuf_rows = max_i32( ibuf_rows, ibuf_top_pad+1+initial_row_load);

	ibp->ibuf_is_rolled = 1;
	// is unrolled? if it's close, just force it to be unrolled
	if( ibuf_rows + win_ht >= ibuf_fullrows){
		ibuf_rows = ibuf_fullrows;
		ibuf_top_pad = top_pad;			// don't need this to be odd.
		ibp->ibuf_is_rolled = 0;
	}

	int ibuf_row_bytes = 128*(ibuf_lpad + 1+ in_wid + ibuf_rpad);
	ibp->ibuf_proper_row_bytes = 128*(left_pad+ 1+ in_wid + right_pad);
	ibp->ibuf_row_bytes = ibuf_row_bytes;
	ibp->ibuf_rows = ibuf_rows;
	ibp->ibuf_row0 = ibuf_top_pad;
	ibp->ibuf_col0_offs = 128*ibuf_lpad;
	ibp->ibuf_proper_offs = 128*(ibuf_lpad-left_pad);
	ibp->ibuf_total_bytes = ibuf_row_bytes * ibuf_rows;

	// scaling
	int winsize = win_ht * win_wid;
	// we want the ceiling of the log2 of that...
	int rsh = ceiling_log2(winsize) + 1;		// rsh for avgpool
	// the mantissa will be 2^(13+rsh), divided by winsize, rounded to nearest
	int frac = (unsigned)( 16384 << rsh)/ (unsigned)winsize;

	frac = (frac+1)>>1;	// guaranteed >= 16384, < 32768  for all winsize <= 65536
	ibp->initial_recip_mant = frac;
	ibp->initial_recip_shift = rsh;
	ibp->recip_mant = frac;
	ibp->recip_shift = rsh;

	// force these to be reconsidered.
	ibp->hvx_specialized_handler = NULL;
	ibp->hvx_specialized_handler_code = 0;
	return 1;
}


// find padding scales for pi = 1 .. n:
//     the value is   -p/(w-p) where  p = max( pi,infeas+1)
//   and is represented with 15 frac bits.
static void
integ_buf_find_padding_scales( int16_t *outp, int n, int winsize, int infeas )
{
	for( int pi = 1; pi <= n; pi++ ){
		int p = max_i32( pi, infeas+1);
		int wmp= winsize-p;
		int16_t scaleval;
		if( wmp <= p ){
			scaleval = -0x8000;		// -1.0
		}else{
			uint32_t recip = lut_reciprocal_i32[wmp-1];	// 1/(w-p) with 31 fractional bits
			recip *= p;		// may now exceed (1<<31) by a bit due to rounding
			scaleval = -(int32_t)( (recip + 0x8000)>>16);
		}
		outp[pi-1] = scaleval;
	}
}
