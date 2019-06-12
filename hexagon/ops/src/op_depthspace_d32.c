/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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


#include <nn_graph.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#include "hvx_inlines.h"


#ifdef V66
#define DEPTHTOSPACE_MAX_THREADS 4
#else
#define DEPTHTOSPACE_MAX_THREADS 2
#endif

//
// DepthToSpace_8_d32
//
// This is given a blocksize_h, blocksize_w (usually they are the same, but they don't need to be)
//
// And it works as follows:
//    - Input depth must factor as blocksize_h*blocksize_w * output_depth
//    - Output height is blocksize_h * input_height  (possibly reduced by clipping)
//    - Output width is blocksize_w * input_width	 (possibly reduced by clipping)
//
// So the operation is as follows (if done in 'flat'):
//     [ b, h_in, w_in, bsH*bsW*d_out]		-> input
//     [ b, h_in, w_in, bsH, bsW, d_out ]	-> relabel depth dim to 3 dims
//     [ b, h_in, bsH, w_in, bsW, d_out ]	-> transpose two dims
//     [ b, h_in*bsH, w_in*bsW,  d_out ]	-> combine dims
//     [ b, h_out, w_out, d_out] 			-> trim h,w if cropping.
//
// BLOCKSIZE:
//   input #1 is a scalar int which gives the blocksize. if it is two ints it is [bsH, bsW].
//
// CROPPING:
//    - an optional input givens [ top_crop, bot_crop, left_crop, right_crop ], reducing output size.
//    - The cropping amounts must be >= 0 and < the block size in the relevant direction
//    - Optionally this input can have a 5th entry; if present (and nonzero) it is the requested 'cropped' output depth,
//      which must be <= the actual output depth as calculated from the input depth and blocksize; furthermore
//      the difference between the two depths cannot exceed 31. If this value is 0 it is ignored.
//     (this is input #4, after in the in_min and in_max inputs).
//
// CONSTRAINTS:
//   The d32 version of depth2space has the following constraints:
//      (1) 'blocksize' input ( and 'crop' input, if present) must be constant.
//      (2) blocksize_w must be one of : 1,2,3,4,8
//      (3) the calculated output depth  in_depth/(blocksize_w * blocksize_h) must be a multiple of 32.
//          OR it may be 16 if the blocksize_w = 2.
//
//-----------------------------------------------------------
//
// BatchToSpace_8_d32
//
// This is given a blocksize_h, blocksize_w (usually they are the same, but they don't need to be)
//
// And it works as follows:
//    - Input batches must factor as out_batches * blocksize_h * blocksize_w
//    - Output height is blocksize_h * input_height  (possibly reduced by clipping)
//    - Output width is blocksize_w * input_width	 (possibly reduced by clipping)
//
// So the operation is as follows (if done in 'flat'):
//     [ bs_H*bs_W*b_out, h_in, w_in, dep] -> input
//     [ bs_H, bs_W, b_out, h_in, w_in, dep] -> relabel batch dim to 3 dims
//     [ b_out, h_in, bs_H, w_in, bs_W, dep] -> transpose five dims
//     [ b_out, h_in*bs_H, w_in*bs_W, dep]	-> combine dims
//     [ b_out, h_out, w_out, dep] 			-> trim h,w if cropping.
//
// BLOCKSIZE:
//   input #1 is a array of scalars [ bsH, bsW ] which gives the blocksize.
//      If it is one int it's [ bsH ] and bsW is =1.
// CROPPING:
//      input #2 is [1,1,2,2]  containing [ [top_crop,bot_crop],[left_crop,right_crop]]
//       (it must be  [1,1,1,2] if the blocksize input is just one int.
// CONSTRAINTS:
//   The d32 version of batch2space has the following constraints:
//      (1) 'blocksize' input ( and 'crop' input, if present) must be constant.
//      (2) blocksize_w must be one of : 1,2,3,4
//       There is no constraint on depth.
//-----------------------------------------------------------
//
// SpaceToBatch_8_d32
//
// This is given a blocksize_h, blocksize_w (usually they are the same, but they don't need to be)
//
// And it works as follows:
//    - output batches will blocksize_h*blocksize_w*vbatches_in;
//    - output height will be (pad_top + input_height + pad_bottom)/blocksize_h  (must divide exactly)
//    - output width will be (pad_left + input_width + pad_right)/blocksize_w  (must divide exactly)
//    - depth is unchanged
//
// So the operation is as follows (if done in 'flat'):
//     [ b_in,  h_out*bs_H,  w_out*bs_W,  dep] -> input with padding added
//     [ b_in,  h_out, bs_H,  w_out, bs_W,  dep] -> relabel to 5 dims
//     [ bs_H,bs_W, b_in, h_out,  w_out,  dep  ] -> transpose five dims
//     [ bs_H*bs_W,*b_in, h_out,  w_out,  dep  ]	-> combine dims
//
// BLOCKSIZE:
//   input #1 is a array of scalars [ bsH, bsW ] which gives the blocksize.
//      If it is one int it's [ bsH ] and bsW is =1.
// CROPPING:
//      input #2 is [1,1,2,2]  containing [ [top_crop,bot_crop],[left_crop,right_crop]]
//       (it must be  [1,1,1,2] if the blocksize input is just one int.
// CONSTRAINTS:
//   The d32 version of space2batch has the following constraints:
//      (1) 'blocksize' input ( and 'crop' input, if present) must be constant.
//      (2) blocksize_w must be one of : 1,2,3,4
//       There is no constraint on depth.
//-----------------------------------------------------------
//
#define MAX_HEIGHT_EXTENTS 4				// max # height units to split into.
struct x2s_d32_info;
typedef void (*width_interleave_loop_funcp)( struct x2s_d32_info const *, uint8_t *,uint8_t const *, int );
//
// 'persistent info' struct. We cache the previous input shape
// to check if the plan can be reused on later runs (reuse also requires that the 'prev_in_wpad0'
// is the same as before).

struct bsz_crop {
	uint16_t blocksize_h, blocksize_w;
	uint16_t crop_top, crop_bottom;			// requested cropping
	uint16_t crop_left, crop_right;
};

struct h_extent_desc {		// describes a height extent
	int hcount;				// # of output rows (may be zero in extent #3, if only 3 in use)
	uint8_t const *in_ptr;	// input ptr		// needs adjust for batch, if >1 batch
	uint8_t      *out_ptr;	// output ptr		// "
};

struct x2s_d32_info {
	// these fields are set up in 'check' method
	uint8_t elbytes;						// size of element, 1 or 2
	uint8_t is_b2s;							// is b2s, not d2s
	uint8_t dtype;							// NN_TYPE_QUINT8 etc
	uint8_t depth32_chunk;					// = 32 for 8-bit processing, = 64 for 16-bit processing.

	struct shape input_shape;				// copy, for validating plan
	struct tensor_format input_format;
	void * input_data;
	struct tensor_addressing tin;		    //' addressing' of the input d32 tensors
	struct tensor_addressing tout;		    //' addressing' of the output d32 tensor

	uint32_t in_total_stride;				// in_batch_stride * in_batches
	struct bsz_crop bsz;
	uint16_t req_out_depth;					// requested out depth (or 0 if not)

	uint16_t out_depth;						// calculated output depth (prior to reduction).
	struct shape output_shape;				// calculated output shape

	// parameters controlling processing of each 'width unit'
	int32_t wunit_left_adjust;				// # of bytes to offset source pointer to get vector-aligned start pointer.
	uint16_t wunit_vlalign;					// vlalign needed at output: 0,32,64 or 96
	uint16_t wunit_vecdrop;					// # of output vectors to drop at start of output row (0..blocksize_w)
	uint16_t wunit_vecout;					// # of vectors stored (w_out+3)/4
	uint16_t wunit_vloops;					// # of 'full vector loops' to run after initial run (>=0)
	int16_t wunit_v_at_end;					// # if vecs to store in 'final loop' 0..blocksize_w-1
	                                        // a value of -1 is a special case: it means we need to align previous & store 1.
											// (no extra read is needed). only occurs when wunit_vecalign is != 0.
	uint16_t wunit_prefrow;					// # of bytes to prefetch in input row
	uint16_t wunit_prefht;					// # of d32 units in the input to a wunit
	int32_t discard_w;

	int32_t input_wunit_span;				// this is in_d32_stride * nd32_out: 'span' of inputs to wunit
	int32_t input_wunit_stride;				// in_d32_stride * nd32_out * bsW : total size of a wunit in input
										// for the case of d=16 out, bsW=2, stride is same as span.
	uint8_t const * input_base;			// base address, adjusted for left crop & padding (vec aligned)
	uint8_t const * input_end_addr;		// base address + batch*stride * batches (used to wrap pointers for b2s)
	int32_t b2s_wrap_offset;			// subtract this to 'wrap'
	width_interleave_loop_funcp width_interleave_loop_fp;

	// how is the work split into h extents...
	struct h_extent_desc hextents[MAX_HEIGHT_EXTENTS];
	int n_extents;						// # of extents actually used (a power of 2:  1,2, .. MAX_HEIGHT_EXTENTS)
	int n_workunits;					// # of workunits (out_batches * n_extents)
	uint32_t intermed_bufrows_per_thread;
	uint32_t intermed_buffer_stride; //How to move from one width to the next in the intermediate buffer

};

struct x2s_d32_runstate {
	struct x2s_d32_info const * info;
	// this is the current work unit
	volatile int cur_work_unit;
	volatile int thrdindex;
	uint8_t * intermed_buffer;			//For stride > 4, interleave in smaller groups
	uint32_t intermed_buffer_size_per_thread; //How much scratch space to we need for each thread
	nn_sem_t done_sem;					// for signalling hvx threads are done.
};
static int get_b2s_dims( struct nn_node *self, struct nn_graph * nn , struct bsz_crop * dst);
static int get_d2s_dims( struct nn_node *self, struct nn_graph * nn , struct bsz_crop * dst, int *dcrop_p);

static inline void width_interleave_1_loop( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t const * inptr, int height_count );
static inline void width_interleave_1_loop_alt( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t const * inptr, int height_count );
static inline void width_interleave_2_loop( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t const * inptr, int height_count );
static inline void width_interleave_3_loop( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t const * inptr, int height_count );
static inline void width_interleave_4_loop( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t const * inptr, int height_count );
static inline void width_interleave_2_to16_loop( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t const * inptr, int height_count );
static inline void width_interleave_8_loop( struct x2s_d32_info const * info,
		uint8_t * out, uint8_t * intermed_buffer, uint8_t const * inptr, int height_count );

static const width_interleave_loop_funcp loop_funcp_table[4] = {
		width_interleave_1_loop,
		width_interleave_2_loop,
		width_interleave_3_loop,
		width_interleave_4_loop
};

static void x2s_d32_workfunc( struct nn_graph * nn, void * rstpv);

// strategy:
// The h-dimension and w-dimension operations are distinct.
//
//  W-dimension:
//    - note that nd32in = bsH * bsW * nd32out
//
//    Each input row consists of  bsH * bsW * nd32out * [ inwidth_total*32]
//    Consider a single w_unit of   bsW * nd32out * [inwidth_total*32]
//    this needs to be converted to   nd32out * [ outwidth_total * 32 ]
//    .. where outwidth_total is approximately  bsW * inwidth_total.
//    This is is done as nd32out passes, each of which reads bsW rows, interleaves them in units
//    of 32, and stores them to the output row. Dealing with left/right cropping, and left-input padding
//    is described below.
//
//  H dimension:
//    This is easy: the input consists of
//         in_height * bsH  "w_unit"s, each being (bsW*nd32out * [inwidth_total*32] as described above;
//    We want the w-processed versions of these to appear in exactly the same order in the output. If top or
//    bottom cropping is applied, we simply skip a given # of w_units at the top or bottom.
//    So the total # of 'w_units' to be processed per batch is equal to the output height.
//   (for batch2space, it's more complex; the same code can be used but the input addressing is different;
//    see note below re input_wunit_span, input_wunit_stride).
//
//
// Width processing:
//   Depending on blocksize_w = (1,2,3,4), we have four different inner loop operations; each of which repeatedly
//   reads bsW vectors (each from separate depth slices in the input), interleaves them, and writes bsW vectors to a single
//   output row. However, we need to discard a certain # of initial width units:
//     - each input row can have 0..3 padding units at the start, meaning 0 .. 3*bsW units need to be discarded
//     - the configuration may additionally require 0..bsW-1 units to be discarded from the output.
//      So, the w-loop needs to discard 0..4*bsW-1 width units from the output of the first iteration.
//      This is done as follows:
//      - if discard is not a multiple of 4, a nonzero vlalign operation is applied at the output (which actually inserts
//        1..3 garbage at the start of row);
//      - discard as many full vectors as needed (this will be 0..bsW)
//     So, the first iteration in the row will not always store bsW vectors. In some cases it will store 0;
//     it will only serve to generate a 'previous' vector for the next operation's vlalign. We also need
//     to consider that the first iteration  may need to drop vectors on the *right* (if the output is narrower than blockW vectors).
//    Each pass across consists of:
//      (1) read one vector each of from bsW separate rows; interleave them to bsW vectors v[0],v[1].. v[bsW-1]
//              use vlalign( v[i], v[i-1], wunit_valign) to get bsW 'aligned' vectors (where 'zero'  is used as v[-1])
//              discard the first 'wunit_vecdrop' of these and store up to wunit_vecout.
//              prev = v[bsW-1]
//      (2) wunit_vloops times (may be 0):
//               -one vector each of from bsW separate rows; interleave them to bsW vectors v[0],v[1].... v[bsW-1]
//               -use vlalign( v[i], v[i-1], wunit_valign) to get bsW 'aligned' vectors (where 'prev' is used as v[-1])
//               - store all of these to output
//               -prev = v[bsW-1]
//      (3) if wunit_v_at_end != 0:
//              - if wunit_v_at_end > 0:
//                     one vector each of from bsW separate rows; interleave them to bsW vectors v[0],v[1].... v[bsW-1]
//              - store valign( v[0], prev, wunit_valign) to output
//              - for k = 1, k < wunit_v_at_end:
//                     store valign( v[k], v[k-1], wunit_valign) to output
//   wunit_v_at_end is the number of 'extra' vectors to store at the end to complete; and is in range 0 .. bsW-1. A value
//   of -1 may be used to indicate that 1 extra output vector is needed, but all of the value in that extra vector are in 'prev', so
//   no interleave operation needs to be done. I.e. all required inputs have already been read.
//
///////////
// The same code does batch2space. In both cases, each output d32 slice is obtained by interleaving bsW
// input d32 slices, in chunks of 32. Each group of nd32_out such slices is an output row
//  and the source data for these is are nd32_out*bsW input slices.
//  (for the in_depth=16 case, the source is one input slice, but we unpack to one of double width).
//
//	However, for d2s:
//        - those slices are always adjacent in the input (a 'wunit') and bsH of them form an input row;
//        for b2s:
//        - the slices are scattered in the input; 'nd32' adjacent slices with spacing of batch_stride.
//
//  in both cases
//       input_wunit_span is the delta between each of the bsH slices that need to be interleaved;
//  	 input_wunit_stride  is the delta to move input pointers, to make the next output slice
//
//    d2s:
//			input_wunit_span =  tin.d32_stride * tout.nd32
//          input_wunit_stride =  tin.d32_stride * tout.nd32 * bsW  [ or *1 in special d_out=16 case]
//    b2s:
//			input_wunit_span =  tin.batch_stride *out_batches
//          input_wunit_stride = tin.batch_sride * out_batches* bsW
// An output row is generated by doing the interleave nd32_out times, advancing both by their d32 strides;
// for d2s, for the next output row, we advance the input ptr by input_wunit_stride, output by out_row_stride
// For b2s it's a bit more complex - we advance the input by input_wunit_stride, and if it falls off the
// end of the world (precalculated input_end_addr) we subtract the length of the world and then advance one row.
//
//
// For 16-bit data:
//  The process is almost the same. The inner loops each read bsW vectors from different locations
//  in the input; then interleave the contents, and write to one output row (with various edge adjustments).
//  The only difference is in how the interleaving is done (which is mostly just a question of giving
//  different values to the vshuff instructions.
// In order to do the strategy calculations for 16-bit, we double the 'w' dimensions and w padding,
// going into those calculations,and mostly everything else is the same
//
//
// setup plan for d2s (or b2s)
static int
setup_x2s_plan( struct nn_node *self, struct nn_graph * nn )
{
	struct x2s_d32_info * info = (struct x2s_d32_info *)self->opaque;
	//int elbytes = 1;

	struct tensor const *input_tensor = self->inputs[0];
	if( shape_matches( &input_tensor->shape , &  info->input_shape)
			&& format_matches( &input_tensor->format, &info->input_format)
			&& input_tensor->data == info->input_data ){
		// same plan still works.
		return 0;
	}
	info->input_shape = input_tensor->shape;
	info->input_format = input_tensor->format;
	info->input_data = input_tensor->data;

	int elbytes = info->elbytes;
	info->tin = (elbytes==1 ?nn_tensor_addressing_d32:nn_tensor_addressing_d32_16b)( input_tensor );

	int is_b2s = info->is_b2s;

	int req_out_depth=0;
	if( !is_b2s){
		int res = get_d2s_dims( self, nn, &info->bsz, & req_out_depth);
		if( res <0) return res;
	}else{
		int res = get_b2s_dims( self, nn, &info->bsz);
		if( res <0) return res;
	}
	// validate things against the actual shape...
	int32_t bsH = info->bsz.blocksize_h;
	int32_t bsW = info->bsz.blocksize_w;
	uint32_t bsProd = mulu32_sat( bsH, bsW);


	uint32_t out_depth, out_batches;
	if( !is_b2s){
		out_batches = info->input_shape.batches;
		// input depth must be divisible by bsProd.
		out_depth = info->input_shape.depth / bsProd;
		if( out_depth < 1 || out_depth* bsProd != info->input_shape.depth) return errlog(nn, "block shape does not divide input depth");

		// restrictions from the implementation:
		//   - output depth must be a multiple of 32 (16 allowed when bsw=2)
		//   - bsW must be 1,2,3 or 4.
		if( out_depth%32 != 0 && (out_depth!=16 || bsW !=2) ){
			return errlog(nn,"limitation: output depth %u not valid (must be a multiple of 32 (or exactly 16, for bsw=2)", (unsigned)out_depth);
		}
	}else{
		// input batches must be divisible by bsProd
		out_depth = info->input_shape.depth;
		out_batches = info->input_shape.batches / bsProd;
		if( out_batches < 1 || out_batches* bsProd != info->input_shape.batches){
			return errlog(nn, "block shape does not divide input batches");
		}
	}
	if( bsW >4 && bsW != 8 ) return errlog(nn,"limitation: blocksize <= 4 or blocksize = 8 in width direction");
	info->out_depth = out_depth;

	// work out the output height/width ( subject to adjustment for cropping)
	int32_t out_height = info->input_shape.height * bsH;
	int32_t out_width = info->input_shape.width * bsW;

	if( req_out_depth > 0){
		if( req_out_depth > out_depth
			|| req_out_depth+ 31 < out_depth){
			return errlog(nn, "can't crop output depth %u to %d", (unsigned)out_depth, req_out_depth);
		}
	}
	info->req_out_depth = req_out_depth;		// save for later.

	// reduce output size
	out_height -= (info->bsz.crop_top + info->bsz.crop_bottom);
	out_width -= (info->bsz.crop_left + info->bsz.crop_right);
	if( out_height < 1 || out_width < 1 ){
		return errlog(nn,"cropping leaves no output");
	}
	// set the actual output shape.

	info->output_shape.batches = out_batches;
	info->output_shape.height = out_height;
	info->output_shape.width = out_width;
	info->output_shape.depth = (info->req_out_depth!=0)? info->req_out_depth : out_depth;

	// now... plan the wunit processing.

	info->wunit_vecout = (elbytes*out_width+3)>>2;		// vectors we need to store.
	//
	// how many values to discard at the front of each row?
	// Deal with crop_left >= bsW by effectively trimming input pixels - add them
	// to the left padding, and if the start point crosses to a new vector, we can end
	// up doing less work on each row.
	//
	int wpad_mod4 = (elbytes*info->input_format.width_pad[0])&3;
	int input_width_needed = info->input_shape.width;		// subject to trimming...
	int wunit_left_adjust = -32*wpad_mod4;		// initial offset to align input pointer.

	int crop_left_eff = info->bsz.crop_left;
	if( crop_left_eff>= bsW){	//'over-cropped' input
		// reduce this to crop_left/bsW input pixels trimmed, then output crop of crop_left%bsW
		int n = crop_left_eff/(unsigned)bsW;		// skip this many w units of input
		crop_left_eff -= n*bsW;						// now 0..bsW-1
		wpad_mod4 += elbytes*n;								// add to input padding - may be >= 4 now
		int left_trim_vectors = wpad_mod4/4u;		// remove any multiple of 4, account as vector trim
		wpad_mod4 &= 3;
		wunit_left_adjust += 128*left_trim_vectors;
		input_width_needed -= n;					// trimmed pixels not accounted as input
		// note: if the above does *not* change  wunit_left adjust, the changes it does make, i.e.
		// wpad_mod4 += n, crop_left_eff -= n*bsW, and input_width_needed -= n,
		// should all wash out to exactly the same result in the calculations below.
	}
	// likewise if we are cropping more than bsW on the right, it reduces the number
	// of input pixels needed, which can affect the accounting here.
	if( info->bsz.crop_right >= bsW){
		input_width_needed -= info->bsz.crop_right/(unsigned)bsW;
	}
	// note: wpad_mod4, wunit_left_adjust have been scaled by 'elbytes';
	// crop_left_eff, input_width_needed have not been scaled.
	//

	// this is the # of initial outputs to discard, based on crop_left and the intra-vector input
	// left padding, both of which may have been adjusted for crop_left >= bsW
	// This must be < bsW*4
	int discard_w = bsW * wpad_mod4 + elbytes*crop_left_eff;		// 0 .. bsW*4-1
	info->discard_w = discard_w;

	int w_read = 4-wpad_mod4;								// w units read in first iter

	int vlalign_ctl = 0;
	int discard_vecs = discard_w >> 2;		// 0 .. bsW-1
	// if discard_w is not a multiple of 4, use vlalign_ctl to insert garbage, and add 1
	//
	int k = discard_w & 3;
	if( k != 0){
		vlalign_ctl = 32*(4-k);		// insert (4-k) units of garbage
		discard_vecs++;				// 0 .. bsW now
	}
	info->wunit_left_adjust = wunit_left_adjust;
	info->wunit_vlalign = vlalign_ctl;
	info->wunit_vecdrop = discard_vecs;
	// # of output vectors remaining to account for.
	int vecs_remain = max_i32(info->wunit_vecout+discard_vecs-bsW, 0 );
	unsigned vloops = (unsigned)vecs_remain/bsW;			// # of full loops
	vecs_remain -= bsW*vloops;					// remainder is 0..bsW-1
	w_read += vloops*4;							// # of width units we've read so far.

	// is there any remainder? if only one output vector remains, it's possible
	// we already have all the data in the 'overlap'.
	if( vecs_remain == 1 && w_read >= elbytes*input_width_needed ){
		vecs_remain = -1;
	}
	info->wunit_vloops = vloops;
	info->wunit_v_at_end = vecs_remain;

	logmsg(nn,3, "input shape: %d %d %d %d; bs %dx%d; crop (%d,%d) (%d,%d); depth req %d  output shape %d %d %d %d",
			(int)info->input_shape.batches, (int)info->input_shape.height,(int)info->input_shape.width, (int)info->input_shape.depth,
			(int)bsH,(int)bsW, info->bsz.crop_top, info->bsz.crop_bottom, info->bsz.crop_left, info->bsz.crop_right, info->req_out_depth,
			(int)info->output_shape.batches, (int)info->output_shape.height,(int)info->output_shape.width, (int)info->output_shape.depth);
	logmsg(nn,3, "wunit_vecout =%d vlalign =%d _vecdrop = %d _vloops = %d _v_at_end = %d _prefrow= %d _prefht = %d",
			info->wunit_vecout, info->wunit_vlalign, info->wunit_vecdrop, info->wunit_vloops, info->wunit_v_at_end,
			info->wunit_prefrow, info->wunit_prefht);
	{
		struct tensor * output_tensor = self->outputs[0];
		int h_pad = 4;
		int w_pad0 = 4;
		if( info->output_shape.height==1) h_pad = 1;
		int w_pad1 = (~info->output_shape.width+1)&3;
		int d_pad1 = (~info->output_shape.depth+1)&31;

		int dtype_out = info->dtype;
		// for 16-bit op: if the input is QINT then so is the output.
		if(dtype_out == NN_TYPE_QUINT8 && input_tensor->format.type == NN_TYPE_QINT8){
			dtype_out = NN_TYPE_QINT8;
		}

		if( tensor_out_prepare_padded_d32( output_tensor, info->output_shape.batches,
				info->output_shape.height, h_pad, h_pad,
				info->output_shape.width, w_pad0, w_pad1,
				info->output_shape.depth, 0, d_pad1 ,dtype_out )!= 0 ){
			return errlog(nn,"failed to prepare output");
		}
		info->tout = (elbytes==1 ?nn_tensor_addressing_d32:nn_tensor_addressing_d32_16b) ( output_tensor );
	}
	// amount to prefetch on each row.
	info->wunit_prefrow = (32*(wpad_mod4 + info->input_shape.width+ 3))& ~127;
	// # of d32 rows in one wunit (for prefetch)
	if( !is_b2s){
		info->wunit_prefht = ((out_depth==16)?1:bsW) * info->tout.nd32;
		// wunit_span is the 'inner span' for a wunit
		info->input_wunit_span = info->tin.d32_stride * info->tout.nd32;
		// wunit_stride is the 'size' of a wunit in the input (amount
		// to advance for next output row)
		info->input_wunit_stride = info->input_wunit_span * bsW;
	}else{
		// alas, can only prefetch one input seg of bsW
		info->wunit_prefht =  info->tout.nd32;
		// here we are interleaving across batches
		info->input_wunit_span = info->tin.batch_stride*out_batches;
		info->input_wunit_stride = info->tin.batch_stride*out_batches*bsW;
	}
	// select the width interleave loop function
	//
	if( !is_b2s && info->out_depth == 16  && bsW == 2 ){
		info->width_interleave_loop_fp = width_interleave_2_to16_loop;
		// span is the same as stride since we are unpacking one d32 slice to one d32 slice here.
		info->input_wunit_stride = info->input_wunit_span;
	}else{
		if( (!is_b2s && info->tout.nd32*32 != info->out_depth) || bsW < 1 || (bsW > 4 && bsW != 8) )
			return errlog(nn,"assumption failed!");
		if (bsW <= 4)
		{
			info->width_interleave_loop_fp = loop_funcp_table[bsW-1];
		}
	}

	{	// set some things up for overall addressing...
		int32_t length_of_all = info->tin.batch_stride * info->input_shape.batches;
		info->in_total_stride = length_of_all;
		info->input_base = info->tin.data + info->wunit_left_adjust;
		info->input_end_addr = info->input_base + length_of_all;
		info->b2s_wrap_offset = length_of_all - info->tin.height_stride;
	}
	// generate the height extents.
#if MAX_HEIGHT_EXTENTS!=4
#error "this needs review"
#endif
	{
		int n_extent;
		unsigned out_row_size = info->tout.height_stride;
		unsigned out_batch_size = info->tout.batch_stride;
		if( out_height >= 8 && out_row_size >= 16384){	// split into 4.
			n_extent = 4;
		}else if( out_height >= 4 && out_batch_size >= 16384){
			n_extent = 2;
			// don't create an odd height if it's the only batch and we have 2 threads
			if( out_height >= 6 && (out_batches>1 || DEPTHTOSPACE_MAX_THREADS==4))
				n_extent = 3;
		}else if( out_batches==1 && out_height >= 2 && out_batch_size >= 4096){
			n_extent = 2;		// split a single small batch if it's not really small.
		}else{
			n_extent = 1;
		}
		// allocate the heights
		{
			info->hextents[3].hcount = 0;	// in case it's 3 we need this
			int h_remain = out_height;
			for( int  i = 0; i < n_extent-1; i++){
				// divide what's left by (n_extent-i), rounding up...
				int hi = ( h_remain + (n_extent-1-i))/(unsigned)(n_extent-i);
				info->hextents[i].hcount = hi;
				h_remain -= hi;
			}
			info->hextents[n_extent-1].hcount = h_remain;
		}
		// find the pointers in each range.
		// These are the input and output pointers, without adjustment for 'batch' offset.
		unsigned max_height_per = 0;
		{
			int hpos_out = 0;
			int hpos_in = info->bsz.crop_top;	// current 'input' pos
			for( int i =0; i < n_extent; i++){
				uint8_t const * iptr = info->input_base;		// base of all
				int row = hpos_in;						// # of rows to bump from start
				if( is_b2s && row >= bsH){				// need modular adjust
					unsigned qrows = row/(unsigned)bsH;	// actual input rows to move..
					iptr += qrows *  info->tin.height_stride;
					row -= qrows * bsH;					// 'row' is now hpos_in % bsH
				}
				iptr += row * info->input_wunit_stride;
				info->hextents[i].in_ptr = iptr;
				info->hextents[i].out_ptr = info->tout.data + hpos_out * info->tout.height_stride;
				// move row indices to next segment.
				int hc = info->hextents[i].hcount;
				max_height_per = max_u32(max_height_per, hc);
				hpos_in += hc;
				hpos_out += hc;
			}
		}
		// consider using static inline void width_interleave_1_loop_alt:
		//   ** only if b2s, only if bsW = 1
		//   It will make 'nd32*bsH' calls to memcpy2d, whereas the regular
		//   one will make one per output row. So,
		//   ** only use if nd32*bsH is less than the typical slice height,
		//     i.e. if n_extent * nd32 *bsH < out_height.
		//     In these cases it should reduce the number of calls to memcpy_2d.
		//
		if( is_b2s && bsW ==1 &&  info->tout.nd32*n_extent * bsH < out_height){
			info->width_interleave_loop_fp  = width_interleave_1_loop_alt;
		}
		// intermediate buffer allocation?
		if( bsW == 8 && !is_b2s){
			//Buffer width stride is discard_w + output width rounded to the nearest multiple of 4 * blocksize_w all multiplied by 32
			uint32_t total_width = info->discard_w + info->output_shape.width* elbytes;
			// round it up to the next multiple of 4*bsW, then multiply by 32
			info->intermed_buffer_stride = ((total_width + (4 * bsW) - 1) & (-(4*bsW))) * 32;
			info->intermed_bufrows_per_thread = max_height_per * info->tout.nd32;
		}
		// the # of extents is recorded as 1,2, or 4; if 3, we use 4
		// and the exec code will need to look at hextents[3].hcount==0 and skip those.
		int n_extent_eff = (n_extent==3)?4: n_extent;
		// one less work unit if we have n_extent. This will keep us from launching
		// 4 threads on V66 to handle 3 units * 1 batch, for instance.
		info->n_workunits = out_batches * n_extent_eff - ((n_extent==3)?1:0);
		info->n_extents = n_extent_eff;
		/*
		for(int i =0; i < n_extent_eff; i++){
			printf(" %d:  %3d   %p %p\n",i, info->hextents[i].hcount, info->hextents[i].in_ptr, info->hextents[i].out_ptr);
		}*/
	}
	return 0;
}

/// this extracts block size and padding from input[1] and input[4]
// for the 'DepthSpace' case.
// if dcrop_p is not null, this is where the 'depth_crop' req is stored if present.
// Values are checked only for >=1 (for dims) and >=0 (for crop).
//
static int
get_d2s_dims( struct nn_node *self, struct nn_graph * nn , struct bsz_crop * dst, int *dcrop_p)
{
	// get the block sizes, validate them.
	struct tensor const *blocksize_tensor = self->inputs[1];		// bs or {bsH, bsW}
	struct tensor const *cropinfo_tensor = (self->n_inputs <5)? NULL : self->inputs[4];

	int bsn = blocksize_tensor->shape.depth;
	int32_t bsH = tensor_get_int32( blocksize_tensor, 0);
	int32_t bsW = bsH;
	if( bsn > 1 ) bsW = tensor_get_int32( blocksize_tensor, 1);

	if( blocksize_tensor->data_size != bsn * sizeof(int32_t) || bsn > 2
			|| bsH <= 0 || bsH > 65535 || bsW <= 0 || bsW > 65535 ){
		return errlog(nn, "bad blocksize tensor");
	}
	dst->blocksize_h = bsH;
	dst->blocksize_w = bsW;
	if( dcrop_p != NULL) * dcrop_p =0;
	if( cropinfo_tensor == NULL){
		dst->crop_top = dst->crop_bottom = dst->crop_left = dst->crop_right =0;
		return 0;
	}
	int ctn = cropinfo_tensor->shape.depth;
	if( ctn < 4 || ctn > ((dcrop_p==NULL)?4:5) || cropinfo_tensor->data_size != ctn*sizeof(int32_t)){
		return errlog(nn,"Bad crop-descriptor tensor");
	}
	// range check them
	int32_t const * croparr = (int32_t const*)cropinfo_tensor->data;
	for( int i =0; i < 4; i++){
		if(croparr[i]  < 0 || croparr[i] > 65535 )
			return errlog( nn, "bad crop element[%d]= %d: must be >=0)", i, (int)croparr[i]);
	}
	dst->crop_top = croparr[0];
	dst->crop_bottom = croparr[1];
	dst->crop_left = croparr[2];
	dst->crop_right = croparr[3];
	if( ctn == 5 ){		// requested depth crop
		int dcropto = croparr[4];
		if( dcropto < 0) return errlog(nn,"bad depth crop value %d\n", dcropto);
		*dcrop_p = dcropto;
	}
	return 0;
}
/// this extracts block size and padding from input[1] and input[2]
// for the 'DepthSpace' case.
static int
get_b2s_dims( struct nn_node *self, struct nn_graph * nn , struct bsz_crop * dst)
{
	struct tensor const *blocksize_tensor = self->inputs[1];		// [bsH] or {bsH, bsW}
	struct tensor const *cropinfo_tensor = self->inputs[2];

	int bsn = blocksize_tensor->shape.depth;
	int32_t bsH = tensor_get_int32( blocksize_tensor, 0);
	int32_t bsW = 1;
	if( bsn > 1 ) bsW = tensor_get_int32( blocksize_tensor, 1);
	if( blocksize_tensor->data_size != bsn * sizeof(int32_t) || bsn > 2
			|| bsH <= 0 || bsH > 65535 || bsW <= 0 || bsW > 65535 ){
		return errlog(nn, "bad blocksize tensor");
	}
	dst->blocksize_h = bsH;
	dst->blocksize_w = bsW;
	// shape of cropinfo must be [1,1,2,bsn]
	if( cropinfo_tensor->shape.depth != 2
	  || cropinfo_tensor->shape.depth != bsn
	  || cropinfo_tensor->data_size != bsn*2*sizeof(int32_t)){
		return errlog(nn,"bad crop tensor");
	}
	int32_t const * carr = (int32_t const*) cropinfo_tensor->data;
	for(int i =0; i < 2*bsn;i++){
		int k = carr[i];
		if( i < 0 || i > 65535) return errlog(nn,"bad crop [%d,%d]=%d",i>>1,i&1,k);
	}
	dst->crop_left = dst->crop_right = 0;
	dst->crop_top = carr[0];
	dst->crop_bottom = carr[1];
	if( bsn == 2){
		dst->crop_left = carr[2];
		dst->crop_right = carr[3];
	}
	return 0;
}



static int
depthspace_d32_execute( struct nn_node *self, struct nn_graph * nn  )
{
	logmsg(nn,2,"depthspace_d32 execute. self=%p ",self);
	// set up the plan, or just check that the existing plan is ok
	int k = setup_x2s_plan( self, nn);
	if(k != 0) return k;

	struct x2s_d32_info * info = (struct x2s_d32_info *)self->opaque;
	//struct tensor const *input_tensor = self->inputs[0];
	//struct tensor * output_tensor = self->outputs[0];


	struct x2s_d32_runstate runstate;
	runstate.info = info;

	int n_work_units = info->n_workunits;
	runstate.cur_work_unit = 0;

	int n_threads = min_i32( n_work_units, DEPTHTOSPACE_MAX_THREADS);
	int bsW = info->bsz.blocksize_w;
	if ( bsW == 8 && !info->is_b2s)
	{
		uint32_t buf_per_thread  = info->intermed_bufrows_per_thread * info->intermed_buffer_stride;
		runstate.intermed_buffer_size_per_thread = buf_per_thread;
		uint32_t required_scratch = DEPTHTOSPACE_MAX_THREADS * buf_per_thread;
		if (nn->scratch_size < required_scratch)
		{
			if (nn_scratch_grow(nn, required_scratch))
				return errlog(nn, "need %d bytes scratch for requested depth to space operation", required_scratch);
		}
	logmsg(nn, 3, "allocated %d bytes of scratch", required_scratch);
	}
	runstate.intermed_buffer = nn->scratch;
	runstate.thrdindex = 0;


	logmsg(nn,3, "%d jobs in %d threads; %d per batch", n_work_units, n_threads, info->n_extents);

	nn_sem_init( & runstate.done_sem, 0);

	for( int i =0; i < n_threads; i++)
		nn_os_work_for_vector( nn, x2s_d32_workfunc, &runstate);
	tensor_copy(self->outputs[1],self->inputs[2]);
	tensor_copy(self->outputs[2],self->inputs[3]);

	nn_sem_wait_n_times(&runstate.done_sem, n_threads);
	logmsg(nn,2,"depthspace_d32 done. self=%p ",self);
	return 0;
}

// hvx work function; used for b2s and d2s
static void
x2s_d32_workfunc( struct nn_graph * nn, void * rstpv)
{
	struct x2s_d32_runstate * rstp = (struct x2s_d32_runstate *)rstpv;
	struct x2s_d32_info const * info = rstp->info;

	int njobs = info->n_workunits;
	int jobno;

	int nhseg = info->n_extents;		// this is a power of 2
	int rsh_segs = Q6_R_ct0_R(nhseg);
	Q6_dcfetch_A( &info->hextents[0].hcount);


	// 'cache' prefetch data: we prefetch one 'wunit' per function call and let the function do the rest
	//
	int pref_width = info->wunit_prefrow;		// width in bytes
	int pref_height = info->wunit_prefht;	// rows in a wunit
	int in_d32_stride = info->tin.d32_stride;
	int input_batch_stride = info->tin.batch_stride;

	width_interleave_loop_funcp  loop_funcp = info->width_interleave_loop_fp;

	int thrid= __sync_fetch_and_add( &rstp->thrdindex, 1);
	uint8_t * intermed_buffer = rstp->intermed_buffer + thrid * rstp->intermed_buffer_size_per_thread;
	while( jobno = __sync_fetch_and_add( &rstp->cur_work_unit, 1),  jobno < njobs ){

		int batchno = jobno >> rsh_segs;		// reduce to batch & seg
		int hslc_no = jobno & ( nhseg-1);

		int h_len = info->hextents[hslc_no].hcount;
		// sometimes this will be zero, e.g if we want 3 segs and padded it to 4.
		if( h_len == 0) continue;

		uint8_t const * in_ptr = info->hextents[hslc_no].in_ptr;
		in_ptr += input_batch_stride *batchno;
		l2pref( in_ptr, pref_height, pref_width, in_d32_stride);
		uint8_t * out = info->hextents[hslc_no].out_ptr;

		out +=  info->tout.batch_stride  *batchno;
		if (info->bsz.blocksize_w == 8)
		{
			width_interleave_8_loop( info, out, intermed_buffer, in_ptr, h_len);
		}
		else
		{
			(*loop_funcp)( info, out, in_ptr, h_len);
		}
		Q6_dcfetch_A( &info->hextents[0].hcount);
	}
	nn_sem_post( & rstp->done_sem);
}


static int
batchspace_d32_execute( struct nn_node *self, struct nn_graph * nn  )
{
	logmsg(nn,2,"batchspace_d32 execute. self=%p ",self);
	// set up the plan, or just check that the existing plan is ok
	int k = setup_x2s_plan( self, nn);
	if(k != 0) return k;

	struct x2s_d32_info * info = (struct x2s_d32_info *)self->opaque;
	//struct tensor const *input_tensor = self->inputs[0];
	//struct tensor * output_tensor = self->outputs[0];


	struct x2s_d32_runstate runstate;
	runstate.info = info;


	int n_work_units = info->n_workunits;
	runstate.cur_work_unit = 0;

	int n_threads = min_i32( n_work_units, DEPTHTOSPACE_MAX_THREADS);


	logmsg(nn,3, "%d jobs in %d threads; %d per batch", n_work_units, n_threads, info->n_extents);

	nn_sem_init( & runstate.done_sem, 0);

	for( int i =0; i < n_threads; i++)
		nn_os_work_for_vector( nn, x2s_d32_workfunc, &runstate);
	tensor_copy(self->outputs[1],self->inputs[3]);
	tensor_copy(self->outputs[2],self->inputs[4]);

	nn_sem_wait_n_times(&runstate.done_sem, n_threads);
	logmsg(nn,2,"batchspace_d32 done. self=%p ",self);
	return 0;
}


//////////////////////////
//////////////////////////
// HVX Width interleave loops - code generation issue
//
// The core operation reads N  vectors from N separate input depth slices (N=blocksize_w)
// interleaves the 32-bit elements, and store the N output vectors to a single
// output depth slice (in general, with a vlalign)
// The first operation in the row needs to be able to drop vectors (and may
// drop all N; this occurs when we only need a partial vector for the next op).
//
//  The original pattern I had for the start of row (example, N=3) was:
//
//		unsigned iv = -vecdrop;
//		HVX_Vector_x3 result = width_interleave_3( *in0++, *in1++, *in2++, invar);
//		HVX_Vector o0 = Q6_V_vlalign_VVR( result.val[0], result.val[0], vlalign_amt);
//		HVX_Vector o1 = Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
//		HVX_Vector o2 = Q6_V_vlalign_VVR( result.val[2], result.val[1], vlalign_amt);
//		revout = result.val[2];
//		if( iv < vecout) *outv++ = o0;
//		if( iv+1 < vecout) *outv++ = o1;
//		if( iv+2 < vecout) *outv++ = o2;
//		for( int j = 0; j < vloops; j++){
//
//  Here, 'iv' is used to make sure that if vecdrop >0 the first 'vecdrop' vectors are not stored
//  (since iv = 0xFFFFFFFF is not < vecout); also, if it happens that vecout is so mall that the
//  last vector(s) should not be stored, this takes care of that too. The compiler generates these
//  conditions efficiently (hoisting iv+1 and iv+2) but the current version (8.2.04) does *not* handle
//  "if(cond) *outv++ = o0;" well - the code is correct but the
//  conditional modification of the pointer is handled poorly (QTOOL-37878)
//  So these have been rearranged to be as follows, with no conditional pointer mod:
//
//		if( iv < vecout) outv[-3] = o0;
//		if( iv+1 < vecout) outv[-2] = o1;
//		if( iv+2 < vecout) outv[-1] = o2;
//		for( int j = 0; j < vloops; j++){
//
//   ... and the 'outv' pointer is modified so that it is correct at the start of the 'for 'j' loop
//    (i.e. it is advanced by (N-vecdrop) vectors, relative to the first actual output vector)
//


//
// This is a pack of loop invariants needed for width_interleave_3
struct intlv3_invariants
{
	HVX_Vector vdel0;   // { 32 of 0, 32 of 64, 32 of 0, 32 of 64}
	HVX_Vector vdel1;	// { 128 of 32 }
	HVX_Vector vdel2;	// { 32 of 64, 32 of 0, 32 of 64, 32 of 0}
	HVX_VectorPred q_first;	// true in 64..95
	HVX_VectorPred q_second;	// true in 32..95
	HVX_VectorPred q_third;	// true in 32..63
};
// in: A0 A1 A2 A3  (each is 32 bytes)
//     B0 B1 B2 B3
// out:  A0 B0 A1 B1, A2 B2 A3 B3
//
// 'chunksize' is 32 for 8-bit ops, and 64 for 16-bit ops
static inline HVX_VectorPair width_interleave_2( HVX_Vector  v0, HVX_Vector v1, int depth32_chunk )
{
	return Q6_W_vshuff_VVR( v1, v0, -depth32_chunk);
}
// in: A0 A1 A2 A3  (each is 32 bytes)
//     B0 B1 B2 B3
//     C0 C1 C2 C3
// out:  A0 B0 C0 A1, B1 C1 A2 B2, C2 A3 B3 C3
//
// Step 1: (apply delta to inputs)
//   A0 A3 A2 A1  			using { 32 of 0, 32 of 64, 32 of 0, 32 of 64}
//   B1 B0 B3 B2			using { 128 of 32 }
//   C2 C1 C0 C3			using { 32 of 64, 32 of 0, 32 of 64, 32 of 0}
//
// Step 2: swap quadrant 2 in second 2 vecs  (using q_first)
//   A0 A3 A2 A1
//   B1 B0 C0 B2
//   C2 C1 B3 C3
// Step 3: swap quadrant 1,2 in first 2 vecs	(using q_second)
//  A0 B0 C0 A1
//  B1 A3 A2 B2
//  C2 C1 B3 C3
// Step 4: swap quadrant 1 in second 2 vecs		(using q_third)
//  A0 B0 C0 A1
//  B1 C1 A2 B2
//  C2 A3 B3 C3
//


static inline HVX_Vector_x3 width_interleave_3( HVX_Vector  v0, HVX_Vector v1, HVX_Vector v2, struct intlv3_invariants invar )
{
	v0 = Q6_V_vdelta_VV( v0, invar.vdel0);
	v1 = Q6_V_vdelta_VV( v1, invar.vdel1);
	v2 = Q6_V_vdelta_VV( v2, invar.vdel2);
	HVX_VectorPair v12 = Q6_W_vswap_QVV( invar.q_first, v2, v1);	// step 2
	HVX_Vector v0_out = Q6_V_vmux_QVV( invar.q_second,  Q6_V_lo_W(v12), v0  );	// step 3
	HVX_Vector v1_tmp = Q6_V_vmux_QVV( invar.q_second,  v0, Q6_V_lo_W(v12)  );
	v12 = Q6_W_vswap_QVV( invar.q_third, Q6_V_hi_W(v12), v1_tmp);				// step 4

	HVX_Vector_x3 result = {{
			v0_out, Q6_V_lo_W(v12), Q6_V_hi_W(v12)
	}};
	return result;
}

// construct the invariants needed for width_interleave_3
//
// For 16-bit data,  elbytes = 2, we want
//  vdel0 = vdel2 = { all 0 }
//  vdel1 = { all 64}
//  q_first  = { { 1} *64 ,{0} * 64 }
//  q_second  = { { 0} *64 ,{1} * 64 }
//  q_third = { all 0 }

static inline struct intlv3_invariants init_intlv3_invariants (int elbytes)
{
	// elbytes = 1 or 2.
	int is_u16 = (elbytes >1);
	unsigned k20202020 = is_u16? 0 : 0x20202020;
	unsigned k40404040 = k20202020 <<1;

	HVX_Vector index = *(HVX_Vector const *)const_Count128;	// {0,1 ..127}
	HVX_VectorPred q23 = Q6_Q_vand_VR( index, 0x40404040);	// true in quadrants 2,3
	HVX_VectorPred q13 = Q6_Q_vand_VR( index, k20202020);	// true in quadrants 1,3 (all 0 for u16)

	struct intlv3_invariants result;
	result.vdel0 = Q6_V_vand_QR( q13, k40404040);		// 32 of 0, 32 of 64, 32 of 0 , 32 of 64
	result.vdel1 = q6op_Vb_vsplat_R(elbytes*32);		// 128 of 32   (or of 64, for u16)
#if __HEXAGON_ARCH__ >= 62
	result.vdel2 = Q6_V_vand_QnR( q13, k40404040);		// 32 of 64, 32 of 0, 32 of 64 , 32 of 0
#else
	result.vdel2 = Q6_V_vand_QR( Q6_Q_not_Q(q13), k40404040);		// 32 of 64, 32 of 0, 32 of 64 , 32 of 0
#endif
	result.q_first = Q6_Q_xor_QQ ( Q6_Q_not_Q( q23),		// quadrant 2 only;  Q0,1 for u16
					Q6_Q_vsetq_R(is_u16?0:96));
	result.q_second = Q6_Q_xor_QQ( q23,q13);		// quadrant 1 & 2			// Q2,3 for u16
	result.q_third = Q6_Q_and_QQn( q13,q23);		// quadrant 1 only		    // all 0 for u16
	return result;
}


// in: A0 A1 A2 A3  (each is 32 bytes)
//     B0 B1 B2 B3
//     C0 C1 C2 C3
//     D0 D1 D2 D3
// out:  A0 B0 C0 D0, A1 B1 C1 D2, A2 B2 C2, D3, A3 B3 C3 D3
//
//  Step 1: transpose units of 64
//     A0 A1 C0 C1 <--
//     B0 B1 D0 D1
//     A2 A3 C2 C3 <--
//     B2 B3 C2 C3
// Step 2: transpose units of 32
//     A0 B0 C0 D0 <--
//     A1 B1 C1 D1 <--
//     A2 B2 C2 D2
//     A3 B3 C3 D3
//
// depth32_chunk = 32 for 8-bit processing.
// For depth_chunk = 64, this will do the equivalent op for 16-bit values
// (8 width units in the four output vectors, instead of 16). It uses 4 vshuff where
// 2 would suffice, but at least it allows sharing the same code.
//
static inline HVX_Vector_x4 width_interleave_4( HVX_Vector  v0, HVX_Vector v1, HVX_Vector v2, HVX_Vector v3, int depth32_chunk)
{
	HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2, v0, 64);
	HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3, v1, 64);

	HVX_VectorPair res01 = Q6_W_vshuff_VVR(  Q6_V_lo_W( sh13), Q6_V_lo_W(sh02),depth32_chunk);
	HVX_VectorPair res23 = Q6_W_vshuff_VVR(  Q6_V_hi_W( sh13), Q6_V_hi_W(sh02),depth32_chunk);

	HVX_Vector_x4 res = {{
			Q6_V_lo_W(res01), Q6_V_hi_W(res01),  Q6_V_lo_W(res23), Q6_V_hi_W(res23)
	}};
	return res;
}

//
// WORK FUNCTION for blocksize_w = 1
// parameters are:
//    runstate pointer
//    batch index
//    output height start index
//    output row count (>=1)
//
// This is a special case since we don't actually interleave anything in width;
// instead we do a single 2d memcopy
// The height of the copy is nd32_out * height_count
// width is output_width* depth32_chunk bytes
// input address needs to be adjusted at the left edge:
//   add 128* wunit_vecdrop, and subtract wunit_vlalign.
//

static inline void
width_interleave_1_loop( struct x2s_d32_info const * info,
		uint8_t * out,				// output address
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_d32_stride = info->tin.d32_stride;
	int32_t output_d32_stride = info->tout.d32_stride;
	int nd32_out = info->tout.nd32;

	// adjust source address to first retained width unit (it may not be vector aligned after)
	int ptr_adj =  info->wunit_vecdrop*128 -info->wunit_vlalign;
	in_ptr += ptr_adj;


	int memcpy_width = info->depth32_chunk* info->output_shape.width;
	int memcpy_height = nd32_out;

	int32_t input_wunit_stride = 0;
	int32_t wrap_offset = 0;
	uint8_t const * input_end_addr = NULL;

	//depth2space can do one giant 2D copy; b2s must do one per output row.
	//
	if( !info->is_b2s){
		memcpy_height *= height_count;
		height_count = 1;
	}else{
		input_wunit_stride = info->input_wunit_stride;
		wrap_offset = info->b2s_wrap_offset;
		input_end_addr = info->input_end_addr + ptr_adj;
	}

	for( int i = 0; i < height_count; i++){
		// this function requires aligned in/out strides, but width and pointers have no alignment
		// constraints
		vmemcpy_2d_asm( memcpy_width, memcpy_height,		// width and height of rectangle to copy
				 out, output_d32_stride, 							// output pointer and stride
				 in_ptr, input_d32_stride);						// input pointer and stride

		in_ptr += input_wunit_stride;
		if( in_ptr >= input_end_addr) in_ptr -= wrap_offset;
		out += output_d32_stride*nd32_out;
	}
}

//
// This is an 'alternative' width_interleave_1_loop
// which is only used for b2s, and in the case where nd32*bsH is less
// than the height_chunk. In this case, rather than issue a 2d memcpy
// per output row, it does one to cover each traversal of the input, repeats that
// per nd32, and then continues these traversals for the whole operation.
// So the # of vmemcpy calls will be bsH*nd32, rather than one per output row.
//
static inline void
width_interleave_1_loop_alt( struct x2s_d32_info const * info,
		uint8_t * out,				// output address
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_d32_stride = info->tin.d32_stride;
	int32_t output_d32_stride = info->tout.d32_stride;
	int32_t output_height_stride = info->tout.height_stride;
	int nd32_out = info->tout.nd32;

	// adjust source address to first retained width unit (it may not be vector aligned after)
	int ptr_adj =  info->wunit_vecdrop*128 -info->wunit_vlalign;
	in_ptr += ptr_adj;


	int memcpy_width = info->depth32_chunk* info->output_shape.width;

	int32_t input_wunit_stride = info->input_wunit_stride;
	int32_t wrap_offset = info->b2s_wrap_offset;
	uint8_t const * input_end_addr = info->input_end_addr + ptr_adj;

	// loop until done....
	while( height_count > 0){
		// how many loops can we do, *exactly*, before in_ptr goes over the 'end_addr'?
		// i.e. the smallest x for which in_ptr  + x* input_wunit_stride >= input_end_addr.
		//
		int hnow = (input_end_addr +  (input_wunit_stride-1)-in_ptr)/(unsigned)input_wunit_stride;
		hnow = min_i32( hnow , height_count);
		// "hnow"  is the number of rows to do on this traversal.
		// each traversal does a rectangle of that height and iterates over the nd32.
		if( hnow <=0) return;		// should never
		for( int i =0; i < nd32_out; i++){
			vmemcpy_2d_asm( memcpy_width, hnow,				// width and height of rectangle to copy
					 out + output_d32_stride*i, 	output_height_stride,		// output pointer and stride
					 in_ptr+  input_d32_stride*i, 	input_wunit_stride);					// input pointer and stride

		}
		// finished a traversal of the input, move to next
		if( height_count < hnow) break;
		height_count -= hnow;
		out += hnow * output_height_stride;
		in_ptr += hnow * input_wunit_stride - wrap_offset;
	}
}



//
// WORK FUNCTION for blocksize_w = 2
// parameters are:
//    runstate pointer
//    batch index
//    output height start index
//    output row count (>=1)
//
// It is expected that a prefetch is in process already for the first output height unit;
// the rest are done as we go.


static inline void
width_interleave_2_loop( struct x2s_d32_info const * info,
		uint8_t * out,				// output address
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_wunit_stride = info->input_wunit_stride;
	int depth32_chunk = info->depth32_chunk;
	int32_t input_wunit_span = info->input_wunit_span;
	int32_t input_d32_stride = info->tin.d32_stride;
	int32_t output_d32_stride = info->tout.d32_stride;
	int nd32_out = info->tout.nd32;
	int32_t pref_height = info->wunit_prefht;
	int32_t pref_width = info->wunit_prefrow;

	// this is for b2s, to do the wrapping at row advance:
	// whenever advancing the input pointer makes it >= input_end_addr, we
	// subtract 'wrap_offset' from it, which wraps it back and advances a row.
	uint8_t const * input_end_addr = info->input_end_addr;
	int32_t wrap_offset = info->b2s_wrap_offset;

	uint8_t const * in_wunit = in_ptr;

	int vlalign_amt = info->wunit_vlalign;
	int vecdrop = info->wunit_vecdrop;			// # of vector stores to skip  0..2
	unsigned vecout = info->wunit_vecout;			// total output vectors (could be as small as 1)
	int vloops = info->wunit_vloops;			// # of 'full loops' to do ( >= 0)
	int v_at_end = info->wunit_v_at_end;			// # of vecs to store at end (range 0..1; or -1 meaning 1 with no read)

	// modify output pointer to be correct for the start of the 'for j' loop
	out += 128*(2-vecdrop);

	HVX_Vector prevout = Q6_V_vzero();
	for( int ih = 0; ih < height_count; ih++){

		uint8_t const * in = in_wunit;
		uint8_t const * in_wunit_next = in_wunit + input_wunit_stride;
		if( in_wunit_next >= input_end_addr) in_wunit_next -= wrap_offset;
		// prefetch next wunit input, if there is a next.
		if( likely(ih+1 < height_count)){
			l2pref( in_wunit_next, pref_height, pref_width, input_d32_stride );
		}

		for(int r = 0; r < nd32_out; r++ ){
			HVX_Vector const* in0 = (HVX_Vector const*)in;
			HVX_Vector const* in1 = (HVX_Vector const*)(in+ input_wunit_span);
			HVX_Vector * outv  = (HVX_Vector *)out;
			// first iteration
			unsigned iv = -vecdrop;
			HVX_VectorPair result = width_interleave_2( *in0++, *in1++, depth32_chunk);
			HVX_Vector o0 = Q6_V_vlalign_VVR( Q6_V_lo_W(result), Q6_V_lo_W(result), vlalign_amt);
			HVX_Vector o1 = Q6_V_vlalign_VVR( Q6_V_hi_W(result), Q6_V_lo_W(result), vlalign_amt);
			prevout = Q6_V_hi_W(result);
			if( iv < vecout) outv[-2] = o0;
			if( (iv+1) < vecout) outv[-1] = o1;
			for( int j = 0; j < vloops; j++){
				result = width_interleave_2( *in0++, *in1++, depth32_chunk);
				outv[0] = Q6_V_vlalign_VVR( Q6_V_lo_W(result), prevout, vlalign_amt);
				outv[1] = Q6_V_vlalign_VVR( Q6_V_hi_W(result), Q6_V_lo_W(result), vlalign_amt);
				prevout = Q6_V_hi_W(result);
				outv += 2;
			}
			if(v_at_end){
				if( v_at_end> 0)
					result = width_interleave_2( *in0, *in1, depth32_chunk);
				outv[0] = Q6_V_vlalign_VVR( Q6_V_lo_W(result), prevout, vlalign_amt);
			}
			in += input_d32_stride;
			out += output_d32_stride;
		}
		in_wunit = in_wunit_next;
	} // for ih
}


//
// WORK FUNCTION for blocksize_w = 2, special case when output depth = 16.
// parameters are:
//    runstate pointer
//    batch index
//    output height start index
//    output row count (>=1)
//
// It is expected that a prefetch is in process already for the first output height unit;
// the rest are done as we go.
// Assuming no h-cropping, this consists of reading the single d32 slice of input and
// writing it out at double width, e.g as below (each symbol represents 16 bytes):
//
//  [ A0 B0 A1 B1 A2 B2 A3 B3 ] -> [A0 X B0 X A1 X B1 X ] [A2 X B2 X A3 X B3 X ]
// This is done by just shuffling the vector with anything (i.e. itself) in 16-byte units.
// (for 16-bit data, do it 32-byte units)

// this is only used in D2S, not B2S.
static inline void
width_interleave_2_to16_loop( struct x2s_d32_info const * info,
		uint8_t * out,				// output address
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_wunit_span = info->input_wunit_span;
	int32_t input_wunit_stride = input_wunit_span;		// same as span for this case.
	int32_t input_d32_stride = info->tin.d32_stride;
	int32_t output_d32_stride = info->tout.d32_stride;
	int nd32_out = info->tout.nd32;
	int32_t pref_height = info->wunit_prefht;
	int32_t pref_width = info->wunit_prefrow;

	uint8_t const * in_wunit = in_ptr;
	uint8_t const * in_limit = in_wunit + height_count * input_wunit_stride;	// to prevent prefetch on last unit

	int vlalign_amt = info->wunit_vlalign;
	int vecdrop = info->wunit_vecdrop;			// # of vector stores to skip  0..2
	unsigned vecout = info->wunit_vecout;			// total output vectors (could be as small as 1)
	int vloops = info->wunit_vloops;			// # of 'full loops' to do ( >= 0)
	int v_at_end = info->wunit_v_at_end;			// # of vecs to store at end (range 0..1; or -1 meaning 1 with no read)

	// modify output pointer to be correct for the start of the 'for j' loop
	out += 128*(2-vecdrop);

	int shufn = -(info->depth32_chunk/2u);	// -16 for 8-bit; -32 for 16 bit

	HVX_Vector prevout = Q6_V_vzero();
	for( int ih = 0; ih < height_count; ih++){

		uint8_t const * in = in_wunit;
		uint8_t const * in_wunit_next = in_wunit + input_wunit_stride;
		// prefetch next wunit input, if there is a next.
		if( likely(in_wunit_next < in_limit)){
			l2pref( in_wunit_next, pref_height, pref_width, input_d32_stride );
		}

		for(int r = 0; r < nd32_out; r++ ){
			HVX_Vector const* in0 = (HVX_Vector const*)in;

			HVX_Vector * outv  = (HVX_Vector *)out;
			// first iteration
			unsigned iv = -vecdrop;
			HVX_Vector vin = *in0++;
			HVX_VectorPair result = Q6_W_vshuff_VVR(vin,vin,shufn);
			HVX_Vector o0 = Q6_V_vlalign_VVR( Q6_V_lo_W(result), Q6_V_lo_W(result), vlalign_amt);
			HVX_Vector o1 = Q6_V_vlalign_VVR( Q6_V_hi_W(result), Q6_V_lo_W(result), vlalign_amt);
			prevout = Q6_V_hi_W(result);
			if( iv < vecout) outv[-2] = o0;
			if( (iv+1) < vecout) outv[-1] = o1;
			for( int j = 0; j < vloops; j++){
				vin = *in0++;
				result = Q6_W_vshuff_VVR(vin,vin,shufn);
				outv[0] = Q6_V_vlalign_VVR( Q6_V_lo_W(result), prevout, vlalign_amt);
				outv[1] = Q6_V_vlalign_VVR( Q6_V_hi_W(result), Q6_V_lo_W(result), vlalign_amt);
				prevout = Q6_V_hi_W(result);
				outv += 2;
			}
			if(v_at_end){
				if( v_at_end> 0){
					vin = *in0;
					result = Q6_W_vshuff_VVR(vin,vin,shufn);
				}
				outv[0] = Q6_V_vlalign_VVR( Q6_V_lo_W(result), prevout, vlalign_amt);
			}
			in += input_d32_stride;
			out += output_d32_stride;
		}
		in_wunit = in_wunit_next;
	} // for ih
}
//
// WORK FUNCTION for blocksize_w = 3
// parameters are:
//    runstate pointer
//    batch index
//    output height start index
//    output row count (>=1)
//
// It is expected that a prefetch is in process already for the first output height unit;
// the rest are done as we go.


static inline void
width_interleave_3_loop( struct x2s_d32_info const * info,
		uint8_t * out,				// output address
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_wunit_stride = info->input_wunit_stride;
	int32_t input_wunit_span = info->input_wunit_span;
	int32_t input_d32_stride = info->tin.d32_stride;
	int32_t output_d32_stride = info->tout.d32_stride;
	int nd32_out = info->tout.nd32;
	int32_t pref_height = info->wunit_prefht;
	int32_t pref_width = info->wunit_prefrow;

	// this is for b2s, to do the wrapping at row advance:
	// whenever advancing the input pointer makes it >= input_end_addr, we
	// subtract 'wrap_offset' from it, which wraps it back and advances a row.
	uint8_t const * input_end_addr = info->input_end_addr;
	int32_t wrap_offset = info->b2s_wrap_offset;

	uint8_t const * in_wunit = in_ptr;

	int vlalign_amt = info->wunit_vlalign;
	int vecdrop = info->wunit_vecdrop;			// # of vector stores to skip  0..3
	unsigned vecout = info->wunit_vecout;			// total output vectors (could be as small as 1)
	int vloops = info->wunit_vloops;			// # of 'full loops' to do ( >= 0)
	int v_at_end = info->wunit_v_at_end;			// # of vecs to store at end (range 0..2; or -1 meaning 1 with no read)


	// modify output pointer to be correct for the start of the 'for j' loop
	out += 128*(3-vecdrop);

	HVX_Vector prevout = Q6_V_vzero();
	struct intlv3_invariants invar = init_intlv3_invariants(info->elbytes);

	for( int ih = 0; ih < height_count; ih++){

		uint8_t const * in = in_wunit;
		uint8_t const * in_wunit_next = in_wunit + input_wunit_stride;
		if( in_wunit_next >= input_end_addr) in_wunit_next -= wrap_offset;
		// prefetch next wunit input, if there is a next.
		if( likely(ih+1 < height_count)){
			l2pref( in_wunit_next, pref_height, pref_width, input_d32_stride );
		}

		for(int r = 0; r < nd32_out; r++ ){
			HVX_Vector const* in0 = (HVX_Vector const*)in;
			HVX_Vector const* in1 = (HVX_Vector const*)(in+ input_wunit_span);
			HVX_Vector const* in2 = (HVX_Vector const*)(in + 2*input_wunit_span);
			HVX_Vector * outv  = (HVX_Vector *)out;
			// first iteration
			unsigned iv = -vecdrop;
			HVX_Vector_x3 result = width_interleave_3( *in0++, *in1++, *in2++, invar);
			HVX_Vector o0 = Q6_V_vlalign_VVR( result.val[0], result.val[0], vlalign_amt);
			HVX_Vector o1 = Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
			HVX_Vector o2 = Q6_V_vlalign_VVR( result.val[2], result.val[1], vlalign_amt);
			prevout = result.val[2];
			if( iv < vecout) outv[-3] = o0;
			iv ++;
			if( iv < vecout) outv[-2] = o1;
			iv ++;
			if( iv < vecout) outv[-1] = o2;
			for( int j = 0; j < vloops; j++){
				result = width_interleave_3( *in0++, *in1++, *in2++, invar);
				outv[0] = Q6_V_vlalign_VVR( result.val[0], prevout, vlalign_amt);
				outv[1] = Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
				outv[2] = Q6_V_vlalign_VVR( result.val[2], result.val[1], vlalign_amt);
				prevout = result.val[2];
				outv += 3;
			}
			if(v_at_end){
				if( v_at_end> 0)
					result = width_interleave_3( *in0, *in1, *in2, invar);
				outv[0] = Q6_V_vlalign_VVR( result.val[0], prevout, vlalign_amt);
				if( v_at_end > 1)
					outv[1]= Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
			}
			in += input_d32_stride;
			out += output_d32_stride;
		}
		in_wunit = in_wunit_next;
	} // for ih
}



//
// WORK FUNCTION for blocksize_w = 4
// parameters are:
//    runstate pointer
//    batch index
//    output height start index
//    output row count (>=1)
//
// It is expected that a prefetch is in process already for the first output height unit;
// the rest are done as we go.


static inline void
width_interleave_4_loop( struct x2s_d32_info const * info,
		uint8_t * out,				// output address
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_wunit_stride = info->input_wunit_stride;
	int32_t input_wunit_span = info->input_wunit_span;
	int32_t input_d32_stride = info->tin.d32_stride;
	int32_t output_d32_stride = info->tout.d32_stride;
	int depth32_chunk = info->depth32_chunk;
	int nd32_out = info->tout.nd32;
	int32_t pref_height = info->wunit_prefht;
	int32_t pref_width = info->wunit_prefrow;

	// this is for b2s, to do the wrapping at row advance:
	// whenever advancing the input pointer makes it >= input_end_addr, we
	// subtract 'wrap_offset' from it, which wraps it back and advances a row.
	uint8_t const * input_end_addr = info->input_end_addr;
	int32_t wrap_offset = info->b2s_wrap_offset;

	uint8_t const * in_wunit = in_ptr;

	int vlalign_amt = info->wunit_vlalign;
	int vecdrop = info->wunit_vecdrop;			// # of vector stores to skip  0..4
	unsigned vecout = info->wunit_vecout;			// total output vectors (could be as small as 1)
	int vloops = info->wunit_vloops;			// # of 'full loops' to do ( >= 0)
	int v_at_end = info->wunit_v_at_end;			// # of vecs to store at end (range 0..3; or -1 meaning 1 with no read)

	// modify output pointer to be correct for the start of the 'for j' loop
	out += 128*(4-vecdrop);

	HVX_Vector prevout = Q6_V_vzero();

	for( int ih = 0; ih < height_count; ih++){

		uint8_t const * in = in_wunit;
		uint8_t const * in_wunit_next = in_wunit + input_wunit_stride;
		if( in_wunit_next >= input_end_addr) in_wunit_next -= wrap_offset;
		// prefetch next wunit input, if there is a next.
		if( likely(ih+1 < height_count)){
			l2pref( in_wunit_next, pref_height, pref_width, input_d32_stride );
		}


		for(int r = 0; r < nd32_out; r++ ){
			HVX_Vector const* in0 = (HVX_Vector const*)in;
			HVX_Vector const* in1 = (HVX_Vector const*)(in+ input_wunit_span);
			HVX_Vector const* in2 = (HVX_Vector const*)(in + 2*input_wunit_span);
			HVX_Vector const* in3 = (HVX_Vector const*)(in + 3*input_wunit_span);
			HVX_Vector * outv  = (HVX_Vector *)out;
			// first iteration
			unsigned iv = -vecdrop;
			HVX_Vector_x4 result = width_interleave_4( *in0++, *in1++, *in2++, *in3++,depth32_chunk);
			HVX_Vector o0 = Q6_V_vlalign_VVR( result.val[0], result.val[0], vlalign_amt);
			HVX_Vector o1 = Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
			HVX_Vector o2 = Q6_V_vlalign_VVR( result.val[2], result.val[1], vlalign_amt);
			HVX_Vector o3 = Q6_V_vlalign_VVR( result.val[3], result.val[2], vlalign_amt);
			prevout = result.val[3];
			if( iv < vecout) outv[-4] = o0;
			if( iv+1 < vecout) outv[-3] = o1;
			if( iv+2 < vecout) outv[-2] = o2;
			if( iv+3 < vecout) outv[-1] = o3;
			for( int j = 0; j < vloops; j++){
				result = width_interleave_4( *in0++, *in1++, *in2++, *in3++,depth32_chunk);
				outv[0] = Q6_V_vlalign_VVR( result.val[0], prevout, vlalign_amt);
				outv[1] = Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
				outv[2] = Q6_V_vlalign_VVR( result.val[2], result.val[1], vlalign_amt);
				outv[3] = Q6_V_vlalign_VVR( result.val[3], result.val[2], vlalign_amt);
				prevout = result.val[3];
				outv += 4;
			}
			if(v_at_end){
				if( v_at_end> 0)
					result = width_interleave_4( *in0, *in1, *in2, *in3,depth32_chunk);
				outv[0] = Q6_V_vlalign_VVR( result.val[0], prevout, vlalign_amt);
				if( v_at_end > 1){
					outv[1]= Q6_V_vlalign_VVR( result.val[1], result.val[0], vlalign_amt);
					if( v_at_end>2)
						outv[2]= Q6_V_vlalign_VVR( result.val[2], result.val[1], vlalign_amt);
				}
			}
			in += input_d32_stride;
			out += output_d32_stride;
		}
		in_wunit = in_wunit_next;
	} // for ih
}

//Only used for depth to space when blocksize_w = 8 (although it can be modified to work with b2s as well)
//Write interleaved chunks of blocksize_w = 4 into an intermidiate buffer, then write out the section we want (after cropping) to the final output
static inline void
width_interleave_8_loop( struct x2s_d32_info const * info,
		uint8_t * out,              // output address
		uint8_t * intermed_ptr,		// address within intermidiate buffer
		uint8_t const * in_ptr,		// input address
	    int height_count )
{
	int32_t input_wunit_stride = info->input_wunit_stride;
	int32_t input_wunit_span = info->input_wunit_span;
	int32_t input_d32_stride = info->tin.d32_stride;
	int depth32_chunk = info->depth32_chunk;
	int nd32_out = info->tout.nd32;
	int32_t pref_height = info->wunit_prefht;
	int32_t pref_width = info->wunit_prefrow;

	uint8_t const * in_wunit = in_ptr;

	int vloops = info->wunit_vloops;			// # of 'full loops' to do ( >= 0)
	// how many here? always one at the start, plus 'vloops', plus one more if v_at_end > 0

	vloops += (info->wunit_v_at_end>0)?2:1;		// number we need to do here

	int32_t intermed_buffer_stride = info->intermed_buffer_stride;

	uint8_t * intermed_copy_ptr = intermed_ptr;
	for( int ih = 0; ih < height_count; ih++){

		uint8_t const * in = in_wunit;
		uint8_t const * in_wunit_next = in_wunit + input_wunit_stride;
		// prefetch next wunit input, if there is a next.
		if( likely(ih+1 < height_count)){
			l2pref( in_wunit_next, pref_height, pref_width, input_d32_stride );
		}

		for(int r = 0; r < nd32_out; r++ ){
			HVX_Vector const* in0 = (HVX_Vector const*)in;
			HVX_Vector const* in1 = (HVX_Vector const*)(in+ input_wunit_span);
			HVX_Vector const* in2 = (HVX_Vector const*)(in + 2*input_wunit_span);
			HVX_Vector const* in3 = (HVX_Vector const*)(in + 3*input_wunit_span);
			HVX_Vector const* in4 = (HVX_Vector const*)(in + 4*input_wunit_span);
			HVX_Vector const* in5 = (HVX_Vector const*)(in+ 5*input_wunit_span);
			HVX_Vector const* in6 = (HVX_Vector const*)(in + 6*input_wunit_span);
			HVX_Vector const* in7 = (HVX_Vector const*)(in + 7*input_wunit_span);

			HVX_Vector * outv  = (HVX_Vector *)intermed_ptr;
			if( depth32_chunk == 32){			// 8 bit case
				for( int j = 0; j < vloops ; j++){
					HVX_Vector_x4 result = width_interleave_4( *in0++, *in1++, *in2++, *in3++,32);
					HVX_Vector_x4 result2 = width_interleave_4( *in4++, *in5++, *in6++, *in7++,32);
					outv[0] = result.val[0];
					outv[1] = result2.val[0];
					outv[2] = result.val[1];
					outv[3] = result2.val[1];
					outv[4] = result.val[2];
					outv[5] = result2.val[2];
					outv[6] = result.val[3];
					outv[7] = result2.val[3];
					outv += 8;
				}
			}else{								// 16-bit case
				for( int j = 0; j < vloops ; j++){
					HVX_VectorPair out04 = Q6_W_vshuff_VVR( *in1++, *in0++,64);
					HVX_VectorPair out15 = Q6_W_vshuff_VVR( *in3++, *in2++,64);
					HVX_VectorPair out26 = Q6_W_vshuff_VVR( *in5++, *in4++,64);
					HVX_VectorPair out37 = Q6_W_vshuff_VVR( *in7++, *in6++,64);
					outv[0] = Q6_V_lo_W(out04);
					outv[4] = Q6_V_hi_W(out04);
					outv[1] = Q6_V_lo_W(out15);
					outv[5] = Q6_V_hi_W(out15);
					outv[2] = Q6_V_lo_W(out26);
					outv[6] = Q6_V_hi_W(out26);
					outv[3] = Q6_V_lo_W(out37);
					outv[7] = Q6_V_hi_W(out37);
					outv += 8;
				}
			}

			in += input_d32_stride;
			intermed_ptr += intermed_buffer_stride;
		}
		in_wunit = in_wunit_next;
	} // for ih
//	for (int h = 0; h < height_count; h++)
	{
		vmemcpy_2d_asm(info->output_shape.width * depth32_chunk, nd32_out * height_count,	// width, height
				out, info->tout.d32_stride,				// dst ptr, stride
				intermed_copy_ptr + (32 * info->discard_w), intermed_buffer_stride);	// src ptr, stride
//		out += info->tout.d32_stride * nd32_out;
//		intermed_copy_ptr += intermed_buffer_stride * nd32_out;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
////////////////////// SpaceTo(Depth,Batch) //////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//
// These two have similar 'inner loop' requirements.
//
// Note for top/bottom padding:
//   - The top and bottom padding translates to entire rows of padding at the
//     top and/or bottom of various batches. To simplify this, we make a table
//     of size [bsH], which contains these values for that segment of the output:
//               - number of top padding rows in that output segment
//               - address of the first 'non-padding' row.
//               - number of non-padding rows (>=0)
//               - number of bottom-padding rows (>=0)
//     Given the row strides, this information is enough to do all the padding ops.
//     This table is ordered so that it is indexed by [inrow%(blocksizeH], i.e. the [0]
//     entry in the table contains the pointers to where the first 'deal' of the first input row would go,
//     The sum of all the 'seg_valid_rows' in the table is always in_height.
//
#define MAX_BSW 4
//
struct batch_seg_desc{
	uint8_t * ptr_begin;		// origins for this batch segment (which is 1/bsH of the entire output)
	int seg_valid_rows;			// # of 'valid' rows in this segment (>=0)
	int pad_top_rows;			// top padding rows >=0
	int pad_bot_rows;			// # of 'bottom' pad rows >=0.
};
struct w_seg_desc {
	int n_valid;			// # of width units to copy from input (>=0)
	int dst_offset;			// offset in the dest row (i.e. to skip padding, and scatter offset)
	uint16_t n_pad_left, n_pad_right;	// left/right padding (>=0)
	int padl_offset, padr_offset;
};
struct s2x_d32_info {
	struct shape input_shape;				// copy, for validating plan
	struct tensor_format input_format;
	void * input_data;
	struct tensor_addressing tin;		    //' addressing' of the input d32 tensors
	struct tensor_addressing tout;		    //' addressing' of the output d32 tensor

	struct bsz_crop bsz;			// contains bsH, bsW, padding

	struct shape output_shape;				// calculated output shape

	int32_t wdeal_span;			// the 'span' used when scattering each input row.
	struct w_seg_desc wdesc[MAX_BSW];

	struct batch_seg_desc * bsdescs;		// pointer to bsH table entries

	uint16_t is_s2b;
	// .... //

	// how is the work split into h extents...
	struct h_extent_desc hextents[MAX_HEIGHT_EXTENTS];

	// work is always divided into in_batches*bsH*2 work units, which is at least 2 but usually at least 4.
	//
	// The even units process the first half of the height of the particular corner, and the top padding, if any;
	// the odd units process the last half of the height, and the bottom padding, if any.
	// in cases with low input height, units could have no rows to do, and often will also have no padding to do.
	// writing of left & right padding is done as part of the  core row processing.
	int n_workunits;
};

struct s2x_d32_runstate {
	struct s2x_d32_info const * info;
	int in_zero;					// zero code (for padding)
	// this is the current work unit
	volatile int cur_work_unit;
	nn_sem_t done_sem;					// for signalling hvx threads are done.
};


// setup plan for d2s (or b2s)
static int
setup_s2x_plan( struct nn_node *self, struct nn_graph * nn, int is_s2b )
{
	struct s2x_d32_info * info = (struct s2x_d32_info *)self->opaque;

	struct tensor const *input_tensor = self->inputs[0];
	if( shape_matches( &input_tensor->shape , &  info->input_shape)
			&& format_matches( &input_tensor->format, &info->input_format)
			&& input_tensor->data == info->input_data ){
		// same plan still works.
		return 0;
	}
	info->input_shape = input_tensor->shape;
	info->input_format = input_tensor->format;
	info->input_data = input_tensor->data;
	info->tin = tensor_addressing_d32( input_tensor );
	info->is_s2b =  is_s2b;

	if( !is_s2b){
		int res = get_d2s_dims( self, nn, &info->bsz, NULL);
		if( res <0) return res;
	}else{
		int res = get_b2s_dims( self, nn, &info->bsz);
		if( res <0) return res;
	}
	// validate things against the actual shape...
	// also work out the output shape.
	int32_t bsH = info->bsz.blocksize_h;
	int32_t bsW = info->bsz.blocksize_w;
	uint32_t bsProd = mulu32_sat( bsH, bsW);

	uint32_t in_height_padded = info->input_shape.height + info->bsz.crop_top + info->bsz.crop_bottom;
	uint32_t in_width_padded = info->input_shape.width + info->bsz.crop_left + info->bsz.crop_right;

	//printf("inh = %d -> %d; bsh = %d\n", (int)info->input_shape.height, (int)in_height_padded, (int)bsH );
	//printf("inw = %d -> %d; bsw = %d\n", (int)info->input_shape.width, (int)in_width_padded, (int)bsW );

	uint32_t out_height = in_height_padded/bsH;
	uint32_t out_width = in_width_padded/bsW;
	if( out_height* bsH != in_height_padded || out_width * bsW != in_width_padded){
		return errlog(nn, "padded shape not divisible by block size");
	}
	info->output_shape.height = out_height;
	info->output_shape.width = out_width;

	if( !is_s2b){
		if((info->input_shape.depth &31)!=0 )
			return errlog(nn,"s2d_d32: input depth must be divisible by 32");
		info->output_shape.depth = info->input_shape.depth * bsProd;
		info->output_shape.batches = info->input_shape.batches;
	}else{
		info->output_shape.depth = info->input_shape.depth;
		info->output_shape.batches = info->input_shape.batches * bsProd;
	}
	if( bsW >4 ) return errlog(nn,"limitation: blocksize <= 4 in width direction");

	{
		struct tensor * output_tensor = self->outputs[0];
		int h_pad = 4;
		int w_pad0 = 4;
		if( info->output_shape.height==1) h_pad = 1;
		int w_pad1 = (~info->output_shape.width+1)&3;
		int d_pad1 = (~info->output_shape.depth+1)&31;
		if( tensor_out_prepare_padded_d32( output_tensor, info->output_shape.batches,
				info->output_shape.height, h_pad, h_pad,
				info->output_shape.width, w_pad0, w_pad1,
				info->output_shape.depth, 0, d_pad1 , NN_TYPE_QUINT8 )!= 0 ){
			return errlog(nn,"failed to prepare output");
		}
		info->tout = tensor_addressing_d32( output_tensor );
	}
	if( !is_s2b){
		// 'deal' output span: distance is input nd32 * output d32 stride
		info->wdeal_span = info->tout.d32_stride * info->tin.nd32;
	}else{
		//'deal' output span: is in_batches * out_batch_stride
		info->wdeal_span = info->tout.batch_stride * info->input_shape.batches;
	}

	// create the bsdescs table, which tells us where to deal each input row, and also
	// where the vertical padding (if any) goes in the output.
	// The output is in bsH segments, each of which has its own entry;
	// for s2b, the segments are each a contiguous subrange of the entire output;
	// for s2d, the segments are interleaved within each output row.
	//
	{
		// table, if allocated, is allocated for a minimum of bsH=8.
		struct batch_seg_desc * bsdd = info->bsdescs;
		if( bsdd == NULL || bsH > 8){		// need alloc or realloc
			unsigned alloc_len = max_i32(bsH,8) *sizeof(struct batch_seg_desc);
			bsdd = (struct batch_seg_desc *)nn_realloc( bsdd, alloc_len);
			if( bsdd == NULL ) return errlog(nn,"alloc failed");
			info->bsdescs = bsdd;
		}

		int tpad = info->bsz.crop_top;
		int in_height= info->input_shape.height;
		int out_height = info->output_shape.height;
		int oh_stride = info->tout.height_stride;

		int seg_stride = info->wdeal_span* bsW;	// the 'bsH' stride
		for(int i = 0; i < bsH; i++){
			int in_h = i + tpad;
			int tpad_rows =in_h/(unsigned)bsH;
			int batchseg = in_h -tpad_rows*bsH;
			int valid_rows = (in_height + (bsH-1-i))/(unsigned)bsH;
			bsdd[i].pad_top_rows= tpad_rows;
			bsdd[i].seg_valid_rows = valid_rows;
			bsdd[i].pad_bot_rows = out_height - (valid_rows+tpad_rows);
			// pointer goes to the proper segment, and skipping the padding rows in this segment.
			bsdd[i].ptr_begin =  info->tout.data + seg_stride * batchseg  +  oh_stride * tpad_rows;
			//printf("%d: %d+%d+%d rows @ %p (seg= %d)\n", i,tpad_rows, valid_rows,  bsdd[i].pad_bot_rows, bsdd[i].ptr_begin, batchseg);
		}
	}
	// build the wdesc table, which determines how each fraction (1/bsW) of the input rows are scattered; each one also has entries
	// for left and right padding.
	{
		int lpad = info->bsz.crop_left;
		int in_width = info->input_shape.width;
		for( int i =0; i < bsW; i++){
			int in_w = i + lpad;
			int lpad_cols =in_w/(unsigned)bsW;
			int batchseg = in_w -lpad_cols*bsW;
			int valid_cols = (in_width + (bsW-1-i))/(unsigned)bsW;
			info->wdesc[i].n_pad_left = lpad_cols;
			info->wdesc[i].n_valid = valid_cols;
			info->wdesc[i].n_pad_right = out_width - (valid_cols + lpad_cols);
			// lpad offset is to the current seg
			int base_offset = batchseg * info->wdeal_span;
			info->wdesc[i].dst_offset = base_offset + 32*lpad_cols;
			info->wdesc[i].padl_offset = base_offset;
			info->wdesc[i].padr_offset  = base_offset + 32*(lpad_cols+valid_cols);
			/*printf("==>%d: %d pad @+0x%X, %d mid @+0x%X, %d pad @+0x%X\n",
					i,lpad_cols,base_offset,
					valid_cols, info->wdesc[i].dst_offset,
					info->wdesc[i].n_pad_right, info->wdesc[i].padr_offset);*/

		}
	}

	/*printf(" in_height_stride = 0x%X; in_d32_stride = 0x%X\n",
			(int)info->tin.height_stride, (int)info->tin.d32_stride);
	printf(" out_height_stride = 0x%X; out_batch_stride = 0x%X; deal_span = 0x%X\n",
			(int)info->tout.height_stride, (int)info->tout.batch_stride, (int)info->wdeal_span );
	*/
	info->n_workunits = info->input_shape.batches * bsH*2;


	/// ...
	return 0;
}

static void s2x_d32_workfunc( struct nn_graph * nn, void * rstpv);

static int
spacebatch_d32_execute( struct nn_node *self, struct nn_graph * nn  )
{
	logmsg(nn,2,"spacebatch_d32 execute. self=%p ",self);
	// set up the plan, or just check that the existing plan is ok
	int k = setup_s2x_plan( self, nn,1);
	if(k != 0) return k;

	struct s2x_d32_info * info = (struct s2x_d32_info *)self->opaque;
	//struct tensor const *input_tensor = self->inputs[0];
	//struct tensor * output_tensor = self->outputs[0];

	float in_qstep;
	struct tensor const *in_min_tensor = self->inputs[3];
	struct tensor const *in_max_tensor = self->inputs[4];

	struct s2x_d32_runstate runstate;
	runstate.info = info;
	runstate.in_zero = get_qu8_level_size_zero(
			tensor_get_float( in_min_tensor, 0),
			tensor_get_float( in_max_tensor, 0), & in_qstep);

	int n_work_units = info->n_workunits;
	runstate.cur_work_unit = 0;

	int n_threads = min_i32( n_work_units, DEPTHTOSPACE_MAX_THREADS);

	logmsg(nn,3, "%d jobs in %d threads", n_work_units, n_threads);

	nn_sem_init( & runstate.done_sem, 0);

	for( int i =0; i < n_threads; i++)
		nn_os_work_for_vector( nn, s2x_d32_workfunc, &runstate);
	tensor_copy(self->outputs[1],in_min_tensor);
	tensor_copy(self->outputs[2],in_max_tensor);

	nn_sem_wait_n_times(&runstate.done_sem, n_threads);
	logmsg(nn,2,"batchspace_d32 done. self=%p ",self);
	return 0;
}

// hvx work function; used for s2b and s2b
static void
s2x_d32_workfunc( struct nn_graph * nn, void * rstpv)
{
	struct s2x_d32_runstate * rstp = (struct s2x_d32_runstate *)rstpv;
	struct s2x_d32_info const * info = rstp->info;
	int njobs = info->n_workunits;
	int jobno;
	int bsH = info->bsz.blocksize_h;
	int bsW = info->bsz.blocksize_w;

	int in_zero= rstp->in_zero;


	int out_d32_stride = info->tout.d32_stride;
	int in_d32_stride = info->tin.d32_stride;
	int in_nd32 = info->tin.nd32;
	int in_height_stride_x_bsH = info->tin.height_stride * bsH;
	int out_height_stride = info->tout.height_stride;

	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, bsH*2);
	while (  jobno = __sync_fetch_and_add( &rstp->cur_work_unit,1),  jobno < njobs){
		int subidx = batchslice_decode_update( &bsdecode, jobno);
		int batch_idx = bsdecode.ibatch;
		int i_bsh = subidx >>1;			// 0 .. bsH-1
		struct batch_seg_desc const * bsd = &info->bsdescs[i_bsh];
		Q6_dcfetch_A(bsd);
		int is_lower = subidx & 1;
		int in_batch_offs = batch_idx * info->tin.batch_stride;
		int out_batch_offs = batch_idx * info->tout.batch_stride;
		uint8_t const * rdp = info->tin.data + in_batch_offs + i_bsh * info->tin.height_stride;
		//printf("*** batch %d bsh = %d  is_lower= %d\n", batch_idx , i_bsh, is_lower);
		uint8_t * out_base = bsd->ptr_begin + out_batch_offs;
		if( !is_lower){
			int n_top_pad = bsd->pad_top_rows;
			if( n_top_pad != 0){
				int wdeal_span = info->wdeal_span;
				int fill_cols = info->output_shape.width * 32;
				int fill_rows = in_nd32*n_top_pad;
				uint8_t * fill_begin = out_base - (out_d32_stride*fill_rows);
				for(int i =0; i < bsW; i++){
					vmemset_2d_asm( fill_begin, in_zero, fill_cols, fill_rows, out_d32_stride );
					fill_begin += wdeal_span;
				}
			}
		}
		int ncopy_rows = bsd->seg_valid_rows;
		uint8_t * out_base_0 = out_base;	// keep for lower padding
#if 1
		// split the work.
		int upper_rows = ncopy_rows>>1;
		if(is_lower){
			ncopy_rows -= upper_rows;
			out_base += upper_rows * out_height_stride;
			rdp += upper_rows * in_height_stride_x_bsH;
		}else{
			ncopy_rows = upper_rows;
		}
#else
		if( !is_lower)
#endif
		if( ncopy_rows > 0){
			//printf("processing %d rows, from %p -> %p\n", ncopy_rows, rdp, out_base );
			// work within a single d32 slice at a time, to keep the working set down.
			for( int id32 = 0; id32< in_nd32; id32++ ){
				uint8_t  * out_slc = out_base + id32*out_d32_stride;
				uint8_t const * rdp_slc = rdp + id32*in_d32_stride;

				for( int i = 0; i < bsW; i++){
					struct w_seg_desc  const *wdp  = &info->wdesc[i];
					int n_pad = wdp->n_pad_left;
					if( n_pad > 0){			// do left padding
						int nwid = n_pad *32;
						int nrows = ncopy_rows;	// rows to do padding on
						vmemset_2d_asm( out_slc + wdp->padl_offset,	// dest
								in_zero, 		// value to fill
								nwid, nrows,	   // size of rectangle
								out_d32_stride*in_nd32);	// dest pitch
					}

					// for(d32)for(bsw)for(width) copy over rows (can use aligned copy)
					// Each call to vmemcpy_2d_asm extracts one column across the entire input span

					uint8_t* out_seg = out_slc+ wdp->dst_offset;
					int n_valid = wdp->n_valid;
					if( n_valid == 0 ) continue;
					for(int j = 0; j < n_valid; j++){
						vmemcpy_2d_asm( 32, ncopy_rows,			// width, height
								out_seg  + j*32,	// output ptr
								out_height_stride,			// output stride;
								rdp_slc + 32*i + j*32*bsW,	// input pointer
								in_height_stride_x_bsH);
					}
					n_pad = wdp->n_pad_right;
					if( n_pad > 0){			// do right padding
						int nwid = n_pad *32;
						int nrows = ncopy_rows;	// rows to do padding on
						vmemset_2d_asm( out_slc + wdp->padr_offset,	// dest
								in_zero, 		// value to fill
								nwid, nrows,	   // size of rectangle
								out_d32_stride*in_nd32);	// dest pitch
					}

				}// for  i
			} // id32 loop
		}
		if(is_lower){
			int n_bot_pad = bsd->pad_bot_rows;
			if( n_bot_pad != 0){
				int wdeal_span = info->wdeal_span;
				int fill_cols = info->output_shape.width * 32;
				int fill_rows = in_nd32*n_bot_pad;
				uint8_t * fill_begin = out_base_0 + bsd->seg_valid_rows * out_height_stride;
				for(int i =0; i < bsW; i++){
					vmemset_2d_asm( fill_begin, in_zero, fill_cols, fill_rows, out_d32_stride );
					fill_begin += wdeal_span;
				}
			}
		}

		//printf(" %d  +%d + %d rows @ %p << %p\n", bsd->pad_top_rows, bsd->pad_bot_rows, bsd->pad_bot_rows, bsd->ptr_begin,rdp);

	}
	nn_sem_post( & rstp->done_sem);
}


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

static int
space2x_d32_dtor(struct nn_node *self, struct nn_graph *nn )
{
	struct  s2x_d32_info * info = (struct s2x_d32_info *)self->opaque;
	if( info != NULL){
		if( info->bsdescs != NULL )
			nn_free( info->bsdescs );
		nn_free(info);
		self->opaque = NULL;
	}
	return node_free_common( self, nn);
}

static int depthspace_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking depth2space_d32 node %p",self);
	int node_type = self->node_type;
	int is_fromspace = (node_type == OP_SpaceToBatchND_8_d32);

	// allocate info struct, all 0
	void * info = nn_calloc( 1,
			is_fromspace? sizeof(struct s2x_d32_info): sizeof(struct x2s_d32_info));
	if( info == NULL ){
		return errlog(nn,"calloc failed");
	}
	if( !is_fromspace ){
		int is_u16 = ( node_type == OP_DepthToSpace_16_d32 );
		struct x2s_d32_info * inf = (struct x2s_d32_info*)info;
		inf->is_b2s = ( node_type == OP_BatchToSpaceND_8_d32);
		inf->elbytes = is_u16? 2:1;
		inf->dtype = is_u16 ? NN_TYPE_QUINT16 :  NN_TYPE_QUINT8;
		inf->depth32_chunk = is_u16? 64:32;
	}
	self->opaque = info;
	logmsg(nn,2,"depthspace_d32 %p check OK",self);
	return 0;
}


// DepthToSpace can have 4 or 5 inputs
// BatchToSpace, SpaceToBatch must have 5.

struct nn_node_ops nn_ops_for_DepthToSpace_8_d32 = {
	.execute = depthspace_d32_execute,
	.check = depthspace_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_CHANSHUFFLE | NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
struct nn_node_ops nn_ops_for_BatchToSpaceND_8_d32 = {
	.execute = batchspace_d32_execute,
	.check = depthspace_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_CHANSHUFFLE | NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
struct nn_node_ops nn_ops_for_SpaceToBatchND_8_d32 = {
	.execute = spacebatch_d32_execute,
	.check = depthspace_d32_check,
	.ctor = node_alloc_common,
	.dtor = space2x_d32_dtor,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_CHANSHUFFLE | NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};


// 16-bit
//
struct nn_node_ops nn_ops_for_DepthToSpace_16_d32 = {
	.execute = depthspace_d32_execute,
	.check = depthspace_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_CHANSHUFFLE | NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
