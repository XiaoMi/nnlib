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
#include "nn_axis.h"


#define CHANNELSHUFFLE_D32_MAX_THREADS 2


//#define TEST_PERFORMANCE

// (this also contains QuantizedSplit_d32)
//
// 'channel shuffle' operator -d32 version
//
// Inputs are:
//     - k (number of input groups)
// 		'n_in' input tensors of shape [b,h,w,d_in]
//  	'n_out' output tensors (all will be [b,h,w,d_in*n_in/n_out] )

// we detemermine the following (all of the divisions must be exact)
//    d1 = k/n_in
//    dq = d_in/d1
//    d0 = dq/n_out
//  and thus the inputs are all of depth d0*d1*n_out  = dq*d1
//               outputs are all of depth d0*d1*n_in  = k *d0
//
//
//  The operation is equivalent to doing the following at each [b,h,w]:
//   -concatenating the 'n_in' inputs to a total depth of k*dq
//   - consider this as k rows of dq
//   - reorder (transpose) to dq rows of k
//   - slice to n_out equal outputs of size [dq/nout*k]  or d0*k
//
//
//-----------------------
//  QuantizedChannelShuffle_8_d32
//-----------------------
//
//   3*n_in+1 inputs:
//     0 - scalar int - 'k' parameter
//     1..n_in            		d32 qu8 tensors of shape [b,h,w,d_in]
//     n_in+1 .. 2*n_in      	float scalar: minima for the inputs
//     n_in*k+1 .. 3*n_in    	float scalar: maxima for the inputs
//   n_out+2 outputs:
//     0..n_out-1				d32 qu8 tensors of shape [b,h,w,d_in*n_in/n_out]
//	     n_out					float scalar: minimum (for all outputs)
//		 n_out+1				float scalar: maximum (for all outputs)
//
//-----------------------
//  QuantizedSplit_d32
//-----------------------
//
//   4 inputs:
//     0 - scalar int - dim to split on (must be 3, for this d32 op)
//     1            	d32 qu8 tensor of shape [b,h,w,d_in]
//     2 				float scalar: minimum for the inputs
//     3			   	float scalar: maximum for the inputs
//
//   n_out+2 outputs:
//     0..n_out-1				d32 qu8 tensors of shape [b,h,w,d_in/n_out]
//	     n_out					float scalar: minimum (for all outputs)
//		 n_out+1				float scalar: maximum (for all outputs)
//
// NOTE: this also support a convention where if input 0 is wired to the same source as input 1,
// instead of to int scalar, then the split is on dim 3.
//

// when scaling qu8:
//    out =  (in*scale+ offs*256 + 16384) >> 15   (saturated to u8)
// There is a chanshuf_qu8_scale for every input;
// when an input doesn't require scaling, then scale < 0
//
struct chanshuf_qu8_scale{
	float in_min, in_max;
	int scale;			// 0 .. 32767
	int offs;			// 0 .. 32576
};
// this assumes that all_min <= in_min < in_max <= all_max.
// when scale is set to 32767, then offs is always 0 and the scaling is a 'no-op'.
// (some scaling situations can arise which are not exactly 1:1 based on the ranges;
// but in which 0 scales to 0 and 255 to 255; these are mapped to the scale=32767, offs = 0 case).
//

static void setup_scale_for_input( struct chanshuf_qu8_scale *sclp, float all_min, float all_max)
{
	float all_range = all_max - all_min;
	float in_range = sclp->in_max - sclp->in_min;
	int scaleval = 32768.0f * in_range / all_range + 0.5f;	// value of scale
	int offs = 0;
	if( scaleval >= 32768){
		scaleval = 32767;		// mark as 'no change'
	}else{
		offs = saturate_i16(roundf_i32( (sclp->in_min-all_min)*(255.0f* 128.0f)/all_range));
		if( scaleval >= 32641){	// could still be 'no-op'
			// if 0->0 and 255->255, it's a no-op
			if( offs < 64 &&  255*scaleval + 256*offs >= 0x7f4000){
				scaleval = 32767;
				offs = 0;
			}
		}
	}
	sclp->scale = scaleval;
	sclp->offs = offs;
}

//
// This are optimized for particular cases, where the outputs are obtained by
// interleaving K = 2 or 4 inputs in a straightforward way. To make this happen,
//     - the 'inputs' may actually be individual inputs, or each may be a 'slice' of an input
//              (e.g. a single input may be sliced 4 ways, two inputs each sliced in 2, to get K=4)
//
// in general each input will be sliced into num_out * k/num_in equal parts; this is a total of num_out*k parts,
// for each output, 'k' of those are combined.
//
// When k=4, we have either 1,2 or 4 inputs, and we have an additional constraint that all inputs
// must have the same batch_stride, height_stride and d32_stride. This reduces the # of live registers needed in the
// inner loop. Since the operator is converted from a 'flat tensor' op with all input tensors being
// of the same shape, there is no reason they should not all have the same padding.
//
// We also have a constraint: when there is more than one input, all inputs must have the same width_before_padding (mod 4).
// The output padding will be the same as the first input's width_padding_before.
// The 'depth_before' padding must be 0 on all inputs.
//
// The slicing is done with 'struct tensor_slice_d32'
// Net result, is we have a certain number of operations, each of which copies data from K input slices
// to one output slices. A given operation or may not involve range scaling, according to which inputs
// is involved. When all inputs to a particular operation are unscaled, a different loop will be used.
//
//
#define CHANSHUF_MAX_K 4			// largest K supported
//
struct chanshuf_d32_suboperation ;
struct chanshuf_d32_runstate;
typedef void  (*chanshuf_exec_funcp)( struct chanshuf_d32_runstate *rstp,
				struct chanshuf_d32_suboperation const * subop,
				int ibatch,								// batch index to process
				int irow);

// these are codes that are used to index a table to pick the proper exec function.
// A table index is the sum of
//     BASE_K   (for the K in effect)
//   + NEED_SCALING  if the operation needs scaling
//  plus an add depending on the channel alignment; this can change between ouput buffers
//     + 0                   .. if all of the inputs are 0-depth aligned.
//     + NEED_ALIGN_SINGLE   .. if all of the inputs are present in the first d32 slice (but may have different aligns)
// or  + NEED_ALIGN_FULL     .. all other cases.
//
//
static const chanshuf_exec_funcp exec_function_table[18];			// the table of function pointers.
enum {
	chanshuf_funcsel_BASE_K_1 = 0,				/// the  base for K = 1 ('split')
	chanshuf_funcsel_BASE_K_2 = 6,				/// the  base for K = 2
	chanshuf_funcsel_BASE_K_4 = 12,				/// the  base for K = 4
	chanshuf_funcsel_NEED_SCALING =3,			/// add this if scaling is needed
	chanshuf_funcsel_NEED_ALIGN_SINGLE=1,		/// add this for 'single' align
	chanshuf_funcsel_NEED_ALIGN_FULL=2,			/// or add this if full align needed
};

struct chanshuf_d32_suboperation {
	struct tensor_addressing tout;						// the output slice
	struct tensor_slice_d32 in_slice[CHANSHUF_MAX_K];		// the input slices
	chanshuf_exec_funcp exec_fp;							// the function to use.
};


struct chanshuf_d32_runstate {

	int num_out;									// number of outputs
	int vecs_wide;									// # of vectors to cover width
	int height;										// height of array
	int height_chunksize;							// 1..height : # of rows per slice when slicing
	int height_chunks;								// # of chunks in height dim (all are height_chunksize, except last)
	struct chanshuf_d32_suboperation * subop;		// pointer to array [outputs]
	uint16_t k_parm;								// number to interleave
	uint16_t need_scaling;							// 0 if no input->output scaling needed.
	uint16_t height_chunk_shift;						// log2 of( height_chunks)

	volatile int cur_jobno;					// current job index
	int total_jobs;							// total # of jobs

	nn_sem_t done_sem;						// set when threads are done.

	// if scaling is needed, the below are the 'k' scaling parms. 'k' records will be filled
	// out even if the number of inputs is less than k. if need_scaling=0, these are not used.
	struct inscaling_desc {
		int:0;		// 32 aligned
		int16_t scale;
		int16_t offset;
	} inscaling[CHANSHUF_MAX_K];
	// scale/offset pre-packed for the hvx loop
	// There are 4 of each.
	uint64_t:0;				// make this 8-byte aligned
	int32_t scale_lo;		// 4 lo-bytes of scale
	int32_t scale_hi;		// 4 hi-bytes of scale
	int32_t offs_02, offs_13;	// 4 16-bit offsets

};
//
static void chanshuf_worker_func( struct nn_graph * nn, void * rstpv);
static int common_chanshuffle_split_d32_execute( struct nn_node *self, struct nn_graph *nn,
		int k_parm, int dq_parm, int d0_parm, int d1_parm , int num_in);
//
//
static int channelshuffle_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *k_tensor = self->inputs[0];
	const struct tensor *in_tensor0 = self->inputs[1];

	int num_in = (unsigned)(self->n_inputs-1)/3;
	int num_out = self->n_outputs-2;
	int k_parm = tensor_get_int32( k_tensor, 0);

	logmsg(nn,2,"chanshuffle_d32 execute. self=%p ",self);

	unsigned depth_in = in_tensor0->shape.depth;
	int d0_parm, d1_parm, dq_parm;
/*printf("in_tens: data = %p w=%d d=%d top_pad = %d left_pad = %d\n",
		in_tensors[0]->data, (int)in_tensors[0]->shape.width,(int)in_tensors[0]->shape.depth,
		in_tensors[0]->format.height_pad[0],in_tensors[0]->format.width_pad[0]);*/

	if(  k_parm <= 1
				|| ( d1_parm = (unsigned)k_parm/num_in,  d1_parm*num_in != k_parm )
				|| ( dq_parm = depth_in/d1_parm,   d1_parm*dq_parm != depth_in)
				|| ( d0_parm = (unsigned)dq_parm/num_out,   d0_parm*num_out != dq_parm)){
		return errlog(nn, "bad chanshuffle parms: k=%d, in_depth=%d, n_in = %d, n_out = %d\n",
				k_parm, depth_in, num_in, num_out );
	}
	// (for now, don't convert to d32 unless k is a value supported!)

	if( k_parm != 2 &&  k_parm !=4){
		return errlog(nn,"chanshuf_d32 : can't use k= %d\n", k_parm);
	}

	return common_chanshuffle_split_d32_execute( self, nn, k_parm, dq_parm, d0_parm, d1_parm, num_in );
}

//
// entry point for 'split_d32'; if the split dimension is 3, this is a chanshuffle with k=1
// and constrained to have one input
//
static int split_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor *in_tensor0 = self->inputs[1];

	int num_out = self->n_outputs-2;

	logmsg(nn,2,"split_d32 execute. self=%p ",self);

	// split is like a degenerate channelshuffle with k= n_in = d1 = 1
	// so depth_in = dq = d0* n_out;
	// depth_out = d0

	// if input 0 is wired to same source as input1, consider the dimno to be 3.
	// ** when converting a Split_8 to Split8_d32, this will need special handling **
	int32_t dim_no =3;
	if( dim_tensor != in_tensor0) {
		dim_no = tensor_get_int32( dim_tensor, 0);
		int res = handle_negative_axes(nn, &dim_no, 1);
    	if (res)
        	return errlog(nn, "split dimension out of range");
		// ** for now we don't convert split to d32, unless it's on dim 3.
		if( dim_no != 3)
			return errlog(nn, "split_d32 must be on dim 3");
	}

	unsigned depth_in = in_tensor0->shape.depth;
	int d0_parm = depth_in/num_out;
	if( num_out*d0_parm != depth_in){
		return errlog(nn, "bad split_d32 parms: can't divide %d by %d\n", (int)depth_in, num_out);
	}
	return common_chanshuffle_split_d32_execute( self, nn,
			1, 			/*k_parm*/
			d0_parm, 	/*dq_parm*/
			d0_parm,
			1, 			/*d1_parm*/
			1); 		/*num_in*/
}




static int
common_chanshuffle_split_d32_execute( struct nn_node *self, struct nn_graph *nn,
		int k_parm, int dq_parm, int d0_parm, int d1_parm , int num_in)
{
	// check all the input shapes match
	const struct tensor **in_tensors = &self->inputs[1];
	struct tensor const *in_tensor0 = in_tensors[0];
	struct tensor ** out_tensors = &self->outputs[0];
	int num_out = self->n_outputs-2;

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	int width_before_padding = in_tensor0->format.width_pad[0];
	int width_before_padding_mod4 = width_before_padding & 3;

	for( int i = 0; i < num_in; i++){
		struct tensor const *ten = in_tensors[i];
		if( ten->format.depth_pad[0] != 0)
			return errlog(nn,"channelshuf input #%d has nonzero depth_before padding", i+1);
		if( i > 0){
			if( ! shape_matches( &in_tensor0->shape, &ten->shape )){
				return errlog(nn,"channelshuf input #%d does not match shape of #1", i+1);
			}
			if( (ten->format.width_pad[0] &3) != width_before_padding_mod4 )
				return errlog(nn,"channelshuf input #%d has incompatible width_before padding", i+1);
		}
	}
	nn_scratch_reset(nn);

	struct shape out_shape = in_tensor0->shape;
	out_shape.depth = d0_parm * d1_parm * num_in;
	logmsg(nn,4,"n_in = %d; n_out = %d; d0,d1 = %d,%d; in_depth = %d, out_depth=%d", 
		num_in, num_out, d0_parm, d1_parm, (int)in_tensor0->shape.depth, (int)out_shape.depth );

	//
	// prepare all the tensor outputs
	//
	int out_depth_after_pad = (-out_shape.depth)&31;
	for( int i = 0; i < num_out; i++ ){
		int k = tensor_out_prepare_padded_d32(out_tensors[i],
			out_shape.batches,
			out_shape.height,  in_tensor0->format.height_pad[0], in_tensor0->format.height_pad[1],
			out_shape.width, width_before_padding,  in_tensor0->format.width_pad[1],
			out_shape.depth, 0,out_depth_after_pad, NN_TYPE_QUINT8);
		if( k!= 0) return errlog( nn, "failed to allocate channelshuf output #%d, size [%d,%d,%d,%d]",
					i, (int)out_shape.batches, (int)out_shape.height, (int)out_shape.width, (int)out_shape.depth);
	}
	// find the range of all the ranges
	//
	float min_overall=0.0f, max_overall = 0.0f;
	struct chanshuf_qu8_scale * qu8_scales = NULL;

	const struct tensor **in_min_tensors = &self->inputs[1+num_in];
	const struct tensor **in_max_tensors = in_min_tensors + num_in;

	qu8_scales = nn_scratch_alloc(nn,num_in * sizeof( struct chanshuf_qu8_scale) );

	min_overall = fminf(0.0f, tensor_get_float( in_min_tensors[0], 0) );
	max_overall = tensor_get_float( in_max_tensors[0], 0);
	qu8_scales[0].in_min = min_overall;
	qu8_scales[0].in_max = max_overall;
	qu8_scales[0].scale = 32767;	// in case num_in = 1
	qu8_scales[0].offs = 0;


	struct chanshuf_d32_runstate runstate;

	runstate.num_out = num_out;
	runstate.need_scaling = 0;
	runstate.vecs_wide = (out_shape.width + width_before_padding_mod4 + 3) >>2;
	runstate.height = out_shape.height;

	{
		int height_chunksize = out_shape.height;
		int height_chunk_shift = 0;
		if( height_chunksize >= 8){	// do in 2 parts
			height_chunk_shift = 1;
			height_chunksize = (height_chunksize+1)>>1;
		}
		runstate.height_chunksize = height_chunksize;
		runstate.height_chunks = 1 << height_chunk_shift;
		runstate.height_chunk_shift = height_chunk_shift;

	}



	if( num_in > 1){
		float tmn = min_overall;
		float tmx = max_overall;
		for( int k = 1; k < num_in; k++){
			float in_min = tensor_get_float( in_min_tensors[k], 0);
			float in_max = tensor_get_float( in_max_tensors[k], 0);
			qu8_scales[k].in_min = in_min;
			qu8_scales[k].in_max = in_max;
			tmn = fminf( tmn, in_min);
			tmx = fmaxf( tmx, in_max);
		}
		min_overall = tmn;
		max_overall = tmx;
		if( min_overall != 0.0f)	// correct range...
			adjust_minmax_for_zero( &min_overall,&max_overall);

		// set up the per-input scales
		for( int j = 0; j < num_in; j++){
			setup_scale_for_input( &qu8_scales[j], min_overall, max_overall);
			logmsg(nn,4,"input %d: (%f..%f) scale = %d, offs = %d",
				j, qu8_scales[j].in_min, qu8_scales[j].in_max,
					qu8_scales[j].scale, qu8_scales[j].offs);
			if( qu8_scales[j].scale != 32767){
				runstate.need_scaling = 1;
			}
		}
	}
	tensor_set_single_float( self->outputs[num_out], min_overall );
	tensor_set_single_float( self->outputs[num_out+1], max_overall );



	// this is the interleave factor; should be 2 or 4 for efficient operation.
	//
	runstate.k_parm = k_parm;

	//
	// set up the runstate 'inscaling' values according to k and num_in
	// (only when needed; thus num_in > 1).
	//     num_in     k
	//       2         2                    interleave 0/1/0/1
	//       2         4                    interleave 0/0/1/1
	//       4         4					1:1 mapping 0/1/2/3
	//
	if( runstate.need_scaling)	// => num_input > 1
	{
		struct inscaling_desc tmp0 = { .scale = qu8_scales[0].scale, .offset = qu8_scales[0].offs};
		struct inscaling_desc tmp1 = { .scale = qu8_scales[1].scale, .offset = qu8_scales[1].offs};
		runstate.inscaling[0] = tmp0;	// always
		runstate.inscaling[1] = tmp1;	// usually
		if( num_in ==2  ){
			runstate.inscaling[3] = tmp1;
			if( k_parm == 2 ){
				runstate.inscaling[2] = tmp0;
			}else{
				runstate.inscaling[1] = tmp0;
				runstate.inscaling[2] = tmp1;	// 0,0,1,1
			}
		}else{ 	// 4->4
			runstate.inscaling[2].scale = qu8_scales[2].scale;
			runstate.inscaling[2].offset = qu8_scales[2].offs;
			runstate.inscaling[3].scale = qu8_scales[3].scale;
			runstate.inscaling[3].offset = qu8_scales[3].offs;
		}
		// now, pack the values for the hvx loop
		// offs_02, offs_13: two offset each
		// scale_lo: the lsbs of the 4 scales
		// scale_hi: the msbs of the 4 scales
		runstate.offs_02 = Q6_R_combine_RlRl( runstate.inscaling[2].offset, runstate.inscaling[0].offset);
		runstate.offs_13 = Q6_R_combine_RlRl( runstate.inscaling[3].offset, runstate.inscaling[1].offset);

		int scl_02 = Q6_R_combine_RlRl( runstate.inscaling[2].scale, runstate.inscaling[0].scale);
		int scl_13 = Q6_R_combine_RlRl( runstate.inscaling[3].scale, runstate.inscaling[1].scale);
		int64_t scl_02x = Q6_P_combine_RR( Q6_R_swiz_R(scl_13), scl_02); // 1L 1H 3L 3H 2H 2L 0H 0L
		int64_t scl_13x = Q6_P_combine_RR( Q6_R_swiz_R(scl_02), scl_13); // 0L 0H 2L 2H 3H 3L 1H 1L
		uint64_t shuffled = Q6_P_shuffeb_PP( scl_13x, scl_02x);
		// lo word is; 3L 2L 1L 0L  (msb to lsb)
		// hi word is: 0H 1H 2H 3H  (msn to lsb)
		runstate.scale_lo = (int32_t)shuffled;
		runstate.scale_hi = Q6_R_swiz_R( (uint32_t)(shuffled>>32));
	}



	// we need to divide the num_in input regions collectively into num_out * k regions,
	// each of width d0, and distribute these among the chanshuf_d32_suboperations
	//
	struct chanshuf_d32_suboperation * shufoptab = nn_scratch_alloc( nn, num_out  * sizeof(struct chanshuf_d32_suboperation));
	if( shufoptab == NULL){
		return errlog(nn,"alloc");
	}
	// output addressing
	for(int i = 0; i < num_out; i++){
		struct chanshuf_d32_suboperation * shufop = &shufoptab[i];
		shufop->tout = tensor_addressing_d32( out_tensors[i]);
		shufop->tout.data -= 32*width_before_padding_mod4;	// align to vector
	}

	{
		int ipos=0;					// current output
		int jpos=0;					// 0..k-1 ; input to the output
		int slice_per_in = d1_parm *num_out;		// # of slices into which each input is to be cut.
		struct chanshuf_d32_suboperation * shufop = shufoptab;
		for( int inp_no = 0; inp_no < num_in; inp_no ++ ){
			struct tensor const * tensor_in = in_tensors[inp_no];
			struct tensor_slice_d32 inslice;
			tensor_slice_from_tensor_d32( &inslice, tensor_in);	// make a slice of the whole thing
			// avoid false reports of 'incompatible strides' when b=1 or h=1
			if( out_shape.batches == 1) inslice.batch_stride = 0;
			if( out_shape.height == 1) inslice.height_stride = 0;
			int dpos = 0;

			for( int i =0; i < slice_per_in; i++ ){
				if( jpos >= k_parm) return errlog(nn,"bad split calc");
				// slice dpos ... dpos + d0-1  from the input.
				struct tensor_slice_d32 * cur_slice = &shufop->in_slice[jpos];
				tensor_slice_on_dimension( cur_slice, &inslice, 3, dpos, d0_parm );
				// compensate all for width padding (align pointer)
				cur_slice->data -= 32*width_before_padding_mod4;
				dpos += d0_parm;
				// advance to next output, cycling back each time we run out (and bumping jpos)
				++shufop;
				if( ++ipos >= num_out){
					ipos = 0;
					shufop = shufoptab;
					jpos ++;
				}
			}
		}
		if( ipos != 0 || jpos != k_parm) return errlog(nn,"bad split calc");
	}
	// if more than one input, we need to make sure they have compatible strides.
	// check the first subop
	if( num_in > 1){
		struct chanshuf_d32_suboperation * shufop = &shufoptab[0];
		int32_t bstride = shufop->in_slice[0].batch_stride;
		int32_t hstride = shufop->in_slice[0].height_stride;
		int32_t d32stride = shufop->in_slice[0].d32_stride;
		for(int i = 1; i < k_parm; i++){
			if(  shufop->in_slice[i].batch_stride  !=  bstride
			   ||	shufop->in_slice[i].height_stride != hstride
			  || shufop->in_slice[i].d32_stride !=d32stride ){
				return errlog(nn,"inputs have incompatible padding");
			}
		}

	}
	// now, for each output, choose an exec function
	{
		int base_func_sel;
		if( k_parm == 4) base_func_sel = chanshuf_funcsel_BASE_K_4;
		else if( k_parm == 2) base_func_sel = chanshuf_funcsel_BASE_K_2;
		else if (k_parm == 1)  base_func_sel = chanshuf_funcsel_BASE_K_1;	// 'split'
		else return errlog(nn,"? impossible k");

		int max_for_single_align= 32-d0_parm;		// could be <=0, in which case 'single align' won't ever be used.
		if( runstate.need_scaling ) base_func_sel += chanshuf_funcsel_NEED_SCALING;
		// for each output, find the maximum depth_before_pad
		for (int i = 0; i < num_out; i++){
			struct chanshuf_d32_suboperation * shufop = &shufoptab[i];
			/* printf("output %d: dbef= %d %d %d %d\n", i, shufop->in_slice[0].depth_pad_before,
					shufop->in_slice[1].depth_pad_before,shufop->in_slice[2].depth_pad_before,shufop->in_slice[3].depth_pad_before); */
			int dpmax = shufop->in_slice[0].depth_pad_before;
			for( int j = 1; j < k_parm; j++) dpmax = max_i32(dpmax, shufop->in_slice[j].depth_pad_before);
			// if all are 0, we don't need align. If all are < d1thresh, then we can use 'single' alignment.
			int func_sel = base_func_sel;
			if( dpmax > 0) {
				func_sel += (dpmax <= max_for_single_align)? chanshuf_funcsel_NEED_ALIGN_SINGLE : chanshuf_funcsel_NEED_ALIGN_FULL;
			}
			shufop->exec_fp = exec_function_table[ func_sel];
		}
	}
	runstate.subop = shufoptab;
	// OK, we can finally start executing something. For threads, the operations are broken as follows:
	// (1) inner : mux across outputs. generally the same outputs will read some of the same input data, so we
	//  want a relatively small chunk here.
	// (2) middle: vertical split of the height dimension into chunks. The number of chunks is a power of 2.
	// (3) outer: batches.
	// the 'outer and middle' are done as a 'batch' index, so we want the 'middle' to be powers of 2 so they can
	// be separated via a shift.
	//


	runstate.cur_jobno = 0;
	runstate.total_jobs = (out_shape.batches << runstate.height_chunk_shift) * num_out;

	int n_threads = min_i32( CHANNELSHUFFLE_D32_MAX_THREADS, runstate.total_jobs);
	nn_sem_init( &runstate.done_sem, 0);

	for( int i = 0; i < n_threads; i++){
		nn_os_work_for_vector( nn, chanshuf_worker_func, &runstate);
	}
	nn_sem_wait_n_times( & runstate.done_sem, n_threads);

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	logmsg(nn,0,"%s cycles = %d   %d in. %d out , total elements = %d) #thr=%d\n",
			k_parm==1? "split":"chanshuffle",
			(end_time-start_time), num_in, num_out,
			(int)(shape_element_count(&out_shape)*num_out), n_threads);
#endif

	logmsg(nn,2,"%s_d32 done. self=%p ",k_parm==1? "split":"chanshuffle",self);

	return 0;
}

// worker thread:
// run the next job, until all are done.
//

static void
chanshuf_worker_func( struct nn_graph * nn, void * rstpv)
{
	struct chanshuf_d32_runstate * rstp = (struct chanshuf_d32_runstate *)rstpv;


	int num_out = rstp->num_out;
	int jobno;
	int numjobs = rstp->total_jobs;
	int hchunk_shift = rstp->height_chunk_shift;
	int hchunk_size = rstp->height_chunksize;

	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, num_out);

	while(  jobno = __sync_fetch_and_add( &rstp->cur_jobno, 1),  jobno < numjobs) {
		int ioutsel = batchslice_decode_update( &bsdecode, jobno);	// jobno % num_out = output #
		int ibat = bsdecode.ibatch;					// jobno/outsel
		// reduce bat ibat into batch, hchunks
		int ihchunk = ibat & (  (1<<hchunk_shift)-1);
		ibat >>= hchunk_shift;

		logmsg(nn,2, "job %d: out=%d, hchunk = %d, batch = %d" , jobno, ioutsel, ihchunk, ibat);

		// select the op...
		struct chanshuf_d32_suboperation const *subop =  &rstp->subop[ioutsel];
		int irow = ihchunk * hchunk_size;		//row to start at

		(*subop->exec_fp)( rstp, subop, ibat, irow);	// run it
	}

	nn_sem_post( &rstp->done_sem);

}




// This does ub->ub scaling on 128 values, supporting
// a different scale/offset for each of 4 lanes
// within each group of 4.
//
// 'scale' is the 16-bit scale used in step (1)
// 'offs'  is 16-bit offset used in step (2) below
//
// nominally it's
///   saturate_u8(   (inp * scale + offs*256 +  16K) >> 15)
//
// This is actually done as
//   (1) prod =  inp * scale >> 8		 (range 0.. 32640 )
//   (2) p2 = addh_sat( prod, offset)
//   (3)    saturate_u8(   (p2 + 64) >> 7 )
//
//   and step (1) is done as
//      (1a)   p0 = inp * scale.lobyte				[uu mul]
//      (1b)   prod = (p0>>8) + in * scale.hibyte	[lsr and us mul]
//
// The computation also supports scale < 0.
// parameters are:
//   scale_lo: this is the low 8 bits of the four scale values, one per byte lane
//   scale_hi: this is the hi 8 bits of the four scale values, one per byte lane
//  off_02:  has offset 0 in low word, offset 1 in high word
//  off_13:  has offset 1 in low word, offset 3 in high word
//
//
static inline HVX_Vector __attribute__((unused))
do_scale_ub( HVX_Vector vin, int scale_lo, int scale_hi, int offs_02, int offs_13 )
{
	// (loop invariant)
	HVX_Vector voffs_02 = Q6_V_vsplat_R(offs_02);
	HVX_Vector voffs_13 = Q6_V_vsplat_R(offs_13);

	// find lo prod
	HVX_VectorPair vprodlo =  Q6_Wuh_vmpy_VubRub( vin, scale_lo);
	// >> 8 bits using vshuffo
	HVX_Vector vprodlo_0 = Q6_Vb_vshuffo_VbVb( Q6_V_vzero(), Q6_V_lo_W(vprodlo));
	HVX_Vector vprodlo_1 = Q6_Vb_vshuffo_VbVb( Q6_V_vzero(), Q6_V_hi_W(vprodlo));

	// add the high prod
	HVX_VectorPair vprod = Q6_Wh_vmpyacc_WhVubRb( Q6_W_vcombine_VV( vprodlo_1, vprodlo_0),
			vin, scale_hi );
	// add the offset and >>7 with sat to u8
	return Q6_Vub_vasr_VhVhR_rnd_sat(
			Q6_Vh_vadd_VhVh_sat(  Q6_V_hi_W( vprod), voffs_13 ),
			Q6_Vh_vadd_VhVh_sat(  Q6_V_lo_W( vprod), voffs_02 ),
			7 );
}


static inline HVX_Vector  *
ptr_advance_bytes(HVX_Vector * p, int bytes )
{
	return (HVX_Vector *)( (uint8_t *)p + bytes);
}
static inline HVX_Vector const *
ptrk_advance_bytes(HVX_Vector const * p, int bytes )
{
	return (HVX_Vector const*)( (uint8_t const *)p + bytes);
}
//
// inner hvx loop for K=4  (will also do K=2)
// This function is inlined with different combinations of k_val, need_scaling
// and alignment_mode (which should be constants in each "call").
//
// need_scaling = 0   if no scaling is needed
//                1   if all 4 inputs are scaled
//
// alignment mode refers to the depth alignment; an input is aligned if the required
// data starts at the beginning of a d32 slice
//
//  aligmnent_mode = 0 : all 4 inputs are aligned
//                 = 1 : all 4 inputs may be misaligned; but d32_loops ==1 (out_depth <= 128)
// 					 and we can do the align with a vror (all of the inputs are present in the first depth slice).
//                 = 2 : all 4 inputs have their own alignment.
//
// The scaling is done on all four inputs after interleaving; this
// reduces the number of invariant registers needed to hold scaling values
// the 'irow' value is the start row for the operation, which is a multiple
// if rstp->height_chunk, and is always less than rstp->height. The number of rows
// processed is rstp->height_chunk, or (rstp->height-irow), whichever is smaller.
//
//  This has been set up to assume alignment_mode ==2; there's another
// function for alignment_mode = 0 or 1, which does the width in the inner loop instead
// of the depth.
//
static inline void __attribute__((always_inline))
channelshuf_d32_k24_align2_template( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop,
		int ibatch,								// batch index to process
		int irow,								// first row to process
		int k_val,								// 2 or 4
		int need_scaling)
{
	int vwide = rstp->vecs_wide;
	int nd32_out = subop->tout.nd32;	// # of output d32 units
	// how many loops to get through the depth dimension?
	int d32_loops = (k_val==4)? ( (nd32_out+3)>>2) : ((nd32_out+1)>>1);

	if( d32_loops < 1 || vwide < 1) return;	// eliminate checks in the loops for these

	int height = rstp->height;
	int nrows = min_i32( rstp->height_chunksize, height-irow);


	int out_height_stride = subop->tout.height_stride;
	int out_d32_stride = subop->tout.d32_stride;
	// find the start addresses
	// (assuming all inputs have same batch/height/d32 stride, just look at the first one).
	int in_height_stride = subop->in_slice[0].height_stride;
	int in_d32_stride = subop->in_slice[0].d32_stride;
	unsigned in_offs = ibatch * subop->in_slice[0].batch_stride + irow * in_height_stride;

	HVX_Vector *rvoutp = (HVX_Vector*)(subop->tout.data + ibatch * subop->tout.batch_stride + irow * out_height_stride );
	HVX_Vector const * rvinp0 = (HVX_Vector const *)(subop->in_slice[0].data + in_offs);
	HVX_Vector const * rvinp1 = (HVX_Vector const *)(subop->in_slice[1].data + in_offs);
	HVX_Vector const * rvinp2 = (HVX_Vector const *)(subop->in_slice[2].data + in_offs);
	HVX_Vector const * rvinp3 = (HVX_Vector const *)(subop->in_slice[3].data + in_offs);

/*
printf("batch %d row %d: %d rows of %d  vecs; nd32=%d al=%d ns=%d iptrs =%p %p %p %p bs=%d hs=%d d32s=%d, optr = %p  bs=%d hs=%d d32s=%d\n",
		ibatch, irow, nrows, vwide, nd32_out, alignment_mode, need_scaling,
		rvinp0, rvinp1, rvinp2, rvinp3, (int)subop->in_slice[0].batch_stride,  in_height_stride, in_d32_stride,
		rvoutp, (int)subop->tout.batch_stride, out_height_stride, out_d32_stride );
*/
	// these are only used if need_scaling !=0
	int scale_lo = 0, scale_hi = 0, offs_02 = 0, offs_13 = 0;
	if( need_scaling){
		scale_lo = rstp->scale_lo;
		scale_hi = rstp->scale_hi;
		offs_02 = rstp->offs_02;
		offs_13 = rstp->offs_13;
	}
	// when k = 2, there are a number of variables in the setup which are never
	// used; we just let the compiler fold them away.


	HVX_Vector vin_prev0, vin_prev1, vin_prev2, vin_prev3;
	vin_prev0 = vin_prev1 = vin_prev2 = vin_prev3 = Q6_V_vzero();

	HVX_VectorPred qalign0, qalign1, qalign2, qalign3;
	HVX_Vector aligndelt0, aligndelt1, aligndelt2, aligndelt3;

	{
		// full align, spanning from two vectors. The execution is
		//  vout = Q6_V_delta_VV(   Q6_V_vmux_QVV( valign, vecin1, vecin0),  aligndelt)
		// The 'mux' selects the values needed from each input, and the 'delta' rotates within each 32-byte
		// lane.
		HVX_Vector count32 = *(HVX_Vector const*)const_Count32;	 // {0,1, ..31, 0,1, ..31, 0, .. 31, 0.. 31 }
		HVX_Vector k31 = q6op_Vb_vsplat_R(31);
		// to get the vdelta code for rotate-down-by-n, each lane needs to be ((i-k)&31)^i
		// where i is the lane 0..31 (The pattern is repeated 4 times). The mux control is i < k.
		int algn = subop->in_slice[0].depth_pad_before;	// 0..31
		HVX_Vector valgn = q6op_Vb_vsplat_R( algn);	// copy to all bytes
		aligndelt0 = Q6_V_vxor_VV( Q6_V_vand_VV(Q6_Vb_vsub_VbVb(count32,valgn),k31),count32);
		qalign0 = Q6_Q_vcmp_gt_VubVub( valgn, count32);	// first 'al0' slots in each lane to come from next d32

		algn = subop->in_slice[1].depth_pad_before;	// 0..31
		valgn = q6op_Vb_vsplat_R( algn);
		aligndelt1 = Q6_V_vxor_VV( Q6_V_vand_VV(Q6_Vb_vsub_VbVb(count32,valgn),k31),count32);
		qalign1 = Q6_Q_vcmp_gt_VubVub( valgn, count32);

		algn = subop->in_slice[2].depth_pad_before;	// 0..31
		valgn = q6op_Vb_vsplat_R( algn);
		aligndelt2 = Q6_V_vxor_VV( Q6_V_vand_VV(Q6_Vb_vsub_VbVb(count32,valgn),k31),count32);
		qalign2 = Q6_Q_vcmp_gt_VubVub( valgn, count32);

		algn = subop->in_slice[3].depth_pad_before;	// 0..31
		valgn = q6op_Vb_vsplat_R( algn);
		aligndelt3 = Q6_V_vxor_VV( Q6_V_vand_VV(Q6_Vb_vsub_VbVb(count32,valgn),k31),count32);
		qalign3 = Q6_Q_vcmp_gt_VubVub( valgn, count32);
	}
	// height loop

	for( int ir = 0; ir < nrows; ir++){
		for (int ic = 0; ic < vwide; ic++){
			// get pointers to start of each 'depth run'
			HVX_Vector *voutp = rvoutp + ic;
			HVX_Vector const * vinp0 = rvinp0 + ic;
			HVX_Vector const * vinp1 = rvinp1 + ic;
			HVX_Vector const * vinp2 = rvinp2 + ic;
			HVX_Vector const * vinp3 = rvinp3 + ic;

			// preload first from each input
			vin_prev0 = *vinp0;  vinp0 = ptrk_advance_bytes( vinp0, in_d32_stride);
			vin_prev1 = *vinp1;  vinp1 = ptrk_advance_bytes( vinp1, in_d32_stride);
			if( k_val > 2){
				vin_prev2 = *vinp2;  vinp2 = ptrk_advance_bytes( vinp2, in_d32_stride);
				vin_prev3 = *vinp3;  vinp3 = ptrk_advance_bytes( vinp3, in_d32_stride);
			}
			// now loop down the depth dimension
			int nd32_remain = nd32_out;
			HVX_Vector vout0,vout1,vout2,vout3;
			for( int id32 = 0; id32 < d32_loops; id32++){
				// (1) get 4 new vectors
				HVX_Vector vin0 = *vinp0;  vinp0 = ptrk_advance_bytes( vinp0, in_d32_stride);
				HVX_Vector vin1 = *vinp1;  vinp1 = ptrk_advance_bytes( vinp1, in_d32_stride);
				HVX_Vector vin2 , vin3;
				if( k_val > 2){
					vin2 = *vinp2;  vinp2 = ptrk_advance_bytes( vinp2, in_d32_stride);
					vin3 = *vinp3;  vinp3 = ptrk_advance_bytes( vinp3, in_d32_stride);
				}else{
					vin2 = vin3 = Q6_V_vzero();	// should not be used anyway
				}
				// (2) do the alignment now
				{
					HVX_Vector tmp = vin0;
					vin0 = Q6_V_vdelta_VV( Q6_V_vmux_QVV(qalign0,vin0,vin_prev0),aligndelt0);
					vin_prev0 = tmp;
					tmp = vin1;
					vin1 = Q6_V_vdelta_VV( Q6_V_vmux_QVV(qalign1,vin1,vin_prev1),aligndelt1);
					vin_prev1 = tmp;
					if( k_val > 2){
						tmp = vin2;
						vin2 = Q6_V_vdelta_VV( Q6_V_vmux_QVV(qalign2,vin2,vin_prev2),aligndelt2);
						vin_prev2 = tmp;
						tmp = vin3;
						vin3 = Q6_V_vdelta_VV( Q6_V_vmux_QVV(qalign3,vin3,vin_prev3),aligndelt3);
						vin_prev3 = tmp;
					}
				}
				// (3) interleave
				if( k_val > 2 ){
					// interleave 0 & 2, and 1 & 3
					HVX_VectorPair v02 = Q6_W_vshuff_VVR( vin2, vin0, 32-1 );
					HVX_VectorPair v13 = Q6_W_vshuff_VVR( vin3, vin1, 32-1 );
					// and then finish interleave
					HVX_VectorPair vshuf_ab = Q6_W_vshuff_VVR( Q6_V_lo_W(v13), Q6_V_lo_W(v02), 32-1);
					HVX_VectorPair vshuf_cd = Q6_W_vshuff_VVR( Q6_V_hi_W(v13), Q6_V_hi_W(v02), 32-1);
					vout0 = Q6_V_lo_W(vshuf_ab);
					vout1 = Q6_V_hi_W(vshuf_ab);
					vout2 = Q6_V_lo_W(vshuf_cd);
					vout3 = Q6_V_hi_W(vshuf_cd);
				}else{
					HVX_VectorPair vshuf_ab = Q6_W_vshuff_VVR( vin1, vin0, 32-1 );
					vout0 = Q6_V_lo_W(vshuf_ab);
					vout1 = Q6_V_hi_W(vshuf_ab);
				}

				// (4) if scaling is needed, apply it now
				if( need_scaling){
					vout0 = do_scale_ub( vout0, scale_lo, scale_hi, offs_02, offs_13);
					vout1 = do_scale_ub( vout1, scale_lo, scale_hi, offs_02, offs_13);
					if( k_val > 2){
						vout2 = do_scale_ub( vout2, scale_lo, scale_hi, offs_02, offs_13);
						vout3 = do_scale_ub( vout3, scale_lo, scale_hi, offs_02, offs_13);
					}
				}
				// (5) store out the results
				// on a last partial iteration, we store only one, and break; and let the code after
				// decide whether to store vout1, vout2 (for k=2, that's not needed).
				//
				*voutp = vout0;		voutp = ptr_advance_bytes( voutp, out_d32_stride);
				if( nd32_remain < k_val) break;
				*voutp = vout1;   	voutp = ptr_advance_bytes( voutp, out_d32_stride);
				if( k_val > 2){
					*voutp = vout2;   	voutp = ptr_advance_bytes( voutp, out_d32_stride);
					*voutp = vout3;   	voutp = ptr_advance_bytes( voutp, out_d32_stride);
				}
				nd32_remain -= k_val;
			}
			// if the last operation had 2 or 3 extra values, we need to store them now.
			// if it had  0 or 1 extra, it's already dealt with.
			if( k_val > 2 &&  nd32_remain >=2){
				*voutp = vout1;
				if( nd32_remain ==3){
					*ptr_advance_bytes( voutp, out_d32_stride) = vout2;
				}
			}
		} // end of column loop
		// move pointers to the next row
		rvoutp = ptr_advance_bytes( rvoutp, out_height_stride );
		rvinp0 = ptrk_advance_bytes( rvinp0, in_height_stride );
		rvinp1 = ptrk_advance_bytes( rvinp1, in_height_stride );
		if( k_val > 2){
			rvinp2 = ptrk_advance_bytes( rvinp2, in_height_stride );
			rvinp3 = ptrk_advance_bytes( rvinp3, in_height_stride );
		}
	} // end of row loop.
}

// Different strategy for alignmode = 0 or 1:
// for these aligmnent modes, we never need to span across d32 chunks
// so 'width' can be the innermost loop.
// Also, the tests in the inner loop to see if less than 'k' output vectors
// should be stored, can make it quite inefficient; so, for the case where scaling
// is not done, any odd depth slices are peeled out to a separate loop.
//
// for height
//     for depth (groups of k output vectors)
//          for width
//              (...) can store partial group at end, only when scaling_mode==1.
//
//     if scaling_mode==0, and d32 output is not a multiple of k:
//          for width
//              (...) only store up to k-1 outputs
//
// NOTE: aligment_mode = -1 means that vror should be used to align, but
// that the d32 loop should *not* be constrained to one cycle (this allows the
// same function to be used for alignment modes 0 and 1). This is used for need_scaling=1
// to reduce the # of functions needed.


static inline void __attribute__((always_inline))
channelshuf_d32_k24_align01_template( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop,
		int ibatch,								// batch index to process
		int irow,								// first row to process
		int k_val,								// 2 or 4
		int need_scaling,
		int alignment_mode )					// 0 or 1 or -1
{
	int vwide = rstp->vecs_wide;
	int nd32_out = subop->tout.nd32;	// # of output d32 units
	// how many loops to get through the depth dimension?
	int d32_loops = (k_val==4)? ( (nd32_out+3)>>2) : ((nd32_out+1)>>1);

	if( alignment_mode ==1) d32_loops= 1;	// assumed to be 1 in this case.

	if( alignment_mode ==-1) alignment_mode = 1; // for -1, allow any # d32_loops but use vror.
	//
	// if we are *not* doing scaling, we always write 'k' depth slices in the main loop
	// and use a second loop to handle the remainder.
	//
	int peel_depth_loop = (need_scaling ==0);
	int final_d32_out = 0;				// no. of d32 outputs in peeled loop
	if(peel_depth_loop){
		// reduce d32_loops to full vecs only. It could be 0.
		d32_loops = (k_val==4)? ( nd32_out>>2) : (nd32_out>>1);
		final_d32_out = nd32_out & (k_val-1);
		if( vwide < 1) return;	// eliminate checks in the loops for this
	}else{
		if( d32_loops < 1 || vwide < 1) return;	// eliminate checks in the loops for these
	}

	int height = rstp->height;
	int nrows = min_i32( rstp->height_chunksize, height-irow);


	int out_height_stride = subop->tout.height_stride;
	int out_d32_stride = subop->tout.d32_stride;
	// find the start addresses
	// (assuming all inputs have same batch/height/d32 stride, just look at the first one).
	int in_height_stride = subop->in_slice[0].height_stride;
	int in_d32_stride = subop->in_slice[0].d32_stride;
	unsigned in_offs = ibatch * subop->in_slice[0].batch_stride + irow * in_height_stride;

	HVX_Vector *rvoutp = (HVX_Vector*)(subop->tout.data + ibatch * subop->tout.batch_stride + irow * out_height_stride );
	HVX_Vector const * rvinp0 = (HVX_Vector const *)(subop->in_slice[0].data + in_offs);
	HVX_Vector const * rvinp1 = (HVX_Vector const *)(subop->in_slice[1].data + in_offs);
	HVX_Vector const * rvinp2 = (HVX_Vector const *)(subop->in_slice[2].data + in_offs);
	HVX_Vector const * rvinp3 = (HVX_Vector const *)(subop->in_slice[3].data + in_offs);
/*
printf("batch %d row %d: %d rows of %d  vecs; nd32=%d al=%d ns=%d iptrs =%p %p %p %p bs=%d hs=%d d32s=%d, optr = %p  bs=%d hs=%d d32s=%d\n",
		ibatch, irow, nrows, vwide, nd32_out, alignment_mode, need_scaling,
		rvinp0, rvinp1, rvinp2, rvinp3, (int)subop->in_slice[0].batch_stride,  in_height_stride, in_d32_stride,
		rvoutp, (int)subop->tout.batch_stride, out_height_stride, out_d32_stride );
*/
	// these are only used if need_scaling !=0
	int scale_lo = 0, scale_hi = 0, offs_02 = 0, offs_13 = 0;
	if( need_scaling){
		scale_lo = rstp->scale_lo;
		scale_hi = rstp->scale_hi;
		offs_02 = rstp->offs_02;
		offs_13 = rstp->offs_13;
	}
	// when k = 2, there are a number of variables in the setup which are never
	// used; we just let the compiler fold them away.
	//
	// these are only used when aligmnent_mode = 1
	//
	int nrot0 =0, nrot1 = 0, nrot2 = 0, nrot3 = 0;

	if( alignment_mode == 1){
		nrot0 =  subop->in_slice[0].depth_pad_before;
		nrot1 =  subop->in_slice[1].depth_pad_before;
		nrot2 =  subop->in_slice[2].depth_pad_before;
		nrot3 =  subop->in_slice[3].depth_pad_before;
	}

	// height loop

	for( int ir = 0; ir < nrows; ir++){
		// d32 loop
		// (null loop when align_mode = 1)
		//
		int nd32_remain = nd32_out;
		for( int id32 = 0; id32 < d32_loops; id32++){
			HVX_Vector *voutp = ptr_advance_bytes( rvoutp, id32 * out_d32_stride*k_val );
			// output pointer for 'odd' depth slice
			HVX_Vector * __restrict voutp1 = ptr_advance_bytes( voutp, out_d32_stride );
			HVX_Vector const * __restrict vinp0 = ptrk_advance_bytes( rvinp0, id32 * in_d32_stride );
			HVX_Vector const * __restrict vinp1 = ptrk_advance_bytes( rvinp1, id32 * in_d32_stride );
			HVX_Vector const * __restrict vinp2 = ptrk_advance_bytes( rvinp2, id32 * in_d32_stride );
			HVX_Vector const * __restrict vinp3 = ptrk_advance_bytes( rvinp3, id32 * in_d32_stride );

			for (int ic = 0; ic < vwide; ic++){
				HVX_Vector vout0,vout1,vout2,vout3;
				// (1) get 4 new vectors
				HVX_Vector vin0 = *vinp0++;
				HVX_Vector vin1 = *vinp1++;
				HVX_Vector vin2 , vin3;
				if( k_val > 2){
					vin2 = *vinp2++;
					vin3 = *vinp3++;
				}else{
					vin2 = vin3 = Q6_V_vzero();	// should not be used anyway
				}
				// (2) if misaligned, do the alignment now
				if( alignment_mode == 1){
					vin0 = Q6_V_vror_VR( vin0, nrot0);
					vin1 = Q6_V_vror_VR( vin1, nrot1);
					if( k_val > 2){
						vin2 = Q6_V_vror_VR( vin2, nrot2);
						vin3 = Q6_V_vror_VR( vin3, nrot3);
					}
				}
				// (3) interleave
				if( k_val > 2 ){
					// interleave 0 & 2, and 1 & 3
					HVX_VectorPair v02 = Q6_W_vshuff_VVR( vin2, vin0, 32-1 );
					HVX_VectorPair v13 = Q6_W_vshuff_VVR( vin3, vin1, 32-1 );
					// and then finish interleave
					HVX_VectorPair vshuf_ab = Q6_W_vshuff_VVR( Q6_V_lo_W(v13), Q6_V_lo_W(v02), 32-1);
					HVX_VectorPair vshuf_cd = Q6_W_vshuff_VVR( Q6_V_hi_W(v13), Q6_V_hi_W(v02), 32-1);
					vout0 = Q6_V_lo_W(vshuf_ab);
					vout1 = Q6_V_hi_W(vshuf_ab);
					vout2 = Q6_V_lo_W(vshuf_cd);
					vout3 = Q6_V_hi_W(vshuf_cd);
				}else{
					HVX_VectorPair vshuf_ab = Q6_W_vshuff_VVR( vin1, vin0, 32-1 );
					vout0 = Q6_V_lo_W(vshuf_ab);
					vout1 = Q6_V_hi_W(vshuf_ab);
				}

				// (4) if scaling is needed, apply it now
				if( need_scaling){
					vout0 = do_scale_ub( vout0, scale_lo, scale_hi, offs_02, offs_13);
					vout1 = do_scale_ub( vout1, scale_lo, scale_hi, offs_02, offs_13);
					if( k_val > 2){
						vout2 = do_scale_ub( vout2, scale_lo, scale_hi, offs_02, offs_13);
						vout3 = do_scale_ub( vout3, scale_lo, scale_hi, offs_02, offs_13);
					}
				}
				// (5) store out the results, to  the various depth slices.
				// when not "peel_depth_loop", suppress stores which are >= nd32_out
				//
				*voutp ++ = vout0;
				if( peel_depth_loop ||  nd32_remain >1){
					*voutp1++ = vout1;
				}
				if( k_val > 2){
					if( peel_depth_loop || nd32_remain >2){
						*ptr_advance_bytes( voutp, out_d32_stride*2 -128) = vout2;
					}
					if( peel_depth_loop || nd32_remain >3){
						*ptr_advance_bytes( voutp1, out_d32_stride*2 -128) = vout3;
					}
				}
			} // end of column loop
			nd32_remain -= k_val;
		} // end of depth loop

		// peeled loop for odd depth, only used when not scaling.
		// when k=4, writes 1..3 outputs; when k=2, only writes 1.
		if( final_d32_out > 0){		// only possible when peel_depth_loop
			HVX_Vector *voutp = ptr_advance_bytes( rvoutp, d32_loops * out_d32_stride*k_val );
			// output pointer for 'odd' depth slice
			HVX_Vector *voutp1 = ptr_advance_bytes( voutp, out_d32_stride );
			HVX_Vector const * vinp0 = ptrk_advance_bytes( rvinp0, d32_loops * in_d32_stride );
			HVX_Vector const * vinp1 = ptrk_advance_bytes( rvinp1, d32_loops * in_d32_stride );
			HVX_Vector const * vinp2 = ptrk_advance_bytes( rvinp2, d32_loops * in_d32_stride );
			HVX_Vector const * vinp3 = ptrk_advance_bytes( rvinp3, d32_loops * in_d32_stride );

			for (int ic = 0; ic < vwide; ic++){
				HVX_Vector vout0,vout1,vout2;
				// (1) get 4 new vectors
				HVX_Vector vin0 = *vinp0++;
				HVX_Vector vin1 = *vinp1++;
				HVX_Vector vin2 , vin3;
				if( k_val > 2){
					vin2 = *vinp2++;
					vin3 = *vinp3++;
				}else{
					vin2 = vin3 = Q6_V_vzero();	// should not be used anyway
				}
				// (2) if misaligned, do the alignment now
				if( alignment_mode == 1){
					vin0 = Q6_V_vror_VR( vin0, nrot0);
					vin1 = Q6_V_vror_VR( vin1, nrot1);
					if( k_val > 2){
						vin2 = Q6_V_vror_VR( vin2, nrot2);
						vin3 = Q6_V_vror_VR( vin3, nrot3);
					}
				}
				// (3) interleave; only need k-1 vectors.
				if( k_val > 2 ){
					// interleave 0 & 2, and 1 & 3
					HVX_VectorPair v02 = Q6_W_vshuff_VVR( vin2, vin0, 32-1 );
					HVX_VectorPair v13 = Q6_W_vshuff_VVR( vin3, vin1, 32-1 );
					// and then finish interleave
					HVX_VectorPair vshuf_ab = Q6_W_vshuff_VVR( Q6_V_lo_W(v13), Q6_V_lo_W(v02), 32-1);
					HVX_VectorPair vshuf_cd = Q6_W_vshuff_VVR( Q6_V_hi_W(v13), Q6_V_hi_W(v02), 32-1);
					vout0 = Q6_V_lo_W(vshuf_ab);
					vout1 = Q6_V_hi_W(vshuf_ab);
					vout2 = Q6_V_lo_W(vshuf_cd);
				}else{
					HVX_VectorPair vshuf_ab = Q6_W_vshuff_VVR( vin1, vin0, 32-1 );
					vout0 = Q6_V_lo_W(vshuf_ab);
				}
				// (4) scaling not applicable
				// (5) store out the results, to  the various depth slices.
				//
				*voutp ++ = vout0;			// for k==2, that is all.
				if( k_val > 2){
					// this is to encourage the compiler to use conditional stores, by making it think
					// it needs to compute vout2 regardless of final_d32_out.
#if defined(__hexagon__)
					asm volatile ("/* %0 */": :"v"(vout2) );
#endif
					if( final_d32_out > 1) *voutp1 = vout1;
					if( final_d32_out > 2) *ptr_advance_bytes( voutp, out_d32_stride*2 -128) = vout2;
					voutp1++;
				}
			} // end of column loop
		} // if final_d32_out

		// move pointers to the next row
		rvoutp = ptr_advance_bytes( rvoutp, out_height_stride );
		rvinp0 = ptrk_advance_bytes( rvinp0, in_height_stride );
		rvinp1 = ptrk_advance_bytes( rvinp1, in_height_stride );
		if( k_val > 2){
			rvinp2 = ptrk_advance_bytes( rvinp2, in_height_stride );
			rvinp3 = ptrk_advance_bytes( rvinp3, in_height_stride );
		}
	} // end of row loop.
}
//
// inner hvx loop for K=1
// (this is only used in 'split' operation; no scaling is needed since there's only one input).
// This function is inlined with different values
// of alignment_mode (which should be constants in each "call").
//
// alignment mode refers to the depth alignment; an input is aligned if the required
// data starts at the beginning of a d32 slice
//
//  aligmnent_mode = 0 : all 4 inputs are aligned
//                 = 1 : all 4 inputs may be misaligned; but d32_loops ==1 (out_depth <= 128)
// 					 and we can do the align with a vror (all of the inputs are present in the first depth slice).
//                 = 2 : all 4 inputs have their own alignment.
//
// the 'irow' value is the start row for the operation, which is a multiple
// if rstp->height_chunk, and is always less than rstp->height. The number of rows
// processed is rstp->height_chunk, or (rstp->height-irow), whichever is smaller.
//
//
//
static inline void __attribute__((always_inline))
channelshuf_d32_k1_template( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop,
		int ibatch,								// batch index to process
		int irow,								// first row to process
		int alignment_mode )
{
	int vwide = rstp->vecs_wide;
	int nd32_out = subop->tout.nd32;	// # of output d32 units
	// how many loops to get through the depth dimension?
	int d32_loops = nd32_out;

	if( alignment_mode ==1) d32_loops= 1;	// assumed to be 1 in this case.

	if( d32_loops < 1 || vwide < 1) return;	// eliminate checks in the loops for these

	int height = rstp->height;
	int nrows = min_i32( rstp->height_chunksize, height-irow);


	int out_height_stride = subop->tout.height_stride;
	int out_d32_stride = subop->tout.d32_stride;
	// find the start addresses
	int in_height_stride = subop->in_slice[0].height_stride;
	int in_d32_stride = subop->in_slice[0].d32_stride;
	unsigned in_offs = ibatch * subop->in_slice[0].batch_stride + irow * in_height_stride;

	HVX_Vector *rvoutp = (HVX_Vector*)(subop->tout.data + ibatch * subop->tout.batch_stride + irow * out_height_stride );
	HVX_Vector const * rvinp0 = (HVX_Vector const *)(subop->in_slice[0].data + in_offs);

	// used only when alignment_mode ==2
	HVX_Vector vin_prev0 = Q6_V_vzero();
	HVX_Vector aligndelt0 = Q6_V_vzero();
	HVX_VectorPred qalign0 = Q6_Q_vand_VR(aligndelt0,0);
	//
	// only used when aligmnent_mode = 1
	//
	int nrot0 =0;

    if( alignment_mode == 1){
		nrot0 =  subop->in_slice[0].depth_pad_before;
	}else{
		// full align, spanning from to vectors. The execution is
		//  vout = Q6_V_delta_VV(   Q6_V_vmux_QVV( valign, vecin1, vecin0),  aligndelt)
		// The 'mux' selects the values needed from each input, and the 'delta' rotates within each 32-byte
		// lane.
		HVX_Vector count32 = *(HVX_Vector const*)const_Count32;	 // {0,1, ..31, 0,1, ..31, 0, .. 31, 0.. 31 }
		HVX_Vector k31 = q6op_Vb_vsplat_R(31);
		// to get the vdelta code for rotate-down-by-n, each lane needs to be ((i-k)&31)^i
		// where i is the lane 0..31 (The pattern is repeated 4 times). The mux control is i < k.
		int algn = subop->in_slice[0].depth_pad_before;	// 0..31
		HVX_Vector valgn = q6op_Vb_vsplat_R( algn);	// copy to all bytes
		aligndelt0 = Q6_V_vxor_VV( Q6_V_vand_VV(Q6_Vb_vsub_VbVb(count32,valgn),k31),count32);
		qalign0 = Q6_Q_vcmp_gt_VubVub( valgn, count32);	// first 'al0' slots in each lane to come from next d32
	}
	// height loop

	for( int ir = 0; ir < nrows; ir++){
		for (int ic = 0; ic < vwide; ic++){
			// get pointers to start of each 'depth run'
			HVX_Vector *voutp = rvoutp + ic;
			HVX_Vector const * vinp0 = rvinp0 + ic;

			if(alignment_mode==2){		// preload first from each input
				vin_prev0 = *vinp0;  vinp0 = ptrk_advance_bytes( vinp0, in_d32_stride);
			}
			// now loop down the depth dimension
			for( int id32 = 0; id32 < d32_loops; id32++){
				// (1) get new vector
				HVX_Vector vin0 = *vinp0;  vinp0 = ptrk_advance_bytes( vinp0, in_d32_stride);
				// (2) if misaligned, do the alignment now
				if( alignment_mode == 1){
					vin0 = Q6_V_vror_VR( vin0, nrot0);
				}else if( alignment_mode ==2){
					HVX_Vector tmp = vin0;
					vin0 = Q6_V_vdelta_VV( Q6_V_vmux_QVV(qalign0,vin0,vin_prev0),aligndelt0);
					vin_prev0 = tmp;
				}
				// (3) (no interleave when K=1)
				// (4) (no scaling when K=1)
				// (5) store out the results
				//
				*voutp = vin0;		voutp = ptr_advance_bytes( voutp, out_d32_stride);
			}
		} // end of column loop
		// move pointers to the next row
		rvoutp = ptr_advance_bytes( rvoutp, out_height_stride );
		rvinp0 = ptrk_advance_bytes( rvinp0, in_height_stride );
	} // end of row loop.
}

// K = 1
static void
channelshuf_d32_k1_scale0_align0( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k1_template( rstp, subop, ibatch, irow, 0);
}
static void
channelshuf_d32_k1_scale0_align1( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k1_template( rstp, subop, ibatch, irow, 1);
}
static void
channelshuf_d32_k1_scale0_align2( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k1_template( rstp, subop, ibatch, irow, 2);
}


// K = 2
static void
channelshuf_d32_k2_scale0_align0( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align01_template( rstp, subop, ibatch, irow, 2, 0,0);
}
static void
channelshuf_d32_k2_scale0_align1( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align01_template( rstp, subop, ibatch, irow, 2, 0,1);
}
static void
channelshuf_d32_k2_scale0_align2( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align2_template( rstp, subop, ibatch, irow,2, 0);
}
// this does alignment_mode = 0 or 1
static void
channelshuf_d32_k2_scale1_align01( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align01_template( rstp, subop, ibatch, irow,2, 1,-1);
}
static void
channelshuf_d32_k2_scale1_align2( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align2_template( rstp, subop, ibatch, irow,2,  1);
}

// K = 4
static void
channelshuf_d32_k4_scale0_align0( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align01_template( rstp, subop, ibatch, irow, 4, 0,0);
}
static void
channelshuf_d32_k4_scale0_align1( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align01_template( rstp, subop, ibatch, irow, 4, 0,1);
}
static void
channelshuf_d32_k4_scale0_align2( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align2_template( rstp, subop, ibatch, irow,4, 0);
}
// this does alignment_mode = 0 or 1

static void
channelshuf_d32_k4_scale1_align01( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align01_template( rstp, subop, ibatch, irow,4, 1,-1);
}
static void
channelshuf_d32_k4_scale1_align2( struct chanshuf_d32_runstate *rstp,
		struct chanshuf_d32_suboperation const * subop, int ibatch, int irow )
{
	channelshuf_d32_k24_align2_template( rstp, subop, ibatch, irow,4,  1);
}


static const chanshuf_exec_funcp exec_function_table[18]=			// the table of function pointers.
{
		channelshuf_d32_k1_scale0_align0,
		channelshuf_d32_k1_scale0_align1,
		channelshuf_d32_k1_scale0_align2,
		channelshuf_d32_k1_scale0_align0,	// these 3 are unused ..
		channelshuf_d32_k1_scale0_align1, 	// .. since we K=1 is only for split ..
		channelshuf_d32_k1_scale0_align2,	// .. and split never needs scaling

		channelshuf_d32_k2_scale0_align0,
		channelshuf_d32_k2_scale0_align1,
		channelshuf_d32_k2_scale0_align2,
		channelshuf_d32_k2_scale1_align01,
		channelshuf_d32_k2_scale1_align01,
		channelshuf_d32_k2_scale1_align2,

		channelshuf_d32_k4_scale0_align0,
		channelshuf_d32_k4_scale0_align1,
		channelshuf_d32_k4_scale0_align2,
		channelshuf_d32_k4_scale1_align01,
		channelshuf_d32_k4_scale1_align01,
		channelshuf_d32_k4_scale1_align2,

};


static int channelshuffle_check_d32(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking channelshuffle_d32 node %p",self);
	int n_in = self->n_inputs;
	if( n_in < 4 ||  (n_in-1)%3u != 0 )
		return errlog(nn, "channelshuffle_d32: needs 3*n+1 inputs, with n >= 1");
	logmsg(nn,2,"channelshuffle_d32 node %p check OK",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedChannelShuffle_8_d32 = {
	.execute = channelshuffle_d32_execute,
	.check = channelshuffle_check_d32,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_CLS_CHANSHUFFLE | NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_GE(4),	// must be 3*k+1
	.n_outputs = NN_IOCOUNT_GE(3),
};


struct nn_node_ops nn_ops_for_QuantizedSplit_8_d32 = {
	.execute = split_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(4),		// must be exactly 4 in
	.n_outputs = NN_IOCOUNT_GE(4),	// at least 4 out
};
