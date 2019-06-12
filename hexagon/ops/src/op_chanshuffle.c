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
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif


//
// 'channel shuffle' operator
//
// Inputs are:
//     - k (number of input groups)
// 		'n_in' input tensors of shape [b,h,w,d_in]
//  	'n_out' output tensors (all will be [b,h,w,d_in*n_in/n_out] )

// we determine the following (all of the divisions must be exact)
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
// 'float' version ChannelShuffle_f
// int32 version ChannelShuffle_int32:
//   n_in+1 inputs:
//     0 - scalar int - 'k' parameter
//     1..n_in            flat tensors of shape [b,h,w,d_in]
//   n_out outputs:
//     0..n_out-1		  flat tensors of shape [b,h,w,d_in*n_in/n_out]
//
//
// 'qu8' version QuantizedChannelShuffle_8
//
//   3*n_in+1 inputs:
//     0 - scalar int - 'k' parameter
//     1..n_in            		flat qu8 tensors of shape [b,h,w,d_in]
//     n_in+1 .. 2*n_in      	float scalar: minima for the inputs
//     n_in*k+1 .. 3*n_in    	float scalar: maxima for the inputs
//   n_out+2 outputs:
//     0..n_out-1				flat qu8 tensors of shape [[b,h,w,d_in*n_in/n_out]
//	     n_out					float scalar: minimum (for all outputs)
//		 n_out+1				float scalar: maximum (for all outputs)
//


// when scaling qu8:
//    out =  (in*scale+ offs) >> 20   (saturated to u8)
// There is a chanshuf_qu8_scale for every input;
// when an input doesn't require scaling, then scale < 0
//
struct chanshuf_qu8_scale{
	float in_min, in_max;
	int scale;
	int offs;
};
// this assumes that all_min <= in_min < in_max <= all_max.

static void setup_scale_for_input( struct chanshuf_qu8_scale *sclp, float all_min, float all_max)
{
	float all_range = all_max - all_min;
	float in_range = sclp->in_max - sclp->in_min;
	int scaleval = 1048576.0f * in_range / all_range + 0.5f;	// value of scale
	if( scaleval == 1048576){
		sclp->scale = -1;		// mark as 'no change'
	}else{
		// + 524288 is a rounding bias for the >>20
		int offs = roundf_i32( (sclp->in_min-all_min)*(255.0f* 1048576.0f)/all_range + 524288.0f);
		if( scaleval >= 1044464){	// could still be 'no-op'
			// if 0->0 and 255->255, it's a no-op
			if( offs < (1<<20) &&  255*scaleval + offs >=(0xFF <<20)){
				sclp->scale = -1;
				return;
			}
		}
		sclp->scale = scaleval;
		sclp->offs = offs;
	}
}


//

static void copy_slice_i32(
		int32_t *out,
		int32_t const * in,
		int nA, int nB, int nC,			// 3 dimensions, all >= 1
		int sAin, int sCin,			// input strides ( sB = 1)
		int sAout, int sBout );		// output strides (sC = 1)

static void copy_slice_qu8(
		struct chanshuf_qu8_scale const *sclp,
		uint8_t *out,
		uint8_t const * in,
		int nA, int nB, int nC,			// 3 dimensions, all >= 1
		int sAin, int sCin,			// input strides ( sB = 1)
		int sAout, int sBout );		// output strides (sC = 1)
//
// execute for 'float' and 'int32' cases (there's no difference between them, other than eltype)
// and for quantized u8.
//
static int channelshuffle_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *k_tensor = self->inputs[0];
	const struct tensor **in_tensors = &self->inputs[1];
	struct tensor ** out_tensors = &self->outputs[0];

	op_type this_optype = self->node_type;
	int is_qu8 = 0;
	int output_eltype = NN_TYPE_FLOAT;
	if( this_optype == OP_ChannelShuffle_int32){
		output_eltype = NN_TYPE_INT32;
	}else if( this_optype == OP_QuantizedChannelShuffle_8 ){
		output_eltype = NN_TYPE_QUINT8;
		is_qu8 = 1;
	}

	int num_in = self->n_inputs-1;
	int num_out = self->n_outputs;
	int k_parm = tensor_get_int32( k_tensor, 0);

	if( is_qu8){
		num_out -= 2;		// deduct for min/max outputs
		num_in = (unsigned)num_in/3;		//actual # inputs
	}

	logmsg(nn,2,"chanshuffle execute. self=%p ",self);

	unsigned depth_in = in_tensors[0]->shape.depth;
	int d0_parm, d1_parm, dq_parm;

	if(  k_parm < 1
				|| ( d1_parm = (unsigned)k_parm/num_in,  d1_parm*num_in != k_parm )
				|| ( dq_parm = depth_in/d1_parm,   d1_parm*dq_parm != depth_in)
				|| ( d0_parm = (unsigned)dq_parm/num_out,   d0_parm*num_out != dq_parm)){
		return errlog(nn, "bad chanshuffle parms: k=%d, in_depth=%d, n_in = %d, n_out = %d\n",
				k_parm, depth_in, num_in, num_out );
	}
	// check all the input shapes match
	for( int i = 1; i < num_in; i++){
		if( ! shape_matches( &in_tensors[0]->shape, &in_tensors[i]->shape )){
			return errlog(nn,"chanshuffle input #%d does not match shape of #1", i+1);
		}
	}
	nn_scratch_reset(nn);

	struct shape out_shape = in_tensors[0]->shape;
	out_shape.depth = k_parm * d0_parm;
	logmsg(nn,4,"n_in = %d; n_out = %d; d0,d1 = %d,%d; in_depth = %d, out_depth=%d", 
		num_in, num_out, d0_parm, d1_parm, (int)depth_in, (int)out_shape.depth );

	//
	// prepare all the tensor outputs
	//
	for( int i = 0; i < num_out; i++ ){
		int k = nn_tensor_out_prepare_normal_fromshape(out_tensors[i],&out_shape, output_eltype);
		if( k!= 0) return errlog( nn, "failed to allocate chanshuffle output #%d, size [%d,%d,%d,%d]",
					i, (int)out_shape.batches, (int)out_shape.height, (int)out_shape.width, (int)out_shape.depth);
	}

	// if qu8, find the range of all the ranges
	//
	float min_overall=0.0f, max_overall = 0.0f;
	struct chanshuf_qu8_scale * qu8_scales = NULL;
	if( is_qu8 ){

		const struct tensor **in_min_tensors = &self->inputs[1+num_in];
		const struct tensor **in_max_tensors = in_min_tensors + num_in;

		qu8_scales = nn_scratch_alloc(nn,num_in * sizeof( struct chanshuf_qu8_scale) );

		min_overall = fminf(0.0f, tensor_get_float( in_min_tensors[0], 0) );
		max_overall = tensor_get_float( in_max_tensors[0], 0);
		qu8_scales[0].in_min = min_overall;
		qu8_scales[0].in_max = max_overall;
		qu8_scales[0].scale = -1;	// in case num_in = 1
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
			for( int k = 0; k < num_in; k++){
				setup_scale_for_input( &qu8_scales[k], min_overall, max_overall);
				logmsg(nn,4,"input %d: (%f..%f) scale = %d, offs = %d",
					k, qu8_scales[k].in_min, qu8_scales[k].in_max, 
						qu8_scales[k].scale, qu8_scales[k].offs);
			}
		}
		tensor_set_single_float( self->outputs[num_out], min_overall );
		tensor_set_single_float( self->outputs[num_out+1], max_overall );
	}

	// conceptually:
	//             [ b, h, w,     d1*d0*n_out]	 	<- n_in of these
	// relabel:    [ b*h*w,  1,  d1*d0*n_out ]		<- n_in of these
	//  concat:    [ b*h*w, n_in, d1*d0*n_out]      (one of)
	// relabel:    [ b*h*w, n_in * d1, d0*n_out]	(one of)
	//  transpose: [ b*h*w,  n_out*d0, d1*n_in]		(one of)
	// relabel:    [ b*h*w,  n_out, d0*d1*n_in]	    (one of)
	// split :     [ b*h*w,  1, d0*d1*n_in]	        (n_out)
	// relabel:    [ b, h, w, d0*d1*n_in ]			(n_out)
	//==============================================================
	// OR this can be seen as n_in * n_out separate copy operations,
	// each from one strided array to another:
	//==============================================================
	// for i_in, i_out:
	//   input array     [ b, h, w,     d1*d0*n_out]		// from input i_in
	// (1) relabel:       [ b*h*w,   d1,  d0*n_out ]
	// (2) extract range  [ b*h*w,   d1,  d0       ]	# extract d0*i_out .. d0*(i_out+1) in last index
	// (3) transpose      [  b*h*w,  d0,    d1     ]
	//
	//      dest array:   [ b, h, w,   d0*d1*n_in ]
	// (4)  reshape       [ b*h*w,   d0 ,  d1*n_in ]
	// (5)  extract range [ b*h*w,   d0,    d1  ]		# extract d1*i_in .. d1 * (i_in+1) from last index
	//
	//  then copy (3) to (5). Each is a 3-dimensional array; they have these 'strides' (in elements):
	//    src_arr_strides = { d0*d1*n_out,  1,  d0*n_out }
	//    dst_arr_strides = { d0*d1*n_in,  d1*n_in,   1 }
	//

	// the operation repeats this many times.
	unsigned outer_count = out_shape.batches * out_shape.height * out_shape.width;
	unsigned outer_pitch_in  = d0_parm * d1_parm * num_out;
	unsigned outer_pitch_out  = d0_parm * d1_parm * num_in;
	unsigned outslice_pitch_in = d0_parm;		// amount to bump inptr per output slice
	unsigned inslice_pitch_out = d1_parm;		// amount to bump optr per input slice

	logmsg(nn,4,"outer_count = %d;  out_pitch_in = %d;  in_pitch_out = %d; oslice_pitch_in = %d; inslice_pitch_out = %d",
			outer_count, outer_pitch_in, outer_pitch_out, outslice_pitch_in, inslice_pitch_out );

	if( !is_qu8 ){
		//
		// The non-qu8 case
		//
		for(int i_out = 0; i_out < num_out; i_out++){
			int32_t *out_data = (int32_t *)out_tensors[i_out]->data;
			for(int i_in = 0; i_in < num_in; i_in++ ){
				int32_t const * in_data = (int32_t const*)in_tensors[i_in]->data;

				copy_slice_i32(
						out_data + i_in * inslice_pitch_out,
						in_data + i_out * outslice_pitch_in,
						outer_count, d0_parm, d1_parm,		// sizes
						outer_pitch_in,  dq_parm,			// input strides
						outer_pitch_out, d1_parm * num_in );	// output strides
			}
		}
	}else{
		for(int i_out = 0; i_out < num_out; i_out++){
			uint8_t *out_data = (uint8_t *)out_tensors[i_out]->data;
			for(int i_in = 0; i_in < num_in; i_in++ ){
				uint8_t const * in_data = (uint8_t const*)in_tensors[i_in]->data;

				copy_slice_qu8(
						&qu8_scales[i_in],
						out_data + i_in * inslice_pitch_out,
						in_data + i_out * outslice_pitch_in,
						outer_count, d0_parm, d1_parm,		// sizes
						outer_pitch_in,  dq_parm,			// input strides
						outer_pitch_out, d1_parm * num_in );	// output strides
			}
		}

	}
	logmsg(nn,2,"chanshuffle done. self=%p ",self);

	return 0;
}

//
// copy a general strided 3D array:
//     [   nA,  nB,   nC ]		 <- dims
//     [ sAin,   1,  SCin]	     <- input stride (elements)
//     [ SAout, SBout,  1 ]		 < output stride (elements)
//
// Since SCout = 1, this has a 'gather' op as the inner loop.
// When nC = 1, it's 2d operation with a 'scatter' as the inner loop.
//

static void copy_slice_i32(
		int32_t *out,
		int32_t const * in,
		int nA, int nB, int nC,			// 3 dimensions, all >= 1
		int sAin, int sCin,			// input strides ( sB = 1)
		int sAout, int sBout )		// output strides (sC = 1)
{
	for( int iA = 0; iA < nA; iA++ ){
		int32_t const * inA = in + sAin*iA;
		int32_t * outA = out + sAout*iA;
		if( nC > 1){			// general case, gather inner loop
			for( int iB = 0; iB < nB; iB++){
				int32_t const * inB = inA + iB;
				int32_t * outB = outA + sBout*iB;
				for( int iC = 0; iC < nC; iC++){
					*outB ++ = *inB;
					inB += sCin;
				}
			}
		}else{
			for( int iB = 0; iB < nB; iB++){
				*outA = *inA ++;
				outA += sBout;
			}
		}
	}
}
//
// same thing but does scaled range conversion in the process of.
static void copy_slice_qu8(
		struct chanshuf_qu8_scale const *sclp,
		uint8_t *out,
		uint8_t const * in,
		int nA, int nB, int nC,			// 3 dimensions, all >= 1
		int sAin, int sCin,			// input strides ( sB = 1)
		int sAout, int sBout )		// output strides (sC = 1)
{
	int scale = sclp->scale;
	int offs = sclp->offs;
	if( scale < 0){		// 'no-op' scaling
		for( int iA = 0; iA < nA; iA++ ){
			uint8_t const * inA = in + sAin*iA;
			uint8_t * outA = out + sAout*iA;
			if( nC > 1){			// general case, gather inner loop
				for( int iB = 0; iB < nB; iB++){
					uint8_t const * inB = inA + iB;
					uint8_t * outB = outA + sBout*iB;
					for( int iC = 0; iC < nC; iC++){
						*outB ++ = *inB;
						inB += sCin;
					}
				}
			}else{
				for( int iB = 0; iB < nB; iB++){
					*outA = *inA ++;
					outA += sBout;
				}
			}
		}
	}else{
		for( int iA = 0; iA < nA; iA++ ){
			uint8_t const * inA = in + sAin*iA;
			uint8_t * outA = out + sAout*iA;
			if( nC > 1){			// general case, gather inner loop
				for( int iB = 0; iB < nB; iB++){
					uint8_t const * inB = inA + iB;
					uint8_t * outB = outA + sBout*iB;
					for( int iC = 0; iC < nC; iC++){
						*outB ++ = saturate_u8( (*inB * scale + offs)>>20);
						inB += sCin;
					}
				}
			}else{
				for( int iB = 0; iB < nB; iB++){
					*outA = saturate_u8( (*inA * scale + offs)>>20);
					inA++;
					outA += sBout;
				}
			}
		}
	}
}


static int channelshuffle_check_qu8(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking channelshuffle node %p",self);
	int n_in = self->n_inputs;
	if( n_in < 4 ||  (n_in-1)%3u != 0 )
		return errlog(nn, "channelshuffle: needs 3*n+1 inputs, with n >= 1");
	logmsg(nn,2,"channelshuffle node %p check OK",self);
	return 0;
}


struct nn_node_ops nn_ops_for_ChannelShuffle_f = {
	.execute = channelshuffle_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(2),
	.n_outputs = NN_IOCOUNT_GE(1),
};

struct nn_node_ops nn_ops_for_ChannelShuffle_int32 = {
	.execute = channelshuffle_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(2),
	.n_outputs = NN_IOCOUNT_GE(1),
};
struct nn_node_ops nn_ops_for_QuantizedChannelShuffle_8 = {
	.execute = channelshuffle_execute,
	.check = channelshuffle_check_qu8,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_CLS_CHANSHUFFLE,
	.n_inputs = NN_IOCOUNT_GE(4),	// must be 3k+1
	.n_outputs = NN_IOCOUNT_GE(3),
};
// the 'ref' version is identical, but will not be transformed in the graph.
struct nn_node_ops nn_ops_for_QuantizedChannelShuffle_8_ref = {
	.execute = channelshuffle_execute,
	.check = channelshuffle_check_qu8,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),		// must be 3k+1
	.n_outputs = NN_IOCOUNT_GE(3),
};

