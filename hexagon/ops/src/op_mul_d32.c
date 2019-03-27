/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
 *
 *
 * This contains implementations for mul_d32
 *
 * Elementwise multiply of two tensors.
 *  - dimensions must match; except it's ok to disagree on batches or height, as long as
 *    the B input has size 1. This will be broadcast to the A dimension.
 *  - the inputs must have identical depth_before padding; and both inputs must
 *   have the same width_before padding (modulo 4).
 *  - padding in the output tensor will be copied from the first input tensor.
 *
 *
 */
#include "hvx_inlines.h"

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>

// scaling parms
// those with [*] are a dependent on the output range;
// the others only on the input range.
//
struct op_scaling_parms {
	int16_t zeroval[2];		// each is 0..255
	int16_t range_bias;		// subtract from product (mod 2^16)
	int16_t post_scale;		// [*] then mul by this with >>15
	int16_t adj_bias;		// [*] add to that (with sat)
	int16_t post_rsh;		// [*]then >> by this with rounding
	float all_in_scale;		// product of input scales
	float all_in_scale_recip;		// recip of all_in_scale
};
// The operation is done as:
//  (1) subtract the 'zeroval' from each input; result is +/-255
//  (2) mul these together; result is +/-64k
//  (3) subtract 'range bias', result is +/32k  (by construction of range_bias)
//  (4) mul by post_scale with >> 15
//  (5) add adj_bias (with saturation)
//  (6) >>post_rsh and saturate to u8.
//
// The 'range bias' is calculated so that (3) falls into i16 range, this allows us to do the
// step (2) multiply modulo 16 bits.
//
// The 'full' result at step (2) is numerically equal to the product of the 'true'
//   inputs, multiplied by all_in_scale = 255/(inrangeA) * 255/(inrangeB).
//
//
// let ps1 = post_scale/32k
//    post_scale_flt = ps1/(2^rsh)
//
// Thus we want:
//     all_in_scale * ps1 >> rsh    ==   255/out_range		 (so the overall scaling is correct).
//     (-range_bias * ps1 + adj_bias)/(2^rsh)   ==  the output 'zero' code
//
// So, for a given output range, we can find
//     post_scale_flt = 255/(out_range* all_in_scale)
//  ... and we find post_scale and post_rsh  such post_scale is 16K..3K
//     and  post_scale * 2^-(15 + post_rsh) ~= post_scale_flt.
//   and then
//     adj_bias = (output_zero) * 2^rsh + range_bias*ps1
//
// typically, post_rsh  is 5..7  (since post_scale_flt is about 1/256 or a bit larger).
// We can 'predict' the output range based on the input range,
// and when it is as predicted, post_rsh is always close to 7 (note that it can't be more than 7, due to the limit of vasr
// operation; in cases where post_scale_flt < 1/256, post_rsh will be 7 and post_scale will be less than 16384).
// In many cases, though, the range of the output will be rather less than so predicted; the large values
// on one side may not line up with small values on the other (and the input may not be full range).
// So the post_rsh may be a smaller number, to get some gain in the operation. We can't have post_rsh < 1,
// so post_scale_flt >0.5 can't be supported. This should not be a problem, since such large gains over-represent
// the precision in the input. This means that out_max - out_min should never be less than 510/all_in_scale
// Currently, we do ranging by
//   (a) starting with a value which is about 1/4 of the predicted range
//   (b) expanding max or min as needed to accomodate results which overflow
//   So the output range will never actually get really small (if the 1/4 estimate os ok,
//    post_rsh will be about 5).
//
// for ranging purposes, we can keep track of the range of the (3) result during the op,
//  and use it to detect clipping and to figure out the actual output range.
//   - check_intermed_range is given min,max from (3) and checks to see if clipping occurred on current scaling;
//   - convert_intermed_to_outval is given min (or max) from (3) and converts to min (or max) float.
// Note that the (3) result does not depend on the choice of output range.
//
//
// this function sets up the scaling for a given output range, using the input-range-dependent
// values already in the struct.


static inline void
set_scaling_for_output_range( struct op_scaling_parms * osp, float out_min, float out_max )
{
	float out_scl = 255.0f/(out_max-out_min);		// out scale
	float out_z = -out_min * out_scl;				// the 'zero point'
	float post_scale_flt = out_scl * osp->all_in_scale_recip;
	int scexp = flt_getexp( post_scale_flt);
	int rsh = min_i32( -scexp,7);	// e.g. 0.11 -> 0.88, rsh = 3

	float rsh_fac = flt_power2(rsh);		// 2.0 ** rsh

	float scmant =  post_scale_flt * rsh_fac *32768.0f ;	// e.g. 0.11 -> 28835.84
	// we need to ensure that 0.99999 becomes 32767, not rounded up to 32768
	osp->post_scale = saturate_i16( roundf_i32(scmant));	// find the 16-bit factor
	osp->post_rsh = rsh;
	// use the quantized scale to find adj_bias

	osp->adj_bias = ((osp->range_bias*osp->post_scale + 16384)>>15) + roundf_i32( out_z * rsh_fac);
}
// this converts a value at step (3) of the process to a float output value.
// - add range_bias to get step (2) product
// - compensate for input scaling. This does not depend on output scaling.
//
static inline float
convert_intermed_to_outval( struct op_scaling_parms const * osp, int val )
{
	return (val + osp->range_bias)*osp->all_in_scale_recip;
}
// this converts the limits at step (3) of the process to a 'quantized' output, using
// the same process as the algo, and checks to see if they are in range for the output.
// returns:
//  0     ok
//  1     min clipping
//  2     max clipping
//  3     both
//
static inline int
check_intermed_range( struct op_scaling_parms const * osp, int minval, int maxval )
{
	minval = ((minval  * osp->post_scale + 16384) >> 15) + osp->adj_bias;
	maxval = ((maxval  * osp->post_scale + 16384) >> 15) + osp->adj_bias;
	int r = osp->post_rsh;
	if( r > 0){
		minval = ((minval>>(r-1)) + 1)>>1;
		maxval = ((maxval>>(r-1)) + 1)>>1;
	}
	return ((minval < 0)? 1: 0) +  ((maxval > 255)? 2: 0);
}

//
// parameter pack for the low-level function
//
struct  core_mul_d32_parms {
	int height;
	int width;
	uint8_t const * ptrA;
	int32_t row_stride_A;
	uint8_t const * ptrB;
	int32_t row_stride_B;
	int32_t d32_stride_B;	// used only when combining depth segs.
	uint8_t* pout;
	int32_t row_stride_out;
	uint16_t  d_here;			// depth in current op; 1..32


	struct op_scaling_parms osp;
	int16_t * minmax;		// store min/max here if not null.
	void (*core_oper_funcp)( struct core_mul_d32_parms const *prmsv );
	nn_sem_t donesem;
};
void core_mul_d32_reference( struct core_mul_d32_parms const * );

void core_mul_d32_hvx( struct core_mul_d32_parms const *);
static void core_oper_thread( struct nn_graph *nn, void * prmsv );

static int mul_d32_execute_common(struct nn_node *self, struct nn_graph *nn, int use_hvx);
//
// 6 inputs:
// 0,1 tensor_A, tensor_B
// 2,3  min_a,   min_b
// 4,5  max_a,   max_b
//
static int mul_d32_execute_ref(struct nn_node *self, struct nn_graph *nn){
	return mul_d32_execute_common( self,nn, 0);
}
static int mul_d32_execute(struct nn_node *self, struct nn_graph *nn){
	return mul_d32_execute_common( self,nn, 1);
}

static int mul_d32_execute_common(struct nn_node *self, struct nn_graph *nn, int use_hvx)
{
	const struct tensor *tensorA = self->inputs[0];
	const struct tensor *tensorB = self->inputs[1];
	const struct tensor **minmax_tensors = &self->inputs[2];		// a_min, a_max, b_min, b_max
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];


	logmsg(nn,2,"mul_d32 execute. self=%p ",self);


	int compat_flags = check_compatible_elementwise_d32( nn, "mul_d32", tensorA, tensorB, compat_ALL| compat_AtoB);

	if(compat_flags<0) return compat_flags;

	int toggleAB = 0;
	// it may be that A is broadcasting onto B
	// in which case we need to swap A,B (and arrange
	// for the min/max to be swapped)
	if( compat_flags & compat_AtoB){
		toggleAB = 1;
		const struct tensor * t = tensorA;
		tensorA = tensorB;
		tensorB = t;
	}
	//logmsg(nn,0,"mul_d32 compat_flags = %X", compat_flags);

	struct shape out_shape = tensorA->shape;
	struct core_mul_d32_parms op_parms;
	struct op_scaling_parms * opscales = &op_parms.osp;

	// do all the scaling calculations which depend on the input ranges.
	//

	float inmin[2], inmax[2];
	float all_in_scale = 1.0f;
	int is_zero_min = 1;

	// scaling...
	for (int i =0; i < 2; i++){
		float in_min = tensor_get_float(minmax_tensors[2*i],0);
		float in_max = tensor_get_float(minmax_tensors[2*i+1],0);
		float rstp = 255.0f/(in_max-in_min);
		all_in_scale *= rstp;
		int zv = saturate_u8(roundf_i32(-rstp*in_min));
		opscales->zeroval[i^toggleAB] = zv;
		if( zv != 0) is_zero_min = 0;
		inmin[i^toggleAB] = fminf(in_min,0.0f);
		inmax[i^toggleAB] = fmaxf(in_max,0.0f);
	}
	opscales->all_in_scale = all_in_scale;
	opscales->all_in_scale_recip = 1.0f/all_in_scale;

	// here is a formula for range_bias
	// which always gives a result which is zero in the 8 lsbs, in range -32512 .. 32512
	// e.g. if both are zero, it evaluates to 32512
	// the range of products, 0..65025 thus becomes -32512 .. 32513 after subtracting this.
	//
	opscales->range_bias =
			(((2*opscales->zeroval[0]-255) * (2*opscales->zeroval[1]-255) +256) >> 9) *256;

	// given in_min0, in_max[0], in_min[1], in_max[1], and assuming
	//   in_min <= 0 ,  in_max >=0 , in_max > in_min:
	// - largest and smallest output values are given by the below
	float out_max_all = fmaxf( inmax[0]*inmax[1], inmin[0]*inmin[1]);		// should be >= 0
	float out_min_all = fminf( inmax[0]*inmin[1], inmin[0]*inmax[1]);		// should be <=0
	out_max_all = fmaxf( out_max_all, out_min_all+ 0.001f);
	/*
	printf("dims= %d:%d:%d:%d\n", (int)out_shape.batches, (int) out_shape.height, (int) out_shape. width, (int) out_shape.depth);
	printf(" tA = %p, hp = %d:%d   wp = %d:%d   dp = %d:%d  layout= %d\n",
			tensorA, tensorA->format.height_pad[0],tensorA->format.height_pad[1],
			tensorA->format.width_pad[0],tensorA->format.width_pad[1],
			tensorA->format.depth_pad[0],tensorA->format.depth_pad[1], tensorA->format.layout);
	printf(" tB = %p, hp = %d:%d   wp = %d:%d   dp = %d:%d  layout= %d\n",
			tensorB, tensorB->format.height_pad[0],tensorB->format.height_pad[1],
			tensorB->format.width_pad[0],tensorB->format.width_pad[1],
			tensorB->format.depth_pad[0],tensorB->format.depth_pad[1], tensorB->format.layout);
	*/

	// figure out the padding
	if( tensorA->format.depth_pad[0]!= 0 || tensorB->format.depth_pad[0] != 0)return errlog(nn,"depth pad before !=0");

	int out_width_pad_before = tensorA->format.width_pad[0];

	int depth_end =  out_shape.depth;

	// number of depth slices we need to do.
	int d32_count = (unsigned)( depth_end + 31)/32;


	//
	// allocate the output tensor
	//  - most padding is copied from input A
	//
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		out_shape.batches,
		out_shape.height, tensorA->format.height_pad[0],tensorA->format.height_pad[1],
		out_shape.width,  tensorA->format.width_pad[0],tensorA->format.width_pad[1],
		out_shape.depth, 0, d32_count*32-depth_end,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,out_shape.batches,out_shape.height,out_shape.width,out_shape.depth);
	}
	///
	/// find all the strides (force row stride B to 0 when broadcasting height)
	///
	int32_t inputA_d32_stride = tensor_d32_stride_d32( tensorA);
	op_parms.row_stride_A = tensor_row_stride_d32( tensorA);

	int32_t inputB_d32_stride = tensor_d32_stride_d32( tensorB);
	int32_t inputB_batch_stride = ((compat_flags& compat_broadcast_B)!=0)? 0: tensor_batch_stride_d32( tensorB);
	op_parms.d32_stride_B = inputB_d32_stride;
	op_parms.row_stride_B = ((compat_flags& compat_broadcast_H)!=0)? 0: tensor_row_stride_d32( tensorB);

	int32_t output_d32_stride = tensor_d32_stride_d32( out_tensor);
	op_parms.row_stride_out = tensor_row_stride_d32( out_tensor);

	op_parms.height = out_shape.height;
	op_parms.width = out_shape.width;
	float out_min, out_max;					// output limits

	//broadcasting from (1,1,*,*) is a special case
	void *alloc_tmp_buf = NULL;
	uint8_t const * B_tensor_base_data = tensor_location_bhw_d32( tensorB, 0,0,0);

	// broadcast from a single value?
	if( (compat_flags & compat_broadcast_ALL) == compat_broadcast_ALL){
		uint8_t b_value_u8 = B_tensor_base_data[ tensorB->format.depth_pad[0]];	// get the value
		float b_value = b_value_u8 * flt_div_255(inmax[1]-inmin[1])  + inmin[1];

		if( fabsf(b_value)> 1e-5f){
			int scale, offs;
			out_min = inmin[0]*b_value;
			out_max = inmax[0]*b_value;
			if( b_value > 0.0f){
				scale = 32767;		// for 'copy'
				offs = 0;
			}else{
				scale = -32768;		// for 1's complement
				offs = 128*255;
				float t = out_min;
				out_min = out_max;
				out_max = t;
			}
			int k = tensor_copy_scaled_d32( nn, tensorA, out_tensor, scale, offs, use_hvx, 3 /*MAX_THREADS */);
			if( k!=0) return k;
			goto done_operation;
		}
	}



	if( (compat_flags & (compat_broadcast_B | compat_broadcast_H))
			== (compat_broadcast_B | compat_broadcast_H) ){
		// both B,H dims are 1 on the B-side.
		// Use temp W buffer if broadcasting on W or D, or if there is misalignment.
		if(  (compat_flags & (compat_broadcast_W | compat_broadcast_D |compat_misalign_ALL ))!= 0 ){
			int32_t new_d32_stride;
			// tensor splatted to full WxD shape, with compatible padding, is constructed in 'scratch'
			// and alloc_tmp_buf -> NULL.
			// if scratch is too small, is done with malloc, alloc_tmp_buf != NULL, and
			// we need to free(alloc_tmp_buf) later.
			uint8_t const * newdata= construct_broadcasted_data_d32( tensorB, tensorA, &new_d32_stride,
					nn->scratch, nn->scratch_size, & alloc_tmp_buf);
			if( newdata == NULL){
				return errlog(nn,"mul_d32: failed to make broadcast row buffer");
			}
			// point 'B' at the new data. batch & row strides are already 0.
			op_parms.d32_stride_B = inputB_d32_stride = new_d32_stride;
			// the new data follows the tensorA alignment.
			B_tensor_base_data = newdata + 32*out_width_pad_before;

			// this cures all misalignments (the copied data is aligned for tensorA).
			compat_flags &= ~compat_misalign_ALL;
		}
	}else{
		if( ((compat_flags& compat_broadcast_W)!=0 && out_shape.width > 1)
		||	 ((compat_flags& compat_broadcast_D)!=0 && out_shape.depth > 1) ){
			return errlog(nn, "mul_d32: can't broadcast on w or d without b & h broadcast");
		}
	}

	int do_side_b_prefetch = 1;

	if( inputB_batch_stride== 0|| op_parms.row_stride_B == 0 )
	{
		do_side_b_prefetch = 0;
	}

	//
	// work is divided up by batches and depth slices into 'work units'.
	// There may be only 1 of these but that's OK.
	// We sequence through these based on current min/max and after each one,
	// if there was an overflow, we reset the min/max to accomodate, and continue with the next.
	// If necessary, we start back at the beginning to re-do  units which overflowed; during the
	// second pass, no range checking is done (since it's faster without).
	// This way it will only need to do the whole thing twice if the highest peak is in the last
	// unit (and if the margin added to previous range isn't enough to cover it).
	//
	// 'clean_work_units' counts the number of units done without overflow; we stop
	// when this is = num_work_units

	int num_work_units = out_shape.batches * d32_count;		// total # of units
	int clean_work_units = 0;
	int second_pass = 0;				// indicates second pass through

	int batch_index = 0;		// 0 .. batches-1
	int depth_index = 0;		// 0 .. d32_count -1

	float out_min0 = out_min_all * 0.25f;		// initial range estimate (prior to padding)
	float out_max0 = out_max_all * 0.25f;

	// a place for the min & max to be stored by the function
	int16_t minmax[2+62] __attribute__((aligned(128))) = {0};// range of internal calc (to detect overflow).

	op_parms.minmax = minmax;	// only on first pass
	// if there are any intra-vector misalignments, use the reference implementation,
	// otherwise use the hvx implementation (if it's enabled).
	op_parms.core_oper_funcp =
			(!use_hvx  || ( compat_flags & compat_misalign_ALL)!= 0 )?core_mul_d32_reference
					: core_mul_d32_hvx;


	while(1){
		if( clean_work_units ==0){		// assume we need to set up scaling
			// first, expand range out_min0 .. out_max0 a little, to reduce
			// likelihood we keep hitting limit. If is_zero_min, we only expand the max.
			float adj = 0.0625f * (out_max0 - out_min0);
			if(is_zero_min){			// special case, no -ve #s
				out_max = out_max0 + 2.0f * adj;
				out_min = 0.0f;
			}else{
				out_max = out_max0 + adj;
				out_min = out_min0 - adj;
				// adjust so that zero point is exact integer
				adjust_minmax_for_zero( & out_min, & out_max );
			}
			logmsg( nn, 2, "%d, %d: Set scaling: %.7f .. %.7f", batch_index, depth_index, out_min, out_max);

			set_scaling_for_output_range( opscales, out_min, out_max);
		}
		// advance pointers - if depth_index = 0, calculate from batch index, otherwise bump by d32_stride
		//
		if( depth_index == 0){
			// (hold B batch index to 0 when broadcasting batch).
			op_parms.ptrA = tensor_location_bhw_d32( tensorA, batch_index, 0,0);
			op_parms.ptrB = B_tensor_base_data +  batch_index * inputB_batch_stride;
			op_parms.pout = tensor_location_bhw_d32( out_tensor, batch_index, 0,0);
		}else{
			op_parms.ptrA += inputA_d32_stride;	// move to next depth slice
			op_parms.pout += output_d32_stride;
			op_parms.ptrB += inputB_d32_stride;
		}

		op_parms.d_here = min_i32( depth_end - 32*depth_index, 32); 		// depth in current slice
		l2pref( op_parms.ptrA, op_parms.height, op_parms.width*32, op_parms.row_stride_A );
		if (do_side_b_prefetch)
			l2pref( op_parms.ptrB, op_parms.height, op_parms.width*32, op_parms.row_stride_B );
		//===============
		nn_sem_init(&op_parms.donesem,0);
		nn_os_work_for_vector(nn, core_oper_thread , &op_parms);
		nn_sem_wait(&op_parms.donesem);
		//===============


		clean_work_units++;	// assuming it was ok

		if( !second_pass){
			// min is stored as -1-min (so vector ops can max-reduce both together)
			int minval = -1-minmax[0];
			int k = check_intermed_range( opscales, minval, minmax[1]);	// check range...
			if( k!=0){
				if( (k&1)!= 0)	// adjust min
					out_min0 = convert_intermed_to_outval( opscales, minval);
				if( (k&2)!= 0)	// adjust max
					out_max0 = convert_intermed_to_outval( opscales, minmax[1]);
				clean_work_units = 0;		// restart from 0 units clean
			}
		}

		// end-of-loop
		//
		if( clean_work_units >= num_work_units)		// all done
			break;
		if( ++ depth_index >= d32_count){	// cycle to next batch
			depth_index = 0;
			if( ++ batch_index >= out_shape.batches){
				batch_index = 0;		// start again (second pass).
				second_pass = 1;
				op_parms.minmax = NULL;	// no minmax calc on second pass
			}
		}
	} // end of work-unit loop

  done_operation:
	if(alloc_tmp_buf != NULL) nn_free( alloc_tmp_buf);

	// store the outputs we decided on.
	tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT);

	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);

	logmsg(nn,2,"mul_d32 %p done",self);
	return 0;
}
// this is called in a thread; it calls the work func via a pointer,
// or core_mul_d32_hvx, and then posts the semaphore.
static void
core_oper_thread( struct nn_graph *nn, void * prmsv )
{
	struct core_mul_d32_parms * prms  = (struct core_mul_d32_parms *)prmsv;
	(*prms->core_oper_funcp)(prms);
	nn_sem_post(&prms->donesem);
}



// reference for the 'core' op.
//
// min & max of the intermediate form are
// stored at minmax, unless minmax is null.
// the 'min' is stored in 1's complement of its actual value, i.e. -1-min.
void
core_mul_d32_reference( struct core_mul_d32_parms const * prms )
{

	int iht, iwid, idep;
	int height = prms->height;
	int width = prms->width;
	uint8_t const * ptrA = prms->ptrA;
	int32_t row_stride_A = prms->row_stride_A;
	uint8_t const * ptrB = prms->ptrB;
	int32_t row_stride_B = prms->row_stride_B;
	uint8_t* pout = prms->pout;
	int32_t row_stride_out = prms->row_stride_out;
	int depth = prms->d_here;


	int za = prms->osp.zeroval[0];
	int zb = prms->osp.zeroval[1];
	int range_bias = prms->osp.range_bias;
	int post_scale = prms->osp.post_scale;
	int adj_bias = prms->osp.adj_bias;
	int post_rsh = prms->osp.post_rsh;
	int rndbias = (1<<post_rsh)>>1;

	int minval = -range_bias;		// use this as the 'baseline' for min/max
	int maxval = -range_bias;		// (corresponds to 'true' zero).

	if( depth == 32){	// full house... do full row in the inner loop.
		depth = 32*width;
		width = 1;
	}

	for(iht = 0; iht < height; iht++ ){

		uint8_t const * rowpa = ptrA + row_stride_A * iht;
		uint8_t const * rowpb = ptrB + row_stride_B * iht;
		uint8_t * rowout = pout + row_stride_out * iht;

		for( iwid = 0; iwid < width; iwid++){
			for( idep = 0; idep < depth; idep++){
				int p1 = (rowpa[idep]-za) * (rowpb[idep]-zb) - range_bias;	// this is i16 range
				minval = min_i32( p1, minval);
				maxval = max_i32( p1, maxval);
				// scale..
				int p2 = (p1 * post_scale + 16384)>>15;
				p2 = saturate_i16( p2 + adj_bias);
				rowout[idep] = saturate_u8(  (p2 + rndbias) >> post_rsh);
			}
			rowpa += 32;
			rowpb += 32;
			rowout += 32;
		}
	}
	int16_t *minmax = prms->minmax;
	if( minmax != NULL){
		minmax[0] = -1-minval;
		minmax[1] = maxval;
	}
}


//
// HVX Version
//
// The first operation
//    (a[i]-za)*(b[i]-zb)-range_bias
//
// is actually done as
//  (za*zb-range_bias) + a[i]*b[i]  -  ( a[i]*zb + b[i]*za)
//
// .. where each product is u8*u8 -> u16, and all the +/- are mod 64K.
// This eliminates the need to zero-extend anything
//
// when 'minmax' is not null, we need it to aligned at a multiple of 128

//--------------- inlines---------------
static inline HVX_VectorPair first_multiply( HVX_Vector vinA, HVX_Vector vinB, int32_t za, int32_t zb, int32_t mulbias){

	HVX_Vector vmb = Q6_V_vsplat_R(mulbias);

	// [(za*zb-range_bias) + a[i]*b[i] ]  -  [ a[i]*zb + b[i]*za ]

	HVX_VectorPair prod1 = Q6_Wuh_vmpyacc_WuhVubVub( Q6_W_vcombine_VV(vmb,vmb), vinA, vinB);

#if __HEXAGON_ARCH__ < 65
	HVX_VectorPair prod2 = Q6_Wuh_vmpyacc_WuhVubRub( Q6_Wuh_vmpy_VubRub( vinA,zb), vinB, za);
#else
	// za = za:zb:za:zb
	HVX_VectorPair prod2 = Q6_Wh_vmpa_WubRub( Q6_W_vcombine_VV(vinB,vinA),za);
#endif

	return Q6_Wh_vsub_WhWh( prod1, prod2);
}

// the rest of the op is: scale both values by post_scale, add adj_bias (with sat) and then >> rsh
static inline HVX_Vector second_multiply( HVX_VectorPair vin, int32_t post_scale, int32_t adj_bias, int rsh )
{
	HVX_Vector vadj = Q6_V_vsplat_R(adj_bias);

	HVX_Vector p0 = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhRh_s1_rnd_sat( Q6_V_lo_W( vin), post_scale ),  vadj );
	HVX_Vector p1 = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhRh_s1_rnd_sat( Q6_V_hi_W( vin), post_scale ),  vadj );
	return Q6_Vub_vasr_VhVhR_rnd_sat( p1, p0, rsh);
}

//-----------------------------------------
void
core_mul_d32_hvx( struct core_mul_d32_parms const * prms )
{

	int height = prms->height;
	int width = prms->width;
	uint8_t const * ptrA = prms->ptrA;
	int32_t row_stride_A = prms->row_stride_A;
	uint8_t const * ptrB = prms->ptrB;
	int32_t row_stride_B = prms->row_stride_B;
	uint8_t* pout = prms->pout;
	int32_t row_stride_out = prms->row_stride_out;

	HVX_VectorPred qmask1 = hvx_make_d32_range_mask( 0, prms->d_here );


	int iht,iwd;
	int za = prms->osp.zeroval[0];
	int zb = prms->osp.zeroval[1];
	int range_bias = prms->osp.range_bias;
	int post_scale = prms->osp.post_scale;
	int adj_bias = prms->osp.adj_bias;
	int post_rsh = prms->osp.post_rsh;

	int mul_bias = za*zb - range_bias;
	int za_splat = Q6_R_vsplatb_R(za);		// za:za:za:za
#if __HEXAGON_ARCH__ < 65
	za = za_splat;		// za:za:za:za
	zb = Q6_R_vsplatb_R(zb);		// zb:zb:zb:zb
#else
	za = (za <<8)| (0xff & zb);		// (zb not used in loop)
	za = Q6_R_combine_RlRl( za,za);	        //  za:zb:za:zb
#endif

	mul_bias  = Q6_R_combine_RlRl(mul_bias,mul_bias);
	post_scale = Q6_R_combine_RlRl( post_scale,post_scale);
	adj_bias = Q6_R_combine_RlRl( adj_bias, adj_bias);


	//
	// find # of vector loops
	//
	int wpad_bytes = (size_t)ptrA & 0x60;	// 0,32,64 or 96
	int wlen = width*32 + wpad_bytes;		// determines right mask
	int nvecw = (unsigned)(wlen-1)/128;		// # of vecs, -1.


	HVX_Vector const *vinpA_0 = (HVX_Vector const *)( ptrA - wpad_bytes);
	HVX_Vector const *vinpB_0 = (HVX_Vector const *)( ptrB - wpad_bytes);
	HVX_Vector *voutp_0 = (HVX_Vector  *)( pout - wpad_bytes);



	if( prms->minmax == NULL ){		// fast version (no min/max calc needed)
		for(iht = 0; iht < height; iht ++){
			HVX_Vector const *vinpA = vinpA_0;
			HVX_Vector const *vinpB = vinpB_0;
			HVX_Vector *voutp = voutp_0;

			//
			// start up..
			//
			HVX_VectorPair vtmp = first_multiply( *vinpA ++, *vinpB++, za,zb, mul_bias );

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				HVX_Vector vout = second_multiply( vtmp, post_scale, adj_bias, post_rsh);
				vtmp = first_multiply( *vinpA ++, *vinpB++, za,zb, mul_bias );
				*voutp++= vout;
			}
			//
			// last one
			//
			*voutp = second_multiply( vtmp, post_scale, adj_bias, post_rsh);

			vinpA_0 = (HVX_Vector const *)(  (char const *)vinpA_0 + row_stride_A);
			vinpB_0 = (HVX_Vector const *)(  (char const *)vinpB_0 + row_stride_B);
			voutp_0 = (HVX_Vector  *)(  (char *)voutp_0 + row_stride_out);
		}
	}else{

		//
		// more complex loop when we are also finding the max range of the intermediate.
		//
		// When finding the min/max, we need to exclude values outside the depth
		// range, or within width padding.
		// This is done as follows:
		//   - keep track of min & max of all vtmp results. min/max are initialized to -range_bias,
		//     which corresponds to zero 'application' result.
		//   - mux is applied to all values read from 'vinpA', forcing values
		// 	  to za if they are outside depth range, or within left-padding. This is sufficient to
		//     force these vtmp value to -range_bias
		//   - for right edge, the final vtmp result is qualified with a mux op.
		//
		HVX_Vector vCenter = q6op_Vh_vsplat_R(-range_bias);
		HVX_Vector vmin = vCenter;
		HVX_Vector vmax = vCenter;
		HVX_Vector vza = Q6_V_vsplat_R(za_splat);	// za_splat is already splat u8->4xu8

		for(iht = 0; iht < height; iht ++){
			HVX_Vector const *vinpA = vinpA_0;
			HVX_Vector const *vinpB = vinpB_0;
			HVX_Vector *voutp = voutp_0;

			//
			// start up..
			//
			HVX_VectorPred qleft =  Q6_Q_vsetq_R(wpad_bytes);
			HVX_VectorPred qmask = Q6_Q_and_QQn( qmask1, qleft);
			HVX_Vector vinA = Q6_V_vmux_QVV( qmask, *vinpA++, vza);	// apply mask - replace 'out' with za
			HVX_VectorPair vtmp = first_multiply( vinA, *vinpB++, za,zb, mul_bias );

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				vmin = Q6_Vh_vmin_VhVh( vmin, Q6_Vh_vmin_VhVh( Q6_V_hi_W(vtmp), Q6_V_lo_W(vtmp)));
				vmax = Q6_Vh_vmax_VhVh( vmax, Q6_Vh_vmax_VhVh( Q6_V_hi_W(vtmp), Q6_V_lo_W(vtmp)));

				HVX_Vector vout = second_multiply( vtmp, post_scale, adj_bias, post_rsh);
				vinA =  Q6_V_vmux_QVV( qmask1, *vinpA++, vza);	// apply 'general' mask
				vtmp = first_multiply( vinA, *vinpB++, za,zb, mul_bias );
				*voutp++ = vout;
			}
			// now, before last min/max op, trim any trailing edge
			// (when nvecw=0, vtmp has already had left masking applied)
			//
			{
				HVX_VectorPred qnright = q6op_Q_vsetq2_R(wlen);
				HVX_Vector vt0 = Q6_V_vmux_QVV( qnright,  Q6_V_lo_W(vtmp), vCenter);
				HVX_Vector vt1 = Q6_V_vmux_QVV( qnright,  Q6_V_hi_W(vtmp), vCenter);
				vmin = Q6_Vh_vmin_VhVh( vmin, Q6_Vh_vmin_VhVh(vt1,vt0));
				vmax = Q6_Vh_vmax_VhVh( vmax, Q6_Vh_vmax_VhVh(vt1,vt0));

				*voutp = second_multiply(  vtmp, post_scale, adj_bias, post_rsh);
			}
			vinpA_0 = (HVX_Vector const *)(  (char const *)vinpA_0 + row_stride_A);
			vinpB_0 = (HVX_Vector const *)(  (char const *)vinpB_0 + row_stride_B);
			voutp_0 = (HVX_Vector  *)(  (char *)voutp_0 + row_stride_out);
		} // end height loop

		//
		// now we need to reduce min/max; we have 64 of each
		// first shuffle ~min with max, and reduce by max oper
		// then we have 32 pairs of {~min, max}
		{
			HVX_VectorPair sh = Q6_Wh_vshuffoe_VhVh( vmax, Q6_V_vnot_V(vmin));
			vmax = Q6_Vh_vmax_VhVh( Q6_V_hi_W(sh), Q6_V_lo_W(sh));
			int k = 4;
			int i;
			//finish the operation across all lanes
			for( i = 0; i < 5 ; i++){
				sh = Q6_W_vshuff_VVR( vmax, vmax, k);
				k <<= 1;
				vmax = Q6_Vh_vmax_VhVh( Q6_V_hi_W(sh), Q6_V_lo_W(sh));
			}
			// now all 32  4-byte lanes have the same (~min,max).

			*(HVX_Vector *)prms->minmax = vmax;
		}
	}
}


static int mul_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	int k;
	logmsg(nn,2,"Checking mul_d32 node %p",self);

	k = node_check_inputs_outputs_n( self,nn, "mul_d32", 6, 3);
	if( k!= 0) return k;

	logmsg(nn,2,"mul_d32 node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedMul_8x8to8_d32 = {
	.execute = mul_d32_execute,
	.check = mul_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedMul_8x8to8_d32_ref = {
	.execute = mul_d32_execute_ref,
	.check = mul_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};
