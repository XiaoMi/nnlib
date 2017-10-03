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
 *
 *
 * This contains implementations for add_d32
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
//#define TEST_PERFORMANCE
#include "hvx_inlines.h"

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>

// If HVX_WITH_BYTEMASKED_STORE is defined, byte-masked stores (in intrinsics) will be used,
// to avoid writing to any bytes which are in the depth-padding or width-padding regions.
// Requires 8.x compiler.
// (not needed; could be faster without)
#ifdef HEXAGON_COMPILER_GE_8_0
//#define HVX_WITH_BYTEMASKED_STORE
#endif

//
// operator to execute.
// the 'subtract_rev' is for use when the op is A-B but A is broadcast to B;
// in that case we reverse the operands, and change the op to op_subtract_rev.
enum addsub_op {
	op_add,
	op_subtract,
	op_subtract_rev
};

// scaling parms
//
struct op_scaling_parms {
	int16_t a_scale;
	int16_t b_scale;
	uint16_t scales_hi;	// repacked for the hvx code.
	uint16_t scales_lo;
	int16_t offset;
	int16_t final_rsh;
	int16_t intermed_zero;	// the step (2) value corresponding to a 'zero'.
	int32_t intermed_min, intermed_max;	// intermed values in this range won't clip at the output.
	float netscale;			// used to transform 'intermediate' result to application result

	// parameters from input ranging
	int16_t zval[2];		// the 'zero' codes for each input
	float stepsize[2];		// the (max-min)/255.0 for each input
};
// The operation is done as:
//   (1) multiply a[i]*256 by ascl, and b[i]*256 by bscl, add products together in 32 bits; result is  >= 0, <2^31
//   (2) >> 16 and treat as  i16  ( >= 0, <= 2^15)
//   (3) add a 16-bit offset using saturated add
//   (4) >>final_rsh and saturate to u8.
// ascl and bscl are the two output scales, relative to a common exponent (defined by final_rsh).
// We have a constraint that each of ascl,bscl must be <= 16383, which means the sum in (1) is < 2^31).
// Also, for 'add', we need their sum to be <= 16383, due to how the hvx calculation works.
// 'offset' is chosen so that 0+0 = 0
//  I.e. after working out the scaling, 'offset' is the amount which gives for 'zero' inputs, a value at (3) of
//    output_zero << final_rsh.
// If the calculated offset falls outside the i16 range, it can be saturated; this situation means that all
// possible outputs shoulde be 0 or 0xff, based on the ranges.
//
// for adaptive ranging, we keep track (at step (2) of the min and max values, which can be translated to application-level
// floats.
//
// To do subtraction with the same datapath:
//   - ascl can range over  0..32767
//   - bscl is a negative number  -32767..0
//   (this will only work if the 'b' mul can be s16 x u16)
//
//
// There is a limitation:
//    a_scale_f and b_scale_f must both <= 63.984, otherwise the method is infeasible
//  (the 'rsh' would need to be < 0).
// In terms of ranges, this means that the output range must be at least 1/63.98. of the largest
// of the input ranges; if we make it at least 1/63.98 of the *sum* of the input ranges (which
// is the actual worst-case output range) then that's sufficient. In order to make it less likely
// to need re-running, we use 1/8 of the hypothetical input range when there's no other information.
//
//
#define OP_ADD_D32_MAX_THREADS 2
#define SCRATCH_RESERVE  (OP_ADD_D32_MAX_THREADS*128)

static inline int
set_scaling_for_output_range( struct op_scaling_parms * osp, int operator, float out_min, float out_max )
{

	// find a scale, b_scale as floats
	float out_scl = 255.0f/(out_max-out_min);		// out scale
	float out_z = -out_min * out_scl;				// the 'zero point'

	//printf("output range = %.6f..%.6f; z = %.6f\n", out_min, out_max, out_z);

	float a_scale_f = out_scl * osp->stepsize[0];
	float b_scale_f = out_scl * osp->stepsize[1];
	float scaleval = a_scale_f+b_scale_f;

	if(operator != op_add ){
		scaleval = fmaxf(a_scale_f, b_scale_f);
		if( operator == op_subtract){
			 b_scale_f = -b_scale_f;
		}else{
			 a_scale_f = -a_scale_f;
		}
	}


	//printf("float scales: %.7f, %.7f\n", a_scale_f, b_scale_f);

	// find exponent. The scale factor here is so that neither of
	// a_scale, b_scale will exceed 16384. (the value is 16384./16380.)
	// scexp should be >= -1 generally; if there is range expansion, it could
	// be less, and we limit rsh to <= 7 which will force smaller a_scale,b_scale.
	//
	// For subtraction, a_scale and b_scale have opposite signs; and both
	// must have abs value < 16384.
	// A value of scexp > 6 is infeasible; this would require rsh < 0
	//
	int scexp = flt_getexp(scaleval* (float)(16384./16380.) );
	if( scexp > 6)				// req. scale is too large
		return -1;
	int rsh = min_i32(6-scexp,7);  // should be 0..6 (maybe 7 sometimes)

	// determine the quantized scale factors
	float rsh_scl = (float)(1<< rsh);
	int a_scale = roundf_i32(a_scale_f  * (rsh_scl * 256.0f));
	int b_scale = roundf_i32(b_scale_f  * (rsh_scl * 256.0f));
	// work out the zero point
	//
	int intzero = (osp->zval[0]*a_scale + osp->zval[1] * b_scale) >> 8;	// step 2 result for 'zero' input
	int offset = roundf_i32( out_z * rsh_scl) - intzero;

	/*printf("scale = %d, %d; offset = %d; final_rsh = %d; int_zero = %d\n",
	a_scale, b_scale, offset, rsh, intzero );*/

	osp->final_rsh = rsh;
	osp->a_scale = a_scale;
	osp->b_scale = b_scale;
	osp->intermed_zero = intzero;
	osp->offset = saturate_i16(offset);
	osp->netscale = 1.0f / (out_scl * rsh_scl);

	// special packing for hvx code
	// scales_lo has the 7 lsbs of each scale (zero extended)
	// scales_hi has bits [14:7] of each scale
	osp->scales_lo = (a_scale & 0x7f) | ( (b_scale & 0x7f) << 8);
	osp->scales_hi = ( (a_scale >>7) & 0xFF) | ( (b_scale <<1) & 0xFF00);
	//
	// find intermed_min, intermed_max ; range of values at (2) which won't
	// clip  at the output.
	// note that these may fall outside the range (0..32767) of (2) results,
	// which just means that clipping can't occur at that endpoint in the particular setup.
	// So it's important to store these as i32, they may not fit in i16.
	//
	int rbias = (1<<rsh)>>1;
	// result from (3) must be at least -rbias to not clip as < 0
	osp->intermed_min = -(offset+rbias);
	// result from (3) must be at most (256<<rsh)-(rbias+1) to not clip as > 255
	osp->intermed_max = osp->intermed_min + (256<<rsh)-1;
	return 0;
}
// this converts a value at step (2) of the process to a float output value.
//
static inline float
convert_intermed_to_outval( struct op_scaling_parms const * osp, int val )
{
	//printf("convert %d:  - %d * %.5f --> %.5f\n", val, osp->intermed_zero,
	//osp->netscale, (val - osp->intermed_zero)*osp->netscale );

	return (val - osp->intermed_zero)*osp->netscale;
}
// this converts the limits at step (2) of the process to a 'quantized' output, using
// the same process as the algo, and checks to see if they are in range for the output.
// returns:
//  0     ok
//  1     min clipping
//  2     max clipping
//  3     both
//
static inline int __attribute__((unused))
check_intermed_range( struct op_scaling_parms const * osp, int minval, int maxval )
{
	return ((minval < osp->intermed_min)? 1: 0) +  ((maxval > osp->intermed_max)? 2: 0);
}
struct core_add_d32_thrinfo {
	struct core_add_d32_runstate * rstp;
	int16_t * minmax;
	uint8_t const * ptrA;
	uint8_t const * ptrB;
	uint8_t* pout;
	uint16_t d_lo, d_hi;			// depth range
// note that d_srcB_lo is always the same as d_lo when using the hvx code; when using the reference
// code, it may differ and the operation may need to gather from two source segments.
	uint16_t d_srcB_lo;
};
//
// parameter pack for the low-level function
//
struct  core_add_d32_runstate {
	int height;
	int width;
	int depth;
	struct tensor_addressing tinA;
	struct tensor_addressing tinB;
	struct tensor_addressing tout;

	struct op_scaling_parms osp;
	void (*core_oper_funcp)( struct core_add_d32_runstate const *rstp, struct core_add_d32_thrinfo * thrp);
	// set when the B input is (1,1,1,D); can use a different inner loop for this case.
	int16_t broadcast_111D;
	int16_t do_side_b_prefetch;
	int16_t find_range;		// nonzero if we are finding range.
	int num_work_units;
	volatile int next_work_unit;
	nn_sem_t donesem;

	struct core_add_d32_thrinfo  thrinfo[OP_ADD_D32_MAX_THREADS];

};

// reference version
static void core_add_d32_reference( struct core_add_d32_runstate const *, struct core_add_d32_thrinfo * thrp );
// reference that can handle 'depth splits'
static void core_add_d32_reference_skewd32( struct core_add_d32_runstate const * , struct core_add_d32_thrinfo * thrp);
// hvx version
static void core_add_d32_hvx( struct core_add_d32_runstate const *, struct core_add_d32_thrinfo * thrp);
// hvx version specialized for b-shape = (1,1,1,d)
static void core_add_d32_hvx_111D( struct core_add_d32_runstate const *, struct core_add_d32_thrinfo * thrp);
static void core_oper_thread( struct nn_graph *nn, void * prmsv );

//static int add_d32_execute_common(struct nn_node *self, struct nn_graph *nn, int use_hvx, int operator);
//
// 6 inputs:
// 0,1 tensor_A, tensor_B
// 2,3  min_a,   max_a
// 4,5  min_b,   max_b
// 6,7  for setting output range (optional)
//
// output min/max may be provided on inputs 6 and 7;
// these are optional; they are considered 'not set' if
//    - # inputs doesn't include it, or the pointer is NULL
//    - value is -INF (for min) or INF for max
//
// if a value is not set, we start with 0.0 (for min) or 0.5
// (for max) and increase as needed when the threshold is exceeded.
//
//

struct addsub_d32_info {
	int16_t min_max_precalc;	// bit 0-> min; bit 1-> max
	int16_t has_run_before;		// true if out_min, out_max are based on previous run
	// if has_run_before, these values are as set on previous run, and are always a proper range
	// if has_run_before=0, only the endpoint(s) specified by min_max_precalc are valid;
	// and if both are valid, they will be a proper range.
	float out_min, out_max;
	// when min and/or max is specified externally, these contain the original
	// specified values. The actual endpoint may move around a bit due to zero correction;
	// the original value is used to keep it from 'drifting' over a series of runs.
	// Note that if an endpoint is specified to be 0, it will
	// not be affected by zero correction.
	float out_min_specified, out_max_specified;

	// save shape analysis from previous run ( if has_run_before)
	struct shape A_shape;
	struct tensor_format A_format;
	struct shape B_shape;
	struct tensor_format B_format;
	int compat_flags;
};

//
//  The main 'execute' function.
//
//
// When the 'B' input has dimension (1,1,1,1), the operation involves adding
//  a scalar to a tensor (or subtracting). This can be simplified; in many cases
// this can be done by copying the u8 data and changing the endpoints. However, given
// that we have a contraint the that 'zero code' is an integer in range 0..255, there
// is some trickiness. The following method is used:
//
//    float b_val = b_code*(bmax-bmin)/255.;		The vale to add (negate if subtracting)
//    float arange = amax-amin;						Existing 'a' range
//    float b_delt_a  = b_val *255/arange			Delta in 'a' units
//    float azero_new = azero - b_delt_a			Where 'azero' is the zero code for a, -amin*255/arange
//
//   Now, in principle, we can add 'b_val' to (amin, amax) to get the new range, and its zero code will
//   be azero_new. But the zero code must be an integer in range 0..255, so we do as as follows:
//
//    (1) if azero_new is in range 0..255, we will use the same quantization (out_range = arange) but we
//      need to adjust the endpoints by a multiple of astep = arange/255. So we do this:
//      	out_min  = -round(azero_new)*astep
//      	out_max  = out_min + arange
//       ... and keep the same encoding
//    (2) if azero_new <  0, it means amin + b_val > 0, so all results from the add are > 0;
//        we must use out_min = 0. We can use out_max = amax + b_val, and will 'squeeze' the coded values
//        to a narrower range, with 255 being invariant:
//           out_min = 0.0
//           out_max = amax + bval
//           float scale_f = arange/out_max
//           float offs_f = 255*(1-scale_f)
//        ... where scale_f, off_f are used to rescale the codes: code' = scale_f * code + offs
///   (3) if azero_new > 255, it means amax + bval < 0, so all results wil be < 0; we must use
//       out_max = 0. We use out_min = amin + b_val, and squeeze the coded values to a narrower range,
//       with zero being invariant.
//           out_min = amin + bval
//           out_max = 0.f
//           float scale_f = -arange/out_min
//           float offs_f = 0
//
// in cases where scale_f is small, <= 0.6, say, it may make sense to allow the full operation to run,  on the
// hopes that the actual range of 'a' will be less than given, and so we'd wind up with less compression of the range.
// (and, before starting, we can re-evaluate the 'worst case' range based on b_val).
// for more severe cases, e.g. scale_f ==0.2, this would make very little difference, even if the range of 'a' is only a small
//  fraction of the indicated range (and is clustered to the proper end of the range) we can only expect the final range to
//  be at best 20% smaller than provided by the above method. Currently, all scalar-add cases are handled as above.
//
//
static int addsub_d32_execute_common(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *tensorA = self->inputs[0];
	const struct tensor *tensorB = self->inputs[1];
	const struct tensor **minmax_tensors = &self->inputs[2];	// Amin, Amax, Bmin, Bmax
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	const float recip255 = 3.9215686e-03f;	// 1/255 as a float
	struct addsub_d32_info * __restrict info  = ( struct addsub_d32_info *)self->opaque;

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	int use_hvx=1;
	int operator = op_add;

	char const * node_name;
	{	// who are we?
		int node_type = self->node_type;
		switch(node_type)
		{
		 case OP_QuantizedAdd_8p8to8_d32:
			break;
		 case OP_QuantizedSub_8p8to8_d32:
			 operator = op_subtract;
			 break;
		 case OP_QuantizedAdd_8p8to8_d32_ref:
			 use_hvx = 0;
			 break;
		 case OP_QuantizedSub_8p8to8_d32_ref:
			 operator = op_subtract;
			 use_hvx = 0;
			 break;
		 default:
			 return errlog(nn,"unexpected node_type = %d", node_type);
		}
		node_name = hexagon_nn_op_names[node_type];
	}

	logmsg(nn,2,"%s execute. self=%p ",node_name,self);

	// we cache the previous analysis of the shapes.

	int compat_flags;
	if( info->has_run_before
			&& shape_matches(&tensorA->shape, &info->A_shape )
			&& format_matches( &tensorA->format, &info->A_format)
			&& shape_matches(&tensorB->shape, &info->B_shape )
			&& format_matches( &tensorB->format ,&info->B_format)){
		compat_flags = info->compat_flags;
	}else{
		compat_flags = check_compatible_elementwise_d32( nn, node_name, tensorA, tensorB, compat_ALL| compat_AtoB);
		if(compat_flags<0)
			return compat_flags;
		info->A_shape = tensorA->shape;
		info->A_format = tensorA->format;
		info->B_shape = tensorB->shape;
		info->B_format = tensorB->format;
		info->compat_flags = compat_flags;
	}

	int toggleAB = 0;
	// it may be that A is broadcasting onto B
	// in which case we need to swap A,B (and arrange
	// for the min/max to be swapped)
	if( compat_flags & compat_AtoB){
		toggleAB = 1;
		const struct tensor * t = tensorA;
		tensorA = tensorB;
		tensorB = t;
		if( operator == op_subtract) operator = op_subtract_rev;
	}
	//logmsg(nn,0,"add_d32 compat_flags = %X", compat_flags);

	struct shape out_shape = tensorA->shape;
	struct core_add_d32_runstate runstate;
	struct op_scaling_parms * opscales = &runstate.osp;


	// do all the scaling calculations which depend on the input ranges.
	//

	float inmin[2], inmax[2];
	int is_zero_min = (operator == op_add)? 1: 0;

	// scaling...
	for (int i =0; i < 2; i++){
		const struct tensor **minmax_tens  = &minmax_tensors[(i^toggleAB)*2]; // select A or B
		float in_min = tensor_get_float(minmax_tens[0],0);
		float in_max = tensor_get_float(minmax_tens[1],0);
		float range = in_max-in_min;
		int zv = saturate_u8(roundf_i32(-255.0f*in_min/range));
		opscales->zval[i] = zv;
		opscales->stepsize[i] = range* recip255;
//printf("input %d: %.6f .. %.6f; z = %d(%.6f); stepsize = %.6f\n", i,
//	in_min, in_max, zv, -255.0f*in_min/range, opscales->stepsize[i]);
		if( zv != 0) is_zero_min = 0;
		inmin[i] = fminf(in_min,0.0f);
		inmax[i] = fmaxf(in_max,0.0f);
	}

	//
	// if either limit is not predetermined, check the worst-case
	// range vs the range we have;
	// if the worst case limit exceeds 16x the current range,
	// extend the current range. This keeps the range measurement
	// from saturating.
	//

	if( info->min_max_precalc < 3 ){
		// given in_min0, in_max[0], in_min[1], in_max[1], and assuming
		//   in_min <= 0 ,  in_max >=0 , in_max > in_min:
		// - largest and smallest output values are given by the below
		float out_max_all;		// should be >= 0
		float out_min_all;		// should be <=0
		switch( operator){
		 default:
		 case op_add:
			out_max_all = inmax[0] + inmax[1];
			out_min_all = inmin[0] + inmin[1];
			break;
		 case op_subtract:
			out_max_all = inmax[0] - inmin[1];
			out_min_all = inmin[0] - inmax[1];
			break;
		 case op_subtract_rev:
			out_max_all = inmax[1] - inmin[0];
			out_min_all = inmin[1] - inmax[0];
			break;
		}

		out_max_all = fmaxf( out_max_all, out_min_all+ 0.001f);

		if( !info->has_run_before){
			// make up our own endpoints; make the range 1/8 of
			// the theoretical range
			float ored_min = 0.125f * out_min_all;
			float ored_max = 0.125f * out_max_all;

			if( info->min_max_precalc == 1){	// move max only
				info->out_min = info->out_min_specified;
				info->out_max = fmax( 0.0f, info->out_min + (ored_max-ored_min));
			}else if( info->min_max_precalc != 0){	// == 2: move min only
				info->out_max = info->out_max_specified;
				info->out_min = fminf(0.0f, info->out_max - (ored_max-ored_min));
			}else{
				info->out_min = ored_min;
				info->out_max = ored_max;
			}
			adjust_minmax_for_zero_with_constraints(
				 &info->out_min, &info->out_max, info->min_max_precalc);
		}else{
			// make sure that the output range is at least 1/32 of the
			// total input range.
			float out_range_all = out_max_all - out_min_all;
			float current_range = info->out_max - info->out_min;
			float d  = out_range_all - 32.0f*current_range;
			if( d > 0){	// expand endpoints
				float r = d/(32.0f*(out_range_all-current_range));
				if ((info->min_max_precalc&1)==0){		// can move min?
					float adj_min = (out_min_all - info->out_min)*r;
					if( adj_min < 0.0f){
						info->out_min  += adj_min;
					}
				}else{	// reset to keeo from drifting
					info->out_min = info->out_min_specified;
				}
				if ((info->min_max_precalc&2)==0){		// can move max ?
					float adj_max = (out_max_all - info->out_max)*r;
					if( adj_max > 0.0f){
						info->out_max  += adj_max;
					}
				}else{
					info->out_max = info->out_max_specified;
				}
				adjust_minmax_for_zero_with_constraints( 
					&info->out_min, &info->out_max, info->min_max_precalc);
			}
		}
	}


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
	int out_width_pad_before = tensorA->format.width_pad[0];
	int out_depth_pad_before = tensorA->format.depth_pad[0];
	int depth_end = out_depth_pad_before + out_shape.depth;

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
		out_shape.depth, out_depth_pad_before, d32_count*32-depth_end,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,out_shape.batches,out_shape.height,out_shape.width,out_shape.depth);
	}

	// get all the addressing info
	//
	runstate.tinA = tensor_addressing_d32( tensorA);
	runstate.tinB = tensor_addressing_d32( tensorB);
	runstate.tout = tensor_addressing_d32( out_tensor);
	//
	// force batch, height strides on B to 0 when broadcasting those dims
	//
	if((compat_flags& compat_broadcast_B)!=0 )
		 runstate.tinB.batch_stride = 0;
	if( (compat_flags& compat_broadcast_H)!=0)
		runstate.tinB.height_stride = 0;

	runstate.height = out_shape.height;
	runstate.width = out_shape.width;
	runstate.depth = out_shape.depth;
	runstate.broadcast_111D = 0;

	float out_min = info->out_min;		// current output range
	float out_max = info->out_max;



	void *alloc_tmp_buf = NULL;
	int ran_again = 0;				// for TEST_PERFORMANCE

	// broadcast from a single value?
	// Only allow when both endpoints are free to move. Otherwise,
	// rely on the general broadcast approach.
	//
	if( info->min_max_precalc == 0 &&  (compat_flags & compat_broadcast_ALL) == compat_broadcast_ALL){
		uint8_t b_value_u8 = runstate.tinB.data[ tensorB->format.depth_pad[0]];	// get the value
		float arange = inmax[0]-inmin[0];
		float astep = arange *recip255;
		// find the actual b value
		float b_val = (b_value_u8 - opscales->zval[1]) * opscales->stepsize[1];
		// subtract?  (for reverse-subtract, we correct later).
		if( operator != op_add) b_val = -b_val;
		float new_zero_code = opscales->zval[0] - b_val / astep;
		int code_scale = 32768;		// will be saturated to i16 later
		int code_offs = 0;

		if( new_zero_code >= -0.25f && new_zero_code <= 255.25f ){	// only shift the endpoints
			// round new_zero_code to form the new zero.
			out_min = -roundf_i32( new_zero_code)*astep;
			out_max = out_min + arange;
		}else {
			if( b_val > 0){		// pin min = 0, extend max
				out_min = 0.0f;
				out_max = inmax[0] + b_val;
			}else{			// pin max = 0, extend min
				out_min = inmin[0] + b_val;
				out_max = 0.0f;
			}
			// find scale for new range
			code_scale = roundf_i32( 32768.0f * arange/(out_max-out_min));
			if( b_val > 0){
				code_scale = saturate_i16(code_scale);
				// make 255 invariant...
				code_offs = 128*255-((255*code_scale) >> 8);
			}
		}
		// codes will be mapped as  out[i] = in[i] * (code_scale/32k)  + (code_offs/128)
		//
		// if reverse sub, change that to 255-(in[i] * (code_scale/32k)  + (code_offs/128))
		//           =  in[i]*( -code_scale/32k) + ((255*128-code_offs)/128)
		//
		if( operator == op_subtract_rev){
			code_scale = -code_scale;
			code_offs = 255*128-code_offs;
			float t = -out_min;
			out_min = -out_max;
			out_max = t;
		}
		code_scale = saturate_i16(code_scale);
		int k = tensor_copy_scaled_d32( nn, tensorA, out_tensor, code_scale, code_offs, use_hvx, OP_ADD_D32_MAX_THREADS);
		if( k!=0) return k;
		goto done_operation;
	}

	// Broadcasting from (1,1,*,*) is a special case
	// Note: we must disable b-side prefetch in all cases that have broadcast_H, otherwise
	// prefetches with row-pitch= 0 will be issued

	if( (compat_flags & (compat_broadcast_B | compat_broadcast_H))
			== (compat_broadcast_B | compat_broadcast_H) ){
		// both B,H dims are 1 on the B-side.
		// Use temp W buffer if broadcasting on W or D, or if there is misalignment.
		if(  (compat_flags & (compat_broadcast_W | compat_broadcast_D |compat_misalign_ALL ))!= 0 ){
			int32_t new_d32_stride;
			// tensor splatted to full WxD shape, with compatible padding, is constructed in 'scratch'
			// and alloc_tmp_buf -> NULL.
			// Skip 'SCRATCH_RESERVE' which is  reserved for per-thread scratch
			// if scratch is too small, is done with malloc, alloc_tmp_buf != NULL, and
			// we need to free(alloc_tmp_buf) later.
			uint8_t * scratch_plus = (uint8_t *) nn->scratch + SCRATCH_RESERVE;
			size_t scratch_remaining =  nn->scratch_size - SCRATCH_RESERVE;
			uint8_t const * newdata= construct_broadcasted_data_d32( tensorB, tensorA, &new_d32_stride,
										 scratch_plus, scratch_remaining, & alloc_tmp_buf);
			if( newdata == NULL){
				return errlog(nn,"%s: failed to make broadcast row buffer",node_name);
			}
			// point 'B' at the new data. batch & row strides are already 0.
			runstate.tinB.d32_stride = new_d32_stride;
			// the new data follows the tensorA alignment.
			runstate.tinB.data = (uint8_t*)newdata + 32*out_width_pad_before;
			runstate.tinB.d0 = out_depth_pad_before;

			// this cures all misalignments (the copied data is aligned for tensorA).
			compat_flags &= ~compat_misalign_ALL;
			if ( (compat_flags & compat_broadcast_W )!= 0 ){	// B is (1,1,1,D)
				runstate.broadcast_111D = 1;
			}
		}else{
			// broadcasting on B & H from actual tensor data. prefetch the whole B side now.
			l2pref( runstate.tinB.data, runstate.tinB.nd32, runstate.width*32, runstate.tinB.d32_stride);
		}
		runstate.do_side_b_prefetch = 0;	// don't prefetch B at run time
	}else{
		if( ((compat_flags& compat_broadcast_W)!=0 && out_shape.width > 1)
		||	 ((compat_flags& compat_broadcast_D)!=0 && out_shape.depth > 1) ){
			return errlog(nn, "%s: can't broadcast on w or d without b & h broadcast", node_name);
		}
		// b side prefetch: do the whole thing now if broadcasting on B or H;
		// otherwise do it as we progress.
		int do_b_prefetch = 1;
		if( (compat_flags& (compat_broadcast_B|compat_broadcast_H))!= 0 ){
			// exactly 1 of compat_broadcast_B, compat_broadcast_H is set...
			int pf_width =  runstate.width*32;
			int pf_stride, pf_height;
			if( compat_flags& compat_broadcast_B ){	// one batch; height is height * nd32
				pf_stride = runstate.tinB.d32_stride;
				pf_height = runstate.height * runstate.tinB.nd32;
			}else{
				// H but not B: load first depth slice from each row.
				pf_stride = runstate.tinB.batch_stride;
				pf_height = out_shape.batches;
			}
			l2pref( runstate.tinB.data, pf_height, pf_width, pf_stride);
			do_b_prefetch =0;
		}
		runstate.do_side_b_prefetch = do_b_prefetch;
	}


	runstate.num_work_units = out_shape.batches * d32_count;		// total # of units

	int num_threads = min_i32( runstate.num_work_units, OP_ADD_D32_MAX_THREADS);


	// a place for the min & max to be stored by the function
	// each thread gets one vector.
	int16_t * minmax_loc = (int16_t *)nn->scratch;
	for(int i = 0; i < num_threads; i++){
		runstate.thrinfo[i].rstp = & runstate;
		runstate.thrinfo[i].minmax = minmax_loc;
		minmax_loc += 64;
	}

	// if there are any intra-vector misalignments, use the reference implementation,
	// otherwise use the hvx implementation (if it's enabled).
	runstate.core_oper_funcp =
			(!use_hvx  || ( compat_flags & compat_misalign_ALL)!= 0 )?core_add_d32_reference
					: runstate.broadcast_111D ?  core_add_d32_hvx_111D: core_add_d32_hvx;
	// if the depth is 'skewed', we need to use a special version of the reference routine.
	if( compat_flags & compat_skewed_D ){
		runstate.core_oper_funcp = core_add_d32_reference_skewd32;
	}
	runstate.find_range = (info->min_max_precalc != 3);	// find range if any endpoint can be moved.


	while(1){
		logmsg( nn, 2, "Set scaling: %.7f .. %.7f", out_min, out_max);

		if( set_scaling_for_output_range( opscales, operator, out_min, out_max) != 0 ){
			// this should happen only when both endpoints are given, and when the range given is
			// a very small fraction (<1/63.9) of the range of either input.
			return errlog(nn, "scaling failed for output range %.7f .. %.7f", out_min, out_max);
		}


		// run all jobs
		runstate.next_work_unit = 0;
		nn_sem_init(&runstate.donesem,0);
		for( int i = 0; i < num_threads;i++)
			nn_os_work_for_vector(nn, core_oper_thread , &runstate.thrinfo[i]);

		for( int i = 0; i < num_threads;i++)
			nn_sem_wait(&runstate.donesem);

		if( !runstate.find_range)
			break;
		//
		// get the min and max values stored by the threads
		// and combine them.
		//
		int minvali = runstate.thrinfo[0].minmax[0];
		int maxvali = runstate.thrinfo[0].minmax[1];
		for (int i  =1 ; i < num_threads; i++ ){
			int16_t const * p = runstate.thrinfo[i].minmax;
			minvali = max_i32( minvali, p[0]);
			maxvali = max_i32( maxvali, p[1]);
		}
		minvali = -1-minvali;

		float actual_min = convert_intermed_to_outval( opscales, minvali);
		float actual_max = convert_intermed_to_outval( opscales, maxvali);
		float out_range = actual_max - actual_min;
		// pad the range out a bit
		out_range = out_range + fmaxf(0.15f*out_range, 1e-4);
		float padded_min = fmin(0.0f, actual_max-out_range);
		float padded_max = fmax(0.0f, actual_min+out_range);
		int any_adj = 0;

		int min_max_precalc = info->min_max_precalc;

		if ( (min_max_precalc & 1) == 0 &&  actual_min < out_min){
			logmsg(nn,2,"adjusting out_min from %f to %f (actual min is %f)", out_min, padded_min, actual_min);
			out_min = padded_min;
			any_adj= 1;
		}
		if ( (min_max_precalc & 2) == 0&& actual_max > out_max){
			logmsg(nn,2,"adjusting out_max from %f to %f (actual max is %f)", out_max, padded_max, actual_max);
			out_max = padded_max;
			any_adj= 1;
		}
		if( !any_adj)
			break;	// all done, if no adjustment needed

		// reset original endpoints, where applic.
		// (so that small tweaks for zero correction don't accumulate)
		//
		if( (min_max_precalc & 1) ) out_min = info->out_min_specified;
		if( (min_max_precalc & 2) ) out_max = info->out_max_specified;
		info->out_min = out_min;
		info->out_max = out_max;

		if( out_min < 0.0f){
			// correct range and reload
			adjust_minmax_for_zero_with_constraints( &info->out_min, & info->out_max, min_max_precalc);
			out_min = info->out_min;
			out_max = info->out_max;
		}
		runstate.find_range = 0;
		// loop and do again
		ran_again = 1;
	}

 done_operation:
	if(alloc_tmp_buf != NULL) nn_free( alloc_tmp_buf);

	// store the outputs we decided on.
	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);
	info->has_run_before = 1;

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("add_d32 %s cycles = %d (elements = %d) ran_again = %d\n",
			use_hvx?"hvx": "ref",(end_time-start_time),
			(int)tensor_element_count(out_tensor), ran_again);
#endif
	logmsg(nn,2,"%s %p done",node_name, self);
	return 0;
}
// this is called in a thread; it calls the work func via a pointer,
// or core_add_d32_hvx, and then posts the semaphore.
static void
core_oper_thread( struct nn_graph *nn, void * thrinfov )
{
	struct core_add_d32_thrinfo * thrinfo = (struct core_add_d32_thrinfo *)thrinfov;
	struct core_add_d32_runstate * rstp  = thrinfo->rstp;
	int ijob;
	int nd32 = rstp->tout.nd32;
	int ibatch, id32;
	int d0_a = rstp->tinA.d0;	// input A and output have same d0
	int d0_b = rstp->tinB.d0;	// usually the same as d0_a; not always
	int dskew = d0_b-d0_a;
	int depth = rstp->depth;	// total depth;

	int16_t * minmaxp = thrinfo->minmax;
	int minmaxbase = rstp->osp.intermed_zero;		// use this as the 'baseline' for min/max
	minmaxp[0] = -1 - minmaxbase;
	minmaxp[1] = minmaxbase;

	int inA_batch_stride = rstp->tinA.batch_stride;
	int inB_batch_stride = rstp->tinB.batch_stride;
	int out_batch_stride = rstp->tout.batch_stride;

	int inA_d32_stride = rstp->tinA.d32_stride;
	int inB_d32_stride = rstp->tinB.d32_stride;
	int out_d32_stride = rstp->tout.d32_stride;

	int pf_height = rstp->height;
	int pf_width = rstp->width*32;

	while( ijob = __sync_fetch_and_add( &rstp->next_work_unit, 1),  ijob < rstp->num_work_units){
		// transform to batch & depth id
		id32 = ijob;
		ibatch = 0;
		if( id32 >= nd32 ){
			ibatch = id32/nd32;
			id32 -= ibatch*nd32;
		}
		uint8_t * ptrA = rstp->tinA.data + ibatch * inA_batch_stride;
		uint8_t * ptrB = rstp->tinB.data + ibatch * inB_batch_stride;
		uint8_t * pout = rstp->tout.data + ibatch * out_batch_stride;

		// set depth range and adjust for depth chunk.
		if( id32 ==0 ){
			thrinfo->d_lo = d0_a;
			thrinfo->d_hi = min_i32(32, d0_a+depth);
			thrinfo->d_srcB_lo = d0_b;
		}else{
			int dn = min_i32( d0_a + depth - 32*id32,32);	// length of this chunk
			thrinfo->d_lo = 0;
			thrinfo->d_hi = dn;
			thrinfo->d_srcB_lo = dskew &31;
			ptrA += id32 * inA_d32_stride;
			// if dskew < 0, start B in previous depth slice
			ptrB += (id32+ (dskew>>31)) * inB_d32_stride;
			pout += id32 * out_d32_stride;

		}
		// l2 prefetch
		l2pref( ptrA, pf_height, pf_width, rstp->tinA.height_stride);
		if( rstp->do_side_b_prefetch)
			l2pref( ptrB, pf_height, pf_width, rstp->tinB.height_stride);
		//
		thrinfo->ptrA = ptrA;
		thrinfo->ptrB = ptrB;
		thrinfo->pout = pout;
		//printf("Thread @ %02x: batch %d, id32 %d [%d..%d]\n", ((unsigned)thrinfo >>3)&0xFF, ibatch, id32,
		// thrinfo->d_lo, thrinfo->d_hi);
		(*rstp->core_oper_funcp)(rstp, thrinfo);
	}
	nn_sem_post(&rstp->donesem);
}

// this is a version of core_add_d32_reference which can handle
// the case where the source bytes are split across two d32 segments.
// it works by making two sub-operations.
// To do this, it is allowed to modify the d_lo, d_hi, d_srcB and pointer
// parms in thrp.
// Note that not all slices will be split.
//
static void
core_add_d32_reference_skewd32( struct core_add_d32_runstate const * rstp , struct core_add_d32_thrinfo * thrp)
{
	int dlo = thrp->d_lo;
	int dhi = thrp->d_hi;
	int depth = dhi - dlo;
	if( depth + thrp->d_srcB_lo <=32 ){	// this one is not split
		core_add_d32_reference(rstp, thrp);
		return;
	}

	int dfirst = (32-thrp->d_srcB_lo);		// do this first
	int dhi2 = dlo + dfirst;				// this is the reduced amount...
	thrp->d_hi = dhi2;

	core_add_d32_reference( rstp, thrp);		// run that
	thrp->ptrB += rstp->tinB.d32_stride;		// advance to next segment of B
	thrp->d_srcB_lo = 0;
	thrp->d_lo = dhi2;				// next output span
	thrp->d_hi = dhi;

	core_add_d32_reference( rstp, thrp);		// run the second one.
}


// reference for the 'core' op.
//
// min & max of the intermediate form are
// stored at minmax, unless minmax is null.
// the 'min' is stored in 1's complement of its actual value, i.e. -1-min.
// Note that this function accumulates min & max on top of the values at thrp->minmax
static void
core_add_d32_reference( struct core_add_d32_runstate const * rstp,  struct core_add_d32_thrinfo * thrp)
{
//   (1) multiply a[i]*256 by ascl, and b[i]*256 by bscl, add products together in 32 bits; result is  >= 0, <2^31
//   (2) >> 16 and treat as  i16  ( >= 0, <= 2^15)
//   (3) add a 16-bit offset using saturated add
//   (4) >>final_rsh and saturate to u8.

	int iht, iwid, idep;
	int height = rstp->height;
	int width = rstp->width;
	uint8_t const * ptrA = thrp->ptrA;
	int32_t row_stride_A = rstp->tinA.height_stride;
	uint8_t const * ptrB = thrp->ptrB;
	int32_t row_stride_B = rstp->tinB.height_stride;
	uint8_t* pout = thrp->pout;
	int32_t row_stride_out = rstp->tout.height_stride;
	// we don't care about alignment here, so just bump all the pointers by dlo
	int dlo = thrp->d_lo;
	int depth = thrp->d_hi - dlo;
	ptrA += dlo;
	ptrB += thrp->d_srcB_lo;
	pout += dlo;

	int a_scale = rstp->osp.a_scale;
	int b_scale = rstp->osp.b_scale;
	int offset = rstp->osp.offset;
	int final_rsh = rstp->osp.final_rsh;
	int rndbias = (1<<final_rsh)>>1;

	int16_t * minmaxp = thrp->minmax;
	int minval = -1-minmaxp[0];
	int maxval = minmaxp[1];

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
				int p1 = (rowpa[idep]*a_scale + rowpb[idep]*b_scale)>>8;	// this is i16 range
				minval = min_i32( p1, minval);
				maxval = max_i32( p1, maxval);
				// offset...
				int p2 = saturate_i16( p1 + offset);
				rowout[idep] = saturate_u8(  (p2 + rndbias) >> final_rsh);
			}
			rowpa += 32;
			rowpb += 32;
			rowout += 32;
		}
	}
	if( rstp->find_range){
		minmaxp[0] = -1-minval;
		minmaxp[1] = maxval;
	}
}

//
// HVX Version
//
//
// when 'minmax' is not null, we need it to aligned at a multple of 4
// (or a multiple of 128, if not HVX_WITH_BYTEMASKED_STORE)

//--------------- inlines---------------

// mul each a byte by 256* a_scale; each b byte by 256 * b_scale; 32 bit result;
// and add them together and >>16.
//
// Rather than doing the input promotion and the four Q6_Wuw_vmpy_VuhRuh, this works as follows:
//  (a) interleave a & b inputs
//  (b) find  hisop = ain[i] * scla_hi  +  bin[i] * sclb_hi;   (16 bit result)
//  (c) find  losop = ain[i] * scla_lo  +  bin[i] * sclb_hi;   (16 bit result)
//     - We then need to find (hisop*128 +losop) >>8
//       which is done as
//   (d) result = avg( hisop,  losop >> 7)
//
//  - all of these are u8 * s8 muls (but the 'scl_lo' are always 0 in the msb)
//   'hisop' won't exceed i16 range, since the sum of the 'high' scales
//    is always <= 127.
//   losop can range up to 0xFD02, ao the >> needs to be lsr.
//
// in the above , scla_lo is bits [6:0] of the actual scale_a, zero extended;
// and scla_hi is bits [14:7]  (scales must be in range -16383 ..  16383);
// sum of scales can't exceed 16384).
// We need to do (b),(c),(d) twice (for odd & even) but the
// advantage of this approach is that (b) and (c) can each be done with a single mpya op,
// directly from the input, which handles both even & odd.
// Also, the same pipe supports b_scale < 0 (for subtract ops).
//
//
static inline HVX_VectorPair first_stage( HVX_Vector vinA, HVX_Vector vinB, int32_t sclab_hi, int32_t sclab_lo)
{
	HVX_VectorPair vinAB = Q6_W_vcombine_VV( vinB, vinA );

	HVX_VectorPair hisop = Q6_Wh_vmpa_WubRb( vinAB, sclab_hi);
	HVX_VectorPair losop = Q6_Wh_vmpa_WubRb( vinAB, sclab_lo);
	HVX_Vector hisop_0 = Q6_V_lo_W(hisop);
	HVX_Vector hisop_1 = Q6_V_hi_W(hisop);
	HVX_Vector losop_0 = Q6_V_lo_W(losop);
	HVX_Vector losop_1 = Q6_V_hi_W(losop);
	// >> these by 7
	losop_0 = Q6_Vuh_vlsr_VuhR( losop_0,7);
	losop_1 = Q6_Vuh_vlsr_VuhR( losop_1,7);
	// average lo & hi to get result.
	return Q6_W_vcombine_VV(
		Q6_Vh_vavg_VhVh(  hisop_1, losop_1),
		Q6_Vh_vavg_VhVh(  hisop_0, losop_0));
}
struct prod_vin_with_scale {
	HVX_VectorPair prod0;
	HVX_VectorPair prod1;
};
// first stage for the case where we are broadcasting from (1,1,1,d):
// in this case 'scale_a' is the scale_a value (+/16k) and vinBs is scale_b * vinB
//
static inline HVX_VectorPair first_stage_111D(  HVX_Vector  vinA,
		struct prod_vin_with_scale vinBs,
		int32_t scale_a)
{
	HVX_VectorPair zxt = Q6_Wb_vshuffoe_VbVb( Q6_V_vzero(), vinA);
	HVX_VectorPair prod0 = Q6_Ww_vmpy_VhRh( Q6_V_lo_W(zxt), scale_a );
	HVX_VectorPair prod1 = Q6_Ww_vmpy_VhRh( Q6_V_hi_W(zxt), scale_a );
	// add to 'b' product and >>8; convert to i16
	HVX_Vector result0 = q6op_Vh_vasr_WwR( Q6_Ww_vadd_WwWw( prod0, vinBs.prod0), 8 );
	HVX_Vector result1 = q6op_Vh_vasr_WwR( Q6_Ww_vadd_WwWw( prod1, vinBs.prod1), 8 );
	return Q6_W_vcombine_VV( result1, result0 );
}
//
// This makes the 'vinBs' input for first_stage_111D
//
static inline
struct prod_vin_with_scale
find_vinB_product( HVX_Vector vinB, int32_t scale_b)
{
	struct prod_vin_with_scale result;
	HVX_VectorPair zxt = Q6_Wb_vshuffoe_VbVb( Q6_V_vzero(), vinB);
	result.prod0 = Q6_Ww_vmpy_VhRh( Q6_V_lo_W(zxt), scale_b );
	result.prod1 = Q6_Ww_vmpy_VhRh( Q6_V_hi_W(zxt), scale_b );
	return result;
}


// the rest of the op is: add offset (with sat) and then >> rsh
static inline HVX_Vector second_stage( HVX_VectorPair vin, int32_t offset, int rsh )
{
	HVX_Vector voffs = Q6_V_vsplat_R(offset);

	HVX_Vector p0 = Q6_Vh_vadd_VhVh_sat( Q6_V_lo_W( vin), voffs);
	HVX_Vector p1 = Q6_Vh_vadd_VhVh_sat( Q6_V_hi_W( vin), voffs );
	return Q6_Vub_vasr_VhVhR_rnd_sat( p1, p0, rsh);
}


#ifdef HVX_WITH_BYTEMASKED_STORE
#define BYTEMASKED_STORE q6op_vstcc_QAV
#else
// this is done as an inline to avoid warnings about unused 'cond' variables
static inline  void BYTEMASKED_STORE(  HVX_VectorPred qunused, HVX_Vector *addr, HVX_Vector v )
{
	*addr = v;
}
#endif
//-----------------------------------------
// NOTE: this is effectively a template
// with 'is_111D' being a constant when it's expanded;
// we expand it for both cases.
//
static inline void __attribute__((always_inline))
core_add_d32_hvx_inline( struct core_add_d32_runstate const * rstp, struct core_add_d32_thrinfo * thrp, int is_111D )
{
	int height = rstp->height;
	int width = rstp->width;
	uint8_t const * ptrA = thrp->ptrA;
	int32_t row_stride_A = rstp->tinA.height_stride;
	uint8_t const * ptrB = thrp->ptrB;
	int32_t row_stride_B = rstp->tinB.height_stride;
	uint8_t* pout = thrp->pout;
	int32_t row_stride_out = rstp->tout.height_stride;

	int dlo = thrp->d_lo;
	int dhi = thrp->d_hi;
	HVX_VectorPred qmask1 = hvx_make_d32_range_mask( dlo, dhi );


	int iht,iwd;
	int scales_hi = rstp->osp.scales_hi;
	int scales_lo = rstp->osp.scales_lo;
	int a_scale = rstp->osp.a_scale;
	int b_scale = rstp->osp.b_scale;
	int offset = rstp->osp.offset;
	scales_hi = Q6_R_combine_RlRl( scales_hi, scales_hi);
	scales_lo = Q6_R_combine_RlRl( scales_lo, scales_lo);
	a_scale = Q6_R_combine_RlRl( a_scale, a_scale);
	b_scale = Q6_R_combine_RlRl( b_scale, b_scale);
	offset = Q6_R_combine_RlRl( offset, offset );

	int final_rsh = rstp->osp.final_rsh;

	//
	// find # of vector loops
	//
	int wpad_bytes = (size_t)ptrA & 0x60;	// 0,32,64 or 96
	int wlen = width*32 + wpad_bytes;		// determines right mask
	int nvecw = (unsigned)(wlen-1)/128;		// # of vecs, -1.


	HVX_Vector const *vinpA_0 = (HVX_Vector const *)( ptrA - wpad_bytes);
	HVX_Vector const *vinpB_0 = (HVX_Vector const *)( ptrB - wpad_bytes);
	HVX_Vector *voutp_0 = (HVX_Vector  *)( pout - wpad_bytes);

	struct prod_vin_with_scale vinBs;
	if ( is_111D){	// get the (only) B side value
		vinBs = find_vinB_product( *vinpB_0, b_scale);
	}

	if( rstp->find_range == 0 ){		// fast version (no min/max calc needed)

		for(iht = 0; iht < height; iht ++){
			HVX_Vector const *vinpA = vinpA_0;
			HVX_Vector const *vinpB = vinpB_0;
			HVX_Vector *voutp = voutp_0;
			//
			// start up..
			//
			HVX_VectorPred qmask = Q6_Q_and_QQn( qmask1, Q6_Q_vsetq_R(wpad_bytes));
			HVX_Vector vinA = *vinpA++;
			HVX_VectorPair vtmp = is_111D? first_stage_111D( vinA, vinBs, a_scale )
										: first_stage( vinA, *vinpB++, scales_hi, scales_lo );

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				HVX_Vector vout = second_stage( vtmp, offset,  final_rsh);
				HVX_Vector vinA = *vinpA++;
				vtmp = is_111D? first_stage_111D( vinA, vinBs, a_scale )
											: first_stage( vinA, *vinpB++, scales_hi, scales_lo );
				BYTEMASKED_STORE( qmask, voutp++, vout);
				qmask= qmask1;
			}
			//
			// last one
			//
			{
				qmask = Q6_Q_and_QQ( qmask, q6op_Q_vsetq2_R(wlen));
				HVX_Vector vout = second_stage( vtmp, offset, final_rsh);
				BYTEMASKED_STORE( qmask, voutp, vout);
			}

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
		//   - keep track of min & max of all vtmp results. min/max are initialized to 'intermed_zero'.
		//     which corresponds to zero 'application' result.
		//     actually we have min and max for odd & slots.
		//   - for 'width' masking: a condition mask is used to force the tmp values to
		//     intermed_zero in the width padding lanes, prior to using them for the min/max calc.
		//   - depth masking is done after, during the horizontal reduction; we first reduce across
		//     the 4 units - so we then have 32 x { ~min, max }, each being from a single depth slot;
		//     and then the depth slots corresponding to padding lanes are forced to {~intermed_zero,intermed_zero}
		//     before the reduction proceeds.
		//
		HVX_Vector vCenter = q6op_Vh_vsplat_R(rstp->osp.intermed_zero);
		HVX_Vector vmin0 = vCenter;
		HVX_Vector vmax0 = vCenter;
		HVX_Vector vmin1 = vCenter;
		HVX_Vector vmax1 = vCenter;

		for(iht = 0; iht < height; iht ++){
			HVX_Vector const *vinpA = vinpA_0;
			HVX_Vector const *vinpB = vinpB_0;
			HVX_Vector *voutp = voutp_0;

			//
			// start up..
			//
			HVX_VectorPred qleft =  Q6_Q_vsetq_R(wpad_bytes);
			HVX_VectorPred qmask = Q6_Q_and_QQn( qmask1, qleft);
			HVX_Vector vinA = *vinpA++;
			HVX_VectorPair vtmp = is_111D? first_stage_111D( vinA, vinBs, a_scale )
										: first_stage( vinA, *vinpB++, scales_hi, scales_lo );
			HVX_Vector vt0  = Q6_V_vmux_QVV( qleft, vCenter, Q6_V_lo_W( vtmp ));
			HVX_Vector vt1  = Q6_V_vmux_QVV( qleft, vCenter, Q6_V_hi_W( vtmp ));

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				vmin0 = Q6_Vh_vmin_VhVh( vmin0, vt0 );
				vmax0 = Q6_Vh_vmax_VhVh( vmax0, vt0 );
				vmin1 = Q6_Vh_vmin_VhVh( vmin1, vt1 );
				vmax1 = Q6_Vh_vmax_VhVh( vmax1, vt1 );

				HVX_Vector vout = second_stage( Q6_W_vcombine_VV(vt1,vt0), offset, final_rsh);
				HVX_Vector vinA = *vinpA++;
				vtmp = is_111D? first_stage_111D( vinA, vinBs, a_scale )
											: first_stage( vinA, *vinpB++, scales_hi, scales_lo );
				BYTEMASKED_STORE( qmask, voutp++, vout);
				qmask= qmask1;
				vt0 = Q6_V_lo_W(vtmp);
				vt1 = Q6_V_hi_W(vtmp);
			}

			// This creates a fake dependency between vt0 and mem here, to suppress
			// a loop optimization that causes ICE in 8.0. It's not needed for 8.1
			HVX_8_0_FAKEDEP_VM( vt0, *voutp);
			// now, before last min/max op, trim any trailing edge
			// (when nvecw=0, vt0,vt1 have already had left masking applied)
			//
			{
				HVX_VectorPred qnright = q6op_Q_vsetq2_R(wlen);
				vt0  = Q6_V_vmux_QVV( qnright, vt0, vCenter);
				vt1  = Q6_V_vmux_QVV( qnright, vt1, vCenter);

				vmin0 = Q6_Vh_vmin_VhVh( vmin0, vt0 );
				vmax0 = Q6_Vh_vmax_VhVh( vmax0, vt0 );
				vmin1 = Q6_Vh_vmin_VhVh( vmin1, vt1 );
				vmax1 = Q6_Vh_vmax_VhVh( vmax1, vt1 );

				qmask = Q6_Q_and_QQ( qmask, qnright);
				HVX_Vector vout = second_stage(  vtmp, offset, final_rsh);
				BYTEMASKED_STORE( qmask, voutp, vout);
			}
			vinpA_0 = (HVX_Vector const *)(  (char const *)vinpA_0 + row_stride_A);
			vinpB_0 = (HVX_Vector const *)(  (char const *)vinpB_0 + row_stride_B);
			voutp_0 = (HVX_Vector  *)(  (char *)voutp_0 + row_stride_out);
		} // end height loop


		// now, each of min and max has 128 values
		// reduce, ignoring values outside the depth range
		//
		//
		HVX_Vector vminmax = hvx_reduce_minmax_h_depthrange( vmin0, vmax0, vmin1, vmax1, dlo, dhi);
		// now all 32  4-byte lanes in vminmax have the same (~min,max).
		// Acc on top of the value there.
		//
		HVX_Vector * vminmaxp = (HVX_Vector *)thrp->minmax;
		*vminmaxp = Q6_Vh_vmax_VhVh( *vminmaxp, vminmax);
	}
}

// expand that inline template with the two cases...
static void
core_add_d32_hvx( struct core_add_d32_runstate const * rstp, struct core_add_d32_thrinfo * thrp )
{
	core_add_d32_hvx_inline( rstp, thrp, 0);
}
static void
core_add_d32_hvx_111D( struct core_add_d32_runstate const * rstp, struct core_add_d32_thrinfo * thrp )
{
	core_add_d32_hvx_inline( rstp, thrp, 1);
}


static int addsub_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	int k;
	char const * nname = hexagon_nn_op_names[self->node_type];

	logmsg(nn,2,"Checking %d node %p",nname,self);

	k = node_check_inputs_range( self,nn, nname, 6, -8);	// 6, + 2 optional
	if( k==0) k = node_check_outputs_n( self,nn, nname, 3);
	if( k!= 0) return k;

	////////////////
	struct addsub_d32_info *info;

	if ((info = nn_calloc(1,sizeof(struct addsub_d32_info))) == NULL) {
		return errlog(nn,"calloc");
	}
	self->opaque = (void*) info;

	info->out_max =0.5f;

	int nin = self->n_inputs;
	const struct tensor *out_min_tensor =  (nin<7)? NULL: self->inputs[6];
	const struct tensor *out_max_tensor = (nin<8)? NULL : self->inputs[7];

	int min_max_precalc = 0;
	// if both min,max are speciified, we must leave  (min,max) as a proper range.
	// (this is only an issue when min < 0 )
	//
	if(out_max_tensor != NULL ){
		float val = tensor_get_float(out_max_tensor,0);
		if( val < INFINITY){
			min_max_precalc |= 2;
			info->out_max = info->out_max_specified = fmaxf(0.0f,val);
		}
	}

	if(out_min_tensor != NULL ){
		float val = tensor_get_float(out_min_tensor,0);
		if( val > -INFINITY){
			min_max_precalc |= 1;
			val  = fminf(0.0f,val);
			info->out_min = info->out_min_specified = val;
			if( min_max_precalc == 3){	// both ends were set
				info->out_max = fmaxf(val + 1e-6f, info->out_max);
				adjust_minmax_for_zero_with_constraints( &info->out_min, & info->out_max,min_max_precalc);
			}
		}
	}
	info->min_max_precalc = min_max_precalc;

	logmsg(nn,2,"@ %p:  min_preset = %d (%f); max_preset = %d (%f)", self,
			min_max_precalc&1, info->out_min,
			(min_max_precalc>>1)&1, info->out_max );

	logmsg(nn,2,"%s node %p check OK",nname,self);
	return 0;
}


static int addsub_d32_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if (self->opaque) nn_free(self->opaque);
	self->opaque = NULL;
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8_d32_ref = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};


struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8_d32_ref = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

