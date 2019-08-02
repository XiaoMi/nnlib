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
/*
 *
 *
 * This contains implementations for add_d32  (QuantizedAdd_8p8to8_d32)
 *
 * Elementwise add of two tensors.
 * Supports any 'broadcast' situation, except that you can't broadcast A->B
 * in some dimensions and B->A in others.
 *
 * Also in here:
 *   QuantizedSub_8p8to8_d32
 *   QuantizedMul_8x8to8_d32 (doesn't have optional inputs 6,7),
 *   QuantizedNeg_8_d32
 *
 */

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

//#define TEST_PERFORMANCE

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include "hvx_inlines.h"

#ifdef HEXAGON_V66
#define OP_ADD_D32_MAX_THREADS 4
#else
#define OP_ADD_D32_MAX_THREADS 2
#endif
//
// operator to execute.
// the 'subtract_rev' is for use when the op is X-Y but X is broadcast to Y;
// in that case we reverse the operands, and change the op to op_subtract_rev.
enum addsub_op {
	op_add,
	op_subtract,
	op_subtract_rev,
	op_mul,

	op_LASTADDSUB = op_subtract_rev	// <= op_LASTADDSUB tests for add or subtract
};


struct elementwise_hvx_thrinfo;
struct elementwise_d32_info;

typedef void  (*core_oper_fp)( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );

typedef int (*setup_scaling_fp)( struct nn_graph *nn, struct nn_node *self);
typedef int (*check_need_rerun_fp)(struct nn_graph *nn, struct elementwise_d32_info *info, int nthreads);

// NOTE: in this code, inputs 'x' and 'y' refer to the first and second inputs;
// 'a' and 'b' are the two inputs, but 'b' is always the smaller-sized input
// in broadcast situations.
//

// scaling parms for add/sub
// note that final_rsh is normally 0..7;
// when 'underrange' !=0, we have a special case
// that needs a different backend; in this case final_rsh can be as small as -6,
// and 'offset' may not fit in i16.
//
struct addsub_scaling_parms {
	int16_t a_scale;
	int16_t b_scale;
	uint16_t scales_hi;	// repacked for the hvx code.
	uint16_t scales_lo;
	int32_t offset;		// this is normally i16, but can be larger when final_rsh <=0
	uint8_t underrange;
	int16_t final_rsh;		// 0..7; or -6..0 when 'underrange'
	int32_t intermed_zero;	// the step (2) value corresponding to a 'zero'.
	int32_t intermed_min, intermed_max;	// intermed values in this range won't clip at the output.
	float netscale;			// used to transform 'intermediate' result to application result

};
// scaling parms for special case of scalar-to-tensor broadcast.
struct bc1111_scaling_parms {
	// these are used for broadcast from (1,1,1,1); nothing else is used.
	// special cases which can be done as copy, 1's complement, or constant-fill
	// are detected from these.
	int16_t scale, offs;

};

// scaling parms for mult;
// those with [*] are a dependent on the output range;
// the others only on the input range.
// The 'ab_zero' from info are used too.
//
struct mul_scaling_parms {
	int16_t range_bias;		// subtract from product (mod 2^16)
	int16_t post_scale;		// [*] then mul by this with >>15
	int16_t adj_bias;		// [*] add to that (with sat)
	int16_t post_rsh;		// [*]then >> by this with rounding
	float all_in_scale;		// product of input scales
	float all_in_scale_recip;		// recip of all_in_scale
};

// each of these describes a vertical slice in a particular batch:
// contains pointers, # of rows to do, and how much to prefetch
//
struct elementwise_chunk_descriptor {
	uint8_t const * data_A;		// a-side source pointer
	uint8_t const * data_B;		// b-side source pointer
	uint8_t * data_out;			// output pointer
	unsigned rows;				// # of of rows
	union {
		uint32_t pf_b_a_ht;
		struct{
			// # of 'rows' to prefetch for A and B. The pfb_ht can be 0, meaning don't prefetch.
			uint16_t pfa_ht;			// a-side prefetch height (width= pf_wid_a, stride= tinA.d32_stride)
			uint16_t pfb_ht;			// b_side prefetch height (wid = pf_wid_b, stride = tinB.d32_stride)
		};
	};
};

//
// This contains the overall 'plan' and includes the scaling, decisions of what special
// case to use, and addressing of the inputs and outputs.
// This needs to be totally rebuilt on first run, or whenever the input shapes or format change;
// If the input ranges (or output range) changes, only the scaling portions need to be
// re-evaluated.
// If we have a 'broadcast from single value' case, this plan contains a pointer to the single
// value, and its previous contents; and if it changes from run to run, the scaling needs to be
// re-done.
//
// Note that the two inputs which specify output range are not allowed to change during runs.
//
struct elementwise_d32_info {
	// these are only for checking if the conditions changed.
	//
	struct shape x_shape;		// previous shape of the 'x' input
	struct tensor_format x_format;
	void * x_data;

	struct shape y_shape;		// previous shape of the 'y' input
	struct tensor_format y_format;
	void * y_data;
	float xy_in_min[2], xy_in_max[2];	// ranges from inputs

	int strategy_valid;
	uint8_t const *single_B_in_p;	// points to single B in val if it's a (1,1,1,1) broadcast; else null
	uint8_t single_B_in;			// previous value.

	struct tensor_addressing tinA;	// 'a' input addressing
	struct tensor_addressing tinB;	// 'b' input addressing
	struct tensor_addressing tout;	// output addressing.
	struct shape out_shape;			// the shape of the output (same as 'A' input)

	int compat_flags;			// shape compatibility (broadcast flags).
	int16_t oper_code;			// the operation code (op_add, op_sub, op_subr)
	int16_t swapXY;				// if true, a is x, b is x;  else a is x,b is y.
	int16_t min_max_precalc;	// bit 0-> min; bit 1-> max; did we get a non-inf input here.
	int16_t has_run_before;		// true if out_min, out_max are based on previous run

	struct elementwise_chunk_descriptor * chunkdescs;	// NULL until allocated
	uint32_t chunkdesc_count;							// number of chunk descriptors
	uint32_t chunkdesc_alloc;							// number we have allocated for.

	// funcs which depend on the operator class
	setup_scaling_fp setup_scaling_funcp;					// set up scaling
	check_need_rerun_fp check_need_rerun_funcp;				// check if we need rerun

	// broadcast mode.
	// Broadcasting in B and H is easy since we can just set the strides to
	// 0 on the B side.
	// There are four strategies in all, with separate code:
	//  (1) broadcast scalar to all is handled specially - but only if out min/max are both unconstrained.
	//      Otherwise it is done as case (2)
	//  (2) if broadcasting from shape (x,1,1,D) or (x,1,1,1) - and this includes from (1,1,1,1) when out
	//      min and/or max are constrained, we use the broadcast_B11D strategy. In this mode a single
	//      b-side d vector value can be used over an entire [h,w,dlo:dhi] range (where dlo:dhi) is a d32 slice).
	//  (3) in all cases broadcasting from W and/or D but *not* from H, or from D but not from W, we
	//      have a special case broadcast_mixed which can handle these.
	//  (4) if no broadcast, or broadcasting only from B and/or H, we use the 'broadcast_general' mode, which only needs to
	//      to set B-side batch_stride or height_stride = 0 as applicable
	// When 'A' and 'B' both have a dim = 1, it can be considered broadcast or not, and we choose whatever's most efficient
	//   (prefer 'b11d' to 'general', and 'general' to 'mixed'.


	enum { mode_broadcast_general,		// general case (include broadcast in B and/or H without W or D)
		mode_broadcast_B11D, 	// broadcast (*,1,1,D) to all, or (B,1,1,1) (or (1,1,1,1) if output constrained).
		mode_broadcast_mixed, 	// all other cases
		mode_broadcast_1111 	// broadcast one to all (but only if out range is not constrained).
	} broadcast_mode;
	// coverage (based on which of bhwd broadcast):
	// 0000	general		0100 general	1000 general	1100 general
	// 0001	mixed		0101 mixed		1001 mixed		1101 mixed
	// 0010	mixed		0110 B11D		1010 mixed		1110 B11D
	// 0011	mixed		0111 B11D		1011 mixed		1111 1111 (or B11D)

	// The  core function for the specific mode
	//  (except _1111)
	//
	core_oper_fp core_oper_funcp;

	// for mode_broadcast_mixed,mode_broadcast_B11D, this is used to generate a vector for the vdelta control needed to do
	// the splatting in w and/or d dims: splat it to all 32-bit lanes then 'and' with const_Count128.
	//
	uint32_t bcast_wd_deltagen;

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

	// scaling
	float ab_stepsize[2];		// 'step size' for A and B inputs
	int16_t ab_zero[2];			// 'zero code' for A and B inputs.
	union {
		struct addsub_scaling_parms addsub_scaling;
		struct mul_scaling_parms mul_scaling;
		struct bc1111_scaling_parms bc1111_scaling;
	};
	// prefetch
	uint16_t pf_wid_a;			// width to prefetch from each 'a' side. multiple of 128, < 64K.
	uint16_t pf_wid_b;			// width to prefetch from each 'b' side.

	//
	// work slicing.
	// mode_broadcast_1111 is a special case. For the rest,
	// we divoide the work into batches * height_chunks work units,
	// which are each 'chunk_rows' rows (or maybe less if it is the last
	// one in the batch). Within each one, the d32 loop is outside the
	// row loop, and covers the whole depth.
	// But - we try to 'flatten' the d32 dimension into height when possible.
	// e.g if both inputs have height = 8 and md32 = 6, it is
	// flattened to effective_height=48 and effective_nd32 = 1,which allows
	// better slicing. This is not possible under some circumstances (in particular
	// we can't do it unless the output depth is a multiple of 32, *or* the
	// output range is fixed; since we can't track the min/max properly in the
	// 'flattened' state otherwise). The 'chunk_rows' is in terms of *flattened*
	// rows, not actual rows.
	//
	int run_height;		// height, or height * nd32 when 'flattened'
	int run_nd32;		// nd32, or 1 when flattened
	int run_depth;		// actual depth, or 32 if flattened.

	int height_chunks;	// the # of chunks each batch
	int chunk_rows;		// rows/batch
	int num_work_units;	// height_chunks * batches.

	uint8_t * minmax_bufs;	// one vector per thread (in scratch)

};
struct elementwise_d32_runstate {
	struct elementwise_d32_info *info;
	volatile int scratch_idx;	// used to pick up scratch buffer(s)
	volatile int jobno;			// sequences of jobs
	int nthreads;				// number of threads in use
	nn_sem_t done_sem;
	int find_range;				// do we need to find range
};
static int addsub_setup_scaling( struct nn_graph *nn, struct nn_node *self);
static int check_addsub_need_rerun( struct nn_graph *nn, struct elementwise_d32_info *info, int nthreads);

static inline int set_addsub_scaling_for_output_range( struct elementwise_d32_info *info, int fixed_range);
static void core_add_d32_general_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_add_d32_B11D_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_add_d32_mixed_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_add_d32_underrange_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );


static int mul_setup_scaling( struct nn_graph *nn, struct nn_node *self);
static int check_mul_need_rerun( struct nn_graph *nn, struct elementwise_d32_info *info, int nthreads);

static void core_mul_d32_general_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_mul_d32_B11D_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_mul_d32_mixed_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );

static int make_chunk_descriptors(  struct nn_graph *nn, struct elementwise_d32_info* info );
// build the 'address' and 'broadcast' portions of the strategy for the current inputs,
// operator is op_add or op_subtract.
//
static int
setup_elementwise_strategy( struct nn_graph *nn, struct nn_node *self, int operator)
{
	struct elementwise_d32_info *info= (struct elementwise_d32_info*)self->opaque;
	struct tensor const *inX_tensor = self->inputs[0];
	struct tensor const *inY_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	info->strategy_valid = 0;
	info->x_shape = inX_tensor->shape;
	info->x_format = inX_tensor->format;
	info->x_data= inX_tensor->data;

	info->y_shape = inY_tensor->shape;
	info->y_format = inY_tensor->format;
	info->y_data= inY_tensor->data;

	if( operator <= op_LASTADDSUB ){
		info->setup_scaling_funcp = addsub_setup_scaling;
		info->check_need_rerun_funcp = check_addsub_need_rerun;
	}else{
		info->setup_scaling_funcp = mul_setup_scaling;
		info->check_need_rerun_funcp = check_mul_need_rerun;
	}

	int compat_flags = check_compatible_elementwise_d32( nn, "tbd", inX_tensor, inY_tensor, compat_ALL| compat_AtoB);
	if( compat_flags < 0)
		return errlog( nn,"unsupported broadcast mode");
	int swapXY = 0;
	// it may be that A is broadcasting onto B
	// in which case we need to swap X,Y to A,B (and arrange
	// for the min/max to be swapped)
	if( compat_flags & compat_AtoB){
		swapXY = 1;
		const struct tensor * t = inX_tensor;
		inX_tensor = inY_tensor;
		inY_tensor = t;
		if( operator == op_subtract) operator = op_subtract_rev;
	}
	info->compat_flags = compat_flags;
	info->out_shape = inX_tensor->shape;
	info->swapXY = swapXY;
	info->oper_code = operator;

	info->tinA = tensor_addressing_d32( inX_tensor);
	info->tinB = tensor_addressing_d32( inY_tensor);
	int nd32 = info->tinA.nd32;
	int depth = info->out_shape.depth;

	//
	// allocate the output tensor
	//  - most padding is copied from input A
	//
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		info->out_shape.batches,
		info->out_shape.height, inX_tensor->format.height_pad[0],inX_tensor->format.height_pad[1],
		info->out_shape.width,  inX_tensor->format.width_pad[0],inX_tensor->format.width_pad[1],
		depth, 0, nd32*32-depth,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,info->out_shape.batches,info->out_shape.height,info->out_shape.width,depth);
	}
	// set up addressing
	info->tout = tensor_addressing_d32( out_tensor);

	// setup out min/max shapes
	{
		struct tensor *out_min_tensor = self->outputs[1];
		struct tensor *out_max_tensor = self->outputs[2];

		struct shape shp1111 = { .batches=1, .height=1, .width=1, .depth=1 };
		if( tensor_out_prepare_normal_fromshape(out_min_tensor, &shp1111, NN_TYPE_FLOAT) != 0
		  ||tensor_out_prepare_normal_fromshape(out_max_tensor, &shp1111, NN_TYPE_FLOAT) != 0 ){
			return errlog(nn,"min or max output too small");
		}
	}

	// decode the compat mode for broadcast.
	int bcast = compat_flags& compat_broadcast_ALL;
	info->single_B_in_p = NULL;
	int bcast_mode;
	if( bcast == compat_broadcast_ALL && info->min_max_precalc == 0 ){
		bcast_mode = mode_broadcast_1111;
		info->single_B_in_p = info->tinB.data;
	}else {
		// if broadcasting from B and/or H, set those strides to 0.
		//
		if(bcast& compat_broadcast_B )
			 info->tinB.batch_stride = 0;
		if( bcast & compat_broadcast_H )
			info->tinB.height_stride = 0;

		// note that 'bcast' will have 1's in for dims which are 1 on both sides.
		// these can be considered broadcast or not, we try to avoid the 'mixed' mode
		// since it's the slowest.
		if((bcast & (compat_broadcast_H|compat_broadcast_W)) == (compat_broadcast_H|compat_broadcast_W)){
			// this is the B11D mode
			bcast_mode = mode_broadcast_B11D;
			info->bcast_wd_deltagen = (bcast & compat_broadcast_D)== 0? 0x60606060	// only in W
					:0x7F7F7F7F;	// splat one to all W and D.
		}else{
			// if dout = 1, we're not really broadcasting on D;
			// likewise for w.
			// Better to map these to general when possible.
			if( depth==1) bcast &= ~compat_broadcast_D;
			if( info->out_shape.width ==1) bcast &=~compat_broadcast_W;
			if( (bcast & (compat_broadcast_W|compat_broadcast_D))!=0 ){
				bcast_mode = mode_broadcast_mixed;
				info->bcast_wd_deltagen =
						(bcast & compat_broadcast_W)== 0? 0x1F1F1F1F	// only in D
						:(bcast & compat_broadcast_D)== 0? 0x60606060	// only in W
						:0x7F7F7F7F;	// splat one to all W and D.
			}else{
				bcast_mode = mode_broadcast_general;
			}
		}
	}
	info->broadcast_mode = bcast_mode;

	static const core_oper_fp addsub_corefuncs[3] = {
			[mode_broadcast_general] = core_add_d32_general_hvx,
			[mode_broadcast_B11D] = core_add_d32_B11D_hvx,
			[mode_broadcast_mixed] = core_add_d32_mixed_hvx };
	static const core_oper_fp mul_corefuncs[3] = {
			[mode_broadcast_general] = core_mul_d32_general_hvx,
			[mode_broadcast_B11D] = core_mul_d32_B11D_hvx,
			[mode_broadcast_mixed] = core_mul_d32_mixed_hvx };

	if( bcast_mode >= mode_broadcast_1111 ){
		info->core_oper_funcp = NULL;
	}else{
		info->core_oper_funcp = ((operator<=op_LASTADDSUB)?addsub_corefuncs:mul_corefuncs)[bcast_mode];
	}

	if( bcast_mode != mode_broadcast_1111){
		{
			// prefetch width
			// expressed as the amount needed in one row (one d32 slice)
			unsigned pf_wid = (  ((size_t)info->tinA.data & 127)		// padding before
								 + info->out_shape.width*32			// actual bytes
								 + 127u) & ~127u;					// round up to mul of 128
			info->pf_wid_a = min_u32( pf_wid, 0xFF80 );			// can't exceed this
			int bwid = (bcast& compat_broadcast_W)?1: info->out_shape.width;
			pf_wid = (  ((size_t)info->tinB.data & 127)		// padding before
					 + bwid*32			// actual bytes
					 + 127u) & ~127u;					// round up to mul of 128
			info->pf_wid_b = min_u32( pf_wid, 0xFF80 );			// can't exceed this
		}



		// work out slicing strategy
		int run_height = info->out_shape.height;
		int run_depth = info->out_shape.depth;
		int run_nd32 = info->tout.nd32;
		// can we flatten the nd32 dim into height?
		// - only makes sense if nd32 > 1
		// - not possible if depth%32!=0 and min_max_precalc !=3
		// always possible if height=1; regardless of depth broadcast or not.
		// Otherwise it's possible if:
		//     -'A' strides are clean
		//     - one of
		//           -B height= B depth = 1 (broadcasting one to all depth/height)
		//  		 -B height =out_height, Bdepth = out_depth, B strides 'clean'
		// (by 'clean' strides; height_stride= nd32*d32_stride).
		if( run_nd32 >1 && ((run_depth&31)==0 || info->min_max_precalc==3)){
			int can_flatten = 0;
			// TODO: backing this off until an issue is solved: (1,1,14,128)+(1,1,1,128) doesn't work
			if( 0 && run_height ==1 ){
				can_flatten = 1;
			}else if( info->tinA.height_stride == info->tinA.d32_stride*run_nd32){
				struct shape const *bshape = swapXY? &info->x_shape : &info->y_shape;
				if( bshape->height ==1 ){
					can_flatten= bshape->depth==1;
				}else{
					if( bshape->height== run_height && bshape->depth==run_depth
							&&  info->tinB.height_stride == info->tinB.d32_stride*run_nd32){
						can_flatten =1;
					}
				}
			}
			// implement the flattening. set all height strides to d32 strides.
			if( can_flatten){
				run_height *= run_nd32;
				run_nd32= 1;
				run_depth = 32;
				info->tout.height_stride = info->tout.d32_stride;
				info->tinA.height_stride = info->tinA.d32_stride;
				if (info->tinB.height_stride != 0) // only if not broadcasting h
					info->tinB.height_stride = info->tinB.d32_stride;
			}
		}
		info->run_height = run_height;
		info->run_depth =  run_depth;
		info->run_nd32 = run_nd32;
		unsigned batches = info->out_shape.batches;
		// how many rows per work unit?
		unsigned rowsize = info->tout.height_stride;
		unsigned chunk_good_size = (32*1024);
		unsigned chunk_rows = (rowsize*2>chunk_good_size)?1
				: (rowsize * run_height <= chunk_good_size)? run_height
				: chunk_good_size/rowsize;
		unsigned height_chunks = (chunk_rows >= run_height)?1
				: ( run_height + (chunk_rows-1))/chunk_rows;
		// if we don't have enough chunks, see if we can make them smaller
	 	// Also we want an even total.
	 	// this is not necessary when batches >= 2*OP_ADD_D32_MAX_THREADS (if we have 7 smallish
	 	// batches, we don't want to split them all in half just to make an even batch count)
	 	//
	 	if( batches < 2*OP_ADD_D32_MAX_THREADS ){
			while( (batches*height_chunks < OP_ADD_D32_MAX_THREADS  ||  ((batches*height_chunks)&1)!=0)  && height_chunks*2 <= run_height ){
				height_chunks++;
			}
		}
		// now we know height_chunks, find an even size for the chunk
		chunk_rows = (run_height+ (height_chunks-1))/height_chunks;

		info->height_chunks = height_chunks;
		info->chunk_rows = chunk_rows;
		info->num_work_units = batches * height_chunks;
		logmsg(nn,2,"%d batches of %d rows each; do each in %d chunks of %d rows * nd32=%d",
				batches, run_height,height_chunks, chunk_rows, run_nd32);

		int res = make_chunk_descriptors( nn, info );
		if( res != 0) return res;
	}
	nn_scratch_reset(nn);
	info->minmax_bufs = nn_scratch_alloc(nn, sizeof(HVX_Vector)*OP_ADD_D32_MAX_THREADS);
	if( info->minmax_bufs == NULL) return errlog(nn,"scratch_alloc");

	info->strategy_valid = -1;	// still needs scaling info.
	return 0;
}

// make the array of chunk descriptors
// each contains input & output pointers, and a row count.
// Using this array, each 'job' can consist of processing one job, while performing
// a prefetch for a later job.
static int
make_chunk_descriptors(  struct nn_graph *nn, struct elementwise_d32_info* info )
{
	int height_chunks = info->height_chunks;
	int batches = info->out_shape.batches;
	int compat_flags = info->compat_flags;


	unsigned alloc_descs = (height_chunks* batches + 15u)&~15u;	// alloc for this many

	struct elementwise_chunk_descriptor *chunkdescs = info->chunkdescs;

	if( info->chunkdesc_alloc < alloc_descs ){	// make it bigger
		void * new_arr = nn_realloc(info->chunkdescs, sizeof( struct elementwise_chunk_descriptor) * alloc_descs);
		if( new_arr == NULL) return errlog(nn, "alloc fail: chunkdescs");
		info->chunkdescs = chunkdescs = (struct elementwise_chunk_descriptor*)new_arr;
		info->chunkdesc_alloc = alloc_descs;
	}
	// fill out the first batch of these

	uint8_t const * inpA0 = info->tinA.data;
	int inA_height_stride = info->tinA.height_stride;

	uint8_t const * inpB0 = info->tinB.data;
	int inB_height_stride = info->tinB.height_stride;

	uint8_t * outp0 = info->tout.data;
	int out_height_stride = info->tout.height_stride;


	int chunk_rows = info->chunk_rows;
	int run_height = info->run_height;

	//printf("%d rows in %d chunks of %d; %d batches (allocated %d)\n", run_height, height_chunks, chunk_rows, batches, alloc_descs );

	// some sanity checks
	if( batches < 1 || chunk_rows < 1 || height_chunks < 1 || run_height > height_chunks*chunk_rows
			|| run_height <= height_chunks*(chunk_rows-1) ){
		return errlog(nn,"bad chunk divs: b=%d, h=%d as %d of %d", batches, run_height, height_chunks, chunk_rows);
	}
	// instead of doing 'chunk_rows' at a time all the way down, change from chunk_rows to chunk_rows-1
	// when it becomes adequate to do so.
	// e.g. 480 sliced in 70 of 7, would be 60 of 7, followed by 10 of 6.
	int curr_chunkrows = chunk_rows;
	int smaller_chunkrows = chunk_rows-1;
	int hpos = 0;
	int hposx = run_height-height_chunks*(chunk_rows-1); // used to determine when to switch
	int nd32 = info->run_nd32;
	int b_nd32 = nd32;
	int b_pfadd = 0;
	if( compat_flags & compat_broadcast_H){		// broadcasting from height
		b_nd32 = 0;							// no running prefetch of B.
	}else if( compat_flags & compat_broadcast_D){
		b_nd32 = 0;							// only one 'depth' needs prefetching on B,
		b_pfadd = 1;
	}
	for( int i =0; i < height_chunks; i++){
		chunkdescs[i].data_A = inpA0 + inA_height_stride*hpos;
		chunkdescs[i].data_B = inpB0 + inB_height_stride*hpos;
		chunkdescs[i].data_out = outp0 + out_height_stride*hpos;
		chunkdescs[i].rows = curr_chunkrows;
		chunkdescs[i].pfa_ht = curr_chunkrows * nd32;	// # of 'rows' to prefetch, a
		chunkdescs[i].pfb_ht = curr_chunkrows * b_nd32 +b_pfadd;	// # of 'rows' to prefetch, b
		hpos += curr_chunkrows;
		hposx += smaller_chunkrows;
		if( hposx == hpos) curr_chunkrows = smaller_chunkrows;
	}
	if( hpos != run_height) return errlog(nn, "h division failed!");

	// if height_broadcasting, we filled in pfb_ht as zero on all entries; go fill in the first
	// one properly now. It's one d-slice (if broadcasting on D as well) or nd32 slices if not.
	if(compat_flags & compat_broadcast_H){
		int pf_b_ht  = (compat_flags & compat_broadcast_D) ? 1: nd32;
		chunkdescs[0].pfb_ht = pf_b_ht;
	}

	info->chunkdesc_count = height_chunks * batches;
	// do the batches by just adding the offsets to the previous...
	if( batches > 1){
		int inA_batch_stride = info->tinA.batch_stride;
		int inB_batch_stride = info->tinB.batch_stride;
		int out_batch_stride = info->tout.batch_stride;
		struct elementwise_chunk_descriptor const *rdp = &chunkdescs[0];
		struct elementwise_chunk_descriptor *wrp = &chunkdescs[height_chunks];
		for(int  i = 0; i < (batches-1)*height_chunks; i++ ){
			unsigned rows = rdp->rows;
			uint32_t pf = rdp->pf_b_a_ht;
			wrp->data_A = rdp->data_A + inA_batch_stride;
			wrp->data_B = rdp->data_B + inB_batch_stride;
			wrp->data_out = rdp->data_out + out_batch_stride;
			wrp->rows = rows;
			wrp->pf_b_a_ht = pf;
			rdp ++;
			wrp ++;
		}
	}
	/*
	for( int i =0; i < info->chunkdesc_count; i++){
		printf(" %2d: %2d  %3d %p %p %p\n",
				i/height_chunks, i% height_chunks,
				chunkdescs[i].rows,chunkdescs[i].data_A,chunkdescs[i].data_B,chunkdescs[i].data_out);
	}*/

	return 0;
}


///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////.///// For Add/Sub //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

//
// setup the scaling for add32 or subtract32
// This needs to be redone if
//   - strategy is rebuilt entirely;
//   - any of the input ranges change
//   - we are in mode_broadcast_1111 and *info->single_B_ptr has changed
//      (no longer info->single_B_in).
//
// For broadcast_1111, the operation involves adding
//  a scalar to a tensor (or subtracting). This can be simplified; in many cases
// this can be done by copying the u8 data and changing the endpoints. However, given
// that we have a constraint the that 'zero code' is an integer in range 0..255, there
// is some trickiness. A method equivalent to the below is used:
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
// In cases where either the out_min or out_max is specified, we handle the (1111) broadcast case as the
// general B11D broadcast case (since it may be that we need to use a gain > 1 in those cases).
//
static int
addsub_setup_scaling( struct nn_graph *nn, struct nn_node *self)
{
	struct elementwise_d32_info *info= (struct elementwise_d32_info*)self->opaque;
	// reload these...
	float a_in_min, a_in_max, b_in_min, b_in_max;

	info->xy_in_min[0] = a_in_min = tensor_get_float( self->inputs[2],0);
	info->xy_in_max[0] = a_in_max = tensor_get_float( self->inputs[3],0);
	info->xy_in_min[1] = b_in_min = tensor_get_float( self->inputs[4],0);
	info->xy_in_max[1] = b_in_max = tensor_get_float( self->inputs[5],0);
	if( info->swapXY){
		float t = a_in_min; a_in_min = b_in_min; b_in_min = t;
		t = a_in_max; a_in_max = b_in_max; b_in_max = t;
	}
	info->ab_zero[0] = get_qu8_level_size_zero( a_in_min, a_in_max, &info->ab_stepsize[0] );
	info->ab_zero[1] = get_qu8_level_size_zero( b_in_min, b_in_max, &info->ab_stepsize[1] );

	int operator = info->oper_code;

	if( info->broadcast_mode == mode_broadcast_1111){
		// just add or subract a single value. Sometimes we can
		// just move the endpoints; sometimes we need to 'squish'
		// the codes (e.g if input range is -0.5 .. 1.0 and we are adding
		// 0.8, we will need output ramge of 0..1.8 and codes will need scaling.
		float astep = info->ab_stepsize[0];
		int b_value_u8 = *info->single_B_in_p;
		info->single_B_in = b_value_u8;		// so we know if it changes.
		// find the actual b value
		float b_val = (b_value_u8 - info->ab_zero[1])* info->ab_stepsize[1];
		// subtract?  (for reverse-subtract, we correct later).
		if( operator != op_add) b_val = -b_val;
		float new_zero_code = info->ab_zero[0] - b_val / astep;
		int code_scale = 32768;		// will be saturated to i16 later
		int code_offs = 0;
		float out_min, out_max;
		float arange = a_in_max-a_in_min;

		if( new_zero_code >= -0.25f && new_zero_code <= 255.25f ){	// only shift the endpoints
			// round new_zero_code to form the new zero.
			out_min = -roundf_i32( new_zero_code)*astep;
			out_max = out_min + arange;
		}else {
			if( b_val > 0){		// pin min = 0, extend max
				out_min = 0.0f;
				out_max = a_in_max + b_val;
			}else{			// pin max = 0, extend min
				out_min = a_in_min + b_val;
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
		if( code_scale <=128 && code_scale >= -128){
			// maybe it's a singularity; make this clear to the 'back end'.
			int from0 = (code_offs+64)>>7;	// 0 maps to this
			int from255 = (code_scale*255 + code_offs*256 + 16384 )>>15;
			if( from0 ==from255 ){
				code_scale = 0;
			}
		}
		//printf("scaling for 1111 done: +%.5f, scale = %d, offs = %d; out range %f ... %f\n",b_val, code_scale, code_offs, out_min, out_max);
		info->bc1111_scaling.scale = saturate_i16(code_scale);
		info->bc1111_scaling.offs = code_offs;
		info->strategy_valid = 1;
		info->out_min = out_min;
		info->out_max = out_max;
		return 0;
	}
	// Scaling for the general case.
	//
	// if either limit is not predetermined, check the worst-case
	// range vs the range we have;
	// if the worst case limit exceeds 16x the current range,
	// extend the current range. This keeps the range measurement
	// from saturating.
	//

	if( info->min_max_precalc < 3 ){
		// given {a,b}_in_{min,max}, and assuming
		//   in_min <= 0 ,  in_max >=0 , in_max > in_min:
		// - largest and smallest output values are given by the below
		float out_max_all;		// should be >= 0
		float out_min_all;		// should be <=0
		switch( operator){
		 default:
		 case op_add:
			out_max_all = a_in_max + b_in_max;
			out_min_all = a_in_min + b_in_min;
			break;
		 case op_subtract:
			out_max_all = a_in_max - b_in_min;
			out_min_all = a_in_min - b_in_max;
			break;
		 case op_subtract_rev:
			out_max_all = b_in_max - a_in_min;
			out_min_all = b_in_min - a_in_max;
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
			info->has_run_before= 1;
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
	int res =set_addsub_scaling_for_output_range( info, info->min_max_precalc ==3);
	if( res != 0) return errlog(nn,"scaling failed");
	info->strategy_valid = 1;
	return 0;
}


// The add/sub operation is done as:
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
// Normally 'final_rsh' must be in range 0..7;
// we allow it to be as small as -5 when 'fixed_range' is true (meaning the output
// range is fixed). This will need a special back-end to handle; the special back-end
// finds initial SOP in 32 bits, >>(8-k), then add offset and >>(final_rsh+k), where k  = 1-final_rsh.
// So, 'offset' needs to be larger by a factor of 2^k, and may not fit in 16 bits.
// For this mode, 'undderange' is set.
//
// The case where final_rsh=0 is handled in two different ways:
//   if fixed_range = 0, it is handled as the normal case, but when fixed_range !=0,
//   it is treated as 'underrange' case (and the 'offset' will be 2x larger as a result).
// This allows us to get a slightly better result in the 'underrange' case, using a proper
// round operation at the end.
//

static inline int
set_addsub_scaling_for_output_range( struct elementwise_d32_info *info, int fixed_range )
{
	struct addsub_scaling_parms * osp = &info->addsub_scaling;
	float out_min = info->out_min;
	float out_max = info->out_max;
	int operator = info->oper_code;

	float out_scl = 255.0f/(out_max-out_min);		// out scale
	float out_z = -out_min * out_scl;				// the 'zero point'

	//printf("output range = %.6f..%.6f; z = %.6f\n", out_min, out_max, out_z);

	float a_scale_f = out_scl * info->ab_stepsize[0];
	float b_scale_f = out_scl * info->ab_stepsize[1];
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
	if( scexp > 6 && ( !fixed_range || scexp>12) )		// req. scale is too large
		return -1;
	int rsh = min_i32(6-scexp,7);  // should be 0..6 (maybe 7 sometimes); or -6..0 when underrange

	// important: rsh=0 is 'underrange' when fixed_range, and not otherwise (this lets us get
	// a rounding up in, using an actual final_rsh of 1, when fixed_range)
	int preshift = 0;
	if (fixed_range && rsh <=0){
		preshift = 1-rsh;
	}
	int is_underrange = preshift != 0;
	osp->underrange = is_underrange;

	// determine the quantized scale factors
	float rsh_scl = flt_ldexp(1.0f,rsh);
	int a_scale = roundf_i32(a_scale_f  * (rsh_scl * 256.0f));
	int b_scale = roundf_i32(b_scale_f  * (rsh_scl * 256.0f));
	// work out the zero point
	//
	float rsh_scl2 = is_underrange? 2.0f : rsh_scl;	// rsh_scl *2^preshift

	int intzero = (info->ab_zero[0]*a_scale + info->ab_zero[1] * b_scale) >> (8-preshift);	// step 2 result for 'zero' input
	int offset = roundf_i32( out_z * rsh_scl2) - intzero;

	/*printf("scale = %d, %d; offset = %d; final_rsh = %d; int_zero = %d\n",
	a_scale, b_scale, offset, rsh, intzero );*/

	osp->final_rsh = rsh;
	osp->a_scale = a_scale;
	osp->b_scale = b_scale;
	osp->intermed_zero = intzero;
	osp->offset = is_underrange? offset: saturate_i16(offset);
	osp->netscale = 1.0f / (out_scl * rsh_scl);

	// special packing for hvx code
	// scales_lo has the 7 lsbs of each scale (zero extended)
	// scales_hi has bits [14:7] of each scale
	osp->scales_lo = (a_scale & 0x7f) | ( (b_scale & 0x7f) << 8);
	osp->scales_hi = ( (a_scale >>7) & 0xFF) | ( (b_scale <<1) & 0xFF00);
	// --> underrange is only supported when output range is fixed; so the below params
	// --> are not meaningful in underrange mode.
	//
	// find intermed_min, intermed_max ; range of values at (2) which won't
	// clip  at the output.
	// note that these may fall outside the range (0..32767) of (2) results,
	// which just means that clipping can't occur at that endpoint in the particular setup.
	// So it's important to store these as i32, they may not fit in i16.
	//
	int rbias = is_underrange?0: ((1<<rsh)>>1);
	// result from (3) must be at least -rbias to not clip as < 0
	osp->intermed_min = -(offset+rbias);
	// result from (3) must be at most (256<<rsh)-(rbias+1) to not clip as > 255
	osp->intermed_max = osp->intermed_min + (1<<(rsh+8))-1;
	return 0;
}
// this converts a value at step (2) of the process to a float output value.
//
static inline float __attribute__((unused))
convert_intermed_to_outval( struct addsub_scaling_parms const * osp, int val )
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
check_intermed_range( struct addsub_scaling_parms const * osp, int minval, int maxval )
{
	return ((minval < osp->intermed_min)? 1: 0) +  ((maxval > osp->intermed_max)? 2: 0);
}
static int
check_addsub_need_rerun( struct nn_graph *nn, struct elementwise_d32_info *info, int nthreads)
{
	// collect the min and max ...
	int16_t const * mmp = (int16_t const *)info->minmax_bufs;
	int minvali = mmp[0];
	int maxvali = mmp[1];
	for( int i = 1; i < nthreads; i++){
		minvali = max_i32(minvali, mmp[64*i]);	// max, since min is encoded as ~min
		maxvali = max_i32(maxvali, mmp[64*i+1]);
	}
	minvali = -1-minvali;			// now is proper min

	float actual_min = convert_intermed_to_outval( &info->addsub_scaling, minvali);
	float actual_max = convert_intermed_to_outval( &info->addsub_scaling, maxvali);
	float out_range = actual_max - actual_min;

	logmsg(nn,3,"new range found is: %d.. %d  [%f ... %f]\n", minvali, maxvali, actual_min, actual_max + out_range*0);

	// pad the range out a bit
	out_range = out_range + fmaxf(0.15f*out_range, 1e-4);
	float padded_min = fmin(0.0f, actual_max-out_range);
	float padded_max = fmax(0.0f, actual_min+out_range);
	int any_adj = 0;
	float out_min = info->out_min;
	float out_max = info->out_max;

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
		return 0;	// all done, if no adjustment needed

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
	}
	logmsg(nn,3,"rerange to %f  ... %f\n", info->out_min, info->out_max);
	int res = set_addsub_scaling_for_output_range(info, 0);
	if( res <0) return -1;
	return 1;
}
///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// For Multiply //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
static inline int set_mul_scaling_for_output_range( struct elementwise_d32_info *info);

static int
mul_setup_scaling( struct nn_graph *nn, struct nn_node *self)
{
	struct elementwise_d32_info *info= (struct elementwise_d32_info*)self->opaque;
	// reload these...
	float a_in_min, a_in_max, b_in_min, b_in_max;

	info->xy_in_min[0] = a_in_min = tensor_get_float( self->inputs[2],0);
	info->xy_in_max[0] = a_in_max = tensor_get_float( self->inputs[3],0);
	info->xy_in_min[1] = b_in_min = tensor_get_float( self->inputs[4],0);
	info->xy_in_max[1] = b_in_max = tensor_get_float( self->inputs[5],0);
	if( info->swapXY){
		float t = a_in_min; a_in_min = b_in_min; b_in_min = t;
		t = a_in_max; a_in_max = b_in_max; b_in_max = t;
	}
	info->ab_zero[0] = get_qu8_level_size_zero( a_in_min, a_in_max, &info->ab_stepsize[0] );
	info->ab_zero[1] = get_qu8_level_size_zero( b_in_min, b_in_max, &info->ab_stepsize[1] );


	if( info->broadcast_mode == mode_broadcast_1111){
		// just multiply by single value; this is done by copying
		// the data (and 1's complement, if the factor < 0).

		int b_value_u8 = *info->single_B_in_p;
		info->single_B_in = b_value_u8;		// so we know if it changes.
		// find the actual b value
		float b_val = (b_value_u8 - info->ab_zero[1])* info->ab_stepsize[1];
		int code_scale = 32767;
		int code_offs = 0;

		float out_min = a_in_min * b_val;
		float out_max = a_in_max * b_val;

		if( b_val <= 0.0f ){
			if( b_val == 0.0f){	// make it all 0
				code_scale= 0;
				out_max = 0.5f;
			}else{
				// mul by -ve # ? reverse the endpoints
				float t = out_min;
				out_min = out_max;
				out_max = t;
				// and arrange for 1's complement of the data.
				code_scale = -32768;
				code_offs = 255*128;
			}
		}

		//printf("scaling for 1111 done: +%.5f, scale = %d, offs = %d; out range %f ... %f\n",b_val, code_scale, code_offs, out_min, out_max);
		info->bc1111_scaling.scale = code_scale;
		info->bc1111_scaling.offs = code_offs;
		info->strategy_valid = 1;
		info->out_min = out_min;
		info->out_max = out_max;
		return 0;
	}
	// Scaling for the general case.
	//
	// if either limit is not predetermined, check the worst-case
	// range vs the range we have;
	// if the worst case limit exceeds 16x the current range,
	// extend the current range. This keeps the range measurement
	// from saturating.
	//
	// do all the scaling calculations which depend on the input ranges.
	//

	float all_in_scale = (255.0f*255.0f)/((a_in_max-a_in_min)*(b_in_max-b_in_min));
	info->mul_scaling.all_in_scale = all_in_scale;
	info->mul_scaling.all_in_scale_recip = info->ab_stepsize[0]*info->ab_stepsize[1];

	// here is a formula for range_bias
	// which always gives a result which is zero in the 8 lsbs, in range -32512 .. 32512
	// e.g. if both are zero, it evaluates to 32512
	// the range of products, 0..65025 thus becomes -32512 .. 32513 after subtracting this.
	//
	info->mul_scaling.range_bias =
			(((2*info->ab_zero[0]-255) * (2*info->ab_zero[1]-255) +256) >> 9) *256;


	if( info->min_max_precalc < 3 ){
		// given {a,b}_in_{min,max}, and assuming
		//   in_min <= 0 ,  in_max >=0 , in_max > in_min:
		// - largest and smallest output values are given by the below
		float out_max_all = fmaxf( a_in_max * b_in_max, a_in_min * b_in_min);
		float out_min_all = fminf( a_in_max * b_in_min, a_in_min * b_in_max);

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
			info->has_run_before= 1;
		}else{
			// @@ does mul have a constraint that output range must be at least
			// a certain fraction of worst-case?
			/*
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
			}*/
		}
	}
	int res =set_mul_scaling_for_output_range( info );
	if( res != 0) return errlog(nn,"scaling failed");

	logmsg(nn,3,"scale for %f * %f -> [%f..%f]:  * %d >> %d; adj_bias = %d, range_bias =%d\n",
			info->ab_stepsize[0], info->ab_stepsize[1], info->out_min, info->out_max,
			info->mul_scaling.post_scale, info->mul_scaling.post_rsh,
			info->mul_scaling.adj_bias, info->mul_scaling.range_bias);

	info->strategy_valid = 1;
	return 0;
}

static inline int
set_mul_scaling_for_output_range( struct elementwise_d32_info *info)
{
	struct mul_scaling_parms * osp = &info->mul_scaling;
	float out_min = info->out_min;
	float out_max = info->out_max;

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
	return 0;
}

static inline float
convert_mul_intermed_to_outval( struct mul_scaling_parms const * osp, int val )
{
	return (val + osp->range_bias)*osp->all_in_scale_recip;
}

static int
check_mul_need_rerun( struct nn_graph *nn, struct elementwise_d32_info *info, int nthreads)
{
	struct mul_scaling_parms * osp = &info->mul_scaling;
	// collect the min and max ...
	int16_t const * mmp = (int16_t const *)info->minmax_bufs;
	int minvali = mmp[0];
	int maxvali = mmp[1];
	for( int i = 1; i < nthreads; i++){
		minvali = max_i32(minvali, mmp[64*i]);	// max, since min is encoded as ~min
		maxvali = max_i32(maxvali, mmp[64*i+1]);
	}
	minvali = -1-minvali;			// now is proper min
	//
	// see if there was an overflow when converting to u8..
	int minval8 = ((minvali  * osp->post_scale + 16384) >> 15) + osp->adj_bias;
	int maxval8 = ((maxvali  * osp->post_scale + 16384) >> 15) + osp->adj_bias;
	int r = osp->post_rsh;
	if( r > 0){
		minval8 = ((minval8>>(r-1)) + 1)>>1;
		maxval8 = ((maxval8>>(r-1)) + 1)>>1;
	}
	int min_max_precalc = info->min_max_precalc;

	int any_adj = 0;
	float out_min = info->out_min;
	float out_max = info->out_max;

	float actual_min = convert_mul_intermed_to_outval( osp, minvali);
	float actual_max = convert_mul_intermed_to_outval( osp, maxvali);
	float out_range = actual_max - actual_min;
	logmsg(nn,3,"new range found is: %d.. %d  [%f ... %f]\n", minvali, maxvali, actual_min, actual_max);

	// pad the range out a bit
	out_range = out_range + fmaxf(0.15f*out_range, 1e-4);
	float padded_min = fmin(0.0f, actual_max-out_range);
	float padded_max = fmax(0.0f, actual_min+out_range);

	if ( (min_max_precalc & 1) == 0 &&  minval8 < 0){
		logmsg(nn,2,"adjusting out_min from %f to %f (actual min is %f)", out_min, padded_min, actual_min);
		out_min = padded_min;
		any_adj= 1;
	}
	if ( (min_max_precalc & 2) == 0&& maxval8 > 255){
		logmsg(nn,2,"adjusting out_max from %f to %f (actual max is %f)", out_max, padded_max, actual_max);
		out_max = padded_max;
		any_adj= 1;
	}

	if( !any_adj)
		return 0;	// all done, if no adjustment needed

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
	}
	logmsg(nn,3,"rerange to %f  ... %f\n", info->out_min, info->out_max);
	int res = set_mul_scaling_for_output_range(info);
	if( res <0) return -1;
	return 1;
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//
// check strategy for addsub cases. Returns 0 if not valid and needs to be rebuilt, 1 if ok
// If the shapes has not changed, but input ranges have, redo the scaling and return 1.
//
static int elementwise_strategy_valid_check(struct nn_node *self, struct nn_graph *nn,struct elementwise_d32_info *info)
{
	struct tensor const *inX_tensor = self->inputs[0];
	struct tensor const *inY_tensor = self->inputs[1];

	// if not valid, or if any shapes have changed, it's not valid.
	if( info->strategy_valid ==0
	  ||  !shape_matches(&inX_tensor->shape, &info->x_shape)
	   || !format_matches(&inX_tensor->format, &info->x_format)
	   || !shape_matches(&inY_tensor->shape, &info->y_shape)
       || !format_matches(&inY_tensor->format, &info->y_format)){
		return 0;
	}
	// check if data has moved.
	// may be a way to adjust strategy for this .. but it's unlikely to happen anyway.
	if( inX_tensor->data != info->x_data || inY_tensor->data != info->y_data) return 0;

	// shape-related strategy is ok. Check the input ranges; if all are unchanged
	// then we don't need to recalc scaling.
	// Also, if single_B_in_p is not null, the value it points to must
	// (still) be  single_B_in.
	if(  tensor_get_float(self->inputs[2],0) == info->xy_in_min[0]
	   && tensor_get_float(self->inputs[3],0) == info->xy_in_max[0]
	   && tensor_get_float(self->inputs[4],0) == info->xy_in_min[1]
	   && tensor_get_float(self->inputs[5],0) == info->xy_in_max[1]
	   && ((info->single_B_in_p==NULL) || (*info->single_B_in_p == info->single_B_in))){
		return 1;	// strategy still good.
	}
	// redo scaling for new ranges (or new single 'B' value).
	// Call the right function, according to op.
	return (*info->setup_scaling_funcp)(nn,self);
}
static void elementwise_d32_worker_func( struct nn_graph *nn, void * rstpv);

//
// common execute for d32 and subtract (and maybe Mul will go here later...)
//
//

static int addsub_d32_execute_common(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	struct elementwise_d32_info * __restrict info  = ( struct elementwise_d32_info *)self->opaque;

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
		 case OP_QuantizedMul_8x8to8_d32:
			 operator = op_mul;
			 use_hvx = 0;
			 break;
		 case OP_QuantizedMul_8x8to8_d32_ref:
			 operator = op_mul;
			 use_hvx = 0;
			 break;
		 default:
			 return errlog(nn,"unexpected node_type = %d", node_type);
		}
		node_name = hexagon_nn_op_names[node_type];
	}

	logmsg(nn,2,"%s execute. self=%p ",node_name,self);

	// is strategy valid?
	//  returns 0 for no, 1 for yes, -1 for error.
	// Note that 'valid_check' may recalc scaling if the
	// strategy is otherwise ok.
	//
	int res = elementwise_strategy_valid_check(self,nn, info);
	if( unlikely(res <=0)){
		if (res < 0)
			return res;			// failed and reported
		res = setup_elementwise_strategy( nn, self, operator);
		if( res == 0){
			res =(info->setup_scaling_funcp)( nn,self);
		}
		if( res !=0 ) return -1;
	}

	// strategy is good to go..
	// if it's a '1111' mode, we can just do that
	if( info->broadcast_mode == mode_broadcast_1111 ){
		tensor_set_float( out_min_tensor, 0, info->out_min );
		tensor_set_float( out_max_tensor, 0, info->out_max );
		struct tensor const * inA_tensor= self->inputs[(info->swapXY)?1:0];
		int code_scale = info->bc1111_scaling.scale;
		int code_offs = info->bc1111_scaling.offs;
		res = tensor_copy_scaled_d32( nn, inA_tensor, out_tensor, code_scale, code_offs, use_hvx, OP_ADD_D32_MAX_THREADS);
		return res;
	}
	logmsg(nn,3,"bcast_mode = %d compat = 0x%X: running %d rows of nd32=%d; (depth=%d) batches = %d, height in %d chunks of %d (total %d)\n",
			 (int)info->broadcast_mode, info->compat_flags, info->run_height,info->run_nd32, info->run_depth,
			(int)info->out_shape.batches, info->height_chunks, info->chunk_rows, info->num_work_units );

	if( info->core_oper_funcp == NULL) return errlog(nn,"did not set function");

	int need_ranging = (info->min_max_precalc ==3)? 0:1;
	int ran_again = 0;
	while(1){  // maybe once, maybe twice

		int nthreads = min_i32(OP_ADD_D32_MAX_THREADS, info->num_work_units );

		struct elementwise_d32_runstate runstate;
		runstate.info = info;
		runstate.scratch_idx = 0;
		runstate.jobno = 0;
		runstate.nthreads = nthreads;
		runstate.find_range = need_ranging;
		nn_sem_init( &runstate.done_sem, 0 );

		for( int i =0; i < nthreads; i++){
			nn_os_work_for_vector(nn,elementwise_d32_worker_func,&runstate);
		}
		nn_sem_wait_n_times( &runstate.done_sem, nthreads);
		if (need_ranging==0)
			break;
		int need_rerun = (*info->check_need_rerun_funcp)( nn, info, nthreads );
		if( need_rerun <= 0){
			if( need_rerun < 0) return -1;
			break;
		}
		ran_again = 1;	// only for TEST_PERFORMANCE
		need_ranging = 0;
	}
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("add_d32 %s cycles = %d (elements = %d) ran_again = %d\n",
			use_hvx?"hvx": "ref",(end_time-start_time),
			(int)tensor_element_count(out_tensor), ran_again);
#endif

	tensor_set_float( out_min_tensor, 0, info->out_min );
	tensor_set_float( out_max_tensor, 0, info->out_max );
	return 0;
	//return errlog(nn,"broadcast_mode = %d not implemented yet", (int)info->broadcast_mode);
}

//
// this struct gathers the things that the hvx code needs, so we can make
// a call to it with a small # of parms
// (these are all the things that don't change from one slice to the next)
//
struct elementwise_hvx_thrinfo {
	struct elementwise_d32_info *info;	// to get scaling parms
	uint8_t* minmax_buf;
	int inA_height_stride;
	int inA_d32_stride;
	int inB_height_stride;
	int inB_d32_stride;
	int out_height_stride;
	int out_d32_stride;
	int width;
	int depth;		// depth units to process.
	int find_range;		// do we need to find the range
};
static inline uint8_t const *
align_uint8_k_ptr( uint8_t const *p )
{
	return (uint8_t const *)( (size_t)p & ~(size_t)127);
}

static void
elementwise_d32_worker_func( struct nn_graph *nn, void * rstpv)
{
	struct elementwise_d32_runstate *rstp = (struct elementwise_d32_runstate*)rstpv;
	struct elementwise_d32_info *info = rstp->info;
	// trhrix is our thread index, 0..nthreads-1.
	int nthreads =rstp->nthreads;

	int thridx = __sync_fetch_and_add( & rstp->scratch_idx,1);

	struct elementwise_hvx_thrinfo thrinfo;
	thrinfo.info = info;
	thrinfo.minmax_buf = info->minmax_bufs + sizeof(HVX_Vector)*thridx;

	*(HVX_Vector *)thrinfo.minmax_buf = q6op_Vh_vsplat_R(0x8000);

	int jobidx;
	int njobs = info->chunkdesc_count;

	// hoist all the things we need
	thrinfo.inA_d32_stride = info->tinA.d32_stride;
	thrinfo.inA_height_stride = info->tinA.height_stride;
	thrinfo.inB_d32_stride = info->tinB.d32_stride;
	thrinfo.inB_height_stride = info->tinB.height_stride;
	thrinfo.out_d32_stride = info->tout.d32_stride;
	thrinfo.out_height_stride = info->tout.height_stride;

	thrinfo.width = info->out_shape.width;
	thrinfo.depth = info->run_depth;
	thrinfo.find_range = rstp->find_range;

	core_oper_fp funcp = info->core_oper_funcp;

	if( info->oper_code <=op_LASTADDSUB && info->addsub_scaling.underrange)
		funcp = core_add_d32_underrange_hvx;

	unsigned pfa_wid = info->pf_wid_a;
	unsigned pfa_stride = info->tinA.d32_stride;
	unsigned pfb_wid = info->pf_wid_b;
	unsigned pfb_stride = info->tinA.d32_stride;
	struct elementwise_chunk_descriptor const * chunks = info->chunkdescs;
	struct elementwise_chunk_descriptor const * last_chunk = &chunks[njobs-1];
	struct elementwise_chunk_descriptor const * pending_chunk = NULL;

	// The logic in the loop here is:
	//   (1) get a chunk from the pool using sync_fetch_and_add; get a pointer
	//     to its desc (this_chunk)
	//   (2) wait for any prefetch to finish
	//   (3) issue l2 prefetch for that chunk
	//   (4) if there's a valid *previous* job( pending_chunk), then do that one.
	//   (5) The pointer obtained in (1) becomes the new 'pending_chunk'
	//  At the end, if there's a pending_chunk, we need to do that.


	while( jobidx = __sync_fetch_and_add(&rstp->jobno,1),  jobidx < njobs){
		struct elementwise_chunk_descriptor const * this_chunk = &chunks[jobidx];
		struct elementwise_chunk_descriptor const * next_chunk_maybe = this_chunk + nthreads;

		if(pending_chunk != NULL) Q6_dcfetch_A(pending_chunk);

		wait_for_l2fetch();

#if defined(__hexagon__)
		asm volatile("hintjr(%0)"::"r"(funcp));
#endif

		// find the input pointers; prefetch the 'new' job
		uint8_t const  *inpA = this_chunk->data_A;
		uint8_t const  *inpB = this_chunk->data_B;
		uint32_t pf_b_a_ht = this_chunk->pf_b_a_ht;
		unsigned pf_b_rows = pf_b_a_ht >>16;
		if( pf_b_rows != 0){
			l2fetch( align_uint8_k_ptr(inpB), pfb_stride, pfb_wid, pf_b_rows);
			if(0)printf("thrd %d .... l2fetch(%p,%d,%d,%d)\n",
					thridx, align_uint8_k_ptr(inpB), pfb_stride, pfb_wid, pf_b_rows);
		}
		l2fetch( align_uint8_k_ptr(inpA), pfa_stride, pfa_wid, (uint16_t)pf_b_a_ht);
		if(0)printf("thrd %d l2fetch(%p,%d,%d,%d)\n",
				thridx, align_uint8_k_ptr(inpA), pfa_stride, pfa_wid, (uint16_t)pf_b_a_ht);
		if(pending_chunk!=NULL){
			// output pointer
			uint8_t  *outp = pending_chunk->data_out;
			uint8_t const  *inpA = pending_chunk->data_A;
			uint8_t const  *inpB = pending_chunk->data_B;
			if(0)printf("thrd %d : %d rows %p + %p\n", thridx,  pending_chunk->rows, inpA, inpB);
			(*funcp)( &thrinfo, outp, inpA, inpB, pending_chunk->rows);
		}
		pending_chunk = this_chunk;
		// try to avoid cache miss on next descriptor in L1
		if( next_chunk_maybe <= last_chunk) Q6_dcfetch_A(next_chunk_maybe);
	} // end of work units
	// usually one pending...
	if( pending_chunk != NULL){
		if(0)printf("thrd %d :: %d rows %p + %p\n", thridx,  pending_chunk->rows, pending_chunk->data_A, pending_chunk->data_B);

		(*funcp)( &thrinfo, pending_chunk->data_out, pending_chunk->data_A, pending_chunk->data_B, pending_chunk->rows);
	}


	// lateral reduce on minmax.. contains 32 of { ~min, max }
	if( rstp->find_range){
		HVX_Vector mmax_all = *(HVX_Vector *)thrinfo.minmax_buf;
		for(int i = 0; i < 5; i++){
			HVX_VectorPair vdealt = Q6_W_vdeal_VVR(mmax_all,mmax_all,-4 );
			mmax_all  = Q6_Vh_vmax_VhVh( Q6_V_hi_W(vdealt),Q6_V_lo_W(vdealt));
		}
		 *(HVX_Vector *)thrinfo.minmax_buf = mmax_all;
	}
	nn_sem_post( &rstp->done_sem);
}



#if 0 // of historical interest...

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
	int height = thrp->rows;
	int width = rstp->width;
	uint8_t const * ptrA = thrp->ptrA;
	int32_t row_stride_A = rstp->tinA.height_stride;
	uint8_t const * ptrB = thrp->ptrB;
	int32_t row_stride_B = rstp->tinB.height_stride;
	uint8_t* pout = thrp->pout;
	int32_t row_stride_out = rstp->tout.height_stride;
	int depth = thrp->d_num;

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
#endif
//
// HVX Version
//
//
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////.///// For Add/Sub //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
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

static inline HVX_Vector * __attribute__((always_inline))
vecptr_add_bytes( HVX_Vector * p, int32_t offs )
{
	return (HVX_Vector *)( (char *)p + offs);
}
static inline HVX_Vector const * __attribute__((always_inline))
vecptrk_add_bytes( HVX_Vector const* p, int32_t offs )
{
	return (HVX_Vector const *)( (char const *)p + offs);
}
// Actual HVX core;
// this does 'height' rows; the width and depth are
// as stored in the threadinfo struct.
//
//-----------------------------------------
// NOTE: this is effectively a template
// with 'broadcast_mode' being a constant when it's expanded;
// we expand it for three cases.
//
// for broadcast_mode = broadcast_mode_general:
//  - both A and B are full tensors (but B could be height-broadcasted, by having a zero height stride)
//  - B could have a different width padding on left, and we need to deal with that here.
//
// for broadcast_mode = broadcast_mode_B11D:
//  - Broadcasting on height and depth. So we need to get a single group of 32
//    values for depth dimension, outside the loops. We read a vector there
//    and process it via info->bcast_wd_deltagen, which broadcasts to w, and will
//    also broadcast in D if applicable ( based on bcast_wd_deltagen).
//
// for broadcast_mode = broadcast_mode_mixed:
//    this applies when we are not broadcasting on height, but are broadcasting on W and/or D; or if we are
//    broadcasting on height and depth, but not on width. In this case we use a delta on the B reads
//    to broadcast D and/or W as applicable; we also need to conditionally advance the B pointer, or not,
//    across the W loop.
//
//
//
//
static inline void __attribute__((always_inline))
core_add_d32_hvx_inline( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height, int broadcast_mode )
{
	struct elementwise_d32_info const * info =	thrp->info;
	int is_B11D = (broadcast_mode == mode_broadcast_B11D);
	int is_mixed = broadcast_mode == mode_broadcast_mixed;

	int width = thrp->width;
	int32_t row_stride_A =  thrp->inA_height_stride;
	int32_t row_stride_B =  thrp->inB_height_stride;
	int32_t row_stride_out = thrp->out_height_stride;
	int32_t inB_d32_stride = thrp->inB_d32_stride;

	// when broadcasting from W and/or D in addition to B & H, we construct a single dummy
	// row which can be be traversed in W dimension (H & B strides set to 0). 111D is a special case
	// where the B dat is loop invariant.
	// to enable broadcast from W and/or D without broadcasting from both H and B, we can't make
	// a dummy row. To handle that, a variant of this template is made with is_WorD_bcast=1.
	// That code will traverse the B side, while broadcasting from W and/or D. W broadcast causes
	// w_bvec_inc to become 0 instead of 1, disabling horizontal traversal; and it causes a 'splat 32'
	// to all 4 outputs' to be used in the wd_splat vector.
	// D broadcast causes wd_splat to contain a 'splat one to 32' effect as well, which can be combined
	// with the w splat.
	// ** this is not yet implemented. wd_splat should found in planning **.
	int w_bvec_inc = 1;		// # of vecs to bump b ptr to traverse w.
	HVX_Vector wd_splat = Q6_V_vzero();	// only used if is_WorD_bcast
	if( broadcast_mode != mode_broadcast_general){
		// suppress B_d32_stride when broadcasting depth
		if( info->bcast_wd_deltagen & 1){
			inB_d32_stride = 0;
		}
		// disable incrementing B in W direction, in broadcast_mixed if we are broadcasting in w dim.
		// this can determined from bit 5 of nfo->bcast_wd_deltagen
		if( broadcast_mode == mode_broadcast_mixed)
			if(info->bcast_wd_deltagen &0x20 ) w_bvec_inc = 0;
		wd_splat= Q6_V_vand_VV(*(HVX_Vector const*)const_Count128, Q6_V_vsplat_R(info->bcast_wd_deltagen));
	}

	// ======== mode =========== is_B11D  is_mixed   w_bvec_inc   wd_splat
	// mode_broadcast_general      0          0        1            0 [1]
	// mode_broadcast_B11D         1          0        1 [1]       valid
	// mode_broadcast_mixed        0          1        ? [2]       valid
	// notes:
	//   [1] not used
	//   [2] 0 if broadcasting W, 1 if not.


	int iht,iwd;
	int scales_hi = info->addsub_scaling.scales_hi;
	int scales_lo = info->addsub_scaling.scales_lo;
	int a_scale = info->addsub_scaling.a_scale;
	int b_scale =info->addsub_scaling.b_scale;
	int offset = info->addsub_scaling.offset;

	scales_hi = Q6_R_combine_RlRl( scales_hi, scales_hi);
	scales_lo = Q6_R_combine_RlRl( scales_lo, scales_lo);
	a_scale = Q6_R_combine_RlRl( a_scale, a_scale);
	b_scale = Q6_R_combine_RlRl( b_scale, b_scale);
	offset = Q6_R_combine_RlRl( offset, offset );

	int depth = thrp->depth;
	int nd32 = (depth+31)/32u;		// no of slices to do.
	int final_rsh = info->addsub_scaling.final_rsh;

	//
	// find # of vector loops in w dimension
	//
	int wpad_bytes = (size_t)ptrA & 0x60;	// 0,32,64 or 96
	int wpad_bytes_B = (size_t)ptrB & 0x60;
	int vlalign_val = 0;
	int wlen = width*32 + wpad_bytes;		// determines right mask
	int nvecw = (unsigned)(wlen-1)/128;		// # of vecs, -1.


	HVX_Vector const *vinpA_0 = (HVX_Vector const *)( ptrA - wpad_bytes);
	HVX_Vector const *vinpB_0 = NULL;
	HVX_Vector *voutp_0 = (HVX_Vector  *)( pout - wpad_bytes);

	struct prod_vin_with_scale vinBs;
	vinpB_0 = (HVX_Vector const *)( ptrB - wpad_bytes_B);	// aligned pointer
	if( !is_B11D){
		if( is_mixed && w_bvec_inc == 0){					// broadcasting w...
			vlalign_val = 128-wpad_bytes_B;					// vlalign needs to align B to 0
		}else{
			vlalign_val = wpad_bytes - wpad_bytes_B;		// for aligning b input to A
			if( vlalign_val < 0) vinpB_0 ++;				// point to first in-loop load of B.
		}
	}

	// we have two loops for with or without 'find_range'.
	// But only the 'mode_broadcast_general really benefits from not
	// needing to find the range; other modes we just use the 'with min/max'
	// loop to save code space, and suppress some of the ranging related ops
	// when find_range = 0.
	//
	int find_range = thrp->find_range;

	if(broadcast_mode == mode_broadcast_general && find_range == 0 ){		// fast version (no min/max calc needed)
		for(int id32 = 0; id32 < nd32; id32++){
			HVX_Vector const *vinpA_d = vinpA_0;
			HVX_Vector const *vinpB_d = vinpB_0;
			HVX_Vector *voutp_d = voutp_0;

			if( is_B11D){
				// get the (only) B side value (just 32 bytes needed) and splat to all.
				HVX_Vector vbin = Q6_V_vror_VR( *vinpB_d,wpad_bytes_B);
				vbin = Q6_V_vdelta_VV( vbin, wd_splat);
				vinBs = find_vinB_product( vbin, b_scale);
			}
			for(iht = 0; iht < height; iht ++){
				HVX_Vector const *vinpA = vinpA_d;
				HVX_Vector const *vinpB = vinpB_d;
				HVX_Vector *voutp = voutp_d;
				HVX_Vector vb_prev = Q6_V_vzero();

				//
				// start up..
				//
				HVX_Vector vinA = *vinpA++;
				HVX_VectorPair vtmp;
				HVX_Vector vbalign;
				if( is_B11D){
					vtmp = first_stage_111D( vinA, vinBs, a_scale );
				}else{
					HVX_Vector vinb = *vinpB++;
					vb_prev = (vlalign_val <0)? vinpB[-2]: vinb;
					vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
					if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
					vtmp = first_stage( vinA, vbalign, scales_hi, scales_lo );
					if( w_bvec_inc) vb_prev = vinb;
				}

				// loop (zero times or more)
				for( iwd = 0; iwd < nvecw; iwd++ ){
					HVX_Vector vout = second_stage( vtmp, offset,  final_rsh);
					HVX_Vector vinA = *vinpA++;
					if( is_B11D){
						vtmp = first_stage_111D( vinA, vinBs, a_scale );
					}else{
						if( w_bvec_inc){
							HVX_Vector vinb = *vinpB++;
							vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
							if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
							vb_prev = vinb;
						}
						vtmp = first_stage( vinA, vbalign, scales_hi, scales_lo );
					}
					*voutp ++ = vout;
				}
				//
				// last one
				//
				*voutp= second_stage( vtmp, offset, final_rsh);

				vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
				vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
				voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
			}
			vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
			vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
			voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);
		} // for id32
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
		HVX_Vector minmax_all = Q6_V_vzero();
		HVX_Vector vCenter = Q6_V_vzero();
		if( find_range){
			minmax_all = *(HVX_Vector *)thrp->minmax_buf;
			vCenter = q6op_Vh_vsplat_R(info->addsub_scaling.intermed_zero);
		}
		HVX_Vector vmin0 = vCenter;
		HVX_Vector vmax0 = vCenter;
		HVX_Vector vmin1 = vCenter;
		HVX_Vector vmax1 = vCenter;

		for(int id32 = 0; id32 < nd32; id32++){
			HVX_Vector const *vinpA_d = vinpA_0;
			HVX_Vector const *vinpB_d = vinpB_0;
			HVX_Vector *voutp_d = voutp_0;

			if( is_B11D){
				// get the (only) B side value (just 32 bytes needed) and splat to all.
				HVX_Vector vbin = Q6_V_vror_VR( *vinpB_d,wpad_bytes_B);
				vbin = Q6_V_vdelta_VV( vbin, wd_splat);
				vinBs = find_vinB_product( vbin, b_scale);
			}
			for(iht = 0; iht < height; iht ++){
				HVX_Vector const *vinpA = vinpA_d;
				HVX_Vector const *vinpB = vinpB_d;
				HVX_Vector *voutp = voutp_d;

				HVX_Vector vb_prev = Q6_V_vzero();
				//
				// start up..
				//
				HVX_VectorPred qleft =  Q6_Q_vsetq_R(wpad_bytes);
				HVX_Vector vinA = *vinpA++;
				HVX_VectorPair vtmp;
				HVX_Vector vbalign;
				if( is_B11D){
					vtmp = first_stage_111D( vinA, vinBs, a_scale );
				}else{
					HVX_Vector vinb = *vinpB++;
					vb_prev = (vlalign_val <0)? vinpB[-2]: vinb;
					vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
					if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
					vtmp = first_stage( vinA, vbalign, scales_hi, scales_lo );
					if( w_bvec_inc) vb_prev = vinb;
				}
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
					if( is_B11D){
						vtmp = first_stage_111D( vinA, vinBs, a_scale );
					}else{
						if( w_bvec_inc){
							HVX_Vector vinb = *vinpB++;
							vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
							if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
							vb_prev = vinb;
						}
						vtmp = first_stage( vinA, vbalign, scales_hi, scales_lo );
					}
					*voutp ++ = vout;
					vt0 = Q6_V_lo_W(vtmp);
					vt1 = Q6_V_hi_W(vtmp);
				}

				// This creates a fake dependency between vt0 and mem here, to suppress
				// a loop optimization that causes ICE in 8.0. It's not needed for 8.1
				//HVX_8_0_FAKEDEP_VM( vt0, *voutp);
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

					*voutp = second_stage(  vtmp, offset, final_rsh);
				}
				vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
				vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
				voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
			} // end height loop
			vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
			vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
			voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);

			// now, each of min and max has 128 values
			// reduce, ignoring values outside the depth range
			//
			int dnum = min_i32(32,depth-id32*32);	// if < 32, we have an odd last slice

			HVX_Vector vminmax_this = hvx_reduce_minmax_h_over_width( vmin0, vmax0, vmin1, vmax1);
			// we now have (~min,max),(~min,max).. for all 32 in the right order.

			// if we are on the last iteration, dnum is 1..31 and only the first
			// 'dnum' of the values are valid. So combine them like this...
			// (no effect when dnum = 32).
			vminmax_this = Q6_V_vlalign_VVR( vminmax_this, minmax_all, 4*(32-dnum));
			minmax_all = Q6_Vh_vmax_VhVh( minmax_all, vminmax_this);

		}// depth loop
		if( find_range)*(HVX_Vector *)thrp->minmax_buf = minmax_all;
	}
}

// expand that inline template with the three cases...
static void
core_add_d32_general_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height )
{
	core_add_d32_hvx_inline( thrp, pout, ptrA, ptrB, height, mode_broadcast_general);
}
static void
core_add_d32_B11D_hvx(struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height)
{
	core_add_d32_hvx_inline( thrp, pout, ptrA, ptrB, height, mode_broadcast_B11D);
}
static void
core_add_d32_mixed_hvx(struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height)
{
	core_add_d32_hvx_inline( thrp, pout, ptrA, ptrB, height, mode_broadcast_mixed);
}

/////////////////////////////////////////
// 'underrange' add computation, for scales which are out of range for the main one
// (1) read a[i],b[i], zero extend them;
// (2) find sca*a[i] + scb*b[i] as a 32-bit quantity
//  (3) >> by initial_rsh
//  (4) add offset and sat to i16
//  (5) >>final_rsh, round, convert to u8
//

// this does (1) (2) (3)
static inline HVX_Vector_x4
first_stage_underrange( HVX_Vector va, HVX_Vector vb, int scalea, int scaleb, int initial_rsh)
{
	HVX_VectorPair xta = Q6_Wb_vshuffoe_VbVb( Q6_V_vzero(), va );
	HVX_VectorPair xtb = Q6_Wb_vshuffoe_VbVb( Q6_V_vzero(), vb );
	HVX_VectorPair p02 = Q6_Ww_vmpy_VhRh( Q6_V_lo_W(xta), scalea);	// even 32-bit prods
	HVX_VectorPair p13 = Q6_Ww_vmpy_VhRh( Q6_V_hi_W(xta), scalea);	// odd 32-bit prods
	p02 = Q6_Ww_vmpyacc_WwVhRh_sat( p02, Q6_V_lo_W(xtb), scaleb);
	p13 = Q6_Ww_vmpyacc_WwVhRh_sat( p13, Q6_V_hi_W(xtb), scaleb);

	HVX_Vector_x4 result;
	result.val[0] = Q6_Vw_vasr_VwR( Q6_V_lo_W(p02), initial_rsh);
	result.val[1] = Q6_Vw_vasr_VwR( Q6_V_lo_W(p13), initial_rsh);
	result.val[2] = Q6_Vw_vasr_VwR( Q6_V_hi_W(p02), initial_rsh);
	result.val[3] = Q6_Vw_vasr_VwR( Q6_V_hi_W(p13), initial_rsh);
	return result;
}
static inline HVX_Vector
second_stage_underrange( HVX_Vector_x4 first,int offset, int final_rsh )
{
	HVX_Vector voffs = Q6_V_vsplat_R( offset);
	HVX_Vector p0= Q6_Vw_vadd_VwVw_sat( first.val[0], voffs );
	HVX_Vector p1= Q6_Vw_vadd_VwVw_sat( first.val[1], voffs );
	HVX_Vector p2= Q6_Vw_vadd_VwVw_sat( first.val[2], voffs );
	HVX_Vector p3= Q6_Vw_vadd_VwVw_sat( first.val[3], voffs );
	HVX_Vector s02 = Q6_Vh_vsat_VwVw( p2,p0);
	HVX_Vector s13 = Q6_Vh_vsat_VwVw( p3,p1);
	return Q6_Vub_vasr_VhVhR_rnd_sat( s13,s02,final_rsh);
}


// This is a 'underrange' case which can handle all modes (it uses the 'mixed' strategy, and can handle
// final_rsh < 0. It is used only when output  min/max are preset, so it does not need
// to find min/max in the loop.

static void
core_add_d32_underrange_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height )
{
	struct elementwise_d32_info const * info =	thrp->info;


	int width = thrp->width;
	int32_t row_stride_A =  thrp->inA_height_stride;
	int32_t row_stride_B =  thrp->inB_height_stride;
	int32_t row_stride_out = thrp->out_height_stride;
	int32_t inB_d32_stride = thrp->inB_d32_stride;

	// when broadcasting from W and/or D in addition to B & H, we construct a single dummy
	// row which can be be traversed in W dimension (H & B strides set to 0). 111D is a special case
	// where the B dat is loop invariant.
	// to enable broadcast from W and/or D without broadcasting from both H and B, we can't make
	// a dummy row. To handle that, a variant of this template is made with is_WorD_bcast=1.
	// That code will traverse the B side, while broadcasting from W and/or D. W broadcast causes
	// w_bvec_inc to become 0 instead of 1, disabling horizontal traversal; and it causes a 'splat 32'
	// to all 4 outputs' to be used in the wd_splat vector.
	// D broadcast causes wd_splat to contain a 'splat one to 32' effect as well, which can be combined
	// with the w splat.
	// ** this is not yet implemented. wd_splat should found in planning **.
	int w_bvec_inc = 1;		// # of vecs to bump b ptr to traverse w.
	// suppress B_d32_stride when broadcasting depth
	if( info->bcast_wd_deltagen & 1){
		inB_d32_stride = 0;
	}
	// disable incrementing B in W direction, in broadcast_mixed if we are broadcasting in w dim.
	// this can determined from bit 5 of nfo->bcast_wd_deltagen
	if(info->bcast_wd_deltagen &0x20 ) w_bvec_inc = 0;
	HVX_Vector wd_splat= Q6_V_vand_VV(*(HVX_Vector const*)const_Count128, Q6_V_vsplat_R(info->bcast_wd_deltagen));


	int iht,iwd;
	int a_scale = info->addsub_scaling.a_scale;
	int b_scale =info->addsub_scaling.b_scale;
	int offset = info->addsub_scaling.offset;

	a_scale = Q6_R_combine_RlRl( a_scale,a_scale);
	b_scale = Q6_R_combine_RlRl( b_scale,b_scale);

	int depth = thrp->depth;
	int nd32 = (depth+31)/32u;		// no of slices to do.
	int initial_rsh = 8;
	int final_rsh = info->addsub_scaling.final_rsh;
	// if final_rsh <1, make it 1 and reduce inital_rsh.
	if( info->addsub_scaling.underrange){
		initial_rsh += final_rsh-1;
		final_rsh = 1;
	}
	//
	// find # of vector loops in w dimension
	//
	int wpad_bytes = (size_t)ptrA & 0x60;	// 0,32,64 or 96
	int wpad_bytes_B = (size_t)ptrB & 0x60;
	int vlalign_val = 0;
	int wlen = width*32 + wpad_bytes;		// determines right mask
	int nvecw = (unsigned)(wlen-1)/128;		// # of vecs, -1.


	HVX_Vector const *vinpA_0 = (HVX_Vector const *)( ptrA - wpad_bytes);
	HVX_Vector const *vinpB_0 = NULL;
	HVX_Vector *voutp_0 = (HVX_Vector  *)( pout - wpad_bytes);

	vinpB_0 = (HVX_Vector const *)( ptrB - wpad_bytes_B);	// aligned pointer
	if(  w_bvec_inc == 0){					// broadcasting w...
		vlalign_val = 128-wpad_bytes_B;					// vlalign needs to align B to 0
	}else{
		vlalign_val = wpad_bytes - wpad_bytes_B;		// for aligning b input to A
		if( vlalign_val < 0) vinpB_0 ++;				// point to first in-loop load of B.
	}

	for(int id32 = 0; id32 < nd32; id32++){
		HVX_Vector const *vinpA_d = vinpA_0;
		HVX_Vector const *vinpB_d = vinpB_0;
		HVX_Vector *voutp_d = voutp_0;

		for(iht = 0; iht < height; iht ++){
			HVX_Vector const *vinpA = vinpA_d;
			HVX_Vector const *vinpB = vinpB_d;
			HVX_Vector *voutp = voutp_d;
			HVX_Vector vb_prev = Q6_V_vzero();
			//
			// start up..
			//
			HVX_Vector vinA = *vinpA++;
			HVX_Vector_x4 vtmp;
			HVX_Vector vbalign;
			HVX_Vector vinb = *vinpB++;
			vb_prev = (vlalign_val <0)? vinpB[-2]: vinb;
			vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
			vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
			vtmp = first_stage_underrange( vinA, vbalign, a_scale, b_scale, initial_rsh  );
			if( w_bvec_inc) vb_prev = vinb;

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				HVX_Vector vout = second_stage_underrange( vtmp, offset,  final_rsh);
				HVX_Vector vinA = *vinpA++;
				if( w_bvec_inc){
					HVX_Vector vinb = *vinpB++;
					vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
					vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
					vb_prev = vinb;
				}
				vtmp = first_stage_underrange( vinA, vbalign, a_scale, b_scale, initial_rsh );
				*voutp ++ = vout;
			}
			//
			// last one
			//
			*voutp= second_stage_underrange( vtmp, offset, final_rsh);

			vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
			vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
			voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
		}
		vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
		vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
		voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);
	} // for id32
}
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////.///// For Multiply /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

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


// The first operation
//    (a[i]-za)*(b[i]-zb)-range_bias
//
// is actually done as
//  (za*zb-range_bias) + a[i]*b[i]  -  ( a[i]*zb + b[i]*za)
//
// .. where each product is u8*u8 -> u16, and all the +/- are mod 64K.
// This eliminates the need to zero-extend anything
//
// For the broadcast where 'b' side is invariant (same 128 values) over a work
// unit, we could do the op as
//
//    P = (a[i]-za)*(b[i]-zb)-range_bias
//   = [ za*zb- range_bias - za*b[i]] + a[i]*b[i] - zb*a[i]
//  .. so we can precalculate the quantity in [], which needs to be in two vectors.
//  I'm not sure that would be any faster, though.
//
//

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

static inline void __attribute__((always_inline))
core_mul_d32_hvx_inline( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height, int broadcast_mode )
{
	struct elementwise_d32_info const * info =	thrp->info;
	int is_B11D = (broadcast_mode == mode_broadcast_B11D);
	int is_mixed = broadcast_mode == mode_broadcast_mixed;

	int width = thrp->width;
	int32_t row_stride_A =  thrp->inA_height_stride;
	int32_t row_stride_B =  thrp->inB_height_stride;
	int32_t row_stride_out = thrp->out_height_stride;
	int32_t inB_d32_stride = thrp->inB_d32_stride;

	// when broadcasting from W and/or D in addition to B & H, we construct a single dummy
	// row which can be be traversed in W dimension (H & B strides set to 0). 111D is a special case
	// where the B dat is loop invariant.
	// to enable broadcast from W and/or D without broadcasting from both H and B, we can't make
	// a dummy row. To handle that, a variant of this template is made with is_WorD_bcast=1.
	// That code will traverse the B side, while broadcasting from W and/or D. W broadcast causes
	// w_bvec_inc to become 0 instead of 1, disabling horizontal traversal; and it causes a 'splat 32'
	// to all 4 outputs' to be used in the wd_splat vector.
	// D broadcast causes wd_splat to contain a 'splat one to 32' effect as well, which can be combined
	// with the w splat.
	// ** this is not yet implemented. wd_splat should found in planning **.
	int w_bvec_inc = 1;		// # of vecs to bump b ptr to traverse w.
	HVX_Vector wd_splat = Q6_V_vzero();	// only used if is_WorD_bcast
	if( broadcast_mode != mode_broadcast_general){
		// suppress B_d32_stride when broadcasting depth
		if( info->bcast_wd_deltagen & 1){
			inB_d32_stride = 0;
		}
		// disable incrementing B in W direction, in broadcast_mixed if we are broadcasting in w dim.
		// this can determined from bit 5 of nfo->bcast_wd_deltagen
		if( broadcast_mode == mode_broadcast_mixed)
			if(info->bcast_wd_deltagen &0x20 ) w_bvec_inc = 0;
		wd_splat= Q6_V_vand_VV(*(HVX_Vector const*)const_Count128, Q6_V_vsplat_R(info->bcast_wd_deltagen));
	}

	// ======== mode =========== is_B11D  is_mixed   w_bvec_inc   wd_splat
	// mode_broadcast_general      0          0        1            0 [1]
	// mode_broadcast_B11D         1          0        1 [1]       valid
	// mode_broadcast_mixed        0          1        ? [2]       valid
	// notes:
	//   [1] not used
	//   [2] 0 if broadcasting W, 1 if not.


	int iht,iwd;
	int za = info->ab_zero[0];
	int zb = info->ab_zero[1];
	int range_bias = info->mul_scaling.range_bias;
	int post_scale = info->mul_scaling.post_scale;
	int adj_bias = info->mul_scaling.adj_bias;
	int post_rsh = info->mul_scaling.post_rsh;

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

	int depth = thrp->depth;
	int nd32 = (depth+31)/32u;		// no of slices to do.

	//
	// find # of vector loops in w dimension
	//
	int wpad_bytes = (size_t)ptrA & 0x60;	// 0,32,64 or 96
	int wpad_bytes_B = (size_t)ptrB & 0x60;
	int vlalign_val = 0;
	int wlen = width*32 + wpad_bytes;		// determines right mask
	int nvecw = (unsigned)(wlen-1)/128;		// # of vecs, -1.


	HVX_Vector const *vinpA_0 = (HVX_Vector const *)( ptrA - wpad_bytes);
	HVX_Vector const *vinpB_0 = NULL;
	HVX_Vector *voutp_0 = (HVX_Vector  *)( pout - wpad_bytes);

	HVX_Vector vinBs;		// used in B11D mode only
	vinpB_0 = (HVX_Vector const *)( ptrB - wpad_bytes_B);	// aligned pointer
	if( !is_B11D){
		if( is_mixed && w_bvec_inc == 0){					// broadcasting w...
			vlalign_val = 128-wpad_bytes_B;					// vlalign needs to align B to 0
		}else{
			vlalign_val = wpad_bytes - wpad_bytes_B;		// for aligning b input to A
			if( vlalign_val < 0) vinpB_0 ++;				// point to first in-loop load of B.
		}
	}

	// we have two loops for with or without 'find_range'.
	// But only the 'mode_broadcast_general really benefits from not
	// needing to find the range; other modes we just use the 'with min/max'
	// loop to save code space, and suppress some of the ranging related ops
	// when find_range = 0.
	//
	int find_range = thrp->find_range;

	if(broadcast_mode == mode_broadcast_general && find_range == 0 ){		// fast version (no min/max calc needed)
		for(int id32 = 0; id32 < nd32; id32++){
			HVX_Vector const *vinpA_d = vinpA_0;
			HVX_Vector const *vinpB_d = vinpB_0;
			HVX_Vector *voutp_d = voutp_0;

			if( is_B11D){
				// get the (only) B side value (just 32 bytes needed) and splat to all.
				HVX_Vector vbin = Q6_V_vror_VR( *vinpB_d,wpad_bytes_B);
				vinBs = Q6_V_vdelta_VV( vbin, wd_splat);
			}
			for(iht = 0; iht < height; iht ++){
				HVX_Vector const *vinpA = vinpA_d;
				HVX_Vector const *vinpB = vinpB_d;
				HVX_Vector *voutp = voutp_d;
				HVX_Vector vb_prev = Q6_V_vzero();

				//
				// start up..
				//
				HVX_Vector vinA = *vinpA++;
				HVX_VectorPair vtmp;
				HVX_Vector vbalign;
				if( is_B11D){
					vbalign = vinBs;
				}else{
					HVX_Vector vinb = *vinpB++;
					vb_prev = (vlalign_val <0)? vinpB[-2]: vinb;
					vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
					if( w_bvec_inc) vb_prev = vinb;
					if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
				}
				vtmp = first_multiply( vinA, vbalign, za,zb, mul_bias );

				// loop (zero times or more)
				for( iwd = 0; iwd < nvecw; iwd++ ){
					HVX_Vector vout = second_multiply( vtmp, post_scale, adj_bias, post_rsh);
					HVX_Vector vinA = *vinpA++;
					if( is_B11D){
						vbalign = vinBs;
					}else if( w_bvec_inc){
						HVX_Vector vinb = *vinpB++;
						vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
						if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
						vb_prev = vinb;
					}
					vtmp = first_multiply( vinA, vbalign,  za,zb, mul_bias );
					*voutp ++ = vout;
				}
				//
				// last one
				//
				*voutp= second_multiply( vtmp, post_scale, adj_bias, post_rsh);

				vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
				vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
				voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
			}
			vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
			vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
			voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);
		} // for id32
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
		HVX_Vector minmax_all = Q6_V_vzero();
		HVX_Vector vCenter = Q6_V_vzero();
		if( find_range){
			minmax_all = *(HVX_Vector *)thrp->minmax_buf;
			vCenter = q6op_Vh_vsplat_R(-range_bias);
		}
		HVX_Vector vza = Q6_V_vsplat_R(za_splat);	// za_splat is already splat u8->4xu8
		HVX_Vector vmin0 = vCenter;
		HVX_Vector vmax0 = vCenter;
		HVX_Vector vmin1 = vCenter;
		HVX_Vector vmax1 = vCenter;

		for(int id32 = 0; id32 < nd32; id32++){
			HVX_Vector const *vinpA_d = vinpA_0;
			HVX_Vector const *vinpB_d = vinpB_0;
			HVX_Vector *voutp_d = voutp_0;

			if( is_B11D){
				// get the (only) B side value (just 32 bytes needed) and splat to all.
				HVX_Vector vbin = Q6_V_vror_VR( *vinpB_d,wpad_bytes_B);
				vinBs = Q6_V_vdelta_VV( vbin, wd_splat);
			}
			for(iht = 0; iht < height; iht ++){
				HVX_Vector const *vinpA = vinpA_d;
				HVX_Vector const *vinpB = vinpB_d;
				HVX_Vector *voutp = voutp_d;

				HVX_Vector vb_prev = Q6_V_vzero();
				//
				// start up..
				//
				HVX_VectorPred qleft =  Q6_Q_vsetq_R(wpad_bytes);
				HVX_Vector vinA = *vinpA++;
				HVX_VectorPair vtmp;
				HVX_Vector vbalign;
				vinA = Q6_V_vmux_QVV( qleft, vza, vinA );	// apply mask - replace out lanes with za

				if( is_B11D){
					vbalign = vinBs;
				}else{
					HVX_Vector vinb = *vinpB++;
					vb_prev = (vlalign_val <0)? vinpB[-2]: vinb;
					vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
					if( w_bvec_inc) vb_prev = vinb;
					if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
				}
				vtmp = first_multiply( vinA, vbalign, za,zb, mul_bias );

				HVX_Vector vt0  = Q6_V_vmux_QVV( qleft, vCenter, Q6_V_lo_W( vtmp ));
				HVX_Vector vt1  = Q6_V_vmux_QVV( qleft, vCenter, Q6_V_hi_W( vtmp ));

				// loop (zero times or more)
				for( iwd = 0; iwd < nvecw; iwd++ ){
					vmin0 = Q6_Vh_vmin_VhVh( vmin0, vt0 );
					vmax0 = Q6_Vh_vmax_VhVh( vmax0, vt0 );
					vmin1 = Q6_Vh_vmin_VhVh( vmin1, vt1 );
					vmax1 = Q6_Vh_vmax_VhVh( vmax1, vt1 );
					HVX_Vector vout = second_multiply( Q6_W_vcombine_VV(vt1,vt0), post_scale, adj_bias, post_rsh);

					HVX_Vector vinA = *vinpA++;
					if( is_B11D){
						vbalign = vinBs;
					}else if( w_bvec_inc){
						HVX_Vector vinb = *vinpB++;
						vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
						if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
						vb_prev = vinb;
					}
					vtmp = first_multiply( vinA, vbalign, za,zb, mul_bias );
					*voutp ++ = vout;
					vt0 = Q6_V_lo_W(vtmp);
					vt1 = Q6_V_hi_W(vtmp);
				}

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

					*voutp = second_multiply(  Q6_W_vcombine_VV(vt1,vt0),post_scale, adj_bias, post_rsh);
				}
				vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
				vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
				voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
			} // end height loop
			vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
			vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
			voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);

			// now, each of min and max has 128 values
			// reduce, ignoring values outside the depth range
			//
			int dnum = min_i32(32,depth-id32*32);	// if < 32, we have an odd last slice

			HVX_Vector vminmax_this = hvx_reduce_minmax_h_over_width( vmin0, vmax0, vmin1, vmax1);
			// we now have (~min,max),(~min,max).. for all 32 in the right order.

			// if we are on the last iteration, dnum is 1..31 and only the first
			// 'dnum' of the values are valid. So combine them like this...
			// (no effect when dnum = 32).
			vminmax_this = Q6_V_vlalign_VVR( vminmax_this, minmax_all, 4*(32-dnum));
			minmax_all = Q6_Vh_vmax_VhVh( minmax_all, vminmax_this);

		}// depth loop
		if( find_range)*(HVX_Vector *)thrp->minmax_buf = minmax_all;
	}
}
static void
core_mul_d32_general_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height )
{
	core_mul_d32_hvx_inline( thrp, pout, ptrA, ptrB, height, mode_broadcast_general);
}
static void
core_mul_d32_B11D_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height )
{
	core_mul_d32_hvx_inline( thrp, pout, ptrA, ptrB, height, mode_broadcast_B11D);
}
static void
core_mul_d32_mixed_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height )
{
	core_mul_d32_hvx_inline( thrp, pout, ptrA, ptrB, height, mode_broadcast_mixed);
}
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

static void __attribute__((cold))
addsub_d32_dealloc_info (struct nn_node *self)
{
	struct elementwise_d32_info * info = (struct elementwise_d32_info*)self->opaque;
	if( info != NULL){
		if( info->chunkdescs !=NULL) nn_free(info->chunkdescs);
		nn_free(info);
		self->opaque = NULL;
	}
}

static int addsub_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	int node_type = self->node_type;
	char const * nname = hexagon_nn_op_names[node_type];

	logmsg(nn,2,"Checking %s node 0x%X",nname,self->node_id);
	// Mul doesn't (currently) have the two optional out_min, out_max inputs.
	// (these ranges are enforced in the ctor now)(
	////////////////
	struct elementwise_d32_info *info;
	if( self->opaque !=NULL){
		addsub_d32_dealloc_info(self);
	}

	if ((info = nn_calloc(1,sizeof(struct elementwise_d32_info))) == NULL) {
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

	logmsg(nn,2,"%s node 0x%X check OK",nname,self->node_id);
	return 0;
}
static int __attribute__((cold))
addsub_d32_dtor(struct nn_node *self, struct nn_graph *nn)
{
	 addsub_d32_dealloc_info(self);
	 return node_free_common(self,nn);
}
/////////////////////////////////////////////////////////////////////////////////////////
// Code for QuantizedNeg_8_d32
/////////////////////////////////////////////////////////////////////////////////////////

static int neg_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor const * in_min_tensor = self->inputs[1];
	struct tensor const * in_max_tensor = self->inputs[2];
	struct tensor * out_tensor = self->outputs[0];
	struct tensor * out_min_tensor = self->outputs[1];
	struct tensor * out_max_tensor = self->outputs[2];

	float minval = tensor_get_float(in_min_tensor,0);
	float maxval = tensor_get_float(in_max_tensor,0);

	if( tensor_out_prepare_padded_d32(out_tensor,
			in_tensor->shape.batches,
			in_tensor->shape.height, in_tensor->format.height_pad[0],in_tensor->format.height_pad[1],
			in_tensor->shape.width, in_tensor->format.width_pad[0],in_tensor->format.width_pad[1],
			in_tensor->shape.depth, in_tensor->format.depth_pad[0],in_tensor->format.depth_pad[1],
				NN_TYPE_QUINT8)!= 0){
		return errlog(nn,"output too small");
	}
	// invert range...
	tensor_set_single_float( out_min_tensor, -maxval);
	tensor_set_single_float( out_max_tensor, -minval);
	// now we just want 1's complement.
	int use_hvx = 1;
	int k = tensor_copy_scaled_d32( nn, in_tensor, out_tensor, -32768, 32640, use_hvx, OP_ADD_D32_MAX_THREADS);
	return k;
}


////////////// ADD ///////////////////
struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8_d32_ref = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};

////////////// SUB ///////////////////

struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};


struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8_d32_ref = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};

////////////// MUL ///////////////////

struct nn_node_ops nn_ops_for_QuantizedMul_8x8to8_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedMul_8x8to8_d32_ref = {
	.execute = addsub_d32_execute_common,
	.check = addsub_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
//////////// NEG ///////////////////

struct nn_node_ops nn_ops_for_QuantizedNeg_8_d32 = {
	.execute = neg_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
