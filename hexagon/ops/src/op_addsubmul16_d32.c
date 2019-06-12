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
 * This contains implementations for add16 in d32  (QuantizedAdd_u16_d32)
 *
 * Elementwise add of two tensors.
 * Supports any 'broadcast' situation, except that you can't broadcast A->B
 * in some dimensions and B->A in others.
 *
 * Also in here:
 *   QuantizedSub_u16_d32
 *   QuantizedMul_u16_d32
 *
 */

// 6 inputs:
// 0,1 tensor_A, tensor_B
// 2,3  min_a,   max_a
// 4,5  min_b,   max_b
// 6,7  for setting output range
//
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
struct eltwise16_d32_info;

typedef void  (*core_oper_fp)( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );

typedef int (*setup_scaling_fp)( struct nn_graph *nn, struct nn_node *self);
typedef int (*check_need_rerun_fp)(struct nn_graph *nn, struct eltwise16_d32_info *info, int nthreads);

// NOTE: in this code, inputs 'x' and 'y' refer to the first and second inputs;
// 'a' and 'b' are the two inputs, but 'b' is always the smaller-sized input
// in broadcast situations.
//


struct addsub16_scaling_parms {
	int16_t scalea;				// 'a' scale (always >=0, <= 32767
	int16_t scaleb;				// 'b' scale ( 0..32767; or negative in s16 subtract)
	int16_t rsh;				//  rsh amount 0..15
	int16_t nota;				// [u16] only in u16 op: -1 if b-a, 0 if a+b or a-b
	int16_t notb;				// [u16] only in u16 op: -1 if a-b, 0 if a+b or b-a.
	int32_t delta;				// [u16] .
	int32_t s32a,s32b;			// [u16] 32-bit a,b scales used in vector+scalar
	int32_t delta_revscalar;	// [u16] delta for (scalar)a +/- (vector)b

};


// scaling parms for mult.
// see comments above mul_setup_scaling
struct mul16_scaling_parms {
	int32_t mulscale;			// the 32 bit scale
	uint16_t zeroa,zerob;
	uint16_t zero_out;
	uint32_t preadd;			// add this in the modulo add
	uint32_t postadd;			// add this in the 'average' step.
};


// each of these describes a vertical slice in a particular batch:
// contains pointers, # of rows to do, and how much to prefetch
//
struct eltwise16_chunk_descriptor {
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
struct eltwise16_d32_info {
	//
	uint16_t base_oper_code;	// op_add,op_subtract,op_mul - set in 'check'
	uint16_t strategy_valid;
	// these are only for checking if the conditions changed.
	//
	struct shape x_shape;		// previous shape of the 'x' input
	struct tensor_format x_format;
	void * x_data;

	struct shape y_shape;		// previous shape of the 'y' input
	struct tensor_format y_format;
	void * y_data;
	float xy_in_min[2], xy_in_max[2];	// ranges from inputs

	struct tensor_addressing tinA;	// 'a' input addressing
	struct tensor_addressing tinB;	// 'b' input addressing
	struct tensor_addressing tout;	// output addressing.
	struct shape out_shape;			// the shape of the output (same as 'A' input)

	int compat_flags;			// shape compatibility (broadcast flags).
	int16_t oper_code;			// the operation code (op_add, op_sub, op_subr)
	int16_t swapXY;				// if true, a is x, b is x;  else a is x,b is y.

	struct eltwise16_chunk_descriptor * chunkdescs;	// NULL until allocated
	uint32_t chunkdesc_count;							// number of chunk descriptors
	uint32_t chunkdesc_alloc;							// number we have allocated for.

	// funcs which depend on the operator class
	setup_scaling_fp setup_scaling_funcp;					// set up scaling

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

	// output range
	//
	float out_min, out_max;

	// scaling
	float ab_stepsize[2];		// 'step size' for A and B inputs
	uint16_t ab_zero[2];			// 'zero code' for A and B inputs.
	float out_stepsize;
	uint16_t out_zero;
	union {
		struct addsub16_scaling_parms addsub_scaling;
		struct mul16_scaling_parms mul_scaling;
	};
	// prefetch
	uint16_t pf_wid_a;			// width to prefetch from each 'a' side. multiple of 128, < 64K.
	uint16_t pf_wid_b;			// width to prefetch from each 'b' side.

	//
	// work slicing.
	// we divide the work into batches * height_chunks work units,
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

};
struct elementwise_d32_runstate {
	struct eltwise16_d32_info *info;
	volatile int scratch_idx;	// used to pick up scratch buffer(s)
	volatile int jobno;			// sequences of jobs
	int nthreads;				// number of threads in use
	nn_sem_t done_sem;
	int find_range;				// do we need to find range
};
static int addsub_setup_scaling( struct nn_graph *nn, struct nn_node *self);

static inline int set_addsub_scaling( struct nn_graph * nn,  struct eltwise16_d32_info *info);
static void core_add_d32_general_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_add_d32_B11D_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_add_d32_mixed_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );


static int mul_setup_scaling( struct nn_graph *nn, struct nn_node *self);

static void core_mul_d32_general_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_mul_d32_B11D_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );
static void core_mul_d32_mixed_hvx( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height );

static int make_chunk_descriptors(  struct nn_graph *nn, struct eltwise16_d32_info* info );
// build the 'address' and 'broadcast' portions of the strategy for the current inputs,
// operator is op_add or op_subtract.
//
static int
setup_elementwise_strategy( struct nn_graph *nn, struct nn_node *self)
{
	struct eltwise16_d32_info *info= (struct eltwise16_d32_info*)self->opaque;
	struct tensor const *inX_tensor = self->inputs[0];
	struct tensor const *inY_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	int operator = info->base_oper_code;		// op_add, op_subtract, or op_mul

	info->strategy_valid = 0;
	info->x_shape = inX_tensor->shape;
	info->x_format = inX_tensor->format;
	info->x_data= inX_tensor->data;

	info->y_shape = inY_tensor->shape;
	info->y_format = inY_tensor->format;
	info->y_data= inY_tensor->data;

	if( operator <= op_LASTADDSUB ){
		info->setup_scaling_funcp = addsub_setup_scaling;

	}else{
		info->setup_scaling_funcp = mul_setup_scaling;
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

	info->tinA = tensor_addressing_d32_16b( inX_tensor);
	info->tinB = tensor_addressing_d32_16b( inY_tensor);
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
		NN_TYPE_QUINT16) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,info->out_shape.batches,info->out_shape.height,info->out_shape.width,depth);
	}
	// set up addressing
	info->tout = tensor_addressing_d32_16b( out_tensor);

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

	int bcast_mode;
	{
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
			info->bcast_wd_deltagen = (bcast & compat_broadcast_D)== 0? 0x40404040	// only in W
					:0x7E7E7E7E;	// splat one to all W and D.
		}else{
			// if dout = 1, we're not really broadcasting on D;
			// likewise for w.
			// Better to map these to general when possible.
			if( depth==1) bcast &= ~compat_broadcast_D;
			if( info->out_shape.width ==1) bcast &=~compat_broadcast_W;
			if( (bcast & (compat_broadcast_W|compat_broadcast_D))!=0 ){
				bcast_mode = mode_broadcast_mixed;
				info->bcast_wd_deltagen =
						(bcast & compat_broadcast_W)== 0? 0x3E3E3E3E	// only in D
						:(bcast & compat_broadcast_D)== 0? 0x40404040	// only in W
						:0x7E7E7E7E;	// splat one to all W and D.
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

	info->core_oper_funcp = ((operator<=op_LASTADDSUB)?addsub_corefuncs:mul_corefuncs)[bcast_mode];

	{
		// prefetch width
		// expressed as the amount needed in one row (one d32 slice)
		unsigned pf_wid = (  ((size_t)info->tinA.data & 127)		// padding before
							 + info->out_shape.width*32*2			// actual bytes
							 + 127u) & ~127u;					// round up to mul of 128
		info->pf_wid_a = min_u32( pf_wid, 0xFF80 );			// can't exceed this
		int bwid = (bcast& compat_broadcast_W)?1: info->out_shape.width;
		pf_wid = (  ((size_t)info->tinB.data & 127)		// padding before
				 + bwid*32*2			// actual bytes
				 + 127u) & ~127u;					// round up to mul of 128
		info->pf_wid_b = min_u32( pf_wid, 0xFF80 );			// can't exceed this
	}



	// work out slicing strategy
	int run_height = info->out_shape.height;
	int run_depth = info->out_shape.depth;
	int run_nd32 = info->tout.nd32;
	// can we flatten the nd32 dim into height?
	// - only makes sense if nd32 > 1
	// always possible if height=1; regardless of depth broadcast or not.
	// Otherwise it's possible if:
	//     -'A' strides are clean
	//     - one of
	//           -B height= B depth = 1 (broadcasting one to all depth/height)
	//  		 -B height =out_height, Bdepth = out_depth, B strides 'clean'
	// (by 'clean' strides; height_stride= nd32*d32_stride).
	if( run_nd32 >1 ){
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

	info->strategy_valid = -1;	// still needs scaling info.
	return 0;
}

// make the array of chunk descriptors
// each contains input & output pointers, and a row count.
// Using this array, each 'job' can consist of processing one job, while performing
// a prefetch for a later job.
static int
make_chunk_descriptors(  struct nn_graph *nn, struct eltwise16_d32_info* info )
{
	int height_chunks = info->height_chunks;
	int batches = info->out_shape.batches;
	int compat_flags = info->compat_flags;


	unsigned alloc_descs = (height_chunks* batches + 15u)&~15u;	// alloc for this many

	struct eltwise16_chunk_descriptor *chunkdescs = info->chunkdescs;

	if( info->chunkdesc_alloc < alloc_descs ){	// make it bigger
		void * new_arr = nn_realloc(info->chunkdescs, sizeof( struct eltwise16_chunk_descriptor) * alloc_descs);
		if( new_arr == NULL) return errlog(nn, "alloc fail: chunkdescs");
		info->chunkdescs = chunkdescs = (struct eltwise16_chunk_descriptor*)new_arr;
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
		struct eltwise16_chunk_descriptor const *rdp = &chunkdescs[0];
		struct eltwise16_chunk_descriptor *wrp = &chunkdescs[height_chunks];
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
// setup the scaling for add or subtract
// This needs to be redone if
//   - strategy is rebuilt entirely;
//   - any of the input ranges change
//
static int
addsub_setup_scaling( struct nn_graph *nn, struct nn_node *self)
{
	struct eltwise16_d32_info *info= (struct eltwise16_d32_info*)self->opaque;
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
	float out_min = tensor_get_float( self->inputs[6],0);
	float out_max = tensor_get_float( self->inputs[7],0);
	info->out_min = out_min;
	info->out_max = out_max;
	info->ab_zero[0] = get_qu16_level_size_zero( a_in_min, a_in_max, &info->ab_stepsize[0] );
	info->ab_zero[1] = get_qu16_level_size_zero( b_in_min, b_in_max, &info->ab_stepsize[1] );
	info->out_zero =  get_qu16_level_size_zero( out_min, out_max, &info->out_stepsize );

	int res =set_addsub_scaling( nn, info );
	if( res != 0) return errlog(nn,"scaling failed");
	info->strategy_valid = 1;
	return 0;
}


//
// 16 bit unsigned (asymmetric) add is done as below:
//     uint32_t sop0 = a * scalea + b * scaleb
//     int32_t sop = (sop>>1)	+ delta			// saturating add
//     result = saturate_u16(  (sop >> rsh ))     // (>> done with rounding)
//
//  here:
//    scalea = a_qstep/out_qstep * (1<<(rsh+1))
//    scaleb = b_qstep/out_qstep * (1<<(rsh+1))
//    delta = (zero_out<<rsh) -  (zero_a*scalea - zero_b * scaleb)/2
//
// The multiplies are done by u16*u16; since scalea, scaleb are both <= 32767, no unsigned overflow
// can occur when adding them, but the '+delta' should be done with saturation.
// for subtraction, we take the 1's complement of b before doing the above, so we are really finding
//     sop =  a* scalea - b *scaleb + 65535*scaleb
//  .. and therefore
//    	delta = (zero_out<<rsh) -  zero_a*scale - (65535-zero_b) * scaleb
//

static inline int
set_addsub_scaling(struct nn_graph * nn, struct eltwise16_d32_info *info)
{
	struct addsub16_scaling_parms * osp = &info->addsub_scaling;


	float a_qstep = info->ab_stepsize[0];
	float b_qstep = info->ab_stepsize[1];
	float out_qstep = info->out_stepsize;

	int a_zero = info->ab_zero[0];
	int b_zero = info->ab_zero[1];
	int out_zero = info->out_zero;


	float min_outstep = fmaxf( a_qstep, b_qstep) * (float)(1./8192.);
	if( out_qstep < min_outstep){
		out_qstep = min_outstep;
		info->out_min = (float)(-out_zero)* out_qstep;
		info->out_max = (float)(65536-out_zero)*out_qstep;
		logmsg(nn,0,"NOTE: expanding range to %f .. %f so scaling is feasible\n", info->out_min, info->out_max );
	}
	float a_thru_scale = a_qstep / out_qstep;
	float b_thru_scale = b_qstep / out_qstep;

	int k = -1;

	// the 1.0004 is to prevent e.g. thru_scale = 0.99999 from giving scale = 32768 instead of 16384
	int rsh = 15-max_i32(k,flt_getexp( 1.00004f * fmaxf(a_thru_scale, b_thru_scale)));
	// rsh is 1..15 ( min = 1 due to 'min_outstep' ) or  1..6 for u16
	int scale_a = min_i32(32767, roundf_u32( flt_ldexp( a_thru_scale, rsh )));
	int scale_b = min_i32(32767, roundf_u32( flt_ldexp( b_thru_scale, rsh )));

	int is_sub = info->oper_code == op_subtract;
	int is_rsub = info->oper_code == op_subtract_rev;

	osp->scalea = scale_a;
	osp->scaleb = scale_b;
	rsh+= k;
	osp->rsh = rsh;

	logmsg(nn,3,"ina = [%.6g @ %d]  inb = [ %.6g @ %d]  out= (%f...%f)[ %.6g @ %d]",
		a_qstep,a_zero,  b_qstep,b_zero,
		info->out_min, info->out_max, out_qstep, out_zero);
	logmsg(nn,3,"a * %f;  b* %f; scale_a = %d scale_b = %d  rsh = %d", a_thru_scale, b_thru_scale,
		scale_a, scale_b, rsh );



		// 'through gains' with 18 fractional bits, used by vec+scalar cases
	int sa =  roundf_u32( a_thru_scale * (float)(1<<18) );
	osp->s32a = is_rsub ? -sa: sa;
	int sb = roundf_u32( b_thru_scale * (float)(1<<18) );
	osp->s32b = is_sub? -sb:sb;
	// is a u16 operation...
	osp->nota = is_rsub?-1:0;
	osp->notb = is_sub?-1:0;
	int azero_adj = is_rsub? (65535-a_zero): a_zero;
	int bzero_adj = is_sub? (65535-b_zero): b_zero;
	int32_t delta= (out_zero << rsh) - ( (uint32_t)(azero_adj*scale_a + bzero_adj *scale_b)>>1);
	osp->delta = delta;
	if( is_sub){
		// special delta for (scalar)a - (vector)b case.
		delta= (out_zero << rsh) -( (int32_t)(a_zero*scale_a - b_zero*scale_b)>>1);
	}
	osp->delta_revscalar = delta;
	logmsg(nn,3,"nota = %d; notb = %d; delta = %d\n", osp->nota, osp->notb, (int)osp->delta );
	return 0;

}

///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// For Multiply //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// unsigned 16 mul - fairly messy.
// This could be done as
//     (a[i]-azero)*(b[i]-bzero) * (some scale) + out_zero
//   ... but this is problematic since the product needs 33 bits signed.
//
//   so we start with
//     tsum = (a[i]-azero)*(b[i]-bzero)+ range_bias
//
//   (using partial products, and modulo 2^32, without saturation; and range_bias is just
//    a precalculated value that ensures the whole range fits in i32)
//
//    and then
//        scaled = (tsum * mulfac )>>31
//        result = ( scaled  + postadd )>> 3   (saturated to u16)
//
//        so postadd =  outzero*8 - (range_bias*mulfac >> 31)
//
//    We need to ensure that postadd fits in i32, which means mulfac
//    must be less than (4095/4096); so the overall gain can't exceed
//    4095/32k. 'normal' overall gain is around 1/32k.
//
//    The 'tsum' is done as
//       tsum =  a[i]*b[i] - (a[i]*bzero + b[i]*azero)  + (azero*bzero+range_bias)
//    .. where all of the multiplies are u16*u16->u32, and all the adds are done without saturation.
//
//
// a method of finding range_bias:
//  range_bias =   - ( 2*azero-65535)*(2*bzero-65535)/2
//  (done as signed; this requires some attention, to avoid signed overflow).
//  Or use  - ((2*azero-65535)*(bzero-32768) + (azero-32768))
//   ... which doesn't overflow.
//
//
static int
mul_setup_scaling( struct nn_graph *nn, struct nn_node *self)
{
	struct eltwise16_d32_info *info= (struct eltwise16_d32_info*)self->opaque;
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
	float out_min, out_max;
	int a_zero, b_zero, out_zero;
	info->out_min = out_min = tensor_get_float( self->inputs[6],0);
	info->out_max = out_max = tensor_get_float( self->inputs[7],0);
	info->ab_zero[0] = a_zero =  get_qu16_level_size_zero( a_in_min, a_in_max, &info->ab_stepsize[0] );
	info->ab_zero[1] = b_zero = get_qu16_level_size_zero( b_in_min, b_in_max, &info->ab_stepsize[1] );
	info->out_zero = out_zero = get_qu16_level_size_zero( out_min, out_max, &info->out_stepsize );


	 float ab_step8 = 8.0f*info->ab_stepsize[0] * info->ab_stepsize[1];
	 float out_qstep = info->out_stepsize;

	 float max_scale = 4095./4096.;

	 float scale;
	 if( ab_step8 > out_qstep*max_scale){
		 out_qstep = ab_step8/max_scale;
		 float tmp= out_zero;
		 info->out_min = -out_qstep* tmp;
		 info->out_max = out_qstep*(65536.0f-tmp);
		 logmsg(nn,0,"NOTE: expanding output range to %f .. %f so scaling is feasible\n",
				 info->out_min, info->out_max );
		 scale = max_scale;
	 }else{
		 scale = ab_step8/out_qstep;
	 }
	 info->mul_scaling.mulscale = min_i32(0x7fffffff, roundf_u32( scale * (float)(1u<<31)));

	 info->mul_scaling.zeroa = a_zero;
	 info->mul_scaling.zerob = b_zero;
	 info->mul_scaling.zero_out = out_zero;
	 // u16 mode.
	 int range_bias = (32768-a_zero) - (2*a_zero-65535)*(b_zero-32768);
	 info->mul_scaling.preadd = (uint32_t)range_bias + (uint32_t)a_zero*(uint32_t)b_zero;

	 // scale range-bias using mul_scale
	 int rbscaled = ((int64_t)range_bias * info->mul_scaling.mulscale + (1<<30))>>31;
	 info->mul_scaling.postadd = 8*out_zero - rbscaled;

	/*logmsg(nn,3,"scale for %f * %f -> [%f..%f]:  * %d >> %d; adj_bias = %d, range_bias =%d\n",
			info->ab_stepsize[0], info->ab_stepsize[1], info->out_min, info->out_max,
			info->mul_scaling.post_scale, info->mul_scaling.post_rsh,
			info->mul_scaling.adj_bias, info->mul_scaling.range_bias);
	*/
	info->strategy_valid = 1;
	return 0;
}





///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//
// check strategy for addsub cases. Returns 0 if not valid and needs to be rebuilt, 1 if ok
// If the shapes has not changed, but input ranges have, redo the scaling and return 1.
//
static int elementwise_strategy_valid_check(struct nn_node *self, struct nn_graph *nn,struct eltwise16_d32_info *info)
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

	if(  tensor_get_float(self->inputs[2],0) == info->xy_in_min[0]
	   && tensor_get_float(self->inputs[3],0) == info->xy_in_max[0]
	   && tensor_get_float(self->inputs[4],0) == info->xy_in_min[1]
	   && tensor_get_float(self->inputs[5],0) == info->xy_in_max[1]
	   && tensor_get_float(self->inputs[6],0) == info->out_min
	   && tensor_get_float(self->inputs[7],0) == info->out_max ){
		return 1;	// strategy still good.
	}
	// redo scaling for new ranges
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
	//struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	struct eltwise16_d32_info * __restrict info  = ( struct eltwise16_d32_info *)self->opaque;

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif



	logmsg(nn,2,"%s execute. nid = 0x%X ",hexagon_nn_op_names[self->node_type],(unsigned)self->node_id);

	// is strategy valid?
	//  returns 0 for no, 1 for yes, -1 for error.
	// Note that 'valid_check' may recalc scaling if the
	// strategy is otherwise ok.
	//
	int res = elementwise_strategy_valid_check(self,nn, info);
	if( unlikely(res <=0)){
		if (res < 0)
			return res;			// failed and reported
		res = setup_elementwise_strategy( nn, self );
		if( res == 0){
			res =(info->setup_scaling_funcp)( nn,self);
		}
		if( res !=0 ) return -1;
	}

	// strategy is good to go..

	logmsg(nn,3,"bcast_mode = %d compat = 0x%X: running %d rows of nd32=%d; (depth=%d) batches = %d, height in %d chunks of %d (total %d)\n",
			 (int)info->broadcast_mode, info->compat_flags, info->run_height,info->run_nd32, info->run_depth,
			(int)info->out_shape.batches, info->height_chunks, info->chunk_rows, info->num_work_units );

	if( info->core_oper_funcp == NULL) return errlog(nn,"did not set function");

	{

		int nthreads = min_i32(OP_ADD_D32_MAX_THREADS, info->num_work_units );

		struct elementwise_d32_runstate runstate;
		runstate.info = info;
		runstate.scratch_idx = 0;
		runstate.jobno = 0;
		runstate.nthreads = nthreads;
		nn_sem_init( &runstate.done_sem, 0 );

		for( int i =0; i < nthreads; i++){
			nn_os_work_for_vector(nn,elementwise_d32_worker_func,&runstate);
		}
		nn_sem_wait_n_times( &runstate.done_sem, nthreads);
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
	struct eltwise16_d32_info *info;	// to get scaling parms
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
	struct eltwise16_d32_info *info = rstp->info;
	// trhrix is our thread index, 0..nthreads-1.
	int nthreads =rstp->nthreads;

	int thridx = __sync_fetch_and_add( & rstp->scratch_idx,1);

	struct elementwise_hvx_thrinfo thrinfo;
	thrinfo.info = info;

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


	unsigned pfa_wid = info->pf_wid_a;
	unsigned pfa_stride = info->tinA.d32_stride;
	unsigned pfb_wid = info->pf_wid_b;
	unsigned pfb_stride = info->tinA.d32_stride;
	struct eltwise16_chunk_descriptor const * chunks = info->chunkdescs;
	struct eltwise16_chunk_descriptor const * last_chunk = &chunks[njobs-1];
	struct eltwise16_chunk_descriptor const * pending_chunk = NULL;

	// The logic in the loop here is:
	//   (1) get a chunk from the pool using sync_fetch_and_add; get a pointer
	//     to its desc (this_chunk)
	//   (2) wait for any prefetch to finish
	//   (3) issue l2 prefetch for that chunk
	//   (4) if there's a valid *previous* job( pending_chunk), then do that one.
	//   (5) The pointer obtained in (1) becomes the new 'pending_chunk'
	//  At the end, if there's a pending_chunk, we need to do that.


	while( jobidx = __sync_fetch_and_add(&rstp->jobno,1),  jobidx < njobs){
		struct eltwise16_chunk_descriptor const * this_chunk = &chunks[jobidx];
		struct eltwise16_chunk_descriptor const * next_chunk_maybe = this_chunk + nthreads;

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

	nn_sem_post( &rstp->done_sem);
}


//
// HVX Version
//
//
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////.///// For Add/Sub //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//--------------- inlines---------------


// first stage of 'add16'
static inline HVX_VectorPair addsub_first_stage( HVX_Vector vinA, HVX_Vector vinB, HVX_Vector xorA, HVX_Vector xorB, int32_t kA, int32_t kB )
{
	HVX_Vector modA = Q6_V_vxor_VV(vinA,xorA);
	HVX_Vector modB = Q6_V_vxor_VV(vinB,xorB);
	return  Q6_Wuw_vmpyacc_WuwVuhRuh( Q6_Wuw_vmpy_VuhRuh(modB,kB),modA,kA);
}
// second stage
static inline HVX_Vector addsub_second_stage( HVX_VectorPair first, HVX_Vector vdelta, int rsh)
{
	HVX_Vector sum0 = Q6_Vw_vadd_VwVw_sat( Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(first),1), vdelta );
	HVX_Vector sum1 = Q6_Vw_vadd_VwVw_sat( Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(first),1), vdelta );
#if __HEXAGON_ARCH__ >= 62
	return Q6_Vuh_vasr_VwVwR_rnd_sat( sum1, sum0, rsh );
#else
	return  Q6_V_vxor_VV(Q6_Vh_vasr_VwVwR_rnd_sat( sum1, sum0, rsh ),Q6_V_vsplat_R(0x80008000));
#endif
}
// this is addsub_first_stage decomposed into B-dependent and A-dependent,
// for use when broadcasting from B
static inline HVX_VectorPair addsub_vinB_with_scale( HVX_Vector vinB, HVX_Vector xorB, int32_t kB )
{
	HVX_Vector modB = Q6_V_vxor_VV(vinB,xorB);
	return  Q6_Wuw_vmpy_VuhRuh(modB,kB);
}
static inline HVX_VectorPair addsub_first_stage_111D( HVX_Vector vinA,  HVX_VectorPair bscaled, HVX_Vector xorA, int32_t kA )
{
	HVX_Vector modA = Q6_V_vxor_VV(vinA,xorA);
	return  Q6_Wuw_vmpyacc_WuwVuhRuh( bscaled,modA,kA);
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
	struct eltwise16_d32_info const * info =	thrp->info;
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
	// to all 2 outputs' to be used in the wd_splat vector.
	// D broadcast causes wd_splat to contain a 'splat one to 32' effect as well, which can be combined
	// with the w splat.
	// ** this is not yet implemented. wd_splat should found in planning **.
	int w_bvec_inc = 1;		// # of vecs to bump b ptr to traverse w.
	HVX_Vector wd_splat = Q6_V_vzero();	// only used if is_WorD_bcast
	if( broadcast_mode != mode_broadcast_general){
		// suppress B_d32_stride when broadcasting depth
		if( info->bcast_wd_deltagen & 2){
			inB_d32_stride = 0;
		}
		// disable incrementing B in W direction, in broadcast_mixed if we are broadcasting in w dim.
		// this can determined from bit 6 of nfo->bcast_wd_deltagen
		if( broadcast_mode == mode_broadcast_mixed)
			if(info->bcast_wd_deltagen &0x40 ) w_bvec_inc = 0;
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
	int scale_a = info->addsub_scaling.scalea;
	int scale_b = info->addsub_scaling.scaleb;

	scale_a = Q6_R_combine_RlRl( scale_a, scale_a);
	scale_b = Q6_R_combine_RlRl( scale_b, scale_b);

	int delta =  info->addsub_scaling.delta;
	int final_rsh = info->addsub_scaling.rsh;
	// the v60 version does sat to int16 and then add 0x8000
	// at the end; so it needs a compensated delta.
#if __HEXAGON_ARCH__ < 62
	delta -= 0x8000<<final_rsh;		// shift to 'int16' range
#endif
	HVX_Vector vdelta = Q6_V_vsplat_R(delta);

	int32_t tmp = Q6_R_combine_RlRl( info->addsub_scaling.notb, info->addsub_scaling.nota );
	HVX_Vector xorA = Q6_V_vsplat_R( tmp );
	HVX_Vector xorB = Q6_Vh_vshuffo_VhVh( xorA, xorA);	// 'odd' lanes are xor B
	xorA = Q6_Vh_vshuffe_VhVh( xorA, xorA);	// even lanes are xor A

	int depth = thrp->depth;
	int nd32 = (depth+31)/32u;		// no of slices to do.

	//
	// find # of vector loops in w dimension
	//
	int wpad_bytes = (size_t)ptrA & 0x40;	// 0 or 64
	int wpad_bytes_B = (size_t)ptrB & 0x40;
	int vlalign_val = 0;
	int wlen = width*32*2 + wpad_bytes;		// determines right mask
	int nvecw = (unsigned)(wlen-1)/128;		// # of vecs, -1.


	HVX_Vector const *vinpA_0 = (HVX_Vector const *)( ptrA - wpad_bytes);
	HVX_Vector const *vinpB_0 = NULL;
	HVX_Vector *voutp_0 = (HVX_Vector  *)( pout - wpad_bytes);

	HVX_VectorPair vinBs;
	vinpB_0 = (HVX_Vector const *)( ptrB - wpad_bytes_B);	// aligned pointer
	if( !is_B11D){
		if( is_mixed && w_bvec_inc == 0){					// broadcasting w...
			vlalign_val = 128-wpad_bytes_B;					// vlalign needs to align B to 0
		}else{
			vlalign_val = wpad_bytes - wpad_bytes_B;		// for aligning b input to A
			if( vlalign_val < 0) vinpB_0 ++;				// point to first in-loop load of B.
		}
	}

	for(int id32 = 0; id32 < nd32; id32++){
		HVX_Vector const *vinpA_d = vinpA_0;
		HVX_Vector const *vinpB_d = vinpB_0;
		HVX_Vector *voutp_d = voutp_0;

		if( is_B11D){
			// get the (only) B side value (just 32 bytes needed) and splat to all.
			HVX_Vector vbin = Q6_V_vror_VR( *vinpB_d,wpad_bytes_B);
			vbin = Q6_V_vdelta_VV( vbin, wd_splat);
			vinBs = addsub_vinB_with_scale( vbin, xorB, scale_b);
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
				vtmp = addsub_first_stage_111D( vinA, vinBs, xorA, scale_a );
			}else{
				HVX_Vector vinb = *vinpB++;
				vb_prev = (vlalign_val <0)? vinpB[-2]: vinb;
				vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
				if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
				vtmp = addsub_first_stage( vinA, vbalign, xorA, xorB,  scale_a, scale_b );
				if( w_bvec_inc) vb_prev = vinb;
			}

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				HVX_Vector vout = addsub_second_stage( vtmp, vdelta,  final_rsh);
				HVX_Vector vinA = *vinpA++;
				if( is_B11D){
					vtmp = addsub_first_stage_111D( vinA, vinBs, xorA, scale_a );
				}else{
					if( w_bvec_inc){
						HVX_Vector vinb = *vinpB++;
						vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
						if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
						vb_prev = vinb;
					}
					vtmp = addsub_first_stage( vinA, vbalign, xorA, xorB,  scale_a, scale_b );
				}
				*voutp ++ = vout;
			}
			//
			// last one
			//
			*voutp= addsub_second_stage( vtmp, vdelta,  final_rsh);

			vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
			vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
			voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
		}
		vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
		vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
		voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);
	} // for id32
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


///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////.///// For Multiply /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


//--------------- inlines---------------
static inline HVX_VectorPair
mulu16_stage1( HVX_Vector va, HVX_Vector vb,
		uint32_t zaa, uint32_t zbb, HVX_Vector vpreadd)
{
	// all adds/sub done modulo u32!
	//    preadd + a*b - (a*zb + b*za)
	HVX_VectorPair sumn = Q6_Wuw_vmpy_VuhRuh( va,zbb);	// a * zb
	sumn = Q6_Wuw_vmpyacc_WuwVuhRuh(sumn, vb,zaa);	// + b * za
	HVX_VectorPair sump = Q6_Wuw_vmpy_VuhVuh( va,vb);
	sump = Q6_W_vcombine_VV(
			Q6_Vw_vadd_VwVw( Q6_V_hi_W(sump), vpreadd),
			Q6_Vw_vadd_VwVw( Q6_V_lo_W(sump), vpreadd));
	return Q6_Ww_vsub_WwWw( sump, sumn);
}
static inline HVX_Vector
mulu16_stage2(HVX_VectorPair stage1, HVX_Vector postadd, HVX_Vector mulscale)
{
	HVX_Vector t0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_lo_W(stage1), mulscale);
	HVX_Vector t1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_hi_W(stage1), mulscale);
	t0 = Q6_Vw_vadd_VwVw_sat( t0, postadd);
	t1 = Q6_Vw_vadd_VwVw_sat( t1, postadd);
#if __HEXAGON_ARCH__ < 62
	HVX_Vector t = Q6_Vh_vasr_VwVwR_rnd_sat( t1, t0, 3);
	return Q6_V_vxor_VV( t, Q6_V_vsplat_R(0x80008000));
#else
	return Q6_Vuh_vasr_VwVwR_rnd_sat( t1, t0, 3);
#endif
}

static inline void __attribute__((always_inline))
core_mul_d32_hvx_inline( struct elementwise_hvx_thrinfo const * thrp,
		uint8_t *pout, uint8_t const * ptrA, uint8_t const *ptrB, int height, int broadcast_mode )
{
	struct eltwise16_d32_info const * info =	thrp->info;
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
		if( info->bcast_wd_deltagen & 2){
			inB_d32_stride = 0;
		}
		// disable incrementing B in W direction, in broadcast_mixed if we are broadcasting in w dim.
		// this can determined from bit 6 of nfo->bcast_wd_deltagen
		if( broadcast_mode == mode_broadcast_mixed)
			if(info->bcast_wd_deltagen &0x40 ) w_bvec_inc = 0;
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
	uint32_t zaa = Q6_R_combine_RlRl(info->mul_scaling.zeroa, info->mul_scaling.zeroa);
	uint32_t zbb = Q6_R_combine_RlRl(info->mul_scaling.zerob, info->mul_scaling.zerob);
	int post_add = info->mul_scaling.postadd;
#if __HEXAGON_ARCH__ < 62		// because we need to sat to int16 on v60
	post_add -= 0x8000 <<3;
#endif
	HVX_Vector vpostadd = Q6_V_vsplat_R( post_add);
	HVX_Vector vpreadd = Q6_V_vsplat_R(info->mul_scaling.preadd);
	HVX_Vector vscale = Q6_V_vsplat_R(info->mul_scaling.mulscale);


	int depth = thrp->depth;
	int nd32 = (depth+31)/32u;		// no of slices to do.

	//
	// find # of vector loops in w dimension
	//
	int wpad_bytes = (size_t)ptrA & 0x40;	// 0 or 64
	int wpad_bytes_B = (size_t)ptrB & 0x40;
	int vlalign_val = 0;
	int wlen = width*32*2 + wpad_bytes;		// determines right mask
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
			vtmp = mulu16_stage1( vinA, vbalign, zaa,zbb, vpreadd );

			// loop (zero times or more)
			for( iwd = 0; iwd < nvecw; iwd++ ){
				HVX_Vector vout = mulu16_stage2( vtmp, vpostadd, vscale);
				HVX_Vector vinA = *vinpA++;
				if( is_B11D){
					vbalign = vinBs;
				}else if( w_bvec_inc){
					HVX_Vector vinb = *vinpB++;
					vbalign = Q6_V_vlalign_VVR(vinb, vb_prev,vlalign_val);
					if( is_mixed ) vbalign = Q6_V_vdelta_VV( vbalign, wd_splat);
					vb_prev = vinb;
				}
				vtmp = mulu16_stage1( vinA, vbalign,  zaa,zbb, vpreadd );
				*voutp ++ = vout;
			}
			//
			// last one
			//
			*voutp= mulu16_stage2( vtmp, vpostadd, vscale);

			vinpA_d = vecptrk_add_bytes(vinpA_d, row_stride_A);
			vinpB_d = vecptrk_add_bytes(vinpB_d, row_stride_B);
			voutp_d = vecptr_add_bytes(voutp_d, row_stride_out);
		}
		vinpA_0 = vecptrk_add_bytes(vinpA_0,thrp->inA_d32_stride);
		vinpB_0 = vecptrk_add_bytes(vinpB_0,inB_d32_stride);
		voutp_0 = vecptr_add_bytes(voutp_0,thrp->out_d32_stride);
	} // for id32
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
addsub16_d32_dealloc_info (struct nn_node *self)
{
	struct eltwise16_d32_info * info = (struct eltwise16_d32_info*)self->opaque;
	if( info != NULL){
		if( info->chunkdescs !=NULL) nn_free(info->chunkdescs);
		nn_free(info);
		self->opaque = NULL;
	}
}

static int addsub16_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	int node_type = self->node_type;
	char const * nname = hexagon_nn_op_names[node_type];

	logmsg(nn,2,"Checking %s node 0x%X",nname,self->node_id);

	////////////////
	struct eltwise16_d32_info *info;
	if( self->opaque !=NULL){
		addsub16_d32_dealloc_info(self);
	}

	if ((info = nn_calloc(1,sizeof(struct eltwise16_d32_info))) == NULL) {
		return errlog(nn,"calloc");
	}
	self->opaque = (void*) info;

	{	// who are we?
		int base_oper_code;
		int node_type = self->node_type;
		switch(node_type)
		{
		 case OP_QuantizedAdd_u16_d32:
			 base_oper_code = op_add;
			break;
		 case OP_QuantizedSub_u16_d32:
			 base_oper_code = op_subtract;
			 break;
		 case OP_QuantizedMul_u16_d32:
			 base_oper_code = op_mul;
			 break;
		 default:
			 return errlog(nn,"unexpected node_type = %d", node_type);
		}
		info->base_oper_code = base_oper_code;
	}

	logmsg(nn,2,"%s node 0x%X check OK",nname,self->node_id);
	return 0;
}
static int __attribute__((cold))
addsub16_d32_dtor(struct nn_node *self, struct nn_graph *nn)
{
	 addsub16_d32_dealloc_info(self);
	 return node_free_common(self,nn);
}
#if 0
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
	int k = tensor_copy_scaled_d32( nn, in_tensor, 8p8to8out_tensor, -32768, 32640, use_hvx, OP_ADD_D32_MAX_THREADS);
	return k;
}
#endif


////////////// ADD ///////////////////
struct nn_node_ops nn_ops_for_QuantizedAdd_u16_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub16_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub16_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};

////////////// SUB ///////////////////

struct nn_node_ops nn_ops_for_QuantizedSub_u16_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub16_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub16_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};



////////////// MUL ///////////////////

struct nn_node_ops nn_ops_for_QuantizedMul_u16_d32 = {
	.execute = addsub_d32_execute_common,
	.check = addsub16_d32_check,
	.ctor = node_alloc_common,
	.dtor = addsub16_d32_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};

#if 0
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

#endif
