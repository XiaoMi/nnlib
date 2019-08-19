
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains operations on the graph
 */

#include <nn_broadcast.h>
#include <nn_graph.h>
#include "quantize.h"

#define ROUNDUP(size) ((size + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1))

/* Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data */
// returns 0 when not prepared to use hvx
// returns 1 when prepared to use hvx
// returns 2 when scratch buffer error occured
int nn_check_prepare_hvx_opt(
		struct nn_graph *nn,
		const struct tensor *a_tensor,
		const struct tensor *b_tensor,
		struct tensor *out_tensor,
		const uint8_t *a_data,
		const uint8_t *b_data,
		struct hvx_info *opt_info) {

	uint32_t ab, ah, aw, ad;
	uint32_t bb, bh, bw, bd;
	int opt_flag=0;
	uint8_t *a_data_pad = NULL;
	uint8_t *b_data_pad = NULL;
	int elements, bhw, i;
	int a_const_value = 0;
	int b_const_value = 0;

	tensor_get_shape(a_tensor,&ab,&ah,&aw,&ad);
	tensor_get_shape(b_tensor,&bb,&bh,&bw,&bd);
	//t2 =  nn_os_get_cycles(nn);

	/* One of the operands is scalar */
	if((ab==1)&&(ah==1)&&(aw==1)&&(ad==1))	{
		elements = bb*bh*bw*bd;
		if (nn_scratch_grow(nn, ROUNDUP(elements+ALIGN_SIZE))){ //extra align size for 1 more fetching
			return 2;
		}

		b_data_pad = (uint8_t *)b_data;// since b_data is already aligned to 128, no need to create padded buffer
		a_data_pad = nn->scratch;

		tensor_set_shape(out_tensor,bb,bh,bw,bd);

		opt_flag = 1;
		a_const_value = a_data[0];
	}
	else if((bb==1)&&(bh==1)&&(bw==1)&&(bd==1))	{
		elements = ab*ah*aw*ad;
		if (nn_scratch_grow(nn, ROUNDUP(elements+ALIGN_SIZE))){ //extra align size for 1 more fetching
			return 2;
		}

		a_data_pad = (uint8_t *)a_data;// since a_data is already aligned to 128, no need to create padded buffer
		b_data_pad = (uint8_t *)nn->scratch;

		tensor_set_shape(out_tensor,ab,ah,aw,ad);

		opt_flag = 1;
		b_const_value = b_data[0];
	}

	/* Both operands are of same dimensions */
	else if ((ab==bb)&&(ah==bh)&&(aw==bw)&&(ad==bd))	{
		//logmsg(nn,0,"Entering qmul asm - a=%p, b=%p, elem=%d, elem_pad=%d", a_data_pad, b_data_pad, elements, elements_pad);
		//t3 =  nn_os_get_cycles(nn);
		elements = ab*ah*aw*ad;

		a_data_pad = (uint8_t *)a_data;// since a_data is already aligned to 128, no need to create padded buffer
		b_data_pad = (uint8_t *)b_data;// since b_data is already aligned to 128, no need to create padded buffer

		tensor_set_shape(out_tensor,ab,ah,aw,ad);

		opt_flag = 1;
	}

	/* Depth matches on both operands - Broadcast elements in one operand to match dimensions of other operand */
	else if ((bb==1)&&(bh==1)&&(bw==1)&&(bd==ad))	{

		elements = ab*ah*aw*ad;
		bhw = ab*ah*aw;
		if (nn_scratch_grow(nn, ROUNDUP(elements+ALIGN_SIZE))){ //extra align size for 1 more fetching
			return 2;
		}

		a_data_pad = (uint8_t *)a_data;// since a_data is already aligned to 128, no need to create padded buffer
		b_data_pad = (uint8_t *)nn->scratch;

		// Broadcast elements in b to match a dimensions
		for(i=0;i<bhw;i++) {
			vmemcpy_asm(b_data_pad,b_data,bd);
			b_data_pad += bd;
		}
		b_data_pad = (uint8_t *)nn->scratch;//pad_and_align(a_data_pad, elements_pad);

		tensor_set_shape(out_tensor,ab,ah,aw,ad);

		opt_flag = 1;
		//printf("Depth optimization: a: %lux%lux%lux%lu b: %lux%lux%lux%lu\n", ab,ah,aw,ad, bb,bh,bw,bd);

	}
	else if ((ab==1)&&(ah==1)&&(aw==1)&&(bd==ad))	{

		elements = bb*bh*bw*bd;
		bhw = bb*bh*bw;
		if (nn_scratch_grow(nn, ROUNDUP(elements+ALIGN_SIZE))){ //extra align size for 1 more fetching
			return 2;
		}

		b_data_pad = (uint8_t *)b_data ;// since b_data is already aligned to 128, no need to create padded buffer
		a_data_pad = nn->scratch;

		// Broadcast elements in b to match a dimensions
		for(i=0;i<bhw;i++) {
			vmemcpy_asm(a_data_pad,a_data,ad);
			a_data_pad += ad;
		}
		a_data_pad = nn->scratch;

		tensor_set_shape(out_tensor,bb,bh,bw,bd);

		opt_flag = 1;
		//printf("Depth optimization: a: %lux%lux%lux%lu b: %lux%lux%lux%lu\n", ab,ah,aw,ad, bb,bh,bw,bd);

	}

	if(opt_flag) {
		opt_info->a_data_pad = a_data_pad;
		opt_info->b_data_pad = b_data_pad;
		opt_info->elements = elements;
		opt_info->a_const_value = a_const_value;
		opt_info->b_const_value = b_const_value;
	}

	return opt_flag;
}
#ifdef NEW_BROADCAST
//
// This handles elementwise operations, possibly with broadcast, on arbitrary type & operator.
// Caller must supply pointers to these routines, each of which does 'n' ops:
//
//
//   op_stride_11( T *out, T const *in1, T const * in2 , int n, void *opaque )
//       n times:  *out++ = *in1++ [OP]  *in2++
//
//   op_stride_10(  T *out, T const *in1, T const * in2 , int n, void *opaque )
//         *out++ = *in1++ [OP]  *in2 ,    n_times
//
//   op_rev_stride_01(  T *out, T const *in1 T const * in2 , int n, void *opaque )
//         *out++ = *in2 [OP]  *in1++ ,    n_times
//
//  Note that 'op_rev_stride_01 does *in2 (op) *in1, with a zero stride
//  on in2; if the op commutes, you can use the same function for both op_stride_10.
//  and op_rev_stride_01.
//
//
// strategy:
//   - if shapes are both (b,h,w,d),   make one call to op_stride_11;
//   - if we have  (b,h,w,d) and (1,1,1,1)
//                      make one call to op_stride_01
//   - if we have  (b,w,h,d) and (1,1,1,d)
//         make (b*h*w) calls to op_stride_01
//
//
// There are fields in the "elementwise_funcs" to supply hvx function pointers, and to describe conditions
// under which they may be used in place of the scalar functions. But there is currently no
// support in here for that.
//
// Normally functabp->scratch_elbytes is zero.
// for operations that need to construct a temp intermediate in scratch;
//    - scratch_elbytes must be set to the elementsize in scratch
//    - this function will size the output tensor based functabp->out_typecode.
//      *But* if out_elbytes is 0, this step will be skipped, and no reference is made to self->outputs[0].
//    - this routine will allocate a scratch area of based on the result shape and scratch_elbytes.
//      (caller must do nn_scratch_reset before calling)
//    - this routine will then do the broadcasted op using the supplied functions, and assuming an output
//      elementsize of scratch_elbytes, to the temp area.
//    - IMPORTANT: the 'opaque' pointer must point to a struct with a pointer as its first member. This is where the
//      intermediate buffer address is returned.
//
static void broadcast_work_for_hvx( struct nn_graph * nn, void * rstpv);
typedef void (*inner_loop_funcp)( void *out, void const *in1, void const *in2, int n, void *opaque);
struct bcast_runstate
{
	uint8_t const * inp_a;
	uint8_t const * inp_b;
	uint8_t * outp;
	unsigned effective_dim[4];
	unsigned effective_stride_a[4];
	unsigned effective_stride_b[4];
	unsigned in_elbytes, out_elbytes;
	void *opaque;
	inner_loop_funcp inner_loop_hvx_fp;
	nn_sem_t done_sem;
};
int nn_elementwise_with_broadcast(
	struct nn_node *self,
	struct nn_graph *nn,
	struct elementwise_funcs const * functabp,
	void *intermed_a,
	void *intermed_b,
	void *opaque)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor;
	inner_loop_funcp inner_loop_fp, inner_loop_hvx_fp;
	struct shape outshape;
	int err = 0;

	// compare shapes and develop output shape.
	// Dimensions must match, unless one of them is 1.

	unsigned a_one= 0, b_one = 0;
	unsigned n_all = 1;
	for( int i =0; i < 4; i++){
		int asize = a_tensor->shape.dimension[i];
		int bsize = b_tensor->shape.dimension[i];
		unsigned m = 1<<i;
		if( asize == 1) a_one |= m;
		if( bsize == 1) b_one |= m;
		if( asize != bsize &&  ((a_one|b_one)&m)== 0){
			err = 1;
			break;
		}
		unsigned dimn = max_i32(asize,bsize);
		outshape.dimension[i] = dimn;
		n_all *= dimn;
	}
	if (err) {
		return errlog(nn,"incompatible shapes (%dx%dx%dx%d) (%dx%dx%dx%d)",
			a_tensor->shape.batches,
			a_tensor->shape.height,
			a_tensor->shape.width,
			a_tensor->shape.depth,
			b_tensor->shape.batches,
			b_tensor->shape.height,
			b_tensor->shape.width,
			b_tensor->shape.depth);
	}
	unsigned out_elbytes = functabp->out_elbytes;
	uint8_t * outp = NULL;
	if( out_elbytes !=0 ){
		out_tensor = self->outputs[0];
		if( tensor_out_prepare_normal_fromshape( out_tensor, &outshape, functabp->out_typecode)!= 0)\
			return errlog(nn,"out too small (id=%x): %d > %d",self->node_id,n_all *out_elbytes,out_tensor->max_size);
		outp = (uint8_t*) out_tensor->data;
	}
	if( functabp->scratch_elbytes != 0){	// special handling
		// allocate intermediate scratch of n_all * scratch_elbytes
		out_elbytes = functabp->scratch_elbytes;
		outp = (uint8_t*)nn_scratch_alloc(nn, out_elbytes * n_all);
		if( outp == NULL){
			return errlog(nn,"scratch alloc failed for %d x %d bytes", n_all, out_elbytes);
		}
		*(void**)opaque = (void*)outp;		// save the intermediate result for caller.
	}
	if( outp == NULL) return errlog(nn,"broadcast: no output address!");

	uint8_t const *inp_a = (uint8_t*) a_tensor->data;
	if (intermed_a != NULL){
		inp_a = (uint8_t *) intermed_a;
	}
	uint8_t const *inp_b = (uint8_t*) b_tensor->data;
	if (intermed_b != NULL){
		inp_b = (uint8_t *) intermed_b;
	}

	struct bcast_runstate rst = {
			.inp_a = inp_a,  // a & b may get reversed, below...
			.inp_b = inp_b,
			.outp = outp,
			.effective_dim = { 1,1,1,1},
			.effective_stride_a = {0,0,0,0},
			.effective_stride_b = {0,0,0,0},
			.opaque = opaque,
			.out_elbytes = out_elbytes,
			.in_elbytes = functabp->in_elbytes
	};

	// deal with common special cases, that can each be done with a single call:
	//  a_one == b_one: shapes are identical
	// a_one = 0xF : broadcast from scalar  A
	// b_one = 0xF: broadcast from scalar B
	if( a_one == b_one || max_i32(a_one, b_one) == 0xF){
		if( a_one == b_one){
			inner_loop_fp = functabp->op_stride_11;
			inner_loop_hvx_fp = functabp->op_hvx_stride_11;
		}else if( b_one == 0xF){	// broadcast b->a
			inner_loop_fp = functabp->op_stride_10;
			inner_loop_hvx_fp = functabp->op_hvx_stride_10;
		}else{
			inner_loop_fp = functabp->op_rev_stride_01;
			inner_loop_hvx_fp = functabp->op_hvx_rev_stride_01;
			uint8_t const * tmp = inp_a; inp_a = inp_b; inp_b = tmp;
			rst.inp_a = inp_a;
			rst.inp_b = inp_b;
		}
		if( inner_loop_hvx_fp != NULL && n_all >= functabp->minlen_hvx ){
			rst.inner_loop_hvx_fp = inner_loop_hvx_fp;
			rst.effective_dim[0] = n_all;
			nn_sem_init(&rst.done_sem,0);
			nn_os_work_for_vector( nn,broadcast_work_for_hvx, &rst);
			nn_sem_wait(&rst.done_sem);
		}else{
			(*inner_loop_fp)( outp, inp_a, inp_b, n_all, opaque);
		}
		return 0;
	}
	//
	// now deal with the general case.
	// We have a_one != b_one, and neither are 1111.
	//
	// Try to do it with as few calls as possible, and consolidating loops;
	// and the output will always be generated in sequence.
	//
	//
	// we start by constructing a description of all the loops, starting from the rightmost dim,
	// with all compatible adjacent dims merged. Each loop will have a count, an a_stride and a b_stride;
	// one the of the strides will be zero when broadcast is done on that dim.
	//
	// for instance (1,5,3,2) + (2,5,1,1)  becomes
	//                      n   a_stride    b_stride
	//   inner loop :       2      1          0
	//                      3      2          0
	//                      5      6          1
	//   outerloop:         2      0          5
	// .. but the first two are merged so it becomes
	//                      n   a_stride    b_stride
	//   inner loop :       6      1          0
	//                      5      6          1
	//   outerloop:         2      0          5
	// and then each 'inner loop' becomes a function call.
	//
	//
	unsigned a_strideacc = 1;
	unsigned b_strideacc = 1;
	int eff_loopn = 0;			// number of actual loops identified so far

	// if the inner loop is going to be a broadcast a->b, then reverse a, b
	// so it's broadcast b->a
	// (cheaper to do this now, than later).
	// condition is: if there's a dim with A->B broadcast which is higher
	// than any dim which has either B->A broadcast, or 'flat'; then reverse.
	// Since these are mutually exclusive on a given dim, we can do the
	// test by comparing magnitudes of bitmasks.

	int a_b_reverse = 0;
	if(   (a_one & ~b_one) > (0xF ^ a_one)){
		unsigned t = a_one; a_one = b_one; b_one = t;
		uint8_t const * tp = inp_a; inp_a = inp_b; inp_b = tp;
		a_b_reverse = 1;
		rst.inp_a = inp_a;
		rst.inp_b = inp_b;
	}

	// 'mode' for each dimension coded as:
	//    0   A,B both = 1  	(skip these dims)
	//    1   A=size,  B = 1	(broadcast from B)
	//    2   A =1,   B = size	(broadcast fromn A)
	//    3   A = B = size, > 1  (both flat)
	// after skipping the 'mode 0' dims, each time a dim has the same
	// mode as previous, we can just glom its count onto the previous.

	int prev_mode = 0;

	for( int dimn = 3; dimn >= 0; dimn -- ){
		unsigned m = (1 << dimn);
		int new_mode = ((a_one&m)? 0:1) | ((b_one&m)? 0:2);
		unsigned size = outshape.dimension[dimn];
		if (new_mode != 0){
			if( new_mode == prev_mode){		// no change in mode...
				rst.effective_dim[eff_loopn-1] *= size;	// glom it onto previous loop.
			}else{	// add a loop level
				rst.effective_dim[ eff_loopn] = size;
				rst.effective_stride_a[eff_loopn] = (a_one&m)? 0 : a_strideacc;
				rst.effective_stride_b[eff_loopn] = (b_one&m)? 0 : b_strideacc;
				eff_loopn++;
				prev_mode = new_mode;
			}
			// grow the strides, even if no new loop added.
			if( (a_one&m)== 0 ){
				a_strideacc *= size;
			}
			if( (b_one&m)== 0 ){
				b_strideacc *= size;
			}
		}
	}

#if 0
	printf("a_shape = [ %d %d %d %d]\n",
			(int)a_tensor->shape.batches, (int)a_tensor->shape.height, (int)a_tensor->shape.width, (int)a_tensor->shape.depth );
	printf("b_shape = [ %d %d %d %d]\n",
			(int)b_tensor->shape.batches, (int)b_tensor->shape.height, (int)b_tensor->shape.width, (int)b_tensor->shape.depth );
	printf(" after (rev = %d)\n", a_b_reverse);
	for( int i= 0; i < 4; i++ ){
		printf("  %3d:  %3d   %3d\n", effective_dim[i], effective_stride_a[i], effective_stride_b[i]);
	}
#endif
	// OK, now we should have at least 2 loops
	// (cases which can be done in one, are already dealt with).
	// And the inner loop (first one) should have effective_dim > 1.
	// The inner loop's A_stride must be 1. (broadcast from A should have been flipped).
	// The B stride should be 0 or 1.
	if( eff_loopn <2  || rst.effective_dim[0] < 2 || rst.effective_stride_a[0] != 1 || rst.effective_stride_b[0] >= 2)
		return errlog(nn,"loop analysis failure");
	// figure out what function to use based on the inner loop
	int bcast_inner = 0;
	if( rst.effective_stride_b[0] == 0 ){
		bcast_inner = 1;
		if(a_b_reverse){
			inner_loop_fp = functabp->op_rev_stride_01;
			inner_loop_hvx_fp = functabp->op_hvx_rev_stride_01;
		}else{
			inner_loop_fp = functabp->op_stride_10;
			inner_loop_hvx_fp = functabp->op_hvx_stride_10;
		}
	}else{
		inner_loop_fp = functabp->op_stride_11;	// full elementwise inner loop.
		inner_loop_hvx_fp = functabp->op_hvx_stride_11;	// full elementwise inner loop.
	}


	unsigned in_elbytes = functabp->in_elbytes;
	int inner_n = rst.effective_dim[0];
	int use_hvx = 0;
	if( inner_loop_hvx_fp != NULL 	// can maybe use hvx?
		&& inner_n >= functabp->minlen_hvx ){	// big enough chunks?
		rst.inner_loop_hvx_fp = inner_loop_hvx_fp;
		use_hvx = 1;
		if( functabp->hvx_need_align ){  // pointers must be aligned.
			use_hvx = 0;			// need to insert checking for this.
		}
	}

	// multiply the input strides by in_elbytes
	if(in_elbytes > 1){
		rst.effective_stride_a[3] *= in_elbytes;
		rst.effective_stride_b[3] *= in_elbytes;
		rst.effective_stride_a[2] *= in_elbytes;
		rst.effective_stride_b[2] *= in_elbytes;
		rst.effective_stride_a[1] *= in_elbytes;
		rst.effective_stride_b[1] *= in_elbytes;
	}
	// if using hvx, spawn that loop instead
	if( use_hvx){
		nn_sem_init(&rst.done_sem,0);
		nn_os_work_for_vector( nn,broadcast_work_for_hvx, &rst);
		nn_sem_wait(&rst.done_sem);
	}else{
		int out_bump = inner_n * out_elbytes;
		int n1 = rst.effective_dim[1];
		int n2 = rst.effective_dim[2];
		int a_stride1 = rst.effective_stride_a[1];
		int a_stride2 = rst.effective_stride_a[2];
		int b_stride1 = rst.effective_stride_b[1];
		int b_stride2 = rst.effective_stride_b[2];

		// in most cases the outer loop (or two loops) will run only once.

		for( int i3 = 0; i3 < rst.effective_dim[3]; i3++ ){
			uint8_t const * aptr0 = inp_a + i3 * rst.effective_stride_a[3];
			uint8_t const * bptr0 = inp_b + i3 * rst.effective_stride_b[3];

			for( int i2 = 0; i2 < n2; i2++){
				for( int i1 = 0; i1 < n1; i1++){
					uint8_t const * in_a = aptr0 + a_stride1 * i1 + a_stride2 * i2;
					uint8_t const * in_b = bptr0 + b_stride1 * i1 + b_stride2 * i2;
					(*inner_loop_fp)( outp, in_a, in_b, inner_n, opaque);
					outp += out_bump;
				}
			}
		}
	}
	return 0;
}

// work function for hvx
// does the same thing as the scalar loop, but in an hvx thread, using
// the hvx inner_loop_fp.
//
static void
broadcast_work_for_hvx( struct nn_graph * nn, void * rstpv)
{
	struct bcast_runstate *rstp= (struct bcast_runstate*)rstpv;
	int a_stride1 = rstp->effective_stride_a[1];
	int a_stride2 = rstp->effective_stride_a[2];
	int b_stride1 = rstp->effective_stride_b[1];
	int b_stride2 = rstp->effective_stride_b[2];

	int inner_n = rstp->effective_dim[0];
	int out_bump = inner_n * rstp->out_elbytes;
	int n1 = rstp->effective_dim[1];
	int n2 = rstp->effective_dim[2];
	uint8_t * outp = rstp->outp;
	inner_loop_funcp inner_loop_hvx_fp = rstp->inner_loop_hvx_fp;
	void * opaque = rstp->opaque;
	// in most cases the outer loop (or two loops) will run only once.

	for( int i3 = 0; i3 < rstp->effective_dim[3]; i3++ ){
		uint8_t const * aptr0 = rstp->inp_a + i3 * rstp->effective_stride_a[3];
		uint8_t const * bptr0 = rstp->inp_b + i3 * rstp->effective_stride_b[3];

		for( int i2 = 0; i2 < n2; i2++){
			for( int i1 = 0; i1 < n1; i1++){
				uint8_t const * in_a = aptr0 + a_stride1 * i1 + a_stride2 * i2;
				uint8_t const * in_b = bptr0 + b_stride1 * i1 + b_stride2 * i2;
				(*inner_loop_hvx_fp)( outp, in_a, in_b, inner_n, opaque);
				outp += out_bump;
			}
		}
	}
	nn_sem_post( &rstp->done_sem);
}


#endif



