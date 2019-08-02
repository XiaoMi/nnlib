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
//

/*
 * This contains implementations for quantized concat node
 */
#include <string.h>

#include "hvx_inlines.h"
#include <nn_asm_ops.h>
#include <nn_graph.h>
#include <quantize.h>

//#define CONCAT_REPORT_RUNTIME

// HVX intrinsic code is working (but will crash older compilers...)
// When this is removed, common alignment cases are still handled by
// hvx code in scalemem_d32.S, and weird alignment cases will be done by 'reference' code.
#ifdef HEXAGON_COMPILER_GE_8_0
#define CONCAT_HAS_HVX_INTRIN 1
#endif

// must be >=1.
// Each input is assigned to a thread; each time a thread finishes copying
// an input, it starts on the next pending input (if any).
//
#ifdef HEXAGON_V66
#define CONCAT_MAX_THREADS 4
#else
#define CONCAT_MAX_THREADS 2
#endif
//
// concat operation for d32 format
// Output composed by concatenating all the inputs along a given dimension
//  - input tensors must match output along all the other dims
//  - on the specified dim, the output tensor's size will be the sum of the
//    input tensor dimensions.
//
// Strategy is:
//  - For each input, create a 'tensor slice' view which represents the portion
//     of the object selected by the current slice; this will be the same shape as the input
//  - do a 'tensor copy' from the input to that slice, rescaling if needed.
//
// The 'tensor copy' will usually involve one call per batch to the basic scaled_copy_d32 operation;
// but if the 'depth' is split across multiple chunks in output it will take more.
//
// The 'ref' and 'hvx' are identical except for which scaled_copy_d32 function is used.
//

//=================================================
//
// The basic scaled_copy_d32 does this:
//  given h,w,nd,
// ptr_in, rowstride_in, d32_stride_in,
// ptr_out, rowstride_out,
//  scl_off
//
//
//    - read h*w groups of nd bytes each from the input, with a stride of 32 in the
//        width dimension. and a given row stride in the height dimension (vector aligned).
//    - each byte is scaled according to scl_off (see below)
//   -  nd is in range 1..32; and it is guaranteed that adding nd to ptr_out will not
//     cross a 32 boundary.
//   - Let 'a' be the lower 5 bits of ptr_in.
//     * if nd+a  <= 32, the input values are contiguous (adding nd to ptr_in will not cross
//        a 32 boundary) and d32_stride_in is ignored.
//     * if nd+a  > 32, then 32-a values are taken starting at ptr_in, and the remaining nd+a-32
//       values are taken from the start of the next group (starting at ptr_in + d32_stride-a)
//  scaling is done as  (in[i]*scale + offs*256)/32k (rounded)
//    where scale is a 16-bit signed value (>0) stored in the lower 16 bits of scl_off
//   and offs is a 16-bit signed value stored in the upper 16 bits.

//
// This is the reference version of scaled_copy_d32
//
void
scaled_copy_d32_reference(
		uint8_t * ptr_out, int32_t row_stride_out,
		uint8_t const * ptr_in, int32_t row_stride_in, int32_t d32_stride_in,
		int32_t height,
		int32_t width,
		int32_t  nd,
		int32_t scl_off)	// scale in 16 lsbs, offset in upper 16

{
	int in_align = (size_t) ptr_in % 32;		// input alignment
	// in the reference version, if the input is split, we just call ourselves recursively to do first
	// copy, and then proceed to do the second.
	if( nd + in_align> 32){
		int do_first = 32-in_align;
		scaled_copy_d32_reference(			// first section
				ptr_out, row_stride_out,
				ptr_in, row_stride_in, 0,
				height, width, do_first,
				scl_off);
		// adjust to do the second...
		ptr_in += d32_stride_in - in_align;	// move to next input chunk
		ptr_out += do_first;
		nd -=  do_first;
	}
	// if nd =32, then we can change nd to 32*width and width to 1
	if( nd == 32){
		nd *= width;
		width = 1;
	}
	//
	// ok, now it's a single copy.
	// each op is scaled as ( in[0] * gain + offs + 16k) >>15
	// scl is a positive value with 15 fractional bits (and in range 0..32767)
	// if offset == 0, and scl >= 32704, the scaling is a no-op.
	//
	int32_t offset = scl_off >> 16;				// extract offset
	if( offset == 0 && scl_off >= 32704){		// can just do a copy
		int ih,iw;
		for( ih = 0; ih <  height; ih++){
			uint8_t * rp_out= ptr_out + row_stride_out * ih;
			uint8_t const * rp_in = ptr_in + row_stride_in * ih;
			for( iw = 0; iw < width; iw++){
				memcpy( rp_out + iw*32, rp_in + iw*32, nd );
			}
		}
	}else{		// need to do the actual scaling
		int scale = (int16_t)scl_off;		// extract 16 lsb
		offset = offset*256+ (1<<14);		//  convert, add rounding bias
		int ih,iw,id;
		for( ih = 0; ih <  height; ih++){
			uint8_t * rp_out= ptr_out + row_stride_out * ih;
			uint8_t const * rp_in = ptr_in + row_stride_in * ih;
			for( iw = 0; iw < width; iw++){
				for( id = 0; id < nd; id++){
					int inval = rp_in[id];
					int outval = (inval * scale + offset) >> 15;
					rp_out[id] = saturate_u8( outval);
				}
				rp_out += 32;
				rp_in += 32;
			}
		}
	}
}
//
// 'scalemem_d32' is used in the common case where
//  nd = 32 and input, output are both vec aligned (so d32_stride is not needed).
// the 'reference' version just supplies the extra parms to scaled_copy_d32_reference.
//
static void scalemem_d32_reference(
		uint8_t * ptr_out, int32_t stride_out,
		uint8_t const * ptr_in, int32_t stride_in,
		int32_t height,
		int32_t width,
		int32_t scl_off)
{
	scaled_copy_d32_reference( ptr_out, stride_out, ptr_in, stride_in, 0, height,width, 32, scl_off);
}

#ifdef CONCAT_HAS_HVX_INTRIN
// forward reference to routine using intrinsics
void
scaled_copy_d32_hvx(
		uint8_t * ptr_out, int32_t row_stride_out,
		uint8_t const * ptr_in, int32_t row_stride_in, int32_t d32_stride_in,
		int32_t height,
		int32_t width,
		int32_t  nd,
		int32_t scl_off);	// scale in 16 lsbs, offset in upper 16
#endif




//=====================================================

// The node has an array of these - 1 per input - attached in "concat_info"
struct concat_input_descriptor {
	struct tensor_slice_d32 out_slice;		// a slice of the output.
	float in_min, in_max;					// input range.
	int index;			// used to reorder the inputs, to do the largest first
};

struct concat_info {
	struct concat_input_descriptor *descriptors;
	struct nn_early_work *next_earlywork;	// Work requested by future node
};


// this is 'run-desc' - shared by threads;
// each thread is passed a pointer to one of the elements of thrinfo (which are per-thread)
// and each of those has a pointer to the run_desc.

struct concat_run_desc {
	struct nn_node *self;
	int n_inputs;		// actual number of inputs
	float out_min, out_max;	// output range
	float out_level_recip;
	volatile int input_next;		// input to process next.

	// pointer to the function which does the work.
	void (*scaled_copy_fp)(
			uint8_t *, int32_t,
			uint8_t const * , int32_t , int32_t,
			int32_t, int32_t, int32_t, int32_t );
	void (*scaled_aligned_copy_fp)(
			uint8_t *, int32_t,
			uint8_t const * , int32_t ,
			int32_t, int32_t, int32_t );

	// one of these per thread
	struct concat_run_thread {
		struct concat_run_desc * rundescp;	// points to container.
		nn_sem_t done_sem;
	} thrinfo[CONCAT_MAX_THREADS];

};

static void concat_earlywork_v(struct nn_graph *nn, void *vinfo)
{
	struct concat_info *info = vinfo;
	struct nn_early_work *work = info->next_earlywork;
	if (work == NULL) return;
	if (work->vtcm_addr != nn->vtcm_ptr) return;
	if (work->src_addr == NULL) return;
	if (work->dst_addr == NULL) return;
	if (work->bytes == 0) return;
	nn_graph_memcpy(nn,work->dst_addr,work->src_addr,work->bytes);
	work->valid = 1;
}
static int concat_earlywork(struct nn_graph *nn, void *vinfo)
{
	concat_earlywork_v(nn,vinfo);
	return 0;
}

//
// this is called in thread, does individual input copies.
// It does
//   while ( (input_no = (*rundescp->input_next) ++ , input_no < rundescp->n_inputs) ){
//     ... process input_no ..
//   }
//  except that the ++ is done with a locked add, so multiple threads can call this function.
//
// The 'input_no' thus obtained is a sequence number, and mapped through the 'index' entry
// in indescs to get an actual input #. when the # of inputs is greater than the # of threads,
// this mapping is used to ensure we process the largest inputs first.
//
static void concat_copy_inputs( struct nn_graph *nn, void *thrinfo)
{

	struct concat_run_thread * thrdesc = (struct concat_run_thread *)thrinfo;
	struct concat_run_desc * rundescp = thrdesc->rundescp;
	struct nn_node *self = rundescp->self;
	struct concat_info *info = self->opaque;
	struct concat_input_descriptor const *indescs = info->descriptors;
	int input_seq;

	// get next input to process.
	while( (  input_seq =  __sync_fetch_and_add(&rundescp->input_next,1), input_seq < rundescp->n_inputs) ){
		// map through 'index' to get input #
		int input_select = indescs[input_seq].index;
		// get the input tensor and slice desc for this operation. Note that both have the same shape.
		const struct concat_input_descriptor *indesc =  & indescs[input_select];
		const struct tensor *input_tensor = self->inputs[1+input_select];

		// determine scaling
		float in_level = (indesc->in_max-indesc->in_min)/255.0f;
		int offset = (int) ( (indesc->in_min- rundescp->out_min)*rundescp->out_level_recip * 128.0f);
		int gain = (int) (rundescp->out_level_recip*in_level*32768.0f/*0x1.0p15f*/);
		gain = ((unsigned)gain <= 32767u )? gain: 32767;
		// scale,offs are packed into 32 bits
		int32_t scl_off = gain + (offset <<16);


		int ib;
		int batches = indesc->out_slice.shape.batches;
		unsigned height = indesc->out_slice.shape.height;
		unsigned width  = indesc->out_slice.shape.width;
		unsigned depth = indesc->out_slice.shape.depth;

		int in_batch_stride = tensor_batch_stride_d32(input_tensor);
		int in_row_stride = tensor_row_stride_d32(input_tensor);
		int in_d32_stride = tensor_d32_stride_d32(input_tensor);
		uint8_t const * ptr_in_base = tensor_location_bhw_d32(input_tensor,0,0,0);


		int in_d_offs = input_tensor->format.depth_pad[0];		// # of bytes to offset


		int out_row_stride = indesc->out_slice.height_stride;
		int out_d32_stride = indesc->out_slice.d32_stride;
		unsigned out_d_offs = indesc->out_slice.depth_pad_before;

		// both in & out have the same depth, but they could be different d_offs, and so the total # of 'chunks' needed could
		// be different. Operations are broken up according to the number of slices in the *output* side
		//  Example (each column represents 4):
		//
		//  copy    .....***|********|*.......		in_d_offs = 20, depth= 48
		//  to      .*******|*****...               out_d_offs = 4, depth = 48
		//
		// is done with two depth passes;
		//  first with idpos = 20, odpos = 4, dnow = 28  (12 from 1st input chunk, 16 from second)
		//   2nd  with idpos = 16, odpos = 0, dnow = 20  (16 from 2nd input chunk, 4 from third)
		//

		// @@ note: in some cases it may make sense to transpose height & batch loops,
		// so that the operations in the depth-pass loop are smaller, and sources will remain in the cache.
		// This is just a matter of swapping height & batches, and also the batch_stride and row strides.
		// The condition would be: more than one iteration of the depth loop, and out_d_offs != in_d_offs.
		// This is assuming batches < height to start with, of course.
		// In cases where batches=1 and we want to do this, and height is a multiple of 2 or 4, we can
		// synthesize a split of the height into 'batches'.
		//
		// batch loop.
		uint8_t const *prev_prefetch = NULL;

		for(ib = 0; ib < batches; ib++){
			// get the batch base pointers.
			// these are 32-aligned pointers.
			uint8_t *ptr_out = indesc->out_slice.data + ib * indesc->out_slice.batch_stride;
			uint8_t const *ptr_in = ptr_in_base + ib *in_batch_stride;

			// depth slice loop
			unsigned idpos = in_d_offs;		// current position in input (0..31)
			unsigned odpos = out_d_offs;	// current position in output (0..31)
			unsigned dremain = depth;		// depth remaining to process.
			while(1){
				// determine # to process.
				unsigned dnow = min_u32( dremain, 32-odpos );		// all of it, or all can fit in output.

				// prefetch
				uint8_t const * pin = ptr_in + idpos;
				uint8_t const * pin_pf = pin;
				if( idpos + dnow > 32){	// need the 2nd input data...
					// get 1st if not already.
					if( pin_pf != prev_prefetch) l2pref( pin_pf, height, width*32, in_row_stride );
					pin_pf += in_d32_stride;
				}
				l2pref( pin_pf, height, width*32, in_row_stride );
				prev_prefetch = pin_pf;

				uint8_t * pout = ptr_out + odpos;

				if(  (((size_t)pout | (size_t)pin) &127)== 0 && dnow ==32 ){
					// fully aligned case...
					(*rundescp->scaled_aligned_copy_fp)(
							pout,  out_row_stride,
							pin , in_row_stride,
							height, width, scl_off);

				}else{
					(*rundescp->scaled_copy_fp)(
							pout,  out_row_stride,
							pin , in_row_stride, in_d32_stride ,
							height, width, dnow, scl_off);
				}
				if( dnow >= dremain ) break;			// we are done
				dremain -= dnow;
				// advance to next output group. This is always to a boundary.
				ptr_out += out_d32_stride;
				odpos = 0;			// all subsequent out are at offs 0
				// advance input group. May or may not cross a boundary.
				idpos += dnow;
				if( idpos >= 32){
					idpos -= 32;
					ptr_in += in_d32_stride;
				}
			} // end of depth loop
		} // end of batch loop
	} // end of input loop.


	// signal complete in thread.
	nn_sem_post(&thrdesc->done_sem);
}


// This node takes 'n_in' input tensors and concatenates them on a given dimension;
// the shapes must match on all others.
// It is set up with 3*n_in+1 actual inputs:

//   input
//     0		          -  a single int32 value containing the dimension on which to cat
//     1      ...  n_in   - the actual data inputs
//     n_in+1 ... 2*n_in  - the 'min' values for the inputs (scalar float)
//   2*n_in+1 ... 3*n_in  - the 'max' values for the inputs (scalar float)
//
//  3 outputs:
//     0   - the actual data
//     1   - the 'min' values for the output (scalar float)
//     2   - the 'max' values for the output (scalar float)
//

static int concat_execute(struct nn_node *self, struct nn_graph *nn, int with_hvx)
{
	int n_input_tensors = (self->n_inputs-1)/3;
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	const struct tensor **min_tensors = &self->inputs[1+n_input_tensors];
	const struct tensor **max_tensors = &self->inputs[1+2*n_input_tensors];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	struct concat_info *info = self->opaque;
	struct concat_input_descriptor *indescs = info->descriptors;

	struct shape out_shape;
	int32_t k;
	int32_t i;
	logmsg(nn,2,"concat execute. self=%p ",self);

	int concat_dim = tensor_get_int32(dim_tensor,0);

	// check the dims of all inputs, find the output shape. This also
	// range checks 'concat_dim'.
	//
	k = find_concat_shape( input_tensors, n_input_tensors, concat_dim, &out_shape );
	if( k < 0){
		if( k <= -2) {
			// mismatch size on a particular dim
			return errlog(nn,"mismatch on tensor dim %d, concat on %d", (-2)-k , concat_dim);
		}
		return errlog( nn, "bad concat dim: %d", concat_dim);
	}
	float out_min = tensor_get_float(min_tensors[0],0);
	float out_max = tensor_get_float(max_tensors[0],0);

	// find the combined range of all inputs...
	// and record them in the indescs.
	for (i = 0; i < n_input_tensors; i++) {
		if (!tensor_is_d32(input_tensors[i])) {
			return errlog(nn,"need d32 inputs");
		}
		float in_min = tensor_get_float(min_tensors[i],0);
		float in_max = tensor_get_float(max_tensors[i],0);
		indescs[i].in_min = in_min;
		indescs[i].in_max = in_max;
		indescs[i].index = i;

		out_min = fminf(out_min,in_min);
		out_max = fmaxf(out_max,in_max);
	}
	if (out_min > 0.0f) {
		out_min = 0.0f; // comport with op_quantize use of quantize_adjust_range setting minval = fminf(0.0f,min);
	}else if(out_min < 0.0f ){
		// ensure that repr. of 0 is exact, by adjusting endpoints as needed.
		float xout_min = out_min;
		float xout_max = out_max;
		if( adjust_minmax_for_zero(&xout_min, &xout_max)> 0){
			out_min = xout_min;
			out_max = xout_max;
		}
	}

	tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT);

	// allocate output tensor.
	//
	// (1) determine padding
	int hpad_out = 4;
	int wpad_out_left = 4;
	int dpad_out_before = 0;
	// special cases to avoid excessive padding
	if( out_shape.height == 1 ) hpad_out = 0;
	if( out_shape.width == 1 ) wpad_out_left = 0;

	// (2) inferred padding
	int wpad_out_right = (-(int32_t)(wpad_out_left+out_shape.width))&3;
	int dpad_out_after = (-(int32_t)(dpad_out_before+out_shape.depth))&31;

	if (tensor_out_prepare_padded_d32(
		out_tensor,
		out_shape.batches,
		out_shape.height,hpad_out,hpad_out,
		out_shape.width,wpad_out_left,wpad_out_right,
		out_shape.depth,dpad_out_before, dpad_out_after,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,out_shape.batches,out_shape.height,out_shape.width,out_shape.depth);
	}
	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);

	// now generate all the slice descriptors.
	//
	{ 	int slice_base = 0;
		int res =-1;

		for (i = 0; i < n_input_tensors; i++) {
			// need to clone each subsequent slice from the first before making it
			if( i > 0) indescs[i].out_slice = indescs[0].out_slice;
			// this should not give an error since we've checked all the things.
			// it should return 1 on the last iteration.
			res =  tensor_slice_progressive_d32( & indescs[i].out_slice, // output slice
					out_tensor,								// tensor being sliced
					&input_tensors[i]->shape,			// shape to match
					concat_dim,							// dimension to slice along
					&slice_base);							// keeps track of slice base.
			if( res <0) break;
		}
		if(res != 1) return errlog( nn, "error making slices:%d; i=%d of %d", res, i, n_input_tensors);
	}

	//
	// ok, now the work consists of doing a 'tensor copy' from each of the input tensors to each
	// of the 'out_slices' which are already prepared.
	// Each thread will process an input, and when done, go do the next input; until all
	// are processed. locking primitive is used to keep two threads from processing the same input.
	//
	// Set up the 'rundesc' (which is shared by threads) and the thread info (which is not)
	// Each thrinfo has a pointer to the rundesc.
	struct concat_run_desc rundesc;
	rundesc.self = self;
	rundesc.n_inputs = n_input_tensors;
	rundesc.input_next = 0;
	rundesc.out_min = out_min;
	rundesc.out_max = out_max;
	rundesc.out_level_recip = 255.0f/(out_max-out_min);


	// when ref & hvx are implemented, this will be the only difference.
	if( with_hvx){
#ifdef CONCAT_HAS_HVX_INTRIN
		rundesc.scaled_copy_fp = scaled_copy_d32_hvx;
#else
		rundesc.scaled_copy_fp = scaled_copy_d32_reference;
#endif
		rundesc.scaled_aligned_copy_fp = scalemem_d32_hvx;
	}else{
		rundesc.scaled_copy_fp = scaled_copy_d32_reference;
		rundesc.scaled_aligned_copy_fp = scalemem_d32_reference;
	}

	int num_actual_threads = min_i32( CONCAT_MAX_THREADS, n_input_tensors);

	// if we have more inputs than threads (and more than one thread)
	// use the 'index' field of the input descs to reorder the inputs
	// so that we do the smallest ones last; it's then more likely that
	// the loads will be balanced between the threads.
	//
	if( CONCAT_MAX_THREADS > 1 &&  num_actual_threads < n_input_tensors){	// sort in decreasing order
		uint32_t tmparray[16];
		int i,j;
		int nsort = min_i32(16, n_input_tensors);
		// set each array entry to 'i' in the lower byte, and the dim in the upper 24
		// bits; sort in decreasing order, then unpack the lower bytes to become the index.
		// if there are more than 16 inputs, the excess ones at the front are not sorted.
		int iskip = n_input_tensors - nsort;	//normally 0
		for(  i = 0; i < nsort; i++ ){
			tmparray[i] = i + indescs[iskip+i].out_slice.shape.dimension[concat_dim]*256;
		}
		int ns = nsort;	// prefix remaining to sort.
		// move the smallest one to the end; repeat on shorter list.
		// We only need to do this until the first
		// 'num_threads' are the largest.
		do{
			ns--;
			uint32_t xa = tmparray[ns];
			uint32_t x = xa;
			int xi = ns;
			for( j = 0; j < ns; j++ ){
				uint32_t y = tmparray[j];
				if( y < x){  x = y; xi = j; }
			}
			if( xi < ns){
				tmparray[xi]=xa;
				tmparray[ns]=x;
			}
		}while( ns > num_actual_threads);
		// put them back
		for(i = 0; i < nsort; i++){
			indescs[iskip+i].index = (uint8_t) tmparray[i] + iskip;
		}
	}


	for( i = 0; i < num_actual_threads; i++ ){
		rundesc.thrinfo[i].rundescp = &rundesc;
		nn_sem_init(&rundesc.thrinfo[i].done_sem, 0);
		nn_os_work_for_vector(nn,concat_copy_inputs,&rundesc.thrinfo[i]);
	}
	nn_os_vector_call(nn,concat_earlywork,info);
	for( i = 0; i < num_actual_threads; i++ ){
		nn_sem_wait(&rundesc.thrinfo[i].done_sem);
	}


	logmsg(nn,2,"concat %p done",self);
	return 0;
}

static int concat_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	return concat_execute( self,nn, 0);
}

static int concat_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
#ifdef CONCAT_REPORT_RUNTIME
	uint64_t time_0 = nn_os_get_cycles(nn);
#endif

	int res = concat_execute( self,nn, 1);
#ifdef CONCAT_REPORT_RUNTIME
	if( res == 0){
		uint64_t time_1 = nn_os_get_cycles(nn);
		logmsg(nn,0,"concat_d32 runtime = %llu", (unsigned long long)(time_1 - time_0));
	}
#endif
	return res;
}
static int concat_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking concat_d32 node %p",self);

	// must be 3*n+1 inputs, where n >= 1

	int n_in = (self->n_inputs - 1) /3;	// actual # of inputs
	if (n_in < 1 || (self->n_inputs - 1) % 3 !=0 )
		return errlog(nn,"concat_d32: inputs must be 3*n+1, n>=1");

	struct concat_input_descriptor *indescs = nn_calloc(n_in, sizeof(*indescs));
	if (indescs == NULL) return errlog( nn, "can't allocate input descs");
	struct concat_info *info = nn_calloc(1,sizeof(*info));
	if (info == NULL) {
		nn_free(indescs);
		return errlog(nn,"can't allocate info");
	}
	info->descriptors = indescs;
	self->opaque = info;

	logmsg(nn,2,"concat_d32 node %p check OK",self);
	return 0;
}

static int concat_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct concat_info *info = self->opaque;
	if (info == NULL) return node_free_common(self,nn);
	if (info->descriptors) nn_free(info->descriptors);
	nn_free(info);
	self->opaque = NULL;
	return node_free_common(self,nn);
}

static int concat_earlywork_register(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work)
{
	struct concat_info *info = self->opaque;
	if (info == NULL) return errlog(nn,"Oops: no supernode info available");
	info->next_earlywork = work;
	return 0;
}

// inputs must be 3*k + 1,   k>=1
// IOCOUNT verifies >= 4.
struct nn_node_ops nn_ops_for_QuantizedConcat_8_d32 = {
	.execute = concat_execute_hvx,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = concat_dtor,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
	.earlywork_register = concat_earlywork_register,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};


struct nn_node_ops nn_ops_for_QuantizedConcat_8_d32_ref = {
	.execute = concat_execute_ref,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = concat_dtor,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

#ifdef CONCAT_HAS_HVX_INTRIN


///////////// HVX operator ////////////////////////
// This is only used for the 'general' case; the most common case
//   - depth copy = 32 (full slice)
//   - left padding in width dimension is a multiple of 4
// .. is done by scalemem_d32_hvx which is hand-coded asm


// inlines

//
// This does ub->ub scaling on 128 values.
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
//
static inline HVX_Vector __attribute__((unused))
do_scale_ub( HVX_Vector vin, int scale, int offs )
{
	// (loop invariant)
	HVX_Vector voffs = q6op_Vh_vsplat_R(offs);
	int32_t scale_lo = Q6_R_vsplatb_R( scale );
	int32_t scale_hi = Q6_R_vsplatb_R( scale>>8 );

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
			Q6_Vh_vadd_VhVh_sat(  Q6_V_hi_W( vprod), voffs ),
			Q6_Vh_vadd_VhVh_sat(  Q6_V_lo_W( vprod), voffs ),
			7 );
}

///
/// strategy for the HVX version:
///  If single-source:
//      (1) load input vector, and rotate as needed to align in each 32
//      (2) scale values (if applicable)
//      (3) combine with previous (to account for misaligned width)
//  The store ops need to account for start/end masking in 'width' direction,
//  as well as the lane masks.
//
//
// if double source (inputs gathered from two depth slices):
//	   (1) load two input vectors; combine with mux so that needed lanes are 'wrapped'
//          around each lane; rotate as needed within each group of 32 to match output.
//          E.g.
//               |...........ABCDE|  first vector
//               |FGHIJ...........|  second vector
//               |FGHIJ......ABCDE|   combined with mux  (condition from hvx_make_d32_mask)
//               |.ABCDEFGHIJ.....|   rotated via vdelta  (code from hvx_make_rorcode_in_d32)
//      (2),(3) as before
//
//
//  The 'combine' in step (3) is done with a vlalign; if in/out are aligned, this
// will be an vlalign(0) and will select the current result. The result generated
//   is stored on the next loop iteration; there is always a final store to store
//  the last result in the loop (this could be a short write). The first output is
//  generated before the loop starts, and stored in the first loop iteration
// (or after, if the loop count is zero).
//
// There are four different loops, single/double source; and with/without scaling.
//
//
//
// The general loop (to handle horizontal misalignment), repeated per 'height' iteration:
//
//    // set up prolog mask
//    qmask = qmask_d32 & <<prolog masking>		// disable initial output writes.
//
//    if wskew <0:
//			vL = process( vin[-1] );		// scaling and/or merging, d32 align
//    vR = process( *vin++ );
//    vout = vlalign( vR,vL, wskew) 		// generate next vout
//    repeat (w_vec_iters) times:
//          if( qmask) *vout++  =  vout;		// store vector
//          qmask = qmaskd32;					// eliminate 'prolog' masking if any
//          vL = vR;
//          vR = process( *vin++ );				// scaling and/or merging, d32 align
//          vout = vlalign( vR,vL, wskew) 		// generate next vout
///   // At the end,
//       if extra_right_store:
//          if( qmask) *vout++  =  vout;		// store vector
//          qmask = qmaskd32;					// eliminate 'prolog' masking if any
//          vout = vlalign( vR,vR, wskew) 		// generate next vout
//       // final store:
//       // note that qmask may still contain prolog mask (if w_vec_iters, extra_right_store both 0)
//       qmask &= <<epilog_masking>>
//       if( qmask) *vout++  =  vout;		// store vector
//
//
//
//
//  wskew = w_off_out - w_off_in       <-this is -96, -64 ...   64, 96
//   it is used for the valign, and if <0, we do an extra read at the start.
//    (vinp is adjusted to skip that extra read).
//    In fact the first few ops need to be done as below...
//
//    vL=vR = process( *vin );
//    if wskew <0:
//			vL = process( vin[-1] );
//    vin++;
//    vout = vlalign( vR,vL, wskew) 		// generate next vout
//  ... since we want vR = vL when vskew >= 0, to handle a special short-line case.
//
void
scaled_copy_d32_hvx(
		uint8_t * ptr_out, int32_t row_stride_out,
		uint8_t const * ptr_in, int32_t row_stride_in, int32_t d32_stride_in,
		int32_t height,
		int32_t width,
		int32_t  nd,
		int32_t scl_off)	// scale in 16 lsbs, offset in upper 16
{
	/*printf("[%d x %d]: out %p [%d], in %p [%d:%d] nd =%d scl_off = %x\n",
			(int)height, (int)width, ptr_out, (int) row_stride_out,
			ptr_in, (int)row_stride_in, (int)d32_stride_in,
			(int)nd, (unsigned)scl_off);
	*/

	int offs = scl_off >> 16;
	int scale = (int16_t)scl_off;
	// if needs_scale = 0, then  'scaling' is an identity operation.
	int needs_scale = (offs!=0 || scale < 32704)?1:0;


	// offsets in depth dimension
	// caller guarantees that d_off_out + nd <=32, but not for d_off_in.
	//
	int d_off_in = (size_t) ptr_in & 31;
	int d_off_out = (size_t) ptr_out & 31;
	// offsets in width direction :  (0,1,2,3) *32

	int w_off_in = (size_t) ptr_in & (128-32);
	int w_off_out = (size_t) ptr_out &(128-32);
	// align pointers
	HVX_Vector const  *vinp0 = (HVX_Vector const *)(  (size_t) ptr_in & ~(size_t)127);
	HVX_Vector  *voutp0 = (HVX_Vector  *)(  (size_t) ptr_out & ~(size_t)127);

	// make general write mask (selects the lanes within each d32)
	HVX_VectorPred qmaskd32 = hvx_make_d32_range_mask( d_off_out, d_off_out + nd);
	/*printf("%p + %d + %d   <-  %p  + %d + %d\n", voutp0, w_off_out, d_off_out,
		vinp0, w_off_in, d_off_in); */
	//
	int wskew = w_off_out - w_off_in;
	// find loop count
	// # of loops required to read whole row, and then subtract 1 if wskew < 0
	// (will need to reduce by 1 later)
	int wloopcnt = (unsigned)( w_off_in + 32*width + 3*32)/128u  -  ((wskew <0) ? 1: 0);
	//
	// extra_right_store if output line is longer than wloopcount
	int extra_right_store = ( w_off_out + 32* width ) > wloopcnt*128;
	int line_end_pos = w_off_out + 32* width;
	//  now subtract 1 from wloopcount to get the actual count (1 value is read
	// prior to loop).
	// Result will be >= 0 except in a handful of small cases, which give -1:
	// -  entire input fits in one word
	// -  w_off_out < w_off_in  ( values need to move left)
	// width can only be 1,2, or 3 with these constraints. To handle these,
	//    - change loop count to 0
	//    - add 128 to wskew (eliminates extra load at start)
	//    - clear extra_right_store (which is always 1 in these situations)
	//    - the first valign will have both inputs the same and will move the data to the left.
	//
	wloopcnt -= 1;
	if( wloopcnt < 0 ){
		wloopcnt = 0;
		wskew += 128;
		extra_right_store = 0;
	}
	if( wskew < 0) vinp0 ++;		//  adjust pointer  ('first' will be picked up at [-1])
	//printf("wloopcnt = %d; wskew = %d; ers = %d; lep = %d\n", wloopcnt, wskew, extra_right_store, line_end_pos);
	int irow,j;
	HVX_Vector const * __restrict vinp;
	HVX_Vector *  __restrict  voutp;


	// 'single source' case is more common.
	//
	if( d_off_in + nd <= 32)
	{
		int vrorn = d_off_in - d_off_out;	// vror needed to align input to output.
		// two different loops, according to whether scaling is actually needed
		if( needs_scale){
			for( irow = 0; irow < height; irow ++){
				vinp = (HVX_Vector const*)( (char const*)vinp0 +  irow*row_stride_in );
				voutp = (HVX_Vector *)( (char*)voutp0 +  irow*row_stride_out );

				// make the 'initial' write mask. After first store in loop, is set to qmaskd32.

				HVX_VectorPred qmask = Q6_Q_and_QQn( qmaskd32, Q6_Q_vsetq_R(w_off_out));
				// first load
				HVX_Vector vR = do_scale_ub( Q6_V_vror_VR( vinp[0], vrorn), scale,offs);
				HVX_Vector vL = vR;
				if( wskew < 0 ){
					vL = do_scale_ub( Q6_V_vror_VR( vinp[-1], vrorn), scale,offs);
				}
				vinp++;
				HVX_Vector vout = Q6_V_vlalign_VVR( vR, vL, wskew );
				for( j = 0; j < wloopcnt; j++ ){
					q6op_vstcc_QAV( qmask, voutp++, vout);
					qmask =  qmaskd32;
					vL = vR;
					vR = do_scale_ub( Q6_V_vror_VR( *vinp++, vrorn), scale,offs);
					vout = Q6_V_vlalign_VVR( vR, vL, wskew );
				}
				if( extra_right_store ){
					q6op_vstcc_QAV( qmask, voutp++, vout);
					qmask =  qmaskd32;
					vout = Q6_V_vlalign_VVR( vR, vR, wskew );
				}
				// final store ... first 'and' the current mask to trim the end.
				// (this may also be the first store).
				qmask = Q6_Q_and_QQ( qmask, q6op_Q_vsetq2_R( line_end_pos));
				q6op_vstcc_QAV( qmask, voutp, vout);
			} // for irow
		}else{ // !needs_scale
			for( irow = 0; irow < height; irow ++){
				vinp = (HVX_Vector const*)( (char const*)vinp0 +  irow*row_stride_in );
				voutp = (HVX_Vector *)( (char*)voutp0 +  irow*row_stride_out );

				// make the 'initial' write mask. After first store in loop, is set to qmaskd32.

				HVX_VectorPred qmask = Q6_Q_and_QQn( qmaskd32, Q6_Q_vsetq_R(w_off_out));
				// first load
				HVX_Vector vR = Q6_V_vror_VR( vinp[0], vrorn);
				HVX_Vector vL = vR;
				if( wskew < 0 ){
					vL = Q6_V_vror_VR( vinp[-1], vrorn);
				}
				vinp++;
				HVX_Vector vout = Q6_V_vlalign_VVR( vR, vL, wskew );
				for( j = 0; j < wloopcnt; j++ ){
					q6op_vstcc_QAV( qmask, voutp++, vout);
					qmask =  qmaskd32;
					vL = vR;
					vR = Q6_V_vror_VR( *vinp++, vrorn);
					vout = Q6_V_vlalign_VVR( vR, vL, wskew );
				}
				if( extra_right_store ){
					q6op_vstcc_QAV( qmask, voutp++, vout);
					qmask =  qmaskd32;
					vout = Q6_V_vlalign_VVR( vR, vR, wskew );
				}
				// final store ... first 'and' the current mask to trim the end.
				// (this may also be the first store).
				qmask = Q6_Q_and_QQ( qmask, q6op_Q_vsetq2_R( line_end_pos));
				q6op_vstcc_QAV( qmask, voutp, vout);
			}// for irow
		}
	}else{

		// double source case
		// This is less common, so the 'scaling not needed' test is applied only for the inner
		// loop.
		HVX_Vector vrotcode = hvx_make_rorcode_in_d32( d_off_in - d_off_out);
		HVX_VectorPred qcombine = hvx_make_d32_mask(d_off_in );  // selects lanes from the second input

		for( irow = 0; irow < height; irow ++){
			vinp = (HVX_Vector const*)( (char const*)vinp0 +  irow*row_stride_in );
			voutp = (HVX_Vector *)( (char*)voutp0 +  irow*row_stride_out );
			HVX_Vector const  *vinp1 = (HVX_Vector const *)(  (size_t) vinp  + d32_stride_in);	// next section.

			// make the 'initial' write mask. After first store in loop, is set to qmaskd32.

			HVX_VectorPred qmask = Q6_Q_and_QQn( qmaskd32, Q6_Q_vsetq_R(w_off_out));
			// first load
			HVX_Vector vload = Q6_V_vmux_QVV( qcombine, vinp1[0], vinp[0]);
			HVX_Vector vR = do_scale_ub( Q6_V_vdelta_VV( vload,vrotcode), scale,offs);
			HVX_Vector vL = vR;
			if( wskew < 0 ){
				vload = Q6_V_vmux_QVV( qcombine, vinp1[-1], vinp[-1]);
				vL = do_scale_ub( Q6_V_vdelta_VV( vload,vrotcode), scale,offs);
			}
			vinp++;
			vinp1++;
			HVX_Vector vout = Q6_V_vlalign_VVR( vR, vL, wskew );

			if( wloopcnt > 0){
				if( needs_scale){
					for( j = 0; j < wloopcnt; j++ ){
						q6op_vstcc_QAV( qmask, voutp++, vout);
						qmask =  qmaskd32;
						vL = vR;
						vload = Q6_V_vmux_QVV( qcombine, *vinp1++, *vinp++);
						vR = do_scale_ub( Q6_V_vdelta_VV( vload,vrotcode), scale,offs);
						vout = Q6_V_vlalign_VVR( vR, vL, wskew );
					}
				}else{	// same inner loop w/o scaling
					for( j = 0; j < wloopcnt; j++ ){
						q6op_vstcc_QAV( qmask, voutp++, vout);
						qmask =  qmaskd32;
						vL = vR;
						vload = Q6_V_vmux_QVV( qcombine, *vinp1++, *vinp++);
						vR = Q6_V_vdelta_VV( vload,vrotcode);
						vout = Q6_V_vlalign_VVR( vR, vL, wskew );
					}
				}
			}
			if( extra_right_store ){
				q6op_vstcc_QAV( qmask, voutp++, vout);
				qmask =  qmaskd32;
				vout = Q6_V_vlalign_VVR( vR, vR, wskew );
			}
			// final store ... first 'and' the current mask to trim the end.
			// (this may also be the first store).
			qmask = Q6_Q_and_QQ( qmask, q6op_Q_vsetq2_R( line_end_pos));
			q6op_vstcc_QAV( qmask, voutp, vout);
		}
	}
}

#endif
