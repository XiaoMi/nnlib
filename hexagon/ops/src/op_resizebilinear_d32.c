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

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include "hvx_inlines.h"
#include "quantize.h"
#include "nn_asm_ops.h"
#define LRN_MAXTHREADS 2

#if defined(__hexagon__)
#define min(a, b) (((a)<(b)) ? (a) : (b))
#endif

//#define TEST_PERFORMANCE
/*
 * resize_bilinear
 * -  Input tensor (with min and max)
 * - new_dims tensor (with height and width )
 */
struct bilin_runstate;
typedef void (*run_slice_fp)( uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height );

struct bilin_runstate {
	struct tensor_addressing tin;
	struct tensor_addressing tout;

	int depth;
	int batches;

	int in_ht;		// input height, width
	int in_width;
	int out_ht;		// output size
	int out_width;

	run_slice_fp run_slice_func;
	int with_hvx;

	nn_sem_t done_sem;
	int jobs;		// total # of 'subsections' to run
	volatile int next_job;	 // index of next subsection
};

static void bilin_interp_work(struct nn_graph *nn, void *rstpv);
static void do_bilin_interp_x_single_slice(uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height);
static void do_bilin_interp_x2_single_slice(uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height);
void do_bilin_interp_x2_single_slice_HVX(uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height);


static int resize_bilinear_d32_execute(struct nn_node *self, struct nn_graph *nn)
{

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *newdims_tensor = self->inputs[3];

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	logmsg(nn,2,"lrn_d32 execute. node=%p id=%x",self,self->node_id);

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

	// get new dims
	int new_height = tensor_get_int32( newdims_tensor, 0);
	int new_width = tensor_get_int32( newdims_tensor, 1);

	int width_pad_before = in_tensor->format.width_pad[0];
	int out_width_pad_before = 4;
	int out_width_pad_after = (-(out_width_pad_before + new_width))&3;

	// construct output tensor, just like input
	//
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		batches,
		new_height, in_tensor->format.height_pad[0],in_tensor->format.height_pad[1],
		new_width,  out_width_pad_before,out_width_pad_after,
		depth, in_tensor->format.depth_pad[0],in_tensor->format.depth_pad[1],
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,batches,height,width,depth);
	}

	struct bilin_runstate runstate;
	runstate.tin = tensor_addressing_d32(in_tensor );
	runstate.tout = tensor_addressing_d32(out_tensor );
	runstate.in_ht = height;
	runstate.in_width = width;
	runstate.out_ht = new_height;
	runstate.out_width = new_width;
	runstate.batches = batches;
	runstate.depth = depth;
	runstate.jobs = batches * runstate.tin.nd32;
	runstate.next_job = 0;
	runstate.run_slice_func = do_bilin_interp_x2_single_slice_HVX;
	runstate.with_hvx = 1;

	// we require aligned vectors for hvx. the output is always aligned, with pad_before = 4;
	// If the input padding is multiple of 4; we're ok; if it's an odd multiple of 2,
	// we can fix by expanding the input on the left by 2, and the output by 4.
	//

	if (new_height != height * 2
		|| new_width != width * 2) {
		runstate.run_slice_func = do_bilin_interp_x_single_slice;	// use float code
		runstate.with_hvx = 0;
		logmsg(nn, 2, "resizebilinear d32 scalar.");
	}
	else if( (width_pad_before&1) != 0 ){
		runstate.run_slice_func = do_bilin_interp_x2_single_slice;	// use scalar code
		runstate.with_hvx = 0;
	}else if((width_pad_before&2) != 0 ){
		runstate.in_width += 2;
		runstate.tin.data -= 64;	// include 2 pixels in 'gutter'
		runstate.out_width += 4;
		runstate.tout.data -= 128;	// put 4 pixels in 'gutter'.
	}

	int n_threads = min_i32(2, runstate.jobs);
	nn_sem_init( &runstate.done_sem,0);
	for(int i =0; i < n_threads; i++)
		nn_os_work_for_vector( nn, bilin_interp_work, &runstate );

	// copy the min and max through
	tensor_copy( out_min_tensor, in_min_tensor );
	tensor_copy( out_max_tensor, in_max_tensor );
	for(int i =0; i < n_threads; i++)
		nn_sem_wait( &runstate.done_sem);

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("resizebilinear_d32  cycles = %d (elements = %d) w=(%d->%d) h=(%d->%d)\n",
		(end_time-start_time),	(int)tensor_element_count(out_tensor), width, new_width, height, new_height);
#endif

	logmsg(nn,2,"lrn_d32 %p done",self);
	return 0;
}

static void
bilin_interp_work( struct nn_graph *nn, void *rstpv )
{
	struct bilin_runstate *rstp = (struct bilin_runstate *)rstpv;

	int nd32 = rstp->tin.nd32;				// # of depth slices to do
	///int depth = rstp->depth + rstp->tin.d0;		// include any pre-padding;

	run_slice_fp run_slice_func = rstp->run_slice_func;

	int job_idx;

	uint32_t pf_stride = rstp->tin.height_stride;
	uint32_t pf_width = rstp->in_width * 32;
	uint32_t pf_height = rstp->in_ht;
	uint32_t in_width = rstp->in_width;
	uint32_t out_width = rstp->out_width;
	uint32_t out_height = rstp->out_ht;
	int in_row_stride = rstp->tin.height_stride;
	int out_row_stride = rstp->tout.height_stride;


	while( job_idx = __sync_fetch_and_add(&rstp->next_job, 1), job_idx < rstp->jobs) {
		int batch_idx = 0;
		int d32_idx = job_idx;
		if( d32_idx >= nd32){
			batch_idx = (unsigned)job_idx/nd32;
			d32_idx -= batch_idx* nd32;
		}
		// ok, get pointers;
		uint8_t const *in_ptr = rstp->tin.data  + batch_idx * rstp->tin.batch_stride  + d32_idx * rstp->tin.d32_stride;
		l2fetch( in_ptr, pf_stride, pf_width, pf_height);

		uint8_t * out_ptr =     rstp->tout.data + batch_idx * rstp->tout.batch_stride + d32_idx * rstp->tout.d32_stride;

		(*run_slice_func)(in_ptr, in_row_stride, in_width, pf_height, out_ptr, out_row_stride, out_width, out_height);

	}
	nn_sem_post( &rstp->done_sem);
}


// 'reference' single slice.
// Note, pointers here are x32 aligned but not generally 128 aligned
//
static void
do_bilin_interp_x_single_slice( uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height )
{
	int xstep = (in_width << 16) / out_width;
	int ystep = (in_height << 16) / out_height;

	for (int irow = 0, ycoord = 0; irow < out_height; irow++, ycoord += ystep) {
		int y0 = (ycoord >> 16);
		int y1 = min(y0 + 1, in_height - 1);
		y0 *= in_row_stride;
		y1 *= in_row_stride;
		int yfrac0 = (ycoord >> 8) & 0xff;
		for (int icol = 0, xcoord = 0; icol < out_width; icol++, xcoord += xstep) {
			int x0 = xcoord >> 16;
			int x1 = min(x0 + 1, in_width - 1);
			int xfrac0 = (xcoord >> 8) & 0xff;

			for (int d = 0; d < 32; d++) {
				unsigned char v00 = in_ptr[y0 + x0 * 32 + d];
				unsigned char v01 = in_ptr[y0 + x1 * 32 + d];
				unsigned char v10 = in_ptr[y1 + x0 * 32 + d];
				unsigned char v11 = in_ptr[y1 + x1 * 32 + d];
				unsigned short v0 = (v00 + v00 * (255-xfrac0) + v01 * xfrac0);
				unsigned short v1 = (v10 + v10 * (255 - xfrac0) + v11 * xfrac0);
				out_ptr[irow*out_row_stride + icol * 32 + d] = min(255, (v0 + v0 * (255 - yfrac0) + v1 * yfrac0 + (1<<15)) >> 16);
			}
		}
	}
}


static void
do_bilin_interp_x2_single_slice( uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height )
{
	int64_t * prev_outp = NULL;

	for( int  irow = 0; irow < in_height; irow++ ){
		int64_t *outp = (int64_t *)(out_ptr + out_row_stride * irow*2);
		// interpolate horizontally
		// only process 2 x uint64 (16 pixels) per pass.
		//
		int64_t const *inp = (int64_t const *)(in_ptr + in_row_stride * irow);
		for( int k = 0; k < 2; k ++){
			int64_t a0 = inp[2*k];
			int64_t a1 = inp[2*k+1];
			for(int j = 0; j < in_width-1; j++){
				outp[8*j + 2*k]= a0;
				outp[8*j + 2*k + 1]= a1;
				int64_t b0 = inp[4*j + 2*k + 4];		// get next input
				int64_t b1 = inp[4*j + 2*k + 5];
				outp[8*j + 2*k + 4]= Q6_P_vavgub_PP( a0, b0);
				outp[8*j + 2*k + 5]= Q6_P_vavgub_PP( a1, b1);
				a0 = b0;
				a1 = b1;
			}
			outp[in_width*8 -8 +  2*k]= a0;			// last input to 2nd last output
			outp[in_width*8 -8 + 2*k + 1]= a1;
			outp[in_width*8 -4 +  2*k]= a0;			// andt to last as well.
			outp[in_width*8 -4 + 2*k + 1]= a1;
		}
		if( irow >= 1){
			int64_t  *op1 = (int64_t *)(out_ptr + out_row_stride *(irow*2-1));
			// irow = 1 .. in_ht-1
			// prev_outp = output irow*2-2
			// outp = output irow*2
			// op1 = output irow*2-1
			// interpolate vertically between surrounding two rows.
			// (rounding down in the h, up in the v, to balance the bias)

			for(int j = 0; j < in_width*4; j++){
				int64_t x0 = Q6_P_vavgub_PP_rnd( prev_outp[2*j], outp[2*j]);
				int64_t x1 = Q6_P_vavgub_PP_rnd( prev_outp[2*j+1], outp[2*j+1]);
				op1[2*j] = x0;
				op1[2*j+1] = x1;
			}
		}
		prev_outp= outp;
	}
	// prev_outp points to the last (even) output row.
	int64_t *outp_last = (int64_t *)(out_ptr + out_row_stride * (in_height*2-1));
	// make klockwork happy
	if (prev_outp) memcpy(outp_last, prev_outp, in_width*2 * 32 );	// last row same as previous)
}

static int resize_bilinear_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking resize_bilinear_d32 node %p",self);

	int k = node_check_inputs_outputs_n( self, nn, "lrn_d32", 4, 3 );
	if( k != 0 ) return k;
	logmsg(nn,2,"resize_bilinear_d32 %p check OK",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedResizeBilinear_8_d32 = {
	.execute = resize_bilinear_d32_execute,
	.check = resize_bilinear_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
