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

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include "quantize.h"
#include "nn_asm_ops.h"
#define LRN_MAXTHREADS 2

/*
 * LRN:
 * * Input tensor (with min and max)
 * * window_shape
 * * Bias
 * * Alpha
 * * Beta
 * out = in / (bias + alpha * (sum(foreach (input element determined by window_shape)**2)))**beta
 *
 * Note: the first float in the 'window' tensor is presumed to contain a window size; the value
 * is truncated to integer, and 'alpha' is divided by this amount. The value must be >= 1.
 */


//
// there is an 'opaque' block which is allocated to give
// working space; it starts with this header and is 128 aligned.
// It is not allocated until needed, and is reallocated to larger
// size when necessary. Fields (other than work_area_size) added to
// this will be preserved over reallocation.
//
struct lrn_d32_work
{
	uint32_t work_area_size;	// the work area following the header
};

//
// ensure that the work area is at least 'size' bytes, and return pointer
// to the usable part of it.
//
static void *
allocate_work_area( struct nn_node *self, uint32_t size )
{
	const unsigned hdr_offs = (sizeof(struct lrn_d32_work) + 127) &~127u;
	void *currentp = self->opaque;
	size = (size + 127)& ~127u;
	struct lrn_d32_work tmp = {size,};

	if( currentp != NULL ){
		struct lrn_d32_work *wp = (struct lrn_d32_work *)currentp;
		if( wp->work_area_size < size ){
			tmp = *wp;
			tmp.work_area_size = size;
			nn_free(currentp);
			currentp = NULL;
		}
	}
	if( currentp == NULL){
		currentp = nn_memalign(128,hdr_offs + size);
		if( currentp == NULL){
			self->opaque = NULL;
			return NULL;
		}
		*(struct lrn_d32_work *)currentp = tmp;
		self->opaque = currentp;
	}
	return (void*)( (char*) currentp + hdr_offs);
}

// state to be passed to threads. Each thread gets a pointer
// to one of the thrinfo.
//
struct lrn_d32_runstate{
	int batches, height;
	int width;				// including padding
	int depth;				// including padding

	struct tensor_addressing tin;		// addressing for input vector
	struct tensor_addressing tout;

	//int16_t *intermed_buf;	// batches * height * width * depth

	// scaling & parms
	int16_t in_offset;		// the input code (0..255) which represents 0.
	int16_t win_radius;
	uint32_t ikappa;
	int isigma,ibeta;	// scaled parms
	int16_t out_offset;	// for scaling output
	int16_t out_recip;
	uint32_t depth_range;	// sets pre/post depth padding in hvx routine

	volatile int next_batchindex;	// used to share work among threads

	struct lrn_d32_thrinfo {
		struct lrn_d32_runstate *stt;
		//int16_t *minmax_mem;	// buffer to use for minmax
		int32_t *tmp_buffer;
		nn_sem_t done_sem;
	} thrinfo[2];
};

static void run_lrn_oper( struct nn_graph *nn, void *info);


static int lrn_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct lrn_d32_runstate runstate;

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *shape_tensor = self->inputs[3];
	const struct tensor *bias_tensor = self->inputs[4];
	const struct tensor *alpha_tensor = self->inputs[5];
	const struct tensor *beta_tensor = self->inputs[6];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	const float bias = tensor_get_float(bias_tensor,0);
	const float alpha_in = tensor_get_float(alpha_tensor,0);	// subject to /winsize
	const float beta = tensor_get_float(beta_tensor,0);
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);

	// compensate alpha for window size
	const int32_t window_size = (int32_t) tensor_get_float(shape_tensor, 0);
	const float alpha = (window_size <=1)? alpha_in : alpha_in/(float)window_size;

	float in_step = flt_div_255(in_max - in_min);
	int in_zero = roundf_i32( -in_min*255.0f/(in_max-in_min));

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

	int window_depth = shape_tensor->shape.depth;
	if( shape_tensor->shape.width != 1
		|| shape_tensor->shape.height != 1
		|| shape_tensor->shape.batches != 1){
		return errlog( nn, "lrn_d32: can only sum on depth");
	}
	int width_pad_before = in_tensor->format.width_pad[0];
	// construct output tensor, just like input
	//
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		batches,
		height, in_tensor->format.height_pad[0],in_tensor->format.height_pad[1],
		width,  width_pad_before,in_tensor->format.width_pad[1],
		depth, in_tensor->format.depth_pad[0],in_tensor->format.depth_pad[1],
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,batches,height,width,depth);
	}

	// input tensor dimensions
	int width_total = tensor_w_total_d32(in_tensor);	// a multiple of 4
	int depth_total = tensor_d_total_d32(in_tensor);	// a multiple of 32

	// if width_pad_before >=4, we can reduce it by a multiple of 4,
	// and reduce width accordingly.
	//
	int wid_before_excess = width_pad_before & ~3;	// round down to multiple of 4
	width_pad_before -= wid_before_excess;
	width -= wid_before_excess;

	runstate.win_radius = (window_depth-1)>>1;		// e.g. winsize = 5 => radius = 2
	runstate.batches = batches;
	runstate.height = height;
	runstate.width = width_total;
	runstate.depth = depth_total;

	runstate.tin = tensor_addressing_d32( in_tensor);
	runstate.tout = tensor_addressing_d32( out_tensor );
	// for this block we want the pointers to the start of the width padding.
	// (which may have been reduced), so pointers will be aligned.
	runstate.tin.data -= width_pad_before*32;
	runstate.tout.data -= width_pad_before*32;


	// the hvx routine requires common d32 & row strides; we allocated the output
	// the same as input, so this should be ok.
	if( runstate.tout.height_stride != runstate.tin.height_stride
		|| runstate.tout.d32_stride != runstate.tin.d32_stride ){
		return errlog(nn, "lrn_d32 can't support differing row or d32 strides");
	}


	// depth range
	// (parameter for asm):
	//    - lo 16 bits are depth_pad_pefore (0..31)
	//    - hi 16 bits are ( 32 - depth_pad_after)  1..31
	// Except if both are zero, it should be 0 instead of (32 <<16)+0.
	{
		int d_lo = in_tensor->format.depth_pad[0];
		int d_hi = in_tensor->format.depth_pad[1];
		int depth_range = 0;
		if( d_lo != 0 || d_hi != 0 ){
			depth_range += ((32-d_hi)<<16 ) | (uint16_t)d_lo;
		}
		runstate.depth_range = depth_range;
	}
	// the 'work' area consists of:
	//   'tmpbuf' for the asm function  ( 1 per thread )
	// The size is depth_total * 4 * int32_t
	// (the asm code works in chunks of w=4, d=depth)
	//
	int num_threads = min_i32( LRN_MAXTHREADS, batches);

	uint32_t tmp_buffer_size = depth_total * 4 *4;		// per thread
	//uint32_t intermed_buffer_size = batches * height * width_total * depth_total * 2;	// need int16's here
	uint32_t all_work_size= num_threads*(0*2*128 + tmp_buffer_size);
		//+ intermed_buffer_size;

	uint8_t *work_area;
	if(all_work_size <= nn->scratch_size){
		work_area = (uint8_t*)nn->scratch;
	}else{
		work_area = allocate_work_area( self, all_work_size);
		if( work_area == NULL){
			return errlog(nn, "failed to allocate %u bytes", (unsigned)all_work_size);
		}
	}
	// note: the work area is either in 'scratch' or is attached to 'opaque', and will
	// be freed in dtor
	// so no need to free it in here.
	// FIXME: remove all the allocate_work_area stuff; scratch will be large enough.

	// break up to sub sections
	for(int i =0; i < num_threads; i++){
		runstate.thrinfo[i].stt = & runstate;
		//runstate.thrinfo[i].minmax_mem = (int16_t*)work_area;
		//work_area += 2*128;
		runstate.thrinfo[i].tmp_buffer = (int32_t*)work_area;
		work_area += tmp_buffer_size;
	}
	//runstate.intermed_buf = (int16_t*)work_area;

	//
	// now figure out scaling
	//
	// we want to find
	//  ( bias  +  alpha * sum{ (step * q[i])^2} )^beta
	//     (where q[i] = in[i] - in_zero) and 'step' is the input step)
	//
	// This is
	//  (bias  + alpha*step^2  * sum{q[i]^2}  )^beta
	// or
	//  (bias/(alpha*step^2)   +  sum(q[i]^2)  )^beta  * (alpha*step^2)^beta
	//
	// or
	//  (  kappa + sumq(q[i]^2) )^beta  * (alpha*step^2)^beta
	//
	// with kappa = bias/(alpha*step^2)
	//
	// which can be written as
	// pow2(   log2( kappa + sumq(q[i]^2) ) * beta   +     log2(alpha*step^2)*beta );
	//
	// but we want to divide q[i] by this, and also multiply by the step size,
	// so we can multiply q[i] by
	//
	// pow2(   log2( kappa + sumq(q[i]^2) ) * -beta   -    [log2(alpha*step^2)*beta - log2(step)] );
	//
	// =  pow2(   log2( kappa + sumq(q[i]^2) ) * -beta   -   sigma );
	//
	// where sigma = log2(alpha*step^2)*beta - log2(step)
	//
	logmsg(nn,2,"lrn_d32: bias=%f, alpha=%f, beta=%f, windep=%d", bias, alpha, beta, window_depth);
	logmsg(nn,2,"input range = %f .. %f", in_min, in_max);

	// now figure out the output scaling. Largest output values occur when a large input
	// has adjacent zeros. for beta <0.5, the limit can be found as
	///  maxval = maxval/ ( bias + alpha*maxval*maxval)^beta
	///  minval = minval/ ( bias + alpha*minval*minval)^beta
	// if beta > 0.5, the function has a peak at  x = sqrt( bias/(alpha*(2*beta-1)) )
	// so we will evaluate limits there if
	//      x^2 * alpha *(2*beta-1) > bias
	// for x = in_max or in_min.
	//
	float out_max;
	float out_min = 0.0f;
	{
		float peak_point = 9e30f;
		if( beta > 0.5f){
			float a2b =alpha* (2.0f*beta-1.0f);
			int mxlim = fmaxf( in_max, -in_min);
			if(mxlim * mxlim * a2b > bias){	// min or max may be over the peak...
				peak_point = sqrtf(bias/a2b);
			}
		}
		float tmp = fminf(in_max, peak_point);
		out_max = tmp * powf( bias + alpha * tmp*tmp, -beta);
		if( in_min < 0.0f){
			tmp = fmaxf(in_min, -peak_point);
			out_min = tmp * powf( bias + alpha * tmp*tmp, -beta);
			adjust_minmax_for_zero( &out_min, & out_max);
		}
	}
	//
	// Now. We generate the 'out_recip' as 4096*255/(out_max-out_min)
	// and that needs to be < 32k and ideally around 4k so that the final multiply
	// will eat the less reliable lower bits of the previous product.
	// We can do this by defining 'out_scale_fudge, an amount by which we are going
	// to boost the output by adjusting sigma (sigma is subracted before 2^n is done,
	// so subtracting log2(out_scale_fudge) from sigma does the trick).
	// Then the out_recip is 4096*255/((out_max-out_min)*out_scale_fudge)
	// and we can just force it to be 4096.
	//
	//
	float out_scale_fudge = 255.0f/(out_max-out_min);

	float log2_255 = 7.99435344f;
	float log2step = log2f( in_max - in_min)- log2_255;

	float gamma = log2f(alpha) + 2.0f * log2step;
	float sigma =  beta * gamma - log2step - log2f(out_scale_fudge);
	float kappa = bias/(alpha*in_step*in_step);

	logmsg(nn,2," out_fudge = %f", out_scale_fudge);

	runstate.ikappa= (uint32_t)(kappa+0.5f);
	// range-check kappa...
	// it must be possible to add kappa to the largest sum of squares and have the result fit in 32 unsigned.
	//
	int max_sum = max_i32( -in_zero, 255-in_zero);
	max_sum = max_sum*max_sum * runstate.win_radius;	// max sum-of-squares in the calc.
	if ( kappa >= 4294967296.0f || (unsigned long long)runstate.ikappa + max_sum >= ((unsigned long long)1 <<32) ){
		float scale = powf(bias,-beta);
		logmsg(nn,0,"Bias much too large compared with alpha/input.  Converting to copy / scale by %f...",scale);
		tensor_copy(out_tensor,in_tensor);
		tensor_set_single_float(out_min_tensor,in_min*scale);
		tensor_set_single_float(out_max_tensor,in_max*scale);
		return 0;
	}

	runstate.isigma  = roundf_i32( sigma * 32768.0f);
	runstate.ibeta = roundf_i32( beta * -2147483648.0f);

	logmsg(nn,2,"ikappa = %u, isigma=%d, ibeta = %d", runstate.ikappa, runstate.isigma, runstate.ibeta );


	// ok, we now have output range...
	// the zero point should be in range 0..255
	//
	float outrange = (out_max-out_min);
	float oscale = 255.0f/outrange;
	//runstate.out_recip = 4096;	// we forced to it to 4096... roundf_i32( oscale * 4096.0f/out_scale_fudge);
	runstate.out_recip = 4096;	// we forced to it to 4096... roundf_i32( oscale * 4096.0f/out_scale_fudge);
	runstate.out_offset = saturate_u8( roundf_i32( -out_min * oscale));
	runstate.in_offset = in_zero;

	logmsg( nn,2, "output range= %f..%f  recip = %d zero = %d", out_min, out_max, runstate.out_recip, runstate.out_offset);

	runstate.next_batchindex = 0;

	for(int i =0; i < num_threads; i++){
		nn_sem_init( &runstate.thrinfo[i].done_sem, 0);
		nn_os_work_for_vector(nn,run_lrn_oper,&runstate.thrinfo[i]);
	}

	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	for( int i = 0; i < num_threads; i++ ){
		nn_sem_wait(&runstate.thrinfo[i].done_sem);
	}
	logmsg(nn,2,"lrn_d32 %p done",self);
	return 0;
}

static
void
run_lrn_oper( struct nn_graph *nn, void *info)
{
	struct lrn_d32_thrinfo * thrinfo = (struct lrn_d32_thrinfo *)info;
	struct lrn_d32_runstate * runstate = thrinfo->stt;

	uint8_t const * inbuf = runstate->tin.data;
	uint8_t *outbuf = runstate->tout.data;
	int in_batch_stride = runstate->tin.batch_stride;
	int out_batch_stride = runstate->tout.batch_stride;

	int depth = runstate->depth;
	int width = runstate->width;
	int height = runstate->height;
	int win_radius = runstate->win_radius;
	int in_offset = -runstate->in_offset;// asm routine defines these with opposite sign
	int out_offset = -runstate->out_offset;
	int32_t * tmp_buffer = thrinfo->tmp_buffer;

	int ibatch;

	while(  ibatch = __sync_fetch_and_add( &runstate->next_batchindex,1), ibatch < runstate-> batches ){
		lrn_d32_hvx( inbuf + in_batch_stride * ibatch,
				depth,
				in_offset,
				win_radius,
				tmp_buffer,
				outbuf + out_batch_stride * ibatch,
				runstate->ikappa,
				runstate->isigma,
				runstate->ibeta,
				runstate->out_recip,
				out_offset,
				runstate->tin.d32_stride,
				runstate->tin.height_stride,
				width,
				height,
				runstate->depth_range
				);
	}
	nn_sem_post(&thrinfo->done_sem);
}


static int lrn_d32_dtor(struct nn_node *self, struct nn_graph *nn)
{
	void *opq = self->opaque;
	if (opq != NULL) {
		nn_free(opq);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

static int lrn_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking lrn_d32 node %p",self);

	//int k = node_check_inputs_outputs_n( self, nn, "lrn_d32", 7, 3 );
	//if( k != 0 ) return k;

	const int32_t window_size = (int32_t) tensor_get_float(self->inputs[3], 0);
	if (window_size < 1) {
		return errlog(nn, "LRN invalid window size (< 1)"); // int(window_size)>=1 check
	}
	/* Checked in function relative to range... */
	const float bias = tensor_get_float(self->inputs[4], 0);
	if (bias < 1e-6f) {
		return errlog(nn, "LRN unsupported bias-value (< 1e-6)"); // bias> 0 check
	}
	const float alpha = tensor_get_float(self->inputs[5], 0);
	if (alpha <= 0.0f) {
		return errlog(nn, "LRN unsupported alpha-value (<= 0.0)"); // alpha>0 check
	}
	const float beta = tensor_get_float(self->inputs[6], 0);
	if (beta <= 0.0f || beta >= 1.0f) {
		return errlog(nn, "LRN unsupported beta-value (<= 0.0 or >= 1)"); // beta>0 check
	}
	for (uint32_t i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) {
			return errlog(nn,"input %d NULL",i);
		}
	}
	logmsg(nn,2,"lrn_d32 %p check OK",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedLRN_8_d32 = {
	.execute = lrn_d32_execute,
	.check = lrn_d32_check,
	.ctor = node_alloc_common,
	.dtor = lrn_d32_dtor,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
