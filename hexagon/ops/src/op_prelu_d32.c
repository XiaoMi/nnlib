/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
 * This contains implementations for quantized Prelu node
 */
#include <nn_graph.h>
#include <quantize.h>
#include <string.h>
#if defined(USE_OS_LINUX)
#include <malloc.h>
#endif

struct tdata {
	struct nn_node *self;
	const uint8_t *indata;
	uint8_t *outdata;
	uint8_t *alphabuf;
	int32_t in_next_row;
	int32_t in_next_d32;
	int32_t out_next_row;
	int32_t out_next_d32;
	int32_t in_qzero;
	int32_t out_qzero;
	int32_t d32_iters;
	int32_t d_iters;
	int32_t h_iters;
	int32_t shift;
	uint32_t shrink;
	nn_sem_t donesem;
};

struct nodeinfo {
	uint8_t *alphabuf;
	int shift;
	float alpha_mult;
	float alpha_mag;
};

void prelu_hvx_d32(
	uint8_t *outdata,
	const uint8_t *indata,
	int32_t in_next_row,
	int32_t in_next_d32,
	const uint8_t *alphabuf,
	int32_t in_qzero,
	int32_t d32iters,
	int32_t d_iters,
	int32_t h_iters,
	uint32_t shrink,
	int32_t out_qzero,
	int32_t alpha_shift);

static void prelu_execute_slice(struct nn_graph *nn, void *vinfo)
{
	struct tdata *td = vinfo;
	prelu_hvx_d32(
		td->outdata,
		td->indata,
		td->in_next_row,
		td->in_next_d32,
		td->alphabuf,
		td->in_qzero,
		td->d32_iters,
		td->d_iters,
		td->h_iters,
		td->shrink,
		td->out_qzero,
		td->shift);
	nn_sem_post(&td->donesem);
}




static int prelu_opt_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	uint32_t i;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	uint8_t in_qzero = quantize_uint8(0.0f,in_min,in_max);
	struct nodeinfo *nodeinfo = self->opaque;
	uint8_t *alphabuf = nodeinfo->alphabuf;
	int shift = nodeinfo->shift;

	int b = in_tensor->shape.batches;
	int h = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int w = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int d = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];
	//int h_total = h+h_pad_before+h_pad_after;
	int w_total = w+w_pad_before+w_pad_after;
	int d_total = d+d_pad_before+d_pad_after;

	float out_max = fmaxf(in_max,-in_min*nodeinfo->alpha_mag);
	float out_min = fminf(in_min,in_min*nodeinfo->alpha_mag);

	uint8_t out_qzero = quantize_uint8(0.0f,out_min,out_max);

	float in_range = (in_max-in_min);
	float out_range = (out_max-out_min);
	float shrink_factor = in_range/out_range;
	uint32_t fixed_shrink_factor = Q6_R_satub_R(fast_roundf(128*shrink_factor));
	logmsg(nn,2,"in_range=%f out_range=%f shrink factor: %f fixed: %x shift: %d",in_range,out_range,shrink_factor,fixed_shrink_factor,shift);

	if (tensor_get_float(in_min_tensor,0) < -tensor_get_float(in_max_tensor,0)) {
		logmsg(nn,1,"Caution: min < -max");
	}

	logmsg(nn,2,"d: %d<%d>%d ==> %d\n",d_pad_before,d,d_pad_after,d_total);
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		b,
		h,h_pad_before,h_pad_after,
		w,w_pad_before,w_pad_after,
		d,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out prepare fail");
	}

	int32_t in_next_row = tensor_row_stride_d32(in_tensor);
	int32_t in_next_d32 = tensor_d32_stride_d32(in_tensor);
	int32_t out_next_row = tensor_row_stride_d32(out_tensor);
	int32_t out_next_d32 = tensor_d32_stride_d32(out_tensor);

	struct tdata td = {
		.self = self,
		.indata = tensor_location_d32(in_tensor,0,0,-w_pad_before,0),
		.outdata = tensor_location_d32(out_tensor,0,0,-w_pad_before,0),
		.in_next_row = in_next_row,
		.in_next_d32 = in_next_d32,
		.out_next_row = out_next_row,
		.out_next_d32 = out_next_d32,
		.in_qzero = in_qzero,
		.out_qzero = out_qzero,
		.d32_iters = w_total / 4,
		.d_iters = d_total / 32,
		.h_iters = h,
		.alphabuf = alphabuf,
		.shrink = fixed_shrink_factor,
		.shift = shift,
	};
	nn_sem_init(&td.donesem,0);

	logmsg(nn,2,"Prelu execute. self=%p ",self);

	tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT);
	//tensor_set_float(out_min_tensor,0,nodeinfo->alpha_mag*tensor_get_float(in_min_tensor,0));
	//tensor_set_float(out_max_tensor,0,nodeinfo->alpha_mag*tensor_get_float(in_max_tensor,0));
	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);

	for (i = 0; i < b; i++) {
		td.indata = tensor_location_d32(in_tensor,i,0,-w_pad_before,0),
		td.outdata = tensor_location_d32(out_tensor,i,0,-w_pad_before,0),
		nn_os_work_for_vector(nn,prelu_execute_slice,&td);
		nn_sem_wait(&td.donesem);
	}

	logmsg(nn,2,"Prelu %p done",self);
	return 0;
}

static int prelu_check_opt(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 4) return errlog(nn,"maxpool wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"maxpool wrong # outs");
	const struct tensor *in_alpha_tensor = self->inputs[3];
	int32_t alpha_depth = in_alpha_tensor->shape.depth;
	int32_t alpha_depth_roundup = (alpha_depth + 31) & ~31;
	const float *alphas = in_alpha_tensor->data;
	struct nodeinfo *nodeinfo = self->opaque;
	uint8_t *alpha_frac_buf = NULL;
	int i,j,w;
	float maxmag = 1.0;
	float magrecip = 1.0;
	int maxmag_bad = 1;
	int shift = 0;
	if (nodeinfo == NULL) {
		if ((nodeinfo = nn_malloc(sizeof(*nodeinfo))) == NULL) {
			return errlog(nn,"can't alloc nodeinfo");
		}
		if ((alpha_frac_buf = nn_memalign(128,alpha_depth_roundup*4)) == NULL) {
			nn_free(nodeinfo);
			return errlog(nn,"can't allocate alpha buf");
		}
		nodeinfo->alphabuf = alpha_frac_buf;
		self->opaque = nodeinfo;
	} else {
		alpha_frac_buf = nodeinfo->alphabuf;
	}
	/*
	 * Find the maximum magnitude of the alphas 
	 * We will use this to scale the alpha values down and the output min/max up
	 * We also need to calculate the shift down of the positive input values
	 * All this is typically unnecessary, alpha input is likely small.
	 */
	while (maxmag_bad) {
		maxmag_bad = 0;
		for (i = 0; i < alpha_depth; i++) {
			if (fabsf(alphas[i]) > maxmag) {
				maxmag_bad = 1;
				maxmag *= 2.0f;
				shift++;
				break;
			}
		}
	}
	magrecip = 1.0f/maxmag;
	logmsg(nn,3,"maxmag=%f shift=%d maxmag_bad=%d",maxmag,shift,maxmag_bad);
	for (i = 0; i < alpha_depth; i++) {
		if (alphas[i] > maxmag) return errlog(nn,"alpha must be <= 1.0f");
		if (alphas[i] < -maxmag) return errlog(nn,"alpha must be >= -1.0f");
	}
	nodeinfo->alpha_mult = magrecip;
	nodeinfo->alpha_mag = maxmag;
	nodeinfo->shift = shift;
	/*
	 * Create our buffer of alpha values.  We duplicate 4x so that it lines up with the D32 format.
	 * Alpha values are also turned into 8 bit signed (q7) format.
	 */
	for (i = 0; i < alpha_depth_roundup; i += 32) {
		for (j = 0; j < 32; j++) {
			float alphaval;
			if ((i+j) >= alpha_depth) alphaval = alphas[0];
			else alphaval = alphas[i+j];
			logmsg(nn,3,"alphas[%d+%d] = %f --> %x",i,j,alphaval,Q6_R_satb_R(128*(alphaval*magrecip)+0.5f));
			for (w = 0; w < 4; w++) {
				alpha_frac_buf[i*4+w*32+j] = 
					Q6_R_satb_R((int)(128 * (alphaval * magrecip) + 0.5f));	// b / ub / shift amt
			}
		}
	}
	return 0;
}

static int prelu_ref_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *in_alpha_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	const float *alphas = in_alpha_tensor->data;
	uint8_t quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	int32_t val;
	int32_t b,h,w,d;
	const uint8_t *in_data;
	uint8_t *out_data;

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int width = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int depth = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];

	if (tensor_out_prepare_padded_d32(
		out_tensor,
		batches,
		height,h_pad_before,h_pad_after,
		width,w_pad_before,w_pad_after,
		depth,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out prepare fail");
	}

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	for (b = 0; b < batches; b++) {
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				for (d = 0; d < depth; d++) {
					in_data = tensor_location_d32(in_tensor,b,h,w,d);
					out_data = tensor_location_d32(out_tensor,b,h,w,d);
					val = *in_data;
					val = val - quantized_zero;
					if (val < 0) val = (val * alphas[d]) - 0.5f;
					val = val + quantized_zero;
					if (val < 0) val = 0;
					if (val > 255) val = 255;
					*out_data = val;
				}
			}
		}
	}
	logmsg(nn,2,"Prelu id %x ref done",self->node_id);
	return 0;
}

static int prelu_check(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 4) return errlog(nn,"maxpool wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"maxpool wrong # outs");
	return 0;
}

static int prelu_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct nodeinfo *nodeinfo = self->opaque;
	if (nodeinfo) {
		if (nodeinfo->alphabuf) nn_free(nodeinfo->alphabuf);
		nn_free(nodeinfo);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}
	

struct nn_node_ops nn_ops_for_QuantizedPRelu_8_d32 = {
	.execute = prelu_opt_execute,
	.check = prelu_check_opt,
	.ctor = node_alloc_common,
	.dtor = prelu_dtor,
};

struct nn_node_ops nn_ops_for_QuantizedPRelu_8_d32_ref = {
	.execute = prelu_ref_execute,
	.check = prelu_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

