/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
#include <quantize.h>
#include <math.h>


static int batchspace_s2b_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *strides_tensor = self->inputs[1];
	const struct tensor *pad_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	int32_t subsample_dims = strides_tensor->shape.depth;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_batches = in_tensor->shape.batches;
	int32_t h_start,w_start;
	int32_t h,w,b;
	int32_t h_stride = 1;
	int32_t w_stride = 1;
	int32_t copy_size = in_depth * sizeof(float);

	int32_t in_pad_top = tensor_get_int32(pad_tensor,0);
	int32_t in_pad_bottom = tensor_get_int32(pad_tensor,1);
	int32_t in_pad_left = 0;
	int32_t in_pad_right = 0;

	const float *in_base = (const float *)in_tensor->data;
	float *out = (float *)out_tensor->data;

	if (subsample_dims >= 3) return errlog(nn,"don't support that many space dimensions");
	if (subsample_dims == 1) {
		h_stride = tensor_get_int32(strides_tensor,0);
		w_stride = 1;
	}
	if (subsample_dims == 2) {
		h_stride = tensor_get_int32(strides_tensor,0);
		w_stride = tensor_get_int32(strides_tensor,1);
		in_pad_left = tensor_get_int32(pad_tensor,2);
		in_pad_right = tensor_get_int32(pad_tensor,3);
	}

	int32_t out_batches = in_batches * h_stride * w_stride;
	int32_t out_height = (in_height + in_pad_top + in_pad_bottom) / h_stride;
	int32_t out_width = (in_width + in_pad_left + in_pad_right) / w_stride;
	int32_t out_depth = in_depth;
	int32_t out_elements = out_batches * out_width * out_height * out_depth;

	if (subsample_dims < 1) return errlog(nn,"bad stride shape");
	if (strides_tensor->shape.width != 1) return errlog(nn,"bad stride shape");
	if (strides_tensor->shape.height != 1) return errlog(nn,"bad stride shape");
	if (strides_tensor->shape.batches != 1) return errlog(nn,"bad stride shape");
	if (pad_tensor->shape.depth != 2) return errlog(nn,"bad pad shape");
	if (pad_tensor->shape.width != subsample_dims) return errlog(nn,"bad pad shape");
	if (pad_tensor->shape.height != 1) return errlog(nn,"bad pad shape");
	if (pad_tensor->shape.batches != 1) return errlog(nn,"bad pad shape");
	if (((in_height + in_pad_top + in_pad_bottom) % h_stride) != 0) {
		return errlog(nn,"height not evenly divisible");
	}
	if (((in_width + in_pad_left + in_pad_right) % w_stride) != 0) {
		return errlog(nn,"width not evenly divisible");
	}
	for (h_start = 0; h_start < h_stride; h_start++) {
		int h_begin = h_start - in_pad_top;
		int h_end = in_height + in_pad_bottom;
		for (w_start = 0; w_start < w_stride; w_start++) {
			int w_begin = w_start - in_pad_left;
			int w_end = in_width + in_pad_right;
			for (b = 0; b < in_batches; b++) {
				for (h = h_begin; h < h_end; h += h_stride) {
					for (w = w_begin; w < w_end; w += w_stride) {
						const float *in = in_base 
							+ b*in_depth*in_width*in_height
							+ h*in_width*in_depth
							+ w*in_depth;
						if (h < 0) memset(out,0,copy_size);
						else if (h >= in_height) memset(out,0,copy_size);
						else if (w < 0) memset(out,0,copy_size);
						else if (w >= in_width) memset(out,0,copy_size);
						else memcpy(out,in,copy_size);
						out += copy_size / sizeof(*out);
					}
				}
			}
		}
	}
	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_elements * sizeof(float);
	return 0;
}

static int batchspace_b2s_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *strides_tensor = self->inputs[1];
	const struct tensor *pad_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	int32_t subsample_dims = strides_tensor->shape.depth;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_batches = in_tensor->shape.batches;
	//int32_t h_start,w_start;
	int32_t h,w,b;
	int32_t h_stride = 1;
	int32_t w_stride = 1;
	int32_t copy_size = in_depth * sizeof(float);

	int32_t in_pad_top = tensor_get_int32(pad_tensor,0);
	int32_t in_pad_bottom = tensor_get_int32(pad_tensor,1);
	int32_t in_pad_left = 0;
	int32_t in_pad_right = 0;

	int32_t out_batches;
	int32_t out_depth = in_depth;
	int32_t out_height;
	int32_t out_width;

	const float *in_base = (const float *)in_tensor->data;
	float *out_base = (float *)out_tensor->data;

	if (subsample_dims >= 3) return errlog(nn,"don't support that many space dimensions");
	if (subsample_dims == 1) {
		h_stride = tensor_get_int32(strides_tensor,0);
		w_stride = 1;
	}
	if (subsample_dims == 2) {
		h_stride = tensor_get_int32(strides_tensor,0);
		w_stride = tensor_get_int32(strides_tensor,1);
		in_pad_left = tensor_get_int32(pad_tensor,2);
		in_pad_right = tensor_get_int32(pad_tensor,3);
	}
	out_height = h_stride * in_height - in_pad_top - in_pad_bottom;
	out_width = w_stride * in_width - in_pad_left - in_pad_right;
	out_batches = in_batches / (h_stride * w_stride);
	int32_t out_elements = out_batches * out_width * out_height * out_depth;

	if (subsample_dims < 1) return errlog(nn,"bad stride shape");
	if (strides_tensor->shape.width != 1) return errlog(nn,"bad stride shape");
	if (strides_tensor->shape.height != 1) return errlog(nn,"bad stride shape");
	if (strides_tensor->shape.batches != 1) return errlog(nn,"bad stride shape");
	if (pad_tensor->shape.depth != 2) return errlog(nn,"bad pad shape");
	if (pad_tensor->shape.width != subsample_dims) return errlog(nn,"bad pad shape");
	if (pad_tensor->shape.height != 1) return errlog(nn,"bad pad shape");
	if (pad_tensor->shape.batches != 1) return errlog(nn,"bad pad shape");
	if ((in_batches % (h_stride * w_stride)) != 0) {
		return errlog(nn,"batches not evenly divisible");
	}

	for (b = 0; b < out_batches; b++) {
		for (h = 0; h < out_height; h++) {
			for (w = 0; w < out_width; w++) {
				int h_in = (h+in_pad_top)/h_stride;
				int w_in = (w+in_pad_left)/w_stride;
				int b_in = ((h%h_stride)*w_stride + (w%w_stride))*out_batches+b;
				float *out = out_base 
					+ b*out_depth*out_width*out_height
					+ h*out_width*out_depth
					+ w*out_depth;
				const float *in = in_base
					+ b_in*in_depth*in_width*in_height
					+ h_in*in_depth*in_width
					+ w_in*in_depth;
				memcpy(out,in,copy_size);
			}
		}
	}

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_elements * sizeof(float);
	return 0;
}

static int batchspace_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking batchspace node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"batchspace %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_BatchToSpaceND_f = {
	SFINIT(.execute, batchspace_b2s_execute),
	SFINIT(  .check, batchspace_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_SpaceToBatchND_f = {
	SFINIT(.execute, batchspace_s2b_execute),
	SFINIT(  .check, batchspace_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

