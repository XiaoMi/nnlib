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


#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include "nn_gentranspose.h"

static int batchspace_s2b_execute(struct nn_node *self, struct nn_graph *nn, int elementsize, int dtype)
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
	int32_t copy_size = in_depth * elementsize;

	int32_t in_pad_top = tensor_get_int32(pad_tensor,0);
	int32_t in_pad_bottom = tensor_get_int32(pad_tensor,1);
	int32_t in_pad_left = 0;
	int32_t in_pad_right = 0;

	uint8_t quantized_zero = 0;
	if (dtype == NN_TYPE_QUINT8) {
		const struct tensor *min_in_tensor = self->inputs[3];
		const struct tensor *max_in_tensor = self->inputs[4];
		float in_max_float = tensor_get_float(max_in_tensor,0);
		float in_min_float = tensor_get_float(min_in_tensor,0);
		quantized_zero = quantize_uint8(0.0f,in_min_float,in_max_float);
	}

	const char *in_base = in_tensor->data;
	char *out = out_tensor->data;

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

	if (tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth,dtype)!=0){
		return errlog(nn,"failed to prepare output");
	}

	if( in_pad_top == 0 && in_pad_bottom == 0 && in_pad_left == 0 && in_pad_right == 0){
		struct nn_transpose_desc txdesc;
		// reshape as below:
		uint32_t dims[5] = { in_batches * out_height, h_stride, out_width, w_stride, in_depth };
		// transpose to:   { h_stride, w_stride, in_batches* out_height, out_width, in_depth };
		int32_t perm_arr[5] = { 1, 3, 0, 2, 4 };
		int res = nn_transpose_analyze_direct( &txdesc,elementsize, perm_arr, 5, dims,5 );
		if(res ==0){
			if( txdesc.buffer_needed > nn->scratch_size)
				res = nn_scratch_grow(nn, txdesc.buffer_needed);
			if( res == 0)
				res = nn_transpose_execute( nn, &txdesc, nn->scratch, (uint8_t*) out,(uint8_t const*)in_base);
		}
		if( res != 0)return errlog(nn,"transpose failed");
		return 0;
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
						if (h < 0 || h >= in_height || w < 0 || w >= in_width){
							memset(out,quantized_zero,copy_size);
						}else{
							const char *in = in_base
								+ elementsize * (b*in_depth*in_width*in_height
								+ h*in_width*in_depth
								+ w*in_depth);
							memcpy(out,in,copy_size);
						}
						out += copy_size / sizeof(*out);
					}
				}
			}
		}
	}
	return 0;
}

static int batchspace_b2s_execute(struct nn_node *self, struct nn_graph *nn, int elementsize, int dtype)
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
	int32_t copy_size = in_depth * elementsize;

	int32_t in_pad_top = tensor_get_int32(pad_tensor,0);
	int32_t in_pad_bottom = tensor_get_int32(pad_tensor,1);
	int32_t in_pad_left = 0;
	int32_t in_pad_right = 0;

	int32_t out_batches;
	int32_t out_depth = in_depth;
	int32_t out_height;
	int32_t out_width;

	const char *in_base = in_tensor->data;
	char *out_base = out_tensor->data;

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
	if (tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth,dtype)!=0){
		return errlog(nn,"failed to prepare output");
	}

	if( in_pad_top == 0 && in_pad_bottom == 0 && in_pad_left == 0 && in_pad_right == 0){
		struct nn_transpose_desc txdesc;
		// reshape as below:
		uint32_t dims[5] = { h_stride, w_stride, out_batches* in_height, in_width, in_depth };
		// transpose to :  { out_batches* in_height, h_stride, in_width, w_stride, in_depth }
		int32_t perm_arr[5] = { 2, 0, 3, 1, 4 };
		int res = nn_transpose_analyze_direct( &txdesc,elementsize,
                perm_arr, 5, dims,5 );
		if(res ==0){
			if( txdesc.buffer_needed > nn->scratch_size)
				res = nn_scratch_grow(nn, txdesc.buffer_needed);
			if( res == 0)
				res = nn_transpose_execute( nn, &txdesc, nn->scratch, (uint8_t*) out_base,(uint8_t const*)in_base);
		}
		if( res != 0)return errlog(nn,"transpose failed");
		return 0;
	}

	for (b = 0; b < out_batches; b++) {
		for (h = 0; h < out_height; h++) {
			unsigned hp = h + in_pad_top;		// h including padding.
			int h_in = hp/h_stride;				// h from the actual input
			int h_rem = hp-h_stride*h_in;		// = hp%h_stride
			for (w = 0; w < out_width; w++) {
				unsigned wp = w + in_pad_left;		// w including padding.
				int w_in = wp/w_stride;
				int w_rem = wp-w_stride*w_in;
				int b_in = (h_rem*w_stride + w_rem)*out_batches+b;
				char *out = out_base 
					+ elementsize*(b*out_depth*out_width*out_height
					+ h*out_width*out_depth
					+ w*out_depth);
				const char *in = in_base
					+ elementsize*(b_in*in_depth*in_width*in_height
					+ h_in*in_depth*in_width
					+ w_in*in_depth);
				memcpy(out,in,copy_size);
			}
		}
	}

	return 0;
}


static int batchspace_b2s_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return batchspace_b2s_execute(self,nn,sizeof(float),NN_TYPE_FLOAT);
}

static int batchspace_s2b_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return batchspace_s2b_execute(self,nn,sizeof(float),NN_TYPE_FLOAT);
}

static int batchspace_b2s_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[3]);
	tensor_copy(self->outputs[2],self->inputs[4]);
	return batchspace_b2s_execute(self,nn,sizeof(uint8_t),NN_TYPE_QUINT8);
}

static int batchspace_s2b_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[3]);
	tensor_copy(self->outputs[2],self->inputs[4]);
	return batchspace_s2b_execute(self,nn,sizeof(uint8_t),NN_TYPE_QUINT8);
}


struct nn_node_ops nn_ops_for_BatchToSpaceND_f = {
	.execute = batchspace_b2s_execute_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_SpaceToBatchND_f = {
	.execute = batchspace_s2b_execute_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_BatchToSpaceND_8 = {
	.execute = batchspace_b2s_execute_8,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};

struct nn_node_ops nn_ops_for_SpaceToBatchND_8 = {
	.execute = batchspace_s2b_execute_8,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};

