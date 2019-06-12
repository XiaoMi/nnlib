
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
#include <nn_graph_hmx.h>
#include <string.h>

static int convert_d32_to_tile_hvx(struct nn_graph *nn, void *vself)
{
	const struct nn_node *self = vself;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];

	int32_t out_batches = in_batches;
	int32_t out_width = in_width;
	int32_t out_height = in_height;
	int32_t out_depth = in_depth;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-(out_left_pad+in_width))&7;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = 4 + ((-in_height)&7);
	int32_t out_depth_before_pad = in_depth_before_pad;
	int32_t out_depth_after_pad = in_depth_after_pad;

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int w, h, d, b;

	for (b = 0; b < in_batches; b++) {
		for (h = 0; h < in_height; h++) {
			for (d = 0; d < in_depth_total; d+=32) { 
				HVX_Vector *in  = (HVX_Vector *)tensor_location_d32(in_tensor,b,h,0,d);
				for (w = 0; w < in_width; w+=4) { 
					HVX_Vector *out = (HVX_Vector *)tensor_location_tile(out_tensor,b,h,w,d, 0);
					*out = *in++;
				}
			}
		}
	}
	return 0;
}


static int convert_tile_to_d32_hvx(struct nn_graph *nn, void *vself)
{
	const struct nn_node *self = vself;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad  = in_tensor->format.depth_pad[1];

	int32_t out_batches = in_batches;
	int32_t out_width = in_width;
	int32_t out_height = in_height;
	int32_t out_depth = in_depth;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-in_width)&3; 
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = in_depth_before_pad;
	int32_t out_depth_after_pad = in_depth_after_pad;

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;

	int w, h, d, b;
	for (b = 0; b < in_batches; b++) {
		for (d = 0; d < in_depth_total; d+=32) { 
			for (h = 0; h < in_height; h++) {
				HVX_Vector *out  = (HVX_Vector *)tensor_location_d32(out_tensor,b,h,0,d);
				for (w = 0; w < in_width; w+=4) { 
					HVX_Vector *in = (HVX_Vector *)tensor_location_tile(in_tensor,b,h,w,d, 0);
					*out++ = *in;
				}
			}
		}
	}
	return 0;
}

static int convert_from_tile_hvx(struct nn_graph *nn, void *vself)
{
	const struct nn_node *self = vself;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	if (tensor_out_prepare_normal(out_tensor,in_batches,in_height,in_width,in_depth,NN_TYPE_QUINT8)) {
		return errlog(nn,"can't prepare output bhwd=%d,%d,%d,%d out_size=%d",
			in_batches,in_height,in_width,in_depth,out_tensor->max_size);
	}

	const uint8_t *in;
	uint8_t *out = out_tensor->data;
	int d_pad_before = in_tensor->format.depth_pad[0];

	int b,h,w;
	for (b = 0; b < in_batches; b++) {
		for (h = 0; h < in_height; h++) {
			for (w = 0; w < in_width; w++) {
				int d = d_pad_before;
				int depth = 32-d; 
				while(depth > 0){
					in = tensor_location_tile(in_tensor,b,h,w,d, 0);
					memcpy(out,in,depth);
					out += depth;
					d += depth;
					depth = (in_depth-d) > 32 ? 32 : (in_depth-d);
				}
			}
		}
	}
	return 0;
}

static int convert_d32toTile_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_os_vector_call(nn,convert_d32_to_tile_hvx,self);
}

static int convert_Tiletod32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_os_vector_call(nn,convert_tile_to_d32_hvx,self);
}


static int convert_from_tile_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_os_vector_call(nn,convert_from_tile_hvx,self);
}


struct nn_node_ops nn_ops_for_Convert_d32toTile = {
	.execute = convert_d32toTile_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,5),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_D32_INPUT,
};

struct nn_node_ops nn_ops_for_Convert_Tiletod32 = {
	.execute = convert_Tiletod32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,5),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_D32_OUTPUT,
};

struct nn_node_ops nn_ops_for_Convert_from_tile = {
	.execute = convert_from_tile_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,5),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_D32_INPUT,
};

