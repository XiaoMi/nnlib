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

/*
 * Given a start offset and width and a stride for each dimention in the input tensor,
 * create a new output tensor with just the slice specified.
 * 
 * There were a lot of fancy features in the docs for another environment,
 * so we'll do a subset.
 */

#include <nn_graph.h>
#include <string.h>

static int strided_slice_impl(
	struct nn_node *self,
	struct nn_graph *nn,
	int element_size)
{
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *start_tensor = self->inputs[1];
	const struct tensor *stop_tensor = self->inputs[2];
	const struct tensor *step_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	int b_in = input_tensor->shape.batches;
	int h_in = input_tensor->shape.height;
	int w_in = input_tensor->shape.width;
	int d_in = input_tensor->shape.depth;
	const char *in = (const char *)input_tensor->data;
	char *out = (char *)out_tensor->data;
	int32_t order = start_tensor->shape.depth;
	int b_start = (order < 4) ? 0 : tensor_get_int32(start_tensor,order-4);
	int h_start = (order < 3) ? 0 : tensor_get_int32(start_tensor,order-3);
	int w_start = (order < 2) ? 0 : tensor_get_int32(start_tensor,order-2);
	int d_start = (order < 1) ? 0 : tensor_get_int32(start_tensor,order-1);
	int b_stop = (order < 4) ? b_in : tensor_get_int32(stop_tensor,order-4);
	int h_stop = (order < 3) ? h_in : tensor_get_int32(stop_tensor,order-3);
	int w_stop = (order < 2) ? w_in : tensor_get_int32(stop_tensor,order-2);
	int d_stop = (order < 1) ? d_in : tensor_get_int32(stop_tensor,order-1);
	int b_step = (order < 4) ? 1 : tensor_get_int32(step_tensor,order-4);
	int h_step = (order < 3) ? 1 : tensor_get_int32(step_tensor,order-3);
	int w_step = (order < 2) ? 1 : tensor_get_int32(step_tensor,order-2);
	int d_step = (order < 1) ? 1 : tensor_get_int32(step_tensor,order-1);

	int b_out = (b_stop - b_start + b_step - 1) / (b_step);
	int h_out = (h_stop - h_start + h_step - 1) / (h_step);
	int w_out = (w_stop - w_start + w_step - 1) / (w_step);
	int d_out = (d_stop - d_start + d_step - 1) / (d_step);

	int out_elements = b_out*h_out*w_out*d_out;
	uint32_t total_bytes = out_elements * element_size;
	int b,h,w,d;
	int offset;

	if (b_stop == 0) b_stop = b_in;
	if (w_stop == 0) w_stop = w_in;
	if (h_stop == 0) h_stop = h_in;
	if (d_stop == 0) d_stop = d_in;

	logmsg(nn,2,"start_tensor: %d %d %d %d stop: %d %d %d %d step: %d %d %d %d",
		tensor_get_int32(start_tensor,0),
		tensor_get_int32(start_tensor,1),
		tensor_get_int32(start_tensor,2),
		tensor_get_int32(start_tensor,3),
		tensor_get_int32(stop_tensor,0),
		tensor_get_int32(stop_tensor,1),
		tensor_get_int32(stop_tensor,2),
		tensor_get_int32(stop_tensor,3),
		tensor_get_int32(step_tensor,0),
		tensor_get_int32(step_tensor,1),
		tensor_get_int32(step_tensor,2),
		tensor_get_int32(step_tensor,3));
	logmsg(nn,2,"slice node %p execute order=%d in=%dx%dx%dx%d start=%dx%dx%dx%d stop=%dx%dx%dx%d step=%dx%dx%dx%d out=%dx%dx%dx%d", 
		self,order,
		b_in,h_in,w_in,d_in,
		b_start,h_start,w_start,d_start,
		b_stop,h_stop,w_stop,d_stop,
		b_step,h_step,w_step,d_step,
		b_out,h_out,w_out,d_out);
	if (total_bytes > out_tensor->max_size) {
		return errlog(nn,"out too small, %d > %d",total_bytes,out_tensor->max_size);
	}

	tensor_set_shape(out_tensor,b_out,h_out,w_out,d_out);
	out_tensor->data_size = total_bytes;

	for (b = b_start; b < b_stop; b += b_step) {
		for (h = h_start; h < h_stop; h += h_step) {
			for (w = w_start; w < w_stop; w += w_step) {
				for (d = d_start; d < d_stop; d += d_step) {
					offset = element_size*(b*h_in*w_in*d_in 
						+ h*w_in*d_in
						+ w*d_in
						+ d);
					memcpy(out,in+offset,element_size);
					out += element_size;
				}
			}
		}
	}

	return 0;
}

static int sslice_execute_4b(struct nn_node *self, struct nn_graph *nn)
{
	return strided_slice_impl(self,nn,4);
}

static int sslice_execute_1b(struct nn_node *self, struct nn_graph *nn)
{
	return strided_slice_impl(self,nn,1);
}

static int sslice_execute_q8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[4]);
	tensor_copy(self->outputs[2],self->inputs[5]);
	return strided_slice_impl(self,nn,1);
}

static int sslice_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 4) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int sslice_check_q8(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 6) return errlog(nn,"num inputs");
	if (self->n_outputs != 3) return errlog(nn,"num outputs");
	return 0;
}


struct nn_node_ops nn_ops_for_StridedSlice_f = {
	SFINIT(.execute, sslice_execute_4b),
	SFINIT(  .check, sslice_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_StridedSlice_int32 = {
	SFINIT(.execute, sslice_execute_4b),
	SFINIT(  .check, sslice_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_StridedSlice_uint8 = {
	SFINIT(.execute, sslice_execute_1b),
	SFINIT(  .check, sslice_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedStridedSlice_8 = {
	SFINIT(.execute, sslice_execute_q8),
	SFINIT(  .check, sslice_check_q8),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};
