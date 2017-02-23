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
	int32_t start = tensor_get_int32(start_tensor,0);
	int32_t stop = tensor_get_int32(stop_tensor,0);
	int32_t step = tensor_get_int32(step_tensor,0);
	int32_t out_elements = (stop - start + (step-1))/step;
	const char *in = input_tensor->data;
	char *out = out_tensor->data;
	uint32_t total_bytes = out_elements * element_size;
	int i,j;
	int32_t out_off,in_off;

	if (total_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	if (b_in > 1) return errlog(nn,"strided slice: Only 1D for now.");
	if (h_in > 1) return errlog(nn,"strided slice: Only 1D for now.");
	if (w_in > 1) return errlog(nn,"strided slice: Only 1D for now.");
	if (d_in <= start) return errlog(nn,"bad start");
	if (d_in < stop) return errlog(nn,"bad stop");

	logmsg(nn,2,"slice node %p execute %d,%d,%d",
		self,(int)start,(int)stop,(int)step);

	tensor_set_shape(out_tensor,1,1,1,out_elements);
	out_tensor->data_size = total_bytes;

	for (i = start, j = 0; j < out_elements; j++, i += step) {
		in_off = i*element_size;
		out_off = j*element_size;
		memcpy(out+out_off,in+in_off,element_size);
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
	.execute = sslice_execute_4b,
	.check = sslice_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_StridedSlice_int32 = {
	.execute = sslice_execute_4b,
	.check = sslice_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_StridedSlice_uint8 = {
	.execute = sslice_execute_1b,
	.check = sslice_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedStridedSlice_8 = {
	.execute = sslice_execute_q8,
	.check = sslice_check_q8,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};



