
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
 * Given a start offset and width for each dimention in the input tensor,
 * create a new output tensor with just the slice specified.
 * 
 */

#include <nn_graph.h>
#include <string.h>

static int slice_impl(
	struct nn_node *self,
	struct nn_graph *nn,
	int element_size)
{
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *start_tensor = self->inputs[1];
	const struct tensor *size_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	int b,b_in,b_start,b_size;
	int h,h_in,h_start,h_size;
	int w,w_in,w_start,w_size;
	int d_in,d_start,d_size;
	uint32_t total_bytes;
	uint32_t offset;
	const char *data = (const char *)input_tensor->data;
	const char *in;
	char *out = (char *)out_tensor->data;
	int order_skip = 4-start_tensor->shape.depth;
	logmsg(nn,2,"slice node %p execute",self);
	b_in = input_tensor->shape.batches;
	h_in = input_tensor->shape.height;
	w_in = input_tensor->shape.width;
	d_in = input_tensor->shape.depth;
	b_start = (order_skip > 0) ? 0 : tensor_get_int32(start_tensor,0-order_skip);
	h_start = (order_skip > 1) ? 0 : tensor_get_int32(start_tensor,1-order_skip);
	w_start = (order_skip > 2) ? 0 : tensor_get_int32(start_tensor,2-order_skip);
	d_start = (order_skip > 3) ? 0 : tensor_get_int32(start_tensor,3-order_skip);
	b_size = (order_skip > 0) ? -1 : tensor_get_int32(size_tensor,0-order_skip);
	h_size = (order_skip > 1) ? -1 : tensor_get_int32(size_tensor,1-order_skip);
	w_size = (order_skip > 2) ? -1 : tensor_get_int32(size_tensor,2-order_skip);
	d_size = (order_skip > 3) ? -1 : tensor_get_int32(size_tensor,3-order_skip);
	if (b_size == -1) b_size = b_in - b_start;
	if (h_size == -1) h_size = h_in - h_start;
	if (w_size == -1) w_size = w_in - w_start;
	if (d_size == -1) d_size = d_in - d_start;

	logmsg(nn,2,"in/start/size: b: %d/%d/%d h: %d/%d/%d w: %d/%d/%d d: %d/%d/%d order_skip=%d",
		b_in,b_start,b_size,
		h_in,h_start,h_size,
		w_in,w_start,w_size,
		d_in,d_start,d_size,
		order_skip);

	total_bytes = b_size * h_size * w_size * d_size * element_size;

	if (total_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	if (b_size <= 0) return errlog(nn,"bad b_size");
	if (h_size <= 0) return errlog(nn,"bad h_size");
	if (w_size <= 0) return errlog(nn,"bad w_size");
	if (d_size <= 0) return errlog(nn,"bad d_size");
	if ((b_start+b_size) > b_in) return errlog(nn,"in b too small");
	if ((h_start+h_size) > h_in) return errlog(nn,"in h too small");
	if ((w_start+w_size) > w_in) return errlog(nn,"in w too small");
	if ((d_start+d_size) > d_in) return errlog(nn,"in d too small");

	tensor_set_shape(out_tensor,b_size,h_size,w_size,d_size);
	out_tensor->data_size = total_bytes;

	for (b = 0; b < b_size; b++) {
		for (h = 0; h < h_size; h++) {
			for (w = 0; w < w_size; w++) {
				offset  = (b_start+b)*(d_in*w_in*h_in) 
					+ (h_start+h)*(d_in*w_in)
					+ (w_start+w)*(d_in)
					+ (d_start);
				offset *= element_size;
				in = data+offset;
				memcpy(out,in,d_size*element_size);
				out += d_size * element_size;
			}
		}
	}
	return 0;
}

static int slice_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return slice_impl(self,nn,sizeof(float));
}

static int slice_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	return slice_impl(self,nn,sizeof(uint8_t));
}

static int slice_execute_int32(struct nn_node *self, struct nn_graph *nn)
{
	return slice_impl(self,nn,sizeof(int32_t));
}

static int slice_execute_q8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[3]);
	tensor_copy(self->outputs[2],self->inputs[4]);
	return slice_impl(self,nn,sizeof(uint8_t));
}

static int slice_check_f(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int slice_check_8(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int slice_check_int32(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int slice_check_q8(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 5) return errlog(nn,"num inputs");
	if (self->n_outputs != 3) return errlog(nn,"num outputs");
	return 0;
}


struct nn_node_ops nn_ops_for_Slice_f = {
	SFINIT(.execute, slice_execute_f),
	SFINIT(  .check, slice_check_f),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Slice_8 = {
	SFINIT(.execute, slice_execute_8),
	SFINIT(  .check, slice_check_8),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Slice_int32 = {
	SFINIT(.execute, slice_execute_int32),
	SFINIT(  .check, slice_check_int32),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedSlice_8 = {
	SFINIT(.execute, slice_execute_q8),
	SFINIT(  .check, slice_check_q8),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

