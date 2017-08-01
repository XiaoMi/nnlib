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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a node to flatten to a 1D array... basically fancy nop
 */

static int reshape_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_shape = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t b = self->inputs[0]->shape.batches;
	uint32_t h = self->inputs[0]->shape.height;
	uint32_t w = self->inputs[0]->shape.width;
	uint32_t d = self->inputs[0]->shape.depth;
	uint32_t elements = b*h*w*d;
	int32_t new_rank = in_shape->shape.depth;
	int32_t newb = (new_rank < 4) ? 1 : tensor_get_int32(in_shape,new_rank-4);
	int32_t newh = (new_rank < 3) ? 1 : tensor_get_int32(in_shape,new_rank-3);
	int32_t neww = (new_rank < 2) ? 1 : tensor_get_int32(in_shape,new_rank-2);
	int32_t newd = (new_rank < 1) ? 1 : tensor_get_int32(in_shape,new_rank-1);
	int32_t new_elements = newb*newh*neww*newd;
	int32_t num_negs = (newb < 0) + (newh < 0) + (neww < 0) + (newd < 0);
	int32_t unknown_dim;
	logmsg(nn,2,"(q)reshape execute. self=%p ",self);
	if (out_tensor->max_size < in_tensor->data_size) {
		return errlog(nn,"out too small");
	}
	if (num_negs > 1) return errlog(nn,"too many unknown dimensions");
	unknown_dim = elements/-new_elements;
	if (newb < 0) newb = unknown_dim;
	if (newh < 0) newh = unknown_dim;
	if (neww < 0) neww = unknown_dim;
	if (newd < 0) newd = unknown_dim;

	/* Copy input tensor to output */
	tensor_set_shape(out_tensor,newb,newh,neww,newd);
	out_tensor->data_size = in_tensor->data_size;
	vmemcpy_asm(out_tensor->data,in_tensor->data,in_tensor->data_size);
	/* Handle quantized version */
	if (self->n_outputs == 3) {
		if (tensor_copy(self->outputs[1],self->inputs[2]) != 0) {
			return errlog(nn,"bad extra copy");
		}
		if (tensor_copy(self->outputs[2],self->inputs[3]) != 0) {
			return errlog(nn,"bad extra copy");
		}
	}
	logmsg(nn,2,"qreshape %dx%dx%dx%x (%dx%dx%dx%d) --> %dx%dx%dx%d",
		b,h,w,d,
		in_shape->shape.batches,
		in_shape->shape.height,
		in_shape->shape.width,
		in_shape->shape.depth,
		newb,newh,neww,newd);
	return 0;
}

static int reshape_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking reshape node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"reshape node %p check OK",self);
	return 0;
}

static int qreshape_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking Qreshape node %p",self);
	if (self->n_inputs != 4) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"Qreshape node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Reshape = {
	SFINIT(.execute, reshape_execute),
	SFINIT(  .check, reshape_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedReshape = {
	SFINIT(.execute, reshape_execute),
	SFINIT(  .check, qreshape_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

