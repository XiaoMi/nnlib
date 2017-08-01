
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

static int pack_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Check to make sure shapes are compatible */
	const struct tensor *t0 = self->inputs[0];
	const struct tensor *t;
	struct tensor *out_tensor = self->outputs[0];
	int n_inputs = self->n_inputs;
	int depth = t0->shape.depth;
	int width = t0->shape.width;
	int height = t0->shape.height;
	int batches = t0->shape.batches;
	int out_depth = depth;
	int out_width = width;
	int out_height = height;
	int out_batches = batches;
	unsigned int total_bytes = t0->data_size;
	char *out = (char *)out_tensor->data;
	int i;
	for (i = 1; i < n_inputs; i++) {
		t = self->inputs[i];
		if (t->shape.batches != batches) return errlog(nn,"bad shape");
		if (t->shape.height != height) return errlog(nn,"bad shape");
		if (t->shape.width != width) return errlog(nn,"bad shape");
		if (t->shape.depth != depth) return errlog(nn,"bad shape");
		if (t->data_size != t0->data_size) return errlog(nn,"bad size");
		total_bytes += t->data_size;
	}
	/* Check output size is OK */
	if (out_tensor->max_size < total_bytes) return errlog(nn,"out too small");
	/* Copy data */
	for (i = 0; i < n_inputs; i++) {
		t = self->inputs[i];
		memcpy(out,t->data,t->data_size);
		out += t->data_size;
	}
	/* Assume we want to expand least significant unity dimension */
	if (out_depth == 1) out_depth = n_inputs;
	else if (out_width == 1) out_width = n_inputs;
	else if (out_height == 1) out_height = n_inputs;
	else if (out_batches == 1) out_batches = n_inputs;
	else return errlog(nn,"Only support 4D, out of dimensions");
	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = total_bytes;
	return 0;
}

static int pack_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking pack node %p",self);
	if (self->n_inputs < 1) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

struct nn_node_ops nn_ops_for_Pack_f = {
	SFINIT(.execute, pack_execute),
	SFINIT(  .check, pack_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Pack_int32 = {
	SFINIT(.execute, pack_execute),
	SFINIT(  .check, pack_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

