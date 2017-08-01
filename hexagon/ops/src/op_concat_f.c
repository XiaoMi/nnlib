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


static inline uint32_t tensor_elements(const struct tensor *t)
{
	return t->shape.batches
		* t->shape.height
		* t->shape.width
		* t->shape.depth;
}

static int concat_do_execute(
	struct nn_node *self,
	struct nn_graph *nn,
	const struct tensor *dim_tensor,
	const struct tensor **input_tensors,
	int n_input_tensors,
	int elementsize)
{
	const struct tensor *t;
	struct tensor *out_tensor = self->outputs[0];
	int32_t in_width = input_tensors[0]->shape.width;
	int32_t in_height = input_tensors[0]->shape.height;
	int32_t in_batches = input_tensors[0]->shape.batches;
	int32_t i;
	int32_t j;
	//float in_off; 
	int32_t out_elements = 0;
	int32_t out_bytes = 0;
	int32_t total_depth = 0;
	int32_t iters = in_width * in_height * in_batches;
	const float *in_data;
	float *out_data = (float *)out_tensor->data;
	logmsg(nn,2,"concat execute. self=%p ",self);
	if (tensor_get_int32(dim_tensor,0) != 3) {
		if (!((in_width == 1) && (in_height == 1))) return errlog(nn,"only depth: %d (%dx%dx%dx%d)",tensor_get_int32(dim_tensor,0),in_batches,in_height,in_width,input_tensors[0]->shape.depth);
	}
	for (i = 0; i < n_input_tensors; i++) {
		if (input_tensors[i]->shape.width != in_width) {
			return errlog(nn,"width mismatch tensor %d",i);
		}
		if (input_tensors[i]->shape.height != in_height) {
			return errlog(nn,"height mismatch tensor %d",i);
		}
		if (input_tensors[i]->shape.batches != in_batches) {
			return errlog(nn,"batches mismatch tensor %d",i);
		}
		out_elements += tensor_elements(input_tensors[i]);
		total_depth += input_tensors[i]->shape.depth;
	}
	out_bytes = out_elements * elementsize;
	if (out_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,in_batches,in_height,in_width,total_depth);
	out_tensor->data_size = out_bytes;
	for (i = 0; i < n_input_tensors; i++) {
		t = input_tensors[i];
		in_data = (const float *)t->data;
		uint32_t t_depth_bytes = t->shape.depth * elementsize;
		for (j = 0; j < iters; j++) {
			memcpy(out_data + j*total_depth,in_data,t_depth_bytes);
			in_data += t->shape.depth;
		}
		out_data += t->shape.depth;
	}
	logmsg(nn,2,"concat %p done",self);
	return 0;
}

static int concat_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1);
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	return concat_do_execute(self,nn,dim_tensor,input_tensors,n_input_tensors,sizeof(float));
}

static int concatv2_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1);
	const struct tensor *dim_tensor = self->inputs[n_input_tensors];
	const struct tensor **input_tensors = &self->inputs[0];
	return concat_do_execute(self,nn,dim_tensor,input_tensors,n_input_tensors,sizeof(float));
}

static int concatv2_int_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1);
	const struct tensor *dim_tensor = self->inputs[n_input_tensors];
	const struct tensor **input_tensors = &self->inputs[0];
	return concat_do_execute(self,nn,dim_tensor,input_tensors,n_input_tensors,sizeof(int32_t));
}

static int concat_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking concat node %p",self);
	if (self->n_inputs < 2) return errlog(nn,"at least 1 input");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	for (i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) return errlog(nn,"NULL input");
	}
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) return errlog(nn,"NULL output");
	}
	logmsg(nn,2,"concat node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Concat_f = {
	SFINIT(.execute, concat_execute),
	SFINIT(  .check, concat_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_ConcatV2_f = {
	SFINIT(.execute, concatv2_execute),
	SFINIT(  .check, concat_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_ConcatV2_int32 = {
	SFINIT(.execute, concatv2_int_execute),
	SFINIT(  .check, concat_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

