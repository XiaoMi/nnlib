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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for variable nodes.
 */

#include <nn_graph.h>
#include <stdlib.h>

static int variable_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"variable execute. self=%p ",self);
	/* Nothing to do! */
	return 0;
}

static int variable_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking variable node %p",self);
	if (self->n_inputs > self->n_outputs) return errlog(nn,"too many inputs");
	for (i = 0; (i < self->n_inputs) && (i < self->n_outputs); i++) {
		if (tensor_copy(self->outputs[i],self->inputs[i]) != 0) {
			return errlog(nn,"out too small");
		}
	}
	return 0;
}

static struct nn_node *variable_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	int i;
	uint32_t data_size = 0;
	uint8_t *p;
	struct nn_node *self;
	if ((self = node_alloc_common(
		nn,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs)) == NULL) {
		errlog(nn,"alloc node");
		return NULL;
	}
	for (i = 0; i < num_outputs; i++) {
		data_size += nn_align_up(128,outputs[0].max_size);
	}
	if ((self->opaque = memalign(128,data_size)) == NULL) {
			errlog(nn,"tensor storage");
	}
	p = (uint8_t *)self->opaque;
	for (i = 0; i < num_outputs; i++) {
		self->outputs[i]->data = p;
		p += nn_align_up(128,outputs[i].max_size);
	}
	return self;
}

static int variable_dtor(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"variable node %p dtor",self);
	if (self->opaque) free(self->opaque);
	return node_free_common(self,nn);
}

int nn_variable_read(
	struct nn_graph *nn,
	struct nn_node *self,
	int output_index,
	uint32_t *b_out,
	uint32_t *h_out,
	uint32_t *w_out,
	uint32_t *d_out,
	uint8_t *data_out,
	uint32_t data_out_max,
	uint32_t *data_out_len)
{
	struct tensor *data = self->outputs[output_index];
	if (self->node_id != OP_Variable) return errlog(nn,"Not a variable");
	if (data_out_max < data->data_size) return errlog(nn,"too small");
	*b_out = data->shape.batches;
	*h_out = data->shape.height;
	*w_out = data->shape.width;
	*d_out = data->shape.depth;
	*data_out_len = data->data_size;
	memcpy(data_out,data->data,data->data_size);
	return 0;
}

int nn_variable_write(
	struct nn_graph *nn,
	struct nn_node *self,
	int output_index,
	uint32_t b,
	uint32_t h,
	uint32_t w,
	uint32_t d,
	const uint8_t *data_in,
	uint32_t data_in_size)
{
	struct tensor *data = self->outputs[output_index];
	if (self->node_id != OP_Variable) return errlog(nn,"Not a variable");
	if (data->max_size < data_in_size) return errlog(nn,"too small");
	data->shape.batches = b;
	data->shape.height = h;
	data->shape.width = w;
	data->shape.depth = d;
	data->data_size = data_in_size;
	memcpy(data->data,data_in,data_in_size);
	return 0;
}

static int assign_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Copy odd inputs to even inputs */
	int i;
	logmsg(nn,2,"assign execute. self=%p inputs=%d",self,self->n_inputs);
	for (i = 0; i < self->n_inputs; i += 2) {
		if (tensor_copy((struct tensor *)self->inputs[i],self->inputs[i+1]) != 0) {
			return errlog(nn,"can't copy to input %d",i);
		}
	}
	/* Copy input to output... or fudge output ptr in setup? */
	for (i = 0; i < self->n_outputs; i++) {
		if (tensor_copy(self->outputs[i],self->inputs[2*i+1]) != 0) {
			return errlog(nn,"can't copy to output %d",i);
		}
	}
	return 0;
}

static int assign_check(struct nn_node *self, struct nn_graph *nn)
{
	/* Check 2N inputs 1N output */
	if (self->n_inputs & 1) return errlog(nn,"bad # inputs (odd)");
	if (self->n_outputs > self->n_inputs/2) return errlog(nn,"too many outs");
	/* FIXME: Check that even inputs point to Variable nodes */
	return 0;
}


struct nn_node_ops nn_ops_for_Assign = {
	SFINIT(.execute, assign_execute),
	SFINIT(  .check, assign_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Variable = {
	SFINIT(.execute, variable_execute),
	SFINIT(  .check, variable_check),
	SFINIT(   .ctor, variable_ctor),
	SFINIT(   .dtor, variable_dtor),
};


