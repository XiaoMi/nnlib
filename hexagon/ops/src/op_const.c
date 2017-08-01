
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
 * This contains the code for constant nodes.
 */

#include <nn_graph.h>
#include <stdlib.h>

static int const_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Nothing to do! */
	return 0;
}

static int const_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking const node %p",self);
	if (self->inputs != NULL) {
		return errlog(nn,0,"const: fatal: inputs");
	}
	if (self->outputs == NULL) {
		return errlog(nn,0,"const: fatal: NULL outputs");
	}
	if (self->outputs[0] == NULL) {
		return errlog(nn,0,"const: fatal: NULL output 0");
	}
	logmsg(nn,2,"const node %p check OK",self);
	return 0;
}
struct nn_node *hexagon_nn_const_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len)
{
	struct nn_node *self;
	struct tensor *const_tensor;
	struct tensor tmp_tensor;
	tensor_set_shape(&tmp_tensor,batches,height,width,depth);
	tmp_tensor.data = (uint8_t *)data;
	tmp_tensor.max_size = tmp_tensor.data_size = data_len;
	if ((const_tensor = tensor_dup(&tmp_tensor)) == NULL) {
		errlog(nn,"can't alloc tensor");
		return NULL;
	}
	if ((self = alloc_node(node_id,OP_Const,NN_PAD_NA)) == NULL) {
		errlog(nn,"cant alloc node");
		return NULL;
	}
	self->n_inputs = 0;
	self->n_outputs = 1;
	self->outputs = &const_tensor->self;
	self->inputs = NULL;
	self->input_refs = NULL;
	return self;
}

static struct nn_node *const_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	logmsg(nn,0,"OOPS: called regular const ctor. don't do that.");
	return NULL;
}

static int const_dtor(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"const node %p dtor",self);
	tensor_free(self->outputs[0]);
	free(self);
	return 0;
}

struct nn_node_ops nn_ops_for_Const = {
	SFINIT(.execute, const_execute),
	SFINIT(  .check, const_check),
	SFINIT(   .ctor, const_ctor),
	SFINIT(   .dtor, const_dtor),
};


