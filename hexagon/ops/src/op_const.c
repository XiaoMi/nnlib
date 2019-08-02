
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for constant nodes.
 */

#include <nn_graph.h>
#include <stdlib.h>
#include "nn_const_prep_share.h"

static int const_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Nothing to do! */
	return 0;
}

static int const_check(struct nn_node *self, struct nn_graph *nn)
{
	if (self->inputs != NULL) {
		return errlog(nn,0,"const: fatal: inputs");
	}
	if (self->outputs == NULL) {
		return errlog(nn,0,"const: fatal: NULL outputs");
	}
	if (self->outputs[0] == NULL) {
		return errlog(nn,0,"const: fatal: NULL output 0");
	}
	return 0;
}

struct nn_node *hexagon_nn_empty_const_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len)
{
	struct nn_node *self;
	struct tensor *const_tensor;
	struct tensor tmp_tensor;
	struct output *outdefs;

	uint32_t allsize = mulu32_x4_sat(batches,height,width,depth);
	if( allsize == 0) allsize = 1;	// @@ for now; avoid /0
	if( allsize == 0 || allsize >= (1u<<28))
		return NULL;

	tensor_set_shape(&tmp_tensor,batches,height,width,depth);
	if ((const_tensor = tensor_alloc(&tmp_tensor.shape,data_len)) == NULL) {
		return NULL;
	}
	if ((self = alloc_node(node_id,OP_Const,NN_PAD_NA)) == NULL) {
		tensor_free(const_tensor);
		errlog(nn,"cant alloc node");
		return NULL;
	}
	if ((outdefs = nn_calloc(1,sizeof(struct output))) == NULL) {
		tensor_free(const_tensor);
		nn_free(self);
		errlog(nn,"can't alloc out defs");
		return NULL;
	}
	outdefs[0].rank = 4;
	outdefs[0].max_sizes[0] = batches;
	outdefs[0].max_sizes[1] = height;
	outdefs[0].max_sizes[2] = width;
	outdefs[0].max_sizes[3] = depth;
	outdefs[0].elementsize = data_len/allsize;
	switch (outdefs[0].elementsize) {
	case 1: const_tensor->format.type = NN_TYPE_QUINT8; break;
	case 2: const_tensor->format.type = NN_TYPE_QUINT16; break;
	default: const_tensor->format.type = NN_TYPE_VOID; break;  // Void has size=4
	}
	self->n_inputs = 0;
	self->noderefhash = 0;
	self->n_outputs = 1;
	self->outputs = &const_tensor->self;
	self->output_defs = outdefs;
	self->inputs = NULL;
	self->input_refs = NULL;
	self->executions = 0;
	self->perfcounter = 0;
	logmsg(nn,9,"DEBUG: Const node output at %p is %d*%d*%d*%d",
	       self->outputs[0],
	       self->outputs[0]->shape.batches,
	       self->outputs[0]->shape.height,
	       self->outputs[0]->shape.width,
	       self->outputs[0]->shape.depth
		);
	return self;
}


int hexagon_nn_populate_const(
	struct nn_graph *nn,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset)
{
	const struct nn_node *node;
	if ((node = get_node(nn, node_id)) == NULL){
		errlog(nn, "get node failed");
		return -1;
	}
	uint8_t *start = (uint8_t *) node->outputs[0]->data + target_offset;
	memcpy(start, data, data_len);
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
	struct nn_node *self = hexagon_nn_empty_const_ctor(nn,node_id,batches,height,width,depth,data_len);
	if (self) memcpy(self->outputs[0]->data,data,data_len);
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
	logmsg(nn,9,"const node %p dtor id=%x",self,self->node_id);
	if( self->opaque != NULL )
		nn_cpshare_decref( nn, self->opaque);
	tensor_free(self->outputs[0]);
	nn_free(self->output_defs);
	del_node_from_hash(nn,self->node_id, self);
	nn_free(self);
	return 0;
}

struct nn_node_ops nn_ops_for_Const = {
	.execute = const_execute,
	.check = const_check,
	.ctor = const_ctor,
	.dtor = const_dtor,
	// these won't be used, since we don't use the node ctor.
	// but, for completeness...
	.n_inputs = NN_IOCOUNT(0),
	.n_outputs = NN_IOCOUNT(1),
};


