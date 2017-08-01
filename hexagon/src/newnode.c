
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
 * This contains the code to append a node.
 */

#include <nn_graph.h>
#include <stdlib.h>

struct nn_node *alloc_node(uint32_t node_id, 
	op_type operation, padding_type padding)
{
	struct nn_node *newnode;
	if ((newnode = (struct nn_node *)malloc(sizeof(*newnode))) == NULL) {
		return newnode;
	}
	newnode->node_id = node_id;
	newnode->ops = optab[operation];
	newnode->node_type = operation;
	newnode->padding = padding;
	newnode->perfcounter = 0;
	newnode->executions = 0;
	newnode->opaque = NULL;
	return newnode;
}

static inline void free_inputs(struct nn_node *node)
{
	if (node->inputs) free(node->inputs);
	if (node->input_refs) free(node->input_refs);
}

static inline int alloc_inputs(
	struct nn_graph *nn,
	struct nn_node *newnode, 
	uint32_t n, 
	const struct input *inputs)
{
	unsigned int tmpsize;
	int i;
	newnode->n_inputs = n;
	newnode->inputs = NULL;
	newnode->input_refs = NULL;
	if (n == 0) {
		return 0;
	}
	tmpsize = n*sizeof(newnode->input_refs[0]);
	/* allocate inputs */
	if ((newnode->input_refs = (struct input *)calloc(1,tmpsize)) == NULL) {
		return errlog(nn,"input refs alloc failed");
	}
	if ((newnode->inputs = (const struct tensor **)calloc(n,sizeof(void *))) == NULL) {
		free(newnode->input_refs);
		return errlog(nn,"input ptr storage alloc failed");
	}

	/* Copy input refs */
	for (i = 0; i < n; i++) {
		if (inputs[i].src_id == 0) {
			/* Or we could handle and dup tensor here */
			free(newnode->input_refs);
			free(newnode->inputs);
			return errlog(nn,"fatal: const tensor in generic input");
		}
		newnode->input_refs[i] = inputs[i];
	}
	return 0;
}

static inline void free_outputs(struct nn_node *node)
{
	int i;
	for (i = 0; i < node->n_outputs; i++) {
		node->outputs[i]->data = NULL;
		tensor_free(node->outputs[i]);
	}
	if (node->outputs) free(node->outputs);
}


static inline int alloc_outputs(
	struct nn_graph *nn,
	struct nn_node *newnode, 
	uint32_t n, 
	const struct output *outputs)
{
	int i;
	struct shape tshape;
	tshape.depth = tshape.width = tshape.height = tshape.batches = 0;
	newnode->n_outputs = n;
	if (n == 0) {
		newnode->outputs = NULL;
		return 0;
	}
	/* Allocate outputs */
	if ((newnode->outputs = (struct tensor **)calloc(n,sizeof(void *))) == NULL) {
		return errlog(nn,"output ptr storage alloc failed");
	}
	/* Allocate outputs */
	/*
	 * Allocate base tensor struct but don't allocate storage until later.
	 * We could postpone longer, but this works pretty well.
	 */
	for (i = 0; i < n; i++) {
		if ((newnode->outputs[i] = tensor_alloc(&tshape,0)) == NULL) {
			goto err_free_allocated_outputs;
		}
		newnode->outputs[i]->max_size = outputs[i].max_size;
	}
	return 0;
err_free_allocated_outputs:
	for (i = 0; i < n; i++) {
		if (newnode->outputs[i]) tensor_free(newnode->outputs[i]);
	}
	free(newnode->outputs);
	return errlog(nn,"output tensor malloc failed");
}

struct nn_node *node_alloc_common(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	struct nn_node *newnode;
	if ((newnode = alloc_node(node_id,operation,padding)) == NULL) {
		errlog(nn,"common alloc id %x malloc fail",node_id);
		return NULL;
	}
	if (alloc_inputs(nn, newnode, num_inputs, inputs) != 0) {
		errlog(nn,"input alloc failed");
		goto err_free_node;
	}
	if (alloc_outputs(nn, newnode, num_outputs, outputs) != 0) {
		errlog(nn,"output alloc failed");
		goto err_free_inputs;
	}
	return newnode;
err_free_inputs:
	free_inputs(newnode);
err_free_node:
	free(newnode);
	return NULL;
}

int node_free_common(struct nn_node *node, struct nn_graph *nn)
{
	logmsg(nn,3,"freeing node %p",node);
	free_inputs(node);
	free_outputs(node);
	free(node);
	return 0;
}

static inline void node_append(struct nn_node **ptr, struct nn_node *newnode)
{
	newnode->next = NULL;
	if (*ptr == NULL) *ptr = newnode;
	else node_append(&((*ptr)->next),newnode);
}

int do_append_node(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	/* Allocate new node */
	/* Set default parameters and ops */
	/* Call node->ctor(node) */
	struct nn_node *node;
	if ((node = optab[operation]->ctor(
		nn,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs)) == NULL) {
		return errlog(nn,"node id=0x%x ctor fail",node_id);
	}
	node_append(&(nn->head),node);
	return 0;
}

extern struct nn_node *hexagon_nn_const_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len);

int do_append_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len)
{
	/* Allocate new node */
	/* Set default parameters and ops */
	/* Call node->ctor(node) */
	struct nn_node *node;
	if ((node = hexagon_nn_const_ctor(
		nn,
		node_id,
		batches,
		height,
		width,
		depth,
		data,
		data_len)) == NULL) {
		return errlog(nn,"node id=0x%x ctor fail",node_id);
	}
	node_append(&(nn->head),node);
	return 0;
}


int do_teardown(struct nn_graph *nn)
{
	struct nn_node *node;
	struct nn_node *nextnode;
	int err;
	nn_os_workers_kill(nn);
	nn->state = NN_GRAPH_INVALID;
	node = nn->head;
	while (node != NULL) {
		nextnode = node->next;
		if ((err = node->ops->dtor(node,nn)) != 0) {
			return errlog(nn,"dtor failed in teardown");
		}
		node = nextnode;
	}
	allocator_teardown(nn);
	free(nn->scratch);
	free(nn->logbuf);
	free(nn);
	return 0;
}

