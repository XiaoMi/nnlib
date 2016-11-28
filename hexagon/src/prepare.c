
/*
 * Copyright (c) 2016, The Linux Foundation. All rights reserved.
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
 * This contains things applicable to the interface outside the DSP.
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <quantize.h>

// int hexagon_nn_prepare(nn_id id);


int const_depth_extend_8(struct nn_node *node, int amt, int val)
{
	struct tensor *t = node->outputs[0];
	int b = t->shape.batches;
	int w = t->shape.width;
	int h = t->shape.height;
	int d = t->shape.depth;
	uint8_t *srcdata = t->data;
	int i;
	int new_size = b*w*h*(d+amt);
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = malloc(new_size + 128)) == NULL) return -1;
	for (i = 0, dstdata=new_data; i < b*w*h; i++) {
		memcpy(dstdata,srcdata,d);
		dstdata += d;
		srcdata += d;
		memset(dstdata,val,amt);
		dstdata += amt;
	}
	free(t->data);
	t->data = new_data;
	t->shape.depth = d+amt;
	node->outputs[0]->max_size = new_size;
	return 0;
}

int const_width_extend_8(struct nn_node *node, int amt, int val)
{
	struct tensor *t = node->outputs[0];
	int b = t->shape.batches;
	int w = t->shape.width;
	int h = t->shape.height;
	int d = t->shape.depth;
	uint8_t *srcdata = t->data;
	int i;
	int new_size = b*(w+amt)*h*d;
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = malloc(new_size + 128)) == NULL) return -1;
	for (i = 0, dstdata=new_data; i < b*h; i++) {
		memcpy(dstdata,srcdata,w*d);
		dstdata += w*d;
		srcdata += w*d;
		memset(dstdata,val,amt*d);
		dstdata += amt*d;
	}
	free(t->data);
	t->data = new_data;
	t->shape.width = w+amt;
	node->outputs[0]->max_size = new_size;
	return 0;
}
/*
 * find_node returns a node that matches node_id or NULL otherwise 
 */
static inline struct nn_node *find_node(struct nn_graph *nn, uint32_t node_id)
{
	struct nn_node *node;
	for (node = nn->head; node != NULL; node = node->next) {
		if (node->node_id == node_id) break;
	}
	return node;
}

static inline int prepare_input(struct nn_graph *nn, struct nn_node *node, int i)
{
	struct input inp;
	struct nn_node *srcnode;
	inp = node->input_refs[i];
	if (inp.src_id != NODE_ID_RESERVED_CONSTANT) {
		if ((srcnode = find_node(nn,inp.src_id)) == NULL) {
			return errlog(nn,"FATAL: source node ID 0x%x from node %x not found",inp.src_id,node->node_id);
		}
		if (srcnode->n_outputs <= inp.output_idx) {
			return errlog(nn,"FATAL: Bad output %d on node ID 0x%x\n",inp.output_idx,inp.src_id);
		}
		node->inputs[i] = srcnode->outputs[inp.output_idx];
		return 0;
	} else {
		logmsg(nn,1,"nn %p node %p idx %d: not handling constant input");
		return 0;
	}
}

static int prepare_inputs(struct nn_graph *nn)
{
	struct nn_node *node;
	int i;
	int err;
	for (node = nn->head; node != NULL; node = node->next) {
		for (i = 0; i < node->n_inputs; i++) {
			if ((err = prepare_input(nn,node,i)) != 0) return err;
		}
	}
	return 0;
}
#if 0
static int run_op_setup(struct nn_graph *nn)
{
	struct nn_node *node;
	int err;
	for (node = nn->head; node != NULL; node = node->next) {
		if ((err = node->ops->setup(node)) != 0) return err;
	}
}
#endif

static int run_op_check(struct nn_graph *nn)
{
	struct nn_node *node;
	int err;
	for (node = nn->head; node != NULL; node = node->next) {
		if ((err = node->ops->check(node,nn)) != 0) return err;
	}
	return 0;
}

static inline int check_all_outputs(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	struct nn_node *consumer)
{
	int i;
	for (i = 0; i < producer->n_outputs; i++) {
		if (find_first_consumer(nn,producer,i) != consumer) return -1;
		if (find_last_consumer(nn,producer,i) != consumer) return -1;
	}
	return 0;
}

static void try_make_supernode(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	struct nn_node *conv_node = *conv_node_p;
	struct nn_node *qdown0_node;
	struct nn_node *biasadd_node;
	struct nn_node *qdown1_node;
	struct nn_node *relu_node;
	struct nn_node *supernode;
	struct input new_inputs[11];
	struct output new_outputs[3];
	int i;
	int num_inputs = 10;
	op_type operation = OP_Supernode_8x8p8to8;
	/* Make sure start node is the right kind... */
	if (conv_node->node_type != OP_QuantizedConv2d_8x8to32) return;
	logmsg(nn,9,"found conv2d\n");
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	if ((qdown0_node = find_first_consumer(nn,conv_node,0)) == conv_node) return;
	/* Do all the ouptuts go to a single consumer? */
	if (check_all_outputs(nn,conv_node,qdown0_node) != 0) return;
	/* Is it the right type? */
	if (qdown0_node->node_type != OP_QuantizeDownAndShrinkRange_32to8) return;
	logmsg(nn,9,"found qdown0\n");
	/* Now repeat for QuantizedBiasAdd */
	if ((biasadd_node = find_first_consumer(nn,qdown0_node,0)) == qdown0_node) return;
	if (check_all_outputs(nn,qdown0_node,biasadd_node) != 0) return;
	if (biasadd_node->node_type != OP_QuantizedBiasAdd_8p8to32) return;
	logmsg(nn,9,"found biasadd\n");
	/* And repeat for QuantizeDown #1 */
	if ((qdown1_node = find_first_consumer(nn,biasadd_node,0)) == biasadd_node) return;
	if (check_all_outputs(nn,biasadd_node,qdown1_node) != 0) return;
	if (qdown1_node->node_type != OP_QuantizeDownAndShrinkRange_32to8) return;
	logmsg(nn,9,"found qdown1\n");
	/* Now repeat for Relu */
	if ((relu_node = find_first_consumer(nn,qdown1_node,0)) == qdown1_node) return;
	if (check_all_outputs(nn,qdown1_node,relu_node) != 0) return;
	if ((relu_node->node_type != OP_QuantizedRelu_8)
		&& (relu_node->node_type != OP_QuantizedReluX_8)) return;
	logmsg(nn,9,"found relu\n");
	/*** WOO we are a good candidate to make a supernode */
	/* 
	 * Embiggen the inputs
	 * Copy the inputs from the nodes:
	 * * all the input args from conv
	 * * Followed by values/min/max for biasadd
	 */
	for (i = 0; i < 7; i++) {
		new_inputs[i] = conv_node->input_refs[i];
	}
	new_inputs[7] = biasadd_node->input_refs[1];
	new_inputs[8] = biasadd_node->input_refs[4];
	new_inputs[9] = biasadd_node->input_refs[5];
	if (relu_node->node_type == OP_QuantizedReluX_8) {
		num_inputs = 11;
		new_inputs[10] = relu_node->input_refs[3];
	}
	/* FIXME: struct vals / ordering. */
	/* FIXME: Time to merge fastrpc branch back into master */
	for (i = 0; i < 3; i++) {
		new_outputs[i].max_size = relu_node->outputs[i]->max_size;
		new_outputs[i].unused = 0;
	}
	/* Reuse the outputs & ID from the relu node */
	/* Allocate new node */
	if ((supernode = optab[operation]->ctor(
		nn,
		relu_node->node_id,
		operation,
		conv_node->padding,
		num_inputs,
		3,
		new_inputs,
		new_outputs)) == NULL) return;
	/* Clean up the old not needed nodes */
	*conv_node_p = supernode;
	supernode->next = relu_node->next;
#define DO_DTOR(NODE,NN) NODE->ops->dtor(NODE,NN)
	DO_DTOR(conv_node,nn);
	DO_DTOR(qdown0_node,nn);
	DO_DTOR(biasadd_node,nn);
	DO_DTOR(qdown1_node,nn);
	DO_DTOR(relu_node,nn);
#undef DO_DTOR
	logmsg(nn,3,"Created supernode id=%x (was relu ID)",supernode->node_id);
}

static int make_supernodes(struct nn_graph *nn)
{
	struct nn_node **root;
	for (root = &nn->head; *root != NULL; root = &((*root)->next)) {
		try_make_supernode(nn,root);
	}
	return 0;
}

/* static */ int try_pad_bad_supernodes(struct nn_graph *nn, struct nn_node **src_node_p)
{
	struct nn_node *src_node = *src_node_p;
	struct nn_node *dst_node;
	int depth;
	int depth_pad;
	int pad_amt;
	struct nn_node *src_filts;
	struct nn_node *src_bias;
	struct nn_node *dst_filts;
	struct nn_node *dst_filts_min;
	struct nn_node *dst_filts_max;
	float filt_min_val;
	float filt_max_val;
	int dst_filt_offset;
	/* Make sure start node is the right kind... */
	if (src_node->node_type != OP_Supernode_8x8p8to8) return 0;
	/* Find the consumer node */
	if ((dst_node = find_first_consumer(nn,src_node,0)) == src_node) return 0;
	/* Do all the ouptuts go to a single consumer? */
	if (check_all_outputs(nn,src_node,dst_node) != 0) return 0;
	/* Is it the right type? */
	if (dst_node->node_type != OP_Supernode_8x8p8to8) return 0;
	/* Don't pad if sufficiently aligned */
	if ((src_bias = find_node(nn,src_node->input_refs[7].src_id)) == NULL) return 0;
	if (((depth = src_bias->outputs[0]->shape.depth) % 32) == 0) return 0;
	/* Don't pad too much, it increases later node complexity a lot */
	/* FIXME: needs better support for quantizing back down in op_supernode. */
	//if (depth <= 48) return 0;
	//if (depth == 80) return 0;
	/* FIXME: make sure const nodes are not referenced more than once, or dup them? */
	depth_pad = (depth + 31) & (~31);
	pad_amt = depth_pad - depth;
	/* Find dst_filt_max / dst_filt_min */
	if ((src_filts = find_node(nn,src_node->input_refs[1].src_id)) == NULL) return 0;
	if ((src_bias = find_node(nn,src_node->input_refs[7].src_id)) == NULL) return 0;
	if ((dst_filts = find_node(nn,dst_node->input_refs[1].src_id)) == NULL) return 0;
	if ((dst_filts_min = find_node(nn,dst_node->input_refs[4].src_id)) == NULL) return 0;
	if ((dst_filts_max = find_node(nn,dst_node->input_refs[5].src_id)) == NULL) return 0;
	filt_min_val = tensor_get_float(dst_filts_min->outputs[0],0);
	filt_max_val = tensor_get_float(dst_filts_max->outputs[0],0);
	dst_filt_offset = quantize_uint8(0.0f,filt_min_val,filt_max_val);
	logmsg(nn,2,"found bad depth %d @ nodeid %x padding by %d to %d offset %d\n",
		depth,
		src_node->node_id,
		pad_amt,
		depth_pad,
		dst_filt_offset);
	if (const_depth_extend_8(src_bias,pad_amt,0) != 0) return -1;
	if (const_depth_extend_8(src_filts,pad_amt,0) != 0) return -1;
	if (const_width_extend_8(dst_filts,pad_amt,dst_filt_offset) != 0) return -1;
	src_node->outputs[0]->max_size = ((uint64_t)src_node->outputs[0]->max_size * depth_pad) / depth;
	logmsg(nn,2,"Successfully prepadded supernodes %x and %x\n",src_node->node_id,dst_node->node_id);
	return 0;
}

static int pad_bad_supernodes(struct nn_graph *nn)
{
	struct nn_node **root;
	for (root = &nn->head; *root != NULL; root = &((*root)->next)) {
		if (try_pad_bad_supernodes(nn,root) != 0) return errlog(nn,"bad failure");
	}
	return 0;
}

static int optimize(struct nn_graph *nn)
{
	int err;
	if ((err = make_supernodes(nn)) != 0) return err;
	if ((err = pad_bad_supernodes(nn)) != 0) return err;
	return 0;
}

int do_prepare(struct nn_graph *nn)
{
	int err;
	//h2_vecaccess_state_t vecstate;
	//h2_vecaccess_ret_t vecret;
	//h2_vecaccess_init(&vecstate,H2_VECACCESS_HVX_128);
	nn_os_hvx_power_on(nn);
	if (nn->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(nn,"prepare: Graph not under construction");
	}
	//if ((err = run_op_setup(nn)) != 0) return err; /* FIXME: needed? Or just call ctor? */
	//vecret = h2_vecaccess_acquire(&vecstate);
	if ((err = optimize(nn)) != 0) return err;
	if ((err = allocate_graph_storage(nn)) != 0) return err;
	if ((err = prepare_inputs(nn)) != 0) return err;
	if ((err = run_op_check(nn)) != 0) return err;
	nn_os_hvx_power_off(nn);
	//h2_vecaccess_release(&vecstate,vecret.idx);
	nn->state = NN_GRAPH_PREPARED;
	return 0;
}



