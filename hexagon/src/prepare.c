
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
 * This contains things applicable to the interface outside the DSP.
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <quantize.h>

// int hexagon_nn_prepare(nn_id id);

int create_const_float_op(struct nn_graph *nn, const float const_float)
{
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	if ((do_append_const_node(nn,new_node_id,1,1,1,1,(const uint8_t *)&const_float,sizeof(const_float))) != 0) {
		return errlog(nn,"whoops, can't make the right node");
	}
	return new_node_id;
}

int get_inf_node(struct nn_graph *nn)
{
	if (nn->const_inf_id != 0) return nn->const_inf_id;
	return (nn->const_inf_id = create_const_float_op(nn,INFINITY));
}

int get_ninf_node(struct nn_graph *nn)
{
	if (nn->const_ninf_id != 0) return nn->const_ninf_id;
	return (nn->const_ninf_id = create_const_float_op(nn,-INFINITY));
}

int get_zero_node(struct nn_graph *nn)
{
	if (nn->const_zero_id != 0) return nn->const_zero_id;
	return (nn->const_zero_id = create_const_float_op(nn,0.0f));
}


int const_depth_extend_8(struct nn_node *node, int amt, int val)
{
	struct tensor *t = node->outputs[0];
	int b = t->shape.batches;
	int w = t->shape.width;
	int h = t->shape.height;
	int d = t->shape.depth;
	uint8_t *srcdata = (uint8_t *)t->data;
	int i;
	int new_size = b*w*h*(d+amt);
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = (uint8_t *)malloc(new_size + 128)) == NULL) return -1;
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
	uint8_t *srcdata = (uint8_t *)t->data;
	int i;
	int new_size = b*(w+amt)*h*d;
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = (uint8_t *)malloc(new_size + 128)) == NULL) return -1;
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

int check_same_inputs(struct nn_graph *nn, struct nn_node *a, struct nn_node *b, int n_inputs)
{
	int i;
	if (a->n_inputs < n_inputs) return -1;
	if (b->n_inputs < n_inputs) return -1;
	for (i = 0; i < n_inputs; i++) {
		if (a->input_refs[i].src_id != b->input_refs[i].src_id) return -1;
		if (a->input_refs[i].output_idx != b->input_refs[i].output_idx) return -1;
	}
	return 0;
}

/*
 * EJP: we really need a new syntax here for pattern matching graph parts 
 */
#define DO_DTOR(NODE,NN) NODE->ops->dtor(NODE,NN)

static void try_fuse_requantization_range(struct nn_graph *nn, struct nn_node **requant_range_node_p)
{
	struct nn_node *requant_range_node = *requant_range_node_p;
	struct nn_node *requantize_node;
	struct nn_node *new_node;
	struct output new_outputs[3];
	int i;
	op_type operation = OP_QuantizeDownAndShrinkRange_32to8;
	/* Make sure the start node is the right kind */
	if (requant_range_node->node_type != OP_RequantizationRange_32) return;
	logmsg(nn,0,"found requantizationrange");
	/* Make sure the consumer is a requantization node and there's only one of them */
	if ((requantize_node = find_first_consumer(nn,requant_range_node,0)) == requant_range_node) return;
	if (check_all_outputs(nn,requant_range_node,requantize_node) != 0) return;
	if (requantize_node->node_type != OP_Requantize_32to8) return;
	/* Make sure the inputs are pointing to the right place */
	if (check_same_inputs(nn,requant_range_node,requantize_node,3) != 0) return;
	logmsg(nn,0,"Found matching requantize");
	/* Just to make sure we're not cheating, we will create a new op and dtor the old ones */
	for (i = 0; i < 3; i++) {
		new_outputs[i].max_size = requantize_node->outputs[i]->max_size;
		new_outputs[i].unused = 0;
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		requantize_node->node_id,
		operation,
		requantize_node->padding,
		requant_range_node->n_inputs,
		3,
		requant_range_node->input_refs,
		new_outputs)) == NULL) return;
	*requant_range_node_p = new_node;
	new_node->next = requantize_node->next;
	DO_DTOR(requant_range_node,nn);
	DO_DTOR(requantize_node,nn);
	logmsg(nn,3,"Changed to autorequantize node id=%x (was requantize ID)",new_node->node_id);
}

/* Turn Requantize with const max value followed by Relu into the same Requantize followed by ReluX */
/* EJP: FIXME: could support Requantize->ReluX where we take the min of the two max values */
static void try_make_reluX(struct nn_graph *nn, struct nn_node **requant_node_p)
{
	struct nn_node *requantize_node = *requant_node_p;
	struct nn_node *relu_node;
	struct nn_node *max_node;
	struct nn_node *new_node;
	struct input new_inputs[4];
	struct output new_outputs[3];
	uint32_t max_node_id;
	int i;
	op_type operation = OP_QuantizedReluX_8;
	if (requantize_node->node_type != OP_Requantize_32to8) return;
	/* Find range max node */
	if (requantize_node->input_refs[4].output_idx != 0) return;
	max_node_id = requantize_node->input_refs[4].src_id;
	if ((max_node = find_node(nn,max_node_id)) == NULL) return;
	/* Make sure range max is const */
	if (max_node->node_type != OP_Const) return;
	/* Make sure consumer is relu and there's only one of them */
	if ((relu_node = find_first_consumer(nn,requantize_node,0)) == requantize_node) return;
	if (check_all_outputs(nn,requantize_node,relu_node) != 0) return;
	if (relu_node->node_type != OP_QuantizedRelu_8) return;
	logmsg(nn,9,"found matching relu");
	/* Create inputs and outputs */
	for (i = 0; i < 3; i++) {
		new_outputs[i].max_size = relu_node->outputs[i]->max_size;
		new_outputs[i].unused = 0;
		new_inputs[i] = relu_node->input_refs[i];
	}
	new_inputs[3] = requantize_node->input_refs[4];
	if ((new_node = optab[operation]->ctor(
		nn,
		relu_node->node_id,
		operation,
		requantize_node->padding,
		4,
		3,
		new_inputs,
		new_outputs)) == NULL) return;
	requantize_node->next = new_node;
	new_node->next = relu_node->next;
	DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Changed requantize w/ const max --> Relu to --> ReluX");
}

static inline int is_requantize_op(const struct nn_node *node)
{
	if (node->node_type == OP_QuantizeDownAndShrinkRange_32to8) return 1;
	if (node->node_type == OP_Requantize_32to8) return 1;
	return 0;
}

static inline struct input requantize_op_min_input(struct nn_graph *nn, struct nn_node *qdown1_node)
{
	struct input ninf_input = {
		get_ninf_node(nn),
		0,
	};
	if (qdown1_node->node_type == OP_QuantizeDownAndShrinkRange_32to8) return ninf_input;
	if (qdown1_node->node_type == OP_Requantize_32to8) return qdown1_node->input_refs[3];
	logmsg(nn,0,"Oops???");
	return ninf_input;
}

static inline struct input requantize_op_max_input(struct nn_graph *nn, struct nn_node *qdown1_node)
{
	struct input inf_input = {
		get_inf_node(nn),
		0,
	};
	if (qdown1_node->node_type == OP_QuantizeDownAndShrinkRange_32to8) return inf_input;
	if (qdown1_node->node_type == OP_Requantize_32to8) return qdown1_node->input_refs[4];
	logmsg(nn,0,"Oops???");
	return inf_input;
};

static inline struct input gen_zero_input(struct nn_graph *nn)
{
	struct input zero_input = {
		get_zero_node(nn),
		0,
	};
	return zero_input;
};

static void try_make_supernode(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	struct nn_node *conv_node = *conv_node_p;
	struct nn_node *qdown0_node;
	struct nn_node *biasadd_node;
	struct nn_node *qdown1_node;
	struct nn_node *relu_node;
	struct nn_node *supernode;
	struct nn_node *lastop;
	const int num_inputs = 12;
	struct input new_inputs[num_inputs];
	struct input min_input;
	struct input max_input;
	struct output new_outputs[3];
	int i;
	op_type operation = OP_Supernode_8x8p8to8;
	/* Make sure start node is the right kind... */
	if (conv_node->node_type != OP_QuantizedConv2d_8x8to32) return;
	logmsg(nn,4,"found conv2d id=%x",conv_node->node_id);
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	if ((qdown0_node = find_first_consumer(nn,conv_node,0)) == conv_node) return;
	/* Do all the ouptuts go to a single consumer? */
	if (check_all_outputs(nn,conv_node,qdown0_node) != 0) return;
	/* Is it the right type? */
	if (!is_requantize_op(qdown0_node)) return;
	logmsg(nn,4,"found qdown0");
	/* Now repeat for QuantizedBiasAdd */
	if ((biasadd_node = find_first_consumer(nn,qdown0_node,0)) == qdown0_node) return;
	if (check_all_outputs(nn,qdown0_node,biasadd_node) != 0) return;
	if (biasadd_node->node_type != OP_QuantizedBiasAdd_8p8to32) return;
	logmsg(nn,4,"found biasadd");
	/* And repeat for QuantizeDown #1 */
	if ((qdown1_node = find_first_consumer(nn,biasadd_node,0)) == biasadd_node) return;
	if (check_all_outputs(nn,biasadd_node,qdown1_node) != 0) return;
	if (!is_requantize_op(qdown1_node)) return;
	logmsg(nn,4,"found qdown1");
	min_input = requantize_op_min_input(nn,qdown1_node);
	max_input = requantize_op_max_input(nn,qdown1_node);
	/* Now repeat for Relu */
	logmsg(nn,4,"checking for relu");
	if (((relu_node = find_first_consumer(nn,qdown1_node,0)) == qdown1_node) 
	|| (check_all_outputs(nn,qdown1_node,relu_node) != 0)
	|| ((relu_node->node_type != OP_QuantizedRelu_8)
		&& (relu_node->node_type != OP_QuantizedReluX_8))) {
		logmsg(nn,4,"RELU missing");
		lastop = qdown1_node;
		relu_node = NULL;
	} else {
		logmsg(nn,4,"found relu\n");
		lastop = relu_node;
		min_input = gen_zero_input(nn);
		if (relu_node->node_type == OP_QuantizedReluX_8) {
			max_input = relu_node->input_refs[3];
		}
	}
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
	new_inputs[10] = min_input;
	new_inputs[11] = max_input;
	/* FIXME: struct vals / ordering. */
	/* FIXME: Time to merge fastrpc branch back into master */
	for (i = 0; i < 3; i++) {
		new_outputs[i].max_size = lastop->outputs[i]->max_size;
		new_outputs[i].unused = 0;
	}
	/* Reuse the outputs & ID from the relu node */
	/* Allocate new node */
	if ((supernode = optab[operation]->ctor(
		nn,
		lastop->node_id,
		operation,
		conv_node->padding,
		num_inputs,
		3,
		new_inputs,
		new_outputs)) == NULL) return;
	/* Clean up the old not needed nodes */
	supernode->next = lastop->next;
	*conv_node_p = supernode;
	DO_DTOR(conv_node,nn);
	DO_DTOR(qdown0_node,nn);
	DO_DTOR(biasadd_node,nn);
	DO_DTOR(qdown1_node,nn);
	if (relu_node) DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Created supernode id=%x (was relu ID)",supernode->node_id);
}

static inline int graph_iterator(struct nn_graph *nn, void (*f)(struct nn_graph *nn, struct nn_node **trynode))
{
	struct nn_node **root;
	for (root = &nn->head; *root != NULL; root = &((*root)->next)) {
		//logmsg(nn,0,"root=%p *root=%p",root,*root);
		f(nn,root);
	}
	return 0;
}

static int make_autorequantize(struct nn_graph *nn)
{
	return graph_iterator(nn,try_fuse_requantization_range);
}

static int make_supernodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_supernode);
}

static int make_reluX_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_reluX);
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

static void do_mark_biasadd_node(struct nn_graph *nn, struct nn_node **add_node_p)
{
	struct nn_node *add_node = *add_node_p;
	struct nn_node *bias_val_node;
	struct nn_node *new_node;
	struct tensor *bias_val;
	struct output new_outputs[3];
	int i;
	int operation = OP_QuantizedBiasAdd_8p8to32;
	if (add_node->node_type != OP_QuantizedAdd_8p8to32) return;
	logmsg(nn,3,"found quantized add id=%x",add_node->node_id);
	if ((bias_val_node = find_node(nn,add_node->input_refs[1].src_id)) == NULL) return;
	if (bias_val_node->node_type != OP_Const) return;
	logmsg(nn,3,"found const src1");
	bias_val = bias_val_node->outputs[0];
	logmsg(nn,3,"src1 shape: %dx%dx%dx%d",
		bias_val->shape.batches,
		bias_val->shape.height,
		bias_val->shape.width,
		bias_val->shape.depth);
	if (bias_val->shape.batches != 1) return;
	if (bias_val->shape.height != 1) return;
	if (bias_val->shape.width != 1) return;
	/* We could probably handle add with scalar, but for now just ditch it */
	if (bias_val->shape.depth <= 1) return;
	logmsg(nn,3,"const src1 OK");
	for (i = 0; i < 3; i++) {
		new_outputs[i].max_size = add_node->outputs[i]->max_size;
		new_outputs[i].unused = 0;
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		add_node->node_id,
		(op_type)operation,
		add_node->padding,
		add_node->n_inputs,
		add_node->n_outputs,
		add_node->input_refs,
		new_outputs)) == NULL) {
		errlog(nn,"ctor fail");
		return;
	}
	new_node->next = add_node->next;
	*add_node_p = new_node;
	DO_DTOR(add_node,nn);
	logmsg(nn,2,"Converted QAdd to QBiasAdd id=%x",new_node->node_id);
	return;
}

static int mark_biasadd_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,do_mark_biasadd_node);
}

static void try_make_qadd_supernode(struct nn_graph *nn, struct nn_node **qadd_node_p)
{
	struct nn_node *qadd_node = *qadd_node_p;
	struct nn_node *qdown_node;
	struct nn_node *relu_node;
	struct nn_node *supernode;
	struct nn_node *lastop;
	const int num_inputs = 8;
	struct input new_inputs[num_inputs];
	struct input min_input;
	struct input max_input;
	struct output new_outputs[3];
	int i;
	op_type operation = OP_QuantizedAdd_8p8to8;
	/* Make sure start node is the right kind... */
	if (qadd_node->node_type != OP_QuantizedAdd_8p8to32) return;
	logmsg(nn,4,"found add id=%x",qadd_node->node_id);
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	if ((qdown_node = find_first_consumer(nn,qadd_node,0)) == qadd_node) return;
	/* Do all the ouptuts go to a single consumer? */
	if (check_all_outputs(nn,qadd_node,qdown_node) != 0) return;
	/* Is it the right type? */
	if (!is_requantize_op(qdown_node)) return;
	logmsg(nn,4,"found qdown");
	/* Now repeat for Relu */
	min_input = requantize_op_min_input(nn,qdown_node);
	max_input = requantize_op_max_input(nn,qdown_node);
	logmsg(nn,4,"checking for relu");
	if (((relu_node = find_first_consumer(nn,qdown_node,0)) == qdown_node) 
	|| (check_all_outputs(nn,qdown_node,relu_node) != 0)
	|| ((relu_node->node_type != OP_QuantizedRelu_8)
		&& (relu_node->node_type != OP_QuantizedReluX_8))) {
		logmsg(nn,4,"RELU missing");
		lastop = qdown_node;
		relu_node = NULL;
	} else {
		logmsg(nn,4,"found relu\n");
		lastop = relu_node;
		min_input = gen_zero_input(nn);
		if (relu_node->node_type == OP_QuantizedReluX_8) {
			max_input = relu_node->input_refs[3];
		}
	}
	/*** WOO we are a good candidate to make a supernode */
	/* 
	 * Embiggen the inputs
	 * Copy the inputs from the nodes:
	 * * all the input args from conv
	 * * Followed by values/min/max for biasadd
	 */
	for (i = 0; i < 6; i++) {
		new_inputs[i] = qadd_node->input_refs[i];
	}
	new_inputs[6] = min_input;
	new_inputs[7] = max_input;
	/* FIXME: struct vals / ordering. */
	/* FIXME: Time to merge fastrpc branch back into master */
	for (i = 0; i < 3; i++) {
		new_outputs[i].max_size = lastop->outputs[i]->max_size;
		new_outputs[i].unused = 0;
	}
	/* Reuse the outputs & ID from the relu node */
	/* Allocate new node */
	if ((supernode = optab[operation]->ctor(
		nn,
		lastop->node_id,
		operation,
		qadd_node->padding,
		num_inputs,
		3,
		new_inputs,
		new_outputs)) == NULL) return;
	/* Clean up the old not needed nodes */
	supernode->next = lastop->next;
	*qadd_node_p = supernode;
	DO_DTOR(qadd_node,nn);
	DO_DTOR(qdown_node,nn);
	if (relu_node) DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Created qadd supernode id=%x (was relu ID)",supernode->node_id);
}

static int make_qadd_supernodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_qadd_supernode);
}



#undef DO_DTOR


static int optimize(struct nn_graph *nn)
{
	int err;
	if ((err = make_autorequantize(nn)) != 0) return err;
	if ((err = make_reluX_nodes(nn)) != 0) return err;
	if ((err = mark_biasadd_nodes(nn)) != 0) return err;
	if ((err = make_supernodes(nn)) != 0) return err;
	if ((err = make_qadd_supernodes(nn)) != 0) return err;
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



