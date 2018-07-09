
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
	logmsg(nn,4,"Creating id %x for const %f",new_node_id,const_float);
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
	uint8_t *srcdata = t->data;
	int i;
	int new_size = b*w*h*(d+amt);
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = nn_malloc(new_size + 128)) == NULL) return -1;
	for (i = 0, dstdata=new_data; i < b*w*h; i++) {
		memcpy(dstdata,srcdata,d);
		dstdata += d;
		srcdata += d;
		memset(dstdata,val,amt);
		dstdata += amt;
	}
	nn_free(t->data);
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
	if ((new_data = nn_malloc(new_size + 128)) == NULL) return -1;
	for (i = 0, dstdata=new_data; i < b*h; i++) {
		memcpy(dstdata,srcdata,w*d);
		dstdata += w*d;
		srcdata += w*d;
		memset(dstdata,val,amt*d);
		dstdata += amt*d;
	}
	nn_free(t->data);
	t->data = new_data;
	t->shape.width = w+amt;
	node->outputs[0]->max_size = new_size;
	return 0;
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
        // Could use: if (! check_single_consumer(nn, producer, i, consumer) return -1;
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
#define DO_DTOR(NODE,NN) ((NODE)->ops->dtor(NODE,NN))

static int try_fuse_requantization_range(struct nn_graph *nn, struct nn_node **requant_range_node_p)
{
	struct nn_node *requant_range_node = *requant_range_node_p;
	struct nn_node *requantize_node;
	struct nn_node *new_node;
	struct output new_outputs[3];
	int i;
	op_type operation = OP_QuantizeDownAndShrinkRange_32to8;
	/* Make sure the start node is the right kind */
	if (requant_range_node->node_type != OP_RequantizationRange_32) return 0;
	logmsg(nn,9,"found requantizationrange");
	/* Make sure the consumer is a requantization node and there's only one of them */
	if ((requantize_node = find_first_consumer(nn,requant_range_node,0)) == requant_range_node) return 0;
	if (check_all_outputs(nn,requant_range_node,requantize_node) != 0) return 0;
	if (requantize_node->node_type != OP_Requantize_32to8) return 0;
	/* Make sure the inputs are pointing to the right place */
	if (check_same_inputs(nn,requant_range_node,requantize_node,3) != 0) return 0;
	logmsg(nn,9,"Found matching requantize");
	/* Just to make sure we're not cheating, we will create a new op and dtor the old ones */
	for (i = 0; i < 3; i++) {
		new_outputs[i] = requantize_node->output_defs[i];
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		requantize_node->node_id,
		operation,
		requantize_node->padding,
		requant_range_node->n_inputs,
		3,
		requant_range_node->input_refs,
		new_outputs)) == NULL) return errlog(nn,"new node ctor");
	*requant_range_node_p = new_node;
	new_node->next = requantize_node->next;
	DO_DTOR(requant_range_node,nn);
	DO_DTOR(requantize_node,nn);
	logmsg(nn,3,"Changed to autorequantize node id=%x (was requantize ID)",new_node->node_id);
	return 0;
}

/* Turn Requantize with const max value followed by Relu into the same Requantize followed by ReluX */
/* EJP: FIXME: could support Requantize->ReluX where we take the min of the two max values */
static int try_make_reluX(struct nn_graph *nn, struct nn_node **requant_node_p)
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
	if (requantize_node->node_type != OP_Requantize_32to8) return 0;
	/* Find range max node */
	if (requantize_node->input_refs[4].output_idx != 0) return 0;
	max_node_id = requantize_node->input_refs[4].src_id;
	if ((max_node = find_node(nn,max_node_id)) == NULL) return 0;
	/* Make sure range max is const */
	if (max_node->node_type != OP_Const) return 0;
	/* Make sure consumer is relu and there's only one of them */
	if ((relu_node = find_first_consumer(nn,requantize_node,0)) == requantize_node) return 0;
	if (check_all_outputs(nn,requantize_node,relu_node) != 0) return 0;
	if (relu_node->node_type != OP_QuantizedRelu_8) return 0;
	logmsg(nn,9,"found matching relu");
	/* Create inputs and outputs */
	for (i = 0; i < 3; i++) {
		new_outputs[i] = relu_node->output_defs[i];
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
		new_outputs)) == NULL) return errlog(nn,"ctor fail");
	requantize_node->next = new_node;
	new_node->next = relu_node->next;
	DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Changed requantize w/ const max --> Relu to --> ReluX");
	return 0;
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
		.src_id = get_ninf_node(nn),
		.output_idx = 0,
	};
	if (qdown1_node->node_type == OP_QuantizeDownAndShrinkRange_32to8) return ninf_input;
	if (qdown1_node->node_type == OP_Requantize_32to8) return qdown1_node->input_refs[3];
	logmsg(nn,0,"Oops???");
	return ninf_input;
}

static inline struct input requantize_op_max_input(struct nn_graph *nn, struct nn_node *qdown1_node)
{
	struct input inf_input = {
		.src_id = get_inf_node(nn),
		.output_idx = 0,
	};
	if (qdown1_node->node_type == OP_QuantizeDownAndShrinkRange_32to8) return inf_input;
	if (qdown1_node->node_type == OP_Requantize_32to8) return qdown1_node->input_refs[4];
	logmsg(nn,0,"Oops???");
	return inf_input;
};

static inline struct input gen_zero_input(struct nn_graph *nn)
{
	struct input zero_input = {
		.src_id = get_zero_node(nn),
		.output_idx = 0,
	};
	return zero_input;
};

static int try_make_supernode_bias32_flavored(struct nn_graph *nn, struct nn_node **conv_node_p, op_type old_op_flavor, op_type new_op_flavor, int (*extra_checks)(struct nn_graph *, struct nn_node *))
{
	struct nn_node *conv_node = *conv_node_p;
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
	op_type operation = new_op_flavor;
	/* Make sure start node is the right kind... */
	if (conv_node->node_type != old_op_flavor) return 0;
	logmsg(nn,9,"found conv flavor %d",old_op_flavor);
	if ((extra_checks != NULL) && (extra_checks(nn,conv_node) != 0)) return 0;
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	/* Is it the right type? */
	/* Now repeat for QuantizedBiasAdd */
	if ((biasadd_node = find_first_consumer(nn,conv_node,0)) == conv_node) return 0;
	if (check_all_outputs(nn,conv_node,biasadd_node) != 0) return 0;
	if (biasadd_node->node_type != OP_QuantizedBiasAdd_32p32to32) return 0;
	/* EJP: FIXME later: make biasadd optional */
	logmsg(nn,9,"found biasadd");
	/* And repeat for QuantizeDown #1 */
	if ((qdown1_node = find_first_consumer(nn,biasadd_node,0)) == biasadd_node) return 0;
	if (check_all_outputs(nn,biasadd_node,qdown1_node) != 0) return 0;
	if (!is_requantize_op(qdown1_node)) return 0;
	min_input = requantize_op_min_input(nn,qdown1_node);
	max_input = requantize_op_max_input(nn,qdown1_node);
	logmsg(nn,9,"found qdown1");
	/* EJP: FIXME: optimize RELU to Requantize_32to8 with const 0 / INF */
	/* EJP: FIXME: optimize RELUX to Requantize_32to8 with const 0 / MAX */
	/* EJP: FIXME: allow plain autorequantize with -INF / INF */
	/* Now repeat for Relu */
	if (((relu_node = find_first_consumer(nn,qdown1_node,0)) == qdown1_node)
	|| (check_all_outputs(nn,qdown1_node,relu_node) != 0) 
	|| ((relu_node->node_type != OP_QuantizedRelu_8)
		&& (relu_node->node_type != OP_QuantizedReluX_8)
		&& (relu_node->node_type != OP_QuantizedClamp_8))) {
		relu_node = NULL;
		lastop = qdown1_node;
	} else {
		logmsg(nn,9,"found relu/clamp\n");
		lastop = relu_node;
		if (relu_node->node_type == OP_QuantizedRelu_8) {
			min_input = gen_zero_input(nn);
		} else if (relu_node->node_type == OP_QuantizedReluX_8) {
			min_input = gen_zero_input(nn);
			max_input = relu_node->input_refs[3];
		} else if (relu_node->node_type == OP_QuantizedClamp_8) {
			min_input = relu_node->input_refs[3];
			max_input = relu_node->input_refs[4];
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
		new_outputs[i] = lastop->output_defs[i];
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
		new_outputs)) == NULL) return errlog(nn,"ctor fail");
	/* Clean up the old not needed nodes */
	supernode->next = lastop->next;
	*conv_node_p = supernode;
	DO_DTOR(conv_node,nn);
	DO_DTOR(biasadd_node,nn);
	DO_DTOR(qdown1_node,nn);
	if (relu_node) DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Created supernode id=%x (was relu ID)",supernode->node_id);
	return 0;
}

static int try_make_supernode_flavored(struct nn_graph *nn, struct nn_node **conv_node_p, op_type old_op_flavor, op_type new_op_flavor, int (*extra_checks)(struct nn_graph *, struct nn_node *))
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
	op_type operation = new_op_flavor;
	/* Make sure start node is the right kind... */
	if (conv_node->node_type != old_op_flavor) return 0;
	logmsg(nn,9,"found conv flavor %d (node id=%d)",old_op_flavor,conv_node->node_id);
	if ((extra_checks != NULL) && (extra_checks(nn,conv_node) != 0)) return 0;
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	if ((qdown0_node = find_first_consumer(nn,conv_node,0)) == conv_node) return 0;
	/* Do all the ouptuts go to a single consumer? */
	if (check_all_outputs(nn,conv_node,qdown0_node) != 0) return 0;
	/* Is it the right type? */
	if (!is_requantize_op(qdown0_node)) return 0;
	logmsg(nn,9,"found qdown0");
	/* Now repeat for QuantizedBiasAdd */
	if ((biasadd_node = find_first_consumer(nn,qdown0_node,0)) == qdown0_node) {
		logmsg(nn,9,"reason: biasaddnode != qdown0_node");
		return 0;
	}
	if (check_all_outputs(nn,qdown0_node,biasadd_node) != 0) {
		logmsg(nn,9,"reason: outputs don't check");
		return 0;
	}
	if (biasadd_node->node_type != OP_QuantizedBiasAdd_8p8to32) {
		logmsg(nn,9,"reason: biasadd_node is not biasadd");
		return 0;
	}
	/* EJP: FIXME later: make biasadd optional */
	logmsg(nn,9,"found biasadd");
	/* And repeat for QuantizeDown #1 */
	if ((qdown1_node = find_first_consumer(nn,biasadd_node,0)) == biasadd_node) return 0;
	if (check_all_outputs(nn,biasadd_node,qdown1_node) != 0) return 0;
	if (!is_requantize_op(qdown1_node)) return 0;
	min_input = requantize_op_min_input(nn,qdown1_node);
	max_input = requantize_op_max_input(nn,qdown1_node);
	logmsg(nn,9,"found qdown1");
	/* EJP: FIXME: optimize RELU to Requantize_32to8 with const 0 / INF */
	/* EJP: FIXME: optimize RELUX to Requantize_32to8 with const 0 / MAX */
	/* EJP: FIXME: allow plain autorequantize with -INF / INF */
	/* Now repeat for Relu */
	if (((relu_node = find_first_consumer(nn,qdown1_node,0)) == qdown1_node)
	|| (check_all_outputs(nn,qdown1_node,relu_node) != 0) 
	|| ((relu_node->node_type != OP_QuantizedRelu_8)
		&& (relu_node->node_type != OP_QuantizedReluX_8)
		&& (relu_node->node_type != OP_QuantizedClamp_8))) {
		relu_node = NULL;
		lastop = qdown1_node;
	} else {
		logmsg(nn,9,"found relu/clamp\n");
		lastop = relu_node;
		if (relu_node->node_type == OP_QuantizedRelu_8) {
			min_input = gen_zero_input(nn);
		} else if (relu_node->node_type == OP_QuantizedReluX_8) {
			min_input = gen_zero_input(nn);
			max_input = relu_node->input_refs[3];
		} else if (relu_node->node_type == OP_QuantizedClamp_8) {
			min_input = relu_node->input_refs[3];
			max_input = relu_node->input_refs[4];
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
		new_outputs[i] = lastop->output_defs[i]; //TODO - When requantize_node->outputs is array of tensors with rank, copy rank and all sizes from it.
	}
	/* Round up depth */
	//new_outputs[0]->depth = (depth + 31) & (~31);
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
		new_outputs)) == NULL) return errlog(nn,"ctor fail");
	/* Clean up the old not needed nodes */
	supernode->next = lastop->next;
	*conv_node_p = supernode;
	DO_DTOR(conv_node,nn);
	DO_DTOR(qdown0_node,nn);
	DO_DTOR(biasadd_node,nn);
	DO_DTOR(qdown1_node,nn);
	if (relu_node) DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Created supernode id=%x (was relu ID)",supernode->node_id);
	return 0;
}

static inline int warnlog(struct nn_graph *nn, const char *msg)
{
	logmsg(nn,2,msg);
	return -1;
}

static int dwconv_extrachecks(struct nn_graph *nn, struct nn_node *conv_node)
{
	struct nn_node *weights_node;
	struct nn_node *stride_node;
	int weights_id = conv_node->input_refs[1].src_id;
	int weights_idx = conv_node->input_refs[1].output_idx;
	int stride_id = conv_node->input_refs[6].src_id;
	int stride_idx = conv_node->input_refs[6].output_idx;
	if ((weights_node = find_node(nn,weights_id)) == NULL) return warnlog(nn,"weights node not found");
	if ((stride_node = find_node(nn,stride_id)) == NULL) return warnlog(nn,"stride node not found");
	if (weights_node->node_type != OP_Const) return warnlog(nn,"weights not const");
	if (stride_node->node_type != OP_Const) return warnlog(nn,"stride not const");
	if (stride_node->outputs[stride_idx]->shape.width > 2) return warnlog(nn,"horiz stride");
	if (weights_node->output_defs[weights_idx].max_sizes[3] != 1) return warnlog(nn,"depth mult");
	if (weights_node->output_defs[weights_idx].max_sizes[1] != 3) return warnlog(nn,"filt width");
	//if ((weights_node->output_defs[weights_idx].max_sizes[2] % 32) != 0) return errlog(nn,"FIXME: for supernode, depth must be mult of 32 for now.");
	return 0;
}

static int try_make_supernode(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	return try_make_supernode_flavored(nn,conv_node_p,OP_QuantizedConv2d_8x8to32,OP_Supernode_8x8p8to8,NULL);
}

static int try_make_dwise_supernode(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	return try_make_supernode_flavored(nn,conv_node_p,OP_QuantizedDepthwiseConv2d_8x8to32,OP_DepthwiseSupernode_8x8p8to8,dwconv_extrachecks);
}

static int try_make_supernode_bias32(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	return try_make_supernode_bias32_flavored(nn,conv_node_p,OP_QuantizedConv2d_8x8to32,OP_Supernode_8x8p32to8,NULL);
}

static int try_make_dwise_supernode_bias32(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	return try_make_supernode_bias32_flavored(nn,conv_node_p,OP_QuantizedDepthwiseConv2d_8x8to32,OP_DepthwiseSupernode_8x8p32to8,dwconv_extrachecks);
}

static inline int graph_iterator(struct nn_graph *nn, int (*f)(struct nn_graph *nn, struct nn_node **trynode))
{
	struct nn_node **root;
	for (root = &nn->head;  *root != NULL; root = (*root == NULL) ? root : &((*root)->next)) {
		if (f(nn,root) != 0) return errlog(nn,"function returned error");
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

static int make_dwise_supernodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_dwise_supernode);
}

static int make_supernodes_bias32(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_supernode_bias32);
}

static int make_dwise_supernodes_bias32(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_dwise_supernode_bias32);
}

static int make_reluX_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_reluX);
}

static int try_pad_bad_supernodes(struct nn_graph *nn, struct nn_node **src_node_p)
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
	logmsg(nn,2,"found bad depth %d @ nodeid %x padding by %d to %d offset %d",
		depth,
		src_node->node_id,
		pad_amt,
		depth_pad,
		dst_filt_offset);
	if (const_depth_extend_8(src_bias,pad_amt,0) != 0) return -1;
	if (const_depth_extend_8(src_filts,pad_amt,0) != 0) return -1;
	if (const_width_extend_8(dst_filts,pad_amt,dst_filt_offset) != 0) return -1;
	src_node->outputs[0]->max_size = ((uint64_t)src_node->outputs[0]->max_size * depth_pad) / depth;
	logmsg(nn,2,"Successfully prepadded supernodes %x and %x",src_node->node_id,dst_node->node_id);
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

static int depth32_replacement_op(int try_op)
{
	if (try_op == OP_Supernode_8x8p8to8) return OP_Supernode_8x8p8to8_d32;
	if (try_op == OP_Supernode_8x8p32to8) return OP_Supernode_8x8p32to8_d32;
	if (try_op == OP_DepthwiseSupernode_8x8p8to8) return OP_DepthwiseSupernode_8x8p8to8_d32;
	if (try_op == OP_DepthwiseSupernode_8x8p32to8) return OP_DepthwiseSupernode_8x8p32to8_d32;
	if (try_op == OP_QuantizedMaxPool_8) return OP_QuantizedMaxPool_8_d32;
	if (try_op == OP_QuantizedAvgPool_8) return OP_QuantizedAvgPool_8_d32;
	if (try_op == OP_QuantizedPRelu_8) return OP_QuantizedPRelu_8_d32;
	if (try_op == OP_QuantizedLRN_8) return OP_QuantizedLRN_8_d32;
	if (try_op == OP_QuantizedAdd_8p8to8) return OP_QuantizedAdd_8p8to8_d32;
	return -1;
}

static struct nn_node *create_convert(
	struct nn_graph *nn,
	int src_id,
	int output_idx,
	struct shape outsize,
	uint32_t operation)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp = { .src_id = src_id, .output_idx = output_idx, };
	struct output outp = {
		.rank = 4,
		.max_sizes = {
			outsize.batches,
			outsize.height,
			outsize.width,
			outsize.depth,
			0,
		},
		.elementsize = 4,  /* TODO - take size as input? */
		.zero_offset = 0,
		.stepsize = 0,
	};
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,1,1,&inp,&outp);
	return new_node;
}

static struct nn_node *create_convert_to_d32(struct nn_graph *nn, int src_id, int output_idx)
{
	struct shape outsize;
	struct nn_node *producer;
	// Validate that data producer feeding this node has been initialized
	if ((producer = find_node(nn,src_id)) == NULL) {
		logmsg(nn,0,"producer not found");
		return NULL;
	}
	outsize.batches = producer->output_defs[output_idx].max_sizes[0] + MAX_PADDING_BATCHES;
	outsize.height  = producer->output_defs[output_idx].max_sizes[1] + MAX_PADDING_HEIGHT;
	outsize.width   = producer->output_defs[output_idx].max_sizes[2] + MAX_PADDING_WIDTH;
	outsize.depth   = producer->output_defs[output_idx].max_sizes[3] + MAX_PADDING_DEPTH;
	return create_convert(nn,src_id,output_idx,outsize,OP_Convert_to_d32);
}

static struct nn_node *create_convert_from_d32(struct nn_graph *nn, int src_id, int output_idx, struct shape outsize)
{
	return create_convert(nn,src_id,output_idx,outsize,OP_Convert_from_d32);
}

static struct nn_node *create_autoquantize(struct nn_graph *nn, struct nn_node *srcnode, int src_idx)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp = { .src_id = srcnode->node_id, .output_idx = src_idx, };
	struct output outp[3];
	uint32_t operation = OP_AutoQuantize;
	const struct output float_scal_output = {
		.rank = 4,
		.max_sizes[0] = 1,
		.max_sizes[1] = 1,
		.max_sizes[2] = 1,
		.max_sizes[3] = 1,
		.elementsize = 4,
	};
	/* output 0 is the same size, but elementsize 1 */
	outp[0] = srcnode->output_defs[0];
	outp[0].elementsize = 1;
	/* Ouptut 1 and 2 are float max and min */
	outp[1] = float_scal_output;
	outp[2] = float_scal_output;
	logmsg(nn,4,"creating autoquant op %x",new_node_id);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,1,3,&inp,&outp[0]);
	return new_node;
}

static struct nn_node *create_requantize(struct nn_graph *nn, struct nn_node *srcnode)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp[3] = {
		{ .src_id = srcnode->node_id, .output_idx = 0, },
		{ .src_id = srcnode->node_id, .output_idx = 1, },
		{ .src_id = srcnode->node_id, .output_idx = 2, },};
	struct output outp[3];
	const struct output float_scal_output = {
		.rank = 4,
		.max_sizes[0] = 1,
		.max_sizes[1] = 1,
		.max_sizes[2] = 1,
		.max_sizes[3] = 1,
		.elementsize = 4,
	};
	uint32_t operation = OP_QuantizeDownAndShrinkRange_32to8;
	/* output 0 is the same size, but elementsize 1 */
	outp[0] = srcnode->output_defs[0];
	outp[0].elementsize = 1;
	/* Ouptut 1 and 2 are float max and min */
	outp[1] = float_scal_output;
	outp[2] = float_scal_output;
	logmsg(nn,4,"creating requant op %x",new_node_id);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,3,3,&inp[0],&outp[0]);
	return new_node;
}

static struct nn_node *create_dequantize(struct nn_graph *nn, struct nn_node *srcnode)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp[3] = {
		{ .src_id = srcnode->node_id, .output_idx = 0, },
		{ .src_id = srcnode->node_id, .output_idx = 1, },
		{ .src_id = srcnode->node_id, .output_idx = 2, },};
	struct output outp = srcnode->output_defs[0];
	uint32_t operation = OP_Dequantize;
	/* output 0 is the same size, but elementsize 4 */
	outp.elementsize = 4;
	logmsg(nn,4,"creating dequant op %x",new_node_id);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,3,1,&inp[0],&outp);
	return new_node;
}

static int change_refs(struct nn_graph *nn, int old_id, int old_out_idx, int new_id, int new_out_idx)
{
	struct nn_node *node;
	int i;
	for (node = nn->head; node != NULL; node = node->next) {
		for (i = 0; i < node->n_inputs; i++) {
			if ((node->input_refs[i].src_id == old_id) 
				&& (node->input_refs[i].output_idx == old_out_idx)) {
				//logmsg(nn,0,"changing node %x to %x:%d",node->node_id,new_id,new_out_idx);
				node->input_refs[i].src_id = new_id;
				node->input_refs[i].output_idx = new_out_idx;
			}
		}
	}
	return 0;
}

// TODO - Move this into some appropriate shared header
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


/*
 * Find eligible op:
 * * Is op correct?
 * * Are input dims OK?
 * Create convert_to_d32 before
 * Create convert_from_d32 after
 * Change source to convert_to_d32 output
 * Point all consumers to new node 
 */

static int do_convert_concat_depth32(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *convert_to_headnode = NULL;
	struct nn_node *srcnode = *nodeptr;
	struct nn_node *convert_to_node;
	struct nn_node *convert_from_node;
	struct nn_node *new_d32node;
	struct nn_node *tmp;
	int n_inputs = (srcnode->n_inputs-1)/3;
	int i;
	int src_id;
	int src_oidx;
	struct output new_outputs[srcnode->n_outputs];
	struct input new_inputs[srcnode->n_inputs];
	int32_t new_operation = OP_QuantizedConcat_8_d32;

	logmsg(nn, 2, "Converting %s %p to %s", hexagon_nn_op_names[srcnode->node_type], srcnode, hexagon_nn_op_names[new_operation]);
	if ((srcnode->output_defs[0].max_sizes[1] == 1) && (srcnode->output_defs[0].max_sizes[2] == 1)) {
		logmsg(nn,2,"Don't try and convert 1x1xD tensors to D32 format");
		return 0;
	}

	memcpy(new_inputs,srcnode->input_refs,srcnode->n_inputs*sizeof(new_inputs[0]));
	for (i = 0; i < srcnode->n_outputs; i++) {
		new_outputs[i] = srcnode->output_defs[i];
	}
	new_outputs[0].max_sizes[0] += MAX_PADDING_BATCHES;
	new_outputs[0].max_sizes[1] += MAX_PADDING_HEIGHT;
	new_outputs[0].max_sizes[2] += MAX_PADDING_WIDTH;
	new_outputs[0].max_sizes[3] += MAX_PADDING_DEPTH;

	// Loop over the number or d32 inputs (e.g. 3 d32s if there are 10 input tensors)
	//   See above if this doesn't make sense:  local n_inputs is ~1/3 srcnode->n_inputs
	for (i = 0; i < n_inputs; i++) {
		src_id = srcnode->input_refs[1+i].src_id;
		src_oidx = srcnode->input_refs[1+i].output_idx;
		if ((convert_to_node = create_convert_to_d32(nn,src_id,src_oidx)) == NULL) {
			return errlog(nn,"Can't make new d32 node");
		}
		new_inputs[1+i].src_id = convert_to_node->node_id;
		new_inputs[1+i].output_idx = 0;
		convert_to_node->next = convert_to_headnode;
		convert_to_headnode = convert_to_node;
	}
	if (convert_to_headnode == NULL) return errlog(nn,"no conversions to d32?");

	// Create the new operation that operates in d32-format
	if ((new_d32node = optab[new_operation]->ctor(
		nn,
		srcnode->node_id,
		new_operation,
		srcnode->padding,
		srcnode->n_inputs,
		srcnode->n_outputs,
		new_inputs,
		new_outputs)) == NULL) {
		return errlog(nn,"Can't make new d32 node");
	}

	// Create the conversion back from d32 format
	if ((convert_from_node = create_convert_from_d32(
		nn,
		new_d32node->node_id,
		0,
		srcnode->outputs[0]->shape)) == NULL) {
		return errlog(nn,"Can't make convert to d32");
	}

	//
	// Stitch in the conversion-to-d32, d32-operation, conversion-from-d32
	//

	// Change input of d32 op to point to converted one
	// Because the linked-list from headnode is assembled backwards,
	//   We need to reverse the order as we attach the inputs coming from those nodes
	// The "n_inputs-i" starts from index=1 because index-0 is the 1,1,1,1 dimensions-tensor (not a depth32)
	tmp = convert_to_headnode;
	for (i = 0; i < n_inputs; i++, tmp = tmp->next) {
		new_d32node->input_refs[n_inputs-i].src_id = convert_to_node->node_id;
		new_d32node->input_refs[n_inputs-i].output_idx = 0;

		// Change inputs of d32 op to point to outputs of conversion
		new_d32node->input_refs[n_inputs-i].src_id = tmp->node_id;
		new_d32node->input_refs[n_inputs-i].output_idx = 0;
		new_d32node->inputs[n_inputs-i] = tmp->outputs[0];
	}

	change_refs(nn,srcnode->node_id,0,convert_from_node->node_id,0);
	for (tmp = convert_to_headnode; tmp->next != NULL; tmp = tmp->next) {
		// Look for end-node
	}
	tmp->next = new_d32node;
	new_d32node->next = convert_from_node;
	convert_from_node->next = srcnode->next;
	*nodeptr = convert_to_headnode;
	//logmsg(nn,0,"concat conversion ... so far so good?");
	DO_DTOR(srcnode,nn);
	return 0;
}

static int do_convert_to_short_conv(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *srcnode = *nodeptr;
	int32_t new_operation = OP_InputSupernode_8x8p8to8_outd32;
	struct nn_node *convert_from_node;
	struct nn_node *new_d32node;
	struct nn_node *stride;
	int i;
	struct output new_outputs[srcnode->n_outputs];
	struct shape d32_shape;
	struct nn_node *weights;
	int stride_width;

	// DISABLE FOR NOW
	if (0) return 0;
	if (srcnode->node_type == OP_Supernode_8x8p32to8) {
		new_operation = OP_InputSupernode_8x8p32to8_outd32;
	}

	logmsg(nn,2,"checking for %p to short conv conversion",srcnode);
	if ((srcnode->node_type != OP_Supernode_8x8p8to8) && (srcnode->node_type != OP_Supernode_8x8p32to8)) {
		return 0;
	}
	/* Ensure output depth multiple of 32 for now... */
	if ((weights = find_node(nn,srcnode->input_refs[1].src_id)) == NULL) {
		return errlog(nn,"weights not found");
	}
	if (weights->node_type != OP_Const) return errlog(nn,"supernode weights not const");
	//if ((weights->outputs[0]->shape.filt_batches % 32) != 0) return 0;
	//if (weights->outputs[0]->shape.filt_batches < 32) return 0;
	/* Ensure stride even / > 1 */
	if ((stride = find_node(nn,srcnode->input_refs[6].src_id)) == NULL) {
		return errlog(nn,"stride not found");
	}
	if (stride->node_type != OP_Const) return errlog(nn,"supernode stride not const");
	stride_width = stride->outputs[0]->shape.width;
	/* Odd strides greater than 1 are unsupported for now */
	if ((stride_width > 1) && ((stride_width & 1) != 0)) return 0;
	//int stride_height;
	//stride_height = stride->outputs[0]->shape.height;
	//if ((stride->outputs[0]->shape.height > 1) return 0;

	/* EJP: FIXME: copy pasta */
        // TODO - HOW is all this calculation of sizing used.... Is it all correct?
	d32_shape.batches = srcnode->outputs[0]->shape.batches + MAX_PADDING_BATCHES;
	d32_shape.height  = srcnode->outputs[0]->shape.height  + MAX_PADDING_HEIGHT;
	d32_shape.width   = srcnode->outputs[0]->shape.width   + MAX_PADDING_WIDTH;
	d32_shape.depth   = srcnode->outputs[0]->shape.depth   + MAX_PADDING_DEPTH;
	logmsg(nn, 10, "Will create d32_out padded from %p to %dx%dx%dx%d",
	       srcnode->outputs[0],
	       d32_shape.batches,d32_shape.height ,d32_shape.width  ,d32_shape.depth  );
	new_outputs[0].rank = 4;
	new_outputs[0].max_sizes[0] = d32_shape.batches;
	new_outputs[0].max_sizes[1] = d32_shape.height;
	new_outputs[0].max_sizes[2] = d32_shape.width;
	new_outputs[0].max_sizes[3] = d32_shape.depth;
	new_outputs[0].max_sizes[4] = 0;
	new_outputs[0].elementsize = sizeof(char);
	new_outputs[0].zero_offset = 0;
	new_outputs[0].stepsize = 0;
	for (i = 1; i < srcnode->n_outputs; i++) {
		new_outputs[i] = srcnode->output_defs[i];
	}

	// Hack to ensure 16KB minimum allocation for this output
	// TODO - Is it a bug that we resize new_outputs[0] here, but not d32_shape, which is
	//    later used for create_convert_to_d32?
	uint32_t size_per_depth =
		new_outputs[0].elementsize *
		new_outputs[0].max_sizes[0] *
		new_outputs[0].max_sizes[1] *
		new_outputs[0].max_sizes[2];
	if ( size_per_depth * new_outputs[0].max_sizes[3] < 16 * 1024 ) {
		// round up to nearest depth for >=16KB
		new_outputs[0].max_sizes[3] = ( ((16 * 1024) + (size_per_depth - 1)) / size_per_depth);
	}

	// Create the new operation that operates in d32-format
	if ((new_d32node = optab[new_operation]->ctor(
		nn,
		srcnode->node_id,
		new_operation,
		srcnode->padding,
		srcnode->n_inputs,
		srcnode->n_outputs,
		srcnode->input_refs,
		new_outputs)) == NULL) {
		return errlog(nn,"can't make new d32node");
	}
	// Create the conversion back from d32 format
	//logmsg(nn,0,"srcnode: %d max_size: %d convert_to max_size: %d",srcnode->node_id,srcnode->outputs[0]->max_size,convert_to_node->outputs[0]->max_size);
	if ((convert_from_node = create_convert_from_d32(
		nn,
		new_d32node->node_id,
		0,
		srcnode->outputs[0]->shape)) == NULL) {
		return errlog(nn,"can't make convert from d32");
	}
	change_refs(nn,new_d32node->node_id,0,convert_from_node->node_id,0);
	new_d32node->next = convert_from_node;
	convert_from_node->next = srcnode->next;
	*nodeptr = new_d32node;
	DO_DTOR(srcnode,nn);
	return 0;
}


/*
  Take an input node (nodeptr) and transform it into a new subgraph
  which performs a similar operation but using d32-format, with
  conversion operators sandwiching the main operation.
  e.g. "Add" becomes "into_d32 -> d32_Add -> from_d32"
*/
static int do_convert_to_depth32(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *srcnode = *nodeptr;
	int32_t new_operation;
	struct nn_node *convert_to_node;
	struct nn_node *convert_to_node2 = NULL;
	struct nn_node *convert_from_node;
	struct nn_node *new_d32node;
	int i;
	int elementwise = 0;
	struct output new_outputs[srcnode->n_outputs];
	struct shape d32_shape;

	// Does this operation have a d32 version?
	if (srcnode->node_type == OP_QuantizedConcat_8) return do_convert_concat_depth32(nn,nodeptr);
	if ((new_operation = depth32_replacement_op(srcnode->node_type)) < 0) return 0;
	logmsg(nn, 2, "Converting %s %p to %s", hexagon_nn_op_names[srcnode->node_type], srcnode, hexagon_nn_op_names[new_operation]);

        // TODO - HOW is all this calculation of sizing used.... Is it all correct?
	d32_shape.batches = srcnode->outputs[0]->shape.batches + MAX_PADDING_BATCHES;
	d32_shape.height  = srcnode->outputs[0]->shape.height  + MAX_PADDING_HEIGHT;
	d32_shape.width   = srcnode->outputs[0]->shape.width   + MAX_PADDING_WIDTH;
	d32_shape.depth   = srcnode->outputs[0]->shape.depth   + MAX_PADDING_DEPTH;

	logmsg(nn, 10, "Will create d32_out padded from %p to %dx%dx%dx%d",
	       srcnode->outputs[0],
	       d32_shape.batches,d32_shape.height ,d32_shape.width  ,d32_shape.depth  );
	new_outputs[0].rank = 4;
	new_outputs[0].max_sizes[0] = d32_shape.batches;
	new_outputs[0].max_sizes[1] = d32_shape.height;
	new_outputs[0].max_sizes[2] = d32_shape.width;
	new_outputs[0].max_sizes[3] = d32_shape.depth;
	new_outputs[0].max_sizes[4] = 0;
	new_outputs[0].elementsize = sizeof(char);
	new_outputs[0].zero_offset = 0;
	new_outputs[0].stepsize = 0;
	for (i = 1; i < srcnode->n_outputs; i++) {
		new_outputs[i] = srcnode->output_defs[i];
	}

#if 0
	// Hack to ensure 16KB minimum allocation for this output
	// TODO - Is it a bug that we resize new_outputs[0] here, but not d32_shape, which is
	//    later used for create_convert_to_d32?
	uint32_t size_per_depth =
		new_outputs[0].elementsize *
		new_outputs[0].max_sizes[0] *
		new_outputs[0].max_sizes[1] *
		new_outputs[0].max_sizes[2];
	if ( size_per_depth * new_outputs[0].max_sizes[3] < 16 * 1024 ) {
		// round up to nearest depth for >=16KB
		new_outputs[0].max_sizes[3] = ( ((16 * 1024) + (size_per_depth - 1)) / size_per_depth);
	}
#endif
	elementwise = (new_operation == OP_QuantizedAdd_8p8to8_d32);
	if (new_operation == OP_QuantizedAdd_8p8to8_d32) {
		/* Try to ensure that the shapes are the same */
		struct nn_node *src0;
		struct nn_node *src1;
		if ((src0 = find_node(nn,srcnode->input_refs[0].src_id)) == NULL) return errlog(nn,"src0 not found");
		if ((src1 = find_node(nn,srcnode->input_refs[1].src_id)) == NULL) return errlog(nn,"src0 not found");
		if (src0->output_defs[0].max_sizes[0] != src1->output_defs[0].max_sizes[0]) return 0;
		if (src0->output_defs[0].max_sizes[1] != src1->output_defs[0].max_sizes[1]) return 0;
		if (src0->output_defs[0].max_sizes[2] != src1->output_defs[0].max_sizes[2]) return 0;
		if (src0->output_defs[0].max_sizes[3] != src1->output_defs[0].max_sizes[3]) return 0;
	}
	if (new_operation == OP_QuantizedLRN_8_d32) {
		struct nn_node *window;
		if ((window = find_node(nn,srcnode->input_refs[3].src_id)) == NULL) {
			return errlog(nn,"window not found");
		}
		if (window->node_type != OP_Const) return errlog(nn,"LRN window not const");
		if (window->outputs[0]->shape.width != 1) return 0;
		if (window->outputs[0]->shape.height != 1) return 0;
		
	}
	if ((new_operation == OP_Supernode_8x8p8to8_d32) || (new_operation == OP_Supernode_8x8p32to8_d32)) {
		struct nn_node *weights;
		if ((weights = find_node(nn,srcnode->input_refs[1].src_id)) == NULL) {
			return errlog(nn,"weights not found");
		}
		if (weights->node_type != OP_Const) return errlog(nn,"supernode weights not const");
#if 0
		/* EJP: hopefully depth padding will save us */
		/* SKIP when the weights are not a multiple of 32 */
		if ((weights->outputs[0]->shape.filt_depth % 32) != 0) return 0;
		if ((weights->outputs[0]->shape.filt_batches % 32) != 0) return 0;
#endif
		/* If the input depth is small, we want a different op */
		if (weights->outputs[0]->shape.filt_depth <= 4) return do_convert_to_short_conv(nn,nodeptr);
		//if (weights->outputs[0]->shape.filt_depth == 3) return do_convert_to_short_conv(nn,nodeptr);
		//if (weights->outputs[0]->shape.filt_depth < 16) return 0;
	}
	// Don't convert avgpool unless stride=1 and output_width=3
	if (new_operation == OP_QuantizedAvgPool_8_d32) {
		struct nn_node *window;
		struct nn_node *stride;
		if ((window = find_node(nn,srcnode->input_refs[3].src_id)) == NULL) {
			return errlog(nn,"src not found");
		}
		if (window->node_type != OP_Const) return errlog(nn,"avgpool window not const");
		if (window->outputs[0]->shape.width != 3) return 0;
		if (window->outputs[0]->shape.height != 3) return 0;
		if ((stride = find_node(nn,srcnode->input_refs[4].src_id)) == NULL) {
			return errlog(nn,"src not found");
		}
		if (stride->node_type != OP_Const) return errlog(nn,"avgpool stride not const");
		if (stride->outputs[0]->shape.width != 1) return 0;
	}

	// Create the new operation that operates in d32-format
	if ((new_d32node = optab[new_operation]->ctor(
		nn,
		srcnode->node_id,
		new_operation,
		srcnode->padding,
		srcnode->n_inputs,
		srcnode->n_outputs,
		srcnode->input_refs,
		new_outputs)) == NULL) {
		return errlog(nn,"can't make new d32node");
	}
	// Create the conversion into d32 format
	if ((convert_to_node = create_convert_to_d32(
		nn,
		srcnode->input_refs[0].src_id,
		srcnode->input_refs[0].output_idx)) == NULL) {
		return errlog(nn,"can't make convert to d32");
	}
	if (elementwise) {
		if ((convert_to_node2 = create_convert_to_d32(
			nn,
			srcnode->input_refs[1].src_id,
			srcnode->input_refs[1].output_idx)) == NULL) {
			return errlog(nn,"can't make convert to d32");
		}
	}
	// Create the conversion back from d32 format
	//logmsg(nn,0,"srcnode: %d max_size: %d convert_to max_size: %d",srcnode->node_id,srcnode->outputs[0]->max_size,convert_to_node->outputs[0]->max_size);
	if ((convert_from_node = create_convert_from_d32(
		nn,
		new_d32node->node_id,
		0,
		srcnode->outputs[0]->shape)) == NULL) {
		return errlog(nn,"can't make convert from d32");
	}


	/* Change input of d32 op to point to converted one */
	new_d32node->input_refs[0].src_id = convert_to_node->node_id;
	new_d32node->input_refs[0].output_idx = 0;
	if (elementwise) {
		new_d32node->input_refs[1].src_id = convert_to_node2->node_id;
		new_d32node->input_refs[1].output_idx = 0;
	}

	// Stitch in the conversion-to-d32, d32-operation, conversion-from-d32
	/* Before we stictch in the new elements, rewrite the graph to point consumers the right way */
	/* We do this before we stich in so that we don't rewrite the convert_from to point to itself */
	change_refs(nn,new_d32node->node_id,0,convert_from_node->node_id,0);
	if (convert_to_node2) {
		convert_to_node2->next = new_d32node;
		convert_to_node->next = convert_to_node2;
	} else {
		convert_to_node->next = new_d32node;
	}
	new_d32node->next = convert_from_node;
	convert_from_node->next = srcnode->next;
	*nodeptr = convert_to_node;
	DO_DTOR(srcnode,nn);
	return 0;
}

static int convert_to_depth32(struct nn_graph *nn)
{
	return graph_iterator(nn,do_convert_to_depth32);
}

static int do_mark_biasadd_node(struct nn_graph *nn, struct nn_node **add_node_p)
{
	struct nn_node *add_node = *add_node_p;
	struct nn_node *bias_val_node;
	struct nn_node *new_node;
	struct tensor *bias_val;
	struct output new_outputs[3];
	int i;
	int operation = OP_QuantizedBiasAdd_8p8to32;
	if (add_node->node_type != OP_QuantizedAdd_8p8to32) return 0;
	if ((bias_val_node = find_node(nn,add_node->input_refs[1].src_id)) == NULL) return 0;
	if (bias_val_node->node_type != OP_Const) return 0;
	bias_val = bias_val_node->outputs[0];
	if (bias_val->shape.batches != 1) return 0;
	if (bias_val->shape.height != 1) return 0;
	if (bias_val->shape.width != 1) return 0;
	/* We could probably handle add with scalar, but for now just ditch it */
	if (bias_val->shape.depth <= 1) return 0;
	for (i = 0; i < 3; i++) {
		new_outputs[i] = add_node->output_defs[i];
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		add_node->node_id,
		operation,
		add_node->padding,
		add_node->n_inputs,
		add_node->n_outputs,
		add_node->input_refs,
		new_outputs)) == NULL) return errlog(nn,"ctor fail");
	new_node->next = add_node->next;
	*add_node_p = new_node;
	DO_DTOR(add_node,nn);
	logmsg(nn,2,"Converted QAdd to QBiasAdd id=%x",new_node->node_id);
	return 0;
}

static int mark_biasadd_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,do_mark_biasadd_node);
}

static int try_make_qadd_supernode(struct nn_graph *nn, struct nn_node **qadd_node_p)
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
	if (qadd_node->node_type != OP_QuantizedAdd_8p8to32) return 0;
	logmsg(nn,4,"found add id=%x",qadd_node->node_id);
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	if ((qdown_node = find_first_consumer(nn,qadd_node,0)) == qadd_node) return 0;
	/* Do all the ouptuts go to a single consumer? */
	if (check_all_outputs(nn,qadd_node,qdown_node) != 0) return 0;
	/* Is it the right type? */
	if (!is_requantize_op(qdown_node)) return 0;
	logmsg(nn,4,"found qdown");
	/* Now repeat for Relu */
	min_input = requantize_op_min_input(nn,qdown_node);
	max_input = requantize_op_max_input(nn,qdown_node);
	logmsg(nn,4,"checking for relu");
	if (((relu_node = find_first_consumer(nn,qdown_node,0)) == qdown_node) 
	|| (check_all_outputs(nn,qdown_node,relu_node) != 0)
	|| ((relu_node->node_type != OP_QuantizedRelu_8)
		&& (relu_node->node_type != OP_QuantizedReluX_8)
		&& (relu_node->node_type != OP_QuantizedClamp_8))) {
		logmsg(nn,4,"RELU/Clamp missing");
		lastop = qdown_node;
		relu_node = NULL;
	} else {
		logmsg(nn,4,"found relu/clamp\n");
		lastop = relu_node;
		if (relu_node->node_type == OP_QuantizedRelu_8) {
			min_input = gen_zero_input(nn);
		} else if (relu_node->node_type == OP_QuantizedReluX_8) {
			min_input = gen_zero_input(nn);
			max_input = relu_node->input_refs[3];
		} else if (relu_node->node_type == OP_QuantizedClamp_8) {
			min_input = relu_node->input_refs[3];
			max_input = relu_node->input_refs[4];
		} else {
			return errlog(nn,"Oops, bad activation");
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
		new_outputs[i] = lastop->output_defs[i];
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
		new_outputs)) == NULL) return 0;
	/* Clean up the old not needed nodes */
	supernode->next = lastop->next;
	*qadd_node_p = supernode;
	DO_DTOR(qadd_node,nn);
	DO_DTOR(qdown_node,nn);
	if (relu_node) DO_DTOR(relu_node,nn);
	logmsg(nn,3,"Created qadd supernode id=%x (was relu ID)",supernode->node_id);
	return 0;
}

static int make_qadd_supernodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_qadd_supernode);
}



static int try_make_autoquantize(struct nn_graph *nn, struct nn_node **quantize_node_p)
{
	struct nn_node *quantize_node = *quantize_node_p;
	struct nn_node *min_node;
	struct nn_node *max_node;
	struct nn_node *new_node;
	op_type operation = OP_AutoQuantize;
	struct output new_outputs[3];
	int i;
	if (quantize_node->node_type != OP_Quantize) return 0;
	logmsg(nn,4,"Found quantize node id=%x",quantize_node->node_id);
	/* Try to find min and max nodes, otherwise abort */
	if ((min_node = find_node(nn,quantize_node->input_refs[1].src_id)) == NULL) return 0;
	if ((max_node = find_node(nn,quantize_node->input_refs[2].src_id)) == NULL) return 0;
	/* Check that Min and Max nodes are Min and Max ops */
	if (min_node->node_type != OP_Min_f) return 0;
	if (max_node->node_type != OP_Max_f) return 0;
	for (i = 0; i < 3; i++) {
		new_outputs[i] = quantize_node->output_defs[i];
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		quantize_node->node_id,
		operation,
		quantize_node->padding,
		1,	// inputs
		3,	// outputs
		&quantize_node->input_refs[0],
		new_outputs)) == NULL) return errlog(nn,"ctor fail");
	new_node->next = quantize_node->next;
	*quantize_node_p = new_node;
	DO_DTOR(quantize_node,nn);
	logmsg(nn,2,"Formed AutoQUantize id=%x",new_node->node_id);
	return 0;
}


static int make_autoquantize(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_autoquantize);
}

static int try_make_quantized_dwise(struct nn_graph *nn, struct nn_node **dwise_node_p)
{
	struct nn_node *dwise_node = *dwise_node_p;
	struct nn_node *input_node;
	struct nn_node *filter_node;
	struct nn_node *new_quant_in_node;
	struct nn_node *new_quant_filter_node;
	struct nn_node *new_dwise_node;
	struct nn_node *new_requant_node;
	struct nn_node *new_dequant_node;
	op_type operation = OP_QuantizedDepthwiseConv2d_8x8to32;
	struct output new_outputs[3];
	struct output float_scal_output = {
		.rank = 4,
		.max_sizes[0] = 1,
		.max_sizes[1] = 1,
		.max_sizes[2] = 1,
		.max_sizes[3] = 1,
		.elementsize = 4,
	};
	struct input new_inputs[7];
	if (dwise_node->node_type != OP_DepthwiseConv2d_f) return 0;
	if (dwise_node->n_inputs < 3) return 0;
	uint32_t input_id = dwise_node->input_refs[0].src_id;
	uint32_t input_idx = dwise_node->input_refs[0].output_idx;
	uint32_t filter_id = dwise_node->input_refs[1].src_id;
	uint32_t filter_idx = dwise_node->input_refs[1].output_idx;
	uint32_t stride_id = dwise_node->input_refs[2].src_id;
	uint32_t stride_idx = dwise_node->input_refs[2].output_idx;
	logmsg(nn,4,"Found dconv2d_f node id=%x",dwise_node->node_id);

	/* Try to find min and max nodes, otherwise abort */
	if ((input_node = find_node(nn,input_id)) == NULL) return 0;
	if ((filter_node = find_node(nn,filter_id)) == NULL) return 0;

	logmsg(nn,4,"Found input node %x filter node %x and stride node %x",
		input_node->node_id,
		filter_node->node_id,
		stride_id);

	/* Quantized ops: output 0 is same shape but element size 4 */
	new_outputs[0] = dwise_node->output_defs[0];
	new_outputs[0].elementsize = 4;

	/* Quantized ops: output 1 and 2 are min/max float vals */
	new_outputs[1] = float_scal_output;
	new_outputs[2] = float_scal_output;

	if ((new_quant_in_node = create_autoquantize(nn,input_node,input_idx)) == NULL) {
		return errlog(nn,"can't create autoquant");
	}
	if ((new_quant_filter_node = create_autoquantize(nn,filter_node,filter_idx)) == NULL) {
		return errlog(nn,"can't create autoquant");
	}
	logmsg(nn,4,"made quant in %x and quant filter %x",
		new_quant_in_node->node_id,
		new_quant_filter_node->node_id); 

	new_inputs[0].src_id = new_quant_in_node->node_id;
	new_inputs[1].src_id = new_quant_filter_node->node_id;
	new_inputs[2].src_id = new_quant_in_node->node_id;
	new_inputs[3].src_id = new_quant_in_node->node_id;
	new_inputs[4].src_id = new_quant_filter_node->node_id;
	new_inputs[5].src_id = new_quant_filter_node->node_id;
	new_inputs[6].src_id = stride_id;

	new_inputs[0].output_idx = 0;
	new_inputs[1].output_idx = 0;
	new_inputs[2].output_idx = 1;
	new_inputs[3].output_idx = 2;
	new_inputs[4].output_idx = 1;
	new_inputs[5].output_idx = 2;
	new_inputs[6].output_idx = stride_idx;
	if ((new_dwise_node = optab[operation]->ctor(
		nn,
		dwise_node->node_id,
		operation,
		dwise_node->padding,
		7,	// inputs
		3,	// outputs
		new_inputs,
		new_outputs)) == NULL) return errlog(nn,"ctor fail");

	if ((new_requant_node = create_requantize(nn,new_dwise_node)) == NULL) {
		return errlog(nn,"can't create requant");
	}
	if ((new_dequant_node = create_dequantize(nn,new_requant_node)) == NULL) {
		return errlog(nn,"can't create dequant");
	}
	change_refs(nn,dwise_node->node_id,0,new_dequant_node->node_id,0);
	new_dequant_node->next = dwise_node->next;
	new_requant_node->next = new_dequant_node;
	new_dwise_node->next = new_requant_node;
	new_quant_in_node->next = new_dwise_node;
	new_quant_filter_node->next = new_quant_in_node;
	*dwise_node_p = new_quant_filter_node;
	DO_DTOR(dwise_node,nn);
	logmsg(nn,2,"Formed QuantizedDepthwiseConv2d id=%x",new_dwise_node->node_id);
	return 0;
}


static int make_quantized_dwise(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_quantized_dwise);
}


/*
 * Find Requantize_32to8 with Const min/max --> Dequantize --> Quantize with Const Min/Max that are the same values
 */

static int try_get_const_float_val(struct nn_graph *nn, uint32_t src_id, float *val_out)
{
	struct nn_node *node;
	if ((node = find_node(nn,src_id)) == NULL) return -1;
	if (node->node_type != OP_Const) return -1;
	if (node->outputs[0]->shape.batches != 1) return -1;
	if (node->outputs[0]->shape.height != 1) return -1;
	if (node->outputs[0]->shape.width != 1) return -1;
	if (node->outputs[0]->shape.depth != 1) return -1;
	if (node->outputs[0]->max_size != 4) return -1;
	memcpy(val_out,node->outputs[0]->data,4);
	return 0;
}

static int do_remove_unnecessary_dequant_quants(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *quantize_node = *nodeptr;
	struct nn_node *dequantize_node;
	struct nn_node *requantize_node;
	float requantize_min_val;
	float requantize_max_val;
	float quantize_min_val;
	float quantize_max_val;
	uint32_t src_id;
	if (quantize_node->node_type != OP_Quantize) return 0;
	logmsg(nn,4,"Found quantize ID %x",quantize_node->node_id);
	src_id = quantize_node->input_refs[0].src_id;
	if ((dequantize_node=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (dequantize_node->node_type != OP_Dequantize) return 0;
	logmsg(nn,4,"Found dequantize ID %x",dequantize_node->node_id);
	src_id = dequantize_node->input_refs[0].src_id;
	if ((requantize_node=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (requantize_node->node_type != OP_Requantize_32to8) return 0;
	if (try_get_const_float_val(nn,requantize_node->input_refs[3].src_id,&requantize_min_val) != 0) return 0;
	if (try_get_const_float_val(nn,requantize_node->input_refs[4].src_id,&requantize_max_val) != 0) return 0;
	if (try_get_const_float_val(nn,quantize_node->input_refs[1].src_id,&quantize_min_val) != 0) return 0;
	if (try_get_const_float_val(nn,quantize_node->input_refs[2].src_id,&quantize_max_val) != 0) return 0;
	logmsg(nn,4,"requantize_min=%f requantize_max=%f quantize_min=%f quantize_max=%f",
		requantize_min_val,requantize_max_val,quantize_min_val,quantize_max_val);
	if (requantize_min_val != quantize_min_val) return 0;
	if (requantize_max_val != quantize_max_val) return 0;
	logmsg(nn,4,"changing %d refs to %d...",quantize_node->node_id,requantize_node->node_id);
	change_refs(nn,quantize_node->node_id,0,requantize_node->node_id,0);
	change_refs(nn,quantize_node->node_id,1,requantize_node->node_id,1);
	change_refs(nn,quantize_node->node_id,2,requantize_node->node_id,2);
	return 0;
}

static int remove_unnecessary_dequant_quants(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_unnecessary_dequant_quants);
}


/*
 * Find Dequantize followed by AutoQuantize
 */

static int do_remove_unnecessary_quants(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	struct nn_node *srcnode;
	uint32_t src_id;
	uint32_t srcsrc_id;
	uint32_t srcsrc_idx;
	int i;
	if (node->node_type != OP_AutoQuantize) return 0;
	src_id = node->input_refs[0].src_id;
	if ((srcnode=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (srcnode->node_type != OP_Dequantize) {
		//logmsg(nn,0,"src node %x not from_32 for to_d32 node %x",srcnode->node_id,node->node_id);
		return 0;
	}
	for (i = 0; i < 3; i++) {
		srcsrc_id = srcnode->input_refs[i].src_id;
		srcsrc_idx = srcnode->input_refs[i].output_idx;
		logmsg(nn,4,"quants: trying to convert %x:%d to %x:%d everywhere...",
			node->node_id,i,srcsrc_id,srcsrc_idx);
		if (change_refs(nn,node->node_id,i,srcsrc_id,srcsrc_idx) != 0) {
			return errlog(nn,"change refs failed");
		}
	}
	return 0;
}

static int remove_unnecessary_quants(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_unnecessary_quants);
}

/*
 * Find convert_to_d32 nodes
 * Find producer
 * If producer is not convert from D32, continue
 * Point all consumers to producer's input
 * The op should become dead
 * Producer might also become dead
 */

static int do_remove_unnecessary_d32_converts(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	struct nn_node *srcnode;
	uint32_t src_id;
	uint32_t srcsrc_id;
	uint32_t srcsrc_idx;
	if (node->node_type != OP_Convert_to_d32) return 0;
	src_id = node->input_refs[0].src_id;
	if ((srcnode=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (srcnode->node_type != OP_Convert_from_d32) {
		//logmsg(nn,0,"src node %x not from_32 for to_d32 node %x",srcnode->node_id,node->node_id);
		return 0;
	}
	srcsrc_id = srcnode->input_refs[0].src_id;
	srcsrc_idx = srcnode->input_refs[0].output_idx;
	//logmsg(nn,0,"trying to convert %x:0 to %x:%d everywhere...",node->node_id,srcsrc_id,srcsrc_idx);
	return change_refs(nn,node->node_id,0,srcsrc_id,srcsrc_idx);
}

static int remove_unnecessary_d32_converts(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_unnecessary_d32_converts);
}

static int clear_ref(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	node->refs = 0;
	return 0;
}

static int bump_refs(struct nn_graph *nn, struct nn_node **nodeptr)
{
	int i;
	struct nn_node *consumer = *nodeptr;
	struct nn_node *producer;
	int node_id;
	for (i = 0; i < consumer->n_inputs; i++) {
		node_id = consumer->input_refs[i].src_id;
		if ((producer = find_node(nn,node_id)) == NULL) {
			return errlog(nn,"can't find id 0x%x",node_id);
		}
		producer->refs++;
	}
	return 0;
}


/*
 * To update reference counts:
 * * Clear all reference counts
 * * Go through the graph.  For each input, increment reference count 
 */
static void update_refs(struct nn_graph *nn)
{
	graph_iterator(nn,clear_ref);
	graph_iterator(nn,bump_refs);
}

/*
 * Calculate reference counts
 * Find nodes that are dead:
 * * Node must have more than zero outputs (OUTPUT, PPRINT, CHECK nodes, etc)
 * * Every output has zero references
 * Remove Nodes
 * If we removed more than zero nodes, repeat... requires plumbing, just do it 8 times
 */

static int do_remove_dead_node(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	//logmsg(nn,0,"nodeptr=%p node=%p",nodeptr,node);
	if (node == NULL) return 0;
	if (likely(node->refs != 0)) return 0;
	if (node->n_outputs == 0) return 0;
	logmsg(nn,8,"freeing %p node->next=%p",node,node->next);
	*nodeptr = node->next;
	DO_DTOR(node,nn);
	return do_remove_dead_node(nn,nodeptr);
}

static int remove_dead_nodes(struct nn_graph *nn)
{

	int i;
	for (i = 0; i < 8; i++) {
		update_refs(nn);
		graph_iterator(nn,do_remove_dead_node);
	}
	return 0;
}

static int optimize(struct nn_graph *nn)
{
	int err;
	if ((err = make_autorequantize(nn)) != 0) return err;
	if ((err = make_autoquantize(nn)) != 0) return err;
	if ((err = make_quantized_dwise(nn)) != 0) return err;
	if ((err = remove_unnecessary_quants(nn)) != 0) return err;
	if ((err = remove_unnecessary_dequant_quants(nn)) != 0) return err;
	if ((err = remove_dead_nodes(nn)) != 0) return err;
	if ((err = make_reluX_nodes(nn)) != 0) return err;
	if ((err = mark_biasadd_nodes(nn)) != 0) return err;
	if ((err = make_supernodes(nn)) != 0) return err;
	if ((err = make_dwise_supernodes(nn)) != 0) return err;
	if ((err = make_supernodes_bias32(nn)) != 0) return err;
	if ((err = make_dwise_supernodes_bias32(nn)) != 0) return err;
	if ((err = make_qadd_supernodes(nn)) != 0) return err;
	if ((err = pad_bad_supernodes(nn)) != 0) return err;
	if ((err = convert_to_depth32(nn)) != 0) return err;
	if ((err = remove_unnecessary_d32_converts(nn)) != 0) return err;
	if ((err = remove_dead_nodes(nn)) != 0) return err;
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

	// append nonconst list to const list
	nn->tail->next = nn->nonconst_head;
	nn->tail = nn->nonconst_tail;
	//if ((err = run_op_setup(nn)) != 0) return err; /* FIXME: needed? Or just call ctor? */
	//vecret = h2_vecaccess_acquire(&vecstate);
	if ((err = optimize(nn)) != 0) return err;
	if ((err = allocate_graph_storage(nn)) != 0) return err;
	if ((err = prepare_inputs(nn)) != 0) return err;
	if ((err = run_op_check(nn)) != 0) return err;
	nn_os_hvx_power_off(nn);
	//h2_vecaccess_release(&vecstate,vecret.idx);
	nn->state = NN_GRAPH_PREPARED;
#ifdef SHOWY_DEBUG
	graphviz_print_graph(nn);
#endif
	return 0;
}



