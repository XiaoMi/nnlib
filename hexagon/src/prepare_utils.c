
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
 */
#include <nn_prepare.h>

void __attribute__((noinline))
make_outputdesc_from_shape(struct output *outp, struct shape const *shp, int elsize, int add_d32_padding_unused)
{
	outp->rank = 4;
	outp->max_sizes[0] = shp->batches;
	outp->max_sizes[1] = shp->height;
	outp->max_sizes[2] = shp->width;
	outp->max_sizes[3] = shp->depth;
	for (int i = 4; i < (int)(sizeof(outp->max_sizes) / sizeof(outp->max_sizes[0])); i++)
	{
		outp->max_sizes[i] = 0;
	}
	outp->elementsize = elsize;
	outp->zero_offset = 0;
	outp->stepsize = 0.0f;
	//if( add_d32_padding ) output_add_d32_padding(outp);
}

void __attribute__((noinline))
shape_from_outdesc(struct shape *shp, struct output const *outp, int add_d32_padding)
{
	shp->batches = outp->max_sizes[0];
	shp->height = outp->max_sizes[1];
	shp->width = outp->max_sizes[2];
	shp->depth = outp->max_sizes[3];
	//if( add_d32_padding) shape_add_d32_padding(shp);
}

struct nn_node *create_node(
	struct nn_graph *nn,
	uint32_t node_id,			// uses this node_id; or assigns one, if this is 0.
	uint32_t operation,
	padding_type padding,
	int num_inputs,
	int num_outputs,
	struct input *input_refs,
	struct output *output_defs)
{
	if (operation >= NN_OPS_MAX)
		return NULL;
	if( node_id == 0)
		node_id = nn_graph_new_internal_node_id(nn);
	return optab[operation]->ctor(nn, node_id, operation, padding, num_inputs, num_outputs, input_refs, output_defs);
}
struct nn_node *create_convert(
	struct nn_graph *nn,
	int src_id,
	int output_idx,
	struct shape outsize,
	uint32_t operation)
{
	struct input inp = {
		.src_id = src_id,
		.output_idx = output_idx,
	};
	struct output outp;
	make_outputdesc_from_shape(&outp, &outsize, /*elsize=*/1, /*pad_d32=*/0);
	struct nn_node *new_node = create_node(nn, 0, operation, NN_PAD_NA, 1, 1, &inp, &outp);
	return new_node;
}

struct nn_node *__attribute__((noinline))
find_node_must_be_Const(struct nn_graph *nn, uint32_t node_id)
{
	return find_node_must_be(nn, node_id, OP_Const);
}

extern struct nn_node *find_node_must_be(struct nn_graph *nn, uint32_t node_id, op_type ntype);

struct nn_node *__attribute__((noinline))
find_node_must_be_Const_from_ref(struct nn_graph *nn, struct input const *iref)
{
	if (iref->output_idx != 0)
		return NULL;
	return find_node_must_be(nn, iref->src_id, OP_Const);
}

void __attribute__((cold)) free_node_array(struct nn_node **node_list, uint32_t num_nodes)
{
	for (uint32_t i = 0; i < num_nodes; i++)
	{
		if (node_list[i] != NULL)
			nn_free(node_list[i]);
	}
}
