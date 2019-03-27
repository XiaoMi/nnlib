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

#ifndef NN_PREPARE_H
#define NN_PREPARE_H 1

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains common routines used in the graph prepare step 
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <quantize.h>
#include <math.h>
#include <stdio.h>
#include "nn_oemnode.h"
#include "transpose_conv_procweights.h"
#include "nn_axis.h"

// output desc for scalar floats.
static const struct output Output_ScalarFloat = {
	.rank = 4,
	.max_sizes = {1, 1, 1, 1},
	.elementsize = 4};

void make_outputdesc_from_shape(struct output *outp, struct shape const *shp, int elsize, int add_d32_padding_unused);

// extract shape from output desc; optionally add d32 padding
void shape_from_outdesc(struct shape *shp, struct output const *outp, int add_d32_padding);

struct nn_node *create_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t operation,
	padding_type padding,
	int num_inputs,
	int num_outputs,
	struct input *input_refs,
	struct output *output_defs);

//
// currently all the uses of this are for Convert to/from d32;
// so elsize =1.
//
struct nn_node *create_convert(
	struct nn_graph *nn,
	int src_id,
	int output_idx,
	struct shape outsize,
	uint32_t operation);

// find a node, but only if node_type == ntype
inline struct nn_node *
find_node_must_be(struct nn_graph *nn, uint32_t node_id, op_type ntype)
{
	struct nn_node *n = find_node(nn, node_id);
	if (n != NULL && n->node_type == ntype)
		return n;
	return NULL;
}

// find a node, but only if node_type == OP_Const
struct nn_node *find_node_must_be_Const(struct nn_graph *nn, uint32_t node_id);
// find a const node from a struct input. returrns
// NULL if the src_id != 0.
//
struct nn_node *find_node_must_be_Const_from_ref(struct nn_graph *nn, struct input const *iref);

void free_node_array(struct nn_node **node_list, uint32_t num_nodes);
#endif //NN_PREPARE_H
