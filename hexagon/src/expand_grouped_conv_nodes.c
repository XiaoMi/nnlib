
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

#include "expand_nodes.h"
#include "nn_prepare.h"
#include "data_utils.h"

int expand_grouped_conv_nodes(struct nn_graph *nn, struct nn_node **grouped_conv_node_p)
{
    struct nn_node *grouped_conv_node = *grouped_conv_node_p;
    if(grouped_conv_node->node_type != OP_QuantizedGroupedConv2d_8x8p32to8){
        return 0;
    }

    /* find the preceeding node */
    uint32_t src_id = grouped_conv_node->input_refs[0].src_id;
    struct nn_node *producer;
    if((producer = find_node(nn, src_id)) == NULL)
        return errlog(nn, "Cannot find src_id for GroupedConv2d");

    /* get node for num_groups */
    uint32_t num_groups_nid = grouped_conv_node->input_refs[10].src_id;
    struct nn_node *num_groups_node = find_node_must_be_Const(nn, num_groups_nid);
    uint32_t num_groups = tensor_get_int32(num_groups_node->outputs[0], 0);

    /* some constant properties for convenience */
    const struct shape *constant_shape = &num_groups_node->outputs[0]->shape;
    const int split_num_outputs = num_groups+2;
    const int concat_num_inputs = 3*num_groups+1;
    const int num_new_nodes = 1 + 3*num_groups + 1; // split, G x (conv>biasadd>shrink), concat

    /* add node for split dimension */
    uint32_t split_dim = 3;
    uint32_t split_dim_nid = create_const_int32_op(nn, split_dim); 

    /* Pre-split the filter and bias and add const nodes */
    const struct tensor *filter_tensor = grouped_conv_node->inputs[1];
    const struct tensor *bias_tensor = grouped_conv_node->inputs[2];
    uint32_t filter_split_height = filter_tensor->shape.filt_height * filter_tensor->shape.filt_width;
    uint32_t bias_split_height = 1;
    if (filter_tensor->data_size % num_groups != 0 || bias_tensor->data_size % num_groups != 0)
        return errlog(nn, "Weights and biases must both be divisible by groups");
    uint32_t filter_split_size = filter_tensor->data_size / num_groups;
    uint32_t bias_split_size = bias_tensor->data_size / num_groups;
    uint8_t *filter_splits = (uint8_t *) nn_memalign(sizeof(HVX_Vector), filter_tensor->data_size);
    uint8_t *bias_splits = (uint8_t *) nn_memalign(sizeof(HVX_Vector), bias_tensor->data_size);
    if( split_data(filter_tensor->data, filter_splits, filter_split_height, filter_tensor->shape.filt_depth, filter_tensor->shape.filt_batches, num_groups, 2) &
        split_data(bias_tensor->data, bias_splits, bias_split_height, bias_tensor->shape.width, bias_tensor->shape.depth, num_groups, 2) ) {
        nn_free(filter_splits);
        nn_free(bias_splits);
        return errlog(nn, "Bias pre-split failed!");
    }

    uint32_t filter_split_nids[num_groups];
    uint32_t bias_split_nids[num_groups];
    uint32_t i;
    for(i=0; i<num_groups; i++) {
       filter_split_nids[i] = nn_graph_new_internal_node_id(nn); 
       bias_split_nids[i] = nn_graph_new_internal_node_id(nn); 
       if( do_prepend_const_node(nn, filter_split_nids[i], filter_tensor->shape.filt_height, filter_tensor->shape.filt_width, filter_tensor->shape.filt_depth, filter_tensor->shape.filt_batches / num_groups, &filter_splits[i * filter_split_size], filter_split_size) &
           do_prepend_const_node(nn, bias_split_nids[i], 1, 1, 1, bias_tensor->shape.depth / num_groups, &bias_splits[i * bias_split_size], bias_split_size) ) {
            nn_free(filter_splits);
            nn_free(bias_splits);
           return errlog(nn, "Failed to prepend const nodes for split filter and bias");
       }
    }

    /* Split node for data tensor - inputs */
    struct input data_split_input_refs[4];
    data_split_input_refs[0] = (struct input){ split_dim_nid, 0 }; //split dim=3
    data_split_input_refs[1] = grouped_conv_node->input_refs[0]; //data tensor
    data_split_input_refs[2] = grouped_conv_node->input_refs[3]; //data min
    data_split_input_refs[3] = grouped_conv_node->input_refs[4]; //data max

    /* Split node for data tensor - outputs */
    struct output data_split_output_defs[split_num_outputs];
    struct shape data_split_outshape = grouped_conv_node->inputs[0]->shape; 
    data_split_outshape.depth /= num_groups;

    /* make output descriptors for each split node */
    for(i=0; i<num_groups; i++) {
        make_outputdesc_from_shape(&data_split_output_defs[i], &data_split_outshape, sizeof(uint8_t), 0); 
    }

    //mins and maxes
    make_outputdesc_from_shape(&data_split_output_defs[num_groups], constant_shape, sizeof(float), 0);
    make_outputdesc_from_shape(&data_split_output_defs[num_groups+1], constant_shape, sizeof(float), 0);

    /* add Split node to graph */
    struct nn_node *data_split_node = create_node(nn, 0, OP_QuantizedSplit_8, NN_PAD_NA, 4, split_num_outputs, data_split_input_refs, data_split_output_defs); 

    /* (Conv2d + BiasAdd + Shrink) x num_groups */
    // all this should be converted to supernodes
    struct input conv_input_refs[7];
    struct output conv_output_defs[3];
    struct nn_node *conv_nodes[num_groups];
    struct shape conv_outshape = grouped_conv_node->outputs[0]->shape;
    if (conv_outshape.depth % num_groups != 0)
        return errlog(nn, "Conv output depth %d should be a multiple of num groups %d", conv_outshape.depth, num_groups);
    conv_outshape.depth /= num_groups;

    struct input biasadd_input_refs[6];
    struct output biasadd_output_defs[3];
    struct nn_node *biasadd_nodes[num_groups];

    struct input shrink_input_refs[3];
    struct output shrink_output_defs[3];
    struct nn_node *shrink_nodes[num_groups];

    for(i=0; i<num_groups; i++) {
        // Conv2d
        conv_input_refs[0] = (struct input){ data_split_node->node_id, i }; //data
        conv_input_refs[1] = (struct input){ filter_split_nids[i], 0 };
        conv_input_refs[2] = (struct input){ data_split_node->node_id, num_groups }; //data min
        conv_input_refs[3] = (struct input){ data_split_node->node_id, num_groups+1 }; //data max
        conv_input_refs[4] = grouped_conv_node->input_refs[5]; //filter min
        conv_input_refs[5] = grouped_conv_node->input_refs[6]; //filter max 
        conv_input_refs[6] = grouped_conv_node->input_refs[9]; //stride

        make_outputdesc_from_shape(&conv_output_defs[0], &conv_outshape, sizeof(uint32_t), 0); //conv out
        make_outputdesc_from_shape(&conv_output_defs[1], constant_shape, sizeof(float), 0); //min
        make_outputdesc_from_shape(&conv_output_defs[2], constant_shape, sizeof(float), 0); //max

        conv_nodes[i] = create_node(nn, 0, OP_QuantizedConv2d_8x8to32, NN_PAD_VALID, 7, 3, conv_input_refs, conv_output_defs); 
        // BiasAdd
        biasadd_input_refs[0] = (struct input){ conv_nodes[i]->node_id, 0 }; //conv out
        biasadd_input_refs[1] = (struct input){ bias_split_nids[i], 0 }; //bias
        biasadd_input_refs[2] = (struct input){ conv_nodes[i]->node_id, 1 }; //conv min
        biasadd_input_refs[3] = (struct input){ conv_nodes[i]->node_id, 2 }; //conv max
        biasadd_input_refs[4] = grouped_conv_node->input_refs[7]; //bias min
        biasadd_input_refs[5] = grouped_conv_node->input_refs[8]; //bias max

        make_outputdesc_from_shape(&biasadd_output_defs[0], &conv_outshape, sizeof(uint32_t), 0); //bias out
        make_outputdesc_from_shape(&biasadd_output_defs[1], constant_shape, sizeof(float), 0); //min
        make_outputdesc_from_shape(&biasadd_output_defs[2], constant_shape, sizeof(float), 0); //max

        biasadd_nodes[i] = create_node(nn, 0, OP_QuantizedBiasAdd_32p32to32, NN_PAD_NA, 6, 3, biasadd_input_refs, biasadd_output_defs);

        // Shrink 32to8
        shrink_input_refs[0] = (struct input){ biasadd_nodes[i]->node_id, 0 }; //biasadd out
        shrink_input_refs[1] = (struct input){ biasadd_nodes[i]->node_id, 1 }; //biasadd min
        shrink_input_refs[2] = (struct input){ biasadd_nodes[i]->node_id, 2 }; //biasadd max

        make_outputdesc_from_shape(&shrink_output_defs[0], &conv_outshape, sizeof(uint8_t), 0); //shrink out
        make_outputdesc_from_shape(&shrink_output_defs[1], constant_shape, sizeof(float), 0); //min
        make_outputdesc_from_shape(&shrink_output_defs[2], constant_shape, sizeof(float), 0); //max

        shrink_nodes[i] = create_node(nn, 0, OP_QuantizeDownAndShrinkRange_32to8, NN_PAD_NA, 3, 3, shrink_input_refs, shrink_output_defs);

    }

    /* Concat - inputs */
    struct input concat_input_refs[concat_num_inputs];
    concat_input_refs[0] = (struct input){ split_dim_nid, 0 };

    for(i=1; i<=num_groups; i++) {
        concat_input_refs[i] = (struct input){ shrink_nodes[i-1]->node_id, 0 }; //tensor pieces
        concat_input_refs[num_groups+i] = (struct input){ shrink_nodes[i-1]->node_id, 1 }; //mins
        concat_input_refs[2*num_groups+i] = (struct input){ shrink_nodes[i-1]->node_id, 2 }; //maxes
    }

    /* Concat - outputs */
    struct output concat_output_defs[3];
    make_outputdesc_from_shape(&concat_output_defs[0], &grouped_conv_node->outputs[0]->shape, sizeof(uint8_t), 0);
    make_outputdesc_from_shape(&concat_output_defs[1], constant_shape, sizeof(float), 0);
    make_outputdesc_from_shape(&concat_output_defs[2], constant_shape, sizeof(float), 0);

    /* Add Concat node */
    struct nn_node *concat_node = create_node(nn, 0,  OP_QuantizedConcat_8, NN_PAD_NA, concat_num_inputs, 3, concat_input_refs, concat_output_defs);

    /* Replace GroupedConv node with new sequence */
    // make the list of new nodes in sequence
    struct nn_node *new_nodes[num_new_nodes];
    new_nodes[0] = data_split_node;

    for(i=0; i<num_groups; i++) {
        new_nodes[i+1] = conv_nodes[i];
        new_nodes[i+1+num_groups] = biasadd_nodes[i];
        new_nodes[i+1+2*num_groups] = shrink_nodes[i];
    }
    new_nodes[num_new_nodes-1] = concat_node;

    // collect new input refs for subesquent ops
    struct input new_input_refs[3] = {
        { concat_node->node_id, 0 },
        { concat_node->node_id, 1 },
        { concat_node->node_id, 2 }
    };

    // do replacement
    change_multi_output_refs_table(nn, grouped_conv_node, grouped_conv_node->node_id, 3, new_input_refs);
    replace_node_with_sequence(nn, grouped_conv_node_p, grouped_conv_node, new_nodes, num_new_nodes);

    return 0;
}
