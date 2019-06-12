
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

#define GROUPEDCONV_MAX_GROUPS 1024 

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
    if(NULL == (producer = find_node(nn, src_id)))
        return errlog(nn, "Cannot find parent node for OP_QuantizedGroupedConv2d_8x8p32to8!");

    /* get node for num_groups */
    uint32_t num_groups_nid = grouped_conv_node->input_refs[10].src_id;
    struct nn_node *num_groups_node = find_node_must_be_Const(nn, num_groups_nid);
    uint32_t num_groups = tensor_get_int32(num_groups_node->outputs[0], 0);

    /* some constant properties for convenience */
    const int split_num_outputs = num_groups+2;
    const int concat_num_inputs = 3*num_groups+1;
    uint32_t has_channel_scale = (grouped_conv_node->n_inputs == 14) ? 1 : 0;
    uint32_t num_core_nodes = 4;

    const int num_new_nodes = 1 + num_core_nodes*num_groups + 1; // split, G x (conv>(maybe) channel scale>biasadd>requant), concat

    /* add node for split dimension */
    const uint32_t split_dim = 3;
    const uint32_t split_dim_nid = create_const_int32_op(nn, split_dim); 

    /* Get the tensors for filter and bias */
    struct nn_node *filter_node, *bias_node, *channel_scale_node = NULL;
    if(NULL == (filter_node = find_node_must_be_Const(nn, grouped_conv_node->input_refs[1].src_id))) {
        return errlog(nn, "Unable to find const filter node!");
    }
    else if (NULL == (bias_node = find_node_must_be_Const(nn, grouped_conv_node->input_refs[2].src_id))) {
        return errlog(nn, "Unable to find const bias node!");
    }
    if (has_channel_scale) {
        if (NULL == (channel_scale_node = find_node_must_be_Const(nn, grouped_conv_node->input_refs[13].src_id))) {
            return errlog(nn, "Unable to find const channel scale node!");
        }
    }
    const struct tensor *filter_tensor = filter_node->outputs[0];
    const struct tensor *bias_tensor = bias_node->outputs[0];
    const struct tensor *channel_scale_tensor = NULL;
    if (has_channel_scale) {
        channel_scale_tensor = channel_scale_node->outputs[0];
    }

    /* Check the dimensions of inputs */
    const uint32_t in_depth = producer->outputs[grouped_conv_node->input_refs[0].output_idx]->shape.depth;
    const uint32_t out_depth = filter_tensor->shape.filt_batches;
    const uint32_t split_data_depth = in_depth / num_groups;

    uint32_t groups_limit = min_u32(min_u32(in_depth, out_depth), GROUPEDCONV_MAX_GROUPS);
    if(num_groups < 1 || num_groups > groups_limit) {
        return errlog(nn, "Bad num_groups: %d : Must be an integer in the range [1, %d]", num_groups,  groups_limit);
    }
    if(in_depth % num_groups != 0) {
        return errlog(nn, "Input depth (%d) must be divisible by num_groups(%d)!", in_depth, num_groups);
    }
    if(out_depth % num_groups != 0) {
        return errlog(nn, "Number of output channels in filter(%d) must be divisible by num_groups(%d)", out_depth, num_groups);
    }
    if(bias_tensor->shape.depth != out_depth) {
        return errlog(nn, "Depth of bias(%d) must be equal to output depth (%d)!", bias_tensor->shape.depth, out_depth);
    }
    if(filter_tensor->shape.filt_depth != split_data_depth) {
        return errlog(nn, "Depth (dimension 2) of filter should be in_depth / num_groups (%d)", split_data_depth);
    }

    /* Pre-split the filter and bias and add const nodes, if groups > 1 */
    uint32_t filter_split_nids[num_groups];
    uint32_t bias_split_nids[num_groups];
    uint32_t channel_scale_nids[num_groups];
    uint32_t i;

    if(num_groups == 1) {
        filter_split_nids[0] = filter_node->node_id;
        bias_split_nids[0] = bias_node->node_id;
        if (has_channel_scale)
            channel_scale_nids[0] = channel_scale_node->node_id;
    }
    else {
        //combine first two dimensions for split_data function
        uint32_t filter_split_height = filter_tensor->shape.filt_height * filter_tensor->shape.filt_width;
        uint32_t bias_split_height = 1;
        uint32_t channel_scale_split_height = 1;
        //size of each split
        uint32_t filter_split_size = filter_tensor->data_size / num_groups;
        uint32_t bias_split_size = bias_tensor->data_size / num_groups;
        uint32_t channel_scale_split_size;
        uint32_t channel_scale_buffer_size;
        if (has_channel_scale) {
            channel_scale_split_size = channel_scale_tensor->data_size / num_groups;
            channel_scale_buffer_size = roundf(channel_scale_split_size / sizeof(HVX_Vector) + 0.5f) * sizeof(HVX_Vector);
        }


        //size of smallest 128-byte aligned buffer to contain each split
        uint32_t filt_split_buffer_size = roundf(filter_split_size / sizeof(HVX_Vector) + 0.5f) * sizeof(HVX_Vector);
        uint32_t bias_split_buffer_size = roundf(bias_split_size * sizeof(int32_t) / sizeof(HVX_Vector) + 0.5f) * sizeof(HVX_Vector);
        uint8_t *filter_splits[num_groups], *bias_splits[num_groups], *channel_scale_splits[num_groups];

        if(nn_split_data_hvx_aligned(nn, filter_tensor->data, filter_splits, sizeof(uint8_t), filt_split_buffer_size, filter_split_height, 
            filter_tensor->shape.filt_depth, out_depth, num_groups, 2)) {
            return errlog(nn, "Failed to pre-split filter tensor!");
        } 
        if (nn_split_data_hvx_aligned(nn, bias_tensor->data, bias_splits, sizeof(int32_t), bias_split_buffer_size, bias_split_height, 
                bias_tensor->shape.width, bias_tensor->shape.depth, num_groups, 2)) {
            free_splits(filter_splits, num_groups);
            return errlog(nn, "Failed to pre-split bias tensor!");
        }
        if (has_channel_scale && nn_split_data_hvx_aligned(nn, channel_scale_tensor->data, channel_scale_splits, sizeof(float), channel_scale_buffer_size, channel_scale_split_height, 
                channel_scale_tensor->shape.width, channel_scale_tensor->shape.depth, num_groups, 2)) {
            free_splits(channel_scale_splits, num_groups);
            return errlog(nn, "Failed to pre-split channel scale tensor!");
        }

        for(i=0; i<num_groups; i++) {
           filter_split_nids[i] = nn_graph_new_internal_node_id(nn); 
           bias_split_nids[i] = nn_graph_new_internal_node_id(nn);
           channel_scale_nids[i] = nn_graph_new_internal_node_id(nn); 
           if(do_prepend_const_node(nn, filter_split_nids[i], filter_tensor->shape.filt_height, 
                filter_tensor->shape.filt_width, filter_tensor->shape.filt_depth, out_depth / num_groups, 
                filter_splits[i], filter_tensor->data_size / num_groups)) {
                free_splits(filter_splits, num_groups); free_splits(bias_splits, num_groups);
                return errlog(nn, "Failed to prepend const node for the %dth split of filter tensor", i);
            }
            if(do_prepend_const_node(nn, bias_split_nids[i], 1, 1, 1, bias_tensor->shape.depth / num_groups, 
                    bias_splits[i], bias_tensor->data_size / num_groups) ) {
                free_splits(filter_splits, num_groups); free_splits(bias_splits, num_groups);
                return errlog(nn, "Failed to prepend const node for the %dth split of bias tensor", i);
            }
            if(has_channel_scale && do_prepend_const_node(nn, channel_scale_nids[i], 1, 1, 1, channel_scale_tensor->shape.depth / num_groups, 
                    channel_scale_splits[i], channel_scale_tensor->data_size / num_groups) ) {
                free_splits(filter_splits, num_groups); free_splits(bias_splits, num_groups); free_splits(channel_scale_splits, num_groups);
                return errlog(nn, "Failed to prepend const node for the %dth split of channel scale tensor", i);
            }
        }
        // Should already be memcopyed into the const nodes' outputs, can free now
        free_splits(filter_splits, num_groups);
        free_splits(bias_splits, num_groups);
        if (has_channel_scale)
            free_splits(channel_scale_splits, num_groups);
    }

    /* Split node for data tensor - inputs */
    struct input data_split_input_refs[4];
    data_split_input_refs[0] = (struct input){ split_dim_nid, 0 }; //split dim=3
    data_split_input_refs[1] = grouped_conv_node->input_refs[0]; //data tensor
    data_split_input_refs[2] = grouped_conv_node->input_refs[3]; //data min
    data_split_input_refs[3] = grouped_conv_node->input_refs[4]; //data max

    /* Split node for data tensor - outputs */
    struct output data_split_output_defs[split_num_outputs];

    /* make output descriptors for each split node */
    for(i=0; i<num_groups; i++) {
        data_split_output_defs[i] = producer->output_defs[grouped_conv_node->input_refs[0].output_idx];
        data_split_output_defs[i].max_sizes[3] = split_data_depth;
    }

    // split output min and max
    data_split_output_defs[num_groups] = data_split_output_defs[num_groups+1] = Output_ScalarFloat;

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

    struct input channel_scale_input_refs[4];
    struct output channel_scale_output_defs[3];
    struct nn_node *channel_scale_nodes[num_groups];

    struct input biasadd_input_refs[6];
    struct output biasadd_output_defs[3];
    struct nn_node *biasadd_nodes[num_groups];

    struct input requant_input_refs[5];
    struct output requant_output_defs[3];
    struct nn_node *requant_nodes[num_groups];
    
    struct nn_node *biasdd_predecessor;

    for(i=0; i<num_groups; i++) {
        // Conv2d
        conv_input_refs[0] = (struct input){ data_split_node->node_id, i }; //data
        conv_input_refs[1] = (struct input){ filter_split_nids[i], 0 }; //filter
        conv_input_refs[2] = (struct input){ data_split_node->node_id, num_groups }; //data min
        conv_input_refs[3] = (struct input){ data_split_node->node_id, num_groups+1 }; //data max
        conv_input_refs[4] = grouped_conv_node->input_refs[5]; //filter min
        conv_input_refs[5] = grouped_conv_node->input_refs[6]; //filter max 
        conv_input_refs[6] = grouped_conv_node->input_refs[9]; //stride

        make_outputdesc_from_shape(&conv_output_defs[0], &conv_outshape, sizeof(uint32_t), 0); //conv out
        conv_output_defs[1] = conv_output_defs[2] = Output_ScalarFloat; //min and max
        conv_nodes[i] = create_node(nn, 0, OP_QuantizedConv2d_8x8to32, grouped_conv_node->padding, 7, 3, conv_input_refs, conv_output_defs); 
        biasdd_predecessor = conv_nodes[i];

        //Channel Scale
        if (has_channel_scale) {
            channel_scale_input_refs[0] = (struct input){ conv_nodes[i]->node_id, 0 }; //conv out
            channel_scale_input_refs[1] = (struct input) { channel_scale_nids[i], 0}; //channel scales
            channel_scale_input_refs[2] = (struct input){ conv_nodes[i]->node_id, 1 }; //conv min
            channel_scale_input_refs[3] = (struct input){ conv_nodes[i]->node_id, 2 }; //conv max

            make_outputdesc_from_shape(&channel_scale_output_defs[0], &conv_outshape, sizeof(int32_t), 0);
            channel_scale_output_defs[1] = channel_scale_output_defs[2] = Output_ScalarFloat; //min and max
            channel_scale_nodes[i] = create_node(nn, 0, OP_QuantizedChannelScale_32xf, NN_PAD_NA, 4, 3, channel_scale_input_refs, channel_scale_output_defs);
            biasdd_predecessor = channel_scale_nodes[i];
        }
        // BiasAdd
        biasadd_input_refs[0] = (struct input){ biasdd_predecessor->node_id, 0 }; //conv out
        biasadd_input_refs[1] = (struct input){ bias_split_nids[i], 0 }; //bias
        biasadd_input_refs[2] = (struct input){ biasdd_predecessor->node_id, 1 }; //conv min
        biasadd_input_refs[3] = (struct input){ biasdd_predecessor->node_id, 2 }; //conv max
        biasadd_input_refs[4] = grouped_conv_node->input_refs[7]; //bias min
        biasadd_input_refs[5] = grouped_conv_node->input_refs[8]; //bias max

        make_outputdesc_from_shape(&biasadd_output_defs[0], &conv_outshape, sizeof(uint32_t), 0); //biasadd out
        biasadd_output_defs[1] = biasadd_output_defs[2] = Output_ScalarFloat; //min and max
        biasadd_nodes[i] = create_node(nn, 0, OP_QuantizedBiasAdd_32p32to32, NN_PAD_NA, 6, 3, biasadd_input_refs, biasadd_output_defs);

        // Requant 32to8
        requant_input_refs[0] = (struct input){ biasadd_nodes[i]->node_id, 0 }; //biasadd out
        requant_input_refs[1] = (struct input){ biasadd_nodes[i]->node_id, 1 }; //biasadd min
        requant_input_refs[2] = (struct input){ biasadd_nodes[i]->node_id, 2 }; //biasadd max
        requant_input_refs[3] = grouped_conv_node->input_refs[11]; //speciefied requant min
        requant_input_refs[4] = grouped_conv_node->input_refs[12]; //specified requant max

        make_outputdesc_from_shape(&requant_output_defs[0], &conv_outshape, sizeof(uint8_t), 0); //requant out
        requant_output_defs[1] = requant_output_defs[2] = Output_ScalarFloat; //min and max
        requant_nodes[i] = create_node(nn, 0, OP_Requantize_32to8, NN_PAD_NA, 5, 3, requant_input_refs, requant_output_defs);
    }

    /* Concat - inputs */
    struct input concat_input_refs[concat_num_inputs];
    concat_input_refs[0] = (struct input){ split_dim_nid, 0 };

    for(i=1; i<=num_groups; i++) {
        concat_input_refs[i] = (struct input){ requant_nodes[i-1]->node_id, 0 }; //tensor pieces
        concat_input_refs[num_groups+i] = (struct input){ requant_nodes[i-1]->node_id, 1 }; //mins
        concat_input_refs[2*num_groups+i] = (struct input){ requant_nodes[i-1]->node_id, 2 }; //maxes
    }

    /* Concat - outputs */
    struct output concat_output_defs[3];
    concat_output_defs[0] = grouped_conv_node->output_defs[0];
    concat_output_defs[1] = concat_output_defs[2] = Output_ScalarFloat;

    /* Add Concat node */
    struct nn_node *concat_node = create_node(nn, 0,  OP_QuantizedConcat_8, NN_PAD_NA, concat_num_inputs, 3, concat_input_refs, concat_output_defs);

    /* Replace GroupedConv node with new sequence */
    // make the list of new nodes in sequence
    struct nn_node *new_nodes[num_new_nodes];
    new_nodes[0] = data_split_node;

    for(i=0; i<num_groups; i++) {
        new_nodes[i+1] = conv_nodes[i];
        new_nodes[i+1+num_groups] = has_channel_scale ? channel_scale_nodes[i] : NULL;
        new_nodes[i+1+2*num_groups] = biasadd_nodes[i];
        new_nodes[i+1+3*num_groups] = requant_nodes[i];
    }
    new_nodes[num_new_nodes-1] = concat_node;

    // collect new input refs for subesquent ops
    struct input new_input_refs[3] = {
        { concat_node->node_id, 0 },
        { concat_node->node_id, 1 },
        { concat_node->node_id, 2 }
    };

    // do replacement
    change_multi_output_refs_table(nn, grouped_conv_node, grouped_conv_node->node_id, num_core_nodes, new_input_refs);
    replace_node_with_sequence(nn, grouped_conv_node_p, grouped_conv_node, new_nodes, num_new_nodes);

    return 0;
}
