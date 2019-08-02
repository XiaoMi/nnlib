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

#include "expand_nodes.h"
#include "nn_prepare.h"

int expand_dilated_conv_nodes(struct nn_graph *nn, struct nn_node **dilated_conv_node_p)
{
    /* for graph_iterator: go back if not the right node */
    struct nn_node *dilated_conv_node = *dilated_conv_node_p;
    if(dilated_conv_node->node_type != OP_QuantizedDilatedConv2d_8x8p32to8){
        return 0;
    }

    /* find the preceeding node */
    uint32_t src_id = dilated_conv_node->input_refs[0].src_id;
    struct nn_node *producer;
    if(NULL == (producer = find_node(nn, src_id)))
        return errlog(nn, "Cannot find parent node for OP_QuantizedDilatedConv2d_8x8p32to8!");

    /* Check the sanity of inputs */
    const struct nn_node *const dilation_node = find_node_must_be_Const(nn, dilated_conv_node->input_refs[10].src_id);
    const struct tensor *const dilation_tensor = dilation_node->outputs[0]; 
    const struct nn_node *const filter_node = find_node_must_be_Const(nn, dilated_conv_node->input_refs[1].src_id);
    const struct tensor *const filter_tensor = filter_node->outputs[0];
    const struct tensor *const data_tensor = producer->outputs[dilated_conv_node->input_refs[0].output_idx];
    const int32_t dilation_h = tensor_get_int32(dilation_tensor, 0);
    const int32_t dilation_w = tensor_get_int32(dilation_tensor, 1);
    if(dilation_h < 1 || dilation_w < 1) {
        return errlog(nn, "Dilation factors must be >= 1! Current dilation: (%d, %d)", dilation_h, dilation_h);
    }

    // Only support 1x1 stride
    const struct nn_node *const stride_node = find_node_must_be_Const(nn, dilated_conv_node->input_refs[9].src_id);
    const struct tensor *const stride_tensor = stride_node->outputs[0];
    const uint32_t stride_w = stride_tensor->shape.width;
    const uint32_t stride_h = stride_tensor->shape.height;
    if(stride_w != 1 || stride_h != 1) {
        return errlog(nn, "Only 1x1 stride is supported currently! Given stride: (%d x %d)", stride_w, stride_h);
    }

    // Calculate paddings
    int32_t pad_top = (data_tensor->shape.height % dilation_h == 0) ? 0 : dilation_h - (data_tensor->shape.height % dilation_h);
    int32_t pad_left = (data_tensor->shape.width % dilation_w == 0) ? 0 : dilation_w - (data_tensor->shape.width % dilation_w);

    const uint32_t pad_crop_nid = nn_graph_new_internal_node_id(nn);
    int32_t *const pad_crop_data = (int32_t *) do_prepend_const_node_ptr(nn, pad_crop_nid, 1, 1, 2, 2, NULL, 4*sizeof(uint32_t))->data;
    pad_crop_data[0] = pad_top;
    pad_crop_data[1] = 0;
    pad_crop_data[2] = pad_left;
    pad_crop_data[3] = 0;

    // Space to batch transformation on the input
    struct input s2b_input_refs[5] = {
        dilated_conv_node->input_refs[0], //input data
        dilated_conv_node->input_refs[10], //block sizes / dilation factors
        { pad_crop_nid, 0 }, //padding
        dilated_conv_node->input_refs[3], //data min
        dilated_conv_node->input_refs[4] //data max
    };

    struct shape s2b_out_shape = {
        .batches = dilation_h * dilation_w * data_tensor->shape.batches,
        .height = (data_tensor->shape.height + pad_top) / dilation_h,
        .width = (data_tensor->shape.width + pad_left) / dilation_w,
        .depth = data_tensor->shape.depth
    };

    struct output s2b_output_defs[3];
    make_outputdesc_from_shape(&s2b_output_defs[0], &s2b_out_shape, sizeof(uint8_t), 0);
    s2b_output_defs[1] = s2b_output_defs[2] = Output_ScalarFloat;

    struct nn_node *s2b_node = create_node(nn, 0, OP_SpaceToBatchND_8, NN_PAD_NA, 5, 3, s2b_input_refs, s2b_output_defs);

    // Convoluton
    struct input conv_input_refs[7] = {
        { s2b_node->node_id, 0 }, //s2b out
        dilated_conv_node->input_refs[1], //filter
        { s2b_node->node_id, 1 }, //s2b out min
        { s2b_node->node_id, 2 }, //s2b out max
        dilated_conv_node->input_refs[5], //filter min
        dilated_conv_node->input_refs[6], //filter max
        dilated_conv_node->input_refs[9] //strides
    };

    struct shape conv_out_shape = {
        .batches = s2b_out_shape.batches,
        .height = (dilated_conv_node->padding == NN_PAD_SAME) ? s2b_out_shape.height / stride_h : (s2b_out_shape.height - filter_tensor->shape.filt_height) / stride_h + 1, 
        .width = (dilated_conv_node->padding == NN_PAD_SAME) ? s2b_out_shape.width / stride_h : (s2b_out_shape.width - filter_tensor->shape.filt_width) / stride_w + 1, 
        .depth = filter_tensor->shape.filt_batches
    };

    struct output conv_output_defs[3];
    make_outputdesc_from_shape(&conv_output_defs[0], &conv_out_shape, sizeof(uint32_t), 0);
    conv_output_defs[1] = conv_output_defs[2] = Output_ScalarFloat; //min and max
    struct nn_node* conv_node = create_node(nn, 0, OP_QuantizedConv2d_8x8to32, dilated_conv_node->padding, 7, 3, conv_input_refs, conv_output_defs);
    struct nn_node *biasadd_predecessor = conv_node;

    uint32_t has_channel_scale = (dilated_conv_node->n_inputs == 14) ? 1 : 0;
    struct nn_node *channel_scale_node = NULL;
    if (has_channel_scale) {
            struct input channel_scale_input_refs[4];
            struct output channel_scale_output_defs[3];
            channel_scale_input_refs[0] = (struct input){ conv_node->node_id, 0 }; //conv out
            channel_scale_input_refs[1] = dilated_conv_node->input_refs[13]; //channel scales
            channel_scale_input_refs[2] = (struct input){ conv_node->node_id, 1 }; //conv min
            channel_scale_input_refs[3] = (struct input){ conv_node->node_id, 2 }; //conv max

            make_outputdesc_from_shape(&channel_scale_output_defs[0], &conv_out_shape, sizeof(int32_t), 0);
            channel_scale_output_defs[1] = channel_scale_output_defs[2] = Output_ScalarFloat; //min and max
            channel_scale_node = create_node(nn, 0, OP_QuantizedChannelScale_32xf, NN_PAD_NA, 4, 3, channel_scale_input_refs, channel_scale_output_defs);
            biasadd_predecessor = channel_scale_node;
        }

    // BiasAdd
    struct input biasadd_input_refs[6] = {
        { biasadd_predecessor->node_id, 0 }, //conv out
        dilated_conv_node->input_refs[2], //bias 
        { biasadd_predecessor->node_id, 1 }, //conv out min
        { biasadd_predecessor->node_id, 2 }, //conv out max
        dilated_conv_node->input_refs[7], //bias min
        dilated_conv_node->input_refs[8] //bias max
    };

    struct output biasadd_output_defs[3] = {
        conv_output_defs[0], //same shape & data_size as conv out
        Output_ScalarFloat, //min
        Output_ScalarFloat //max
    };

    struct nn_node *biasadd_node = create_node(nn, 0, OP_QuantizedBiasAdd_32p32to32, NN_PAD_NA, 6, 3, biasadd_input_refs, biasadd_output_defs);

    // Requantize 32bit to 8bit 
    struct input requant_input_refs[5] = {
        { biasadd_node->node_id, 0 }, //biasadd out
        { biasadd_node->node_id, 1 }, //biasadd out min
        { biasadd_node->node_id, 2 }, //biasadd out max
        dilated_conv_node->input_refs[11], //requant output min
        dilated_conv_node->input_refs[12], //requant output max
    };

    struct output requant_output_defs[3];
    make_outputdesc_from_shape(&requant_output_defs[0], &conv_out_shape, sizeof(uint8_t), 0);
    requant_output_defs[1] = requant_output_defs[2] = Output_ScalarFloat;

    struct nn_node *requant_node = create_node(nn, 0, OP_Requantize_32to8, NN_PAD_NA, 5, 3, requant_input_refs, requant_output_defs);

    // Batch to Space
    struct input b2s_input_refs[5] = {
        { requant_node->node_id, 0 }, //requant out
        dilated_conv_node->input_refs[10], //block sizes
        { pad_crop_nid, 0 }, // crop paddings from s2b
        { requant_node->node_id, 1 }, //requant out min
        { requant_node->node_id, 2 } //requant out max
    };

    struct shape b2s_out_shape = {
        .batches = conv_out_shape.batches / (dilation_h * dilation_w),
        .height = conv_out_shape.height * dilation_h - pad_top,
        .width = conv_out_shape.width * dilation_w - pad_left,
        .depth = conv_out_shape.depth
    };

    struct output b2s_output_defs[3];
    make_outputdesc_from_shape(&b2s_output_defs[0], &b2s_out_shape, sizeof(uint8_t), 0);
    b2s_output_defs[1] = b2s_output_defs[2] = Output_ScalarFloat;
    struct nn_node *b2s_node = create_node(nn, 0, OP_BatchToSpaceND_8, NN_PAD_NA, 5, 3, b2s_input_refs, b2s_output_defs);

    struct nn_node *new_nodes[6];
    new_nodes[0] = s2b_node;
    new_nodes[1] = conv_node;
    new_nodes[2] = channel_scale_node;
    new_nodes[3] = biasadd_node;
    new_nodes[4] = requant_node;
    new_nodes[5] = b2s_node;

    struct input new_input_refs[3] = {
        { b2s_node->node_id, 0 },
        { b2s_node->node_id, 1 },
        { b2s_node->node_id, 2 }
    };

    change_multi_output_refs_table(nn, dilated_conv_node, dilated_conv_node->node_id, 3, new_input_refs);
    replace_node_with_sequence(nn, dilated_conv_node_p, dilated_conv_node, new_nodes, 6);

    return 0;
}
