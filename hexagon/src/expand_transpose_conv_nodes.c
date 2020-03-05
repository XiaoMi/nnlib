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
#include "nn_prepare.h"
#include "expand_nodes.h"
#include <nn_graph.h>
#include <nn_graph_types.h>
#include <data_utils.h>

struct expand_transpose_conv_info
{
    int32_t pad_op;
    int32_t shuffle_op;
    int32_t conv_op;
    int32_t channel_scale_op;
    int32_t bias_add_op;
    int32_t shrink_op;
    int32_t requant_op;
    int32_t d2s_op;
    int32_t elbytes;
    int32_t elbytes_bias;
    int32_t allow_num_filter_pad;
    int32_t has_depthwise_path;
};

static const struct expand_transpose_conv_info transpose_conv_8_8_info = {
    .pad_op = OP_QuantizedPad_8,
    .shuffle_op = OP_AxisShuffle_8,
    .conv_op = OP_QuantizedConv2d_8x8to32,
    .channel_scale_op = OP_QuantizedChannelScale_32xf,
    .shrink_op = OP_QuantizeDownAndShrinkRange_32to8,
    .bias_add_op = OP_QuantizedBiasAdd_8p8to32,
    .requant_op = OP_Requantize_32to8,
    .d2s_op = OP_DepthToSpace_8,
    .elbytes = sizeof(uint8_t),
    .elbytes_bias = sizeof(uint8_t),
    .allow_num_filter_pad = 1,
    .has_depthwise_path = 1};

static const struct expand_transpose_conv_info transpose_conv_8_32_info = {
    .pad_op = OP_QuantizedPad_8,
    .shuffle_op = OP_AxisShuffle_8,
    .conv_op = OP_QuantizedConv2d_8x8to32,
    .channel_scale_op = OP_QuantizedChannelScale_32xf,
    .bias_add_op = OP_QuantizedBiasAdd_32p32to32,
    .requant_op = OP_Requantize_32to8,
    .d2s_op = OP_DepthToSpace_8,
    .elbytes = sizeof(uint8_t),
    .elbytes_bias = sizeof(int32_t),
    .allow_num_filter_pad = 1,
    .has_depthwise_path = 1};

static const struct expand_transpose_conv_info transpose_conv_16_info = {
    .pad_op = OP_QuantizedPad_u16,
    .shuffle_op = OP_AxisShuffle_16,
    .conv_op = OP_QuantizedConv2d_16x16to32,
    .channel_scale_op = OP_QuantizedChannelScale_32xf,
    .bias_add_op = OP_QuantizedBiasAdd_32p32to32,
    .requant_op = OP_Requantize_32tou16,
    .d2s_op = OP_DepthToSpace_16,
    .elbytes = sizeof(uint16_t),
    .elbytes_bias = sizeof(int32_t),
    .allow_num_filter_pad = 0,
    .has_depthwise_path = 0};


/*
 *  In all cases weights are rotated 180 degrees and Conv input is padded by (fh-1,fw-1)
 *  In all cases pad is optional and shrink op only used in 8 bit bias cases.
 *  In all depthwise cases filters are (if needed) padded to height of 2 and width of 3 or 5 or 7.
 *
 *  Case 1:
 *      No groups and stride=(1,1)
 *
 *      pad -> conv -> channel_scale -> shrink -> bias_add -> requant
 *
 *
 *  Case 2:
 *      No groups and stride>1
 *      Weights are transformed by space-to-batch by a factor of strides
 *
 *      pad -> conv -> channel_scale -> shrink -> bias_add -> requant -> depth_to_space
 *
 *
 *  Case 3:
 *      Groups and strides=(1,1)
 *      Weights are split on input depth axis
 *
 *                    -> conv -> channel_scale -> shrink -> bias_add -> requant \
 *      pad -> split -| #groups                                                  |-> concat
 *                    -> conv -> channel_scale -> shrink -> bias_add -> requant /
 *
 *
 *  Case 4:
 *      Groups and strides>1
 *      Weights are transformed by space-to-batch by a factor of strides
 *      Weights are split on input depth axis
 *      Biasesa and channel scales split along depth
 *                    -> conv -> channel_scale -> shrink -> bias_add -> requant -> depth_to_space \
 *      pad -> split -| #groups                                                                    |-> concat
 *                    -> conv -> channel_scale -> shrink -> bias_add -> requant -> depth_to_space /
 *
 *
 *  Case 5:
 *      Normal Depthwise case (in_depth==groups==out_depth) and stride=(1,1)
 *
 *     pad -> DWconv -> channel_scale -> shrink -> bias_add -> requant
 *
 *
 *  Case 6:
 *      Normal Depthwise case (in_depth==groups==out_depth) and stride>1
 *      Using subkerneling method. Weights are rearranged into 5-dimensional tensor where 5th dim is the # of subkernels.
 *      # of subkernels == stride_h*stride_w.
 *
 *            -> DWconv -> channel_scale -> shrink -> bias_add -> requant \
 *      pad -| # stride_w*stride_h                                         |-> concat  -> depth_to_space
 *            -> DWconv -> channel_scale -> shrink -> bias_add -> requant /
 *
 *
 *  Case 7:
 *      Channel Multiplier Depthwise case (in_depth==2*groups==2*out_depth) and strides=(1,1)
 *      input and weights are reshaped by:
 *          - channel shuffle with k=in_depth/2
 *          - reshape (b,h,w,d) -> (b,h,w*2,d/2)
 *      DWconv uses stride=(1,2)
 *
 *      channel_shuffle -> reshape -> pad -> DWconv -> channel_scale -> shrink -> bias_add -> requant
 *
 *
 *  Case 8:
 *      Channel Multiplier Depthwise case (in_depth==2*groups==2*out_depth) and strides>1
 *      Using subkerneling method. Weights are rearranged into 5-dimensional tensor where 5th dim is the # of subkernels.
 *      # of subkernels == stride_h*stride_w.
 *      input and weights are reshaped by:
 *          - channel shuffle with k=in_depth/2
 *          - reshape (b,h,w,d) -> (b,h,w*2,d/2)
 *      DWconv uses stride=(1,2)
 *
 *                                          -> DWconv -> channel_scale -> shrink -> bias_add -> requant \
 *      channel_shuffle -> reshape -> pad -| # stride_w*stride_h                                         |-> concat  -> depth_to_space
 *                                          -> DWconv -> channel_scale -> shrink -> bias_add -> requant /
 *
 */
int expand_transpose_conv_nodes(struct nn_graph *nn, struct nn_node **transpose_conv_node_p)
{
    struct nn_node *transpose_conv_node = *transpose_conv_node_p;
    int ntyp = transpose_conv_node->node_type;
    const struct expand_transpose_conv_info *info;
    struct input const *original_inputs = transpose_conv_node->input_refs;
    struct nn_node *producer = NULL;
    struct nn_node *channel_scale_node = NULL;
    struct nn_node *pad_node = NULL;
    struct nn_node *split_node = NULL;
    struct nn_node *reshape_node = NULL;
    struct nn_node *shuffle_node = NULL;
    struct shape input_shape;
    struct shape original_output_shape;
    struct shape conv_output_shape;
    struct shape d2s_output_shape;
    uint32_t src_id = transpose_conv_node->input_refs[0].src_id;
    uint8_t has_channel_scale = (transpose_conv_node->n_inputs >= 14) ? 1 : 0;
    uint8_t has_groups = (transpose_conv_node->n_inputs >= 15) ? 1 : 0;
    uint32_t groups = 1;
    uint32_t use_depthwise = 0;
    uint32_t use_subkerneling = 0;
    uint32_t need_reshape = 0;
    uint32_t num_subkernels = 0;
    uint32_t pad_num_filters = 0;
    uint32_t chmul = 1; // channel multipler
    uint32_t qzero = 0;
    uint8_t bias_qzero8 = 0;
    int32_t group_depth; // output depth of the conv for a group
    int32_t d2s_output_depth; // output depth after d2s operation


    if (ntyp == OP_QuantizedTransposeConv2d_8x8p8to8) {
        info = &transpose_conv_8_8_info;
    } else if (ntyp == OP_QuantizedTransposeConv2d_8x8p32to8) {
        info = &transpose_conv_8_32_info;
    } else if (ntyp == OP_QuantizedTransposeConv2d_16x16p32to16) {
        info = &transpose_conv_16_info;
    } else {
        return errlog(nn, "Unsupported transpose conv2d data type");
    }
    int32_t conv_op = info->conv_op;


    if ((producer = find_node(nn, src_id)) == NULL)
        return errlog(nn, "src id not found transpose conv stuff");
    struct nn_node *weight_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[1].src_id);
    struct nn_node *weight_min_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[4].src_id);
    struct nn_node *weight_max_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[5].src_id);
    struct nn_node *paddings_arr_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[6].src_id);
    struct nn_node *strides_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[7].src_id);
    struct nn_node *bias_data_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[8].src_id);
    struct nn_node *bias_min_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[9].src_id);
    struct nn_node *bias_max_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[10].src_id);
    if (has_channel_scale) {
        channel_scale_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[13].src_id);
    }
    if (has_groups)    {
        struct nn_node *group_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[14].src_id);
        groups = tensor_get_int32(group_node->outputs[0], 0);
    }

    input_shape = producer->outputs[transpose_conv_node->input_refs[0].output_idx]->shape;
    int32_t *paddings_arr = paddings_arr_node->outputs[0]->data;
    uint8_t *bias_data = bias_data_node->outputs[0]->data;
    uint32_t stride_h = strides_node->outputs[0]->shape.height;
    uint32_t stride_w = strides_node->outputs[0]->shape.width;
    uint32_t num_filters = weight_node->outputs[0]->shape.batches;
    uint32_t filt_height = weight_node->outputs[0]->shape.height;
    uint32_t filt_width = weight_node->outputs[0]->shape.width;
    uint32_t filt_depth = weight_node->outputs[0]->shape.depth;
    float weight_min = tensor_get_float(weight_min_node->outputs[0], 0);
    float weight_max = tensor_get_float(weight_max_node->outputs[0], 0);
    float bias_min = tensor_get_float(bias_min_node->outputs[0], 0);
    float bias_max = tensor_get_float(bias_max_node->outputs[0], 0);

    if (stride_h < 1 || stride_w < 1) return errlog(nn, "Bad strides %d %d for transpose conv node", stride_h, stride_w);
    uint32_t stride_wxh = stride_w * stride_h;
    uint32_t padded_filt_height = roundup(filt_height, stride_h);
    uint32_t padded_filt_width = roundup(filt_width, stride_w);
    uint32_t padded_filt_depth = roundup(filt_depth, groups);
    uint32_t depth_padding = padded_filt_depth - filt_depth;
    pad_num_filters = ((1 < stride_w && stride_w < 5) || stride_w == 8) ? 1 : 0;
    pad_num_filters &= info->allow_num_filter_pad;
    uint32_t padded_num_filters = (pad_num_filters) ? roundup(num_filters, 32) : num_filters;

    // Calculate quantized 0 for use in padding biases & weights
    if (info->elbytes == sizeof(uint8_t)) {
        qzero = quantize_uint8(0.0f, weight_min, weight_max);
        bias_qzero8 = quantize_uint8(0.0f, bias_min, bias_max);
    } else if (info->elbytes == sizeof(uint16_t))    {
        qzero = quantize_uint16(0.0f, weight_min, weight_max);
    } else {
        return errlog(nn, "Unsupported data type for weights. Cannot find quantized zero");
    }

    shape_from_outdesc(&original_output_shape, &transpose_conv_node->output_defs[0], 0);
    // regular Depthwise case: (input_depth == groups == output_depth)
    if(groups > 1 && groups == input_shape.depth && groups == original_output_shape.depth && info->has_depthwise_path) {
        conv_op = OP_QuantizedDepthwiseConv2d_8x8to32;
        groups = 1;
        use_depthwise = 1;
        pad_num_filters = 0;
        // Dont pad in depthwise case
        padded_num_filters = num_filters;
        padded_filt_depth = filt_depth;
        depth_padding = 0;
        if (stride_wxh > 1) {
            // For the depthwise with striding case we use subkerneling
            use_subkerneling = 1;
            num_subkernels = stride_wxh;
            stride_wxh = 1;  // striding will be handled by subkerneling ==> treat strides as 1
        }
    }

    // Depthwise Channel multiple case (input_depth == 2 * groups == 2 * output_depth)
    uint32_t depthwise_chmul_filt_width = (filt_width * 2) / stride_w;
    if (groups > 1 && 2 * groups == input_shape.depth && groups == original_output_shape.depth && info->has_depthwise_path && depthwise_chmul_filt_width < 7 && padded_filt_height / stride_h >= 2) {
            chmul = 2; // currently only support channel multiplier of 2 because dwconv only supports strides <= 2
            // reshape input & weights and use depthwise with DWCONV stride_w == 2 strategy
            conv_op = OP_QuantizedDepthwiseConv2d_8x8to32;
            groups = 1;
            use_depthwise = 1;
            pad_num_filters = 0;
            padded_num_filters = num_filters;
            padded_filt_depth = filt_depth;
            depth_padding = 0;
            need_reshape = 1;
            if (stride_wxh > 1) {
                // For the depthwise with striding case we use subkerneling
                use_subkerneling = 1;
                num_subkernels = stride_wxh;
                stride_wxh = 1;  // striding will be handled by subkerneling ==> treat strides as 1
            }
    }

    uint32_t iterations = use_subkerneling ? num_subkernels : groups;
    uint32_t num_core_nodes = (stride_wxh > 1) ? 6 : 5; // extra d2s node if stride_wxh>1
    uint32_t num_nodes_to_replace = iterations * num_core_nodes + 5; // 5 extra nodes are: pad, (shuffle & reshape), split, concat and d2s
    struct nn_node *new_nodes[num_nodes_to_replace];
    memset(new_nodes, 0, num_nodes_to_replace * sizeof(struct nn_node *));


    // input needs to be channel shuffled and reshaped in the case of depthwise with channel multiplier
    if (need_reshape) {
        const uint32_t shuffle_axis_nid = create_const_int32_op(nn, 3);
        const uint32_t shuffle_groups_nid = create_const_int32_op(nn, input_shape.depth / chmul);

        struct input data_shuffle_input_refs[5];
        data_shuffle_input_refs[0] = original_inputs[0];
        data_shuffle_input_refs[1] = (struct input){shuffle_axis_nid, 0};
        data_shuffle_input_refs[2] = (struct input){shuffle_groups_nid, 0};
        data_shuffle_input_refs[3] = original_inputs[2];
        data_shuffle_input_refs[4] = original_inputs[3];

        struct output shuffle_output_refs[3];
        make_outputdesc_from_shape(&shuffle_output_refs[0], &input_shape, info->elbytes, 0);
        shuffle_output_refs[1] = Output_ScalarFloat;
        shuffle_output_refs[2] = Output_ScalarFloat;

        shuffle_node = create_node(nn, 0, info->shuffle_op, NN_PAD_NA, 5, 3, data_shuffle_input_refs, shuffle_output_refs);


        input_shape.width *= chmul;
        input_shape.depth /= chmul;
        padded_filt_depth = filt_depth = filt_depth / chmul;
        padded_filt_width *= chmul;
        // Create const node for new input shape
        uint32_t new_shape[4] = {input_shape.batches, input_shape.height, input_shape.width, input_shape.depth};
        uint32_t new_input_shape_id = nn_graph_new_internal_node_id(nn);
        if ((do_prepend_const_node(nn, new_input_shape_id, 1, 1, 1, 4, (const uint8_t *) new_shape, 4 * sizeof(uint32_t))) != 0) {
            errlog(nn, "Can't make a const scalar node");
            return 0;
        }

        struct input data_reshape_input_refs[4];
        data_reshape_input_refs[0] = (struct input){shuffle_node->node_id, 0};
        data_reshape_input_refs[1] = (struct input){new_input_shape_id, 0};
        data_reshape_input_refs[2] = (struct input){shuffle_node->node_id, 1};
        data_reshape_input_refs[3] = (struct input){shuffle_node->node_id, 2};

        struct output reshape_output_refs[3];
        make_outputdesc_from_shape(&reshape_output_refs[0], &input_shape, info->elbytes, 0);
        reshape_output_refs[1] = Output_ScalarFloat;
        reshape_output_refs[2] = Output_ScalarFloat;
        reshape_node = create_node(nn, 0, OP_QuantizedReshape, NN_PAD_NA, 4, 3, data_reshape_input_refs, reshape_output_refs);
    }

    if (use_depthwise) {
        // In DW case group_depth is full depth of the transpose conv op
        group_depth = filt_depth * padded_num_filters * stride_wxh;
        d2s_output_depth = num_filters * filt_depth;
    } else {
        group_depth = padded_num_filters * stride_wxh;
        d2s_output_depth = num_filters;
    }

    //r, s, c, k,  -> (fh, fw, din, dout)
    uint32_t k = group_depth;
    uint32_t r = padded_filt_height / stride_h;
    uint32_t s = padded_filt_width / stride_w;
    uint32_t c = padded_filt_depth / groups;
    uint32_t kernel_padding_height = 0;
    uint32_t kernel_padding_width = 0;
    if (use_depthwise) {
        // depthwise conv currently only supports channel multiplier of 1
        k = 1;
        r = (padded_filt_height / stride_h);
        s = (padded_filt_width / stride_w);
        c = padded_filt_depth;

        // pad kernel width and height up to optimal depthwise shapes
        uint32_t pre_pad_s = s;
        uint32_t pre_pad_r = r;
        if (s <= 3) s = 3;
        else if (s <= 5) s = 5;
        else if (s <= 7) s = 7;

        if (r <= 2) r = 2;
        kernel_padding_height = r - pre_pad_r;
        kernel_padding_width = s - pre_pad_s;
    }
    // Transform the filters using process_transpose_conv_filter
    struct transpose_conv_filter_parms tcfparms;

    uint32_t padded_filt_size = padded_num_filters * stride_h * stride_w * padded_filt_height/stride_h * padded_filt_width/stride_w * padded_filt_depth * info->elbytes;
    uint32_t filter_buffer_size = padded_filt_size;
    if (use_depthwise) {
        filter_buffer_size  = padded_num_filters * stride_h * stride_w * (padded_filt_height / stride_h + kernel_padding_height) * (padded_filt_width / stride_h + kernel_padding_width) * padded_filt_depth * info->elbytes;
    }

    uint8_t *padded_filters = nn_memalign(128, filter_buffer_size);
    tcfparms.filt_tensor = weight_node->outputs[0];
    tcfparms.out_data = padded_filters;
    tcfparms.zero_offset = qzero;
    tcfparms.block_h = stride_h;
    tcfparms.block_w = stride_w;
    tcfparms.elbytes = info->elbytes;
    tcfparms.use_subkerneling = use_subkerneling;
    tcfparms.use_depthwise = use_depthwise;
    tcfparms.chmul = chmul;
    tcfparms.pad_num_filters = pad_num_filters;
    tcfparms.data_size = padded_filt_size;
    tcfparms.groups = groups;
    int vv = nn_os_vector_acquire();
    process_transpose_conv_filter(nn, &tcfparms);
    nn_os_vector_release(vv);
    uint32_t kernel_size = r * s * c * k;
    uint8_t *weight_data = tcfparms.out_data;
    uint8_t **split_weights = (uint8_t **) padded_filters;

    // Create const node for new strides
    uint32_t new_strides[4] = {1, 1, chmul, 1};
    uint32_t new_strides_id = nn_graph_new_internal_node_id(nn);
    if ((do_prepend_const_node(nn, new_strides_id, 1, 1, chmul, 1, (const uint8_t *)new_strides, 4 * sizeof(uint32_t))) != 0) {
        errlog(nn, "Can't make a const scalar node");
        return 0;
    }

    uint32_t outsize_conv_height = (input_shape.height - 1) + ((r - kernel_padding_height) - 1) + 1;
    uint32_t outsize_conv_width = (input_shape.width - 1) + ((s - kernel_padding_width) - chmul) + 1;
    uint32_t d2s_crop_bottom = (input_shape.height - 1 + (r - kernel_padding_height)) * stride_h - paddings_arr[0] - original_output_shape.height;
    uint32_t d2s_crop_right = (input_shape.width/chmul - 1 + (s - kernel_padding_width)/chmul) * stride_w - paddings_arr[2] - original_output_shape.width;


    uint32_t pad_input_top = 0;
    uint32_t pad_input_bottom = 0;
    uint32_t pad_input_left = 0;
    uint32_t pad_input_right = 0;
    //If strides are 1, we need to adjust the padding amount around the input (SAME)
    if (stride_wxh == 1 && !use_subkerneling) {
        pad_input_top = paddings_arr[0];
        pad_input_bottom = paddings_arr[1];
        pad_input_left = paddings_arr[2];
        pad_input_right = paddings_arr[3];
        outsize_conv_height = original_output_shape.height;
        outsize_conv_width = original_output_shape.width;
    }

    uint32_t paddings[8] = {0, 0, r - 1 - pad_input_top, r - 1 - pad_input_bottom - kernel_padding_height, s - chmul - chmul*pad_input_left , s - chmul - chmul*pad_input_right - kernel_padding_width, 0, depth_padding};
    input_shape.height += 2 * ((r - kernel_padding_height) - 1) - pad_input_top - pad_input_bottom + kernel_padding_height;
    input_shape.width += 2 * ((s - kernel_padding_width) - chmul) - pad_input_left - pad_input_right + kernel_padding_width;
    input_shape.depth += depth_padding;

    unsigned pad_node_nid = 0;
    // if padding needed in either height or width create const node to contain the padding tensor
    if (paddings[2] != 0 || paddings[3] != 0 || paddings[4] != 0 || paddings[5] != 0 || paddings[7] != 0) {
        uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
        if ((do_prepend_const_node(nn, new_node_id, 1, 1, 4, 2, (const uint8_t *)paddings, 8 * sizeof(int32_t))) != 0)
            return errlog(nn, "Can't make a const scalar node");
        struct output pad_output_refs[3];
        make_outputdesc_from_shape(&pad_output_refs[0], &input_shape, info->elbytes, 0);
        pad_output_refs[1] = Output_ScalarFloat;
        pad_output_refs[2] = Output_ScalarFloat;

        struct input pad_input_refs[4] = {
            original_inputs[0],
            original_inputs[2],
            original_inputs[3],
            {new_node_id, 0}};
        if (need_reshape) {
            pad_input_refs[0] = (struct input){reshape_node->node_id, 0};
            pad_input_refs[1] = (struct input){reshape_node->node_id, 1};
            pad_input_refs[2] = (struct input){reshape_node->node_id, 2};
        }

        pad_node = create_node(nn, 0, info->pad_op, NN_PAD_NA, 4, 3, pad_input_refs, pad_output_refs);
        pad_node_nid = pad_node->node_id;
    }

    if (need_reshape) {
        new_nodes[0] = shuffle_node;
        new_nodes[1] = reshape_node;
        new_nodes[2] = pad_node; // may be NULL, if pad not needed
    } else {
        new_nodes[0] = pad_node; // may be NULL, if pad not needed
    }


    // Process Channel Scales and Biases
    float *padded_channel_scales = NULL;
    uint8_t *padded_biases = nn_memalign(128, groups * group_depth * info->elbytes_bias);
    struct nn_node *split_channel_scales[groups];
    struct nn_node *split_biases[groups];
    uint32_t new_bias_id;
    uint32_t new_channel_scale_id;
    uint32_t num_biases = bias_data_node->outputs[0]->shape.depth / groups;

    if (has_channel_scale) {
        float *channel_scales_data = (float *) channel_scale_node->outputs[0]->data;
        uint32_t num_channel_scales = channel_scale_node->outputs[0]->shape.depth / groups;
        padded_channel_scales = nn_memalign(128, groups * group_depth * sizeof(float));

        // prefill with 1.0f
        for (int i = 0; i < groups * group_depth; i++) {
            padded_channel_scales[i] = 1.0f;
        }

        // copy & duplicate channel scales to match s2b changes to weights caused by striding and grouping
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < stride_wxh; i++) {
                memcpy(padded_channel_scales + i * padded_num_filters + (g * group_depth), channel_scales_data + (g * num_channel_scales), num_channel_scales * sizeof(float));
            }
        }

        // create channel scale data const nodes for each group
        for (int g = 0; g < groups; g++) {
            new_channel_scale_id = nn_graph_new_internal_node_id(nn);
            if ((do_prepend_const_node(nn, new_channel_scale_id, 1, 1, 1, group_depth, (const uint8_t *) (padded_channel_scales + g * group_depth), group_depth * sizeof(float))) != 0) {
                errlog(nn, "Can't make a padded const node for channel scales");
                return 0;
            }
            split_channel_scales[g] = find_node_must_be_Const(nn,new_channel_scale_id);
        }
    }


    // prefill with quantized zeros and
    // copy biases to match s2b changes to weights caused by striding and grouping
    if (info->elbytes_bias == sizeof(uint8_t)) {
        for (int i = 0; i < groups * group_depth; i++) {
            padded_biases[i] = bias_qzero8;
        }
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < stride_wxh; i++) {
                memcpy(padded_biases + i * padded_num_filters + (g * group_depth), bias_data + (g * num_biases), num_biases * info->elbytes_bias);
            }
        }
    } else {
        int32_t *padded_biases32 = (int32_t *) padded_biases;
        for (int i = 0; i < groups * group_depth; i++) {
            padded_biases32[i] = 0;
        }
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < stride_wxh; i++) {
                memcpy(padded_biases32 + i * padded_num_filters + (g * group_depth), bias_data + (g * num_biases), num_biases * info->elbytes_bias);
            }
        }
    }

    // create bias data const nodes for each group
    for (int g = 0; g < groups; g++) {
        new_bias_id = nn_graph_new_internal_node_id(nn);
        if ((do_prepend_const_node(nn, new_bias_id, 1, 1, 1, group_depth, (const uint8_t *) (padded_biases + g * group_depth), group_depth * info->elbytes_bias)) != 0) {
            errlog(nn, "Can't make a padded const node for biases");
            return 0;
        }
        split_biases[g] = find_node_must_be_Const(nn,new_bias_id);
    }

    struct nn_node *conv_2d_node = NULL;
    struct nn_node *shrink1_predecessor_node = NULL;
    struct nn_node *bias_add_predecessor_node = NULL;
    struct nn_node *shrink1_node = NULL;
    struct nn_node *bias_add_node = NULL;
    struct nn_node *shrink_node = NULL;


    const uint32_t split_dim = 3; // split on the depth
    const uint32_t split_dim_nid = create_const_int32_op(nn, split_dim);
    if (groups > 1)    {
        /* add node for split dimension */
        const int split_num_outputs = groups + 2;

         /* Split node for data tensor - inputs */
        struct input data_split_input_refs[4];
        data_split_input_refs[0] = (struct input){ split_dim_nid, 0 }; //split dim=3
        data_split_input_refs[1] = original_inputs[0]; //data tensor
        data_split_input_refs[2] = original_inputs[2]; //data min
        data_split_input_refs[3] = original_inputs[3]; //data max
        if (pad_node_nid != 0) {
            data_split_input_refs[1] = (struct input){pad_node->node_id, 0};
            data_split_input_refs[2] = (struct input){pad_node->node_id, 1};
            data_split_input_refs[3] = (struct input){pad_node->node_id, 2};
        }

        /* Split node for data tensor - outputs */
        struct output data_split_output_defs[split_num_outputs];

        /* make output descriptors for each split node */
        for(int i=0; i < groups; i++) {
            make_outputdesc_from_shape(&data_split_output_defs[i], &input_shape, info->elbytes, 0);
            data_split_output_defs[i].max_sizes[3] = c;
        }

        // split output min and max
        data_split_output_defs[groups] = data_split_output_defs[groups+1] = Output_ScalarFloat;

        /* add Split node to graph */
        split_node = create_node(nn, 0, OP_QuantizedSplit_8, NN_PAD_NA, 4, split_num_outputs, data_split_input_refs, data_split_output_defs);
        new_nodes[1] = split_node;
    }


    shape_from_outdesc(&conv_output_shape, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
    conv_output_shape.height = outsize_conv_height;
    conv_output_shape.width = outsize_conv_width;
    conv_output_shape.depth = group_depth;

    // in the regular grouped cases we will create groups # of seperate tracks
    // in the case of subkerneling we will split the kernel by stride_wxh times and create that many seperate tracks
    for (int i = 0; i < iterations; i++) {
        int const_node_idx = use_subkerneling ? 0 : i;
        uint32_t offset = 0;
        if (groups > 1) {
            weight_data = split_weights[i];
        }
        if (use_subkerneling) {
            offset = i * kernel_size;
        }
        //Conv2d
        uint32_t new_weight_node_id = nn_graph_new_internal_node_id(nn);
        if ((do_prepend_const_node(nn, new_weight_node_id, r, s, c, k, (const uint8_t *) weight_data + offset, kernel_size * info->elbytes)) != 0){
            free_node_array(new_nodes, num_nodes_to_replace);
            return errlog(nn, "Can't make a const weights node for rearranged weights");
        }

        struct input conv_input_refs[7] = {
                original_inputs[0],
                {new_weight_node_id, 0},
                original_inputs[2],
                original_inputs[3],
                original_inputs[4],
                original_inputs[5],
                {new_strides_id, 0}
        };
        if (need_reshape) {
            conv_input_refs[0] = (struct input){reshape_node->node_id, 0};
            conv_input_refs[2] = (struct input){reshape_node->node_id, 1};
            conv_input_refs[3] = (struct input){reshape_node->node_id, 2};
        }
        if (pad_node_nid != 0) {
            conv_input_refs[0] = (struct input){pad_node->node_id, 0};
            conv_input_refs[2] = (struct input){pad_node->node_id, 1};
            conv_input_refs[3] = (struct input){pad_node->node_id, 2};
        }
        if (groups > 1)    {
            conv_input_refs[0] = (struct input){split_node->node_id, i};
            conv_input_refs[2] = (struct input){split_node->node_id, groups};
            conv_input_refs[3] = (struct input){split_node->node_id, groups + 1};
        }

        struct output conv_output_refs[3];
        make_outputdesc_from_shape(&conv_output_refs[0], &conv_output_shape, sizeof(int32_t), 0);
        conv_output_refs[1] = Output_ScalarFloat;
        conv_output_refs[2] = Output_ScalarFloat;
        conv_2d_node = create_node(nn, 0, conv_op, NN_PAD_VALID, 7, 3, conv_input_refs, conv_output_refs);
        bias_add_predecessor_node = conv_2d_node;
        shrink1_predecessor_node = conv_2d_node;
        //Channel Scale
        if (has_channel_scale) {
            struct output channel_scale_output_refs[3];
            make_outputdesc_from_shape(&channel_scale_output_refs[0], &conv_output_shape, sizeof(int32_t), 0);
            channel_scale_output_refs[1] = Output_ScalarFloat;
            channel_scale_output_refs[2] = Output_ScalarFloat;
            struct input channel_scale_input_refs[4] = {
                    {conv_2d_node->node_id, 0},
                    {split_channel_scales[const_node_idx]->node_id, 0},
                    {conv_2d_node->node_id, 1},
                    {conv_2d_node->node_id, 2}};
            channel_scale_node = create_node(nn, 0, info->channel_scale_op, NN_PAD_NA, 4, 3, channel_scale_input_refs, channel_scale_output_refs);
            bias_add_predecessor_node = channel_scale_node;
            shrink1_predecessor_node = channel_scale_node;
        }
        if (info->bias_add_op == OP_QuantizedBiasAdd_8p8to32) {
            struct output shrink1_output_refs[3];
            make_outputdesc_from_shape(&shrink1_output_refs[0], &conv_output_shape, sizeof(uint8_t), 0);
            shrink1_output_refs[1] = Output_ScalarFloat;
            shrink1_output_refs[2] = Output_ScalarFloat;
            struct input shrink1_input_refs[3] = {
                    {shrink1_predecessor_node->node_id, 0},
                    {shrink1_predecessor_node->node_id, 1},
                    {shrink1_predecessor_node->node_id, 2},
            };
            shrink1_node = create_node(nn, 0, info->shrink_op, NN_PAD_NA, 3, 3, shrink1_input_refs, shrink1_output_refs);
            bias_add_predecessor_node = shrink1_node;
        }

        //Bias add
        struct output bias_add_output_refs[3];
        make_outputdesc_from_shape(&bias_add_output_refs[0], &conv_output_shape, sizeof(int32_t), 0);
        bias_add_output_refs[1] = Output_ScalarFloat;
        bias_add_output_refs[2] = Output_ScalarFloat;

        struct input bias_add_input_refs[6] = {
                {bias_add_predecessor_node->node_id, 0},
                {split_biases[const_node_idx]->node_id, 0},
                {bias_add_predecessor_node->node_id, 1},
                {bias_add_predecessor_node->node_id, 2},
                original_inputs[9],
                original_inputs[10]};
        bias_add_node = create_node(nn, 0, info->bias_add_op, NN_PAD_NA, 6, 3, bias_add_input_refs, bias_add_output_refs);

        //Shrink
        struct input shrink_input_refs[5] = {
                {bias_add_node->node_id, 0},
                {bias_add_node->node_id, 1},
                {bias_add_node->node_id, 2},
                original_inputs[11],
                original_inputs[12]};
        struct output shrink_output_refs[3];
        make_outputdesc_from_shape(&shrink_output_refs[0], &conv_output_shape, info->elbytes, 0);
        shrink_output_refs[1] = Output_ScalarFloat;
        shrink_output_refs[2] = Output_ScalarFloat;
        // shrink_node inherits the original node id, if it's the output node.
        uint32_t is_output_node = stride_wxh == 1 && groups == 1 && !use_subkerneling;
        shrink_node = create_node(nn,
                                  is_output_node ? transpose_conv_node->node_id : 0,
                                  info->requant_op, NN_PAD_NA, 5, 3, shrink_input_refs, shrink_output_refs);
        new_nodes[3 + i * num_core_nodes] = conv_2d_node;
        new_nodes[3 + i * num_core_nodes + 1] = channel_scale_node;
        new_nodes[3 + i * num_core_nodes + 2] = shrink1_node;
        new_nodes[3 + i * num_core_nodes + 3] = bias_add_node;
        new_nodes[3 + i * num_core_nodes + 4] = shrink_node;

        // Depth to space before concat only when stride_wxh > 1
        // For subkerneling, d2s happens after concat
        if (stride_wxh > 1)    {
            uint32_t blocksize[2] = {stride_h, stride_w};
            uint32_t block_nid = nn_graph_new_internal_node_id(nn);
            if ((do_prepend_const_node(nn, block_nid, 1, 1, 1, 2, (const uint8_t *)blocksize, 2 * sizeof(uint32_t))) != 0)
            {
                free_node_array(new_nodes, num_nodes_to_replace);
                return errlog(nn, "Failed to append const node for d2s");
            }
            int32_t d2s_num_paddings = (pad_num_filters ? 5 : 4);
            int32_t padding_top = paddings_arr[0];
            int32_t padding_bottom = d2s_crop_bottom;
            int32_t padding_left = paddings_arr[2];
            int32_t padding_right = d2s_crop_right;
            int32_t d2s_paddings[5] = {padding_top, padding_bottom, padding_left, padding_right, d2s_output_depth};

            uint32_t d2s_paddings_id = nn_graph_new_internal_node_id(nn);
            if ((do_prepend_const_node(nn, d2s_paddings_id, 1, 1, 1, d2s_num_paddings, (const uint8_t *)d2s_paddings, d2s_num_paddings * sizeof(int32_t))) != 0)
            {
                free_node_array(new_nodes, num_nodes_to_replace);
                return errlog(nn, "Failed to append const node for d2s");
            }


            shape_from_outdesc(&d2s_output_shape, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
            d2s_output_shape.depth = d2s_output_shape.depth / iterations;
            if (pad_num_filters)
                d2s_output_shape.depth = roundup(d2s_output_shape.depth, 32) * stride_wxh;

            struct output d2s_output_refs[3];
            make_outputdesc_from_shape(&d2s_output_refs[0], &d2s_output_shape, info->elbytes, 0);
            d2s_output_refs[1] = Output_ScalarFloat;
            d2s_output_refs[2] = Output_ScalarFloat;
            struct input d2s_input_refs[5] = {
                    {shrink_node->node_id, 0},
                    {block_nid, 0},
                    {shrink_node->node_id, 1},
                    {shrink_node->node_id, 2},
                    {d2s_paddings_id, 0}};

            unsigned d2s_nid;
            if (iterations > 1) {
                d2s_nid = 0;
            } else {
                // final node thats node id of original transpose conv op
                d2s_nid = transpose_conv_node->node_id;
                d2s_output_refs[0] = transpose_conv_node->output_defs[0];
            }
            struct nn_node *d2s_node = create_node(nn, d2s_nid, info->d2s_op, NN_PAD_NA, 5, 3, d2s_input_refs, d2s_output_refs);
            new_nodes[3 + i * num_core_nodes + 5] = d2s_node;
        }
    }

    // if iterations > 1 then separate tracks need to be concatenated together
    if (iterations > 1)    {
        const int concat_num_inputs = 3*iterations+1;
        unsigned concat_id;
        struct output concat_output_refs[3];
        struct shape concat_out_shape;
        concat_out_shape = (stride_wxh > 1) ? d2s_output_shape : conv_output_shape;

        if (use_subkerneling) {
            // Final node will be d2s not concat
            concat_id = 0;
        } else {
            // Final node will be concat. Use node id of original transpose conv op
            concat_id = transpose_conv_node->node_id;
        }
        concat_out_shape.depth *= iterations;
        make_outputdesc_from_shape(&concat_output_refs[0], &concat_out_shape, sizeof(uint8_t), 0);
        concat_output_refs[1] = Output_ScalarFloat;
        concat_output_refs[2] = Output_ScalarFloat;
        //And we concat here
        struct input concat_input_refs[concat_num_inputs];
        concat_input_refs[0] = (struct input){ split_dim_nid, 0 };
        int offset = (stride_wxh > 1) ? 5 : 4;



        for(int i = 1; i <= iterations; i++) {
            concat_input_refs[i] = (struct input){ new_nodes[3 + (i-1) * num_core_nodes + offset]->node_id, 0 }; //tensor pieces
            concat_input_refs[iterations+i] = (struct input){ new_nodes[3 + (i-1) * num_core_nodes + offset]->node_id, 1 }; //mins
            concat_input_refs[2*iterations+i] = (struct input){ new_nodes[3 + (i-1) * num_core_nodes + offset]->node_id, 2 }; //maxes
        }

        struct nn_node *concat_node = create_node(nn, concat_id,  OP_QuantizedConcat_8, NN_PAD_NA, concat_num_inputs, 3, concat_input_refs, concat_output_refs);
        uint32_t position = use_subkerneling ? 2 : 1;
        new_nodes[num_nodes_to_replace - position] = concat_node;
    }

    // When using subkerneling, depth2space happens after concat
    if (use_subkerneling) {
        uint32_t blocksize[2] = {stride_h, stride_w};
        uint32_t block_nid = nn_graph_new_internal_node_id(nn);
        if ((do_prepend_const_node(nn, block_nid, 1, 1, 1, 2, (const uint8_t *)blocksize, 2 * sizeof(uint32_t))) != 0) {
            free_node_array(new_nodes, num_nodes_to_replace);
            return errlog(nn, "Failed to append const node for d2s");
        }
        int32_t d2s_num_paddings = (pad_num_filters ? 5 : 4);
        int32_t padding_top = paddings_arr[0];
        int32_t padding_bottom = d2s_crop_bottom;
        int32_t padding_left = paddings_arr[2];
        int32_t padding_right = d2s_crop_right;
        int32_t d2s_paddings[5] = {padding_top, padding_bottom, padding_left, padding_right, d2s_output_depth};
        uint32_t d2s_paddings_id = nn_graph_new_internal_node_id(nn);
        if ((do_prepend_const_node(nn, d2s_paddings_id, 1, 1, 1, d2s_num_paddings, (const uint8_t *)d2s_paddings, d2s_num_paddings * sizeof(int32_t))) != 0)
        {
            free_node_array(new_nodes, num_nodes_to_replace);
            return errlog(nn, "Failed to append const node for d2s");
        }
        struct nn_node *concat_node = new_nodes[num_nodes_to_replace - 2];
        shape_from_outdesc(&d2s_output_shape, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
        if (pad_num_filters)
            d2s_output_shape.depth = roundup(d2s_output_shape.depth, 32) * stride_wxh;

        struct input d2s_input_refs[5] = {
                {concat_node->node_id, 0},
                {block_nid, 0},
                {concat_node->node_id, 1},
                {concat_node->node_id, 2},
                {d2s_paddings_id, 0}
        };

        // final node thats node id of original transpose conv op
        struct nn_node *d2s_node = create_node(nn, transpose_conv_node->node_id, info->d2s_op, NN_PAD_NA, 5, 3, d2s_input_refs, transpose_conv_node->output_defs);
        new_nodes[num_nodes_to_replace - 1] = d2s_node;
    }
    replace_node_with_sequence(nn, transpose_conv_node_p, transpose_conv_node, new_nodes, num_nodes_to_replace);
    if (padded_channel_scales) {
        nn_free(padded_channel_scales);
    }
    if (padded_biases) {
        nn_free(padded_biases);
    }
    if (groups > 1) {
        free_splits(split_weights, groups);
    }
    if (padded_filters) {
        nn_free(padded_filters);
    }
    return 0;
}
