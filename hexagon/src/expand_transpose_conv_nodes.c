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

int expand_transpose_conv_nodes(struct nn_graph *nn, struct nn_node **transpose_conv_node_p)
{
	struct nn_node *transpose_conv_node = *transpose_conv_node_p;
	struct nn_node *producer;
	uint32_t src_id;
	struct nn_node *paddings_arr_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[6].src_id);
	int32_t * paddings_arr = paddings_arr_node->outputs[0]->data;
	struct nn_node *strides_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[7].src_id);
	uint32_t stride_h = strides_node->outputs[0]->shape.height;
	uint32_t stride_w = strides_node->outputs[0]->shape.width;
	uint32_t stride_wxh = stride_w * stride_h;
	if (stride_h < 1 || stride_w < 1)
		return errlog(nn, "Bad strides %d %d for transpose conv node", stride_h, stride_w);
	uint32_t num_nodes_to_replace = (stride_h == 1 && stride_w == 1) ? 4 : 3 * stride_h * stride_w + 3;
	struct nn_node *weight_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[1].src_id);
	uint8_t pad_num_filters __attribute__((unused)) = (1 < stride_h && stride_h < 5 && 1 < stride_w && stride_w < 5) ? 1 : 0;
	uint32_t num_filters = weight_node->outputs[0]->shape.batches;
	uint32_t filt_height = weight_node->outputs[0]->shape.height;
	uint32_t filt_width = weight_node->outputs[0]->shape.width;
	uint32_t filt_depth = weight_node->outputs[0]->shape.depth;
	uint32_t padded_num_filters = (pad_num_filters) ? roundup(num_filters, 32) : num_filters;
	uint32_t padded_filt_height = roundup(filt_height, stride_h);
	uint32_t padded_filt_width = roundup(filt_width, stride_w);
	uint32_t padded_filt_depth = filt_depth;
	uint32_t filt_padding_h = padded_filt_height - filt_height;
	uint32_t filt_padding_w = padded_filt_width - filt_width;
	uint32_t padded_filt_size = padded_num_filters * padded_filt_height * padded_filt_width * padded_filt_depth;
	uint32_t placeholder_weight_node_id = nn_graph_new_internal_node_id(nn);
	if ((do_prepend_const_node(nn, placeholder_weight_node_id, padded_num_filters, padded_filt_height, padded_filt_width, padded_filt_depth, NULL, padded_filt_size)) != 0)
	{
		return errlog(nn, "Can't allocate placeholder weight node for transpose conv");
	}
	struct nn_node *placeholder_weight_node = find_node_must_be_Const(nn, placeholder_weight_node_id);
	struct transpose_conv_filter_parms tcfparms;

	struct nn_node *weight_min_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[4].src_id);
	struct nn_node *weight_max_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[5].src_id);
	float weight_min = tensor_get_float(weight_min_node->outputs[0], 0);
	float weight_max = tensor_get_float(weight_max_node->outputs[0], 0);

	uint32_t qzero = quantize_uint8(0.0f, weight_min, weight_max);

	tcfparms.filt_tensor = weight_node->outputs[0];
	tcfparms.out_data = placeholder_weight_node->outputs[0]->data;
	tcfparms.zero_offset = qzero;
	tcfparms.strides_tensor = strides_node->outputs[0];
	tcfparms.data_size = padded_filt_size;
	nn_sem_init(&tcfparms.done_sem, 0);
	int vv = nn_os_vector_acquire();
	process_tranpose_conv_filter(nn, &tcfparms);
	nn_os_vector_release(vv);

	src_id = transpose_conv_node->input_refs[0].src_id;
	if ((producer = find_node(nn, src_id)) == NULL)
		return errlog(nn, "src id not found transpose conv stuff");
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	uint32_t k = padded_num_filters;
	uint32_t r = padded_filt_height / stride_h;
	uint32_t s = padded_filt_width / stride_w;
	uint32_t c = padded_filt_depth;

	struct shape insize;
	struct shape orig_outsize;
	struct shape outsize_conv;

	insize = producer->outputs[transpose_conv_node->input_refs[0].output_idx]->shape;
	uint32_t new_strides[4] = {1, 1, 1, 1};
	uint32_t new_strides_id = nn_graph_new_internal_node_id(nn);
	shape_from_outdesc(&orig_outsize, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
	if ((do_prepend_const_node(nn, new_strides_id, 1, 1, 1, 1, (const uint8_t *)new_strides, 4 * sizeof(uint32_t))) != 0) {
		errlog(nn, "Can't make a const scalar node");
		return 0;
	}
	uint32_t outsize_conv_height = (insize.height - 1) + (r - 1) + 1;
	uint32_t outsize_conv_width = (insize.width - 1) + (s - 1) + 1;

	uint32_t pad_input_top = 0;
	uint32_t pad_input_bottom = 0;
	uint32_t pad_input_left = 0;
	uint32_t pad_input_right = 0;
	//If strides are 1, we need to adjust the padding amount around the input (SAME)
	if (stride_wxh == 1) {
		pad_input_top = paddings_arr[0];
		pad_input_bottom = paddings_arr[1];
		pad_input_left = paddings_arr[2];
		pad_input_right = paddings_arr[3];
		outsize_conv_height = orig_outsize.height;
		outsize_conv_width = orig_outsize.width;
	}

	insize.height += 2 * (r - 1) - pad_input_top - pad_input_bottom;
	insize.width += 2 * (s - 1) - pad_input_left - pad_input_right;
	uint32_t paddings[8] = {0, 0, r - 1 - pad_input_top, r - 1 - pad_input_bottom, s - 1 - pad_input_left, s - 1 - pad_input_right, 0, 0};
	if ((do_prepend_const_node(nn, new_node_id, 1, 1, 4, 2, (const uint8_t *)paddings, 8 * sizeof(int32_t))) != 0)
		return errlog(nn, "Can't make a const scalar node");

	struct input *new_inputs = transpose_conv_node->input_refs;

	struct output pad_output_refs[3];
	make_outputdesc_from_shape(&pad_output_refs[0], &insize, sizeof(uint8_t), 0);
	pad_output_refs[1] = Output_ScalarFloat;
	pad_output_refs[2] = Output_ScalarFloat;

	struct input pad_input_refs[4] = {
		{new_inputs[0].src_id, 0},
		{new_inputs[2].src_id, 0},
		{new_inputs[3].src_id, 0},
		{new_node_id, 0}
	};

	struct nn_node *pad_node = create_node(nn, 0, OP_QuantizedPad_8, NN_PAD_NA, 4, 3, pad_input_refs, pad_output_refs);
	struct nn_node *conv_2d_node;
	struct nn_node *bias_add_node;
	struct nn_node *shrink_node;
	shape_from_outdesc(&outsize_conv, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
	outsize_conv.height = outsize_conv_height;
	outsize_conv.width = outsize_conv_width;
	if (1 < stride_h && stride_h < 5 && 1 < stride_w && stride_w < 5)
	    outsize_conv.depth = roundup(outsize_conv.depth, 32);
	uint32_t subkernel_size = r * s * c * k;
	uint8_t *weight_data = tcfparms.out_data;
	struct nn_node *new_nodes[num_nodes_to_replace];
	memset(new_nodes, NULL, num_nodes_to_replace * sizeof(*new_nodes));
	new_nodes[0] = pad_node;
	for (int i = 0; i < stride_wxh; i++) {
		//Conv2d
		uint32_t new_weight_node_id = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, new_weight_node_id, r, s, c, k, (const uint8_t *)weight_data + i * subkernel_size, subkernel_size * sizeof(uint8_t))) != 0)
		{
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Can't make a const weights node for subkernel %d", i);
		}
		struct input conv_input_refs[7] = {
				{pad_node->node_id, 0},
				{new_weight_node_id, 0},
				{pad_node->node_id, 1},
				{pad_node->node_id, 2},
				new_inputs[4],
				new_inputs[5],
				{stride_wxh > 1 ? new_strides_id : new_inputs[7].src_id, 0}
		};
		struct output conv_output_refs[3];
		make_outputdesc_from_shape(&conv_output_refs[0], &outsize_conv, sizeof(int32_t), 0);
		conv_output_refs[1] = Output_ScalarFloat;
		conv_output_refs[2] = Output_ScalarFloat;
		conv_2d_node = create_node(nn, 0, OP_QuantizedConv2d_8x8to32, NN_PAD_VALID, 7, 3, conv_input_refs, conv_output_refs);

		//Bias add
		struct output bias_add_output_refs[3];
		make_outputdesc_from_shape(&bias_add_output_refs[0], &outsize_conv, sizeof(int32_t), 0);
		bias_add_output_refs[1] = Output_ScalarFloat;
		bias_add_output_refs[2] = Output_ScalarFloat;
		struct input bias_add_input_refs[6] = {
				{conv_2d_node->node_id, 0},
				new_inputs[8],
				{conv_2d_node->node_id, 1},
				{conv_2d_node->node_id, 2},
				new_inputs[9],
				new_inputs[10]
		};
		bias_add_node = create_node(nn, 0, OP_QuantizedBiasAdd_32p32to32, NN_PAD_NA, 6, 3, bias_add_input_refs, bias_add_output_refs);

		//Shrink
		struct input shrink_input_refs[5] = {
				{bias_add_node->node_id, 0},
				{bias_add_node->node_id, 1},
				{bias_add_node->node_id, 2},
				new_inputs[11],
				new_inputs[12]
		};
		struct output shrink_output_refs[3];
		make_outputdesc_from_shape(&shrink_output_refs[0], &outsize_conv, 1, 0);
		shrink_output_refs[1] = Output_ScalarFloat;
		shrink_output_refs[2] = Output_ScalarFloat;
		shrink_node = create_node(nn, 0, OP_Requantize_32to8, NN_PAD_NA, 5, 3, shrink_input_refs, shrink_output_refs);
		new_nodes[3 * i + 1] = conv_2d_node;
		new_nodes[3 * i + 2] = bias_add_node;
		new_nodes[3 * i + 3] = shrink_node;
	}
	struct input new_inprefs[3] = {
		{shrink_node->node_id, 0},
		{shrink_node->node_id, 1},
		{shrink_node->node_id, 2}
	};

	if (stride_wxh > 1) {
		uint32_t depth_dim[1] = { 3 };
		uint32_t depth_dim_nid = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, depth_dim_nid, 1, 1, 1, 1, (const uint8_t *)depth_dim, sizeof(uint32_t))) != 0) {
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Failed to append const node concat dim");
		}

		struct input concat_input_refs[3 * stride_wxh + 1];
		concat_input_refs[0] = (struct input){depth_dim_nid, 0};
		for (int i = 0; i < stride_wxh; i++) {
			concat_input_refs[0*stride_wxh+i+1] = (struct input) { new_nodes[3*i+3]->node_id, 0 }; // in
			concat_input_refs[1*stride_wxh+i+1] = (struct input) { new_nodes[3*i+3]->node_id, 1 }; //mins
			concat_input_refs[2*stride_wxh+i+1] = (struct input) { new_nodes[3*i+3]->node_id, 2 }; //maxes
		}

		struct output concat_output_refs[3];
		struct shape concat_out_shape = outsize_conv;
		concat_out_shape.depth *= stride_wxh;
		make_outputdesc_from_shape(&concat_output_refs[0], &concat_out_shape, sizeof(uint8_t), 0);
		concat_output_refs[1] = Output_ScalarFloat;
		concat_output_refs[2] = Output_ScalarFloat;
		struct nn_node *concat_node = create_node(nn, 0, OP_QuantizedConcat_8, NN_PAD_NA, 3*stride_wxh+1, 3, concat_input_refs, concat_output_refs);
		new_nodes[3 * stride_wxh + 1] = concat_node;

		uint32_t blocksize[2] = {stride_h, stride_w};
		uint32_t block_nid = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, block_nid, 1, 1, 1, 2, (const uint8_t *)blocksize, 2 * sizeof(uint32_t))) != 0) {
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Failed to append const node for d2s");
		}
		int32_t d2s_num_paddings = (stride_h < 5 && stride_w < 5 ? 5 : 4);
		int32_t padding_top = paddings_arr[0];
		int32_t padding_bottom = paddings_arr[1] + filt_padding_h;
		int32_t padding_left = paddings_arr[2];
		int32_t padding_right = paddings_arr[3] + filt_padding_w;
		int32_t d2s_paddings[5] = { padding_top, padding_bottom, padding_left, padding_right, 0 };
		if (d2s_num_paddings == 5)
			d2s_paddings[4] = weight_node->outputs[0]->shape.batches;
		uint32_t d2s_paddings_id = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, d2s_paddings_id, 1, 1, 1, d2s_num_paddings, (const uint8_t *)d2s_paddings, d2s_num_paddings * sizeof(int32_t))) != 0)
		{
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Failed to append const node for d2s");
		}

		struct input d2s_input_refs[5];
		d2s_input_refs[0] = (struct input) { concat_node->node_id, 0 };
		d2s_input_refs[1] = (struct input) { block_nid, 0 };
		d2s_input_refs[2] = (struct input) { concat_node->node_id, 1 };
		d2s_input_refs[3] = (struct input) { concat_node->node_id, 2 };
		d2s_input_refs[4] = (struct input) { d2s_paddings_id, 0 };
		struct output d2s_output_refs[3];
		make_outputdesc_from_shape(&d2s_output_refs[0], &transpose_conv_node->outputs[0]->shape, sizeof(uint8_t), 0);
		d2s_output_refs[1] = Output_ScalarFloat;
		d2s_output_refs[2] = Output_ScalarFloat;
		struct nn_node *d2s_node = create_node(nn, 0, OP_DepthToSpace_8, NN_PAD_NA, 5, 3, d2s_input_refs, d2s_output_refs);
		new_nodes[3 * stride_wxh + 2] = d2s_node;
		new_inprefs[0] = (struct input){d2s_node->node_id, 0};
		new_inprefs[1] = (struct input){d2s_node->node_id, 1};
		new_inprefs[2] = (struct input){d2s_node->node_id, 2};
	}

	change_multi_output_refs_table(nn, transpose_conv_node, transpose_conv_node->node_id, 3, new_inprefs);
	replace_node_with_sequence(nn, transpose_conv_node_p, transpose_conv_node, new_nodes, num_nodes_to_replace);
	return 0;
}

int expand_transpose_conv16_nodes(struct nn_graph *nn, struct nn_node **transpose_conv_node_p)
{
	struct nn_node *transpose_conv_node = *transpose_conv_node_p;
	struct nn_node *producer;
	uint32_t src_id;
	struct nn_node *paddings_arr_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[6].src_id);
	int32_t * paddings_arr = paddings_arr_node->outputs[0]->data;
	struct nn_node *strides_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[7].src_id);
	uint32_t stride_h = strides_node->outputs[0]->shape.height;
	uint32_t stride_w = strides_node->outputs[0]->shape.width;
	uint32_t stride_wxh = stride_w * stride_h;
	if (stride_h < 1 || stride_w < 1)
		return errlog(nn, "Bad strides %d %d for transpose conv node", stride_h, stride_w);
	uint32_t num_nodes_to_replace = (stride_h == 1 && stride_w == 1) ? 4 : 3 * stride_h * stride_w + 3;
	struct nn_node *weight_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[1].src_id);
	uint8_t pad_num_filters __attribute__((unused)) = (1 < stride_h && stride_h < 5 && 1 < stride_w && stride_w < 5) ? 1 : 0;
	uint32_t num_filters = weight_node->outputs[0]->shape.batches;
	uint32_t filt_height = weight_node->outputs[0]->shape.height;
	uint32_t filt_width = weight_node->outputs[0]->shape.width;
	uint32_t filt_depth = weight_node->outputs[0]->shape.depth;
//	uint32_t padded_num_filters = (pad_num_filters) ? roundup(num_filters, 32) : num_filters;
	uint32_t padded_num_filters = num_filters; // TODO
	uint32_t padded_filt_height = roundup(filt_height, stride_h);
	uint32_t padded_filt_width = roundup(filt_width, stride_w);
	uint32_t padded_filt_depth = filt_depth;
	uint32_t filt_padding_h = padded_filt_height - filt_height;
	uint32_t filt_padding_w = padded_filt_width - filt_width;
	uint32_t padded_filt_size = padded_num_filters * padded_filt_height * padded_filt_width * padded_filt_depth * sizeof(uint16_t);
	uint32_t placeholder_weight_node_id = nn_graph_new_internal_node_id(nn);
	if ((do_prepend_const_node(nn, placeholder_weight_node_id, padded_num_filters, padded_filt_height, padded_filt_width, padded_filt_depth, NULL, padded_filt_size)) != 0)
	{
		return errlog(nn, "Can't allocate placeholder weight node for transpose conv");
	}
	struct nn_node *placeholder_weight_node = find_node_must_be_Const(nn, placeholder_weight_node_id);
	struct transpose_conv_filter_parms tcfparms;

	struct nn_node *weight_min_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[4].src_id);
	struct nn_node *weight_max_node = find_node_must_be_Const(nn, transpose_conv_node->input_refs[5].src_id);
	float weight_min = tensor_get_float(weight_min_node->outputs[0], 0);
	float weight_max = tensor_get_float(weight_max_node->outputs[0], 0);

	uint32_t qzero = quantize_uint16(0.0f, weight_min, weight_max);

	tcfparms.filt_tensor = weight_node->outputs[0];
	tcfparms.out_data = placeholder_weight_node->outputs[0]->data;
	tcfparms.zero_offset = qzero;
	tcfparms.strides_tensor = strides_node->outputs[0];
	tcfparms.data_size = padded_filt_size;
	nn_sem_init(&tcfparms.done_sem, 0);
	int vv = nn_os_vector_acquire();
	process_tranpose_conv16_filter(nn, &tcfparms);
	nn_os_vector_release(vv);

	src_id = transpose_conv_node->input_refs[0].src_id;
	if ((producer = find_node(nn, src_id)) == NULL)
		return errlog(nn, "src id not found transpose conv stuff");
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	uint32_t k = padded_num_filters;
	uint32_t r = padded_filt_height / stride_h;
	uint32_t s = padded_filt_width / stride_w;
	uint32_t c = padded_filt_depth;

	struct shape insize;
	struct shape orig_outsize;
	struct shape outsize_conv;

	insize = producer->outputs[transpose_conv_node->input_refs[0].output_idx]->shape;
	uint32_t new_strides[4] = { 1, 1, 1, 1 };
	uint32_t new_strides_id = nn_graph_new_internal_node_id(nn);
	shape_from_outdesc(&orig_outsize, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
	if ((do_prepend_const_node(nn, new_strides_id, 1, 1, 1, 1, (const uint8_t *)new_strides, 4 * sizeof(uint32_t))) != 0) {
		errlog(nn, "Can't make a const scalar node");
		return 0;
	}
	uint32_t outsize_conv_height = (insize.height - 1) + (r - 1) + 1;
	uint32_t outsize_conv_width = (insize.width - 1) + (s - 1) + 1;

	uint32_t pad_input_top = 0;
	uint32_t pad_input_bottom = 0;
	uint32_t pad_input_left = 0;
	uint32_t pad_input_right = 0;
	//If strides are 1, we need to adjust the padding amount around the input (SAME)
	if (stride_wxh == 1) {
		pad_input_top = paddings_arr[0];
		pad_input_bottom = paddings_arr[1];
		pad_input_left = paddings_arr[2];
		pad_input_right = paddings_arr[3];
		outsize_conv_height = orig_outsize.height;
		outsize_conv_width = orig_outsize.width;
	}

	insize.height += 2 * (r - 1) - pad_input_top - pad_input_bottom;
	insize.width += 2 * (s - 1) - pad_input_left - pad_input_right;
	uint32_t paddings[8] = { 0, 0, r - 1 - pad_input_top, r - 1 - pad_input_bottom, s - 1 - pad_input_left, s - 1 - pad_input_right, 0, 0 };
	if ((do_prepend_const_node(nn, new_node_id, 1, 1, 4, 2, (const uint8_t *)paddings, 8 * sizeof(int32_t))) != 0)
		return errlog(nn, "Can't make a const scalar node");

	struct input *new_inputs = transpose_conv_node->input_refs;

	struct output pad_output_refs[3];
	make_outputdesc_from_shape(&pad_output_refs[0], &insize, sizeof(uint16_t), 0);
	pad_output_refs[1] = Output_ScalarFloat;
	pad_output_refs[2] = Output_ScalarFloat;

	struct input pad_input_refs[4] = {
		{new_inputs[0].src_id, 0},
		{new_inputs[2].src_id, 0},
		{new_inputs[3].src_id, 0},
		{new_node_id, 0}
	};

	struct nn_node *pad_node = create_node(nn, 0, OP_QuantizedPad_u16, NN_PAD_NA, 4, 3, pad_input_refs, pad_output_refs);
	struct nn_node *conv_2d_node;
	struct nn_node *bias_add_node;
	struct nn_node *shrink_node;
	shape_from_outdesc(&outsize_conv, &transpose_conv_node->output_defs[0], /*add_d32_pad=*/0);
	outsize_conv.height = outsize_conv_height;
	outsize_conv.width = outsize_conv_width;
	// TODO
	//if (1 < stride_h && stride_h < 5 && 1 < stride_w && stride_w < 5)
	//	outsize_conv.depth = roundup(outsize_conv.depth, 32);
	uint32_t subkernel_size = r * s * c * k;
	uint16_t *weight_data = (uint16_t *)tcfparms.out_data;
	struct nn_node *new_nodes[num_nodes_to_replace];
	memset(new_nodes, NULL, num_nodes_to_replace*sizeof(*new_nodes));
	new_nodes[0] = pad_node;

	for (int i = 0; i < stride_wxh; i++) {
		// Conv2d
		uint32_t new_weight_node_id = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, new_weight_node_id, r, s, c, k, (const uint8_t *)(weight_data + i * subkernel_size), subkernel_size * sizeof(uint16_t))) != 0) {
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Can't make a const weights node for subkernel %d", i);
		}
		struct input conv_input_refs[7] = {
			{pad_node->node_id, 0}, // in
			{new_weight_node_id, 0}, // filt
			{pad_node->node_id, 1}, // in_min
			{pad_node->node_id, 2}, // in_max
			new_inputs[4], // filt_min
			new_inputs[5], // filt_max
			{stride_wxh > 1 ? new_strides_id : new_inputs[7].src_id, 0} // stride
		};
		struct output conv_output_refs[3];
		make_outputdesc_from_shape(&conv_output_refs[0], &outsize_conv, sizeof(int32_t), 0);
		conv_output_refs[1] = Output_ScalarFloat;
		conv_output_refs[2] = Output_ScalarFloat;
		conv_2d_node = create_node(nn, 0, OP_QuantizedConv2d_16x16to32, NN_PAD_VALID, 7, 3, conv_input_refs, conv_output_refs);

		//Bias add
		struct output bias_add_output_refs[3];
		make_outputdesc_from_shape(&bias_add_output_refs[0], &outsize_conv, sizeof(int32_t), 0);
		bias_add_output_refs[1] = Output_ScalarFloat;
		bias_add_output_refs[2] = Output_ScalarFloat;
		struct input bias_add_input_refs[6] = {
			{conv_2d_node->node_id, 0}, // in
			new_inputs[8], // bias
			{conv_2d_node->node_id, 1}, // in min
			{conv_2d_node->node_id, 2}, // in max
			new_inputs[9], // bias min
			new_inputs[10] // bias max
		};
		bias_add_node = create_node(nn, 0, OP_QuantizedBiasAdd_32p32to32, NN_PAD_NA, 6, 3, bias_add_input_refs, bias_add_output_refs);

		//Shrink
		struct input shrink_input_refs[5] = {
			{bias_add_node->node_id, 0}, // in
			{bias_add_node->node_id, 1}, // in min
			{bias_add_node->node_id, 2}, // in max
			new_inputs[11], // out min
			new_inputs[12]  // out max
		};
		struct output shrink_output_refs[3];
		make_outputdesc_from_shape(&shrink_output_refs[0], &outsize_conv, sizeof(uint16_t), 0);
		shrink_output_refs[1] = Output_ScalarFloat;
		shrink_output_refs[2] = Output_ScalarFloat;
		shrink_node = create_node(nn, 0, OP_Requantize_32tou16, NN_PAD_NA, 5, 3, shrink_input_refs, shrink_output_refs);

		new_nodes[3 * i + 1] = conv_2d_node;
		new_nodes[3 * i + 2] = bias_add_node;
		new_nodes[3 * i + 3] = shrink_node;
	}

	struct input new_inprefs[3] = {
		{shrink_node->node_id, 0},
		{shrink_node->node_id, 1},
		{shrink_node->node_id, 2}
	};

	if (stride_wxh > 1) {
		uint32_t depth_dim[1] = { 3 };
		uint32_t depth_dim_nid = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, depth_dim_nid, 1, 1, 1, 1, (const uint8_t *)depth_dim, sizeof(uint32_t))) != 0) {
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Failed to append const node concat dim");
		}

		struct input concat_input_refs[3 * stride_wxh + 1];
		concat_input_refs[0] = (struct input) { depth_dim_nid, 0 };
		for (int i = 0; i < stride_wxh; i++) {
			concat_input_refs[0*stride_wxh+i+1] = (struct input){ new_nodes[3*i+3]->node_id, 0 }; // in
			concat_input_refs[1*stride_wxh+i+1] = (struct input){ new_nodes[3*i+3]->node_id, 1 }; //mins
			concat_input_refs[2*stride_wxh+i+1] = (struct input){ new_nodes[3*i+3]->node_id, 2 }; //maxes
		}

		struct output concat_output_refs[3];
		struct shape concat_out_shape = outsize_conv;
		concat_out_shape.depth *= stride_wxh;
		make_outputdesc_from_shape(&concat_output_refs[0], &concat_out_shape, sizeof(uint16_t), 0);
		concat_output_refs[1] = Output_ScalarFloat;
		concat_output_refs[2] = Output_ScalarFloat;
		struct nn_node *concat_node = create_node(nn, 0, OP_QuantizedConcat_u16, NN_PAD_NA, 3*stride_wxh+1, 3, concat_input_refs, concat_output_refs);
		new_nodes[3 * stride_wxh + 1] = concat_node;

		uint32_t blocksize[2] = { stride_h, stride_w };
		uint32_t block_nid = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, block_nid, 1, 1, 1, 2, (const uint8_t *)blocksize, 2 * sizeof(uint32_t))) != 0) {
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Failed to append const node for d2s");
		}
		//int32_t d2s_num_paddings = (stride_h < 5 && stride_w < 5 ? 5 : 4);
		int32_t d2s_num_paddings = 4; // TODO: non d32 only have 4 params
		int32_t padding_top = paddings_arr[0];
		int32_t padding_bottom = paddings_arr[1] + filt_padding_h;
		int32_t padding_left = paddings_arr[2];
		int32_t padding_right = paddings_arr[3] + filt_padding_w;
		int32_t d2s_paddings[5] = { padding_top, padding_bottom, padding_left, padding_right, 0 };
		if (d2s_num_paddings == 5)
			d2s_paddings[4] = weight_node->outputs[0]->shape.batches;
		uint32_t d2s_paddings_id = nn_graph_new_internal_node_id(nn);
		if ((do_prepend_const_node(nn, d2s_paddings_id, 1, 1, 1, d2s_num_paddings, (const uint8_t *)d2s_paddings, d2s_num_paddings * sizeof(int32_t))) != 0) {
			free_node_array(new_nodes, num_nodes_to_replace);
			return errlog(nn, "Failed to append const node for d2s");
		}

		struct input d2s_input_refs[5];
		d2s_input_refs[0] = (struct input) { concat_node->node_id, 0 };
		d2s_input_refs[1] = (struct input) { block_nid, 0 };
		d2s_input_refs[2] = (struct input) { concat_node->node_id, 1 };
		d2s_input_refs[3] = (struct input) { concat_node->node_id, 2 };
		d2s_input_refs[4] = (struct input) { d2s_paddings_id, 0 };
		struct output d2s_output_refs[3];
		make_outputdesc_from_shape(&d2s_output_refs[0], &transpose_conv_node->outputs[0]->shape, sizeof(uint16_t), 0);
		d2s_output_refs[1] = Output_ScalarFloat;
		d2s_output_refs[2] = Output_ScalarFloat;
		struct nn_node *d2s_node = create_node(nn, 0, OP_DepthToSpace_16, NN_PAD_NA, 5, 3, d2s_input_refs, d2s_output_refs);
		new_nodes[3 * stride_wxh + 2] = d2s_node;

		new_inprefs[0] = (struct input) { d2s_node->node_id, 0 };
		new_inprefs[1] = (struct input) { d2s_node->node_id, 1 };
		new_inprefs[2] = (struct input) { d2s_node->node_id, 2 };
	}

	change_multi_output_refs_table(nn, transpose_conv_node, transpose_conv_node->node_id, 3, new_inprefs);
	replace_node_with_sequence(nn, transpose_conv_node_p, transpose_conv_node, new_nodes, num_nodes_to_replace);
	return 0;
}
