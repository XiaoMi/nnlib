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

#ifndef HEXAGON_NN_HEXNN_DSP_API_H
#define HEXAGON_NN_HEXNN_DSP_API_H

#ifdef __QAIC_HEADER
#undef __QAIC_HEADER
#endif

#ifndef __QAIC_STUB_EXPORT
#define __QAIC_STUB_EXPORT
#endif // __QAIC_STUB_EXPORT

#define __QAIC_HEADER(ff) stub_ ## ff

#include "hexagon_nn.h"

__QAIC_STUB_EXPORT int hexagon_nn_config_impl(void);
__QAIC_STUB_EXPORT int hexagon_nn_config_with_options_impl(const hexagon_nn_uint_option* uint_options,
        int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen);
__QAIC_STUB_EXPORT int hexagon_nn_graph_config_impl(hexagon_nn_nn_id id, const hexagon_nn_uint_option* uint_options,
        int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen);
__QAIC_STUB_EXPORT int hexagon_nn_get_dsp_offset_impl(unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr);
__QAIC_STUB_EXPORT int hexagon_nn_init_impl(hexagon_nn_nn_id* g);
__QAIC_STUB_EXPORT int hexagon_nn_set_debug_level_impl(hexagon_nn_nn_id id, int level);
__QAIC_STUB_EXPORT int hexagon_nn_snpprint_impl(hexagon_nn_nn_id id, unsigned char* buf, int bufLen);
__QAIC_STUB_EXPORT int hexagon_nn_getlog_impl(hexagon_nn_nn_id id, unsigned char* buf, int bufLen);
__QAIC_STUB_EXPORT int hexagon_nn_append_node_impl(hexagon_nn_nn_id id, unsigned int node_id, unsigned int operation,
        hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen);
__QAIC_STUB_EXPORT int hexagon_nn_append_const_node_impl(hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches,
        unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen);
__QAIC_STUB_EXPORT int hexagon_nn_append_empty_const_node_impl(hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size);
__QAIC_STUB_EXPORT int hexagon_nn_populate_const_node_impl(hexagon_nn_nn_id id, unsigned int node_id,
        const unsigned char* data, int dataLen, unsigned int target_offset);
__QAIC_STUB_EXPORT int hexagon_nn_prepare_impl(hexagon_nn_nn_id id);
__QAIC_STUB_EXPORT int hexagon_nn_execute_impl(hexagon_nn_nn_id id,
        unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in,
        const unsigned char* data_in, int data_inLen, unsigned int* batches_out,
        unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out,
        int data_outLen, unsigned int* data_len_out);
__QAIC_STUB_EXPORT int hexagon_nn_teardown_impl(hexagon_nn_nn_id id);
__QAIC_STUB_EXPORT int hexagon_nn_variable_read_impl(hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out,
        unsigned char* data_out, int data_outLen, unsigned int* data_len_out);
__QAIC_STUB_EXPORT int hexagon_nn_variable_write_impl(hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth,
        const unsigned char* data_in, int data_inLen);
__QAIC_STUB_EXPORT int hexagon_nn_variable_write_flat_impl(hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, const unsigned char* data_in, int data_inLen);
__QAIC_STUB_EXPORT int hexagon_nn_set_powersave_level_impl(unsigned int level);
__QAIC_STUB_EXPORT int hexagon_nn_set_powersave_details_impl(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency);
__QAIC_STUB_EXPORT int hexagon_nn_get_perfinfo_impl(hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items);
__QAIC_STUB_EXPORT int hexagon_nn_reset_perfinfo_impl(hexagon_nn_nn_id id, unsigned int event);
__QAIC_STUB_EXPORT int hexagon_nn_last_execution_cycles_impl(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi);
__QAIC_STUB_EXPORT int hexagon_nn_version_impl(int* ver);
__QAIC_STUB_EXPORT int hexagon_nn_op_name_to_id_impl(const char* name, unsigned int* node_id);
__QAIC_STUB_EXPORT int hexagon_nn_op_id_to_name_impl(unsigned int node_id, char* name, int nameLen);
__QAIC_STUB_EXPORT int hexagon_nn_get_num_nodes_in_graph_impl(hexagon_nn_nn_id id, unsigned int* num_nodes);
__QAIC_STUB_EXPORT int hexagon_nn_disable_dcvs_impl(void);
__QAIC_STUB_EXPORT int hexagon_nn_GetHexagonBinaryVersion_impl(int* ver);
__QAIC_STUB_EXPORT int hexagon_nn_PrintLog_impl(const unsigned char* buf, int bufLen);
__QAIC_STUB_EXPORT int hexagon_nn_execute_new_impl(hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen);
__QAIC_STUB_EXPORT int hexagon_nn_execute_with_info_impl(hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info);
__QAIC_STUB_EXPORT int hexagon_nn_init_with_info_impl(hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info);
__QAIC_STUB_EXPORT int hexagon_nn_get_nodetype_impl(hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type);
__QAIC_STUB_EXPORT int hexagon_nn_multi_execution_cycles_impl(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi);
__QAIC_STUB_EXPORT int hexagon_nn_get_power_impl(int type);
__QAIC_STUB_EXPORT int hexagon_nn_set_graph_option_impl(hexagon_nn_nn_id id, const char* name, int value);

#endif //HEXAGON_NN_HEXNN_DSP_API_H
