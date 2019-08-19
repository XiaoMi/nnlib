
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
#include "hexnn_dsp_domains_api.h"

__QAIC_STUB_EXPORT int hexagon_nn_domains_open(const char* uri, remote_handle64* h)
{
    return hexagon_nn_domains_open_impl(uri, h);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_close(remote_handle64 h)
{
    return hexagon_nn_domains_close_impl(h);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_config(remote_handle64 _h)
{
    return hexagon_nn_domains_config_impl(_h);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_config_with_options(remote_handle64 _h, const hexagon_nn_uint_option* uint_options,
        int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return hexagon_nn_domains_config_with_options_impl(_h, uint_options, uint_optionsLen, string_options, string_optionsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_graph_config(remote_handle64 _h, hexagon_nn_nn_id id,
        const hexagon_nn_uint_option* uint_options, int uint_optionsLen,
        const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return hexagon_nn_domains_graph_config_impl(_h, id, uint_options, uint_optionsLen, string_options, string_optionsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_dsp_offset(remote_handle64 _h, unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr)
{
    return hexagon_nn_domains_get_dsp_offset_impl(_h, libhexagon_addr, fastrpc_shell_addr);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_init(remote_handle64 _h, hexagon_nn_nn_id* g)
{
    return hexagon_nn_domains_init_impl(_h, g);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_debug_level(remote_handle64 _h, hexagon_nn_nn_id id, int level)
{
    return hexagon_nn_domains_set_debug_level_impl(_h, id, level);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_snpprint(remote_handle64 _h, hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return hexagon_nn_domains_snpprint_impl(_h, id, buf, bufLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_getlog(remote_handle64 _h, hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return hexagon_nn_domains_getlog_impl(_h, id, buf, bufLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen,
        const hexagon_nn_output* outputs, int outputsLen)
{
    return hexagon_nn_domains_append_node_impl(_h, id, node_id, operation, padding, inputs, inputsLen, outputs, outputsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_const_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen)
{
    return hexagon_nn_domains_append_const_node_impl(_h, id, node_id, batches, height, width, depth, data, dataLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_empty_const_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size)
{
    return hexagon_nn_domains_append_empty_const_node_impl(_h, id, node_id, batches, height, width, depth, size);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_populate_const_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        const unsigned char* data, int dataLen, unsigned int target_offset)
{
    return hexagon_nn_domains_populate_const_node_impl(_h, id, node_id, data, dataLen, target_offset);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_prepare(remote_handle64 _h, hexagon_nn_nn_id id)
{
    return hexagon_nn_domains_prepare_impl(_h, id);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute(remote_handle64 _h, hexagon_nn_nn_id id,
        unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in,
        const unsigned char* data_in, int data_inLen,
        unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out,
        unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return hexagon_nn_domains_execute_impl(_h, id, batches_in, height_in, width_in, depth_in, data_in, data_inLen,
            batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_teardown(remote_handle64 _h, hexagon_nn_nn_id id)
{
    return hexagon_nn_domains_teardown_impl(_h, id);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_read(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out,
        unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return hexagon_nn_domains_variable_read_impl(_h, id, node_id, output_index,
            batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_write(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth,
        const unsigned char* data_in, int data_inLen)
{
    return hexagon_nn_domains_variable_write_impl(_h, id, node_id,
            output_index, batches, height, width, depth, data_in, data_inLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_write_flat(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, const unsigned char* data_in, int data_inLen)
{
    return hexagon_nn_domains_variable_write_flat_impl(_h, id, node_id, output_index, data_in, data_inLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_powersave_level(remote_handle64 _h, unsigned int level)
{
    return hexagon_nn_domains_set_powersave_level_impl(_h, level);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_powersave_details(remote_handle64 _h,
        hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    return hexagon_nn_domains_set_powersave_details_impl(_h, corner, dcvs, latency);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_perfinfo(remote_handle64 _h, hexagon_nn_nn_id id,
        hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items)
{
    return hexagon_nn_domains_get_perfinfo_impl(_h, id, info_out, info_outLen, n_items);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_reset_perfinfo(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int event)
{
    return hexagon_nn_domains_reset_perfinfo_impl(_h, id, event);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_last_execution_cycles(remote_handle64 _h, hexagon_nn_nn_id id,
        unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return hexagon_nn_domains_last_execution_cycles_impl(_h, id, cycles_lo, cycles_hi);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_version(remote_handle64 _h, int* ver)
{
    return hexagon_nn_domains_version_impl(_h, ver);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_op_name_to_id(remote_handle64 _h, const char* name, unsigned int* node_id)
{
    return hexagon_nn_domains_op_name_to_id_impl(_h, name, node_id);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_op_id_to_name(remote_handle64 _h, unsigned int node_id, char* name, int nameLen)
{
    return hexagon_nn_domains_op_id_to_name_impl(_h, node_id, name, nameLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_num_nodes_in_graph(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* num_nodes)
{
    return hexagon_nn_domains_get_num_nodes_in_graph_impl(_h, id, num_nodes);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_disable_dcvs(remote_handle64 _h)
{
    return hexagon_nn_domains_disable_dcvs_impl(_h);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_GetHexagonBinaryVersion(remote_handle64 _h, int* ver)
{
    return hexagon_nn_domains_GetHexagonBinaryVersion_impl(_h, ver);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_PrintLog(remote_handle64 _h, const unsigned char* buf, int bufLen)
{
    return hexagon_nn_domains_PrintLog_impl(_h, buf, bufLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_new(remote_handle64 _h, hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen)
{
    return hexagon_nn_domains_execute_new_impl(_h, id, inputs, inputsLen, outputs, outputsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_with_info(remote_handle64 _h, hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info)
{
    return hexagon_nn_domains_execute_with_info_impl(_h, id, inputs, inputsLen, outputs, outputsLen, execute_info);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_init_with_info(remote_handle64 _h, hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info)
{
    return hexagon_nn_domains_init_with_info_impl(_h, g, info);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_nodetype(remote_handle64 _h, hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type)
{
    return hexagon_nn_domains_get_nodetype_impl(_h, graph_id, node_id, node_type);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_multi_execution_cycles(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return hexagon_nn_domains_multi_execution_cycles_impl(_h, id, cycles_lo, cycles_hi);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_power(remote_handle64 _h, int type)
{
    return hexagon_nn_domains_get_power_impl(_h, type);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_graph_option(remote_handle64 _h, hexagon_nn_nn_id id, const char* name, int value)
{
    return hexagon_nn_domains_set_graph_option_impl(_h, id, name, value);
}
