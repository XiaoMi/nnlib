
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
#include "hexnn_graph_wrapper.hpp"


__QAIC_STUB_EXPORT int hexagon_nn_domains_open_impl(const char* uri, remote_handle64* h)
{
    return(stub_hexagon_nn_domains_open(uri, h));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_close_impl(remote_handle64 h)
{
    return(stub_hexagon_nn_domains_close(h));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_config_impl(remote_handle64 _h)
{
    return(stub_hexagon_nn_domains_config(_h));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_config_with_options_impl(remote_handle64 _h, const hexagon_nn_uint_option* uint_options,
        int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return(stub_hexagon_nn_domains_config_with_options(_h, uint_options, uint_optionsLen, string_options, string_optionsLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_graph_config_impl(remote_handle64 _h, hexagon_nn_nn_id id,
        const hexagon_nn_uint_option* uint_options, int uint_optionsLen,
        const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return(stub_hexagon_nn_domains_graph_config(_h, id, uint_options, uint_optionsLen, string_options, string_optionsLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_dsp_offset_impl(remote_handle64 _h, unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr)
{
    return(stub_hexagon_nn_domains_get_dsp_offset(_h, libhexagon_addr, fastrpc_shell_addr));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_init_impl(remote_handle64 _h, hexagon_nn_nn_id* g)
{
    int sts;
    sts = stub_hexagon_nn_domains_init(_h, g);

    if(sts == 0)
    {
        sts = add_nn_id(_h, *g);
    }
    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_debug_level_impl(remote_handle64 _h, hexagon_nn_nn_id id, int level)
{
    return(stub_hexagon_nn_domains_set_debug_level(_h, id, level));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_snpprint_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return(stub_hexagon_nn_domains_snpprint(_h, id, buf, bufLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_getlog_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return(stub_hexagon_nn_domains_getlog(_h, id, buf, bufLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_node_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen,
        const hexagon_nn_output* outputs, int outputsLen)
{
    int sts = -1;
    batch_ops_params params;

    params.op = HEXNN_BATCH_OP_APPEND_NODE;
    params.node_id = node_id;
    params.operation = operation;
    params.padding = padding;
    params.inputs = (hexagon_nn_input*)inputs;
    params.inputsLen = inputsLen;
    params.outputs = (hexagon_nn_output*)outputs;
    params.outputsLen = outputsLen;
    sts = batch_append_ops(_h, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_const_node_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen)
{
    int sts = -1;
    batch_ops_params params;

    params.op = HEXNN_BATCH_OP_APPEND_CONST_NODE;
    params.node_id = node_id;
    params.batches = batches;
    params.height = height;
    params.width = width;
    params.depth = depth;
    params.data = (unsigned char*)data;
    params.dataLen = dataLen;
    sts = batch_append_ops(_h, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_empty_const_node_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size)
{
    int sts = -1;
    batch_ops_params params;

    params.op = HEXNN_BATCH_OP_APPEND_EMPTY_CONST_NODE;
    params.node_id = node_id;
    params.batches = batches;
    params.height = height;
    params.width = width;
    params.depth = depth;
    params.size = size;
    sts = batch_append_ops(_h, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_populate_const_node_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        const unsigned char* data, int dataLen, unsigned int target_offset)
{
    int sts = -1;
    batch_ops_params params;

    params.op = HEXNN_BATCH_OP_POPULATE_CONST_NODE;
    params.node_id = node_id;
    params.data = (unsigned char*)data;
    params.dataLen = dataLen;
    params.target_offset = target_offset;
    sts =batch_append_ops(_h, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_prepare_impl(remote_handle64 _h, hexagon_nn_nn_id id)
{
    int sts = -1;
    // setup batch ops' ion memory
    unsigned char *poi;
    unsigned int size;
    copy_batch_ops_to_ion_memory(_h, id, &poi, &size);

    sts = stub_hexagon_nn_domains_populate_graph(_h, id, poi, size);

    // free up the memory
    free_batch_op_memory(_h, id);
    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_impl(remote_handle64 _h, hexagon_nn_nn_id id,
        unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in,
        const unsigned char* data_in, int data_inLen,
        unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out,
        unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return(stub_hexagon_nn_domains_execute(_h, id, batches_in, height_in, width_in, depth_in, data_in, data_inLen,
            batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_teardown_impl(remote_handle64 _h, hexagon_nn_nn_id id)
{
    int sts;
    sts = stub_hexagon_nn_domains_teardown(_h, id);

    if(sts == 0)
    {
        sts = remove_nn_id(_h, id);
    }
    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_read_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out,
        unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return(stub_hexagon_nn_domains_variable_read(_h, id, node_id, output_index,
            batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_write_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth,
        const unsigned char* data_in, int data_inLen)
{
    return(stub_hexagon_nn_domains_variable_write(_h, id, node_id,
            output_index, batches, height, width, depth, data_in, data_inLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_write_flat_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, const unsigned char* data_in, int data_inLen)
{
    return(stub_hexagon_nn_domains_variable_write_flat(_h, id, node_id, output_index, data_in, data_inLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_powersave_level_impl(remote_handle64 _h, unsigned int level)
{
    return(stub_hexagon_nn_domains_set_powersave_level(_h, level));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_powersave_details_impl(remote_handle64 _h,
        hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    return(stub_hexagon_nn_domains_set_powersave_details(_h, corner, dcvs, latency));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_perfinfo_impl(remote_handle64 _h, hexagon_nn_nn_id id,
        hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items)
{
    return(stub_hexagon_nn_domains_get_perfinfo(_h, id, info_out, info_outLen, n_items));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_reset_perfinfo_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int event)
{
    return(stub_hexagon_nn_domains_reset_perfinfo(_h, id, event));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_last_execution_cycles_impl(remote_handle64 _h, hexagon_nn_nn_id id,
        unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return(stub_hexagon_nn_domains_last_execution_cycles(_h, id, cycles_lo, cycles_hi));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_version_impl(remote_handle64 _h, int* ver)
{
    return(stub_hexagon_nn_domains_version(_h, ver));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_op_name_to_id_impl(remote_handle64 _h, const char* name, unsigned int* node_id)
{
    return(stub_hexagon_nn_domains_op_name_to_id(_h, name, node_id));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_op_id_to_name_impl(remote_handle64 _h, unsigned int node_id, char* name, int nameLen)
{
    return(stub_hexagon_nn_domains_op_id_to_name(_h, node_id, name, nameLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_num_nodes_in_graph_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* num_nodes)
{
    return(stub_hexagon_nn_domains_get_num_nodes_in_graph(_h, id, num_nodes));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_disable_dcvs_impl(remote_handle64 _h)
{
    return(stub_hexagon_nn_domains_disable_dcvs(_h));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_GetHexagonBinaryVersion_impl(remote_handle64 _h, int* ver)
{
    return(stub_hexagon_nn_domains_GetHexagonBinaryVersion(_h, ver));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_PrintLog_impl(remote_handle64 _h, const unsigned char* buf, int bufLen)
{
    return(stub_hexagon_nn_domains_PrintLog(_h, buf, bufLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_new_impl(remote_handle64 _h, hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen)
{
    return(stub_hexagon_nn_domains_execute_new(_h, id, inputs, inputsLen, outputs, outputsLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_with_info_impl(remote_handle64 _h, hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info)
{
    return(stub_hexagon_nn_domains_execute_with_info(_h, id, inputs, inputsLen, outputs, outputsLen, execute_info));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_init_with_info_impl(remote_handle64 _h, hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info)
{
    return(stub_hexagon_nn_domains_init_with_info(_h, g, info));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_nodetype_impl(remote_handle64 _h, hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type)
{
    return(stub_hexagon_nn_domains_get_nodetype(_h, graph_id, node_id, node_type));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_multi_execution_cycles_impl(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return(stub_hexagon_nn_domains_multi_execution_cycles(_h, id, cycles_lo, cycles_hi));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_power_impl(remote_handle64 _h, int type)
{
    return(stub_hexagon_nn_domains_get_power(_h, type));
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_graph_option_impl(remote_handle64 _h, hexagon_nn_nn_id id, const char* name, int value)
{
    return(stub_hexagon_nn_domains_set_graph_option(_h, id, name, value));
}

#ifdef  __QAIC_STUB
#undef __QAIC_STUB
#endif //__QAIC_STUB
#define __QAIC_STUB(ff) stub_ ## ff

#include "hexagon_nn_domains_stub.c"
