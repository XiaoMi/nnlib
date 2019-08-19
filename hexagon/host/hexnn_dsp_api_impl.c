
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
#include "hexnn_dsp_api.h"
#include "hexnn_graph_wrapper.hpp"


__QAIC_STUB_EXPORT int hexagon_nn_config_impl(void)
{
    return stub_hexagon_nn_config();
}

__QAIC_STUB_EXPORT int hexagon_nn_config_with_options_impl(const hexagon_nn_uint_option* uint_options,
        int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return(stub_hexagon_nn_config_with_options(uint_options, uint_optionsLen, string_options, string_optionsLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_graph_config_impl(hexagon_nn_nn_id id, const hexagon_nn_uint_option* uint_options,
        int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return(stub_hexagon_nn_graph_config(id, uint_options, uint_optionsLen, string_options, string_optionsLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_get_dsp_offset_impl(unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr)
{
    return(stub_hexagon_nn_get_dsp_offset(libhexagon_addr, fastrpc_shell_addr));
}

__QAIC_STUB_EXPORT int hexagon_nn_init_impl(hexagon_nn_nn_id* g)
{
    int sts;
    sts = stub_hexagon_nn_init(g);

    if(sts == 0)
    {
        sts = add_nn_id(0, *g);
    }
    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_set_debug_level_impl(hexagon_nn_nn_id id, int level)
{
    return(stub_hexagon_nn_set_debug_level(id, level));
}

__QAIC_STUB_EXPORT int hexagon_nn_snpprint_impl(hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return(stub_hexagon_nn_snpprint(id, buf, bufLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_getlog_impl(hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return(stub_hexagon_nn_getlog(id, buf, bufLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_append_node_impl(hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int operation, hexagon_nn_padding_type padding,
        const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen)
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
    sts = batch_append_ops(0, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_append_const_node_impl(hexagon_nn_nn_id id, unsigned int node_id,
        unsigned int batches, unsigned int height, unsigned int width, unsigned int depth,
        const unsigned char* data, int dataLen)
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
    sts = batch_append_ops(0, id, params);

    return sts;
}


__QAIC_STUB_EXPORT int hexagon_nn_append_empty_const_node_impl(hexagon_nn_nn_id id, unsigned int node_id,
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
    sts = batch_append_ops(0, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_populate_const_node_impl(hexagon_nn_nn_id id, unsigned int node_id,
        const unsigned char* data, int dataLen, unsigned int target_offset)
{
    int sts = -1;
    batch_ops_params params;

    params.op = HEXNN_BATCH_OP_POPULATE_CONST_NODE;
    params.node_id = node_id;
    params.data = (unsigned char*)data;
    params.dataLen = dataLen;
    params.target_offset = target_offset;
    sts = batch_append_ops(0, id, params);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_prepare_impl(hexagon_nn_nn_id id)
{
    int sts = -1;
    // setup batch ops' ion memory
    unsigned char *poi;
    unsigned int size;

    // allocate ION memory and copy
    sts = copy_batch_ops_to_ion_memory(0,id, &poi, &size);

    if (sts == 0) {
        sts = stub_hexagon_nn_populate_graph(id, poi, size);
    }

    // free up the memory
    free_batch_op_memory(0, id);

    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_execute_impl(hexagon_nn_nn_id id,
        unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in,
        const unsigned char* data_in, int data_inLen, unsigned int* batches_out,
        unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out,
        int data_outLen, unsigned int* data_len_out)
{
    return(stub_hexagon_nn_execute(id, batches_in, height_in, width_in, depth_in, data_in, data_inLen,
            batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out));
}

__QAIC_STUB_EXPORT int hexagon_nn_teardown_impl(hexagon_nn_nn_id id)
{
    int sts;
    sts = stub_hexagon_nn_teardown(id);

    if(sts == 0)
    {
        sts = remove_nn_id(0, id);
    }
    return sts;
}

__QAIC_STUB_EXPORT int hexagon_nn_variable_read_impl(hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out,
        unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return(stub_hexagon_nn_variable_read(id, node_id, output_index, batches_out, height_out, width_out, depth_out,
            data_out, data_outLen, data_len_out));
}

__QAIC_STUB_EXPORT int hexagon_nn_variable_write_impl(hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth,
        const unsigned char* data_in, int data_inLen)
{
    return(stub_hexagon_nn_variable_write(id, node_id, output_index, batches, height, width, depth, data_in, data_inLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_variable_write_flat_impl(hexagon_nn_nn_id id, unsigned int node_id,
        int output_index, const unsigned char* data_in, int data_inLen)
{
    return(stub_hexagon_nn_variable_write_flat(id, node_id, output_index, data_in, data_inLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_set_powersave_level_impl(unsigned int level)
{
    return(stub_hexagon_nn_set_powersave_level(level));
}

__QAIC_STUB_EXPORT int hexagon_nn_set_powersave_details_impl(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    return(stub_hexagon_nn_set_powersave_details(corner, dcvs, latency));
}

__QAIC_STUB_EXPORT int hexagon_nn_get_perfinfo_impl(hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items)
{
    return(stub_hexagon_nn_get_perfinfo(id, info_out, info_outLen, n_items));
}

__QAIC_STUB_EXPORT int hexagon_nn_reset_perfinfo_impl(hexagon_nn_nn_id id, unsigned int event)
{
    return(stub_hexagon_nn_reset_perfinfo(id, event));
}

__QAIC_STUB_EXPORT int hexagon_nn_last_execution_cycles_impl(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return(stub_hexagon_nn_last_execution_cycles(id,cycles_lo, cycles_hi));
}

__QAIC_STUB_EXPORT int hexagon_nn_version_impl(int* ver)
{
    return(stub_hexagon_nn_version(ver));
}

__QAIC_STUB_EXPORT int hexagon_nn_op_name_to_id_impl(const char* name, unsigned int* node_id)
{
    return(stub_hexagon_nn_op_name_to_id(name, node_id));
}

__QAIC_STUB_EXPORT int hexagon_nn_op_id_to_name_impl(unsigned int node_id, char* name, int nameLen)
{
    return(stub_hexagon_nn_op_id_to_name(node_id, name, nameLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_get_num_nodes_in_graph_impl(hexagon_nn_nn_id id, unsigned int* num_nodes)
{
    return(stub_hexagon_nn_get_num_nodes_in_graph(id, num_nodes));
}

__QAIC_STUB_EXPORT int hexagon_nn_disable_dcvs_impl(void)
{
    return(stub_hexagon_nn_disable_dcvs());
}

__QAIC_STUB_EXPORT int hexagon_nn_GetHexagonBinaryVersion_impl(int* ver)
{
    return(stub_hexagon_nn_GetHexagonBinaryVersion(ver));
}

__QAIC_STUB_EXPORT int hexagon_nn_PrintLog_impl(const unsigned char* buf, int bufLen)
{
    return(stub_hexagon_nn_PrintLog(buf, bufLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_execute_new_impl(hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen)
{
    return(stub_hexagon_nn_execute_new(id, inputs, inputsLen, outputs, outputsLen));
}

__QAIC_STUB_EXPORT int hexagon_nn_execute_with_info_impl(hexagon_nn_nn_id id,
        const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info)
{
    return(stub_hexagon_nn_execute_with_info(id, inputs, inputsLen, outputs, outputsLen, execute_info));
}

__QAIC_STUB_EXPORT int hexagon_nn_init_with_info_impl(hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info)
{
    return(stub_hexagon_nn_init_with_info(g, info));
}

__QAIC_STUB_EXPORT int hexagon_nn_get_nodetype_impl(hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type)
{
    return(stub_hexagon_nn_get_nodetype(graph_id, node_id, node_type));
}

__QAIC_STUB_EXPORT int hexagon_nn_multi_execution_cycles_impl(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return(stub_hexagon_nn_multi_execution_cycles(id, cycles_lo, cycles_hi));
}

__QAIC_STUB_EXPORT int hexagon_nn_get_power_impl(int type)
{
    return(stub_hexagon_nn_get_power(type));
}

__QAIC_STUB_EXPORT int hexagon_nn_set_graph_option_impl(hexagon_nn_nn_id id, const char* name, int value)
{
    return(stub_hexagon_nn_set_graph_option(id, name, value));
}

#ifdef  __QAIC_STUB
#undef __QAIC_STUB
#endif //__QAIC_STUB
#define __QAIC_STUB(ff) stub_ ## ff

#include "hexagon_nn_stub.c"
