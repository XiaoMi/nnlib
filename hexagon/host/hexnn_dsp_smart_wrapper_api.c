
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
#include "unistd.h"
#include "stdlib.h"
#include "sys/stat.h"
#include "fcntl.h"

#include "hexnn_dsp_smart_wrapper_api.h"

#define select_stub_fn(domains_stub_fnptr, non_domains_stub_fnptr, h, ...) \
    (domains) ? (*domains_stub_fnptr)(h, ##__VA_ARGS__) : \
                (*non_domains_stub_fnptr)(__VA_ARGS__)
#define CHECK_DOMAINS_AND_OPEN_HANDLE \
    if (check_and_open_handle() != 0) {\
        return -1;\
    }

static remote_handle64 h = 0x0;
static int domains = 1;


soc_model get_soc_model() {
    soc_model s_soc_model = UNKNOWN_SOC;

    int fd;
    if (!access("/sys/devices/soc0/soc_id", F_OK)) {
        fd = open("/sys/devices/soc0/soc_id", O_RDONLY);
    } 
    else {
        fd = open("/sys/devices/system/soc/soc0/id", O_RDONLY);
    }

    if (fd != -1) {
        char raw_buf[5];
        int soc_id;
        read(fd, raw_buf,4);
        raw_buf[4] = 0;
        soc_id = atoi(raw_buf);
        close(fd);

        if (IS_QTI_SOC_SDM855(soc_id))
            s_soc_model = SD855;
        else if (IS_QTI_SOC_SDM845(soc_id))
            s_soc_model = SD845;
        else if (IS_QTI_SOC_8998(soc_id))
            s_soc_model = SD835;
        else if (IS_QTI_SOC_8996(soc_id))
            s_soc_model = SD820;
        else if (IS_QTI_SOC_SDM710(soc_id))
            s_soc_model = SD710;
        else if (IS_QTI_SOC_SDM670(soc_id))
            s_soc_model = SD670;
        else if (IS_QTI_SOC_SDM660(soc_id))
            s_soc_model = SD660;
        else if (IS_QTI_SOC_SM6150(soc_id))
            s_soc_model = SD6150;
        else if (IS_QTI_SOC_SM7150(soc_id))
            s_soc_model = SD7150;
        else if (IS_QTI_SOC_QCS405(soc_id))
            s_soc_model = QCS405;
        else if (IS_QTI_SOC_SM6125(soc_id))
            s_soc_model = SD6125;
        else if (IS_QTI_SOC_QCS403(soc_id))
            s_soc_model = QCS403;
        else
            s_soc_model = UNKNOWN_SOC;
    }
    return s_soc_model;
}

int get_skel_handle(remote_handle64*  handle) {
    soc_model model = get_soc_model();

    char* uri = NULL;
    switch(model) {
        case SD820:
            // non-domains case
            domains = 0;
            return 0;
        case SD660:
            uri = "file:///libhexagon_nn_skel.so?hexagon_nn_domains_skel_handle_invoke&_modver=1.0&_dom=cdsp";
            break;
        case SD835:
            uri = "file:///libhexagon_nn_skel.so?hexagon_nn_domains_skel_handle_invoke&_modver=1.0&_dom=adsp";
            break;
        case SD845:
        case SD670:
        case SD710:
        case SD7150:
            uri = "file:///libhexagon_nn_skel_v65.so?hexagon_nn_domains_skel_handle_invoke&_modver=1.0&_dom=cdsp";
            break;
        case SD6150:
        case SD855:
        case QCS405:
        case SD6125:
        case QCS403:
            uri = "file:///libhexagon_nn_skel_v66.so?hexagon_nn_domains_skel_handle_invoke&_modver=1.0&_dom=cdsp";
            break;
        default:
            uri = NULL;
    }

    if (uri == NULL){
        return -1;
    }

    int rc = hexagon_nn_domains_open_impl(uri, handle);
    if (rc != 0 || *handle == 0x0) {
        return -1;
    }

    return 0;
}

int check_and_open_handle() {
    if (domains && h == 0x0) {
        return get_skel_handle(&h);
    }

    return 0;
}

__QAIC_STUB_EXPORT int hexagon_nn_config(void)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_config_fnptr, hexagon_nn_config_fnptr, h);
}

__QAIC_STUB_EXPORT int hexagon_nn_config_with_options(const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_config_with_options_fnptr, hexagon_nn_config_with_options_fnptr, h, uint_options, uint_optionsLen, string_options, string_optionsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_graph_config(hexagon_nn_nn_id id, const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_graph_config_fnptr, hexagon_nn_graph_config_fnptr, h, id, uint_options, uint_optionsLen, string_options, string_optionsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_get_dsp_offset(unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_get_dsp_offset_fnptr, hexagon_nn_get_dsp_offset_fnptr, h, libhexagon_addr, fastrpc_shell_addr);
}

__QAIC_STUB_EXPORT int hexagon_nn_init(hexagon_nn_nn_id* g)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_init_fnptr, hexagon_nn_init_fnptr, h, g);
}

__QAIC_STUB_EXPORT int hexagon_nn_set_debug_level(hexagon_nn_nn_id id, int level)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_set_debug_level_fnptr, hexagon_nn_set_debug_level_fnptr, h, id, level);
}

__QAIC_STUB_EXPORT int hexagon_nn_snpprint(hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_snpprint_fnptr, hexagon_nn_snpprint_fnptr, h, id, buf, bufLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_getlog(hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_getlog_fnptr, hexagon_nn_getlog_fnptr, h, id, buf, bufLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_append_node(hexagon_nn_nn_id id, unsigned int node_id, unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_append_node_fnptr, hexagon_nn_append_node_fnptr, h, id, node_id, operation, padding, inputs, inputsLen, outputs, outputsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_append_const_node(hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_append_const_node_fnptr, hexagon_nn_append_const_node_fnptr, h, id, node_id, batches, height, width, depth, data, dataLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_append_empty_const_node(hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_append_empty_const_node_fnptr, hexagon_nn_append_empty_const_node_fnptr, h, id, node_id, batches, height, width, depth, size);
}

__QAIC_STUB_EXPORT int hexagon_nn_populate_const_node(hexagon_nn_nn_id id, unsigned int node_id, const unsigned char* data, int dataLen, unsigned int target_offset)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_populate_const_node_fnptr, hexagon_nn_populate_const_node_fnptr, h, id, node_id, data, dataLen, target_offset);
}

__QAIC_STUB_EXPORT int hexagon_nn_prepare(hexagon_nn_nn_id id)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_prepare_fnptr, hexagon_nn_prepare_fnptr, h, id);
}

__QAIC_STUB_EXPORT int hexagon_nn_execute(hexagon_nn_nn_id id, unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in, const unsigned char* data_in, int data_inLen, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_execute_fnptr, hexagon_nn_execute_fnptr, h, id, batches_in, height_in, width_in, depth_in, data_in, data_inLen, batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out);
}

__QAIC_STUB_EXPORT int hexagon_nn_teardown(hexagon_nn_nn_id id)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_teardown_fnptr, hexagon_nn_teardown_fnptr, h, id);
}

__QAIC_STUB_EXPORT int hexagon_nn_variable_read(hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_variable_read_fnptr, hexagon_nn_variable_read_fnptr, h, id, node_id, output_index, batches_out, height_out, width_out, depth_out, data_out, data_outLen, data_len_out);
}

__QAIC_STUB_EXPORT int hexagon_nn_variable_write(hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data_in, int data_inLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_variable_write_fnptr, hexagon_nn_variable_write_fnptr, h, id, node_id, output_index, batches, height, width, depth, data_in, data_inLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_variable_write_flat(hexagon_nn_nn_id id, unsigned int node_id, int output_index, const unsigned char* data_in, int data_inLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_variable_write_flat_fnptr, hexagon_nn_variable_write_flat_fnptr, h, id, node_id, output_index, data_in, data_inLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_set_powersave_level(unsigned int level)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_set_powersave_level_fnptr, hexagon_nn_set_powersave_level_fnptr, h, level);
}

__QAIC_STUB_EXPORT int hexagon_nn_set_powersave_details(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_set_powersave_details_fnptr, hexagon_nn_set_powersave_details_fnptr, h, corner, dcvs, latency);
}

__QAIC_STUB_EXPORT int hexagon_nn_get_perfinfo(hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_get_perfinfo_fnptr, hexagon_nn_get_perfinfo_fnptr, h, id, info_out, info_outLen, n_items);
}

__QAIC_STUB_EXPORT int hexagon_nn_reset_perfinfo(hexagon_nn_nn_id id, unsigned int event)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_reset_perfinfo_fnptr, hexagon_nn_reset_perfinfo_fnptr, h, id, event);
}

__QAIC_STUB_EXPORT int hexagon_nn_last_execution_cycles(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_last_execution_cycles_fnptr, hexagon_nn_last_execution_cycles_fnptr, h, id, cycles_lo, cycles_hi);
}

__QAIC_STUB_EXPORT int hexagon_nn_version(int* ver)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_version_fnptr, hexagon_nn_version_fnptr, h, ver);
}

__QAIC_STUB_EXPORT int hexagon_nn_op_name_to_id(const char* name, unsigned int* node_id)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_op_name_to_id_fnptr, hexagon_nn_op_name_to_id_fnptr, h, name, node_id);
}

__QAIC_STUB_EXPORT int hexagon_nn_op_id_to_name(unsigned int node_id, char* name, int nameLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_op_id_to_name_fnptr, hexagon_nn_op_id_to_name_fnptr, h, node_id, name, nameLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_get_num_nodes_in_graph(hexagon_nn_nn_id id, unsigned int* num_nodes)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_get_num_nodes_in_graph_fnptr, hexagon_nn_get_num_nodes_in_graph_fnptr, h, id, num_nodes);
}

__QAIC_STUB_EXPORT int hexagon_nn_disable_dcvs(void)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_disable_dcvs_fnptr, hexagon_nn_disable_dcvs_fnptr, h);
}

__QAIC_STUB_EXPORT int hexagon_nn_GetHexagonBinaryVersion(int* ver)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_GetHexagonBinaryVersion_fnptr, hexagon_nn_GetHexagonBinaryVersion_fnptr, h, ver);
}

__QAIC_STUB_EXPORT int hexagon_nn_PrintLog(const unsigned char* buf, int bufLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_PrintLog_fnptr, hexagon_nn_PrintLog_fnptr, h, buf, bufLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_execute_new(hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_execute_new_fnptr, hexagon_nn_execute_new_fnptr, h, id, inputs, inputsLen, outputs, outputsLen);
}

__QAIC_STUB_EXPORT int hexagon_nn_execute_with_info(hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_execute_with_info_fnptr, hexagon_nn_execute_with_info_fnptr, h, id, inputs, inputsLen, outputs, outputsLen, execute_info);
}

__QAIC_STUB_EXPORT int hexagon_nn_init_with_info(hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_init_with_info_fnptr, hexagon_nn_init_with_info_fnptr, h, g, info);
}

__QAIC_STUB_EXPORT int hexagon_nn_get_nodetype(hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_get_nodetype_fnptr, hexagon_nn_get_nodetype_fnptr, h, graph_id, node_id, node_type);
}

__QAIC_STUB_EXPORT int hexagon_nn_multi_execution_cycles(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_multi_execution_cycles_fnptr, hexagon_nn_multi_execution_cycles_fnptr, h, id, cycles_lo, cycles_hi);
}

__QAIC_STUB_EXPORT int hexagon_nn_get_power(int type)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_get_power_fnptr, hexagon_nn_get_power_fnptr, h, type);
}

__QAIC_STUB_EXPORT int hexagon_nn_set_graph_option(hexagon_nn_nn_id id, const char* name, int value)
{
    CHECK_DOMAINS_AND_OPEN_HANDLE
    return select_stub_fn(hexagon_nn_domains_set_graph_option_fnptr, hexagon_nn_set_graph_option_fnptr, h, id, name, value);
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_config(remote_handle64 _h)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_config_with_options(remote_handle64 _h, const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_graph_config(remote_handle64 _h, hexagon_nn_nn_id id, const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_dsp_offset(remote_handle64 _h, unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_init(remote_handle64 _h, hexagon_nn_nn_id* g)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_debug_level(remote_handle64 _h, hexagon_nn_nn_id id, int level)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_snpprint(remote_handle64 _h, hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_getlog(remote_handle64 _h, hexagon_nn_nn_id id, unsigned char* buf, int bufLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_const_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_append_empty_const_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_populate_const_node(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, const unsigned char* data, int dataLen, unsigned int target_offset)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_prepare(remote_handle64 _h, hexagon_nn_nn_id id)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in, const unsigned char* data_in, int data_inLen, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_teardown(remote_handle64 _h, hexagon_nn_nn_id id)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_read(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_write(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data_in, int data_inLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_variable_write_flat(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id, int output_index, const unsigned char* data_in, int data_inLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_powersave_level(remote_handle64 _h, unsigned int level)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_powersave_details(remote_handle64 _h, hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_perfinfo(remote_handle64 _h, hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_reset_perfinfo(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int event)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_last_execution_cycles(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_version(remote_handle64 _h, int* ver)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_op_name_to_id(remote_handle64 _h, const char* name, unsigned int* node_id)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_op_id_to_name(remote_handle64 _h, unsigned int node_id, char* name, int nameLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_num_nodes_in_graph(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* num_nodes)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_disable_dcvs(remote_handle64 _h)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_GetHexagonBinaryVersion(remote_handle64 _h, int* ver)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_PrintLog(remote_handle64 _h, const unsigned char* buf, int bufLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_new(remote_handle64 _h, hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_execute_with_info(remote_handle64 _h, hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_init_with_info(remote_handle64 _h, hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_nodetype(remote_handle64 _h, hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_multi_execution_cycles(remote_handle64 _h, hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_get_power(remote_handle64 _h, int type)
{
    return -1;
}

__QAIC_STUB_EXPORT int hexagon_nn_domains_set_graph_option(remote_handle64 _h, hexagon_nn_nn_id id, const char* name, int value)
{
    return -1;
}
