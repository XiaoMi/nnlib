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

#ifndef HEXAGON_NN_HEXNN_DSP_SMART_WRAPPER_API_H
#define HEXAGON_NN_HEXNN_DSP_SMART_WRAPPER_API_H

#include "hexagon_nn.h"
#include "hexnn_dsp_api.h"
#include "hexnn_dsp_domains_api.h"
#include "hexnn_soc_defines.h"

#ifndef __QAIC_STUB_EXPORT
#define __QAIC_STUB_EXPORT
#endif // __QAIC_STUB_EXPORT

// Function pointers
__QAIC_STUB_EXPORT int (*hexagon_nn_config_fnptr)(void) = &hexagon_nn_config_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_config_with_options_fnptr)(const hexagon_nn_uint_option*, int, const hexagon_nn_string_option*, int) = &hexagon_nn_config_with_options_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_graph_config_fnptr)(hexagon_nn_nn_id, const hexagon_nn_uint_option*, int, const hexagon_nn_string_option*, int) = &hexagon_nn_graph_config_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_get_dsp_offset_fnptr)(unsigned int*, unsigned int*) = &hexagon_nn_get_dsp_offset_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_init_fnptr)(hexagon_nn_nn_id*) = &hexagon_nn_init_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_set_debug_level_fnptr)(hexagon_nn_nn_id, int) = &hexagon_nn_set_debug_level_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_snpprint_fnptr)(hexagon_nn_nn_id, unsigned char*, int) = &hexagon_nn_snpprint_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_getlog_fnptr)(hexagon_nn_nn_id, unsigned char*, int) = &hexagon_nn_getlog_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_append_node_fnptr)(hexagon_nn_nn_id, unsigned int, unsigned int, hexagon_nn_padding_type, const hexagon_nn_input*, int, const hexagon_nn_output*, int) = &hexagon_nn_append_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_append_const_node_fnptr)(hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int) = &hexagon_nn_append_const_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_append_empty_const_node_fnptr)(hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) = &hexagon_nn_append_empty_const_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_populate_const_node_fnptr)(hexagon_nn_nn_id, unsigned int, const unsigned char*, int, unsigned int) = &hexagon_nn_populate_const_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_prepare_fnptr)(hexagon_nn_nn_id id) = &hexagon_nn_prepare_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_execute_fnptr)(hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned char*, int, unsigned int*) = &hexagon_nn_execute_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_teardown_fnptr)(hexagon_nn_nn_id) = &hexagon_nn_teardown_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_variable_read_fnptr)(hexagon_nn_nn_id, unsigned int, int, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned char*, int, unsigned int*) = &hexagon_nn_variable_read_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_variable_write_fnptr)(hexagon_nn_nn_id, unsigned int, int, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int) = &hexagon_nn_variable_write_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_variable_write_flat_fnptr)(hexagon_nn_nn_id, unsigned int, int, const unsigned char*, int) = &hexagon_nn_variable_write_flat_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_set_powersave_level_fnptr)(unsigned int) = &hexagon_nn_set_powersave_level_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_set_powersave_details_fnptr)(hexagon_nn_corner_type, hexagon_nn_dcvs_type, unsigned int) = &hexagon_nn_set_powersave_details_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_get_perfinfo_fnptr)(hexagon_nn_nn_id, hexagon_nn_perfinfo*, int, unsigned int*) = &hexagon_nn_get_perfinfo_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_reset_perfinfo_fnptr)(hexagon_nn_nn_id, unsigned int) = &hexagon_nn_reset_perfinfo_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_last_execution_cycles_fnptr)(hexagon_nn_nn_id, unsigned int*, unsigned int*) = &hexagon_nn_last_execution_cycles_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_version_fnptr)(int*) = &hexagon_nn_version_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_op_name_to_id_fnptr)(const char*, unsigned int*) = &hexagon_nn_op_name_to_id_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_op_id_to_name_fnptr)(unsigned int, char*, int) = &hexagon_nn_op_id_to_name_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_get_num_nodes_in_graph_fnptr)(hexagon_nn_nn_id, unsigned int*) = &hexagon_nn_get_num_nodes_in_graph_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_disable_dcvs_fnptr)(void) = &hexagon_nn_disable_dcvs_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_GetHexagonBinaryVersion_fnptr)(int*) = &hexagon_nn_GetHexagonBinaryVersion_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_PrintLog_fnptr)(const unsigned char*, int) = &hexagon_nn_PrintLog_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_execute_new_fnptr)(hexagon_nn_nn_id, const hexagon_nn_tensordef*, int, hexagon_nn_tensordef*, int) = &hexagon_nn_execute_new_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_execute_with_info_fnptr)(hexagon_nn_nn_id, const hexagon_nn_tensordef*, int, hexagon_nn_tensordef*, int, hexagon_nn_execute_info*) = &hexagon_nn_execute_with_info_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_init_with_info_fnptr)(hexagon_nn_nn_id*, const hexagon_nn_initinfo*) = &hexagon_nn_init_with_info_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_get_nodetype_fnptr)(hexagon_nn_nn_id, hexagon_nn_nn_id, unsigned int*) = &hexagon_nn_get_nodetype_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_multi_execution_cycles_fnptr)(hexagon_nn_nn_id, unsigned int*, unsigned int*) = &hexagon_nn_multi_execution_cycles_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_get_power_fnptr)(int) = &hexagon_nn_get_power_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_set_graph_option_fnptr)(hexagon_nn_nn_id, const char*, int) = &hexagon_nn_set_graph_option_impl;

__QAIC_STUB_EXPORT int (*hexagon_nn_domains_config_fnptr)(remote_handle64) = &hexagon_nn_domains_config_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_config_with_options_fnptr)(remote_handle64, const hexagon_nn_uint_option*, int, const hexagon_nn_string_option*, int) = &hexagon_nn_domains_config_with_options_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_graph_config_fnptr)(remote_handle64, hexagon_nn_nn_id, const hexagon_nn_uint_option*, int, const hexagon_nn_string_option*, int) = &hexagon_nn_domains_graph_config_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_get_dsp_offset_fnptr)(remote_handle64, unsigned int*, unsigned int* ) = &hexagon_nn_domains_get_dsp_offset_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_init_fnptr)(remote_handle64, hexagon_nn_nn_id*) = &hexagon_nn_domains_init_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_set_debug_level_fnptr)(remote_handle64, hexagon_nn_nn_id, int) = &hexagon_nn_domains_set_debug_level_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_snpprint_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned char*, int) = &hexagon_nn_domains_snpprint_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_getlog_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned char*, int) = &hexagon_nn_domains_getlog_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_append_node_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, unsigned int, hexagon_nn_padding_type, const hexagon_nn_input*, int, const hexagon_nn_output*, int) = &hexagon_nn_domains_append_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_append_const_node_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int) = &hexagon_nn_domains_append_const_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_append_empty_const_node_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) = &hexagon_nn_domains_append_empty_const_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_populate_const_node_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, const unsigned char*, int, unsigned int) = &hexagon_nn_domains_populate_const_node_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_prepare_fnptr)(remote_handle64, hexagon_nn_nn_id) = &hexagon_nn_domains_prepare_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_execute_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned char*, int, unsigned int*) = &hexagon_nn_domains_execute_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_teardown_fnptr)(remote_handle64, hexagon_nn_nn_id) = &hexagon_nn_domains_teardown_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_variable_read_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, int, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned char*, int, unsigned int*) = &hexagon_nn_domains_variable_read_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_variable_write_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, int, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int) = &hexagon_nn_domains_variable_write_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_variable_write_flat_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int, int, const unsigned char*, int) = &hexagon_nn_domains_variable_write_flat_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_set_powersave_level_fnptr)(remote_handle64, unsigned int) = &hexagon_nn_domains_set_powersave_level_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_set_powersave_details_fnptr)(remote_handle64, hexagon_nn_corner_type, hexagon_nn_dcvs_type, unsigned int) = &hexagon_nn_domains_set_powersave_details_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_get_perfinfo_fnptr)(remote_handle64, hexagon_nn_nn_id, hexagon_nn_perfinfo*, int, unsigned int*) = &hexagon_nn_domains_get_perfinfo_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_reset_perfinfo_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int) = &hexagon_nn_domains_reset_perfinfo_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_last_execution_cycles_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int*, unsigned int*) = &hexagon_nn_domains_last_execution_cycles_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_version_fnptr)(remote_handle64, int*) = &hexagon_nn_domains_version_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_op_name_to_id_fnptr)(remote_handle64, const char*, unsigned int*) = &hexagon_nn_domains_op_name_to_id_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_op_id_to_name_fnptr)(remote_handle64, unsigned int, char*, int) = &hexagon_nn_domains_op_id_to_name_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_get_num_nodes_in_graph_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int*) = &hexagon_nn_domains_get_num_nodes_in_graph_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_disable_dcvs_fnptr)(remote_handle64) = &hexagon_nn_domains_disable_dcvs_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_GetHexagonBinaryVersion_fnptr)(remote_handle64, int*) = &hexagon_nn_domains_GetHexagonBinaryVersion_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_PrintLog_fnptr)(remote_handle64, const unsigned char*, int) = &hexagon_nn_domains_PrintLog_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_execute_new_fnptr)(remote_handle64, hexagon_nn_nn_id, const hexagon_nn_tensordef*, int, hexagon_nn_tensordef*, int) = &hexagon_nn_domains_execute_new_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_execute_with_info_fnptr)(remote_handle64, hexagon_nn_nn_id, const hexagon_nn_tensordef*, int, hexagon_nn_tensordef*, int, hexagon_nn_execute_info*) = &hexagon_nn_domains_execute_with_info_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_init_with_info_fnptr)(remote_handle64, hexagon_nn_nn_id*, const hexagon_nn_initinfo*) = &hexagon_nn_domains_init_with_info_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_get_nodetype_fnptr)(remote_handle64, hexagon_nn_nn_id, hexagon_nn_nn_id, unsigned int*) = &hexagon_nn_domains_get_nodetype_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_multi_execution_cycles_fnptr)(remote_handle64, hexagon_nn_nn_id, unsigned int*, unsigned int*) = &hexagon_nn_domains_multi_execution_cycles_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_get_power_fnptr)(remote_handle64, int) = &hexagon_nn_domains_get_power_impl;
__QAIC_STUB_EXPORT int (*hexagon_nn_domains_set_graph_option_fnptr)(remote_handle64, hexagon_nn_nn_id, const char*, int) = &hexagon_nn_domains_set_graph_option_impl;

#endif //HEXAGON_NN_HEXNN_DSP_SMART_WRAPPER_API_H
