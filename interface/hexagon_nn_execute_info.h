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

// this file contains enums and structs used in the extraInfo part of hexagon_nn_execute_info

#ifndef HEXAGON_NN_EXECUTE_INFO_H
#define HEXAGON_NN_EXECUTE_INFO_H

enum hexagon_nn_execute_udo_err_type {
        UDO_EXE_OP_EXECUTE_FAILED,
        UDO_EXE_LIB_NOT_REGISTERED,
        UDO_EXE_INVALID_INPUTS_OUTPUTS_QUANTIZATION_TYPE
};

// struct in extraInfo of hexagon_nn_execute_info if result is NN_EXECUTE_ERROR
struct hexagon_nn_execute_error_complete_info {
        uint32_t exe_failure_node_id;
        uint32_t exe_failure_node_op_type;
};

// struct in extraInfo of hexagon_nn_execute_info if result is NN_EXECUTE_BUFFER_SIZE_ERROR
struct hexagon_nn_execute_buffer_size_error_complete_info {
        uint32_t exe_failure_node_id;
        uint32_t exe_failure_node_op_type;
};

// struct in extraInfo of hexagon_nn_execute_info if result is NN_EXECUTE_UDO_ERROR
struct hexagon_nn_execute_udo_error_complete_info {
        uint32_t exe_failure_node_id;
        enum hexagon_nn_execute_udo_err_type exe_failure_udo_err_type;
        int32_t exe_failure_snpe_udo_err_code;   // only applies if error type is UDO_EXE_OP_EXECUTE_FAILED
};

/*
 * struct in extraInfo of hexagon_nn_execute_info 
 * if result is NN_EXECUTE_ERROR, NN_EXECUTE_BUFFER_SIZE_ERROR, or NN_EXECUTE_UDO_ERROR
 */
struct hexagon_nn_execute_complete_info {
        union {
                struct hexagon_nn_execute_error_complete_info exe_err_info;
                struct hexagon_nn_execute_buffer_size_error_complete_info buffer_size_err_info;
                struct hexagon_nn_execute_udo_error_complete_info udo_err_info;
        };
};

#endif
