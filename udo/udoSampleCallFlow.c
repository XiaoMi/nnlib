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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hexagon_nn.h"
#include "hexagon_nn_execute_info.h"
#include "hexagon_nn_ops.h"
#include "SnpeUdo/UdoFlatten.h"
#include "SnpeUdo/UdoBase.h"
#include "rpcmem.h"

static void errorCodeFail(const char *file, int line, const char *expr, int code) {
    fprintf(stderr, "%s(%d): %s failed with %d\n", file, line, expr, code);
}

#define RUN_HEXNN_API(e, x) \
   do { \
     if ( (e = x) != 0 ) { \
       errorCodeFail(__FILE__, __LINE__, #x, e); \
       goto exit; \
     } \
     else { \
       printf("-->  %s success!!\n", #x); \
     } \
   } while(0);

#define RUN_HEXNN_API_EXIT(e, x) \
   do { \
     if ( (e = x) != 0 ) { \
       errorCodeFail(__FILE__, __LINE__, #x, e); \
     } \
     else { \
       printf("-->  %s success!!\n", #x); \
     } \
   } while(0);

#define CHECK_UDO_RES(e, x) \
   do { \
     if ( e != UDO_SUCCESS ) { \
       errorCodeFail(__FILE__, __LINE__, #x, e); \
       goto exit; \
     } \
   } while(0);


int main(int argc, char* argv[])
{
        rpcmem_init();

	int result = 0;
	int nnid = 0;
        enum hexagon_nn_udo_err udoRes = 999;

        uint8_t i [] = {2,3,4,5,6,7};  // input
        uint8_t r [] = {6,7,8,9,10,11};  // expected result
        char udoPackageName1[] = "udoExampleLib";
        char udoOpType1[] = "opUdoPlusOne";
        char udoOpType2[] = "opUdoPlusStat";

        SnpeUdo_Param_t staticParam1;
        char udoStaticName1[] = "plusTwo";
        staticParam1.paramType = SNPE_UDO_PARAMTYPE_SCALAR;
        (staticParam1.scalarParam).dataType = SNPE_UDO_DATATYPE_UINT_8;
        ((staticParam1.scalarParam).dataValue).uint8Value = (uint8_t)2;
        staticParam1.paramName = udoStaticName1;

        SnpeUdo_Param_t staticParam2;
        char udoStaticName2[] = "useD32";
        staticParam2.paramType = SNPE_UDO_PARAMTYPE_SCALAR;
        (staticParam2.scalarParam).dataType = SNPE_UDO_DATATYPE_UINT_8;
        ((staticParam2.scalarParam).dataValue).uint8Value = (uint8_t)1;
        staticParam2.paramName = udoStaticName2;

        SnpeUdo_Param_t* staticParamsList1[] = {&staticParam1, &staticParam2};
        uint32_t numStatic1 = 2;
        uint32_t flattenedStaticSize1 = 0;
        void* flattenedStatic1 = NULL;

        // static parameters need to be flattned before passing through fastrpc
        if (SnpeUdo_flattenStaticParams (staticParamsList1, numStatic1, &flattenedStaticSize1, &flattenedStatic1) != 0) {
                printf("flatten static params for udo library %s, udo op %s, node id %d failed\n", udoPackageName1, udoOpType2, 3);
                goto exit;
        }


	RUN_HEXNN_API(result, hexagon_nn_config());

	RUN_HEXNN_API(result, hexagon_nn_udo_register_lib("/udoLibs/udoExampleImplLib.so", &udoRes));
        CHECK_UDO_RES(udoRes, reg_lib_1);

	RUN_HEXNN_API(result, hexagon_nn_init(&nnid));

	RUN_HEXNN_API(result, hexagon_nn_set_debug_level(nnid, 5));

	RUN_HEXNN_API(result, hexagon_nn_append_const_node(nnid, 1, 1, 2, 3, 1, i, 6));

        static hexagon_nn_input layer_1_input_list[] = {
                { .src_id = 1, .output_idx = 0, },
        };
        static hexagon_nn_output layer_1_output_list[] = {
                { .rank = 4, .max_sizes = {10,10,10,10}, .elementsize = sizeof(uint8_t), },
        };

        RUN_HEXNN_API(result, hexagon_nn_append_node(
            nnid,           // Graph handle we're appending into
            2,             // Node identifier (any unique uint32)
            OP_Nop,           // Operation of this node (e.g. Concat, Relu)
            NN_PAD_NA,          // Padding type for this node
            layer_1_input_list,   // The list of inputs to this node
            1,                  //   How many elements in input list
            layer_1_output_list, // The list of outputs from this node
            1                   //   How many elements in output list
            ));

        static hexagon_nn_input layer_2_input_list[] = {
                { .src_id = 2, .output_idx = 0, },
        };
        static hexagon_nn_output layer_2_output_list[] = {
                { .rank = 4, .max_sizes = {10,10,10,10}, .elementsize = sizeof(uint8_t), },
        };



	RUN_HEXNN_API(result, hexagon_nn_append_udo_node(nnid, 3, udoPackageName1, udoOpType1, NULL, 0, layer_2_input_list, 1, layer_2_output_list, 1, &udoRes));
        CHECK_UDO_RES(udoRes, append_udo_node_1);

        static hexagon_nn_input layer_3_input_list[] = {
                { .src_id = 3, .output_idx = 0, },
        };
        static hexagon_nn_output layer_3_output_list[] = {
                { .rank = 4, .max_sizes = {10,10,10,10}, .elementsize = sizeof(uint8_t), },
        };

        RUN_HEXNN_API(result, hexagon_nn_append_udo_node(nnid, 4, udoPackageName1, udoOpType1, NULL, 0, layer_3_input_list, 1, layer_3_output_list, 1, &udoRes));
        CHECK_UDO_RES(udoRes, append_udo_node_2);


        float q_min_in [] = {10};
        float q_max_in [] = {1000};
        float q_min_out [] = {50};
        float q_max_out [] = {5000};
	RUN_HEXNN_API(result, hexagon_nn_append_const_node(nnid, 5, 1, 1, 1, 1, (uint8_t*)q_min_in, sizeof(float)));
	RUN_HEXNN_API(result, hexagon_nn_append_const_node(nnid, 6, 1, 1, 1, 1, (uint8_t*)q_max_in, sizeof(float)));
	RUN_HEXNN_API(result, hexagon_nn_append_const_node(nnid, 7, 1, 1, 1, 1, (uint8_t*)q_min_out, sizeof(float)));
	RUN_HEXNN_API(result, hexagon_nn_append_const_node(nnid, 8, 1, 1, 1, 1, (uint8_t*)q_max_out, sizeof(float)));

        static hexagon_nn_input layer_4_input_list[] = {
                { .src_id = 4, .output_idx = 0, },
                { .src_id = 5, .output_idx = 0, },
                { .src_id = 6, .output_idx = 0, },
                { .src_id = 7, .output_idx = 0, },
                { .src_id = 8, .output_idx = 0, },
        };
        static hexagon_nn_output layer_4_output_list[] = {
                { .rank = 4, .max_sizes = {10,10,10,10}, .elementsize = sizeof(uint8_t), },
                { .rank = 4, .max_sizes = {1,1,1,1}, .elementsize = sizeof(float), },
                { .rank = 4, .max_sizes = {1,1,1,1}, .elementsize = sizeof(float), },
        };

        // op_udo_plus_stat expects 1 TF8 input and 1 TF8 output. min and max are provided as hexnn tensors.
        // expected hexnn inputs and outputs configuration as follows
        // 5 inputs (inData, inMin, inMax, outMin, outMax) and 3 outputs (outData, outMin, outMax)
	RUN_HEXNN_API(result, hexagon_nn_append_udo_node(nnid, 9, udoPackageName1, udoOpType2, flattenedStatic1, flattenedStaticSize1, layer_4_input_list, 5, layer_4_output_list, 3, &udoRes));
        CHECK_UDO_RES(udoRes, append_udo_node_3);


	RUN_HEXNN_API(result, hexagon_nn_append_const_node(nnid, 10, 1, 2, 3, 1, r ,6));

        static hexagon_nn_input layer_out_input_list[] = {
                { .src_id = 9, .output_idx = 0, },
                { .src_id = 10, .output_idx = 0, },
        };

        RUN_HEXNN_API(result, hexagon_nn_append_node(
            nnid,           
            11,             
            OP_Close_quint8,     
            NN_PAD_NA,         
            layer_out_input_list,   
            2,                  
            NULL, 
            0                 
            ));

        static hexagon_nn_input layer_out_min_input_list[] = {
                { .src_id = 9, .output_idx = 1, },
                { .src_id = 7, .output_idx = 0, },
        };

        RUN_HEXNN_API(result, hexagon_nn_append_node(
            nnid,
            12,
            OP_Close_f,
            NN_PAD_NA,
            layer_out_min_input_list,
            2,
            NULL,
            0
            ));

        static hexagon_nn_input layer_out_max_input_list[] = {
                { .src_id = 9, .output_idx = 2, },
                { .src_id = 8, .output_idx = 0, },
        };

        RUN_HEXNN_API(result, hexagon_nn_append_node(
            nnid,
            13,
            OP_Close_f,
            NN_PAD_NA,
            layer_out_max_input_list,
            2,
            NULL,
            0
            ));


	RUN_HEXNN_API(result, hexagon_nn_prepare(nnid));
        printf("EXECUTE NEW RESULT IS    %d\n", hexagon_nn_execute_new(nnid, NULL, 0, NULL, 0));

        // execute_with_info API supports UDO
        struct hexagon_nn_execute_info e_info;
        struct hexagon_nn_execute_complete_info c_info;
        e_info.extraInfo = (unsigned char*)&c_info;
        e_info.extraInfoLen = sizeof(struct hexagon_nn_execute_complete_info);
	RUN_HEXNN_API(result, hexagon_nn_execute_with_info(nnid, NULL, 0, NULL, 0, &e_info));
        printf("EXECUTE WITH INFO RESULT IS      %d\n", e_info.result);
        if (e_info.result == NN_EXECUTE_UDO_ERROR) {
                printf("EXECUTE WITH INFO FAILED:   NODE ID  %d,   UDO ERR TYPE  %d,   SNPE UDO ERR CODE   %d\n", c_info.udo_err_info.exe_failure_node_id, c_info.udo_err_info.exe_failure_udo_err_type, c_info.udo_err_info.exe_failure_snpe_udo_err_code);
        } else if (e_info.result == NN_EXECUTE_BUFFER_SIZE_ERROR || e_info.result == NN_EXECUTE_ERROR) {
                printf("EXECUTE WITH INFO FAILED:   NODE ID  %d,   NODE OP TYPE  %d\n", c_info.exe_err_info.exe_failure_node_id, c_info.exe_err_info.exe_failure_node_op_type);
        }

exit:
	if(nnid != 0)  RUN_HEXNN_API_EXIT(result, hexagon_nn_teardown(nnid));
        // option 1: free udo libs individually
        RUN_HEXNN_API_EXIT(result, hexagon_nn_free_udo_individual_lib(udoPackageName1, &udoRes));
        if(udoRes!=UDO_SUCCESS){
                errorCodeFail(__FILE__, __LINE__, "free_udo_individual_lib", udoRes);
        }
        // option 2: free udo libs all together
	RUN_HEXNN_API_EXIT(result, hexagon_nn_free_udo_libs(&udoRes));
        if(udoRes!=UDO_SUCCESS){
                errorCodeFail(__FILE__, __LINE__, "free_udo_libs", udoRes);
        }
        SnpeUdo_freeFlattenedStaticParams(&flattenedStatic1);
	return result;
}
