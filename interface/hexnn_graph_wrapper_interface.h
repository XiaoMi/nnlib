
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
#ifndef HEXAGON_NN_NN_GRAPH_WRAPPER_H
#define HEXAGON_NN_NN_GRAPH_WRAPPER_H

#include "hexagon_nn.h"

typedef enum hexnn_batch_op_type {
    // WARNING - DANGER!  Do not alter existing enum-IDs.
    //                    These IDs should be stable across builds
    //                    So you should:
    //                    Add new items to BOTTOM
    HEXNN_BATCH_OP_APPEND_NOPE,
    HEXNN_BATCH_OP_APPEND_NODE,
    HEXNN_BATCH_OP_APPEND_CONST_NODE,
    HEXNN_BATCH_OP_APPEND_EMPTY_CONST_NODE,
    HEXNN_BATCH_OP_POPULATE_CONST_NODE,

    //  /\.
    //   \___ Add NEW entries HERE, at BOTTOM
    HEXNN_BATCH_OP_MAX = 0x7fffffff

}hexnn_batch_op_type;

typedef struct _append_node_params {
    unsigned int node_id;
    unsigned int operation;
    hexagon_nn_padding_type padding;
    unsigned int inputsLen;        // size of hexagon_nn_input in num of bytes
    unsigned int outputsLen;       // size of hexagon_nn_output in num of bytes
    unsigned int structPaddingFor8byteAlignment;
}append_node_params;

typedef struct _append_const_node_params {
    unsigned int node_id;
    unsigned int batches;
    unsigned int height;
    unsigned int width;
    unsigned int depth;
    unsigned int dataLen;
}append_const_node_params;

typedef struct _append_empty_const_node_params {
    unsigned int node_id;
    unsigned int batches;
    unsigned int height;
    unsigned int width;
    unsigned int depth;
    unsigned int size;
}append_empty_const_node_params;

typedef struct _populate_const_node_params {
    unsigned int node_id;
    unsigned int dataLen;          // size of data in num of bytes
    unsigned int target_offset;
    unsigned int structPaddingFor8byteAlignment;
}populate_const_node_params;

#pragma pack(1)
typedef struct _flat_batch_ops_params
{
    hexnn_batch_op_type op;
    unsigned int structPaddingFor8byteAlignment;
    union {
        append_node_params              an_params;
        append_const_node_params        acn_params;
        append_empty_const_node_params  aecn_params;
        populate_const_node_params      pcn_params;
    } U;
    unsigned char   c[1];   // this is 1st varialbe length. the buffer size is aligned to 8 bytes
                            // 2nd buffer append after (i.e. for append_mode command)
}flat_batch_ops_params;
#pragma pack()

typedef struct _batch_ops_params
{
    hexnn_batch_op_type op;
    unsigned int node_id;
    unsigned int operation;
    hexagon_nn_padding_type padding;
    hexagon_nn_input* inputs;
    unsigned int inputsLen;
    hexagon_nn_output* outputs;
    unsigned int outputsLen;
    unsigned int batches;
    unsigned int height;
    unsigned int width;
    unsigned int depth;
    unsigned int size;
    unsigned char* data;
    unsigned int dataLen;
    unsigned int target_offset;
}batch_ops_params;


#endif //HEXAGON_NN_NN_GRAPH_WRAPPER_H
