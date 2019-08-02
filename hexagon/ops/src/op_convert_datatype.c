/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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
#include <nn_graph.h>
#include <string.h>
#include <math.h>
//
// This contains conversions to 'float32 flat tensors'.
//
////////////////////////////////////////////////////////////////////////////
//  convert_int32_f:
//       input 0:  i32 flat tensor, any shape
//       output 0: float tensor, same shape as input 0.
//  Note:
//       it does force cast directly, which will lose LSB accuracy.
//       not suggested for large int range input (> 2^23)
//
static int convert_int32_f_execute(struct nn_node *self, struct nn_graph *nn)
{
    struct tensor const * in_tensor = self->inputs[0];
    struct tensor * out_tensor = self->outputs[0];

    if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT) != 0){
        return errlog(nn,"output too small");
    }

    const int *in_data = in_tensor->data;
    float *out_data = out_tensor->data;
    int count = tensor_element_count(in_tensor);

    for( int i = 0; i < count; i++ ){
        out_data[i] = (float)(in_data[i]);
    }
    return 0;
}



struct nn_node_ops nn_ops_for_Convert_int32_f = {
    .execute = convert_int32_f_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(1),
    .n_outputs = NN_IOCOUNT(1),
};
