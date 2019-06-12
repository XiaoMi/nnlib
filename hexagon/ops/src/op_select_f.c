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


#include <nn_graph.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

static int select_execute(struct nn_node *self, struct nn_graph *nn)
{
    // Unpack the input and output tensors of this node.
    const struct tensor *c_tensor = self->inputs[0];
    const struct tensor *x_tensor = self->inputs[1];
    const struct tensor *y_tensor = self->inputs[2];
    struct tensor *out_tensor = self->outputs[0];

    // Set the size of the output tensor.
    if (tensor_out_prepare_normal_fromshape(out_tensor, &y_tensor->shape, NN_TYPE_FLOAT) !=0) return errlog(nn,"out too small");

    int i;
    char *cond = (char *) c_tensor->data;
    float *in1 = (float *) x_tensor->data;
    float *in2 = (float *) y_tensor->data;
    float *out = (float *) out_tensor->data;
    for (i=0; i<out_tensor->shape.batches *
                out_tensor->shape.height *
                out_tensor->shape.width *
                out_tensor->shape.depth; i++) {
        *(out+i) = *(cond+i) ? *(in1+i) : *(in2+i);
    }

    return 0;
}


struct nn_node_ops nn_ops_for_Select_f = {
        .execute = select_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(1),
};
