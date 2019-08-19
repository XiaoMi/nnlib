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
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <quantize.h>
#include <nn_broadcast.h>
#include <nn_reduction.h>
#include "hvx_inlines.h"
#include "quantize.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the elementwise select op 
 */

typedef struct select_info {
    int a_offset;
    int b_offset;
    int a_mult;
    int b_mult;
    int shift;
    int qzero;
} select_info;

struct select_tdata {
    struct nn_node *self;
    const struct tensor *c_tensor;
    const struct tensor *a_tensor;
    const struct tensor *b_tensor;
    struct tensor *out_tensor;
    struct select_info *info;
    nn_sem_t donesem;
};

static inline void qselect_hvx(
        const uint8_t *c,
        const uint8_t *a,
        const uint8_t *b,
        uint8_t *out,
        void *vselect_info,
        int32_t elem) {
    const struct select_info *info = vselect_info;

    HVX_VectorPred *ptr_c = (HVX_VectorPred *) c;
    HVX_Vector *ptr_a = (HVX_Vector *) a;
    HVX_Vector *ptr_b = (HVX_Vector *) b;

    int a_offset = info->a_offset;
    int b_offset = info->b_offset;
    int a_mult = info->a_mult;
    int b_mult = info->b_mult;
    int shift = info->shift;
    int qzero = info->qzero;
    int loopcount = elem / ALIGN_SIZE;
    int leftovers = elem % ALIGN_SIZE;

    /* Assumption: Range of a_offset and b_offset to be with between 0 to 255 */
    a_offset = (a_offset << 8) | a_offset;
    b_offset = (b_offset << 8) | b_offset;
    a_offset = Q6_R_combine_RlRl(a_offset, a_offset);
    b_offset = Q6_R_combine_RlRl(b_offset, b_offset);
    a_mult = Q6_R_combine_RlRl(a_mult, a_mult);
    b_mult = Q6_R_combine_RlRl(b_mult, b_mult);

    HVX_Vector va_offset = Q6_V_vsplat_R(a_offset);
    HVX_Vector vb_offset = Q6_V_vsplat_R(b_offset);
    HVX_Vector va_mult = Q6_V_vsplat_R(a_mult);
    HVX_Vector vb_mult = Q6_V_vsplat_R(b_mult);
    HVX_Vector vqzero = q6op_Vh_vsplat_R(qzero);

    // loop through each ${sizeof(HVX_Vector)} bytes
    l2fetch(a, 128, 128, (elem + 127) / 128u);
    l2fetch(b, 128, 128, (elem + 127) / 128u);

    int i;
    for (i = 0; i < loopcount; i++) {
        HVX_VectorPred qc = *ptr_c++;
        HVX_Vector ind_a = *ptr_a++;
        HVX_Vector ind_b = *ptr_b++;

        HVX_Vector va = jointly_quantize(ind_a, va_offset, va_mult, vqzero, shift);
        HVX_Vector vb = jointly_quantize(ind_b, vb_offset, vb_mult, vqzero, shift);

        HVX_Vector res = Q6_V_vmux_QVV(qc, va, vb);
        *((HVX_Vector *)(&out[i * sizeof(HVX_Vector)])) = res;
    }
    if (leftovers) {
        HVX_VectorPred qc = *ptr_c++;
        HVX_Vector ind_a = *ptr_a++;
        HVX_Vector ind_b = *ptr_b++;

        HVX_Vector va = jointly_quantize(ind_a, va_offset, va_mult, vqzero, shift);
        HVX_Vector vb = jointly_quantize(ind_b, vb_offset, vb_mult, vqzero, shift);

        HVX_Vector res = Q6_V_vmux_QVV(qc, va, vb);
        q6op_vstu_variable_ARV(&out[i * sizeof(HVX_Vector)], leftovers, res);
    }
}

static void qselect_thread_process(struct nn_graph *nn, void *vtdata) {

    struct select_tdata *td = vtdata;
    const struct tensor *c_tensor = td->c_tensor;
    const struct tensor *a_tensor = td->a_tensor;
    const struct tensor *b_tensor = td->b_tensor;
    struct tensor *out_tensor = td->out_tensor;
    const uint8_t *c_data = c_tensor->data;
    const uint8_t *a_data = a_tensor->data;
    const uint8_t *b_data = b_tensor->data;
    uint8_t *out_data = out_tensor->data;
    struct select_info *info = td->info;
    int elements = out_tensor->shape.batches *
                   out_tensor->shape.height *
                   out_tensor->shape.width *
                   out_tensor->shape.depth;

    qselect_hvx(c_data, a_data, b_data, out_data, info, elements);
    nn_sem_post(&td->donesem);
}

static int select_execute(struct nn_node *self, struct nn_graph *nn) {
    // Unpack the input and output tensors of this node.
    const struct tensor *c_tensor = self->inputs[0];
    const struct tensor *a_tensor = self->inputs[1];
    const struct tensor *a_min_tensor = self->inputs[2];
    const struct tensor *a_max_tensor = self->inputs[3];
    const struct tensor *b_tensor = self->inputs[4];
    const struct tensor *b_min_tensor = self->inputs[5];
    const struct tensor *b_max_tensor = self->inputs[6];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    // calculate quantization parameters and output parameters
    float a_min_float = tensor_get_float(a_min_tensor, 0);
    float a_max_float = tensor_get_float(a_max_tensor, 0);
    float b_min_float = tensor_get_float(b_min_tensor, 0);
    float b_max_float = tensor_get_float(b_max_tensor, 0);
    float a_level_size = (a_max_float - a_min_float) / 255;
    float b_level_size = (b_max_float - b_min_float) / 255;
    float out_min = fminf(a_min_float, b_min_float);
    float out_max = fmaxf(a_max_float, b_max_float);

    // sets the min and max so 0 is exactly quantized to the exact int 0
    adjust_minmax_for_zero(&out_min, &out_max);

    float out_level_size = (out_max - out_min) / 255;
    struct select_info info;
    info.a_offset = quantize_uint8(0.0f, a_min_float, a_max_float);
    info.b_offset = quantize_uint8(0.0f, b_min_float, b_max_float);
    info.shift = 12;
    info.a_mult = ((float) (1 << info.shift)) * (a_level_size / out_level_size);
    info.b_mult = ((float) (1 << info.shift)) * (b_level_size / out_level_size);
    info.qzero = quantize_uint8(0.0f, out_min, out_max);
    struct select_tdata td = {
            .self = self,
            .c_tensor = c_tensor,
            .a_tensor = a_tensor,
            .b_tensor = b_tensor,
            .out_tensor = out_tensor,
            .info = &info,
    };

    // set output sizes
    tensor_set_single_float(out_min_tensor, out_min);
    tensor_set_single_float(out_max_tensor, out_max);

    // Set the size of the output tensor.
    if (tensor_out_prepare_normal_fromshape(out_tensor, &b_tensor->shape, NN_TYPE_QINT8) != 0) {
        return errlog(nn, "failed to prepare output");
    }

    nn_sem_init(&td.donesem, 0);
    nn_os_work_for_vector(nn, qselect_thread_process, &td);
    nn_sem_wait(&td.donesem);

    return 0;

}


struct nn_node_ops nn_ops_for_Select_8 = {
        .execute = select_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(7),
        .n_outputs = NN_IOCOUNT(3),
};
