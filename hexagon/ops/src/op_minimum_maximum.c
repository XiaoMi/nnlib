
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
 * OTHERWISE) ARISING IN ANY WAY vout_val OF THE USE OF THIS SOFTWARE, EVEN
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
 * This contains the elementwise minimum and maximum ops
 */
#if defined(__hexagon__)
typedef long HVX_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN *)(A))
#endif

typedef struct minmax_info
{
    int a_offset;
    int b_offset;
    int a_mult;
    int b_mult;
    int shift;
    int qzero;
} minmax_info;

struct minmax_tdata
{
    struct nn_node *self;
    const struct tensor *a_tensor;
    const struct tensor *b_tensor;
    struct tensor *out_tensor;
    int opt_flag;
    struct minmax_info *info;
    int op;
    nn_sem_t donesem;
};

enum MinMaxOp
{
    MINIMUM,
    MAXIMUM
};

/*
    We have two independently quantized arrays. We need to compute
    joint_a = (((a - a_offset) * a_mult) >> shift) + qzero
    joint_b = (((b - b_offset) * b_mult) >> shift) + qzero
    to yield both values quantized in the same range so we can compare values
    max(joint_a, joint_b)
*/
static inline HVX_Vector jointly_quantize(HVX_Vector ind_a, HVX_Vector va_offset, HVX_Vector va_mult, HVX_Vector vqzero, int shift)
{
    // (a - a_offset), (b - offset) ~ ub*ub->2h
    HVX_VectorPair a_sub = Q6_Wh_vsub_VubVub(ind_a, va_offset);

    // ((a - offset) * a_mult) ~ h*h->w
    HVX_VectorPair a_mult_res_lo = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(a_sub), va_mult);
    HVX_VectorPair a_mult_res_hi = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(a_sub), va_mult);

    // (((a - offset) * a_mult)) >> shift) ~ w*w->h
    HVX_Vector shift_a_lo = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(a_mult_res_lo), Q6_V_lo_W(a_mult_res_lo), shift);
    HVX_Vector shift_a_hi = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(a_mult_res_hi), Q6_V_lo_W(a_mult_res_hi), shift);

    // shuffle high and low to convert from words to half words
    // uint8 [x,0,y,0,z,0,...], [] -> shuffle -> [a,b,c,....0,0,0...]
    HVX_VectorPair ashuff = Q6_W_vshuff_VVR(shift_a_hi, shift_a_lo, -2);

    // (((a - offset) * a_mult)) >> shift) + qzero ~ h*h->h
    HVX_Vector joint_a_even = Q6_Vh_vadd_VhVh_sat(Q6_V_lo_W(ashuff), vqzero); // (((a - a_offset) * a_mult) >> shift) + qzero
    HVX_Vector joint_a_odd = Q6_Vh_vadd_VhVh_sat(Q6_V_hi_W(ashuff), vqzero);  // (((a - a_offset) * a_mult) >> shift) + qzero

    // deal high and low to restore element order
    HVX_Vector joint_a_lo = Q6_Vb_vdeal_Vb(joint_a_even);
    HVX_Vector joint_a_hi = Q6_Vb_vdeal_Vb(joint_a_odd);

    // Each vector now has 64 elements.
    // rotate high to the last 64 bits and mux together
    // HVX_Vector rotated_a = Q6_V_vror_VR(joint_a_hi, 64);
    HVX_Vector va = Q6_V_vmux_QVV(Q6_Q_vsetq_R(64), joint_a_lo, Q6_V_vror_VR(joint_a_hi, 64));
    return va;
}

static inline void qminmax_hvx(
    uint8_t *a,
    uint8_t *b,
    uint8_t *out,
    void *vminmax_info,
    int32_t elem,
    int32_t a_const_value,
    int32_t b_const_value,
    int op)
{
    const struct minmax_info *info = vminmax_info;

    HVX_Vector *ptr_a = (HVX_Vector *)a;
    HVX_Vector *ptr_b = (HVX_Vector *)b;

    int a_offset = info->a_offset;
    int b_offset = info->b_offset;
    int a_mult = info->a_mult;
    int b_mult = info->b_mult;
    int shift = info->shift;
    int qzero = info->qzero;
    int loopcount = elem >> 7; /* 7 - log2(ALIGN_SIZE) */
    int leftovers = elem % ALIGN_SIZE;

    // splat all constant values to vectors
    a_const_value = (a_const_value << 8) | a_const_value;
    a_const_value = Q6_R_combine_RlRl(a_const_value, a_const_value);
    HVX_Vector vaconst_val = Q6_V_vsplat_R(a_const_value);
    b_const_value = (b_const_value << 8) | b_const_value;
    b_const_value = Q6_R_combine_RlRl(b_const_value, b_const_value);
    HVX_Vector vbconst_val = q6op_Vh_vsplat_R(b_const_value);
    /* Assumption: Range of a_offset and b_offset to be with between 0 to 255 */
    a_offset = (a_offset << 8) | a_offset;
    b_offset = (b_offset << 8) | b_offset;
    qzero = (qzero << 8) | qzero;
    a_offset = Q6_R_combine_RlRl(a_offset, a_offset);
    b_offset = Q6_R_combine_RlRl(b_offset, b_offset);
    a_mult = Q6_R_combine_RlRl(a_mult, a_mult);
    b_mult = Q6_R_combine_RlRl(b_mult, b_mult);
    HVX_Vector va_offset = Q6_V_vsplat_R(a_offset);
    HVX_Vector vb_offset = Q6_V_vsplat_R(b_offset);
    HVX_Vector va_mult = Q6_V_vsplat_R(a_mult);
    HVX_Vector vb_mult = Q6_V_vsplat_R(b_mult);
    HVX_Vector vqzero = q6op_Vh_vsplat_R(qzero);

    // loop through each 128 bytes
    int i = 0;
    for (i = 0; i < loopcount; i++)
    {
        HVX_Vector ind_a = (a_const_value != 0) ? vaconst_val : *ptr_a++;
        HVX_Vector ind_b = (b_const_value != 0) ? vbconst_val : *ptr_b++;

        HVX_Vector va = jointly_quantize(ind_a, va_offset, va_mult, vqzero, shift);
        HVX_Vector vb = jointly_quantize(ind_b, vb_offset, vb_mult, vqzero, shift);
        HVX_Vector res = op == MINIMUM ? Q6_Vub_vmin_VubVub(va, vb) : Q6_Vub_vmax_VubVub(va, vb);
        vmemu(&out[i * 128]) = res;
    }
    if (leftovers)
    {
        HVX_Vector ind_a = (a_const_value != 0) ? vaconst_val : *ptr_a++;
        HVX_Vector ind_b = (b_const_value != 0) ? vbconst_val : *ptr_b++;

        HVX_Vector va = jointly_quantize(ind_a, va_offset, va_mult, vqzero, shift);
        HVX_Vector vb = jointly_quantize(ind_b, vb_offset, vb_mult, vqzero, shift);

        HVX_Vector res = op == MINIMUM ? Q6_Vub_vmin_VubVub(va, vb) : Q6_Vub_vmax_VubVub(va, vb);
        q6op_vstu_variable_ARV(&out[i * 128], leftovers, res);
    }
}

static inline uint8_t q8maximum_helper(uint8_t a, uint8_t b, void *vminmax_info)
{
    const struct minmax_info *info = vminmax_info;
    int a_offset = info->a_offset;
    int b_offset = info->b_offset;
    int a_mult = info->a_mult;
    int b_mult = info->b_mult;
    int shift = info->shift;
    int qzero = info->qzero;
    int aval = (((a - a_offset) * a_mult) >> shift) + qzero;
    int bval = (((b - b_offset) * b_mult) >> shift) + qzero;

    uint8_t ret = max_u32(aval, bval);

    return ret;
}

static inline uint8_t q8minimum_helper(uint8_t a, uint8_t b, void *vminmax_info)
{
    const struct minmax_info *info = vminmax_info;
    int a_offset = info->a_offset;
    int b_offset = info->b_offset;
    int a_mult = info->a_mult;
    int b_mult = info->b_mult;
    int shift = info->shift;
    int qzero = info->qzero;
    int aval = (((a - a_offset) * a_mult) >> shift) + qzero;
    int bval = (((b - b_offset) * b_mult) >> shift) + qzero;

    uint8_t ret = min_u32(aval, bval);

    return ret;
}

static void qminmax_thread_process(struct nn_graph *nn, void *vtdata)
{

    struct minmax_tdata *td = vtdata;
    int op = td->op;
    const struct tensor *a_tensor = td->a_tensor;
    const struct tensor *b_tensor = td->b_tensor;
    struct tensor *out_tensor = td->out_tensor;
    const uint8_t *a_data = a_tensor->data;
    const uint8_t *b_data = b_tensor->data;
    uint8_t *out_data = out_tensor->data;
    struct minmax_info *info = td->info;
    int elements, a_const_value, b_const_value;
    struct hvx_info opt_info;
    uint8_t *a_data_pad;
    uint8_t *b_data_pad;

    // Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data
    td->opt_flag = check_prepare_hvx_opt(nn, a_tensor, b_tensor, out_tensor, a_data, b_data, &opt_info);
    a_data_pad = opt_info.a_data_pad;
    b_data_pad = opt_info.b_data_pad;
    elements = opt_info.elements;
    a_const_value = opt_info.a_const_value;
    b_const_value = opt_info.b_const_value;
    if (td->opt_flag == 1)
    {
        qminmax_hvx(a_data_pad, b_data_pad, out_data, info, elements, a_const_value, b_const_value, op);
    }
    nn_sem_post(&td->donesem);
}

static int minmax_execute(struct nn_node *self, struct nn_graph *nn, int op)
{
    int retval;
    const struct tensor *a_tensor = self->inputs[0];
    const struct tensor *b_tensor = self->inputs[1];
    const struct tensor *a_min_tensor = self->inputs[2];
    const struct tensor *a_max_tensor = self->inputs[3];
    const struct tensor *b_min_tensor = self->inputs[4];
    const struct tensor *b_max_tensor = self->inputs[5];
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
    struct minmax_info info;
    info.a_offset = quantize_uint8(0.0f, a_min_float, a_max_float);
    info.b_offset = quantize_uint8(0.0f, b_min_float, b_max_float);
    info.shift = 12;
    info.a_mult = ((float)(1 << info.shift)) * (a_level_size / out_level_size) + 0.5f;
    info.b_mult = ((float)(1 << info.shift)) * (b_level_size / out_level_size) + 0.5f;
    info.qzero = -out_min * (255 / (out_max - out_min)) + 0.5f;
    struct minmax_tdata td = {
        .self = self,
        .a_tensor = a_tensor,
        .b_tensor = b_tensor,
        .out_tensor = out_tensor,
        .opt_flag = 0,
        .info = &info,
        .op = op,
    };

    // set output sizes
    tensor_set_single_float(out_min_tensor, out_min);
    tensor_set_single_float(out_max_tensor, out_max);
    if (tensor_out_prepare_normal(out_tensor, fmaxf(a_tensor->shape.batches, b_tensor->shape.batches), fmaxf(a_tensor->shape.height, b_tensor->shape.height), fmaxf(a_tensor->shape.width, b_tensor->shape.width), fmaxf(a_tensor->shape.depth, b_tensor->shape.depth), NN_TYPE_QUINT8) != 0)
    {
        return errlog(nn, "failed to prepare output");
    }

    nn_sem_init(&td.donesem, 0);
    nn_os_work_for_vector(nn, qminmax_thread_process, &td);
    nn_sem_wait(&td.donesem);
    retval = 0;
    if (td.opt_flag == 2)
    {
        return -1;
    }

    if (td.opt_flag == 1)
    {
        retval = 0;
    }
    else
    {
        if (op == MINIMUM)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8maximum_helper, &info);
        }
        else
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8minimum_helper, &info);
        }
    }

    return retval;
}

static int maximum_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum MinMaxOp op = MAXIMUM;
    return minmax_execute(self, nn, op);
}

static int minimum_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum MinMaxOp op = MINIMUM;
    return minmax_execute(self, nn, op);
}

static int minmax_q8_check(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn, 2, "Minimum/Maximum node %p", self);
    if (self->n_inputs != 6)
        return errlog(nn, "wrong # inputs");
    if (self->n_outputs != 3)
        return errlog(nn, "wrong # outputs");
    logmsg(nn, 2, "Minimum/Maximum %p check OK", self);
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedMinimum_8 = {
    .execute = minimum_q8_execute,
    .check = minmax_q8_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedMaximum_8 = {
    .execute = maximum_q8_execute,
    .check = minmax_q8_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};
