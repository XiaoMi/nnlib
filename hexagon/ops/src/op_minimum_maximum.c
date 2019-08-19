
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

#define SHIFT 12

BROADCAST_STRIDE_11_FUNC( maximum_stride_11, uint8_t, uint8_t, max_u32)
BROADCAST_STRIDE_10_FUNC( maximum_stride_10, uint8_t, uint8_t, max_u32)
BROADCAST_REV_STRIDE_01_FUNC( maximum_rev_stride_01, uint8_t, uint8_t, max_u32)

BROADCAST_STRIDE_11_FUNC( minimum_stride_11, uint8_t, uint8_t, min_u32)
BROADCAST_STRIDE_10_FUNC( minimum_stride_10, uint8_t, uint8_t, min_u32)
BROADCAST_REV_STRIDE_01_FUNC( minimum_rev_stride_01, uint8_t, uint8_t, min_u32)

static const struct elementwise_funcs maximum_funcs = {
    .op_stride_11 = maximum_stride_11,
    .op_stride_10 = maximum_stride_10,
    .op_rev_stride_01 = maximum_rev_stride_01,
    .in_elbytes = 1,
    .out_elbytes = 1,
    .out_typecode = NN_TYPE_QUINT8
};

static const struct elementwise_funcs minimum_funcs = {
    .op_stride_11 = minimum_stride_11,
    .op_stride_10 = minimum_stride_10,
    .op_rev_stride_01 = minimum_rev_stride_01,
    .in_elbytes = 1,
    .out_elbytes = 1,
    .out_typecode = NN_TYPE_QUINT8
};

typedef struct minmax_info
{
    int32_t a_tensor_size;
    int32_t b_tensor_size;
    uint8_t *a_tensor_intermed_b_buffer;
    uint8_t *b_tensor_intermed_b_buffer;
    uint8_t* a_buffer_ptr; //points to the data buffer that is used to do the operation
    uint8_t* b_buffer_ptr;
    float a_min_float;
    float a_max_float;
    float b_min_float;
    float b_max_float;
    float out_min_float;
    float out_max_float;
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

static inline void qminmax_hvx(
    uint8_t *a,
    uint8_t *b,
    uint8_t *out,
    int32_t elem,
    int32_t a_const_value,
    int32_t b_const_value,
    int op)
{
    HVX_Vector *ptr_a = (HVX_Vector *)a;
    HVX_Vector *ptr_b = (HVX_Vector *)b;

    int loopcount = elem / ALIGN_SIZE;
    int leftovers = elem % ALIGN_SIZE;

    // splat all constant values to vectors
    a_const_value = (a_const_value << 8) | a_const_value;
    a_const_value = Q6_R_combine_RlRl(a_const_value, a_const_value);
    HVX_Vector vaconst_val = Q6_V_vsplat_R(a_const_value);
    b_const_value = (b_const_value << 8) | b_const_value;
    b_const_value = Q6_R_combine_RlRl(b_const_value, b_const_value);
    HVX_Vector vbconst_val = q6op_Vh_vsplat_R(b_const_value);

    // loop through each 128 bytes
    int i = 0;
    for (i = 0; i < loopcount; i++)
    {
        HVX_Vector ind_a = (a_const_value != 0) ? vaconst_val : *ptr_a++;
        HVX_Vector ind_b = (b_const_value != 0) ? vbconst_val : *ptr_b++;



        HVX_Vector res = op == MINIMUM ? Q6_Vub_vmin_VubVub(ind_a, ind_b) : Q6_Vub_vmax_VubVub(ind_a, ind_b);
        *((HVX_Vector *)(&out[i * sizeof(HVX_Vector)])) = res;
    }
    if (leftovers)
    {
        HVX_Vector ind_a = (a_const_value != 0) ? vaconst_val : *ptr_a++;
        HVX_Vector ind_b = (b_const_value != 0) ? vbconst_val : *ptr_b++;

        HVX_Vector res = op == MINIMUM ? Q6_Vub_vmin_VubVub(ind_a, ind_b) : Q6_Vub_vmax_VubVub(ind_a, ind_b);
        q6op_vstu_variable_ARV(&out[i * sizeof(HVX_Vector)], leftovers, res);
    }
}

static void requantize_to_outrange(float in_min, float in_max, float out_min, float out_max,
        const uint8_t* in_data, uint8_t* requant_data, uint32_t in_size) {

    int loopcount = in_size / ALIGN_SIZE;
    int leftovers = in_size % ALIGN_SIZE;
    float in_step = flt_div_255(in_max - in_min);
    int in_offset = get_qu8_level_size_zero(in_min, in_max, &in_step);
    float out_step = flt_div_255(out_max - out_min);
    int out_offset =get_qu8_level_size_zero(out_min, out_max, &out_step);

    in_offset = (in_offset << 8) | in_offset;
    in_offset = Q6_R_combine_RlRl(in_offset, in_offset);
    int mult = round(((float)(1 << SHIFT)) * (in_step / out_step));

    mult = Q6_R_combine_RlRl(mult, mult);

    HVX_Vector v_in_offset = Q6_V_vsplat_R(in_offset);
    HVX_Vector v_mult = Q6_V_vsplat_R(mult);
    HVX_Vector v_out_offset = q6op_Vh_vsplat_R(out_offset);
    l2fetch(in_data, 128, 128, (in_size + 127) / 128u);
    HVX_Vector *ptr = (HVX_Vector *)in_data;

    int i = 0;
    for (i = 0; i < loopcount; i++) {
        HVX_Vector v_in = *ptr;
        ptr++;

        HVX_Vector v_requant = jointly_quantize(v_in, v_in_offset, v_mult, v_out_offset, SHIFT);
        *((HVX_Vector *)&requant_data[i*sizeof(HVX_Vector)]) = v_requant;
    }
    if (leftovers) {
        HVX_Vector v_in = *ptr;
        HVX_Vector v_requant = jointly_quantize(v_in, v_in_offset, v_mult, v_out_offset, SHIFT);
        q6op_vstu_variable_ARV(&requant_data[i * sizeof(HVX_Vector)], leftovers, v_requant);
    }
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
    uint8_t *a_requant_data = info->a_tensor_intermed_b_buffer;
    uint8_t *b_requant_data = info->b_tensor_intermed_b_buffer;

    int elements, a_const_value, b_const_value;

    struct hvx_info opt_info;
    uint8_t *a_data_pad;
    uint8_t *b_data_pad;

    if(info->a_min_float != info->out_min_float || info->a_max_float != info->out_max_float) {

        requantize_to_outrange(info->a_min_float, info->a_max_float, info->out_min_float, info->out_max_float,
                               a_data, a_requant_data, info->a_tensor_size);
        info->a_buffer_ptr = a_requant_data;
    }
    else {
        info->a_buffer_ptr = (uint8_t*)a_data;
    }

    if(info->b_min_float != info->out_min_float || info->b_max_float != info->out_max_float) {

        requantize_to_outrange(info->b_min_float, info->b_max_float, info->out_min_float, info->out_max_float,
                               b_data, b_requant_data, info->b_tensor_size);
        info->b_buffer_ptr = b_requant_data;
    }
    else {
        info->b_buffer_ptr = (uint8_t*)b_data;
    }

    // Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data
    td->opt_flag = check_prepare_hvx_opt(nn, a_tensor, b_tensor, out_tensor, info->a_buffer_ptr, info->b_buffer_ptr, &opt_info);
    a_data_pad = opt_info.a_data_pad;
    b_data_pad = opt_info.b_data_pad;
    elements = opt_info.elements;
    a_const_value = opt_info.a_const_value;
    b_const_value = opt_info.b_const_value;

    if (td->opt_flag == 1)
    {
        qminmax_hvx(a_data_pad, b_data_pad, out_data, elements, a_const_value, b_const_value, op);
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

    float out_min = 0.0f;
    float out_max = 0.0f;
    if (self->n_inputs == 8) {
        out_min = tensor_get_float(self->inputs[6],0);
        out_max = tensor_get_float(self->inputs[7],0);
    }
    else if (self->n_inputs == 7) {
        out_min = tensor_get_float(self->inputs[6],0);
        out_max = fmaxf(a_max_float, b_max_float);
    }
    else {
        out_min = fminf(a_min_float, b_min_float);
        out_max = fmaxf(a_max_float, b_max_float);
    }

    if(out_min >= out_max) return errlog(nn, "minimum/maximum output min >= output max");

    uint32_t a_tensor_size = a_tensor->data_size;
    uint32_t b_tensor_size = b_tensor->data_size;

    uint32_t intermed_buffer_size = max_u32(a_tensor_size, b_tensor_size) + 128;
    uint8_t* a_tensor_intermed_b_buffer = nn_memalign(128, intermed_buffer_size);   //use memalign here since broadcast uses nn_scratch
    uint8_t* b_tensor_intermed_b_buffer = nn_memalign(128, intermed_buffer_size);

    struct minmax_info info;
    info.a_tensor_intermed_b_buffer = a_tensor_intermed_b_buffer;
    info.b_tensor_intermed_b_buffer = b_tensor_intermed_b_buffer;
    info.a_tensor_size = a_tensor_size;
    info.b_tensor_size = b_tensor_size;
    info.a_min_float = a_min_float;
    info.a_max_float = a_max_float;
    info.b_min_float = b_min_float;
    info.b_max_float = b_max_float;
    info.out_min_float = out_min;
    info.out_max_float = out_max;

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
            retval = nn_elementwise_with_broadcast(self, nn, &minimum_funcs, info.a_buffer_ptr, info.b_buffer_ptr, NULL);
        }
        else
        {
            retval = nn_elementwise_with_broadcast(self, nn, &maximum_funcs, info.a_buffer_ptr, info.b_buffer_ptr, NULL);
        }
    }

    if (a_tensor_intermed_b_buffer)
        nn_free(a_tensor_intermed_b_buffer);
    if (b_tensor_intermed_b_buffer)
        nn_free(b_tensor_intermed_b_buffer);
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

struct nn_node_ops nn_ops_for_QuantizedMinimum_8 = {
    .execute = minimum_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT_RANGE(6, 8),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedMaximum_8 = {
    .execute = maximum_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT_RANGE(6, 8),
    .n_outputs = NN_IOCOUNT(3),
};
