
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
 * OTHERWISE) ARISING IN ANY WAY vout_val OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <nn_graph.h>
#include <nn_broadcast.h>
#include "hvx_inlines.h"
#include "quantize.h"

#if defined(__hexagon__)
#include "hexagon_types.h"

#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the elementwise comparison ops
 */

typedef HVX_VectorPred (*FUNC_PTR)(HVX_Vector, HVX_Vector);
typedef void (*EXEC_FUNC)(struct nn_graph *, void * );

typedef struct compare_info
{
    int a_const_value;
    int b_const_value;
    uint32_t elem;
    uint32_t b_tensor_size;
    float a_min_float;
    float a_max_float;
    float b_min_float;
    float b_max_float;
    float *b_tensor_intermed_buffer;
} compare_info;

struct compare_tdata
{
    struct nn_node *self;
    const uint8_t * a;
    const uint8_t * b;
    const struct tensor * a_tensor;
    const struct tensor * b_tensor;
    struct tensor *out_tensor;
    int opt_flag;
    struct compare_info *info;
    EXEC_FUNC compare_fp;
    nn_sem_t donesem;
};

enum CompareOp
{
    EQUAL,
    NOT_EQUAL,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL
};

static inline void compare_hvx_template(struct nn_graph *nn, void *vtd, FUNC_PTR reduction_func)
{
    struct compare_tdata *td = (struct compare_tdata *)vtd;
    struct compare_info *info = td->info;
    HVX_Vector *ptr_a = (HVX_Vector *)(td->a);
    HVX_Vector *ptr_b = (HVX_Vector *)(td->b);
    int a_const_value = info->a_const_value;
    int b_const_value = info->b_const_value;
    uint8_t *out = td->out_tensor->data;
    int elem = info->elem;
    int loopcount = elem / sizeof(HVX_Vector);
    int leftovers = elem % sizeof(HVX_Vector);

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
        HVX_Vector * outv = (HVX_Vector *)&out[i * sizeof(HVX_Vector)];
        HVX_Vector ind_a = (a_const_value != 0) ? vaconst_val : *ptr_a++;
        HVX_Vector ind_b = (b_const_value != 0) ? vbconst_val : *ptr_b++;
        *outv = reduction_func(ind_a, ind_b);
    }
    if (leftovers)
    {
        HVX_Vector * outv = (HVX_Vector *)&out[i * sizeof(HVX_Vector)];
        HVX_Vector ind_a = (a_const_value != 0) ? vaconst_val : *ptr_a++;
        HVX_Vector ind_b = (b_const_value != 0) ? vbconst_val : *ptr_b++;
        q6op_vstu_variable_ARV(outv, leftovers, reduction_func(ind_a, ind_b));
    }
}

static inline uint8_t q8_equal_helper(uint8_t a, uint8_t b, void *info)
{
    return (uint8_t)(a == b);
}
static inline uint8_t q8_not_equal_helper(uint8_t a, uint8_t b, void *info)
{
    return (uint8_t)(a != b);
}
static inline uint8_t q8_less_helper(uint8_t a, uint8_t b, void *info)
{
    return (uint8_t)(a < b);
}
static inline uint8_t q8_less_equal_helper(uint8_t a, uint8_t b, void *info)
{
    return (uint8_t)(a <= b);
}
static inline uint8_t q8_greater_helper(uint8_t a, uint8_t b, void *info)
{
    return (uint8_t)(a > b);
}
static inline uint8_t q8_greater_equal_helper(uint8_t a, uint8_t b, void *info)
{
    return (uint8_t)(a >= b);
}

static inline HVX_Vector  __attribute__((always_inline))
eltwise_vec_equal(HVX_Vector v1, HVX_Vector v2)
{
	return 	Q6_V_vand_QR(Q6_Q_vcmp_eq_VbVb(v1, v2), 0x01010101);
}

static inline HVX_Vector  __attribute__((always_inline))
eltwise_vec_not_equal(HVX_Vector v1, HVX_Vector v2)
{
    HVX_VectorPred pred = Q6_Q_vcmp_eq_VbVb(v1, v2);
    return Q6_V_vmux_QVV(pred, Q6_V_vsplat_R(0), Q6_V_vsplat_R(0x01010101));
}
static inline HVX_VectorPred  __attribute__((always_inline))
eltwise_vec_less(HVX_Vector v1, HVX_Vector v2)
{
     HVX_VectorPred pred = Q6_Q_vcmp_eq_VbVb(v1, v2);
     HVX_VectorPred pred2 = Q6_Q_vcmp_gtor_QVubVub(pred, v1, v2);
     return Q6_V_vmux_QVV(pred2, Q6_V_vsplat_R(0), Q6_V_vsplat_R(0x01010101));
}
static inline HVX_VectorPred  __attribute__((always_inline))
eltwise_vec_less_equal(HVX_Vector v1, HVX_Vector v2)
{
    HVX_VectorPred pred = Q6_Q_vcmp_gt_VubVub(v1,v2);
    return Q6_V_vmux_QVV(pred, Q6_V_vsplat_R(0), Q6_V_vsplat_R(0x01010101));
}
static inline HVX_Vector  __attribute__((always_inline))
eltwise_vec_greater(HVX_Vector v1, HVX_Vector v2)
{
	return Q6_V_vand_QR(Q6_Q_vcmp_gt_VubVub(v1, v2), 0x01010101);
}
static inline HVX_Vector  __attribute__((always_inline))
eltwise_vec_greater_equal(HVX_Vector v1, HVX_Vector v2)
{
     HVX_VectorPred pred = Q6_Q_vcmp_eq_VbVb(v1, v2);
     HVX_VectorPred pred2 = Q6_Q_vcmp_gtor_QVubVub(pred, v1, v2);
     return Q6_V_vmux_QVV(pred2, Q6_V_vsplat_R(0x01010101), Q6_V_vsplat_R(0));

}

static void
compare_hvx_EQUAL(struct nn_graph *nn, void *vtd)
{
	compare_hvx_template( nn, vtd,eltwise_vec_equal );
}

static void
compare_hvx_NOT_EQUAL(struct nn_graph *nn, void *vtd)
{
	compare_hvx_template( nn, vtd,eltwise_vec_not_equal );
}

static void
compare_hvx_LESS(struct nn_graph *nn, void *vtd)
{
	compare_hvx_template( nn, vtd,eltwise_vec_less );
}
static void
compare_hvx_LESS_EQUAL(struct nn_graph *nn, void *vtd)
{
	compare_hvx_template( nn, vtd,eltwise_vec_less_equal );
}
static void
compare_hvx_GREATER(struct nn_graph *nn, void *vtd)
{
	compare_hvx_template( nn, vtd,eltwise_vec_greater );
}
static void
compare_hvx_GREATER_EQUAL(struct nn_graph *nn, void *vtd)
{
	compare_hvx_template( nn, vtd,eltwise_vec_greater_equal );
}

static void qcompare_thread_process(struct nn_graph *nn, void *vtdata)
{

    struct compare_tdata *td = (struct compare_tdata *)vtdata;

    EXEC_FUNC compare_fp = td->compare_fp;
    const struct tensor *a_tensor = td->a_tensor;
    const struct tensor *b_tensor = td->b_tensor;
    struct tensor *out_tensor = td->out_tensor;
    const uint8_t *a_data = a_tensor->data;
    const uint8_t *b_data = b_tensor->data;
    int elements, a_const_value, b_const_value;
    struct hvx_info opt_info;
    struct compare_info *info = td->info;
    uint8_t *a_data_pad;
    uint8_t *b_data_pad;
    float b_step = fmaxf(0.0001f, flt_div_255(info->b_max_float - info->b_min_float));
    int b_offset = saturate_u8(roundf_i32(-info->b_min_float/b_step));

    //If a and b are using different scales, we requantize b into the range of a, so that we can avoid any rounding errors that might impact the comparison
    if(info->a_min_float != info->b_min_float || info->a_max_float != info->b_max_float)
    {
        l2fetch( b_data, 128,128, (info->b_tensor_size+127)/128u);
        hvx_do_dequantize(b_data, info->b_tensor_intermed_buffer, info->b_tensor_size, b_offset, b_step);

        struct hvx_quant_parms qparms;
        float fbuf[2];
        fbuf[0] = -info->a_min_float;
        fbuf[1] = info->a_max_float;
        if( find_scaling_for_hvx_quant(fbuf, &qparms) !=0 ){
		    errlog(nn,"inf or NaN input, to compare quantize");
        }
        quantize_floats_to_8b_asm(info->b_tensor_intermed_buffer, (uint8_t*)b_data, info->b_tensor_size, qparms.min_offset, qparms.common_exp, qparms.scaling);
    }

    // Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data
    td->opt_flag = check_prepare_hvx_opt(nn, a_tensor, b_tensor, out_tensor, a_data, b_data, &opt_info);
    a_data_pad = opt_info.a_data_pad;
    b_data_pad = opt_info.b_data_pad;
    elements = opt_info.elements;
    a_const_value = opt_info.a_const_value;
    b_const_value = opt_info.b_const_value;
    td->info->elem = elements;
    if (td->opt_flag == 1)
    {
        td->a = a_data_pad;
        td->b = b_data_pad;
        td->info->a_const_value = a_const_value;
        td->info->b_const_value = b_const_value;
        (*compare_fp)(nn, (void *)td);
    }
    nn_sem_post(&td->donesem);
}

static int compare_execute(struct nn_node *self, struct nn_graph *nn, int op)
{
    int retval;
    const struct tensor *a_tensor = self->inputs[0];
    const struct tensor *b_tensor = self->inputs[1];
    const struct tensor *a_min_tensor = self->inputs[2];
    const struct tensor *a_max_tensor = self->inputs[3];
    const struct tensor *b_min_tensor = self->inputs[4];
    const struct tensor *b_max_tensor = self->inputs[5];
    struct tensor *out_tensor = self->outputs[0];

    //Determine which function will handle the comparison depending on its type
    EXEC_FUNC compare_fp = NULL;
    switch(op)
    {
        case EQUAL:
            compare_fp = compare_hvx_EQUAL;
            break;
        case NOT_EQUAL:
            compare_fp = compare_hvx_NOT_EQUAL;
            break;
        case LESS:
            compare_fp = compare_hvx_LESS;
            break;
        case LESS_EQUAL:
            compare_fp = compare_hvx_LESS_EQUAL;
            break;
        case GREATER:
            compare_fp = compare_hvx_GREATER;
            break;
        case GREATER_EQUAL:
            compare_fp = compare_hvx_GREATER_EQUAL;
            break;
        default:
            break;
    }

    // calculate quantization parameters and output parameters
    struct compare_info info;
    info.a_min_float = tensor_get_float(a_min_tensor, 0);
    info.a_max_float = tensor_get_float(a_max_tensor, 0);
    info.b_min_float = tensor_get_float(b_min_tensor, 0);
    info.b_max_float = tensor_get_float(b_max_tensor, 0);
    info.b_tensor_size = b_tensor->shape.batches * b_tensor->shape.height * b_tensor->shape.width * b_tensor->shape.depth;
    if (nn_scratch_grow(nn, info.b_tensor_size*sizeof(float)))
    {
        errlog(nn, "Failed to grow scratch for temporary comparison buffer");
    }
    else
    {
        info.b_tensor_intermed_buffer = nn->scratch;
    }


    struct compare_tdata td = {
        .self = self,
        .a_tensor = a_tensor,
        .b_tensor = b_tensor,
        .out_tensor = out_tensor,
        .opt_flag = 0,
        .info = &info,
        .compare_fp = compare_fp,
    };
    uint32_t out_batches = max_i32(a_tensor->shape.batches, b_tensor->shape.batches);
    uint32_t out_height = max_i32(a_tensor->shape.height, b_tensor->shape.height);
    uint32_t out_width = max_i32(a_tensor->shape.width, b_tensor->shape.width);
    uint32_t out_depth = max_i32(a_tensor->shape.depth, b_tensor->shape.depth);

    if (tensor_out_prepare_normal(out_tensor, out_batches, out_height, out_width, out_depth, NN_TYPE_QUINT8) != 0)
    {
        return errlog(nn, "failed to prepare output of size %d %d %d %d", out_batches, out_height, out_width, out_depth);
    }

    nn_sem_init(&td.donesem, 0);
    nn_os_work_for_vector(nn, qcompare_thread_process, &td);
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
        if (op == EQUAL)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8_equal_helper, &info);
        }
        else if (op == NOT_EQUAL)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8_not_equal_helper, &info);
        }
        else if (op == LESS)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8_less_helper, &info);
        }
        else if (op == LESS_EQUAL)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8_less_equal_helper, &info);
        }
        else if (op == GREATER)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8_greater_helper, &info);
        }
        else if (op == GREATER_EQUAL)
        {
            retval = broadcast_elementwise_execute_quint8(self, nn, q8_greater_equal_helper, &info);
        }
        else
        {
            retval = errlog(nn, "Unsupported compare op %d", op);
        }
    }

    return retval;
}

static int equal_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum CompareOp op = EQUAL;
    return compare_execute(self, nn, op);
}
static int not_equal_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum CompareOp op = NOT_EQUAL;
    return compare_execute(self, nn, op);
}
static int less_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum CompareOp op = LESS;
    return compare_execute(self, nn, op);
}
static int less_equal_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum CompareOp op = LESS_EQUAL;
    return compare_execute(self, nn, op);
}
static int greater_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum CompareOp op = GREATER;
    return compare_execute(self, nn, op);
}
static int greater_equal_q8_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum CompareOp op = GREATER_EQUAL;
    return compare_execute(self, nn, op);
}

struct nn_node_ops nn_ops_for_QuantizedEqual_8 = {
    .execute = equal_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_QuantizedNotEqual_8 = {
    .execute = not_equal_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_QuantizedLess_8 = {
    .execute = less_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_QuantizedLessEqual_8 = {
    .execute = less_equal_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_QuantizedGreater_8 = {
    .execute = greater_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_QuantizedGreaterEqual_8 = {
    .execute = greater_equal_q8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT(1),
};
