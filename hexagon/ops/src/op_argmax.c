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
#include <quantize.h>
#include <math.h>
#include <quantize.h>
#include "nn_axis.h"

#ifdef HEXAGON_V66
#define ARGMAX_MAX_THREADS 4
#else
#define ARGMAX_MAX_THREADS 2
#endif

#include "hvx_inlines.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif
typedef HVX_Vector (*FUNC_PTR)(HVX_Vector, HVX_Vector);

#define AXIS_DEPTH_IDX   3

typedef struct
{
    int* dst;
    const uint8_t* src;
} argmax_job_info;

typedef struct
{
    argmax_job_info* jobs;
    int jobs_len;
    int work_dim;
    int hasleftover;
    nn_sem_t done_sem;
} argmax_thread_info;

static inline HVX_Vector  __attribute__((always_inline))
eltwise_vec_min(HVX_Vector v1, HVX_Vector v2)
{
    return Q6_Vub_vmin_VubVub(v1, v2);
}

static inline HVX_Vector  __attribute__((always_inline))
eltwise_vec_max(HVX_Vector v1, HVX_Vector v2)
{
    return Q6_Vub_vmax_VubVub(v1, v2);
}

//To find the max/min of a vector horizontally, perform log2 steps of rotation followed by a max/min
static HVX_Vector  __attribute__((always_inline))
horizontal_peak_vec (HVX_Vector in, FUNC_PTR peak_func) {
    HVX_Vector in_rot = Q6_V_vror_VR(in, 64);
    HVX_Vector val_result = peak_func(in, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 32);
    val_result = peak_func(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 16);
    val_result = peak_func(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 8);
    val_result = peak_func(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 4);
    val_result = peak_func(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 2);
    val_result = peak_func(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 1);
    val_result = peak_func(val_result, in_rot);
    return val_result;
}

static HVX_Vector  __attribute__((always_inline))
horizontal_max_vecpack (HVX_Vector in) {
    HVX_Vector in_rot = Q6_V_vror_VR(in, 64);
    HVX_Vector val_result = Q6_Vuh_vmax_VuhVuh(in, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 32);
    val_result = Q6_Vuh_vmax_VuhVuh(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 16);
    val_result = Q6_Vuh_vmax_VuhVuh(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 8);
    val_result = Q6_Vuh_vmax_VuhVuh(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 4);
    val_result = Q6_Vuh_vmax_VuhVuh(val_result, in_rot);
    in_rot = Q6_V_vror_VR(val_result, 2);
    val_result = Q6_Vuh_vmax_VuhVuh(val_result, in_rot);
    return val_result;
}

static uint8_t __attribute__((always_inline))
vec_element_extract(HVX_Vector v, int idx)
{
    union {
        HVX_Vector as_v;
        uint8_t as_u8[128];
    } uu;
    uu.as_v = v;
    return uu.as_u8[idx];
}
////////////////////////////////////////////
// optimized for argmax dim <= 128 case especially
// pack value and index together when search and extract max
static void __attribute__((always_inline))
horizontal_argmax128_hvx(argmax_job_info *jobinfo, const int dim) {
    int *dst = jobinfo->dst;
    uint8_t const *src = jobinfo->src;
    HVX_Vector val_result = Q6_V_vzero();
    val_result = Q6_V_vmux_QVV(Q6_Q_vsetq_R(dim), vmemu(src), val_result);
    // pack index-uint8 and value-uint8 information in uint16 format, index is in least significant part
    HVX_VectorPair vmmshuf = Q6_Wb_vshuffoe_VbVb(val_result, *(HVX_Vector const *)const_InverseCount128);
    // find out the maximum uint16 pair which contains both value and index
    HVX_Vector half_vmax_pack = Q6_Vuh_vmax_VuhVuh(Q6_V_lo_W(vmmshuf),Q6_V_hi_W(vmmshuf));
    HVX_Vector vmax_pack = horizontal_max_vecpack(half_vmax_pack);
    // use inverse index to ensure the index of max search logic is >, not >=
    *dst = 127-vec_element_extract(vmax_pack, 0);
}
// common approach to search max at first
// get masked index table by vcmp, extract index from it
static void __attribute__((always_inline))
horizontal_argmax_hvx(argmax_job_info *jobinfo, const int dim) {
    int *dst = jobinfo->dst;
    uint8_t const *src = jobinfo->src;
    int maxIdx = 0;
    HVX_Vector val_result = Q6_V_vzero();
    int32_t xd;
    const int leftovers = dim % sizeof(HVX_Vector);
    for (xd = 0; xd + 128 <=dim; xd +=128) {
        val_result = Q6_Vub_vmax_VubVub(vmemu(&src[xd]), val_result);
    }
    if (leftovers) {
        HVX_Vector last_result = Q6_Vub_vmax_VubVub(vmemu(&src[xd]), val_result);
        // use the last_result only in valid lanes
        val_result = Q6_V_vmux_QVV( Q6_Q_vsetq_R(leftovers), last_result, val_result);
    }

    val_result = horizontal_peak_vec(val_result, eltwise_vec_max);
    // max value locates at the index of vector
    uint8_t maxVal = vec_element_extract(val_result, 0);
    HVX_Vector max_Vector = q6op_Vb_vsplat_R(maxVal);
    HVX_Vector index_table = *(HVX_Vector const *)const_Count128;    // {0,1 ..127}
    // set initial index value to 128
    HVX_Vector index_init = q6op_Vb_vsplat_R(128);
    // Find corresponding index of max value
    for (xd = 0; xd <=dim; xd +=128) {
        HVX_VectorPred qcmpmask = Q6_Q_vcmp_eq_VbVb(vmemu(&src[xd]), max_Vector);
        HVX_Vector index_select = Q6_V_vmux_QVV(qcmpmask,index_table, index_init);
        // get the minimum index out of elements which equal to maximum value
        HVX_Vector index_result = horizontal_peak_vec(index_select, eltwise_vec_min);
        uint8_t vecMaxIdx = vec_element_extract(index_result, 0);
        maxIdx += vecMaxIdx;
        if (vecMaxIdx != 128) {
            break;
        }
    }

    *dst = maxIdx;
}

static void argmax_thread_work(struct nn_graph *nn, void *work_info) {
    argmax_thread_info* info = work_info;
    argmax_job_info* job_ptr = info->jobs;
    Q6_dcfetch_A(job_ptr);
    int jobs_len = info->jobs_len;
    const int dim = info->work_dim;
    if (info->hasleftover) {
        for(int i = 0; i < jobs_len; i++){
            Q6_dcfetch_A(&job_ptr[i+1]);
            horizontal_argmax_hvx(&job_ptr[i], dim);
        }
    }
    else {
        for(int i = 0; i < jobs_len; i++){
            Q6_dcfetch_A(&job_ptr[i+1]);
            horizontal_argmax128_hvx(&job_ptr[i], dim);
        }
    }
    nn_sem_post(&info->done_sem);
}

static inline int maxIdx_8toInt32 (const uint8_t *src, int dimSize, int stride)
{
    uint8_t maxVal = *src;
    src += stride;
    int maxIdx = 0;
    for (int i = 1; i < dimSize; i++,src+=stride) {
        if (*src > maxVal) {
            maxVal = *src;
            maxIdx = i;
        }
    }
    return maxIdx;
}

static inline uint8_t maxIdx_8 (const uint8_t *src, int dimSize, int stride)
{
    uint8_t maxVal = *src;
    src += stride;
    uint8_t maxIdx = 0;
    for (uint8_t i = 1; i < dimSize; i++,src+=stride) {
        if (*src > maxVal) {
            maxVal = *src;
            maxIdx = i;
        }
    }
    return maxIdx;
}

// input - quantized uint8 data, output - int32 index data
static int argmax_execute_8toInt32(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *axis_tensor = self->inputs[1];
    struct tensor *out_tensor = self->outputs[0];


    int32_t axis_idx0 = tensor_get_int32(axis_tensor,0);
    int res = handle_negative_axes(nn, &axis_idx0, 1);
    if (res)
        return errlog(nn, "ArgMax_8toInt32: axis is out of range \n");
    int axis_idx = axis_idx0;

    const struct shape inshape = in_tensor->shape;
    struct shape outshape = in_tensor->shape;
    outshape.dimension[axis_idx] = 1;

    int dim = inshape.dimension[axis_idx];
    int stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > axis_idx; i--)
    {
        stride *= inshape.dimension[i];
    }

    if (tensor_out_prepare_normal_fromshape(out_tensor,&outshape,NN_TYPE_INT32) != 0) {
        return errlog(nn,"failed to prepare output");
    }

    int inputIdx = 0;
    int counter = 0;
    int *out_data = out_tensor->data;

    const uint8_t *in_base = in_tensor->data;
    int dataLength = in_tensor->data_size/sizeof(uint8_t);
    if (stride == 1) {
        // Multi-thread just for common case - stride=1 (mostly argmax on depth)
        // If input depth=1, and argmax on width, it is also taken as common case
        int workloads_len = inshape.batches*inshape.height*inshape.width*inshape.depth/dim;
        int num_threads = min_i32(workloads_len, ARGMAX_MAX_THREADS);
        int workload_for_worker[ARGMAX_MAX_THREADS];
        int average_workload_per_worker = workloads_len / num_threads;
        int extra_work = workloads_len % num_threads;
        for(int i = 0; i < num_threads; i++){
            workload_for_worker[i] = average_workload_per_worker;
        }
        for(int i = 0; i < extra_work; i++){
            workload_for_worker[i] += 1;
        }

        //calc the argmax axis max search job positions
        argmax_job_info* jobs_ptr = (argmax_job_info*)nn->scratch;
        while (inputIdx < dataLength) {
            jobs_ptr->dst = out_data++;
            jobs_ptr->src = in_base+inputIdx;
            jobs_ptr++;
            inputIdx += dim;
        }

        int offset = 0;
        argmax_thread_info thrinfo[ARGMAX_MAX_THREADS];
        const int leftovers = (dim > sizeof(HVX_Vector));

        for(int i = 0; i < num_threads; i++) {
            nn_sem_init(&thrinfo[i].done_sem, 0);
            thrinfo[i].jobs_len = workload_for_worker[i];
            thrinfo[i].jobs = ((argmax_job_info*)nn->scratch) + offset;
            thrinfo[i].work_dim = dim;
            thrinfo[i].hasleftover = leftovers;
            offset += workload_for_worker[i];
            nn_os_work_for_vector(nn, argmax_thread_work, &thrinfo[i]);
        }

        for(int i =0; i < num_threads; i++) {
            nn_sem_wait(&thrinfo[i].done_sem);
        }
    }
    else {
        while (inputIdx < dataLength) {
            if (counter < stride) {
                *out_data++ = maxIdx_8toInt32(in_base+inputIdx,dim,stride);
                counter++;
                inputIdx++;
            } else {
                counter = 0;
                inputIdx += (dim - 1) * stride;
            }
        }
    }

    return 0;
}

static int argmax_execute_8(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *axis_tensor = self->inputs[1];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    int32_t axis_idx0 = tensor_get_int32(axis_tensor,0);
    int res = handle_negative_axes(nn, &axis_idx0, 1);
    if (res)
        return errlog(nn, "ArgMax_8: axis is out of range \n");
    int axis_idx = axis_idx0;

    const struct shape inshape = in_tensor->shape;
    struct shape outshape = in_tensor->shape;
    outshape.dimension[axis_idx] = 1;

    int dim = inshape.dimension[axis_idx];
    if (dim > 256) {
        return errlog(nn,"argmax_8 does not support dimension > 256 because of q8 resolution");
    }

    int stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > axis_idx; i--)
    {
        stride *= inshape.dimension[i];
    }

    tensor_set_single_float(out_min_tensor, 0.0f);
    tensor_set_single_float(out_max_tensor, 255.0f);

    if (tensor_out_prepare_normal_fromshape(out_tensor,&outshape,NN_TYPE_QUINT8) != 0) {
        return errlog(nn,"failed to prepare output");
    }

    int inputIdx = 0;
    int counter = 0;
    uint8_t *out_data = out_tensor->data;
    const uint8_t *in_base = in_tensor->data;
    int dataLength = in_tensor->data_size/sizeof(uint8_t);

    while (inputIdx < dataLength) {
        if (counter < stride) {
            *out_data++ = maxIdx_8(in_base+inputIdx,dim,stride);
            counter++;
            inputIdx++;
        } else {
            counter = 0;
            inputIdx += (dim - 1) * stride;
        }
    }

    return 0;
}

static int argmax_check_8toInt32(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"Checking argmax node %p",self);

    const struct shape inshape = self->inputs[0]->shape;
    const int axis_idx = tensor_get_int32(self->inputs[1],0);
    int dim = inshape.dimension[axis_idx];
    int stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > axis_idx; i--)
    {
        stride *= inshape.dimension[i];
    }
    if (stride == 1) {
        // Grow scratch in prepare
        // Multi-thread just for common case - stride=1 (mostly argmax on depth)
        // If input depth=1, and argmax on width, it is also taken as common case
        int workloads_len = inshape.batches*inshape.height*inshape.width*inshape.depth/dim;
        if(nn_scratch_grow(nn, sizeof(argmax_job_info) * workloads_len)) {
            return errlog(nn,"insufficient scratch");
        }
    }
    logmsg(nn,2,"argmax %p check OK",self);
    return 0;
}


struct nn_node_ops nn_ops_for_ArgMax_8toInt32 = {
    .execute = argmax_execute_8toInt32,
    .check = argmax_check_8toInt32,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_ArgMax_8 = {
    .execute = argmax_execute_8,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(3),
};
