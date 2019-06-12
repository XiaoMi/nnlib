
/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains implementations for quantized unary min node
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include "hvx_inlines.h"
#include "nn_axis.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#include "hvx_inlines.h"
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif
#define NUM_DIMS 4
typedef HVX_Vector (*FUNC_PTR)(HVX_Vector, HVX_Vector);
typedef void (*EXEC_FUNC)(struct nn_graph *, void * );



enum ReductionOp
{
    MIN,
    MAX
};
struct tdata {
    uint8_t *in_data;
    uint8_t *out_data;
    int32_t reduction_batches;
    int32_t num_blobs;
    int32_t blob_size;
    uint8_t init_val;
    nn_sem_t donesem;
};

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
//To find the min of a vector horizontally, perform log2 steps of rotation followed by a min

static HVX_Vector  __attribute__((always_inline))
horizontal_reduce_vec (HVX_Vector in, FUNC_PTR reduction_func) {
    HVX_Vector in_rot = Q6_V_vror_VR(in, 64);
    HVX_Vector cur_result = reduction_func(in, in_rot);
    in_rot = Q6_V_vror_VR(cur_result, 32);
    cur_result = reduction_func(cur_result, in_rot);
    in_rot = Q6_V_vror_VR(cur_result, 16);
    cur_result = reduction_func(cur_result, in_rot);
    in_rot = Q6_V_vror_VR(cur_result, 8);
    cur_result = reduction_func(cur_result, in_rot);
    in_rot = Q6_V_vror_VR(cur_result, 4);
    cur_result = reduction_func(cur_result, in_rot);
    in_rot = Q6_V_vror_VR(cur_result, 2);
    cur_result = reduction_func(cur_result, in_rot);
    in_rot = Q6_V_vror_VR(cur_result, 1);
    cur_result = reduction_func(cur_result, in_rot);
    return cur_result;
}

static uint8_t __attribute__((unused))
vec_extract( HVX_Vector v)
{
    union {
        HVX_Vector v;
        uint8_t as_u8[128];
    } uu = { v };
    return uu.as_u8[0];
}
////////////////////////////////////////////

static void __attribute__((always_inline))
reduce_all_axes_hvx_template(struct nn_graph *nn, void *vtd, FUNC_PTR reduction_func) {

    struct tdata *td = vtd;
    uint8_t *in_data = td->in_data;
    uint8_t *out_data = td->out_data;
    const int blob_size = td->blob_size;
    uint8_t init_val = td->init_val;
    const int leftovers = blob_size % sizeof(HVX_Vector);
    int xd;
    HVX_Vector cur_result = q6op_Vb_vsplat_R(init_val);
    for (xd = 0; xd + 128 <=blob_size; xd +=128) {
        cur_result = reduction_func(*(HVX_Vector * ) & in_data[xd], cur_result);
    }
    if (leftovers) {
        HVX_Vector in = *(HVX_Vector * )&in_data[xd];
        HVX_Vector last_result = reduction_func( in, cur_result);
        // use the last_result only in valid lanes
        cur_result = Q6_V_vmux_QVV( Q6_Q_vsetq_R(leftovers), last_result, cur_result);
    }
    cur_result = horizontal_reduce_vec(cur_result, reduction_func);
    out_data[0] = vec_extract(cur_result);
    nn_sem_post(&td->donesem);
}

static void reduce_all_axes_hvx_MIN(struct nn_graph *nn, void *vtd) {
	reduce_all_axes_hvx_template(nn,vtd, eltwise_vec_min);
}
static void reduce_all_axes_hvx_MAX(struct nn_graph *nn, void *vtd) {
	reduce_all_axes_hvx_template(nn,vtd, eltwise_vec_max);
}
//////////////////////////////////////////////////////

static void  __attribute__((always_inline))
reduce_single_axis_hvx_template(struct nn_graph *nn, void *vtd, FUNC_PTR reduction_func) {

    struct tdata *td = vtd;
    uint8_t *in_data = td->in_data;
    uint8_t *out_data = td->out_data;
    const int num_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
    uint8_t init_val = td->init_val;
    const int leftovers = blob_size % sizeof(HVX_Vector);
    int xd;
    HVX_Vector cur_result = q6op_Vb_vsplat_R(init_val);

    for (int n = 0; n < num_batches; n++) {
        for (xd = 0; xd + 128 <=blob_size; xd +=128) {
            cur_result = q6op_Vb_vsplat_R(init_val);
            for (int i = 0; i < num_blobs; i++) {
                cur_result = reduction_func( vmemu(&in_data[i * blob_size + xd]), cur_result);
            }
            vmemu(&out_data[xd]) = cur_result;
        }
        if (leftovers) {
            cur_result = q6op_Vb_vsplat_R(init_val);
            // we don't need to mask extra bytes since they won't be stored.
            for (int i = 0; i < num_blobs; i++) {
                HVX_Vector in = vmemu(&in_data[i * blob_size + xd]);
                cur_result = reduction_func(in, cur_result);
            }
            HVX_Vector *outp = (HVX_Vector*)&out_data[xd];
            // do unaligned store of first 1..127 bytes in cur_result
            q6op_vstu_variable_ARV( outp, leftovers, cur_result);
        }
        in_data += num_blobs * blob_size;
        out_data += blob_size;
    }
    nn_sem_post(&td->donesem);
}

static void
reduce_single_axis_hvx_MIN(struct nn_graph *nn, void *vtd)
{
	reduce_single_axis_hvx_template( nn, vtd,eltwise_vec_min );
}
static void
reduce_single_axis_hvx_MAX(struct nn_graph *nn, void *vtd)
{
	reduce_single_axis_hvx_template( nn, vtd,eltwise_vec_max );
}
////////////////////////////////////////////////////////////////////

static int reduction_execute(struct nn_node *self, struct nn_graph *nn, int reduction_type)
{
    uint8_t init_val = 0;

    EXEC_FUNC reduce_all_axes_fp;
    EXEC_FUNC reduce_single_axis_fp;

    if (reduction_type == MAX)
    {
    	reduce_all_axes_fp =  reduce_all_axes_hvx_MAX;
    	reduce_single_axis_fp =  reduce_single_axis_hvx_MAX;
    }
    else if (reduction_type == MIN)
    {
    	init_val = 255;
      	reduce_all_axes_fp =  reduce_all_axes_hvx_MIN;
        reduce_single_axis_fp =  reduce_single_axis_hvx_MIN;
    }
    else
    {
        return errlog(nn, "Specified currently unsupported reduction function");
    }
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *axes_tensor = self->inputs[3];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_height = in_tensor->shape.height;
    int32_t in_width = in_tensor->shape.width;
    int32_t in_depth = in_tensor->shape.depth;
    int32_t* axes_ori = (int32_t*)axes_tensor->data;
    int32_t axes_size = axes_tensor->data_size / sizeof(int32_t);
    uint8_t *in_data = in_tensor->data;
    uint8_t *out_data_final = out_tensor->data;
    int32_t elemcount = in_batches * in_height * in_width * in_depth;

    //Check that the final reduction dims are sane
    int32_t modified_shape_final[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
    int32_t modified_data_size = 1;

    // Handle negative axes by re-interpreting them as positive axes
    if( handle_negative_axes(nn, axes_ori, axes_size)!=0) return -1;

    // Remove duplicate axes
    int32_t distinctive_axes_size=0;
    int32_t appeared[] = {0,0,0,0};
    for (int i = 0; i < axes_size; i++) {
        modified_shape_final[axes_ori[i]] = 1;
 	if (appeared[axes_ori[i]]==0){
	    distinctive_axes_size+=1;
	    appeared[axes_ori[i]]=1;
	}
    }
    int32_t appeared2[] = {0,0,0,0};
    int32_t axes[distinctive_axes_size];
    int j=0;
    for (int i = 0; i < axes_size; i++) {
 	if (appeared2[axes_ori[i]]==0){
	    appeared2[axes_ori[i]]=1;
	    axes[j] = axes_ori[i];
	    j++;
	}
    }
    axes_size = distinctive_axes_size;
    for (int i = 0; i < NUM_DIMS; i++) {
        modified_data_size *= modified_shape_final[i];
    }
    if (axes_size > NUM_DIMS)
        return errlog(nn, "Number of elements in axes tensor is %d, support a maximum of 4", axes_size);
    int32_t out_batches = out_tensor->shape.batches;
    int32_t out_height = out_tensor->shape.height;
    int32_t out_width = out_tensor->shape.width;
    int32_t out_depth = out_tensor->shape.depth;
    int32_t out_data_size = out_batches * out_height * out_width * out_depth;

    if (out_data_size != modified_data_size) {
        return errlog(nn, "Output tensor is of size %d, but expected output tensor of size %d", out_data_size, modified_data_size);
    }
    if (nn_scratch_grow(nn, elemcount)){
        return errlog(nn,"failed to get scratch");
    }
    uint8_t *out_data= nn->scratch;

    if (tensor_out_prepare_normal_fromshape( out_tensor, & out_tensor->shape, NN_TYPE_QUINT8 )!= 0)
        return errlog(nn,"out too small");
    if (axes_size == 4) {
        struct tdata td = {
                .in_data = in_data,
                .out_data = out_data_final,
                .reduction_batches = 1,
                .num_blobs = 1,
                .blob_size = elemcount,
                .init_val = init_val
        };
        nn_sem_init(&td.donesem, 0);
        nn_os_work_for_vector(nn, reduce_all_axes_fp, &td);
        nn_sem_wait(&td.donesem);
    }
    else {
        int32_t modified_shape[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
        for (int a = 0; a < axes_size; a++) {
            if (a == axes_size - 1) {
                out_data = out_data_final;
            }
            int32_t axis = axes[a];
            int32_t reduction_batches = 1;
            int32_t blob_size = 1;
            for (int i = 0; i < axis; i++) {
                reduction_batches *= modified_shape[i];
            }
            for (int i = axis+1; i < NUM_DIMS; i++) {
                blob_size *= modified_shape[i];
            }

            struct tdata td = {
                    .in_data = in_data,
                    .out_data = out_data,
                    .reduction_batches = reduction_batches,
                    .num_blobs = modified_shape[axis],
                    .blob_size = blob_size,
                    .init_val = init_val
            };
            nn_sem_init(&td.donesem, 0);
            nn_os_work_for_vector(nn, reduce_single_axis_fp, &td);
            nn_sem_wait(&td.donesem);
            in_data = out_data;
            // set new shape as the axis is reduced
            modified_shape[axis] = 1;
        }
    }

    tensor_copy(out_min_tensor,in_min_tensor);
    tensor_copy(out_max_tensor,in_max_tensor);
    return 0;
}
static int min_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum ReductionOp op = MIN;
    return reduction_execute(self, nn, op);
}

static int max_execute(struct nn_node *self, struct nn_graph *nn)
{
    enum ReductionOp op = MAX;
    return reduction_execute(self, nn, op);
}


struct nn_node_ops nn_ops_for_QuantizedMax_8 = {
        .execute = max_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(4),
        .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedMin_8 = {
        .execute = min_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(4),
        .n_outputs = NN_IOCOUNT(3),
};
