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

////////////////////////////////////////
#error "CURRENTLY UNUSED - 'Transpose' is a functional superset, and has a faster implementation."
////////////////////////////////////////

static inline void copy_f(const struct tensor *in_tensor, size_t inputIdx,
        struct tensor *out_tensor, size_t outputIdx){
    tensor_set_float(out_tensor,outputIdx,tensor_get_float(in_tensor,inputIdx));
}
static inline void copy_8(const struct tensor *in_tensor, size_t inputIdx,
        struct tensor *out_tensor, size_t outputIdx){
    const uint8_t* in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    out_data[outputIdx] = in_data[inputIdx];
}

static inline int permute_execute_generic_ref(struct nn_node *self, struct nn_graph *nn,
        void (*copier)(const struct tensor*, size_t, struct tensor *, size_t )){

    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *order = self->inputs[1];
    struct tensor *out_tensor = self->outputs[0];
    uint32_t in_b = in_tensor->shape.batches;
    uint32_t in_h = in_tensor->shape.height;
    uint32_t in_w = in_tensor->shape.width;
    uint32_t in_d = in_tensor->shape.depth;

    uint32_t order_b = tensor_get_int32(order,0);
    uint32_t order_h = tensor_get_int32(order,1);
    uint32_t order_w = tensor_get_int32(order,2);
    uint32_t order_d = tensor_get_int32(order,3);

    uint32_t in_stride_b = in_h*in_w*in_d;
    uint32_t in_stride_h = in_w*in_d;
    uint32_t in_stride_w = in_d;
    uint32_t in_stride_d = 1;

    uint32_t in_dims[4] = {in_b,in_h,in_w,in_d};
    uint32_t out_b = in_dims[order_b];
    uint32_t out_h = in_dims[order_h];
    uint32_t out_w = in_dims[order_w];
    uint32_t out_d = in_dims[order_d];

    uint32_t out_stride_b = out_h*out_w*out_d;
    uint32_t out_stride_h = out_w*out_d;
    uint32_t out_stride_w = out_d;
    uint32_t out_stride_d = 1;

    logmsg(nn,2,"(q)permute execute. self=%p ",self);
    if (out_tensor->max_size < in_tensor->data_size) {
        return errlog(nn,"out too small");
    }

    /* Copy input tensor to output */
    tensor_set_shape(out_tensor,out_b,out_h,out_w,out_d);
    out_tensor->data_size = in_tensor->data_size;

    size_t w = 0, y = 0, x = 0, z = 0;
    size_t outputIdx = 0, inputIdx = 0;
    size_t* outputAxisPtr[4]={&w,&y,&x,&z};
    for( w = 0; w < in_b; ++w ){
        for( y = 0; y < in_h; ++y ){
            for( x = 0; x < in_w; ++x ){
                for( z = 0; z < in_d; ++z ){
                    outputIdx =
                        *(outputAxisPtr[order_b])*out_stride_b +
                        *(outputAxisPtr[order_h])*out_stride_h +
                        *(outputAxisPtr[order_w])*out_stride_w +
                        *(outputAxisPtr[order_d])*out_stride_d;
                    inputIdx = w*in_stride_b + y*in_stride_h + x*in_stride_w + z*in_stride_d;
                    copier(in_tensor,inputIdx,out_tensor,outputIdx);
                }
            }
        }
    }

    logmsg(nn,2,"qpermute %dx%dx%dx%x (%dx%dx%dx%d) --> %dx%dx%dx%d",
        in_b,in_h,in_w,in_d,
        order->shape.batches,
        order->shape.height,
        order->shape.width,
        order->shape.depth,
        out_b,out_h,out_w,out_d);
    return 0;
}

static int permute_execute_ref_8(struct nn_node *self, struct nn_graph *nn){
    int32_t retval = permute_execute_generic_ref(self, nn, copy_8);
    if (retval){
        return retval;
    }
    /*Handle min and max*/
    if (tensor_copy(self->outputs[1],self->inputs[2]) != 0) {
        return errlog(nn,"failed to copy min");
    }
    if (tensor_copy(self->outputs[2],self->inputs[3]) != 0) {
        return errlog(nn,"failed to copy max");
    }
    return 0;
}

static int permute_execute_ref_f(struct nn_node *self, struct nn_graph *nn){
    return permute_execute_generic_ref(self, nn, copy_f);
}

static inline int common_permute_checks(struct nn_node *self, struct nn_graph* nn){
    const struct tensor *order = self->inputs[1];
    if (order->shape.depth != 4){
        return errlog(nn,"not all permute valuess given. Expected 4 values got %dd",order->shape.depth);
    }
    uint32_t order_b = tensor_get_int32(order,0);
    uint32_t order_h = tensor_get_int32(order,1);
    uint32_t order_w = tensor_get_int32(order,2);
    uint32_t order_d = tensor_get_int32(order,3);


    /*checking if duplicates present in order values
        for example: order values of [1,3,2,3] is invalid
        but [1,3,2,0] is valid
    */
    uint8_t duplicates_present[4] = {0,0,0,0};
    uint32_t orders[4] = {order_b,order_h,order_w,order_d};
    for (int8_t i = 0; i < 4; i++){
        uint32_t current = orders[i];

        if (current > 3){
            return errlog(nn,"invalid order value given %d for dimension: %d",current,i);
        }
        if (duplicates_present[current]){
            return errlog(nn,"order value: %d used more than once",current);
        }
        duplicates_present[current] = 1;
    }
    return 0;
}

static int permute_check(struct nn_node *self, struct nn_graph *nn){
    logmsg(nn,2,"Checking permute node %p",self);

    if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
    if (self->n_outputs != 1) return errlog(nn,"wrong # outputs %d",self->n_outputs);
    int32_t retval = common_permute_checks(self,nn);
    if (retval){
        return retval;
    }
    logmsg(nn,2,"permute node %p check OK",self);
    return 0;
}

static int qpermute_check(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"Checking Qpermute node %p",self);
    if (self->n_inputs != 4) return errlog(nn,"wrong # inputs");
    if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
    int32_t retval = common_permute_checks(self,nn);
    if (retval){
        return retval;
    }
    logmsg(nn,2,"Qpermute node %p check OK",self);
    return 0;
}

////////////////////////////////////////
#error "CURRENTLY UNUSED - 'Transpose' is a functional superset, and has a faster implementation."
////////////////////////////////////////

struct nn_node_ops nn_ops_for_Permute_f = {
    .execute = permute_execute_ref_f,
    .check = permute_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};


struct nn_node_ops nn_ops_for_QuantizedPermute_8 = {
    .execute = permute_execute_ref_8,
    .check = qpermute_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};

