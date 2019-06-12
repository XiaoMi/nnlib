/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
#include <stdio.h>
#include <stdlib.h>

#define INPUT_IDX 0
#define INPUT_MIN_IDX 1
#define INPUT_MAX_IDX 2
#define PAD_IDX 3
#define TILE_IDX 4
#define OUTPUT_DATA_IDX 0
#define OUTPUT_MIN_IDX 1
#define OUTPUT_MAX_IDX 2
#define OUTPUT_SHAPE_IDX 3

static int implode_batch_ref_execute(struct nn_node *self, struct nn_graph *nn){
    const struct tensor *input_tensor = self->inputs[INPUT_IDX];
    const struct tensor *min_tensor = self->inputs[INPUT_MIN_IDX];
    const struct tensor *max_tensor = self->inputs[INPUT_MAX_IDX];
    const struct tensor *pad_tensor = self->inputs[PAD_IDX];
    const struct tensor *tile_tensor = self->inputs[TILE_IDX];
    struct tensor *output_tensor = self->outputs[OUTPUT_DATA_IDX];
    struct tensor *shape_tensor = self->outputs[OUTPUT_SHAPE_IDX];
    uint8_t* in_data = input_tensor->data;
    uint8_t* out_data = output_tensor->data;

    tensor_copy(self->outputs[OUTPUT_MIN_IDX],min_tensor);
    tensor_copy(self->outputs[OUTPUT_MAX_IDX],max_tensor);

    float min_f = tensor_get_float(min_tensor,0);
    float max_f = tensor_get_float(max_tensor,0);
    float pad_f = tensor_get_float(pad_tensor,0);
    if (min_f > pad_f || max_f < pad_f){
        return errlog(nn, "pad value %f does not fall in range of input (%f to %f)",pad_f,min_f,max_f);
    }
    uint8_t pad_value = quantize_uint8(pad_f,min_f,max_f);

    int32_t in_height = input_tensor->shape.height;
    int32_t pad_height = pad_tensor->shape.height;
    int32_t tile_height = tile_tensor->shape.height;
    int32_t big_height = in_height + pad_height;
    int32_t out_height = tile_height*(big_height)-pad_height;

    int32_t in_width = input_tensor->shape.width;
    int32_t pad_width = pad_tensor->shape.width;
    int32_t tile_width = tile_tensor->shape.width;
    int32_t big_width = in_width + pad_width;
    int32_t out_width = tile_width*big_width-pad_width;

    int32_t depth = input_tensor->shape.depth;
    int32_t total_batches = input_tensor->shape.batches;
    int32_t in_size = in_height * in_width * depth;
    int32_t out_size = out_height * out_width * depth;
    if(output_tensor->max_size < out_size){
        return errlog(nn,"out too small %d < %d",output_tensor->max_size,out_size);
    }
    output_tensor->data_size = out_size;
    memset(out_data,pad_value,out_size);
    int32_t in_row_bytes = in_width * depth * sizeof (uint8_t);
    int32_t out_row_bytes = out_width * depth * sizeof (uint8_t);
    int32_t data_pad_per_width_bytes = big_width * depth * sizeof (uint8_t);
    // int32_t data_pad_per_width_bytes = tile_height * big_height;
    tensor_set_shape(output_tensor,1,out_height,out_width,depth);
    tensor_set_shape(shape_tensor,1,1,1,4);

    float shape[] = {total_batches,in_height,in_width,depth};
    if(shape_tensor->max_size < sizeof(shape)){
        return errlog(nn,"shape out too small %d < %d",shape_tensor->max_size,sizeof(shape));
    }
    shape_tensor->data_size = sizeof(shape);
    for (int i = 0; i < 4;i++){
        tensor_set_float(shape_tensor,i,shape[i]);
    }
    int32_t batch_height_index,batch_width_index, current_num_batches;
    int32_t out_batched_row, out_batched_col,out_cur_batch_byte_index,in_cur_batch_byte_index;
    int32_t out_row_index,in_row_index,in_byte_index,out_byte_index;

    for (batch_height_index=0; batch_height_index<tile_height; batch_height_index++){
        out_batched_row = big_height * batch_height_index;
        for (batch_width_index=0; batch_width_index<tile_width; batch_width_index++){
            current_num_batches = batch_height_index * tile_width + batch_width_index;
            if (current_num_batches >total_batches){
                break;
            }
            out_batched_col = big_width * batch_width_index;
            out_cur_batch_byte_index = data_pad_per_width_bytes*batch_width_index;
            in_cur_batch_byte_index = current_num_batches * in_size;

            for (in_row_index = 0; in_row_index <in_height; in_row_index++){
                out_row_index = out_batched_row + in_row_index;
                in_byte_index =  in_cur_batch_byte_index+ in_row_index * in_row_bytes;
                out_byte_index = out_cur_batch_byte_index + out_row_index * out_row_bytes;
                memcpy(out_data+out_byte_index,in_data+in_byte_index,in_row_bytes);
            }

        }
    }
    return 0;
}

struct implode_worker{
    struct nn_node *self;
    uint8_t* in_data;
    uint8_t* out_data;

    int32_t in_height;
    int32_t pad_height;
    int32_t tile_height;
    int32_t out_height;

    int32_t in_width;
    int32_t pad_width;
    int32_t tile_width;
    int32_t out_width;
    int32_t depth;
    int32_t total_batches;
    uint8_t pad_value;
};
static void implode_batch_hvx(struct nn_graph *nn, void *vinfo){
    struct implode_worker *inf=(struct implode_worker*)vinfo;
    uint8_t* in_data=inf->in_data;
    uint8_t* out_data=inf->out_data;

    int32_t in_height=inf->in_height;
    int32_t pad_height=inf->pad_height;
    int32_t tile_height=inf->tile_height;
    int32_t out_height=inf->out_height;

    int32_t in_width=inf->in_width;
    int32_t pad_width=inf->pad_width;
    int32_t tile_width=inf->tile_width;
    int32_t out_width=inf->out_width;

    int32_t depth=inf->depth;
    int32_t total_batches=inf->total_batches;
    uint8_t pad_value=inf->pad_value;

    int32_t in_size=in_height*in_width*depth;
    int32_t out_size=out_height*out_width*depth;
    vmemset_asm(out_data,pad_value,out_size);

    int32_t in_row_bytes = in_width * depth * sizeof (uint8_t);
    int32_t out_row_bytes = out_width * depth * sizeof (uint8_t);
    int32_t data_pad_per_width_bytes = (in_width+pad_width) * depth * sizeof (uint8_t);

    int32_t batch_height_index,batch_width_index, current_num_batches;
    int32_t out_batched_row, out_batched_col,out_cur_batch_byte_index,in_cur_batch_byte_index;
    int32_t out_row_index,in_row_index,in_byte_index,out_byte_index;

    for (batch_height_index=0; batch_height_index<tile_height; batch_height_index++){
        out_batched_row = (in_height+pad_height) * batch_height_index;
        for (batch_width_index=0; batch_width_index<tile_width; batch_width_index++){
            current_num_batches = batch_height_index * tile_width + batch_width_index;
            if (current_num_batches >total_batches){
                break;
            }
            out_batched_col = (in_width+pad_width) * batch_width_index;
            out_cur_batch_byte_index = data_pad_per_width_bytes*batch_width_index;
            in_cur_batch_byte_index = current_num_batches * in_size;

            for (in_row_index = 0; in_row_index <in_height; in_row_index++){
                out_row_index = out_batched_row + in_row_index;
                in_byte_index =  in_cur_batch_byte_index+ in_row_index * in_row_bytes;
                out_byte_index = out_cur_batch_byte_index + out_row_index * out_row_bytes;
                vmemcpy_asm(out_data+out_byte_index,in_data+in_byte_index,in_row_bytes);
            }
        }
    }
    nn_sem_post(inf->self->opaque);
}

static int implode_batch_hvx_execute(struct nn_node *self, struct nn_graph *nn){
    const struct tensor *input_tensor = self->inputs[INPUT_IDX];
    const struct tensor *min_tensor = self->inputs[INPUT_MIN_IDX];
    const struct tensor *max_tensor = self->inputs[INPUT_MAX_IDX];
    const struct tensor *pad_tensor = self->inputs[PAD_IDX];
    const struct tensor *tile_tensor = self->inputs[TILE_IDX];
    struct tensor *output_tensor = self->outputs[OUTPUT_DATA_IDX];
    struct tensor *shape_tensor = self->outputs[OUTPUT_SHAPE_IDX];
    uint8_t* in_data = input_tensor->data;
    uint8_t* out_data = output_tensor->data;

    tensor_copy(self->outputs[OUTPUT_MIN_IDX],min_tensor);
    tensor_copy(self->outputs[OUTPUT_MAX_IDX],max_tensor);

    float min_f = tensor_get_float(min_tensor,0);
    float max_f = tensor_get_float(max_tensor,0);
    float pad_f = tensor_get_float(pad_tensor,0);
    if (min_f > pad_f || max_f < pad_f){
        return errlog(nn, "pad value %f does not fall in range of input (%f to %f)",pad_f,min_f,max_f);
    }
    uint8_t pad_value = quantize_uint8(pad_f,min_f,max_f);

    int32_t in_height = input_tensor->shape.height;
    int32_t pad_height = pad_tensor->shape.height;
    int32_t tile_height = tile_tensor->shape.height;
    int32_t out_height = tile_height*(in_height + pad_height)-pad_height;

    int32_t in_width = input_tensor->shape.width;
    int32_t pad_width = pad_tensor->shape.width;
    int32_t tile_width = tile_tensor->shape.width;
    int32_t out_width = tile_width*(in_width + pad_width)-pad_width;

    int32_t depth = input_tensor->shape.depth;
    int32_t total_batches = input_tensor->shape.batches;
    int32_t out_size = out_height * out_width * depth;
    if(output_tensor->max_size < out_size){
        return errlog(nn,"out too small %d < %d",output_tensor->max_size,out_size);
    }
    output_tensor->data_size = out_size;
    tensor_set_shape(output_tensor,1,out_height,out_width,depth);
    tensor_set_shape(shape_tensor,1,1,1,4);

    float shape[] = {total_batches,in_height,in_width,depth};
    if(shape_tensor->max_size < sizeof(shape)){
        return errlog(nn,"shape out too small %d < %d",shape_tensor->max_size,sizeof(shape));
    }
    shape_tensor->data_size = sizeof(shape);
    memcpy(shape_tensor->data,shape,sizeof(shape));
    struct implode_worker td={
        .self=self,
        .in_data=in_data,
        .out_data=out_data,

        .in_height=in_height,
        .pad_height=pad_height,
        .tile_height=tile_height,
        .out_height=out_height,

        .in_width=in_width,
        .pad_width=pad_width,
        .tile_width=tile_width,
        .out_width=out_width,

        .depth=depth,
        .total_batches=total_batches,
        .pad_value=pad_value
    };
    nn_sem_t sem;
    nn_sem_init(&sem,0);
    self->opaque = &sem;
    nn_os_work_for_vector(nn, implode_batch_hvx, &td);
    nn_sem_wait(&sem);
    self->opaque = NULL;
    return 0;
}



struct nn_node_ops nn_ops_for_Implode_8_ref = {
    .execute = implode_batch_ref_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(5),
    .n_outputs = NN_IOCOUNT(4),
};
struct nn_node_ops nn_ops_for_Implode_8 = {
    .execute = implode_batch_hvx_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(5),
    .n_outputs = NN_IOCOUNT(4),
};
