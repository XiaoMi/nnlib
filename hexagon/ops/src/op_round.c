/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (mulject to the limitations in the
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
#include <math.h>
#include <stdbool.h>

static int ceil_f_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    out_tensor->shape.batches = in_tensor->shape.batches;
    out_tensor->shape.height = in_tensor->shape.height;
    out_tensor->shape.width = in_tensor->shape.width;
    out_tensor->shape.depth = in_tensor->shape.depth;
    out_tensor->data_size = out_tensor->shape.batches *
		out_tensor->shape.height *
		out_tensor->shape.width *
		out_tensor->shape.depth *
        sizeof(float);
    out_tensor->format.layout = NN_LAYOUT_PLAIN;
    out_tensor->format.type = NN_TYPE_FLOAT;

    int i, count;
    float *inf  = (float *) in_tensor->data;
    float *outf = (float *) out_tensor->data;
    count = in_tensor->data_size/tensor_type_size(in_tensor->format.type);
    for (i=0; i<count; i++) {
        *(outf++) = ceilf(*(inf++));
    }

    return 0;
}

static int floor_f_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    out_tensor->shape.batches = in_tensor->shape.batches;
    out_tensor->shape.height = in_tensor->shape.height;
    out_tensor->shape.width = in_tensor->shape.width;
    out_tensor->shape.depth = in_tensor->shape.depth;
    out_tensor->data_size = out_tensor->shape.batches *
		out_tensor->shape.height *
		out_tensor->shape.width *
		out_tensor->shape.depth *
        sizeof(float);
    out_tensor->format.layout = NN_LAYOUT_PLAIN;
    out_tensor->format.type = NN_TYPE_FLOAT;

    int i, count;
    float *inf  = (float *) in_tensor->data;
    float *outf = (float *) out_tensor->data;
    count = in_tensor->data_size/tensor_type_size(in_tensor->format.type);
    for (i=0; i<count; i++) {
        *(outf++) = floorf(*(inf++));
    }

    return 0;
}

static int round_f_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    if (nn_tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_FLOAT)) {
		return errlog( nn, "failed to allocate round_f output, size [%d,%d,%d,%d]",
                       (int)in_tensor->shape.batches, (int)in_tensor->shape.height,
                       (int)in_tensor->shape.width, (int)in_tensor->shape.depth);
    }

    int i, count;
    count = in_tensor->data_size/tensor_type_size(in_tensor->format.type);
    float *inf  = (float *) in_tensor->data;
    float *outf = (float *) out_tensor->data;
    for (i=0; i<count; i++) {
        const float raw = *(inf++);
        float rounded = roundf(raw);
        //math.h's roundf rounds away from zero, but TF does toward even
        //so if we rounded a half to an odd number, need to fix it up
        // Note: if difference isn't exactly +/-0.5f, then we didn't round
        //       a half and don't need to fix anything
        // Note2: Not using fenv/nearbyintf because some platforms don't support it (eg. SD820)
        const float difference = raw - rounded;
        if(unlikely( fabsf(difference)==0.5f)){   // usually will not
            if( ((int32_t)rounded )%2){           // was changed to odd..
                rounded += 2.0f * difference;     // round the other way
            }
        }
        *(outf++) = rounded;
    }
    
    return 0;
}


struct nn_node_ops nn_ops_for_Ceil_f = {
	.execute = ceil_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Floor_f = {
	.execute = floor_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Round_f = {
	.execute = round_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};
