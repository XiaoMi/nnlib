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

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float_mathops.h>

#define BOX_IDX 0
#define ANCHOR_IDX 1
#define SCALE_IDX 2

#define OUTPUT_IDX 0
#define NUM_BOXDELTA_IN_BOXES 4


static int box_decoder_execute_f(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *boxes_tensor = self->inputs[BOX_IDX];
    const struct tensor *anchors_tensor = self->inputs[ANCHOR_IDX];
    const struct tensor *scale_tensor = self->inputs[SCALE_IDX];

    struct tensor *out_box_tensor = self->outputs[OUTPUT_IDX];

    const float* boxes = boxes_tensor->data;
    const float* anchors = anchors_tensor->data;
    const float* scale = scale_tensor->data;
    float* out_boxes = out_box_tensor->data;

    tensor_set_shape(out_box_tensor,boxes_tensor->shape.batches,boxes_tensor->shape.height,boxes_tensor->shape.width,boxes_tensor->shape.depth);
    out_box_tensor->data_size = boxes_tensor->shape.width * boxes_tensor->shape.height * boxes_tensor->shape.depth * boxes_tensor->shape.batches * sizeof(float);

    const int32_t num_boxes = boxes_tensor->shape.width;
    const int32_t batches = boxes_tensor->shape.batches;

    for (int32_t batch = 0; batch < batches; batch++) {
        const int batchOffset = batch * (num_boxes * NUM_BOXDELTA_IN_BOXES);

        for (int32_t i = 0; i < num_boxes; i++) {
            const int idxOffset = batchOffset + i * sizeof(float);

            const float anchorY0 = anchors[idxOffset + 0];
            const float anchorX0 = anchors[idxOffset + 1];
            const float anchorY1 = anchors[idxOffset + 2];
            const float anchorX1 = anchors[idxOffset + 3];

            const float anchorW = anchorX1 - anchorX0;
            const float anchorH = anchorY1 - anchorY0;
            const float anchorCenterY = anchorY0 + anchorH / 2;
            const float anchorCenterX = anchorX0 + anchorW / 2;
            const float boxDeltaY = boxes[idxOffset + 0] / scale[0];
            const float boxDeltaX = boxes[idxOffset + 1] / scale[1];
            const float boxDeltaH = boxes[idxOffset + 2] / scale[2];
            const float boxDeltaW = boxes[idxOffset + 3] / scale[3];
            const float boxH = fast_exp(boxDeltaH) * anchorH;
            const float boxW = fast_exp(boxDeltaW) * anchorW;
            const float boxCenterY = boxDeltaY * anchorH + anchorCenterY;
            const float boxCenterX = boxDeltaX * anchorW + anchorCenterX;

            const float decoded_box_Y0 = boxCenterY - boxH / 2;
            const float decoded_box_X0 = boxCenterX - boxW / 2;
            const float decoded_box_Y1 = boxCenterY + boxH / 2;
            const float decoded_box_X1 = boxCenterX + boxW / 2;
            out_boxes[idxOffset + 0] = (float) decoded_box_Y0;
            out_boxes[idxOffset + 1] = (float) decoded_box_X0;
            out_boxes[idxOffset + 2] = (float) decoded_box_Y1;
            out_boxes[idxOffset + 3] = (float) decoded_box_X1;
        }
    }
    return 0;

}

struct nn_node_ops nn_ops_for_Box_Decoder_f = {
        .execute = box_decoder_execute_f,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(1),
};
