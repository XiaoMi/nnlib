
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
#include <math.h>

/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains an axis-aligned bbox tranform op
 */

#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

typedef struct {
    float x1, y1, x2, y2;
} BoxEncodingCorner;

typedef struct {
    float w, h, x, y;
} BoxEncodingCenter;

static int axis_aligned_bbox_transform_execute(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"axis_aligned_bbox_transform execute. self=%p ",self);

    const struct tensor *boxes_input_tensor = self->inputs[0];
    const struct tensor *deltas_input_tensor = self->inputs[1];
    const struct tensor *batch_splits_input_tensor = self->inputs[2];
    const struct tensor *image_info_input_tensor = self->inputs[3];

    struct tensor *boxes_output_tensor = self->outputs[0];

    float *boxes_input = boxes_input_tensor->data;
    float *deltas_input = deltas_input_tensor->data;
    int *batch_splits = batch_splits_input_tensor->data;
    float *image_info = image_info_input_tensor->data;

    float *boxes_output = boxes_output_tensor->data;

    tensor_out_prepare_normal(boxes_output_tensor,	1,1,deltas_input_tensor->shape.width,deltas_input_tensor->shape.depth, NN_TYPE_FLOAT);

    const uint32_t roiLength = 4;
    const uint32_t imageLength = 2;

    uint32_t numClasses = deltas_input_tensor->shape.depth / roiLength;
    uint32_t numBatches = image_info_input_tensor->shape.width;

    int boxes_tensor_size = boxes_input_tensor->shape.batches * boxes_input_tensor->shape.height * boxes_input_tensor->shape.width * boxes_input_tensor->shape.depth;

    const float* roiDataEnd = boxes_input + boxes_tensor_size;
    const float* deltas = deltas_input;
    float* outPtr = boxes_output;
    uint32_t roiIndex = 0;
    for (const float* roiBase = boxes_input; roiBase < roiDataEnd; roiBase += roiLength, roiIndex++) {

        uint32_t batchIndex = batch_splits[roiIndex];

        // Check for malformed data
        // 1. invalid batch id
        // 2. Invalid region: x2 <= x1 || y2 <= y1
        if (batchIndex >= numBatches) return errlog(nn,"batch index is not less than total batches");
        if (roiBase[0] > roiBase[2]) return errlog(nn,"malformed ROI: x1 is not less than x2");
        if (roiBase[1] > roiBase[3]) return errlog(nn,"malformed ROI: y1 is not less than y2");

        const float* imageInfoBase = image_info + batchIndex * imageLength;
        float imageHeight = imageInfoBase[0];
        float imageWidth = imageInfoBase[1];

        BoxEncodingCenter roiBefore;
        roiBefore.w = roiBase[2] - roiBase[0];
        roiBefore.h = roiBase[3] - roiBase[1];
        roiBefore.x = (roiBase[0] + roiBase[2]) / 2;
        roiBefore.y = (roiBase[1] + roiBase[3]) / 2;

        for (uint32_t i = 0; i < numClasses; i++) {

            BoxEncodingCenter roiAfterCentered;
            roiAfterCentered.w = expf(deltas[2]) * roiBefore.w;
            roiAfterCentered.h = expf(deltas[3]) * roiBefore.h;
            roiAfterCentered.x = roiBefore.x + deltas[0] * roiBefore.w;
            roiAfterCentered.y = roiBefore.y + deltas[1] * roiBefore.h;

            BoxEncodingCorner roiAfter;
            roiAfter.x1 = roiAfterCentered.x - roiAfterCentered.w / 2;
            roiAfter.y1 = roiAfterCentered.y - roiAfterCentered.h / 2;
            roiAfter.x2 = roiAfterCentered.x + roiAfterCentered.w / 2;
            roiAfter.y2 = roiAfterCentered.y + roiAfterCentered.h / 2;

            outPtr[0] = MIN(MAX(roiAfter.x1, 0.0f), imageWidth);
            outPtr[1] = MIN(MAX(roiAfter.y1, 0.0f), imageHeight);
            outPtr[2] = MIN(MAX(roiAfter.x2, 0.0f), imageWidth);
            outPtr[3] = MIN(MAX(roiAfter.y2, 0.0f), imageHeight);
            deltas += roiLength;
            outPtr += roiLength;
        }
    }

    return 0;
}

struct nn_node_ops nn_ops_for_AxisAlignedBBoxTransform_f = {
        .execute = axis_aligned_bbox_transform_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(4),
        .n_outputs = NN_IOCOUNT(1),
};

