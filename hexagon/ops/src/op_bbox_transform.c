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

#define MIN(A,B) (A < B ? A : B)
#define ALIGN_SIZE 128
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))

struct confidence_order_t{
    float confidence;
    int order;
};
#define TRANSFORMATION_LIMIT_LOG 4.1351666
#define TRANSFORMATION_LIMIT_VAL 62.5

#define DATA_PER_ROI 5
#define DATA_PER_BOX 4

#define ROI_IDX 0
#define DELTA_IDX 1
#define IMINFO_IDX 2
#define WEIGHTS_IDX 3
#define APPLY_SCALE_IDX 4
#define CORRECT_COORDS_IDX 5

#define OUTPUT_BOX_IDX 0
#define OUTPUT_NUM_IDX 1
static int bbox_transform_execute_f(struct nn_node *self, struct nn_graph *nn){
    const struct tensor *roi_tensor = self->inputs[ROI_IDX];
    const struct tensor *deltas_tensor = self->inputs[DELTA_IDX];
    const struct tensor *im_info_tensor = self->inputs[IMINFO_IDX];
    const struct tensor *weights_tensor = self->inputs[WEIGHTS_IDX];

    struct tensor *out_box_tensor = self->outputs[OUTPUT_BOX_IDX];
    struct tensor *out_num_tensor = NULL;

    const float* rois = roi_tensor->data; // in the format [[x1,y1,x2,y2]...] for each roi
    const float* deltas = deltas_tensor->data; // transformation (translation and scaling) for each roi and classification
    const float* weights = weights_tensor->data;
    float* boxes = out_box_tensor->data;

    //transform offset for the x2,y2 coordinates of the transformed boxes
    float transform_offset = 0.0f;
    int32_t should_correct_coords = tensor_get_int32(self->inputs[CORRECT_COORDS_IDX], 0);
    if (should_correct_coords) transform_offset = 1.0f;
    
    //if 0, new coordinates not scaled to fit original image
    int32_t should_apply_scale = tensor_get_int32(self->inputs[APPLY_SCALE_IDX], 0);

    int32_t max_num_roi = roi_tensor->shape.width;
    int32_t roi_depth = roi_tensor->shape.depth; //4 or 5
    int32_t batches = im_info_tensor->shape.batches;
    int32_t num_classes = deltas_tensor->shape.depth / DATA_PER_BOX;

    int32_t out_data_size = max_num_roi * num_classes * DATA_PER_BOX * sizeof(float);
    if(out_box_tensor->max_size < out_data_size ){
        return errlog(nn,"box out too small. actual=%d  expected=%d ", out_box_tensor->max_size,out_data_size);
    }

    tensor_set_shape(out_box_tensor,1,1,max_num_roi,num_classes*DATA_PER_BOX);
    out_box_tensor->data_size = out_data_size;

    if(self->n_outputs == 2){
        out_num_tensor = self->outputs[OUTPUT_NUM_IDX];
        if(out_num_tensor->max_size <  batches * sizeof(float)){
            return errlog(nn,"2 outputs given but last tensor does not have enough space for a %d floats",batches);
        }
        tensor_set_shape(out_num_tensor,batches,1,1,1);
        out_num_tensor->data_size=sizeof(float)*batches;
    }

    float img_scale_before = tensor_get_float(im_info_tensor,2);
    if (img_scale_before <= 0){
        return errlog(nn,"expected img scale > 0, found %f", img_scale_before);
    }
    float img_scale_after = should_apply_scale ? img_scale_before : 1.0f;
    float img_height = roundf(tensor_get_float(im_info_tensor, 0) / img_scale_before);
    float img_width = roundf(tensor_get_float(im_info_tensor, 1) / img_scale_before);
    float scaled_rois[DATA_PER_BOX];
    int32_t roi_depth_offset = roi_depth - DATA_PER_BOX; // 0 or 1
    int32_t roi_count=0;

    //for each input roi, transform (translate and scale) it based on the deltas
    for (int32_t i = 0; i < max_num_roi; ++i){
        if (rois[1]==-1.0f) {
        // op_proposal gives us boxes that are filled with -1 when invalid. 
        // fill corresponding output boxes with -1
            for (int k =0; k<num_classes*DATA_PER_BOX;k++){
                boxes[k]=-1.0f;
            }
            boxes += DATA_PER_BOX*num_classes;
            rois += roi_depth;
            deltas += DATA_PER_BOX*num_classes;

            continue;
        }
        roi_count++;
        for (int j = 0; j < DATA_PER_BOX; j++){
            //rois are scaled to a different height
            scaled_rois[j] = rois[j+roi_depth_offset]/img_scale_before;
        }
        // turing a roi from ((x1,y1),(x2,y2)) format into (x-center,y-center,height,width) format
        float w = scaled_rois[2] - scaled_rois[0] + 1.0f;
        float h = scaled_rois[3] - scaled_rois[1] + 1.0f;
        float x_ctr = scaled_rois[0] + 0.5f * w;
        float y_ctr = scaled_rois[1] + 0.5f * h;
        //for each roi, there is a different delta predicted by the network for each classification
        //each roi is transformed by each of these dletas
        for ( int k = 0; k< num_classes; k++){
            float dx = deltas[0]/weights[0];
            float dy = deltas[1]/weights[1];
            float dw = deltas[2]/weights[2];
            float dh = deltas[3]/weights[3];

            //translating the center
            float pred_ctr_x = x_ctr + w * dx;
            float pred_ctr_y = y_ctr + h * dy;

            //scaling the dimensions. scaling deltas provided in log format
            float pred_w = 0;
            float pred_h = 0;
            if(dw > -1.0f && dw < 1.0f) {
                //optimiztion: Quadratic approximation of e^x works well for small values of x
                pred_w = w * (1 + dw + (dw * dw) / 2);
            }
            else if( dw > TRANSFORMATION_LIMIT_LOG){
                pred_w = w*TRANSFORMATION_LIMIT_VAL;
            }
            else {
                pred_w = w * expf(dw);
            }
            if(dh > -1.0f && dw < 1.0f) {
                pred_h = h * (1 + dh + (dh * dh) / 2);
            }
            else if(dh > TRANSFORMATION_LIMIT_LOG){
                pred_h = h*TRANSFORMATION_LIMIT_VAL;
            }
            else {
                pred_h = h * expf(dh);
            }

            // transform (x-center,y-center,height,width) back to (x1,y1),(x2,y2) and clip it into the original image boundaries
            boxes[0] = fmaxf(fminf(pred_ctr_x - 0.5f * pred_w, img_width - 1.0f), 0.0f)*img_scale_after;
            boxes[1] = fmaxf(fminf(pred_ctr_y - 0.5f * pred_h, img_height - 1.0f), 0.0f)*img_scale_after;
            boxes[2] = fmaxf(fminf(pred_ctr_x + 0.5f * pred_w - transform_offset, img_width - 1.0f), 0.0f)*img_scale_after;
            boxes[3] = fmaxf(fminf(pred_ctr_y + 0.5f * pred_h - transform_offset, img_height - 1.0f), 0.0f)*img_scale_after;
            //pointers to next box
            deltas += DATA_PER_BOX;
            boxes += DATA_PER_BOX;
        }
        rois += roi_depth;
    }

    if(out_num_tensor){
        tensor_set_float(out_num_tensor,0,(float)roi_count);
    }
    return 0;
}


// must have 6 inputs, and 1 or 2 outputs

struct nn_node_ops nn_ops_for_Bbox_Transform_f = {
    .execute = bbox_transform_execute_f,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(6),
    .n_outputs = NN_IOCOUNT_RANGE(1,2),

};
