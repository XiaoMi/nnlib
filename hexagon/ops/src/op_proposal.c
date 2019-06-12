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

#define MIN(A,B) (A < B ? A : B)
#define ALIGN_SIZE 128
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))

struct confidence_order_t{
    float confidence;
    int order;
};

#define DATA_PER_ROI 5
int compare_confidence(const void* v1,const void* v2){
    float c1 = ((struct confidence_order_t*) v1)->confidence;
    float c2 = ((struct confidence_order_t*) v2)->confidence;
    float diff = c1 - c2;
    if (diff > 0) return -1;
    if (diff < 0) return 1;
    return 0;
}

static inline void
non_max_suppression(float* output_boxes,float* output_scores,
                    int32_t* num_output_boxes,
                    float *pred_boxes,
                    int32_t num_pred_boxes,
                    int32_t num_roi,
                    float threshold,
                    struct confidence_order_t *ordered_idx,
                    uint8_t *suppressed,
                    float *bbox_area){
    for (int32_t idx_i = 0; idx_i < num_pred_boxes; ++idx_i){
        int32_t pred_idx = 4 * ordered_idx[idx_i].order;
        bbox_area[idx_i] = (pred_boxes[pred_idx + 2] - pred_boxes[pred_idx] + 1.0f) *
                            (pred_boxes[pred_idx + 3] - pred_boxes[pred_idx + 1] + 1.0f);
    }

    for (int32_t idx_i = 0; idx_i < num_pred_boxes; ++idx_i){
        int32_t pred_idx = 4 * ordered_idx[idx_i].order;

        if (suppressed[idx_i]){
            continue;
        }
        *output_boxes=0;//batch id set to 0 to prevent garbage values;
        memcpy(output_boxes+1, pred_boxes + (pred_idx), 4 * sizeof(float));
        *output_scores = ordered_idx[idx_i].confidence;
        (*num_output_boxes)++;
        if (*num_output_boxes >= num_roi) {
            return; // collected enough bboxes
        }
        output_boxes += DATA_PER_ROI; // depth dim: 5, i.e., [batch id (dont care), x1, y1, x2, y2]
        output_scores++;

        float s1 = bbox_area[idx_i];
        float ix1 = pred_boxes[pred_idx];
        float iy1 = pred_boxes[pred_idx + 1];
        float ix2 = pred_boxes[pred_idx + 2];
        float iy2 = pred_boxes[pred_idx + 3];
        for (int32_t idx_j = idx_i + 1; idx_j < num_pred_boxes; ++idx_j){
            int32_t pred_idx2 = 4 * ordered_idx[idx_j].order;

            if (suppressed[idx_j]){
                continue;
            }

            float s2 = bbox_area[idx_j];

            float x1 = fmaxf(ix1, pred_boxes[pred_idx2]);
            float y1 = fmaxf(iy1, pred_boxes[pred_idx2 + 1]);
            float x2 = fminf(ix2, pred_boxes[pred_idx2 + 2]);
            float y2 = fminf(iy2, pred_boxes[pred_idx2 + 3]);

            float width = fmaxf(x2 - x1 + 1.0f, 0.0f);
            float height = fmaxf(y2 - y1 + 1.0f, 0.0f);
            float inter = width * height;
            float IOU = inter / (s1 + s2 - inter);
            if (IOU >= threshold){
                suppressed[idx_j] = 1;
            }
        }
    }
}

#define CLS_IDX 0
#define BBOX_IDX 1
#define IMINFO_IDX 2
#define ANCHOR_IDX 3
#define SRTIDE_IDX 4
#define MAX_ROI_IDX 5
#define MAX_PROPOSAL_IDX 6
#define THRESHOLD_IDX 7
#define MIN_BBOX_IDX 8
#define CORRECT_COORDS_IDX 9

#define OUTPUT_ROI_IDX 0
#define OUTPUT_PROB_IDX 1
#define OUTPUT_NUM_IDX 2
static int proposal_execute_f(struct nn_node *self, struct nn_graph *nn){
    const struct tensor *cls_score_tensor = self->inputs[CLS_IDX];
    const struct tensor *bbox_deltas_tensor = self->inputs[BBOX_IDX];
    const struct tensor *im_info_tensor = self->inputs[IMINFO_IDX];
    const struct tensor *anchor_tensor = self->inputs[ANCHOR_IDX];
    const struct tensor *stride_tensor = self->inputs[SRTIDE_IDX];
    const struct tensor *max_num_roi_tensor = self->inputs[MAX_ROI_IDX];
    const struct tensor *max_num_proposals_tensor = self->inputs[MAX_PROPOSAL_IDX];
    const struct tensor *threshold_tensor = self->inputs[THRESHOLD_IDX];
    const struct tensor *min_bbox_size_tensor = self->inputs[MIN_BBOX_IDX];

    struct tensor *out_roi_tensor = self->outputs[OUTPUT_ROI_IDX];
    struct tensor *out_prob_tensor = self->outputs[OUTPUT_PROB_IDX];
    struct tensor *out_num_tensor = NULL;

    float* sorted_roi_data = out_roi_tensor->data;
    float* sorted_prob_data = out_prob_tensor->data;

    float transform_offset = 0.0f;
    if(self->n_inputs == 10){
        int32_t should_correct_coords = tensor_get_int32(self->inputs[CORRECT_COORDS_IDX],0);
        if (should_correct_coords) transform_offset = 1.0f;
    }

    uint32_t cls_h = cls_score_tensor->shape.height;
    uint32_t cls_w = cls_score_tensor->shape.width;
    uint32_t cls_d = cls_score_tensor->shape.depth;

    float *cls_score = cls_score_tensor->data;
    float *bbox_deltas = bbox_deltas_tensor->data;

    float feature_stride = tensor_get_float(stride_tensor, 0);
    int32_t max_num_roi = tensor_get_int32(max_num_roi_tensor, 0);
    uint32_t anchor_num = anchor_tensor->shape.width;
    int32_t max_num_proposals = tensor_get_int32(max_num_proposals_tensor, 0);
    float nms_iou_threshold = tensor_get_float(threshold_tensor, 0);
    float minimum_bbox_size = tensor_get_float(min_bbox_size_tensor, 0);

    if(out_roi_tensor->max_size < max_num_roi * DATA_PER_ROI * sizeof(float)){
        return errlog(nn,"roi out too small. actual=%d  expected=%d ", out_roi_tensor->max_size, max_num_roi * DATA_PER_ROI * sizeof(float));
    }
    if(out_prob_tensor->max_size < max_num_roi * sizeof(float)){
        return errlog(nn,"prob out too small. actual=%d expected=%d ", out_prob_tensor->max_size,max_num_roi * sizeof(float));
    }

    tensor_set_shape(out_roi_tensor,1,1,max_num_roi,DATA_PER_ROI);
    tensor_set_shape(out_prob_tensor,1,1,1,max_num_roi);

    if(self->n_outputs == 3){
        out_num_tensor = self->outputs[OUTPUT_NUM_IDX];
        if(out_num_tensor->max_size <  sizeof(float)){
            return errlog(nn,"3 outputs given but last tensor does not have enough space for a float");
        }
        tensor_set_shape(out_num_tensor,1,1,1,1);
        out_num_tensor->data_size=sizeof(float);
    }

    if (anchor_num != cls_d && anchor_num * 2 != cls_d){
        return errlog(nn,"number of channels in cls does not match the number of anchors in each cell.");
    }
    if (anchor_num * 4 != bbox_deltas_tensor->shape.depth){
        return errlog(nn,"number of channels in bbox does not match the number of anchors in each cell.");
    }

    size_t proposals_size = anchor_num * cls_h * cls_w * 4 * sizeof(float);
    size_t scores_size = anchor_num * cls_h * cls_w * sizeof(float);
    size_t confidence_order_size = anchor_num * cls_h * cls_w   * sizeof(struct confidence_order_t);
    size_t suppressed_size = max_num_proposals * sizeof(uint8_t);
    size_t bbox_area_size = max_num_proposals * sizeof(float);
    size_t total_size = proposals_size + scores_size + confidence_order_size + suppressed_size + bbox_area_size;

    if(nn_scratch_grow(nn,total_size)){
        return errlog(nn,"failed to get scratch");
    }

    float *proposals = nn->scratch ;
    float *scores = (float *) (proposals+ proposals_size);
    struct confidence_order_t *confidence_order_ptr = (struct confidence_order_t *)(scores+ scores_size);
    uint8_t *suppressed = (uint8_t *)(confidence_order_ptr+ confidence_order_size);
    float *bbox_area = (float *) (suppressed+suppressed_size);

    float img_height = tensor_get_float(im_info_tensor,0);
    float img_width = tensor_get_float(im_info_tensor,1);
    float img_scale_percent = tensor_get_float(im_info_tensor,2);
    float scaled_min_size = minimum_bbox_size * img_scale_percent / 100.0f;

    float *bbox_deltas_ptr = bbox_deltas;
    float *proposals_ptr = proposals;
    float *scores_ptr = scores;
    int32_t is_py_faster_rcnn = ((2*anchor_num) == (cls_d)) ? 1 : 0;
    const float* cls_score_foreground = cls_score + (is_py_faster_rcnn ? anchor_num : 0);
    int32_t num_proposals = 0; // keep track the number of proposals collected
    for (int32_t i = 0; i < cls_h; ++i){
        float y_offset = i * feature_stride;
        for (int32_t j = 0; j < cls_w; ++j){
            float x_offset = j * feature_stride;
            float* anchor_ptr = anchor_tensor->data;
            for (int32_t k = 0; k < anchor_num; ++k){

                // bbox_transform_inv
                float w = anchor_ptr[2] - anchor_ptr[0] + 1.0f;
                float h = anchor_ptr[3] - anchor_ptr[1] + 1.0f;
                float x_ctr = anchor_ptr[0] + 0.5f * w;
                float y_ctr = anchor_ptr[1] + 0.5f * h;

                float dx = bbox_deltas_ptr[0];
                float dy = bbox_deltas_ptr[1];
                float dw = bbox_deltas_ptr[2];
                float dh = bbox_deltas_ptr[3];

                float y_shift = y_offset + y_ctr;
                float x_shift = x_offset + x_ctr;

                float pred_ctr_x = x_shift + w * dx;
                float pred_ctr_y = y_shift + h * dy;
                float pred_w = 0;
                float pred_h = 0;

                //Quadratic approximation of e^x works well for small values of x
                if(dw > -1.0f && dw < 1.0f) {
                    pred_w = w * (1 + dw + (dw * dw) / 2);
                }
                else {
                    pred_w = w * expf(dw);
                }
                if(dh > -1.0f && dw < 1.0f) {
                    pred_h = h * (1 + dh + (dh * dh) / 2);
                }
                else {
                    pred_h = h * expf(dh);
                }

                // clip predicted boxes to image
                proposals_ptr[0] = fmaxf(fminf(pred_ctr_x - 0.5f * pred_w, img_width - 1.0f), 0.0f);
                proposals_ptr[1] = fmaxf(fminf(pred_ctr_y - 0.5f * pred_h, img_height - 1.0f), 0.0f);
                proposals_ptr[2] = fmaxf(fminf(pred_ctr_x + 0.5f * pred_w - transform_offset, img_width - 1.0f), 0.0f);
                proposals_ptr[3] = fmaxf(fminf(pred_ctr_y + 0.5f * pred_h - transform_offset, img_height - 1.0f), 0.0f);

                // ignore the predicted bboxes with height or width smaller than scaled_min_size
                float ws = proposals_ptr[2] - proposals_ptr[0] + 1.0f;
                float hs = proposals_ptr[3] - proposals_ptr[1] + 1.0f;
                if (ws >= scaled_min_size && hs >= scaled_min_size){
                    *scores_ptr++ = cls_score_foreground[k];
                    proposals_ptr += 4;
                    num_proposals++;
                }

                // shift ptr to process next anchor
                anchor_ptr += 4;
                bbox_deltas_ptr += 4;
            }
            cls_score_foreground += cls_d; // depth is equal to 2 * anchor_num for py-faster-rcnn and anchor-num for caffe2 frcnn
        }
    }

    // sort the confidence from high to low and keep track of ordered index
    for (int o = 0; o < num_proposals; o++){
        confidence_order_ptr[o].confidence = scores[o];
        confidence_order_ptr[o].order = o;
    }
    l2fetch(confidence_order_ptr,1,num_proposals * sizeof(struct confidence_order_t),1);
    qsort(confidence_order_ptr,num_proposals, sizeof(struct confidence_order_t), compare_confidence);
    int32_t num_output_boxes = 0;
    memset(suppressed,0,suppressed_size);
    non_max_suppression(sorted_roi_data, sorted_prob_data,
                        &num_output_boxes,
                        proposals,
                        MIN(max_num_proposals, num_proposals),
                        max_num_roi,
                        nms_iou_threshold,
                        confidence_order_ptr,
                        suppressed,
                        bbox_area);

    //setting remaining spaces to 0;
    for (int i = num_output_boxes; i < max_num_roi; i ++){
        int j = 5*i;
        sorted_roi_data[j] = 0.0;
        sorted_prob_data[i] = 0.0;
        sorted_roi_data[j+1] = sorted_roi_data[j+2] = sorted_roi_data[j+3] = sorted_roi_data[j+4] = -1.0;
    }

    sorted_roi_data[0]  = (is_py_faster_rcnn) ? (float)(num_output_boxes) : sorted_roi_data[0];
    out_roi_tensor->data_size = max_num_roi*DATA_PER_ROI*sizeof(float);
    out_prob_tensor->data_size = max_num_roi*sizeof(float);
    if(out_num_tensor){
        tensor_set_float(out_num_tensor,0,(float)num_output_boxes);
    }
    return 0;
}


struct nn_node_ops nn_ops_for_Proposal_f = {
    .execute = proposal_execute_f,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT_RANGE(9,10),
    .n_outputs = NN_IOCOUNT_RANGE(2,3),
};
