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
#include <quantize.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(A,B) (A < B ? A : B)
#define ALIGN_SIZE 128
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))

struct score_order_t{
            int16_t order;
            uint8_t score;
};

#define DATA_PER_ROI 4
#define COORD_SCALE 0.125f
// #define COORD_OFFSET 0.0f
#define DEQUANT_COORD(COORD) (COORD*COORD_SCALE)
#define QUANT_COORD_POSITIVE(COORD)((int16_t)(COORD/COORD_SCALE+0.5f))
#define QUANT_COORD_NEGATIVE(COORD)((int16_t)(COORD/COORD_SCALE-0.5f))

int compare_score(const void* v1,const void* v2){
    uint32_t c1 = ((struct score_order_t*) v1)->score;
    uint32_t c2 = ((struct score_order_t*) v2)->score;
    if (c1 > c2) return -1;
    if (c2 > c1) return 1;
    return 0;
}


static inline float __attribute__((unused))expq(uint8_t q,float min, float max){
    float f = (max-min)/255.0f *q+min;
    return expf(f);
    // if(f > -1.0f && f < 1.0f) {
    //     return (1 + f + (f * f) / 2);
    // }
    // else {
    //     // 
    //     return expf(f);
    // }
}
static inline float __attribute__((unused))quantmul(float a, uint8_t q,float min, float max){
    float f = (max-min)/255.0f *q+min;
    return a*f;
}

static inline void box_flt_corner_from_center(float*dest_corner, const float* src_center){
    float x_center= src_center[0];
    float y_center = src_center[1];
    float width = src_center[2];
    float height = src_center[3];
    float* x_corner_1 = dest_corner+0;
    float* y_corner_1 = dest_corner+1;
    float* x_corner_2 = dest_corner+2;
    float* y_corner_2 = dest_corner+3;
    *x_corner_1 = x_center-width/2.0f;
    *y_corner_1 = y_center-height/2.0f;
    *x_corner_2 = x_center+width/2.0f;
    *y_corner_2 = y_center+height/2.0f;
}
static inline void box_flt_center_from_corner(float*dest_center, const float* src_corner){
    float x_corner_1= src_corner[0];
    float y_corner_1 = src_corner[1];
    float x_corner_2 = src_corner[2];
    float y_corner_2 = src_corner[3];
    float* x_center = dest_center+0;
    float* y_center = dest_center+1;
    float* width = dest_center+2;
    float* height = dest_center+3;

    *width = x_corner_2-x_corner_1;
    *height = y_corner_2 -y_corner_1;
    *x_center =(x_corner_2+x_corner_1)/2.0f;
    *y_center =(y_corner_2+y_corner_1)/2.0f;
}
static inline void box_flt_tranform(float * roi_predicted,const float *roi_base,const uint8_t* delta, const float delta_min, const float delta_max){
    uint8_t dx = delta[0];
    uint8_t dy = delta[1];
    uint8_t dw = delta[2];
    uint8_t dh = delta[3];
    float roi_base_x_ctr = roi_base[0];
    float roi_base_y_ctr = roi_base[1];
    float roi_base_width = roi_base[2];
    float roi_base_height = roi_base[3];
    float* roi_pred_x_ctr = roi_predicted;
    float* roi_pred_y_ctr = roi_predicted+1;
    float *roi_pred_width = roi_predicted+2;
    float *roi_pred_height = roi_predicted+3;

    *roi_pred_x_ctr = roi_base_x_ctr + quantmul(roi_base_width,dx,delta_min,delta_max);
    *roi_pred_y_ctr = roi_base_y_ctr + quantmul(roi_base_height,dy,delta_min,delta_max);
    *roi_pred_width = roi_base_width * expq(dw,delta_min,delta_max);
    *roi_pred_height = roi_base_height * expq(dh,delta_min,delta_max);
}
static inline void box_flt_clip(float* clipped_boxes_corner, const float* unclippled_boxes_center,
            float clip_begin_x, float clip_end_x, float clip_begin_y, float clip_end_y){
    box_flt_corner_from_center(clipped_boxes_corner,unclippled_boxes_center);
    clipped_boxes_corner[0] = fminf(fmaxf(clipped_boxes_corner[0],clip_begin_x),clip_end_x);
    clipped_boxes_corner[1] = fminf(fmaxf(clipped_boxes_corner[1],clip_begin_y),clip_end_y);
    clipped_boxes_corner[2] = fminf(fmaxf(clipped_boxes_corner[2],clip_begin_x),clip_end_x);
    clipped_boxes_corner[3] = fminf(fmaxf(clipped_boxes_corner[3],clip_begin_y),clip_end_y);
}
static inline __attribute__((unused)) void printcnr(const float* boxes,int32_t num){
    for(int i =0; i<num;i++ ){
        const float *b = &boxes[i*DATA_PER_ROI];
        printf("i:%d, x1:%f, y1:%f, x2:%f, y2:%f\n",i,b[0],b[1],b[2],b[3]);
    }
}
static inline void box_flt_filter_boxes_store_area(
    float* area,int32_t* suppressed,
    const float * boxes,const struct score_order_t* selection,
    const int32_t array_size, const float box_min_size){
    for (int i =0; i<array_size;i++){
        float tempbox_center[DATA_PER_ROI];
        int box_idx = selection[i].order;
        const float*box_corner=&boxes[DATA_PER_ROI*box_idx];
        box_flt_center_from_corner(tempbox_center,box_corner);
        float width = tempbox_center[2];
        float height = tempbox_center[3];
        area[i]=width*height;
        int32_t small = (width<box_min_size)||(height<box_min_size);
        if (small){
            suppressed[i] = 1;
        }
    }
}
static inline __attribute__((unused)) void printctr(float* boxes,int32_t num){
    for(int i =0; i<num;i++ ){
        float *b = &boxes[i*DATA_PER_ROI];
        printf("i:%d, xc:%f, yc:%f, w:%f, h:%f\n",i,b[0],b[1],b[2],b[3]);
    }
}
static inline void
flt_non_max_suppression(float* output_boxes,uint8_t* output_scores,
                    int32_t* num_output_boxes,
                    float *pred_boxes,
                    int32_t num_pred_boxes,
                    int32_t num_roi,
                    float threshold,
                    struct score_order_t *ordered_idx,
                    int32_t *suppressed,
                    float *bbox_area){

    for (int32_t idx_i = 0; idx_i < num_pred_boxes; ++idx_i){
        int32_t pred_idx = DATA_PER_ROI * ordered_idx[idx_i].order;

        if (suppressed[idx_i]){
            continue;
        }
        memcpy(output_boxes, pred_boxes + (pred_idx), DATA_PER_ROI * sizeof(float));
        *output_scores = ordered_idx[idx_i].score;
        (*num_output_boxes)++;
        if (*num_output_boxes >= num_roi) {
            return; // collected enough bboxes
        }
        output_boxes += DATA_PER_ROI; // depth dim: 4, i.e., [ x1, y1, x2, y2]
        output_scores++;

        float s1 = bbox_area[idx_i];
        float ix1 = pred_boxes[pred_idx];
        float iy1 = pred_boxes[pred_idx + 1];
        float ix2 = pred_boxes[pred_idx + 2];
        float iy2 = pred_boxes[pred_idx + 3];
        for (int32_t idx_j = idx_i + 1; idx_j < num_pred_boxes; ++idx_j){
            int32_t pred_idx2 = DATA_PER_ROI * ordered_idx[idx_j].order;

            if (suppressed[idx_j]){
                continue;
            }

            float s2 = bbox_area[idx_j];
            float x1 = fmaxf(ix1, pred_boxes[pred_idx2]);
            float y1 = fmaxf(iy1, pred_boxes[pred_idx2 + 1]);
            float x2 = fminf(ix2, pred_boxes[pred_idx2 + 2]);
            float y2 = fminf(iy2, pred_boxes[pred_idx2 + 3]);

            float width = fmaxf(x2 - x1, 0.0f);
            float height = fmaxf(y2 - y1, 0.0f);
            float areaIntersect = width * height;
            float areastruct = s2 +s1 - areaIntersect;
            float IOU = areaIntersect / areastruct;
            if (IOU >= threshold){
                suppressed[idx_j] = 1;
            }
        }
    }
}
#define SCORE_DATA_IDX 0
#define SCORE_MIN_IDX 1
#define SCORE_MAX_IDX 2

#define DELTAS_DATA_IDX 3
#define DELTAS_MIN_IDX 4
#define DELTAS_MAX_IDX 5

#define ANCHOR_DATA_IDX 6
#define ANCHOR_MIN_IDX 7
#define ANCHOR_MAX_IDX 8

#define IMSIZE_DATA_IDX 9
#define IMSIZE_MIN_IDX 10
#define IMSIZE_MAX_IDX 11
#define IMSTRIDE_H_IDX 12
#define IMSTRIDE_W_IDX 13

#define MAX_PROPOSAL_IDX 14
#define MAX_ROI_IDX 15

#define THRESHOLD_IDX 16
#define MIN_BBOX_SIDE_IDX 17

#define N_IN 18

#define OUTPUT_ROI_DATA_IDX 3
#define OUTPUT_ROI_MIN_IDX 4
#define OUTPUT_ROI_MAX_IDX 5

#define OUTPUT_SCORE_DATA_IDX 0
#define OUTPUT_SCORE_MIN_IDX 1
#define OUTPUT_SCORE_MAX_IDX 2

#define OUTPUT_NUM_IDX 6
#define N_OUT 7

static int proposal_execute_8(struct nn_node *self, struct nn_graph *nn){
    const struct tensor *score_tensor = self->inputs[SCORE_DATA_IDX];
    const struct tensor *deltas_tensor = self->inputs[DELTAS_DATA_IDX];
    const struct tensor *im_size_tensor = self->inputs[IMSIZE_DATA_IDX];

    const struct tensor *anchor_tensor = self->inputs[ANCHOR_DATA_IDX];
    const struct tensor *max_num_roi_tensor = self->inputs[MAX_ROI_IDX];
    const struct tensor *max_num_proposals_tensor = self->inputs[MAX_PROPOSAL_IDX];
    float threshold = tensor_get_float(self->inputs[THRESHOLD_IDX],0);
    float min_size = tensor_get_float(self->inputs[MIN_BBOX_SIDE_IDX],0);
    float height_stride = tensor_get_float(self->inputs[IMSTRIDE_H_IDX],0);
    float width_stride = tensor_get_float(self->inputs[IMSTRIDE_W_IDX],0);
    struct tensor *out_roi_tensor = self->outputs[OUTPUT_ROI_DATA_IDX];
    struct tensor *out_score_tensor = self->outputs[OUTPUT_SCORE_DATA_IDX];
    struct tensor *batch_split_tensor = self->outputs[OUTPUT_NUM_IDX];
    int32_t* batch_split = batch_split_tensor->data;

    uint16_t* sorted_roi_data = out_roi_tensor->data;
    uint8_t* sorted_score_data = out_score_tensor->data;
    uint32_t batches = score_tensor->shape.batches;

    uint32_t featuremap_height = score_tensor->shape.height;
    uint32_t featuremap_width = score_tensor->shape.width;
    uint32_t featuremap_depth = score_tensor->shape.depth;

    uint8_t *score = score_tensor->data;
    uint8_t *bbox_deltas = deltas_tensor->data;

    int32_t max_num_roi = tensor_get_int32(max_num_roi_tensor, 0);
    uint32_t anchor_num = anchor_tensor->shape.width;
    int32_t max_num_proposals = tensor_get_int32(max_num_proposals_tensor, 0);
    if (max_num_roi <1){
        return errlog(nn,"maximum rois must be greater than 0");
    }
    int32_t IMAGE_ROI_LEN=max_num_roi * DATA_PER_ROI;
    int32_t IMAGE_SCORE_LEN=max_num_roi;
    if(out_roi_tensor->max_size < max_num_roi * DATA_PER_ROI * sizeof(int16_t)){
        return errlog(nn,"roi out too small. actual=%d  expected=%d ", out_roi_tensor->max_size, max_num_roi * DATA_PER_ROI * sizeof(int16_t));
    }
    if(out_score_tensor->max_size < max_num_roi * sizeof(uint8_t)){
        return errlog(nn,"score out too small. actual=%d expected=%d ", out_score_tensor->max_size,max_num_roi * sizeof(uint8_t));
    }
    if(batch_split_tensor->max_size < max_num_roi * sizeof(int32_t)){
        return errlog(nn,"batch split out too small. actual=%d expected=%d ", batch_split_tensor->max_size,max_num_roi * sizeof(int32_t));
    }
    float score_min = tensor_get_float(self->inputs[SCORE_MIN_IDX],0);
    float score_max = tensor_get_float(self->inputs[SCORE_MAX_IDX],0);
    float delta_min = tensor_get_float(self->inputs[DELTAS_MIN_IDX],0);
    float delta_max = tensor_get_float(self->inputs[DELTAS_MAX_IDX],0);
    float roi_min = 0.f;
    float roi_max = 8191.875f;

    tensor_set_single_float(self->outputs[OUTPUT_ROI_MIN_IDX],roi_min);
    tensor_set_single_float(self->outputs[OUTPUT_ROI_MAX_IDX],roi_max);
    tensor_set_single_float(self->outputs[OUTPUT_SCORE_MIN_IDX],score_min);
    tensor_set_single_float(self->outputs[OUTPUT_SCORE_MAX_IDX],score_max);

    if (featuremap_depth != anchor_num){
        return errlog(nn,"number of channels in cls %d does not match the number of anchors in each cell %d .",featuremap_depth,anchor_num);
    }
    if (anchor_num * DATA_PER_ROI != deltas_tensor->shape.depth){
        return errlog(nn,"number of channels in bbox does not match the number of anchors in each cell.");
    }
    if(nn->loopstack.n == 0 || (nn->loopstack.n > 0 && nn->loopstack.entries[nn->loopstack.n-1].nodep != self)){
        nn_loopstack_push(nn, self, 1, batches);
    }
    uint32_t iteration = nn_loopstack_get_itercount(nn);

    score += iteration*featuremap_height*featuremap_width*anchor_num;
    bbox_deltas += iteration*DATA_PER_ROI*featuremap_height*featuremap_width*anchor_num;

    int32_t max_pre_nms = ((max_num_proposals >0)?max_num_proposals:anchor_num * featuremap_height * featuremap_width);
    size_t all_coords_size = anchor_num * featuremap_height * featuremap_width * DATA_PER_ROI * sizeof(float);
    size_t anchor_dequantized_size = anchor_num * DATA_PER_ROI * sizeof(float);
    size_t roi_base_size =all_coords_size;
    size_t proposals_size = all_coords_size;
    size_t scores_size = anchor_num * featuremap_height * featuremap_width * sizeof(int32_t);
    size_t score_order_size = anchor_num * featuremap_height * featuremap_width   * sizeof(struct score_order_t);
    size_t suppressed_size = max_pre_nms * sizeof(int32_t);
    size_t bbox_area_size = max_pre_nms * sizeof(float);
    size_t roi_temp_float_size = max_num_roi * DATA_PER_ROI * sizeof(float);
    size_t roi_split_int_size = IMAGE_ROI_LEN*sizeof(int16_t);
    size_t score_split_size = IMAGE_SCORE_LEN *sizeof(uint8_t);
    size_t total_size = (anchor_dequantized_size)+anchor_dequantized_size+all_coords_size*2 + scores_size + score_order_size
                        + suppressed_size + bbox_area_size + roi_temp_float_size+score_split_size+roi_split_int_size;

    if(nn_scratch_grow(nn,total_size)){
        return errlog(nn,"failed to get scratch");
    }
    nn_scratch_reset(nn);

    float* anchor_dequantized = nn_scratch_alloc(nn,anchor_dequantized_size);
    float*anchors_transformed = (float *) nn_scratch_alloc(nn,anchor_dequantized_size);
    float *roi_bases_ctr = (float *) nn_scratch_alloc(nn,roi_base_size);
    float *proposals = (float *) nn_scratch_alloc(nn,proposals_size);
    struct score_order_t *score_order_ptr = nn_scratch_alloc(nn,score_order_size);
    int32_t *suppressed = (int32_t*)nn_scratch_alloc(nn,suppressed_size);
    float *bbox_area = (float *) nn_scratch_alloc(nn,bbox_area_size);
    float *roi_temp_float = (float *) nn_scratch_alloc(nn,roi_temp_float_size);
    int16_t* roi_split_int = (int16_t*) nn_scratch_alloc(nn,roi_split_int_size);
    uint8_t* score_split = (uint8_t *)nn_scratch_alloc(nn,score_split_size);
    int32_t boxes_in_image=0;

    int16_t* anchor_ptr = anchor_tensor->data;
    for (int a =0; a< anchor_num*DATA_PER_ROI;a++ ){
        anchor_dequantized[a]=DEQUANT_COORD(*anchor_ptr);
        anchor_ptr++;
    }
    for (int a =0; a<anchor_num;a++){
        box_flt_center_from_corner(anchors_transformed+DATA_PER_ROI*a,anchor_dequantized+DATA_PER_ROI*a);
    }

    for (int h =0; h < featuremap_height;h++){
        float y_shift = h * height_stride;
        for (int w =0; w < featuremap_width;w++){
            int j = (h*featuremap_width+w)*anchor_num*DATA_PER_ROI;
            float x_shift = w * width_stride;
            float * roi_base_ctr_ptr = roi_bases_ctr+j;
            memcpy(roi_base_ctr_ptr,anchors_transformed,anchor_dequantized_size);
            for (int a=0; a<anchor_num; a++){
                roi_base_ctr_ptr[a*DATA_PER_ROI]+=x_shift;
                roi_base_ctr_ptr[a*DATA_PER_ROI+1]+=y_shift;
            }
        }
    }

    uint8_t *bbox_deltas_ptr = bbox_deltas;
    uint16_t* im_info_dat= im_size_tensor->data;
    uint8_t*score_split_ptr=score_split;
    int16_t* roi_split_int_ptr=roi_split_int;

    float img_height = DEQUANT_COORD(im_info_dat[2*iteration]);
    float img_width = DEQUANT_COORD(im_info_dat[2*iteration+1]);
    int32_t num_proposals_generated = 0;
    float *proposals_cnr_ptr = proposals;
    memset(proposals,0,proposals_size);
    memset(score_order_ptr,0,score_order_size);
    memset(roi_temp_float,0,roi_temp_float_size);
    memset(suppressed,0,suppressed_size);
    memset(roi_split_int,0,roi_split_int_size);

    struct score_order_t *score_order_ptr2 = score_order_ptr;
    float *roi_base_ctr_ptr = roi_bases_ctr;
    for (int32_t i = 0; i < featuremap_height*featuremap_width*anchor_num; ++i){
        float roi_temp_ctr[DATA_PER_ROI];
        box_flt_tranform(roi_temp_ctr, roi_base_ctr_ptr, bbox_deltas_ptr, delta_min, delta_max);
        box_flt_clip(proposals_cnr_ptr, roi_temp_ctr, 0, img_width, 0, img_height);

        score_order_ptr2->score = *score;
        score_order_ptr2->order = num_proposals_generated;
        num_proposals_generated++;

        roi_base_ctr_ptr += DATA_PER_ROI;
        bbox_deltas_ptr += DATA_PER_ROI;
        score_order_ptr2++;
        proposals_cnr_ptr+=DATA_PER_ROI;
        score++;
    }
    qsort(score_order_ptr,num_proposals_generated, sizeof(struct score_order_t), compare_score);

    //ignore boxes that are too far forward
    num_proposals_generated = max_num_proposals > 0? MIN(num_proposals_generated,max_num_proposals): num_proposals_generated;

    //suppress boxes that are too small
    box_flt_filter_boxes_store_area(bbox_area,suppressed,
        proposals,score_order_ptr,num_proposals_generated,min_size);

    //suppress boxes that overlap in descending order of scores
    flt_non_max_suppression(roi_temp_float, score_split_ptr,
                        &boxes_in_image,
                        proposals,
                        num_proposals_generated,
                        max_num_roi,
                        threshold,
                        score_order_ptr,
                        suppressed,
                        bbox_area);


    for (int i =0; i < boxes_in_image*DATA_PER_ROI; i++){
        roi_split_int_ptr[i] = QUANT_COORD_POSITIVE(roi_temp_float[i]);
    }

    if(tensor_out_prepare_normal(out_roi_tensor, 1,1,boxes_in_image,DATA_PER_ROI,NN_TYPE_QINT16)){
        return errlog(nn, "ohno coulndn't prepare rois");
    }
    if(tensor_out_prepare_normal(out_score_tensor, 1,1,1,boxes_in_image,NN_TYPE_QUINT8)){
        return errlog(nn, "ohno coulndn't prepare scores");
    }
    if(tensor_out_prepare_normal(batch_split_tensor, 1,1,1,boxes_in_image,NN_TYPE_INT32)){
        return errlog(nn, "ohno coulndn't prepare batch_split");
    }
    //alternative
    //     tensor_set_shape(out_roi_tensor,batches,1,max_num_roi,DATA_PER_ROI);
    // tensor_set_shape(out_score_tensor,batches,1,1,max_num_roi);
    // atch_split_tensor->data_size=batches*max_num_roi*sizeof(int32_t);
    // out_roi_tensor->data_size = max_num_roi*DATA_PER_ROI*sizeof(int16_t);
    // out_score_tensor->data_size = batches*max_num_roi*sizeof(uint8_t);

    int32_t roi_len=boxes_in_image*DATA_PER_ROI;
    int32_t score_size=boxes_in_image*sizeof(uint8_t);
    memcpy(sorted_roi_data,roi_split_int,roi_len*sizeof(int16_t));
    memcpy(sorted_score_data,score_split,score_size);
    for(int i =0; i< boxes_in_image;i++){
        batch_split[i]=iteration;
    }

    nn_scratch_reset(nn);

    if(iteration == batches - 1)
        nn_loopstack_set_next_batches(nn, 0);
    else
        nn_loopstack_set_next_batches(nn, 1);

    return 0;
}


struct nn_node_ops nn_ops_for_Proposal_q8q16 = {
    .execute = proposal_execute_8,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(N_IN),
    .n_outputs = NN_IOCOUNT(N_OUT),
    .flags = NN_NODE_FLAG_CLS_LOOP_CONTROL_NODE | NN_NODE_FLAG_CLS_DYNAMIC_TENSOR,
};
