
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
#include <stdlib.h>
#ifndef __hexagon__
#include <malloc.h>

#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a box with nms limit op.
 */

#define DATA_PER_ROI 4

typedef struct{
    uint16_t* coordinates;
	int class_id;
	int status; // 0 = not processed, -1 = rejected, >0 = the order it was selected in
	float score;
} nms_box;

typedef struct{
	nms_box* start;
	uint32_t size;
} nms_box_list;

static int load_boxes_scores(uint16_t * boxes, uint8_t* scores, float scores_min, float scores_max, int total_boxes, int total_classes, float score_threshold, nms_box_list* boxes_list_out, nms_box* boxes_out_base){
	int total = 0;
    float step = (scores_max-scores_min)/255.f;

	nms_box* boxes_out_i;

	boxes_out_i = boxes_out_base;

	for(int i = 1; i < total_classes; i++){
		int n = 0;
		for(int j = 0; j < total_boxes; j++){
			float score_float = scores_min + scores[i+j*total_classes] * step;
			if (score_float <= score_threshold){
				continue;
			}
            boxes_out_i[n].coordinates = boxes + DATA_PER_ROI*i + j*DATA_PER_ROI*total_classes;
			boxes_out_i[n].class_id = i;
			boxes_out_i[n].status = 0;
			boxes_out_i[n].score = score_float;
			n++;
		}
		boxes_list_out[i-1].start = boxes_out_i;
		boxes_list_out[i-1].size = n;
		boxes_out_i+=n;
        total+=n;
	}

    return total;
}

float get_area(nms_box *b){
	return (b->coordinates[2] - b->coordinates[0]) * (b->coordinates[3] - b->coordinates[1]);
}

float get_intersection_area_i16(nms_box *r1, nms_box *r2){

	if(r1->coordinates[2] < r2->coordinates[0] ||
	   r1->coordinates[0] > r2->coordinates[2] ||
	   r1->coordinates[3] < r2->coordinates[1] ||
	   r1->coordinates[1] > r2->coordinates[3]){
		return 0.f;
	}

	uint16_t xx1 = r1->coordinates[0] > r2->coordinates[0] ? r1->coordinates[0] : r2->coordinates[0];
	uint16_t yy1 = r1->coordinates[1] > r2->coordinates[1] ? r1->coordinates[1] : r2->coordinates[1];
	uint16_t xx2 = r1->coordinates[2] > r2->coordinates[2] ? r2->coordinates[2] : r1->coordinates[2];
	uint16_t yy2 = r1->coordinates[3] > r2->coordinates[3] ? r2->coordinates[3] : r1->coordinates[3];

	int w = xx2 - xx1;
	int h = yy2 - yy1;

	w = w < 0 ? 0 : w;
	h = h < 0 ? 0 : h;

	return (float)(w * h);
}

static int get_max_coefficient_u8(nms_box_list* box_list){
	int result = -1;
	float max_score = 0;
	nms_box *boxes = box_list->start;

	for(int i = 0; i < box_list->size; i++){
		if(boxes[i].status == 0 && boxes[i].score > max_score){
			result = i;
			max_score = boxes[i].score;
		}
	}

	return result;
}

static int soft_nms_u16(nms_box_list* box_list,
	float overlap_thresh, float score_thresh, float sigma, int method, int max_selection){

	if(box_list->size <= 0){
		return 0;
	}

	nms_box* boxes = box_list->start;
	int num_selected = 0;
	int candidates_remaining = box_list->size;

	while(candidates_remaining > 0 && num_selected < max_selection){
		int current_candidate_index = get_max_coefficient_u8(box_list);
		
		if(current_candidate_index == -1){
			break;
		}

		num_selected++;
		boxes[current_candidate_index].status = num_selected;
		
		for(int i = 0; i < box_list->size; i++){

			if(boxes[i].status != 0){
				continue;
			}

            float area_i = get_area(boxes+i);
            float area_c = get_area(boxes+current_candidate_index);
			float inter = get_intersection_area_i16(boxes+i, boxes+current_candidate_index);
            float ovr = inter / (area_c + area_i - inter);
			float weight;
			switch (method) {
				case 1: // Linear
					weight = (ovr < overlap_thresh) ? 1.0f : (1.0f - ovr);
					break;
				case 2: // Gaussian
					weight = expf(-1.0f * ovr * ovr / sigma);
					break;
				default: // Original NMS
					weight = (ovr < overlap_thresh) ? 1.0f : 0.0f;
			}

			boxes[i].score *= weight;
			
			if(boxes[i].score < score_thresh){
				boxes[i].status = -1;
				candidates_remaining--;
			}
		}
		
		candidates_remaining--;
	}

	return num_selected;
}

int compare_box_scores(const void *a, const void *b){
	nms_box box_a = *(nms_box*)a;
	nms_box box_b = *(nms_box*)b;

	if(box_a.status < 1 && box_b.status < 1)
		return 0;
	else if(box_a.status < 1)
		return 1;
	else if(box_b.status < 1)
		return -1;
	else
		return box_b.score - box_a.score > 0 ? 1 : -1;
}

int compare_box_classes(const void *a, const void *b){
    nms_box box_a = *(nms_box*)a;
    nms_box box_b = *(nms_box*)b;

    return box_b.class_id == box_a.class_id ? (box_b.score - box_a.score > 0 ? 1 : -1) : box_a.class_id - box_b.class_id;
}

static int box_with_nms_limit_q8q16_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"box_with_nms_limit_execute execute. self=%p ",self);
	
	const struct tensor *scores_input_tensor       = self->inputs[0];
	const struct tensor *boxes_input_tensor        = self->inputs[1];
    const struct tensor *batch_info_input_tensor   = self->inputs[2];

	struct tensor *scores_output_tensor       = self->outputs[0];
	struct tensor *boxes_output_tensor        = self->outputs[1];
	struct tensor *classes_output_tensor      = self->outputs[2];
	struct tensor *batch_info_output_tensor   = self->outputs[3];
	
	uint8_t *scores_input = scores_input_tensor->data;
	uint16_t *boxes_input = boxes_input_tensor->data;
	int *batch_info_data  = batch_info_input_tensor->data;

	float score_threshold              = tensor_get_float(self->inputs[3], 0);
	int detections_per_image           = tensor_get_int32(self->inputs[4], 0);
	unsigned int soft_nms_method       = tensor_get_int32(self->inputs[5], 0);
	float nms_threshold                = tensor_get_float(self->inputs[6], 0);
	float soft_nms_sigma               = tensor_get_float(self->inputs[7], 0);
	float soft_nms_min_score_threshold = tensor_get_float(self->inputs[8], 0);
    float scores_min                   = tensor_get_float(self->inputs[9], 0);
    float scores_max                   = tensor_get_float(self->inputs[10], 0);
    float boxes_min                    = tensor_get_float(self->inputs[11], 0);
    float boxes_max                    = tensor_get_float(self->inputs[12], 0);

	uint8_t *scores_output     = scores_output_tensor->data;
	uint16_t *boxes_output     = boxes_output_tensor->data;
	int32_t *classes_output    = classes_output_tensor->data;
	int32_t *batch_info_output = batch_info_output_tensor->data;

	int total_boxes = scores_input_tensor->shape.width;
	int num_classes = scores_input_tensor->shape.depth;
	float scores_step = (scores_max - scores_min) / 255.f;

    if (scores_input_tensor->shape.batches != 1 || scores_input_tensor->shape.height != 1) return errlog(nn,"scores_input_tensor improperly formed");
    if (boxes_input_tensor->shape.batches != 1 || boxes_input_tensor->shape.height != 1) return errlog(nn,"boxes_input_tensor improperly formed");
	if (batch_info_input_tensor->shape.batches != 1 || batch_info_input_tensor->shape.height != 1 || batch_info_input_tensor->shape.width != 1) return errlog(nn,"batch_info_input_tensor improperly formed");
    if (scores_input_tensor->shape.width != boxes_input_tensor->shape.width || boxes_input_tensor->shape.width != batch_info_input_tensor->shape.depth) return errlog(nn,"number of boxes inconsistent between tensors");
    if (scores_input_tensor->shape.depth * DATA_PER_ROI != boxes_input_tensor->shape.depth) return errlog(nn,"boxes_input_tensor depth is not equal to 4 x number of classes");
    if (detections_per_image < 1) return errlog(nn,"detections per image must be greater than 0");

	tensor_set_single_float(self->outputs[4],scores_min);
	tensor_set_single_float(self->outputs[5],scores_max);
	tensor_set_single_float(self->outputs[6],boxes_min);
	tensor_set_single_float(self->outputs[7],boxes_max);


	nms_box_list* box_lists = nn_malloc(num_classes*sizeof(nms_box_list));

	int boxes_processed = 0, total_keep = 0, keep_idx = 0;
	while(boxes_processed < total_boxes){

		int batch_size = 0, keep_from_batch = 0;
		while (batch_info_data[batch_size] == *batch_info_data && batch_size < total_boxes) batch_size++;

		nms_box* boxes_base = nn_malloc(batch_size*num_classes*sizeof(nms_box));
		int total_loaded = load_boxes_scores(boxes_input+boxes_processed*DATA_PER_ROI*num_classes, scores_input+boxes_processed*num_classes, scores_min, scores_max, batch_size, num_classes, score_threshold, box_lists, boxes_base);

		for (int i = 1; i < num_classes; i++){
			keep_from_batch += soft_nms_u16(box_lists+i-1, nms_threshold, soft_nms_min_score_threshold, soft_nms_sigma, soft_nms_method, detections_per_image);
		}

		qsort(boxes_base, total_loaded, sizeof(nms_box), compare_box_scores);

		if (keep_from_batch > detections_per_image) {
			keep_from_batch = detections_per_image;
		}

        qsort(boxes_base, keep_from_batch, sizeof(nms_box), compare_box_classes);

        for(int i = 0; i < keep_from_batch; i++){
			scores_output[keep_idx] = round((boxes_base[i].score - scores_min) / scores_step);
			boxes_output[keep_idx*DATA_PER_ROI]   = boxes_base[i].coordinates[0];
			boxes_output[keep_idx*DATA_PER_ROI+1] = boxes_base[i].coordinates[1];
			boxes_output[keep_idx*DATA_PER_ROI+2] = boxes_base[i].coordinates[2];
			boxes_output[keep_idx*DATA_PER_ROI+3] = boxes_base[i].coordinates[3];
			classes_output[keep_idx] = boxes_base[i].class_id;
			batch_info_output[keep_idx] = *batch_info_data;
			keep_idx++;
		}

		batch_info_data += batch_size;
		boxes_processed += batch_size;
		total_keep += keep_from_batch;

		nn_free(boxes_base);
	}

	nn_free(box_lists);

	tensor_out_prepare_normal(scores_output_tensor, 		1,1,1,total_keep, NN_TYPE_QUINT8);
	tensor_out_prepare_normal(boxes_output_tensor, 			1,1,total_keep,DATA_PER_ROI, NN_TYPE_QUINT16);
	tensor_out_prepare_normal(classes_output_tensor, 		1,1,1,total_keep, NN_TYPE_INT32);
	tensor_out_prepare_normal(batch_info_output_tensor, 	1,1,1,total_keep, NN_TYPE_INT32);

	return 0;
}

struct nn_node_ops nn_ops_for_BoxWithNmsLimit_q8q16 = {
	.execute = box_with_nms_limit_q8q16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(13),
	.n_outputs = NN_IOCOUNT(8),
	.flags = NN_NODE_FLAG_CLS_DYNAMIC_TENSOR,
};
