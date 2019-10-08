
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
#include "2d_geometry.h"
#ifndef __hexagon__
#include <malloc.h>
#include <2d_geometry.h>

#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a box with nms limit op.
 */


static void load_upright_boxes(float* boxes, int total_boxes, int total_classes, rectangle** result){

	for(int i = 0; i < total_classes; i++){
		rectangle* boxes_i = nn_malloc(total_boxes*sizeof(rectangle));
		for(int j = 0; j < total_boxes; j++){
			boxes_i[j] = create_upright_rectangle(boxes + 4*i + j*4*total_classes);
		}
		result[i] = boxes_i;
	}
}

static void load_rotated_boxes(float* boxes, int total_boxes, int total_classes, rectangle** result){

	for(int i = 0; i < total_classes; i++){
		rectangle* boxes_i = nn_malloc(total_boxes*sizeof(rectangle));
		for(int j = 0; j < total_boxes; j++){
			boxes_i[j] = create_rotated_rectangle(boxes + 5*i + j*5*total_classes);
		}
		result[i] = boxes_i;
	}
}

static void load_scores(float* scores, int total_boxes, int total_classes, float** result){

	for(int i = 0; i < total_classes; i++){
		float* scores_i = nn_malloc(total_boxes*sizeof(float));
		for(int j = 0; j < total_boxes; j++){
			scores_i[j] = scores[i+j*total_classes];
		}
		result[i] = scores_i;
	}
}

static void get_indices_gt(float* array, int start, int end, float threshold, int* result, int* result_size){

	int count = 0;
	for (int i = start; i < end; i++) {
		if (array[i] > threshold) {
			result[count] = i;
			count++;
		}
	}

	*result_size = count;
}

typedef struct {
	int index;
	float score;
} box_score;

static int compare_box_scores (const void * a, const void * b) {
	box_score bs_a = *(box_score*)a;
	box_score bs_b = *(box_score*)b;
   return ( bs_b.score - bs_a.score >= 0.0 ? 1 : -1 );
}

static void sort_indices(int* indices, int num_indices, float* scores){

	box_score *box_scores = nn_malloc(num_indices*sizeof(box_score));
	int i;
	for(i = 0; i < num_indices; i++){
		box_scores[i].index = indices[i];
		box_scores[i].score = scores[indices[i]];
	}
	
	qsort(box_scores, num_indices, sizeof(box_score), compare_box_scores);
	
	for(i = 0; i < num_indices; i++){
		indices[i] = box_scores[i].index;
	}

	nn_free(box_scores);
}

static int get_max_coefficient(float *scores, int *candidate_indices, int num_candidates, int *status){
	int result = -1;
	float max_score = 0;
	
	for(int i = 0; i < num_candidates; i++){
		if(status[i] == 0 && scores[candidate_indices[i]] > max_score){
			result = i;
			max_score = scores[candidate_indices[i]];
		}
	}

	return result;
}

static void nms(rectangle *boxes, int *candidate_indices, int num_candidates, float threshold, int keep_max, int *result, int *result_size){

	if(num_candidates <= 0){
		*result_size = 0;
		return;
	}

	float *areas = nn_malloc(num_candidates * sizeof(float));
	int num_selected = 0;

	for(int i = 0; i < num_candidates; i++){
		areas[i] = (boxes[candidate_indices[i]].height  + 1.f) * (boxes[candidate_indices[i]].width + 1.f);
    }

	for(int current_candidate_index = 0; current_candidate_index < num_candidates; current_candidate_index++){
		int current_candidate = candidate_indices[current_candidate_index];

		if(current_candidate == -1){
			continue;
		}

		num_selected++;
		
		for(int i = current_candidate_index + 1; i < num_candidates; i++){

			if(candidate_indices[i] == -1){
				continue;
			}

			float inter = get_rectangles_intersection_area(boxes[candidate_indices[i]], boxes[current_candidate]);
			float ovr = inter / (areas[current_candidate_index] + areas[i] - inter);

			if(ovr > threshold){
				candidate_indices[i] = -1;
			}
		}

		if(keep_max > 0 && num_selected >= keep_max){
			break;
		}
	}
	
	nn_free(areas);

	int result_index = 0;
	for(int i = 0; i < num_candidates; i++){
		if(candidate_indices[i] != -1){
			result[result_index] = candidate_indices[i];
			result_index++;
			
			if(result_index == num_selected){
				break;
			}
		}
	}
	
	*result_size = num_selected;
}

static void soft_nms(rectangle *boxes, float *scores, int *candidate_indices, int num_candidates, float overlap_thresh, float score_thresh, float sigma, int method, int* result, int *result_size){

	if(num_candidates <= 0){
		*result_size = 0;
		return;
	}

	float *areas = nn_malloc(num_candidates * sizeof(float));
	int *status = nn_malloc(num_candidates * sizeof(int));	// 0 = not processed, -1 = rejected, >0 = the order it was selected in
	int num_selected = 0;
	int candidates_remaining = num_candidates;

	for(int i = 0; i < num_candidates; i++){
		areas[i] = (boxes[candidate_indices[i]].height + 1.f) * (boxes[candidate_indices[i]].width + 1.f);
		status[i] = 0;
	}


	while(candidates_remaining >0){
		int current_candidate_index = get_max_coefficient(scores, candidate_indices, num_candidates, status);
		
		if(current_candidate_index == -1){
			break;
		}

		int current_candidate = candidate_indices[current_candidate_index];

		num_selected++;
		status[current_candidate_index] = num_selected;
		
		for(int i = 0; i < num_candidates; i++){

			if(status[i] != 0){
				continue;
			}

			float inter = get_rectangles_intersection_area(boxes[candidate_indices[i]], boxes[current_candidate]);
			float ovr = inter / (areas[current_candidate_index] + areas[i] - inter);
			
			float weight;
			switch (method) {
				case 1: // Linear
					weight = (ovr > overlap_thresh) ? (1.0f - ovr) : 1.0f;
					break;
				case 2: // Gaussian
					weight = expf(-ovr * ovr / sigma);
					break;
				default: // Original NMS
					weight = (ovr > overlap_thresh) ? 0.0f : 1.0f;
			}
			scores[candidate_indices[i]] *= weight;
			
			if(scores[candidate_indices[i]] < score_thresh){
				status[i] = -1;
				candidates_remaining--;
			}
		}
		
		candidates_remaining--;
	}

	for(int i = 0; i < num_candidates; i++){
		if(status[i] > 0){
			result[status[i]-1] = candidate_indices[i];
		}
	}
	
	*result_size = num_selected;

	nn_free(areas);
	nn_free(status);
}

typedef struct {
	int class;
	int index;
	float score;
} keep_index;

int compare_keep_index(const void *a, const void *b){
	keep_index index_a = *(keep_index*)a;
	keep_index index_b = *(keep_index*)b;
	return ( index_b.score - index_a.score > 0.0 ? 1 : -1 );
}

static int box_with_nms_limit_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"box_with_nms_limit_execute execute. self=%p ",self);
	
	const struct tensor *scores_input_tensor = self->inputs[0];
	const struct tensor *boxes_input_tensor = self->inputs[1];
    const struct tensor *batch_splits_input_tensor = self->inputs[2];
	
	float *scores_input = scores_input_tensor->data;
	float *boxes_input = boxes_input_tensor->data;
	float *batch_splits_data = batch_splits_input_tensor->data;

	int batch_size = batch_splits_input_tensor->shape.batches * batch_splits_input_tensor->shape.height * batch_splits_input_tensor->shape.width * batch_splits_input_tensor->shape.depth;

	float score_threshold = tensor_get_float(self->inputs[3], 0);
	float nms_threshold = tensor_get_float(self->inputs[4], 0);
	int detections_per_image = tensor_get_int32(self->inputs[5], 0);
	int soft_nms_enabled = tensor_get_int32(self->inputs[6], 0);
	unsigned int soft_nms_method = tensor_get_int32(self->inputs[7], 0);
	float soft_nms_sigma = tensor_get_float(self->inputs[8], 0);
	float soft_nms_min_score_threshold = tensor_get_float(self->inputs[9], 0);
    int rotated = tensor_get_int32(self->inputs[10], 0);
	
	struct tensor *scores_output_tensor = self->outputs[0];
	struct tensor *boxes_output_tensor = self->outputs[1];
	struct tensor *classes_output_tensor = self->outputs[2];
	struct tensor *batch_splits_output_tensor = self->outputs[3];

	const int box_dim = rotated ? 5 : 4;

    if (scores_input_tensor->shape.height != 1) return errlog(nn,"scores_input_tensor height is not 1");
    if (boxes_input_tensor->shape.height != 1) return errlog(nn,"boxes_input_tensor height is not 1");
    if (scores_input_tensor->shape.width != boxes_input_tensor->shape.width) return errlog(nn,"scores_input_tensor width != boxes_input_tensor width");
    if (scores_input_tensor->shape.depth * box_dim != boxes_input_tensor->shape.depth) return errlog(nn,"scores_input_tensor depth x box_dim != boxes_input_tensor depth");

	float *scores_output = scores_output_tensor->data;
	float *boxes_output = boxes_output_tensor->data;
	float *classes_output = classes_output_tensor->data;
	float *batch_splits_output = batch_splits_output_tensor->data;

	int total_boxes = scores_input_tensor->shape.width;
	int num_classes = scores_input_tensor->shape.depth;

	float batch_splits_sum = 0;
	for(int i = 0; i < batch_size; i++){
		batch_splits_sum += batch_splits_data[i];
	}

	if ((int)batch_splits_sum != total_boxes) return errlog(nn,"batch_splits sum != scores_input_tensor width");
	
	struct tensor *out_keeps_tensor;
	struct tensor *out_keeps_size_tensor;
	int *keeps_output;
	int *keeps_size_output;
	if (self->n_outputs > 4){
		out_keeps_tensor = self->outputs[4];
		out_keeps_size_tensor = self->outputs[5];
		keeps_output = out_keeps_tensor->data;
		keeps_size_output = out_keeps_size_tensor->data;	
	}

	int **keeps = nn_malloc(num_classes * sizeof(int*));
	
	int *total_keep_per_class = nn_malloc(num_classes * sizeof(int*));

	rectangle **boxes = nn_malloc(num_classes*sizeof(rectangle*));
	if(rotated)
		load_rotated_boxes(boxes_input, total_boxes, num_classes, boxes);
	else
		load_upright_boxes(boxes_input, total_boxes, num_classes, boxes);
	
	float **scores = nn_malloc(num_classes*sizeof(float*));
	load_scores(scores_input, total_boxes, num_classes, scores);

	int max_keep = (num_classes-1)*total_boxes;

	tensor_out_prepare_normal(scores_output_tensor, 		1,1,1,max_keep, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(boxes_output_tensor, 			1,1,max_keep,box_dim, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(classes_output_tensor, 		1,1,1,max_keep, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(batch_splits_output_tensor, 	1,1,1,batch_size, NN_TYPE_FLOAT);

	memset(scores_output, 0, max_keep*4);
	memset(boxes_output, 0, max_keep*box_dim*4);
	memset(classes_output, 0, max_keep*4);

	if (self->n_outputs > 4){
		tensor_out_prepare_normal(out_keeps_tensor, 				1,1,1,max_keep,  NN_TYPE_INT32);
		tensor_out_prepare_normal(out_keeps_size_tensor, 			1,1,batch_size,num_classes, NN_TYPE_INT32);

		memset(keeps_output, 0, max_keep*4);
	}

	int keep_idx = 0;
    int offset = 0;
	for (int b = 0; b < batch_size; ++b){
		int total_keep = 0;
		int num_boxes = batch_splits_data[b];
		int* indices = nn_malloc(num_boxes*sizeof(int));

		batch_splits_output[b] = 0;

		for (int i = 1; i < num_classes; i++){

			int ind_size, result_size = 0;
			total_keep_per_class[i] = 0;
			keeps[i] = nn_malloc(num_boxes*sizeof(int));

			get_indices_gt(scores[i], offset, offset + num_boxes, score_threshold, indices, &ind_size);

			if (soft_nms_enabled){
				soft_nms(boxes[i], scores[i], indices, ind_size, nms_threshold, soft_nms_min_score_threshold, soft_nms_sigma, soft_nms_method, keeps[i], &result_size);
			}
			else{
				int keep_max = detections_per_image > 0 ? detections_per_image : -1;
				sort_indices(indices, ind_size, scores[i]);
				nms(boxes[i], indices, ind_size, nms_threshold, keep_max, keeps[i], &result_size);
			}

			total_keep_per_class[i] += result_size;
			batch_splits_output[b] += result_size;
			total_keep += result_size;
		}

		nn_free(indices);

        if (detections_per_image > 0 && total_keep > detections_per_image) {
            keep_index* all_scores = nn_malloc(total_keep * sizeof(keep_index));
            int idx = 0;
            for(int i = 1; i < num_classes; i++){
                for(int j = 0; j < total_keep_per_class[i]; j++){
                    all_scores[idx] = (keep_index){.class = i, .index = keeps[i][j], .score = scores[i][keeps[i][j]] };
                    idx++;
                }
            }

            qsort(all_scores, total_keep, sizeof(keep_index), compare_keep_index);

            total_keep = 0;
			batch_splits_output[b] = 0;
            for(int i = 1; i < num_classes; i++){
                total_keep_per_class[i] = 0;
            }

            for(int i = 0; i < detections_per_image; i++){
                keeps[all_scores[i].class][total_keep_per_class[all_scores[i].class]] = all_scores[i].index;
                total_keep++;
				batch_splits_output[b]++;
                total_keep_per_class[all_scores[i].class]++;
            }

            nn_free(all_scores);
        }

		for(int i = 1; i < num_classes; i++){
			for(int j = 0; j < total_keep_per_class[i]; j++){
				scores_output[keep_idx] = scores[i][keeps[i][j]];
				memcpy(boxes_output + box_dim*keep_idx, boxes_input + i*box_dim + num_classes*box_dim*keeps[i][j], box_dim*sizeof(float));
				classes_output[keep_idx] = i;

				if (self->n_outputs > 4){
					keeps_output[keep_idx] = keeps[i][j] - offset;
				}

				keep_idx++;
			}

			if (self->n_outputs > 4){
				keeps_size_output[i + b*num_classes] = total_keep_per_class[i];
			}
			
			nn_free(keeps[i]);
		}

        offset += num_boxes;
	}

	for(int i = 0; i < num_classes; i++){
		nn_free(boxes[i]);
		nn_free(scores[i]);
	}
	nn_free(boxes);
	nn_free(scores);

	nn_free(keeps);
    nn_free(total_keep_per_class);

	return 0;
}


struct nn_node_ops nn_ops_for_BoxWithNmsLimit_f = {
	.execute = box_with_nms_limit_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT_RANGE(4,6),
};
