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
#include <stdlib.h>
#include <stdbool.h>

/*
 * Mobilenet ssd multi-class non max supression
 */

//The indices of tensors in input
#define OP_MULTICLASSNMS_INPUT_NUM 6
#define INPUT_BOXES_IDX 0
#define INPUT_CLASS_SCORES_IDX 1
#define INPUT_SCORE_THRESHOLD_IDX 2
#define INPUT_IOU_THRESHOLD_IDX 3
#define INPUT_MAX_DETECTION_PER_CLASS 4
#define INPUT_MAX_TOTAL_DETECTIONS 5

//The indices of tensors in output
#define OP_MULTICLASSNMS_OUTPUT_NUM 3
#define OUTPUT_BOXES_IDX 0
#define OUTPUT_SCORES_IDX 1
#define OUTPUT_CLASSES_IDX 2

#define NUM_COORD 4

struct Box {
    float y1;
    float x1;
    float y2;
    float x2;
    float area;
};

void new_box(struct Box* box_pointer, const float y1, const float x1, const float y2, const float x2) {
    box_pointer->y1 = y1;
    box_pointer->x1 = x1;
    box_pointer->y2 = y2;
    box_pointer->x2 = x2;
    box_pointer->area = (x2 - x1) * (y2 - y1);
}

struct BoxRecord {
    float score;
    uint32_t class;
    uint32_t box;   //box index in input
};

void new_boxRecord(struct BoxRecord* boxrecord_pointer, const uint32_t box, const float score, const uint32_t class) {
    boxrecord_pointer->box = box; //box index in input
    boxrecord_pointer->score = score;
    boxrecord_pointer->class = class;
}

struct BoxRecordRef {
    float score;
    size_t boxRecord_idx;
};

void new_boxRecordRef(struct BoxRecordRef* boxRecordIdx_pointer, const float score, const uint32_t idx) {
    boxRecordIdx_pointer->score = score;
    boxRecordIdx_pointer->boxRecord_idx = idx;
}

int compare_box_records(const void* br1,const void* br2) {

    float diff = ((struct BoxRecord*)br1)->score - ((struct BoxRecord*)br2)->score;
    if (diff > 0) {
        return -1;
    }
    if (diff < 0) {
        return 1;
    }
    return 0;
}

int compare_box_records_ref(const void* br1,const void* br2) {
    float diff = ((struct BoxRecordRef*)br1)->score - ((struct BoxRecordRef*)br2)->score;
    if (diff > 0) return -1;
    if (diff < 0) return 1;
    return 0;
}

float calculateIntersection(const struct Box b1, const struct Box b2) {
    float intersect_width = fmaxf(0.0, fminf(b1.x2, b2.x2) - fmaxf(b1.x1, b2.x1));
    float intersect_height = fmaxf(0.0, fminf(b1.y2, b2.y2) - fmaxf(b1.y1, b2.y1));
    float intersection = intersect_width * intersect_height;
    float areaUnion = b1.area + b2.area - intersection;
    return intersection/areaUnion;
}

static int multiclassnms_execute(struct nn_node *self, struct nn_graph *nn) {

    //Input tensors setup
    //[BATCH,1,BOXES,4]
    const uint32_t input_boxes_count = self->inputs[INPUT_BOXES_IDX]->shape.width;                  //number of boxes per image
    float *input_boxes_tensor_data = self->inputs[INPUT_BOXES_IDX]->data;                           //boxes' coordinates

    //[BATCH,1,BOXES,CLASSES]
    const uint32_t batch_count = self->inputs[INPUT_CLASS_SCORES_IDX]->shape.batches;               //number of images
    const uint32_t input_scores_classes_count = self->inputs[INPUT_CLASS_SCORES_IDX]->shape.depth;  //number of scores per box
    float *input_scores_tensor_data = self->inputs[INPUT_CLASS_SCORES_IDX]->data;                   //scores of boxes

    //Input params setup
    const struct tensor *score_threshold_tensor = self->inputs[INPUT_SCORE_THRESHOLD_IDX];
    const struct tensor *iou_threshold_tensor = self->inputs[INPUT_IOU_THRESHOLD_IDX];
    const struct tensor *max_detection_per_class_tensor = self->inputs[INPUT_MAX_DETECTION_PER_CLASS];
    const struct tensor *max_total_detection_tensor = self->inputs[INPUT_MAX_TOTAL_DETECTIONS];

    const float score_threshold = tensor_get_float(score_threshold_tensor, 0);
    const float iou_threshold = tensor_get_float(iou_threshold_tensor, 0);                         //overlap ratio threshold
    const int32_t max_detection_per_class = tensor_get_int32(max_detection_per_class_tensor, 0);   //Maximum number of boxes per class
    const int32_t max_total_detections = tensor_get_int32(max_total_detection_tensor, 0);          //Maximum number of returned boxes

    //Input tensor offset helper
    const uint32_t coordiates_per_batch = input_boxes_count * NUM_COORD;                           //top left corner (x,y), bottom right corner (x,y). Image index * memory chunk for one image in float*
    uint32_t input_boxes_tensor_offset = 0;                                                        //Image index * memory chunk for one image in float*
    float* input_boxes_tensor_batch = input_boxes_tensor_data;                                     //Head pointer of boxes per image

    const size_t scores_per_batch = input_boxes_count * input_scores_classes_count;                //number of scores per image
    uint32_t input_scores_tensor_offset = 0;                                                       //offset per image in float*
    float* box_scores_tensor = input_scores_tensor_data;                                           //Head pointer of scores per image

    //Output tensors setup
    struct tensor *output_boxes_tensor = self->outputs[OUTPUT_BOXES_IDX];
    struct tensor *output_scores_tensor = self->outputs[OUTPUT_SCORES_IDX];
    struct tensor *output_classes_tensor = self->outputs[OUTPUT_CLASSES_IDX];

    float *output_boxes_tensor_data = output_boxes_tensor->data;
    float *output_scores_tensor_data = output_scores_tensor->data;
    float *output_classes_tensor_data = output_classes_tensor->data;

    tensor_set_shape(output_boxes_tensor,batch_count,1,max_total_detections,NUM_COORD);
    tensor_set_shape(output_scores_tensor,batch_count,1,1,max_total_detections);
    tensor_set_shape(output_classes_tensor,batch_count,1,1,max_total_detections);

    output_boxes_tensor->data_size = max_total_detections * 4 * sizeof(float);
    output_scores_tensor->data_size = max_total_detections * sizeof(float);
    output_classes_tensor->data_size = max_total_detections * sizeof(float);

    const int32_t output_max_coords_per_batch = max_total_detections * NUM_COORD;
    size_t output_boxes_tensor_offset = 0;
    size_t output_scores_tensor_offset = 0;
    size_t output_classes_tensor_offset = 0;
    size_t hash_boxes_per_class_offset = 1 + max_detection_per_class;                            //Size of each node in the filtered boxes hash table; 1 represents the number of boxes per class

    //Array size is in byte
    size_t boxes_array_size = batch_count * input_boxes_count * sizeof(struct Box);
    size_t boxes_records_array_size = batch_count * input_boxes_count * input_scores_classes_count * sizeof(struct BoxRecord);
    size_t boxes_records_ref_array_size = batch_count * input_boxes_count * input_scores_classes_count * sizeof(struct BoxRecordRef);
    size_t filtered_boxes_hashtable_size = (1 + max_detection_per_class) * input_scores_classes_count * sizeof(uint32_t);  //(number of filtered boxes per class + max number of boxes per class)*max number of classes
    size_t total_size = boxes_array_size + boxes_records_array_size + boxes_records_ref_array_size + filtered_boxes_hashtable_size;

    if(nn_scratch_grow(nn,total_size)){
        return errlog(nn,"failed to get scratch \n");
    }

    //Assign memory
    uint8_t *head_temp_mem = nn->scratch;
    struct Box *boxes = (struct Box *)head_temp_mem;
    struct BoxRecord *box_records = (struct BoxRecord *)((uint8_t *)boxes + boxes_array_size);
    struct BoxRecordRef *box_records_ref = (struct BoxRecordRef *)((uint8_t *)box_records + boxes_records_array_size);
    uint32_t *filtered_boxes_hashtable = (uint32_t*)((uint8_t *)box_records_ref + boxes_records_ref_array_size);

    size_t box_idx = 0;                                                                          //box array tail index, pointing to the last element
    size_t box_record_idx = 0;                                                                   //box record array tail index
    size_t filtered_boxes_count = 0;
    for (size_t batch = 0; batch < batch_count; ++batch) {

        box_idx = 0;
        box_record_idx = 0;
        filtered_boxes_count = 0;
        input_boxes_tensor_offset = batch * coordiates_per_batch;                                //Image index * memory chunk for one image in float*

        //Validate boxes
        for (size_t i = 0; i < input_boxes_count; ++i) {
            input_boxes_tensor_batch = input_boxes_tensor_data + input_boxes_tensor_offset;
            //keep the coordinates that are in the range of [0,1]
            float y1 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 0], 0.0), 1.0);
            float x1 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 1], 0.0), 1.0);
            float y2 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 2], 0.0), 1.0);
            float x2 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 3], 0.0), 1.0);
            new_box(&boxes[box_idx], y1, x1, y2, x2);
            ++box_idx;
        }//end traverse of boxes

        //Find valid scores
        input_scores_tensor_offset = batch * scores_per_batch;                                  //offset per image in float*
        for (size_t i = 0; i < input_boxes_count; ++i) {                                        //i: box index

            if (0 == boxes[i].area)
                continue;
            box_scores_tensor = input_scores_tensor_data + input_scores_tensor_offset + i * input_scores_classes_count;

            for (int32_t j = 0; j < input_scores_classes_count; ++j) {                          //j: class index
                if (box_scores_tensor[j] < score_threshold)
                    continue;

                new_boxRecord(&box_records[box_record_idx], i, box_scores_tensor[j], (uint32_t)j);
                new_boxRecordRef(&box_records_ref[box_record_idx],box_scores_tensor[j], box_record_idx);
                ++box_record_idx;
            }//end traverse of scores of each box
        }//end traverse of boxes

        //sort box records in descending order
        qsort(box_records_ref,box_record_idx, sizeof(struct BoxRecordRef), compare_box_records_ref);

        //output
        output_boxes_tensor_offset = batch * output_max_coords_per_batch;
        output_scores_tensor_offset = batch * max_total_detections;
        output_classes_tensor_offset = batch * max_total_detections;

        //temp variables
        uint32_t current_class = 0;
        struct Box *currentBox = NULL;
        size_t hashtable_offset = 0;
        size_t  num_boxes_current_class = 0;
        float iou = 0.0;
        size_t hashtable_length = 0;
        bool should_keep = true;
        int32_t temp_box_record_idx = 0;

        memset(filtered_boxes_hashtable, 0, filtered_boxes_hashtable_size);                    //Reset the hash table

        //Traverse the box record array. Keep the boxes that have little overlap area.
        //Use a hash table. The key is the class, the values are the box indices in the class.
        for(int32_t k = 0; k < box_record_idx; ++k) {

            if(filtered_boxes_count == max_total_detections)
                break;

            temp_box_record_idx = box_records_ref[k].boxRecord_idx;
            current_class = box_records[temp_box_record_idx].class;
            hashtable_offset = current_class * hash_boxes_per_class_offset;
            num_boxes_current_class = filtered_boxes_hashtable[hashtable_offset];

            if(num_boxes_current_class == max_detection_per_class)
                continue;

            currentBox = &boxes[box_records[temp_box_record_idx].box];
            should_keep = true;

            //Check with the boxes in the same class, discard if the overlap is too big
            for(int32_t  m = 0; m < num_boxes_current_class; ++m) {

                iou = calculateIntersection(boxes[filtered_boxes_hashtable[hashtable_offset+1+m]], *currentBox);

                if(iou > iou_threshold) {
                    should_keep = false;
                    break;
                }
            }//end traverse of boxes in the same class

            if(should_keep){

                filtered_boxes_hashtable[hashtable_offset + 1 + num_boxes_current_class] = box_records[temp_box_record_idx].box;     //Add box index at the back
                filtered_boxes_hashtable[hashtable_offset]++;                                                                        //Increase the number of boxes in the class

                output_boxes_tensor_data[output_boxes_tensor_offset + hashtable_length * NUM_COORD] = currentBox->y1;
                output_boxes_tensor_data[output_boxes_tensor_offset + hashtable_length * NUM_COORD + 1] = currentBox->x1;
                output_boxes_tensor_data[output_boxes_tensor_offset + hashtable_length * NUM_COORD + 2] = currentBox->y2;
                output_boxes_tensor_data[output_boxes_tensor_offset + hashtable_length * NUM_COORD + 3] = currentBox->x2;
                output_scores_tensor_data[output_scores_tensor_offset + hashtable_length] = box_records[temp_box_record_idx].score;
                output_classes_tensor_data[output_classes_tensor_offset + hashtable_length] = (float)box_records[temp_box_record_idx].class;

                ++filtered_boxes_count;
                ++hashtable_length;
            }
        }//end traverse of box records

        int32_t num_unused_units = max_total_detections - filtered_boxes_count;

        if (0 == num_unused_units) continue;

        float *batch_boxes_ptr =  output_boxes_tensor_data + output_boxes_tensor_offset;
        float *batch_scores_ptr =  output_scores_tensor_data + output_scores_tensor_offset;
        float *batch_classes_ptr =  output_classes_tensor_data + output_classes_tensor_offset;

        memset(batch_boxes_ptr + filtered_boxes_count * NUM_COORD, 0, num_unused_units * NUM_COORD * sizeof(float));
        memset(batch_scores_ptr + filtered_boxes_count, 0, num_unused_units * sizeof(float));
        memset(batch_classes_ptr + filtered_boxes_count, 0, num_unused_units * sizeof(float));

    }//end traverse of images
    return 0;
}


struct nn_node_ops nn_ops_for_MultiClassNms_f = {
	.execute = multiclassnms_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(OP_MULTICLASSNMS_INPUT_NUM),
	.n_outputs = NN_IOCOUNT(OP_MULTICLASSNMS_OUTPUT_NUM),
};

