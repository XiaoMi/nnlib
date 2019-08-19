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
#include <stdbool.h>
#include <math.h>
#include <quantize.h>

#include "hvx_inlines.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif
typedef HVX_Vector (*FUNC_PTR)(HVX_Vector, HVX_Vector);

#ifdef HEXAGON_V66
#define VALIDATION_MAX_THREADS 4
#else
#define VALIDATION_MAX_THREADS 2
#endif

#define HVX_SIZE (sizeof(HVX_Vector))
#define THREAD_USED_VTCM_SIZE    1024
#define TOTAL_USED_VTCM_SIZE     (THREAD_USED_VTCM_SIZE*VALIDATION_MAX_THREADS)

/*
 * Mobilenet ssd multi-class non max supression
 */
//The indices of tensors in input
#define OP_MULTICLASSNMS_INPUT_NUM 8
#define INPUT_BOXES_IDX 0
#define INPUT_CLASS_SCORES_IDX 1
#define INPUT_CLASS_SCORES_MIN_IDX 2
#define INPUT_CLASS_SCORES_MAX_IDX 3
#define INPUT_SCORE_THRESHOLD_IDX 4
#define INPUT_IOU_THRESHOLD_IDX 5
#define INPUT_MAX_DETECTION_PER_CLASS 6
#define INPUT_MAX_TOTAL_DETECTIONS 7

//The indices of tensors in output
#define OP_MULTICLASSNMS_OUTPUT_NUM 5
#define OUTPUT_BOXES_IDX 0
#define OUTPUT_SCORES_IDX 1
#define OUTPUT_SCORES_MIN_IDX 2
#define OUTPUT_SCORES_MAX_IDX 3
#define OUTPUT_CLASSES_IDX 4

#define NUM_COORD 4

struct Box {
    float y1;
    float x1;
    float y2;
    float x2;
    float area;
};

void new_box_8(struct Box* box_pointer, const float y1, const float x1, const float y2, const float x2) {
    box_pointer->y1 = y1;
    box_pointer->x1 = x1;
    box_pointer->y2 = y2;
    box_pointer->x2 = x2;
    box_pointer->area = (x2 - x1) * (y2 - y1);
}

struct BoxRecord_8 {
    uint8_t score;
    uint32_t class;
    uint32_t box;   //box index in input
};

void new_boxRecord_8(struct BoxRecord_8* boxrecord_pointer, const uint32_t box, const uint8_t score, const uint32_t class) {
    boxrecord_pointer->box = box; //box index in input
    boxrecord_pointer->score = score;
    boxrecord_pointer->class = class;
}

struct BoxRecord_8_Ref {
    uint8_t score;
    size_t boxRecord_idx;
};

void new_boxRecord_8_Ref(struct BoxRecord_8_Ref* boxRecordIdx_pointer, const uint8_t score, const uint32_t idx) {
    boxRecordIdx_pointer->score = score;
    boxRecordIdx_pointer->boxRecord_idx = idx;
}

#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
struct general_info {
    uint8_t score_cmpthreshold;
    uint32_t round_classes_count;
    uint32_t leftovers;
};

struct validscore_job {
    uint8_t const * in_data;
    uint32_t * valid_num;
    uint16_t * valid_classidx;
};

struct validscore_thread_info {
    struct general_info generalinfo;
    struct validscore_job * jobinfo;
    void * vtcm_addr;
    uint32_t jobs_len;
    nn_sem_t done_sem;
};

static inline uint32_t round_up4(uint32_t size) {
    return ((size + 3) & (-4));
}

static void __attribute__((always_inline))
score_validation_hvx( struct validscore_job const * job, const uint8_t score_threshold,
                      const uint32_t round_classcnt, const uint32_t leftovers,
                      volatile void * vtcm_addr) {
    uint8_t const * in_data = job->in_data;
    uint8_t validBits = 0;
    uint32_t total_validCnt = 0;

    union {
        HVX_Vector vqbitsum;
        uint8_t sqbitsum_u8[128];
    } uu;

    for (uint32_t xd = 0; xd <= round_classcnt; xd +=HVX_SIZE) {
        HVX_Vector vscore = vmemu(&in_data[xd]);
        if (xd == round_classcnt) {
            // Fill zero at the end of valid left over vector data
            vscore = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(leftovers), vscore, Q6_V_vzero());
        }
        HVX_VectorPred qcmpmask = Q6_Q_vcmp_gt_VubVub(vscore, q6op_Vb_vsplat_R(score_threshold));
        uu.vqbitsum = Q6_Vb_prefixsum_Q(qcmpmask);
        validBits = uu.sqbitsum_u8[HVX_SIZE-1];
        if (validBits) {
            // Find out and store valid class index compactly
            HVX_VectorPair vhscorePair = Q6_Wuh_vzxt_Vub(vscore);
            HVX_VectorPair vhqbitsumPair = Q6_Wuh_vzxt_Vub(uu.vqbitsum);
            HVX_VectorPair vhindexPair = Q6_Wuh_vzxt_Vub(*(HVX_Vector const *)const_Count128);

            // Extend byte to half word to execute vscatter, take even(lo) part at first and then odd(hi)
            HVX_VectorPred qcmplomask = Q6_Q_vcmp_gt_VuhVuh(Q6_V_lo_W(vhscorePair), q6op_Vh_vsplat_R(score_threshold));
            // Add index shift on hvx slice
            HVX_Vector vhindexlo = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_lo_W(vhindexPair), q6op_Vh_vsplat_R(xd));
            // Use vscatter to copy valid index according to valid offset from Q prefixsum, offsetx2 for uint16
            Q6_vscatter_QRMVhV(qcmplomask, (uint32_t)vtcm_addr, THREAD_USED_VTCM_SIZE-2, Q6_Vh_vasl_VhR(Q6_V_lo_W(vhqbitsumPair), 1), vhindexlo);

            HVX_VectorPred qcmphimask = Q6_Q_vcmp_gt_VuhVuh(Q6_V_hi_W(vhscorePair), q6op_Vh_vsplat_R(score_threshold));
            HVX_Vector vhindexhi = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_hi_W(vhindexPair), q6op_Vh_vsplat_R(xd));
            Q6_vscatter_QRMVhV(qcmphimask, (uint32_t)vtcm_addr, THREAD_USED_VTCM_SIZE-2, Q6_Vh_vasl_VhR(Q6_V_hi_W(vhqbitsumPair), 1), vhindexhi);
            // Create a fake store operation on VTCM to have above vscatter completed and sync up
            q6op_scatter_release_A(vtcm_addr);
            vmemcpy_asm((job->valid_classidx+total_validCnt), ((uint16_t *)vtcm_addr+1), validBits*sizeof(uint16_t));

            total_validCnt += validBits;
        }
    }
    *(job->valid_num) = total_validCnt;
}

static void validscore_thread_work(struct nn_graph *nn, void *work_info) {
    struct validscore_thread_info* info = (struct validscore_thread_info *)work_info;
    struct validscore_job const * job_ptr = info->jobinfo;
    Q6_dcfetch_A(job_ptr);
    uint32_t jobs_len = info->jobs_len;
    uint8_t score_cmpthreshold = info->generalinfo.score_cmpthreshold;
    uint32_t round_classes_count = info->generalinfo.round_classes_count;
    uint32_t leftovers = info->generalinfo.leftovers;
    void * vtcm_addr = info->vtcm_addr;
    for(uint32_t i = 0; i < jobs_len; i++){
        Q6_dcfetch_A(&job_ptr[i+1]);
        score_validation_hvx(&job_ptr[i], score_cmpthreshold, round_classes_count, leftovers, vtcm_addr);
    }
    nn_sem_post(&info->done_sem);
}
#endif

int compare_box_records_8_ref(const void* br1,const void* br2) {

    if (((struct BoxRecord_8_Ref*)br1)->score > ((struct BoxRecord_8_Ref*)br2)->score) return -1;
    else if (((struct BoxRecord_8_Ref*)br1)->score < ((struct BoxRecord_8_Ref*)br2)->score) return 1;
    return 0;
}

float calculateIntersection_8(const struct Box b1, const struct Box b2) {
    float intersect_width = fmaxf(0.0, fminf(b1.x2, b2.x2) - fmaxf(b1.x1, b2.x1));
    float intersect_height = fmaxf(0.0, fminf(b1.y2, b2.y2) - fmaxf(b1.y1, b2.y1));
    float intersection = intersect_width * intersect_height;
    float areaUnion = b1.area + b2.area - intersection;
    return intersection/areaUnion;
}

static int multiclassnms_8_execute(struct nn_node *self, struct nn_graph *nn) {
    //Input tensors setup
    //[BATCH,1,BOXES,4]
    const uint32_t input_boxes_count = self->inputs[INPUT_BOXES_IDX]->shape.width;                  //number of boxes per image
    float *input_boxes_tensor_data = self->inputs[INPUT_BOXES_IDX]->data;                           //boxes' coordinates

    //[BATCH,1,BOXES,CLASSES]
    const uint32_t batch_count = self->inputs[INPUT_CLASS_SCORES_IDX]->shape.batches;               //number of images
    const uint32_t input_scores_classes_count = self->inputs[INPUT_CLASS_SCORES_IDX]->shape.depth;  //number of scores per box
    uint8_t *input_scores_tensor_data = self->inputs[INPUT_CLASS_SCORES_IDX]->data;                 //scores of boxes

    //Input params setup
    const struct tensor *input_scores_min_tensor = self->inputs[INPUT_CLASS_SCORES_MIN_IDX];
    const struct tensor *input_scores_max_tensor = self->inputs[INPUT_CLASS_SCORES_MAX_IDX];

    const struct tensor *score_threshold_tensor = self->inputs[INPUT_SCORE_THRESHOLD_IDX];
    const struct tensor *iou_threshold_tensor = self->inputs[INPUT_IOU_THRESHOLD_IDX];
    const struct tensor *max_detection_per_class_tensor = self->inputs[INPUT_MAX_DETECTION_PER_CLASS];
    const struct tensor *max_total_detection_tensor = self->inputs[INPUT_MAX_TOTAL_DETECTIONS];

    const float float_score_threshold = tensor_get_float(score_threshold_tensor, 0);
    const float iou_threshold = tensor_get_float(iou_threshold_tensor, 0);                         //overlap ratio threshold
    const int32_t max_detection_per_class = tensor_get_int32(max_detection_per_class_tensor, 0);   //Maximum number of boxes per class
    const int32_t max_total_detections = tensor_get_int32(max_total_detection_tensor, 0);          //Maximum number of returned boxes
    const float score_min = tensor_get_float(input_scores_min_tensor, 0);
    const float score_max = tensor_get_float(input_scores_max_tensor, 0);

    uint8_t score_threshold = (uint8_t)(float_score_threshold * 255.0f / (score_max-score_min));
    if(float_score_threshold>0 && 0==score_threshold)   score_threshold = 1;    //threshold is too small
    //Input tensor offset helper
    const uint32_t coordiates_per_batch = input_boxes_count * NUM_COORD;                           //top left corner (x,y), bottom right corner (x,y). Image index * memory chunk for one image in float*
    uint32_t input_boxes_tensor_offset = 0;                                                        //Image index * memory chunk for one image in float*
    float* input_boxes_tensor_batch = input_boxes_tensor_data;                                     //Head pointer of boxes per image

    const size_t scores_per_batch = input_boxes_count * input_scores_classes_count;                //number of scores per image
    uint32_t input_scores_tensor_offset = 0;                                                       //offset per image in float*
    uint8_t* box_scores_tensor = input_scores_tensor_data;                                         //Head pointer of scores per image

    //Output tensors setup
    struct tensor *output_boxes_tensor = self->outputs[OUTPUT_BOXES_IDX];
    struct tensor *output_scores_tensor = self->outputs[OUTPUT_SCORES_IDX];
    struct tensor *output_scores_min_tensor = self->outputs[OUTPUT_SCORES_MIN_IDX];
    struct tensor *output_scores_max_tensor = self->outputs[OUTPUT_SCORES_MAX_IDX];
    struct tensor *output_classes_tensor = self->outputs[OUTPUT_CLASSES_IDX];

    float *output_boxes_tensor_data = output_boxes_tensor->data;
    uint8_t *output_scores_tensor_data = output_scores_tensor->data;
    float *output_classes_tensor_data = output_classes_tensor->data;

    if (tensor_out_prepare_normal_fromshape(output_boxes_tensor, &output_boxes_tensor->shape, NN_TYPE_FLOAT) !=0) return errlog(nn,"multiclassnms_8 out too small");
    if (tensor_out_prepare_normal_fromshape(output_scores_tensor, &output_scores_tensor->shape, NN_TYPE_QINT8) !=0) return errlog(nn,"multiclassnms_8 out too small");
    if (tensor_out_prepare_normal_fromshape(output_classes_tensor, &output_classes_tensor->shape, NN_TYPE_FLOAT) !=0) return errlog(nn,"multiclassnms_8 out too small");

    const int32_t output_max_coords_per_batch = max_total_detections * NUM_COORD;
    size_t output_boxes_tensor_offset = 0;
    size_t output_scores_tensor_offset = 0;
    size_t output_classes_tensor_offset = 0;
    size_t hash_boxes_per_class_offset = 1 + max_detection_per_class;                            //Size of each node in the filtered boxes hash table; 1 represents the number of boxes per class

    //Array size is in byte
    size_t boxes_array_size = batch_count * input_boxes_count * sizeof(struct Box);
    size_t boxes_records_array_size = batch_count * input_boxes_count * input_scores_classes_count * sizeof(struct BoxRecord_8);
    size_t boxes_records_ref_array_size = batch_count * input_boxes_count * input_scores_classes_count * sizeof(struct BoxRecord_8_Ref);
    size_t filtered_boxes_hashtable_size = (1 + max_detection_per_class) * input_scores_classes_count * sizeof(uint32_t);  //(number of filtered boxes per class + max number of boxes per class)*max number of classes

#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
    size_t validnum_size = input_boxes_count * sizeof(uint32_t);
    size_t validclassidx_size = input_boxes_count * input_scores_classes_count * sizeof(uint16_t);
    validclassidx_size = round_up4(validclassidx_size);

    uint32_t num_threads = (VALIDATION_MAX_THREADS < input_boxes_count) ? VALIDATION_MAX_THREADS : input_boxes_count;
    uint32_t workload_for_worker[VALIDATION_MAX_THREADS];
    uint32_t average_workload_per_worker = input_boxes_count / num_threads;
    uint32_t extra_work = input_boxes_count % num_threads;
    for(uint32_t i = 0; i < num_threads; i++){
        workload_for_worker[i] = average_workload_per_worker;
    }
    for(uint32_t i = 0; i < extra_work; i++){
        workload_for_worker[i] += 1;
    }
#endif

    //Assign memory
    uint8_t *head_temp_mem = nn->scratch;
    struct Box *boxes = (struct Box *)head_temp_mem;
    struct BoxRecord_8 *box_records = (struct BoxRecord_8 *)((uint8_t *)boxes + boxes_array_size);
    struct BoxRecord_8_Ref *box_records_ref = (struct BoxRecord_8_Ref *)((uint8_t *)box_records + boxes_records_array_size);
    uint32_t *filtered_boxes_hashtable = (uint32_t*)((uint8_t *)box_records_ref + boxes_records_ref_array_size);
#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
    uint32_t *validnum = (uint32_t *)((uint8_t *)filtered_boxes_hashtable + filtered_boxes_hashtable_size);
    uint16_t *validclassidx = (uint16_t *)((uint8_t *)validnum + validnum_size);
    struct validscore_job *validation_job = (struct validscore_job *)((uint8_t *)validclassidx + validclassidx_size);
#endif

    size_t box_record_idx = 0;                                                                   //box record array tail index
    size_t filtered_boxes_count = 0;
    for (size_t batch = 0; batch < batch_count; ++batch) {

        box_record_idx = 0;
        filtered_boxes_count = 0;
        input_boxes_tensor_offset = batch * coordiates_per_batch;                                //Image index * memory chunk for one image in float*
        input_scores_tensor_offset = batch * scores_per_batch;                                  //offset per image in float*
        box_scores_tensor = input_scores_tensor_data + input_scores_tensor_offset;

#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
        uint32_t *output_validnum = validnum;
        uint16_t *output_validclassidx = validclassidx;
        struct validscore_job *job = validation_job;
        for (size_t i = 0; i < input_boxes_count; i++, job++) {
            job->in_data = box_scores_tensor;
            job->valid_num = output_validnum++;
            job->valid_classidx = output_validclassidx;
            box_scores_tensor += input_scores_classes_count;
            output_validclassidx += input_scores_classes_count;
        }

        if (input_scores_classes_count == 0) {
            return errlog(nn,"multiclassnms the class count of input scores per box is zero\n");
        }
        if (score_threshold == 0) {
            // the score threshold (score_threshold-1) used to do comparison (>) has to be >= 0
            score_threshold = 1;
        }
        struct general_info basic_info;
        // because cmp.gt is >, not >=. take quantized scorethreshold -1 as threshold in comparison
        basic_info.score_cmpthreshold = score_threshold - 1;
        basic_info.round_classes_count = input_scores_classes_count / HVX_SIZE * HVX_SIZE;
        basic_info.leftovers = input_scores_classes_count % HVX_SIZE;
        if (basic_info.leftovers == 0) {
            basic_info.round_classes_count -= HVX_SIZE;
            basic_info.leftovers = HVX_SIZE;
        }

        if (nn->vtcm_size < TOTAL_USED_VTCM_SIZE) {
            return errlog(nn,"multiclassnms could not get enough VTCM to validate score \n");
        }
        // To guarantee that OS has mapped the TCM range in a single page
        int offset = 0;
        struct validscore_thread_info thrinfo[VALIDATION_MAX_THREADS];
        for(uint32_t i = 0; i < num_threads; i++) {
            nn_sem_init(&thrinfo[i].done_sem, 0);
            thrinfo[i].jobs_len = workload_for_worker[i];
            thrinfo[i].jobinfo = validation_job + offset;
            thrinfo[i].generalinfo = basic_info;
            thrinfo[i].vtcm_addr = (void *)((uint32_t)nn->vtcm_ptr + i*THREAD_USED_VTCM_SIZE);
            offset += workload_for_worker[i];
            nn_os_work_for_vector(nn, validscore_thread_work, &thrinfo[i]);
        }

        for(uint32_t i =0; i < num_threads; i++) {
            nn_sem_wait(&thrinfo[i].done_sem);
        }
#endif

        box_scores_tensor = input_scores_tensor_data + input_scores_tensor_offset;
        for (size_t i = 0; i < input_boxes_count; ++i, box_scores_tensor += input_scores_classes_count) {                                        //i: box index

            input_boxes_tensor_batch = input_boxes_tensor_data + input_boxes_tensor_offset;
#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
            if (0 == validnum[i])
                continue;
#endif
            //keep the coordinates that are in the range of [0,1]
            float y1 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 0], 0.0), 1.0);
            float x1 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 1], 0.0), 1.0);
            float y2 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 2], 0.0), 1.0);
            float x2 = fminf(fmaxf(input_boxes_tensor_batch[i * NUM_COORD + 3], 0.0), 1.0);
            new_box_8(&boxes[i], y1, x1, y2, x2);

            if (0 == boxes[i].area)
                continue;

#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
            uint16_t * valid_index_array = &(validclassidx[i*input_scores_classes_count]);
            for (uint32_t j = 0; j < validnum[i]; ++j) {                                         //j: valid copy loop count
                // Fetch valid class index directly
                uint32_t valid_index = valid_index_array[j];
                new_boxRecord_8(&box_records[box_record_idx], i, box_scores_tensor[valid_index], valid_index);
                new_boxRecord_8_Ref(&box_records_ref[box_record_idx],box_scores_tensor[valid_index], box_record_idx);
                ++box_record_idx;
            }//end traverse of scores of each box
#else
            for (int32_t j = 0; j < input_scores_classes_count; ++j) {                          //j: class index
                if (box_scores_tensor[j] < score_threshold)
                    continue;

                new_boxRecord_8(&box_records[box_record_idx], i, box_scores_tensor[j], (uint32_t)j);
                new_boxRecord_8_Ref(&box_records_ref[box_record_idx],box_scores_tensor[j], box_record_idx);
                ++box_record_idx;
            }//end traverse of scores of each box
#endif
        }//end traverse of boxes

        //sort box records in descending order
        qsort(box_records_ref,box_record_idx, sizeof(struct BoxRecord_8_Ref), compare_box_records_8_ref);

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

                iou = calculateIntersection_8(boxes[filtered_boxes_hashtable[hashtable_offset+1+m]], *currentBox);

                if(iou > iou_threshold) {
                    should_keep = false;
                    break;
                }
            }//end traverse of boxes in the same class

            if(should_keep){

                filtered_boxes_hashtable[hashtable_offset + 1 + num_boxes_current_class] = box_records[temp_box_record_idx].box;     //Add box index at the back
                filtered_boxes_hashtable[hashtable_offset]++;                                                                        //Increase the number of boxes in the class

                output_boxes_tensor_data[output_boxes_tensor_offset + filtered_boxes_count * NUM_COORD] = currentBox->y1;
                output_boxes_tensor_data[output_boxes_tensor_offset + filtered_boxes_count * NUM_COORD + 1] = currentBox->x1;
                output_boxes_tensor_data[output_boxes_tensor_offset + filtered_boxes_count * NUM_COORD + 2] = currentBox->y2;
                output_boxes_tensor_data[output_boxes_tensor_offset + filtered_boxes_count * NUM_COORD + 3] = currentBox->x2;
                output_scores_tensor_data[output_scores_tensor_offset + filtered_boxes_count] = box_records[temp_box_record_idx].score;
                output_classes_tensor_data[output_classes_tensor_offset + filtered_boxes_count] = (float)box_records[temp_box_record_idx].class;

                ++filtered_boxes_count;
            }
        }//end traverse of box records

        *(float*)(output_scores_min_tensor->data) = score_min;
        *(float*)(output_scores_max_tensor->data) = score_max;

        int32_t num_unused_units = max_total_detections - filtered_boxes_count;
        if (0 == num_unused_units) continue;

        float *batch_boxes_ptr =  output_boxes_tensor_data + output_boxes_tensor_offset;
        uint8_t *batch_scores_ptr =  output_scores_tensor_data + output_scores_tensor_offset;
        float *batch_classes_ptr =  output_classes_tensor_data + output_classes_tensor_offset;

        memset(batch_boxes_ptr + filtered_boxes_count * NUM_COORD, 0, num_unused_units * NUM_COORD * sizeof(float));
        memset(batch_scores_ptr + filtered_boxes_count, 0, num_unused_units * sizeof(uint8_t));
        memset(batch_classes_ptr + filtered_boxes_count, 0, num_unused_units * sizeof(float));

    }//end traverse of images
    return 0;
}

static int multiclassnms_8_check(struct nn_node *self, struct nn_graph *nn)  {
    logmsg(nn,2,"multiclassnms %p scratch enough memory space",self);
    const uint32_t input_boxes_count = self->inputs[INPUT_BOXES_IDX]->shape.width;                  //number of boxes per image
    const uint32_t batch_count = self->inputs[INPUT_CLASS_SCORES_IDX]->shape.batches;               //number of images
    const uint32_t input_scores_classes_count = self->inputs[INPUT_CLASS_SCORES_IDX]->shape.depth;  //number of scores per box
    const int32_t max_detection_per_class = tensor_get_int32(self->inputs[INPUT_MAX_DETECTION_PER_CLASS], 0);   //Maximum number of boxes per class

    size_t boxes_array_size = batch_count * input_boxes_count * sizeof(struct Box);
    size_t boxes_records_array_size = batch_count * input_boxes_count * input_scores_classes_count * sizeof(struct BoxRecord_8);
    size_t boxes_records_ref_array_size = batch_count * input_boxes_count * input_scores_classes_count * sizeof(struct BoxRecord_8_Ref);
    size_t filtered_boxes_hashtable_size = (1 + max_detection_per_class) * input_scores_classes_count * sizeof(uint32_t);  //(number of filtered boxes per class + max number of boxes per class)*max number of classes
    size_t total_size = boxes_array_size + boxes_records_array_size + boxes_records_ref_array_size + filtered_boxes_hashtable_size;
#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
    if (UINT16_MAX < input_scores_classes_count) {
        return errlog(nn,"multiclassnms input class count is too big to get validation with half-word vscatter\n");
    }
    size_t validnum_size = input_boxes_count * sizeof(uint32_t);
    size_t validclassidx_size = input_boxes_count * input_scores_classes_count * sizeof(uint16_t);
    validclassidx_size = round_up4(validclassidx_size);
    size_t validation_work_size = input_boxes_count * sizeof(struct validscore_job);
    total_size += (validnum_size + validclassidx_size + validation_work_size);
#endif
    if(nn_scratch_grow(nn,total_size)){
        return errlog(nn,"multiclassnms failed to get scratch \n");
    }
    logmsg(nn,2,"multiclassnms %p check OK",self);
    return 0;
}

struct nn_node_ops nn_ops_for_MultiClassNms_8 = {
    .execute = multiclassnms_8_execute,
    .check = multiclassnms_8_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_MULTICLASSNMS_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_MULTICLASSNMS_OUTPUT_NUM),
};
