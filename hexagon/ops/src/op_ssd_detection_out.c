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
#include <limits.h>
#include <quantize.h>

#include "float_mathops.h"
#include "nn_pqueue.h"
#include "hvx_inlines.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif

#define THREAD_USED_VTCM_SIZE 1024
#define TOTAL_USED_VTCM_SIZE (THREAD_USED_VTCM_SIZE * NUM_THREADS)

//Struct defs
#define CORNER 0
#define CENTER_SIZE 1
#define CORNER_SIZE 2

#define ALIGN128(a) ((((size_t)a)+127)&(~(size_t)127))

struct bbox
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float width;
    float height;
    float x_center;
    float y_center;
    float size;
};

struct bbox_record_ref
{
    uint32_t batch;
    int32_t class;
    uint32_t idx;
    uint8_t score;
};

struct bbox_record
{
    int32_t class;
    uint32_t *scores;
    struct bbox *bboxes;
};

struct box
{
    float c1;
    float c2;
    float c3;
    float c4;
};

struct nodeinfo
{
    struct bbox *prior_boxes;
    struct box *prior_variances;
    uint32_t num_priors;
    uint32_t is_variance_encoded;
};

typedef void (*decode_bbox_fp)(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox);

struct parse_data
{
    const uint8_t *loc_data;
    float *dequantized_loc_data;
    uint32_t num;
    uint32_t num_preds_per_class;
    uint32_t num_loc_classes;
    uint32_t share_location;
    struct bbox_record *bbox_records;
    struct bbox *prior_bboxes;
    struct box *prior_variances;
    decode_bbox_fp decode_bbox_funcp;
    const uint8_t *conf_data;
    uint32_t num_classes;
    volatile unsigned current_pos;
    nn_sem_t donesem;
};

struct nms_info
{
    uint32_t num;
    uint32_t background_label_id;
    uint32_t num_classes;
    uint32_t share_location;
    struct bbox_record *bbox_records;
    struct bbox_record_ref *bbox_record_refs;
    float eta;
    float nms_threshold;
    uint32_t keep_top_k;
    struct nn_pqueue *score_pqueues;
    struct nn_pqueue *output_pqueues;
    uint32_t num_scores;
    uint32_t nms_top_k;
    uint8_t score_threshold;
    struct bbox_record_ref *filtered_indices;
    volatile unsigned current_class;
    volatile unsigned thr_id;
    nn_sem_t donesem;
};

struct dequant_info
{
    uint8_t const *inp;
    float *outp;
    unsigned numel;
    unsigned chunk;                // chunks to do at once; always a multiple of 128
    volatile unsigned current_pos; // used to share across threads.
    float qstep;
    int qzero;
    nn_sem_t done_sem;
};

static inline int compare_uint32(const void *a, const void *b)
{
    return (*(uint32_t *)a - *(uint32_t *)b);
}

static inline int compare_box_record_ref(const void *a, const void *b)
{
    return (((struct bbox_record_ref *)a)->score - ((struct bbox_record_ref *)b)->score);
}

static inline void construct_bbox(struct bbox *prior_box, float xmin, float ymin, float xmax, float ymax)
{
    prior_box->xmin = xmin;
    prior_box->ymin = ymin;
    prior_box->xmax = xmax;
    prior_box->ymax = ymax;
    prior_box->width = xmax - xmin;
    prior_box->height = ymax - ymin;
    prior_box->x_center = (xmin + xmax) / 2;
    prior_box->y_center = (ymin + ymax) / 2;
}

static inline void set_bbox_size(struct bbox *bb)
{
    if ((bb->xmax < bb->xmin) || (bb->ymax < bb->ymin))
        bb->size = 0;
    else
    {
        float width = bb->xmax - bb->xmin;
        float height = bb->ymax - bb->ymin;
        bb->size = height * width;
    }
}

//Inlines used for decoding bboxes
static inline void __attribute__((always_inline)) decode_bbox_corner_w_var(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox)
{
    decode_bbox->xmin = prior_box.xmin + bb.xmin;
    decode_bbox->ymin = prior_box.ymin + bb.ymin;
    decode_bbox->xmax = prior_box.xmax + bb.xmax;
    decode_bbox->ymax = prior_box.ymax + bb.ymax;
}

static inline void __attribute__((always_inline)) decode_bbox_corner(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox)
{
    decode_bbox->xmin = prior_box.xmin + prior_variances.c1 * bb.xmin;
    decode_bbox->ymin = prior_box.ymin + prior_variances.c2 * bb.ymin;
    decode_bbox->xmax = prior_box.xmax + prior_variances.c3 * bb.xmax;
    decode_bbox->ymax = prior_box.ymax + prior_variances.c4 * bb.ymax;
}

static inline void __attribute__((always_inline)) decode_bbox_center_size_w_var(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox)
{
    float prior_width = prior_box.width;
    float prior_height = prior_box.height;
    float prior_center_x = prior_box.x_center;
    float prior_center_y = prior_box.y_center;
    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    decode_bbox_center_x = bb.xmin * prior_width + prior_center_x;
    decode_bbox_center_y = bb.ymin * prior_height + prior_center_y;
    decode_bbox_width = fast_exp(bb.xmax) * prior_width; //fast_exp
    decode_bbox_height = fast_exp(bb.ymax) * prior_height;
    decode_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2;
    decode_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2;
    decode_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2;
    decode_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2;
}

static inline void __attribute__((always_inline)) decode_bbox_center_size(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox)
{
    float prior_width = prior_box.width;
    float prior_height = prior_box.height;
    float prior_center_x = prior_box.x_center;
    float prior_center_y = prior_box.y_center;
    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    decode_bbox_center_x = prior_variances.c1 * bb.xmin * prior_width + prior_center_x;
    decode_bbox_center_y = prior_variances.c2 * bb.ymin * prior_height + prior_center_y;
    decode_bbox_width = fast_exp(prior_variances.c3 * bb.xmax) * prior_width;
    decode_bbox_height = fast_exp(prior_variances.c4 * bb.ymax) * prior_height;
    decode_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2;
    decode_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2;
    decode_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2;
    decode_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2;
}

static inline void __attribute__((always_inline)) decode_bbox_corner_size_w_var(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox)
{
    float prior_width = prior_box.width;
    float prior_height = prior_box.height;
    decode_bbox->xmin = prior_box.xmin + bb.xmin * prior_width;
    decode_bbox->ymin = prior_box.ymin + bb.ymin * prior_height;
    decode_bbox->xmax = prior_box.xmax + bb.xmax * prior_width;
    decode_bbox->ymax = prior_box.ymax + bb.ymax * prior_height;
}

static inline void __attribute__((always_inline)) decode_bbox_corner_size(struct nn_graph *nn, struct bbox prior_box, struct box prior_variances, const struct bbox bb, struct bbox *decode_bbox)
{
    float prior_width = prior_box.width;
    float prior_height = prior_box.height;
    decode_bbox->xmin = prior_box.xmin + prior_variances.c1 * bb.xmin * prior_width;
    decode_bbox->ymin = prior_box.ymin + prior_variances.c2 * bb.ymin * prior_height;
    decode_bbox->xmax = prior_box.xmax + prior_variances.c3 * bb.xmax * prior_width;
    decode_bbox->ymax = prior_box.ymax + prior_variances.c4 * bb.ymax * prior_height;
}

static inline float compute_jaccard_overlap(struct bbox b1, struct bbox b2)
{
    if (b2.xmin > b1.xmax || b2.xmax < b1.xmin ||
        b2.ymin > b1.ymax || b2.ymax < b1.ymin)
        return 0.0f;
    else
    {
        float intersect_box_xmin = fmaxf(b1.xmin, b2.xmin);
        float intersect_box_ymin = fmaxf(b1.ymin, b2.ymin);
        float intersect_box_xmax = fminf(b1.xmax, b2.xmax);
        float intersect_box_ymax = fminf(b1.ymax, b2.ymax);
        float intersect_width = intersect_box_xmax - intersect_box_xmin;
        float intersect_height = intersect_box_ymax - intersect_box_ymin;
        if (intersect_height > 0 && intersect_width > 0)
        {
            float intersect_size = intersect_height * intersect_width;
            return intersect_size / (b1.size + b2.size - intersect_size);
        }
        return 0.0f;
    }
}

#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
static inline int __attribute__((unused, always_inline)) get_top_k_score_idx_hvx(struct nn_graph *nn, uint32_t *scores, struct nn_pqueue *pqueue, const uint32_t scores_size, const uint8_t score_threshold, void *vtcm_addr)
{
    uint32_t min_score = (255 << 24) - 1;
    uint32_t validated_score_threshold = (score_threshold << 24) - 1;
    uint32_t validBits = 0;
    uint32_t scores_size_roundup = (scores_size + 31) & -32;
    uint32_t leftovers = scores_size % 32;
    uint32_t total_valid_bits = 0;
    union {
        HVX_Vector vqbitsum;
        uint32_t sqbitsum_u32[32];
    } uu;
    for (uint32_t i = 0; i < scores_size_roundup; i += 32)
    {
        HVX_Vector qcmpmask = Q6_V_vsplat_R(0xFFFFFFFF);
        HVX_Vector vin = *(HVX_Vector *)&scores[i];
        if (i == scores_size_roundup - 32)
            vin = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(leftovers), vin, Q6_V_vzero());
        qcmpmask = q6op_V_vand_QV(qcmpmask, Q6_Q_vcmp_gt_VuwVuw(vin, Q6_V_vsplat_R(validated_score_threshold)));
        if (pqueue->size >= pqueue->capacity)
            qcmpmask = q6op_V_vand_QV(qcmpmask, Q6_Q_vcmp_gt_VuwVuw(vin, Q6_V_vsplat_R(min_score & 0xFF000000)));
        HVX_VectorPred qcmpmask_pred = (HVX_VectorPred)qcmpmask;
        uu.vqbitsum = Q6_Vw_prefixsum_Q(qcmpmask_pred);
        validBits = uu.sqbitsum_u32[31] / sizeof(uint32_t);
        if (validBits)
        {
            Q6_vscatter_QRMVwV(qcmpmask_pred, (uint32_t)vtcm_addr, THREAD_USED_VTCM_SIZE - 4, uu.vqbitsum, vin);
            q6op_scatter_release_A(vtcm_addr);
            vmemcpy_asm((scores + total_valid_bits), ((uint32_t *)vtcm_addr + 1), validBits * sizeof(uint32_t));
            for (uint32_t j = 0; j < validBits; j++)
                nn_pqueue_enqueue(nn, pqueue, &scores[total_valid_bits + i]);
            min_score = *(uint32_t *)pqueue->data[pqueue->size - 1];
            total_valid_bits += validBits;
        }
    }
    return pqueue->size;
}
#endif

static inline int __attribute__((unused, always_inline)) get_top_k_score_idx(struct nn_graph *nn, uint32_t *scores, struct nn_pqueue *pqueue, const uint32_t scores_size, const uint8_t score_threshold)
{
    uint32_t score_threshold_idx = score_threshold << 24;
    uint32_t min_score = 255 << 24;
    for (int i = 0; i < scores_size; i++)
    {
        if (scores[i] >= score_threshold_idx)
        {
            min_score = min_u32(scores[i], min_score);
            if (pqueue->size >= pqueue->capacity && scores[i] < (min_score & 0xFF000000))
                continue;
            nn_pqueue_enqueue(nn, pqueue, &scores[i]);
        }
    }
    return pqueue->size;
}

static void apply_nms_fast(struct nn_graph *nn, void *nms_info)
{
    struct nms_info *info = (struct nms_info *)nms_info;
    uint32_t num_classes = info->num_classes;
    uint32_t background_label_id = info->background_label_id;
    uint32_t share_location = info->share_location;
    uint32_t num = info->num;
    struct bbox_record *bbox_records = info->bbox_records;
    float eta = info->eta;
    float nms_threshold = info->nms_threshold;
    uint32_t num_scores = info->num_scores;
    uint8_t score_threshold = info->score_threshold;
    uint8_t min_score = 255;
    struct bbox_record_ref *filtered_indices = info->filtered_indices;
    unsigned chunk = num_classes / NUM_THREADS;
    unsigned thr_id = __sync_fetch_and_add(&info->thr_id, 1);
    struct nn_pqueue *score_pqueue = &info->score_pqueues[thr_id];
    struct nn_pqueue *output_pqueue = &info->output_pqueues[thr_id];
    //Hvx vtcm based score validation currently unsued
    void *thr_vtcm = (void *)((uint32_t)nn->vtcm_ptr + thr_id * THREAD_USED_VTCM_SIZE);
    (void) thr_vtcm;
    unsigned class_cnt;
    while ((class_cnt = __sync_fetch_and_add(&info->current_class, chunk)), class_cnt < num_classes)
    {
        for (uint32_t c = 0; c < min_i32(chunk, num_classes - class_cnt); c++)
        {
            struct bbox_record_ref *filtered_indices_ptr = &filtered_indices[(class_cnt + c) * output_pqueue->capacity];
            if (c + class_cnt == background_label_id)
                continue;
            uint32_t box_offset = share_location ? 0 : class_cnt + c;
            uint32_t *scores = bbox_records[num * num_classes + class_cnt + c].scores;
            struct bbox *bb = bbox_records[num * num_classes + box_offset].bboxes;
            int i = 0;
            uint32_t num_valid_scores = 0;
            // TODO Debug the hvx version
            //#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
            //num_valid_scores = get_top_k_score_idx_hvx(nn, scores, score_pqueue, num_scores, score_threshold, thr_vtcm);
            //#else
            num_valid_scores = get_top_k_score_idx(nn, scores, score_pqueue, num_scores, score_threshold);
            //#endif
            float adaptive_threshold = nms_threshold;
            for (int j = 0; j < num_valid_scores; j++)
            {
                int keep = 1;
                uint32_t score_idx = *(uint32_t *)nn_pqueue_dequeue(nn, score_pqueue);
                uint32_t idx = score_idx & 0xFFFFFF;
                uint8_t score = score_idx >> 24;
                if (output_pqueue->size >= output_pqueue->capacity && score < min_score)
                    continue;
                for (int k = 0; k < i; k++)
                {
                    if (keep)
                    {
                        const int kept_idx = filtered_indices_ptr[k].idx;
                        float overlap = compute_jaccard_overlap(bb[idx], bb[kept_idx]);
                        keep = overlap <= adaptive_threshold;
                    }
                    else
                        break;
                }
                if (keep)
                {
                    if(i < output_pqueue->capacity) {
                        filtered_indices_ptr[i].batch = num;
                        filtered_indices_ptr[i].class = class_cnt + c;
                        filtered_indices_ptr[i].idx = idx;
                        filtered_indices_ptr[i].score = score;
                        nn_pqueue_enqueue(nn, output_pqueue, (void *)&filtered_indices_ptr[i]);
                        min_score = ((struct bbox_record_ref *)(output_pqueue->data[output_pqueue->size - 1]))->score;
                        i++;
                    } else { /* reach the limit of keep_top_k */
                        break;
                    }
                }
                if (keep && eta < 1 && adaptive_threshold > 0.5)
                    adaptive_threshold *= eta;
            }
            nn_pqueue_vclear(nn, score_pqueue);
        }
    }
    nn_sem_post(&info->donesem);
}

static void do_dequantize(struct nn_graph *nn, void *dequant_info)
{
    struct dequant_info *info = (struct dequant_info *)dequant_info;
    uint8_t const *inp0 = info->inp;
    float *outp0 = info->outp;
    unsigned all_numel = info->numel;
    unsigned chunk = info->chunk;
    unsigned pos;
    while (pos = __sync_fetch_and_add(&info->current_pos, chunk), pos < all_numel)
    {
        uint8_t const *inp = inp0 + pos;
        float *outp = outp0 + pos;
        unsigned numel = min_i32(chunk, all_numel - pos);
        l2fetch(inp, 128, 128, (numel + 127) / 128u);
        hvx_do_dequantize(inp, outp, numel, info->qzero, info->qstep);
    }
    nn_sem_post(&info->done_sem);
}

static decode_bbox_fp determine_bbox_decoder_func(struct nn_graph *nn, const uint32_t code_type, const uint32_t variance_encoded_in_target)
{
    if (code_type == CORNER)
    {
        if (variance_encoded_in_target)
            return decode_bbox_corner_w_var;
        else
            return decode_bbox_corner;
    }
    else if (code_type == CENTER_SIZE)
    {
        if (variance_encoded_in_target)
            return decode_bbox_center_size_w_var;
        else
            return decode_bbox_center_size;
    }
    else if (code_type == CORNER_SIZE)
    {
        if (variance_encoded_in_target)
            return decode_bbox_corner_size_w_var;
        else
            return decode_bbox_corner_size;
    }
    else
    {
        errlog(nn, "Unknown code type specified");
        return NULL;
    }
}

static void parse_input_data(struct nn_graph *nn, void *parse_info)
{
    struct parse_data *pinfo = parse_info;
    const uint32_t num = pinfo->num;
    const uint32_t num_classes = pinfo->num_classes;
    const uint32_t share_location = pinfo->share_location;
    const uint8_t *conf_data = pinfo->conf_data;
    const uint32_t num_loc_classes = pinfo->num_loc_classes;
    decode_bbox_fp decode_bbox_funcp = pinfo->decode_bbox_funcp;
    const uint32_t num_preds_per_class = pinfo->num_preds_per_class;
    struct bbox_record *bbox_records = pinfo->bbox_records;
    struct bbox *prior_bboxes = pinfo->prior_bboxes;
    struct box *prior_variances = pinfo->prior_variances;
    float * dequantized_loc_data = pinfo->dequantized_loc_data;

    struct bbox bb;
    struct bbox_record *rec;
    unsigned p;
    int b = 0;
    int batch_idx = 0;
    while (p = __sync_fetch_and_add(&pinfo->current_pos, 1), p < num * num_preds_per_class)
    {
        while(p - batch_idx >= num_preds_per_class)
        {
            b++;
            if(b >= num) goto done;
            batch_idx += num_preds_per_class;
        }
        dequantized_loc_data = &dequantized_loc_data[b * num_loc_classes * 4];
        conf_data = &conf_data[b * num_preds_per_class * num_classes];

        int start_idx = p * num_loc_classes * 4;
        for (int c = 0; c < num_loc_classes; c++)
        {
            rec = &bbox_records[c];
            bb.xmin = dequantized_loc_data[start_idx + c * 4];
            bb.ymin = dequantized_loc_data[start_idx + c * 4 + 1];
            bb.xmax = dequantized_loc_data[start_idx + c * 4 + 2];
            bb.ymax = dequantized_loc_data[start_idx + c * 4 + 3];
            (decode_bbox_funcp)(nn, prior_bboxes[p], prior_variances[p], bb, &rec->bboxes[p]);

            set_bbox_size(&rec->bboxes[p]);
        }
        int start_score_idx = p * num_classes;
        for (int c = 0; c < num_classes; c++)
        {
            rec = &bbox_records[c];
            rec->class = share_location ? -1 : c;
            rec->scores[p] = (uint32_t)((uint32_t)(conf_data[start_score_idx + c] << 24) + p);
        }
    }
done:
    nn_sem_post(&pinfo->donesem);
}

static int ssd_detection_out_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *loc_tensor = self->inputs[0];
    const uint8_t *loc_data = loc_tensor->data;
    const float loc_min = tensor_get_float(self->inputs[1], 0);
    const float loc_max = tensor_get_float(self->inputs[2], 0);
    const struct tensor *conf_tensor = self->inputs[3];
    const uint8_t *conf_data = conf_tensor->data;
    const float conf_min = tensor_get_float(self->inputs[4], 0);
    const float conf_max = tensor_get_float(self->inputs[5], 0);
    const int32_t num_classes = tensor_get_int32(self->inputs[6], 0);
    int32_t share_location = tensor_get_int32(self->inputs[7], 0);
    const int32_t background_label_id = tensor_get_int32(self->inputs[8], 0);
    const float nms_threshold = tensor_get_float(self->inputs[9], 0);
    const int32_t nms_top_k = tensor_get_int32(self->inputs[10], 0);
    const float eta = tensor_get_float(self->inputs[11], 0);
    const int32_t code_type = tensor_get_int32(self->inputs[12], 0);
    const int32_t keep_top_k = tensor_get_int32(self->inputs[14], 0);

    struct tensor *output_batch = self->outputs[0];
    struct tensor *output_class = self->outputs[1];
    struct tensor *output_conf = self->outputs[2];
    struct tensor *output_conf_min = self->outputs[3];
    struct tensor *output_conf_max = self->outputs[4];
    struct tensor *output_location_xmin = self->outputs[5];
    struct tensor *output_location_ymin = self->outputs[6];
    struct tensor *output_location_xmax = self->outputs[7];
    struct tensor *output_location_ymax = self->outputs[8];

    int32_t *output_batch_data = output_batch->data;
    int32_t *output_class_data = output_class->data;
    uint8_t *output_conf_data = output_conf->data;
    float *output_conf_min_data = output_conf_min->data;
    float *output_conf_max_data = output_conf_max->data;
    float *output_location_xmin_data = output_location_xmin->data;
    float *output_location_ymin_data = output_location_ymin->data;
    float *output_location_xmax_data = output_location_xmax->data;
    float *output_location_ymax_data = output_location_ymax->data;
    const uint8_t conf_zero = quantize_uint8(0, conf_min, conf_max);
	
    memset(output_batch_data, 0, output_batch->shape.batches*output_batch->shape.height*output_batch->shape.width*output_batch->shape.depth * sizeof(uint32_t));
    memset(output_class_data, 0, output_class->shape.batches*output_class->shape.height*output_class->shape.width*output_class->shape.depth * sizeof(int32_t));
    memset(output_conf_data, conf_zero, output_conf->shape.batches*output_conf->shape.height*output_conf->shape.width*output_conf->shape.depth * sizeof(uint8_t));
    memset(output_location_xmin_data, 0, output_location_xmin->shape.batches*output_location_xmin->shape.height*output_location_xmin->shape.width*output_location_xmin->shape.depth * sizeof(float));
    memset(output_location_ymin_data, 0, output_location_ymin->shape.batches*output_location_ymin->shape.height*output_location_ymin->shape.width*output_location_ymin->shape.depth * sizeof(float));
    memset(output_location_xmax_data, 0, output_location_xmax->shape.batches*output_location_xmax->shape.height*output_location_xmax->shape.width*output_location_xmax->shape.depth * sizeof(float));
    memset(output_location_ymax_data, 0, output_location_ymax->shape.batches*output_location_ymax->shape.height*output_location_ymax->shape.width*output_location_ymax->shape.depth * sizeof(float));
    

    *output_conf_min_data = conf_min;
    *output_conf_max_data = conf_max;

    const int32_t num = output_class->shape.batches;
    const int32_t variance_encoded_in_target = tensor_get_int32(self->inputs[15], 0);
    const float confidence_threshold = tensor_get_float(self->inputs[16], 0);
    const uint8_t confidence_threshold_q = quantize_uint8(confidence_threshold, conf_min, conf_max);
    const int32_t num_loc_classes = (share_location) ? 1 : num_classes;
    const uint32_t loc_data_size = loc_tensor->shape.batches * loc_tensor->shape.height * loc_tensor->shape.width * loc_tensor->shape.depth;
    unsigned n_loc_vec = (loc_data_size + 127) / 128u;
    unsigned chunk = 256;
    if (n_loc_vec < 512)
        chunk = (n_loc_vec < 32) ? n_loc_vec : ((n_loc_vec + 1) >> 1);
    float loc_step = 0;
    int loc_qzero = get_qu8_level_size_zero(loc_min, loc_max, &loc_step);
#if 0
//TODO: Debug hvx score validation
#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
    if (nn->vtcm_size < TOTAL_USED_VTCM_SIZE)
        return errlog(nn, "ssd detection out could not get enough VTCM to validate score \n");
#endif
#endif

    struct nodeinfo *info = self->opaque;
    int n_nms_threads = min_i32(NUM_THREADS, num_classes);
    size_t dequatized_loc_size = ALIGN128((size_t)loc_data_size * sizeof(float));
    size_t bbox_record_size = ALIGN128((size_t)num * num_classes * sizeof(struct bbox_record)) +
        (size_t)num * num_classes * ALIGN128((size_t)sizeof(uint32_t) * info->num_priors) +
        (size_t)num * num_classes * ALIGN128((size_t)sizeof(struct bbox) * info->num_priors);
    size_t filtered_scores_size = ALIGN128((size_t)n_nms_threads * nms_top_k * sizeof(uint32_t*));
    size_t filtered_indices_size = ALIGN128((size_t)num_classes * keep_top_k * sizeof(struct bbox_record_ref));
    size_t output_pqueue_size = ALIGN128((size_t)n_nms_threads * keep_top_k * sizeof(struct bbox_record_ref*));
    size_t final_output_size = ALIGN128((size_t)keep_top_k * sizeof(struct bbox_record_ref*));
    size_t required_scratch =  dequatized_loc_size + bbox_record_size + filtered_scores_size + filtered_indices_size + output_pqueue_size + final_output_size;
    if (nn->scratch_size < required_scratch)
    {
        if (nn_scratch_grow(nn, required_scratch))
            return errlog(nn, "need %d bytes scratch for detection out", required_scratch);
        nn_scratch_reset(nn);
    }
    struct bbox_record *bbox_records = (struct bbox_record *)nn_scratch_alloc(nn, num * num_classes * sizeof(struct bbox_record));
    for (int i = 0; i < num * num_classes; i++)
    {
        bbox_records[i].scores = (uint32_t *)nn_scratch_alloc(nn, sizeof(uint32_t) * info->num_priors);
        bbox_records[i].bboxes = (struct bbox *)nn_scratch_alloc(nn, sizeof(struct bbox) * info->num_priors);
    }
    uint32_t* *filtered_scores = (uint32_t **)nn_scratch_alloc(nn, n_nms_threads * nms_top_k * sizeof(uint32_t*));
    struct bbox_record_ref *filtered_indices = (struct bbox_record_ref *)nn_scratch_alloc(nn, num_classes * keep_top_k * sizeof(struct bbox_record_ref));
    struct bbox_record_ref* *output_pqueue_mem = (struct bbox_record_ref* *)nn_scratch_alloc(nn, n_nms_threads * keep_top_k * sizeof(struct bbox_record_ref*));
    struct bbox_record_ref* *final_output_pqueue = (struct bbox_record_ref* *)nn_scratch_alloc(nn, keep_top_k * sizeof(struct bbox_record_ref*));

    //Dequantize deltas
    float *dequantized_locs = (float *)nn_scratch_alloc(nn, loc_data_size * sizeof(float));
    struct parse_data pinfo;
    struct dequant_info dinfo;
    dinfo.inp = loc_data;
    dinfo.numel = loc_data_size;
    dinfo.qzero = loc_qzero;
    dinfo.qstep = loc_step;
    dinfo.outp = dequantized_locs;
    dinfo.chunk = 128 * chunk;
    dinfo.current_pos = 0;
    nn_sem_init(&dinfo.done_sem, 0);
    int n_dequant_threads = (n_loc_vec > (NUM_THREADS - 1) * chunk) ? NUM_THREADS : (n_loc_vec + (chunk - 1)) / chunk;
    for (int i = 0; i < n_dequant_threads; i++)
        nn_os_work_for_vector(nn, do_dequantize, &dinfo);
    nn_sem_wait_n_times(&dinfo.done_sem, n_dequant_threads);

    //Decode bboxes and setup global box record struct
    pinfo.decode_bbox_funcp = determine_bbox_decoder_func(nn, code_type, variance_encoded_in_target);
    pinfo.num = num;
    pinfo.num_preds_per_class = info->num_priors;
    pinfo.num_loc_classes = num_loc_classes;
    pinfo.num_classes = num_classes;
    pinfo.share_location = share_location;
    pinfo.bbox_records = bbox_records;
    pinfo.prior_variances = info->prior_variances;
    pinfo.prior_bboxes = info->prior_boxes;
    pinfo.conf_data = conf_data;
    pinfo.dequantized_loc_data = dequantized_locs;
    pinfo.current_pos = 0;
    int n_parse_threads = min_i32(NUM_THREADS, pinfo.num_preds_per_class);
    nn_sem_init(&pinfo.donesem, 0);
    l2fetch(pinfo.prior_bboxes, 1, 1, pinfo.num_preds_per_class * sizeof(float));
    l2fetch(pinfo.prior_variances, 1, 1, pinfo.num_preds_per_class * sizeof(float));
    l2fetch(pinfo.conf_data, 1, pinfo.num_classes, pinfo.num_preds_per_class);
    for (int i = 0; i < n_parse_threads; i++)
        nn_os_work_for_vector(nn, parse_input_data, &pinfo);
    nn_sem_wait_n_times(&pinfo.donesem, n_parse_threads);

    //Apply NMS
    struct nn_pqueue score_pqueue[n_nms_threads];
    struct nn_pqueue output_pqueue[n_nms_threads];
    for (int i = 0; i < n_nms_threads; i++)
    {
        nn_pqueue_init(nn, &score_pqueue[i], compare_uint32, nms_top_k, &filtered_scores[i * nms_top_k]);
        nn_pqueue_init(nn, &output_pqueue[i], compare_box_record_ref, keep_top_k, &output_pqueue_mem[i * keep_top_k]);
    }
    struct nms_info ninfo;
    ninfo.num_classes = num_classes;
    ninfo.background_label_id = background_label_id;
    ninfo.share_location = share_location;
    ninfo.bbox_records = bbox_records;
    ninfo.eta = eta;
    ninfo.nms_threshold = nms_threshold;
    ninfo.num_scores = info->num_priors;
    ninfo.nms_top_k = nms_top_k;
    ninfo.score_threshold = confidence_threshold_q;
    ninfo.keep_top_k = keep_top_k;
    ninfo.score_pqueues = score_pqueue;
    ninfo.output_pqueues = output_pqueue;
    ninfo.filtered_indices = filtered_indices;
    ninfo.thr_id = 0;
    struct nn_pqueue final_queue;
    nn_pqueue_init(nn, &final_queue, compare_box_record_ref, keep_top_k, final_output_pqueue);
    for (int n = 0; n < num; n++) //traverse batches
    {
        ninfo.num = n;
        ninfo.current_class = 0;
        for (int i = 0; i < n_nms_threads; i++)
            nn_pqueue_clear(nn, &output_pqueue[i]);
        nn_sem_init(&ninfo.donesem, 0);
        for (int i = 0; i < n_nms_threads; i++)
            nn_os_work_for_vector(nn, apply_nms_fast, &ninfo);
        nn_sem_wait_n_times(&ninfo.donesem, n_nms_threads);

        if (keep_top_k > -1)
        {
            int batch_out_head = n * keep_top_k; //head index of the outputs in each batch
            int out_idx = 0;
            struct bbox tmp_box;
            for (int i = 0; i < n_nms_threads; i++)
            {
                struct nn_pqueue tmp = output_pqueue[i];
                while(tmp.size)
                {
                    nn_pqueue_enqueue(nn, &final_queue, (struct bbox_record_ref *)nn_pqueue_dequeue(nn, &tmp));
                }
            }

            uint32_t loops = min_u32(final_queue.size, (uint32_t)keep_top_k);
            for (uint32_t i = 0; i < loops; i++)
            {
                struct bbox_record_ref ref = *(struct bbox_record_ref *)nn_pqueue_dequeue(nn, &final_queue);
                out_idx = batch_out_head + i;
                output_batch_data[out_idx] = n;
                output_class_data[out_idx] = ref.class;
                output_conf_data[out_idx] = ref.score;

                if (share_location)
                    tmp_box = bbox_records[n].bboxes[ref.idx];
                else
                    tmp_box = bbox_records[n * num_classes + ref.class].bboxes[ref.idx]; //
                output_location_xmin_data[out_idx] = tmp_box.xmin;
                output_location_ymin_data[out_idx] = tmp_box.ymin;
                output_location_xmax_data[out_idx] = tmp_box.xmax;
                output_location_ymax_data[out_idx] = tmp_box.ymax;
            }
        }
    }
    return 0;
}

//All done at check
static void get_prior_boxes(const float *prior_data, const uint32_t num_priors, struct bbox *prior_boxes, struct box *prior_variances, const uint32_t is_variance_encoded)
{
    for (int i = 0; i < num_priors; i++)
    {
        int start_idx = i * 4;
        construct_bbox(&prior_boxes[i], prior_data[start_idx], prior_data[start_idx + 1], prior_data[start_idx + 2], prior_data[start_idx + 3]);
    }

    if (!is_variance_encoded)
    {
        const float *variance_offset_ptr = &prior_data[4 * num_priors];
        memcpy(prior_variances, variance_offset_ptr, 4 * num_priors * sizeof(float));
    }
}

static int ssd_detection_out_check(struct nn_node *self, struct nn_graph *nn)
{
    //Process prior boxes here
    struct nodeinfo *info = self->opaque;
    const struct tensor *loc_tensor = self->inputs[0];
    if ((info = nn_malloc(sizeof(*info))) == NULL)
        return errlog(nn, "Can't alloc nodeinfo");
    const struct tensor *prior_tensor = self->inputs[13];
    const struct tensor *variance_encoding_tensor = self->inputs[15];
    const float *prior_data = prior_tensor->data;
    info->is_variance_encoded = tensor_get_int32(variance_encoding_tensor, 0);
    info->num_priors = loc_tensor->shape.depth / 4;
    if ((info->prior_boxes = nn_memalign(128, info->num_priors * sizeof(struct bbox))) == NULL)
        return errlog(nn, "Couldn't allocate memory for prior boxes");
    if ((info->prior_variances = nn_memalign(128, 4 * info->num_priors * sizeof(float))) == NULL)
        return errlog(nn, "Couldn't allocate memory for prior variances");
    get_prior_boxes(prior_data, info->num_priors, info->prior_boxes, info->prior_variances, info->is_variance_encoded);
    self->opaque = info;

    return 0;
}

static int ssd_detection_out_dtor(struct nn_node *self, struct nn_graph *nn)
{
    struct nodeinfo *nodeinfo = self->opaque;
    if (nodeinfo)
    {
        if (nodeinfo->prior_boxes)
            nn_free(nodeinfo->prior_boxes);
        if (nodeinfo->prior_variances)
            nn_free(nodeinfo->prior_variances);
        nn_free(nodeinfo);
    }
    self->opaque = NULL;
    return node_free_common(self, nn);
}

struct nn_node_ops nn_ops_for_SsdDetectionOut = {
    .execute = ssd_detection_out_execute,
    .check = ssd_detection_out_check,
    .ctor = node_alloc_common,
    .dtor = ssd_detection_out_dtor,
    .n_inputs = NN_IOCOUNT(17),
    .n_outputs = NN_IOCOUNT(9),
};
