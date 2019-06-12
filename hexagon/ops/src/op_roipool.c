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
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains implementations for quantized roi pooling node
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>

// These constants must match those in DspRoiPoolingLayer.cpp from SNPE
// No great way to share them yet.
#define OP_ROIPOOL_FEAT_DATA_IDX 0
#define OP_ROIPOOL_ROIS_DATA_IDX 1
#define OP_ROIPOOL_SIZE_DATA_IDX 2
#define OP_ROIPOOL_SPATIAL_SCALE_DATA_IDX 3
#define OP_ROIPOOL_FEAT_MIN_IDX  4
#define OP_ROIPOOL_FEAT_MAX_IDX  5
#define OP_ROIPOOL_NUM_OPS 6


// Input 1 is the ROI tensor.  It must be 1x1x1x5
#define ROI_TENSOR_BATCHES 1
#define ROI_TENSOR_HEIGHT 1
#define ROI_TENSOR_WIDTH 1
#define ROI_TENSOR_DEPTH 5

#define LOG_LEVEL 2 // 2 is default. Set to 0 if you just want to see everything

// struct to communicate with nn_os_work_for_vector
struct tdata {
    struct nn_node *self;
    nn_sem_t donesem;
    int res;
};

#if defined(__hexagon__)
static int32_t max(int a, int32_t b) { return((a>b) ? a : b); }
static int32_t min(int32_t a, int32_t b) {return((a<b)?a:b);}
#endif


static int check_roishape(struct nn_graph *nn, const struct tensor* tens, int logval) {
   int res = 0;
   if (tens->shape.batches != ROI_TENSOR_BATCHES) {
       logmsg(nn, logval, "roipool: ROI tensor batch incorrect %d", tens->shape.batches);
       res = -1;
   }
   if (tens->shape.height != ROI_TENSOR_HEIGHT) {
       logmsg(nn, logval, "roipool: ROI tensor height incorrect %d", tens->shape.height );
       res = -1;
   }
   if (tens->shape.width != ROI_TENSOR_WIDTH) {
       logmsg(nn, logval, "roipool: ROI tensor width incorrect %d", tens->shape.width);
       res = -1;
   }
   if (tens->shape.depth != ROI_TENSOR_DEPTH) {
       logmsg(nn, logval, "roipool: ROI tensor depth incorrect %d", tens->shape.depth);
       res = -1;
   }
   return res;
}

static int roipool_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *feat_tensor = self->inputs[OP_ROIPOOL_FEAT_DATA_IDX];
    const struct tensor *rois_tensor = self->inputs[OP_ROIPOOL_ROIS_DATA_IDX];
    const struct tensor *pool_tensor = self->inputs[OP_ROIPOOL_SIZE_DATA_IDX];
    const struct tensor *spatial_scale_tensor = self->inputs[OP_ROIPOOL_SPATIAL_SCALE_DATA_IDX];
    const struct tensor *feat_min_tensor = self->inputs[OP_ROIPOOL_FEAT_MIN_IDX];
    const struct tensor *feat_max_tensor = self->inputs[OP_ROIPOOL_FEAT_MAX_IDX];

    // Check that the region of interest tensor, is 1x1x1x5
    if (check_roishape(nn, self->inputs[OP_ROIPOOL_ROIS_DATA_IDX],0)) return errlog(nn, "roipool ROI tensor dimensions wrong");

    const float spatial_scale = tensor_get_float(spatial_scale_tensor, 0);

    // Setup our output_tensors/out_min/out_max properly
    struct tensor *out_tensor = self->outputs[0];
    int out_batch = feat_tensor->shape.batches;
    int out_width = pool_tensor->shape.width;
    int out_height = pool_tensor->shape.height;
    int out_depth = feat_tensor->shape.depth;
    tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
    out_tensor->data_size = out_batch * out_width * out_height * out_depth;
    if (out_tensor->data_size > out_tensor->max_size) return errlog(nn, "roipool output data size too large");
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    const int32_t feat_height = feat_tensor->shape.height;
    const int32_t feat_width = feat_tensor->shape.width;
    const int32_t feat_depth = feat_tensor->shape.depth;

    const int32_t pooled_height = pool_tensor->shape.height;
    const int32_t pooled_width = pool_tensor->shape.width;

    int32_t roi_start_w = round(tensor_get_float(rois_tensor,1) * spatial_scale);
    int32_t roi_start_h = round(tensor_get_float(rois_tensor,2) * spatial_scale);
    int32_t roi_end_w = round(tensor_get_float(rois_tensor,3) * spatial_scale);
    int32_t roi_end_h = round(tensor_get_float(rois_tensor,4) * spatial_scale);

    int32_t roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int32_t roi_width = max(roi_end_w - roi_start_w + 1, 1);

    // Divide the ROI into (pooled_height) x (pooled_width) cells, each cell is of size: (cell_size_h) x (cell_size_w)
    const float cell_size_h = (float)roi_height / (float)pooled_height;
    const float cell_size_w = (float)roi_width / (float)pooled_width;

    uint8_t* feat_data = feat_tensor->data;
    uint8_t* dst = out_tensor->data;
    for (int32_t ph = 0; ph < pooled_height; ++ph) {
        for (int32_t pw = 0; pw < pooled_width; ++pw) {
            // Compute the pooling region for each cell
            int32_t h_start = (int32_t)(floor((float)(ph) * cell_size_h));
            int32_t h_end = (int32_t)(ceil((float)(ph + 1) * cell_size_h));
            int32_t w_start = (int32_t)(floor((float)(pw) * cell_size_w));
            int32_t w_end = (int32_t)(ceil((float)(pw + 1) * cell_size_w));

            h_start = min(max(h_start + roi_start_h, 0), feat_height);
            h_end = min(max(h_end + roi_start_h, 0), feat_height);
            w_start = min(max(w_start + roi_start_w, 0), feat_width);
            w_end = min(max(w_end + roi_start_w, 0), feat_width);


           int32_t is_empty = (h_end <= h_start) || (w_end <= w_start);
           const int32_t pooled_depth = feat_depth;
           if (is_empty) {
              uint8_t quantized_zero = quantize_uint8(0.0f,tensor_get_float(feat_min_tensor,0),tensor_get_float(feat_max_tensor,0));
              memset( dst, quantized_zero, pooled_depth*sizeof(uint8_t) );
              dst += pooled_depth;
              continue;
           }
           // Max pool in the region.
           for (int32_t c = 0; c < pooled_depth; ++c) {
               // init all pixels to ~ minimum value
               // In quantized land, 0 is minimum enough for maxpool op
               uint8_t v0 = 0;

               for (int32_t h = h_start; h < h_end; ++h) {
                   const uint8_t* poolSrc = feat_data + (h * feat_width + w_start) * feat_depth + c;
                   for (int32_t w = w_start; w < w_end; ++w) {
                       v0 = max(v0, *poolSrc);
                       poolSrc += feat_depth;
                   } // loop w (pool w)
               } // loop h (pool h)
               *dst = v0;
               dst++;
           } // loop c (all channels)
       } // loop pw (output w)
    } // loop ph (output h)

    tensor_copy(out_min_tensor,feat_min_tensor);
    tensor_copy(out_max_tensor,feat_max_tensor);
    return 0;
}

static int roipool_execute_asm_worker(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *feat_tensor = self->inputs[OP_ROIPOOL_FEAT_DATA_IDX];
    const struct tensor *rois_tensor = self->inputs[OP_ROIPOOL_ROIS_DATA_IDX];
    const struct tensor *pool_tensor = self->inputs[OP_ROIPOOL_SIZE_DATA_IDX];
    const struct tensor *spatial_scale_tensor = self->inputs[OP_ROIPOOL_SPATIAL_SCALE_DATA_IDX];
    const struct tensor *feat_min_tensor = self->inputs[OP_ROIPOOL_FEAT_MIN_IDX];
    const struct tensor *feat_max_tensor = self->inputs[OP_ROIPOOL_FEAT_MAX_IDX];

    // Check that the region of interest tensor, is 1x1x1x5
    if (check_roishape(nn, self->inputs[OP_ROIPOOL_ROIS_DATA_IDX],0)) return errlog(nn, "roipool ROI tensor dimensions wrong");

    const float spatial_scale = tensor_get_float(spatial_scale_tensor, 0);

    // Setup our output_tensors/out_min/out_max properly
    struct tensor *out_tensor = self->outputs[0];
    int out_batch = feat_tensor->shape.batches;
    int out_width = pool_tensor->shape.width;
    int out_height = pool_tensor->shape.height;
    int out_depth = feat_tensor->shape.depth;
    tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
    out_tensor->data_size = out_batch * out_width * out_height * out_depth;
    if (out_tensor->data_size > out_tensor->max_size) return errlog(nn, "roipool output data size too large");
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    const int32_t feat_height = feat_tensor->shape.height;
    const int32_t feat_width = feat_tensor->shape.width;
    const int32_t feat_depth = feat_tensor->shape.depth;

    const int32_t pooled_height = pool_tensor->shape.height;
    const int32_t pooled_width = pool_tensor->shape.width;

    int32_t roi_start_w = round(tensor_get_float(rois_tensor,1) * spatial_scale);
    int32_t roi_start_h = round(tensor_get_float(rois_tensor,2) * spatial_scale);
    int32_t roi_end_w = round(tensor_get_float(rois_tensor,3) * spatial_scale);
    int32_t roi_end_h = round(tensor_get_float(rois_tensor,4) * spatial_scale);

    int32_t roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int32_t roi_width = max(roi_end_w - roi_start_w + 1, 1);

    // Divide the ROI into (pooled_height) x (pooled_width) cells, each cell is of size: (cell_size_h) x (cell_size_w)
    const float cell_size_h = (float)roi_height / (float)pooled_height;
    const float cell_size_w = (float)roi_width / (float)pooled_width;
    uint8_t quantized_zero = quantize_uint8(0.0f,tensor_get_float(feat_min_tensor,0),tensor_get_float(feat_max_tensor,0));

    uint8_t* feat_data = feat_tensor->data;
    uint8_t* dst = out_tensor->data;
    for (int32_t ph = 0; ph < pooled_height; ++ph) {
        for (int32_t pw = 0; pw < pooled_width; ++pw) {
            // Compute the pooling region for each cell
            int32_t h_start = (int32_t)(floor((float)(ph) * cell_size_h));
            int32_t h_end = (int32_t)(ceil((float)(ph + 1) * cell_size_h));
            int32_t w_start = (int32_t)(floor((float)(pw) * cell_size_w));
            int32_t w_end = (int32_t)(ceil((float)(pw + 1) * cell_size_w));

            h_start = min(max(h_start + roi_start_h, 0), feat_height);
            h_end = min(max(h_end + roi_start_h, 0), feat_height);
            w_start = min(max(w_start + roi_start_w, 0), feat_width);
            w_end = min(max(w_end + roi_start_w, 0), feat_width);

            int32_t is_empty = (h_end <= h_start) || (w_end <= w_start);
            const int32_t pooled_depth = feat_depth;
            if (is_empty) {
              memset( dst, quantized_zero, pooled_depth*sizeof(uint8_t) );
              dst += pooled_depth;
              continue;
            }

            int32_t input_idx = h_start * feat_depth * feat_width +
                                w_start * feat_depth;
            uint8_t *in0 = &feat_data[input_idx];
            // From op_maxpool
            if ((feat_depth % 128) == 0) {
                 maxpool_aligned_hvx(dst, in0, feat_depth, w_end - w_start, h_end - h_start, feat_width);
            } else {
                maxpool_nonaligned_hvx(dst, in0, feat_depth, w_end - w_start, h_end - h_start, feat_width);
            }
            dst += pooled_depth;
        } // loop pw (output w)
    } // loop ph (output h)
    tensor_copy(out_min_tensor,feat_min_tensor);
    tensor_copy(out_max_tensor,feat_max_tensor);
    return 0;
}

static void roipool_execute_asm_worker_wrapper(struct nn_graph *nn, void *vtdata) {
    struct tdata *td = vtdata;
    td->res = roipool_execute_asm_worker(td->self, nn);
    nn_sem_post(&td->donesem);
}

static int roipool_execute_asm(struct nn_node *self, struct nn_graph *nn)
{
    struct tdata td;
    td.self = self;
    td.res = 0;

    nn_sem_init(&td.donesem,0);
    nn_os_work_for_vector(nn,roipool_execute_asm_worker_wrapper,&td);
    nn_sem_wait(&td.donesem);

    return td.res;
}


struct nn_node_ops nn_ops_for_QuantizedRoiPool_8 = {
    .execute = roipool_execute_asm,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ROIPOOL_NUM_OPS),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedRoiPool_8_ref = {
    .execute = roipool_execute_ref,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ROIPOOL_NUM_OPS),
    .n_outputs = NN_IOCOUNT(3),
};


