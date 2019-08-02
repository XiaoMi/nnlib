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
#include <math.h>
#include <quantize.h>

#define OP_ROIPOOL_FEAT_DATA_IDX 0
#define OP_ROIPOOL_ROIS_DATA_IDX 1
#define OP_ROIPOOL_BATCH_SPLIT_DATA_IDX 2
#define OP_ROIPOOL_OUTPUT_HEIGHT_DATA_IDX 3
#define OP_ROIPOOL_OUTPUT_WIDTH_DATA_IDX 4
#define OP_ROIPOOL_HEIGHT_SCALE_DATA_IDX 5
#define OP_ROIPOOL_WIDTH_SCALE_DATA_IDX 6
#define OP_ROIPOOL_LAYOUT_DATA_IDX 7
#define OP_ROIPOOL_FEAT_MIN_IDX  8
#define OP_ROIPOOL_FEAT_MAX_IDX  9
#define OP_ROIPOOL_ROI_SCALE_OFFSET_IDX  10
#define OP_ROIPOOL_NUM_OPS 11

#define LOG_LEVEL 2 // 2 is default. Set to 0 if you just want to see everything

// struct to communicate with nn_os_work_for_vector
struct tdata {
    struct nn_node *self;
    nn_sem_t donesem;
    int res;
};

static int roipool_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
    // extract inputs
    const struct tensor *feat_tensor = self->inputs[OP_ROIPOOL_FEAT_DATA_IDX];
    const struct tensor *rois_tensor = self->inputs[OP_ROIPOOL_ROIS_DATA_IDX];
    const struct tensor *batch_split_tensor = self->inputs[OP_ROIPOOL_BATCH_SPLIT_DATA_IDX];
    const float out_height = tensor_get_float(self->inputs[OP_ROIPOOL_OUTPUT_HEIGHT_DATA_IDX],0);
    const float out_width = tensor_get_float(self->inputs[OP_ROIPOOL_OUTPUT_WIDTH_DATA_IDX],0);
    const float height_scale = tensor_get_float(self->inputs[OP_ROIPOOL_HEIGHT_SCALE_DATA_IDX],0);
    const float width_scale = tensor_get_float(self->inputs[OP_ROIPOOL_WIDTH_SCALE_DATA_IDX],0);
    // the following line is reserved for future implementation involving layout scalar
    // const float layout_scalar = tensor_get_float(self->inputs[OP_ROIPOOL_LAYOUT_DATA_IDX],0);    
    const struct tensor *feat_min_tensor = self->inputs[OP_ROIPOOL_FEAT_MIN_IDX];
    const struct tensor *feat_max_tensor = self->inputs[OP_ROIPOOL_FEAT_MAX_IDX];
    const float  roi_quantization_scale = tensor_get_float(self->inputs[OP_ROIPOOL_ROI_SCALE_OFFSET_IDX],0);
    const float  roi_quantization_offset = tensor_get_float(self->inputs[OP_ROIPOOL_ROI_SCALE_OFFSET_IDX],1);

    // Setup our output_tensors/out_min/out_max properly
    struct tensor *out_tensor = self->outputs[0];
    int out_batch = rois_tensor->shape.batches;
    int out_depth = feat_tensor->shape.depth;
    tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
    out_tensor->data_size = out_batch * out_width * out_height * out_depth;
    if (out_tensor->data_size > out_tensor->max_size) return errlog(nn, "roipool output data size too large");
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    // process and save input data
    const int32_t feat_batches = feat_tensor->shape.batches;
    const int32_t feat_height = feat_tensor->shape.height;
    const int32_t feat_width = feat_tensor->shape.width;
    const int32_t feat_depth = feat_tensor->shape.depth;

    const int32_t num_rois = rois_tensor->shape.batches;
    const int32_t roiinfo_len = rois_tensor->shape.height;

    int32_t* batch_split_data = batch_split_tensor->data;

    uint8_t* feat_data = feat_tensor->data;
    uint8_t* dst = out_tensor->data;
    uint16_t* rois_data = rois_tensor->data;
    uint16_t* rois_data_end = rois_data + num_rois*roiinfo_len;
    uint32_t roiind = 0;
    const uint32_t roi_dim = 4;

    // loop thru each (batch) roi
    for (uint16_t* roiinfo = rois_data; roiinfo < rois_data_end; roiinfo += roi_dim, roiind ++) {
	// fetch batch id for current roi
        int32_t* id_loc = batch_split_data + roiind;
	int32_t batchid = *id_loc;

	// check for invalid input data, checks are compatible with caffe2
	if (batchid<0 || batchid>=feat_batches) {
	    return errlog(nn,"invalid batch id");
	}

	// save quantized roi coordinate inputs
	uint16_t startw_quantized = roiinfo[0];
	uint16_t starth_quantized = roiinfo[1];
	uint16_t endw_quantized = roiinfo[2];
	uint16_t endh_quantized = roiinfo[3];

	// dequantize roi coordinates
	float startw = (startw_quantized - roi_quantization_offset)*roi_quantization_scale;
	float starth = (starth_quantized - roi_quantization_offset)*roi_quantization_scale;
	float endw = (endw_quantized - roi_quantization_offset)*roi_quantization_scale;
	float endh = (endh_quantized - roi_quantization_offset)*roi_quantization_scale;

	// rescale roi coordinates and round roi coordinates to nearest int values
	int32_t roi_start_w = round(startw*(1.0f/width_scale));
	int32_t roi_start_h = round(starth*(1.0f/height_scale));
	int32_t roi_end_w = round(endw*(1.0f/width_scale));
	int32_t roi_end_h = round(endh*(1.0f/height_scale));

	// calculate roi height and width, round to 1 if less than 1 (<1 considered as malformat)
    	int32_t roi_height = max_i32(roi_end_h - roi_start_h + 1, 1);
    	int32_t roi_width = max_i32(roi_end_w - roi_start_w + 1, 1);

    	// kernel size is cell_size_h x cell_size_w
	const float cell_size_h = (float)roi_height / out_height; 
    	const float cell_size_w = (float)roi_width / out_width;

	// batch_base is a pointer for input data at batchid
	uint8_t* batch_base = feat_data + batchid*feat_height*feat_width*feat_depth;

	// kernel goes across the input data
    	for (int32_t ph = 0; ph < out_height; ++ph) {
            for (int32_t pw = 0; pw < out_width; ++pw) {
            	// Compute the pooling region for each cell, and at start and end coordinates are at least 1 apart
            	int32_t h_start = (int32_t)(floorf((float)(ph) * cell_size_h));
            	int32_t h_end = (int32_t)(ceilf((float)(ph + 1) * cell_size_h));
            	int32_t w_start = (int32_t)(floorf((float)(pw) * cell_size_w));
            	int32_t w_end = (int32_t)(ceilf((float)(pw + 1) * cell_size_w));

		// limit the coordinates to be within (0, input height) for h and (0, input width) for w
            	h_start = min_i32(max_i32(h_start + roi_start_h, 0), feat_height);
            	h_end = min_i32(max_i32(h_end + roi_start_h, 0), feat_height);
            	w_start = min_i32(max_i32(w_start + roi_start_w, 0), feat_width);
            	w_end = min_i32(max_i32(w_end + roi_start_w, 0), feat_width);

		// if pooling kernel is out of boundary, output float 0.0 in quantized number
                if(w_end==w_start || h_start==h_end){
                	uint8_t quantized_zero = quantize_uint8(0.0f,tensor_get_float(feat_min_tensor,0),tensor_get_float(feat_max_tensor,0));
              		memset( dst, quantized_zero, out_depth*sizeof(uint8_t) );
              		dst += out_depth;
              		continue;
           	}

		// Max pool in the region
		for (int32_t c = 0; c < out_depth; ++c) {
		    // init all pixels to ~ minimum value
		    // In quantized land, 0 is minimum enough for maxpool op
		    uint8_t v0 = 0;

		    for (int32_t h = h_start; h < h_end; ++h) {
		        uint8_t* poolSrc = batch_base + (h * feat_width + w_start) * feat_depth + c;
		        for (int32_t w = w_start; w < w_end; ++w) {
		            v0 = max_i32(v0, *poolSrc);
		            poolSrc += feat_depth;
		        } // loop w (pool w)
		    } // loop h (pool h)
		    *dst = v0;
		    dst++;
		} // loop c (all channels)

            } // loop pw (output w)
        } // loop ph (output h)
    }

    tensor_copy(out_min_tensor,feat_min_tensor);
    tensor_copy(out_max_tensor,feat_max_tensor);

    return 0;
}



static int roipool_execute_asm_worker(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *feat_tensor = self->inputs[OP_ROIPOOL_FEAT_DATA_IDX];
    const struct tensor *rois_tensor = self->inputs[OP_ROIPOOL_ROIS_DATA_IDX];
    const struct tensor *batch_split_tensor = self->inputs[OP_ROIPOOL_BATCH_SPLIT_DATA_IDX];
    const float out_height = tensor_get_float(self->inputs[OP_ROIPOOL_OUTPUT_HEIGHT_DATA_IDX],0);
    const float out_width = tensor_get_float(self->inputs[OP_ROIPOOL_OUTPUT_WIDTH_DATA_IDX],0);
    const float height_scale = tensor_get_float(self->inputs[OP_ROIPOOL_HEIGHT_SCALE_DATA_IDX],0);
    const float width_scale = tensor_get_float(self->inputs[OP_ROIPOOL_WIDTH_SCALE_DATA_IDX],0);
    // const float layout_scalar = tensor_get_float(self->inputs[OP_ROIPOOL_LAYOUT_DATA_IDX],0);
    const struct tensor *feat_min_tensor = self->inputs[OP_ROIPOOL_FEAT_MIN_IDX];
    const struct tensor *feat_max_tensor = self->inputs[OP_ROIPOOL_FEAT_MAX_IDX];
    const float  roi_quantization_scale = tensor_get_float(self->inputs[OP_ROIPOOL_ROI_SCALE_OFFSET_IDX],0);
    const float  roi_quantization_offset = tensor_get_float(self->inputs[OP_ROIPOOL_ROI_SCALE_OFFSET_IDX],1);

    // Setup our output_tensors/out_min/out_max properly
    struct tensor *out_tensor = self->outputs[0];
    int out_batch = rois_tensor->shape.batches;
    int out_depth = feat_tensor->shape.depth;
    tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
    out_tensor->data_size = out_batch * out_width * out_height * out_depth;
    if (out_tensor->data_size > out_tensor->max_size) return errlog(nn, "roipool output data size too large");
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    const int32_t feat_batches = feat_tensor->shape.batches;
    const int32_t feat_height = feat_tensor->shape.height;
    const int32_t feat_width = feat_tensor->shape.width;
    const int32_t feat_depth = feat_tensor->shape.depth;

    const int32_t num_rois = rois_tensor->shape.batches;
    const int32_t roiinfo_len = rois_tensor->shape.height;

    int32_t* batch_split_data = batch_split_tensor->data;

    uint8_t* feat_data = feat_tensor->data;
    uint8_t* dst = out_tensor->data;
    uint16_t* rois_data = rois_tensor->data;
    uint16_t* rois_data_end = rois_data + num_rois*roiinfo_len;
    uint32_t roiind = 0;
    const uint32_t roi_dim = 4;

    for (uint16_t* roiinfo = rois_data; roiinfo < rois_data_end; roiinfo += roi_dim, roiind ++) {
        int32_t* id_loc = batch_split_data + roiind;
	int32_t batchid = *id_loc;

	// check for invalid input data, checks are compatible with caffe2
	if (batchid<0 || batchid>=feat_batches) {
	    return errlog(nn,"invalid batch id");
	}

	uint16_t startw_quantized = roiinfo[0];
	uint16_t starth_quantized = roiinfo[1];
	uint16_t endw_quantized = roiinfo[2];
	uint16_t endh_quantized = roiinfo[3];

	float startw = (startw_quantized - roi_quantization_offset)*roi_quantization_scale;
        float starth = (starth_quantized - roi_quantization_offset)*roi_quantization_scale;
        float endw = (endw_quantized - roi_quantization_offset)*roi_quantization_scale;
        float endh = (endh_quantized - roi_quantization_offset)*roi_quantization_scale;

	// rescale roi coordinates and round roi coordinates to nearest int values
	int32_t roi_start_w = round(startw*(1.0f/width_scale));
	int32_t roi_start_h = round(starth*(1.0f/height_scale));
	int32_t roi_end_w = round(endw*(1.0f/width_scale));
	int32_t roi_end_h = round(endh*(1.0f/height_scale));

    	int32_t roi_height = max_i32(roi_end_h - roi_start_h + 1, 1);
    	int32_t roi_width = max_i32(roi_end_w - roi_start_w + 1, 1);

	const float cell_size_h = (float)roi_height / out_height;
    	const float cell_size_w = (float)roi_width / out_width;

	uint8_t* batch_base = feat_data + batchid*feat_height*feat_width*feat_depth;

    	for (int32_t ph = 0; ph < out_height; ++ph) {
            for (int32_t pw = 0; pw < out_width; ++pw) {
            	// Compute the pooling region for each cell
            	int32_t h_start = (int32_t)(floorf((float)(ph) * cell_size_h));
            	int32_t h_end = (int32_t)(ceilf((float)(ph + 1) * cell_size_h));
            	int32_t w_start = (int32_t)(floorf((float)(pw) * cell_size_w));
            	int32_t w_end = (int32_t)(ceilf((float)(pw + 1) * cell_size_w));

            	h_start = min_i32(max_i32(h_start + roi_start_h, 0), feat_height);
            	h_end = min_i32(max_i32(h_end + roi_start_h, 0), feat_height);
            	w_start = min_i32(max_i32(w_start + roi_start_w, 0), feat_width);
            	w_end = min_i32(max_i32(w_end + roi_start_w, 0), feat_width);

		// if pooling kernel is out of boundary, output float 0.0 in quantized number
                if(w_end==w_start || h_start==h_end){
                	uint8_t quantized_zero = quantize_uint8(0.0f,tensor_get_float(feat_min_tensor,0),tensor_get_float(feat_max_tensor,0));
              		memset( dst, quantized_zero, out_depth*sizeof(uint8_t) );
              		dst += out_depth;
              		continue;
           	}

            	int32_t input_idx = h_start * feat_depth * feat_width +
                                w_start * feat_depth;
            	uint8_t *in0 = batch_base+input_idx;

            	// From op_maxpool
            	if ((feat_depth % 128) == 0) {
                	 maxpool_aligned_hvx(dst, in0, feat_depth, w_end - w_start, h_end - h_start, feat_width);
            	} else {
                	maxpool_nonaligned_hvx(dst, in0, feat_depth, w_end - w_start, h_end - h_start, feat_width);
            	}
            	dst += out_depth;
            } // loop pw (output w)
        } // loop ph (output h)
    }

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


struct nn_node_ops nn_ops_for_QuantizedRoiPool_8_v2 = {
    .execute = roipool_execute_asm,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ROIPOOL_NUM_OPS),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedRoiPool_8_ref_v2 = {
    .execute = roipool_execute_ref,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ROIPOOL_NUM_OPS),
    .n_outputs = NN_IOCOUNT(3),
};


