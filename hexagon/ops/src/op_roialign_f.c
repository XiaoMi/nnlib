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
 * This contains implementations for floating point reference and a quantized roi align node
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>
#define ALIGN_SIZE 128
// These constants must match those in DspRoiPoolingLayer.cpp from SNPE
// No great way to share them yet.
#define OP_ROIALIGN_FEAT_DATA_IDX 0
#define OP_ROIALIGN_ROIS_DATA_IDX 1
#define OP_ROIALIGN_SIZE_DATA_IDX 2
#define OP_ROIALIGN_SPATIAL_SCALE_DATA_IDX 3
#define OP_ROIALIGN_SAMPLING_RATIO_DATA_IDX 4
#define OP_ROIALIGN_NUM_INPUTS 5
#define OP_ROIALIGN_NUM_OUTPUTS 2


// Input 1 is the ROI tensor.  It must be 1x1xnx(4 or 5) where n is the maximum number of rois
#define ROI_TENSOR_BATCHES 1
#define ROI_TENSOR_HEIGHT 1
#define ROI_TENSOR_DEPTH_NO_ID 4
#define ROI_TENSOR_DEPTH 5

#define TOP_LEFT_CORNER_IDX 0
#define TOP_RIGHT_CORNER_IDX 1
#define BOTTOM_LEFT_CORNER_IDX 2
#define BOTTOM_RIGHT_CORNER_IDX 3
#define NUM_CORNERS 4
#define NUM_DIMS 4

#define LOG_LEVEL 2 // 2 is default. Set to 0 if you just want to see everything

// struct to communicate with nn_os_work_for_vector
struct tdata {
	struct nn_node *self;
	nn_sem_t donesem;
	int res;
};
struct pooling_region {
	int corners[NUM_CORNERS];
	float corner_weights[NUM_CORNERS];
};

//static int32_t max (int a, int32_t b) { return((a>b) ? a : b); }
//static int32_t min (int32_t a, int32_t b) {return((a<b)?a:b);}

void pre_calc_for_bilinear_interpolate_f(
		struct nn_graph *nn,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		float roi_start_h,
		float roi_start_w,
		float bin_size_h,
		float bin_size_w,
		int roi_bin_grid_h,
		int roi_bin_grid_w,
		struct pooling_region * pooling_regions) {

	int pooling_region_idx = 0;
	for (int ph = 0; ph< pooled_height; ph++) {
		for (int pw = 0; pw< pooled_width; pw++) {
			for (int iy = 0; iy<roi_bin_grid_h; iy++) {
				const float yy = roi_start_h + ph * bin_size_h + ((float) iy + 0.5f) * bin_size_h / (float) roi_bin_grid_h;
				for (int ix = 0; ix<roi_bin_grid_w; ix++) {
					float x = roi_start_w + pw * bin_size_w + ((float) ix + 0.5f) * bin_size_w / (float) roi_bin_grid_w;
					float y = yy;
					struct pooling_region pr;
					if (y < -1.0f || y > height || x < -1.0f || x > width) {
						memset(&pr, 0, sizeof(pr));
						pooling_regions[pooling_region_idx] = pr;
						pooling_region_idx += 1;
						continue;
					}
					if (y<= 0) {
						y = 0;
					}
					if(x <= 0){
						x = 0;
					}
					int y_low =  (int) y;
					int x_low = (int) x;
					int y_high;
					int x_high;
					if (y_low >= height -1) {
						y_high = y_low = height -1;
						y = (float) y_low;
					}
					else {
						y_high = y_low + 1;
					}
					if (x_low >= width - 1) {
						x_high = x_low = width -1;
						x = (float) x_low;
					}
					else{
						x_high = x_low + 1;
					}
					float w1 = (1.0f - (y - y_low)) * (1.0f - (x - x_low));
					float w2 = (1.0f - (y - y_low)) * (x - x_low);
					float w3 = (1.0f - (x - x_low)) * (y - y_low);
					float w4 = (y - y_low) * (x - x_low);
					pr.corners[TOP_LEFT_CORNER_IDX] = y_low * width + x_low;
					pr.corners[TOP_RIGHT_CORNER_IDX] = y_low * width + x_high;
					pr.corners[BOTTOM_LEFT_CORNER_IDX] = y_high * width + x_low;
					pr.corners[BOTTOM_RIGHT_CORNER_IDX] = y_high * width + x_high;
					pr.corner_weights[TOP_LEFT_CORNER_IDX] = w1;
					pr.corner_weights[TOP_RIGHT_CORNER_IDX] = w2;
					pr.corner_weights[BOTTOM_LEFT_CORNER_IDX] = w3;
					pr.corner_weights[BOTTOM_RIGHT_CORNER_IDX] = w4;
					pooling_regions[pooling_region_idx] = pr;

					pooling_region_idx += 1;

				}
			}
		}
	}
}

static int check_roishape(struct nn_graph *nn, const struct tensor* tens, int logval) {
	int res = 0;
	if (tens->shape.batches != ROI_TENSOR_BATCHES) {
		logmsg(nn, logval, "roialign: ROI tensor batch incorrect %d", tens->shape.batches);
		res = -1;
	}
	if (tens->shape.height != ROI_TENSOR_HEIGHT) {
		logmsg(nn, logval, "roialign: ROI tensor height incorrect %d", tens->shape.height );
		res = -1;
	}
	if (!(tens->shape.depth == ROI_TENSOR_DEPTH || tens->shape.depth == ROI_TENSOR_DEPTH_NO_ID)) {
		logmsg(nn, logval, "roialign: ROI tensor depth incorrect %d", tens->shape.depth );
		res = -1;
	}
	return res;
}

static int roialign_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *feat_tensor = self->inputs[OP_ROIALIGN_FEAT_DATA_IDX];
	const struct tensor *rois_tensor = self->inputs[OP_ROIALIGN_ROIS_DATA_IDX];
	const struct tensor *pool_tensor = self->inputs[OP_ROIALIGN_SIZE_DATA_IDX];
	const struct tensor *spatial_scale_tensor = self->inputs[OP_ROIALIGN_SPATIAL_SCALE_DATA_IDX];
	const struct tensor *sampling_ratio_tensor = self->inputs[OP_ROIALIGN_SAMPLING_RATIO_DATA_IDX];

	// Check that the region of interest tensor, is 1x1xnx(4 or 5)
	if (check_roishape(nn, self->inputs[OP_ROIALIGN_ROIS_DATA_IDX],0)) return errlog(nn, "roialign ROI tensor dimensions wrong");

	const float spatial_scale = tensor_get_float(spatial_scale_tensor, 0);
	const int sampling_ratio = tensor_get_int32(sampling_ratio_tensor, 0);

	// Setup our output_tensors/out_min/out_max properly
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_pre_implode_shape = self->outputs[1];

	int out_batch = rois_tensor->shape.width;
	int out_width = pool_tensor->shape.width;
	int out_height = pool_tensor->shape.height;
	int out_depth = feat_tensor->shape.depth;
	tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
	out_tensor->data_size = out_batch * out_width * out_height * out_depth * sizeof(float);
	if (out_tensor->data_size > out_tensor->max_size) return errlog(nn, "roialign output data size too large");
	if (out_pre_implode_shape->max_size<NUM_DIMS*sizeof(float)){
		return errlog(nn, "roialign shape size too small");
	}
	out_pre_implode_shape->data_size = NUM_DIMS*sizeof(float);
	const int32_t feat_height = feat_tensor->shape.height;
	const int32_t feat_width = feat_tensor->shape.width;
	const int32_t feat_depth = feat_tensor->shape.depth;

	const int32_t pooled_height = pool_tensor->shape.height;
	const int32_t pooled_width = pool_tensor->shape.width;
	const int32_t pooled_depth = feat_depth;
	// ** ceilf(int/int) has no effect **
	const int32_t max_sampling_points = (sampling_ratio > 0) ? sampling_ratio * sampling_ratio : ceilf(feat_height/pooled_height) * ceilf(feat_width/pooled_width);
	size_t data_size = feat_depth * sizeof(float);
	//Allocate space for each corner, summed pooling values, and the maximum number of points that could be sampled during bilinear interpolation
	size_t total_data_size = data_size * (NUM_CORNERS + feat_depth) + max_sampling_points * pooled_height * pooled_width * sizeof(struct pooling_region);
	if(nn_scratch_grow(nn, total_data_size)) {
		return errlog(nn, "Failed to get scratch");
	}
	int pooled_size = pooled_height * pooled_width * pooled_depth;
	float *data1 = nn->scratch;
	float *data2 = (float *) data1 + data_size;
	float *data3 = (float *) data2 + data_size;
	float *data4 = (float *) data3 + data_size;
	float *output_vals= (float *) data4 + data_size;
	struct pooling_region *pooling_regions = (struct pooling_region *)(output_vals + data_size);

	for (int n = 0; n < out_batch; n++) {
		int index_n = n * pooled_size;
		uint32_t offset = n * rois_tensor->shape.depth;
		float roi_start_w = tensor_get_float(rois_tensor, offset +1) * spatial_scale;
		float roi_start_h = tensor_get_float(rois_tensor, offset +2) * spatial_scale;
		float roi_end_w = tensor_get_float(rois_tensor, offset + 3) * spatial_scale;
		float roi_end_h = tensor_get_float(rois_tensor, offset + 4) * spatial_scale;
		float roi_height = (roi_end_h - roi_start_h > 1.0f) ? (roi_end_h - roi_start_h) : 1.0f;
		float roi_width = (roi_end_w - roi_start_w > 1.0f) ? (roi_end_w - roi_start_w) : 1.0f;

		// Divide the ROI into (pooled_height) x (pooled_width) cells, each cell is of size: (cell_size_h) x (cell_size_w)
		const float cell_size_h = (float) roi_height / (float) pooled_height;
		const float cell_size_w = (float) roi_width / (float) pooled_width;
		int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_height / pooled_height);
		int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);
		const uint32_t count = roi_bin_grid_h * roi_bin_grid_w;
		pre_calc_for_bilinear_interpolate_f(nn,
				feat_height,
				feat_width,
				pooled_height,
				pooled_width,
				roi_start_h,
				roi_start_w,
				cell_size_h,
				cell_size_w,
				roi_bin_grid_h,
				roi_bin_grid_w,
				pooling_regions);
		float *feat_data = feat_tensor->data;
		float *dst = out_tensor->data;
		int pooling_region_idx = 0;
		for (int32_t ph = 0; ph < pooled_height; ++ph) {
			for (int32_t pw = 0; pw < pooled_width; ++pw) {
				memset(output_vals, 0, feat_depth * sizeof(float));
				for (int iy = 0; iy < roi_bin_grid_h; iy++) {
					for (int ix = 0; ix < roi_bin_grid_w; ix++) {
						struct pooling_region pr = pooling_regions[pooling_region_idx];
						memcpy(data1, feat_data + feat_depth * pr.corners[TOP_LEFT_CORNER_IDX], feat_depth * sizeof(float));
						memcpy(data2, feat_data + feat_depth * pr.corners[TOP_RIGHT_CORNER_IDX], feat_depth * sizeof(float));
						memcpy(data3, feat_data + feat_depth * pr.corners[BOTTOM_LEFT_CORNER_IDX], feat_depth * sizeof(float));
						memcpy(data4, feat_data + feat_depth * pr.corners[BOTTOM_RIGHT_CORNER_IDX], feat_depth * sizeof(float));

						for (int i = 0; i < feat_depth; i++) {

							output_vals[i] += data1[i] * pr.corner_weights[TOP_LEFT_CORNER_IDX]
											  + data2[i] * pr.corner_weights[TOP_RIGHT_CORNER_IDX]
											  + data3[i] * pr.corner_weights[BOTTOM_LEFT_CORNER_IDX]
											  + data4[i] * pr.corner_weights[BOTTOM_RIGHT_CORNER_IDX];
						}
						pooling_region_idx += 1;
					}
				}
				for (int i = 0; i < feat_depth; i++) {

					output_vals[i] = output_vals[i] / count;

				}
				int index_nhw = index_n + (ph * pooled_width + pw) * pooled_depth;
				memcpy(dst + index_nhw, output_vals, pooled_depth * sizeof(float));
			} // loop pw (output w)
		} // loop ph (output h)
	}
	tensor_set_shape(out_pre_implode_shape, 1, 1, 1, NUM_DIMS);
	tensor_set_float(out_pre_implode_shape, 0, out_batch);
	tensor_set_float(out_pre_implode_shape, 1, out_height);
	tensor_set_float(out_pre_implode_shape, 2, out_width);
	tensor_set_float(out_pre_implode_shape, 3, out_depth);
	return 0;
}

struct nn_node_ops nn_ops_for_RoiAlign_f = {
		.execute = roialign_execute_f,
		.check = NULL,
		.ctor = node_alloc_common,
		.dtor = node_free_common,
		.n_inputs = NN_IOCOUNT(OP_ROIALIGN_NUM_INPUTS),
		.n_outputs = NN_IOCOUNT(OP_ROIALIGN_NUM_OUTPUTS),
};

