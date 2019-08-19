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
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif
#define ALIGN_SIZE 128

#define OP_ROIALIGN_NUM_INPUTS 11
#define OP_ROIALIGN_NUM_OUTPUTS 3

// Input 1 is the ROI tensor.  It must be 1x1xnx4 where n is the maximum number of rois
#define ROI_TENSOR_BATCHES 1
#define ROI_TENSOR_HEIGHT 1
#define ROI_TENSOR_DEPTH_NO_ID 4
#define ROI_TENSOR_DEPTH 4

#define TOP_LEFT_CORNER_IDX 0
#define TOP_RIGHT_CORNER_IDX 1
#define BOTTOM_LEFT_CORNER_IDX 2
#define BOTTOM_RIGHT_CORNER_IDX 3
#define NUM_CORNERS 4
#define NUM_DIMS 4
#define NUM_THREADS 1

#define LOG_LEVEL 2 // 2 is default. Set to 0 if you just want to see everything

#define ALIGN_SIZE 128
#define PADDED_SIZE(size, pad) ((size + pad - 1) & ~(pad - 1))

#if defined(__hexagon__)
typedef long HVX_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN *)(A))
#endif

// struct to communicate with nn_os_work_for_vector
struct tdata
{
	struct nn_node *self;
	int whoami;
	nn_sem_t donesem;
	int start_batch;
	int end_batch;
	const struct tensor *feat_tensor;
	const struct tensor *rois_tensor;
	const struct tensor *batch_indices_tensor;
	const struct tensor *out_tensor;
	int32_t pooled_height;
	int32_t pooled_width;
	const float spatial_h_scale;
	const float spatial_w_scale;
	const int sampling_h_ratio;
	const int sampling_w_ratio;
	struct pooling_region *pooling_regions;
	int32_t *output_vals;
	uint8_t *output_vals_q;
};

struct pooling_region
{
	int corners[NUM_CORNERS];
	uint8_t corner_weights[NUM_CORNERS];
};

static inline void pre_calc_for_bilinear_interpolate(
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
	struct pooling_region *pooling_regions)
{

	int pooling_region_idx = 0;
	for (int ph = 0; ph < pooled_height; ph++)
	{
		for (int pw = 0; pw < pooled_width; pw++)
		{
			for (int iy = 0; iy < roi_bin_grid_h; iy++)
			{
				const float yy = roi_start_h + ph * bin_size_h + ((float)iy + 0.5f) * bin_size_h / (float)roi_bin_grid_h;
				for (int ix = 0; ix < roi_bin_grid_w; ix++)
				{
					float x = roi_start_w + pw * bin_size_w + ((float)ix + 0.5f) * bin_size_w / (float)roi_bin_grid_w;
					float y = yy;
					struct pooling_region pr;
					if (y < -1.0 || y > height || x < -1.0 || x > width)
					{
						memset(&pr, 0, sizeof(pr));
						pooling_regions[pooling_region_idx] = pr;
						pooling_region_idx++;
						continue;
					}
					if (y <= 0)
					{
						y = 0;
					}
					if (x <= 0)
					{
						x = 0;
					}
					int y_low = (int)y;
					int x_low = (int)x;
					int y_high;
					int x_high;
					if (y_low >= height - 1)
					{
						y_high = y_low = height - 1;
						y = (float)y_low;
					}
					else
					{
						y_high = y_low + 1;
					}
					if (x_low >= width - 1)
					{
						x_high = x_low = width - 1;
						x = (float)x_low;
					}
					else
					{
						x_high = x_low + 1;
					}
					float ly = y - y_low;
					float lx = x - x_low;
					float hy = 1 - ly;
					float hx = 1 - lx;
					uint8_t w1 = roundf(hy * hx * 255);
					uint8_t w2 = roundf(hy * lx * 255);
					uint8_t w3 = roundf(ly * hx * 255);
					uint8_t w4 = roundf(ly * lx * 255);
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

static int check_roishape(struct nn_graph *nn, const struct tensor *tens, int logval)
{
	int res = 0;
	if (tens->shape.batches != ROI_TENSOR_BATCHES)
	{
		logmsg(nn, logval, "roialign: ROI tensor batch incorrect %d", tens->shape.batches);
		res = -1;
	}
	if (tens->shape.height != ROI_TENSOR_HEIGHT)
	{
		logmsg(nn, logval, "roialign: ROI tensor height incorrect %d", tens->shape.height);
		res = -1;
	}
	if (!(tens->shape.depth == ROI_TENSOR_DEPTH || tens->shape.depth == ROI_TENSOR_DEPTH_NO_ID))
	{
		logmsg(nn, logval, "roialign: ROI tensor depth incorrect %d", tens->shape.depth);
		res = -1;
	}
	return res;
}


static void doRoiAlignBatch_hvx(struct nn_graph *nn, int start_batch, int end_batch,
								const struct tensor *feat_tensor,
								const struct tensor *rois_tensor,
								const struct tensor *batch_indices_tensor,
								const struct tensor *out_tensor,
								int32_t pooled_height,
								int32_t pooled_width,
								float spatial_h_scale,
								float spatial_w_scale,
								int sampling_h_ratio,
								int sampling_w_ratio,
								struct pooling_region *pooling_regions,
								int32_t *output_vals,
								uint8_t *output_vals_q)
{
	const int32_t feat_height = feat_tensor->shape.height;
	const int32_t feat_width = feat_tensor->shape.width;
	const int32_t feat_depth = feat_tensor->shape.depth;
	int32_t pooled_depth = feat_depth;
	int32_t pooled_size = pooled_height * pooled_width * pooled_depth;
	float stepsize;
	float recip_stepsize;
	//Max value of output_vals is 255*255 (since the weights sum to 255). Divide by range(int32) to determine step size
	float in_level_size = 65025 / 4294967296.0f;
	float out_min_val = 0;
	float out_max_val = in_level_size * 65025;
	uint8_t *data1;
	uint8_t *data2;
	uint8_t *data3;
	uint8_t *data4;

	quantize_adjust_range(
		&out_min_val, &out_max_val,
		&stepsize, &recip_stepsize,
		out_min_val, out_max_val);

	for (int n = start_batch; n < end_batch; n++)
	{
		int index_n = n * pooled_size;
		uint32_t offset = n * rois_tensor->shape.depth;
		int roi_batch_ind = ((int32_t*) batch_indices_tensor->data)[n];

		float roi_start_w = (float) ((uint16_t*) rois_tensor->data)[offset + 0] * spatial_w_scale * .125; 
		float roi_start_h = (float) ((uint16_t*) rois_tensor->data)[offset + 1] * spatial_h_scale * .125;
		float roi_end_w = (float) ((uint16_t*) rois_tensor->data)[offset + 2] * spatial_w_scale * .125;
		float roi_end_h = (float) ((uint16_t*) rois_tensor->data)[offset + 3] * spatial_h_scale * .125;
		float roi_height = (roi_end_h - roi_start_h > 1.0) ? (roi_end_h - roi_start_h) : 1.0;
		float roi_width = (roi_end_w - roi_start_w > 1.0) ? (roi_end_w - roi_start_w) : 1.0;

		// sampling_w_ratio = sampling_w_ratio > 0 ? sampling_w_ratio : ceilf(wStepSize);
		// sampling_h_ratio = sampling_h_ratio > 0 ? sampling_h_ratio : ceilf(hStepSize);

		// Divide the ROI into (pooled_height) x (pooled_width) cells, each cell is of size: (cell_size_h) x (cell_size_w)
		const float cell_size_h = (float)roi_height / ((float)pooled_height);
		const float cell_size_w = (float)roi_width / ((float)pooled_width);
		int roi_bin_grid_h = (sampling_h_ratio > 0) ? sampling_h_ratio : ceilf(roi_height / pooled_height);
		int roi_bin_grid_w = (sampling_w_ratio > 0) ? sampling_w_ratio : ceilf(roi_width / pooled_width);

		uint32_t count = roi_bin_grid_h * roi_bin_grid_w;
		count = Q6_R_sath_R(0x8000 / count);
		pre_calc_for_bilinear_interpolate(
			nn,
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
		uint8_t *feat_data = feat_tensor->data;
		feat_data = feat_data + roi_batch_ind * feat_depth * feat_height * feat_width;
		uint8_t *dst = out_tensor->data;
		int pooling_region_idx = 0;
		for (int32_t ph = 0; ph < pooled_height; ++ph)
		{
			for (int32_t pw = 0; pw < pooled_width; ++pw)
			{
				int index_nhw = index_n + (ph * pooled_width + pw) * pooled_depth;

				uint8_t *out = &dst[index_nhw];
				int32_t *cur_interpolation_bin = output_vals;
				for (int iy = 0; iy < roi_bin_grid_h; iy++)
				{
					for (int ix = 0; ix < roi_bin_grid_w; ix++)
					{
						int32_t *inner_interpolation_bin = cur_interpolation_bin;
						struct pooling_region pr = pooling_regions[pooling_region_idx];
						data1 = feat_data + feat_depth * pr.corners[TOP_LEFT_CORNER_IDX];
						data2 = feat_data + feat_depth * pr.corners[TOP_RIGHT_CORNER_IDX];
						data3 = feat_data + feat_depth * pr.corners[BOTTOM_LEFT_CORNER_IDX];
						data4 = feat_data + feat_depth * pr.corners[BOTTOM_RIGHT_CORNER_IDX];
						l2fetch(data1, 1, PADDED_SIZE(feat_depth, 128), pr.corners[TOP_RIGHT_CORNER_IDX] - pr.corners[TOP_LEFT_CORNER_IDX]);
						l2fetch(data3, 1, PADDED_SIZE(feat_depth, 128), pr.corners[BOTTOM_RIGHT_CORNER_IDX] - pr.corners[BOTTOM_LEFT_CORNER_IDX]);
						for (int i = 0; i < feat_depth; i += 128)
						{
							HVX_Vector data1_vec = vmemu((unsigned char *)(data1));
							HVX_Vector data2_vec = vmemu((unsigned char *)(data2));
							HVX_Vector data3_vec = vmemu((unsigned char *)(data3));
							HVX_Vector data4_vec = vmemu((unsigned char *)(data4));

							//In : b0,b1,b2...
							//In : a0,a1,a2...
							//Out (lo): b0a0,b2a2...
							//Out (hi): b1a1,b3a3...
							HVX_VectorPair ab = Q6_W_vshuff_VVR(data2_vec, data1_vec, 1);

							//In : d0,d1,d2...
							//In : c0,c1,c2...
							//Out (lo): d0c0,d2c2...
							//Out (hi): d1c1,d3c3...
							HVX_VectorPair cd = Q6_W_vshuff_VVR(data4_vec, data3_vec, 1);

							//In : d0c0,d2c2...
							//In : b0a0,b2a2...
							//Out: d0c0b0a0,d4c4b4a4...
							//Out: d2c2b2a2,d6c6b6a6...
							HVX_VectorPair abcd = Q6_W_vshuff_VVR(Q6_V_lo_W(cd), Q6_V_lo_W(ab), 2);

							//In : d1c1,d3c3...
							//In : b1a1,b3a3...
							//Out : d1c1b1a1,d5c5b5a5...
							//Out : d3c3b3a3,d7c7b7a7...
							HVX_VectorPair abcd2 = Q6_W_vshuff_VVR(Q6_V_hi_W(cd), Q6_V_hi_W(ab), 2);

							// In : [0, 4,  8 ... 124]
							// In : [2, 6, 10 ... 126]
							// Out : [ 0,  2,  4, ...  62]
							// Out : [64, 66, 68, ... 126]
							HVX_VectorPair abcd_even = Q6_W_vshuff_VVR(Q6_V_hi_W(abcd), Q6_V_lo_W(abcd), -4);

							// In : [1, 5,  9 ... 125]
							// In : [3, 7, 11 ... 127]
							// Out : [ 1,  3,  5, ...  63]
							// Out : [65, 67, 69, ... 127]
							HVX_VectorPair abcd_odd = Q6_W_vshuff_VVR(Q6_V_hi_W(abcd2), Q6_V_lo_W(abcd2), -4);

							// In : [ 0,  2,  4, ...  62]
							// In : [ 1,  3,  5, ...  63]
							// Out : [ 0,   1,  2, ...  31]
							// Out : [32,  33, 34, ...  63]
							HVX_VectorPair abcd_low = Q6_W_vshuff_VVR(Q6_V_lo_W(abcd_odd), Q6_V_lo_W(abcd_even), -4);

							// In : [64, 66, 68, ... 126]
							// In : [65, 67, 69, ... 127]
							// Out : [64, 65, 66, ...  95]
							// Output : [96, 97, 98, ... 127]
							HVX_VectorPair abcd_high = Q6_W_vshuff_VVR(Q6_V_hi_W(abcd_odd), Q6_V_hi_W(abcd_even), -4);

							// We've got the data all shuffled, now do 4 multiplies to process 128 depths
							// If depth is >= 128, we will always move in such a way that what we're loading will be aligned, so we don't have to waste time doing unaligned loads
							if (feat_depth >= 128)
							{
								HVX_Vector *depth_0_31 = (HVX_Vector *)inner_interpolation_bin;
								HVX_Vector *depth_32_63 = (HVX_Vector *)(inner_interpolation_bin + 32);
								HVX_Vector *depth_64_95 = (HVX_Vector *)(inner_interpolation_bin + 64);
								HVX_Vector *depth_96_127 = (HVX_Vector *)(inner_interpolation_bin + 96);
								*depth_0_31 = Q6_Vuw_vrmpy_VubRub(Q6_V_lo_W(abcd_low), *(int32_t *)pr.corner_weights);

								*depth_32_63 = Q6_Vuw_vrmpy_VubRub(Q6_V_hi_W(abcd_low), *(int32_t *)pr.corner_weights);

								*depth_64_95 = Q6_Vuw_vrmpy_VubRub(Q6_V_lo_W(abcd_high), *(int32_t *)pr.corner_weights);

								*depth_96_127 = Q6_Vuw_vrmpy_VubRub(Q6_V_hi_W(abcd_high), *(int32_t *)pr.corner_weights);
							}
							else
							{
								vmemu((int32_t *)inner_interpolation_bin) = Q6_Vuw_vrmpy_VubRub(Q6_V_lo_W(abcd_low), *(int32_t *)pr.corner_weights);
								vmemu((int32_t *)inner_interpolation_bin + 32) = Q6_Vuw_vrmpy_VubRub(Q6_V_hi_W(abcd_low), *(int32_t *)pr.corner_weights);
								vmemu((int32_t *)inner_interpolation_bin + 64) = Q6_Vuw_vrmpy_VubRub(Q6_V_lo_W(abcd_high), *(int32_t *)pr.corner_weights);
								vmemu((int32_t *)inner_interpolation_bin + 96) = Q6_Vuw_vrmpy_VubRub(Q6_V_hi_W(abcd_high), *(int32_t *)pr.corner_weights);
							}

							data1 += 128;
							data2 += 128;
							data3 += 128;
							data4 += 128;
							inner_interpolation_bin += 128;
						}
						cur_interpolation_bin += feat_depth;
						pooling_region_idx++;
					}
				}
				nn_requantize_i32_to_qu8_hvx(output_vals_q, output_vals, feat_depth * roi_bin_grid_h * roi_bin_grid_w,
											 in_level_size, out_min_val, out_max_val);

				l2fetch(output_vals_q, 1, roi_bin_grid_w * roi_bin_grid_h * feat_depth, 1);
				if ((feat_depth % 128) == 0)
				{
					avgpool_aligned_hvx(out, output_vals_q, feat_depth, roi_bin_grid_w, roi_bin_grid_h, roi_bin_grid_w, count);
				}
				else
				{
					avgpool_nonaligned_hvx(out, output_vals_q, feat_depth, roi_bin_grid_w, roi_bin_grid_h, roi_bin_grid_w, count);
				}
			} // loop pw (output w)
		}	 // loop ph (output h)
	}
}

static void roialign_execute_slice_hvx(struct nn_graph *nn, void *vinfo)
{
	struct tdata *info = vinfo;
	doRoiAlignBatch_hvx(nn, info->start_batch, info->end_batch, info->feat_tensor,
						info->rois_tensor, info->batch_indices_tensor, info->out_tensor, info->pooled_height, info->pooled_width,
						info->spatial_h_scale, info->spatial_w_scale, info->sampling_h_ratio, info->sampling_w_ratio, 
						info->pooling_regions, info->output_vals, info->output_vals_q);
	nn_sem_post(&info->donesem);
}

static int roialign_execute(struct nn_node *self, struct nn_graph *nn,
							void (*roialign_execute_f)(struct nn_graph *self, void *vinfo))
{
	const struct tensor *feat_tensor = self->inputs[0];
	const struct tensor *feat_min_tensor = self->inputs[1];
	const struct tensor *feat_max_tensor = self->inputs[2];

	const struct tensor *rois_tensor = self->inputs[3];
	const struct tensor *batch_indices_tensor = self->inputs[4];

	const struct tensor *pooled_h_tensor = self->inputs[5];
	const struct tensor *pooled_w_tensor = self->inputs[6];
	const struct tensor *spatial_h_scale_tensor = self->inputs[7];
	const struct tensor *spatial_w_scale_tensor = self->inputs[8];
	const struct tensor *sampling_h_ratio_tensor = self->inputs[9];
	const struct tensor *sampling_w_ratio_tensor = self->inputs[10];
	struct tensor *out_tensor = self->outputs[0];

	// Check that the region of interest tensor, is 1x1xnx4
	if (check_roishape(nn, self->inputs[3], 0))
		return errlog(nn, "roialign ROI tensor dimensions wrong");
	const int pooled_h = tensor_get_int32(pooled_h_tensor, 0);
	const int pooled_w = tensor_get_int32(pooled_w_tensor, 0);
	const float spatial_h_scale = tensor_get_float(spatial_h_scale_tensor, 0);
	const float spatial_w_scale = tensor_get_float(spatial_w_scale_tensor, 0);
	const int sampling_h_ratio = tensor_get_int32(sampling_h_ratio_tensor, 0);
	const int sampling_w_ratio = tensor_get_int32(sampling_w_ratio_tensor, 0);

	// Setup our output_tensors/out_min/out_max properly
	int out_batch = rois_tensor->shape.width;
	int out_width = pooled_w;
	int out_height = pooled_h;
	int out_depth = feat_tensor->shape.depth;
	tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
	out_tensor->data_size = out_batch * out_width * out_height * out_depth;
	if (out_tensor->data_size > out_tensor->max_size)
		return errlog(nn, "roialign output data size too large");
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	const int32_t feat_height = feat_tensor->shape.height;
	const int32_t feat_width = feat_tensor->shape.width;
	const int32_t feat_depth = feat_tensor->shape.depth;
	const int32_t max_sampling_points = (sampling_h_ratio * sampling_w_ratio > 0) ? sampling_h_ratio * sampling_w_ratio : (feat_height / (float) pooled_h) * (feat_width / (float) pooled_w);	
	struct pooling_region *pooling_regions;
	int32_t *output_vals;
	uint8_t *output_vals_q = NULL;
	size_t output_vals_size_padded = PADDED_SIZE(feat_depth, ALIGN_SIZE) * max_sampling_points * sizeof(int32_t);
	size_t pooling_regions_size_padded = PADDED_SIZE(max_sampling_points * pooled_h * pooled_w * sizeof(struct pooling_region), ALIGN_SIZE);
	size_t output_vals_q_size_padded = PADDED_SIZE(feat_depth * max_sampling_points * sizeof(uint8_t), ALIGN_SIZE);

	//Allocate space for each corner, summed pooling values, and the maximum number of points that could be sampled during bilinear interpolation
	size_t total_data_size = NUM_THREADS * (output_vals_size_padded + pooling_regions_size_padded + output_vals_q_size_padded);
	if (nn_scratch_grow(nn, total_data_size))
	{
		return errlog(nn, "Failed to get scratch");
	}
	output_vals = (int32_t *)nn->scratch;
	pooling_regions = (struct pooling_region *)(output_vals + NUM_THREADS * output_vals_size_padded);
	output_vals_q = (uint8_t *)(pooling_regions + NUM_THREADS * pooling_regions_size_padded);
	
	int num_valid_batches = rois_tensor->shape.width;
	int num_batches_for_thread = num_valid_batches / NUM_THREADS;
	struct tdata worker0_info = {
		.self = self,
		.whoami = 0,
		.start_batch = 0,
		.end_batch = num_batches_for_thread,
		.feat_tensor = feat_tensor,
		.rois_tensor = rois_tensor,
		.batch_indices_tensor = batch_indices_tensor,
		.out_tensor = out_tensor,
		.pooled_height = pooled_h,
		.pooled_width = pooled_w,
		.spatial_h_scale = spatial_h_scale,
		.spatial_w_scale = spatial_w_scale,
		.sampling_h_ratio = sampling_h_ratio,
		.sampling_w_ratio = sampling_w_ratio,
		.output_vals = output_vals,
		.pooling_regions = pooling_regions,
		.output_vals_q = output_vals_q
	};

	nn_sem_init(&worker0_info.donesem, 0);
	nn_os_work_for_vector(nn, roialign_execute_f, &worker0_info);
	nn_sem_wait(&worker0_info.donesem);
	tensor_copy(out_min_tensor, feat_min_tensor);
	tensor_copy(out_max_tensor, feat_max_tensor);

	return 0;
}

static int roialign_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	return roialign_execute(self, nn, roialign_execute_slice_hvx);
}

struct nn_node_ops nn_ops_for_QuantizedRoiAlignV2_8 = {
	.execute = roialign_execute_hvx,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(OP_ROIALIGN_NUM_INPUTS),
	.n_outputs = NN_IOCOUNT(OP_ROIALIGN_NUM_OUTPUTS),
};
