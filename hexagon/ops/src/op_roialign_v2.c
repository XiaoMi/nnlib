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
#include <hvx_inlines.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif

#define OP_ROIALIGN_NUM_INPUTS 13
#define OP_ROIALIGN_NUM_OUTPUTS 3

// Input 1 is the ROI tensor.  It must be 1x1xnx4 where n is the maximum number of rois
#define ROI_TENSOR_BATCHES 1
#define ROI_TENSOR_HEIGHT 1
#define ROI_TENSOR_DEPTH_NO_ID 4
#define ROI_TENSOR_DEPTH 4

#ifdef HEXAGON_V66
#define NUM_MAX_THREADS 4
#else
#define NUM_MAX_THREADS 2
#endif

#define LOG_LEVEL 2 // 2 is default. Set to 0 if you just want to see everything

#if defined(__hexagon__)
typedef long HVX_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN *)(A))
#endif

// struct to communicate with nn_os_work_for_vector
struct tdata
{
	struct nn_node *self;
	nn_sem_t donesem;
	int total_num_rois;
	int total_num_jobs;
	int rois_per_job;
	int cur_unit;
	const struct tensor *feat_tensor;
	const int32_t feat_offset;
	const float feat_scale;
	const struct tensor *rois_tensor;
	const struct tensor *batch_indices_tensor;
	const struct tensor *out_tensor;
	const int32_t output_offset;
	const float output_scale;
	int32_t pooled_height;
	int32_t pooled_width;
	const float spatial_h_scale;
	const float spatial_w_scale;
	const int sampling_h_ratio;
	const int sampling_w_ratio;
};

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

static void doRoiAlignBatch_hvx(struct nn_graph *nn, const int n_rois, const int n_jobs, int rois_per_job, int *cur_unit,
								const struct tensor *feat_tensor,
								const int32_t feat_offset,
								const float feat_scale,
								const struct tensor *rois_tensor,
								const struct tensor *batch_indices_tensor,
								const struct tensor *out_tensor,
								const int32_t out_offset,
								const float out_scale,
								int32_t pooled_height,
								int32_t pooled_width,
								float spatial_h_scale,
								float spatial_w_scale,
								int sampling_h_ratio,
								int sampling_w_ratio)
{
	uint8_t *out_data = out_tensor->data;
	uint8_t *feat_data = feat_tensor->data;
	const int32_t feat_height = feat_tensor->shape.height;
	const int32_t feat_width = feat_tensor->shape.width;
	const int32_t feat_depth = feat_tensor->shape.depth;
	const int32_t leftovers = feat_depth % sizeof(HVX_Vector);
	const float wScale = 1.0f / 255.0f;
	int job_id, start_roi_ind, end_roi_ind;
	uint8_t *dst;

	while (job_id = __sync_fetch_and_add(cur_unit, 1), job_id < n_jobs)
	{
		start_roi_ind = job_id * rois_per_job;
		end_roi_ind = min_i32((job_id + 1) * rois_per_job, n_rois);
		dst = out_data + start_roi_ind * feat_depth * pooled_height * pooled_width;

		for (int n = start_roi_ind; n < end_roi_ind; n++)
		{
			uint32_t offset = n * rois_tensor->shape.depth;
			int roi_batch_ind = ((int32_t *)batch_indices_tensor->data)[n];

			float roi_start_w = (float)((uint16_t *)rois_tensor->data)[offset + 0] * spatial_w_scale * 0.125f;
			float roi_start_h = (float)((uint16_t *)rois_tensor->data)[offset + 1] * spatial_h_scale * 0.125f;
			float roi_end_w = (float)((uint16_t *)rois_tensor->data)[offset + 2] * spatial_w_scale * 0.125f;
			float roi_end_h = (float)((uint16_t *)rois_tensor->data)[offset + 3] * spatial_h_scale * 0.125f;

			if (roi_batch_ind < 0 || roi_batch_ind >= n_rois || roi_start_w > feat_width || roi_end_w > feat_width ||
				roi_start_h > feat_height || roi_start_h > feat_height || roi_start_w > roi_end_w || roi_start_h > roi_start_h)
			{
				errlog(nn, "invalid input data");
				return;
			}

			float roi_height = (roi_end_h - roi_start_h > 1.0) ? (roi_end_h - roi_start_h) : 1.0;
			float roi_width = (roi_end_w - roi_start_w > 1.0) ? (roi_end_w - roi_start_w) : 1.0;

			// Divide the ROI into (pooled_height) x (pooled_width) cells, each cell is of size: (cell_size_h) x (cell_size_w)
			const float cell_size_h = (float)roi_height / ((float)pooled_height);
			const float cell_size_w = (float)roi_width / ((float)pooled_width);
			int roi_bin_grid_h = (sampling_h_ratio > 0) ? sampling_h_ratio : (int)ceilf(roi_height / pooled_height);
			int roi_bin_grid_w = (sampling_w_ratio > 0) ? sampling_w_ratio : (int)ceilf(roi_width / pooled_width);

			uint32_t count = roi_bin_grid_h * roi_bin_grid_w;
			float wBinSize = cell_size_w / roi_bin_grid_w;
			float hBinSize = cell_size_h / roi_bin_grid_h;

			float realMultiplier = fminf(0.99999, (feat_scale * wScale / out_scale / (float)count));
			int32_t outputMultiplier = 0;
			int32_t outputShift = 0;
			if (QuantizeMultiplierSmallerThanOne(realMultiplier, &outputMultiplier, &outputShift) == -1)
			{
				errlog(nn,"Cannot determine quantized multiplier for roi align output");
				return;
			}

			int32_t outTemp_size = feat_depth % 128 == 0 ? feat_depth : ((feat_depth / 128 + 1) * 128);

			const uint8_t *batchBase = feat_data + roi_batch_ind * feat_height * feat_width * feat_depth;
			for (int ph = 0; ph < pooled_height; ph++)
			{
				for (int pw = 0; pw < pooled_width; pw++)
				{
					float wStart = cell_size_w * pw + roi_start_w;
					float wEnd = cell_size_w * (pw + 1) + roi_start_w;
					float hStart = cell_size_h * ph + roi_start_h;
					float hEnd = cell_size_h * (ph + 1) + roi_start_h;
					int32_t outTemp[outTemp_size];
					vmemset_asm(outTemp, 0, outTemp_size * sizeof(int32_t));

					for (float y = hStart + hBinSize / 2; y < hEnd; y += hBinSize)
					{
						for (float x = wStart + wBinSize / 2; x < wEnd; x += wBinSize)
						{

							uint32_t x1 = floorf(x);
							uint32_t y1 = floorf(y);
							uint32_t x2 = x1 + 1;
							uint32_t y2 = y1 + 1;
							float dx1 = x - x1;
							float dy1 = y - y1;
							if (y1 >= feat_height - 1)
							{
								y1 = y2 = feat_height - 1;
								dy1 = 0;
							}
							if (x1 >= feat_width - 1)
							{
								x1 = x2 = feat_width - 1;
								dx1 = 0;
							}
							float dx2 = 1.0f - dx1;
							float dy2 = 1.0f - dy1;

							uint8_t ws[] = {saturate_u8(roundf_i32(dx2 * dy2 / wScale)), saturate_u8(roundf_i32(dx1 * dy2 / wScale)), saturate_u8(roundf_i32(dx2 * dy1 / wScale)), saturate_u8(roundf_i32(dx1 * dy1 / wScale))};
							uint32_t offsets[] = {y1 * feat_width * feat_depth + x1 * feat_depth,
												  y1 * feat_width * feat_depth + x2 * feat_depth,
												  y2 * feat_width * feat_depth + x1 * feat_depth,
												  y2 * feat_width * feat_depth + x2 * feat_depth};

							int32_t weight_sum = 0;
							for (int32_t c = 0; c < 4; c++)
							{
								weight_sum += ws[c];
							}
							HVX_Vector weighted_offset = Q6_V_vsplat_R(feat_offset * weight_sum);

							for (uint32_t i = 0; i < feat_depth; i += 128)
							{
								HVX_Vector c1 = vmemu((unsigned char *)(batchBase + offsets[0] + i));
								HVX_Vector c2 = vmemu((unsigned char *)(batchBase + offsets[1] + i));
								HVX_Vector c3 = vmemu((unsigned char *)(batchBase + offsets[2] + i));
								HVX_Vector c4 = vmemu((unsigned char *)(batchBase + offsets[3] + i));

								//In : b0,b1,b2...
								//In : a0,a1,a2...
								//Out (lo): b0a0,b2a2...
								//Out (hi): b1a1,b3a3...
								HVX_VectorPair ab = Q6_W_vshuff_VVR(c2, c1, 1);

								//In : d0,d1,d2...
								//In : c0,c1,c2...
								//Out (lo): d0c0,d2c2...
								//Out (hi): d1c1,d3c3...
								HVX_VectorPair cd = Q6_W_vshuff_VVR(c4, c3, 1);

								//In : d0c0,d2c2...
								//In : b0a0,b2a2...
								//Out: d0c0b0a0,d4c4b4a4...
								//Out: d2c2b2a2,d6c6b6a6...
								HVX_VectorPair abcd_even_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(cd), Q6_V_lo_W(ab), 2);

								//In : d1c1,d3c3...
								//In : b1a1,b3a3...
								//Out : d1c1b1a1,d5c5b5a5...
								//Out : d3c3b3a3,d7c7b7a7...
								HVX_VectorPair abcd_odd_pair = Q6_W_vshuff_VVR(Q6_V_hi_W(cd), Q6_V_hi_W(ab), 2);

								// In : [0, 4,  8 ... 124]
								// In : [2, 6, 10 ... 126]
								// Out : [ 0,  2,  4, ...  62]
								// Out : [64, 66, 68, ... 126]
								HVX_VectorPair abcd_even = Q6_W_vshuff_VVR(Q6_V_hi_W(abcd_even_pair), Q6_V_lo_W(abcd_even_pair), -4);

								// In : [1, 5,  9 ... 125]
								// In : [3, 7, 11 ... 127]
								// Out : [ 1,  3,  5, ...  63]
								// Out : [65, 67, 69, ... 127]
								HVX_VectorPair abcd_odd = Q6_W_vshuff_VVR(Q6_V_hi_W(abcd_odd_pair), Q6_V_lo_W(abcd_odd_pair), -4);

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
								vmemu((int32_t *)(outTemp + i)) = Q6_Vw_vadd_VwVw_sat(vmemu((int32_t *)(outTemp + i)), Q6_Vw_vsub_VwVw_sat(Q6_Vuw_vrmpy_VubRub(Q6_V_lo_W(abcd_low), *(int32_t *)ws), weighted_offset));

								vmemu((int32_t *)(outTemp + i + 32)) = Q6_Vw_vadd_VwVw_sat(vmemu((int32_t *)(outTemp + i + 32)), Q6_Vw_vsub_VwVw_sat(Q6_Vuw_vrmpy_VubRub(Q6_V_hi_W(abcd_low), *(int32_t *)ws), weighted_offset));

								vmemu((int32_t *)(outTemp + i + 64)) = Q6_Vw_vadd_VwVw_sat(vmemu((int32_t *)(outTemp + i + 64)), Q6_Vw_vsub_VwVw_sat(Q6_Vuw_vrmpy_VubRub(Q6_V_lo_W(abcd_high), *(int32_t *)ws), weighted_offset));

								vmemu((int32_t *)(outTemp + i + 96)) = Q6_Vw_vadd_VwVw_sat(vmemu((int32_t *)(outTemp + i + 96)), Q6_Vw_vsub_VwVw_sat(Q6_Vuw_vrmpy_VubRub(Q6_V_hi_W(abcd_high), *(int32_t *)ws), weighted_offset));
							}
						}
					}

					// take average and cast to output quantization
					HVX_Vector out_offset_vec = Q6_V_vsplat_R(out_offset);
					
					HVX_Vector s1 = Q6_V_vzero();
					HVX_Vector s2 = Q6_V_vzero();
					HVX_Vector s3 = Q6_V_vzero();
					HVX_Vector s4 = Q6_V_vzero();
					int k;

					for (k = 0; k + 128 < feat_depth; k += 128)
					{

						HVX_Vector *depth_0_31 = (HVX_Vector *)(outTemp + k);
						HVX_Vector *depth_32_63 = (HVX_Vector *)(outTemp + k + 32);
						HVX_Vector *depth_64_95 = (HVX_Vector *)(outTemp + k + 64);
						HVX_Vector *depth_96_127 = (HVX_Vector *)(outTemp + k + 96);

						s1 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_0_31, outputMultiplier, -outputShift),out_offset_vec);

						s2 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_32_63, outputMultiplier, -outputShift),out_offset_vec);

						s3 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_64_95, outputMultiplier, -outputShift),out_offset_vec);

						s4 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_96_127, outputMultiplier, -outputShift),out_offset_vec);

						s1 = Q6_Vh_vpack_VwVw_sat(s2, s1);
						s2 = Q6_Vh_vpack_VwVw_sat(s4, s3);

						vmemu(&dst[k]) = Q6_Vub_vpack_VhVh_sat(s2, s1); // 16-bit to 8-bit and store in output
					}

					//Process any leftovers
					{

						HVX_Vector *depth_0_31 = (HVX_Vector *)(outTemp + k);
						HVX_Vector *depth_32_63 = (HVX_Vector *)(outTemp + k + 32);
						HVX_Vector *depth_64_95 = (HVX_Vector *)(outTemp + k + 64);
						HVX_Vector *depth_96_127 = (HVX_Vector *)(outTemp + k + 96);

						s1 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_0_31, outputMultiplier, -outputShift),out_offset_vec);
						s2 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_32_63, outputMultiplier, -outputShift),out_offset_vec);
						s3 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_64_95, outputMultiplier, -outputShift),out_offset_vec);
						s4 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(*depth_96_127, outputMultiplier, -outputShift),out_offset_vec);
						s4 = Q6_Vh_vpacke_VwVw(s4, s3); // take upper 16 bits.
						s1 = Q6_Vh_vpacke_VwVw(s2, s1);
						q6op_vstu_variable_ARV(&dst[k], leftovers, Q6_Vub_vpack_VhVh_sat(s4, s1)); // sat to u8
					}

					dst += feat_depth;

				} // loop pw (output w)
			}	 // loop ph (output h)
		}
	}
}

static void roialign_execute_slice_hvx(struct nn_graph *nn, void *vinfo)
{
	struct tdata *info = vinfo;
	doRoiAlignBatch_hvx(nn, info->total_num_rois, info->total_num_jobs, info->rois_per_job, &(info->cur_unit),
						info->feat_tensor, info->feat_offset, info->feat_scale,
						info->rois_tensor, info->batch_indices_tensor, info->out_tensor, info->output_offset,
						info->output_scale, info->pooled_height, info->pooled_width,
						info->spatial_h_scale, info->spatial_w_scale, info->sampling_h_ratio, info->sampling_w_ratio);
	nn_sem_post(&info->donesem);
}

static int roialign_execute(struct nn_node *self, struct nn_graph *nn)
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
	const struct tensor *output_exp_min_tensor = self->inputs[11];
	const struct tensor *output_exp_max_tensor = self->inputs[12];

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
	const float out_exp_min = tensor_get_float(output_exp_min_tensor, 0);
	const float out_exp_max = tensor_get_float(output_exp_max_tensor, 0);

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

	float feat_min = tensor_get_float(feat_min_tensor, 0);
	float feat_max = tensor_get_float(feat_max_tensor, 0);
	float feat_scale = 0.0f;
	float out_scale = 0.0f;
	int feat_offset = get_qu8_level_size_zero(feat_min, feat_max, &feat_scale);
	int out_offset = get_qu8_level_size_zero(out_exp_min, out_exp_max, &out_scale);

	int num_rois = rois_tensor->shape.width;
	const int rois_per_job = 4;
	int n_jobs = (int32_t)ceilf((float)num_rois / (float)rois_per_job);
	int n_threads = min_i32(n_jobs, NUM_MAX_THREADS);

	struct tdata worker0_info = {
		.self = self,
		.total_num_rois = num_rois,
		.total_num_jobs = n_jobs,
		.rois_per_job = rois_per_job,
		.cur_unit = 0,
		.feat_tensor = feat_tensor,
		.feat_offset = feat_offset,
		.feat_scale = feat_scale,
		.rois_tensor = rois_tensor,
		.batch_indices_tensor = batch_indices_tensor,
		.out_tensor = out_tensor,
		.output_offset = out_offset,
		.output_scale = out_scale,
		.pooled_height = pooled_h,
		.pooled_width = pooled_w,
		.spatial_h_scale = spatial_h_scale,
		.spatial_w_scale = spatial_w_scale,
		.sampling_h_ratio = sampling_h_ratio,
		.sampling_w_ratio = sampling_w_ratio,
	};

	nn_sem_init(&worker0_info.donesem, 0);
	for (int32_t tid = 0; tid < n_threads; tid++)
	{
		nn_os_work_for_vector(nn, roialign_execute_slice_hvx, &worker0_info);
	}
	nn_sem_wait_n_times(&worker0_info.donesem, n_threads);

	tensor_copy(out_min_tensor, output_exp_min_tensor);
	tensor_copy(out_max_tensor, output_exp_max_tensor);

	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedRoiAlignV2_8 = {
	.execute = roialign_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(OP_ROIALIGN_NUM_INPUTS),
	.n_outputs = NN_IOCOUNT(OP_ROIALIGN_NUM_OUTPUTS),
};
