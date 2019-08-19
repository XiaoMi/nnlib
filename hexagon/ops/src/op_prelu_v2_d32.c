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
 * This contains implementations for quantized Prelu node
 */
#include <nn_graph.h>
#include <quantize.h>
#include <string.h>
#if defined(USE_OS_LINUX)
#include <malloc.h>
#endif
// new implementation
#include "hvx_inlines.h"

#ifdef HEXAGON_V66
#define MAX_THREADS 4
#else
#define MAX_THREADS 2
#endif

struct scale_parms
{
	float in_min;
	float in_max;
	float alpha_min;
	float alpha_max;
	float out_min;
	float out_max;
	int16_t in_qzero;
	int16_t alpha_qzero;
	int32_t pos_scale;
	int32_t neg_scale;
	int32_t pos_lsh;
	int32_t neg_lsh;
	int32_t out_offset;
};

struct tdata
{
	struct nn_node *self;
	struct shape opshape;
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	struct scale_parms scaling;
	nn_sem_t init_done;
	int work_units;
	volatile int next_work;
	// each batch is sliced into work units of size h_per_slice
	int h_per_slice;	 // # of height units per slice.
	int slice_per_batch; // # of slices per batch.
	int16_t *alphabuf;
	uint8_t *scratch;
	int32_t d32_iters;
	nn_sem_t donesem;
};

struct nodeinfo
{
	int16_t *alphabuf;
	int alpha_depth;
	float alpha_min, alpha_max; // the actual range
	int32_t prev_num_alphas;
	int preprocessed_alphas;
};

struct alpha_info
{
	struct tensor_addressing tin;
	uint8_t * alphas;
	float alpha_min;
	float alpha_max;
	uint32_t alpha_depth;
	struct nodeinfo * nodeinfo;
	nn_sem_t donesem;
};

//Used to duplicate alpha values
unsigned char alpha_controls[] __attribute__((aligned(64))) = {
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,
0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,
0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
};

/*
* Prelu with quantized alphas works as follows:
*	1) Subtract alpha_offset from each alpha (result in 16 bits, -255,255, usually done at prep time)
*	2) Subtract input_offset from each input (result in 16 bits, -255,255)
*	3) Dependent on input value, mulitply by result from 1) (result in 32 bits)
*	4) Multiply by a set of 32 bit scale factors which are determined as below.
*	   Each scale is either a "positive scale" for all inputs that were positive or
*	   a "negative scale" for inputs that were negative, which takes into account the alpha scale
*	5) Possibly left shift the result (how we determine this left shift is described below)
*   6) Add output offset
*	7) Pack/Saturate to bytes
*
* Scaling factors (and shift) are determined as follows:
*	1) Compute input scale / output scale as a fraction with 31 - lsh fractional bits (positive scale)
*	2) Compute (input scale / output scale) * alpha scale as a fraction with 31 - lsh fractional bits (negative scale)
*	3) Each scale factor has an associated left shift, that will be used when we have gain > 1.0
*
* The alphas (after subtracted alpha offset) are stored as 2 vectors of i16 per d32 slice.
*
*/

static int
set_scaling(struct nn_graph *nn, struct scale_parms *sp,
			float in_min, float in_max,
			float alpha_min, float alpha_max,
			float out_min, float out_max)
{
	float in_scale;
	float alpha_scale;
	float out_scale;
	int in_offset = get_qu8_level_size_zero(in_min, in_max, &in_scale);
	int alpha_offset = get_qu8_level_size_zero(alpha_min, alpha_max, &alpha_scale);
	int out_offset = get_qu8_level_size_zero(out_min, out_max, &out_scale);
	float pos_scale = in_scale / out_scale;
	float neg_scale = pos_scale * alpha_scale;
	int pos_lsh = (pos_scale <= 1.0) ? 0 : flt_getexp(pos_scale);
	int32_t pos_post_scale = roundf_i32(flt_ldexp(pos_scale, 31-pos_lsh));
	int neg_lsh = (neg_scale <= 1.0) ? 0 : flt_getexp(neg_scale);
	int neg_post_scale = roundf_i32(flt_ldexp( neg_scale, 31-neg_lsh));
	sp->in_qzero = in_offset;
	sp->alpha_qzero = alpha_offset;
	sp->pos_scale = pos_post_scale;
	sp->neg_scale = neg_post_scale;
	sp->pos_lsh = pos_lsh;
	sp->neg_lsh = neg_lsh;
	sp->out_offset = out_offset;
	return 0;
}

static inline HVX_Vector
hvx_process_prelu_pos(HVX_Vector vin,
				 int32_t in_qzero,
				 int32_t pos_gain,
				 int32_t out_offset,
				 int pos_lsh) 
{
	HVX_Vector vinqz = q6op_Vh_vsplat_R(in_qzero);
	HVX_Vector vpos = Q6_V_vsplat_R(pos_gain);
	HVX_Vector pos_voutoff = Q6_V_vsplat_R(out_offset);
	HVX_VectorPair in_x128 = Q6_Wuh_vunpack_Vub(vin);
	HVX_Vector x_0_h_lo = Q6_Vh_vsub_VhVh(Q6_V_lo_W(in_x128), vinqz);
	HVX_Vector x_0_h_hi = Q6_Vh_vsub_VhVh(Q6_V_hi_W(in_x128), vinqz);
	HVX_VectorPair x_0_w = Q6_Ww_vunpack_Vh(x_0_h_lo);
	HVX_VectorPair x_1_w = Q6_Ww_vunpack_Vh(x_0_h_hi);
		
	HVX_Vector_x4 result;
	result.val[0] = Q6_V_lo_W(x_0_w);
	result.val[1] = Q6_V_hi_W(x_0_w);
	result.val[2] = Q6_V_lo_W(x_1_w);
	result.val[3] = Q6_V_hi_W(x_1_w);

	if(pos_lsh > 0) {
		result.val[0] = Q6_Vw_vasl_VwR(result.val[0], pos_lsh);
		result.val[1] = Q6_Vw_vasl_VwR(result.val[1], pos_lsh);
		result.val[2] = Q6_Vw_vasl_VwR(result.val[2], pos_lsh);
		result.val[3] = Q6_Vw_vasl_VwR(result.val[3], pos_lsh);
	}
	result.val[0] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[0], vpos), pos_voutoff);
	result.val[1] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[1], vpos), pos_voutoff);
	result.val[2] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[2], vpos), pos_voutoff);
	result.val[3] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[3], vpos), pos_voutoff);

	result.val[3] = Q6_Vh_vpack_VwVw_sat( result.val[3], result.val[2]);
	result.val[1] = Q6_Vh_vpack_VwVw_sat( result.val[1], result.val[0]);
	return Q6_Vub_vpack_VhVh_sat( result.val[3], result.val[1]);	// sat to u8
}


static inline HVX_Vector
hvx_process_prelu_neg(HVX_Vector vin, HVX_Vector valpha0, HVX_Vector valpha1,
				 int16_t in_qzero,
				 int32_t neg_gain,
				 int32_t out_offset,
				 int32_t neg_lsh
				)
{
	HVX_Vector vinqz = q6op_Vh_vsplat_R(in_qzero);
	HVX_Vector vneg = Q6_V_vsplat_R(neg_gain);
	HVX_Vector neg_voutoff = Q6_V_vsplat_R(out_offset);
	HVX_VectorPair in_x128 = Q6_Wuh_vunpack_Vub(vin);
	HVX_Vector x_0_h_lo = Q6_Vh_vsub_VhVh(Q6_V_lo_W(in_x128), vinqz);
	HVX_Vector x_0_h_hi = Q6_Vh_vsub_VhVh(Q6_V_hi_W(in_x128), vinqz);
	//Mul alphas in here
	HVX_VectorPair x_0_w = Q6_Ww_vmpy_VhVh(x_0_h_lo, valpha0);
	HVX_VectorPair x_1_w = Q6_Ww_vmpy_VhVh(x_0_h_hi, valpha1);
	x_0_w = Q6_W_vshuff_VVR(Q6_V_hi_W(x_0_w), Q6_V_lo_W(x_0_w), -4);
	x_1_w = Q6_W_vshuff_VVR(Q6_V_hi_W(x_1_w), Q6_V_lo_W(x_1_w), -4);

	HVX_Vector_x4 result;
	result.val[0] = Q6_V_lo_W(x_0_w);
	result.val[1] = Q6_V_hi_W(x_0_w);
	result.val[2] = Q6_V_lo_W(x_1_w);
	result.val[3] = Q6_V_hi_W(x_1_w);
	if(neg_lsh > 0) {
		result.val[0] = Q6_Vw_vasl_VwR(result.val[0], neg_lsh);
		result.val[1] = Q6_Vw_vasl_VwR(result.val[1], neg_lsh);
		result.val[2] = Q6_Vw_vasl_VwR(result.val[2], neg_lsh);
		result.val[3] = Q6_Vw_vasl_VwR(result.val[3], neg_lsh);
	}
	result.val[0] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[0], vneg), neg_voutoff);
	result.val[1] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[1], vneg), neg_voutoff);
	result.val[2] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[2], vneg), neg_voutoff);
	result.val[3] = Q6_Vw_vadd_VwVw_sat(q6op_Vw_vmpy_VwVw_s1_rnd_sat(result.val[3], vneg), neg_voutoff);

	result.val[3] = Q6_Vh_vpack_VwVw_sat( result.val[3], result.val[2]);
	result.val[1] = Q6_Vh_vpack_VwVw_sat( result.val[1], result.val[0]);
	return Q6_Vub_vpack_VhVh_sat( result.val[3], result.val[1]);	// sat to u8
}

static inline HVX_Vector
hvx_process_prelu(HVX_Vector vin, HVX_Vector valpha0, HVX_Vector valpha1,
				 int16_t in_qzero,
				 int32_t pos_gain,
				 int32_t neg_gain,
				 int32_t out_offset,
				 int pos_lsh,
				 int neg_lsh)
{
	HVX_Vector vinqz_b =  q6op_Vb_vsplat_R(in_qzero);
	HVX_Vector pos_vec = hvx_process_prelu_pos(vin, in_qzero, pos_gain, out_offset, pos_lsh);
	HVX_Vector neg_vec = hvx_process_prelu_neg(vin, valpha0, valpha1, in_qzero, neg_gain, out_offset, neg_lsh);
	HVX_VectorPred pred = Q6_Q_vcmp_eq_VbVb(vin, vinqz_b);
    pred = Q6_Q_vcmp_gtor_QVubVub(pred, vin, vinqz_b);
	return Q6_V_vmux_QVV(pred, pos_vec, neg_vec);
}

static void prelu_hvx_work_func(struct nn_graph *nn, void *vinfo)
{
	// scale the alphas...
	struct tdata *td = vinfo;
	int nd32 = td->tin.nd32;
	int work_unit_index = __sync_fetch_and_add(&td->next_work, 1);
	HVX_Vector const *alphas = (HVX_Vector const *)td->alphabuf;
	uint8_t const *inp0 = td->tin.data;
	uint8_t *outp0 = td->tout.data;
	int batches = td->opshape.batches;
	int height = td->opshape.height;
	int widvecs = td->d32_iters;

	uint32_t in_d32_stride = td->tin.d32_stride;

	uint32_t in_batch_stride = td->tin.batch_stride;
	uint32_t in_height_stride = td->tin.height_stride;
	uint32_t out_batch_stride = td->tout.batch_stride;
	uint32_t out_height_stride = td->tout.height_stride;
	uint32_t out_d32_stride = td->tout.d32_stride;

	int16_t in_qzero = td->scaling.in_qzero;
	int32_t pos_gain = td->scaling.pos_scale;
	int32_t neg_gain = td->scaling.neg_scale;
	int32_t out_offset = td->scaling.out_offset;
	int32_t pos_lsh = td->scaling.pos_lsh;
	int32_t neg_lsh = td->scaling.neg_lsh;
	int slice_per_batch = td->slice_per_batch;
	int h_per_slice = td->h_per_slice;

	int b = 0;
	int batch_workindex = 0; // always = b*slice_per_batch
	int work_units = td->work_units;
	// work unit loop.
	while (work_unit_index < work_units)
	{
		// figure out what batch and height range we are at.
		while ((work_unit_index - batch_workindex) >= slice_per_batch)
		{
			// need to be in the next batch
			b++;
			if (b >= batches)
				goto done;
			batch_workindex += slice_per_batch;
		}
		int h_base = (work_unit_index - batch_workindex) * h_per_slice;
		int h_count = min_i32(height - h_base, h_per_slice);
		uint8_t const *inp_b = inp0 + b * in_batch_stride + h_base * in_height_stride;
		l2fetch(inp_b, in_d32_stride, 128 * widvecs, h_count * nd32);
		uint8_t *outp_b = outp0 + b * out_batch_stride + h_base * out_height_stride;
		for (int h = 0; h < h_count; h++)
		{
			uint8_t const *inp = inp_b + h * in_height_stride;
			uint8_t *outp = outp_b + h * out_height_stride;
			HVX_Vector const *alpha_p = alphas;
			for (int id32 = 0; id32 < nd32; id32++)
			{
				HVX_Vector alph0 = *alpha_p++;
				HVX_Vector alph1 = *alpha_p++;
				HVX_Vector const *vinp = (HVX_Vector const *)inp;
				HVX_Vector *voutp = (HVX_Vector *)outp;
				for (int i = 0; i < widvecs; i++)
				{
					voutp[i] = hvx_process_prelu(vinp[i], alph0, alph1, in_qzero, pos_gain, neg_gain, out_offset, pos_lsh, neg_lsh);
				}
				inp += in_d32_stride;
				outp += out_d32_stride;
			}
		}
		// get next work unit...
		work_unit_index = __sync_fetch_and_add(&td->next_work, 1);
	}
done:
	nn_sem_post(&td->donesem);
}

static void process_alphas(struct nn_graph *nn, void *vinfo)
{
	struct alpha_info *info = vinfo;
	struct tensor_addressing tin = info->tin;
	const uint8_t * alphas = info->alphas;
	struct nodeinfo *nodeinfo = info->nodeinfo;
	int32_t alpha_depth = info->alpha_depth;
	int32_t alpha_depth_roundup = (alpha_depth + 31) & ~31;
	int16_t *alpha_frac_buf = NULL;

	if (nodeinfo == NULL)
	{
		if ((nodeinfo = nn_malloc(sizeof(*nodeinfo))) == NULL)
		{
			errlog(nn, "can't alloc nodeinfo");
			return;
		}
		if ((alpha_frac_buf = nn_memalign(128, alpha_depth_roundup * 4 * 2)) == NULL)
		{
			nn_free(nodeinfo);
			errlog(nn, "can't allocate alpha buf");
			return;
		}
		nodeinfo->alphabuf = alpha_frac_buf;
		nodeinfo->prev_num_alphas = 0;
		nodeinfo->preprocessed_alphas = 0;
	}
	else
	{
		alpha_frac_buf = nodeinfo->alphabuf;
	}
	nodeinfo->alpha_min = info->alpha_min;
	nodeinfo->alpha_max = info->alpha_max;

	if (0 == nodeinfo->prev_num_alphas)
	{
		nodeinfo->prev_num_alphas = alpha_depth;
	}
	if (alpha_depth != nodeinfo->prev_num_alphas)
	{
		errlog(nn, "Changing the number of alphas between executes is currently unsupported. ");
		return;
	}
	nodeinfo->prev_num_alphas = alpha_depth;
	nodeinfo->alpha_depth = alpha_depth;
	float alpha_scale;
	int alpha_offset = get_qu8_level_size_zero(nodeinfo->alpha_min, nodeinfo->alpha_max, &alpha_scale);
	HVX_Vector * outp = (HVX_Vector *)nodeinfo->alphabuf;
	HVX_Vector alpha_voff = q6op_Vb_vsplat_R(alpha_offset);
	int d32_slice_depth = min_i32(alpha_depth, 32);
	HVX_Vector alpha_offset_h = q6op_Vh_vsplat_R(alpha_offset);

	HVX_VectorPred depth_mask = Q6_Q_vsetq_R(d32_slice_depth);

	for (int i = 0; i < tin.nd32; i++)
	{
		HVX_Vector valphas = q6op_V_vldu_A((HVX_Vector *)&alphas[i * d32_slice_depth]);
		HVX_Vector vin = Q6_V_vmux_QVV(depth_mask, valphas, alpha_voff);
		vin =  Q6_V_vrdelta_VV(vin, *(HVX_Vector const *)alpha_controls);
		HVX_VectorPair vin_x2 = Q6_Wuh_vunpack_Vub(vin);
		*outp++ =  Q6_Vh_vsub_VhVh(Q6_V_lo_W(vin_x2), alpha_offset_h);
		*outp++ =  Q6_Vh_vsub_VhVh(Q6_V_hi_W(vin_x2), alpha_offset_h);
	}
	nodeinfo->preprocessed_alphas = 1;
	info->nodeinfo = nodeinfo;
	nn_sem_post(&info->donesem);
}

static int prelu_opt_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct nodeinfo *nodeinfo = self->opaque;
	const struct tensor *in_tensor = self->inputs[0];
	if(NULL == nodeinfo || 0 == nodeinfo->preprocessed_alphas)
	{
		struct alpha_info info;
		const struct tensor *in_alpha_tensor = self->inputs[3];
		info.alphas = in_alpha_tensor->data;
		info.alpha_depth = in_alpha_tensor->shape.depth;
		info.alpha_min = tensor_get_float(self->inputs[4], 0);
		info.alpha_max = tensor_get_float(self->inputs[5], 0);
		info.tin = tensor_addressing_d32(in_tensor);
		info.nodeinfo = self->opaque;
		nn_sem_init(&info.donesem, 0);
		nn_os_work_for_vector(nn, process_alphas, &info);
		nn_sem_wait(&info.donesem);
		nodeinfo = info.nodeinfo;
		nodeinfo->preprocessed_alphas = 0;
	}
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor, 0);
	float in_max = tensor_get_float(in_max_tensor, 0);
	

	int b = in_tensor->shape.batches;
	int h = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int w = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int d = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];
	//int h_total = h+h_pad_before+h_pad_after;
	int w_total = w + w_pad_before + w_pad_after;
	int d_total = d + d_pad_before + d_pad_after;
	if (d != nodeinfo->alpha_depth)
		return errlog(nn, "depth mismatch : alphas=%d, input = %d", nodeinfo->alpha_depth, d);
	// find output range. If not specified,
	// out_max must allow for most negative input * most negative alpha
	// and for most positive input.
	// out_min just allows for most -ve input and most +ve alpha.
	float alpha_min = tensor_get_float(self->inputs[4], 0);
	float alpha_max = tensor_get_float(self->inputs[5], 0);
	float out_min = in_min * nodeinfo->alpha_max;
	float out_max = fmaxf(in_max, in_min * nodeinfo->alpha_min);
	if (self->n_inputs > 6)
	{
		out_min = tensor_get_float(self->inputs[6], 0);
	}
	if (self->n_inputs > 7)
	{
		out_max = tensor_get_float(self->inputs[7], 0);
	}

	// make sure there is a clean zero.
	adjust_minmax_for_zero(&out_min, &out_max);

	struct tdata td;
	logmsg(nn, 2, "d: %d<%d>%d ==> %d\n", d_pad_before, d, d_pad_after, d_total);
	if (tensor_out_prepare_padded_d32(
			out_tensor,
			b,
			h, h_pad_before, h_pad_after,
			w, w_pad_before, w_pad_after,
			d, d_pad_before, d_pad_after,
			NN_TYPE_QUINT8) != 0)
	{
		return errlog(nn, "out prepare fail");
	}

	logmsg(nn, 2, "Prelu execute. self=%p ", self);

	int wskip = w_pad_before & ~3; // skip 4 if wpad = 4

	td.self = self;
	td.opshape = in_tensor->shape;
	td.tin = tensor_addressing_d32(in_tensor);
	td.tout = tensor_addressing_d32(out_tensor);
	td.tin.data -= -32 * (w_pad_before - wskip),
	td.tout.data -= -32 * (w_pad_before - wskip),
	td.d32_iters = (w_total - wskip) / 4u,
	td.alphabuf = nodeinfo->alphabuf;
	td.scratch = nn->scratch;
	td.next_work = 0;
	td.scaling.in_min = in_min;
	td.scaling.in_max = in_max;
	td.scaling.alpha_min = alpha_min;
	td.scaling.alpha_max = alpha_max;
	td.scaling.out_min = out_min;
	td.scaling.out_max = out_max;
	set_scaling(nn, &td.scaling, in_min, in_max, alpha_min, alpha_max, out_min, out_max);

	nn_sem_init(&td.donesem, 0);
	nn_sem_init(&td.init_done, 0);

	// work out how many parts to slice each batch into vertically.
	//
	unsigned row_work = td.d32_iters * td.tin.nd32; // vectors per row
	// this is the chunk size on the h dimension
	int hchunk = (row_work >= 1024) ? 1 : (1024 >> floor_log2(row_work));
	if (hchunk >= h)
	{
		if (b == 1)
		{
			hchunk = (h + 1) >> 1;
		}
		else
		{
			hchunk = h;
		}
	}
	unsigned slice_per_batch = 1;
	if (h > hchunk)
	{
		slice_per_batch = (h + (hchunk - 1)) / (unsigned)hchunk; // >=2
		// redistribute ?
		if (slice_per_batch < 8)
			hchunk = (h + (slice_per_batch - 1)) / slice_per_batch;
	}
	td.h_per_slice = hchunk;
	td.slice_per_batch = slice_per_batch;
	td.work_units = slice_per_batch * b;

	int nthreads = min_i32(td.work_units, MAX_THREADS);

	//printf("%d threads; %d work units; h=%d divided into %d parts of %d\n",
	//		nthreads, td.work_units, h, slice_per_batch, hchunk);

	// prefetch the alpha table
	l2fetch(td.alphabuf, 128, 128, 2 * td.tin.nd32);
	for( int i =0; i < nthreads; i++){
		nn_os_work_for_vector(nn,prelu_hvx_work_func,&td);
	}

	nn_sem_wait_n_times(&td.donesem, nthreads);

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_set_float(out_min_tensor, 0, out_min);
	tensor_set_float(out_max_tensor, 0, out_max);

	logmsg(nn, 2, "Prelu %p done", self);
	return 0;
}

static int prelu_check_opt(struct nn_node *self, struct nn_graph *nn)
{
	if (find_node_must_be_Const_from_ref(nn, &self->input_refs[3]))
	{
		struct alpha_info info;
		const struct tensor *in_tensor = self->inputs[0];
		const struct tensor *in_alpha_tensor = self->inputs[3];
		info.alphas = in_alpha_tensor->data;
		info.alpha_depth = in_alpha_tensor->shape.depth;
		info.alpha_min = tensor_get_float(self->inputs[4], 0);
		info.alpha_max = tensor_get_float(self->inputs[5], 0);
		info.tin = tensor_addressing_d32(in_tensor);
		info.nodeinfo = self->opaque;
		process_alphas(nn, &info);
		self->opaque = info.nodeinfo;
	}
	return 0;
}

static int prelu_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct nodeinfo *nodeinfo = self->opaque;
	if (nodeinfo)
	{
		if (nodeinfo->alphabuf)
			nn_free(nodeinfo->alphabuf);
		nn_free(nodeinfo);
	}
	self->opaque = NULL;
	return node_free_common(self, nn);
}

//This is a dummy. Will always be replaced by d32 version
struct nn_node_ops nn_ops_for_QuantizedPRelu_8_V2 = {
    .execute = NULL,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedPRelu_8_V2_d32 = {
	.execute = prelu_opt_execute,
	.check = prelu_check_opt,
	.ctor = node_alloc_common,
	.dtor = prelu_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};
