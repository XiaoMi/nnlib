
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
 * This contains implementations for quantized concat node
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include "hvx_inlines.h"


#ifdef HEXAGON_V66
#define CONCAT_MAX_THREADS 4
#else
#define CONCAT_MAX_THREADS 2
#endif

struct concat16_state {
	struct nn_node *self;
	int32_t n_inputs;		// actual number of inputs
	float out_min, out_max;	// output range
	float out_level_recip;
	volatile int32_t input_next;		// input to process next.

	uint32_t inner_size;
	uint32_t out_stride;
	uint32_t outer_count;

	int32_t concat_dim;
	const struct tensor **input_tensors;
	const struct tensor **min_tensors;
	const struct tensor **max_tensors;
	struct tensor *out_tensor;

	nn_sem_t done_sem;
};

static int32_t set_output_range(
	struct nn_node *self,
	int32_t concat_dim,
	float *pout_min,
	float *pout_max,
	float *pout_level_recip
)
{
	int32_t  n_input_tensors = (self->n_inputs - 1) / 3;
	const struct tensor **input_tensors = &self->inputs[1];
	const struct tensor **min_tensors = &self->inputs[1 + n_input_tensors];
	const struct tensor **max_tensors = &self->inputs[1 + 2 * n_input_tensors];
	struct tensor *out_tensor = self->outputs[0];
	float out_min = 0.0f;
	float out_max = 0.0f;

	uint32_t outdims[4] = { self->inputs[1]->shape.batches,
							self->inputs[1]->shape.height,
							self->inputs[1]->shape.width,
							self->inputs[1]->shape.depth };

	outdims[concat_dim] = 0;
	for (int32_t i = 0; i < n_input_tensors; i++) {
		for (int32_t k = 0; k < 4; k++) {
			if (concat_dim != k) {
				if (outdims[k] != input_tensors[i]->shape.dimension[k]) {
					return errlog(NULL, "batches mismatch tensor %d, dim %d", i, k);
				}
			}
			else {
				outdims[concat_dim] += input_tensors[i]->shape.dimension[k];
			}
		}
		out_min = fminf(out_min, tensor_get_float(min_tensors[i], 0));
		out_max = fmaxf(out_max, tensor_get_float(max_tensors[i], 0));
	}

	if (tensor_out_prepare_normal(out_tensor, outdims[0], outdims[1], outdims[2], outdims[3], NN_TYPE_QUINT16) != 0) {
		return errlog(NULL, "out too small");
	}
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	tensor_set_single_float(out_min_tensor, out_min);
	tensor_set_single_float(out_max_tensor, out_max);

	*pout_min = out_min;
	*pout_max = out_max;
	*pout_level_recip = 65536.0f / (out_max - out_min);
	return 0;
}

static void find_concat_param(
	int32_t concat_dim,
	int32_t *dims,
	uint32_t *pinner_size,
	uint32_t *pout_stride,
	uint32_t *pouter_count
)
{
	// assume concat_dim = 4
	// dim 0    1    2    3    4    5    6    7
	//     <--outer_count->                      :prod of all dims < concat_dim
	//                         <--out_stride---> :inner_size * dim[concat_dim]
	//                              <----------> inner_size: prod of all dims > concat_dim
	uint32_t inner_size = 0;
	uint32_t out_stride = 0;
	uint32_t outer_count = 1;	// will become inner_size.
	for (int32_t i = 3; i >= 0; --i) {	// depth, width, height, batches
		uint32_t ndim = dims[i];
		uint32_t newcnt = outer_count * ndim;
		if (i == concat_dim) {
			inner_size = outer_count;	// set inner_size, out_stride
			out_stride = newcnt;
			newcnt = 1;					// and restart for out_count.
		}
		outer_count = newcnt;
	}
	*pinner_size = inner_size;
	*pout_stride = out_stride;
	*pouter_count = outer_count;
}

void memconvert16_hvx(
	uint16_t *dsto,
	uint16_t *srco,
	int32_t length,
	int32_t offset,
	int16_t gain,
	int32_t stride,
	int32_t rows
)
{
	offset = Q6_R_combine_RlRl(offset, offset);
	int32_t gains = Q6_R_combine_RlRl(gain, gain);

	HVX_Vector z0, x0, x1, xa, y0;
	HVX_Vector vpredp, vprede, voffset, vgain, v8000, vone, vzero = Q6_V_vzero();
	v8000 = Q6_V_vsplat_R(0x80008000);
	vgain = Q6_V_vsplat_R(gains);
	voffset = Q6_V_vsplat_R(offset);
	vone = Q6_V_vnot_V(vzero);
	x0 = xa = vzero;

	int32_t select, sel0 = 0x01010101;
	int32_t sel1 = sel0 + sel0;

	for (int32_t i = 0; i < rows; i++) {
		uint16_t *src = srco;
		uint16_t *dst = dsto;
		uint32_t srcalign = (size_t)src & 127;
		uint32_t dstalign = (size_t)dst & 127;
		uint32_t end = (dstalign + length * sizeof(uint16_t)) & 127;

		HVX_VectorPred qprolog = Q6_Q_vsetq_R((size_t)dsto);
		HVX_VectorPred qepilog = Q6_Q_vsetq_R(end);
		vpredp = Q6_V_vand_QR(qprolog, sel1);
		vprede = Q6_V_vand_QR(qepilog, sel1);
		qprolog = Q6_Q_or_QQn(qprolog, qepilog);
		vpredp = Q6_V_vandor_VQR(vpredp, qprolog, sel0);
		select = (dstalign + length * sizeof(uint16_t) > 127) ? sel1 : sel0;

		qprolog = Q6_Q_vand_VR(vpredp, select);
		qepilog = Q6_Q_vand_VR(vprede, select);

		int32_t mid = srcalign - dstalign;
		int32_t kernel = max_i32(0, length * sizeof(uint16_t) - end);

		if (mid >= 0) {
			xa = *(HVX_Vector*)src;
			if (((size_t)(src + 64)&-128) < (size_t)(srco + length))
				src += 64;
			xa = Q6_V_vxor_VV(xa, v8000);
			z0 = Q6_Vh_vmpy_VhRh_s1_rnd_sat(xa, gains);
			z0 = Q6_Vh_vadd_VhVh(z0, vgain);
			x0 = Q6_Vuh_vadd_VuhVuh_sat(z0, voffset);
		}
		int32_t j = 0;
		do {
			xa = *(HVX_Vector*)src; src += 64;
			xa = Q6_V_vxor_VV(xa, v8000);
			z0 = Q6_Vh_vmpy_VhRh_s1_rnd_sat(xa, gains);
			z0 = Q6_Vh_vadd_VhVh(z0, vgain);
			x1 = Q6_Vuh_vadd_VuhVuh_sat(z0, voffset);
			y0 = Q6_V_valign_VVR(x1, x0, mid);
			x0 = x1;
			q6op_vstcc_QnAV(qprolog, (HVX_Vector *)dst, y0);
			dst += 64;
			qprolog = Q6_Q_vcmp_eq_VbVb(vone, vzero);
		} while (++j < (kernel + 127) >> 7);
		if (((size_t)src&-128) <= (size_t)(srco + length)) xa = *(HVX_Vector*)src;
		src += 64;
		xa = Q6_V_vxor_VV(xa, v8000);
		z0 = Q6_Vh_vmpy_VhRh_s1_rnd_sat(xa, gains);
		z0 = Q6_Vh_vadd_VhVh(z0, vgain);
		x1 = Q6_Vuh_vadd_VuhVuh_sat(z0, voffset);
		y0 = Q6_V_valign_VVR(x1, x0, mid);
		x0 = x1;
		q6op_vstcc_QAV(qepilog, (HVX_Vector *)dst, y0);

		dsto += stride;
		srco += length;
	}
}

static int concat_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	int32_t n_input_tensors = (self->n_inputs - 1) / 3;
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	struct shape out_shape;		// shape of output tensor
	int32_t concat_dim, i, k;

	logmsg(nn, 2, "concat execute. self=%p ", self);
	concat_dim = tensor_get_int32(dim_tensor, 0);

	// check the dims of all inputs, find the output shape. This also
	// range checks 'concat_dim'.
	//
	k = find_concat_shape(input_tensors, n_input_tensors, concat_dim, &out_shape);
	if (k < 0) {
		if (k <= -2) {
			// mismatch size on a particular dim
			return errlog(nn, "mismatch on tensor dim %d, concat on %d", (-2) - k, concat_dim);
		}
		return errlog(nn, "bad concat dim: %d", concat_dim);
	}

	// find max/min among input vectors
	float out_min = 0.0f, out_max = 0.0f, out_level_recip = 0.0f;
	if (set_output_range(self, concat_dim, &out_min, &out_max, &out_level_recip)) {
		return errlog(nn, "set_output_range error!");
	}

	// find concat traverse param
	uint32_t inner_size, out_stride, outer_count;
	find_concat_param(concat_dim, (int32_t*)out_shape.dimension, &inner_size, &out_stride, &outer_count);

	const struct tensor **min_tensors = &self->inputs[1 + n_input_tensors];
	const struct tensor **max_tensors = &self->inputs[1 + 2 * n_input_tensors];
	uint16_t* out_data = out_tensor->data;
	for (i = 0; i < n_input_tensors; i++) {
		const struct tensor *t = input_tensors[i];
		uint16_t *in_data = t->data;
		int32_t input_dim = t->shape.dimension[concat_dim];
		uint32_t copylen = input_dim * inner_size;

		l2fetch(in_data, copylen * sizeof(uint16_t), copylen * sizeof(uint16_t), outer_count);

		float in_min = tensor_get_float(min_tensors[i], 0);
		float in_max = tensor_get_float(max_tensors[i], 0);
		in_min = fminf(0.0f, in_min); // in_min <= 0.0f
		float in_level = (in_max - in_min) / 65536.0f;

		int32_t offset = max_i32(0, roundf_i32((in_min - out_min)*out_level_recip));
		int32_t gaint = roundf_i32(out_level_recip*in_level* 32768.0f);
		int16_t gain = min_i32(32767, gaint);

		if (gain < 32767) {
			for (int32_t j = 0; j < outer_count; j++) {
				for (k = 0; k < copylen; k++) {
					uint16_t ival = in_data[j*copylen + k];
					uint16_t oval = Q6_R_satuh_R(((ival* gain + (1 << 14)) >> 15) + offset);
					out_data[j*out_stride + k] = oval;
				}
			}
		}
		else {
			for (int32_t j = 0; j < outer_count; j++) {
				memcpy(out_data + j * out_stride, in_data + j * copylen, copylen * sizeof(uint16_t));
			}
		}
		out_data += copylen;
	}

	logmsg(nn, 2, "concat %p done", self);
	return 0;
}

static void concat16_work(
	struct nn_graph *nn,
	void *thrinfo)
{
	struct concat16_state *thrdesc = (struct concat16_state *)thrinfo;

	int32_t concat_dim = thrdesc->concat_dim;
	const struct tensor **input_tensors = thrdesc->input_tensors;
	const struct tensor **min_tensors = thrdesc->min_tensors;
	const struct tensor **max_tensors = thrdesc->max_tensors;
	struct tensor *out_tensor = thrdesc->out_tensor;

	int32_t inner_size = thrdesc->inner_size;
	int32_t out_stride = thrdesc->out_stride;
	int32_t outer_count = thrdesc->outer_count;

	float out_min = thrdesc->out_min;
	float out_level_recip = thrdesc->out_level_recip;
	uint16_t* out_data = out_tensor->data;

	int32_t jobid, prev_thread = 0;
	while (jobid = __sync_fetch_and_add(&thrdesc->input_next, 1), jobid < thrdesc->n_inputs) {
		const struct tensor *t = input_tensors[jobid];
		while (prev_thread < jobid) {
			out_data += input_tensors[prev_thread++]->shape.dimension[concat_dim] * inner_size;
		}
		uint16_t *in_data = t->data;
		int32_t input_dim = t->shape.dimension[concat_dim];
		uint32_t copylen = input_dim * inner_size;

		l2fetch(in_data, copylen * sizeof(uint16_t), copylen * sizeof(uint16_t), outer_count);

		float in_min = tensor_get_float(min_tensors[jobid], 0);
		float in_max = tensor_get_float(max_tensors[jobid], 0);
		in_min = fminf(0.0f, in_min); // in_min <= 0.0f
		float in_level = (in_max - in_min) / 65536.0f;

		int32_t offset = max_i32(0, roundf_i32((in_min - out_min)*out_level_recip));
		int32_t gaint = roundf_i32(out_level_recip*in_level* 32768.0f);
		int16_t gain = min_i32(32767, gaint);

		if (gain < 32767) {
			memconvert16_hvx(
				out_data,
				in_data,
				copylen,
				offset,
				gain,
				out_stride,
				outer_count);
		}
		else {
			vmemcpy_2d_general_asm(
				copylen * sizeof(uint16_t),    // bytes wide
				outer_count,                   // rows
				out_data,                      // destination address, any allowed
				out_stride * sizeof(uint16_t), // row pitch of dest; any allowed
				in_data,                       // source address, any allowed
				copylen * sizeof(uint16_t));   // source stride, any
		}
	}

	// signal complete in thread.
	nn_sem_post(&thrdesc->done_sem);
}

static int concat_execute(struct nn_node *self, struct nn_graph *nn)
{
	int32_t n_input_tensors = (self->n_inputs - 1) / 3;
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	struct shape out_shape;		// shape of output tensor
	int32_t concat_dim, i, k;

	logmsg(nn, 2, "concat execute. self=%p ", self);
	concat_dim = tensor_get_int32(dim_tensor, 0);

	// check the dims of all inputs, find the output shape. This also
	// range checks 'concat_dim'.
	//
	k = find_concat_shape(input_tensors, n_input_tensors, concat_dim, &out_shape);
	if (k < 0) {
		if (k <= -2) {
			// mismatch size on a particular dim
			return errlog(nn, "mismatch on tensor dim %d, concat on %d", (-2) - k, concat_dim);
		}
		return errlog(nn, "bad concat dim: %d", concat_dim);
	}

	// find max/min among input vectors
	float out_min = 0.0f, out_max = 0.0f, out_level_recip = 0.0f;
	if (set_output_range(self, concat_dim, &out_min, &out_max, &out_level_recip)) {
		return errlog(nn, "set_output_range error!");
	}

	// find concat traverse param
	struct concat16_state rundesc;
	find_concat_param(concat_dim, (int32_t*)out_shape.dimension, &rundesc.inner_size, &rundesc.out_stride, &rundesc.outer_count);

	// fire the threads
	rundesc.self = self;
	rundesc.n_inputs = n_input_tensors;
	rundesc.out_min = out_min;
	rundesc.out_max = out_max;
	rundesc.out_level_recip = out_level_recip;
	rundesc.input_next = 0;
	nn_sem_init(&rundesc.done_sem, 0);

	rundesc.concat_dim = concat_dim;
	rundesc.input_tensors = input_tensors;
	rundesc.min_tensors = &self->inputs[1 + n_input_tensors];
	rundesc.max_tensors = &self->inputs[1 + 2 * n_input_tensors];
	rundesc.out_tensor = out_tensor;

	int32_t num_actual_threads = min_i32(CONCAT_MAX_THREADS, n_input_tensors);
	for (i = 0; i < num_actual_threads; i++) {
		nn_os_work_for_vector(nn, concat16_work, &rundesc);
	}
	nn_sem_wait_n_times(&rundesc.done_sem, num_actual_threads);

	logmsg(nn,2,"concat %p done",self);
	return 0;
}

static int concat_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking concat node %p",self);

	// must be 3*n+1 inputs, where n >= 1
	int32_t n_in = (self->n_inputs - 1) /3;	// actual # of inputs
	if (n_in < 1 || (self->n_inputs - 1) % 3 !=0 )
		return errlog(nn,"concat: inputs must be 3*n+1, n>=1");

	logmsg(nn,2,"concat node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedConcat_u16 = {
	.execute = concat_execute,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedConcat_u16_ref = {
	.execute = concat_execute_ref,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
};

