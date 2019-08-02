
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
#include <stdio.h>
#include <math.h>
#include <quantize.h>
#include <nn_broadcast.h>
#include <nn_reduction.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains min and max (floating) ops
 */

static int min_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_reduction_float(self, nn, fminf, INFINITY);
}

static int max_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_reduction_float(self, nn, fmaxf, -INFINITY);
}


BROADCAST_STRIDE_11_FUNC(minimum_f_stride_11, float, float, fminf)
BROADCAST_STRIDE_10_FUNC(minimum_f_stride_10, float, float, fminf)

static const struct elementwise_funcs Minimum_f_funcs = {
	.op_stride_11 = minimum_f_stride_11,
	.op_stride_10 = minimum_f_stride_10,
	.op_rev_stride_01 = minimum_f_stride_10,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode = NN_TYPE_FLOAT};

static int minimum_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast(self, nn, &Minimum_f_funcs,NULL,NULL, NULL);
}
BROADCAST_STRIDE_11_FUNC(maximum_f_stride_11, float, float, fmaxf)
BROADCAST_STRIDE_10_FUNC(maximum_f_stride_10, float, float, fmaxf)

static const struct elementwise_funcs Maximum_f_funcs = {
	.op_stride_11 = maximum_f_stride_11,
	.op_stride_10 = maximum_f_stride_10,
	.op_rev_stride_01 = maximum_f_stride_10,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode = NN_TYPE_FLOAT};

static int maximum_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast(self, nn, &Maximum_f_funcs,NULL,NULL, NULL);
}

struct nn_node_ops nn_ops_for_Min_f = {
	.execute = min_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Max_f = {
	.execute = max_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Min_f_ref = {
	.execute = min_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Max_f_ref = {
	.execute = max_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Minimum_f = {
	.execute = minimum_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Maximum_f = {
	.execute = maximum_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

#define CREATE_REF_OP_MIN_MAX(NAME, OPNAME, OPERATOR)                                      \
	struct NAME##_info                                                                     \
	{                                                                                      \
		int a_offset;                                                                      \
		int b_offset;                                                                      \
		int a_mult;                                                                        \
		int b_mult;                                                                        \
		int shift;                                                                         \
		int qzero;                                                                         \
	};                                                                                     \
                                                                                           \
	static inline uint8_t q8##NAME##_helper(uint8_t a, uint8_t b, void *v##NAME_info)      \
	{                                                                                      \
		const struct NAME##_info *info = v##NAME_info;                                     \
		int a_offset = info->a_offset;                                                     \
		int b_offset = info->b_offset;                                                     \
		int a_mult = info->a_mult;                                                         \
		int b_mult = info->b_mult;                                                         \
		int shift = info->shift;                                                           \
		int qzero = info->qzero;                                                           \
		int aval = (((a - a_offset) * a_mult) >> shift) + qzero;                           \
		int bval = (((b - b_offset) * b_mult) >> shift) + qzero;                           \
                                                                                           \
		uint8_t ret = f##OPERATOR##f(aval, bval);                                          \
                                                                                           \
		return ret;                                                                        \
	}                                                                                      \
                                                                                           \
	static int NAME##_q8_execute_ref(struct nn_node *self, struct nn_graph *nn)            \
	{                                                                                      \
		struct NAME##_info info;                                                           \
		const struct tensor *a_min_tensor = self->inputs[2];                               \
		const struct tensor *a_max_tensor = self->inputs[3];                               \
		const struct tensor *b_min_tensor = self->inputs[4];                               \
		const struct tensor *b_max_tensor = self->inputs[5];                               \
		struct tensor *out_min_tensor = self->outputs[1];                                  \
		struct tensor *out_max_tensor = self->outputs[2];                                  \
		float a_min_float = tensor_get_float(a_min_tensor, 0);                             \
		float a_max_float = tensor_get_float(a_max_tensor, 0);                             \
		float b_min_float = tensor_get_float(b_min_tensor, 0);                             \
		float b_max_float = tensor_get_float(b_max_tensor, 0);                             \
                                                                                           \
		float a_level_size = (a_max_float - a_min_float) / 255;                            \
		float b_level_size = (b_max_float - b_min_float) / 255;                            \
                                                                                           \
		float out_min = fminf(0.0, fminf(a_min_float, b_min_float));                       \
		float out_max = fmaxf(0.0, f##OPERATOR##f(a_max_float, b_max_float));              \
		float out_level_size = (out_max - out_min) / 255;                                  \
		int retval;                                                                        \
                                                                                           \
		tensor_set_single_float(out_min_tensor, out_min);                                  \
		tensor_set_single_float(out_max_tensor, out_max);                                  \
                                                                                           \
		info.a_offset = quantize_uint8(0.0f, a_min_float, a_max_float);                    \
		info.b_offset = quantize_uint8(0.0f, b_min_float, b_max_float);                    \
		info.shift = 12;                                                                   \
		info.a_mult = ((float)(1 << info.shift)) * (a_level_size / out_level_size) + 0.5f; \
		info.b_mult = ((float)(1 << info.shift)) * (b_level_size / out_level_size) + 0.5f; \
		info.qzero = -out_min * (255 / (out_max - out_min)) + 0.5f;                        \
		retval = broadcast_elementwise_execute_quint8(self, nn, q8##NAME##_helper, &info); \
                                                                                           \
		return retval;                                                                     \
	}                                                                                      \
                                                                                           \
	struct nn_node_ops nn_ops_for_Quantized##OPNAME##_8_ref = {                            \
		.execute = NAME##_q8_execute_ref,                                                  \
		.check = NULL,                                                                     \
		.ctor = node_alloc_common,                                                         \
		.dtor = node_free_common,                                                          \
		.n_inputs = NN_IOCOUNT(6),                                                         \
		.n_outputs = NN_IOCOUNT(3),                                                        \
	};

CREATE_REF_OP_MIN_MAX(minimum, Minimum, min);
CREATE_REF_OP_MIN_MAX(maximum, Maximum, max);
