
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
 * This contains implementations for quantized relu node
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

#define USE_ASM
#if !defined(USE_ASM)
static inline uint8_t uint8_t_max(uint8_t a, uint8_t b)
{
	if (a > b) return a;
	else return b;
}

static inline uint8_t uint8_t_min(uint8_t a, uint8_t b)
{
	if (a < b) return a;
	else return b;
}
#else
void relu_kernel(const uint8_t *in_data, uint8_t *out_data, int bytes, uint8_t quantized_zero);
void reluX_kernel(const uint8_t *in_data, uint8_t *out_data, int bytes, uint8_t quantized_zero, uint8_t quantized_max);
#endif


struct tdata {
	const uint8_t *in_data;
	uint8_t *out_data;
	size_t bytes;
	uint8_t quantized_zero;
	uint8_t quantized_max;
	nn_sem_t donesem;
};


static void relu_execute_hvx(struct nn_graph *nn, void *vtd)
{
	struct tdata *td = vtd;
	relu_kernel(td->in_data, td->out_data, td->bytes, td->quantized_zero);
	nn_sem_post(&td->donesem);
}

static void reluX_execute_hvx(struct nn_graph *nn, void *vtd)
{
	struct tdata *td = vtd;
	reluX_kernel(td->in_data, td->out_data, td->bytes, td->quantized_zero,td->quantized_max);
	nn_sem_post(&td->donesem);
}

static int relu_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t bytes = 1* tensor_element_count(in_tensor);
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	uint8_t quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	struct tdata td = {
		.in_data = in_data,
		.out_data = out_data,
		.bytes = bytes,
		.quantized_zero = quantized_zero,
	};
	nn_sem_init(&td.donesem,0);

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"relu execute. self=%p ",self);
	logmsg(nn,2,"relu in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	if( tensor_out_prepare_normal_fromshape( out_tensor, & in_tensor->shape, NN_TYPE_QUINT8 )!= 0)
		return errlog(nn,"out too small");

	nn_os_work_for_vector(nn,relu_execute_hvx,&td);
	nn_sem_wait(&td.donesem);

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	logmsg(nn,2,"relu out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"relu %p done",self);
	return 0;
}

static int clamp_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *min_val_tensor = self->inputs[3];
	const struct tensor *max_val_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t bytes = 1* tensor_element_count(in_tensor);
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float min_val = tensor_get_float(min_val_tensor,0);
	float max_val = tensor_get_float(max_val_tensor,0);
	uint8_t q_min = quantize_uint8(min_val,in_min,in_max);
	uint8_t q_max = quantize_uint8(max_val,in_min,in_max);
	struct tdata td = {
		.in_data = in_data,
		.out_data = out_data,
		.bytes = bytes,
		.quantized_zero = q_min,
		.quantized_max = q_max,
	};
	nn_sem_init(&td.donesem,0);

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"clamp execute. self=%p ",self);

	if( tensor_out_prepare_normal_fromshape( out_tensor, & in_tensor->shape, NN_TYPE_QUINT8 )!= 0)
		return errlog(nn,"out too small");

	nn_os_work_for_vector(nn,reluX_execute_hvx,&td);
	nn_sem_wait(&td.donesem);

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	logmsg(nn,2,"clamp %p done",self);
	return 0;
}

static int reluX_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *max_val_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t bytes = 1* tensor_element_count(in_tensor);
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float max_val = tensor_get_float(max_val_tensor,0);
	uint8_t quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	uint8_t quantized_max = quantize_uint8(max_val,in_min,in_max);
	struct tdata td = {
		.in_data = in_data,
		.out_data = out_data,
		.bytes = bytes,
		.quantized_zero = quantized_zero,
		.quantized_max = quantized_max,
	};
	nn_sem_init(&td.donesem,0);

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"relu execute. self=%p ",self);

	if( tensor_out_prepare_normal_fromshape( out_tensor, & in_tensor->shape, NN_TYPE_QUINT8 )!= 0)
		return errlog(nn,"out too small");

	nn_os_work_for_vector(nn,reluX_execute_hvx,&td);
	nn_sem_wait(&td.donesem);

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	logmsg(nn,2,"relu %p done",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedRelu_8 = {
	.execute = relu_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedReluX_8 = {
	.execute = reluX_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};


struct nn_node_ops nn_ops_for_QuantizedRelu_8_ref = {
	.execute = relu_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedReluX_8_ref = {
	.execute = reluX_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedClamp_8 = {
	.execute = clamp_execute, 
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedClamp_8_ref = {
	.execute = clamp_execute, 
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

