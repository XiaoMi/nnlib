/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include <op_sigmoid.h>
//#define TEST_PERFORMANCE

static int qsigmoid_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	uint32_t i;
	float inval,tmpval,outval;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float stepsize = (in_max - in_min)/255.0f;

	logmsg(nn,2,"sigmoid execute. self=%p ",self);

	if( tensor_out_prepare_normal_fromshape( out_tensor, & in_tensor->shape, NN_TYPE_QUINT8 )!= 0)
		return errlog(nn,"out too small");

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	for (i = 0; i < elements; i++) {
		inval = in_min + stepsize * in_data[i];
		inval *= 0.5f;
		tmpval = tanhf(inval);
		outval = (tmpval + 1.0f) * (255.0f/2.0f) + 0.5f;
		if (outval > 255.0f) outval = 255.0f;
		out_data[i] = outval;
	}
	
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsigmoid ref cycles = %d (elements = %d)\n", (end_time-start_time), elements);
#endif
	
	tensor_set_single_float( out_min_tensor, 0.0f);
	tensor_set_single_float( out_max_tensor, 1.0f);

	logmsg(nn,2,"sigmoid %p done",self);
	return 0;
}

#if 0
struct tdata {
	struct nn_node *self;
	void *in_data;
	void *out_data;
	size_t bytes;
	size_t pad_size;
	float in_min;
	float in_max;
	float out_min;
	float out_max;
	nn_sem_t donesem;
};

static void non_lin_execute_td(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	uint8_t *in_data = td->in_data;
	int8_t *out_data = td->out_data;
	size_t bytes = td->bytes;
	size_t pad_size = td->pad_size;
	float in_min = td->in_min;
	float in_max = td->in_max;
	float out_min = td->out_min;
	float out_max = td->out_max;
	requant_s8u8(out_data, in_data, bytes, in_min, in_max, out_min, out_max);
	non_lin_i_sigmoid_8(out_data, out_data, pad_size);
	nn_sem_post(&td->donesem);
}

static int qsigmoid_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);

	size_t pad_size = (bytes+MAXPAD-1)&~(MAXPAD-1);
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float rng_min = (float)(MIN_RNG_SIGMOID_8);
	float rng_max = (float)(MAX_RNG_SIGMOID_8);
	
	logmsg(nn,2,"sigmoid execute. self=%p ",self);
	if( tensor_out_prepare_normal_fromshape( out_tensor, & in_tensor->shape, NN_TYPE_QUINT8 )!= 0)
		return errlog(nn,"out too small");
	
#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif
	
	struct tdata td = {
		.self = self,
		.in_data = in_data,
		.out_data = out_data,
		.bytes = bytes,
		.pad_size = pad_size,
		.in_min = in_min,
		.in_max = in_max,
		.out_min = rng_min,
		.out_max = rng_max,
	};
	nn_sem_init(&td.donesem,0);
	nn_os_work_for_vector(nn, non_lin_execute_td, &td);
	nn_sem_wait(&td.donesem);
	
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsigmoid hvx cycles = %d (elements = %d)\n", (end_time-start_time), elements);
#endif
	
	tensor_set_single_float( out_min_tensor, 0.0f);
	tensor_set_single_float( out_max_tensor, 1.0f);
	
	logmsg(nn,2,"sigmoid %p done",self);
	return 0;
}
#endif

static int qsigmoid_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking sigmoid node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"sigmoid node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8_ref = {
	.execute = qsigmoid_execute_ref,
	.check = qsigmoid_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8 = {
	.execute = qsigmoid_execute_ref,
	.check = qsigmoid_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

