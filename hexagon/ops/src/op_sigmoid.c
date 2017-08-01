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
#include <op_non_lin_gen_hvx_common.h>
//#define TEST_PERFORMANCE

static int qsigmoid_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = in_tensor->shape.batches 
		* in_tensor->shape.height
		* in_tensor->shape.width
		* in_tensor->shape.depth;
	size_t bytes = elements * sizeof(uint8_t);
	const uint8_t *in_data = (const uint8_t *)in_tensor->data;
	uint8_t *out_data = (uint8_t *)out_tensor->data;
	uint32_t i;
	float inval,tmpval,outval;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float stepsize = (in_max - in_min)/255.0f;

	logmsg(nn,2,"sigmoid execute. self=%p ",self);
	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;

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
	
	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,0.0f);
	out_min_tensor->data_size = sizeof(float);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_max_tensor,0,1.0f);
	out_max_tensor->data_size = sizeof(float);

	logmsg(nn,2,"sigmoid %p done",self);
	return 0;
}

static int qsigmoid_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = in_tensor->shape.batches 
		* in_tensor->shape.height
		* in_tensor->shape.width
		* in_tensor->shape.depth;
	size_t bytes = elements * sizeof(uint8_t);
	size_t pad_size = (bytes+MAXPAD-1)&~(MAXPAD-1);
	const uint8_t *in_data = (const uint8_t *)in_tensor->data;
	uint8_t *out_data = (uint8_t *)out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float rng_min = (float)(MIN_RNG);
	float rng_max = (float)(MAX_RNG);
	
	logmsg(nn,2,"sigmoid execute. self=%p ",self);
	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;
	
#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

#ifdef USE_SCRATCH_PAD
	uint8_t *scratch_pad = nn->scratch;
	vmemcpy_asm(scratch_pad,in_data,bytes);
	requant_u8u8_inplace(scratch_pad, bytes, in_min, in_max, rng_min, rng_max);
	qnonlinear_execute_i(scratch_pad,scratch_pad,pad_size,lut_non_lin_asm_sigmoid);
	vmemcpy_asm(out_data,scratch_pad,bytes);
#else
	vmemcpy_asm(out_data,in_data,bytes);
	requant_u8u8_inplace(out_data,bytes,in_min,in_max,rng_min,rng_max);
	qnonlinear_execute_i(out_data,out_data,pad_size,lut_non_lin_asm_sigmoid);
#endif
	
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsigmoid hvx cycles = %d (elements = %d)\n", (end_time-start_time), elements);
#endif
	
	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,0.0f);
	out_min_tensor->data_size = sizeof(float);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_max_tensor,0,+1.0f);
	out_max_tensor->data_size = sizeof(float);
	
	logmsg(nn,2,"sigmoid %p done",self);
	return 0;
}

static int qsigmoid_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking sigmoid node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"sigmoid node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8_ref = {
	SFINIT(.execute, qsigmoid_execute_ref),
	SFINIT(  .check, qsigmoid_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8 = {
	SFINIT(.execute, qsigmoid_execute_hvx),
	SFINIT(  .check, qsigmoid_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

