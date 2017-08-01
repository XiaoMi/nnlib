
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
void relu_kernel(uint8_t *in_data, uint8_t *out_data, int bytes, uint8_t quantized_zero);
void reluX_kernel(uint8_t *in_data, uint8_t *out_data, int bytes, uint8_t quantized_zero, uint8_t quantized_max);
#endif

static int relu_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t bytes = in_tensor->shape.batches 
		* in_tensor->shape.height
		* in_tensor->shape.width
		* in_tensor->shape.depth;
	uint8_t *in_data = (uint8_t *)in_tensor->data;
	uint8_t *out_data = (uint8_t *)out_tensor->data;
	uint8_t quantized_zero;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"relu execute. self=%p ",self);
	logmsg(nn,2,"relu in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	quantized_zero = quantize_uint8(0.0f,in_min,in_max);

	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;

	relu_kernel(in_data, out_data, bytes, quantized_zero);

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	logmsg(nn,2,"relu out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"relu %p done",self);
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
	size_t bytes = in_tensor->shape.batches 
		* in_tensor->shape.height
		* in_tensor->shape.width
		* in_tensor->shape.depth;
	uint8_t *in_data = (uint8_t *)in_tensor->data;
	uint8_t *out_data = (uint8_t *)out_tensor->data;
	uint8_t quantized_zero;
	uint8_t quantized_max;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float max_val = tensor_get_float(max_val_tensor,0);

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"relu execute. self=%p ",self);

	quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	quantized_max = quantize_uint8(max_val,in_min,in_max);

	if (out_tensor->max_size < bytes) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;

	reluX_kernel(in_data, out_data, bytes, quantized_zero, quantized_max);

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	logmsg(nn,2,"relu %p done",self);
	return 0;
}

static int relu_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking relu node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	for (i = 0; i < 3; i++) {
		if (self->inputs[i] == NULL) return errlog(nn,"NULL input");
		if (self->outputs[i] == NULL) return errlog(nn,"NULL output");
	}
	logmsg(nn,2,"relu node %p check OK",self);
	return 0;
}

static int reluX_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking reluX node %p",self);
	if (self->n_inputs != 4) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	for (i = 0; i < 3; i++) {
		if (self->inputs[i] == NULL) return errlog(nn,"NULL input");
		if (self->outputs[i] == NULL) return errlog(nn,"NULL output");
	}
	if (self->inputs[3] == NULL) return errlog(nn,"NULL input");
	logmsg(nn,2,"reluX node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedRelu_8 = {
	SFINIT(.execute, relu_execute),
	SFINIT(  .check, relu_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedReluX_8 = {
	SFINIT(.execute, reluX_execute),
	SFINIT(  .check, reluX_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};


struct nn_node_ops nn_ops_for_QuantizedRelu_8_ref = {
	SFINIT(.execute, relu_execute),
	SFINIT(  .check, relu_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedReluX_8_ref = {
	SFINIT(.execute, reluX_execute),
	SFINIT(  .check, reluX_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

