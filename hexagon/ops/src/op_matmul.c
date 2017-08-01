
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
 * This contains the code for matrix multiply op
 */

#include <nn_graph.h>
#include <string.h>
#include <stdlib.h>
#include <quantize.h>

#ifndef __hexagon__
#include <malloc.h>
#endif
#define ALIGN_SIZE 128

/* 8x8 matrix multiply --> 32 bits */
#if defined(__hexagon__)
static int min(int a, int b) { return((a<b)?a:b); }
#endif


static inline int matmul_execute(struct nn_node *self, struct nn_graph *nn,
		void (*f)(struct nn_node *self, struct nn_graph *nn, int32_t a_offset, int32_t b_offset))
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *min_a_tensor = self->inputs[2];
	const struct tensor *max_a_tensor = self->inputs[3];
	const struct tensor *min_b_tensor = self->inputs[4];
	const struct tensor *max_b_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	uint32_t a_batches = a_tensor->shape.batches;
	uint32_t a_width = a_tensor->shape.width;
	uint32_t a_height = a_tensor->shape.height;
	uint32_t a_depth = a_tensor->shape.depth;

	uint32_t b_batches = b_tensor->shape.batches;
	uint32_t b_width = b_tensor->shape.width;
	uint32_t b_height = b_tensor->shape.height;
	uint32_t b_depth = b_tensor->shape.depth;

	uint32_t out_batches = a_batches;
	uint32_t out_height = a_height;
	uint32_t out_width = a_width;
	uint32_t out_depth = b_depth;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements*sizeof(int32_t);

	float a_max_float = tensor_get_float(max_a_tensor,0);
	float a_min_float = tensor_get_float(min_a_tensor,0);
	float b_max_float = tensor_get_float(max_b_tensor,0);
	float b_min_float = tensor_get_float(min_b_tensor,0);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float a_level_size = (a_max_float - a_min_float) / 255.0f;
	float b_level_size = (b_max_float - b_min_float) / 255.0f;
	float out_level_size = a_level_size * b_level_size;

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t a_offset = quantize_uint8(0.0f,a_min_float,a_max_float);
	int32_t b_offset = quantize_uint8(0.0f,b_min_float,b_max_float);

	logmsg(nn,2,"matmul execute. self=%p",self);
	logmsg(nn,2,"matmul in dims: %lux%lux%lux%lu * %lux%lux%lux%lu",
		a_batches,a_height,a_width,a_depth,
		b_batches,b_height,b_width,b_depth);
	logmsg(nn,2,"matmul out dims: %lux%lux%lux%lu",
			out_batches,out_height,out_width,out_depth);
	if (a_height != 1) return errlog(nn,"oops, height != 1");
	if (b_height != 1) return errlog(nn,"oops, height != 1");
	if (a_batches != 1) return errlog(nn,"fixme: support batches");
	if (b_batches != 1) return errlog(nn,"fixme: support batches");
	if (out_size > (out_tensor->max_size)) return errlog(nn,"output too small");
	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,out_min_val);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,out_max_val);
	out_min->data_size = sizeof(float);
	out_max->data_size = sizeof(float);

	f(self, nn, a_offset, b_offset);

	logmsg(nn,2,"matmul execute done!");
	return 0;
}

static inline void matmul_ref(
		struct nn_node *self,
		struct nn_graph *nn,
		int32_t a_offset,
		int32_t b_offset)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint8_t *a = (uint8_t *)a_tensor->data;
	uint8_t *b = (uint8_t *)b_tensor->data;
	int32_t *out = (int32_t *)out_tensor->data;

	int32_t adata;
	int32_t bdata;
	int32_t sum;
	int32_t x;
	int32_t y;
	int32_t i;

	uint32_t a_width = a_tensor->shape.width;
	uint32_t a_depth = a_tensor->shape.depth;

	uint32_t b_depth = b_tensor->shape.depth;
	uint32_t out_width = a_width;
	uint32_t out_depth = b_depth;

    logmsg(nn,2,"a_widthxa_depth=%lux%lu a_offset=%ld b_offset=%ld",
    		a_width, a_depth, a_offset, b_offset);

	for (y = 0; y < out_width; y++) {
		for (x = 0; x < out_depth; x++) {
			sum = 0;
			for (i = 0; i < a_depth; i++) {
				adata = a[i+y*a_depth] - a_offset;
				bdata = b[x+i*b_depth] - b_offset;
				sum += adata * bdata;
			}
			out[x+y*out_depth] = sum;
		}
	}
	logmsg(nn,2,"matmul execute ref done!");
}

static inline void matmul_asm(
		struct nn_node *self,
		struct nn_graph *nn,
		int32_t a_offset,
		int32_t b_offset)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint8_t *a = (uint8_t *)a_tensor->data;
	int32_t *out = (int32_t *)out_tensor->data;

	uint32_t a_width = a_tensor->shape.width;
	uint32_t a_depth = a_tensor->shape.depth;
	uint32_t b_depth = b_tensor->shape.depth;
	uint32_t out_width = a_width;
	uint32_t out_depth = b_depth;
	int32_t i;

    //SIM_ACQUIRE_HVX;
    //SIM_SET_HVX_DOUBLE_MODE;

    //int b_depth_pad = (b_depth + 32-1)&~(32-1);
    int a_depth_pad = (a_depth + 16-1)&~(16-1);

	// ASM code does not handle out_width != 1, for now use reference C code
	if(out_width == 1)
	{
		logmsg(nn,2,"Pad A: a_widthxa_depth=%lux%lu,a_widthxa_depth_pad=%lux%d, a_offset=%ld b_offset=%ld",
				a_width, a_depth, a_width,a_depth_pad, a_offset, b_offset);

		for(i = 0; i < out_depth; i+=32)
		  gemvmpybbw_asm(
			a,
			-a_offset,
			((uint8_t *)self->opaque)+i*a_depth_pad,
			-b_offset,
			((int *)out)+i,
			min(32, out_depth-i),
			a_depth_pad);
		//SIM_RELEASE_HVX;
	}
	else
	{
		matmul_ref(self, nn, a_offset, b_offset);
		logmsg(nn,2,"matmul execute asm does not handle out_width != 1, for now use reference C code!");

	}
	logmsg(nn,2,"matmul execute asm done!");
}


static int matmul_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	return matmul_execute(self,nn,matmul_ref);
}

static int matmul_execute_asm(struct nn_node *self, struct nn_graph *nn)
{
	return matmul_execute(self,nn,matmul_asm);
}

static inline void logmsg_input(
	struct nn_graph *nn,
	int logval,
	int index,
	const struct tensor *tens)
{
	logmsg(nn,logval,"input %d: BHWD=%d,%d,%d,%d data %d bytes @ %p",
		index,
		tens->shape.batches,
		tens->shape.height,
		tens->shape.width,
		tens->shape.depth,
		tens->data_size,
		tens->data);
}

static int matmul_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking matmul node %p",self);
	if (self->n_inputs != 6) return errlog(nn,"matmul wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"matmul wrong # outputs");
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	for (i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) {
			return errlog(nn,"input %d NULL",i);
		}
	}
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) {
			return errlog(nn,"output %d NULL",i);
		}
	}
	logmsg(nn,3,"matmul node %p inputs: "
		"[a, b, min_a, max_a, min_b, max_b]:",
		self);
	for (i = 0; i < self->n_inputs; i++) {
		logmsg_input(nn,3,i,self->inputs[i]);
	}


#define BPAD 32
#define APAD 16
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	uint32_t filt_batches = filt_tensor->shape.filt_batches;
	uint32_t filt_depth = filt_tensor->shape.filt_depth;
	uint32_t out_depth = filt_batches;
	uint8_t *filt = (uint8_t *)filt_tensor->data;
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	uint32_t filt_elements = filt_depth;
	uint32_t filt_elements_pad = (filt_elements + APAD - 1) & (~(APAD - 1));
	int out_depth_pad = (out_depth + BPAD - 1) & ~(BPAD-1);
	uint32_t consts_size;
	uint32_t vecinfo;
	filt_elements_pad = (filt_elements_pad < 32)?32:filt_elements_pad;
	consts_size = filt_elements_pad * out_depth_pad;
	if ((self->opaque = memalign(ALIGN_SIZE,consts_size)) == NULL) {
		return errlog(nn,"couldn't allocate buffer for const rearrangement");
	}
	nn_os_hvx_power_on(nn);
	vecinfo = nn_os_vector_acquire();
	logmsg(nn,2,"Pad B: filt_elements=%lu %lu,out_depth=%lu %d, filt_offset=%ld", filt_elements, out_depth, filt_elements_pad,out_depth_pad, filt_offset);
	pad2d(filt,filt_elements,out_depth,(uint8_t*)nn->scratch,filt_elements_pad,out_depth_pad,filt_offset);
	transpack((const uint8_t *)nn->scratch,filt_elements_pad,out_depth_pad,(uint8_t *)self->opaque);
	nn_os_vector_release(vecinfo);
	nn_os_hvx_power_off(nn);
	logmsg(nn,2,"matmul node %p check OK",self);
	return 0;
}


static struct nn_node *matmul_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	logmsg(nn,2,"matmul node id %x ctor",node_id);
	/* FIXME: replace ops pointers with optimized implementations when available */
	return node_alloc_common(
		nn,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs);
}

struct nn_node_ops nn_ops_for_QuantizedMatMul_8x8to32 = {
	SFINIT(.execute, matmul_execute_asm),
	SFINIT(  .check, matmul_check_ref),
	SFINIT(   .ctor, matmul_ctor),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedMatMul_8x8to32_ref = {
	SFINIT(.execute, matmul_execute_ref),
	SFINIT(  .check, matmul_check_ref),
	SFINIT(   .ctor, matmul_ctor),
	SFINIT(   .dtor, node_free_common),
};
