
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
#include <math.h>
#include <quantize.h>


/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains quantize / dequantize ops
 */

static int quantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float min_in = tensor_get_float(min_tensor,0);
	float max_in = tensor_get_float(max_tensor,0);
	float recip_stepsize;
	float min_out;
	float max_out;
	float stepsize;
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const float *in = (const float *)in_tensor->data;
	uint8_t *out = (uint8_t *)out_tensor->data;
	int out_bytes = batches*height*width*depth;
	float inval;
	int i;
	int ival;
	logmsg(nn,2,"quantize execute. self=%p ",self);
	if (out_tensor->max_size < out_bytes) return errlog(nn,"out too small %d < %d",out_tensor->max_size,out_bytes);
	tensor_set_shape(out_tensor,batches,height,width,depth);
	out_tensor->data_size = out_bytes;
	quantize_adjust_range(
		&min_out,&max_out,
		&stepsize,&recip_stepsize,
		min_in,max_in);

	for (i = 0; i < batches*height*width*depth; i++) {
		inval = in[i];
		ival = ((inval - min_out)*recip_stepsize+0.5f);
		if (ival < 0) ival = 0;
		if (ival > 255) ival = 255;
		out[i] = ival;
	}

	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,min_out);
	tensor_set_float(out_max_tensor,0,max_out);
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);
	return 0;
}

static int autoquantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float min_in = 0.0f;
	float max_in = 0.0f;
	float recip_stepsize;
	float min_out;
	float max_out;
	float stepsize;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;
	const float *in = (const float *)in_tensor->data;
	uint8_t *out = (uint8_t *)out_tensor->data;
	int out_bytes = batches*height*width*depth;
	float inval;
	int i;
	int ival;
	logmsg(nn,2,"autoquantize execute. self=%p ",self);
	if (out_tensor->max_size < out_bytes) return errlog(nn,"out too small %d < %d",out_tensor->max_size,out_bytes);
	tensor_set_shape(out_tensor,batches,height,width,depth);
	out_tensor->data_size = out_bytes;

	for (i = 0; i < batches*height*width*depth; i++) {
		min_in = fminf(min_in,in[i]);
		max_in = fmaxf(max_in,in[i]);
	}
	logmsg(nn,2,"min=%f max=%f bhwd=%d,%d,%d,%d",min_in,max_in,batches,height,width,depth);
	quantize_adjust_range(
		&min_out,&max_out,
		&stepsize,&recip_stepsize,
		min_in,max_in);

	for (i = 0; i < batches*height*width*depth; i++) {
		inval = in[i];
		ival = ((inval - min_out)*recip_stepsize+0.5f);
		if (ival < 0) ival = 0;
		if (ival > 255) ival = 255;
		out[i] = ival;
	}

	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,min_out);
	tensor_set_float(out_max_tensor,0,max_out);
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);
	return 0;
}

static int dequantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	float minval = tensor_get_float(min_tensor,0);
	float maxval = tensor_get_float(max_tensor,0);
	float range = fmaxf(0.0001f,maxval-minval);
	float stepsize = range/255.0f;
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const uint8_t *in = (uint8_t *)in_tensor->data;
	float *out = (float *)out_tensor->data;
	int out_bytes = batches*height*width*depth*sizeof(float);
	int i;
	logmsg(nn,2,"dequantize execute. self=%p ",self);
	if (out_tensor->max_size < out_bytes) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,batches,height,width,depth);
	out_tensor->data_size = out_bytes;
	for (i = 0; i < batches*height*width*depth; i++) {
		out[i] = (in[i] * stepsize) + minval;
	}
	return 0;
}

static int dequantize_i32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	float minval = tensor_get_float(min_tensor,0);
	float maxval = tensor_get_float(max_tensor,0);
	float range = fmaxf(0.0001f,maxval-minval);
	float stepsize = range/4294967296.0f/*0x1.0p32f*/;
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const int32_t *in = (const int32_t *)in_tensor->data;
	float *out = (float *)out_tensor->data;
	int out_bytes = batches*height*width*depth*sizeof(float);
	int i;
	logmsg(nn,2,"dequantize 32 execute.");
	if (out_tensor->max_size < out_bytes) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,batches,height,width,depth);
	out_tensor->data_size = out_bytes;
	for (i = 0; i < batches*height*width*depth; i++) {
		out[i] = (in[i] * stepsize);
	}
	return 0;
}


static int quantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking quantize node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"quantize node %p OK",self);
	return 0;
}

static int autoquantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking quantize node %p",self);
	if (self->n_inputs != 1) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"quantize node %p OK",self);
	return 0;
}

static int dequantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking dequantize node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"dequantize node %p OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Quantize = {
	SFINIT(.execute, quantize_execute),
	SFINIT(  .check, quantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Quantize_ref = {
	SFINIT(.execute, quantize_execute),
	SFINIT(  .check, quantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_AutoQuantize = {
	SFINIT(.execute, autoquantize_execute),
	SFINIT(  .check, autoquantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_AutoQuantize_ref = {
	SFINIT(.execute, autoquantize_execute),
	SFINIT(  .check, autoquantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Dequantize = {
	SFINIT(.execute, dequantize_execute),
	SFINIT(  .check, dequantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Dequantize_ref = {
	SFINIT(.execute, dequantize_execute),
	SFINIT(  .check, dequantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Dequantize_qint32_f = {
	SFINIT(.execute, dequantize_i32_execute),
	SFINIT(  .check, dequantize_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};
