
/*
 * Copyright (c) 2016, The Linux Foundation. All rights reserved.
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


static inline uint32_t tensor_elements(const struct tensor *t)
{
	return t->shape.batches
		* t->shape.height
		* t->shape.width
		* t->shape.depth;
}

static inline uint8_t __attribute__((unused)) roundsatu8(float in)
{
	int32_t ival;
	ival = round(in);
	if (ival < 0) ival = 0;
	if (ival > 255) ival = 255;
	return ival;
}

struct tdata {
	struct nn_node *self;
	int whoami;
	float out_min;
	float out_max;
	int total_depth;
	nn_sem_t donesem;
};

static inline void do_convert(
	struct nn_graph *nn,
	uint8_t *out,
	const uint8_t *in, 
	const uint32_t iters,
	const uint32_t depth,
	const uint32_t stride,
	const float in_level,
	const float in_min,
	const float out_min,
	const float recip_level)
{
#if 0
	int i,j;
	uint8_t val;
	for (i = 0; i < iters; i++) {
		for (j = 0; j < depth; j++) {
			val = roundsatu8(((*(in++) * in_level) + in_min - out_min) * recip_level);
			out[j] = val;
		}
		out += stride;
	}
#else
        int offset = (int) ((in_min-out_min)/in_level);
        int gain = (int) (recip_level*in_level*powf(2.0, 15));
        short gains;
        if(gain > 32767) gains  = 32767; else gains = (short) gain;
#ifndef __hexagon__
	int i,j;
	int val, ival;
	for (i = 0; i < iters; i++) {
		for (j = 0; j < depth; j++) {
                        ival = in[j];
			val = ((ival + offset)* gains + (1<<14))>>15;
                        if(val > 255) val = 255; else if(val < 0) val = 0;
			out[j] = val;
		}
                in += depth;
		out += stride;
	}
#else
        memconvert_hvx(out, in, depth, offset, gains,stride, iters);
#endif
#endif
}


static void concat_execute_slice(struct nn_graph *nn, void *vinfo)
{
	struct tdata *info = vinfo;
	struct nn_node *self = info->self;
	int whoami = info->whoami;
	int n_input_tensors = (self->n_inputs-1)/3;
	const struct tensor **input_tensors = &self->inputs[1];
	const struct tensor **min_tensors = &self->inputs[1+n_input_tensors];
	const struct tensor **max_tensors = &self->inputs[1+2*n_input_tensors];
	const struct tensor *t;
	struct tensor *out_tensor = self->outputs[0];
	uint8_t *out_data = out_tensor->data;
	uint32_t i;
	float in_min;
	float in_max; 
	//float in_off; 
	float in_level;
	float out_min = info->out_min;
	float out_max = info->out_max;
	float out_level_recip;
	uint32_t iters = 0;
	uint32_t total_depth = info->total_depth;
	uint32_t stride;

	out_level_recip = 255.0f/(out_max-out_min);
	stride = total_depth;
	for (i = 0; i < n_input_tensors; i++) {
		t = input_tensors[i];
		if ((i & 1) != whoami) {
			out_data += t->shape.depth;
			continue;
		}
		// FIXME: l2fetch(t->data,t->shape.depth*t->shape.width,t->shape.height);
		in_min = tensor_get_float(min_tensors[i],0);
		in_max = tensor_get_float(max_tensors[i],0);
		in_level = (in_max-in_min)/255.0f;
		//in_off = out_min-in_min;
		iters = t->shape.width * t->shape.height * t->shape.batches;
		do_convert(nn,
			out_data,
			t->data,
			iters,
			t->shape.depth,
			stride,
			in_level,
			in_min,
			out_min,
			out_level_recip);
		out_data += t->shape.depth;
	}
	nn_sem_post(&info->donesem);
}

static int concat_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1)/3;
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	const struct tensor **min_tensors = &self->inputs[1+n_input_tensors];
	const struct tensor **max_tensors = &self->inputs[1+2*n_input_tensors];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	uint32_t in_width = self->inputs[1]->shape.width;
	uint32_t in_height = self->inputs[1]->shape.height;
	uint32_t in_batches = self->inputs[1]->shape.batches;
	uint32_t i;
	//float in_off; 
	float out_min = tensor_get_float(min_tensors[0],0);
	float out_max = tensor_get_float(max_tensors[0],0);
	uint32_t out_bytes = 0;
	uint32_t total_depth = 0;
	struct tdata my_info;
	struct tdata worker_info;
	logmsg(nn,2,"concat execute. self=%p ",self);
	if (tensor_get_int32(dim_tensor,0) != 3) return errlog(nn,"only depth");
	for (i = 0; i < n_input_tensors; i++) {
		if (input_tensors[i]->shape.width != in_width) {
			return errlog(nn,"width mismatch tensor %d",i);
		}
		if (input_tensors[i]->shape.height != in_height) {
			return errlog(nn,"height mismatch tensor %d",i);
		}
		if (input_tensors[i]->shape.batches != in_batches) {
			return errlog(nn,"batches mismatch tensor %d",i);
		}
		out_min = fminf(out_min,tensor_get_float(min_tensors[i],0));
		out_max = fmaxf(out_max,tensor_get_float(max_tensors[i],0));
		out_bytes += tensor_elements(input_tensors[i]);
		total_depth += input_tensors[i]->shape.depth;
	}
	if (out_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);
	tensor_set_shape(out_tensor,in_batches,in_height,in_width,total_depth);
	out_tensor->data_size = out_bytes;

	my_info.self = worker_info.self = self;
	my_info.out_min = worker_info.out_min = out_min;
	my_info.out_max = worker_info.out_max = out_max;
	my_info.total_depth = worker_info.total_depth = total_depth;
	my_info.whoami = 0;
	worker_info.whoami = 1;

	nn_sem_init(&worker_info.donesem,0);

	nn_os_work_for_vector(nn,concat_execute_slice,&worker_info);
	//concat_execute_slice(nn,&worker_info);
	concat_execute_slice(nn,&my_info);
	nn_sem_wait(&worker_info.donesem);

	logmsg(nn,2,"concat %p done",self);
	return 0;
}


static int concat_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking concat node %p",self);
	if ((self->n_inputs - 1) % 3) return errlog(nn,"input triplets please");
	if (self->n_inputs < 4) return errlog(nn,"at least 1 input");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	for (i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) return errlog(nn,"NULL input");
	}
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) return errlog(nn,"NULL output");
	}
	logmsg(nn,2,"concat node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedConcat_8 = {
	.execute = concat_execute,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


struct nn_node_ops nn_ops_for_QuantizedConcat_8_ref = {
	.execute = concat_execute,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

