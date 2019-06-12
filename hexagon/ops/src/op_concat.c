
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

static void concat_execute_slice_ref(struct nn_graph *nn, void *vinfo)
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
	float in_level;
	float out_min = info->out_min;
	float out_max = info->out_max;
	float out_level_recip;
	uint32_t iters = 0;
	uint32_t total_depth = info->total_depth;
	uint32_t stride;
	int offset;
	int gain;
	short gains;
	uint8_t *out;
	const uint8_t *in; 
	int k,l;
	int oval, ival;

	out_level_recip = 255.0f/(out_max-out_min);
	stride = total_depth;
	for (i = 0; i < n_input_tensors; i++) {
		t = input_tensors[i];
		if ((i & 1) != whoami) {
			out_data += t->shape.depth;
			continue;
		}
		in_min = tensor_get_float(min_tensors[i],0);
		in_max = tensor_get_float(max_tensors[i],0);
		if (in_min > 0.0f) {
			in_min = 0.0f; // comport with op_quantize use of quantize_adjust_range setting minval = fminf(0.0f,min);
		}
		in_level = flt_div_255(in_max-in_min);
		iters = t->shape.width * t->shape.height * t->shape.batches;
		l2fetch(t->data, t->shape.depth, t->shape.depth, iters);
		offset = max_i32(0,roundf_i32((in_min-out_min)/in_level));
		gain = roundf_i32(out_level_recip*in_level* 32768.0f );
		if (gain > 32767) {
			gains  = 32767; 
		} else {
			gains = (short) gain;
		}
		out = out_data;
		in = t->data;
		for (k = 0; k < iters; k++) {
			for (l = 0; l < t->shape.depth; l++) {
				ival = in[l];
				oval = ((ival + offset)* gains + (1<<14))>>15;
				if (oval > 255) {
					oval = 255; 
				} else if (oval < 0) {
					oval = 0;
				}
				out[l] = oval;
			}
			in += t->shape.depth;
			out += stride;
		}
		out_data += t->shape.depth;
	}
	nn_sem_post(&info->donesem);
}

static void concat_execute_slice_asm(struct nn_graph *nn, void *vinfo)
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
	float in_level;
	float out_min = info->out_min;
	float out_max = info->out_max;
	float out_level_recip;
	uint32_t iters = 0;
	uint32_t total_depth = info->total_depth;
	uint32_t stride;
	int offset;
	int gain;
	short gains;
	uint8_t *out;
	const uint8_t *in; 

	out_level_recip = 255.0f/(out_max-out_min);
	stride = total_depth;
	for (i = 0; i < n_input_tensors; i++) {
		t = input_tensors[i];
		out = out_data;
		in = t->data;
		uint32_t in_depth = t->shape.depth;
		if ((i & 1) != whoami) {
			out_data += in_depth;
			continue;
		}
		in_min = tensor_get_float(min_tensors[i],0);
		in_max = tensor_get_float(max_tensors[i],0);
		if (in_min > 0.0f) {
			in_min = 0.0f; // comport with op_quantize use of quantize_adjust_range setting minval = fminf(0.0f,min);
		}
		in_level = flt_div_255(in_max-in_min);
		iters = t->shape.width * t->shape.height * t->shape.batches;
		l2fetch((void*)in, in_depth, in_depth, iters);
		offset = max_i32(0,roundf_i32((in_min-out_min)/in_level));
		gain = roundf_i32(out_level_recip*in_level*32768.0f/*0x1.0p15f*/);
		if (gain > 32767) {
			gains  = 32767; 
		} else {
			gains = (short) gain;
		}
		if( offset == 0 && gain >= 0x7fc0) {	// is unity gain (0->0, 255->255)
			vmemcpy_2d_general_asm(
					in_depth,			// bytes wide
			      iters,			//rows
			      out,			// destination address, any allowed
			      stride,		// row pitch of dest; any allowed
			      in,			// source address, any allowed
			      in_depth);	// source stride, any
		}else{
			memconvert_hvx(out, in, in_depth, offset, gains, stride, iters);
		}
		out_data += in_depth;
	}
	nn_sem_post(&info->donesem);
}

static int concat_execute(struct nn_node *self, struct nn_graph *nn,
		void (*concat_execute_slice_f)(struct nn_graph *self, void *vinfo))
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
	float out_min = tensor_get_float(min_tensors[0],0);
	float out_max = tensor_get_float(max_tensors[0],0);
	uint32_t total_depth = 0;
	struct tdata worker0_info;
	struct tdata worker1_info;
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
		total_depth += input_tensors[i]->shape.depth;
	}
	if (out_min > 0.0f) {
		out_min = 0.0f; // comport with op_quantize use of quantize_adjust_range setting minval = fminf(0.0f,min);
	}
	if( tensor_out_prepare_normal( out_tensor,in_batches,in_height,in_width,total_depth, NN_TYPE_QUINT8 )!=0){
		return errlog(nn,"out too small");
	}
	tensor_set_single_float(out_min_tensor, out_min);
	tensor_set_single_float(out_max_tensor, out_max);

	worker1_info.self = worker0_info.self = self;
	worker1_info.out_min = worker0_info.out_min = out_min;
	worker1_info.out_max = worker0_info.out_max = out_max;
	worker1_info.total_depth = worker0_info.total_depth = total_depth;
	worker0_info.whoami = 0;
	worker1_info.whoami = 1;

	nn_sem_init(&worker0_info.donesem,0);
	nn_sem_init(&worker1_info.donesem,0);

	nn_os_work_for_vector(nn,concat_execute_slice_f,&worker0_info);
	nn_os_work_for_vector(nn,concat_execute_slice_f,&worker1_info);
	//concat_execute_slice_f(nn,&worker_info);
	//concat_execute_slice_f(nn,&my_info);
	nn_sem_wait(&worker1_info.donesem);
	nn_sem_wait(&worker0_info.donesem);

	logmsg(nn,2,"concat %p done",self);
	return 0;
}

static int concat_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	return concat_execute(self,nn,concat_execute_slice_ref);
}

static int concat_execute_asm(struct nn_node *self, struct nn_graph *nn)
{
	return concat_execute(self,nn,concat_execute_slice_asm);
}

static int concat_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking concat node %p",self);

	// must be 3*n+1 inputs, where n >= 1

	int n_in = (self->n_inputs - 1) /3;	// actual # of inputs
	if (n_in < 1 || (self->n_inputs - 1) % 3 !=0 )
		return errlog(nn,"concat: inputs must be 3*n+1, n>=1");


	logmsg(nn,2,"concat node %p check OK",self);
	return 0;
}

// TODO: remove when deconv is "conv" optimized, i.e. all its nodes can be d32
struct nn_node_ops nn_ops_for_QuantizedConcat_8_nond32 = {
	.execute = concat_execute_asm,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedConcat_8 = {
	.execute = concat_execute_asm,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
};


struct nn_node_ops nn_ops_for_QuantizedConcat_8_ref = {
	.execute = concat_execute_ref,
	.check = concat_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
};

