
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
 * This contains implementations for requantizing
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>

static int do_autorequantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	uint32_t i;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min;
	float out_max;
	float stepsize;
	float recip_stepsize;
	int32_t in_max_val;
	int32_t in_min_val;
	int32_t inval;
	float in_level_size = (in_max - in_min) / 4294967296.0f/*0x1.0p32f*/;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"autorequantize execute. self=%p ",self);
	logmsg(nn,2,"autorequantize in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	if( tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!=0){
		return errlog(nn,"out too small");
	}

	/* Find min and max quantized 32 bit val */
	in_max_val = INT32_MIN;
	in_min_val = INT32_MAX;
	for (i = 0; i < elements; i++) {
		inval = in_data[i];
		if (inval > in_max_val) in_max_val = inval;
		if (inval < in_min_val) in_min_val = inval;
	}
	/* Make sure min val <= 0.0 in floaty land */
	out_min = in_level_size * (float)in_min_val;
	out_max = in_level_size * (float)in_max_val;

	quantize_adjust_range(
		&out_min,&out_max,
		&stepsize,&recip_stepsize,
		out_min,out_max);

	/* Requantize with new range */
	/* For elements < 128, HVX code seem to have issue, for now fall back to reference C code */
	if(elements < 128) {
		for (i = 0; i < elements; i++) {
			/* FIXME: in HVX we will need to do this in fixed point... */
			out_data[i] = quantize_uint8(
				in_level_size* (float)in_data[i],out_min,out_max);
		}
	} else {
		float gain = recip_stepsize*in_level_size;
		int gaini = (int) (gain*2147483648.0f/*0x1.0p31f*/+0.5f);
		/* 
		 * Since out min is not necessarily in_min, we need to make 
		 * sure we have exactly the right offset
		 */
		int offseti = out_min / in_level_size;
		quantize_asm(in_data, offseti, gaini, out_data, elements);
	}
	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"autorequantize out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"autorequantize %p done",self);
	return 0;
}

static int do_requantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *min_val_tensor = self->inputs[3];
	const struct tensor *max_val_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min = tensor_get_float(min_val_tensor,0);
	float out_max = tensor_get_float(max_val_tensor,0);
	float in_level_size = (in_max - in_min) / 4294967296.0f/*0x1.0p32f*/;
	float recip_stepsize;
	float stepsize;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"autorequantize execute. self=%p ",self);
	logmsg(nn,2,"autorequantize in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!= 0 ){
		return errlog(nn,"out too small");
	}

	quantize_adjust_range(
		&out_min,&out_max,
		&stepsize,&recip_stepsize,
		out_min,out_max);

	/* Requantize with new range */
	/* For elements < 128, HVX code seem to have issue, for now fall back to reference C code */
	if(elements < 128)
	{
		uint32_t i;
		for (i = 0; i < elements; i++) {
			/* FIXME: in HVX we will need to do this in fixed point... */
			out_data[i] = quantize_uint8(
				in_level_size* (float)in_data[i],out_min,out_max);
		}
	}
	else
	{
		float gain = (255.f *in_level_size)/(out_max-out_min);
		int gaini = (int) (gain*2147483648.0f/*0x1.0p31f*/+0.5f);
		int offseti = out_min / in_level_size;
		quantize_asm(in_data, offseti, gaini, out_data, elements);
	}

	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"requantize out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"requantize %p done",self);
	return 0;
}

static int do_requantrange_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_min_tensor = self->outputs[0];
	struct tensor *out_max_tensor = self->outputs[1];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint32_t i;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min;
	float out_max;
	int32_t in_max_val;
	int32_t in_min_val;
	int32_t inval;
	float in_level_size = (in_max - in_min) / 4294967296.0f/*0x1.0p32f*/;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"requantrange execute. self=%p ",self);
	logmsg(nn,2,"requantrange in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	/* Find min and max quantized 32 bit val */
	in_max_val = INT32_MIN;
	in_min_val = INT32_MAX;
	for (i = 0; i < elements; i++) {
		inval = in_data[i];
		if (inval > in_max_val) in_max_val = inval;
		if (inval < in_min_val) in_min_val = inval;
	}
	/* Make sure min val <= 0.0 in floaty land */
	out_min = in_level_size * (float)in_min_val;
	out_max = in_level_size * (float)in_max_val;
	if (out_min > 0.0f) out_min = 0.0f;

	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"requantrange out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"requantrange %p done",self);
	return 0;
}

struct tdata {
	int (*f)(struct nn_node *self, struct nn_graph *nn);
	int retval;
	struct nn_node *self;
	nn_sem_t donesem;
};

void worker(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	td->retval = td->f(td->self,nn);
	nn_sem_post(&td->donesem);
}

int launcher(struct nn_node *self, struct nn_graph *nn, 
	int (*f)(struct nn_node *self, struct nn_graph *nn))
{
	struct tdata td = {
		.f = f,
		.self = self,
		.retval = 0,
	};
	nn_sem_init(&td.donesem,0);
	nn_os_work_for_vector(nn,worker,&td);
	nn_sem_wait(&td.donesem);
	return td.retval;
}

static int requantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_requantize_execute);
}

static int requantrange_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_requantrange_execute);
}
static int autorequantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_autorequantize_execute);
}


static int autorequantize_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking autorequantize node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	for (i = 0; i < 3; i++) {
		if (self->inputs[i] == NULL) return errlog(nn,"NULL input");
		if (self->outputs[i] == NULL) return errlog(nn,"NULL output");
	}
	logmsg(nn,2,"autorequantize node %p check OK",self);
	return 0;
}

static int requantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking requantize node %p",self);
	if (self->n_inputs != 5) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"requantize node %p check OK",self);
	return 0;
}

static int requantrange_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking requantrange node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 2) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"requantrange node %p check OK",self);
	return 0;
}

//
// QuantizeDownAndShrinkRange_32to8:
// convert 32-bit quantized to 8-bit quantized, checking the actual
// range of values to find the output range.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0:   qu8 tensor
//  output 1,2:  scalar float, output min & max
//
struct nn_node_ops nn_ops_for_QuantizeDownAndShrinkRange_32to8 = {
	.execute = autorequantize_execute,
	.check = autorequantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizeDownAndShrinkRange_32to8_ref = {
	.execute = autorequantize_execute,
	.check = autorequantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

//
// Requantize_32to8:
// convert 32-bit quantized to 8-bit quantized, relying on
// supplied range (inputs 3 & 4) to determine range.
// The output range may be expanded slightly to produce a
// a properly aligned zero code.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//  input 3,4:  scalar float, output min & max
//
//  output 0:   qu8 tensor
//  output 1,2:  scalar float, input min & max
//
struct nn_node_ops nn_ops_for_Requantize_32to8 = {
	.execute = requantize_execute,
	.check = requantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Requantize_32to8_ref = {
	.execute = requantize_execute,
	.check = requantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
//
// RequantizationRange_32
// find the range of values in qi32 data, and produce this
// range as min,max floats
// Note that the range is expanded if needed so that min <=0,
// but is not adjusted to have an integer zero point.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0,1  scalar float, input actual min & max
//

struct nn_node_ops nn_ops_for_RequantizationRange_32 = {
	.execute = requantrange_execute,
	.check = requantrange_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_RequantizationRange_32_ref = {
	.execute = requantrange_execute,
	.check = requantrange_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


