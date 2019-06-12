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
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

//#define TEST_PERFORMANCE

// this does (slow, reference) 8-bit quantized tanh and sigmoid functions.
// The two are the same except for changes in input and output scaling.

static int qtanh_execute_ref(struct nn_node *self, struct nn_graph *nn)
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
	float stepsize = flt_div_255(in_max - in_min);

	logmsg(nn,2,"tanh/sigmoid execute. self=%p ",self);
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!= 0){
		return errlog(nn,"out too small");
	}

	// datapath is:
	//       tmp = tanh( in[i]*ingain + inoffs );
	//       out = (tmp + 1.0)*outgain;
	// with ingain, inoffs, outgain depending the function we're doing
	// and the input range.
	//
	float ingain = stepsize;
	float inoff =  in_min;
	float outgain;
	float out_min, out_max;

	int is_sigmoid = (self->node_type == OP_QuantizedSigmoid_8_ref);
	if( is_sigmoid){
		ingain *= 0.5f;
		inoff *= 0.5f;
		// we want 0.5*(1+tanh)*255   = (tanh+1)*(255/2)
		outgain = 255.0f*0.5f;
		out_min = 0.0f;
		out_max = 1.0f;
	}else{
		// range is -1.0 to 1.00787 (128/127)  (zero code = 127)
		// so scaling is (tanh+1)*(254/2)  -- max output code is 254.
		out_min = -1.0f;
		out_max = 128.0/127.0;
		outgain = 127.0f;
	}


#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	for (i = 0; i < elements; i++) {
		inval = inoff + ingain * in_data[i];
		tmpval = tanhf(inval);
		outval = (tmpval + 1.0f) * outgain;
		out_data[i] = saturate_u8( roundf_i32(outval));
	}

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qtanh ref cycles = %d (elements = %d)\n", (end_time-start_time), elements);
#endif
	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"tanh/sigmoid %p done",self);
	return 0;
}
/*
static int qtanh_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking tanh/sigmoid node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"tanh/sigmoid node %p check OK",self);
	return 0;
}*/

struct nn_node_ops nn_ops_for_QuantizedTanh_8_ref = {
	.execute = qtanh_execute_ref,
	.check = NULL,// qtanh_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedSigmoid_8_ref = {
	.execute = qtanh_execute_ref,
	.check = NULL,// qtanh_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

// NOTE: the 'flat' QuantizedTanh_8 and QuantizedSigmoid_8 are done with hvx ops in op_tanh_d32.c
//
