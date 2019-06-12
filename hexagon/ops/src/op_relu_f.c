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
#include <quantize.h>

//
// underlying operator for relu:
//   out = max( in , 0 )
//
static void relu_operator( float * out_data, float const * in_data, int elements, void *info)
{
	int i;
	int nloop = (elements-1)>>1;
	float x0 = in_data[0];
	float x1;
	for (i = 0; i < nloop; i++) {
		float x1 = in_data[1];
		out_data[0] = fmaxf(x0,0.0f);
		out_data[1] = fmaxf(x1,0.0f);
		x0 = in_data[2];
		out_data += 2;
		in_data += 2;
	}
	if( (elements&1)== 0){
		x1 = in_data[1];
		*out_data++ =  fmaxf(x0,0.0f);
		x0 = x1;
	}
	*out_data =  fmaxf(x0,0.0f);
}

static int relu_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"relu execute. self=%p ",self);
	int res = nn_generic_unary_float_op( self,nn, relu_operator, NULL,0);
	if (res == 0)
		logmsg(nn,2,"relu %p done",self);
	return res;
}

//
// underlying operator for reluX and clamp_f
//   out = min( maxval, max( in , minval ))
//  where 'info' points to [minval,maxval]
//
void clamp_operator( float * __restrict out_data, float const * __restrict in_data, int elements, void *info)
{
	float const * parms = (float const*)info;
	float minval = parms[0];
	float maxval = parms[1];

	int i;
	int nloop = (elements-1)>>1;
	float x0 = in_data[0];
	float x1;
	for (i = 0; i < nloop; i++) {
		float x1 = in_data[1];
		out_data[0] = fminf(maxval,fmaxf(x0,minval));
		out_data[1] = fminf(maxval,fmaxf(x1,minval));
		x0 = in_data[2];
		out_data += 2;
		in_data += 2;
	}
	if( (elements&1)== 0){
		x1 = in_data[1];
		*out_data++ =  fminf(maxval,fmaxf(x0,minval));
		x0 = x1;
	}
	*out_data =  fminf(maxval,fmaxf(x0,minval));
}

static int reluX_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"reluX execute. self=%p ",self);
	float max_val = tensor_get_float(self->inputs[1],0);
	if( !(max_val > 0.0f))
		return errlog(nn,"reluX limit %f not > 0", max_val);
	float clamp_parms[2] = { 0.0f, max_val };

	int res = nn_generic_unary_float_op( self,nn, clamp_operator, clamp_parms,0);
	if (res == 0)
		logmsg(nn,2,"reluX %p done",self);
	return res;
}

static int clamp_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"clamp execute. self=%p ",self);
	const float minval = tensor_get_float(self->inputs[1],0);
	const float maxval = tensor_get_float(self->inputs[2],0);
	if( !(maxval > minval))
		return errlog(nn,"clamp: min %f not < max %f", minval, maxval);
	float clamp_parms[2] = { minval, maxval };

	int res = nn_generic_unary_float_op( self,nn, clamp_operator, clamp_parms,0);
	if (res == 0)
		logmsg(nn,2,"clamp %p done",self);
	return res;
}


struct nn_node_ops nn_ops_for_Relu_f = {
	.execute = relu_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_ReluX_f = {
	.execute = reluX_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Clamp_f = {
	.execute = clamp_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};
