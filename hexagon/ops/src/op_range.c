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
#include <math.h>

static int range_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *start_tensor = self->inputs[0];
	const struct tensor *limit_tensor = self->inputs[1];
	const struct tensor *delta_tensor = self->inputs[2];
	int32_t start = tensor_get_int32(start_tensor,0);
	int32_t limit = tensor_get_int32(limit_tensor,0);
	int32_t delta = tensor_get_int32(delta_tensor,0);
	struct tensor *out_tensor = self->outputs[0];
	int elements;
	int32_t *out_data = out_tensor->data;
	int i;

	logmsg(nn,2,"range execute. self=%p ",self);
	elements = 1;		// in case start == limit
	if (start < limit) {
		if( delta < 1) delta = 1;
		// (limit-start)/delta, rounding up
		elements = (unsigned)( limit-start-1)/(unsigned)delta + 1;
	} else if( start > limit){
		if( delta > -1) delta = -1;
		elements = (unsigned)( start-limit-1)/(unsigned)(-delta) + 1;
	}

	if( tensor_out_prepare_normal( out_tensor, 1,1,1,elements, NN_TYPE_INT32)!= 0)
		return errlog(nn,"out too small");

	for( i = 0; i < elements; i++ ){
		out_data[i] = start + delta*i;
	}

	logmsg(nn,2,"range %p done",self);
	return 0;
}


struct nn_node_ops nn_ops_for_Range_int32 = {
	.execute = range_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};

