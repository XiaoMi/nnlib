
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
#include <nn_graph.h>
#include <string.h>
#include <math.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains min and max (floating) ops
 */


static inline int minmax_execute(struct nn_node *self, struct nn_graph *nn, float (*f)(float,float))
{
	int i;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int n_elements = in_tensor->shape.depth;
	const float *data = in_tensor->data;
	float minmax = data[0];
	logmsg(nn,2,"min/max execute. self=%p ",self);
	if (in_tensor->shape.batches != 1) return errlog(nn,"want 1D");
	if (in_tensor->shape.width != 1) return errlog(nn,"want 1D");
	if (in_tensor->shape.height != 1) return errlog(nn,"want 1D");
	for (i = 0; i < n_elements; i++) {
		minmax = f(minmax,data[i]);
	}
	tensor_set_shape(out_tensor,1,1,1,1);
	out_tensor->data_size = sizeof(float);
	tensor_set_float(out_tensor,0,minmax);
	return 0;
}

static int min_execute(struct nn_node *self, struct nn_graph *nn)
{
	return minmax_execute(self,nn,fminf);
}

static int max_execute(struct nn_node *self, struct nn_graph *nn)
{
	return minmax_execute(self,nn,fmaxf);
}

static int minmax_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking min/max node %p",self);
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	if (self->inputs[0] == NULL) return errlog(nn,"NULL input 0");
	if (self->outputs[0] == NULL) return errlog(nn,"NULL output 0");
	if (self->n_inputs > 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # inputs");
	logmsg(nn,2,"min/max node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Min_f = {
	.execute = min_execute,
	.check = minmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Min_f_ref = {
	.execute = min_execute,
	.check = minmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Max_f = {
	.execute = max_execute,
	.check = minmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Max_f_ref = {
	.execute = max_execute,
	.check = minmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

