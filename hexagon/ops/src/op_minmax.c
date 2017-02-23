
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
#include <nn_broadcast.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains min and max (floating) ops
 */

#if 0
static inline int minmax_execute(struct nn_node *self, struct nn_graph *nn, float (*f)(float,float))
{
	const struct tensor *in_tensor = self->inputs[0];
	//const struct tensor *reduction_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	int32_t batches = in_tensor->shape.batches;
	int32_t height = in_tensor->shape.height;
	int32_t width = in_tensor->shape.width;
	int out_elements = batches;
	int depth = in_tensor->shape.depth;
	const float *data = in_tensor->data;
	float *out = out_tensor->data;
	float minmax = data[0];
	int i;
	int j;
	size_t bytes = out_elements * sizeof(float);
	logmsg(nn,2,"min/max execute. self=%p ",self);
	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	for (j = 0; j < out_elements; j++) {
		minmax = *data;
		for (i = 0; i < height*width*depth; i++) {
			minmax = f(minmax,*data++);
		}
		*out++ = minmax;
	}
	tensor_set_shape(out_tensor,batches,1,1,1);
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
#else

#include <nn_reduction.h>

static int min_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_reduction_float(self,nn,fminf,INFINITY);
}

static int max_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_reduction_float(self,nn,fmaxf,-INFINITY);
}

#endif

static int minimum_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_f(self,nn,fminf);
}

static int maximum_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_f(self,nn,fmaxf);
}

static int minmax_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking min/max node %p",self);
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	if (self->inputs[0] == NULL) return errlog(nn,"NULL input 0");
	if (self->outputs[0] == NULL) return errlog(nn,"NULL output 0");
	if (self->n_inputs > 3) return errlog(nn,"wrong # inputs");
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


struct nn_node_ops nn_ops_for_Minimum_f = {
	.execute = minimum_execute,
	.check = minmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Maximum_f = {
	.execute = maximum_execute,
	.check = minmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
