
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for an output node.
 */

static int output_execute(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	struct tensor *out;
	const struct tensor *in;
	logmsg(nn,2,"output execute. self=%p ",self);
	if (nn->n_outputs != self->n_inputs) return errlog(nn,"bad # outputs");
	for (i = 0; i < nn->n_outputs; i++) {
		in = self->inputs[i];
		out = &nn->outputs[i];
		if (out->max_size < in->data_size) {
			return errlog(nn,"output %d too small",i);
		}
		out->shape = in->shape;
		out->data_size = in->data_size;
		memcpy(out->data,in->data,in->data_size);
	}
	/* Copy input tensor to output */
	logmsg(nn,2,"copied %d tensors",self->n_inputs);
	return 0;
}

static int output_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking output node %p",self);
	for (i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) return errlog(nn,"output: NULL input");
	}
	logmsg(nn,2,"output node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_OUTPUT = {
	SFINIT(.execute, output_execute),
	SFINIT(  .check, output_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

