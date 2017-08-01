
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
#include <nn_asm_ops.h>
#include <string.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for an input node.
 */

static int input_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"output execute. self=%p ",self);
	int i;
	struct tensor *out;
	const struct tensor *in;
	/* Copy input tensor to output */
	if (nn->n_inputs != self->n_outputs) return errlog(nn,"oops, input #");
	for (i = 0; i < self->n_outputs; i++) {
		out = self->outputs[i];
		in = &nn->inputs[i];
		/* Warning! Inputs come in as max_size not data_size! */
		if (out->max_size < in->max_size) return errlog(nn,"out too small: %d < %d",out->max_size,in->max_size);
		out->shape = in->shape;
		out->data_size = in->max_size;
		vmemcpy_asm(out->data,in->data,in->max_size);
	}
	logmsg(nn,2,"input %d tensors",nn->n_inputs);
	return 0;
}

static int input_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking input node %p",self);
	int i;
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) {
			return errlog(nn,0,"input: fatal: NULL output");
		}
	}
	logmsg(nn,2,"input node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_INPUT = {
	SFINIT(.execute, input_execute),
	SFINIT(  .check, input_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

