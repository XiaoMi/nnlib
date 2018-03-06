
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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

static void input_execute_worker(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	int i;
	struct tensor *out;
	const struct tensor *in;
	/* Copy input tensor to output */
	logmsg(nn,2,"input execute. self=%p ",self);
	for (i = 0; i < self->n_outputs; i++) {
		out = self->outputs[i];
		in = &nn->inputs[i];
		/* Warning! Inputs come in as max_size not data_size! */
		logmsg(nn,9,"in: [%d,%d,%d,%d] size=%d",
			in->shape.batches,
			in->shape.height,
			in->shape.width,
			in->shape.depth,
			in->max_size);
		if (out->max_size < in->max_size) {
			errlog(nn,"out too small (%lu < %lu)", out->max_size, in->max_size);
			out->shape.batches = 0;
			out->shape.height = 0;
			out->shape.width = 0;
			out->shape.depth = 0;
			out->data_size = 0;  // Communicate the failure upward
			return;
		}
		out->shape = in->shape;
		out->data_size = in->max_size;
		vmemcpy_asm(out->data,in->data,in->max_size);
	}
	nn_sem_post(self->opaque);
	logmsg(nn,2,"input %d tensors",nn->n_inputs);
}

static int input_execute(struct nn_node *self, struct nn_graph *nn)
{
	if (nn->n_inputs != self->n_outputs) return errlog(nn,"Expected %d, got %d inputs",self->n_outputs,nn->n_inputs);
	nn_sem_t donesem;
	nn_sem_init(&donesem,0);
	self->opaque = &donesem;
	nn_os_work_for_vector(nn,input_execute_worker,self);
	nn_sem_wait(&donesem);
	self->opaque = NULL;
	for (int i = 0; i < self->n_outputs; i++) {
		if (self->outputs[0]->data_size == 0) {
			return errlog(nn,"op_input: Worker is telling us there was no valid input for output %d",i);
		}
	}
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
	.execute = input_execute,
	.check = input_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

