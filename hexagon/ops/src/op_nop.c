
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a dummy pass-through node ("nop")
 */

static int nop_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"nop execute. self=%p ",self);
	/* These checks for initial bringup, probably not necessary after
	 * framework is going pretty well, since we will have separate checks
	 */
	if (self->inputs == NULL) {
		return errlog(nn,"nop: fatal: NULL inputs");
	}
	if (self->outputs == NULL) {
		return errlog(nn,"nop: fatal: NULL outputs");
	}
	if (self->inputs[0] == NULL) {
		return errlog(nn,"nop: fatal: NULL input 0");
	}
	if (self->outputs[0] == NULL) {
		return errlog(nn,"nop: fatal: NULL output 0");
	}
	/* Copy input tensor to output */
	self->outputs[0]->shape = self->inputs[0]->shape;
	self->outputs[0]->data_size = self->inputs[0]->data_size; // FIXME: check vs. max size
	memcpy(self->outputs[0]->data,self->inputs[0]->data,self->inputs[0]->data_size);
	logmsg(nn,2,"copied tensor %d bytes of data",self->inputs[0]->data_size);
	return 0;
}

static int nop_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking nop node %p",self);
	if (self->inputs == NULL) {
		return errlog(nn,"nop: fatal: NULL inputs");
	}
	if (self->outputs == NULL) {
		return errlog(nn,"nop: fatal: NULL outputs");
	}
	if (self->inputs[0] == NULL) {
		return errlog(nn,"nop: fatal: NULL input 0");
	}
	if (self->outputs[0] == NULL) {
		return errlog(nn,"nop: fatal: NULL output 0");
	}
	logmsg(nn,2,"nop node %p check OK",self);
	return 0;
}

static struct nn_node *nop_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	logmsg(nn,2,"nop node id %x ctor",node_id);
	return node_alloc_common(
		nn,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs);
}

static int nop_dtor(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"nop node %p dtor",self);
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_Nop = {
	.execute = nop_execute,
	.check = nop_check,
	.ctor = nop_ctor,
	.dtor = nop_dtor,
};

