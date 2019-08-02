
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a dummy node that you can attach things to, especially for making nodes not get optimized away.
 */

static int sink_execute(struct nn_node *self, struct nn_graph *nn)
{
	return 0;
}


static int sink_dtor(struct nn_node *self, struct nn_graph *nn)
{
	self->opaque = NULL;
	return node_free_common(self,nn);
}

static int sink_earlywork_register(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work)
{
	if (self->opaque == NULL) {
		/* Maybe this should just be a logmsg and return instead of returning error */
		logmsg(nn,2,"node %p: Oops, no predecessor registered.",self);
		return 0;
	}
	struct nn_node *pred = self->opaque;
	/* Now, pass the buck */
	return pred->ops->earlywork_register(pred,nn,work);
}

static int sink_earlywork_note_pred(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor)
{
	if (predecessor == NULL) return errlog(nn,"Oops: NULL predecessor");
	logmsg(nn,2,"node %p: note predecessor %p",self,predecessor);
	if (predecessor->ops->earlywork_register != NULL) self->opaque = predecessor;
	else logmsg(nn,2,"predecessor %p has no early work registration function",predecessor);
	return 0;
}

struct nn_node_ops nn_ops_for_Sink = {
	.execute = sink_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = sink_dtor,
	.earlywork_note_pred = sink_earlywork_note_pred,
	.earlywork_register = sink_earlywork_register,
	.n_inputs = NN_IOCOUNT_GE(0),		// # any # of inputs
	.n_outputs = NN_IOCOUNT(0),			// no outputs
};

