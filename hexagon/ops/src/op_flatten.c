
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
 * This contains a node to flatten to a 1D array... basically fancy nop
 */

static int flatten_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t batches = self->inputs[0]->shape.batches;
	uint32_t height = self->inputs[0]->shape.height;
	uint32_t width = self->inputs[0]->shape.width;
	uint32_t depth = self->inputs[0]->shape.depth;
	uint32_t elements = batches*height*width*depth;
	uint32_t data_size = in_tensor->data_size;

	logmsg(nn,2,"flatten execute. self=%p ",self);
	if (out_tensor->max_size < data_size) {
		return errlog(nn,"out too small. %p  Need %d, have %d. (for %d*%d*%d*%d)",
			      out_tensor,
			      in_tensor->data_size,
			      out_tensor->max_size,
			      batches, height, width, depth
			);
	}
	/* Copy input tensor to output */
	tensor_set_shape(out_tensor,1,1,1,elements);
	out_tensor->format = in_tensor->format;
	out_tensor->data_size = data_size;
	if( data_size> 0 && in_tensor->data != out_tensor->data){
		struct nn_memcpy_manager  mcman;
		nn_mcmanager_init(nn, &mcman );
		nn_mcmanager_vmemcpy( nn, &mcman, out_tensor->data, in_tensor->data, data_size );
		// wait for copy thread(s) if any
		nn_mcmanager_wait( nn, &mcman );
		//memcpy(out_tensor->data,in_tensor->data,in_tensor->data_size);
	}
	logmsg(nn,2,"copied tensor %d bytes of data",data_size);
	return 0;
}

static int qflatten_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t batches = self->inputs[0]->shape.batches;
	uint32_t height = self->inputs[0]->shape.height;
	uint32_t width = self->inputs[0]->shape.width;
	uint32_t depth = self->inputs[0]->shape.depth;
	uint32_t elements = batches*height*width*depth;
	uint32_t data_size = in_tensor->data_size;

	logmsg(nn,2,"qflatten execute. self=%p ",self);
	if (out_tensor->max_size < data_size) {
		return errlog(nn,"out too small");
	}
	/* Copy input tensor to output */
	tensor_set_shape(out_tensor,1,1,1,elements);
	out_tensor->format = in_tensor->format;
	out_tensor->data_size = data_size;
	if( data_size> 0 && in_tensor->data != out_tensor->data){
		struct nn_memcpy_manager  mcman;
		nn_mcmanager_init(nn, &mcman );
		nn_mcmanager_vmemcpy( nn, &mcman, out_tensor->data, in_tensor->data, data_size );
		// wait for copy thread(s) if any
		nn_mcmanager_wait( nn, &mcman );
		//memcpy(out_tensor->data,in_tensor->data,in_tensor->data_size);
	}
	tensor_copy(self->outputs[1],self->inputs[1]);
	tensor_copy(self->outputs[2],self->inputs[2]);
	logmsg(nn,2,"copied tensor %d bytes of data",data_size);
	return 0;
}


// supports aliasing: so it is possible the input and output
// tensors will be at the same address.

// note: some graphs have a second input
// on Flatten, which is ignored. Not sure
// what it's for but we need to tolerate it
struct nn_node_ops nn_ops_for_Flatten = {
	.execute = flatten_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS,
	.n_inputs = NN_IOCOUNT_RANGE(1,2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedFlatten = {
	.execute = qflatten_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
