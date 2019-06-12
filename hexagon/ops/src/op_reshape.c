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
/*
static void do_reshape(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	vmemcpy_asm(out_tensor->data,in_tensor->data,in_tensor->data_size);
	nn_sem_post(self->opaque);
}*/

static int reshape_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *shape_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t b = in_tensor->shape.batches;
	uint32_t h = in_tensor->shape.height;
	uint32_t w = in_tensor->shape.width;
	uint32_t d = in_tensor->shape.depth;
	uint32_t elements = b*h*w*d;
	int32_t new_rank = shape_tensor->shape.depth;
	if ( new_rank < 1 || new_rank > 4) return errlog(nn, "bad shape input for reshape");

	int32_t new_shape[4] = { 1,1,1,1 };
	int negdim = -1;		// index of unknown dim 0..3; -1 if none
	unsigned new_prod = 1;	// product of 'known' dims

	for( int i = 4-new_rank; i < 4 ; i++){
		int d = tensor_get_int32( shape_tensor, i-(4-new_rank));
		if( d < 0 ){			// -> unspecified; we have to find it.
			if( negdim >= 0 ) return errlog(nn,"too many unknown dimensions");
			negdim = i;
		}else{
			new_shape[i] = d;
			new_prod = mulu32_sat(new_prod,d);		// find prod; protect against overflow.
		}
	}
	if( new_prod == 0 || new_prod == (unsigned)-1){	// shape has 0, or overflowed
		return  errlog(nn, "bad shape input for reshape");
	}
	if( negdim >= 0 ){			// we have an unknown dim
		unsigned unknown_dim = elements/new_prod;
		new_shape[negdim] = unknown_dim;
		new_prod *= unknown_dim;
		if (new_prod != elements){
			return errlog(nn,"reshape: can't determine unknown dim %d  = %u/%u", negdim, (unsigned)elements, new_prod);
		}
	}else{
		if (new_prod != elements){
			return errlog(nn,"reshape: size mismatch");
		}
	}

	logmsg(nn,2,"(q)reshape execute. self=%p ",self);
	if (out_tensor->max_size < in_tensor->data_size) {
		return errlog(nn,"out too small");
	}

	/* Copy input tensor to output */
	// TODO: we don't have a 'tensor_out_prepare' case for this
	// since the element size is unknown, and we can't rely on in_tensor->format.type
	// telling us what it is.

	tensor_set_shape(out_tensor,new_shape[0],new_shape[1],new_shape[2],new_shape[3]);

	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );

	out_tensor->format.raw0 = out_tensor->format.raw1 = 0;
	out_tensor->format.type = in_tensor->format.type;
	unsigned data_size = in_tensor->data_size;
	out_tensor->data_size = data_size;

	// do vector copy (possibly broken across >1 thread, if large)
	if( data_size >0  && out_tensor->data!= in_tensor->data)
		nn_mcmanager_vmemcpy( nn, &mcman, out_tensor->data, in_tensor->data, data_size );

	/* Handle quantized version */
	if (self->n_outputs == 3) {
		if (tensor_copy(self->outputs[1],self->inputs[2]) != 0
		 || tensor_copy(self->outputs[2],self->inputs[3]) != 0) {
			return errlog(nn,"bad extra copy");
		}
	}
	// wait for copy thread(s) if any
	nn_mcmanager_wait( nn, &mcman );

	logmsg(nn,2,"qreshape %dx%dx%dx%d (%dx%dx%dx%d) --> %dx%dx%dx%d",
		b,h,w,d,
		shape_tensor->shape.batches,
		shape_tensor->shape.height,
		shape_tensor->shape.width,
		shape_tensor->shape.depth,
		new_shape[0],new_shape[1],new_shape[2],new_shape[3]);
	return 0;
}


// This supports aliasing: so maybe we will get an execute call with in_tensor->data == out_tensor->data
// and we don't need to copy the data.
//
struct nn_node_ops nn_ops_for_Reshape = {
	.execute = reshape_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};

struct nn_node_ops nn_ops_for_QuantizedReshape = {
	.execute = reshape_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};

