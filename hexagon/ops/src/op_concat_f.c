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

//
// This op concatenates 2 or more tensors on a specified axis
// The axis is specified by another input tensor which is a single integer 0 .. 3
//   0 = batches, 1 = height, 2= width, 3= depth
// All inputs must have a common shape on all dimensions *other* than the specified dimension
// The output shape will be the sum of input dims on the specified direction, and the same on others
//
//   E.g. T1 shape  =  (2, 40, 30, 12 )
//        T2 shape  =  (2, 40, 30, 24 )
//        T3 shape   = (2, 40, 20, 12 )
//        T4 shape   = (2, 40, 30, 14 )

// T1,T2 can be concat on dimension 3, giving (2, 40, 30, 36) as output shape
// T1,T3 can be concat on dimension 2, giving (2,40, 50, 12)
// T1,T2,T4 can be concat on dimension 3, giving (2, 40, 30, 50 )
//

#include <nn_graph.h>
#include <string.h>


static int concat_do_execute(
	struct nn_node *self,
	struct nn_graph *nn,
	const struct tensor *dim_tensor,
	const struct tensor **input_tensors,
	int n_input_tensors,
	int elementsize)
{
	struct shape out_shape;		// shape of output tensor
	const struct tensor *t;
	struct tensor *out_tensor = self->outputs[0];
	int concat_dim, i,k;

	logmsg(nn,2,"concat execute. self=%p ",self);
	concat_dim = tensor_get_int32(dim_tensor,0);

	// check the dims of all inputs, find the output shape. This also
	// range checks 'concat_dim'.
	//
	k = find_concat_shape( input_tensors, n_input_tensors, concat_dim, &out_shape );
	if( k < 0){
		if( k <= -2) {
			// mismatch size on a particular dim
			return errlog(nn,"mismatch on tensor dim %d, concat on %d", (-2)-k , concat_dim);
		}
		return errlog( nn, "bad concat dim: %d", concat_dim);
	}

	// input is 'outer_count' units of idim * inner_size, contiguous
	// output is 'outer_count'units of odim * inner_size, contiguous
	// idim = size of input in selected dim (changes per input)
	// odim = size of output in selected dim (sum of all the idims)
	//  so the copy is
	//   for each input:
	//        outer_count times:
	//             copy 'inner_size*idim' bytes
	//             advance in pointer by inner_size*idim, out ptr by inner_size * odim ( = out_stride)
	//        advance 'base' out ptr by  inner_size*idim
	//

	// 'inner_size' is in bytes, so use byte pointers (memcpy is used for all copies).
	uint8_t const * in_data;
	uint8_t *  out_data = out_tensor->data;


	uint32_t inner_size=0;
	uint32_t out_stride=0;
	uint32_t outer_count;

	// set inner_size, out_stride, outer_count
	// outer_count = prod of all dims < concat_dim
	// inner_size = prod of all dims > concat_dim, and also elementsize
	// out_stride = inner_size * dim[concat_dim]

	outer_count = elementsize;	// this will become inner_size.

	for( i= 3; i >= 0; --i){	// depth, width, height, batches
		uint32_t ndim = out_shape.dimension[i];
		uint32_t newcnt = outer_count * ndim;
		if( i == concat_dim){
			inner_size = outer_count;	// set inner_size, out_stride
			out_stride = newcnt;
			newcnt = 1;					// and restart for out_count.
		}
		outer_count = newcnt;
	}

	// allocate
	int out_typ = ( self->node_type == OP_ConcatV2_int32)? NN_TYPE_INT32: NN_TYPE_FLOAT;
	if(tensor_out_prepare_normal_fromshape( out_tensor,  &out_shape, out_typ) != 0 ){
		return errlog(nn,"out too small");
	}

	// copy
	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman );

	for (i = 0; i < n_input_tensors; i++) {
		t = input_tensors[i];
		in_data = t->data;
		int input_dim = t->shape.dimension[concat_dim];
		uint32_t copylen = input_dim  * inner_size;

		nn_mcmanager_vmemcpy_2d(nn, &mcman,
				copylen, outer_count,	// width, height of rectangle
				out_data, out_stride, 	// output ptr, stride
				in_data, copylen );		// input ptr, stride

		//for( int j = 0; j < outer_count; j++){
		//	memcpy(out_data + out_stride * j, in_data + copylen * j, copylen);
		//}
		out_data += copylen;
	}
	nn_mcmanager_wait( nn, &mcman);

	logmsg(nn,2,"concat %p done",self);
	return 0;
}

static int concat_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1);
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	return concat_do_execute(self,nn,dim_tensor,input_tensors,n_input_tensors,sizeof(float));
}

static int concatv2_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1);
	const struct tensor *dim_tensor = self->inputs[n_input_tensors];
	const struct tensor **input_tensors = &self->inputs[0];
	return concat_do_execute(self,nn,dim_tensor,input_tensors,n_input_tensors,sizeof(float));
}

static int concatv2_int_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_input_tensors = (self->n_inputs-1);
	const struct tensor *dim_tensor = self->inputs[n_input_tensors];
	const struct tensor **input_tensors = &self->inputs[0];
	return concat_do_execute(self,nn,dim_tensor,input_tensors,n_input_tensors,sizeof(int32_t));
}


struct nn_node_ops nn_ops_for_Concat_f = {
	.execute = concat_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(1),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_ConcatV2_f = {
	.execute = concatv2_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(1),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_ConcatV2_int32 = {
	.execute = concatv2_int_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(1),
	.n_outputs = NN_IOCOUNT(1),
};

