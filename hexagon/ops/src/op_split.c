
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

/*
 * Input 0: dimension to split on
 * Input 1: data
 * Evenly divide Data across all Outputs
 */

#include <nn_graph.h>
#include <string.h>
#include "nn_axis.h"

static int split_impl(
	struct nn_node *self,
	struct nn_graph *nn,
	int element_type,
	int element_size,
	int n_outs)
{
	const struct tensor *dimdef_tensor = self->inputs[0];
	const struct tensor *data_tensor = self->inputs[1];
	struct tensor **outs = self->outputs;
	// @@ seems that cnns_alexnet.c is built using Split with in[0] == in[1]
	int32_t dimdef = (dimdef_tensor == data_tensor)? 3:  tensor_get_int32(dimdef_tensor,0);
	const char *in_data = (const char *) data_tensor->data;
	const char *inptr;
	char *outptr;
	int i;
	int j;
	int bytestride;
	int copyn;
	
	int res = handle_negative_axes(nn, &dimdef, 1);
    if (res)
        return errlog(nn, "split dimension out of range");
	logmsg(nn,2,"split node %p execute, dim %d",self, (int)dimdef);

	if( ! ( 0 <= dimdef && dimdef <= 3)){
		return errlog(nn, "split dimension out of range");
	}
	if (n_outs == 1) {
		struct nn_memcpy_manager mcman;
		nn_mcmanager_init(nn, &mcman );
		int res = nn_mcmanager_tensor_copy(nn,&mcman,outs[0],data_tensor);
		nn_mcmanager_wait(nn, &mcman);
		if( res != 0)errlog(nn,"out too small");
		return res;
	}
	struct shape outshape = data_tensor->shape;
	int old_dim = outshape.dimension[dimdef];
	int new_dim =  old_dim / n_outs;
	if( new_dim * n_outs != old_dim){
		return errlog(nn, "uneven split: %d / %d", old_dim, n_outs);
	}
	outshape.dimension[dimdef] = new_dim;

	for (i = 0; i < n_outs; i++) {
		if( tensor_out_prepare_normal_fromshape( outs[i],&outshape, element_type )!=0)
			return errlog(nn,"out %d too small",i);
	}
	// each copy is 'copyn' copies of size 'inner_count * element_size'
	copyn = 1;
	int inner_count = 1;
	for( i = 0; i < 4; i++ ){
		if( i == dimdef){
			copyn = inner_count;
			inner_count = 1;
		}
		inner_count *= outshape.dimension[i];
	}
	bytestride = inner_count * element_size; // length per copy

	// use multi-copy manager to run the copies in vector threads.
	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );

	for (j = 0; j < n_outs; j++) {
		inptr = in_data + j*bytestride;
		outptr = (char *)outs[j]->data;

		if(copyn == 1){
			nn_mcmanager_vmemcpy( nn, &mcman, outptr, inptr, bytestride );
		}else{
			// TODDO: break large copies up into smaller parts.
			nn_mcmanager_vmemcpy_2d( nn, &mcman,
					bytestride, copyn,			// width, height
					outptr,bytestride,			// output & stride
					inptr,	bytestride *n_outs );	// input & stride
		}

	/* scalar version
		for (i = 0; i < copyn; i++) {
			memcpy(outptr,inptr,bytestride);
			outptr += bytestride;
			inptr += (bytestride * n_outs);
		}
	 */
	}

	nn_mcmanager_wait( nn, &mcman );

	return 0;
}

static int split_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	int eltype = (self->node_type == OP_Split_int32 )? NN_TYPE_INT32 : NN_TYPE_FLOAT;
	return split_impl(self,nn,eltype, 4, self->n_outputs);
}

static int qsplit_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	int n_outputs = self->n_outputs;
	tensor_copy(self->outputs[n_outputs-2],self->inputs[2]);
	tensor_copy(self->outputs[n_outputs-1],self->inputs[3]);
	return split_impl(self,nn,NN_TYPE_QUINT8, 1, n_outputs-2);
}



struct nn_node_ops nn_ops_for_Split_f = {
	.execute = split_execute_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT_GE(1),
};

struct nn_node_ops nn_ops_for_Split_int32 = {
	.execute = split_execute_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT_GE(1),
};

struct nn_node_ops nn_ops_for_QuantizedSplit_8 = {
	.execute = qsplit_execute_8,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT_GE(3),
};

