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
#include <string.h>
#include <quantize.h>
#include <math.h>

#include "nn_gentranspose.h"

// this holds a 'cached' strategy for doing the transpose
// and the parameters which make it valid.
// A zeroed version of this is attached to the node
// when it's checked.
// 
struct transpose_info {
	uint8_t strategy_valid;
	struct shape inshape;
	struct shape outshape;
	int n_perm;			// # of dims in the 'tx' permutation
	int32_t perm[4];	// the permutation
	struct nn_transpose_desc  txdesc;
};

static int transpose_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct transpose_info * info = (struct transpose_info*)self->opaque;

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *dims_tensor = self->inputs[1];
		
	//const struct tensor *true_rank_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];

	int elbytes = 4;
	int eltype = NN_TYPE_INT32;
	int optype = self->node_type;
	if( optype == OP_Transpose_f || optype == OP_Permute_f){
		eltype = NN_TYPE_FLOAT;
	}else if( optype == OP_Transpose_8 || optype == OP_QuantizedPermute_8){
		eltype = NN_TYPE_QUINT8;
		elbytes = 1;
		tensor_copy( self->outputs[1], self->inputs[2]);
		tensor_copy( self->outputs[2], self->inputs[3]);
	}

	logmsg(nn,2,"transpose execute. self=%p ",self);
	logmsg(nn,3,"transpose input = %dx%dx%dx%d",
		(int)in_tensor->shape.batches, (int)in_tensor->shape.height,
		(int)in_tensor->shape.width,	(int)in_tensor->shape.depth);

	if ((dims_tensor->shape.batches != 1)
		|| (dims_tensor->shape.height != 1)
		|| (dims_tensor->shape.width != 1)) return errlog(nn,"dims !1d");
	int n_perm = dims_tensor->data_size / sizeof(int32_t);
	if (unlikely(n_perm > 4)) return errlog(nn,"bad transpose control");

	int32_t const *perm_p = (int32_t const *)dims_tensor->data;

	// is it valid & current?
	int strategy_valid = info->strategy_valid;
	if( strategy_valid ){
		strategy_valid = 0;		// until shapes proven to match
		if( shape_matches( &in_tensor->shape, &info->inshape)
			&& n_perm == info->n_perm ){
			strategy_valid = 1;		// until mismatch found
			for( int i= 0; i < n_perm; i++ ){
				if( perm_p[i] != info->perm[i]){
					strategy_valid = info->strategy_valid = 0;
					break;
				}
			}
		}
	}
	int res;
	// if not valid, build it
	
	if( !strategy_valid ){
		info->inshape = in_tensor->shape;
		info->n_perm = n_perm;
		for( int i = 0 ; i < n_perm; i++)
			info->perm[i] = perm_p[i];
		res = nn_transpose_check( perm_p, n_perm, &info->inshape, &info->outshape );
		if( res ) return errlog(nn,"bad transpose control");
		res = nn_transpose_analyze( &info->txdesc, elbytes, info->perm, n_perm, &info->inshape );
		if( res ) return errlog( nn,"transpose error %d", res);
		info->strategy_valid = 1;
	}
	if( tensor_out_prepare_normal_fromshape( out_tensor, &info->outshape, eltype)!=0 ){
		return errlog(nn,"out too small");
	}
	// @@ TODO: check if the transpose needs scratch memory.
	
	res = nn_transpose_execute(nn, &info->txdesc, NULL, out_tensor->data, in_tensor->data );
	if( res != 0) return errlog( nn, "transpose exec error %d", res);

	logmsg(nn,2,"transpose %p done",self);
	return 0;
}

static int transpose_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking transpose node %p",self);
	int k =(self->node_type == OP_Transpose_8 || self->node_type == OP_QuantizedPermute_8)? 2 : 0;

	if (self->n_inputs != k+2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != k+1) return errlog(nn,"wrong # outputs");
	
	void * infop = nn_calloc( 1, sizeof(struct transpose_info) );
	if( infop == NULL ) return errlog(nn,"alloc failed");
	self->opaque = infop;
	logmsg(nn,2,"range transpose %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Transpose_int32 = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
};
struct nn_node_ops nn_ops_for_Transpose_f = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
};

struct nn_node_ops nn_ops_for_Transpose_8 = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
};

// Permute is the same op ! (actually, a subset, since it requires the 'permute'
// control to always have a length of 4)

struct nn_node_ops nn_ops_for_Permute_f = {
    .execute = transpose_execute,
    .check = transpose_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
};


struct nn_node_ops nn_ops_for_QuantizedPermute_8 = {
    .execute = transpose_execute,
    .check = transpose_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
};
