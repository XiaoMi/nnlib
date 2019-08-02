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
#include <quantize.h>
#include <math.h>

#include "nn_gentranspose.h"

// this holds a 'cached' strategy for doing the transpose
// and the parameters which make it valid.
// A zeroed version of this is attached to the node
// when it's checked.
// 
struct transpose_info {
	uint8_t elbytes;		// bytes per element (set in 'check')
	uint8_t eltype;			// type of output element (e.g. NN_TYPE_FLOAT, set in 'check')
	uint8_t isq;			// true if the type has min/max ports (set in 'check')
	uint8_t strategy_valid;
	struct shape inshape;
	struct shape outshape;
	int n_perm;			// # of dims in the 'tx' permutation
	int32_t perm[4];	// the permutation
	struct nn_transpose_desc  txdesc;	// the prepared 'transpose' op
};

static int transpose_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct transpose_info * info = (struct transpose_info*)self->opaque;

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *dims_tensor = self->inputs[1];

	//const struct tensor *true_rank_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];

	logmsg(nn,2,"transpose execute. self=%p ",self);
	if( info->isq){
		tensor_copy( self->outputs[1], self->inputs[2]);
		tensor_copy( self->outputs[2], self->inputs[3]);
	}

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
		res = nn_transpose_analyze( &info->txdesc, info->elbytes, info->perm, n_perm, &info->inshape );
		if( res ) return errlog( nn,"transpose error %d", res);
		if( info->txdesc.buffer_needed > nn->scratch_size){
			if( nn_scratch_grow(nn, info->txdesc.buffer_needed)!=0)
				return errlog(nn,"couldn't grow scratch for transpose\n");
		}
		int eltype = info->eltype;
		if( tensor_out_prepare_normal_fromshape( out_tensor, &info->outshape, eltype)!=0 ){
			return errlog(nn,"out too small");
		}
		// Transpose_16 propagates QINT16 as QINT16
		if(eltype == NN_TYPE_QUINT16 && in_tensor->format.type == NN_TYPE_QINT16){
			out_tensor->format.type = NN_TYPE_QINT16;
		}
		info->strategy_valid = 1;
	}

	res = nn_transpose_execute(nn, &info->txdesc, nn->scratch, out_tensor->data, in_tensor->data );
	if( res != 0) return errlog( nn, "transpose exec error %d", res);

	logmsg(nn,2,"transpose %p done",self);
	return 0;
}

static int transpose_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking transpose node %p",self);

	void * infop = nn_calloc( 1, sizeof(struct transpose_info) );
	if( infop == NULL ) return errlog(nn,"alloc failed");
	self->opaque = infop;

	struct transpose_info * info = (struct transpose_info*)infop;
	int elbytes, eltype, isq;
	switch( self->node_type){
		case OP_Transpose_int32:
			elbytes = sizeof(int32_t);
			eltype = NN_TYPE_INT32;
			isq = 0;
			break;
		case OP_Permute_f:
		case OP_Transpose_f:
			elbytes = sizeof(float);
			eltype = NN_TYPE_FLOAT;
			isq = 0;
			break;
		case OP_QuantizedPermute_8:
		case OP_Transpose_8:
			elbytes = sizeof(uint8_t);
			eltype = NN_TYPE_QUINT8;
			isq = 1;
			break;
		case OP_Transpose_16:
			elbytes = sizeof(uint16_t);
			eltype = NN_TYPE_QUINT16;
			isq = 1;
			break;
		default:
			return errlog(nn,"unexpected node type %d", (int)self->node_type);
	}
	info->elbytes = elbytes;
	info->eltype = eltype;
	info->isq = isq;

	logmsg(nn,2,"transpose %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Transpose_int32 = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_Transpose_f = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Transpose_8 = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_Transpose_16 = {
	.execute = transpose_execute,
	.check = transpose_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

// Permute is the same op ! (actually, a subset, since it requires the 'permute'
// control to always have a length of 4)

struct nn_node_ops nn_ops_for_Permute_f = {
    .execute = transpose_execute,
    .check = transpose_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
    .n_inputs = NN_IOCOUNT(2),
    .n_outputs = NN_IOCOUNT(1),
};


struct nn_node_ops nn_ops_for_QuantizedPermute_8 = {
    .execute = transpose_execute,
    .check = transpose_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(3),
};
