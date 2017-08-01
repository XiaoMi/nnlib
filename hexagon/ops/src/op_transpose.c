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
#include <string.h>
#include <quantize.h>
#include <math.h>

static int transpose_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *dims_tensor = self->inputs[1];
	//const struct tensor *true_rank_tensor = self->inputs[2];
	int32_t b = in_tensor->shape.batches;
	int32_t h = in_tensor->shape.height;
	int32_t w = in_tensor->shape.width;
	int32_t d = in_tensor->shape.depth;
	struct tensor *out_tensor = self->outputs[0];
	int32_t length;
	int32_t length_delta;
	int i;
	int bi,hi,wi,di;
	int bs,hs,ws,ds;
	int ob,oh,ow,od;
	int32_t idxtab[4] = { 0, 1, 2, 3 };
	int32_t strides[4] = { d*w*h, d*w, d, 1 };
	int32_t dims[4] = { b, h, w, d };
	int idx;
	int nd_idx;
	int32_t val;
	int32_t *out_data = (int32_t *)out_tensor->data;

	logmsg(nn,2,"transpose execute. self=%p ",self);
	logmsg(nn,3,"transpose input = %dx%dx%dx%d",
		(int)b,
		(int)h,
		(int)w,
		(int)d);

	if ((dims_tensor->shape.batches != 1) 
		|| (dims_tensor->shape.height != 1)
		|| (dims_tensor->shape.width != 1)) return errlog(nn,"dims !1d");
	if (in_tensor->data_size > out_tensor->max_size) return errlog(nn,"out too small");
	length = dims_tensor->data_size / sizeof(int32_t);
	if (unlikely(length > 4)) return errlog(nn,"oops, too dimensional");

	/* Transpose 1D data? */
	if (unlikely(length == 1)) {
		logmsg(nn,3,"dims length 1? I think that means do nothing.");
		out_tensor->shape = in_tensor->shape;
		out_tensor->data_size = in_tensor->data_size;
		memcpy(out_tensor->data,in_tensor->data,in_tensor->data_size);
		return 0;
	}

	/*
	 * Transpose strategy:
	 * * The ordering array tells me what dimensions go where.
	 * * But the N-D Rank might != 4
	 * * However the ordering array should be 0..(N-1).
	 * * In N-D, rank 0 is most significant.
	 * * We can turn N-D rank into 4D rank by adding zeros and incrementing N-D rank values
	 * * The dims_tensor tells me which input dimensions go in which order
	 * * N-1,...,0 means least significant become most significant
	 * * 0,...,N-1 means NOP
	 * * 0,3,2,1 would mean "leave batches alone but flip within batches"
	 * * We can compute the strides for each input dimension
	 * * Then if we have iterators i,j,k,l we can have an associated stride value
	 * * And as the output traverses linearly we can apply looked-up strides to read values.
	 */
	length_delta = 4-length;
	for (i = 0; i < length; i++) {
		nd_idx = tensor_get_int32(dims_tensor,i);
		idxtab[i+length_delta] = nd_idx + length_delta;
	}
	out_tensor->data_size = in_tensor->data_size;
	/* Compute strides */
	bs = strides[idxtab[0]];
	hs = strides[idxtab[1]];
	ws = strides[idxtab[2]];
	ds = strides[idxtab[3]];
	/* Find output dimensions */
	ob = dims[idxtab[0]];
	oh = dims[idxtab[1]];
	ow = dims[idxtab[2]];
	od = dims[idxtab[3]];
	tensor_set_shape(out_tensor,
		dims[idxtab[0]],
		dims[idxtab[1]],
		dims[idxtab[2]],
		dims[idxtab[3]]);
	logmsg(nn,3,"stride idx = %d,%d,%d,%d strides=%d,%d,%d,%d dims=%d,%d,%d,%d",
		(int)idxtab[0],
		(int)idxtab[1],
		(int)idxtab[2],
		(int)idxtab[3],
		(int)bs,
		(int)hs,
		(int)ws,
		(int)ds,
		(int)ob,
		(int)oh,
		(int)ow,
		(int)od);
	for (bi = 0; bi < ob; bi++) {
		for (hi = 0; hi < oh; hi++) {
			for (wi = 0; wi < ow; wi++) {
				for (di = 0; di < od; di++) {
					idx = bi*bs + hi*hs + wi*ws + di*ds;
					val = tensor_get_int32(in_tensor,idx);
					*out_data++ = val;
				}
			}
		}
	}

	logmsg(nn,2,"transpose %p done",self);
	return 0;
}

static int transpose_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking transpose node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"range transpose %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Transpose_int32 = {
	SFINIT(.execute, transpose_execute),
	SFINIT(  .check, transpose_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Transpose_f = {
	SFINIT(.execute, transpose_execute),
	SFINIT(.check, transpose_check),
	SFINIT(.ctor, node_alloc_common),
	SFINIT(.dtor, node_free_common),
};

