
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
 * Given 'n' input tensors of identical shape, pack them all together on a
 * new dimension.
 * The original rank is determined by finding the rightmost non one axis
 * Implement the pack by expand dim + concat
 * 
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>

#ifdef HEXAGON_V66
#define PACK_MAX_THREADS 4
#else
#define PACK_MAX_THREADS 2
#endif

struct pack_state {
	struct nn_node *self;
	int32_t n_inputs;		// actual number of inputs
	float out_min, out_max;	// output range
	float out_level_recip;
	volatile int32_t input_next;		// input to process next.

	uint32_t inner_size;
	uint32_t out_stride;
	uint32_t outer_count;

	int32_t pack_dim;
	const struct tensor **input_tensors;
	const struct tensor **min_tensors;
	const struct tensor **max_tensors;
	const struct shape *expand_inputdim;
	struct tensor *out_tensor;

	nn_sem_t done_sem;
};

// This function will first expand dim on pack dim and then recalculate pack
// dim as concat_dim. Looks at the shapes in an array of 1 or more 'struct_tensor'
// and, assuming they will be concatenated on dimension 'concat_dim', finds
// the overall shape.
// It also range-checks 'concat_dim' (must be 0..3) and
// ensures that all shapes match in all dims *other* than concat_dim.
//
// returns:
//   0:  ok
//   -1: concat_dim out of range
//   -2..-5: mismatch on dimension 0,1,2,3
static int find_pack_shape(
	const struct tensor **input_tensors,
	int n_input,	// >= 1
	int32_t *pack_dim,
	struct shape *allshape,
	struct shape *expand_input_shape)
{
	// the first input tensor provides 'ref' dims
	int32_t concat_dim = *pack_dim;
	// look for apparent rank for expand dim
	// pack will create new dim, max hexagon tensor apparent rank is 3
	int input_rank = shape_apparent_rank(&(input_tensors[0]->shape));
	// Does not support rank < 3 input since could not tell actual rank in hexagon tensor
	// if original tensor has batch axis and it equal to 1
	input_rank = max_i32(3, input_rank);
	if (concat_dim < 0) {
	   // add 1 because output dimension size is input rank + 1
		concat_dim = concat_dim + input_rank + 1;
	}
	else {
		concat_dim = concat_dim + 3 - input_rank;
	}
	if (!(concat_dim >= 0 && concat_dim <= 3))
		return -1;

	int i, j;
	// Expand dim and save new dim in shape buffer
	for (i = 0; i < n_input; i++) {
		for (j = 0; j < 4; j++) {
			if (j < concat_dim) {
				expand_input_shape[i].dimension[j] = input_tensors[i]->shape.dimension[j+1];
			}
			else if (j == concat_dim) {
				expand_input_shape[i].dimension[j] = 1;
			}
			else {
				expand_input_shape[i].dimension[j] = input_tensors[i]->shape.dimension[j];
			}
		}
	}

	uint32_t ref_batches = expand_input_shape[0].batches;
	uint32_t ref_height = expand_input_shape[0].height;
	uint32_t ref_width = expand_input_shape[0].width;
	uint32_t ref_depth = expand_input_shape[0].depth;

	uint32_t anydel_batches = 0, anydel_height = 0;
	uint32_t anydel_width = 0, anydel_depth = 0;
	uint32_t sum_del = 0;


	// for all the others:
	//  find  del_XX = XX - ref_XX (mod uint32)
	//    - 'or' them all so that we can tell if any are different from ref
	//    - sum them all (across all dims) so we can figure the size of the
	//     concat dimension later.

	for (i = 1; i < n_input; i++)
	{
		uint32_t del_batches = expand_input_shape[i].batches - ref_batches;
		anydel_batches |= del_batches;
		sum_del += del_batches;
		uint32_t del_height = expand_input_shape[i].height - ref_height;
		anydel_height |= del_height;
		sum_del += del_height;
		uint32_t del_width = expand_input_shape[i].width - ref_width;
		anydel_width |= del_width;
		sum_del += del_width;
		uint32_t del_depth = expand_input_shape[i].depth - ref_depth;
		anydel_depth |= del_depth;
		sum_del += del_depth;
	}
	// now:
	//  -all anydel_XX (except in the current dim) must be zero.
	//  -sum_del is the sum of the deltas XX - ref_XX in all dimensions on all inputs.
	//     This contains no contributions from the non-selected
	//     dimensions. So if we add n*ref_XX to this, the result is the sum
	//     of the selected dim across all inputs.

	if (anydel_batches != 0 && concat_dim != 0)
		return -2; // mismatch on batches
	if (anydel_height != 0 && concat_dim != 1)
		return -3; // mismatch on height
	if (anydel_width != 0 && concat_dim != 2)
		return -4; // mismatch on width
	if (anydel_depth != 0 && concat_dim != 3)
		return -5; // mismatch on depth

	// fill out the result...
	// one of these needs to be corrected later

	allshape->batches = ref_batches;
	allshape->height = ref_height;
	allshape->width = ref_width;
	allshape->depth = ref_depth;

	switch (concat_dim)
	{
	case 0:
		allshape->batches = n_input * ref_batches + sum_del;
		break;
	case 1:
		allshape->height = n_input * ref_height + sum_del;
		break;
	case 2:
		allshape->width = n_input * ref_width + sum_del;
		break;
	case 3:
		allshape->depth = n_input * ref_depth + sum_del;
		break;
	}
	*pack_dim = concat_dim;
	return 0;
}

static int32_t set_output_range(
	struct nn_node *self,
	int32_t pack_dim,
	float *pout_min,
	float *pout_max,
	float *pout_level_recip
)
{
	int32_t  n_input_tensors = (self->n_inputs - 1) / 3;
	const struct tensor **min_tensors = &self->inputs[1 + n_input_tensors];
	const struct tensor **max_tensors = &self->inputs[1 + 2 * n_input_tensors];
	float out_min = 0.0f;
	float out_max = 0.0f;

	for (int32_t i = 0; i < n_input_tensors; i++) {
		out_min = fminf(out_min, tensor_get_float(min_tensors[i], 0));
		out_max = fmaxf(out_max, tensor_get_float(max_tensors[i], 0));
	}

	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	tensor_set_single_float(out_min_tensor, out_min);
	tensor_set_single_float(out_max_tensor, out_max);

	*pout_min = out_min;
	*pout_max = out_max;
	*pout_level_recip = 255.0f / (out_max - out_min);
	return 0;
}

static void find_pack_param(
	int32_t pack_dim,
	int32_t *dims,
	uint32_t *pinner_size,
	uint32_t *pout_stride,
	uint32_t *pouter_count
)
{
	// assume concat_dim = 4
	// dim 0    1    2    3    4    5    6    7
	//     <--outer_count->                      :prod of all dims < concat_dim
	//                         <--out_stride---> :inner_size * dim[concat_dim]
	//                              <----------> inner_size: prod of all dims > concat_dim
	uint32_t inner_size = 0;
	uint32_t out_stride = 0;
	uint32_t outer_count = 1;	// will become inner_size.
	for (int32_t i = 3; i >= 0; --i) {	// depth, width, height, batches
		uint32_t ndim = dims[i];
		uint32_t newcnt = outer_count * ndim;
		if (i == pack_dim) {
			inner_size = outer_count;	// set inner_size, out_stride
			out_stride = newcnt;
			newcnt = 1;					// and restart for out_count.
		}
		outer_count = newcnt;
	}
	*pinner_size = inner_size;
	*pout_stride = out_stride;
	*pouter_count = outer_count;
}

static void pack_work(
	struct nn_graph *nn,
	void *thrinfo)
{
	struct pack_state *thrdesc = (struct pack_state *)thrinfo;

	int32_t pack_dim = thrdesc->pack_dim;
	const struct tensor **input_tensors = thrdesc->input_tensors;
	const struct tensor **min_tensors = thrdesc->min_tensors;
	const struct tensor **max_tensors = thrdesc->max_tensors;
	struct tensor *out_tensor = thrdesc->out_tensor;

	int32_t inner_size = thrdesc->inner_size;
	int32_t out_stride = thrdesc->out_stride;
	int32_t outer_count = thrdesc->outer_count;

	float out_min = thrdesc->out_min;
	float out_level_recip = thrdesc->out_level_recip;
	uint8_t* out_data = out_tensor->data;

	int32_t jobid, prev_thread = 0;
	const struct shape *expanddim = thrdesc->expand_inputdim;
	while (jobid = __sync_fetch_and_add(&thrdesc->input_next, 1), jobid < thrdesc->n_inputs) {
		const struct tensor *t = input_tensors[jobid];
		while (prev_thread < jobid) {
			out_data += expanddim[prev_thread++].dimension[pack_dim] * inner_size;
		}
		uint8_t *in_data = t->data;
		int32_t input_dim = expanddim[jobid].dimension[pack_dim];
		uint32_t copylen = input_dim * inner_size;

		l2fetch(in_data, copylen * sizeof(uint16_t), copylen * sizeof(uint16_t), outer_count);

		float in_min = tensor_get_float(min_tensors[jobid], 0);
		float in_max = tensor_get_float(max_tensors[jobid], 0);
		in_min = fminf(0.0f, in_min); // in_min <= 0.0f
		float in_level = flt_div_255(in_max-in_min);

		int32_t offset = max_i32(0, roundf_i32((in_min - out_min)*out_level_recip));
		int32_t gaint = roundf_i32(out_level_recip*in_level* 32768.0f);
		int32_t gain = min_i32(32767, gaint);

		if( offset != 0 || gain < 0x7fc0) {    // scale the input into common range
			memconvert_hvx(
				out_data,
				in_data,
				copylen,
				offset,
				gain,
				out_stride,
				outer_count);
		}
		else {                                 // is unity gain (0->0, 255->255)
			vmemcpy_2d_general_asm(
				copylen,                       // bytes wide
				outer_count,                   // rows
				out_data,                      // destination address, any allowed
				out_stride,                    // row pitch of dest; any allowed
				in_data,                       // source address, any allowed
				copylen);                      // source stride, any
		}
	}

	// signal complete in thread.
	nn_sem_post(&thrdesc->done_sem);
}

static int pack_do_execute(struct nn_node *self, struct nn_graph *nn, int elementsize)
{
	int32_t n_input_tensors = (self->n_inputs - 1);
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	struct shape out_shape;		// shape of output tensor
	struct shape expand_input_shape[n_input_tensors];
	const struct tensor *t;
	int32_t pack_dim, i, k;

	logmsg(nn, 2, "pack execute. self=%p ", self);
	pack_dim = tensor_get_int32(dim_tensor, 0);

	// check the dims of all inputs, find the output shape. This also
	// range checks 'pack_dim'. convert pack_dim to "concat_dim" actually
	// expand dim and replace input tensor shape with new dim in later concat operation
	//
	k = find_pack_shape(input_tensors, n_input_tensors, &pack_dim, &out_shape, &expand_input_shape[0]);
	if (k < 0) {
		if (k <= -2) {
			// mismatch size on a particular dim
			return errlog(nn, "mismatch on expanded tensor dim %d, concat on %d", (-2) - k, pack_dim);
		}
		return errlog(nn, "bad pack dim: %d", pack_dim);
	}
	int out_type = ( self->node_type == OP_Pack_int32 ) ? NN_TYPE_INT32 : NN_TYPE_FLOAT;
	if (tensor_out_prepare_normal_fromshape(out_tensor, &out_shape, out_type) != 0) {
		return errlog(nn,"failed to prepare output");
	} // 0..3

	uint8_t const * in_data;
	uint8_t *  out_data = out_tensor->data;

	// set inner_size, out_stride, outer_count
	// outer_count = prod of all dims < concat_dim
	// inner_size = prod of all dims > concat_dim, and also elementsize
	// out_stride = inner_size * dim[concat_dim]
	uint32_t inner_size;
	uint32_t out_stride;
	uint32_t outer_count;
	find_pack_param(pack_dim, (int32_t*)out_shape.dimension, &inner_size, &out_stride, &outer_count);
	inner_size  *= elementsize;
	out_stride  *= elementsize;
	outer_count *= elementsize;

	// copy
	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman );

	for (i = 0; i < n_input_tensors; i++) {
		t = input_tensors[i];
		in_data = t->data;
		int input_dim = expand_input_shape[i].dimension[pack_dim];
		uint32_t copylen = input_dim  * inner_size;

		nn_mcmanager_vmemcpy_2d(nn, &mcman,
				copylen, outer_count,	// width, height of rectangle
				out_data, out_stride, 	// output ptr, stride
				in_data, copylen );		// input ptr, stride

		out_data += copylen;
	}
	nn_mcmanager_wait( nn, &mcman);

	logmsg(nn, 2, "pack %p done", self);
	return 0;
}

// Expand dim and then do concat on new dim to implement pack
static int pack_quint8_execute(struct nn_node *self, struct nn_graph *nn)
{
	int32_t n_input_tensors = (self->n_inputs - 1) / 3;
	const struct tensor *dim_tensor = self->inputs[0];
	const struct tensor **input_tensors = &self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	struct shape out_shape;		// shape of output tensor
	struct shape expand_input_shape[n_input_tensors];
	int32_t pack_dim, i, k;

	logmsg(nn, 2, "pack quint8 execute. self=%p ", self);
	pack_dim = tensor_get_int32(dim_tensor, 0);

	// check the dims of all inputs, find the output shape. This also
	// range checks 'pack_dim'. convert pack_dim to "concat_dim" actually
	// expand dim and replace input tensor shape with new dim in later concat operation
	//
	k = find_pack_shape(input_tensors, n_input_tensors, &pack_dim, &out_shape, &expand_input_shape[0]);
	if (k < 0) {
		if (k <= -2) {
			// mismatch size on a particular dim
			return errlog(nn, "mismatch on expanded tensor dim %d, concat on %d", (-2) - k, pack_dim);
		}
		return errlog(nn, "bad pack dim: %d", pack_dim);
	}
	if (tensor_out_prepare_normal_fromshape(out_tensor, &out_shape, NN_TYPE_QUINT8) != 0) {
		return errlog(NULL, "out too small");
	}

	// find max/min among input vectors
	float out_min = 0.0f, out_max = 0.0f, out_level_recip = 0.0f;
	if (set_output_range(self, pack_dim, &out_min, &out_max, &out_level_recip)) {
		return errlog(nn, "set_output_range error!");
	}

	// find concat traverse param
	struct pack_state rundesc;
	find_pack_param(pack_dim, (int32_t*)out_shape.dimension, &rundesc.inner_size, &rundesc.out_stride, &rundesc.outer_count);

	// fire the threads
	rundesc.self = self;
	rundesc.n_inputs = n_input_tensors;
	rundesc.out_min = out_min;
	rundesc.out_max = out_max;
	rundesc.out_level_recip = out_level_recip;
	rundesc.input_next = 0;
	rundesc.expand_inputdim = expand_input_shape;
	nn_sem_init(&rundesc.done_sem, 0);

	rundesc.pack_dim = pack_dim;
	rundesc.input_tensors = input_tensors;
	rundesc.min_tensors = &self->inputs[1 + n_input_tensors];
	rundesc.max_tensors = &self->inputs[1 + 2 * n_input_tensors];
	rundesc.out_tensor = out_tensor;

	int32_t num_actual_threads = min_i32(PACK_MAX_THREADS, n_input_tensors);
	for (i = 0; i < num_actual_threads; i++) {
		nn_os_work_for_vector(nn, pack_work, &rundesc);
	}
	nn_sem_wait_n_times(&rundesc.done_sem, num_actual_threads);

	logmsg(nn,2,"pack quint8 %p done",self);
	return 0;
}

static int pack_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking pack node %p",self);

	// must be 3*n+1 inputs, where n >= 1
	int32_t n_in = (self->n_inputs - 1) /3;	// actual # of inputs
	if (n_in < 1 || (self->n_inputs - 1) % 3 !=0 )
		return errlog(nn,"concat: inputs must be 3*n+1, n>=1");

	logmsg(nn,2,"pack node %p check OK",self);
	return 0;
}

static int pack_f_execute(struct nn_node *self, struct nn_graph *nn)
{
   return pack_do_execute(self, nn, sizeof(float));
}

static int pack_int_execute(struct nn_node *self, struct nn_graph *nn)
{
   return pack_do_execute(self, nn, sizeof(int32_t));
}

struct nn_node_ops nn_ops_for_Pack_f = {
	.execute = pack_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Pack_int32 = {
	.execute = pack_int_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedPack_8 = {
	.execute = pack_quint8_execute,
	.check = pack_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
};
