/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
 * This is reverse op of pack.
 * 
 */

#include <nn_graph.h>
#include <string.h>
#include "nn_axis.h"

static int unpack_do_execute(struct nn_node *self, struct nn_graph *nn, int elementsize, int element_type) {
	const struct tensor *input = self->inputs[0];
	const struct tensor *axis_tensor = self->inputs[1];
	const struct tensor *num_tensor = NULL;
	const struct tensor *min_tensor = NULL;
	const struct tensor *max_tensor = NULL;
	int32_t axis_idx0 = tensor_get_int32(axis_tensor, 0);
	struct tensor **outs = self->outputs;

	if (handle_negative_axes(nn, &axis_idx0, 1))
		return errlog(nn, "Unpack: axis is out of range \n");
	int axis = axis_idx0;
	// num of outputs should equal to input dimension at index 'axis'.
	int n_outputs = input->shape.dimension[axis];
	float min_overall, max_overall;
	if (element_type == NN_TYPE_UINT8) {
		if (self->n_inputs == 5) {
			num_tensor = self->inputs[2];
			n_outputs = tensor_get_int32(num_tensor, 0);
			min_tensor = self->inputs[3];
			max_tensor = self->inputs[4];
		}
		else {
			min_tensor = self->inputs[2];
			max_tensor = self->inputs[3];
		}
		min_overall = tensor_get_float(min_tensor, 0);
		max_overall = tensor_get_float(max_tensor, 0);
	}
	else {
		if (self->n_inputs == 3) {
			num_tensor = self->inputs[2];
			n_outputs = tensor_get_int32(num_tensor, 0);
		}
	}
	int outer_size = 1;
	for (int i = 0; i < axis; ++i) {
		outer_size *= input->shape.dimension[i];
	}

	int copy_size = elementsize;
	for (int i = axis + 1; i < 4; ++i) {
		copy_size *= input->shape.dimension[i];
	}

	// left shift the dimension, take first 3 to construct h,w,c of new shape. b of new shape equals to 1.
	struct shape outshape = input->shape;
	if (axis < 3) {
		for (int i = axis; i < 3; i++)
		{
			outshape.dimension[i] = outshape.dimension[i+1];
		}
	}

	for (int i = 3 ; i > 0; i--)
	{
		outshape.dimension[i] = outshape.dimension[i - 1];
	}
	outshape.dimension[0] = 1;

	if (element_type == NN_TYPE_UINT8) {
		for (int i = 0; i < n_outputs; i++) {
			if( tensor_out_prepare_normal_fromshape( outs[i*3], &outshape, element_type )!=0)
				return errlog(nn,"out %d too small",i);
			tensor_set_single_float(self->outputs[3*i+1],min_overall);
			tensor_set_single_float(self->outputs[3*i+2],max_overall);
		}
	}
	else {
		for (int i = 0; i < n_outputs; i++) {
			if( tensor_out_prepare_normal_fromshape( outs[i], &outshape, element_type )!=0)
				return errlog(nn,"out %d too small",i);
		}
	}

	int src_stride = (axis > 0) ? (input->shape.dimension[axis] * copy_size) : copy_size;

	uint8_t const * in_data = input->data;
	struct tensor *out_tensor;
	struct nn_memcpy_manager mcman;
	nn_mcmanager_init( nn, &mcman );
	for (int k = 0; k < n_outputs; ++k) {
		if (element_type == NN_TYPE_UINT8) {
			out_tensor = outs[3*k];
		}
		else {
			out_tensor = outs[k];
		}
		uint8_t *out_data = out_tensor->data;
		nn_mcmanager_vmemcpy_2d(nn, &mcman,
				copy_size, outer_size,	// width, height of rectangle
				out_data, copy_size, 	// output ptr, stride
				in_data, src_stride );		// input ptr, stride
      in_data += copy_size;
	}
	nn_mcmanager_wait( nn, &mcman);

	logmsg(nn, 2, "unpack %p done", self);
	return 0;
}

static int unpack_int_execute(struct nn_node *self, struct nn_graph *nn)
{
	return unpack_do_execute(self, nn, sizeof(int32_t), NN_TYPE_INT32);
}

static int unpack_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	return unpack_do_execute(self, nn, sizeof(float), NN_TYPE_FLOAT);
}

static int unpack_quint8_execute(struct nn_node *self, struct nn_graph *nn)
{
	return unpack_do_execute(self, nn, sizeof(uint8_t), NN_TYPE_UINT8);
}

struct nn_node_ops nn_ops_for_Unpack_f = {
	.execute = unpack_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(2,3),
	.n_outputs = NN_IOCOUNT_GE(1),
};

struct nn_node_ops nn_ops_for_Unpack_int32 = {
	.execute = unpack_int_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(2,3),
	.n_outputs = NN_IOCOUNT_GE(1),
};

struct nn_node_ops nn_ops_for_QuantizedUnpack_8 = {
	.execute = unpack_quint8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT_GE(3),
};
