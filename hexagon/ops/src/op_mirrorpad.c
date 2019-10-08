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

#define BATCH_PAD_BEFORE_IDX 0
#define BATCH_PAD_AFTER_IDX 1
#define HEIGHT_PAD_BEFORE_IDX 2
#define HEIGHT_PAD_AFTER_IDX 3
#define WIDTH_PAD_BEFORE_IDX 4
#define WIDTH_PAD_AFTER_IDX 5
#define DEPTH_PAD_BEFORE_IDX 6
#define DEPTH_PAD_AFTER_IDX 7

struct mirrorpad_info
{
	const char *in_ptr;
	char *out_ptr;
	struct shape in_shape;
	struct shape out_shape;
	int32_t *paddings;
	uint32_t elementsize;
	uint32_t mode;
	nn_sem_t done_sem;
};

static inline void do_mirrorpad(
	struct nn_graph *nn,
	struct nn_memcpy_manager *mcman,
	char *outp,
	const char *inp,
	const int32_t d_in, const int32_t w_in, const int32_t h_in, const int32_t b_in,
	const int32_t d_out, const int32_t w_out, const int32_t h_out, const int32_t b_out,
	const int32_t pre_d, const int32_t post_d,
	const int32_t pre_w, const int32_t post_w,
	const int32_t pre_h, const int32_t post_h,
	const int32_t pre_b, const int32_t post_b,
	const int32_t elem_size,
	const int32_t reflect)
{

	const uint32_t in_d_size = elem_size * d_in;
	const uint32_t in_w_size = in_d_size * w_in;

	const uint32_t out_d_size = elem_size * d_out;
	const uint32_t out_w_size = out_d_size * w_out;
	const uint32_t out_h_size = out_w_size * h_out;

	const uint32_t d_mirror_start_offset = reflect * elem_size;
	const uint32_t w_mirror_start_offset = reflect * out_d_size;
	const uint32_t h_mirror_start_offset = reflect * out_w_size;
	const uint32_t b_mirror_start_offset = reflect * out_h_size;

	char *pad_out;
	const char *pad_in;
	char *cur_out_batch;
	char *cur_out_row;
	char *cur_out_col;
	char *cur_in_col = (char *)inp;

	for (int b = 0; b < b_in; b++)
	{
		cur_out_batch = outp + b * out_h_size + pre_b * out_h_size;

		for (int h = 0; h < h_in; h++)
		{
			cur_out_row = cur_out_batch + h * out_w_size + pre_h * out_w_size;

			//A complete set of cols is processed in this inner loop, so we just need to point it at the appropriate col offset for each row
			cur_out_col = cur_out_row + pre_w * out_d_size;

			//If we have no pre d or post d padding, we're in luck and can do bigger copies
			if (pre_d || post_d)
			{

				for (int w = 0; w < w_in; w++)
				{
					if (pre_d > 0)
					{
						pad_in = cur_in_col + (pre_d - 1) * elem_size + d_mirror_start_offset;
						nn_mcmanager_vmemcpy_2d(nn, mcman,
									elem_size, pre_d,
									cur_out_col, elem_size,
									pad_in, -elem_size);
						
						cur_out_col += pre_d * elem_size;
					}

					nn_mcmanager_vmemcpy(nn, mcman, cur_out_col, cur_in_col, in_d_size);
					cur_in_col += in_d_size;
					cur_out_col += in_d_size;

					if (post_d > 0)
					{
						pad_in = cur_in_col - elem_size - d_mirror_start_offset;
						nn_mcmanager_vmemcpy_2d(nn, mcman,
									elem_size, post_d,
									cur_out_col, elem_size,
									pad_in, -elem_size);
						
						cur_out_col += post_d * elem_size;
					}
				}
			}

			//No depth padding, blast a 2d copy
			else
			{
				nn_mcmanager_vmemcpy_2d(nn, mcman,
							in_d_size, w_in,
							cur_out_col, in_d_size,
							cur_in_col, in_d_size);
				
				cur_in_col += in_w_size;
			}
			//Outer loop copies are reliant on the inner loop being finished before executing, so we need to wait until all inner copies are complete
			nn_mcmanager_wait( nn, mcman );

			if (pre_w > 0)
			{
				pad_in = cur_out_row + pre_w * out_d_size + w_mirror_start_offset + (pre_w - 1) * out_d_size;
				nn_mcmanager_vmemcpy_2d(nn, mcman,
							out_d_size, pre_w,
							cur_out_row, out_d_size,
							pad_in, -out_d_size);
			}

			if (post_w > 0)
			{
				pad_out = cur_out_row + out_d_size * (pre_w + w_in);
				pad_in = pad_out - out_d_size - w_mirror_start_offset;
				nn_mcmanager_vmemcpy_2d(nn, mcman,
							out_d_size, post_w,
							pad_out, out_d_size,
							pad_in, -out_d_size);
			}
			nn_mcmanager_wait( nn, mcman );
		}

		if (pre_h > 0)
		{
			pad_in = cur_out_batch + pre_h * out_w_size + h_mirror_start_offset + (pre_h - 1) * out_w_size;
			nn_mcmanager_vmemcpy_2d(nn, mcman,
						out_w_size, pre_h,
						cur_out_batch, out_w_size,
						pad_in, -out_w_size);
		}

		if (post_h > 0)
		{
			pad_out = cur_out_batch + out_w_size * (pre_h + h_in);
			pad_in = pad_out - out_w_size - h_mirror_start_offset;
			nn_mcmanager_vmemcpy_2d(nn, mcman,
						out_w_size, post_h,
						pad_out, out_w_size,
						pad_in, -out_w_size);
		}
		nn_mcmanager_wait( nn, mcman );
	}

	if (pre_b > 0)
	{
		pad_in = outp + pre_b * out_h_size + b_mirror_start_offset + (pre_b - 1) * out_h_size;
		nn_mcmanager_vmemcpy_2d(nn, mcman,
					out_h_size, pre_b,
					outp, out_h_size,
					pad_in, -out_h_size);
	}

	if (post_b > 0)
	{
		pad_out = outp + out_h_size * (pre_b + b_in);
		pad_in = pad_out - out_h_size - b_mirror_start_offset;
		nn_mcmanager_vmemcpy_2d(nn, mcman,
					out_h_size, post_b,
					pad_out, out_h_size,
					pad_in, -out_h_size);
	}
}

static void do_mirrorpad_wrapper(struct nn_graph *nn, void *info)
{

	struct mirrorpad_info *infop = (struct mirrorpad_info *)info;
	const int32_t d_in = infop->in_shape.depth;
	const int32_t w_in = infop->in_shape.width;
	const int32_t h_in = infop->in_shape.height;
	const int32_t b_in = infop->in_shape.batches;
	const int32_t d_out = infop->out_shape.depth;
	const int32_t w_out = infop->out_shape.width;
	const int32_t h_out = infop->out_shape.height;
	const int32_t b_out = infop->out_shape.batches;
	uint32_t pad_b_before = infop->paddings[BATCH_PAD_BEFORE_IDX];
	uint32_t pad_b_after = infop->paddings[BATCH_PAD_AFTER_IDX];
	uint32_t pad_h_before = infop->paddings[HEIGHT_PAD_BEFORE_IDX];
	uint32_t pad_h_after = infop->paddings[HEIGHT_PAD_AFTER_IDX];
	uint32_t pad_w_before = infop->paddings[WIDTH_PAD_BEFORE_IDX];
	uint32_t pad_w_after = infop->paddings[WIDTH_PAD_AFTER_IDX];
	uint32_t pad_d_before = infop->paddings[DEPTH_PAD_BEFORE_IDX];
	uint32_t pad_d_after = infop->paddings[DEPTH_PAD_AFTER_IDX];

	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman);

	do_mirrorpad(
		nn,
		&mcman,
		infop->out_ptr,
		infop->in_ptr,
		d_in, w_in, h_in, b_in,
		d_out, w_out, h_out, b_out,
		pad_d_before, pad_d_after,
		pad_w_before, pad_w_after,
		pad_h_before, pad_h_after,
		pad_b_before, pad_b_after,
		infop->elementsize,
		infop->mode);

	nn_mcmanager_wait(nn, &mcman);
	nn_sem_post(&infop->done_sem);
}
static int mirrorpad_execute(struct nn_node *self, struct nn_graph *nn, const uint32_t elementsize, int dtype)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *pads_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	if (pads_tensor->shape.depth != 2)
		return errlog(nn, "bad pad tensor");
	unsigned padt_len = pads_tensor->shape.width;
	if (padt_len > 4)
		padt_len = 4; // ignore > 4

	const int32_t *pads = pads_tensor->data;

	// exract the pads, based on w dimension; ensure all are >=0 and
	//
	int32_t padby[4 * 2] = {0, 0, 0, 0, 0, 0, 0, 0};
	for (int i = 0; i < (int)padt_len * 2; i++)
	{
		int p = pads[i];
		if (p < 0)
			return errlog(nn, "pad bad tensor");
		padby[i] = p;
	}
	// find the new shape; validate sanity
	struct shape out_shape;
	uint32_t new_shape_count = 1;
	for (int i = 0; i < 4; i++)
	{
		unsigned p_before = padby[2 * i];
		unsigned p_after = padby[2 * i + 1];
		unsigned old_dim = in_tensor->shape.dimension[i];
		uint64_t all_dim = (uint64_t)old_dim + (uint64_t)p_before + (uint64_t)p_after;
		if (all_dim > (uint64_t)0x7FFFFFFF)
			return errlog(nn, "padded size overflow");
		uint32_t new_dim = (uint32_t)all_dim;
		out_shape.dimension[i] = new_dim;
		new_shape_count = mulu32_sat(new_shape_count, new_dim);
	}
	if (new_shape_count == 0 || new_shape_count == (uint32_t)-1 || mulu32_sat(new_shape_count, elementsize) == (uint32_t)-1)
		return errlog(nn, "padded size overflow");

	struct mirrorpad_info *info = self->opaque;

	info->paddings = padby;
	info->elementsize = elementsize;
	info->mode = self->padding == NN_PAD_MIRROR_REFLECT;
	info->in_shape = in_tensor->shape;
	info->out_shape = out_shape;

	info->in_ptr = in_tensor->data;
	info->out_ptr = out_tensor->data;
	if (pads_tensor->shape.depth != 2)
		return errlog(nn, "bad pad shape");
	if (pads_tensor->shape.width != 4)
		return errlog(nn, "bad pad shape");

	if (padby[BATCH_PAD_BEFORE_IDX] >= info->in_shape.batches)
		return errlog(nn, "batches too small (%d>=%d)", padby[BATCH_PAD_BEFORE_IDX], info->in_shape.batches);
	if (padby[BATCH_PAD_AFTER_IDX] >= info->in_shape.batches)
		return errlog(nn, "batches too small (%d>=%d)", padby[BATCH_PAD_AFTER_IDX], info->in_shape.batches);

	if (padby[HEIGHT_PAD_BEFORE_IDX] >= info->in_shape.height)
		return errlog(nn, "height too small (%d>=%d)", padby[HEIGHT_PAD_BEFORE_IDX], info->in_shape.height);
	if (padby[HEIGHT_PAD_AFTER_IDX] >= info->in_shape.height)
		return errlog(nn, "height too small (%d>=%d)", padby[HEIGHT_PAD_AFTER_IDX], info->in_shape.height);

	if (padby[WIDTH_PAD_BEFORE_IDX] >= info->in_shape.width)
		return errlog(nn, "width too small (%d>=%d)", padby[WIDTH_PAD_AFTER_IDX], info->in_shape.width);
	if (padby[WIDTH_PAD_AFTER_IDX] >= info->in_shape.width)
		return errlog(nn, "width too small (%d>=%d)", padby[WIDTH_PAD_AFTER_IDX], info->in_shape.width);

	if (padby[DEPTH_PAD_BEFORE_IDX] >= info->in_shape.depth)
		return errlog(nn, "depth too small (%d>=%d)", padby[DEPTH_PAD_BEFORE_IDX], info->in_shape.depth);
	if (padby[DEPTH_PAD_AFTER_IDX] >= info->in_shape.depth)
		return errlog(nn, "depth too small (%d>=%d)", padby[DEPTH_PAD_AFTER_IDX], info->in_shape.depth);

	if (tensor_out_prepare_normal(out_tensor, out_shape.batches, out_shape.height, out_shape.width, out_shape.depth, dtype) != 0)
	{
		return errlog(nn, "out too small");
	}
	nn_sem_init(&info->done_sem, 0);
	nn_os_work_for_vector(nn, do_mirrorpad_wrapper, info);
	nn_sem_wait(&info->done_sem);
	return 0;
}

static int mirrorpad_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return mirrorpad_execute(self, nn, sizeof(float), NN_TYPE_FLOAT);
}

static int mirrorpad_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1], self->inputs[2]);
	tensor_copy(self->outputs[2], self->inputs[3]);
	return mirrorpad_execute(self, nn, sizeof(uint8_t), NN_TYPE_QUINT8);
}

static int mirrorpad_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn, 2, "mirrorpad node %p", self);
	if ((self->padding != NN_PAD_MIRROR_REFLECT) && (self->padding != NN_PAD_MIRROR_SYMMETRIC))
	{
		return errlog(nn, "bad mirror pad type");
	}
	void *info = nn_calloc(1, sizeof(struct mirrorpad_info));
	if (info == NULL)
	{
		return errlog(nn, "calloc failed");
	}
	self->opaque = info;
	logmsg(nn, 2, "mirrorpad %p check OK", self);
	return 0;
}
struct nn_node_ops nn_ops_for_MirrorPad_f = {
	.execute = mirrorpad_execute_f,
	.check = mirrorpad_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_MirrorPad_8 = {
	.execute = mirrorpad_execute_8,
	.check = mirrorpad_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};
