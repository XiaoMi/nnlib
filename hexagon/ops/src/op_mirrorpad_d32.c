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
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	int32_t *paddings;
	uint32_t elementsize;
	uint32_t mode;
	nn_sem_t done_sem;
};

static inline void do_mirrorpad(
	struct nn_graph *nn,
	struct nn_memcpy_manager *mcman,
	struct mirrorpad_info *info)
{
	const uint8_t *inp = info->tin.data;
	uint8_t *outp = info->tout.data;
	const uint32_t b_in = info->in_shape.batches;
	const uint32_t h_in = info->in_shape.height;
	const uint32_t w_in = info->in_shape.width;
	const uint32_t pre_h = info->paddings[HEIGHT_PAD_BEFORE_IDX];
	const uint32_t post_h = info->paddings[HEIGHT_PAD_AFTER_IDX];
	const uint32_t pre_w = info->paddings[WIDTH_PAD_BEFORE_IDX];
	const uint32_t post_w = info->paddings[WIDTH_PAD_AFTER_IDX];
	const uint32_t reflect = info->mode;

	const uint32_t w_out = info->out_shape.width;
	const uint32_t w_stride = 32;

	//Pointers used
	uint8_t *outp_core = outp + pre_h * info->tout.height_stride + pre_w * w_stride;
	uint8_t *outp_pre_w = outp_core - pre_w * w_stride;
	uint8_t *outp_post_w = outp_core + w_in * w_stride;
	uint8_t *outp_pre_h = outp;
	uint8_t *outp_post_h = outp_core - pre_w * w_stride + h_in * info->tout.height_stride;

	const uint32_t w_mirror_start_offset = reflect * w_stride;
	const uint32_t h_mirror_start_offset = reflect * info->tout.height_stride;

	for (int b = 0; b < b_in; b++)
	{
		//Core copy
		outp_core = outp + pre_h * info->tout.height_stride + pre_w * w_stride;
		outp_pre_w = outp_core - pre_w * w_stride;
		outp_post_w = outp_core + w_in * w_stride;
		outp_pre_h = outp;

		for (int h = 0; h < h_in; h++)
		{
			nn_mcmanager_vmemcpy_2d(nn, mcman,
                                    		32 * w_in, info->tout.nd32,
                                    		outp_core + h * info->tout.height_stride, info->tout.d32_stride,
                                    		inp + h * info->tin.height_stride, info->tin.d32_stride);
		}
		nn_mcmanager_wait( nn, mcman );

		//Any pre width padding?
		if (pre_w)
		{
			for (int h = 0; h < h_in; h++)
			{
				for (int d = 0; d < info->tout.nd32; d++)
				{
					nn_mcmanager_vmemcpy_2d(nn, mcman,
								w_stride, pre_w,
								outp_pre_w + d * info->tout.d32_stride, w_stride,
								outp_pre_w + d * info->tout.d32_stride + w_stride * (2 * pre_w - 1) + w_mirror_start_offset, -w_stride);
				}
				outp_pre_w += info->tout.height_stride;
			}
		}

		//Any post width padding?
		if (post_w)
		{
			for (int h = 0; h < h_in; h++)
			{
				for (int d = 0; d < info->tout.nd32; d++)
				{
					nn_mcmanager_vmemcpy_2d(nn, mcman,
								w_stride, post_w,
								outp_post_w + d * info->tout.d32_stride, w_stride,
								outp_post_w + d * info->tout.d32_stride - w_stride - w_mirror_start_offset, -w_stride);
				}
				outp_post_w += info->tout.height_stride;
			}
		}
		nn_mcmanager_wait( nn, mcman );

		//Any pre height padding?
		if (pre_h)
		{
			for (int d = 0; d < info->tout.nd32; d++)
			{
				nn_mcmanager_vmemcpy_2d(nn, mcman,
							w_out * w_stride, pre_h,
							outp_pre_h, info->tout.height_stride,
							outp_pre_h + info->tout.height_stride * (2 * pre_h - 1) + h_mirror_start_offset, -info->tout.height_stride);
				outp_pre_h += info->tout.d32_stride;
			}
		}

		//Any post height padding?
		if (post_h)
		{
			for (int d = 0; d < info->tout.nd32; d++)
			{
				nn_mcmanager_vmemcpy_2d(nn, mcman,
							w_out * w_stride, post_h,
							outp_post_h, info->tout.height_stride,
							outp_post_h - info->tout.height_stride - h_mirror_start_offset, -info->tout.height_stride);
				outp_post_h += info->tout.d32_stride;
			}
		}
		nn_mcmanager_wait( nn, mcman );
		outp += info->tout.batch_stride;
		inp += info->tin.batch_stride;
	}
}

static void do_mirrorpad_wrapper(struct nn_graph *nn, void *info)
{

	struct mirrorpad_info *infop = (struct mirrorpad_info *)info;

	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman);

	do_mirrorpad(
		nn,
		&mcman,
		infop);

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

	if (padby[BATCH_PAD_BEFORE_IDX] || padby[BATCH_PAD_AFTER_IDX])
		return errlog(nn, "Can't pad batches");
	if (padby[DEPTH_PAD_BEFORE_IDX] || padby[DEPTH_PAD_AFTER_IDX])
		return errlog(nn, "Can't pad depth");

	if (padby[HEIGHT_PAD_BEFORE_IDX] >= info->in_shape.height)
		return errlog(nn, "height too small (%d>=%d)", padby[HEIGHT_PAD_BEFORE_IDX], info->in_shape.height);
	if (padby[HEIGHT_PAD_AFTER_IDX] >= info->in_shape.height)
		return errlog(nn, "height too small (%d>=%d)", padby[HEIGHT_PAD_AFTER_IDX], info->in_shape.height);

	if (padby[WIDTH_PAD_BEFORE_IDX] >= info->in_shape.width)
		return errlog(nn, "width too small (%d>=%d)", padby[WIDTH_PAD_BEFORE_IDX], info->in_shape.width);
	if (padby[WIDTH_PAD_AFTER_IDX] >= info->in_shape.width)
		return errlog(nn, "width too small (%d>=%d)", padby[WIDTH_PAD_AFTER_IDX], info->in_shape.width);

	int w_in_before_pad = in_tensor->format.width_pad[0];
	int w_out_before_pad = w_in_before_pad;
	int w_out_after_pad = (-(out_shape.width + w_out_before_pad)) & 3;
	int d_out_after_pad = (-out_shape.depth) & 31;

	int res = tensor_out_prepare_padded_d32(out_tensor,
											out_shape.batches,
											out_shape.height, in_tensor->format.height_pad[0], in_tensor->format.height_pad[1],
											out_shape.width, w_out_before_pad, w_out_after_pad,
											out_shape.depth, 0, d_out_after_pad, NN_TYPE_QUINT8);
	info->tin = tensor_addressing_d32(in_tensor);
	info->tout = tensor_addressing_d32(out_tensor);
	if (res != 0)
	{
		return errlog(nn, "output too small");
	}
	nn_sem_init(&info->done_sem, 0);
	nn_os_work_for_vector(nn, do_mirrorpad_wrapper, info);
	nn_sem_wait(&info->done_sem);
	return 0;
}

static int mirrorpad_execute_8_d32(struct nn_node *self, struct nn_graph *nn)
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

struct nn_node_ops nn_ops_for_MirrorPad_8_d32 = {
	.execute = mirrorpad_execute_8_d32,
	.check = mirrorpad_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};
