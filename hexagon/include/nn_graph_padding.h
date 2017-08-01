
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
#ifndef NN_PAD_HELP_H
#define NN_PAD_HELP_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for help with TF-style padding.
 * 
 * 
 */

#include <nn_graph_types.h>

/*
 * SAME is supposed to be the SAME size
 * VALID is only supposed to use VALID pixels, and so when filtering it will be smaller
 * 
 * But it's not so simple when striding.
 *
 * SAME or SAME_CAFFE size is ceil(float(size)/float(stride))
 * VALID size is ceil(float(size - filt_size + 1) / float(stride))
 *
 * We can get the ciel of the float divide by adding (stride-1) 
 * and using integer (floor) division.
 *
 */

static inline int32_t nn_pad_compute_outsize(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	if (padding == NN_PAD_NA) {
		return in_size / stride;
	}
	if ((padding == NN_PAD_SAME) || (padding == NN_PAD_SAME_CAFFE)) {
		return (in_size + (stride - 1)) / stride;
	}
	if (padding == NN_PAD_VALID) {
		/* size - filt_size + 1 + stride - 1: the 1's cancel */
		return (in_size - filt_size + stride) / stride;
	}
	return 0;
}

/* 
 * How much to pad?
 * For SAME size, we want a total padding amount:
 * ((size + stride - 1) / stride) * stride + filt_size - size - stride
 * The floor division means this is a little more tricky to simplify
 * ((size + stride - 1) + filt_size - size - stride) - ((size + stride - 1) % stride)
 * size and stride cancel
 * filt_size - 1 - ((size + stride - 1) % stride)
 * Or the filter size - 1 unless we have some extra pixels from striding.  That makes sense, I guess.
 * padding_before is floor(half(padding)) and padding_after is the remainder.
 */

static inline int32_t nn_pad_compute_total(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	if ((padding == NN_PAD_SAME) || (padding == NN_PAD_SAME_CAFFE)) {
		//return nn_pad_compute_outsize(in_size,filt_size,stride,padding)*stride+filt_size-in_size-stride;
		return filt_size - 1 - ((in_size + stride - 1) % stride);
	}
	return 0;
}

/*
 * SAME_CAFFE is different from SAME (TensorFlow) when there is asymmetrical padding.
 * SAME_CAFFE the additional padding is on the left, in a sense it'll provide symmetrical padding.
 * SAME the additional padding is on the right
 */
static inline int32_t nn_pad_compute_before(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	int round_up = 0;
	if (padding == NN_PAD_SAME_CAFFE) round_up = 1;
	return (nn_pad_compute_total(in_size,filt_size,stride,padding) + round_up)/2;
}

static inline int32_t nn_pad_compute_after(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	return nn_pad_compute_total(in_size,filt_size,stride,padding)
		- nn_pad_compute_before(in_size,filt_size,stride,padding);
}

#endif
