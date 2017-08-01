
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
#ifndef NN_GRAPH_IM2COL_H
#define NN_GRAPH_IM2COL_H 1

void im2col_row(
	uint8_t *out,
	const uint8_t *in,
	int32_t in_x,
	int32_t in_width,
	int32_t filt_width,
	int32_t depth,
	uint8_t zero_val);

void im2col_stripe(
	uint8_t *out,
	const uint8_t *in,
	int32_t in_x,
	int32_t in_width,
	int32_t filt_width,
	int32_t in_y,
	int32_t in_height,
	int32_t filt_height,
	int32_t depth,
	int8_t zero_val);

void im2col_full(uint8_t *out,
	const uint8_t *in,
	int32_t in_height,
	int32_t in_width,
	int32_t in_depth,
	int32_t filt_height,
	int32_t filt_width,
	int32_t stride_width,
	int32_t stride_height,
	int32_t out_height_start,
	int32_t out_height_stop,
	int32_t out_height,
	int32_t out_width,
	int32_t out_depth,
	uint8_t zero_val,
	int32_t padding);


#endif

