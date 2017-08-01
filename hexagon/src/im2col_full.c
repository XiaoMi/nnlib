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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for a full im2col when it's required
 */


#include <nn_graph.h>

void im2col_row(
	uint8_t *out,
	const uint8_t *in,
	int32_t in_x,
	int32_t in_width,
	int32_t filt_width,
	int32_t depth,
	uint8_t zero_val)
{
	uint32_t depth_size = depth * sizeof(*in);
	int32_t x = in_x;
	if (x < 0) {
		memset(out,zero_val,depth_size*-x);
		out += depth_size*-x;
		x = 0;
	}
	for (; x < (in_x + filt_width); x++) {
		if (x >= in_width) break;
		else memcpy(out,in+x*depth,depth_size);
		out += depth_size;
	}
	if (x < (in_x + filt_width)) {
		int cols = in_x + filt_width - in_width;
		memset(out,zero_val,depth_size*cols);
	}
}

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
	int8_t zero_val)
{
	uint32_t depth_size = depth * sizeof(*in);
	uint32_t width_size = filt_width * depth_size;
	int y = in_y;
	if (y < 0) {
		memset(out,zero_val,width_size*-y);
		out += width_size*-y;
		y = 0;
	}
	for (; y < (in_y+filt_height); y++) {
		if (y >= in_height) break;
		else im2col_row(out,
			in+y*depth*in_width,
			in_x,
			in_width,
			filt_width,
			depth,
			zero_val);
		out += width_size;
	}
	if (y < (in_y + filt_height)) {
		int rows = in_y + filt_height - in_height;
		memset(out,zero_val,width_size*rows);
	}
}

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
	int32_t padding)
{
	int32_t out_x;
	int32_t out_y;
	int32_t in_x;
	int32_t in_y;
	//int32_t adj_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	//int32_t adj_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;
	int32_t adj_x = nn_pad_compute_before(out_width,filt_width,stride_width,(padding_type)padding);
	int32_t adj_y = nn_pad_compute_before(out_height,filt_height,stride_height,(padding_type)padding);
	out += out_depth * out_width * out_height_start;
	memset(out,zero_val,(out_height_stop-out_height_start)*out_width*out_depth);
	for (out_y = out_height_start; out_y < out_height_stop; out_y++) {
		in_y = out_y * stride_height - adj_y;
		for (out_x = 0; out_x < out_width; out_x++) {
			in_x = out_x * stride_width - adj_x;
			im2col_stripe(
				out,
				in,
				in_x,
				in_width,
				filt_width,
				in_y,
				in_height,
				filt_height,
				in_depth,
				zero_val);
			out += out_depth;
		}
	}
}
