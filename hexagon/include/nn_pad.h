
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

static inline void do_pad(
    void *outpv,
    const void *inpv,
    const int32_t b_in,
    const int32_t h_in,
    const int32_t w_in,
    const int32_t d_in,
    const int32_t pre_b,
    const int32_t post_b,
    const int32_t pre_h,
    const int32_t post_h,
    const int32_t pre_w,
    const int32_t post_w,
    const int32_t pre_d,
    const int32_t post_d,
    const int32_t element_size,
    const int32_t padval)
{
    const char *in = inpv;
    char *out = outpv;
    int h_out = h_in + pre_h + post_h;
    int w_out = w_in + pre_w + post_w;
    int d_out = d_in + pre_d + post_d;
    int out_depth_size = d_out * element_size;
    int out_width_size = w_out * out_depth_size;
    int out_height_size = h_out * out_width_size;
    int pre_b_size = out_height_size * pre_b;
    int pre_h_size = out_width_size * pre_h;
    int post_b_size = out_height_size * post_b;
    int post_h_size = out_width_size * post_h;
    int pre_w_size = out_depth_size * pre_w;
    int post_w_size = out_depth_size * post_w;
    int pre_d_size = element_size * pre_d;
    int post_d_size = element_size * post_d;
    int in_d_size = d_in * element_size;
    int b, h, w;

    memset(out, padval, pre_b_size);
    out += pre_b_size;
    for (b = 0; b < b_in; b++)
    {
        memset(out, padval, pre_h_size);
        out += pre_h_size;
        for (h = 0; h < h_in; h++)
        {
            memset(out, padval, pre_w_size);
            out += pre_w_size;
            if( d_out == d_in ){
            	memcpy(out, in, in_d_size*w_in);
            	in += in_d_size*w_in;
            	out += in_d_size*w_in;
            }else{
                for (w = 0; w < w_in; w++)
                {
                    memset(out, padval, pre_d_size);
                    out += pre_d_size;
                    memcpy(out, in, in_d_size);
                    in += in_d_size;
                    out += in_d_size;
                    memset(out, padval, post_d_size);
                    out += post_d_size;
                }
            }
            memset(out, padval, post_w_size);
            out += post_w_size;
        }
        memset(out, padval, post_h_size);
        out += post_h_size;
    }
    memset(out, padval, post_b_size);
    out += post_b_size;
}

static void memset16(void * dest, uint16_t value, int32_t size_in_int16_t)
{
	uint16_t *pdst = (uint16_t *)dest;
	for (int i = 0; i < size_in_int16_t; i++) {
		pdst[i] = value;
	}
}

static inline void do_pad16(
	void *outpv,
	const void *inpv,
	const int32_t b_in,
	const int32_t h_in,
	const int32_t w_in,
	const int32_t d_in,
	const int32_t pre_b,
	const int32_t post_b,
	const int32_t pre_h,
	const int32_t post_h,
	const int32_t pre_w,
	const int32_t post_w,
	const int32_t pre_d,
	const int32_t post_d,
	const int32_t element_size,
	const int32_t padval)
{
	const uint16_t *in = inpv;
	uint16_t *out = outpv;
	int h_out = h_in + pre_h + post_h;
	int w_out = w_in + pre_w + post_w;
	int d_out = d_in + pre_d + post_d;
	int out_depth_size = d_out;
	int out_width_size = w_out * out_depth_size;
	int out_height_size = h_out * out_width_size;
	int pre_b_size = out_height_size * pre_b;
	int pre_h_size = out_width_size * pre_h;
	int post_b_size = out_height_size * post_b;
	int post_h_size = out_width_size * post_h;
	int pre_w_size = out_depth_size * pre_w;
	int post_w_size = out_depth_size * post_w;
	int pre_d_size = pre_d;
	int post_d_size = post_d;
	int in_d_size = d_in;
	int b, h, w;

	memset16(out, padval, pre_b_size);
	out += pre_b_size;
	for (b = 0; b < b_in; b++)
	{
		memset16(out, padval, pre_h_size);
		out += pre_h_size;
		for (h = 0; h < h_in; h++)
		{
			memset16(out, padval, pre_w_size);
			out += pre_w_size;
			if (d_out == d_in) {
				memcpy(out, in, in_d_size*w_in*element_size);
				in += in_d_size * w_in;
				out += in_d_size * w_in;
			}
			else {
				for (w = 0; w < w_in; w++)
				{
					memset16(out, padval, pre_d_size);
					out += pre_d_size;
					memcpy(out, in, in_d_size*element_size);
					in += in_d_size;
					out += in_d_size;
					memset16(out, padval, post_d_size);
					out += post_d_size;
				}
			}
			memset16(out, padval, post_w_size);
			out += post_w_size;
		}
		memset16(out, padval, post_h_size);
		out += post_h_size;
	}
	memset16(out, padval, post_b_size);
	out += post_b_size;
}
