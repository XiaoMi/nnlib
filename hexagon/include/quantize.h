
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
#ifndef NN_QUANTIZE_H
#define NN_QUANTIZE_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for common quantization routines
 */

#include <stdint.h>
#include <math.h>

static inline uint8_t quantize_uint8(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(0.0001f,maxval-minval);
	float resize_amt = 255.0f/range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	uint8_t ret = (value_i < 0) ? 0 : ((value_i > 255) ? 255 : value_i);
	return ret;
}

static inline int32_t quantize_uint(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(0.0001f,maxval-minval);
	float resize_amt = 255.0f/range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	int32_t ret = (value_i < 0) ? 0 : value_i;
	return ret;
}

static inline int32_t quantize_int(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(0.0001f,maxval-minval);
	float resize_amt = 255.0f/(range);
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	return value_i;
}

static inline void quantize_adjust_range(float *out_min, float *out_max, float *out_stepsize, float *out_recip_stepsize, float in_min, float in_max)
{
	float minval = fminf(0.0f,in_min);
	float maxval = in_max;
	float range = fmaxf(0.0001f,maxval-minval);
	float stepsize = range/254.0f;
	float recip_stepsize = 254.0f/range;
	// round quantized_zero up so min_out <= minval
	int quantized_zero = ((0.0f - minval) * recip_stepsize) + 0.999;
	float newmin = -quantized_zero * stepsize;
	float newmax = 255.0f * stepsize + newmin;
	*out_min = newmin;
	*out_max = newmax;
	*out_stepsize = stepsize;
	*out_recip_stepsize = recip_stepsize;
}

#endif
