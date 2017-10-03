
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

struct nn_padding_outsize_and_inpadding {
	int32_t out_size;
	int32_t pad_before;
	int32_t pad_after;
};
// This inline finds the output size, before, and after padding
// as a struct.
//
static inline struct nn_padding_outsize_and_inpadding
nn_pad_compute_outsize_padding(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	struct nn_padding_outsize_and_inpadding result;
	result.out_size = 0;
	result.pad_before = 0;
	result.pad_after = 0;

	int32_t delt = in_size - filt_size;
	int32_t pad_n = 0;	//total padding
	if (padding == NN_PAD_VALID) {
		// size = (in-filt)/stride + 1;
		if( delt >= 0)
			result.out_size = (unsigned)delt / (unsigned)stride + 1;
	}else if ((padding == NN_PAD_NA) ||(padding == NN_PAD_SAME) || (padding == NN_PAD_SAME_CAFFE)) {
		int32_t n = in_size;
		if( padding != NN_PAD_NA)
			n += stride-1;
		if( n >= stride){
			int32_t out_size = (unsigned)n / (unsigned)stride;		// >= 1
			pad_n = (out_size-1) * stride - delt;
			result.out_size = out_size;
			if( pad_n > 0){
				int32_t pad_n1 = pad_n;
				if( padding == NN_PAD_SAME_CAFFE)
					++pad_n1;
				pad_n1 >>= 1;
				result.pad_before = pad_n1;
				result.pad_after = pad_n-pad_n1;
			}
		}
	}
	return result;
}

//
// find outsize (returned) and before/after padding, for given situation and padding mode.
//
// pad_before > 0 means left padding
// i.e first output is generated using samples from   -pad_before .. -pad_before+filt_size-1.
// Output 'i' is generated using  i*stride-pad_before   ... i*stride-pad_before + filt_size-1.
//
// note:
//  - return value is >=1 ; or 0 when the situation is impossible
//  - -pad_before, pad_after are always >=0
//
// Impossible cases (assuming in_size, filt_size, stride all >= 1):
//    NA_PAD_NA:    impossible if in_size < stride
//    NA_PAD_SAME:  always possible
//    NA_PAD_VALID: impossible if filt_size > in_size
//

//
// same function , wrapped as an inline without the struct return.
//
static inline int32_t nn_pad_compute_outsize_and_pad(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding,
	int32_t * pad_before_p,
	int32_t * pad_after_p )
{
	struct nn_padding_outsize_and_inpadding tmp = nn_pad_compute_outsize_padding( in_size, filt_size, stride, padding);
	*pad_before_p = tmp.pad_before;
	*pad_after_p = tmp.pad_after;
	return tmp.out_size;
}
//
// same thing but only gives size and before padding
//
static inline int32_t nn_pad_compute_outsize_and_padbefore(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding,
	int32_t * pad_before_p )
{
	struct nn_padding_outsize_and_inpadding tmp = nn_pad_compute_outsize_padding( in_size, filt_size, stride, padding);
	*pad_before_p = tmp.pad_before;
	return tmp.out_size;
}
//
// same thing but only gives size
//
static inline int32_t nn_pad_compute_outsize(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	return nn_pad_compute_outsize_padding( in_size, filt_size, stride, padding).out_size;
}
//
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
 *
 * Note, padding can be negative when filt_size < stride
 *  E.g. in = 20, stride = 4, filt = 3
 *  filt_size - 1 - ((in_size + stride - 1) % stride)   = 2 - 23%4 =  2-3 = -1
 *  In these cases clamp to 0.
 */

static inline int32_t nn_pad_compute_total(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	if ((padding == NN_PAD_SAME) || (padding == NN_PAD_SAME_CAFFE)) {
		//return nn_pad_compute_outsize(in_size,filt_size,stride,padding)*stride+filt_size-in_size-stride;
		int padn = filt_size - 1 - ((in_size + stride - 1) % stride);
		return (padn <0)?0: padn;
	} else if (padding == NN_PAD_VALID) {
		return 0;
		/*
		int32_t outsize = nn_pad_compute_outsize(in_size,filt_size,stride,padding);
		int32_t used_insize = (outsize-1) * stride + filt_size;
		// used_insize is always <= in_size
		if (used_insize <= in_size) return 0;
		return used_insize - in_size;
		*/
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
	return nn_pad_compute_outsize_padding( in_size, filt_size, stride, padding).pad_before;
}

static inline int32_t nn_pad_compute_after(
	int32_t in_size,
	int32_t filt_size,
	int32_t stride,
	padding_type padding)
{
	return nn_pad_compute_outsize_padding( in_size, filt_size, stride, padding).pad_after;
}

//
// This is for the 'deconv' operator; given
// in_size, filt_size, stride, padding type
// it finds the smallest 'out_size' such that
//   nn_pad_compute_outsize( out_size, filt_size, stride, padding) == in_size.
//
// NN_PAD_NA:
//   result is in_size * stride;
// NN_PAD_SAME[_CAFFE]:
//   result is (in_size-1) * stride  + 1
// NN_PAD_VALID:
//    result is (in_size-1)*stride + filt_size
//
// We assume that in_size, filt_size, stride all >=1.
// The only 'invalid' return (0) is from unrecognized padding
//
//
static inline int32_t
nn_pad_compute_outsize_inverse(
		int32_t in_size,
		int32_t filt_size,
		int32_t stride,
		padding_type padding)
{
	int inxst = (in_size-1)*stride;
	if( padding == NN_PAD_NA ){
		return inxst + stride;		// in_size*stride
	}else if((padding == NN_PAD_SAME) || (padding == NN_PAD_SAME_CAFFE)){
		return inxst + 1;			// (in_size-1)*stride + 1;
	}else if( padding == NN_PAD_VALID){
		return inxst + filt_size;	// (in_size-2)*stride + filt_size
	}
	return 0;
}


#endif
