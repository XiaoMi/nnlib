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
#ifndef NN_BULK_TRANSPOSE_H
#define NN_BULK_TRANSPOSE_H 1

#include <stdint.h>

///////////////////////////////////////////////////////////////////////////
// 'bulk transpose' function
//
// for flat tensors, does (b,hin,win,d) -> (b,win,hin,d)
// where 'd' is in bytes
// (if your elements are not 1 byte, just multiply depth by
// the element size to get d).
// If you want to do something like (x,y,z) -> (x,z,y), insert a dummy 'd'
//     (x,y,z,d) ->  (x,z,y,d) where d = 1*elementsize.
// More complex cases can be done using multiple calls to perform_bulk_transpose_2,
// see below.
//
// ***********************************
// **  d must be one of: 1,2,4,8,16 **
// ***********************************
//
// This must be called in an hvx thread; and it's probably not
// the best way to go unless h*w*d is at least 1K; also,
//  - if min(h,w) <= 4 there are better ways to do this,
//  - if min(h,w)*d < 32 it won't use the vectors very efficiently
//  But it's still a good fallback in such cases.
//
// 'work_area' must point to a temp. area, vector aligned, of sufficient
// size. 16384/d bytes is always sufficient, but sometimes you need less; if you
// call with the specific b,h,w,d values and work_area = NULL, it will return
// the # of bytes needed (which is always a multiple of 1K).
//
int
perform_bulk_transpose(
		void *outp,			// output buffer b * w * h * d bytes
		void const *inp,	// input  buffer b * h * w * d bytes
		void * work_area,	// vector aligned work area (see below).
		uint32_t b,			// batches >= 1
		uint32_t h,			// input height >= 1 (outpt width)
		uint32_t w,			// input width >= 1 (output height)
		uint32_t d,			// element size, bytes (power of 1, 1,2,4..16
		int flags);			// reserved for multi-threading tag. For now
							// always pass 0.

// the more sophisticated bulk-transpose routine supports 'gaps' in the b and h dimensions,
// and you pass most parameters in a struct.
// Note that the input & output 'd' stride is always 1;
// the input & output 'w' strides are both = d.
//
// This can be used to construct more complex transposes;
// For instance if you want to transpose [h,r,s,w,d] -> [ w,s,r,h,d]
// then this can be done using in_h_stride = r*s*w*d, out_h_stride = s*r*h*d
// and if 'r' is your batch dimension, in_b_stride = s*w*d and out_b_stride = h*d
// You will then need an outer loop over the 's' dimension,
// using input stride w*d and output stride r*h*d
// In general, 'd' is the last dimension of each tensor - even if you need to insert
// a dummy '1*elementsize' dimension to allow that; 'w' is the 2nd-last dimension
// of the input, and 'h' is the 2nd-last dimension of the output. 'b' can be any dimension
// left over, but the 3rd-last output dimension (or 4th-last, if the 3rd-last is 'w')
// is probably the best choice in general to get better store locality,
//
// This version of the transpose also allows you to crop and transpose in one step,
// e..g [  b, h0+h+h1, w0+w+w1, d] -> [b,w,h,d]
//     in_h_stride = (w0+w+w1)*d                 out_h_sride = h*d
//     in_b_stride = in_h_stride *(h0+h+h1)     out_b_stride = w*h*d
// .. and you need to offset the input pointer by w0*d + h0*in_h_stride
// Likewise, the input can be transposed and dropped into the middle area of a 'padded' tensor,
// by padding the output strides.
//
struct bulk_transpose_parms {
	uint32_t in_dims[4];			// b,h,w,d ; input shape
	int32_t in_h_stride;			// stride of input h in bytes
	int32_t in_b_stride;			// stride of input b in bytes
	int32_t out_h_stride;			// stride of output h (same dim as input w) in bytes
	int32_t out_b_stride;			// stride of output b in bytes
};

// similar to perform_bulk_transpose except that the 7 shape params
// are passed via a struct.
// you can call with work_area = NULL and it will return the work
// area size (or -1).
int
perform_bulk_transpose_2(
		void *outp,			// output buffer
		void const *inp,	// input  buffer
		void * work_area,	// vector aligned work area (see below).
		struct bulk_transpose_parms const * parms,
		int flags);			// reserved for multi-threading tag. For now use 0.


// an internal function used by 'perform_bulk_transpose'.
//
void transpose_rectangle( uint8_t * buffer, int elementsize, int Nh, int Nw);

#endif // NN_BULK_TRANSPOSE_H 1
