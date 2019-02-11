
/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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

#ifndef NN_GRAPH_FLOAT_MATHOPS_H
#define NN_GRAPH_FLOAT_MATHOPS_H

//Implementation of Schraudolph's algorithm for exp(x) with a cubic correction for the mantissa
//This saves a floating point division
//max rel. error [-87.33654, 88.72283] = 8.34e-5
//Most of the magic constants are empirically determined
//12102203.0f ~= 2^23 / ln(2)
//1065353216 ~= 127 * (2^23 / ln(2))

static inline __attribute__((unused,always_inline)) float fast_exp (float x) {
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1065353216);
    int32_t m = (u.i >> 7) & 0xFFFF;
    u.i += ((((((((1277 * m) >> 14) + 14825) * m) >> 14) - 79749) * m) >> 11) - 626;
    return u.f;
}

static inline float linear_interpolate(float s, float e, float t){
	return s+(e-s)*t;//equivalent to (1-t)*s + t*e
}

static inline float bilinear_interpolate(float c00, float c01, float c10, float c11, float tx, float ty){
	return linear_interpolate(linear_interpolate(c00, c01, tx), linear_interpolate(c10, c11, tx), ty);
}

#endif //NN_GRAPH_FLOAT_MATHOPS_H
