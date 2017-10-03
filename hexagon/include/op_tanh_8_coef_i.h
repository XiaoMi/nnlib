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

#ifndef NN_OP_TANH_COEF_H
#define NN_OP_TANH_COEF_H

#ifndef CINTRINSIC_NONLIN_DEFS
#define CINTRINSIC_NONLIN_DEFS
#endif

#ifdef CINTRINSIC_NONLIN_DEFS
const signed char lut_non_lin_tanh_8[256] __attribute__ ((aligned(128))) = {
0,
0,
32,
0,
60,
0,
82,
0,
98,
0,
109,
0,
116,
0,
121,
0,
123,
0,
125,
0,
126,
0,
127,
0,
127,
0,
127,
0,
127,
0,
127,
0,
-127,
0,
-127,
0,
-127,
0,
-127,
0,
-127,
0,
-127,
0,
-126,
0,
-125,
0,
-123,
0,
-121,
0,
-116,
0,
-109,
0,
-98,
0,
-82,
0,
-59,
0,
-31,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
125,
0,
111,
0,
89,
0,
65,
0,
44,
0,
29,
0,
19,
0,
12,
0,
7,
0,
4,
0,
3,
0,
2,
0,
1,
0,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
0,
1,
0,
2,
0,
3,
0,
4,
0,
7,
0,
12,
0,
19,
0,
29,
0,
44,
0,
65,
0,
89,
0,
111,
0,
125,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
};
#endif

#define SCALE (2)

#define SCALE_TANH_8   (2)

#define N_ORDER_TANH_8  (1)

#define DELTAX_NBITS_TANH_8  (3) 
#define DELTAX_LSHIFT_TANH_8  (2) // = (SCALE) 
#define DELTAX_MASK_TANH_8  (0x7) // = ((2^DELTAX_NBITS) - 1) 
#define RNGIDX_NBITS_TANH_8  (5) 
#define RNGIDX_RSHIFT_TANH_8  (3) // = (DELTAX_NBITS) 
#define RNGIDX_MASK_TANH_8  (0xf8) // = (((2^RNGIDX_NBITS) - 1) << RNGIDX_RSHIFT) 

#define MIN_RNG_TANH_8  (-4)
#define MAX_RNG_TANH_8  (4)

#ifdef OUT_Q7_VAL_IS_NEGATIVE_TO_POSITIVE
#undef OUT_Q7_VAL_IS_NEGATIVE_TO_POSITIVE
#endif
#define OUT_Q7_VAL_IS_NEGATIVE_TO_POSITIVE (1)

#endif
