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

#ifndef NN_OP_SIGMOID_COEF_H
#define NN_OP_SIGMOID_COEF_H

#ifndef CINTRINSIC_NONLIN_DEFS
#define CINTRINSIC_NONLIN_DEFS
#endif

#ifdef CINTRINSIC_NONLIN_DEFS
const signed char lut_non_lin_sigmoid_8[256] __attribute__ ((aligned(128))) = {
64,
0,
72,
0,
80,
0,
87,
0,
94,
0,
100,
0,
105,
0,
109,
0,
113,
0,
116,
0,
118,
0,
120,
0,
122,
0,
123,
0,
124,
0,
125,
0,
2,
0,
3,
0,
4,
0,
5,
0,
6,
0,
8,
0,
10,
0,
12,
0,
15,
0,
19,
0,
23,
0,
28,
0,
34,
0,
41,
0,
48,
0,
56,
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
8,
0,
8,
0,
7,
0,
7,
0,
6,
0,
5,
0,
4,
0,
4,
0,
3,
0,
2,
0,
2,
0,
2,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
2,
0,
2,
0,
2,
0,
3,
0,
4,
0,
4,
0,
5,
0,
6,
0,
7,
0,
7,
0,
8,
0,
8,
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

#define SCALE (4)

#define SCALE_SIGMOID_8   (4)

#define N_ORDER_SIGMOID_8  (1)

#define DELTAX_NBITS_SIGMOID_8  (3) 
#define DELTAX_LSHIFT_SIGMOID_8  (4) // = (SCALE) 
#define DELTAX_MASK_SIGMOID_8  (0x7) // = ((2^DELTAX_NBITS) - 1) 
#define RNGIDX_NBITS_SIGMOID_8  (5) 
#define RNGIDX_RSHIFT_SIGMOID_8  (3) // = (DELTAX_NBITS) 
#define RNGIDX_MASK_SIGMOID_8  (0xf8) // = (((2^RNGIDX_NBITS) - 1) << RNGIDX_RSHIFT) 

#define MIN_RNG_SIGMOID_8  (-4)
#define MAX_RNG_SIGMOID_8  (4)

#ifdef OUT_Q7_VAL_IS_NEGATIVE_TO_POSITIVE
#undef OUT_Q7_VAL_IS_NEGATIVE_TO_POSITIVE
#endif
#define OUT_Q7_VAL_IS_NEGATIVE_TO_POSITIVE (0)

#endif
