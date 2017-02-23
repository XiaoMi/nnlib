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

#ifndef NN_OP_SIGMOID_H
#define NN_OP_SIGMOID_H
#define CINTRINSIC_NONLIN_DEFS

#ifdef CINTRINSIC_NONLIN_DEFS
const signed char lut_non_lin_asm_sigmoid[256] __attribute__ ((aligned(128))) = {
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

#define N_ORDER (1)

#define SCALE (2)

#define LOG2_MAX_RNG (2) 
#define INT_MAX_RNG (4) 
#define MAX_RNG (4.0) 

#define INT_MIN_RNG (-4) 
#define MIN_RNG (-4.0) 

#define Q_n (7) 
#define Qbasis (5) 
#define Qpoint (5) 
#define QDIFF_MULT (1) // = (1 << (Qbasis - Qpoint)) 

#define LIM_MAX (127) // = (0x7f) = ((INT_MAX_RNG << Qpoint) - 1) 
#define LIM_MIN (-128) // = (-0x80) = (-INT_MAX_RNG << Qpoint) 

#define LUT_IGNORE_NBITS (0) 

#define DELTAX_NBITS (3) 
#define DELTAX_LSHIFT (4) // = (LOG2_MAX_RNG + SCALE) 
#define DELTAX_MASK (0x7) // = ((2^DELTAX_NBITS) - 1) 
#define DELTAX_MULT (16) // = (1 << DELTAX_LSHIFT) 

#define RNGIDX_NBITS (5) 
#define RNGIDX_RSHIFT (3) // = (DELTAX_NBITS) 
#define RNGIDX_MASK (0xf8) // = (((2^RNGIDX_NBITS) - 1) << RNGIDX_RSHIFT) 
#define RNGIDX_UNSHIFTED_MASK (0x1f) // = ((2^RNGIDX_NBITS) - 1) 
#define RNGIDX_SIGNBITS (0xf0) // = (~((2^(RNGIDX_NBITS-1)) - 1)) 
#define RNGIDX_OFST (0x10) // (INTERVALS/2) if funct non-symmetric else 0

#define OUT_RNG_FLOAT_ZERO_TO_POSITIVE

#endif
