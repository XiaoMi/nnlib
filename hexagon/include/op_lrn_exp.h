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

#ifndef NN_OP_LRN_EXP_H
#define NN_OP_LRN_EXP_H

#ifndef CINTRINSIC_NONLIN_DEFS
#define CINTRINSIC_NONLIN_DEFS
#endif

#ifdef CINTRINSIC_NONLIN_DEFS
const signed char lut_non_lin_asm_lrn_exp10[256] __attribute__ ((aligned(128))) = {
13,
0,
15,
0,
17,
0,
20,
0,
23,
0,
26,
0,
30,
0,
35,
0,
40,
0,
47,
0,
54,
0,
62,
0,
72,
0,
83,
0,
96,
0,
111,
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
127,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
16,
0,
18,
0,
21,
0,
24,
0,
28,
0,
33,
0,
38,
0,
43,
0,
50,
0,
58,
0,
67,
0,
77,
0,
89,
0,
103,
0,
119,
0,
127,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
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

#ifndef NN_OP_LRN_HVX_COMMON
#define N_ORDER (1)

#define SCALE (1)

#define LOG2_MAX_ABS_RNG (0) 
#define INT_MAX_ABS_RNG (1) 

#define LOG2_MAX_RNG (0) 
#define INT_MAX_RNG (1) 
#define MAX_RNG (0.0) 

#define INT_MIN_RNG (-1) 
#define MIN_RNG (-1.0) 

#define Q_n (7) 
#define Qbasis (7) 
#define Qpoint (7) 
#define QDIFF_MULT (1) // = (1 << (Qbasis - Qpoint)) 

#define LIM_MAX (127) // = (0x7f) = ((INT_MAX_ABS_RNG << Qpoint) - 1) 
#define LIM_MIN (-128) // = (-0x80) = (-INT_MAX_ABS_RNG << Qpoint) 

#define LUT_IGNORE_NBITS (0) 

#define DELTAX_NBITS (3) 
#define DELTAX_LSHIFT (1) // = (LOG2_MAX_ABS_RNG + SCALE) 
#define DELTAX_MASK (0x7) // = ((2^DELTAX_NBITS) - 1) 
#define DELTAX_MULT (2) // = (1 << DELTAX_LSHIFT) 

#define RNGIDX_NBITS (5) 
#define RNGIDX_RSHIFT (3) // = (DELTAX_NBITS) 
#define RNGIDX_MASK (0xf8) // = (((2^RNGIDX_NBITS) - 1) << RNGIDX_RSHIFT) 
#define RNGIDX_UNSHIFTED_MASK (0x1f) // = ((2^RNGIDX_NBITS) - 1) 
#define RNGIDX_SIGNBITS (0xf0) // = (~((2^(RNGIDX_NBITS-1)) - 1)) 
#define RNGIDX_OFST (0x10) // (INTERVALS/2) if funct non-symmetric and min_rng less than 0 else 0

#define IN_RNG_FLOAT_NEGATIVE

#define OUT_RNG_FLOAT_ZERO_TO_POSITIVE

#define NN_OP_LRN_HVX_COMMON
#endif

#define N_ORDER_LRN_EXP10 (1)

#define MAX_RNG_LRN_EXP10 (0.0) 
#define MIN_RNG_LRN_EXP10 (-1.0) 

#define QDIFF_MULT_EXP10 (1) // = (1 << (Qbasis - Qpoint))

#define LIM_MAX_EXP10 (127) // = (0x7f) = ((INT_MAX_ABS_RNG << Qpoint) - 1)
#define LIM_MIN_EXP10 (-128) // = (-0x80) = (-INT_MAX_ABS_RNG << Qpoint)

#define DELTAX_NBITS_LRN_EXP10 (3) 
#define DELTAX_LSHIFT_LRN_EXP10 (1) // = (LOG2_MAX_ABS_RNG + SCALE) 
#define DELTAX_MASK_LRN_EXP10 (0x7) // = ((2^DELTAX_NBITS) - 1) 
#define DELTAX_MULT_LRN_EXP10 (2) // = (1 << DELTAX_LSHIFT) 

#define RNGIDX_NBITS_LRN_EXP10 (5) 
#define RNGIDX_RSHIFT_LRN_EXP10 (3) // = (DELTAX_NBITS) 
#define RNGIDX_MASK_LRN_EXP10 (0xf8) // = (((2^RNGIDX_NBITS) - 1) << RNGIDX_RSHIFT) 
#define RNGIDX_UNSHIFTED_MASK_EXP10 (0x1f) // = ((2^RNGIDX_NBITS) - 1)
#define RNGIDX_SIGNBITS_LRN_EXP10 (0xf0) // = (~((2^(RNGIDX_NBITS-1)) - 1))
#define RNGIDX_OFST_LRN_EXP10 (0x10) // (INTERVALS/2) if funct non-symmetric and min_rng less than 0 else 0

#define IN_RNG_FLOAT_NEGATIVE_LRN_EXP10 (1)

#define OUT_RNG_FLOAT_NONNEGATIVE_LRN_EXP10 (1)

#endif
