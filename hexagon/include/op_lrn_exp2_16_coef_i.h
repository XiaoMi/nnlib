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

#ifndef NN_OP_LRN_EXP2_H
#define NN_OP_LRN_EXP2_H

#ifndef CINTRINSIC_NONLIN_DEFS
#define CINTRINSIC_NONLIN_DEFS
#endif

#ifdef CINTRINSIC_NONLIN_DEFS
const short lut_non_lin_exp2_16[192] __attribute__ ((aligned(128))) = {
32767,
0,
31379,
0,
30048,
0,
28774,
0,
27554,
0,
26386,
0,
25268,
0,
24196,
0,
23170,
0,
22188,
0,
21247,
0,
20347,
0,
19484,
0,
18658,
0,
17867,
0,
17109,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
-5677,
0,
-5437,
0,
-5206,
0,
-4985,
0,
-4774,
0,
-4572,
0,
-4378,
0,
-4192,
0,
-4014,
0,
-3844,
0,
-3681,
0,
-3525,
0,
-3376,
0,
-3233,
0,
-3096,
0,
-2964,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
481,
0,
461,
0,
442,
0,
423,
0,
405,
0,
388,
0,
371,
0,
356,
0,
340,
0,
326,
0,
312,
0,
299,
0,
286,
0,
274,
0,
263,
0,
251,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
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

#define N_ORDER_EXP2_16  (2)


#define DELTAX_NBITS_EXP2_16  (11) 
#define DELTAX_LSHIFT_EXP2_16  (2) // = (SCALE) 
#define DELTAX_MASK_EXP2_16  (0x7ff) // = ((2^DELTAX_NBITS) - 1) 
#define RNGIDX_NBITS_EXP2_16  (4) 
#define RNGIDX_RSHIFT_EXP2_16  (11) // = (DELTAX_NBITS) 
#define RNGIDX_MASK_EXP2_16  (0x7800) // = (((2^RNGIDX_NBITS) - 1) << RNGIDX_RSHIFT) 

#endif
