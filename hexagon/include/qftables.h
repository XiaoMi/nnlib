/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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

#include "nn_graph_builtin.h"

//scaler inverse table
static const short int lut_inv_cn[16*4] = {
//---------------- 0.500000 to 0.562500 ----------------------
  16384, -16265, 13718,  //01000 
//---------------- 0.562500 to 0.625000 ----------------------
  14563, -12869, 9816 ,  //01001 
//---------------- 0.625000 to 0.687500 ----------------------
  13107, -10434, 7267 ,  //01010 
//---------------- 0.687500 to 0.750000 ----------------------
  11915, -8630,  5528 ,  //01011 
//---------------- 0.750000 to 0.812500 ----------------------
  10923, -7256,  4304 ,  //01100 
//---------------- 0.812500 to 0.875000 ----------------------
  10082, -6185,  3415 ,  //01101 
//---------------- 0.875000 to 0.937500 ----------------------
  9362,  -5335,  2756 ,  //01110 
//---------------- 0.937500 to 1.000000 ----------------------
  8738,  -4649,  2255 ,  //01111 
//---------------- -1.000000 to -0.937500 ----------------------
  -8192, -4084, -2255 ,  //11000
//---------------- -0.937500 to -0.875000 ----------------------
  -8738, -4649, -2756 ,  //11001 
//---------------- -0.875000 to -0.812500 ----------------------
  -9362, -5335, -3415 ,  //11010 
//---------------- -0.812500 to -0.750000 ----------------------
  -10082,-6185, -4304 ,  //11011 
//---------------- -0.750000 to -0.687500 ----------------------
  -10923,-7248, -5528 ,  //11100 
//---------------- -0.687500 to -0.625000 ----------------------
  -11915,-8630, -7267 ,  //11101 
//---------------- -0.625000 to -0.562500 ----------------------
  -13107,-10434,-9816 ,  //11110 
//---------------- -0.562500 to -0.500000 ----------------------
  -14563,-12865,-13718   //11111 
};


//scalar isqrt table
static const short int lut_isqrt_cn[16*4] = {
 23170, 22479, 21845, 21263, 20724, 20225, 19760, 19326,
 18919, 18536, 18176, 17837, 17515, 17211, 16921, 16646,
-11584,-10577, -9708, -8952, -8289, -7704, -7184, -6721,
 -6305, -5931, -5592, -5284, -5003, -4747, -4511, -4295,
  8661,  7446,  6456,  5641,  4963,  4394,  3912,  3501,
  3148,  2843,  2578,  2346,  2142,  1963,  1803,  1661,
 -6508, -5296, -4359, -3625, -3042, -2575, -2196, -1886,
 -1630, -1416, -1238, -1087,  -959,  -850,  -756,  -675
};

//hvx isqrt table

static const short lut_isqrt_asm[3*64] __attribute__ ((aligned(128))) = {
 23170, 0, 22479, 0, 21845, 0, 21263, 0, 20724, 0, 20225, 0, 19760, 0, 19325, 0,
 18919, 0, 18536, 0, 18176, 0, 17837, 0, 17515, 0, 17211, 0, 16921, 0, 16646, 0,
     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,
     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,
-11570, 0,-10565, 0, -9698, 0, -8944, 0, -8282, 0, -7698, 0, -7180, 0, -6717, 0,
 -6302, 0, -5927, 0, -5589, 0, -5282, 0, -5001, 0, -4745, 0, -4510, 0, -4293, 0,
     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,
     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,
  8052, 0,  6950, 0,  6048, 0,  5302, 0,  4679, 0,  4153, 0,  3707, 0,  3325, 0,
  2996, 0,  2711, 0,  2462,  0, 2244, 0,  2052, 0,  1883, 0,  1732, 0,  1598, 0,
     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,
     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0,     0, 0
};

/*======================================================================*/
/*                              end of file                             */
/*======================================================================*/
