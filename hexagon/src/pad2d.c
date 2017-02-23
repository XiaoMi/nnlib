
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
/*
 */
/*======================================================================*/
/*  FUNCTIONS      : op_pad2d.c                                         */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*  Take a 2d array like  matrix and padd x direction with pad_val      */
/*  pad v direction by pad_val                                          */
/*  Also perform reverse, strip out the edge and bottom not required    */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 08/24/16       created                          */
/*======================================================================*/
/*  C OPTIMIZATIONS                                                     */ 
/*  just use a really fast memcpy routine                               */
/*======================================================================*/
#include <nn_graph.h>
//void vmemcpy_asm(void *dst, void *src, int len);
//void vmemset_asm(void *dst, int val, int len);

/*
       <----------------------M------------------->
       <32>
    
 ^  ^  0000
 |  |  1111
 |  4  2222
 |  v  3333
 |
 K
 |
 |
 |
 v

       <----------------------K*128/4------------------->
       <-----------128-------->
    
 ^  ^  012301230123012301230123
 |
 M/32
 |
 v
 */

void transpack(
  const uint8_t* in_data, int K, int M, uint8_t* out_data)
{
    int x,y,z;

    //out_width = 32*K;
    //out_height = M/32;

    for (x = 0; x < M; x+=32)
    {
      for (y = 0; y < K; y+=4)
      for (z = 0; z < 32; z+=1)
      {
           out_data[32*y+K*x+4*z+0] = in_data[M*y+0*M+x+z];
           out_data[32*y+K*x+4*z+1] = in_data[M*y+1*M+x+z];
           out_data[32*y+K*x+4*z+2] = in_data[M*y+2*M+x+z];
           out_data[32*y+K*x+4*z+3] = in_data[M*y+3*M+x+z];
      }
    }
    return;
}

void pad2d(
  const uint8_t* input_data, int input_height, int input_width,  
  uint8_t* output_data, int output_height, int output_width, int pad_value)
{
    int out_y, pad_x, pad_y ;

    const uint8_t* ptr_in = input_data;
    uint8_t* ptr_out = output_data;
    pad_x = output_width - input_width;
    pad_y = output_height - input_height;
    for (out_y = 0; out_y < input_height; out_y++)
    {
        vmemcpy_asm(ptr_out, ptr_in, input_width);
        ptr_out += input_width;
        ptr_in += input_width;
        vmemset_asm(ptr_out, pad_value, pad_x);
        ptr_out += pad_x;
    }
    for (out_y = 0; out_y < pad_y; out_y++)
    {
       vmemset_asm(ptr_out, pad_value, output_width);
       ptr_out += output_width;
    }
    return;
}

void unpad2d(
    const int* input_data, int input_height, int input_width,  
    int* output_data, int output_height, int output_width)
{
    int out_y ;

    const int* ptr_in = input_data;
    int* ptr_out = output_data;
    for (out_y = 0; out_y < output_height; out_y++)
    {
        vmemcpy_asm(ptr_out, ptr_in, sizeof(int)*output_width);
        ptr_out += output_width;
        ptr_in += input_width;
    }
    return;
}

void unpad2d_bytes(
    const uint8_t* input_data, int input_height, int input_width,  
    uint8_t* output_data, int output_height, int output_width)
{
    int out_y ;

    const uint8_t* ptr_in = input_data;
    uint8_t* ptr_out = output_data;
    for (out_y = 0; out_y < output_height; out_y++)
    {
        vmemcpy_asm(ptr_out, ptr_in, sizeof(uint8_t)*output_width);
        ptr_out += output_width;
        ptr_in += input_width;
    }
    return;
}
