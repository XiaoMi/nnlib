
/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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

void transpack_16(
  const uint16_t* in_data, int K, int M, uint16_t* out_data)
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

static void pad2d_generic(
	void const * input_data,	//  { inh,  inw ,  (elbytes)}
	int input_height,
	int input_width,
	void * output_data,	//  { outh,  outw ,  (elbytes)}
	int output_height,
	int output_width,
	int pad_value,
	int elbytes ) // may be 1,2 or 4 (or any, if pad_value=0)
{
	if( elbytes == 1) pad_value = Q6_R_vsplatb_R(pad_value);
	else if ( elbytes == 2)pad_value = Q6_R_combine_RlRl(pad_value,pad_value);

    const uint8_t* ptr_in = input_data;
    uint8_t* ptr_out = output_data;
    int pad_x = output_width - input_width;
    int pad_y = output_height - input_height;
    if( pad_x > 0){
		vmemcpy_2d_general_asm(
				input_width * elbytes,  input_height,	// rect width, height
				ptr_out, output_width * elbytes, 	// dst address, stride
				ptr_in, input_width * elbytes );
		vmemset_32_2d_general_asm(
				ptr_out + input_width*elbytes,			// location
				pad_value,						// pad value (32 bits)
				pad_x * elbytes, input_height,	// w,h of region
				output_width * elbytes			// stride
			);
    }else{
		vmemcpy_asm( ptr_out, ptr_in, input_height * output_width*elbytes);
    }
    if( pad_y > 0){
		ptr_out += input_height * output_width*elbytes;
		// fill as 'single row'
		vmemset_32_2d_general_asm(
				ptr_out,
				pad_value,
				pad_y*output_width*elbytes, 1,		// width, height
				0);
    }
}

void pad2d(
  const uint8_t* input_data, int input_height, int input_width,
  uint8_t* output_data, int output_height, int output_width, int pad_value)
{
	pad2d_generic( input_data, input_height, input_width,
			output_data, output_height, output_width, pad_value, sizeof(uint8_t));
}


void pad2d_16(
  const uint16_t* input_data, int input_height, int input_width,
  uint16_t* output_data, int output_height, int output_width, int pad_value)
{
	pad2d_generic( input_data, input_height, input_width,
			output_data, output_height, output_width, pad_value, sizeof(uint16_t));
}

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
// unpad: output shape h x w must be <= input shape; right/bottom removed.
//
void unpad2d(
    const int* input_data, int input_height, int input_width,  
    int* output_data, int output_height, int output_width)
{
	int elbytes = sizeof(int);
	vmemcpy_2d_general_asm(
			output_width * elbytes,  output_height,	// rect width, height
			output_data, output_width * elbytes, 	// dst address, stride
			input_data, input_width * elbytes );	// src address, stride
}
void unpad2d_bytes(
	const uint8_t* input_data, int input_height, int input_width,
	uint8_t* output_data, int output_height, int output_width)
{
	int elbytes = sizeof(uint8_t);
	vmemcpy_2d_general_asm(
			output_width * elbytes,  output_height,	// rect width, height
			output_data, output_width * elbytes, 	// dst address, stride
			input_data, input_width * elbytes );	// src address, stride
}
