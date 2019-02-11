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
   GVCONV2DBBB
       - generalized vector convolution - 2dimensional - byte * byte => byte
       - meax and min of the accumulations are tracked.

   Notes
   -----
   Does multiple of 4 outputs in parallel. This code takes the pointer to where the valid 
   data begins.

   Weights are reformed by transpoing so are sequential in memory.

   <------------------------------------------ filt_height = 3 * indepth/32 --------------->
   <------------------------- filt width = 3 ------------------------------------><---><--->
   <---- 32 depth 4 each----><---- 32 depth 4 each----><---- 32 depth 4 each-----><><><><><>
*/
#include <hexagon_protos.h>

void gvconv2dbbb_cn(
     uint8_t * in_buf,    //input activations - aligned to 8bytes
     uint8_t * weights,   //8bit unsigned weights - aligned to 128bytes
     uint8_t * out_buf,   //quantized output - 
     int next_in_width,   //total physical width in depths
     int next_out_width,  //total size of full line out_depth* total out_width
     int out_width,       //amount of work to do 
     int stride_height,   //striding x
     int stride_width,    //striding y
     int in_depth,        //input depth total
     int filt_width,      //2d filter
     int filt_height,     //-sizes
     int out_height,      //number of lines to compute
     int * bias,          //activation bias for accumulators
     int * ptr_minmax,    //ptr for tracking min and max
     int recip,           //1/max for quantization
     int recip_shift,     //shift sum for reciprocal adjustment
     int in_offset,       //activation zero
     int filt_offset      //weight zero
) {
   int out_y, out_x, i;
   int64_t lsum;
   int num_depth32 = in_depth/32;
   int max = -0x7fffffff, min = 0x7fffffff;
   int out_z, in_z, filt_y, filt_x, filt_z;
   int w0, d[4], sum[4]; //4 accumulators in parallele

        for (out_y = 0; out_y < out_height; out_y++)
        {
          for (out_x = 0; out_x < out_width; out_x+=4) //will overrun into out of sample data
          {
            for (out_z = 0; out_z < 32; out_z++)
            {
              //initialize bias
              for(i=0;i<4;i++)sum[i] = bias[out_z];

              //2d filter
              for (filt_y = 0; filt_y < filt_height*num_depth32; filt_y++)
              {
                for (filt_x = 0; filt_x < filt_width; filt_x++)
                {
                  //perform the filtering across 4 accumulators and long the depth
                  for (in_z = 0; in_z < 32; in_z++)
                  {
                    //addressing scheme for transposed weight array
                    filt_z = 32*(filt_width*filt_y + filt_x) + in_z; 
                    w0 = weights[128*(filt_z/4) + filt_z % 4 + 4*out_z];
                    w0 -= filt_offset;   //subtract "zero" 
                    for(i=0;i<4;i++) {
                      d[i] = in_buf[32*((stride_height*out_y*num_depth32+filt_y)*next_in_width+stride_width*(out_x+i)+filt_x) + in_z];
                      d[i] -= in_offset; //subtract zero
                      sum[i] += d[i] * w0;
                    }
                  }//in_z
                }//filt x
              }//filt_y

              //monitor max and min and quantize the 4 accumulations
              for(i=0;i<4;i++) {
                if(sum[i] > max) max = sum[i];
                if(sum[i] < min) min = sum[i];

                sum[i] <<= recip_shift;  //deal with adjusted recip
                lsum = ((int64_t) sum[i]) * ((int64_t) recip) + 0x40000000LL;
                sum[i] = lsum >> 31;
                if(sum[i] <   0) sum[i] = 0;
                if(sum[i] > 255) sum[i] = 255;
                out_buf[32*(out_x+i)+ next_out_width*out_y + out_z] = (uint8_t) sum[i];
              }
            }//out_z
          }//out_x
        }//out_y
        ptr_minmax[0] = max;
        ptr_minmax[1] = min;
    return;
}

/* ----------------------------------------------------------------------------------------- */
