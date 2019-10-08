/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a reference implementation for depthwise conv with a Nx3 kernel and a stride width of 2.
 */

#include <nn_graph.h>
#include <hexagon_protos.h>
#include <stdio.h>

void dwconv2dbbb_s2_3xN_cn(
   const uint8_t *in_buf, 
   const uint8_t  *filt, 
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   int32_t recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * sbuf) 
{
   int out_y, d, out_x, ur, in_val, filt_val;
   int out_z, filt_y, filt_x, cnt;
   int out_width_pad = (out_width+3)&(~3);
   int32_t sum, zum, sum0;
   int64_t lsum ;
   int filt_width = 3;
   int o_filt_width = (filt_width+3)&(~3);
   int buf_offset;
   int stride_width = 2;

    for (out_y = 0; out_y < out_height; out_y++) {
        cnt = out_width;
        for (out_x = 0; out_x < out_width_pad; out_x+=4) {
            cnt -= 4;
            for(d=0; d < depth/32; d++) {
               for (out_z = 0; out_z < 32; out_z++) {
                  for(ur=0; ur < 4; ur++)
                  {
                    sum = (int32_t)bias_sum[32*d+out_z];
                    zum = 0;
                    for (filt_y = 0; filt_y < filt_height; filt_y++) {
                       for (filt_x = 0; filt_x < o_filt_width; filt_x++) {
                          buf_offset = (out_y * stride_height +  filt_y) * next_in_width 
                                      + d * next_in_width_32 
                                      + (out_x*stride_width + ur*stride_width + filt_x) * 32 
                                      + out_z;
                          in_val = in_buf[buf_offset]; 

                          filt_val = filt[32*d*filt_height*o_filt_width
                                           + (o_filt_width*filt_y)*32 
                                           + out_z*4 + 128*(filt_x/4)
                                           + (filt_x % 4)] ;

                          sum += (uint32_t)in_val*(int32_t)filt_val;
                          if(filt_x < filt_width)
                             zum += (uint32_t)in_val*(int32_t)filt_zero;
                       }
                    }
                    sum = sum - zum;
                    if(ur==0) sum0 = sum;
                    if(ur == 1 && !(cnt > -3)) sum = sum0; 
                    if(ur == 2 && !(cnt > -2)) sum = sum0; 
                    if(ur == 3 && !(cnt > -1)) sum = sum0; 

                    sum <<= recip_shift;
                    lsum = (int64_t)sum * ((int64_t)recip_level) + 0x40000000LL;
                    lsum = lsum >> 31;
                    sum = (int)lsum;
                    max[out_z]    = (sum > max[out_z]) ? sum : max[out_z];
                    max[out_z+32] = (sum < max[out_z+32]) ? sum : max[out_z+32];
                    if(lsum < 0) lsum = 0; if(lsum > 0xffll) lsum = 0xffll;
                    out_buf[out_y * next_out_width 
                            + 32 * (out_x+ur) 
                            + d * next_out_width_32 
                            + out_z] = (uint8_t) lsum;
                  }//ur
               }//out_z
            }//d
        }//out_x
    }//out_y
    return;
}
