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
 
#include <nn_graph.h>
#include <hexagon_protos.h>
#include <stdio.h>

#if 0
void dwconv2d_cn(
   uint8_t *in_buf, int in_width, int in_height, int depth, int stride_width, int stride_height, int in_offset,
   int8_t  *filt,   int filt_width, int filt_height,    
   int32_t  *out_buf, int out_width, int out_height, int adj_x, int adj_y)
{
   int out_y, in_y_base, out_x, in_x_base;
   int out_z, filt_y, filt_x, in_element, filt_element, sum;
   int * outstripe;
   uint8_t * instripe;
   int8_t * filtstripe;

         for (out_y = 0; out_y < out_height; out_y++) {
            in_y_base = out_y * stride_height - adj_y;
            for (out_x = 0; out_x < out_width; out_x++) {
               in_x_base = out_x * stride_width - adj_x;
               outstripe = out_buf+(depth*(out_x+ out_width*out_y));
               for (out_z = 0; out_z < depth; out_z++) {
                  sum = 0;
                  for (filt_y = 0; filt_y < filt_height; filt_y++) {
                     if ((in_y_base + filt_y) >= in_height) continue;
                     if ((in_y_base + filt_y) < 0) continue;
                     for (filt_x = 0; filt_x < filt_width; filt_x++) {
                        if ((in_x_base + filt_x) >= in_width) continue;
                        if ((in_x_base + filt_x) < 0) continue;

                        filtstripe = filt+(depth*(filt_x+ filt_width*filt_y));
                        filt_element = filtstripe[out_z];

                        instripe = in_buf+(depth*(in_x_base + filt_x + in_width*(in_y_base + filt_y))) ;
                        in_element = instripe[out_z] - in_offset;

                        sum += in_element*filt_element;
                     }
                  }
                  outstripe[out_z] = sum;
               }
            }
         }
    return;
}
void biasadd_requant_cn(
        uint8_t *out,
        int     *tmp_out,
        int     *biasbuf,
        int     num_patches,
        int     depth,
        int     *fixed_recip_level_size,
        int * max, int * min)
{
        int32_t sum;
        int64_t loutval;
        int32_t outval;
        int32_t i,j, s;
        uint32_t recip_val;
        max[0] = -0x7fffffff; //max
        min[0] =  0x7fffffff; //min
        for (j = 0; j < num_patches; j++) {
                for (i = 0; i < depth; i++) {
                        sum = biasbuf[i] + tmp_out[j*depth+i];
                        if(sum > max[0] )max[0] = sum;
                        if(sum < min[0] )min[0] = sum;
                        tmp_out[j*depth+i] = sum;
                }
        }
        *fixed_recip_level_size = (int) (0x1fff80000000LL / (long long int) (max[0]-min[0]));
        recip_val = *fixed_recip_level_size;
        for (j = 0; j < num_patches; j++) {
                for (i = 0; i < depth; i++) {
                    sum = tmp_out[j*depth+i] - min[0];

                    loutval = (int64_t)sum * (int64_t)*fixed_recip_level_size + 0x40000000LL;

                    outval = loutval >> 31;

                    if (outval < 0) outval = 0;
                    if (outval > 16383) outval = 16383;
                    *out++ = outval;
                }
        }
}
#endif

void dwconv2dbbb_MxN_cn(
   uint8_t *in_buf, 
   uint8_t  *filt, 
   uint8_t  *out_buf,
   int next_in_width,
   int next_out_width,
   int next_in_width_32,
   int next_out_width_32,
   int depth,
   int out_width,
   int out_height,
   int filt_width,
   int filt_height,
   int filt_zero,
   int32_t *bias_sum,
   int32_t *max,
   int recip_level,
   int recip_shift,
   int stride_width, 
   int stride_height) 
{
   int out_y, d, out_x, ur, in_val, filt_val;
   int out_z, filt_y, filt_x, cnt;
   int out_width_pad = (out_width+3)&(~3);
   int32_t sum, zum, sum0;
   int64_t lsum ;
   int o_filt_width = (filt_width+3)&(~3);
   int buf_offset;

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
