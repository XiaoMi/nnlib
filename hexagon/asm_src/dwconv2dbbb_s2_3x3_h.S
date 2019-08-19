/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
 *  FUNCTIONS      : gvconv2dbbb_s2_3x3_asm
 *
 *  DESCRIPTION
 *   Perform 2d depthwise convolution using elements along depth, do only simple
 *   convolution. THis is optimized for case of 3x3 filter horizontal stride of 2.
 *   Sums are scaled and saturated to 8bits. Max and Min accumulations are kept.
 *
 *  ARCHITECTURE   : QDSP6V6x  + HVX
 *
 *  REVISION HISTORY:
 *  =================
 *
 *
 *  Author              Date           Comments
 *  -------------------------------------------------------------
 *  DJH                 06/26/19       created
 *
 *  MEMORY
 *     CODESIZE = 768 bytes
 *     STACK    = 64 bytes
 *     ASSUMPTIONS
 *      width multiple of 4 depth multiple of 32 aligned to 32bytes
 *  BIT ACCURATE C MODEL
 */
#if 0
    void dwconv2dbbb_s2_3x3_cn(
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
       int stride_height)
    {
       int out_y, d, out_x, ur, in_val, filt_val;
       int out_z, filt_y, filt_x, buf_offset;
       int out_width_pad = (out_width+3)&(~3);
       int32_t sum, zum;
       int64_t lsum ;
       int o_filt_width = (filt_width+3)&(~3);
       int stride_width = 2;
    
        for (out_y = 0; out_y < out_height; out_y++) {
            for (out_x = 0; out_x < out_width_pad; out_x+=4) {
                for(d=0; d < depth/32; d++) {
                   for (out_z = 0; out_z < 32; out_z++) {
                      for(ur=0; ur < 4; ur++) {
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
#endif
/* ----------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dbbb_s2_3x3_h.S"
       .global dwconv2dbbb_s2_3x3_asm
       .balign 32
       .type  dwconv2dbbb_s2_3x3_asm, @function
dwconv2dbbb_s2_3x3_asm:
/* ----------------------------------------------------------------------------- */
//io registers
#define   in_buf               r0
#define   filt                 r1
#define   out_buf              r2
#define   next_in_width_depth  r3
#define   next_out_width_depth r4       //currently unused
#define   next_in_width_32     r5
#define   next_out_width_32    r6 
#define   depth                r7 
#define   out_width            r8 
#define   out_height           r9 
#define   filt_height          r6        //nc
#define   filt_zero            r10
#define   bias_sum             r11
#define   ptr_max              r12
#define   recip_level          r13
#define   recip_shift          r14
#define   stride_v             r28  
#define   in_left_skip         r22  
//state scaler registers
#define   ptr_w0               r15
#define   bias_ptr             r16
#define   ptr_x0               r17
#define   ptr_x1               r18
#define   ptr_x2               r19
#define   ptr_xin              r20
#define   ptr_y                r21
#define   next_in_width_depth_stride r28
#define   zzzz                 r10       //111111111111
#define   _zzz                 r10       //111111111111
//vector registers
#define   vrecip                v0
#define   max                   v1
#define   min                   v2
#define   bias_val              v3 
#define   x0                    v4       //___11_______
#define   x1                    v5       //_111________
#define   x2                    v13      //_111________
#define   w0_210                v6       //_______111__
#define   w1_210                v7       //_______111__
#define   w2_210                v8       //_______111__
#define   x0_ba98               v9       //__________11
#define   x0_3210              v10       //111111111111
#define   x1_ba98              v11       //__________11
#define   x1_3210              v12       //111111111111
#define   x2_ba98              v13       //__________11
#define   x2_3210              v14       //111111111111
#define   x0_7654              v30       //111111111111
#define   x0_3232              v27        //111111111111
#define   x0_5432              v27        //111111111111
#define   x0_7676              v28        //111111111111
#define   x0_9876              v29        //111111111111
#define   x1_7654              v30       //111111111111
#define   x1_3232              v26        //111111111111
#define   x1_5432              v27        //111111111111
#define   x1_7676              v29        //111111111111
#define   x1_9876              v29        //111111111111
#define   x2_7654              v30       //111111111111
#define   x2_3232              v26        //111111111111
#define   x2_5432              v27        //111111111111
#define   x2_7676              v28        //111111111111
#define   x2_9876              v29        //111111111111
#define   s0                   v15       //111111111111
#define   s1                   v16       //111111111111
#define   s2                   v17       //111111111111
#define   s3                   v18       //111111111111
#define   z0                   v19       //111111111111
#define   z1                   v20       //111111111111
#define   z2                   v21       //111111111111
#define   z3                   v22       //111111111111
#define   d0                   v23       //
#define   d1                   v24       //
#define   d1d0                 v23       //
#define   d2                   v25       //
#define   d3                   v26       //
#define   d3d2                 v25       //
#define   d3210                v25       //
#define SSR        <<1:rnd:sat:shift     //simplfy mpy instruction
#define PV(VSRC) .word (0x1DFFE020+VSRC)
#define PS(SSRC) .word (0x1DFFE100+SSRC)   //debug sca reg
/* =========================================================================== */
   {  allocframe(#56)                               //0th entry on sbuf (56+8)/4=20
   } {
      memd(sp+#0)  = r17:16
      memd(sp+#8)  = r19:18
   } {
      memd(sp+#16) = r21:20
      memd(sp+#24) = r23:22
   } {
      next_out_width_32 = memw(sp+#16<<2)           //
      depth  = memw(sp+#17<<2)                      //
      p2 = !cmp.eq(r0,r0)                           //
   } {
      out_width = memw(sp+#18<<2)                   //
      out_height = memw(sp+#19<<2)                  //
      depth = lsr(depth, #5)                        //depth/32
      ptr_y = out_buf                               //
   } {
      filt_zero = memw(sp+#21<<2)                   //
      bias_sum = memw(sp+#22<<2)                    //
      loop1(.L_depth, depth)
      out_width = add(out_width, #3)
   } {
      ptr_max = memw(sp+#23<<2)                     //
      recip_level = memw(sp+#24<<2)                 //
      zzzz  = vsplatb(filt_zero)
      out_width = lsr(out_width, #2)
   } {
      recip_shift = memw(sp+#25<<2)                 //
      max = vmem(ptr_max+#0)
      ptr_xin = in_buf
   } {
      stride_v = memw(sp+#26<<2)                    //
      in_left_skip= memw(sp+#28<<2)                 //
      _zzz  = lsr(zzzz, #8)
      bias_ptr = bias_sum                           //
   } {
      next_in_width_depth_stride =                  \
                 mpyi(next_in_width_depth, stride_v)//
      min = vmem(ptr_max+#1)
      _zzz  = asl(_zzz, in_left_skip)
      ptr_w0 = filt
   } {
      vrecip = vmem(recip_level)
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
.L_depth:
   {  x0.tmp = vmem (ptr_xin+#0):nt                 //
      ptr_x0 = add(ptr_xin, #128)                   //
      x0.b = vshuff(x0.b)
      ptr_x1 = add(ptr_xin, next_in_width_depth)    //[0, 4]
   } {
      x1.tmp = vmem (ptr_x1++#1)
      x1.b = vshuff(x1.b)
      ptr_x2 = add(ptr_x1, next_in_width_depth)     //[0, 4]
   } {
      x2.tmp = vmem (ptr_x2++#1)
      x2.b = vshuff(x2.b)
   } {
      w0_210.tmp = vmem(ptr_w0++#1)
      w0_210.w = vasl(w0_210.w, in_left_skip)
      x0_3210.b = vshuff(x0.b)
   } {
      x0.tmp = vmem (ptr_x0++#1):nt                 //[0, 3]
      x0.b = vshuff(x0.b)                           //[0, 3]
   } {
      w1_210.tmp = vmem(ptr_w0++#1)
      w1_210.w = vasl(w1_210.w, in_left_skip)
      x1_3210.b = vshuff(x1.b)
      x0_3232.h = vshuffo(x0_3210.h, x0_3210.h)     //[0, 5]
   } {
      x0_7654.b = vshuff(x0.b)                      //[0, 5]
      p3 = sp1loop0(.L_width, out_width)
      bias_val = vmem(bias_ptr++#1)                 //
      ptr_xin = add(ptr_xin, next_in_width_32)      //[DEPTH]
   } {
      x1.tmp = vmem (ptr_x0++#1)                    //[0, 4]
      x1.b = vshuff(x1.b)                           //[0, 4]
      x0_5432.h = vshuffe(x0_7654.h,x0_3232.h)      //[0, 7]
   } {
      w2_210.tmp = vmem(ptr_w0++#1)
      w2_210.w = vasl(w2_210.w, in_left_skip)
      x2_3210.b = vshuff(x2.b)
      s0.uw  = vrmpy(x0_3210.ub, w0_210.ub)         //[0, 6]filter even output
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
   {  s2.uw  = vrmpy(x0_7654.ub, w0_210.ub)         //[0, 8]filter even output
      x0_ba98.b = vshuff(x1.b)                      //[0, 6]
      x0_7676.h = vshuffo(x0_7654.h, x0_7654.h)     //[0, 7]
   } {
      z0.uw  = vrmpy(x0_3210.ub, _zzz.ub)           //[0, 7]filter even output
      z1.uw  = vrmpy(x0_5432.ub, _zzz.ub)           //[0,10]filter even output
      x0_9876.h = vshuffe(x0_ba98.h,x0_7676.h)      //[0, 8]
      d3d2.h = vpack(d3.w, d2.w):sat                //[WIDTH, E]
   } {
      s1.uw  = vrmpy(x0_5432.ub, w0_210.ub)         //[0, 9]filter even output
      x0.tmp = vmem (ptr_x1++#1)                    //[0, 8]
      x0.b = vshuff(x0.b)                           //[0, 8]
   } {
      z2.uw  = vrmpy(x0_7654.ub, _zzz.ub)           //[0, 7]filter even output
      z3.uw  = vrmpy(x0_9876.ub, _zzz.ub)           //[0,10]filter even output
      x1.tmp = vmem(ptr_x1++#1)                     //[0, 9]
      x1.b = vshuff(x1.b)                           //[0, 9]
   } {
      s3.uw  = vrmpy(x0_9876.ub, w0_210.ub)         //[0,11]filter even output
      x1_7654.b = vshuff(x0.b)                      //[0,10]
      x1_3232.h = vshuffo(x1_3210.h, x1_3210.h)     //[0,10]
   } {
      s0.uw += vrmpy(x1_3210.ub, w1_210.ub)         //[0,12]filter even output
      x1_5432.h = vshuffe(x1_7654.h,x1_3232.h)      //[0,12]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat          //[WIDTH, E]
      if(p2) vmem (ptr_y++#1) = d3210.new           //[WIDTH, E]
   } {
      z0.uw += vrmpy(x1_3210.ub, _zzz.ub)           //[0,13]filter even output
      z2.uw += vrmpy(x1_7654.ub, _zzz.ub)           //[0,13]filter even output
      x1_ba98.b = vshuff(x1.b)                      //[0,11]
      x1_7676.h = vshuffo(x1_7654.h, x1_7654.h)     //[0,11]
   } {
      s2.uw += vrmpy(x1_7654.ub, w1_210.ub)         //[0,14]filter even output
      x1_9876.h = vshuffe(x1_ba98.h,x1_7676.h)      //[0,14]
      x0_3210 = x0_ba98                             //[0, 9]
      if(!p3)ptr_y = out_buf                        //
   } {
      s1.uw += vrmpy(x1_5432.ub, w1_210.ub)         //[0,15]filter even output
      x1_3210 = x1_ba98                             //[0,15]
      x0.tmp = vmem (ptr_x2++#1)                    //[0,14]
      x0.b = vshuff(x0.b)                           //[0,14]
   } {
      s3.uw += vrmpy(x1_9876.ub, w1_210.ub)         //[0,17]filter even output
      x1.tmp = vmem (ptr_x2++#1)                    //[0,15]
      x1.b = vshuff(x1.b)                           //[0,15]
      x2_3210 = x2_ba98                             //[0,21]
   } {
      z1.uw += vrmpy(x1_5432.ub, _zzz.ub)           //[0,16]filter even output
      z3.uw += vrmpy(x1_9876.ub, _zzz.ub)           //[0,16]filter even output
      x2_7654.b = vshuff(x0.b)                      //[0,16]
      x2_3232.h = vshuffo(x2_3210.h, x2_3210.h)     //[0,16]
   } {
      s0.uw += vrmpy(x2_3210.ub, w2_210.ub)         //[0,18]filter even output
      x2_5432.h = vshuffe(x2_7654.h,x2_3232.h)      //[0,18]
      x2_7676.h = vshuffo(x2_7654.h, x2_7654.h)     //[0,17]
      if(!p3)out_buf=add(out_buf,next_out_width_32) //[DEPTH]
   } {
      z0.uw += vrmpy(x2_3210.ub, _zzz.ub)           //[0,19]filter even output
      z2.uw += vrmpy(x2_7654.ub, _zzz.ub)           //[0,19]filter even output
      s0.w = vadd(s0.w, bias_val.w)                 //[WIDTH, P]
      x2_ba98.b = vshuff(x1.b)                      //[0,17]
   } {
      s2.uw += vrmpy(x2_7654.ub, w2_210.ub)         //[0,20]filter even output
      x2_9876.h = vshuffe(x2_ba98.h,x2_7676.h)      //[0,20]
      s0.w = vsub(s0.w, z0.w)                       //
      p2 = cmp.eq(r0,r0)                            //
   } {
      s1.uw += vrmpy(x2_5432.ub, w2_210.ub)         //[0,21]filter even output
      s2.w = vadd(s2.w, bias_val.w)                 //[WIDTH, P]
      x0.tmp = vmem (ptr_x0++#1):nt                 //[1, 3]
      x0.b = vshuff(x0.b)                           //[1, 3]
   } {
      z1.uw += vrmpy(x2_5432.ub, _zzz.ub)           //[0,22]filter even output
      z3.uw += vrmpy(x2_9876.ub, _zzz.ub)           //[0,22]filter even output
      s0.w = vasl(s0.w, recip_shift)                //
      s2.w = vsub(s2.w, z2.w)                       //
   } {
      s3.uw += vrmpy(x2_9876.ub, w2_210.ub)         //[0,23]filter even output
      s1.w = vadd(s1.w, bias_val.w)                 //[WIDTH, P]
      x1.tmp = vmem (ptr_x0++#1):nt                 //[1, 4]
      x1.b = vshuff(x1.b)                           //[1, 4]
   } {
      d0.w = vmpye(s0.w, vrecip.uh)                 //[0,15]multiply by 1/max
      s1.w = vsub(s1.w, z1.w)                       //
      s2.w = vasl(s2.w, recip_shift)                //
   } {
      d0.w += vmpyo(s0.w, vrecip.h):SSR             //[0,17]3
      s3.w = vadd(s3.w, bias_val.w)                 //[WIDTH, P]
      x0_3232.h = vshuffo(x0_3210.h, x0_3210.h)     //[1, 5]
   } {
      d2.w = vmpye(s2.w, vrecip.uh)                 //[0,15]multiply by 1/max
      s3.w = vsub(s3.w, z3.w)                       //
      s1.w = vasl(s1.w, recip_shift)                //
   } {
      d2.w += vmpyo(s2.w, vrecip.h):SSR             //[0,17]3
      min.w = vmin(min.w, d0.w)                     //[0,22]8               //0+2+1
      max.w = vmax(max.w, d0.w)                     //[0,18]5               //0+2+1
   } {
      d1.w = vmpye(s1.w, vrecip.uh)                 //[0,22]multiply by 1/max
      min.w = vmin(min.w, d2.w)                     //[0,22]8               //0+2+1
      s3.w = vasl(s3.w, recip_shift)                //
   } {
      d1.w += vmpyo(s1.w, vrecip.h):SSR             //[0,23]9
      max.w = vmax(max.w, d2.w)                     //[0,18]5               //0+2+1
      x0_7654.b = vshuff(x0.b)                      //[1, 5]
   } {
      min.w = vmin(min.w, d1.w)                     //[0,27]13              //0+2+1
      max.w = vmax(max.w, d1.w)                     //[0,26]12              //0+2+1
      d3.w = vmpye(s3.w, vrecip.uh)                 //[0,22]multiply by 1/max
   } {
      d1d0.h = vpack(d1.w, d0.w):sat                //[0,27]
      d3.w += vmpyo(s3.w, vrecip.h):SSR             //[0,23]9
      x0_5432.h = vshuffe(x0_7654.h,x0_3232.h)      //[1, 7]
   } {
      max.w = vmax(max.w, d3.w)                     //[0,26]12              //0+2+1
      min.w = vmin(min.w, d3.w)                     //[0,27]13              //0+2+1
      s0.uw  = vrmpy(x0_3210.ub, w0_210.ub)         //[1, 6]filter even output
   }:endloop0:endloop1 //end width, depth
/* ----------------------------------------------------------------------------- */
   {  d3d2.h = vpack(d3.w, d2.w):sat                //[WIDTH, E]
      out_height = add(out_height, #-1)             //
      p2 = !cmp.eq(r0,r0)                           //
      bias_ptr = bias_sum                           //
   } {
      p0 = cmp.eq(out_height, #0)                   //
      in_buf=add(in_buf,next_in_width_depth_stride) //stride 
      lc1 = depth                                   //loop1(.L_depth, depth)   
      ptr_w0 = filt                                 //
   } {
      ptr_xin = in_buf                              //
      d3210.ub = vpack(d3d2.h, d1d0.h):sat          //[WIDTH, E]
      vmem (ptr_y+#0) = d3210.new                   //[WIDTH, E]
      if(!p0) jump .L_height                        //
   }//end height
/* ----------------------------------------------------------------------------- */
   {  r17:16 = memd(sp+#0)                          //restore
      vmem(ptr_max+#0) = max                        //
   } {
      r19:18 = memd(sp+#8)                          //restore
      vmem(ptr_max+#1) = min                        //
   } {
      r21:20 = memd(sp+#16)                         //restore
      r23:22 = memd(sp+#24)                         //restore
   } {
      dealloc_return                                //return
   }
/* ----------------------------------------------------------------------------- */
.L_end:
      .size dwconv2dbbb_s2_3x3_asm, .L_end-dwconv2dbbb_s2_3x3_asm
/* ----------------------------------------------------------------------------- */