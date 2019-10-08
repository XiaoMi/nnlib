/*
 * Copyright (c) 2016,2017,2018 The Linux Foundation. All rights reserved.
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
/*======================================================================*/
/*  FUNCTIONS      : dwconv2dbbb_s2_5xN_asm                             */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Depthwise filter stride 2xM, filter size 5xN         */
/*                 input and output ptr non aligned output width        */
/*                 padded, max and min found only on valid range        */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 4.30.19                                         */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->                                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =   880 bytes                                           */
/*     STACK    =   48  bytes                                           */
/*     ASSUMPTIONS                                                      */
/*        none                                                          */
/*  C MODEL                                                             */
/*     exact match to assembly code                                     */
/*======================================================================*/
#if 0
void dwconv2dhhh_s2_cn(
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
   int filt_height,
   int filt_zero,
   int32_t *bias_sum,
   int32_t *max,
   int recip_level,
   int recip_shift,
   int stride_height)
{
   int out_y, d, out_x, ur, in_val, filt_val;
   int out_z, filt_y, filt_x, cnt;
   int out_width_pad = (out_width+3)&(~3);
   int64_t lsum;
   int32_t sum, zum, sum0;
   int filt_width = 5;
   int o_filt_width = (filt_width+3)&(~3);

    for (out_y = 0; out_y < out_height; out_y++) {
        cnt = out_width;
        for (out_x = 0; out_x < out_width_pad; out_x+=4) {
            cnt -= 4;
            for(d=0; d < depth/32; d++) {
               for (out_z = 0; out_z < 32; out_z++) {
                  for(ur=0; ur < 4; ur++)
                  {
                    zum = 0;
                    sum  = bias_sum[32*d+out_z];
                    for (filt_y = 0; filt_y < filt_height; filt_y++) {
                       for (filt_x = 0; filt_x < o_filt_width; filt_x++) {
                          in_val = in_buf[(out_y*stride_height +  filt_y) * next_in_width
                                          + d * next_in_width_32
                                          + (out_x*2 + ur*2 + filt_x) * 32
                                          + out_z];
                          filt_val = filt[32*d*filt_height*o_filt_width +
                                          (o_filt_width*filt_y)*32 +
                                          out_z*4 + 128*(filt_x/4) + (filt_x & 3)] ;
                          sum += in_val*filt_val;
                          if(filt_x < filt_width)
                              zum += in_val*filt_zero;
                       }
                    }
                    sum = sum - zum;
                    if(ur==0)sum0 = sum;
                    if(ur == 1 && !(cnt > -3)) sum = sum0;
                    if(ur == 2 && !(cnt > -2)) sum = sum0;
                    if(ur == 3 && !(cnt > -1)) sum = sum0;

                    sum <<= recip_shift;
                    lsum = (int64_t)sum * (int64_t)recip_level + 0x40000000LL;
                    lsum = lsum >> 31;
                    sum = (int) lsum;
                    if(sum > max[out_z   ]) max[out_z   ] = sum;
                    if(sum < max[out_z+32]) max[out_z+32] = sum;
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
       .text
       .file "dwconv2dbbb_s2_5xN_h.S"
       .global dwconv2dbbb_s2_5xN_asm
       .balign 32
       .type  dwconv2dbbb_s2_5xN_asm, @function
dwconv2dbbb_s2_5xN_asm:
/* ----------------------------------------------------------------------------- */
//input output registers
#define   in_buf               r0        //
#define   filt                 r1        //
#define   out_buf              r2        //
#define   next_in_width_depth  r3        //
#define   next_out_width_depth r4        //currently unused
#define   next_in_width_32     r5        //
#define   next_out_width_32    r10       //
#define   depth                r11       //
#define   out_width            r12       //
#define   out_height           r13       //
#define   filt_height          r25       //
#define   filt_zero            r7        //
#define   bias_sum             r14       //
#define   ptr_max              r15       //
#define   recip_level          r10       //
#define   recip_shift          r8        //
#define   stride_v             r28       //
//scaler registers
#define   ptr_w0               r16       //
#define   ptr_w1               r17       //
#define   col_count            r9        //
#define   c16                  r6        //
#define   bias_ptr             r18       //
#define   ptr_x0               r19       //
#define   ptr_x1               r20       //
#define   ptr_xin              r22       //
#define   ptr_y                r23       //
#define   depth_cnt            r26       //
#define   filt_size            r11       //
#define   next_in_width_depth_stride r28 //
#define   zzzz                 r7        //
#define   ___z                 r27       //
#define   _zzz                 r21       //
#define   zzz_                 r4        //
#define   zz__                 r15       //
#define   out_width4           r24       //
//vector registers
#define   vrecip                v0       //
#define   vshamt_vec            v1       //
#define   max                   v2       //
#define   min                   v3       //
#define   bias_val              v4       //
#define   w_432                v27       //
#define   w___4                v8        //
#define   w3210                v28       //
#define   w10__                v7        //
#define   x0                   v30       //
#define   x1                   v29       //
#define   x2                   v5        //
#define   x3x1x2x0             v30       //
#define   x3x2x1x0             v30       //
#define   x7x5x6x4             v29       //
#define   x7x6x5x4             v10       //
#define   xbxax9x8             v5        //
#define   xbx9xax8             v5        //
#define   s0                   v12       //
#define   s1                   v13       //
#define   s2                   v14       //
#define   s3                   v15       //
#define   z0                   v16       //
#define   z1                   v17       //
#define   z2                   v18       //
#define   z3                   v19       //
#define   d0                   v20       //
#define   d1                   v21       //
#define   d1d0                 v24       //
#define   d2                   v22       //
#define   d3                   v23       //
#define   d3d2                 v25       //
#define   d3210                v25       //
#define SSR        <<1:rnd:sat:shift     //simplfy mpy instruction
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca reg

/* =========================================================================== */
   {  allocframe(#56)                              //0th entry on stack (56+8)/4=20
      c16= #16                                     //
      out_width = memw(sp+#2<<2)                   //
   } {
      M0 = next_in_width_depth                     //
      memd(sp+#32) = r25:24                        //
      memd(sp+#40) = r27:26                        //
      out_width4 = add(out_width, #3)              //
   } {
      memd(sp+#16) = r21:20                        //
      memd(sp+#24) = r23:22                        //
      out_width4 = lsr(out_width4, #2)             //
   } {
      memd(sp+#0)  = r17:16                        //
      memd(sp+#8)  = r19:18                        //
   } {
      stride_v = memw(sp+#26<<2)                   //
      filt_zero = memw(sp+#21<<2)                  //
   } {
      zzzz  = vsplatb(filt_zero)                   //
      next_out_width_32 = memw(sp+#16<<2)          //
   } {
      next_in_width_depth_stride = mpyi(next_in_width_depth,stride_v) //
      _zzz  = lsr(zzzz, #8)                        //
      depth  = memw(sp+#17<<2)                     //
   } {
      out_height = memw(sp+#19<<2)                 //
      ptr_max = memw(sp+#23<<2)                    //
      depth = lsr(depth, #5)                       //depth/32
   } {
      memw(sp+#17<<2) = depth                      //
      max = vmem(ptr_max+#0)                       //
      ___z  = lsr(_zzz, #16)                       //
   } {
      min = vmem(ptr_max+#1)                       //
      zz__  = asl(zzzz, #16)                       //
   } {
      filt_size = memw(sp+#20<<2)                  //
      recip_shift = memw(sp+#25<<2)                //
   } {
      filt_height = add(filt_size, #-1)            //
      bias_sum = memw(sp+#22<<2)                   //
      depth_cnt = memw(sp+#17<<2)                  //depth
      vshamt_vec= vsplat(recip_shift)              //
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {
      recip_level = memw(sp+#24<<2)
   }
   {  bias_ptr = bias_sum                          //
      ptr_xin = in_buf                             //
      ptr_w0 = filt                                //
      loop1(.L_width, out_width4)                  //
   } {
      out_height = add(out_height, #-1)            //
      col_count = out_width                        //
      x2 = vmemu(ptr_xin+#2)                       //[0, 0]
   }
/* ----------------------------------------------------------------------------- */
.L_depth:
   {
      vrecip = vmem(recip_level++#1)
      ptr_x0 = ptr_xin                             //
      ptr_y = out_buf                              //
      memw(sp+#52) = recip_level
   } {  
      x1 = vmemu(ptr_xin+#1)                       //[0, 1]
   } {
      xbx9xax8.b = vshuff(x2.b)                    //[0, 2]
      bias_val = vmem(bias_ptr++#1)                //
      p3 = !cmp.eq(r0, r0)                         //
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
   {  w3210.cur = vmem(ptr_w0+#0)                  //[0, 3]
      w_432.uw = vlsr(w3210.uw, c16)               //[0, 5]
      xbxax9x8.b = vshuff(xbx9xax8.b)              //[0, 5]
   } {
      w___4.cur     = vmem(ptr_w0+#1)              //[0, 4]
      w_432.w += vasl(w___4.w, c16)                //[0, 6]
      x7x5x6x4.b = vshuff(x1.b)                    //[0, 6]
      ptr_x1 = ptr_x0                              //[WIDTH, P]
   } {
      z2.uw  = vrmpy(xbxax9x8.ub, ___z.ub)         //[0, 7]
      z3.uw  = vrmpy(xbxax9x8.ub, _zzz.ub)         //[0, 7]
      x0 = vmemu(ptr_x1++M0)                       //[0, 7]
   } {
      s3.uw  = vrmpy(xbxax9x8.ub, w_432.ub)        //[0, 8]
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[0, 8]
      ptr_x0 = add(ptr_x0, #256)                   //[WIDTH,P]+8 32 depths for stride 2
      col_count = add(col_count, #-4)              //[WIDTH]
   } {
      s2.uw  = vrmpy(xbxax9x8.ub, w___4.ub)        //[0, 9]
      x3x1x2x0.b = vshuff(x0.b)                    //[0, 9]
      s0 = bias_val                                //[WIDTH, P]
      s3.w = vadd(s3.w, bias_val.w)                //[WIDTH, P]
   } {
      s2.w = vadd(s2.w, bias_val.w)                //[WIDTH, P]
      s0.uw += vrmpy(x7x6x5x4.ub, w___4.ub)        //[0,10]
      loop0(.L_vert, filt_height)                  //[WIDTH, P]
      ptr_w1 = add(ptr_w0, #256)                   //[WIDTH, P]
   } {
      z0.uw = vrmpy(x7x6x5x4.ub, ___z.ub)          //[0,11]
      z1.uw = vrmpy(x7x6x5x4.ub, _zzz.ub)          //[0,11]
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[0,11]
      s1 = bias_val                                //[WIDTH, P]
   } 
/* --------------------------------------------------------------------------- */
      .balign 32
.L_vert:
   {  s2.uw += vrmpy(x7x6x5x4.ub, w3210.ub)        //[0,12]
      x2 = vmemu(ptr_x1+#2)                        //[1, 0]
   } {
      s1.uw += vrmpy(x7x6x5x4.ub, w_432.ub)        //[0,13]
      x1 = vmemu(ptr_x1+#1)                        //[1, 1]
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w3210.ub)        //[0,14]
      w10__.w = vasl(w3210.w, c16)                 //[0,14]
      xbx9xax8.b = vshuff(x2.b)                    //[1, 2]
   } {
      z2.uw+= vrmpy(x7x6x5x4.ub, zzzz.ub)          //[0,15]
      z3.uw+= vrmpy(x7x6x5x4.ub, zz__.ub)          //[0,15]
      w3210 = vmem(ptr_w1++#1)                     //[1, 3]
   } {
      s3.uw += vrmpy(x7x6x5x4.ub, w10__.ub)        //[0,16]
      w___4     = vmem(ptr_w1++#1)                 //[1, 4]
   } {
      s1.uw += vrmpy(x3x2x1x0.ub, w10__.ub)        //[0,17]
      w_432.uw = vlsr(w3210.uw, c16)               //[1, 5]
      xbxax9x8.b = vshuff(xbx9xax8.b)              //[1, 5]
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, zzzz.ub)         //[0,18]
      z1.uw += vrmpy(x3x2x1x0.ub, zz__.ub)         //[0,18]
      w_432.w += vasl(w___4.w, c16)                //[1, 6]
      x7x5x6x4.b = vshuff(x1.b)                    //[1, 6]
   } {
      z2.uw += vrmpy(xbxax9x8.ub, ___z.ub)         //[1, 7]
      z3.uw += vrmpy(xbxax9x8.ub, _zzz.ub)         //[1, 7]
      x0 = vmemu(ptr_x1++M0)                       //[1, 7]
   } {
      s3.uw += vrmpy(xbxax9x8.ub, w_432.ub)        //[1, 8]
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[1, 8]
   } {
      s2.uw += vrmpy(xbxax9x8.ub, w___4.ub)        //[1, 9]
      x3x1x2x0.b = vshuff(x0.b)                    //[1, 9]
   } {
      s0.uw += vrmpy(x7x6x5x4.ub, w___4.ub)        //[1,10]
   } {
      z0.uw+= vrmpy(x7x6x5x4.ub, ___z.ub)          //[1,11]
      z1.uw+= vrmpy(x7x6x5x4.ub, _zzz.ub)          //[1,11]
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[1,11]
   }:endloop0  
/* --------------------------------------------------------------------------- */
   {  s0.uw += vrmpy(x3x2x1x0.ub, w3210.ub)        //[1,14]
      p0 = !cmp.gt(col_count, #-2)                 //
      d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
   } {
      d1d0.h = vpack(d1.w, d0.w):sat               //[0,27]
      z0.uw += vrmpy(x3x2x1x0.ub, zzzz.ub)         //[1,18]
      z2.uw += vrmpy(x7x6x5x4.ub, zzzz.ub)         //[1,15]
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w3210.ub)        //[1,12]
      w10__.w = vasl(w3210.w, c16)                 //[1,14]
      s0.w = vsub(s0.w, z0.w)                      //[WIDTH]
   } {
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
      s2.w = vsub(s2.w, z2.w)                      //[WIDTH,E]
      s1.uw += vrmpy(x7x6x5x4.ub, w_432.ub)        //[1,13]
   } {
      if(p0) s2 = s0
      z1.uw += vrmpy(x3x2x1x0.ub, zz__.ub)         //[1,18]
      z3.uw += vrmpy(x7x6x5x4.ub, zz__.ub)         //[1,15]
      s0.w = vasl(s0.w, vshamt_vec.w)              //
   } {
      if(p3) vmemu(ptr_y++#1) = d3210              //[WIDTH, E]
      s3.uw += vrmpy(x7x6x5x4.ub, w10__.ub)        //[1,16]
   } {
      s2.w = vasl(s2.w, vshamt_vec.w)              //[WIDTH]
      s1.uw += vrmpy(x3x2x1x0.ub, w10__.ub)        //[1,17]
      s3.w = vsub(s3.w, z3.w)                      //[WIDTH]
      p0 = !cmp.gt(col_count, #-3)                 //[WIDTH]
   } {
      d0.w = vmpye(s0.w, vrecip.uh)                //[WIDTH]multiply by 1/max
      s1.w = vsub(s1.w, z1.w)                      //[WIDTH]
      x2 = vmemu(ptr_x0+#2)                        //[P, 0]
   } {
      d0.w += vmpyo(s0.w, vrecip.h):SSR            //[WIDTH]
      s3.w = vasl(s3.w, vshamt_vec.w)              //[WIDTH]
      if(p0) s1 = s0                               //[WIDTH] if over gen'd write valid val
      p0 = !cmp.gt(col_count, #-1)                 //[WIDTH]
   } {
      if(p0) s3 = s0                               //[WIDTH]
      d2.w = vmpye(s2.w, vrecip.uh)                //[WIDTH]multiply by 1/max
   } {
      d2.w += vmpyo(s2.w, vrecip.h):SSR            //[WIDTH]
      min.w = vmin(min.w, d0.w)                    //[WIDTH]
      max.w = vmax(max.w, d0.w)                    //[WIDTH]
   } {
      min.w = vmin(min.w, d2.w)                    //[WIDTH]
      s1.w = vasl(s1.w, vshamt_vec.w)              //WIDTH
      d3.w = vmpye(s3.w, vrecip.uh)                //[WIDTH]multiply by 1/max
   } {
      max.w = vmax(max.w, d2.w)                    //[WIDTH]
      d3.w += vmpyo(s3.w, vrecip.h):SSR            //[WIDTH]
      x1 = vmemu(ptr_x0+#1)                        //[P, 1]
   } {
      max.w = vmax(max.w, d3.w)                    //[WIDTH]
      min.w = vmin(min.w, d3.w)                    //[WIDTH]
      d1.w = vmpye(s1.w, vrecip.uh)                //[WIDTH]multiply by 1/max
   } {
      p3 = cmp.eq(r0, r0)                          //[WIDTH]
      d1.w += vmpyo(s1.w, vrecip.h):SSR            //[WIDTH]
      xbx9xax8.b = vshuff(x2.b)                    //[P, 2]
   } {
      max.w = vmax(max.w, d1.w)                    //[WIDTH]
      min.w = vmin(min.w, d1.w)                    //[WIDTH]
   }:endloop1 //end width
/* --------------------------------------------------------------------------- */
   {  d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      col_count = out_width                        //[DEPTH,P]
      ptr_xin = add(ptr_xin, next_in_width_32)     //[DEPTH]
      next_out_width_32 = memw(sp+#16<<2)
   } {
      d1d0.h = vpack(d1.w, d0.w):sat               //[WIDTH,E]
      ptr_w0 += asl(filt_size, #8)                 //[DEPTH,E]filt_size = filt_height*256 //4*3*64
      loop1(.L_width, out_width4)                  //[DEPTH]
   } {
      depth_cnt = add(depth_cnt, #-1)              //[DEPTH,E]
      out_buf = add(out_buf, next_out_width_32)    //[DEPTH]
      x2 = vmemu(ptr_xin+#2)                       //[P, 0]
   } {
      p0 = cmp.eq(depth_cnt, #0)                   //[DEPTH,E]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
      recip_level = memw(sp+#52)
   } {
      vmemu(ptr_y+#0) = d3210                      //[WIDTH, E]save final activations
      if(!p0) jump .L_depth                        //[DEPTH,E]next depth
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                  //have all rows been processed
      depth_cnt = memw(sp+#17<<2)                  //reload depth count
      in_buf=add(in_buf,next_in_width_depth_stride)//vertical stride for input ptr
      if(!p0.new) jump:nt .L_height                //next height
   }//end height
/* ----------------------------------------------------------------------------- */
      ptr_max = memw(sp+#23<<2)                    //ptr to max and mins
   {
      r17:16 = memd(sp+#0)                         //restore
      vmem(ptr_max+#0) = max                       //save max vec
   } {
      r19:18 = memd(sp+#8)                         //restore
      vmem(ptr_max+#1) = min                       //save min vec
   } {
      r21:20 = memd(sp+#16)                        //restore
      r23:22 = memd(sp+#24)                        //restore
   } {
      r25:24 = memd(sp+#32)                        //restore
      r27:26 = memd(sp+#40)                        //restore
   } {
      dealloc_return                               //return
   }
/* ----------------------------------------------------------------------------- */
.L_end:
      .size dwconv2dbbb_s2_5xN_asm, .L_end-dwconv2dbbb_s2_5xN_asm
/* ----------------------------------------------------------------------------- */
/* =========================================================================== */