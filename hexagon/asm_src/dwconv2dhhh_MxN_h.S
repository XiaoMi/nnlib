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

#if 0
void dwconv2dhhh_cn(
   uint16_t *in_buf,
   int16_t  *filt,
   uint16_t  *out_buf,
   int next_in_width,
   int next_out_width,
   int next_in_width_32,
   int next_out_width_32,
   int depth,
   int out_width,
   int out_height,
   int *bias_sum,
   int *max,
   int recip_level) {
   int out_y, d, out_x, in_val, filt_val;
   int out_z, filt_y, filt_x, tsum;
   int64_t sum;

    for (out_y = 0; out_y < out_height; out_y++) {
        for (out_x = 0; out_x < out_width; out_x++) {
            for(d=0; d < depth/32; d++) {
               for (out_z = 0; out_z < 32; out_z++) {
                  sum = bias_sum[32*d+out_z];
                  for (filt_y = 0; filt_y < 3; filt_y++) {
                     for (filt_x = 0; filt_x < 3; filt_x++) {
                        in_val = in_buf[(out_y +  filt_y) * next_in_width + d * next_in_width_32 + (out_x + filt_x) * 32 + out_z];

                        filt_val = filt[32*d*3*4 + (filt_x + 4*filt_y)*32 + out_z] ; //0122,0122,0122  
                        sum += in_val*filt_val;
                     }
                  }
                  tsum = sat(sum);
                  max[0]  = (tsum > max[0]) ? tsum : max[0];
                  max[32] = (tsum < max[32]) ? tsum : max[32];
                  sum = sum * (int64_t)recip_level + 0x40000000LL;
                  sum = sum >> 31;
                  if(sum < 0) sum = 0; if(sum > 0xffffll) sum = 0xffffll;
                  out_buf[out_y * next_out_width + 32 * out_x + d * next_out_width_32 + out_z] = (uint16_t) (sum>>2);
               }
            }
        }
    }
    return;
}
#endif
/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dhhh_MxN_h.S"
       .global dwconv2dhhh_MxN_asm
       .balign 32
       .type  dwconv2dhhh_MxN_asm, @function
dwconv2dhhh_MxN_asm:
/* ----------------------------------------------------------------------------- */
//CALLING REGS
#define   in_buf                     r0     //
#define   filt                       r1     //
#define   out_buf                    r2     //
#define   next_in_width_depth        r3     //
#define   next_out_width_depth       r4     //currently unused
#define   next_in_width_32           r5     //
#define   next_out_width_32          r10    //
#define   depth                      r17    //
#define   in_depth                   r11    //
#define   out_width                  r12    //
#define   out_height                 r13    //
#define   filt_width                 r24    //
#define   filt_height                r25    //
#define   bias_sum                   r14    //
#define   ptr_max                    r15    //
#define   recip_level                r16    //
#define   recip_shift                r8     //
#define   stride_v_h                 r28    //scratch holds packed vert and horz stride
//SCALER REGS
#define   ptr_w0                     r16    //
#define   ptr_w1                     r17    //
#define   c64                        r7     //
#define   c4                         r6     //
#define   s8                         r9     //
#define   bias_ptr                   r18    //
#define   ptr_x0                     r19    //
#define   ptr_x1                     r20    //
#define   ptr_x2                     r21    //
#define   ptr_xin                    r22    //
#define   ptr_y                      r23    //
#define   out_depth_cnt              r26    //       
#define   width_cnt                  r27    //
#define   filt_size                  r8     //
#define   next_in_width_depth_stride r4     //
#define   stride_h                   r28    //
//VECTOR REGS
#define   vec128                     v1     //
#define   w01_w00                    v0     //
#define   x01_00                     v6     //
#define   x03_02                     v7     //
#define   x02_01                     v8     //
#define   xh01_h00_xl01_l00          v21:20 //
#define   xh01_h00                   v21    //
#define            xl01_l00          v20    //
#define   xh02_h01_xl02_l01          v5:4   //
#define   xh02_h01                   v5     //
#define            xl02_l01          v4     //
#define   m01l_m00l                  v11:10 //
#define   m01l                       v11    //
#define        m00l                  v10    //
#define   m01h_m00h                  v13:12 //
#define   m01h                       v13    //
#define        m00h                  v12    //
#define   m11l_m10l                  v15:14 //
#define   m11l                       v15    //
#define        m10l                  v14    //
#define   m11h_m10h                  v3:2   //
#define   m11h                       v3     //
#define        m10h                  v2     //
#define   l1_l0                      v11:10 //
#define   l1                         v11    //
#define      l0                      v10    //
#define   h1_h0                      v13:12 //
#define   h1                         v13    //
#define      h0                      v12    //
#define   vrecip                     v9     //
#define   vshamt_vec                 v25    //
#define   max                        v16    //
#define   min                        v17    //
#define   d0                         v29    //
#define   d1                         v30    //
#define   d1d0                       v28    //
#define   bias_val                   v18    //
#define   s0                         v26    //
#define   s1                         v27    //
#define SSR        <<1:rnd:sat:shift        //simplfy mpy instruction
#define PV(VSRC) .word (0x1DFFE020+VSRC)    //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC)    //debug scaler reg
/* ----------------------------------------------------------------------------- */
   {  allocframe(#56)                            //0th entry on stack (56+8)/4=20
      c4 = #-4                                   //
      s8 = #128                                  //
   } {
      memd(sp+#0)  = r17:16                      //
      memd(sp+#8)  = r19:18                      //
      vec128 = vsplat(s8)                        //
      s8 = #8                                    //
   } {
      memd(sp+#16) = r21:20                      //
      memd(sp+#24) = r23:22                      //
   } {
      memd(sp+#32) = r25:24                      //
      depth  = memw(sp+#17<<2)                   //  
   } {
      memd(sp+#40) = r27:26                      //
      next_out_width_32 = memw(sp+#16<<2)        //
      depth = zxth(depth)                        //
      in_depth = lsr(depth, #21)                 //in depth / 32
   } {
      depth = lsr(depth, #5)                     //out depth/32
      out_width = memw(sp+#18<<2)                //
      out_height = memw(sp+#19<<2)               //
   } {
      out_depth_cnt = depth                      //
      recip_shift = memw(sp+#25<<2)              //
      filt_width = memw(sp+#20<<2)               //
      out_width = add(out_width, #1)             //
   } {
      filt_height = memw(sp+#21<<2)              //
      bias_sum = memw(sp+#22<<2)                 // 
      filt_width = add(filt_width, #1)           //round up to even
      out_width = lsr(out_width, #1)             //
   } {
      vshamt_vec= vsplat(recip_shift)            //
      ptr_max = memw(sp+#23<<2)                  //     
      recip_level = memw(sp+#24<<2)              //
      filt_width = lsr(filt_width, #1)           //divide by 2
   } {
      memw(sp+#17<<2) = depth                    //  
      vrecip = vsplat(recip_level)               //
      stride_v_h = memw(sp+#26<<2)               //
      filt_size = mpyi(filt_width, filt_height)  //
   } {
      c64 = zxth(stride_v_h)                     //horizontal stride
      next_in_width_depth_stride=lsr(stride_v_h,#16) //
      filt_size = asl(filt_size, #7)             //
      filt_width = add(filt_width, #-1)          //account fopr sware pipeline
   } {
      p0 = cmp.eq(c64, #1)                       //iss stride 1
      next_in_width_depth_stride=mpyi(next_in_width_depth_stride, next_in_width_depth) //
      stride_h = #256                            //
      max = vmem(ptr_max+#0)                     //
   } {
      memw(sp+#48) = in_depth                    //
      c64 = mux(p0, #64, #0)                     //0 for s2 64 for s1
      if(p0) stride_h = #128                     //
      min = vmem(ptr_max+#1)                     //
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {  bias_ptr = bias_sum                        //[HEIGHT]
      ptr_xin = in_buf                           //[HEIGHT]
      ptr_w0 = filt                              //[HEIGHT]
      out_height = add(out_height, #-1)          //[HEIGHT]
   }
/* ----------------------------------------------------------------------------- */
.L_depth:
   {  bias_val = vmem(bias_ptr++#1)              //[DEPTH]
      ptr_x0 = ptr_xin                           //[DEPTH]
      ptr_xin = add(ptr_xin, next_in_width_32)   //[DEPTH]
      in_depth = add(in_depth, #-1)              //[DEPTH]
   } {
      m00l = vec128                              //[WIDTH]
      ptr_y = out_buf                            //[DEPTH]
      out_buf = add(out_buf, next_out_width_32)  //[DEPTH]
      ptr_x1 = ptr_x0                            //[WIDTH]
   } {
      loop1(.L_vert, filt_height)                //[WIDTH]
      m01l = vec128                              //[WIDTH]
      ptr_w1 = ptr_w0                            //[WIDTH]
      width_cnt = out_width                      //[WIDTH]
   } {  
      loop0(.L_horz, filt_width)                 //[0, P]
      x01_00 = vmemu(ptr_x1+#0)                  //[0, P]16bits, 64 values, d32
      ptr_x2 = add(ptr_x1, #128)                 //[0, P]
   } {
      x03_02 = vmemu(ptr_x2++#1)                 //[0, 0]16bits, 64 values, d32
      m01h = #0                                  //[WIDTH]
      ptr_x0 = add(ptr_x0, stride_h)             //[WIDTH]advance by 2*stride 32 depths
   } {
      xh01_h00_xl01_l00.uh = vzxt(x01_00.ub)     //[0, 1]
      ptr_x1 = add(ptr_x1, next_in_width_depth)  //[0, P]
      m00h = #0                                  //[WIDTH]
      out_depth_cnt = add(out_depth_cnt, #-1)    //[DEPTH]
   } {
      x02_01 = vlalign(x03_02, x01_00, c64)      //[0, 2]16bits, 64 values, d32
      w01_w00 = vmem(ptr_w1++#1)                 //[0, 2]
      m11l = vec128                              //[WIDTH]
      m10l = vec128                              //[WIDTH]
   } {
      p1 = cmp.eq(out_depth_cnt, #0)             //[DEPTH]
      xh02_h01_xl02_l01.uh = vzxt(x02_01.ub)     //[0, 3]
      m11h_m10h = vcombine(m01h,m01h)            //[WIDTH]
      p3 = !cmp.eq(r0, r0)                       //[WIDTH]
   } 
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_width:                                        //
.L_vert:                                         //
.L_horz:                                         //
   {  m01l_m00l.w += vmpy(w01_w00.h,xl01_l00.uh) //[0, 4]64,128
      x01_00 = x03_02                            //[0, 4]
      x03_02 = vmemu(ptr_x2++#1)                 //[1, 0]16bits, 64 values, d32
   } {
      m01h_m00h.w += vmpy(w01_w00.h,xh01_h00.uh) //[0, 5]64,128
      xh01_h00_xl01_l00.uh = vzxt(x01_00.ub)     //[1, 1]
   } {
      m11l_m10l.w += vmpy(w01_w00.h,xl02_l01.uh) //[0, 6]64,128
      x02_01 = vlalign(x03_02, x01_00, c64)      //[1, 2]16bits, 64 values, d32
      w01_w00 = vmem(ptr_w1++#1)                 //[1, 2]
   } {
      w01_w00.tmp= vmem(ptr_w1+#-2)              //[0, 7]
      m11h_m10h.w += vmpy(w01_w00.h,xh02_h01.uh) //[0, 7]64,128
      xh02_h01_xl02_l01.uh = vzxt(x02_01.ub)     //[1, 3]
   }:endloop0 
/* ----------------------------------------------------------------------------- */
   {  loop0(.L_horz, filt_width)                 //[0, P]
      x01_00 = vmemu(ptr_x1+#0)                  //[0, P]16bits, 64 values, d32
      ptr_x2 = add(ptr_x1, #128)                 //[0, P]
   } {
      m01l_m00l.w += vmpy(w01_w00.h,xl01_l00.uh) //[1, 4]64,128
      x03_02 = vmemu(ptr_x2++#1)                 //[0, 0]16bits, 64 values, d32
   } {
      m01h_m00h.w += vmpy(w01_w00.h,xh01_h00.uh) //[1, 5]64,128
      xh01_h00_xl01_l00.uh = vzxt(x01_00.ub)     //[0, 1]
      ptr_x1 = add(ptr_x1, next_in_width_depth)  //[0, P]
   } {
      m11l_m10l.w += vmpy(w01_w00.h,xl02_l01.uh) //[1, 6]64,128
      x02_01 = vlalign(x03_02, x01_00, c64)      //[0, 2]16bits, 64 values, d32
      w01_w00 = vmem(ptr_w1++#1)                 //[0, 2]
   } {
      w01_w00.tmp = vmem(ptr_w1+#-2)             //[1, 7]
      m11h_m10h.w += vmpy(w01_w00.h,xh02_h01.uh) //[1, 7]64,128
      xh02_h01_xl02_l01.uh = vzxt(x02_01.ub)     //[0, 3]
   }:endloop1
/* ----------------------------------------------------------------------------- */
   {  m00l.w = vasr(m00l.w, s8)                  //WIDTH, E]
      ptr_w1 = ptr_w0                            //[WIDTH, P]
      ptr_x1 = ptr_x0                            //[WIDTH, P]
      p2 = cmp.eq(in_depth, #0)                  //has depth mpy couted down to 0
   } {
      m00h.w = vadd(m00h.w, m00l.w)              //WIDTH, E]
      m01l.w = vasr(m01l.w, s8)                  //WIDTH, E]
      x01_00 = vmemu(ptr_x1+#0)                  //[0, P]16bits, 64 values, d32
   } {
      m01h.w = vadd(m01h.w, m01l.w)              //WIDTH, E]
      m10l.w = vasr(m10l.w, s8)                  //WIDTH, E]
      ptr_x0 = add(ptr_x0, stride_h)             //[WIDTH, P]advance by 2*stride 32 depths
      ptr_x2 = add(ptr_x1, #128)                 //[0, P]
   } {
      m10h.w = vadd(m10h.w, m10l.w)              //WIDTH, E]
      m11l.w = vasr(m11l.w, s8)                  //WIDTH, E]
      x03_02 = vmemu(ptr_x2++#1)                 //[0, 0]16bits, 64 values, d32
   } {
      l1_l0 = vshuff(m01h,m00h, c4)              //[WIDTH, E]
      m11h.w = vadd(m11h.w, m11l.w)              //[WIDTH, E]
      ptr_x1 = add(ptr_x1, next_in_width_depth)  //[0, P]
   } {
      l0.w = vadd(l0.w, l1.w)                    //[WIDTH, E]
      m01l = vec128                              //[WIDTH, P]
      xh01_h00_xl01_l00.uh = vzxt(x01_00.ub)     //[0, 1]
   } {
      s0.w = vadd(l0.w, bias_val.w)              //[WIDTH, E]
      h1_h0 = vshuff(m11h,m10h, c4)              //[WIDTH, E]
      m00l = vec128                              //[WIDTH, P]
   } {
      h0.w = vadd(h0.w, h1.w)                    //[WIDTH, E]
      loop1(.L_vert, filt_height)                //[WIDTH, P]
      x02_01 = vlalign(x03_02, x01_00, c64)      //[0, 2]16bits, 64 values, d32
   } {
      d1d0.uh = vpack(d1.w, d0.w):sat            //[WIDTH, E]
      s0.w = vasl(s0.w, vshamt_vec.w)            //[WIDTH, E]
      s1.w = vadd(h0.w, bias_val.w)              //[WIDTH, E]
      w01_w00 = vmem(ptr_w1++#1)                 //[0, 2]
   } {
      m01h = #0                                  //[WIDTH, P]
      m00h = #0                                  //[WIDTH, P]
      m11l = vec128                              //[WIDTH, P]
      m10l = vec128                              //[WIDTH, P]
   } {
      s1.w = vasl(s1.w, vshamt_vec.w)            //[WIDTH, E]
      d0.w = vmpye(s0.w, vrecip.uh)              //[WIDTH, E]
      if(p3) vmemu(ptr_y++#1) = d1d0             //[WIDTH, E]
   } {
      xh02_h01_xl02_l01.uh = vzxt(x02_01.ub)     //[0, 3]
      d0.w += vmpyo(s0.w, vrecip.h):SSR          //[WIDTH, E]
      width_cnt = add(width_cnt, #-1)            //[WIDTH, E]
   } {
      p0 = cmp.eq(width_cnt, #0)                 //[WIDTH, E]
      max.w = vmax(max.w, d0.w)                  //[WIDTH, E]
      d1.w = vmpye(s1.w, vrecip.uh)              //[WIDTH, E]
      m11h = #0                                  //[WIDTH, P]
   } {
      min.w = vmin(min.w, d0.w)                  //[WIDTH, E]
      d1.w += vmpyo(s1.w, vrecip.h):SSR          //[WIDTH, E]
      m10h = #0                                  //[WIDTH, P]
      p3 = cmp.eq(r0, r0)                        //[WIDTH]
   } {
      max.w = vmax(max.w, d1.w)                  //[WIDTH, E]
      min.w = vmin(min.w, d1.w)                  //[WIDTH, E]
      if(!p0) jump .L_width                      //[WIDTH, E]
   }//end width
/* ----------------------------------------------------------------------------- */
   {
      d1d0.uh = vpack(d1.w, d0.w):sat            //[WIDTH, E]
      if(p2) in_depth = memw(sp+#48)             //reload mpy counter
      if(p2) ptr_xin = in_buf                    //[HEIGHT]
   } {
      ptr_w0 = add(ptr_w0, filt_size)            //[DEPTH]filt_size = filt_height*256 //4*3*64
      vmemu(ptr_y+#0) = d1d0                     //[WIDTH, E]
      if(!p1) jump .L_depth                      //[DEPTH]
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                //[HEIGHT]
      in_buf=add(in_buf,next_in_width_depth_stride)//HEIGHT stride
      out_depth_cnt = memw(sp+#17<<2)            //[HEIGHT]out depth 
      if(!p0.new) jump:t .L_height               //[HEIGHT]
   }//end height
/* ----------------------------------------------------------------------------- */
   {  vmem(ptr_max+#0) = max                     //save off discovered max
      r17:16 = memd(sp+#0)                       //restore
   } {
      vmem(ptr_max+#1) = min                     //save off discovered max
      r19:18 = memd(sp+#8)                       //restore
   } {
      r21:20 = memd(sp+#16)                      //restore
      r23:22 = memd(sp+#24)                      //restore
   } {
      r25:24 = memd(sp+#32)                      //restore
      r27:26 = memd(sp+#40)                      //restore
   } {
      dealloc_return                             //return
   }
/* ----------------------------------------------------------------------------- */
.L_end:
      .size dwconv2dhhh_MxN_asm, .L_end-dwconv2dhhh_MxN_asm
/* ----------------------------------------------------------------------------- */
   
