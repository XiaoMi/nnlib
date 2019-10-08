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

        .text
        .file "inconv2dbbb_d32_v60_h.S"
        .global inconv2db2b2b2_v60_asm
        .balign 32
        .type  inconv2db2b2b2_v60_asm, @function
/* parameters:
 *  r0       const uint8_t * in_bufe
 *  r1       const uint8_t * in_bufo
 *  r2       const uint8_t * weights
 *  r3       uint16_t * out_bufe
 *  r4       int in_width
 *  r5       int out_width_stride_depth
 * PARMW(0)  int out_width
 * PARMW(1)  int stride_h_w
 * PARMW(2)  int in_depth
 * PARMW(3)  int filt_width
 * PARMW(4)  int filt_height
 * PARMW(5)  int out_height
 * PARMW(6)  const int32_t * bias_add
 * PARMW(7)  const int32_t *suma
 * PARMW(8)  int32_t next_suma_row
 * PARMW(9)  int32_t * ptr_minmax
 * PARMW(10) int32_t recip
 * PARMW(11) int recip_shift
 */
inconv2db2b2b2_v60_asm:
/*=============================================================================*/
#define in_bufe                r0
#define in_bufo                r1
#define in_bufo_in_bufe        r1:0
#define weights                r2
#define out_bufe               r3
#define in_width               r4
#define out_width_stride_depth r5
#define recip_shift            r6
#define recip                  recip_shift
#define c8                     r7
#define xl03x00                r8
#define sum0                   xl03x00
#define sum1                   xl03x00
#define sum2                   xl03x00
#define sum3                   xl03x00
#define xl13x10                r9
#define xl13x10_xl03x00        r9:8
#define xl23x20                r10
#define xl33x30                r11
#define xl33x30_xl23x20        r11:10
#define xh03x00                r12
#define xh13x10                r13
#define xh13x10_xh03x00        r13:12
#define xh23x20                r14
#define xh33x30                r15
#define xh33x30_xh23x20        r15:14
#define in_width_depth         r16
#define in_depth               r17
#define stride_h_w             r18
#define stride_h               stride_h_w
#define in_width_stride_depth  r19
#define next_outputs           r20
#define filt_height            r21
#define next_suma_row          r22
#define bias_add               r23
#define suma                   bias_add
#define ptr_xl0                r24
#define stride_w               ptr_xl0
#define ptr_xh0                r25
#define ptr_xh0_ptr_xl0        r25:24
#define ptr_w                  r26
#define filt_width             r27
#define out_x4                 r28
#define out_height             r30
#define active_sum             r31

/*=============================================================================*/
#define shl0                   v0
#define shl1                   v1
#define shl2                   v2
#define shl3                   v3
#define shh0                   v4
#define shh1                   v5
#define shh2                   v6
#define shh3                   v7
#define w0l                    v8
#define w0h                    v9
#define wsum                   v10
#define recipvec               v11
#define out0                   v12
#define out1                   v13
#define out2                   v14
#define out3                   v15
#define sy0                    v16
#define sy1                    v17
#define sy2                    v18
#define sy3                    v19
#define max_val                v20
#define min_val                v21
#define vzero                  v22
#define c80                    v23
/*=============================================================================*/
#define SS                     (13*8)
#define SS8                    (13*8+8)
#define PARMW(n) sp+#(SS8+4*(n))

// TODO: prefetch
// TODO: min/max
    {
      allocframe(#SS)                               //
      memd(R29+#0*8-SS8) = R17:16                   //
      r6 = ##0x7fffffff                             //
    } {
      memd(sp+#1*8)  = r19:18                       //
      memd(sp+#2*8) = r21:20                        //
      min_val = vsplat(r6)                          //
      vzero = #0                                    //
    } {
      memd(sp+#3*8) = r23:22                        //
      memd(sp+#4*8) = r25:24                        //
      max_val.w = vsub(vzero.w,min_val.w)           //
      r6 = #0x80                                    //
    } {
      memd(sp+#5*8) = r27:26                        //
      memd(R29+#6*8) = R31:30                       //
      c80 = vsplat(r6)                              //
      out_width_stride_depth = add(out_width_stride_depth,out_width_stride_depth)//
    } {
      in_depth = memw(PARMW(2))                     //
      stride_h_w = memw(PARMW(1))                   //
      c8 = #8                                       //
    } {
      in_width_depth = mpyi(in_depth,in_width)      //
      stride_w = zxth(stride_h_w)                   //
      filt_height = memw(PARMW(4))                  //
      recip = memw(PARMW(10))                       //
    } {
      stride_h = asrh(stride_h_w)                   //
      next_outputs = mpyi(stride_w,#-16)            //
      next_suma_row = memw(PARMW(8))                //
      bias_add = memw(PARMW(6))                     //
    } {
      in_width_stride_depth = mpyi(in_width_depth, stride_h)//
      next_outputs += mpyi(in_width_depth,filt_height)//
      wsum = vmem(bias_add+#0)                      //
      active_sum = memw(PARMW(7))                   //
    } {
      filt_width = memw(PARMW(3))                   //
      recipvec = vsplat(recip)                      //
      in_width_depth = add(in_width_depth,#-3*4)    //
      out_x4 = memw(PARMW(0))                       //
    } {
      in_width_depth -= mpyi(filt_width,#4)         //
      memw(sp+#56) = out_bufe                       //
      stride_w = mpyi(stride_w,#4)                  //
      out_x4 = add(out_x4,#3)                       //
    } {
      out_height = memw(PARMW(5))                   //
      out_x4 = and(out_x4,#-4)                      //
      m0 = stride_w                                 //
      stride_w = neg(stride_w)                      //
    } {
      recip_shift = memw(PARMW(11))                 //
      next_suma_row += mpyi(out_x4,stride_w)        //
    }
/* ---------------------------------------------------------------------------- */
    .balign 32
.L_height:
    {
      out_bufe = memw(sp+#56)                       //
      memw(sp+#56) += out_width_stride_depth        //
      ptr_xh0_ptr_xl0 = in_bufo_in_bufe             //
    } {
      out_x4 = memw(PARMW(0))                       //
      out_height = add(out_height, #-1)             //
      in_bufe = add(in_bufe,in_width_stride_depth)  //
      in_bufo = add(in_bufo,in_width_stride_depth)  //
    }
/* ---------------------------------------------------------------------------- */
    .balign 32
.L_width:
    {
      sum0 = memw(active_sum++M0)                   //
      ptr_w = weights                               //
    } {
      shh0 = vsplat(sum0)                           //
      sum1 = memw(active_sum++M0)                   //
    } {
      shh1 = vsplat(sum1)                           //
      sum2 = memw(active_sum++M0)                   //
      shl0 = c80                                    //
      shl1 = c80                                    //
    } {
      shh2 = vsplat(sum2)                           //
      sum3 = memw(active_sum++M0)                   //
      shl2 = c80                                    //
      shl3 = c80                                    //
    } {
      shh3 = vsplat(sum3)                           //
      loop1(.L_filt_height,filt_height)             //
    } {
      shh0.w = vadd(shh0.w,wsum.w)                  //
      shh1.w = vadd(shh1.w,wsum.w)                  //
      shh2.w = vadd(shh2.w,wsum.w)                  //
      shh3.w = vadd(shh3.w,wsum.w)                  //
    }
/* ---------------------------------------------------------------------------- */
    .balign 32
.L_filt_height:
    .falign
    {
      xl13x10_xl03x00 = memd(ptr_xl0++#8)           //
      xh13x10_xh03x00 = memd(ptr_xh0++#8)           //
    } {
      xl23x20 = memw(ptr_xl0++#4)                   //
      xh23x20 = memw(ptr_xh0++#4)                   //
      p3 = sp1loop0(.L_filt_width,filt_width)       //
    }
/* ---------------------------------------------------------------------------- */
    .balign 32
.L_filt_width:
    {
      w0h.cur = vmem(ptr_w+#1)                      //
      shl0.uw += vrmpy(w0h.ub,xl03x00.ub)           //
      shl1.uw += vrmpy(w0h.ub,xl13x10.ub)           //
      xl13x10_xl03x00 = combine(xl23x20,xl13x10)    //
    } {
      w0l.cur = vmem(ptr_w++#2)                     //
      shl0.uw += vrmpy(w0l.ub,xh03x00.ub)           //
      shl1.uw += vrmpy(w0l.ub,xh13x10.ub)           //
      xl33x30 = memw(ptr_xl0++#4)                   //
    } {
      shl2.uw += vrmpy(w0h.ub,xl23x20.ub)           //
      shl3.uw += vrmpy(w0h.ub,xl33x30.ub)           //
      xl23x20 = xl33x30                             //
    } {
      shh0.uw += vrmpy(w0h.ub,xh03x00.ub)           //
      shh1.uw += vrmpy(w0h.ub,xh13x10.ub)           //
      xh13x10_xh03x00 = combine(xh23x20, xh13x10)   //
      xh33x30 = memw(ptr_xh0++#4)                   //
    } {
      shl2.uw += vrmpy(w0l.ub,xh23x20.ub)           //
      shh2.uw += vrmpy(w0h.ub,xh23x20.ub)           //
      xh23x20 = xh33x30                             //
    } {
      shl3.uw += vrmpy(w0l.ub,xh33x30.ub)           //
      shh3.uw += vrmpy(w0h.ub,xh33x30.ub)           //
    }:endloop0
/* ---------------------------------------------------------------------------- */
    {
      ptr_xl0 = add(ptr_xl0, in_width_depth)        //
      ptr_xh0 = add(ptr_xh0, in_width_depth)        //
    }:endloop1
/* ---------------------------------------------------------------------------- */
    {
      ptr_xl0 = sub(ptr_xl0, next_outputs)          //
      ptr_xh0 = sub(ptr_xh0, next_outputs)          //
      shh0.w += vasr(shl0.w,c8)                     //
      out_x4 = add(out_x4, #-4)                     //
    } {
      shh1.w += vasr(shl1.w,c8)                     //
      p1 = cmp.gt(out_x4, #0)                       //
      max_val.w = vmax(max_val.w,shh0.w)            //
      min_val.w = vmin(min_val.w,shh0.w)            //
    } {
      out0.w = vasl(shh0.w,recip_shift)             //
      max_val.w = vmax(max_val.w,shh1.w)            //
      min_val.w = vmin(min_val.w,shh1.w)            //
    } {
      out1.w = vasl(shh1.w,recip_shift)             //
    } {
      shh2.w += vasr(shl2.w,c8)                     //
      sy0.w = vmpye(out0.w,recipvec.uh)             //
    } {
      shh3.w += vasr(shl3.w,c8)                     //
      sy1.w = vmpye(out1.w,recipvec.uh)             //
      max_val.w = vmax(max_val.w,shh2.w)            //
    } {
      out2.w = vasl(shh2.w,recip_shift)             //
      sy0.w+= vmpyo(out0.w, recipvec.h):<<1:rnd:sat:shift//
      min_val.w = vmin(min_val.w,shh2.w)            //
    } {
      out3.w = vasl(shh3.w,recip_shift)             //
      max_val.w = vmax(max_val.w,shh3.w)            //
      min_val.w = vmin(min_val.w,shh3.w)            //
    } {
      sy1.w+= vmpyo(out1.w, recipvec.h):<<1:rnd:sat:shift//
    } {
      sy2.w = vmpye(out2.w,recipvec.uh)             //
    } {
      sy3.w = vmpye(out3.w,recipvec.uh)             //
    } {
      sy2.w+= vmpyo(out2.w, recipvec.h):<<1:rnd:sat:shift//
    } {
      sy3.w+= vmpyo(out3.w, recipvec.h):<<1:rnd:sat:shift//
    } {
      sy0.uh = vpack(sy1.w,sy0.w):sat               //
      vmem(out_bufe++#1) = sy0.new                  //
    } {
      sy2.uh = vpack(sy3.w,sy2.w):sat               //
      vmem(out_bufe++#1) = sy2.new                  //
      if(p1) jump:t .L_width                        //
    }                                               //end cols per line
/* ---------------------------------------------------------------------------- */
    {
      active_sum = add(active_sum, next_suma_row)   //
      p1 = cmp.gt(out_height, #0)                   //
      if(p1.new) jump:t .L_height                   //
    }                                               //end lines per block
/* ---------------------------------------------------------------------------- */
    {
      r0 = memw(PARMW(9))                           // ptr_minmax
    } {
      v0 = vmem(r0+#0)                              //
    } {
      v1 = vmem(r0+#1)                              //
      max_val.w = vmax(max_val.w,v0.w)              //
    } {
      min_val.w = vmin(min_val.w,v1.w)              //
      r7 = #64                                      //
      loop0(.minmax_lp,#5)                          //
    }
    .falign
.minmax_lp:
    {
      v0 = vror(max_val,r7)                         //
    } {
      v1 = vror(min_val,r7)                         //
      r7 = asr(r7,#1)                               //
    } {
      max_val.w = vmax(max_val.w,v0.w)              //
      min_val.w = vmin(min_val.w,v1.w)              //
    }:endloop0
    {
      vmem(r0+#0) = max_val                         //
      R17:16 = memd(R29+#0*8)                       // restore callee-saved registers
    } {
      vmem(r0+#1) = min_val                         //
      R19:18 = memd(R29+#1*8)                       // restore callee-saved registers
    } {
      R21:20 = memd(R29+#2*8)                       // restore callee-saved registers
      R23:22 = memd(R29+#3*8)                       // restore callee-saved registers
    } {
      R25:24 = memd(R29+#4*8)                       // restore callee-saved registers
      R31:30 = memd(R29+#6*8)                       // restore callee-saved registers
    } {
      R27:26 = memd(R29+#5*8)                       // restore callee-saved registers
      DEALLOC_RETURN                                // return
    }
.L_end:
/*=============================================================================*/
      .size inconv2db2b2b2_v60_asm, .L_end-inconv2db2b2b2_v60_asm
/*=============================================================================*/

