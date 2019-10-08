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
 *  FUNCTIONS      : gvconv2db2b2b2_d32_v65_asm
 *
 *  DESCRIPTION
 *    Perform 2d convolution using elements of size in_depth. Results are
 *    scaled and saturated to 8bits. Max and Min accumulations are kept.
 *
 *  ARCHITECTURE   : QDSP6V60  + HVX
 *
 *  INPUT          : R0 : uint8_t       *in_bufe
 *                   R1 : uint8_t       *in_bufo
 *                   R2 : uint8_t       *out_bufe
 *                   R3 : uint8_t       *out_bufo
 *                   R4 : uint8_t        *weights
 *                   R5 : int32_t        in_width
 *               SP+#0  : int32_t        next_out_width
 *               SP+#4  : int32_t        out_width
 *               SP+#8  : int32_t        stride_h_w
 *               SP+#12 : int32_t        in_depth
 *               SP+#16 : int32_t        filt_width
 *               SP+#20 : int32_t        filt_height
 *               SP+#24 : int32_t        out_height
 *               SP+#28 : const int32_t *biasbuf
 *               SP+#32 : const int32_t *suma
 *               SP+#36 : int32_t        next_suma_row
 *               SP+#40 : int32_t       *ptr_minmax
 *               SP+#44 : int32_t        recip
 *               SP+#48 : int32_t        recip_shift
 */
        .text
        .global gvconv2db2b2b2_d32_v65_asm
        .balign 32
        .type  gvconv2db2b2b2_d32_v65_asm, @function
gvconv2db2b2b2_d32_v65_asm:
/*=============================================================================*/
#define SS                  (13*8)
#define APTR                (SS+8)
#define in_bufe                r0
#define in_bufo                r1
#define in_bufo_in_bufe        r1:0
#define out_bufe               r2
#define out_bufo               r3
#define out_bufo_out_bufe      r3:2
#define weights                r4
#define in_width               r5
#define c8w                    r7
#define xl03x00                r8
#define xh03x00                xl03x00
#define xl23x20                xl03x00
#define xh23x20                xl03x00
#define suma0                  xl03x00
#define in_width_stride_depth  xl03x00
#define sumabuf                xl03x00
#define xl07x04                r9
#define xl27x24                xl07x04
#define xh27x24                xl07x04
#define xh07x04                xl07x04
#define next_out_width         xl07x04
#define suma1                  xl07x04
#define xl07x04_xl03x00        r9:8
#define xh07x04_xh03x00        xl07x04_xl03x00
#define xl27x24_xl23x20        xl07x04_xl03x00
#define xl13x10_xl03x00        xl07x04_xl03x00
#define xh27x24_xh23x20        xl07x04_xl03x00
#define xl33x30_xl23x20        xl07x04_xl03x00
#define xh13x10_xh03x00        xl07x04_xl03x00
#define xh33x30_xh23x20        xl07x04_xl03x00
#define next_out_width_in_width_stride_depth xl07x04_xl03x00
#define xl13x10                r10
#define xh13x10                xl13x10
#define in_bufet               xl13x10
#define out_bufet              xl13x10
#define recip_shift            xl13x10
#define suma3                  xl13x10
#define xl17x14                r11
#define xh17x14                xl17x14
#define in_bufot               xl17x14
#define out_bufot              xl17x14
#define recipshiftval          xl17x14
#define suma2                  xl17x14
#define xl17x14_xl13x10        r11:10
#define xl17x14_xl07x04        xl17x14_xl13x10
#define xh17x14_xh13x10        xl17x14_xl13x10
#define in_bufoet              xl17x14_xl13x10
#define out_bufoet             xl17x14_xl13x10
#define xh17x14_xh07x04        xl17x14_xl13x10
#define recipshiftval_recip_shift xl17x14_xl13x10
#define ptr_xl0                r12
#define ptr_xh0                r13
#define ptr_xl0_ptr_xh0        r13:12
#define ptr_xl1                r14
#define ptr_xh1                r15
#define ptr_wl                 r16
#define sumat                  ptr_wl
#define ptr_wh                 r17
#define sumainc                ptr_wh
#define ptr_wh_wl              r17:16
#define sumainc_sumat          ptr_wh_wl
#define filt_wid               r18
#define filt_ht                r19
#define suma                   r20
#define fetch_ptrh0            r20
#define stride_w               r21
#define ptr_ze                 r22
#define ptr_zo                 r23
#define fetch_ptrl0            ptr_zo
#define ptr_zo_ptr_ze          r23:22
#define out_y                  r24
#define out_x4                 r25
#define xl33x30                r26
#define xh33x30                xl33x30
#define xl37x34                r27
#define xh37x34                xl37x34
#define xl37x34_xl33x30        r27:26
#define xl37x34_xl27x24        xl37x34_xl33x30
#define xh37x34_xh33x30        xl37x34_xl33x30
#define xh37x34_xh27x24        xl37x34_xl33x30
#define stride_w4              r28
#define in_width_4             r30
#define next_outputs           r31
/*=============================================================================*/
#define sll0                   v0
#define sll1                   v1
#define sll1_sll0              v1:0
#define sll2                   v2
#define sll3                   v3
#define sll3_sll2              v3:2
#define shl0                   v4
#define shl1                   v5
#define shl1_shl0              v5:4
#define shl2                   v6
#define shl3                   v7
#define shl3_shl2              v7:6
#define shh0                   v8
#define s0                     shh0
#define vsuma0                 shh0
#define shh1                   v9
#define s1                     shh1
#define vsuma1                 shh1
#define shh1_shh0              v9:8
#define shh2                   v10
#define s2                     shh2
#define vsuma2                 shh2
#define shh3                   v11
#define vsuma3                 shh3
#define shh3_shh2              v11:10
#define wh0                    v12
#define wh1                    v13
#define wl0                    v14
#define wl1                    v15
#define min_val                v16
#define max_val                v17
#define recipvec               v18
#define wsum                   v19
#define constw80               v20
#define sk                     v21
#define y0                     v22
#define y1                     v23
#define y1y0                   v23:22
#define y2                     v24
#define y3                     v25
#define s3                     v26
/*=============================================================================*/
#define off_ptr_wl                ( 7*8+0)
#define off_ptr_wh                ( 7*8+4)
#define off_in_bufe               ( 8*8+0)
#define off_in_bufo               ( 8*8+4)
#define off_out_bufe              ( 9*8+0)
#define off_out_bufo              ( 9*8+4)
#define off_in_width_stride_depth (10*8+0)
#define off_next_out_width        (10*8+4)
#define off_recip                 (11*8+0)
/*=============================================================================*/
    { allocframe(#SS)                                   //
      memd(R29+#0*8-APTR) = R17:16                      //
      r8 = #0x80                                        //
      sll0 = #0                                         //
    }
    { memd(R29+#1*8) = R19:18                           //
      memd(R29+#2*8) = R21:20                           //
      constw80 = VSPLAT(r8)                             //
      sll1 = #0                                         //
    }
    { memd(R29+#3*8) = R23:22                           //
      memd(R29+#4*8) = R25:24                           //
      sll2 = #0                                         //
      sll3 = #0                                         //
    }
    { memd(R29+#5*8) = R27:26                           //
      memd(R29+#6*8) = R31:30                           //
      shl0 = constw80                                   //
      shl1 = constw80                                   //
    }
    { r9:8 = memd(r29+#APTR+0)                          // out_width|next_out_width
      r11:10 = memd(r29+#APTR+8)                        // in_depth|stride_h_w
      shl2 = constw80                                   //
      shl3 = constw80                                   //
    }
    { r13:12 = memd(r29+#APTR+16)                       // filt_height|filt_width
      r15:14 = memd(r29+#APTR+24)                       // biasbuf|out_height
      r22 = zxth(r10)                                   // stride_w
      r10 = lsr(r10,#16)                                // stride_h
    }
    { r1:0 = memd(r29+#APTR+40)                         // recip|ptr_minmax
      memd(r29+#off_in_bufe) = in_bufo_in_bufe          //
      filt_wid = asl(r12,#2)                            //filt_wid = filt_width*4
      filt_ht = asr(r11,#5)                             // in_depth>>5
    }
    { r6 = memw(r29+#APTR+48)                           // recip_shift
      max_val = vmem(r0+#0)                             //
      r22 = asl(r22,#2)                                 // stride_w*4
      r7 = #1                                           //
    }
    { min_val = vmem(r0+#1)                             //
      recipvec = VSPLAT(r1)                             //
      memw(r29+#APTR+8) = r22                           // stride_w
      stride_w4 = asl(r22,#4)                           // stride_w4*2
    }
    { wsum = vmem(r15+#0)                               // biasbuf
      r23 = mpyi(in_width,r11)                          // in_width,in_depth
      in_width_4 = asl(in_width,#5)                     //
      memd(r29+#off_out_bufe) = out_bufo_out_bufe       //
    }
    { r23 = mpyi(r23,r10)                               //in_width,in_depth*stride_h
      r7 = asl(r7,r6)                                   //
      memw(r29+#off_next_out_width) = r8                //
      memw(r29+#off_ptr_wl) = weights                   //
    }
    { memw(r29+#off_in_width_stride_depth) = r23        //
      r7 = combine(r7.l,r7.l)                           //
      ptr_wh = mpyi(filt_wid,r13)                       // filt_wid*filt_height
      filt_ht = mpyi(r13,filt_ht)                       // filt_height*in_depth>>5
    }
    { memd(r29+#off_recip) = r7:6                       //
      ptr_wh = mpyi(ptr_wh,r11)                         // filt_width*filt_height*in_depth
      next_outputs = mpyi(filt_ht,in_width_4)           //
      nop                                               //
    }
    { ptr_wh = addasl(weights,ptr_wh,#3)                //
      next_outputs += mpyi(stride_w4,#-2)               // -2*(stride_w4*2)
      stride_w = memw(r29+#APTR+8)                      //
    }
    { memw(r29+#off_ptr_wh) = ptr_wh                    //
      m0 = stride_w4                                    // stride_w4*2
      r22 = sub(#8,stride_w4)                           // -2*stride_w4
    }
    { m1 = r22                                          // -2*tsride_w4+8
      out_y = memw(r29+#APTR+24)                        // out_height
      c8w = #8                                          //
      stride_w4 = asr(stride_w4,#1)                     //
    }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_height:
    { ptr_xl0_ptr_xh0 = memd(r29+#off_in_bufe)          //
      next_out_width_in_width_stride_depth = memd(r29+#off_in_width_stride_depth)//
      out_y = add(out_y, #-1)                           //
      p0 = cmp.eq(out_y,#1)                             // last iteration?
    }
    { in_bufet = add(ptr_xl0,in_width_stride_depth)     //
      in_bufot = add(ptr_xh0,in_width_stride_depth)     //
      sumainc_sumat = memd(r29+#APTR+32)                //
      p2 = cmp.gt(r0,r0)                                //
    }
    { ptr_zo_ptr_ze = memd(r29+#off_out_bufe)           //
      memd(r29+#off_in_bufe) = in_bufoet                //
      sumat = add(sumat,sumainc)                        //
      loop1(.L_filt_height,filt_ht)                     //[p2]
    }
    { out_x4 = memw(r29+#APTR+4)                        //
      if (!p0) memw(r29+#APTR+32) = sumat               //
      out_bufet = addasl(ptr_ze,next_out_width,#1)      //
      out_bufot = add(ptr_zo,next_out_width)            //
    }
    { memd(r29+#off_out_bufe) = out_bufoet              //
      out_x4 = add(out_x4,#-4)                          //[p2]
      p1 = cmp.gt(out_x4,#4)                            //[p2] last iteration
      ptr_wh_wl = memd(r29+#off_ptr_wl)                 //[p2]
    }

/* ---------------------------------------------------------------------------- */
        .balign 32
.L_width:
    { y0.uh = vpack(y1.w,y0.w):sat                      //
      if (p2) vmem(ptr_ze++#1) = y0.new                 //
      shh0 = wsum                                       //
      shh1 = wsum                                       //
    }
    { y1.uh = vpack(y3.w,y2.w):sat                      //
      if (p2) vmem(ptr_ze++#1) = y1.new                 //
      shh2 = wsum                                       //
      shh3 = wsum                                       //
    }
.L_filt_height:
    { xl07x04_xl03x00 = memd(ptr_xl0+#0)                //[p0]
      xl17x14_xl13x10 = memd(ptr_xl0+stride_w4<<#0)     //[p0]
      ptr_xl1 = addasl(ptr_xl0,stride_w,#4)             //
      p3 = sp1loop0(.L_filt_width, filt_wid)            //
    }
    { ptr_xh1 = ptr_xh0                                 //
      ptr_xh0 = add(ptr_xh0,in_width_4)                 //
      xl07x04 = xl13x10                                 //
      xl13x10 = xl07x04                                 //
    }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_filt_width:
    { wh0.cur = vmem(ptr_wh++#1)                        //
      shl1_shl0.w += vrmpy(wh0.b,xl13x10_xl03x00.ub)    //
      xl37x34_xl33x30 = memd(ptr_xl1+stride_w4<<#0)     //
      if (!p3) fetch_ptrh0 = ptr_xh0
    }
    { dcfetch(fetch_ptrh0+#0)                           //
      fetch_ptrh0 = add(fetch_ptrh0,#64)                //
      if (!p3) fetch_ptrl0 = add(ptr_xl0,in_width_4)    //
      if (!p3) ptr_xl0 = add(ptr_xl0,in_width_4)        //
    }
#define FAST_16B_CONV 1
#if defined(FAST_16B_CONV)
    { wh1.cur = vmem(ptr_wh++#1)                        //
      shl1_shl0.w += vrmpy(wh1.b,xl17x14_xl07x04.ub)    //
      xl27x24_xl23x20 = memd(ptr_xl1++M1)               //
    }
    { wl0 = vmem(ptr_wl++#1)                            //
      dcfetch(fetch_ptrl0+#0)                           //
      xl27x24 = xl33x30                                 //
      xl33x30 = xl27x24                                 //
    }
    { shl3_shl2.w += vrmpy(wh0.b,xl33x30_xl23x20.ub)    //
      xh07x04_xh03x00 = memd(ptr_xh1++M0)               //
      xh17x14_xh13x10 = memd(ptr_xh1+stride_w4<<#0)     //
      fetch_ptrl0 = add(fetch_ptrl0,#64)                //
    }
    { wl1 = vmem(ptr_wl++#1)                            //
      shl3_shl2.w += vrmpy(wh1.b,xl37x34_xl27x24.ub)    //
      xh07x04 = xh13x10                                 //
      xh13x10 = xh07x04                                 //
    }
#else
    { wl0.cur = vmem(ptr_wl++#1)                        //
      sll1_sll0.w += vrmpy(wl0.b,xl13x10_xl03x00.ub)    //L*L
      xl27x24_xl23x20 = memd(ptr_xl1++M1)               //
    }
    { wh1.cur = vmem(ptr_wh++#1)                        //
      shl1_shl0.w += vrmpy(wh1.b,xl17x14_xl07x04.ub)    //
      xl27x24 = xl33x30                                 //
      xl33x30 = xl27x24                                 //
    }
    { wl1.cur = vmem(ptr_wl++#1)                        //
      sll1_sll0.w += vrmpy(wl1.b,xl17x14_xl07x04.ub)    //L*L
      dcfetch(fetch_ptrl0+#0)                           //
      fetch_ptrl0 = add(fetch_ptrl0,#64)                //
    }
    { shl3_shl2.w += vrmpy(wh0.b,xl33x30_xl23x20.ub)    //
      sll3_sll2.w += vrmpy(wl0.b,xl33x30_xl23x20.ub)    //L*L
      xh07x04_xh03x00 = memd(ptr_xh1++M0)               //
      xh17x14_xh13x10 = memd(ptr_xh1+stride_w4<<#0)     //
    }
    { shl3_shl2.w += vrmpy(wh1.b,xl37x34_xl27x24.ub)    //
      sll3_sll2.w += vrmpy(wl1.b,xl37x34_xl27x24.ub)    //L*L
      xh07x04 = xh13x10                                 //
      xh13x10 = xh07x04                                 //
    }
#endif
    { shl1_shl0.w += vrmpy(wl0.b,xh13x10_xh03x00.ub)    //
      shh1_shh0.w += vrmpy(wh0.b,xh13x10_xh03x00.ub)    //
      xh27x24_xh23x20 = memd(ptr_xh1++M1)               //
      xh37x34_xh33x30 = memd(ptr_xh1+stride_w4<<#0)     //
    }
    { shl1_shl0.w += vrmpy(wl1.b,xh17x14_xh07x04.ub)    //
      shh1_shh0.w += vrmpy(wh1.b,xh17x14_xh07x04.ub)    //
      xh27x24 = xh33x30                                 //
      xh33x30 = xh27x24                                 //
    }
    { shl3_shl2.w += vrmpy(wl0.b,xh33x30_xh23x20.ub)    //
      shh3_shh2.w += vrmpy(wh0.b,xh33x30_xh23x20.ub)    //
      xl07x04_xl03x00 = memd(ptr_xl1++M0)               //[p0]
      xl17x14_xl13x10 = memd(ptr_xl1+stride_w4<<#0)     //[p0]
    }
    { shl3_shl2.w += vrmpy(wl1.b,xh37x34_xh27x24.ub)    //
      shh3_shh2.w += vrmpy(wh1.b,xh37x34_xh27x24.ub)    //
      xl07x04 = xl13x10                                 //
      xl13x10 = xl07x04                                 //
    }:endloop0:endloop1
    { shl0.w += vasr(sll0.w,c8w)                        //
      recipshiftval_recip_shift = memd(r29+#off_recip)  //
      sll0 = #0                                         //
    }
    { shl1.w += vasr(sll1.w,c8w)                        //
      ptr_xl0 = sub(ptr_xl0,next_outputs)               //
      ptr_xh0 = sub(ptr_xh0,next_outputs)               //
      sll1 = #0                                         //
    }
    { shh0.w += vasr(shl0.w,c8w)                        //
      loop1(.L_filt_height,filt_ht)                     //[p2]
      shl0 = constw80                                   //
    }
    { shh1.w += vasr(shl1.w,c8w)                        //
      min_val.w = vmin(min_val.w,shh0.w)                //
      max_val.w = vmax(max_val.w,shh0.w)                //
      sk = shh0                                         //
    }
    { s0.w = vmpyi(shh0.w,recipshiftval.h)              //s0=Q6_Vw_vasl_VwR(shh0.h,recip_shift)
      shl2.w += vasr(sll2.w,c8w)                        //
      shl1 = constw80                                   //
      p2 = cmp.gt(out_x4,#1-4)                          //should s1 be included ?
    }
    { if (p2) sk = shh1                                 //
      s1.w = vmpyi(shh1.w,recipshiftval.h)              //s1=Q6_Vw_vasl_VwR(shh1.h,recip_shift)
      shl3.w += vasr(sll3.w,c8w)                        //
    }
    { shh2.w += vasr(shl2.w,c8w)                        //
      y0.w = vmpye(s0.w,recipvec.uh)                    //
      min_val.w = vmin(min_val.w,sk.w)                  //
      p2 = cmp.gt(out_x4,#2-4)                          //should s2 be included ?
    }
    { shh3.w += vasr(shl3.w,c8w)                        //
      y0.w += vmpyo(s0.w,recipvec.h):<<1:rnd:sat:shift  //
      max_val.w = vmax(max_val.w,sk.w)                  //
    }
    { s2.w = vasl(shh2.w,recip_shift)                   //
      y1.w = vmpye(s1.w,recipvec.uh)                    //
      if (p2) sk = shh2                                 //
      p2 = cmp.gt(out_x4,#3-4)                          //should s3 be included ?
    }
    { s3.w = vasl(shh3.w,recip_shift)                   //
      y1.w += vmpyo(s1.w,recipvec.h):<<1:rnd:sat:shift  //
      min_val.w = vmin(min_val.w,sk.w)                  //
      out_x4 = add(out_x4,#-4)                          //[p2]
    }
    { y2.w = vmpye(s2.w,recipvec.uh)                    //
      max_val.w = vmax(max_val.w,sk.w)                  //
      if (p2) sk = shh3                                 //
      ptr_wh_wl = memd(r29+#off_ptr_wl)                 //[p2]
    }
    { y2.w += vmpyo(s2.w,recipvec.h):<<1:rnd:sat:shift  //
      min_val.w = vmin(min_val.w,sk.w)                  //
      max_val.w = vmax(max_val.w,sk.w)                  //
      p2 = p1                                           //
    }
    { y3.w = vmpye(s3.w,recipvec.uh)                    //
      sll2 = #0                                         //
      sll3 = #0                                         //
      p1 = cmp.gt(out_x4,#0)                            //[p2] last iteration
    }
    { y3.w += vmpyo(s3.w,recipvec.h):<<1:rnd:sat:shift  //
      shl2 = constw80                                   //
      shl3 = constw80                                   //
      if (p2) jump:t .L_width                           //
    }
/* ---------------------------------------------------------------------------- */
    { y0.uh = vpack(y1.w,y0.w):sat                      //
      shh0 = wsum                                       //
      vmem(ptr_ze++#1) = y0.new                         //
    }
    { y1.uh = vpack(y3.w,y2.w):sat                      //
      vmem(ptr_ze++#1) = y1.new                         //
      p0 = cmp.eq(out_y, #0)                            //
      if(!p0.new) jump:t .L_height                      //
    }                                                   //end lines per block
/* ---------------------------------------------------------------------------- */
    { r0 = memw(r29+#APTR+40)                           // ptr_minmax
    }
    { vmem(r0+#0) = max_val                             //
      R17:16 = memd(R29+#0*8)                           // restore callee-saved registers
    }
    { vmem(r0+#1) = min_val                             //
      R19:18 = memd(R29+#1*8)                           // restore callee-saved registers
    }
    { R21:20 = memd(R29+#2*8)                           // restore callee-saved registers
      R23:22 = memd(R29+#3*8)                           // restore callee-saved registers
    }
    { R25:24 = memd(R29+#4*8)                           // restore callee-saved registers
      R31:30 = memd(R29+#6*8)                           // restore callee-saved registers
    }
    { R27:26 = memd(R29+#5*8)                           // restore callee-saved registers
      DEALLOC_RETURN                                    // return
    }
.L_end:
/*=============================================================================*/
      .size gvconv2db2b2b2_d32_v65_asm, .L_end-gvconv2db2b2b2_d32_v65_asm
/*=============================================================================*/
