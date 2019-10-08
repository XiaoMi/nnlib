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
 *  FUNCTIONS      : dwconv2dbbb_s1_7xN_asm
 *                                          
 *  DESCRIPTION                            
 *   Mx7 depthwise filter stride horz = 1 stride vert is anyting, vertic filter is 
 *   anything. 
 *  CODESIZE                            
 *   896 bytes
 *  STACK SIZE
 *   7*2*128 + 64 = 1845 bytes
 *                                       
 *  ARCHITECTURE   : QDSP6V60+  + HVX     
 *
 *  REVISION HISTORY:                                    
 *  =================                                   
 *                                                     
 *  Author           Date      Comments       
 *  ------------------------------------------------------------- 
 *  DJH              05/20/19  created   
 *  -------------------------------------------------------------   */


/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dbbb_s1_7xN_h.S"
       .global dwconv2dbbb_s1_7xN_asm
       .balign 32
       .type  dwconv2dbbb_s1_7xN_asm, @function
dwconv2dbbb_s1_7xN_asm:
/* ----------------------------------------------------------------------------- */
//I/O regs
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
#define   sbuf                 r14       //
//Scaler Regs
#define   ptr_w0               r16       //
#define   ptr_w1               r17       //
#define   col_count            r9        //
#define   c8                   r6        //
#define   c24                  r4        //
#define   bias_ptr             r20       //
#define   ptr_x0               r18       //
#define   ptr_x1               r19       //
#define   ptr_x1ptr_x0         r19:18    //
#define   ptr_xin              r22       //
#define   ptr_y                r23       //
#define   depth_cnt            r26       //
#define   filt_size            r8        //
#define   next_in_width_depth_stride r28 //
#define   zzzz                 r7        //
#define   _zzz                 r27       //
#define   zzz_                 r15       //
#define   out_width4           r11       //
//Vector Regs
#define   vrecip                v0       //
#define   vshamt_vec            v1       //
#define   max                   v2       //
#define   min                   v3       //
#define   bias_val              v4       //
#define   w_654                 v8       //
#define   w6543                v26       //
#define   w3210                v28       //
#define   x0                   v30       //
#define   x1                   v29       //
#define   x2                   v5        //
#define   w210_                v11       //
#define   x3x2x1x0             v30       //
#define   x7x6x5x4             v10       //
#define   xbxax9x8             v5        //
#define   x3x2x3x2             v6        //
#define   x5x4x3x2             v6        //
#define   x7x6x7x6             v9        //
#define   x9x8x7x6             v9        //
#define   xbxaxbxa             v7        //
#define   xdxcxbxa             v7        //
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
   } {
      memd(sp+#0)  = r17:16                        //
      memd(sp+#8)  = r19:18                        //
   } {
      memd(sp+#16) = r21:20                        //
      memd(sp+#24) = r23:22                        //
   } {
      memd(sp+#32) = r25:24                        //
      memd(sp+#40) = r27:26                        //
      c8 = #8                                      //
      c24 = #24                                    //
   } {
      depth  = memw(sp+#17<<2)                     //
   } {
      depth = lsr(depth, #5)                       //depth/32
      out_width = memw(sp+#18<<2)                  //
      filt_zero = memw(sp+#21<<2)                  //
   } {
      memw(sp+#17<<2) = depth                      //
      out_height = memw(sp+#19<<2)                 //
      out_width4 = add(out_width, #3)              //
      zzzz  = vsplatb(filt_zero)                   //
   } {
      recip_shift = memw(sp+#25<<2)                //
      depth_cnt = memw(sp+#17<<2)                  //depth
      out_width4 = lsr(out_width4, #2)             //
      _zzz  = lsr(zzzz, #8)                        //
   } {
      vshamt_vec= vsplat(recip_shift)              //
      stride_v = memw(sp+#26<<2)                   //
      ptr_max = memw(sp+#23<<2)                    //
   } {
      filt_size = memw(sp+#20<<2)                  //
      max = vmem(ptr_max+#0)                       //
      next_in_width_depth_stride = mpyi(next_in_width_depth,stride_v) //
   } {
      min = vmem(ptr_max+#1)                       //
      zzz_  = asl(zzzz, #8)                        //
      filt_height = add(filt_size, #-1)            //
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {  bias_ptr = memw(sp+#22<<2)                   //
      ptr_xin = in_buf                             //
      ptr_w0 = filt                                //
      recip_level = memw(sp+#24<<2)
   }
/* ----------------------------------------------------------------------------- */
.L_depth:
   {
      vrecip = vmem(recip_level++#1)
      loop1(.L_width, out_width4)                  //
      memw(sp+#52) = recip_level
   }
   {  bias_val = vmem(bias_ptr++#1)                //
      ptr_x1ptr_x0 = combine(ptr_xin, ptr_xin)     //
      sbuf = memw(sp+#27<<2)                       //
      loop0(.L_init, filt_size)                    //
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_init:
   {  x0 = vmemu(ptr_x1+#0)                        //0
   } {
      x1 = vmemu(ptr_x1+#1)                        //1
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //1
   } {
      x0.b = vshuff(x0.b)                          //2
   } {
      x1.b = vshuff(x1.b)                          //3]
   } {
      x3x2x1x0.b = vshuff(x0.b)                    //4
      vmem(sbuf++#1) = x3x2x1x0.new                //next xb-8
   } {
      x7x6x5x4.b = vshuff(x1.b)                    //5
      vmem(sbuf++#1) = x7x6x5x4.new                //next xb-8
   }:endloop0
   {  sbuf = memw(sp+#27<<2)                       //
      w3210 = vmem(ptr_w0+#0)                      //[1, 1]
      ptr_x1 = ptr_x0                              //[WIDTH, P]
      p3 = !cmp.eq(r0, r0)                         //
   } {
      x3x2x1x0 = vmem(sbuf+#0)                     //[0, 0]
      ptr_x0 = add(ptr_x0, #128)                   //[WIDTH,P]+8 32 depths for stride 2
      ptr_y = out_buf                              //
      col_count = out_width                        //
   }
/* --------------------------------------------------------------------------- */
.L_width:
   {  z0.uw  = vrmpy(x3x2x1x0.ub, zzzz.ub)         //[0, 3]filter even output
      z1.uw  = vrmpy(x3x2x1x0.ub, zzz_.ub)         //[0, 3]filter even output
      w210_.w = vasl(w3210.w, c8)                  //[0, 3]
      x7x6x5x4 = vmem(sbuf+#1)                     //[0, 3]
   } {
      s0.uw  = vrmpy(x3x2x1x0.ub, w3210.ub)        //[0, 4]filter even output
      w_654     = vmem(ptr_w0+#1)                  //[0, 4]
      w6543.uw = vlsr(w3210.uw, c24)               //[0, 4]
      ptr_w1 = add(ptr_w0, #256)                   //[WIDTH, P]
   } {
      s1.uw  = vrmpy(x3x2x1x0.ub, w210_.ub)        //[0, 5]filter even output
      x2 = vmemu(ptr_x1+#2)                        //[0, 5]
      s0.w = vadd(bias_val.w, s0.w)                //[WIDTH, P]
   } {
      z0.uw+= vrmpy(x7x6x5x4.ub, _zzz.ub)          //[0, 6]filter even output
      z1.uw+= vrmpy(x7x6x5x4.ub, zzzz.ub)          //[0, 6]filter even output
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[0, 6]
      w6543.w += vasl(w_654.w, c8)                 //[0, 6]
   } {
      s0.uw += vrmpy(x7x6x5x4.ub, w_654.ub)        //[0, 7]filter even output
      x5x4x3x2.h = vshuffe(x7x6x5x4.h, x3x2x3x2.h) //[0, 7]
      x2.b = vshuff(x2.b)                          //[0, 7]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[0, 7]
   } {
      s1.uw += vrmpy(x7x6x5x4.ub, w6543.ub)        //[0, 8]filter even output
      vmem(sbuf++#1) = x7x6x5x4                    //[0, 8]next xb-8
      s3 = bias_val                                //[WIDTH, P]
      loop0(.L_vert, filt_height)                  //[WIDTH, P]
   } {
      s2.uw  = vrmpy(x5x4x3x2.ub, w3210.ub)        //[0,10]filter even output
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[0, 9]
      xbxax9x8.b = vshuff(x2.b)                    //[0, 9]
      vmem(sbuf++#1) = xbxax9x8.new                //[0,10]next xb-8
   } {
      z2.uw  = vrmpy(x5x4x3x2.ub, zzzz.ub)         //[0, 9]filter even output
      z3.uw  = vrmpy(x5x4x3x2.ub, zzz_.ub)         //[0, 9]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h, x7x6x7x6.h) //[0,10]
      s1.w = vadd(bias_val.w, s1.w)                //[WIDTH, P]
   } 
/* --------------------------------------------------------------------------- */
      .balign 32
.L_vert:
   {  s3.uw += vrmpy(x5x4x3x2.ub, w210_.ub)        //[0,11]filter even output
   } {
      z2.uw+= vrmpy(x9x8x7x6.ub, _zzz.ub)          //[0,12]filter even output
      z3.uw+= vrmpy(x9x8x7x6.ub, zzzz.ub)          //[0,12]filter even output
      x3x2x1x0 = vmem(sbuf+#0)                     //[1, 0]
   } {
      s2.uw += vrmpy(x9x8x7x6.ub, w_654.ub)        //[0,13]filter even output
      w3210 = vmem(ptr_w1++#1)                     //[1, 1]
   } {
      s3.uw += vrmpy(x9x8x7x6.ub, w6543.ub)        //[0,13]filter even output
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, zzzz.ub)         //[1, 3]filter even output
      z1.uw += vrmpy(x3x2x1x0.ub, zzz_.ub)         //[1, 3]filter even output
      w210_.w = vasl(w3210.w, c8)                  //[1, 3]
      x7x6x5x4 = vmem(sbuf+#1)                     //[1, 3]
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w3210.ub)        //[1, 4]filter even output
      w_654     = vmem(ptr_w1++#1)                 //[1, 4]
      w6543.uw = vlsr(w3210.uw, c24)               //[1, 4]
   } {
      s1.uw += vrmpy(x3x2x1x0.ub, w210_.ub)        //[1, 5]filter even output
      x2 = vmemu(ptr_x1+#2)                        //[1, 5]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[1, 5]
   } {
      z0.uw+= vrmpy(x7x6x5x4.ub, _zzz.ub)          //[1, 6]filter even output
      z1.uw+= vrmpy(x7x6x5x4.ub, zzzz.ub)          //[1, 6]filter even output
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[1, 6]
      w6543.w += vasl(w_654.w, c8)                 //[1, 6]
   } {
      s0.uw += vrmpy(x7x6x5x4.ub, w_654.ub)        //[1, 7]filter even output
      x5x4x3x2.h = vshuffe(x7x6x5x4.h, x3x2x3x2.h) //[1, 7]
      x2.b = vshuff(x2.b)                          //[1, 7]
   } {
      s1.uw += vrmpy(x7x6x5x4.ub, w6543.ub)        //[1, 8]filter even output
      vmem(sbuf++#1) = x7x6x5x4                    //[1, 8]next xb-8
   } {
      z2.uw += vrmpy(x5x4x3x2.ub, zzzz.ub)         //[1, 9]filter even output
      z3.uw += vrmpy(x5x4x3x2.ub, zzz_.ub)         //[1, 9]filter even output
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[1, 9]
      xbxax9x8.b = vshuff(x2.b)                    //[1, 9]
   } {
      s2.uw += vrmpy(x5x4x3x2.ub, w3210.ub)        //[1,10]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h, x7x6x7x6.h) //[1,10]
      vmem(sbuf++#1) = xbxax9x8                    //[1,10]next xb-8
   }:endloop0   
/* --------------------------------------------------------------------------- */
   {  s3.uw += vrmpy(x5x4x3x2.ub, w210_.ub)        //[1,11]filter even output
      s0.w = vsub(s0.w, z0.w)                      //[WIDTH]
      s1.w = vsub(s1.w, z1.w)                      //[WIDTH]
      col_count = add(col_count, #-4)              //[WIDTH]
   } {
      z2.uw+= vrmpy(x9x8x7x6.ub, _zzz.ub)          //[1,12]filter even output
      z3.uw+= vrmpy(x9x8x7x6.ub, zzzz.ub)          //[1,12]filter even output
      s2.w = vadd(s2.w, bias_val.w)                //[WIDTH, P]
      d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
   } {
      s2.uw += vrmpy(x9x8x7x6.ub, w_654.ub)        //[1,13]filter even output
      s0.w = vasl(s0.w, vshamt_vec.w)              //
      s3.w = vsub(s3.w, z3.w)                      //[WIDTH]
   } {
      s3.uw += vrmpy(x9x8x7x6.ub, w6543.ub)        //[0,13]filter even output
      s2.w = vsub(s2.w, z2.w)                      //[WIDTH,E]
      s1.w = vasl(s1.w, vshamt_vec.w)              //
      p0 = !cmp.gt(col_count, #-3)                 //
   } {
      d0.w = vmpye(s0.w, vrecip.uh)                //[W,15]multiply by 1/max
      if(p0) s1 = s0                               //[WIDTH] if over gen'd write valid val
      s2.w = vasl(s2.w, vshamt_vec.w)              //[WIDTH]
      p0 = !cmp.gt(col_count, #-2)                 //
   } {
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
      d0.w += vmpyo(s0.w, vrecip.h):SSR            //[W,17]3
      if(p0) s2 = s0                               //
   } {
      if(p3) vmemu(ptr_y++#1) = d3210              //[WIDTH, E]
      d1.w = vmpye(s1.w, vrecip.uh)                //[0,22]multiply by 1/max
      min.w = vmin(min.w, d0.w)                    //[0,22]8               //0+2+1
   } {
      d1.w += vmpyo(s1.w, vrecip.h):SSR            //[0,23]9
      s3.w = vasl(s3.w, vshamt_vec.w)              //
      max.w = vmax(max.w, d0.w)                    //[0,18]5               //0+2+1
      p0 = !cmp.gt(col_count, #-1)                 //
   } {
      d2.w = vmpye(s2.w, vrecip.uh)                //[0,15]multiply by 1/max
      if(p0) s3 = s0                               //
      min.w = vmin(min.w, d1.w)                    //[0,27]13              //0+2+1
      sbuf = memw(sp+#27<<2)                       //[WIDTH,P]
   } {
      d2.w += vmpyo(s2.w, vrecip.h):SSR            //[0,17]3
      x3x2x1x0 = vmem(sbuf+#0)                     //[P, 0]
      max.w = vmax(max.w, d1.w)                    //[0,26]12              //0+2+1
      p3 = cmp.eq(r0, r0)                          //
   } {
      d3.w = vmpye(s3.w, vrecip.uh)                //[0,22]multiply by 1/max
      max.w = vmax(max.w, d2.w)                    //[0,18]5               //0+2+1
      min.w = vmin(min.w, d2.w)                    //[0,22]8               //0+2+1
   } {
      d1d0.h = vpack(d1.w, d0.w):sat               //[0,27]
      d3.w += vmpyo(s3.w, vrecip.h):SSR            //[0,23]9
      w3210 = vmem(ptr_w0+#0)                      //[1, 1]
      ptr_x1 = ptr_x0                              //[WIDTH, P]
   } {
      min.w = vmin(min.w, d3.w)                    //[0,27]13              //0+2+1
      max.w = vmax(max.w, d3.w)                    //[0,26]12              //0+2+1
      ptr_x0 = add(ptr_x0, #128)                   //[WIDTH,P]+8 32 depths for stride 2
   }:endloop1 //end width
/* --------------------------------------------------------------------------- */
   {  d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      ptr_w0 += asl(filt_size, #8)                 //[DEPTH,E]filt_size = filt_height*256 //4*3*64
      ptr_xin = add(ptr_xin, next_in_width_32)     //[DEPTH]
      next_out_width_32 = memw(sp+#16<<2) 
   } {
      depth_cnt = add(depth_cnt, #-1)              //[DEPTH,E]
      out_buf = add(out_buf, next_out_width_32)    //[DEPTH]
      loop1(.L_width, out_width4)                  //[DEPTH]
   } {
      p0 = cmp.eq(depth_cnt, #0)                   //[DEPTH,E]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
      recip_level = memw(sp+#52)
   } {
      vmemu(ptr_y+#0) = d3210                      //[WIDTH, E]
      if(!p0) jump .L_depth                        //[DEPTH,E]
      if(p0) out_height = add(out_height, #-1)     //[HEIGHT]
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                  //height
      depth_cnt = memw(sp+#17<<2)                  //depth
      in_buf=add(in_buf,next_in_width_depth_stride)//stride 
      if(!p0.new) jump:nt .L_height                //height
   }//end height
/* ----------------------------------------------------------------------------- */
   {  ptr_max = memw(sp+#23<<2)                    //
   } {
      r17:16 = memd(sp+#0)                         //restore
      vmem(ptr_max+#0) = max                       //
   } {
      r19:18 = memd(sp+#8)                         //restore
      vmem(ptr_max+#1) = min                       //
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
      .size dwconv2dbbb_s1_7xN_asm, .L_end-dwconv2dbbb_s1_7xN_asm
/* ----------------------------------------------------------------------------- */
