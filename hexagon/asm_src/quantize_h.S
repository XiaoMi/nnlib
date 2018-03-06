
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
 */
/*======================================================================*/
/*                                                                      */
/*======================================================================*/
/*  FUNCTIONS      : quantize_asm                                       */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                Compute quantized range from in range to 8bit unsigned*/
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 09/12/16       created                          */
/*======================================================================*/
/*  IDEAL-LIST SCHEDULED CYCLE-COUNT:                                   */
/*     ->  N/32+7                                                       */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 272 bytes                                             */
/*     ASSUMPTIONS                                                      */
/*        y and x are 128 byte aligned                                  */
/*  C MODEL                                                             */
/*======================================================================*/
#if 0
void quantize_cn(unsigned char * out, int * input, int in_max_val, 
                 int in_min_val, float in_level_size, int n)
{
       float out_min;
       float out_max;
       int i;
        /* Make sure min val <= 0.0 in floaty land */
        out_min = in_level_size * (float)in_min_val;
        out_max = in_level_size * (float)in_max_val;
        if (out_min > 0.0f) out_min = 0.0f;
        /* Requantize with new range */

        for (i = 0; i < n; i++) {
          /* We want 0.0 -- 255.0 to resize to 0..255 */
          float resize_amt = 255.f/(out_max-out_min);
          float value_f = (in_level_size*input[i] - out_min) * resize_amt;
          int value_i = roundf(value_f);
          out[i] = (value_i < 0) ? 0 : ((value_i > 255) ? 255 : value_i);
        }
  return;
}
#endif
/*======================================================================*/
          .global quantize_asm
          .balign 32
          .type  quantize_asm, @function
quantize_asm:
/* ==================================================================== */
#define ptr_x    r0   //pointer to input
#define offset   r1   //qpoint of input data
#define gain     r2   //output data
#define ptr_y    r3   //
#define n        r4   //number of points > 128
#define lsb7     r5
#define u0       v0
#define u1       v1
#define u2       v2
#define u3       v3
#define w0       v4
#define w1       v5
#define w2       v6
#define w3       v7
#define voffset  v8
#define vgain    v9
#define h3h1     v10
#define h2h0     v11
#define b3b2b1b0 v12
/* ==================================================================== */
       { voffset = vsplat(offset)                    //[P, ]
         q0 = vsetq(n)                               //
         lsb7 = #127                                 //
       } {
         u0.tmp = vmem(ptr_x++#1)                    //[0, 0]
         u0.w = vsub(u0.w, voffset.w)                //[0, 0]
         p0 = bitsclr(n, lsb7)                       //
       } {
         vgain = vsplat(gain)                        //[P, ]
         n = add(n, #127)                            //
       } {
         w0.w =vmpye(u0.w,vgain.uh)                  //[0, 2]
         u1.tmp = vmem(ptr_x++#1)                    //[0, 2]
         u1.w = vsub(u1.w, voffset.w)                //[0, 2]
         n = lsr(n, #7)                              //[P, 7]
       } {
         w0.w+=vmpyo(u0.w,vgain.h):<<1:rnd:sat:shift //[0, 3]
         if(!p0) n = add(n, #-1)                     //
       } {
         w1.w =vmpye(u1.w,vgain.uh)                  //[0, 4]
         u2.tmp = vmem(ptr_x++#1)                    //[0, 4]
         u2.w = vsub(u2.w, voffset.w)                //[0, 4]
       } {
         w1.w+=vmpyo(u1.w,vgain.h):<<1:rnd:sat:shift //[0, 5]
         loop0(.L_loopN, n)                          //[P,10]
       } 
/* ==================================================================== */
       .balign 32
.L_loopN:
       {
         w2.w =vmpye(u2.w,vgain.uh)                  //[0, 6]
         u3.tmp = vmem(ptr_x++#1)                    //[0, 6]
         u3.w = vsub(u3.w, voffset.w)                //[0, 6]
       } {
         h2h0.h = vpack(w1.w, w0.w):sat              //[0, 7]
         w2.w+=vmpyo(u2.w,vgain.h):<<1:rnd:sat:shift //[0, 7]
       } {
         w3.w =vmpye(u3.w,vgain.uh)                  //[0, 8]
         u0.tmp = vmem(ptr_x++#1)                    //[1, 0]
         u0.w = vsub(u0.w, voffset.w)                //[1, 0]
       } {
         w3.w+=vmpyo(u3.w,vgain.h):<<1:rnd:sat:shift //[0, 9]
       } {
         w0.w =vmpye(u0.w,vgain.uh)                  //[1, 2]
         u1.tmp = vmem(ptr_x++#1)                    //[1, 2]
         u1.w = vsub(u1.w, voffset.w)                //[1, 2]
       } {
         w0.w+=vmpyo(u0.w,vgain.h):<<1:rnd:sat:shift //[1, 3]
         h3h1.h = vpack(w3.w, w2.w):sat              //[0,11] 
       } {
         w1.w =vmpye(u1.w,vgain.uh)                  //[1, 4]
         u2.tmp = vmem(ptr_x++#1)                    //[1, 4]
         u2.w = vsub(u2.w, voffset.w)                //[1, 4]
       } {
         b3b2b1b0.ub = vpack(h3h1.h, h2h0.h):sat     //[0,13]
         vmem(ptr_y++#1) = b3b2b1b0.new              //[0,13]
         w1.w+=vmpyo(u1.w,vgain.h):<<1:rnd:sat:shift //[1, 5]
       }:endloop0 
/*======================================================================*/
       {
         w2.w =vmpye(u2.w,vgain.uh)                  //[1, 6]
         u3.tmp = vmem(ptr_x++#1)                    //[1, 6]
         u3.w = vsub(u3.w, voffset.w)                //[1, 6]
	}{
	 if(p0) jumpr  r31                           //[E, 6]
	}{
       } {
         h2h0.h = vpack(w1.w, w0.w):sat              //[1, 7]
         w2.w+=vmpyo(u2.w,vgain.h):<<1:rnd:sat:shift //[1, 7]
       } {
         w3.w =vmpye(u3.w,vgain.uh)                  //[1, 8]
       } {
         w3.w+=vmpyo(u3.w,vgain.h):<<1:rnd:sat:shift //[1, 9]
       } {
         h3h1.h = vpack(w3.w, w2.w):sat              //[1,11] 
       } {
         b3b2b1b0.ub = vpack(h3h1.h, h2h0.h):sat     //[1,13]
       } {
         if(q0) vmem(ptr_y+#0) = b3b2b1b0            //[1,13]
	}{
         jumpr r31                                   //Q
       }
/*======================================================================*/
.L_end:
      .size quantize_asm, .L_end-quantize_asm
