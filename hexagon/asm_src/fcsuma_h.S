
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
    take aligned vector non multiple of 32, pad and also compute sum
 */

        .text
        .global fcsuma_asm
        .type fcsuma_asm,@function
        .p2align 6
fcsuma_asm:

#define  ptr_in   r0       
#define  width    r1       
#define  ptr_sum  r2
#define  ws       r5
#define  spare    r6
#define  c1111    r7

#define  vzero    v0
#define  d0       v2
#define  d1       v3
#define  s1_s0    v5:4
#define  s1       v5
#define  s0       v4

       {
         spare = and(width, #0x7f)        //  _____111
         width = lsr(width, #7)           // multiple of 128
       } {
         ws = #-4                         //word deal 
         q0 = vsetq(spare)                //  _____111
         p0 = cmp.eq(spare, #0)           //it is a multple of 128 
         loop0(.L_sumnpad, width)         //
       } {
         p1 = cmp.eq(width, #0)           //less than 128
         s0 = #0                          //init sum
       } {
         vzero = #0                       //
         c1111 = #-1                      //sum bytes to word and neg
         if(p1) jump .L_spare_only        //
       }
       .balign 32
.L_sumnpad:
       { d1 = vmemu(ptr_in++#1)           //
       } {
         nop                              //
       } {
         s0.w += vrmpy(d1.ub, c1111.b)    //
       }:endloop0
.L_spare_only:
       { loop0(.L_sum, #5)                //
         d1 = vmemu(ptr_in+#0)            //
       } {
         d1 = vmux(q0, d1, vzero)         //
       } {
         if( p0)c1111 = #0                //
       } {
         s0.w += vrmpy(d1.ub, c1111.b)    //
       }
.L_sum:
       { s1_s0 = vdeal(s0, s0, ws)        //
       } {
         s0.w = vadd(s0.w, s1.w)          //
       }:endloop0
       { vmem(ptr_sum+#0) = s0            //
         jumpr r31                        //
       }
.L_end:
     .size fcsuma_asm,.L_end-fcsuma_asm

