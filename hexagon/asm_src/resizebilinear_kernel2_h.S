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
/*  FUNCTIONS      : resizebilinear_kernel2                             */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  KG                  07/12/17       created                          */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*                                                                      */
/*     ->                                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =  bytes                                                */
/*     ASSUMPTIONS                                                      */
/*  C MODEL                                                             */
/*======================================================================*/
#define LOG2VLEN       7
#define VLEN           (1<<LOG2VLEN)
#define pSrc0          r0
#define pDst0          r1
#define b_in           r2
#define newheight      r3
#define newwidth       r4
#define d_in           r5
#define negd_in        r6
#define const2w        r7
#define pSrc1          r8
#define pDst1          r9
#define h_in           r10
#define w_in           r11
#define lp0cnt         r12
#define h_iter         r13
#define b_iter         r14
#define vostride       r15
#define const1b        R28

#define sL0cur         v0
#define sL1cur         v1
#define dL10cur        v1:0
#define sL0nxt         v2
#define sL1nxt         v3
#define sL0s1          v4
#define sL1s1          v5
#define dL10s1         v5:4
#define sAcc_L         v6
#define sAcc_H         v7
#define dAcc           v7:6
#define sL0Odd         v8
#define sL1Odd         v9
#define sL1Even        v10
#define sOut0_L        v12
#define sOut0_H        v13
#define dOut0          v13:12
#define sOut1_L        v14
#define sOut1_H        v15
#define dOut1          v15:14

    .text
    .globl    resizebilinear_kernel2
    .falign
    .type    resizebilinear_kernel2,@function
resizebilinear_kernel2:
    { h_in = lsr(newheight,#1)                      //
      w_in = lsr(newwidth,#1)                       //
      r9:8 = r1:0                                   // pSrc1 = pSrc0; pDst1 = pDst0;
      r7 = #2                                       //
    }{
      pSrc1 += mpyi(w_in,d_in)                      //
      pDst1 += mpyi(newwidth,d_in)                  //
      negd_in = sub(#128,d_in)                      //
    }{
      const1b = ##0x1010101                         //
      q0 = vsetq(negd_in)                           //
      lp0cnt = mpyi(w_in,d_in)                      //
    }{
      lp0cnt = lsr(lp0cnt,#LOG2VLEN)                //
      vostride = add(lp0cnt,lp0cnt)                 //
      negd_in = add(negd_in,#-128)                  //
    }{
      lp0cnt = add(lp0cnt,#-1)                      //
      b_iter = #0                                   //
    }
.rb_lp2:
    {
      loop1(.rb_lp1,h_in)                           //
      b_iter = add(b_iter,#1)                       //
      h_iter = #1                                   //
    }
    .falign
.rb_lp1:
    {
      sL0cur = vmem(pSrc0++#1)                      //
      h_iter = add(h_iter,#1)                       //
      p2 = cmp.gt(h_in,h_iter)                      //
      if (!p2.new) pSrc1 = pSrc0                    //
    }{
      sL1cur = vmem(pSrc1++#1)                      //
      p3 = sp1loop0(.rb_lp0,lp0cnt)                 //
      p2 = cmp.gt(lp0cnt,#0); if (!p2.new) jump:nt .rb_lp0end//
    }
    .falign
.rb_lp0:
    {
      sL0nxt.cur = vmem(pSrc0++#1)                  //
      sL0s1 = valign(sL0nxt,sL0cur,d_in)            //
    }{
      sL1nxt.cur = vmem(pSrc1++#1)                  //
      sL1s1 = valign(sL1nxt,sL1cur,d_in)            //
      dAcc.h = vmpa(dL10cur.ub,const1b.b)           //
    }{
      sL0Odd.ub = vavg(sL0cur.ub,sL0s1.ub):rnd      //
      if p3 vmem(pDst0++#1) = sOut0_H               //[2]
      dOut1 = vshuff(sL1Odd,sL1Even,negd_in)        //[2]
    }{
      dAcc.h += vmpa(dL10s1.ub,const1b.b)           //
      if p3 vmem(pDst1++#1) = sOut1_L               //[2]
    }{
      dOut0 = vshuff(sL0Odd,sL0cur,negd_in)         //
      sL1Even.ub = vavg(sL0cur.ub,sL1cur.ub):rnd    //
      if p3 vmem(pDst1++#1) = sOut1_H               //[2]
    }{
      sL1Odd.ub = vasr(sAcc_H.h,sAcc_L.h,const2w):rnd:sat//
      vmem(pDst0++#1) = sOut0_L                     //
      sL1cur = sL1nxt                               //
      sL0cur = sL0nxt                               //
    }:endloop0
.rb_lp0end:
    {
      sL0s1 = valign(sL0nxt,sL0cur,d_in)            //
      if p3 vmem(pDst0++#1) = sOut0_H               //[2]
    }{
      sL1s1 = valign(sL1nxt,sL1cur,d_in)            //
      sL0s1 = vmux(Q0,sL0s1,sL0cur)                 //
    }{
      sL0Odd.ub = vavg(sL0cur.ub,sL0s1.ub):rnd      //
      sL1s1 = vmux(Q0,sL1s1,sL1cur)                 //
      dOut1 = vshuff(sL1Odd,sL1Even,negd_in)        //[2]
      if p3 vmem(pDst1++#1) = sOut1_L.new           //[2]
    }{
      dAcc.h = vmpa(dL10cur.ub,const1b.b)           //
      sL1Even.ub = vavg(sL0cur.ub,sL1cur.ub):rnd    //
      if p3 vmem(pDst1++#1) = sOut1_H               //[2]
    }{
      dOut0 = vshuff(sL0Odd,sL0cur,negd_in)         //
      dAcc.h += vmpa(dL10s1.ub,const1b.b)           //
    }{
      vmem(pDst0++#1) = sOut0_L                     //
    }{
      sL1Odd.ub = vasr(sAcc_H.h,sAcc_L.h,const2w):rnd:sat//
      vmem(pDst0++#1) = sOut0_H                     //
    }{
      dOut1 = vshuff(sL1Odd,sL1Even,negd_in)        //
      vmem(pDst1++#1) = sOut1_L.new                 //
    }{
      vmem(pDst1+#0) = sOut1_H                      //
      pDst0 = add(pDst0,vostride)                   //
      pDst1 += add(vostride,#VLEN)                  //
    }:endloop1
    {
      p0 = cmp.gt(b_in,b_iter); if (p0.new) jump:t .rb_lp2//
    }{
      jumpr r31                                     //
    }
.resizebilinear_kernel2_end:
    .size    resizebilinear_kernel2, .resizebilinear_kernel2_end-resizebilinear_kernel2


