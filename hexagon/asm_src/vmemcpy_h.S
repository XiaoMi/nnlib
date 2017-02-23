
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
          .global vmemcpy_asm
          .type   vmemcpy_asm, @function
          .balign 32
vmemcpy_asm: 
/* ============================================================================ */
#define dst         r0 //dest ptr
#define src         r1 //src ptr
#define length      r2 //num bytes
/* ============================================================================ */
#define srcalign    r4
#define dstalign    r5
#define mid         r6
#define end         r7
#define sel0        r8
#define kernel      r3
#define sel1        r9
#define dsto        r10
#define x0          v0
#define x1          v1
#define y0          v2
#define vpredp      v3
#define vprede      v4
#define qprolog     q0
#define qepilog     q1
/* ============================================================================ */
   { sel0 = ##0x01010101              //position of qprolog
     qprolog =vsetq(dst)              //qprolog vec predicate __|---
     end = add(length, dst)           //last byte of block
   } {
     dstalign = and(dst, #127)        //alignment of dst
     sel1 = add(sel0, sel0)           //position of modified vec predicates
     qepilog = vsetq(end)             //setup epilog vec predicate
     dsto = add(dst, length)          //updated destination ptr
   } {
     srcalign = and(src, #127)        //alignment of src
     end = and(end, #127)             //alignment of last byte
     vpredp = vand(qprolog, sel1)     //write prolog pred into vreg
     vprede = vand(qepilog, sel1)     //write epilog pred into vreg
   } {
     mid = sub(srcalign, dstalign)    //shift up or down src data
     dstalign = add(dstalign, length) //amount of total data
     kernel = sub(length, end)        //bytes in loop0
     qprolog = or(qprolog, !qepilog)  //modified proglog if no kernel
   } {
     vpredp|= vand(qprolog, sel0)     //store modified prolog
     p2 = cmp.gt(dstalign, #127)      //if > 127 dont use modified prolog
     if(!p2.new) sel1 = sel0          //dont choose modfied
     kernel = add(kernel, #127)       //round kernel up to 128 nearest
   } {
     x0 = vmem(src+#0)                //load first block of input data
     kernel= lsr(kernel, #7)          //kernel in blocks of 128
     qprolog = vand(vpredp, sel1)     //select the qprolog
     p1 = cmp.gt(mid, #-1)            //see if we shift down
   } {
     qepilog = vand(vprede, sel1)     //choose correct qepilog
     if(p1) src = add(src, #128)      //if shift up force reload
     loop0(.L_blocks, kernel)         //start main loop
   }
/* ============================================================================ */
   .balign 32
.L_blocks:
   { x1.tmp = vmem(src++#1)           //load next bloc
     y0 = valign(x1, x0, mid)         //align using the offset mid
     x0 = x1                          //reuse x1 in next loop
   } {
     if(!qprolog) vmem(dst++#1) = y0  //do prolog load as part of main loop
     qprolog = and(qprolog, !qprolog) //make all subsequent prologs true
   }:endloop0
/* ============================================================================ */
   { x1.tmp = vmem(src+#0)            //load next or reload data
     y0 = valign(x1, x0, mid)         //aligne for final output
   } {
     if(qepilog) vmem(dst+#0) = y0    //store out epilog data
     r0 = dsto                        //return updated pointer
	}{
     jumpr r31                        //return to caller
   }
.L_end:
/*==============================================================================*/
      .size vmemcpy_asm, .L_end-vmemcpy_asm
