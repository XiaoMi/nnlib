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
 */

/* ============================================================================ */
          .global vmemset_nt_asm
          .type   vmemset_nt_asm, @function
          .balign 32
vmemset_nt_asm: 
/* ============================================================================ */
#define dst         r0
#define src         r1
#define length      r2
/* ============================================================================ */
#define dstalign    r5
#define end         r7
#define sel0        r8
#define kernel      r3
#define sel1        r9
#define sel2        r4
#define dsto        r10
#define y0          v2
#define vpredp      v3
#define vprede      v4
#define qprolog     q0
#define qepilog     q1
/* ============================================================================ */
   { sel0 = ##0x01010101              //position of qprolog
     src = vsplatb(src)
     end = add(length, dst)           //last byte of block
   } {
     qprolog =vsetq(dst)              //qprolog vec predicate __|---
     y0 = vsplat(src)
     sel1 = add(sel0, sel0)           //position of modified vec predicates
     end = and(end, #127)             //alignment of last byte
   } {
     dstalign = and(dst, #127)        //alignment of dst
     qepilog = vsetq(end)             //setup epilog vec predicate
     vpredp = vand(qprolog, sel1)     //write prolog pred into vreg
     length -= add(end, #-127)        //round kernel up to 128 nearest
   } {
     vprede = vand(qepilog, sel1)     //write epilog pred into vreg
     qprolog = or(qprolog, !qepilog)  //modified proglog if no kernel
     length= lsr(length, #7)          //kernel in blocks of 128
     dstalign = add(dstalign, length) //amount of total data
   } {
     vpredp|= vand(qprolog, sel0)     //store modified prolog
     loop0(.L_blocks, length)         //start main loop
     p2 = cmp.gt(dstalign, #127)      //if > 127 dont use modified prolog
     if(!p2.new) sel1 = sel0          //dont choose modfied
   } {
     qprolog = vand(vpredp, sel1)     //select the qprolog
     qepilog = vand(vprede, sel1)     //choose correct qepilog
   }
/* ============================================================================ */
   .balign 32
.L_blocks:
   { 
     if(!qprolog) vmem(dst++#1):nt = y0  //do prolog load as part of main loop
     qprolog = and(qprolog, !qprolog) //make all subsequent prologs true
   }:endloop0
/* ============================================================================ */
   { 
     if(qepilog) vmem(dst+#0):nt = y0 //store out epilog data
   }{
     jumpr r31                        //return to caller
   }
.L_end:
/* ============================================================================ */
      .size vmemset_nt_asm, .L_end-vmemset_nt_asm
/* ============================================================================ */
