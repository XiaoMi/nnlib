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
 * This operation does a group copy and scale.
 *  memconvert_hvx( uint8_t * dsto,
 *                  uint8_t const *srco,
 *                int depth,
 *                int offset,
 *				  int16_t gain,
 *                int stride,
 *                int iters.
 * Input is iters*depth elements at 'srco', contiguous;
 * Output is iters groups of depth elements each
 *        group i starts at dsto + i*stride
 *        each group is contiguous
 *
 * The scaling done in each group is
 *     tmp =   ( in[i] + offset) * gain
 *     out[i] = clip_to_u8(   tmp/32K (with rounding))
 *
 *   i.e.
 *      for( int i = 0; i < iters; i++ )
 *         for (int  j = 0; j< depth; j++ )
 *             dsto[ i*stride + j ]  = scale( srco[i*depth + j], offset,gain)
 *
 */
          .global memconvert_hvx
          .type   memconvert_hvx, @function
          .balign 32
memconvert_hvx: 
/* ============================================================================ */
#define dsto        r0 //dest ptr
#define srco        r1 //src ptr
#define length      r2 //depth
#define offset      r3 //offset
#define gain        r4 //gain
#define stride      r5 //stride
#define iters       r6 //num of depths
/* ============================================================================ */
#define src         r6
#define srcalign    r13
#define dstalign    r14
#define mid         r7
#define end         r8
#define sel0        r9
#define kernel      r10
#define sel1        r11
#define c128        r12
#define dst         r15
#define x0          v0
#define x1          v1
#define y0          v2
#define xa          v12
#define vpredp      v3
#define vprede      v4
#define z1z0        v7:6
#define z1          v7
#define z0          v6
#define voffset     v5
#define z3          v9
#define z2          v8
#define vone        v10
#define vzero       v11
#define qprolog     q0
#define qepilog     q1
/* ============================================================================ */
   {
     offset = combine(offset.L, offset.L)
     gain = combine(gain.L, gain.L)
     iters = memw(sp+#0)                  //
     vzero = #0
   } {
     q3 = and(q3, !q3)                    //disable first store
     vone = vnot(vzero)
     sel0 = ##0x01010101                  //position of qprolog
   } {
     loop1(.L_iters, iters)
     voffset = vsplat(offset)
     c128 = #128
     dst = dsto
   } {
     src = srco
     qprolog =vsetq(dsto)                  //qprolog vec predicate __|---
     end = add(length, dsto)               //last byte of block
     dstalign = and(dsto, #127)            //alignment of dst
   } { 
     sel1 = add(sel0, sel0)               //position of modified vec predicates
     qepilog = vsetq(end)                 //setup epilog vec predicate
     srcalign = and(src, #127)            //alignment of src
     end = and(end, #127)                 //alignment of last byte
   } {
     vpredp = vand(qprolog, sel1)         //write prolog pred into vreg
     vprede = vand(qepilog, sel1)         //write epilog pred into vreg
     xa.tmp = vmem(src+#0)                //load first block of input data
     z1z0.b = vshuffoe(vzero.b, xa.b)     //[P
   } 
   .balign 32
.L_iters:
   {
     mid = sub(srcalign, dstalign)        //shift up or down src data
     z0.h = vadd(z0.h, voffset.h)         //[P
     qprolog = or(qprolog, !qepilog)      //modified proglog if no kernel
     dstalign = add(dstalign, length)     //amount of total data
   } {
     z1.h = vadd(z1.h, voffset.h)         //[P
     p1 = cmp.gt(mid, #-1)                //see if we shift down
     vpredp|= vand(qprolog, sel0)         //store modified prolog
     kernel = sub(length, end)            //bytes in loop0
   } {
     z2.h = vmpy(z0.h, gain.h):<<1:rnd:sat//[P
     if(p1) src = add(src, c128)          //if shift up force reload
     kernel = add(kernel, #127)           //round kernel up to 128 nearest
     p2 = cmp.gt(dstalign, #127)          //if > 127 dont use modified prolog
   } {
     z3.h = vmpy(z1.h, gain.h):<<1:rnd:sat//[P
     xa.tmp = vmem(src++#1)               //[0, 0]load next bloc
     z1z0.b = vshuffoe(vzero.b, xa.b)             //[P
     if(!p2) sel1 = sel0                  //dont choose modfied
   } {
     z0.h = vadd(z0.h, voffset.h)         //[0, 1]
     kernel= lsr(kernel, #7)              //kernel in blocks of 128
     qprolog = vand(vpredp, sel1)         //select the qprolog
   } {
     x0.ub = vsat(z3.h, z2.h)        //[P
     z1.h = vadd(z1.h, voffset.h)         //[0, 2]
     qepilog = vand(vprede, sel1)         //choose correct qepilog
   } {
     if(q3) vmem(dst+#0) = y0             //[1, 9]store out epilog data
     z2.h = vmpy(z0.h, gain.h):<<1:rnd:sat//[0, 3]
     loop0(.L_blocks, kernel)             //start main loop
   } {
     dst = dsto
     dsto = add(dsto, stride)
     srco = add(srco, length)
     z3.h = vmpy(z1.h, gain.h):<<1:rnd:sat//[0, 4]
   }
/* ============================================================================ */
   .balign 32
.L_blocks:
   {
     xa.tmp = vmem(src++#1)               //[1, 0]load next bloc
     z1z0.b = vshuffoe(vzero.b, xa.b)             //[P
   } {
     x1.ub = vsat(z3.h, z2.h)        //[0, 6]
     z0.h = vadd(z0.h, voffset.h)         //[1, 1]
   } {
     z1.h = vadd(z1.h, voffset.h)         //[1, 2]
     q3 = or(qepilog, qepilog)
   } {
     y0 = valign(x1, x0, mid)             //[0, 8]align using the offset mid
     x0 = x1                              //[0, 8]reuse x1 in next loop
     z2.h = vmpy(z0.h, gain.h):<<1:rnd:sat//[1, 3]
   } {
     if(!qprolog) vmem(dst++#1) = y0      //[0, 9]do prolog load as part of main loop
     qprolog = vcmp.eq(vone.b, vzero.b)   //[0, 9]and(qprolog, !qprolog) all subsequent prologs true
     z3.h = vmpy(z1.h, gain.h):<<1:rnd:sat//[1, 4]
   }:endloop0
/* ============================================================================ */
   {
     src = srco
     qprolog =vsetq(dsto)                  //qprolog vec predicate __|---
     end = add(length, dsto)               //last byte of block
     dstalign = and(dsto, #127)            //alignment of dst
   } {
     x1.ub = vsat(z3.h, z2.h)             //[1, 6]
     sel1 = add(sel0, sel0)               //position of modified vec predicates
     qepilog = vsetq(end)                 //setup epilog vec predicate
     srcalign = and(src, #127)            //alignment of src
   } {
     end = and(end, #127)                 //alignment of last byte
     vpredp = vand(qprolog, sel1)         //write prolog pred into vreg
     vprede = vand(qepilog, sel1)         //write epilog pred into vreg
   } {
     xa.tmp = vmem(src+#0)                //load first block of input data
     z1z0.b = vshuffoe(vzero.b, xa.b)     //[P
     y0 = valign(x1, x0, mid)             //[1, 8]align for final output
   }:endloop1
   {
     if(q3) vmem(dst+#0) = y0             //[1, 9]store out epilog data
	}{
     jumpr r31
   }
/*==============================================================================*/
.L_end:
      .size memconvert_hvx, .L_end-memconvert_hvx
