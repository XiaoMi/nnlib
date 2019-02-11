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
          .global vmemcpy_asm
          .type   vmemcpy_asm, @function
          .balign 32
vmemcpy_asm: 
/* ============================================================================ */
#define dst         r7 //dest ptr (moved from r0)
#define src         r1 //src ptr
#define length      r2 //num bytes
/* ============================================================================ */
#define srcalign    r4
#define dstalign    r5
#define mid         r6
#define end         r0      // put 'dst+length' in r0, so always ready to return

#define temp        r8
#define nv_dst_m2   r11     //  # of dest vectors, -2
#define nv_src_m2   r12     // # of src vectors, -2
#define len_min_129 r13     // length - 129

#define x0          v0
#define x1          v1
#define y0          v2
#define qprolog     q0
#define qepilog     q1
/*
 *  void * vmemcpy( void * dst, void * src, int length )
 *
 * returns dst + length  (note that memcpy returns dst).
 *
 * 'int length' because 'negative' or zero length values will result in no copies.
 * There are no restrictions on length or alignment of source/dest buffer; other than
 * that they must not overlap.
 *
 * This is written to only read the vectors that contain source bytes, and only write vectors
 * containing part of the dest buffer.
 * HOWEVER: if the 'length' parameter is <=0, the function will read one vector from src
 * and not perform any stores.
 *
 * To avoid reading vectors that do not contain source bytes, the v62+ variant uses conditional
 * vector loads; the v60 variant sometimes uses conditional branches, sometimes adjusts the source
 * pointer to re-read the first or last input vector.
 *
 * We always have 'prolog' and 'epilog' - first and last output vectors; each
 * stores at least one byte.
 *   - The case where there is only one output vector
 *       (and one or two input vectors) is handled separately.
 *   - The case where srcalign=dstalign is also handled separately.
 *   - For the general case,
 *      - write prolog  (based on one or two input vectors)
 *         (one if srcalign < dstalign, 2 if srcalign > dstalign
 *      - inner loop (0 or more, each needing one new source vector)
 *      - epilog; may or may not need to read one more vector for this.
 *
 * The following are found:
 *    srcalign, dstalign:       src & 127, dst & 127
 *    mid                       srcalign-dstalign               Used for 'valign' in loop
 *    nv_dst_m2                 = (dstalign + length-129)>>7    2 less than # dest vectors needed
 *    nv_src_m2                 = (srcalign + length-129)>>7    2 less than # src vectors needed
 *
 * nv_dst_m2, nv_src_m2 can only differ by 1.
 *         mid
 *          0                 nv_src_m2 == nv_dst_m2
 *          > 0               nv_src_m2 == nv_dst_m2 or nv_dst_m2+1
 *          < 0               nv_src_m2 == nv_dst_m2 or nv_dst_m2-1
 *
 *  When nv_dst_m2 == -1, this is a special case (only 1 output vector); which is handled using
 * a single store, and 1 or 2 loads (two when nv_src_m2 == 0, which is only possible when srcalign > dstalign)
 *
 * When nv_dst_m2 >= 0, we have a special case for mid == 0 (both buffers have the same alignment); otherwise
 * the general case:
 *    - prolog reads one vector, and one more when mid > 0.
 *          nv_src_m2_adj = nv_src_m2 - (1 if mid>0; 0 otherwise)
 *    - middle loop: nv_dst_m2 times; each does one vec load and full store.
 *     - the epilog requires a vector load whenever nv_src_m2_adj >= nv_dst_m2.
 */
/* ============================================================================ */
   {
     dst = r0;                              // dst pointer, move to another reg
     srcalign = and(src,#127)
     dstalign = and(r0, #127)
     len_min_129 = add(length,#-129)
   } {
     end = add(r0,length);                  // this will be return value
     nv_dst_m2 = add(dstalign,len_min_129)
     mid = sub(srcalign,dstalign)           // difference in alignment
     qprolog =vsetq(dst)                    //qprolog vec predicate __|---
   } {
     nv_dst_m2 = asr( nv_dst_m2, #7)        // # of dest vecs-2  (>= -1)
     p1 = cmp.gt(dstalign,srcalign)         // is mid < 0?
#if __HEXAGON_ARCH__ >= 62
     qepilog =vsetq2(end)
     x0 =  vmem(src++#1)
   }
#else
     temp = and( end,#127)
     qepilog =vsetq(end)
   } {
     p3 = cmp.eq(temp,#0)
     x0 =  vmem(src++#1)
     if( !p3.new ) jump:t .L_qset0
   } {
     qepilog = or(qepilog,!qepilog)         // must be all 1 if end is aligned
   }
.L_qset0:
#endif
   {
     nv_src_m2 = add(srcalign,len_min_129)
     p2 = cmp.eq(mid,#0)                    // src_align = dst_align?
     p0 = !cmp.gt(nv_dst_m2,#-1)            // true if only one output vector
     if(p0.new) jump:nt .L_dst1             // go handle 1-output-vector case
   }
////--> at least 2 output vectors: 1 prolog, nv_dst_m2 'middle', 1 epilog
   {
     loop0( .L_samealign_loop, nv_dst_m2 )  // set up 'same align' loop
     if(p2) jump:t .L_samealign             // go handle 'same alignment' case
#if __HEXAGON_ARCH__ >= 62
     x1 = x0;
#endif
   }
#if __HEXAGON_ARCH__ >= 62
   {
     loop0( .L_cpyloop, nv_dst_m2 )         // set up general case loop
     if(!p1) nv_src_m2 = add(nv_src_m2,#-128)  // adjust this...
     if(!p1) x1 = vmem(src++#1)             // conditionally load 2nd vector
   }
#else
   {
     if(!p1) src = add(src,#128)            // bump to next if we need extra load
     loop0( .L_cpyloop, nv_dst_m2 )         // set up general case loop
   } {
     if(!p1) nv_src_m2 = add(nv_src_m2,#-128)  // adjust this...
     x1 = vmem(src+#-1)                     // load next (or reload first)
   }
#endif
   {
     p2 = !cmp.gt(nv_dst_m2,#0)             // true if no loops
     nv_src_m2 = asr( nv_src_m2, #7)        // find # src vectors needed
     y0 = valign( x1,x0,mid)                // align data for first store
     x0 = x1;
   } {
     // store prolog, and skip loop where applicable
     p0 = cmp.gt( nv_dst_m2, nv_src_m2 )    // when true, don't read vector for epilog.
     if( !qprolog )vmem(dst++#1) = y0
     if( p2 ) jump:nt .L_endcpy
   }
   .balign 32
.L_cpyloop:
   {
     x1.cur = vmem(src++#1)
     y0  =valign( x1,x0,mid);
     x0 = x1;
   } {
     vmem(dst++#1) = y0;
   } :endloop0
.L_endcpy:
    // optional read to feed epilog
#if __HEXAGON_ARCH__ >= 62
   {
     if(!p0) x1 = vmem(src+#0)
   } {
     y0  =valign( x1,x0,mid);
   }
#else
   {
     if(p0) src = add(src,#-128)            // avoid over-read: back up one vec
   } {
     x1.cur = vmem(src+#0)
     y0  =valign( x1,x0,mid)
   }
#endif
   {
     if( qepilog) vmem(dst+#0) = y0;
     jumpr r31;
   }
/////////////////////////////////////
//
// Same-alignment src & dest case
//
.L_samealign:
   {
     p2 = !cmp.gt(nv_dst_m2,#0)
     if(!qprolog) vmem(dst++#1) = x0;       // store first
     if(p2.new) jump:nt .L_samealign_0      // skip if no loops
     x0 = vmem(src++#1)
   }
    // common alignment loop
    .balign 32
.L_samealign_loop:
   {
     vmem(dst++#1) = x0;
     x0 = vmem(src++ #1)
   } :endloop0
.L_samealign_0:
   {
     if(qepilog) vmem(dst) = x0;
     jumpr r31;
   }
/////////////////////////////////////
// one vector output cases
// Need to read an extra vector if (and only if) nv_src_m2 >=0.
//
// Here we detect length <=0 (p1 and p0 will both be false).
//
.L_dst1:
   {
     p1 = cmp.gt(length,#0)					// any at all?
     p0 = cmp.gt(nv_src_m2,#-1)             // if srcalign + length-129 >= 0, need 2nd source vector
     x1 = x0
#if __HEXAGON_ARCH__ >= 62   // use conditional vector load
   } {
     if( p0 ) x1 = vmem(src +#0)
   }
#else
     if( !p0.new) jump:t .L_dst1_skiprd
   } {
     x1 = vmem(src +#0)
   }
.L_dst1_skiprd:
#endif
   {
     if( !p1 ) jumpr r31					// done if length <= 0
     qprolog = or(qprolog, !qepilog )       // combine epilog -> prolog
     y0 = valign( x1,x0, mid)
   } {
     if( !qprolog) vmem(dst) = y0;
     jumpr r31;
   }
.L_end:
/*==============================================================================*/
      .size vmemcpy_asm, .L_end-vmemcpy_asm
