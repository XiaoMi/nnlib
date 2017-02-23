
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
/*  FUNCTIONS      : gemmpybbw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform gemm matrix vector multiply, result  32bits  */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*  DJH                 05/10/16       added post add for x and y offset*/
/*  DJH                 07/10/16       rewrote to do mat-vec mult       */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->  5*K/512+24                                                   */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 1040 bytes                                            */
/*     STACK    = 48 bytes                                              */
/*     ASSUMPTIONS                                                      */
/*        y and z are 128 byte aligned                                  */
/*        x is16byte aligned                                            */
/*        K%16=0 M%32=0                                                 */
/*  C MODEL                                                             */
/*======================================================================*/
#if 0
void gemvmpybbw_cn(
   uint8 * a, int a_off, uint8 * b, int b_off, int * c, int M, int K) {
    int i, j, k;
    int32 sum;
    uint8 a_val, b_val;

    for (j=0; j < M; j++) {
            sum = 0;
            for (k=0; k < K; k++) {
              a_val = a[k];
              b_val = b[k*M+j];
              sum  += a_val * b_val ;
            }
            c[j] = sum;
        }
    }
    return;
}
#endif
/*=============================================================================*/
        .text
        .file "gemvmpybbw_h.S"
        .global gemvmpybbw_asm
        .balign 32
        .type  gemvmpybbw_asm, @function
gemvmpybbw_asm:

/*=============================================================================*/
#define ptr_a         r0     //data
#define a_offset      r1 
#define ptr_be        r2     //weights must be pre processwed and transposed
#define b_offset      r3
#define ptr_c         r4     //results
#define m             r5     //can be < 32 will write less
#define k             r6     //k % 16
/*=============================================================================*/
#define ki            r7     //
#define ptr_bo        r8     //
#defne  c0101         r9
#define sum1sum0      r15:14
#define sum1          r15
#define sum0          r14
#define c0101         r16
#define mask          r17
#define x7x4x3x0      r11:10 //
#define xfxcxbx8      r13:12 //
#define x3x0          r10    //
#define x7x4          r11    //
#define xbx8          r12    //
#define xfxc          r13    //
/*=============================================================================*/
#define y0            v0     //
#define y1            v1     //
#define y2            v2     //
#define y3            v3     //
#define bsum          v5
#define z0            v4     //
#define asum          v6
#define va_offset     v7
/*=============================================================================*/
       {
           k = memw(sp+#0)
           allocframe(#16)
       } {
           memd(sp+#0) = r17:16
           mask = #127
           m = asl(m, #2)                    //ints
       } {
           q3 = vsetq(m)                     //
           p0 = bitsclr(m, mask)
           if(!p0.new) jump:nt .L_32          //
       } {
           q3 = vcmp.eq(v0.b, v0.b)          //enable all bits
       }
/*============================================================================*/
       .balign 32
.L_32:
       {
           ptr_bo = add(ptr_be, #128)        //[ , P]
           ki = lsr(k, #4)                   //k / 16
       } {
           dcfetch(ptr_a+#4<<5)              //[0, 0]prefetch next line
           ki = add(ki, #-1)                 //
       } {
           c0101 = ##0x01010101         
           x7x4x3x0 = memd(ptr_a++#8)        //[0, 1]
           loop0(.L_loopK, ki)               //[P, 9]ki is k1/4 - 2
       } {
           y0.tmp = vmem(ptr_be++#2)         //[0, 2]32x4
           z0.uw = vrmpy(y0.ub, x3x0.ub)     //[0, 2]
           bsum.uw = vrmpy(y0.ub, c0101.ub)  //[0, 2]
           sum0 = #0
       } {
           y1.tmp = vmem(ptr_bo++#2)         //[0, 3]32x4
           z0.uw += vrmpy(y1.ub, x7x4.ub)    //[0, 3]
           xfxcxbx8 = memd(ptr_a++#8)        //[0, 3]
           bsum.uw += vrmpy(y1.ub, c0101.ub) //[0, 3]
       } {
           y2.tmp = vmem(ptr_be++#2)         //[0, 4]32x4
           z0.uw += vrmpy(y2.ub, xbx8.ub)    //[0, 4]
           bsum.uw += vrmpy(y2.ub, c0101.ub) //[0, 4]
           sum1 = #0
       }
/*============================================================================*/
       .balign 32
.L_loopK:
       {
           y3.tmp = vmem(ptr_bo++#2)         //[0, 5]32x4
           z0.uw += vrmpy(y3.ub, xfxc.ub)    //[0, 5]
           bsum.uw += vrmpy(y3.ub, c0101.ub) //[0, 5]
           dcfetch(ptr_a+#4<<5)              //[1, 0]prefetch next line
       } {
           sum1sum0+=vraddub(xfxcxbx8,x7x4x3x0)//[0,6]
           x7x4x3x0 = memd(ptr_a++#8)        //[1, 1]
       } {
           y0.tmp = vmem(ptr_be++#2)         //[1, 2]32x4
           z0.uw += vrmpy(y0.ub, x3x0.ub)    //[1, 2]
           bsum.uw += vrmpy(y0.ub, c0101.ub) //[1, 2]
       } {
           y1.tmp = vmem(ptr_bo++#2)         //[1, 3]32x4
           z0.uw += vrmpy(y1.ub, x7x4.ub)    //[1, 3]
           xfxcxbx8 = memd(ptr_a++#8)        //[1, 3]
           bsum.uw += vrmpy(y1.ub, c0101.ub) //[1, 3]
       } {
           y2.tmp = vmem(ptr_be++#2)         //[1, 4]32x4
           z0.uw += vrmpy(y2.ub, xbx8.ub)    //[1, 4]
           bsum.uw += vrmpy(y2.ub, c0101.ub) //[1, 4]
       }:endloop0 
/*=============================================================================*/
       {   y3.tmp = vmem(ptr_bo++#2)         //[1, 5]32x4
           z0.uw += vrmpy(y3.ub, xfxc.ub)    //[1, 5]
           bsum.uw += vrmpy(y3.ub, c0101.ub) //[1, 5]
       } {
           sum1sum0+=vraddub(xfxcxbx8,x7x4x3x0)//[1,6]
       } {
           sum0 = add(sum0, sum1)
       } {
           sum0 = mpyi(sum0, b_offset)
           sum1 = mpyi(a_offset, b_offset)
       } {
           sum0 += mpyi(sum1, k)
           a_offset = combine(a_offset.L, a_offset.L)
       } {
           asum = vsplat(sum0)
           va_offset = vsplat(a_offset)
       } {
           z0.w = vadd(z0.w, asum.w)
           bsum.w = vmpyio(bsum.w, va_offset.h)
       } {
           z0.w = vadd(z0.w, bsum.w)
           r17:16 = memd(sp+#0) 
       } {
           if(q3) vmem(ptr_c+#0) = z0        //[E,16]
       } {
           dealloc_return
       }
.L_end:
/*=============================================================================*/
      .size gemvmpybbw_asm, .L_end-gemvmpybbw_asm
