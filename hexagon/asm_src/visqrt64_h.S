/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
    Perform inverse square root on 64 elements based on qf16 type.
*/
#if 0
void isqrt_cn(
  short *ptr_xm, short *ptr_ein, short *ptr_ym, short *ptr_ye)
{
  int i;
  short u, x, mant, expnt, y, y2, range, isqrt2 = 0x5a83;

        u = ptr_xm[0];
        if(u<=0)u = 1;
        expnt = norm16(u);
        mant = u << expnt;
        range = (mant >> 10) & 0xf;
        x = (mant<<1) & 0x7ff;
        y =                 lut_isqrt_cn[range+2*16];
        y = mpyrsat(y, x) + lut_isqrt_cn[range+1*16];
        y = mpyrsat(y, x) + lut_isqrt_cn[range+0*16];
        y2 = mpyrsat(isqrt2, y);
        if(!(expnt & 1)) y = y2;
        expnt = 15-expnt;
        ptr_ye[0] = (expnt>>1)-1;  
        ptr_ym[0] = y;
  return;
}
#endif
/*======================================================================*/
          .global visqrt64_asm
          .balign 32
          .type  visqrt64_asm, @function
visqrt64_asm:
/*======================================================================*/
#define ptr_xm    r0 //ptr to input data
#define ptr_expin r1 //qpoint of input data
#define ptr_ym    r2 //mantissa of output data
#define ptr_ye    r3 //corresponding exponents of output data
#define ptr_isqrt r4 //.global lut_isqrt_asm
/*======================================================================*/
#define c3        r6
#define c3c       r5
#define cf        r8
#define c0        r7
#define c2        r9
#define sqrt0p5   r10
#define c1        r12
#define c15       r13
#define c30       r14
#define isqrt0    v0
#define isqrt1    v1
#define isqrt2    v2
#define vcf       v4
#define vc7ff     v16
#define d0        v5
#define d1        v6
#define i1i0      v7
#define qexp0     v8
#define qexp1     v9
#define c21_c20   v11:10
#define c21       v11
#define     c20   v10
#define c11_c10   v13:12
#define c11       v13
#define     c10   v12
#define c01_c00   v15:14
#define c01       v15
#define     c00   v14
#define vzero     v19
#define vone      v17
#define y1y0      v21:20
#define y1        v21
#define y0        v20
#define ny0       v22
#define ny1       v23
#define vc15      v25
#define vc30      v26
#define x1        v27
#define x0        v24
#define bit1      v29
#define bit0      v28
/*======================================================================*/
       {
         c1 = ##0x40004000
         vzero = #0
       } {
         d0.cur = vmem(ptr_xm+#0)                 //[0, 1]
         q0 = vcmp.gt(d0.h, vzero.h)              //[0, 1]
         c30 = ##0x001d001d
       } {
         qexp0 = vmem(ptr_expin+#0)               //[0, 2]
         vone = vsplat(c1)
         vc30 = vsplat(c30)
         c1.L = #0x0001
       } {
         c3c.L= #0x3c3c
         c1.H = #0x0001
         d0 = vmux(q0, d0, vone)                  //[0, 3]
         qexp0 = vmux(q0, qexp0, vc30)            //[0, 3]
       } {
         c3c.H= #0x3c3c
         vone = vsplat(c1)
         i1i0.b = vshuffo(d1.b, d0.b)             //[0, 4]
       } {
         bit0 = vand(vone, qexp0)                 //[0, 5]
         vcf = vsplat(c3c)
       } {
         cf = ##0x07ff07ff
         qexp0.h = vsub(vzero.h, qexp0.h)
         i1i0    = vand(i1i0,vcf)                 //[0, 6]
       } {
         c2 = #2
         vc7ff = vsplat(cf)
         d0.h = vadd(d0.h, d0.h)                  //[0, 7]
       } {
         isqrt2 = vmem(ptr_isqrt+#2)
         i1i0.uh = vlsr(i1i0.uh, c2)              //[0, 8]
         c0 = #0
       } {
         x0 = vand(d0, vc7ff)                     //[0,9]
         qexp0.h = vasr(qexp0.h, c1)              //[0,9]
       } {
         c21_c20.h = vlut16(i1i0.b, isqrt2.h, c0) //[0,10]
       } {
         qexp0.h = vsub(qexp0.h, vone.h)          //[0,11]
         vmem(ptr_ye++#1) = qexp0.new             //[0,11]
       } {
         isqrt1.tmp = vmem(ptr_isqrt+#1)
         c11_c10.h = vlut16(i1i0.b, isqrt1.h, c0) //[0,12]
         y0.h = vmpy(x0.h, c20.h):<<1:rnd:sat     //[0,12]
       } {
         y0.h = vadd(y0.h, c10.h)                 //[0,13]
         q0  = vcmp.eq(vzero.h, bit0.h)           //[0,13]
       } {
         isqrt0.tmp = vmem(ptr_isqrt+#0)
         c01_c00.h = vlut16(i1i0.b, isqrt0.h, c0) //[0,14]
       } {
         y0.h = vmpy(x0.h, y0.h):<<1:rnd:sat      //[0,15]
       } {
         y0.h = vadd(y0.h, c00.h)                 //[0,16]
       } {
         sqrt0p5 = ##0x5a835a83                   //1/sqrt(2)
       } {
         ny0.h = vmpy(y0.h, sqrt0p5.h):<<1:rnd:sat//[0,18]
       } {
         y0 = vmux(q0, y0, ny0)                   //[0,19]
         vmem(ptr_ym+#0) = y0.new                 //[0,19]
       }
         jumpr r31
/*----------------------------------------------------------------*/
.L_end:
      .size visqrt64_asm, .L_end-visqrt64_asm
/*======================================================================*/
/*                            end of file                               */
/*======================================================================*/
