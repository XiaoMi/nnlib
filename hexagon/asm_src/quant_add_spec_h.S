/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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
  Behavioral Model
  ----------------
  speculative add two streams with guess of max and min
 */
#if 0
void quant_add_spec_casm(uint8_t *aq, float amax, float amin,
                         uint8_t *bq, float bmax, float bmin,
                         float gmax, float gmin,   //guess
                         uint8_t * cq, float *cmax, float *cmin, int length)
{
   float stepa = amax-amin;
   float stepb = bmax-bmin;
   float step, lmin, lmax;
   float alpha  = stepa/stepb;
   short ialpha = 128.0*alpha;
   float kappa  = 128.0*alpha +(255.0*amin + 255.0*bmin)/stepb ;
   short ikappa = (int) (kappa+.0); //+ialpha is because input is 08 ^

   //compute local max,min by updating local
   lmin = (gmin * 255.0)/stepb;
   lmax = (gmax * 255.0)/stepb;
   step = lmax - lmin;

   float frecip = (2.0 * 255.0 * 32768.0) / step;
   float foffset = (2.0 * 255.0 * lmin) / step;
   short recip = (int) (frecip +0.0);
   short offset = (int) (foffset-1.0) ;
   quant_add_spec_asm(aq, bq, ialpha, ikappa, offset, recip, cq, ptr_max, length);

   lmax = (float)ptr_max[0];
   lmin = (float)ptr_max[64];
   //turn back to global max
   *cmin = (lmin*stepb)/255.0;
   *cmax = (lmax*stepb)/255.0;
   return;
}
#endif
/*
     Methods+Notes
     -------------
      for(i=0; i < length; i++) {
         a = aq[i];
         c = ((ialpha * a<<8 + 0x4000)>>7) + bq[i] + ikappa;
         if(c > omax) omax = c;
         if(c < omin) omin = c;
         int oval = ((tmp[i]-imin) * recip + 0x4000)>>15;
         if(oval > 255) cq[i] = 255; else if (oval < 0) cq[i] = 0; else cq[i] = (uint8_t) oval;
      }
 */
/* -------------------------------------------------------------*/
         .text
         .file "quant_add_spec_h.S"
         .global quant_add_spec_asm
         .balign 32
         .type quant_add_spec_asm, @function
/* -------------------------------------------------------------*/
quant_add_spec_asm:
/* -------------------------------------------------------------*/
#define ptr_a         r0    //pointer to input data
#define ptr_b         r1    //
#define alpha         r2    //
#define kappa         r3    //pointer to input zero
#define offset        r4
#define recip         r5
#define ptr_c         r9
#define ptr_minmax    r8    //pointer to output unquantized output
#define length        r6    //
#define c2            r7
/* -------------------------------------------------------------*/
/* -------------------------------------------------------------*/
#define xh1xh0        v9:8
#define yh1yh0        v7:6
#define xh1           v9
#define xh0           v8
#define yh1           v7
#define yh0           v6
#define wh1wh0        v5:4
#define wh1           v5
#define wh0           v4
#define zh1zh0        v15:14
#define zh1           v15
#define zh0           v14
#define uh1uh0        v21:20
#define uh1           v21
#define uh0           v20
#define xb1xb0        v16
#define yb1yb0        v17
#define zb1zb0        v18
#define mino_mine     v11:10
#define maxo_maxe     v13:12
#define mino          v11
#define mine          v10
#define maxo          v13
#define maxe          v12
#define maxe0         v22
#define vzero         v3
#define vkappa        v0
#define vc80          v1
#define valpha        v2
#define voffset       v19
#define VEXP          <<1:rnd:sat
/* -------------------------------------------------------------*/
   {
       r10 = ##0x7fff7fff
       r11 = ##0x80808080
   } {
       vzero = #0
       mine = vsplat(r10)
       ptr_c = memw(sp+#0<<2)
       vc80 = vsplat(r11)
   } {
       maxe.h = vsub(vzero.h, mine.h)
       xb1xb0.tmp = vmem(ptr_a++#1)                //[0, 0]
       xb1xb0 = vxor(xb1xb0, vc80)                 //[0, 0]
       length= memw(sp+#2<<2)                 
   } {
       alpha = combine(alpha.L, alpha.L)
       kappa = combine(kappa.L, kappa.L)
       xh1xh0.b = vshuffoe(xb1xb0.b, vzero.b)      //[0, 1]
       length = add(length, #-1)                   // round down
   } {
       vkappa = vsplat(kappa)
       yb1yb0.tmp = vmem(ptr_b++#1)                //[0, 2]
       yh1yh0.b = vshuffoe(vzero.b, yb1yb0.b)      //[0, 2]
       length = lsr(length, #7)                    // truncate off bits
   } {
       maxe0 = maxe
       xh0.h = vmpy(xh0.h, alpha.h):VEXP           //[0, 3]
       wh0.h = vadd(yh0.h, vkappa.h)               //[0, 3]
       loop0(.L_quant, length)                     //[P, 5]
   } {
       offset = combine(offset.L, offset.L)
       recip = combine(recip.L, recip.L)
       xh1.h = vmpy(xh1.h, alpha.h):VEXP           //[0, 4]
       zh0.h = vadd(xh0.h, wh0.h)                  //[0, 4]
   } {
       voffset = vsplat(offset)                    //[P, 5]
       wh1.h = vadd(yh1.h, vkappa.h)               //[0, 5]
       maxe0.h = vmax(maxe0.h, zh0.h)              //[0, 5]
       p0 = cmp.eq(length,#0) ; if (p0.new) jump:nt .Loopbot
   } 
/* -------------------------------------------------------------*/
       .balign 32
.L_quant:
   {   uh0.h = vmpy(zh0.h, recip.h):VEXP           //[0, 6]
       zh1.h = vadd(xh1.h, wh1.h)                  //[0, 6]
       xb1xb0.tmp = vmem(ptr_a++#1)                //[1, 0]
       xb1xb0 = vxor(xb1xb0, vc80)                 //[1, 0]
   } {
       mine.h = vmin(mine.h, zh0.h)                //[0, 7]
       uh0.h = vsub(uh0.h, voffset.h)             //[0, 7]
       xh1xh0.b = vshuffoe(xb1xb0.b, vzero.b)      //[1, 1]
   } {
       uh1.h = vmpy(zh1.h, recip.h):VEXP           //[0, 8]
       yb1yb0.tmp = vmem(ptr_b++#1)                //[1, 2]
       yh1yh0.b = vshuffoe(vzero.b, yb1yb0.b)      //[1, 2]
   } {
       uh1.h = vsub(uh1.h, voffset.h)             //[0, 9]
       xh0.h = vmpy(xh0.h, alpha.h):VEXP           //[1, 3]
       wh0.h = vadd(yh0.h, vkappa.h)               //[1, 3]
   } {
       zb1zb0.ub = vsat(uh1.h, uh0.h)              //[0,10]
       vmem(ptr_c++#1) = zb1zb0.new                //[0,10]
       xh1.h = vmpy(xh1.h, alpha.h):VEXP           //[1, 4]
       zh0.h = vadd(xh0.h, wh0.h)                  //[1, 4]
   } {
       maxe.h = vmax(maxe.h, zh1.h)                //[0,11]
       mine.h = vmin(mine.h, zh1.h)                //[0,11]
       wh1.h = vadd(yh1.h, vkappa.h)               //[1, 5]
       maxe0.h = vmax(maxe0.h, zh0.h)              //[1, 5]
   }:endloop0
.Loopbot:
/* -------------------------------------------------------------*/
   {   uh0.h = vmpy(zh0.h, recip.h):VEXP           //[1, 6]
       zh1.h = vadd(xh1.h, wh1.h)                  //[1, 6]
   } {
       mine.h = vmin(mine.h, zh0.h)                //[1, 7]
       maxe.h = vmax(maxe.h, zh1.h)                //[1,11]
       uh0.h = vsub(uh0.h, voffset.h)             //[1, 7]
   } {
       loop0(.L_minmax, #5)
       uh1.h = vmpy(zh1.h, recip.h):VEXP           //[1, 8]
       mine.h = vmin(mine.h, zh1.h)                //[1,11]
   } {
       ptr_minmax = memw(sp+#1<<2)                 //[E, 11]
       uh1.h = vsub(uh1.h, voffset.h)             //[1, 9]
       c2 = #-2
   } {
       zb1zb0.ub = vsat(uh1.h, uh0.h)              //[1,10]
       vmem(ptr_c++#1) = zb1zb0.new                //[1,10]
       maxe.h = vmax(maxe0.h, maxe.h)
   }
/* -------------------------------------------------------------*/
.L_minmax:
   {
       maxo_maxe = vdeal(maxe, maxe, c2)            //[0, 0]
   } {
       mino_mine = vdeal(mine, mine, c2)            //[0, 1]
       maxe.h = vmax(maxe.h, maxo.h)                //[0, 1]
   } {
       mine.h = vmin(mine.h, mino.h)                //[0, 2]
   }:endloop0
   {   maxo_maxe = vdeal(maxe, maxe, c2)            //[1, 0]
   } {
       mino_mine = vdeal(mine, mine, c2)            //[1, 1]
       maxe.h = vmax(maxe.h, maxo.h)                //[1, 1]
       vmem(ptr_minmax+#0)= maxe.new
   } {
       mine.h = vmin(mine.h, mino.h)                //[1, 2]
       vmem(ptr_minmax+#1)= mine.new
   } {
       jumpr r31
   }
.L_end:
/* -------------------------------------------------------------*/
      .size quant_add_spec_asm, .L_end-quant_add_spec_asm

/*=============================================================================*/
