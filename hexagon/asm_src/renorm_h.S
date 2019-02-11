/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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

#if 0
        Apply final combined requant and adjust channels
        y = (gamma*(x / sigma - u / sigma) - beta) - gmin) * 255 / (gmax-gmin)
#endif
/* =================================================================================== */
          .global   renorm_asm
          .balign   32
          .type     renorm_asm, @function   
/* =================================================================================== */
renorm_asm:
/* =================================================================================== */
#define   in_data         r0      //ptr raw input
#define   width32         r1      //num points in current d32 line    
#define   next_width      r2      //next depth32 chunk stride
#define   height          r3      //hieght of acts    
#define   out_data        r4      //ptr normalized result
#define   ptr_mean        r5      //mean mantissa / exponent
#define   ptr_rsd         r6      //1/sqrt mantissa / exponent
#define   c15             r10     //constants 15
#define   n               r9      //number of itns
#define   in_data1        r11     //ptr to input
#define   out_data1       r8      //ptr to output
#define   cm2             r7      //-2 for dealing hwords
#define   cm128           r12     //-128
#define   mean_shift      r3      //value to shift left mean
#define   rsd_shift       r13     //value to shift right qrsd
#define   l2cntrl         r15:14  //l2 fetch controls
#define   l2cntrl_h       r15     //
#define   l2cntrl_l       r14     //
#define   fetch           r6      //
/* =================================================================================== */
#define   vzero           v21     //nil
#define   qrsd_mant       v4      //recip stdev mantissa
#define   qrsd_exp        v5      //recip stdev exponent
#define   vc15            v16     //rounding value 1 << (qe+14)
#define   qmean_exp       v12     //mean exponent
#define   qmean_mant      v14     //mean mantissa
#define   data127_0b      v20     //data in bytes
#define   data127_0h      v1:0    //expanded to hwords after * -128
#define   data127_0h_hi   v1      //hi reg
#define   data127_0h_lo   v0      //lo reg
#define   phi             v7      //products
#define   plo             v2      //products
#define   phis            v8      //products
#define   plos            v3      //products
#define   nhis            v18     //products
#define   nlos            v13     //products
#define   phic            v9      //products
#define   ploc            v6      //products
#define   pb              v16     //final values to be stored
#define   vmean_shift     v30     //mean adjust
#define   vrsd_shift      v19     //1/sd adjust
#define   qmean           v26     //temp
#define   qmean_mant_o_e  v23:22  //mean mantissas
#define   qmean_mant_e    v22     //mean mantissas
#define   qmean_mant_o    v23     //mean mantissas
#define   qrsd_mant_o_e   v25:24  //1/sd mantissas
#define   qrsd_mant_e     v24     //1/sd mantissas
#define   qrsd_mant_o     v25     //1/sd mantissas
#define   qrsd_exp_o_e    v29:28  //1/sd exponents
#define   qrsd_exp_o      v29     //1/sd exponents
#define   qrsd_exp_e      v28     //1/sd exponents
#define PV(VSRC) .word (0x1DFFE020+VSRC)
#if __HEXAGON_ARCH__ > 60
#define   MSHIFT      0x00010001  //1 bits of fraction for mean
#define   QSHIFT      0x00060006  //increase right shift by 6
#else
#define   MSHIFT      0x00010001  //expose 1 bit of fraction of mean
#define   QSHIFT      0x00060006  //make -ve shift rights go away
#endif
/* =================================================================================== */
        { ptr_rsd = memw(sp+#0<<2)                                  //[S,0]ptr to recip sqrt sd
          c15 = ##0x000f000f                                        //[S,0]15
          vzero = #0                                                //[S,1]nil
        } {
          vc15 = vsplat(c15);                                       //[S,1]constant 15
          qrsd_mant = vmem(ptr_rsd+#0)                              //[S,1]load recip sd mant
          cm128 = ##0x80808080                                      //-128 splatted
        } {
          loop1(.L_height, height)                                  //[S,4]iterate over hegith lines
          qrsd_exp = vmem(ptr_rsd+#1)                               //[S,2]
          cm2 = #-2                                                 //hwords
          l2cntrl_l = #1                                            //1 line at a time
        } {
          rsd_shift = ##QSHIFT                                      //assert rsd exp rightt by some
          mean_shift = ##MSHIFT                                     //assert mean left by some
        } {
          vmean_shift = vsplat(mean_shift)                          //mean exp adjust
          l2cntrl_h = next_width                                    //stride
          l2cntrl_l = combine(width32.L, l2cntrl_l.L)               //width , height
        } {
          qmean.tmp = vmem(ptr_mean+#1)                             //[S,3]load mean exp
          qmean_exp.h = vadd(qmean.h, vc15.h)                       //[S,3]float -> int
          n = lsr(width32, #7)                                      //[S,3]128 bytes per itn.
        } {
          fetch = addasl(in_data, next_width, #1)                   //2 lines ahead
          vrsd_shift = vsplat(rsd_shift)                            //splat the rsd expoent adjust
          qmean_exp.h = vsub(qmean_exp.h, vmean_shift.h)            //decrese mean exp
        } {
          l2fetch(fetch, l2cntrl)                                   //fetch initial line
        } {
          qmean.tmp = vmem(ptr_mean+#0)                             //[S,4]loaf mean mant
          qmean_mant.h = vasr(qmean.h, qmean_exp.h)                 //[S,4]convert to int mant >> exp
          qrsd_exp.h = vadd(qrsd_exp.h, vrsd_shift.h)               //add extra right shift to exp  
          q0 = vcmp.gt(qmean_exp.h, vc15.h)                         //if mean_exp > 15 flush to 0
        } {
          qrsd_mant_o_e = vdeal(qrsd_mant, qrsd_mant, cm2)          //match the input o/e data
          qmean_mant.h = vsub(vzero.h, qmean_mant.h)                //mean = -mean 
          qrsd_exp.h = vmin(qrsd_exp.h, vc15.h)                     //[S,2]prevent wrap if > 15
        } {
          qrsd_exp_o_e = vdeal(qrsd_exp, qrsd_exp, cm2)             //match the input o/e data
          qmean_mant = vmux(q0, vzero, qmean_mant)                  //if exp > 15 flush mant
          p3 = sp1loop0(.L_renorm, n)                               //start new loop
        } {
          qmean_mant_o_e = vdeal(qmean_mant, qmean_mant, cm2)       //match the input o/e data
          data127_0b.tmp = vmem(in_data+#0)                         //[P, 0]in_data += 128;
          data127_0h.h = vmpy(data127_0b.ub, cm128.b)               //[P, 0]
          in_data1 = add(in_data, #128)                             //[P, 1]
        } 
/* =========================================================================== */
        .balign 32
.L_height:
.L_renorm:                                                          //for (i = 0; i < height*width/4; i++) {
        { plo.h = vmpy(qrsd_mant_e.h, data127_0h_lo.h):<<1:rnd:sat  //[0, 0](x * qrsd_mant)>>15
          if(!p3) in_data = add(in_data, next_width)                //[P, 1]update in ptr
          if(!p3) out_data1 = out_data                              //[P, 1]
        } {
          phi.h = vmpy(qrsd_mant_o.h, data127_0h_hi.h):<<1:rnd:sat  //[0, 1](x * qrsd_mant)>>15
          pb.ub = vasr(phic.h, ploc.h, mean_shift):rnd:sat          //[0, 1]
          if(p3) vmem(out_data1++#1):nt = pb.new                    //[0, 1]out_data+=128;
        } {
          plos.h = vasr(plo.h, qrsd_exp_e.h)                        //[0, 2]apply exponent shift
          if(!p3) out_data = add(out_data, next_width)              //[P, 1]update line ptr
        } {
          phis.h = vasr(phi.h, qrsd_exp_o.h)                        //[0, 3]apply exponent shift
          data127_0b.tmp = vmem(in_data1++#1)                       //[0, 3]in_data += 128;
          data127_0h.h = vmpy(data127_0b.ub, cm128.b)               //[0, 3] * 128 (-1 <<7)
        } {
          ploc.h = vsub(qmean_mant_e.h, plos.h)                     //[0, 4]x/sigma - u / sigma
          phic.h = vsub(qmean_mant_o.h, phis.h)                     //[0, 4] 
        }:endloop0
/* =========================================================================== */
        { p3 = sp1loop0(.L_renorm, n)                               //start new loop
          data127_0b.tmp = vmem(in_data+#0)                         //[P, 0]in_data += 128;
          data127_0h.h = vmpy(data127_0b.ub, cm128.b)               //[P, 0]*-128 for << 7
          in_data1 = add(in_data, #128)                             //[P, 1]
        } {
          l2fetch(fetch, l2cntrl)                                   //fetch next line
          fetch = add(fetch, next_width)                            //lines ahead
        } {
          pb.ub = vasr(phic.h, ploc.h, mean_shift):rnd:sat          //[E, 1]correct for shift
          vmem(out_data1+#0):nt = pb.new                            //[E, 0]out_data+=128;
        }:endloop1
/* =========================================================================== */
          jumpr r31                                                 //return
/* =========================================================================== */
.L_end:
      .size renorm_asm, .L_end-renorm_asm
