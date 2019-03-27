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
    Compute sum of data and sum of squares of data for each depth position
    used for later mean and sd. convert result to qf16 format.
  */

#if 0
                for (d = 0; d < depth; d++) { sumg[d] = sum2g[d] = 0; }
                for (h = 0; h < height; h++) {
                    for (w = 0; w < width; w++) {
                        for (d = 0; d < depth; d++) {
                            tmp = *in_data++;
                            sumg[d] += tmp;         //23bits
                            sum2g[d] += tmp*tmp;    //30bits
                        }
                    }
                }
#endif
/*======================================================================*/
                .global getstats_asm
                .balign 32
                .type  getstats_asm, @function
getstats_asm:
/*======================================================================*/
#define in_data          r0
#define width32          r1
#define next_width       r2
#define height           r3
#define ptr_mean         r4
#define ptr_var          r5
#define ptr_max          r8
#define ptr_min          r9

#define n                r1
#define byte             r6
#define hwrd             r7
#define dprd             r15
#define c31              r11
#define next_width2      r12
#define in_data1         r13
#define in_data2         r14

#define exp0hi_expnt0    v21:20
#define exp2hi_expnt2    v23:22
#define sum0_sum0lo      v25:24
#define sum2_sum2lo      v27:26
#define u                v26
#define expnt0           v20
#define expnt2           v22
#define sum0             v25
#define sum1             v26
#define sum2             v27
#define lsum             v28
#define gsuml_prev       v31    //previous global sum low
#define gsuml            v29    //global 64bit sum low half
#define gsumh            v30    //global 64bit sum high half
#define din              v0
#define vzero            v1
#define vones            v2
#define delta            v3
#define pr1              v4
#define pr2              v5
#define d7d6d5d4         v7
#define d3d2d1d0         v6
#define a7a6a5a4a3a2a1a0 v9:8
#define a7a6a5a4         v9
#define a3a2a1a0         v8
#define b7b6b5b4b3b2b1b0 v13:12
#define b7b6b5b4         v13
#define b3b2b1b0         v12
#define vc31             v15
#define maxo_maxe        v17:16
#define maxo             v17
#define maxe             v16
#define max_exp          v7
#define mino_mine        v19:18
#define mino             v19
#define mine             v18
#define min_exp          v6
#define PV(VSRC) .word (0x1DFFE020+VSRC) 
/*======================================================================*/
             {
               byte = #-1                                       //
               n = lsr(width32, #7)                             //1/128
               dprd = ##0x01010101                              //
             } {
               n = add(n, #-1)                                  //
               next_width2 = add(next_width, next_width)        //
               height = lsr(height, #1)                         //
               lsum = #0                                        //
             } {
               loop0(.L_width, n)                               //
               c31 = #31                                        //31
               q0 = and(q0, !q0)                                //clear q0 
               p2 = cmp.eq(n, #0)                               //
             } {
               d3d2d1d0 = vmem(in_data+#0)                      //[P,0]b3 b2 b1 b0
               in_data1 = add(in_data, #128)                    //
               in_data2 = add(in_data, next_width)              //
               height = add(height, #-1)                        //
             } {
               vc31 = vsplat(c31)                               //
               vzero = #0                                       //
               gsuml = #0                                       //
               gsumh = #0                                       //
             } {
               d7d6d5d4.cur = vmem(in_data2++#1)                //[0,0]b3 b2 b1 b0
               a7a6a5a4a3a2a1a0=vshuff(d7d6d5d4,d3d2d1d0,byte)  //[0,0]
               loop1(.L_height, height)                         //
               sum0 = #0                                        //clear mean
             } {
               gsuml_prev = #0                                  //
               maxe = d3d2d1d0                                  //[0,1]
               mine = d3d2d1d0                                  //[0,1]
               d3d2d1d0 = vmem(in_data1++#1)                    //[0,1]b3 b2 b1 b0
             } {
               b7b6b5b4b3b2b1b0 =vshuff(a7a6a5a4,a3a2a1a0,byte) //[0,2]
               in_data = add(in_data, next_width2)              //
               sum1 = #0                                        //clear mean
               c31 = #1                                         //
             } {
               maxe.ub = vmax(maxe.ub, d7d6d5d4.ub)             //[0,3]
               mine.ub = vmin(mine.ub, d7d6d5d4.ub)             //[0,3]
               sum2 = #0                                        //
               vones = vsplat(c31)                              //set incre reg
             } 
/*======================================================================*/
           .balign 32
.L_height:
             {
               if(p2) jump .L_skip
               gsuml.w = vadd(gsuml.w, lsum.w)                  //new sum
             }
.L_width:
             { sum0.uw += vrmpy(b3b2b1b0.ub, dprd.ub)           //[0,4]
               d7d6d5d4.cur = vmem(in_data2++#1)                //[1,0]b3 b2 b1 b0
               a7a6a5a4a3a2a1a0=vshuff(d7d6d5d4,d3d2d1d0,byte)  //[1,0]
             } {
               sum0.uw += vrmpy(b7b6b5b4.ub, dprd.ub)           //[0,5]
               maxe.ub = vmax(maxe.ub, d3d2d1d0.ub)             //[1,1]
               mine.ub = vmin(mine.ub, d3d2d1d0.ub)             //[1,1]
               d3d2d1d0 = vmem(in_data1++#1)                    //[1,1]b3 b2 b1 b0
             } {
               pr1.uw = vrmpy(b3b2b1b0.ub, b3b2b1b0.ub)         //[0,6]
               pr2.uw = vrmpy(b7b6b5b4.ub, b7b6b5b4.ub)         //[0,6]
               b7b6b5b4b3b2b1b0 =vshuff(a7a6a5a4,a3a2a1a0,byte) //[1,2]
             } {
               sum1.w = vadd(sum1.w, pr1.w)                     //[0,7]
               sum2.w = vadd(sum2.w, pr2.w)                     //[0,7]
               maxe.ub = vmax(maxe.ub, d7d6d5d4.ub)             //[1,3]
               mine.ub = vmin(mine.ub, d7d6d5d4.ub)             //[1,3]
             }:endloop0
/*======================================================================*/
.L_skip:
             { d3d2d1d0 = vmem(in_data+#0)                      //[P,0]b3 b2 b1 b0
               in_data1 = add(in_data, #128)                    //[P,0]
               in_data2 = add(in_data, next_width)              //[P,0]
               q0 = vcmp.gt(gsuml_prev.uw, gsuml.uw)            //
             } {
               loop0(.L_width, n)                               //[P,0]
               in_data = add(in_data, next_width2)              //[P,0]
               delta = vmux(q0, vones, vzero)                   //
               gsuml_prev = gsuml                               //record for next time
             } {
               d7d6d5d4.cur = vmem(in_data2++#1)                //[P,0]b3 b2 b1 b0
               sum0.uw += vrmpy(b3b2b1b0.ub, dprd.ub)           //[1,4]
               a7a6a5a4a3a2a1a0=vshuff(d7d6d5d4,d3d2d1d0,byte)  //[P,0]
             } {
               sum0.uw += vrmpy(b7b6b5b4.ub, dprd.ub)           //[1,5]
               maxe.ub = vmax(maxe.ub, d3d2d1d0.ub)             //[P,1]
               mine.ub = vmin(mine.ub, d3d2d1d0.ub)             //[P,1]
               d3d2d1d0 = vmem(in_data1++#1)                    //[P,1]b3 b2 b1 b0
             } {
               pr1.uw = vrmpy(b3b2b1b0.ub, b3b2b1b0.ub)         //[1,6]
               pr2.uw = vrmpy(b7b6b5b4.ub, b7b6b5b4.ub)         //[1,6]
               b7b6b5b4b3b2b1b0 =vshuff(a7a6a5a4,a3a2a1a0,byte) //[P,2]
             } {
               sum1.w = vadd(sum1.w, pr1.w)                     //[0,7]
               sum2.w = vadd(sum2.w, pr2.w)                     //[0,7]
               maxe.ub = vmax(maxe.ub, d7d6d5d4.ub)             //[P,3]
               mine.ub = vmin(mine.ub, d7d6d5d4.ub)             //[P,3]
             } {
               gsumh.w = vadd(gsumh.w, delta.w)                 //
               lsum.w = vadd(sum2.w, sum1.w)                    //line sum
               sum1 = #0                                        //clear mean
               sum2 = #0                                        //clear mean
             }:endloop1
             { gsuml.w = vadd(gsuml.w, lsum.w)                  //new sum
               loop0(.L_width1, n)                              //[P,0]
               if(p2) jump .L_skip2                             //
             }
/*======================================================================*/
.L_width1:
             { sum0.uw += vrmpy(b3b2b1b0.ub, dprd.ub)           //[0,4]
               d7d6d5d4.cur = vmem(in_data2++#1)                //[1,0]b3 b2 b1 b0
               a7a6a5a4a3a2a1a0=vshuff(d7d6d5d4,d3d2d1d0,byte)  //[1,0]
             } {
               sum0.uw += vrmpy(b7b6b5b4.ub, dprd.ub)           //[0,5]
               maxe.ub = vmax(maxe.ub, d3d2d1d0.ub)             //[1,1]
               mine.ub = vmin(mine.ub, d3d2d1d0.ub)             //[1,1]
               d3d2d1d0 = vmem(in_data1++#1)                    //[1,1]b3 b2 b1 b0
             } {
               pr1.uw = vrmpy(b3b2b1b0.ub, b3b2b1b0.ub)         //[0,6]
               pr2.uw = vrmpy(b7b6b5b4.ub, b7b6b5b4.ub)         //[0,6]
               b7b6b5b4b3b2b1b0 =vshuff(a7a6a5a4,a3a2a1a0,byte) //[1,2]
             } {
               sum1.w = vadd(sum1.w, pr1.w)                     //[0,7]
               sum2.w = vadd(sum2.w, pr2.w)                     //[0,7]
               maxe.ub = vmax(maxe.ub, d7d6d5d4.ub)             //[1,3]
               mine.ub = vmin(mine.ub, d7d6d5d4.ub)             //[1,3]
             }:endloop0
/*======================================================================*/
.L_skip2:
             { q0 = vcmp.gt(gsuml_prev.uw, gsuml.uw)            //
             } {
               delta = vmux(q0, vones, vzero)                   //
               gsuml_prev = gsuml                               //record for next time
             } {
               sum0.uw += vrmpy(b3b2b1b0.ub, dprd.ub)           //[1,4]
             } {
               sum0.uw += vrmpy(b7b6b5b4.ub, dprd.ub)           //[1,5]
             } {
               pr1.uw = vrmpy(b3b2b1b0.ub, b3b2b1b0.ub)         //[1,6]
               pr2.uw = vrmpy(b7b6b5b4.ub, b7b6b5b4.ub)         //[1,6]
             } {
               sum1.w = vadd(sum1.w, pr1.w)                     //[0,7]
               sum2.w = vadd(sum2.w, pr2.w)                     //[0,7]
             } {
               gsumh.w = vadd(gsumh.w, delta.w)                 //
               lsum.w = vadd(sum2.w, sum1.w)                    //line sum
             } {
               gsuml.w = vadd(gsuml.w, lsum.w)       
             } {
               q0 = vcmp.gt(gsuml_prev.uw, gsuml.uw)            //
               gsuml_prev = gsuml                               //record for next time
             } {
               delta = vmux(q0, vones, vzero)                   //
             } {
               gsumh.w = vadd(gsumh.w, delta.w)                 //
               maxo_maxe =vshuff(vzero, maxe, byte)
             } {
               maxe.uh = vmax(maxe.uh, maxo.uh)
               mino_mine =vshuff(vzero, mine, byte)
             } {
               mine.uh = vmin(mine.uh, mino.uh)
               hwrd = #-2
             } {
               maxo_maxe =vshuff(vzero, maxe, hwrd)
             } {
               mino_mine =vshuff(vzero, mine, hwrd)
             } {
               maxe.w = vmax(maxe.w, maxo.w)
               mine.w = vmin(mine.w, mino.w)
               expnt0.w = vnormamt(sum0.w)                      //convert to qf16
             } {
               q0 = vcmp.eq(mine.w, maxe.w)                     //are max and min same 
             } {
               mine = vmux(q0, vzero, mine)                     //then clamp min to 0
             } {
               max_exp.w = vnormamt(maxe.w)
               ptr_max = memw(sp+#0<<2)
             } {
               min_exp.w = vnormamt(mine.w)
               ptr_min = memw(sp+#1<<2)
             } {
               maxe.w = vasl(maxe.w, max_exp.w)
               max_exp.w = vsub(max_exp.w, vc31.w)
             } {
               mine.w = vasl(mine.w, min_exp.w)
               min_exp.w = vsub(min_exp.w, vc31.w)
             } {
               exp2hi_expnt2 = vdeal(max_exp, max_exp, hwrd)    //(short)expnt pack 2xdepth
               vmem(ptr_max+#1) = expnt2.new                    //
             } {
               maxo_maxe = vdeal(maxe, maxe, hwrd)
               vmem(ptr_max+#0) = maxo.new                      //
             } {
               exp2hi_expnt2 = vdeal(min_exp, min_exp, hwrd)    //(short)expnt pack 2xdepth
               vmem(ptr_min+#1) = expnt2.new                    //
             } {
               mino_mine = vdeal(mine, mine, hwrd)
               vmem(ptr_min+#0) = mino.new                      //
             } {
               sum0.w = vasl(sum0.w, expnt0.w)                  //sum << norml
               expnt0.w = vsub(expnt0.w, vc31.w)
               q0 = vcmp.eq(gsumh.w, vzero.w)                   //is hi = 0 convert 64bit into qf16
             } {
               u = vmux(q0, gsuml, gsumh)                       //u = xlo else xhi 
             } {
               exp0hi_expnt0 = vdeal(expnt0, expnt0, hwrd)      //(short)expnt pack 2xdepth
               vmem(ptr_mean+#1) = expnt0.new                   //exp of sum 16bits
             } {
               expnt2.uw = vcl0(u.uw)                           //count lead zero
             } {
               vc31.w = vadd(vc31.w, vones.w)                   //31+1
             } {
               u.w = vasl(u.w, expnt2.w)                        //sum << exp
               expnt2.w = vsub(expnt2.w, vc31.w)                //exp - 32
               expnt0.w = vsub(vc31.w, expnt2.w)                //32 - exp
             } {
               sum0_sum0lo = vdeal(sum0, sum0, hwrd)            //sum0 >> 16, pack 2xdepth
               vmem(ptr_mean+#0) = sum0.new                     //mant of sum0 16bits
             } {
               gsuml.w = vlsr(gsuml.w, expnt0.w)                //sum >> 32-exp 
               vc31 = vmux(q0, vzero, vc31)                     //if hi !=0 sub 32
             } {
               expnt2.w = vsub(expnt2.w, vc31.w)                //
               gsuml = vmux(q0, vzero, gsuml)                   //
             } {
               exp2hi_expnt2 = vdeal(expnt2, expnt2, hwrd)      //(short)expnt pack 2xdepth
               vmem(ptr_var+#1) = expnt2.new                    //exponenet of sum2 16bits
               u = vor(u, gsuml)                                //
             } {
               u.w = vlsr(u.w, vones.w)                         //sum << norml
             } {
               sum2_sum2lo = vdeal(u, u, hwrd)                  //sum2 >> 16, pack 2xdepth
               vmem(ptr_var+#0) = sum2.new                      //mant of sum2 16bits
             } {
               jumpr r31                                        //
             }
/*======================================================================*/
/*                               End fo File                            */
/*======================================================================*/
