
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
        .file "fullyconnected.S"
/*======================================================================*/
/*  FUNCTIONS      : fully_connected_asm                                */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform matrix vector multiply, result  8bits        */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  MZ                  09/07/17       created                          */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->                                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =      bytes                                            */
/*     STACK    =      bytes                                            */
/*     ASSUMPTIONS                                                      */
/*        y and z are 128 byte aligned                                  */
/*        x is16byte aligned                                            */
/*  C MODEL                                                             */
/*======================================================================*/

/*=============================================================================*/
#define ptr_d           r0     //data
#define ptr_w           r1     //weights must be pre processwed and transposed 
#define ptr_o           r2 
#define batch           r3
#define in_depth        r4
#define out_depth       r5
#define ptr_m           r6
#define recip           r7
#define recip_ptr_m     r7:6
#define ptr_suma        r8
#define ptr_bias        r9
#define ptrs_bias_suma  r9:8
/*=============================================================================*/
#define ptr_wk          r7       
#define k               r7
#define l2feparam_w_l   r10
#define stride          r11
#define l2feparam_w     r11:10
#define lc1             r12
#define out_d           r13
#define d3210           r14
#define d7654           r15
#define d7654_d3210     r15:14
#define count           r16
#define lc0             r17
#define lc0_count       r17:16
#define c7fffffff       r18
#define sumaval         r18
#define ptr_w_fe        r19
#define l2f_w_win0      r20
#define l2f_w_win1      r21
#define indepthd8mod    r21
#define indepthd8       r28 
/*=============================================================================*/
#define ssum0_0         v0
#define ssum1_0         v1
#define ssum2_0         v2
#define ssum3_0         v3
#define sbias0          v4
#define sbias1          v4
#define sbias2          v4
#define sbias3          v4
#define sw0             v4
#define sw1             v4
#define sw2             v4
#define sw3             v4
#define ssumaval        v5
#define srecip          V6
#define smin            v8            
#define smax            v9
#define smin_t          v10
#define smax_t          v11
#define ssumr0_0        v10           
#define ssumr1_0        v11           
#define ssumr2_0        v12           
#define ssumr3_0        v13           
#define ssumrh10_0      v14
#define ssumrh32_0      v15
#define sout0           v14
/*=============================================================================*/
    .text
    .global fully_connected_asm
    .balign 32
    .type  fully_connected_asm, @function
fully_connected_asm:
        {
           recip_ptr_m = memd(sp+#0)                    //
           ptrs_bias_suma = memd(sp+#8)                 //
           sp = add(sp,#-3*8)                           //
           indepthd8 = lsr(in_depth,#3)                 //
        }{
           memd(sp+#0) = r17:16                         //
           memd(sp+#8) = r19:18                         //
           stride = asl(out_depth,#2)                   //
        }{
           memd(sp+#16) = r21:20                        //
           indepthd8mod = and(indepthd8,#15)            // 
           c7fffffff = ##0x7fffffff                     //
        }{
           m0 = stride                                  //
           p0 = cmp.eq(indepthd8mod,#0)                 //
           lc1 = lsr(indepthd8,#4)                      //
        }{
           l2f_w_win0 = ##0x02000010                    // for fetch a block of width=512,height=16
           if  p0 indepthd8mod = #16                    //
           smin = vsplat(c7fffffff)                     //
        }{
           if !p0 lc1 = add(lc1,#1)                     //ceil((in_depth/8)/16)
           l2f_w_win1=combine(l2f_w_win0.h,indepthd8mod.l)//
           smax = vnot(smin)                            //
           srecip = vsplat(recip)                       //
        }{
           p0 = cmp.eq(lc1,#1)                          //
           if p0.new l2f_w_win0 = l2f_w_win1            //
        }
        
/*============================================================================*/
.fully_connected_batches_loop:                          // TODO: optimize when batch>1
        {    
           out_d = out_depth                            //
           sumaval = memw(ptr_suma++#4)                 //
           l2feparam_w_l = l2f_w_win0                   //
        }{
           ssumaval = vsplat(sumaval)                   //
           p3 = !cmp.gt(out_d,#0)                       //
        }
        .balign 32
.fully_connected_outdepth_loop:
        {  
           loop1(.fully_connected_indepth_loop,lc1)     //
           ptr_wk = add(ptr_w,#4*128)                   //
           sbias0.tmp = vmem(ptr_bias++#1)              //
           ssum0_0.w = vadd(sbias0.w,ssumaval.w)        //
        }{
           out_d = add(out_d,#-128)                     //
           sbias1.tmp = vmem(ptr_bias++#1)              //
           ssum1_0.w = vadd(sbias1.w,ssumaval.w)        //
           count = indepthd8                            //
        }{
           lc0 = #16                                    // 
           sbias2.tmp = vmem(ptr_bias++#1)              //
           ssum2_0.w = vadd(sbias2.w,ssumaval.w)        //
           nop
        }{
           sbias3.tmp = vmem(ptr_bias++#1)              //
           ssum3_0.w = vadd(sbias3.w,ssumaval.w)        //
           d7654_d3210 = memd(ptr_d++#8)                //
           nop
        }
        .balign 32
.fully_connected_indepth_loop:
        {
           lc0 = min(lc0,count)                         //
           p0 = cmp.gt(count,#32)                       //
           if !p0.new l2feparam_w_l = l2f_w_win1        //
           ptr_w_fe = addasl(ptr_w,stride,#4)           //
        }{
           loop0(.fully_connected_indepth_innerloop,lc0)//
           p0 = cmp.gt(count,#16)                       //  p0 = not last iteration?
           if !p0.new ptr_w_fe = ptr_wk                 //
           if !p0.new l2feparam_w_l = l2f_w_win0        //
        }{
           p0 = not(p0)                                 // if last block,  
           p0 = !cmp.gt(out_d,#0)                       // then set to 0 to cancel l2fetch
           if p0.new l2feparam_w_l = #0                 // 
           if p0.new ptr_w_fe = ptr_w                   //
        }{
           l2fetch(ptr_w_fe,l2feparam_w)                //
           nop; nop; nop
        }
        .balign 32
.fully_connected_indepth_innerloop:
        { 
           sw3.tmp = vmem(ptr_w+#3)                     //
           ssum3_0.uw += vrmpy(sw3.ub,d3210.ub)         //
        }{
           sw2.tmp = vmem(ptr_w+#2)                     //
           ssum2_0.uw += vrmpy(sw2.ub,d3210.ub)         //
        }{
           sw1.tmp = vmem(ptr_w+#1)                     //
           ssum1_0.uw += vrmpy(sw1.ub,d3210.ub)         //
        }{
           sw0.tmp = vmem(ptr_w++m0)                    //
           ssum0_0.uw += vrmpy(sw0.ub,d3210.ub)         //
           count = add(count,#-1)                       //
        }{
           sw3.tmp = vmem(ptr_w+#3)                     //
           ssum3_0.uw += vrmpy(sw3.ub,d7654.ub)         //
           p0 = cmp.eq(count,#0)                        //
        }{
           sw2.tmp = vmem(ptr_w+#2)                     //
           ssum2_0.uw += vrmpy(sw2.ub,d7654.ub)         //
        }{
           sw1.tmp = vmem(ptr_w+#1)                     //
           ssum1_0.uw += vrmpy(sw1.ub,d7654.ub)         //
        }{
           sw0.tmp = vmem(ptr_w++m0)                    //
           ssum0_0.uw += vrmpy(sw0.ub,d7654.ub)         //
           if !p0 d7654_d3210 = memd(ptr_d++#8)         //
        }:endloop0:endloop1

.fully_connected_indepth_lpend:
        {
           ssumr3_0.w = vmpye(ssum3_0.w,srecip.uh)      //
           ssumrh32_0.h=vpack(ssumr3_0.w,ssumr2_0.w):sat//[2]
        }{
           ssumr2_0.w = vmpye(ssum2_0.w,srecip.uh)      //
           ssumrh10_0.h=vpack(ssumr1_0.w,ssumr0_0.w):sat//[2]
        }{
           ssumr3_0.w += vmpyo(ssum3_0.w,srecip.h):<<1:rnd:sat:shift        
           smax.w = vmax(smax.w,ssum3_0.w)              //
           smin.w = vmin(smin.w,ssum3_0.w)              //
        }{
           ssumr2_0.w += vmpyo(ssum2_0.w,srecip.h):<<1:rnd:sat:shift        
           sout0.ub=vpack(ssumrh32_0.h,ssumrh10_0.h):sat//[2]
           if p3 vmem(ptr_o++#1) = sout0.new            //[2]
        }{
           ssumr1_0.w = vmpye(ssum1_0.w,srecip.uh)      //
           smax.w = vmax(smax.w,ssum2_0.w)              //
           smin.w = vmin(smin.w,ssum2_0.w)              //
        }{
           ssumr0_0.w = vmpye(ssum0_0.w,srecip.uh)      //
           smax.w = vmax(smax.w,ssum1_0.w)              //
           smin.w = vmin(smin.w,ssum1_0.w)              //
        }{
           ssumr1_0.w += vmpyo(ssum1_0.w,srecip.h):<<1:rnd:sat:shift        
           smax.w = vmax(smax.w,ssum0_0.w)              //
           smin.w = vmin(smin.w,ssum0_0.w)              //
           p3 = cmp.gt(out_d,#0)                        //
        }{
           ssumr0_0.w += vmpyo(ssum0_0.w,srecip.h):<<1:rnd:sat:shift        
           ptr_w = ptr_wk                               //
           if p3 ptr_d = sub(ptr_d,in_depth)            //
           if p3 jump .fully_connected_outdepth_loop    //
        }
.fully_connected_outdepth_lpend:
        {
           ssumrh32_0.h=vpack(ssumr3_0.w,ssumr2_0.w):sat//[2]
           batch = add(batch,#-1)                       //
        }{
           ssumrh10_0.h=vpack(ssumr1_0.w,ssumr0_0.w):sat//[2]
           ptr_w    = sub(ptr_w,stride)                 //
           ptr_bias = sub(ptr_bias,stride)              //
           p0 = cmp.gt(batch,#0)                        //
        }{
           sout0.ub=vpack(ssumrh32_0.h,ssumrh10_0.h):sat//[2]
           vmem(ptr_o++#1) = sout0.new                  //[2]
           if p0 jump .fully_connected_batches_loop     //       
        }
/*=============================================================================*/
        {
           loop0(.fully_connect_reducemax_loop,#5)      //
           k = #64                                      //
           r17:16 = memd(sp+#0)                         //
           r19:18 = memd(sp+#8)                         //
        }

        .falign
.fully_connect_reducemax_loop:
        {  
           smax_t = vror(smax,k)                        //
        }{
           smin_t = vror(smin,k)                        //
           smax.w = vmax(smax.w,smax_t.w)               //
        }{
           smin.w = vmin(smin.w,smin_t.w)               //
           k = lsr(k,#1)                                //
        }:endloop0

        {  
           vmem(ptr_m+#0) = smax                        //
        }{
           vmem(ptr_m+#1) = smin                        //
        }{
           r21:20 = memd(sp+#16)                        //
           sp = add(sp,#3*8)                            //
           jumpr r31                                    //
        }
.fully_connected_asm_end:
/*=============================================================================*/
      .size fully_connected_asm, .-fully_connected_asm

