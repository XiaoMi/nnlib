/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
 
#define ine8_ptr     r0
#define poute        ine8_ptr
#define ino8_ptr     r1
#define pouto        ino8_ptr
#define in16_ptr     r2
#define pinput       in16_ptr
#define width        r3
#define height       r4
#define depth        r5
#define c_1          r6
#define c127         r8
#define lpcnt        c127
#define c128         r9

#define sIn0         v0
#define sIn1         v1
#define sOut_L       v2
#define sOut_H       v3
#define dOut         v3:2
#define sPrev0       v4
#define sShift0      v5
#define dOut2        v7:6
#define sOut2_L      v6
#define sOut2_H      v7

	.text
	.global d32_16_to_88_cn
	.type d32_16_to_88_cn,@function
	.p2align 6
d32_16_to_88_cn:
    {                                               
      c127 = #127                                   //
      c_1 = #-1                                     //
      width = asr(width,#2)                         //
      depth = asr(depth,#5)                         //
    }                                             
    {                                               
      c128 = #128                                   //
      p0 = bitsclr(in16_ptr,c127)                   //
      lpcnt = mpyi(width,depth)                     //
    }                                             
    {                                               
      lpcnt = mpyi(lpcnt,height)                    //
      if (!p0) jump d32_16_to_88_cn_unalign         //
    }                                             
    {                                               
      p3 = sp1loop0(d32_16_to_88_cn_A_lp,lpcnt)     //
    }                                             
d32_16_to_88_cn_A_lp:
    {                                               
      sIn0 = vmem(pinput++#1)                       //
      vmem(poute+#0) = sOut_L                       //
      if (p3) poute = add(poute,c128)               //[2]
    }                                             
    {                                               
      if (p3) vmem(pouto++#1) = sOut_H              //[2]
    }                                             
    {                                               
      sIn1.tmp = vmem(pinput++#1)                   //
      dOut = vdeal(sIn1,sIn0,c_1)                   //
    }:endloop0                                    
    {                                               
      vmem(poute++#1) = sOut_L                      //
    }                                             
    {                                               
      vmem(pouto++#1) = sOut_H                      //
      jumpr r31                                     //
    }                                             
    
	.p2align 6
d32_16_to_88_cn_unalign:    
    {                                               
      sIn1 = vmem(pinput++#1)                       //
      p3 = sp1loop0(d32_16_to_88_cn_B_lp,lpcnt)     //
    }                                             
d32_16_to_88_cn_B_lp:
    {                                               
      sIn0.cur = vmem(pinput++#1)                   //
      sShift0 = valign(sIn0, sIn1, pinput)          //
    }                                             
    {                                               
      dOut2 = vdeal(sOut_H,sOut_L,c_1)              //[2]
      if (p3) vmem(poute++#1) = sOut2_L.new         //[2]
      sOut_L = sShift0                              //
    }                                             
    {                                               
      sIn1.cur = vmem(pinput++#1)                   //
      sOut_H = valign(sIn1, sIn0, pinput)           //
      vmem(pouto+#0) = sOut2_H                      //[2]
      if (p3) pouto = add(pouto,c128)               //[2]
    }:endloop0                                    
    {                                               
      dOut2 = vdeal(sOut_H,sOut_L,c_1)              //[e]
      vmem(poute++#1) = sOut2_L.new                 //[e]
    }                                             
    {                                               
      vmem(pouto++#1) = sOut2_H                     //[e]
      jumpr r31                                     //
    }                                             

	.size d32_16_to_88_cn,.-d32_16_to_88_cn
