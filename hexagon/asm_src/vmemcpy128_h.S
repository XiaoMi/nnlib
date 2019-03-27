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
          .global vmemcpy_128
          .type   vmemcpy_128, @function
          .balign 32
vmemcpy_128: 
/* ============================================================================ */
#define dst         r0 //dest ptr
#define src         r1 //src ptr
#define length      r2 //num bytes
#define kernel      r3
#define rest        r4
/* ============================================================================ */
#define x0          v0
#define y0          v1
/* ============================================================================ */
    { kernel = lsr(length, #(7+2))                  // (length>>7)/4
      rest = extractu(length,#2,#7)                 // (length>>7)%4
    }{
      p1 = cmp.gt(rest,#0)                          //
      p0 = cmp.gt(kernel,#0)                        //
      if (!p0.new) jump:nt .remain                  //
    }{
      loop0(.L_copy_loop, kernel)                   //
      nop; nop; nop                                 //
    }
/* ============================================================================ */
   .balign 32
.L_copy_loop:
    { x0.tmp = vmem(src++#1):nt                     //load next bloc
      y0 = x0                                       //
      vmem(dst++#1) = y0.new                        //store out 
      nop                                           //
    }{
      x0.tmp = vmem(src++#1):nt                     //load next bloc
      y0 = x0                                       //
      vmem(dst++#1) = y0.new                        //store out 
      nop                                           //
    }{
      x0.tmp = vmem(src++#1):nt                     //load next bloc
      y0 = x0                                       //
      vmem(dst++#1) = y0.new                        //store out 
      nop                                           //
    }{
      x0.tmp = vmem(src++#1):nt                     //load next bloc
      y0 = x0                                       //
      vmem(dst++#1) = y0.new                        //store out 
      nop                                           //
    }:endloop0

.remain:
    { if (!p1) jumpr r31                            //
      p2 = cmp.gt(rest,#1)                          //
      p3 = cmp.gt(rest,#2)                          //
    }{
      x0.tmp = vmem(src+#0):nt                      //load next bloc
      y0 = x0                                       //
      vmem(dst+#0) = y0.new                         //store out 
      if (!p2) jumpr r31                            //
    }{
      x0.tmp = vmem(src+#1):nt                      //load next bloc
      y0 = x0                                       //
      vmem(dst+#1) = y0.new                         //store out 
      if (!p3) jumpr r31                            //
    }{
      x0.tmp = vmem(src+#2):nt                      //load next bloc
      y0 = x0                                       //
      vmem(dst+#2) = y0.new                         //store out 
      jumpr r31                                     //
    }
.L_end:
/*==============================================================================*/
      .size vmemcpy_128, .L_end-vmemcpy_128
