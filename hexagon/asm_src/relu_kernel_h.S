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
/*  FUNCTIONS      : relu_kernel                                        */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  KG                  06/26/17       created                          */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*                                                                      */
/*     ->                                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =  bytes                                                */
/*     ASSUMPTIONS                                                      */
/*  C MODEL                                                             */
/*======================================================================*/
#define LOG2VLEN       7
#define VLEN           (1<<LOG2VLEN)
#define in_data        r0
#define out_data       r1
#define bytes          r2
#define quantized_zero r3
#define quantized_max  r4
#define bytes1         r8
#define sQuantizedZero v0
#define sQuantizedMax  v1
#define sIn            v2
#define sOut           v3
#define sOut2          v4
    .text
    .globl    relu_kernel
    .falign
    .type    relu_kernel,@function
relu_kernel:
    {
      quantized_zero = vsplatb(quantized_zero)      //
      bytes = add(bytes, #VLEN-1)                   //
      p0 = cmp.gt(bytes, #0)                        //
      if (!p0.new) jumpr:nt r31                     //
    } {
      bytes = lsr(bytes, #LOG2VLEN)                 //
      sQuantizedZero = vsplat(quantized_zero)       //
    } {
      loop0(.relu_kernel_lp0, bytes)                //
    }
    .falign
.relu_kernel_lp0:
    {
      sIn.cur = vmem(in_data++#1)                   //
      sOut.ub = vmax(sIn.ub,sQuantizedZero.ub)      //
      vmem(out_data++#1) = sOut.new                 //
    }:endloop0
    {
      jumpr r31                                     //
    }
.relu_kernel_end:
    .size    relu_kernel, .relu_kernel_end-relu_kernel


    .globl    reluX_kernel
    .falign
    .type    reluX_kernel,@function
reluX_kernel:
    {
      quantized_zero = vsplatb(quantized_zero)      //
      bytes1 = add(bytes, #VLEN-1)                  //
      p0 = cmp.gt(bytes, #0)                        //
      if (!p0.new) jumpr:nt r31                     //
    } {
      bytes1 = lsr(bytes1, #LOG2VLEN)               //
      quantized_max = vsplatb(quantized_max)        //
    } {
      sQuantizedZero = vsplat(quantized_zero)       //
      sQuantizedMax = vsplat(quantized_max)         //
    } {
      loop0(.reluX_kernel_lp0, bytes1)              //
      sIn.cur = vmem(in_data++#1)                   //
      sOut.ub = vmax(sIn.ub,sQuantizedZero.ub)      //
    }
    .falign
.reluX_kernel_lp0:
    {
      sIn.cur = vmem(in_data++#1)                   //
      sOut.ub = vmax(sIn.ub,sQuantizedZero.ub)      //
      sOut2.ub = vmin(sOut.ub,sQuantizedMax.ub)     //[2]
      vmem(out_data++#1) = sOut2.new                //[2]
    }:endloop0
    {
      jumpr r31                                     //
    }
.reluX_kernel_end:
    .size    reluX_kernel, .reluX_kernel_end-reluX_kernel

