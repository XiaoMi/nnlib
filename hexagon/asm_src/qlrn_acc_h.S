
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
/*  FUNCTIONS      : qlrn_acc_asm                                       */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 X matrix to be squared horizontally and accumulated. */
/*                                                                      */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*======================================================================*/
#if 0
int32_t qlrn_acc_cn(int32_t numelem, const int16_t *ptr_xvec)
{
    int32_t xvec;
    int32_t xacc = 0;
    for (int32_t i = 0; i < numelem; i++) {
        xvec = *ptr_xvec;
        ptr_xvec += 2;
        xacc += (int32_t)(xvec * xvec);
    }
    return xacc;
}
#endif
/*=============================================================================*/
        .text
        .file "qlrn_acc_h.S"
        .global qlrn_acc_asm
        .type  qlrn_acc_asm, @function
/*=============================================================================*/
#define xacc          r0     //accumulation summation value
#define numelem       r0     //numelem is number of elements to be squared+summed
#define ptr_xvec      r1     //ptr to vector of elements
#define xvec          r6     //
#define retreg        r31    //
/*=============================================================================*/
qlrn_acc_asm:
        {
           loop0(.L_loop0,numelem)                      //loop-config
           dcfetch(ptr_xvec+#0)                         //prefetch
           xacc = xor(xacc,xacc)                        //xacc=0;
           xvec = memh(ptr_xvec++#4)                    //xvec=*ptr_xvec;ptr_xvec+=2;
        }
.L_loop0:
        {
           xvec = mpyi(xvec,xvec)                        //(int32_t)(xvec*xvec);
        }
        {
           xacc = add(xacc,xvec)                        //xacc+=(int32_t)(xvec*xvec)
           xvec = memh(ptr_xvec++#4)                    //xvec=*ptr_xvec;ptr_xvec+=2;
        }:endloop0
        {
           jumpr retreg                                 //return;(xacc computation done)
        }
.L_end:
/*=============================================================================*/
      .size qlrn_acc_asm, .L_end-qlrn_acc_asm
