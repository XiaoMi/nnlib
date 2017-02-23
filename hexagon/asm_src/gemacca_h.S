
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
/*  FUNCTIONS      : gemacca_asm                                        */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 X matrix to be accumulated horizontally and added to */
/*                   sum.                                               */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*  DJH                 05/10/16       added post add for xsum          */
/*======================================================================*/
/*  IDEAL-CYCLE-COUNT:                                                  */
/*     ->  3*K*N/16+6*N+8                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 288 bytes                                             */
/*     STACK    =  64 bytes                                             */
/*  ASSUMPTIONS                                                         */
/*        x, xsum is 8byte aligned                                      */
/*        N%4=0 K%8=0                                                   */
/*  C MODEL                                                             */
/*       N = Nlen                                                       */
/*       K = Klen | Kstride                                             */
/*======================================================================*/
#if 0
void gemacca_cn(uchar *x, int N, int K, int * xsum, int y_offset)
{
    int i,, k;
    int ksize = 0xffff&(K >> 16);
    int kstride = 0xffff& K;
    int x_val;
    int sum;

    for (i=0; i < N; i++) {
        sum = xsum[i];
        for (k=0; k < ksize; k++) {
           x_val = x[i*kstride+k];
           sum  += x_val ;
        }
        xsum[i] = sum*y_offset;
    }
}
#endif
/*=============================================================================*/
        .text
        .file "gemacca_h.S"
        .global gemacca_asm
        .balign 32
        .type  gemacca_asm, @function
gemacca_asm:
/*=============================================================================*/
#define ptr_x         r0     //
#define n             r1     //n is number of rows to be summed
#define k             r2     //k | kstride
#define ptr_xsum      r3     //
#define y_offset      r4
#define kjump         r6     //kstride
#define ki            r7     //
#define kstride       r8     //alias to k1
#define mkk           M1     //
#define kk_1          M0     //skip back
#define c32_kstride   r10    //11111111
#define c8_kstride    r11    //11111111
#define l1xptri       r12    //
#define l1xptr        r13    //11111111
#define kstride2      r9     //
#define x07x04_x03x00 r21:20 //111111--
#define x07x04        r21    //111111--
#define x03x00        r20    //11------
#define x17x14_x13x10 r15:14 //-1111111
#define x17x14        r15    //-1111111
#define x13x10        r14    //-111----
#define x27x24_x23x20 r21:20 //-------1
#define x27x24        r21    //-------1
#define x23x20        r20    //-------1
#define x37x34_x33x30 r15:14 //--------
#define x37x34        r15    //--------
#define x33x30        r14    //--------
#define sum01_sum00   r17:16
#define sum11_sum10   r19:18
#define sum01         r17
#define sum00         r16
#define sum11         r19
#define sum10         r18
/*=============================================================================*/
       {
           allocframe(#32)                              //
           kjump = lsr(k, #16)                          //size of k
           n = lsr(n, #1)                               //divide by 2
           kstride = zxth(k)                            //
       } {
           memd(sp+#0)  = r17:16                        //
           memd(sp+#8)  = r19:18                        //
           loop1(.L_loopN, n)                           //[ , P]for(i=0; i < n; i+=2){
           ki = lsr(k, #20)                             //k/16
       } {
           memd(sp+#16) = r21:20                        //
           kstride2 = asl(kstride, #1)                  //2*kstride
           l1xptr = addasl(ptr_x, kstride, #1)          //l1 fetch 2 klines ahead
           c32_kstride = sub(#32, kstride)             //zag back to next column of lines
       } {
           kjump = sub(kstride2, kjump)                 //+32 - 4*k
           c8_kstride = sub(#16, kstride)               //zag back to next column of dwords
           mkk = kstride                                //stride k   
           p2 = cmp.eq(r0, r0)                          //or(p2, !p2) 
       } {
           kk_1 = c8_kstride                            //
           l1xptr = addasl(l1xptr, kstride, #1)         //[ , P]advance by 2k strip
           l1xptri = l1xptr                             //[ , P]make temp copy
       }
/*=============================================================================*/
        .balign 32
.L_loopN:
       {
           sum01_sum00 = combine(#0, #0)                //
           sum11_sum10 = combine(#0, #0)                //
           loop0(.L_loopK, ki)                          //[ , P]ki is k1/2 - 2
       }
/*============================================================================*/
        .balign 32
.L_loopK:
       {   
           x17x14_x13x10 = memd(ptr_x+#8)               //[0,0]
           x07x04_x03x00 = memd(ptr_x++mkk)             //[0,0]
       } {
           sum01_sum00 +=vraddub(x17x14_x13x10, x07x04_x03x00) //[0,1]
           x27x24_x23x20 = memd(ptr_x++kk_1)            //[0,3]6
           x37x34_x33x30 = memd(ptr_x+#8)               //[0,3]  
           p2 = not(p2)                                 //[0,3]
       } {
           sum11_sum10+=vraddub(x37x34_x33x30, x27x24_x23x20) //[0,4]
           //dcfetch(l1xptri+#0)                          //[0,1]prefetch next line
           if(!p2)l1xptri = add(l1xptri, kstride)       //[0,1]nex line
           if(p2) l1xptri = add(l1xptri,c32_kstride)    //[0,3]
       }:endloop0
       {
           sum00 = add(sum01, sum00)                    //
           sum01 = add(sum11, sum10)                    //
           ptr_x = add(ptr_x, kjump)                    //skip back to next row 
       } {
           sum00 = mpyi(sum00, y_offset)                //
           sum01 = mpyi(sum01, y_offset)                //
           sum11_sum10 = memd(ptr_xsum+#0)              //
       } {
           sum00 = add(sum00, sum10)                    //
           sum01 = add(sum01, sum11)                    //
           l1xptri = l1xptr                             //[ , P]make temp copy
       } {
           l1xptr = addasl(l1xptr, kstride, #1)         //[ , P]advance by 2k strip
           memd(ptr_xsum++#1<<3) = sum01_sum00          //
           p2 = or(p2, !p2)                             //
       }:endloop1
/*=============================================================================*/
       {
           r17:16 = memd(sp+#0)                         //restore stack and return
           r19:18 = memd(sp+#8)                         //Q
       } {
           r21:20 = memd(sp+#16)                        //Q
       } {
           dealloc_return                               //Q
       }
.L_end:
/*=============================================================================*/
      .size gemacca_asm, .L_end-gemacca_asm
