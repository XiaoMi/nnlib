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
/*======================================================================*/
/*  FUNCTIONS      : gvmsumimw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform vector sum on input stream, result at 32bits */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*  DJH                 05/10/16       added post add for x and y offset*/
/*  DJH                 07/10/16       rewrote pre-transpose            */
/*  DJH                 09/16/16       fix over prefetch by 16 now 8    */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->  16*K*N/32+11*N/4+24                                          */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 1040 bytes                                            */
/*     STACK    = 48 bytes                                              */
/*     ASSUMPTIONS                                                      */
/*        y and z are 128 byte aligned                                  */
/*        x is 8byte aligned                                            */
/*        N%4=0 K%8=0 M%128=0                                           */
/*  C MODEL                                                             */
/*       N = Nlen                                                       */
/*       K = Klen | Kstride                                             */
/*       M = Mlen | Mstride                                             */
/*======================================================================*/
#if 0
void gvmsumimw_cn(uint8 * a, int * c, int N, int K, uchar y_offset, int z_offset) {
    int i, j, k;
    int32 sum;
    uint8 a_val, b_val;

    for (j=0; j < M; j++) {
        for (i=0; i < N; i++) {
            sum = 0;
            for (k=0; k < K; k++) {
              a_val = a[i*K+k];
              sum  += a_val ;
            }
            c[i*M+j] = sum*y_offset + z_offset;
        }
    }
    return;
}
#endif
/*=============================================================================*/
        .text
        .file "gvmsumimw_h.S"
        .global gvmsumimw_asm
        .balign 32
        .type  gvmsumimw_asm, @function
gvmsumimw_asm:
/*=============================================================================*/
#define ptr_x         r0    //data
#define ptr_xsum      r1    //results
#define out_width     r2    //out_width 
#define skip_back     r3    //skip back to next line wrt the stride, pad and filt_width and depth
#define stride        r4    //stride*depth  
#define filt_width    r5    //filt_width*depth elements in the filter length
#define out_height    r6    //number of vertical lines to perform
#define filt_offset   r7    //8bit value to be subtracted
#define z_offset      r8    //32bit value to be added K*xo*yo
/*=============================================================================*/
#define ki            r9    //
#define ptr_x0        r10   //
#define sum           r11   //
#define sum1_sum0     r17:16//
#define sum1          r17   //
#define sum0          r16   //
                            //
#define x07x04x03x00  r13:12//
#define x07x04        r13   //
#define x03x00        r12   //
#define x0fx0cx0bx08  r15:14//
#define x0fx0c        r15   //
#define x0bx08        r14   //
/*=============================================================================*/
       {   
           out_height = memw(sp+#0<<2)      //
           filt_offset = memw(sp+#1<<2)     //
       } {
           z_offset = memw(sp+#2<<2)         //extract filt_width*depth 
           allocframe(#16)                   //
           ki = lsr(filt_width, #4)          //k / 16
       } {
           memd(sp+#0)  = r17:16             //
           memd(sp+#8)  = r19:18             //
           ki = add(ki, #-1)                 //
       }
/*============================================================================*/
        .balign 32
.L_height:
       { 
           loop1(.L_width, out_width)       //[ , P]for(i=0; i < n; i+=4){
           out_height = add(out_height, #-1)
       }
        .balign 32
.L_width:
       {
           ptr_x0 = ptr_x
           ptr_x = add(ptr_x, stride)        //ptr_x += stride
           loop0(.L_filt_width, ki)          //[P, 9]ki is k1/4 - 2
       } {
           sum1_sum0 = combine(#0, #0)       //
           x0fx0cx0bx08 = memd(ptr_x0+#8)    //[0, 0]
           x07x04x03x00 = memd(ptr_x0++#16)  //[0, 0]
       }
        .balign 32
.L_filt_width:
       {
           sum1_sum0 += vraddub(x0fx0cx0bx08, x07x04x03x00) //[1,0]
           x0fx0cx0bx08 = memd(ptr_x0+#8)    //[0, 1]
           x07x04x03x00 = memd(ptr_x0++#16)  //[0, 1]
       }:endloop0 
       {
           sum1_sum0 += vraddub(x0fx0cx0bx08, x07x04x03x00) //[1,1]
       } {
           sum0 = add(sum0, sum1)
           sum = z_offset
       } {
           sum += mpyi(sum0, filt_offset)       //
       } {
           memw(ptr_xsum++#1<<2) = sum       //[E,  ]
       }:endloop1
       {
           ptr_x = add(ptr_x, skip_back)     //[E,  ]next line
           p1 = cmp.eq(out_height, #0)
           if(!p1.new) jump:t .L_height
       }
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)              //restore stack
           r19:18 = memd(sp+#8)              //Q
       } {
           dealloc_return                    //Q
       }
.L_end:
/*=============================================================================*/
      .size gvmsumimw_asm, .L_end-gvmsumimw_asm
