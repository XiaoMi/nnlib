
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
/*  FUNCTIONS      : gemmpybbw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform gemm matrix multiply, result left at 32bits  */
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
void gemmpybbw_cn(uint8 * a, uint8 * b, int * c, int N, int M, int K) {
    int i, j, k;
    int32 sum;
    uint8 a_val, b_val;

    for (j=0; j < M; j++) {
        for (i=0; i < N; i++) {
            sum = 0;
            for (k=0; k < K; k++) {
              a_val = a[i*K+k];
              b_val = b[k*M+j];
              sum  += a_val * b_val ;
            }
            c[i*M+j] = sum;
        }
    }
    return;
}
#endif
/*=============================================================================*/
        .text
        .file "gemmpybbw_h.S"
        .global gemmpybbw_asm
        .balign 32
        .type  gemmpybbw_asm, @function
gemmpybbw_asm:
/*=============================================================================*/
#define ptr_x         r0    //data
#define ptr_yi        r1    //weights
#define ptr_z         r2    //results
#define n             r3    //n %8 number of patches
#define m             r4    //is stride of weights matrix k*32 always 32 wide
#define k             r5    //ksize %16 | k - stride
/*=============================================================================*/
#define ksize         r28    //amount of data in this job
#define ki            r9     //
#define kstride7      r8     //
#define kjump         r4     //16-8kstride  
#define ptr_y         r6     //
#define l1xptri0      r7     //
#define l1xptri0_ptr_y r7:6  //
#define l1xptri1      r10    //
#define l1xptri2      r11    //
#define l1xptri3      r12    //
#define l1xptr        r13    //
#define skip          r14    //
#define back          r15    //
#define kk            M0     //
#define mm            M1     //

#define x07x04x03x00  r17:16 //1111-----------1
#define x0fx0cx0bx08  r23:22 //11-------------1
#define x17x14x13x10  r19:18 //1111------------
#define x1fx1cx1bx18  r21:20 //11--------------
#define x27x24x23x20  r21:20 //---111----------
#define x2fx2cx2bx28  r23:22 //---11111--------
#define x37x34x33x30  r19:18 //----11----------
#define x3fx3cx3bx38  r17:16 //----1111--------
#define x47x44x43x40  r21:20 //-------111------
#define x4fx4cx4bx48  r19:18 //-------11111----
#define x57x54x53x50  r25:24 //--------11------
#define x5fx5cx5bx58  r17:16 //--------1111----
#define x67x64x63x60  r23:22 //----------1111--
#define x6fx6cx6bx68  r25:24 //----------111111
#define x77x74x73x70  r27:26 //-----------111--
#define x7fx7cx7bx78  r21:20 //-----------11111
#define x03x00        r16  //1111-----------1
#define x0bx08        r22  //11-------------1
#define x13x10        r18  //1111------------
#define x1bx18        r20  //11--------------
#define x23x20        r20  //---111----------
#define x2bx28        r22  //---11111--------
#define x33x30        r18  //----11----------
#define x3bx38        r16  //----1111--------
#define x43x40        r20  //-------111------
#define x4bx48        r18  //-------11111----
#define x53x50        r24  //--------11------
#define x5bx58        r16  //--------1111----
#define x63x60        r22  //----------111---
#define x6bx68        r24  //----------11111-
#define x73x70        r26  //-----------111--
#define x7bx78        r20  //-----------11111
#define x07x04        r17  //1111-----------1
#define x0fx0c        r23  //11-------------1
#define x17x14        r19  //1111------------
#define x1fx1c        r21  //11--------------
#define x27x24        r21  //---111----------
#define x2fx2c        r23  //---11111--------
#define x37x34        r19  //----11----------
#define x3fx3c        r17  //----1111--------
#define x47x44        r21  //-------111------
#define x4fx4c        r19  //-------11111----
#define x57x54        r25  //--------11------
#define x5fx5c        r17  //--------1111----
#define x67x64        r23  //----------111---
#define x6fx6c        r25  //----------11111-
#define x77x74        r27  //-----------111--
#define x7fx7c        r21  //-----------11111
/*=============================================================================*/
#define z0            v0   //
#define z1            v1   //
#define z1z0          v1:0 //
#define z2            v2   //
#define z3            v3   //
#define z3z2          v3:2 //
#define z4            v4   //
#define z5            v5   //
#define z5z4          v5:4 //
#define z6            v6   //
#define z7            v7   //
#define z7z6          v7:6 //
#define y0            v8   //
#define y1            v9   //
#define y2            v10  //
#define y3            v11  //
#define vzero         v12  //
/*=============================================================================*/
       {   allocframe(#64)                   //
       } {
           memw(sp+#48) = r28                //
           vzero = #0                        //
           ksize = lsr(k, #16)               //extract work
           k = zxth(k)                       //extract stride
       } {
           m = asl(m, #2)                    //ints
           kk = k                            //stride k
       } {
           ki = lsr(ksize, #4)               //k / 16
           kstride7 = asl(k, #3)             //3*kstride
           memd(sp+#0)  = r17:16             //
       } {
           kstride7 = sub(kstride7, k)       //
           l1xptr = ptr_x                    //asl(ptr_x, k, #3) //#48)  //l1 fetch 32 +26 bytes ahead
           n = lsr(n, #3)                    //divide by 8
           memd(sp+#8)  = r19:18             //
       } {
           mm = m                            //
           kjump = sub(#16, kstride7)        //zag back to next column of lines
           ki = add(ki, #-1)                 //
       } { 
           memd(sp+#16) = r21:20             //
           kstride7 += sub(k, ksize)         //correction factor
           k = add(k, k)                     //2*k
           l1xptri0 = add(l1xptr, #80)       //[ , P]advance first fetches by 64
       } {
           loop1(.L_loopN, n)                //[ , P]for(i=0; i < n; i+=4){
           l1xptri1 = add(l1xptr, k)         //[ , P]make temp copy
           memd(sp+#24) = r23:22             //
           memd(sp+#32) = r25:24             //
       } {
           memd(sp+#40) = r27:26             //
           ptr_y = ptr_yi                    //[ , P]
           loop0(.L_loopK, ki)               //[P, 9]ki is k1/4 - 2
           l1xptri2 = add(l1xptri1, k)       //[ , P]make temp copy
       } {
           y0 = vmem(ptr_y++#2)              //[0, 0]32x4
           dcfetch(l1xptri0+#0)              //[0, 0]prefetch next line
           skip = lsr(k, #1)                 //next line += k
           l1xptri3 = add(l1xptri2, k)       //[ , P]make temp copy
       } {
           y1 = vmem(ptr_y+#-1)              //[0, 1]32x4
           l1xptri0 = add(l1xptri0, skip)    //[0, 1]next line
           back = sub(#32, skip)             //skip back
           l1xptr= addasl(l1xptr,k,#2)       //[P,  ]advance by 8k strip
       } {
           x0fx0cx0bx08 = memd(ptr_x+#8)     //[0, 2]
           x07x04x03x00 = memd(ptr_x++kk)    //[0, 2]
           z1z0 = vcombine(vzero, vzero)     //[P, 0]
           z3z2 = vcombine(vzero, vzero)     //[P, 0]
       } {
           x1fx1cx1bx18 = memd(ptr_x+#8)     //[0, 3]
           x17x14x13x10 = memd(ptr_x++kk)    //[0, 3]
           z5z4 = vcombine(vzero, vzero)     //[P, 0]
           z7z6 = vcombine(vzero, vzero)     //[P, 0]
       }
/*============================================================================*/
        .balign 32
.L_loopN:
.L_loopK:
       {
           y2.cur = vmem(ptr_y++#2)          //[0, 4]32x4
           z0.uw += vrmpy(y2.ub, x0bx08.ub)  //[0, 4]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)  //[0, 4]
           dcfetch(l1xptri1+#0)              //[0, 4]prefetch next line
       } {
           y3.cur = vmem(ptr_y+#-1)          //[0, 5]32x4
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)  //[0, 5]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)  //[0, 5]
           l1xptri1 = add(l1xptri1, skip)    //[0, 5]next line
       } {
           z0.uw += vrmpy(y0.ub, x03x00.ub)  //[0, 6]
           z1.uw += vrmpy(y0.ub, x13x10.ub)  //[0, 6]
           x2fx2cx2bx28 = memd(ptr_x+#8)     //[0, 6]
           x27x24x23x20 = memd(ptr_x++kk)    //[0, 6]
       } {
           z0.uw += vrmpy(y1.ub, x07x04.ub)  //[0, 7]
           z1.uw += vrmpy(y1.ub, x17x14.ub)  //[0, 7]
           x3fx3cx3bx38 = memd(ptr_x+#8)     //[0, 7]
           x37x34x33x30 = memd(ptr_x++kk)    //[0, 7]
       } {
           z2.uw += vrmpy(y0.ub, x23x20.ub)  //[0, 8]
           z3.uw += vrmpy(y0.ub, x33x30.ub)  //[0, 8]
           dcfetch(l1xptri2+#0)              //[0, 8]prefetch next line
           l1xptri2 = add(l1xptri2, skip)    //[0, 8]next line
       } {
           z2.uw += vrmpy(y1.ub, x27x24.ub)  //[0, 9]
           z3.uw += vrmpy(y1.ub, x37x34.ub)  //[0, 9]
           dcfetch(l1xptri3+#0)              //[0, 9]prefetch next line
           l1xptri3 = add(l1xptri3, skip)    //[0, 9]next line
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)  //[0,10]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)  //[0,10]
           x4fx4cx4bx48 = memd(ptr_x+#8)     //[0,10]
           x47x44x43x40 = memd(ptr_x++kk)    //[0,10]
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)  //[0,11]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)  //[0,11]
           x5fx5cx5bx58 = memd(ptr_x+#8)     //[0,11]
           x57x54x53x50 = memd(ptr_x++kk)    //[0,11]
       } {
           z4.uw += vrmpy(y0.ub, x43x40.ub)  //[0,12]
           z5.uw += vrmpy(y0.ub, x53x50.ub)  //[0,12]
           skip = back                       //[0,12]next line
           back = skip                       //[0,12]previous line + 32
       } {
           z4.uw += vrmpy(y1.ub, x47x44.ub)  //[0,13]
           z5.uw += vrmpy(y1.ub, x57x54.ub)  //[0,13]
           x6fx6cx6bx68 = memd(ptr_x+#8)     //[0,13]
           x67x64x63x60 = memd(ptr_x++kk)    //[0,13]
       } {
           z4.uw += vrmpy(y2.ub, x4bx48.ub)  //[0,14]
           z5.uw += vrmpy(y2.ub, x5bx58.ub)  //[0,14]
           x7fx7cx7bx78 = memd(ptr_x+#8)     //[0,14]
           x77x74x73x70 = memd(ptr_x+#0)     //[0,14]
       } {
           z4.uw += vrmpy(y3.ub, x4fx4c.ub)  //[0,15]
           z5.uw += vrmpy(y3.ub, x5fx5c.ub)  //[0,15]
           ptr_x = add(ptr_x, kjump)         //[0,15]
       } {
           z6.uw += vrmpy(y0.ub, x63x60.ub)  //[0,16]
           z7.uw += vrmpy(y0.ub, x73x70.ub)  //[0,16]
           y0 = vmem(ptr_y++#2)              //[1, 0]32x4
           dcfetch(l1xptri0+#0)              //[1, 0]prefetch next line
       } {
           z6.uw += vrmpy(y1.ub, x67x64.ub)  //[0,17]
           z7.uw += vrmpy(y1.ub, x77x74.ub)  //[0,17]
           y1 = vmem(ptr_y+#-1)              //[1, 1]32x4
           l1xptri0 = add(l1xptri0, skip)    //[1, 1]next line
       } {
           z6.uw += vrmpy(y2.ub, x6bx68.ub)  //[0,18]
           z7.uw += vrmpy(y2.ub, x7bx78.ub)  //[0,18]
           x0fx0cx0bx08 = memd(ptr_x+#8)     //[1, 2]
           x07x04x03x00 = memd(ptr_x++kk)    //[1, 2]
       } {
           z6.uw += vrmpy(y3.ub, x6fx6c.ub)  //[0,19]
           z7.uw += vrmpy(y3.ub, x7fx7c.ub)  //[0,19]
           x1fx1cx1bx18 = memd(ptr_x+#8)     //[1, 3]
           x17x14x13x10 = memd(ptr_x++kk)    //[1, 3]
       }:endloop0 
       {
           y2.cur = vmem(ptr_y++#2)          //[1, 4]32x4
           z0.uw += vrmpy(y2.ub, x0bx08.ub)  //[1, 4]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)  //[1, 4]
           l1xptri0=l1xptr                   //[P,  ]
       } {
           y3.cur = vmem(ptr_y+#-1)          //[1, 5]32x4
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)  //[1, 5]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)  //[1, 5]
           l1xptri1 = add(l1xptr, k)         //[ , P]make temp copy
       } {
           z0.uw += vrmpy(y0.ub, x03x00.ub)  //[1, 6]
           z1.uw += vrmpy(y0.ub, x13x10.ub)  //[1, 6]
           x2fx2cx2bx28 = memd(ptr_x+#8)     //[1, 6]
           x27x24x23x20 = memd(ptr_x++kk)    //[1, 6]
       } {
           z0.uw += vrmpy(y1.ub, x07x04.ub)  //[1, 7]
           z1.uw += vrmpy(y1.ub, x17x14.ub)  //[1, 7]
           x3fx3cx3bx38 = memd(ptr_x+#8)     //[1, 7]
           x37x34x33x30 = memd(ptr_x++kk)    //[1, 7]
       } {
           z2.uw += vrmpy(y0.ub, x23x20.ub)  //[1, 8]
           z3.uw += vrmpy(y0.ub, x33x30.ub)  //[1, 8]
           l1xptri2 = add(l1xptri1, k)       //[ , P]make temp copy
       } {
           z2.uw += vrmpy(y1.ub, x27x24.ub)  //[1, 9]
           z3.uw += vrmpy(y1.ub, x37x34.ub)  //[1, 9]
           l1xptri3 = add(l1xptri2, k)       //[ , P]make temp copy
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)  //[1,10]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)  //[1,10]
           x4fx4cx4bx48 = memd(ptr_x+#8)     //[1,10]
           x47x44x43x40 = memd(ptr_x++kk)    //[1,10]
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)  //[1,11]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)  //[1,11]
           x5fx5cx5bx58 = memd(ptr_x+#8)     //[1,11]
           x57x54x53x50 = memd(ptr_x++kk)    //[1,11]
       } {
           z4.uw += vrmpy(y0.ub, x43x40.ub)  //[1,12]
           z5.uw += vrmpy(y0.ub, x53x50.ub)  //[1,12]
           vmem(ptr_z++mm) = z0              //[E,  ]
           z0 = #0
       } {
           z4.uw += vrmpy(y1.ub, x47x44.ub)  //[1,13]
           z5.uw += vrmpy(y1.ub, x57x54.ub)  //[1,13]
           x6fx6cx6bx68 = memd(ptr_x+#8)     //[1,13]
           x67x64x63x60 = memd(ptr_x++kk)    //[1,13]
       } {
           z4.uw += vrmpy(y2.ub, x4bx48.ub)  //[1,14]
           z5.uw += vrmpy(y2.ub, x5bx58.ub)  //[1,14]
           x7fx7cx7bx78 = memd(ptr_x+#8)     //[1,14]
           x77x74x73x70 = memd(ptr_x+#0)     //[1,14]
       } {
           z4.uw += vrmpy(y3.ub, x4fx4c.ub)  //[1,15]
           z5.uw += vrmpy(y3.ub, x5fx5c.ub)  //[1,15]
           ptr_x = add(ptr_x, kjump)         //[1,15]
           vmem(ptr_z++mm) = z1              //[E,  ]
       } {
           z6.uw += vrmpy(y0.ub, x63x60.ub)  //[1,16]
           z7.uw += vrmpy(y0.ub, x73x70.ub)  //[1,16]
           vmem(ptr_z++mm) = z2              //[E,  ]
           ptr_x = add(ptr_x, kstride7)      //[E,  ]
       } {
           z6.uw += vrmpy(y1.ub, x67x64.ub)  //[1,17]
           z7.uw += vrmpy(y1.ub, x77x74.ub)  //[1,17]
           vmem(ptr_z++mm) = z3              //[E,  ]
           z1 = #0                           //[P, 0]
       } {
           z6.uw += vrmpy(y2.ub, x6bx68.ub)  //[1,18]
           z7.uw += vrmpy(y2.ub, x7bx78.ub)  //[1,18]
           vmem(ptr_z++mm) = z4              //[E,  ]
           z2 = #0                           //[P, 0]
       } {
           z6.uw += vrmpy(y3.ub, x6fx6c.ub)  //[1,19]
           z7.uw += vrmpy(y3.ub, x7fx7c.ub)  //[1,19]
           vmem(ptr_z++mm) = z5              //[E,  ]
           z3 = #0                           //[P, 0]
       } {
           l1xptr= addasl(l1xptr,k,#2)       //[P,  ]advance by 8k strip
           vmem(ptr_z++mm) = z6              //[E,  ]
           skip = lsr(k, #1)                 //[P,  ]
       } {
           back = sub(#32, skip)             //
           vmem(ptr_z++mm) = z7              //[E,  ]
           loop0(.L_prefetch, #3)            //2 lines ahead
       }
       .balign 32
.L_prefetch:
       {  
           dcfetch(l1xptri0+#0)              //[E, 4]prefetch next line
           l1xptri0 = add(l1xptri0, skip)
       } {
           dcfetch(l1xptri1+#0)              //[E, 4]prefetch next line
           l1xptri1 = add(l1xptri1, skip)
       } {
           dcfetch(l1xptri2+#0)              //[E, 9]prefetch next line
           l1xptri2 = add(l1xptri2, skip)
       } {
           dcfetch(l1xptri3+#0)              //[E, 8]prefetch next line
           l1xptri3 = add(l1xptri3, skip)
       } {
           dcfetch(l1xptri0+#0)              //[E, 4]prefetch next line
           l1xptri0 = add(l1xptri0, back)
       } {
           dcfetch(l1xptri1+#0)              //[E, 4]prefetch next line
           l1xptri1 = add(l1xptri1, back)
       } {
           dcfetch(l1xptri2+#0)              //[E, 9]prefetch next line
           l1xptri2 = add(l1xptri2, back)
       } {
           dcfetch(l1xptri3+#0)              //[E, 8]prefetch next line
           l1xptri3 = add(l1xptri3, back)
       }:endloop0
       {
           x0fx0cx0bx08 = memd(ptr_x+#8)     //[0, 2]
           dcfetch(l1xptri0+#0)              //[0, 0]prefetch next line
           ptr_y = ptr_yi                    //
           z4 = #0
       } {
           y0 = vmem(ptr_y++#2)              //[0, 0]32x4
           x07x04x03x00 = memd(ptr_x++kk)    //[0, 2]
           l1xptri0 = add(l1xptri0, skip)    //[0, 1]next line
           z5 = #0
       } {
           y1 = vmem(ptr_y+#-1)              //[0, 1]32x4
           x1fx1cx1bx18 = memd(ptr_x+#8)     //[0, 3]
           z6 = #0
       } {
           z7 = #0
           x17x14x13x10 = memd(ptr_x++kk)    //[0, 3]
           loop0(.L_loopK, ki)               //[P, 9]ki is k1/4 - 2
#ifdef TOOL_V_8_0_X_ASSEMBLER_ISSUE_WORKAROUND_EXTRA_NOP
		   } {
           nop  //NOTE: THIS EXTRA NOP PACKET ADDED TO WORK AROUND TOOL V.8.0.X ISSUE (JIRA) [QTOOL-27821] - NOP NOT NEEDED FOR TOOLS V.7.2 & V8.1
#endif
       }:endloop1
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)              //restore stack
           r19:18 = memd(sp+#8)              //Q
       } {
           r21:20 = memd(sp+#16)             //Q
           r23:22 = memd(sp+#24)             //Q
       } {
           r25:24 = memd(sp+#32)             //Q
           r27:26 = memd(sp+#40)             //Q
       } {
           r28 = memw(sp+#48)                //Q
       } {
           dealloc_return                    //Q
       }
.L_end:
/*=============================================================================*/
      .size gemmpybbw_asm, .L_end-gemmpybbw_asm
