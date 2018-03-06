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
/*  FUNCTIONS      : gemmpybbw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform gemm matrix multiply accumulate,             */
/*                 result left at 32bits                                */
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
/*     ->  w*D*N/256+11*N/4+24                                          */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 1056 bytes                                            */
/*     STACK    = 48 bytes                                              */
/*     ASSUMPTIONS                                                      */
/*        y and z are 128 byte aligned                                  */
/*        x is 8byte aligned                                            */
/*        N%4=0 K%16=0 M%32=0                                           */
/*  C MODEL                                                             */
/*======================================================================*/
#if 0
void gvmmpybbw_cn(uint8 * a, uint8 * b, int * c, int N, int M, int K) {
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
        .file "gvmmacbbw_h.S"
        .global gvmmacbbw_asm
        .balign 32
        .type  gvmmacbbw_asm, @function
gvmmacbbw_asm:
/*=============================================================================*/
#define ptr_x         r0    //data
#define ptr_yi        r1    //weights
#define ptr_z         r2    //results
#define out_width     r3    //out_width
#define skip_back     r4    //(out_width4 - outwidth-1)*skip back*stride*depth
#define m             r5    //is stride of weights matrix *32 always 32 wide
#define stride        r6    //stride*depth
#define filt_width    r7    //depth*filt_width
#define out_height    r8    //number of vertical lines to perform
#define PREFETCH      64    //hwo far ahead to fetch data of ptrs
/*=============================================================================*/
#define ki            r9     //
#define ptr_x0        r21
#define ptr_x1        r20
#define ptr_x2        r22
#define ptr_x3        r11
#define ptr_y         r10     //
#define out_width4    r24
#define col_count     r23
                             //01234567
#define x07x04x03x00  r13:12 //11-----1
#define x07x04        r13    //11-----1
#define x03x00        r12    //1------1
#define x0fx0cx0bx08  r15:14 //1111---1
#define x0fx0c        r15    //1111---1
#define x0bx08        r14    //111----1
#define x17x14x13x10  r19:18 //11------
#define x17x14        r19    //11------
#define x13x10        r18    //1-------
#define x1fx1cx1bx18  r17:16 //1111----
#define x1fx1c        r17    //1111----
#define x1bx18        r16    //111-----
#define x27x24x23x20  r13:12 //---111--
#define x27x24        r13    //---111--
#define x23x20        r12    //---11---
#define x2fx2cx2bx28  r19:18 //---1111-
#define x2fx2c        r19    //---11111
#define x2bx28        r18    //---1111-
#define x37x34x33x30  r15:14 //----11--
#define x37x34        r15    //----11--
#define x33x30        r14    //----1---
#define x3fx3cx3bx38  r17:16 //----1111
#define x3fx3c        r17    //----1111
#define x3bx38        r16    //----111-
/*=============================================================================*/
#define z0            v0   //
#define z1            v1   //
#define z1z0          v1:0 //
#define z2            v2   //
#define z3            v3   //
#define z3z2          v3:2 //
#define y0            v8   //
#define y1            v9   //
#define y2            v10  //
#define y3            v11  //
#define vzero         v12  //
#define WO            1
/*=============================================================================*/
       {
           stride = memw(sp+#0<<2)           //extract stride*depth
           filt_width = memw(sp+#1<<2)       //extract filt_width*depth
           m = asl(m, #2)                    //ints
       } { 
           out_height = memw(sp+#2<<2)       //number of output lines
           allocframe(#64)                   //
       } {
           M0 = m                            //
           memd(sp+#0)  = r17:16             //
           memd(sp+#8)  = r19:18             //
       } {
           ki = lsr(filt_width, #4)          //k / 16
           m = mpyi(m, #-3)
           memd(sp+#16) = r21:20             //
           memd(sp+#24) = r23:22             //
       } {
           memd(sp+#32) = r25:24             //
           memd(sp+#40) = r27:26             //
           vzero = #0                        //
           out_width4 = add(out_width, #3)
       } {
           M1 = m                            //
           out_width4 = lsr(out_width4, #2)
           ki = add(ki, #-1)                 //
       }
/*============================================================================*/
        .balign 32
.L_height:
       { 
           loop1(.L_width, out_width4)       //[ , P]for(i=0; i < n; i+=4){
           col_count = out_width
           out_height = add(out_height, #-1)
           ptr_y = ptr_yi                    //[ , P]
       }
        .balign 32
.L_width:
       {
#if WO
           y0 = vmem(ptr_y++#2)              //[0, 0]32x4
#endif
           dcfetch(ptr_x+#PREFETCH)          //[0, 0]prefetch next line
       } {
#if WO
           y1 = vmem(ptr_y+#-1)              //[0, 1]32x4
#endif
           dcfetch(ptr_x1+#PREFETCH)         //[0, 1]prefetch next line
           ptr_x0 = ptr_x
           ptr_x1 = add(ptr_x, stride)       // x1 = x0 + depth*stride
       } {
           x0fx0cx0bx08 = memd(ptr_x0+#8)    //[0, 2]
           z0 = vmem(ptr_z++M0)
#if WO
#endif
       } {
           z1 = vmem(ptr_z++M0)
#if WO
#endif
           x07x04x03x00 = memd(ptr_x0++#16)  //[0, 2]
           ptr_x2 = add(ptr_x1, stride)      // x2 = x1 + depth*stride
           ptr_x3 = addasl(ptr_x1, stride,#1)// x3 = x2 + depth*stride
       } {
           z2 = vmem(ptr_z++M0)
#if WO
#endif
           x1fx1cx1bx18 = memd(ptr_x1+#8)    //[0, 3]
       } {
           z3 = vmem(ptr_z++M1)
#if WO
#endif
           x17x14x13x10 = memd(ptr_x1++#16)  //[0, 3]
           loop0(.L_filt_width, ki)          //[P, 9]ki is
           ptr_x = addasl(ptr_x, stride, #2) //ptr_x += 4*stride
       } 
        .balign 32
.L_filt_width:
       {
           dcfetch(ptr_x2+#PREFETCH)         //[0, 4]prefetch next line
           z0.uw += vrmpy(y0.ub, x03x00.ub)  //[0, 4]
           z1.uw += vrmpy(y0.ub, x13x10.ub)  //[0, 4]
#if WO
           y2 = vmem(ptr_y++#2)              //[0, 4]32x4
#endif
       } {
           dcfetch(ptr_x3+#PREFETCH)         //[0, 5]prefetch next line
           z0.uw += vrmpy(y1.ub, x07x04.ub)  //[0, 5]
           z1.uw += vrmpy(y1.ub, x17x14.ub)  //[0, 5]
#if WO
           y3 = vmem(ptr_y+#-1)              //[0, 5]32x4
#endif
       } {
           z0.uw += vrmpy(y2.ub, x0bx08.ub)  //[0, 6]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)  //[0, 6]
           x2fx2cx2bx28 = memd(ptr_x2+#8)    //[0, 6]
           x27x24x23x20 = memd(ptr_x2++#16)  //[0, 6]
#if WO
#endif
       } {
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)  //[0, 7]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)  //[0, 7]
           x3fx3cx3bx38 = memd(ptr_x3+#8)    //[0, 7]
           x37x34x33x30 = memd(ptr_x3++#16)  //[0, 7]
#if WO
#endif
       } {
           z2.uw += vrmpy(y0.ub, x23x20.ub)  //[0, 8]
           z3.uw += vrmpy(y0.ub, x33x30.ub)  //[0, 8]
#if WO
           y0 = vmem(ptr_y++#2)              //[1, 0]32x4
#endif
           dcfetch(ptr_x0+#PREFETCH)         //[1, 0]prefetch next line
       } {
           z2.uw += vrmpy(y1.ub, x27x24.ub)  //[0, 9]
           z3.uw += vrmpy(y1.ub, x37x34.ub)  //[0, 9]
#if WO
           y1 = vmem(ptr_y+#-1)              //[1, 1]32x4
#endif
           dcfetch(ptr_x1+#PREFETCH)         //[1, 1]prefetch next line
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)  //[0,10]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)  //[0,10]
           x0fx0cx0bx08 = memd(ptr_x0+#8)    //[1, 2]
           x07x04x03x00 = memd(ptr_x0++#16)  //[1, 2]
#if WO
#endif
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)  //[0,11]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)  //[0,11]
           x1fx1cx1bx18 = memd(ptr_x1+#8)    //[1, 3]
           x17x14x13x10 = memd(ptr_x1++#16)  //[1, 3]
#if WO
#endif
       }:endloop0 
       {
           dcfetch(ptr_x2+#PREFETCH)         //[1, 4]prefetch next line
           z0.uw += vrmpy(y0.ub, x03x00.ub)  //[1, 4]
           z1.uw += vrmpy(y0.ub, x13x10.ub)  //[1, 4]
#if WO
           y2 = vmem(ptr_y++#2)              //[1, 4]32x4
#endif
       } {
           dcfetch(ptr_x3+#PREFETCH)         //[1, 5]prefetch next line
           z0.uw += vrmpy(y1.ub, x07x04.ub)  //[1, 5]
           z1.uw += vrmpy(y1.ub, x17x14.ub)  //[1, 5]
#if WO
           y3 = vmem(ptr_y+#-1)              //[1, 5]32x4
#endif
       } {
           z0.uw += vrmpy(y2.ub, x0bx08.ub)  //[1, 6]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)  //[1, 6]
           x2fx2cx2bx28 = memd(ptr_x2+#8)    //[1, 6]
           x27x24x23x20 = memd(ptr_x2++#16)  //[1, 6]
#if WO
#endif
       } {
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)  //[1, 7]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)  //[1, 7]
           x3fx3cx3bx38 = memd(ptr_x3+#8)    //[1, 7]
           x37x34x33x30 = memd(ptr_x3++#16)  //[1, 7]
       } {
           vmem(ptr_z++M0) = z0              //[E,  ]
#if WO
#endif
           z2.uw += vrmpy(y0.ub, x23x20.ub)  //[1, 8]
           z3.uw += vrmpy(y0.ub, x33x30.ub)  //[1, 8]
           p0 = cmp.gt(col_count, #1)
       } {
           if(p0)vmem(ptr_z++M0) = z1        //[E,  ]
#if WO
#endif
           z2.uw += vrmpy(y1.ub, x27x24.ub)  //[1, 9]
           z3.uw += vrmpy(y1.ub, x37x34.ub)  //[1, 9]
           p0 = cmp.gt(col_count, #2)
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)  //[1,10]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)  //[1,10]
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)  //[1,11]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)  //[1,11]
           if(p0)vmem(ptr_z++M0) = z2.new    //[E,  ]
#if WO
#endif
           p0 = cmp.gt(col_count, #3)        //
       } {
           if(p0)vmem(ptr_z++M0) = z3        //[E,  ]
#if WO
#endif
           col_count = add(col_count, #-4)   //
           ptr_y = ptr_yi                    //[ , P]
       }:endloop1
       {
           ptr_x = add(ptr_x, skip_back)     //[E,  ]
           p1 = cmp.eq(out_height, #0)
           if(!p1.new) jump:t .L_height
       }
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
           dealloc_return                    //Q
       }
.L_end:
/*=============================================================================*/
      .size gvmmacbbw_asm, .L_end-gvmmacbbw_asm
