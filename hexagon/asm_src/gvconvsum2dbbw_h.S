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
/*  FUNCTIONS      : gvmmpybbw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform gvm vector matrix multiply, result left at   */
/*                 32bits                                               */
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
/*     ->  K*N/256+11*N/4+24                                            */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 960 bytes                                             */
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
        .file "gvconvsum2dbbw_h.S"
        .global gvconvsum2dbbw_asm
        .balign 32
        .type  gvconvsum2dbbw_asm, @function
gvconvsum2dbbw_asm:
/*=============================================================================*/
#define ptr_x         r0    //data
#define ptr_yi        r1    //weights
#define ptr_z         r2    //results
#define in_width      r3    //(pad_x+in_width) * depth
#define out_width     r4    //out_width 
#define m             r5    //is stride of the output matrix always mult of 32
#define stride_depth  r8    //0 stride|depth  between computations
#define filt_width    r6    //1 depth*filt_width  
#define filt_height   r6    //2 filt_hieght lines per filter
#define out_height    r9    //3 number of vertical lines to perform
#define ptr_datasum   r10   //4
#define ptr_weightsum r11   //5
#define ptr_max       r16   //6
#define in_offset     r14   //7
#define zsum          r6    //8
/*=============================================================================*/
#define filt_skip     r13   //the skip back after the fot_width is done for next filt_y
#define stride3_1     r12   //used in prefetch
#define ptr_x0        r11
#define stride4       r13   //
#define stride        r25
#define next_outputs  r23   //jump to input ptr for next set of outputs
#define ptr_y         r9    //
#define col_count     r22
#define c4            r6 
#define mstride       r15
#define fetch_count   r7
#define pre_x         r28
#define PREFETCH      64

// r10 r7 
#define sum1_sum0     r1:0
#define sum1          r1
#define sum0          r0
#define sum3_sum2     r5:4
#define sum3          r5
#define sum2          r4
#define sum5_sum4     r25:24
#define sum5          r25
#define sum4          r24
#define sum7_sum6     r27:26
#define sum7          r27
#define sum6          r26

#define MSTRIDE       M0     //stride*depth
#define M4STRIDE_1    M1     //3*stride*depth-16     0-1-2-3

                             //01234567
#define x07x04x03x00  r21:20 //11-----1
#define x07x04        r21    //11-----1
#define x03x00        r20    //1------1
#define x0fx0cx0bx08  r15:14 //1111---1
#define x0fx0c        r15    //1111---1
#define x0bx08        r14    //111----1
#define x17x14x13x10  r19:18 //11------
#define x17x14        r19    //11------
#define x13x10        r18    //1-------
#define x1fx1cx1bx18  r17:16 //1111----
#define x1fx1c        r17    //1111----
#define x1bx18        r16    //111-----
#define x27x24x23x20  r21:20 //---111--
#define x27x24        r21    //---111--
#define x23x20        r20    //---11---
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
#define z0            v0     //
#define z1            v1     //
#define z1z0          v1:0   //
#define z2            v2     //
#define z3            v3     //
#define z3z2          v3:2   //
#define x0            v4     //
#define x1            v5     //
#define x2            v6     //
#define x3            v7     //
#define y0            v8     //
#define y1            v9     //
#define y2            v10    //
#define y3            v11    //
#define vwsum         v15    //
#define maxomaxe      v13:12 //
#define maxe          v12    //
#define maxo          v13    //
#define vc8000        v14    //
/*=============================================================================*/

       {
           stride_depth = memw(sp+#0<<2)          //extract stride*depth 
           dcfetch(ptr_x)                         //
       } {   
           filt_height = memw(sp+#2<<2)           //extract filt_height 
           //out_height = memw(sp+#3<<2)            //number of output lines
           m = asl(m, #2)                         //in ints
           dcfetch(ptr_x+#32)                         //
       } {
           ptr_weightsum = memw(sp+#5<<2)         //ptr pre computed weight sum
           allocframe(#72)                        // 20<<2
       } {
           memd(sp+#32) = r25:24                  //
           memd(sp+#0)  = r17:16                  //
           stride = lsr(stride_depth, #16)        //
       } {
           memd(sp+#16) = r21:20                  //
           memd(sp+#24) = r23:22                  //
           stride_depth = mpy(stride_depth.H, stride_depth.L)
       } {
           M0 = stride_depth                      //
           memd(sp+#8)  = r19:18                  //
           memd(sp+#40) = r27:26                  //
       } {
           vwsum = vmem(ptr_weightsum+#0)         //
           stride3_1 = addasl(stride_depth, stride_depth,#1)  //3*stride
           r16 = ##0x80000001                     //max negative
       } {
           stride3_1 = sub(#16, stride3_1)        //
           next_outputs = mpyi(filt_height, in_width) 
           vc8000 = vsplat(r16)                   //
           memw(sp+#56) = out_width               //
       } {
           filt_width = memw(sp+#21<<2)           //extract filt_width*depth 
           M1 = stride3_1                         // add to
       } {
           stride3_1 = add(stride3_1, #16)        //used for line prefetch
           stride4= asl(stride_depth, #1)         //4-2*stride
           memw(sp+#48) = ptr_x                   //
           memw(sp+#52) = ptr_yi                  //
       } {
           memw(sp+#60) = m                       //
           next_outputs = sub(next_outputs, stride4)
           filt_skip = sub(filt_width, in_width)
           filt_width = lsr(filt_width, #4)       //filt_width / 16
       } {
           memw(sp+#64) = r28
           ptr_max = memw(sp+#26<<2)              //ptr pre computed max value in output
           filt_width = add(filt_width, #-1)
           p3 = cmp.gt(stride_depth, #96)         //is !(D <= 96) heuristic to fix fall behind fetch
       } {
           if(p3) stride3_1 = sub(stride3_1, stride_depth) //used for line prefetch
           maxe= vmem(ptr_max+#0)
           in_width = mpyi(in_width, stride)      //
           memw(sp+#21<<2) = filt_width           //extract filt_width*depth /16 - 1
       }
/*============================================================================*/
        .balign 32
.L_height:
       { 
           ptr_x0 = memw(sp+#48)                  //
           memw(sp+#23<<2) -= #1                  //out_height = add(out_height, #-1)      //
       } {
           col_count = memw(sp+#56)               //out_width 
           memw(sp+#48) += in_width               //ptr_x += in_width
           pre_x = add(ptr_x0, #PREFETCH)
       }
        .balign 32
.L_width:
       {
           ptr_y = memw(sp+#52)                   //ptr_yi //[P, 0] initialize filter pointer
           filt_height = memw(sp+#22<<2)          //extract filt_height 
           fetch_count = #0
       } {
           y0 = vmem(ptr_y++#2)                   //[0, 0]32x4
           z1z0 = vcombine(vwsum, vwsum)          //[P, 0]
           dcfetch(pre_x)
       } {
           loop1(.L_filt_height, filt_height)     //[P, 0]for(filt_y=0; filt_y < n; filt_y+=1){
           y1 = vmem(ptr_y+#-1)                   //[0, 1]32x4
           z3z2 = vcombine(vwsum, vwsum)          //[P, 0]
           pre_x = add(pre_x, stride_depth)
       } {
           x0fx0cx0bx08 = memd(ptr_x0+#8)         //[0, 2]
           x07x04x03x00 = memd(ptr_x0++MSTRIDE)   //[0, 2]
           sum1_sum0 = combine(#0, #0)            //[P, 0]
           sum3_sum2 = combine(#0, #0)            //[P, 0]
       } {
           sum5_sum4 = combine(#0, #0)            //[P, 0]
           sum7_sum6 = combine(#0, #0)            //[P, 0]
           x1fx1cx1bx18 = memd(ptr_x0+#8)         //[0, 3]
           x17x14x13x10 = memd(ptr_x0++MSTRIDE)   //[0, 3]
       } 
        .balign 32
.L_filt_height:
       {  
           filt_width = memw(sp+#21<<2)           //extract filt_width*depth /16 - 1
           sum1_sum0 += vraddub(x0fx0cx0bx08, x07x04x03x00) //[0, 4]
           dcfetch(pre_x)
           fetch_count = add(fetch_count, #1)
       } {
           loop0(.L_filt_width, filt_width)       //[P, 0]ki is k1/16 - 1
           z0.uw += vrmpy(y0.ub, x03x00.ub)       //[0, 4]
           pre_x = add(pre_x, stride_depth)
           p3 = cmp.eq(fetch_count, #2)           //
       } {
           sum3_sum2 += vraddub(x1fx1cx1bx18, x17x14x13x10) //[0, 5]
           z1.uw += vrmpy(y0.ub, x13x10.ub)       //[0, 5]
           y2 = vmem(ptr_y++#2)                   //[0, 5]32x4
           if(p3) fetch_count = #0
       } {
           z0.uw += vrmpy(y1.ub, x07x04.ub)       //[0, 6]
           z1.uw += vrmpy(y1.ub, x17x14.ub)       //[0, 6]
           y3 = vmem(ptr_y+#-1)                   //[0, 6]32x4
           if(p3) pre_x = add(pre_x, stride3_1)
       } {
           z0.uw += vrmpy(y2.ub, x0bx08.ub)       //[0, 7]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)       //[0, 7]
           x2fx2cx2bx28 = memd(ptr_x0+#8)         //[0, 7]
           x27x24x23x20 = memd(ptr_x0++MSTRIDE)   //[0, 7]
       } {
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)       //[0, 8]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)       //[0, 8]
           x3fx3cx3bx38 = memd(ptr_x0+#8)         //[0, 8]
           x37x34x33x30 = memd(ptr_x0++M4STRIDE_1)//[0, 8]
       } {
           sum5_sum4 += vraddub(x2fx2cx2bx28, x27x24x23x20) //[0, 9]
           z2.uw += vrmpy(y0.ub, x23x20.ub)       //[0, 9]
       } 
       .balign 32
.L_filt_width:
       {
           sum7_sum6 += vraddub(x3fx3cx3bx38, x37x34x33x30) //[0,10]
           z3.uw += vrmpy(y0.ub, x33x30.ub)       //[0,10]
           y0 = vmem(ptr_y++#2)                   //[1, 0]32x4
           dcfetch(pre_x)
       } {
           z2.uw += vrmpy(y1.ub, x27x24.ub)       //[0,11]
           z3.uw += vrmpy(y1.ub, x37x34.ub)       //[0,11]
           y1 = vmem(ptr_y+#-1)                   //[1, 1]32x4
           pre_x = add(pre_x, stride_depth)
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)       //[0,12]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)       //[0,12]
           x0fx0cx0bx08 = memd(ptr_x0+#8)         //[1, 2]
           x07x04x03x00 = memd(ptr_x0++MSTRIDE)   //[1, 2]
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)       //[0,13]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)       //[0,13]
           x1fx1cx1bx18 = memd(ptr_x0+#8)         //[1, 3]
           x17x14x13x10 = memd(ptr_x0++MSTRIDE)   //[1, 3]
       } {
           sum1_sum0 += vraddub(x0fx0cx0bx08, x07x04x03x00) //[1, 4]
           z0.uw += vrmpy(y0.ub, x03x00.ub)       //[1, 4]
           dcfetch(pre_x)
           pre_x = add(pre_x, stride_depth)
       } {
           sum3_sum2 += vraddub(x1fx1cx1bx18, x17x14x13x10) //[1, 5]
           z1.uw += vrmpy(y0.ub, x13x10.ub)       //[1, 5]
           y2 = vmem(ptr_y++#2)                   //[1, 5]32x4
           fetch_count = add(fetch_count, #1)
       } {
           z0.uw += vrmpy(y1.ub, x07x04.ub)       //[1, 6]
           z1.uw += vrmpy(y1.ub, x17x14.ub)       //[1, 6]
           y3 = vmem(ptr_y+#-1)                   //[1, 6]32x4
           p3 = cmp.eq(fetch_count #2)
       } {
           z0.uw += vrmpy(y2.ub, x0bx08.ub)       //[1, 7]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)       //[1, 7]
           x2fx2cx2bx28 = memd(ptr_x0+#8)         //[1, 7]
           x27x24x23x20 = memd(ptr_x0++MSTRIDE)   //[1, 7]
       } {
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)       //[1, 8]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)       //[1, 8]
           x3fx3cx3bx38 = memd(ptr_x0+#8)         //[1, 8]
           x37x34x33x30 = memd(ptr_x0++M4STRIDE_1)//[1, 8]
       } {
           sum5_sum4 += vraddub(x2fx2cx2bx28, x27x24x23x20) //[1, 9]
           z2.uw += vrmpy(y0.ub, x23x20.ub)       //[1, 9]
           if(p3) fetch_count = #0                //[1, 9]
           if(p3) pre_x = add(pre_x, stride3_1)   //[1, 9]
       }:endloop0 
       {
           sum7_sum6 += vraddub(x3fx3cx3bx38, x37x34x33x30) //[1,10]
           z3.uw += vrmpy(y0.ub, x33x30.ub)       //[1,10]
           y0 = vmem(ptr_y++#2)                   //[0, 0]32x4
           ptr_x0 = sub(ptr_x0, filt_skip)        //[E, 0]move to next line ptr_y keeps going
       } {
           z2.uw += vrmpy(y1.ub, x27x24.ub)       //[1,11]
           z3.uw += vrmpy(y1.ub, x37x34.ub)       //[1,11]
           y1 = vmem(ptr_y+#-1)                   //[0, 1]32x4
           dcfetch(ptr_x0+#PREFETCH)
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)       //[1,12]
           pre_x = add(ptr_x0, #PREFETCH)
           fetch_count = #0
       } {
           z3.uw += vrmpy(y2.ub, x3bx38.ub)       //[1,12]
           x0fx0cx0bx08 = memd(ptr_x0+#8)         //[0, 2]
           x07x04x03x00 = memd(ptr_x0++MSTRIDE)   //[0, 2]
           pre_x = add(pre_x, stride_depth)
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)       //[1,13]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)       //[1,13]
           x1fx1cx1bx18 = memd(ptr_x0+#8)         //[0, 3]
           x17x14x13x10 = memd(ptr_x0++MSTRIDE)   //[0, 3]
       }:endloop1
       {
           ptr_x0 = sub(ptr_x0, next_outputs)     //
           in_offset = memw(sp+#27<<2)            //+18+7
           zsum = memw(sp+#28<<2)                 //+18+8res as zsum
       } {
           sum0 = zsum                           //
           sum1 = add(sum0, sum1)                //
           ptr_datasum = memw(sp+#24<<2)         //data sum ptr
       } {
           pre_x = add(ptr_x0, #PREFETCH)        //pre_x, next_outputs)      // 
           sum0 += mpyi(in_offset, sum1)         //
           mstride = memw(sp+#60)                // result matrix stride copied into separete reg
       } {
           memw(ptr_datasum++#1<<2) = sum0       //
           x0 = vsplat(sum0)                     //
       } {
           z0.w = vadd(z0.w, x0.w)               //
           vmem(ptr_z+#0):nt = z0.new            //[E,  ]
           ptr_z = add(ptr_z, mstride)                  //
           p0 = cmp.gt(col_count, #1)             //
       } {
           maxe.w = vmax(maxe.w, z0.w)            //
           sum2 = zsum                           //
           sum3 = add(sum2, sum3)                //
           dcfetch(ptr_x0)
       } {
           sum2 += mpyi(in_offset, sum3)         //
           dcfetch(ptr_x0+#32)
       } {
           if(p0)memw(ptr_datasum++#1<<2) = sum2 //
           x1 = vsplat(sum2)                     //
       } {
           z1.w = vadd(z1.w, x1.w)               //
           if(p0)vmem(ptr_z+#0):nt = z1.new      //[E,  ]
           if(p0)ptr_z = add(ptr_z, mstride)            //
           p1 = cmp.gt(col_count, #2)             //
       } {
           if(!p0) z1 = vc8000                    //
           sum4 = zsum                           //
           sum5 = add(sum4, sum5)                //
       } {
           sum4 += mpyi(in_offset, sum5)         //
       } {
           maxe.w = vmax(maxe.w, z1.w)            //
           if(p1)memw(ptr_datasum++#1<<2) = sum4 //
           x2 = vsplat(sum4)                     //
       } {
           z2.w = vadd(z2.w, x2.w)               //
           if(p1)vmem(ptr_z+#0):nt = z2.new      //[E,  ]
           if(p1)ptr_z = add(ptr_z, mstride)            //
           p0 = cmp.gt(col_count, #3)             //
       } {
           if(!p1) z2 = vc8000                    //
           sum6 = zsum                           //
           sum7 = add(sum6, sum7)                //
       } {
           sum6 += mpyi(in_offset, sum7)         //
       } {
           maxe.w = vmax(maxe.w, z2.w)            //
           if(p0)memw(ptr_datasum++#1<<2) = sum6 //
           x3 = vsplat(sum6)                     //
       } {
           z3.w = vadd(z3.w, x3.w)               //
           if(p0)vmem(ptr_z+#0):nt = z3.new      //[E,  ]
           if(p0)ptr_z = add(ptr_z, mstride)            //
       } {
           if(!p0) z3 = vc8000                    //
           col_count = add(col_count, #-4)        //
           memw(sp+#24<<2) = ptr_datasum          //data sum ptr
       } {
           maxe.w = vmax(maxe.w, z3.w)            //
           p2 = cmp.gt(col_count, #0)             //
           if(p2.new) jump:t .L_width             //
       }//end cols per line
       {
           out_height = memw(sp+#23<<2)           //
       } {
           p1 = cmp.eq(out_height, #0)            //
           if(!p1.new) jump:t .L_height           //
       }//end lines per block
       {
           loop0(.L_peak, #5)                     //[P, 0]
           c4 = #4                                //
           ptr_max = memw(sp+#26<<2)              //ptr pre computed max value in output
       }
.L_peak:
       {
           maxomaxe=vshuff(maxe,maxe,c4)          //[0, 0]
       } {
           maxe.w = vmax(maxo.w, maxe.w)          //[0, 1]
           c4 = add(c4, c4)                       //[0, 1]
       }:endloop0
       {   vmem(ptr_max+#0) = maxe                //[E, 0]
       }
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)                   //restore stack
           r19:18 = memd(sp+#8)                   //Q
       } {
           r21:20 = memd(sp+#16)                  //Q
           r23:22 = memd(sp+#24)                  //Q
       } {    
           r25:24 = memd(sp+#32)                  //Q
           r27:26 = memd(sp+#40)                  //Q
       } {
           r28 = memw(sp+#64)
           dealloc_return                         //Q
       }
.L_end:
/*=============================================================================*/
      .size gvconvsum2dbbw_asm, .L_end-gvconvsum2dbbw_asm
