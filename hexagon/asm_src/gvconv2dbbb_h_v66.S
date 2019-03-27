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

/*======================================================================*/
/*  FUNCTIONS      : gvmmpybbw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform gvm vector matrix multiply, result is        */
/*                 saturated to 8bits                                   */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V66  + HVX                                    */
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
/*  DJH                 01/30/16       installed splatter Z buffer a    */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->  K*N/512+17*N/4+36                                            */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 976 bytes                                             */
/*     STACK    = 64 bytes                                              */
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
        .file "gvconv2dbbb_h.S"
        .global gvconv2dbbb_v66_asm
        .balign 32
        .type  gvconv2dbbb_v66_asm, @function
gvconv2dbbb_v66_asm:
/*=============================================================================*/
#define ptr_x         r0    //data
#define ptr_yi        r1    //weights
#define ptr_z         r2    //results
#define in_width      r3    //(pad_x+in_width) * depth
#define out_width     r4    //out_width 
#define m             r5    //is stride of the output matrix always mult of 32
#define stride_depth  r6    //0 stride|depth  between computations
#define filt_width    r7    //1 depth*filt_width  
#define filt_height   r8    //2 filt_hieght lines per filter
#define out_height    r9    //3 number of vertical lines to perform
#define ptr_datasum   r10   //4 
#define ptr_weightsum r11   //5 
#define ptr_max       r12   //6 
#define ptr_biasbuf   r14   //7  sat8 ((0x8000 + (x + biass)*recip_level)>>16)
#define recip_level   r15   //8 
/*=============================================================================*/
#define PV32(VSRC) .word (0x1DFFE020 + VSRC)
#define PZ(ZSRC)   .word (0x1DFFE1E0 + ZSRC)
#define PS(SSRC)   .word (0x1DFFE100 + SSRC)
#define PD(SSRC)   .word (0x1DFFE120 + SSRC)
#define sel           r8
#define len           r9
#define in_width_stride r13 //in_width * stride for next output
#define ptr_x0        r11
#define stride4       r13   //
#define stride        r25
#define next_outputs  r23   //jump to input ptr for next set of outputs
#define ptr_y         r24   //
#define col_count     r22
#define xsuma         r0     //kernel sum * filt_offset computed externally
#define xsumb         r21    //kernel sum * filt_offset computed externally
#define round_amt     r6     //amount to add to bias buf odffset computation
#define one           r17
#define  c4           r26
#define row0          r14
#define row1          r15
#define row2          r16
#define row3          r17
#define mpy_cntrl     r18
#define mpy_cntrl2    r27
#define mpy_cntrl3    r20
/*=============================================================================*/
#define s0            v0     //
#define s1            v1     //
#define s1s0          v1:0   //
#define s2            v2     //
#define s3            v3     //
#define s3s2          v3:2   //
#define s3s2s1s0      v1:0   //
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
#define biasvec       v18    //
#define recipvec      v16    //
#define rndvec        v17    //
#define vpreds        v19    //
/*=============================================================================*/
       {
           sel = ##0x01010101                     // entry 0
           len = #32                              //
       } {
           q0 = vsetq(len);                       // 1000
           len = #64                              //
           round_amt = ##0x00008000               //
       } {
           vpreds = vand(q0, sel)                 //
           q2 = vsetq(len);                       // 1100
           len = #96                              //
           rndvec = vsplat(round_amt)             //
       } {
           q1 = and(q2, !q0)                      // 0100
           q3 = vsetq(len)                        // 1110
           sel = add(sel, sel)                    //02020202
       } {
           vpreds|= vand(q1, sel)                 //
           q2 = and(q3, !q2)                      // 0010
           q3 = not(q3)                           // 0001
           sel = add(sel, sel)                    //04040404
       } {
           vpreds|= vand(q2, sel)                 //
           sel = add(sel, sel)                    //08080808
       } {
           vpreds|= vand(q3, sel)                 // entry 3 10101010 selects all zero
           stride_depth = memw(sp+#0<<2)          //extract stride*depth 
           filt_width = memw(sp+#1<<2)            //extract filt_width*depth 
       } {   
           filt_height = memw(sp+#2<<2)           //extract filt_height 
           out_height = memw(sp+#3<<2)            //number of output lines
           p0 = cmp.eq(filt_width, #1)
       } {
           ptr_datasum = memw(sp+#4<<2)           //data sum ptr
           ptr_weightsum = memw(sp+#5<<2)         //ptr pre computed weight sum
           filt_width = mpy(filt_width.L, stride_depth.L)
       } {
           ptr_max = memw(sp+#6<<2)               //ptr pre computed max value in output
           ptr_biasbuf = memw(sp+#7<<2)           //read in the ptr to the bias buffer value
       } {
           biasvec = vmem(ptr_biasbuf+#0)         //
           recip_level = memw(sp+#8<<2)           // 
           p3 = cmp.gt(filt_width, #192)
       } {
           recipvec = vsplat(recip_level)         //
           allocframe(#72)                        //
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
           memw(sp+#48) = ptr_x                   //
           memw(sp+#52) = ptr_yi                  //
       } {
           vwsum = vmem(ptr_weightsum+#0)         //
           r16 = ##0x80000001                     //max negative
           c4 = #0x4
       } {
           next_outputs = mpyi(filt_height, in_width) 
           vc8000 = vsplat(r16)                   //
           memw(sp+#56) = out_width               //
       } {
           stride4= asl(stride_depth, #2)         //4-2*stride to corrct for outper pipeline
           M0 = stride_depth
       } {
           M1 = m                                 //outdepth
           next_outputs = sub(next_outputs, stride4)
           filt_width = lsr(filt_width, #5)       //filt_width / 32
       } {
           maxe= vmem(ptr_max+#0)
           in_width_stride = mpyi(in_width, stride)      //
           filt_width = add(filt_width, #-1)
       }
/*===========o=================================================================*/
        .balign 32
.L_height:
       { 
           ptr_x0 = memw(sp+#48)                  //ptr_x 
           out_height = add(out_height, #-1)      //
       } {
           col_count = memw(sp+#56)               //out_width 
           memw(sp+#48) += in_width_stride     //ptr_x=add(ptr_x,in_width) //ptr_x+=in_width
       }
        .balign 32
.L_width:
       {
           ptr_y = memw(sp+#52)                   //ptr_y = ptr_yi initialize filter pointer
           loop1(.L_filt_height, filt_height)     //[P, 0]for(filt_y=0; filt_y < n; filt_y+=1){
           ptr_x = ptr_x0
       } {
           mpy_cntrl = combine(stride_depth.L, ptr_x.L)
           row0 = add(ptr_x, #32)
           z1:0 = vmem(ptr_x++M0)
       } {
           row1 = add(ptr_x, #32)
           z3:2 = vmem(ptr_x++M0)
           s1s0 = vcombine(vwsum, vwsum)          //[P, 0]
           p0 = cmp.eq(filt_width, #0)
       } {
           row2 = add(ptr_x, #32)
           z5:4 = vmem(ptr_x++M0)
           s3s2 = vcombine(vwsum, vwsum)          //[P, 0]
           mpy_cntrl = vsubh(mpy_cntrl, c4)
       } 
       .balign 32
.L_filt_height:
       {
           loop0(.L_filt_width, filt_width)       //[P, 0]ki is k1/16 - 1
           row3 = add(ptr_x, #32)
           z7:6 = vmem(ptr_x+#0)
           if(p0) jump .L_s32
       }
       .balign 32
.L_filt_width:
       {   y0.tmp = vmem(ptr_y++#2)                     //[1, 0]32x4
           s3s2s1s0.uw += vrmpy_tmp(y0.ub, ++mpy_cntrl.ub)
       } {
           y1.tmp = vmem(ptr_y+#-1)                     //[1, 1]32x4
           s3s2s1s0.uw += vrmpy_tmp(y1.ub, ++mpy_cntrl.ub)
       } {
           y2.tmp = vmem(ptr_y++#2)                   //[1, 4]32x4
           s3s2s1s0.uw += vrmpy_tmp(y2.ub, ++mpy_cntrl.ub)
       } {
           y3.tmp = vmem(ptr_y+#-1)                   //[1, 4]32x4
           s3s2s1s0.uw += vrmpy_tmp(y3.ub, ++mpy_cntrl.ub)
       } {
           y0.tmp = vmem(ptr_y++#2)                   //[1, 0]32x4
           s3s2s1s0.uw += vrmpy_tmp(y0.ub, ++mpy_cntrl.ub)
           z1:0 = vmem_fifo(row0++#32) 
       } {
           y1.tmp = vmem(ptr_y+#-1)                   //[1, 1]32x4
           s3s2s1s0.uw += vrmpy_tmp(y1.ub, ++mpy_cntrl.ub)
           z3:2 = vmem_fifo(row1++#32) 
       } {
           y2.tmp = vmem(ptr_y++#2)                   //[1, 4]32x4
           s3s2s1s0.uw += vrmpy_tmp(y2.ub, ++mpy_cntrl.ub)
           z5:4 = vmem_fifo(row2++#32) 
       } {
           y3.tmp = vmem(ptr_y+#-1)                   //[1, 5]32x4
           s3s2s1s0.uw += vrmpy_tmp(y3.ub, ++mpy_cntrl.ub)
           z7:6 = vmem_fifo(row3++#32) 
       }:endloop0 
       .balign 32
.L_s32:
       {   y0.tmp = vmem(ptr_y++#2)                     //[1, 0]32x4
           s3s2s1s0.uw += vrmpy_tmp(y0.ub, ++mpy_cntrl.ub)
           ptr_x0 = add(ptr_x0, in_width)        //[E, 0]move to next line ptr_y keeps going
       } {
           y1.tmp = vmem(ptr_y+#-1)                     //[1, 1]32x4
           s3s2s1s0.uw += vrmpy_tmp(y1.ub, ++mpy_cntrl.ub)
           ptr_x = ptr_x0
       } {
           y2.tmp = vmem(ptr_y++#2)                   //[1, 4]32x4
           s3s2s1s0.uw += vrmpy_tmp(y2.ub, ++mpy_cntrl.ub)
       } {
           y3.tmp = vmem(ptr_y+#-1)                   //[1, 4]32x4
           s3s2s1s0.uw += vrmpy_tmp(y3.ub, ++mpy_cntrl.ub)
           mpy_cntrl2 = combine(stride_depth.L, ptr_x0.L)
       } {
           y0.tmp = vmem(ptr_y++#2)                   //[1, 0]32x4
           s3s2s1s0.uw += vrmpy_tmp(y0.ub, ++mpy_cntrl.ub)
       } {
           y1.tmp = vmem(ptr_y+#-1)                   //[1, 1]32x4
           s3s2s1s0.uw += vrmpy_tmp(y1.ub, ++mpy_cntrl.ub)
           mpy_cntrl3 = add(mpy_cntrl, #12)
           row0 = add(ptr_x0, #32)
       } {
           y2.tmp = vmem(ptr_y++#2)                   //[1, 4]32x4
           s3s2s1s0.uw += vrmpy_tmp(y2.ub, ++mpy_cntrl.ub)
       } {
           y3.tmp = vmem(ptr_y+#-1)                   //[1, 5]32x4
           s3s2s1s0.uw += vrmpy_tmp(y3.ub, mpy_cntrl3.ub)
           mpy_cntrl = vsubh(mpy_cntrl2, c4)
           z1:0 = vmem(ptr_x++M0)
       } {
           z3:2 = vmem(ptr_x++M0)
           row1 = add(ptr_x, #32)
       } {
           row2 = add(ptr_x, #32)
           z5:4 = vmem(ptr_x++M0)
       }:endloop1
       {
           ptr_x0 = sub(ptr_x0, next_outputs)     //reset data ptr to next set of 4
           xsuma = memw(ptr_datasum++#1<<2)       //#0 
       } {
           x0 = vsplat(xsuma)                     //#0 
           p0 = cmp.gt(col_count, #1)             //#1  are there at least 2 levt?
           if(p0.new) xsumb = memw(ptr_datasum++#1<<2) //#1 
       } {
           s0.w = vadd(s0.w, x0.w)                //#0 add data sum
           x1 = vsplat(xsumb)                     //#1 
           y1 = rndvec                            //#1 out1 = 0x8000
       } {
           maxe.w = vmax(maxe.w, s0.w)            //#0 see if z0 is max
           s0.w = vadd(s0.w, biasvec.w)           //#0 add data sum
           p1 = cmp.gt(col_count, #2)             //#2 
           if(p1.new) xsuma = memw(ptr_datasum++#1<<2) //#2 
       } {
           s1.w = vadd(s1.w, x1.w)                //#1 
           x2 = vsplat(xsuma)                     //#2 
           p2 = cmp.gt(col_count, #3)             //#3 
           if(p2.new) xsumb = memw(ptr_datasum++#1<<2)//#3 
       } {
           if(!p0) s1 = vc8000                    //#1 
           x1.w = vadd(s1.w, biasvec.w)           //#1 add data sum
           x3 = vsplat(xsumb)                     //#3 
           y0 = rndvec                            //#0 out0 = 0x8000
       } {
           y0.w += vmpyie(s0.w, recipvec.uh)      //#0 
           s2.w = vadd(s2.w, x2.w)                //#2 
       } {
           y1.w += vmpyie(x1.w, recipvec.uh)      //#1 
           x2.w = vadd(s2.w, biasvec.w)           //#2 add data sum
       } {
           y0.h = vpacko(y0.w, y0.w)              //#0 >>16
           if(!p1) s2 = vc8000                    //#2 
           s3.w = vadd(s3.w, x3.w)                //#3 
       } {
           y1.h = vpacko(y1.w, y1.w)              //#1 >>16
           if(!p2) s3 = vc8000                    //#3 
           x3.w = vadd(s3.w, biasvec.w)           //#3 add data sum
           y2 = rndvec                            //#2 out2 = 0x8000
       } {
           y0.ub = vpack(y0.h, y0.h):sat          //#0 sat8  <0, >255
           y2.w += vmpyie(x2.w, recipvec.uh)      //#2 
           y3 = rndvec                            //#3 out3 = 0x8000
       } {
           y1.ub = vpack(y1.h, y1.h):sat          //#1 sat8  <0, >255
           y3.w += vmpyie(x3.w, recipvec.uh)      //#3 
       } {
           maxe.w = vmax(maxe.w, s1.w)            //#1 
           y2.h = vpacko(y2.w, y2.w)              //#2 >>16
       } {
           vmem32(ptr_z+#0)= y0                   //#0 [E,  ]store first 32bytes
           ptr_z = add(ptr_z, m)                  //#0 
           maxe.w = vmax(maxe.w, s2.w)            //#2 
           y3.h = vpacko(y3.w, y3.w)              //#3 >>16
       } {
           if(p0) vmem32(ptr_z+#0) = y1           //#1 [E,  ]store 2nd 32bytes
           if(p0) ptr_z = add(ptr_z, m)           //#1 
           y2.ub = vpack(y2.h, y2.h):sat          //#2 sat8  <0, >255
           maxe.w = vmax(maxe.w, s3.w)            //#3 
       } {
           if(p1) vmem32(ptr_z+#0) = y2           //#2 [E,  ]store 2nd 32bytes
           if(p1) ptr_z = add(ptr_z, m)           //#2 
           y3.ub = vpack(y3.h, y3.h):sat          //#3 sat8  <0, >255
           col_count = add(col_count, #-4)        //
       } {
           if(p2) vmem32(ptr_z+#0) = y3           //#3 [E,  ]store 2nd 32bytes
           if(p2) ptr_z = add(ptr_z, m)           //#3 
           p3 = cmp.gt(col_count, #0)             //
           if(p3.new) jump:t .L_width             //
       }//end cols per line
       {
           p1 = cmp.eq(out_height, #0)            //
           if(!p1.new) jump:t .L_height           //
       }//end lines per block
       {
           loop0(.L_peak, #5)                     //[P, 0]
           r6 = #4                                //
       }
.L_peak:
       {
           maxomaxe=vshuff(maxe,maxe,r6)          //[0, 0]
       } {
           maxe.w = vmax(maxo.w, maxe.w)          //[0, 1]
           r6 = add(r6, r6)                       //[0, 1]
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
           dealloc_return                         //Q
       }
.L_end:
/*=============================================================================*/
      .size gvconv2dbbb_v66_asm, .L_end-gvconv2dbbb_v66_asm
/*=============================================================================*/
