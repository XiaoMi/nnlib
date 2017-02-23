
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
#define ptrIn     r0
#define ptrOut    r1
#define xoffset   r2
#define ptrDelta  r3
#define startorow r4
#define numorows  r5
#define endrows   r23

#define selp      r6   //used to select the predicates
#define sel0      r7
#define sel1      r8
#define sel2      r9
#define sel3      r10
#define sel4      r11
#define sel5      r12
#define sel6      r13
#define sel7      r14
#define ptrRow0   r15
#define ptrRow1   r16
#define rot0      r17
#define optr      r18
#define cm21      r19
#define cm0       r20
#define row_cnt   r21
#define line_cnt  r22
#define offset    r15
#define iwidth    r16

#define vpred0    v0
#define in0       v1
#define in1       v2
#define min0      v3
#define min1      v4
#define out01     v5
#define out23     v8
#define out45     v11
#define out67     v14
#define vxoffset  v15
#define vpred1    v16
#define vpred2    v17
#define vpred3    v18
#define vpred4    v19
#define vpred5    v20
#define vpred6    v21
#define vpred7    v22

#define vqp       q1
#define vq0       q0
#define vq1       q3
#define vq2       q3
#define vq3       q3
#define vq4       q3
#define vq5       q0
#define vq6       q2
#define vq7       q1
#define vq8       q2
/* ========================================================================= */
          .global im2col7732_asm
          .type   im2col7732_asm, @function
          .balign 32
im2col7732_asm:
/* ============================================================================ */
       {
         allocframe(#64)
         xoffset = vsplatb(xoffset)
       } {
         memd(sp+#0) = r17:16
         memd(sp+#8) = r19:18
         sel0 = #160
       } {
         memd(sp+#16)= r21:20
         memd(sp+#24)= r23:22
       } {
         memd(sp+#32)= r25:24
         memd(sp+#40)= r27:26
         M1 = sel0
         sel0 = #32
       } {
         vpred0 = vmem(ptrDelta+#0)
         M0 = sel0
         offset = #-2
       } {
         vxoffset = vsplat(xoffset)
         vpred1 = vmem(ptrDelta+#1)
         offset += add(startorow, startorow)
       } {
         sel0 = ##0x01010101
         vpred2 = vmem(ptrDelta+#2)
         iwidth = #672
       } {
         sel1 = add(sel0, sel0)
         sel2 = asl(sel0, #2)
         sel3 = asl(sel0, #3)
         vpred3 = vmem(ptrDelta+#3)
       } {
         sel4 = add(sel3, sel3)
         sel5 = asl(sel3, #2)
         sel6 = asl(sel3, #3)
         vpred4 = vmem(ptrDelta+#4)
       } {
         vpred5 = vmem(ptrDelta+#5)
         sel7 = asl(sel4, #3)
         p0 = and(p0, !p0)               //epilog
         endrows = add(startorow, numorows)
       } {
         vpred6 = vmem(ptrDelta+#6)
         vqp = vand(vpred1, sel1)        //extract the pre mask
         p2 = cmp.eq(startorow, #0)      //on prolog
         offset = mpyi(offset, iwidth)
       } {
         p1 = and(p1, !p1)               //off epilog
         vpred7 = vmem(ptrDelta+#7)
         line_cnt = startorow
         ptrIn = add(ptrIn, offset)
       }
         
/* ============================================================================ */
.L_line_loop:
/* ============================================================================ *
       0td and 1st row of filter
 * ============================================================================ */
       {
         cm21 = #-21
         ptrRow0 = ptrIn
         optr = add(ptrOut,#0)
         selp = mux(p2,sel7,sel1)      //select the prolog padding
       } {
         row_cnt = #0;
         ptrRow0 = add(ptrRow0, #-6)   //retard by 6
         ptrRow1 = add(ptrRow0, #666)  //3*224-6
         loop1(.L_outer0, #7)
       }
         .balign 32
.L_outer0:
       {
         in0 = vmemu(ptrRow0)
         ptrRow0 = add(ptrRow0, #96)
         vqp = vand(vpred1, selp)
       } {
         in0 = vmux(vqp, vxoffset, in0)
         in1 = vmemu(ptrRow1)
         p3 = cmp.eq(row_cnt, #5)       //ultimate stage
       } {
         ptrRow1 = add(ptrRow1, #96)
         in1 = vmux(vqp, vxoffset, in1)
         rot0 = #0
         selp = mux(p3,sel2, sel3)       //from there always choose the data
       } {
         selp = mux(p2, sel7, selp)      //either ficre prolog = offset or normal
         row_cnt = add(row_cnt, #1)
         in1 = vror(in1, cm21)
       } {
         vq0 = vand(vpred0, sel0)        //[0, 0]
         in1 = vror(in1, rot0)           //[0, 1]
         rot0 = #-26                     //[0, 1]
       } {
         loop0(.L_loop00, #4)            //create 16 x 42 values
         out01 = vmux(vq0, in0, in1)     //[0, 2]
         vq1 = vand(vpred0, sel1)        //[0, 2]
         in0 = vror(in0, rot0)           //[0, 2]
       } 
       .balign 32
.L_loop00:
       {
         if(vq1) vmem(optr++M1) = out01  //advance 0
         in1 = vror(in1, rot0)           //[0, 3]
         vq2 = vand(vpred0, sel2)        //[0, 3]
       } {
         out23 = vmux(vq2, in0, in1)     //[0, 4]
         vq3 = vand(vpred0, sel3)        //[0, 4]
         in0 = vror(in0, rot0)           //[0, 4]
       } {
         if(vq3) vmem(optr++M1) = out23  //advance 160
         in1 = vror(in1, rot0)           //[0, 5]
         vq4 = vand(vpred0, sel4)        //[0, 5]
       } {
         out45 = vmux(vq4, in0, in1)     //[0, 6]
         vq5 = vand(vpred0, sel5)        //[0, 6]
         in0 = vror(in0, rot0)           //[0, 6]
       } {
         if(vq5) vmem(optr++M1) = out45  //advance 320
         in1 = vror(in1, rot0)           //[0, 7]
         vq6 = vand(vpred0, sel6)        //[0, 7]
       } {
         out67 = vmux(vq6, in0, in1)     //[0, 8]
         vq8 = vand(vpred1, sel0)        //[0, 8]
         in0 = vror(in0, rot0)           //[1, 0]
         vq0 = vand(vpred0, sel0)        //[1, 0]
       } {
         if(vq8) vmem(optr+#1) = out67   //[0, 9]
         vq7 = vand(vpred0, sel7)        //[1, 9]
         in1 = vror(in1, rot0)           //[1, 1]
         rot0 = #-26                     //[1, 1]
       } {
         if(vq7) vmem(optr++M1) = out67  //advance 480
         out01 = vmux(vq0, in0, in1)     //[1, 2]
         vq1 = vand(vpred0, sel1)        //[1, 2]
         in0 = vror(in0, rot0)           //[1, 2]
       }:endloop0:endloop1
/* ============================================================================ *
       2nd and 3rd row of filter
 * ============================================================================ */
       {
         ptrRow0 = add(ptrIn, #1344)   //+0
         optr = add(ptrOut,#0)
         selp = sel1
         row_cnt = #0;
       } {
         cm21 = #-21 //-63                   //3*21
         ptrRow0 = add(ptrRow0, #-6)  //-6
         ptrRow1 = add(ptrRow0, #666)  //3*224-6
         loop1(.L_outer1, #7)
       }
         .balign 32
.L_outer1:
       {
         in0 = vmemu(ptrRow0)
         ptrRow0 = add(ptrRow0, #96)
         vqp = vand(vpred1, selp)
       } {
         in0 = vmux(vqp, vxoffset, in0)
         in1 = vmemu(ptrRow1)
         ptrRow1 = add(ptrRow1, #96)
       } {
         p3 = cmp.eq(row_cnt, #5) 
         in1 = vmux(vqp, vxoffset, in1)
       } {
         selp = mux(p3,sel2,sel3)        //from there always choose the data
         in1 = vror(in1, cm21)
         rot0 = #-42
       } {
         row_cnt = add(row_cnt, #1);
         in0 = vror(in0, rot0)
         vq0 = vand(vpred2, sel0)
         loop0(.L_loop10, #4)            //create 16 x 42 values
       } 
       .balign 32
.L_loop10:
       {
         in1 = vror(in1, rot0)
         rot0 = #-26
       } {
         out01 = vmux(vq0, in0, in1)
         vq1 = vand(vpred2, sel1)
         in0 = vror(in0, rot0)
       } {
         if(vq1) vmem(optr++M1) = out01   //advance 0
         in1 = vror(in1, rot0)
         vq2 = vand(vpred2, sel2)
       } {
         out23 = vmux(vq2, in0, in1)
         vq3 = vand(vpred2, sel3)
         in0= vror(in0, rot0)
       } {
         if(vq3) vmem(optr++M1) = out23   //advance 160
         in1 = vror(in1, rot0)
         vq4 = vand(vpred2, sel4)
       } {
         out45 = vmux(vq4, in0, in1)
         in0= vror(in0, rot0)
         vq6 = vand(vpred2, sel6)
       } {
         if(vq6) vmem(optr+#1) = out45   //
         vq5 = vand(vpred2, sel5)
         in1 = vror(in1, rot0)
         vq7 = vand(vpred2, sel7)
       } {
         if(vq5) vmem(optr) = out45       //advance 320
         out67 = vmux(vq7, in0, in1)
         vq8 = vand(vpred3, sel0)
         optr = add(optr, #288)
       } {
         if(vq8) vmem(optr++M0) = out67   //advance 480
         in0 = vror(in0, rot0)            //[1, 0]
         vq0 = vand(vpred2, sel0)         //[1, 0]
       }:endloop0:endloop1
/* ============================================================================ *
       4th and 5th row of filter
 * ============================================================================ */
       {
         cm21 = #-21 //105                  //5*21
         ptrRow0 = add(ptrIn, #2688)   //3*224 * 4
         optr = add(ptrOut,#0)
         row_cnt = #0
       } {
         selp = mux(p1,sel7,sel1)      //pad the last row 
         ptrRow0 = add(ptrRow0, #-6)   //retard by 6
         ptrRow1 = add(ptrRow0, #666)  //3*224-6
         loop1(.L_outer2, #7)
       }
         .balign 32
.L_outer2:
       {
         in0 = vmemu(ptrRow0)
         ptrRow0 = add(ptrRow0, #96)
         vqp = vand(vpred1, selp)
       } {
         in1 = vmemu(ptrRow1)
         ptrRow1 = add(ptrRow1, #96)
         in0 = vmux(vqp, vxoffset, in0)
       } {
         p3 = cmp.eq(row_cnt, #5)
         in1 = vmux(vqp, vxoffset, in1)
         row_cnt = add(row_cnt, #1)
       } {
         in1 = vror(in1, cm21)
         rot0 = #-84
       } {
         selp = mux(p3,sel2,sel3)        //from there always choose the data
         in0 = vror(in0, rot0)
         vq0 = vand(vpred4, sel0)
       } {
         selp = mux(p1, sel7, selp)
         in1 = vror(in1, rot0)
         rot0 = #-26
         loop0(.L_loop20, #4)            //create 16 x 42 values
       } 
       .balign 32
.L_loop20:
       {
         out01 = vmux(vq0, in0, in1)
         vq1 = vand(vpred4, sel1)
         in0 = vror(in0, rot0)
       } {
         if(vq1) vmem(optr++M1) = out01   //advance 0
         in1 = vror(in1, rot0)
         vq2 = vand(vpred4, sel2)
       } {
         out23 = vmux(vq2, in0, in1)
         vq4 = vand(vpred4, sel4)
         in0= vror(in0, rot0)
         vq6 = vand(vpred4, sel6)
       } {
         if(vq4) vmem(optr+#1) = out23    //advance 160
         vq3 = vand(vpred4, sel3)
         in1 = vror(in1, rot0)
         vq5 = vand(vpred4, sel5)
       } {
         if(vq3) vmem(optr) = out23       //advance 160
         optr = add(optr, #288)
         out45 = vmux(vq5, in0, in1)
         in0= vror(in0, rot0)
       } {
	 in1 = vror(in1, rot0)
         vq7 = vand(vpred4, sel7)
         if(vq6) vmem(optr++M1) = out45   //advance 320
       } {
         out67 = vmux(vq7, in0, in1)
         vq8 = vand(vpred5, sel0)
         in0 = vror(in0, rot0)            //[1, 0]
         vq0 = vand(vpred4, sel0)         //[1, 0]
       } {
         if(vq8) vmem(optr++M0) = out67   //advance 480
         in1 = vror(in1, rot0)            //[1, 1]
       }:endloop0:endloop1
/* ============================================================================ *
       6th row of filter and 13 pixel pad to make 160 pixel block
 * ============================================================================ */
       {
         ptrRow0 = add(ptrIn, #4032)   //6*3*224 
         optr = add(ptrOut,#0) 
         row_cnt = #0
       } {
         selp = mux(p0, sel7, sel1)
         ptrRow0 = add(ptrRow0, #-6)   //element 4
         loop1(.L_outer3, #7)
       }
         .balign 32
.L_outer3:
       {
         in0 = vmemu(ptrRow0)
         ptrRow0 = add(ptrRow0, #96)
         vqp = vand(vpred1, selp)
       } {
         in0 = vmux(vqp, vxoffset, in0)
         in1 = vxoffset
         p3 = cmp.eq(row_cnt, #5)
         rot0 = #2                       //-126 6x21
       } {
         selp = mux(p3, sel2, sel3)
         loop0(.L_loop30, #4)            //create 16 x 42 values
         in0 = vror(in0, rot0)            //[0, 0]
         vq0 = vand(vpred6, sel0)         //[0, 0]
       } {
         rot0 = #-26          //[0, 0]
       } {
         row_cnt = add(row_cnt, #1)
         selp = mux(p0, sel7, selp)
         out01 = vmux(vq0, in0, in1)      //[0, 1]
         vq2 = vand(vpred6, sel2)         //[0, 1]
       } 
       .balign 32
.L_loop30:
       {
         if(vq2) vmem(optr+#1) = out01   //advance 160
         vq1 = vand(vpred6, sel1)
         in0 = vror(in0, rot0)
       } {
         if(vq1) vmem(optr++M1) = out01   //advance 0
         vq3 = vand(vpred6, sel3)
       } {
         optr = add(optr, #128)
         out23 = vmux(vq3, in0, in1)
         in0= vror(in0, rot0)
         vq4 = vand(vpred6, sel4)
       } {
         if(vq4) vmem(optr++M1) = out23   //advance 160
         vq5 = vand(vpred6, sel5)
       } {
         out45 = vmux(vq5, in0, in1)
         in0= vror(in0, rot0)
         vq6 = vand(vpred6, sel6)
         vq7 = vand(vpred6, sel7)
       } {
         if(vq6) vmem(optr++M1) = out45   //advance 320
       } {
         out67 = vmux(vq7, in0, in1)
         vq8 = vand(vpred7, sel0)
         in0 = vror(in0, rot0)            //[1, 0]
         vq0 = vand(vpred6, sel0)         //[1, 0]
       } {
         if(vq8) vmem(optr++M0) = out67   //advance 480 
         out01 = vmux(vq0, in0, in1)      //[1, 1]
         vq2 = vand(vpred6, sel2)         //[1, 1]
       }:endloop0:endloop1
/* ============================================================================ */
       {
         line_cnt = add(line_cnt, #1)
       } {
         p0 = cmp.eq(line_cnt, endrows)
         ptrOut = add(ptrOut,#17920)     //add 112*160 
         p2 = and(p2, !p2)
       } {
         p1 = cmp.gt(line_cnt, #110)      //is this the last lines
         ptrIn  = add(ptrIn,  #1344)     //add 224*3 
         if(!p0) jump:t .L_line_loop
         p0 = cmp.gt(line_cnt, #109)      //is this the last 2 line
       }
/* ============================================================================ */
.L_quit:
       {
         r17:16 = memd(sp+#0)
         r19:18 = memd(sp+#8)
       } {
         r21:20 = memd(sp+#16)
         r23:22 = memd(sp+#24)
       } {
         r25:24 = memd(sp+#32)
         r27:26 = memd(sp+#40)
       }
         dealloc_return
/* ============================================================================ */
.L_end:
     .size im2col3332_asm, .L_end-im2col3332_asm
