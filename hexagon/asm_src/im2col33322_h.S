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
/*
      Function
      --------
      im2col for filter 3x3 depth 3 pading = 2 stride = 2 non same no padding
      pack rows 0,1,2 into sequence, advance by 1 element repack

      void im2col33322_hvx(uchar  * in, uchar * im2col_data, 
                          int x_offset, uchar * delta_tables,
                          startoutrow, numoutrows); 

      each output row contains 149 packed elements, 32bytes each
 ===============================================================================*/
#define ptrIn     r0
#define ptrOut    r1
#define xoffset   r2
#define ptrDelta  r3
#define startorow r4
#define numorows  r5
//scaler regs
#define sel0      r7
#define sel1      r8
#define sel2      r9
#define sel3      r10
#define sel4      r11
#define sel5      r12
#define sel6      r13
#define sel7      r27
#define sel_line0 r2 
#define sel_line2 r6 
#define offset    r15
#define ptrRow0   r15
#define ptrRow1   r16
#define ptrRow2   r17
#define rot0      r18
#define optr      r19
#define cm18      r20
#define cm0       r25
#define cm9       r21
#define cm32      r22
#define iwidth    r23
#define owidth    r24
#define row_cnt   r26
#define line_count r3
//vector regs
#define vpred0    v0
#define vpred1    v1
#define vpred2    v2
#define vpred3    v3
#define in0       v4
#define in1       v5
#define in2       v6
#define out012    v7
#define vxoffset  v8
#define vpredo    v9
#define vq0       q2
#define vq1       q3
#define vq2       q3
#define vq3       q3
#define vq31      q1   //select prolog,kernle,epilog padding
#define vq5       q3
#define vq6       q3
/* ============================================================================ */
          .global im2col33322_hvx
          .type   im2col33322_hvx, @function
          .balign 32
im2col33322_hvx:
/* ============================================================================ */
       { allocframe(#64)
       } {
         vpredo = vmem(ptrDelta+#0)
         xoffset = vsplatb(xoffset)
       } {
         memd(sp+#0) = r17:16
         memd(sp+#8) = r19:18
         sel0 = #32
         vxoffset = vsplat(xoffset)
       } {
         memd(sp+#16)= r21:20
         memd(sp+#24)= r23:22
       } {
         M1 = sel0
         memd(sp+#32)= r25:24
         memd(sp+#40)= r27:26
         sel0 = #120
       } {
         M0 = sel0
         iwidth = #897
         sel0 = ##0x01010101
       } {
         sel1 = add(sel0, sel0)
         sel2 = asl(sel0, #2)
         sel3 = asl(sel0, #3)
         owidth = #4768      //149*32
       } {
         sel4 = add(sel3, sel3)
         sel5 = asl(sel3, #2)
         sel6 = asl(sel3, #3)
       } {
         sel7 = add(sel6, sel6)
         line_count = startorow
         numorows = add(numorows, startorow)
       } 
/* ============================================================================ */
.L_lines:
       { 
         optr = ptrOut
         cm0 = and(ptrOut,#96)              //is it 0, 32,64 or 96byte alignment
         ptrOut = add(ptrOut,owidth) 
         offset = add(line_count, line_count)
       } {
         cm0 = sub(#0, cm0)
         offset = mpyi(offset, iwidth)
       } {
         vpred0 = vror(vpredo, cm0)  //adjust guidance for 64b alignment
         ptrRow0 = add(ptrIn, offset)     //retard pointer
         cm32 = #-32
         cm9 = add(cm0, #-9)
       } {
         ptrRow1 = add(ptrRow0, iwidth)  //0*299*3-3
         vpred1 = vror(vpred0, cm32)
       } {
         ptrRow2 = add(ptrRow1, iwidth) //1*299*3-3
         vpred2 = vror(vpred1, cm32)
         cm18= add(cm0, #-18)
       } {
         vpred3 = vror(vpred2, cm32)
         row_cnt = #0
         loop1(.L_outer, #8)
       }
/* ============================================================================ */
         .balign 32
.L_outer:
       { in0 = vmemu(ptrRow0++M0)
       } {
         in1 = vmemu(ptrRow1++M0)
       } {
         in2 = vmemu(ptrRow2++M0)
         rot0 =#-26
       } {
         in0 = vror(in0, cm0)
         loop0(.L_loop0, #5)              //create 16 x 42 values
       } {
         in1 = vror(in1, cm9)
         vq31 = vand(vpred1, sel3)        //[0,5]
       } {
         in2 = vror(in2, cm18)
         vq1 = vand(vpred0, sel1)         //[0,0]
       } {
         out012 = vmux(vq1, in1, in0)     //[0,1]
         vq2 = vand(vpred0, sel2)         //[0,1]
         in0 = vror(in0, rot0)            //[0,1]
       } 
/* ============================================================================ */
        .balign 32
.L_loop0:
       { out012 = vmux(vq2, in2, out012)  //[0,2]
         vq3 = vand(vpred0, sel3)         //[0,2]
         in1 = vror(in1, rot0)            //[0,2]
         row_cnt = add(row_cnt, #32)      //[0,2]
       } {
         out012 =vmux(vq3,vxoffset,out012)//[0,3]
         vq0 = vand(vpred0, sel0)         //[0,3]
         in2 = vror(in2, rot0)            //[0,3]
         vq1 = vand(vpred1, sel1)         //[0,3]
       } {
         if(vq0) vmem(optr++M1) = out012  //[0,4]advance 0
         out012 = vmux(vq1, in1, in0)     //[0,4]
         vq2 = vand(vpred1, sel2)         //[0,4]
         in0 = vror(in0, rot0)            //[0,4]
       } {
         out012 = vmux(vq2, in2, out012)  //[0,5]
         in1 = vror(in1, rot0)            //[0,5]
         p2 = cmp.eq(row_cnt, owidth)     //[0,5]
         if(p2.new) jump:nt .L_break      //[0,8]
       } {
         out012 =vmux(vq31,vxoffset,out012)//[0,6]
         vq0 = vand(vpred1, sel0)         //[0,6]
         in2 = vror(in2, rot0)            //[0,6]
         vq1 = vand(vpred2, sel1)         //[0,6]
       } {
         if(vq0) vmem(optr++M1) = out012  //[0,7]advance 1
         out012 = vmux(vq1, in1, in0)     //[0,7]
         vq2 = vand(vpred2, sel2)         //[0,7]
         in0 = vror(in0, rot0)            //[0,7]
       } {
         out012 = vmux(vq2, in2, out012)  //[0,8]
         vq3 = vand(vpred2, sel3)         //[0,8]
         in1 = vror(in1, rot0)            //[0,8]
       } {
         out012 =vmux(vq3,vxoffset,out012)//[0,9]
         vq0 = vand(vpred2, sel0)         //[0,9]
         in2 = vror(in2, rot0)            //[0,9]
         vq1 = vand(vpred3, sel1)         //[0,9]
       } {
         if(vq0) vmem(optr++M1) = out012  //[0,10]advance 2
         out012 = vmux(vq1, in1, in0)     //[0,10]
         vq2 = vand(vpred3, sel2)         //[0,10]
         in0 = vror(in0, rot0)            //[0,10]
       } { 
         out012 = vmux(vq2, in2, out012)  //[0,11]
         vq3 = vand(vpred3, sel3)         //[0,11]
         in1 = vror(in1, rot0)            //[0,11]
         row_cnt = add(row_cnt, #96)      //[0,8]
       } {
         out012 =vmux(vq3,vxoffset,out012)//[0,12]
         vq0 = vand(vpred3, sel0)         //[0,12]
         in2 = vror(in2, rot0)            //[0,12]
         vq1 = vand(vpred0, sel1)         //[1,0]
       } {
         if(vq0) vmem(optr++M1) = out012  //[0,13]advance 3
         out012 = vmux(vq1, in1, in0)     //[1,1]
         vq2 = vand(vpred0, sel2)         //[1,1]
         in0 = vror(in0, rot0)            //[1,1]
       }:endloop0:endloop1
/* ============================================================================ */
        .balign 32
.L_break:
       {
         line_count = add(line_count, #1)
       } {
         p0 = cmp.eq(line_count,numorows)
         if(!p0.new) jump:t .L_lines
       }
/* ============================================================================ */
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
     .size im2col33322_hvx, .L_end-im2col33322_hvx
