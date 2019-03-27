/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
  Memory
     CODESIZE 304 bytes
     STACK    16 bytes

     Description
      Stride 1 or 2 Shuffle the stream with itself to feed the new vrmpy ops in V65
      and while we are at it, do the alignment to save ops
*/
/* --------------------------------------------------------------------------- */
         .text
         .global repstream2_asm
         .balign 32
         .type repstream2_asm, @function
repstream2_asm:
#if 1
/* --------------------------------------------------------------------------- */
#define ptr_x0            r0   //input data raw 
#define ptr_z1            r1   //output cbuf pointer aligned/shuffled data
#define next_in_width     r2   //width of padded input
#define in_depth          r3   //input depth
#define buf_fill          r4   //number of lines to fill
#define rpad_lpad         r5   //right and left pad on input requirement packed
#define stride_width      r13  //stride_width
#define cbuf_base         r14  //base of the circular buffer
#define buf_height        r15  //number of total logical lines
#define inzero            r12  //activation zero value
/* --------------------------------------------------------------------------- */
#define cbuf_eob          r16  //end of circ buf 
#define cbuf_size         r17  //size of cicular buffer
#define width             r10  //width in 128byte block
#define width_cnt         r4   //width left in 128byte block
#define rpad              r11  //right pad used to minimize stray maxes
#define lpad              r6   //left pad that gets removed
#define cm4               r7   //shuffle ints
#define ptr_x1            r8   //temp input ptr
#define buf_width         r9   //width of circ buffer 64*(next_in_width+3-lpad+rpad)&(-4)
/* --------------------------------------------------------------------------- */
#define x3x2x1x0          v0   //input data
#define x7x6x5x4          v1   //next input data
#define y3y2y1y0          v4   //aligned input data
#define y7y6y5y4          v5   //delayed aligned inout data
#define ybyay9y8          v6   //delayed by 2 aligned data
#define z73z62            v3   //shuffled delayed input
#define z51z40            v2   //shuffled delayed input
#define z73z62_z51z40     v3:2 //shuffled delayed input
#define vin_zero          v7
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug vec reg
/* --------------------------------------------------------------------------- */
   {  allocframe(#56)                                 //0th entry on stack is (112+8)/4=30 ints
   } {
      memd(sp+#0<<2) = r17:16                         //
      memd(sp+#2<<2) = r19:18                         //
   } {
      width = add(next_in_width, #3)                  //[S, 0]round to nearest 4
      inzero = memw(sp+#19<<2)                        //[S, 0]
   } {
      width = sub(width.L, rpad_lpad.L)               //[S, 1]next_in_width-lpad+3
      inzero = vsplatb(inzero)                        //[S, 2]
   } {
      rpad = lsr(rpad_lpad, #18)                      //[S, 1]next_in_width-lpad+3+rpad
      width = lsr(width, #2)                          //[S, 2]number of 128byte blks to fetch 
      buf_height = memw(sp+#18<<2)                    //[S, 0]
   } {
      cbuf_size = mpyi(buf_height, in_depth)          // 
      rpad = add(rpad, width)                         //
      buf_fill = mpyi(buf_fill, in_depth)             //
   } {
      vin_zero = vsplat(inzero)                       //[S, 3]
      cbuf_size = mpyi(cbuf_size, rpad)               // 
      cm4 = #-4                                       //[S, 1]shuffle ints
   } {
      cbuf_size = asl(cbuf_size, #3)                  //
      lpad = asl(rpad_lpad, #5)                       //[S, 1]alignment % 128
      stride_width = memw(sp+#16<<2)                  //[S, 0] stride_width 
      cbuf_base = memw(sp+#17<<2)                     //[S, 0]
   } {
      cbuf_eob = add(cbuf_base, cbuf_size)            //
      p0 = cmp.eq(stride_width, #1)                   //[S, 2]is stride = 1? (or 2)
      buf_fill = lsr(buf_fill, #5)                    //
   } {
      cbuf_eob = add(cbuf_eob, #-4)                   //
      loop1(.L_rows, buf_fill)                        //[S, 3]loop over num lines
   }
/* --------------------------------------------------------------------------- */
   .balign 32
.L_rows:
   {  p3 =sp1loop0(.L_cols, rpad)                     //[P, 0]inner loop width
      width_cnt = add(width, #-1)                     //[P, 0]
      x3x2x1x0 = vmem(ptr_x0+#0)                      //[P, 0]load 1st 128
      x7x6x5x4 = vin_zero                             //[P, 0]
   } {
      ptr_x1 = add(ptr_x0, #128)                      //[P, 1]
      p1 = cmp.gt(width_cnt, #0)                      //[P, 1]
      width_cnt = add(width_cnt, #-1)                 //[P, 1]
   } {
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt       //[P, 2]stride = 1 data
      y3y2y1y0 = valign(x7x6x5x4,x3x2x1x0, lpad)      //[P, 2]aligned data next 128
      x3x2x1x0 = x7x6x5x4                             //[P, 2]update pipe
      x7x6x5x4 = vin_zero                             //[P, 2]
   } {
      p1 = cmp.gt(width_cnt, #0)                      //[P, 3]
      width_cnt = add(width_cnt, #-1)                 //[P, 3]
      if(p0) jump .L_cols                             //[P, 3]skip 1 load
      if(p0) ybyay9y8 = y3y2y1y0                      //[P, 3]update 1 stage pipe y0 = y2
   } {
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt       //[P, 4]stride = 2 data
      y7y6y5y4 = valign(x7x6x5x4,x3x2x1x0, lpad)      //[P, 4]aligned data next 128
      x3x2x1x0 = x7x6x5x4                             //[P, 4]update pipe
      x7x6x5x4 = vin_zero                             //[P, 4]
   } {
      p1 = cmp.gt(width_cnt, #0)                      //[P, 5]
      width_cnt = add(width_cnt, #-1)                 //[P, 5]
   } 
.L_cols:
   {  
      if( p0) y3y2y1y0 = ybyay9y8                     //[0, 0]update 1 stage pipe y0 = y2
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt       //[0, 0]
      ybyay9y8 = valign(x7x6x5x4,x3x2x1x0, lpad)      //[0, 0]aligned data next 128
      x3x2x1x0 = x7x6x5x4                             //[0, 0]update pipe
   } { 
      if(p3) vmem(ptr_z1++#1) = z73z62                //[0, 1]empty pipe
      p1 = cmp.gt(width_cnt, #0)                      //[0, 1]
      width_cnt = add(width_cnt, #-1)                 //[0, 1]
      x7x6x5x4 = vin_zero                             //[0, 1]
   } { 
      z73z62_z51z40= vshuff(ybyay9y8,y3y2y1y0,cm4)    //[0, 2]shuffle up for new vrmpy
      vmem(ptr_z1++#1) = z51z40.new                   //[0, 2]empty pipe
      if(!p0) y3y2y1y0 = y7y6y5y4                     //[0, 2]update 2 stage shift reg
      if(!p0) y7y6y5y4 = ybyay9y8                     //[0, 2]y0 = y1 || y2 = y2
   }:endloop0
/* --------------------------------------------------------------------------- */
   {  vmem(ptr_z1++#1) = z73z62                       //[E, 0]empty pipe
      ptr_x0 = addasl(ptr_x0, next_in_width, #5)      //[E, 0]update input next row
   } {
      p2 = cmp.gt(ptr_z1, cbuf_eob)                   //if circ buf write and end of buf
      if(p2.new) ptr_z1 = sub(ptr_z1, cbuf_size)      //then subtract buf size take to 1st row
   }:endloop1
   {
      r17:16 = memd(sp+#0<<2)                         //
      r19:18 = memd(sp+#2<<2)                         //
   } {
      dealloc_return                                  //[T, 0]return to caller
   }
/* --------------------------------------------------------------------------- */
#else
//older previous version
/* --------------------------------------------------------------------------- */
#define ptr_x0            r0   //input data raw 
#define ptr_z1            r1   //output pointer aligned/shuffled data
#define next_in_width     r2   //width of padded input
#define buf_height        r3   //number of total lines
#define rpad_lpad         r4   //right and left pad on input requirement packed
#define stride            r5   //stride_width
#define inzero            r12
#define width             r10  //width in 128byte block
#define width_cnt         r4   //width left in 128byte block
#define rpad              r11
#define lpad              r6 
#define cm4               r7   //shuffle ints
#define ptr_x1            r8   //temp input ptr
#define in_width_32       r9   //total width in bytes of input
/* --------------------------------------------------------------------------- */
#define x3x2x1x0          v0   //input data
#define x7x6x5x4          v1   //next input data
#define y3y2y1y0          v4   //aligned input data
#define y7y6y5y4          v5   //delayed aligned inout data
#define ybyay9y8          v6   //delayed by 2 aligned data
#define z73z62            v3   //shuffled delayed input
#define z51z40            v2   //shuffled delayed input
#define z73z62_z51z40     v3:2 //shuffled delayed input
#define vin_zero             v7
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug vec reg
/* --------------------------------------------------------------------------- */
   {  in_width_32 = asl(next_in_width, #5)            //[S, 0] distance to next row
      inzero = memw(sp+#0<<2)                         //[S, 0]
      width = add(next_in_width, #3)                  //[S, 0]round to nearest 4
      rpad = lsr(rpad_lpad, #18)                      //[S, 0]extract right pad
   } { 
      width = sub(width.L, rpad_lpad.L)               //[S, 1]next_in_width-lpad+3
      cm4 = #-4                                       //[S, 1]shuffle ints
      lpad = asl(rpad_lpad, #5)                       //[S, 1]alignment % 128
   } {  
      inzero = vsplatb(inzero)                        //[S, 2]
      p0 = cmp.eq(stride, #1)                         //[S, 2]is stride = 1? (or 2)
      width = lsr(width, #2)                          //[S, 2]number of 128byte blks to fetch 
   } {  
      vin_zero = vsplat(inzero)                       //[S, 3]
      loop1(.L_rows, buf_height)                      //[S, 3]loop over num lines
      rpad = add(rpad, width)                         //[S, 3]account for prolog
   }
/* --------------------------------------------------------------------------- */
   .balign 32
.L_rows:
   {  p3 =sp1loop0(.L_cols, rpad)                     //[P, 0]inner loop width
      width_cnt = add(width, #-1)                     //[P, 0]
      x3x2x1x0 = vmem(ptr_x0+#0)                      //[P, 0]load 1st 128
      x7x6x5x4 = vin_zero                             //[P, 0]
   } {
      ptr_x1 = add(ptr_x0, #128)                      //[P, 1]
      p1 = cmp.gt(width_cnt, #0)                      //[P, 1]
      width_cnt = add(width_cnt, #-1)                 //[P, 1]
   } {
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt       //[P, 2]stride = 1 data
      y3y2y1y0 = valign(x7x6x5x4,x3x2x1x0, lpad)      //[P, 2]aligned data next 128
      x3x2x1x0 = x7x6x5x4                             //[P, 2]update pipe
      x7x6x5x4 = vin_zero                             //[P, 2]
   } {
      p1 = cmp.gt(width_cnt, #0)                      //[P, 3]
      width_cnt = add(width_cnt, #-1)                 //[P, 3]
      if(p0) jump .L_cols                             //[P, 3]skip 1 load
      if(p0) ybyay9y8 = y3y2y1y0                      //[P, 3]update 1 stage pipe y0 = y2
   } {
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt       //[P, 4]stride = 2 data
      y7y6y5y4 = valign(x7x6x5x4,x3x2x1x0, lpad)      //[P, 4]aligned data next 128
      x3x2x1x0 = x7x6x5x4                             //[P, 4]update pipe
      x7x6x5x4 = vin_zero                             //[P, 4]
   } {
      p1 = cmp.gt(width_cnt, #0)                      //[P, 5]
      width_cnt = add(width_cnt, #-1)                 //[P, 5]
   } 
.L_cols:
   {  
      if( p0) y3y2y1y0 = ybyay9y8                     //[0, 0]update 1 stage pipe y0 = y2
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt       //[0, 0]
      ybyay9y8 = valign(x7x6x5x4,x3x2x1x0, lpad)      //[0, 0]aligned data next 128
      x3x2x1x0 = x7x6x5x4                             //[0, 0]update pipe
   } { 
      if(p3) vmem(ptr_z1++#1) = z73z62                //[0, 1]empty pipe
      p1 = cmp.gt(width_cnt, #0)                      //[0, 1]
      width_cnt = add(width_cnt, #-1)                 //[0, 1]
      x7x6x5x4 = vin_zero                             //[0, 1]
   } { 
      z73z62_z51z40= vshuff(ybyay9y8,y3y2y1y0,cm4)    //[0, 2]shuffle up for new vrmpy
      vmem(ptr_z1++#1) = z51z40.new                   //[0, 2]empty pipe
      if(!p0) y3y2y1y0 = y7y6y5y4                     //[0, 2]update 2 stage shift reg
      if(!p0) y7y6y5y4 = ybyay9y8                     //[0, 2]y0 = y1 || y2 = y2
   }:endloop0
/* --------------------------------------------------------------------------- */
   {  vmem(ptr_z1++#1) = z73z62                       //[E, 0]empty pipe
      ptr_x0 = add(ptr_x0, in_width_32)               //[E, 0]update input next row
   }:endloop1
   {  jumpr r31                                       //[T, 0]return to caller
   }
#endif
/* --------------------------------------------------------------------------- */
.L_end:
      .size repstream2_asm, .L_end-repstream2_asm
