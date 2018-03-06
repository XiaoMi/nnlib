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
     CODESIZE 160 bytes
     STACK    0 bytes

     Description
      Stride 1 or 2 Shuffle the stream with itself to feed the new vrmpy ops
      and while we are at it, do the alignment to save ops
     void repstream2_asm(uint8_t *x, uint8_t *y,
                         int w, int d, int fill_height, int rpad_lpad, int stride_w,
                         uint8_t *, 
                         int buf_height, int in_offset);

*/
/* --------------------------------------------------------------------------- */
         .text
         .global repstreamN_asm
         .balign 32
         .type repstreamN_asm, @function
repstreamN_asm:
/* --------------------------------------------------------------------------- */
#define ptr_x0            r0   //input data raw 
#define ptr_z1            r1   //output cbuf pointer aligned/shuffled data
#define next_in_width     r2   //width of padded input
#define in_depth          r3   //input depth
#define buf_fill          r4   //number of lines to fill
#define rpad_lpad         r5   //right and left pad on input requirement packed
#define stride_width      r4   //stride_width any value
#define cbuf_base         r14  //base of the circular buffer
#define buf_height        r15  //number of total logical lines
#define inzero            r12  //activation zero value
#define num_acc           r18  //number of accumulators to interleave for must be even
/* --------------------------------------------------------------------------- */
#define cbuf_eob          r16  //end of circ buf 
#define cbuf_size         r17  //size of cicular buffer
#define width             r10  //width in 128byte block
#define width_cnt         r13  //width left in 128byte block
#define rpad              r11  //right pad used to minimize stray maxes
#define lpad              r6   //left pad that gets removed
#define cm4               r7   //shuffle ints
#define ptr_x1            r8   //temp input ptr
#define ptr_x2            r9   //width of circ buffer 64*(next_in_width+3-lpad+rpad)&(-4)
#define end_stream        r19  //number of 128b blocks for 2nd stream to advance
/* --------------------------------------------------------------------------- */
#define x3x2x1x0          v0   //input data
#define x7x6x5x4          v1   //next input data
#define y3y2y1y0          v4   //aligned input data
#define y7y6y5y4          v5   //delayed aligned inout data
#define u3u2u1u0          v8   //delayed by 2 aligned data
#define v3v2v1v0          v6   //delayed by 2 aligned data
#define y3x3y2x2          v3   //shuffled delayed input
#define y1x1y0x0          v2   //shuffled delayed input
#define y3x3y2x2_y1x1y0x0 v3:2 //shuffled delayed input
#define vin_zero          v7
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug vec reg
/* --------------------------------------------------------------------------- */
   {  allocframe(#56)                                    //0th entry on stack is (112+8)/4=30 ints
      width = add(next_in_width, #3)                     //[S, 0]round to nearest 4
   } {
      width = sub(width.L, rpad_lpad.L)                  //[S, 1]next_in_width-lpad+3
      memd(sp+#0<<2) = r17:16                            //[S, 1]save
      inzero = memw(sp+#19<<2)                           //[S, 8]get zero of activations
   } {
      rpad = lsr(rpad_lpad, #18)                         //[S, 2]next_in_width-lpad+3+rpad
      width = lsr(width, #2)                             //[S, 2]number of 128byte blks to fetch 
      buf_height = memw(sp+#18<<2)                       //[S, 2]height of buffer num 32depth
   } {
      memd(sp+#2<<2) = r19:18                            //[S, 3]save
      cbuf_size = mpyi(buf_height, in_depth)             //[S, 3]total size of buffer
      rpad = add(rpad, width)                            //[S, 3]total width of active buffer
      inzero = vsplatb(inzero)                           //[S, 3]
   } {
      buf_fill = mpyi(buf_fill, in_depth)                //[S, 4]
      cbuf_size = mpyi(cbuf_size, rpad)                  //[S, 4]
      lpad = zxth(rpad_lpad)                             //[S, 4]alignment % 128  
   } {
      buf_fill = lsr(buf_fill, #5)                       //[S, 5]num in_depth*hiehgt/32
      lpad = asl(lpad, #5)                               //[S, 5]alignment*32
      num_acc = memw(sp+#20<<2)                          //[S, 5]how many accumulators must we parallelize
   } {
      num_acc = asl(num_acc, #4)                         //[S, 6]if 6 then shift 2 stream by 96bytes
      loop1(.L_rows, buf_fill)                           //[S, 6]loop over num lines
      stride_width = memw(sp+#16<<2)                     //[S, 6] stride_width 
   } {
      stride_width = mpyi(stride_width, num_acc)         //[S, 7]get start of buffer to interleave
      lpad = and(lpad, #127)                             //[S, 7]round padding to 128
      cbuf_base = memw(sp+#17<<2)                        //[S, 7]get buffer base
      cbuf_size = asl(cbuf_size, #3)                     //[S, 7]complete buffer size
   } {
      stride_width = add(stride_width, lpad)             //[S, 8]compute offset to upper accumulators
      cbuf_eob = add(cbuf_base, cbuf_size)               //[S, 8]figure end of buffer
   } {
      vin_zero = vsplat(inzero)                          //[S, 9]splat activation zero for pad
      cm4 = #-4                                          //[S, 9]shuffle ints
      cbuf_eob = add(cbuf_eob, #-4)                      //[S, 9]dela with >= case
      end_stream = lsr(stride_width, #7)                 //[S, 9]when 2nd stream hits end of buffer
   }
/* --------------------------------------------------------------------------- */
   .balign 32
.L_rows:
   {  p3 =sp1loop0(.L_cols, rpad)                        //[P, 0]inner loop width
      width_cnt = add(width, #-1)                        //[P, 0]
   } {
      ptr_x2 = add(ptr_x0, stride_width)                 //[P, 1]
      x3x2x1x0 = vmem(ptr_x0+#0)                         //[0, 0]load 1st 128
      ptr_x1 = add(ptr_x0, #128)                         //[P, 1]
   } {
      y3y2y1y0 = vmem(ptr_x2++#1)                        //[P, 2]load 1st 128
      p1 = cmp.gt(width_cnt, #0)                         //[0, 1]
      p2 = cmp.gt(width_cnt, end_stream)                 //[0, 1]
      x7x6x5x4 = vin_zero                                //[0, 1]
   }
.L_cols:
   {  width_cnt = add(width_cnt, #-1)                    //[0, 2]
      if(p3) vmem(ptr_z1++#1) = y3x3y2x2                 //[0, 2]empty pipe
      y7y6y5y4 = vin_zero                                //[0, 2]
   } {  
      if(p1) x7x6x5x4.tmp = vmem(ptr_x1++#1):nt          //[0, 3]
      u3u2u1u0 = valign(x7x6x5x4,x3x2x1x0, lpad)         //[0, 3]aligned data next 128
      x3x2x1x0 = x7x6x5x4                                //[0, 3]update pipe
   } { 
      if(p2) y7y6y5y4.tmp = vmem(ptr_x2++#1):nt          //[0, 4]
      v3v2v1v0 = valign(y7y6y5y4,y3y2y1y0, stride_width) //[0, 4]aligned data next 128
      y3y2y1y0 = y7y6y5y4                                //[0, 4]update pipe
      x7x6x5x4 = vin_zero                                //[0, 4]
   } { 
      y3x3y2x2_y1x1y0x0 = vshuff(v3v2v1v0,u3u2u1u0,cm4)  //[0, 5]shuffle up for new vrmpy
      vmem(ptr_z1++#1) = y1x1y0x0.new                    //[0, 5]empty pipe
      p1 = cmp.gt(width_cnt, #0)                         //[0, 5]
      p2 = cmp.gt(width_cnt, end_stream)                 //[0, 5]
   }:endloop0
/* --------------------------------------------------------------------------- */
   {  vmem(ptr_z1++#1) = y3x3y2x2                        //[0, 1]empty pipe
      ptr_x0 = addasl(ptr_x0, next_in_width, #5)         //[E, 0]update input next row
   } {
      p2 = cmp.gt(ptr_z1, cbuf_eob)                      //[E, 1]if circ buf write and end of buf
      if(p2.new) ptr_z1 = sub(ptr_z1, cbuf_size)         //[E, 1]then subtract buf size take to 1st row
   }:endloop1
   {
      r17:16 = memd(sp+#0<<2)                            //[T, 0]
      r19:18 = memd(sp+#2<<2)                            //[T, 0]
   } {
      dealloc_return                                     //[T, 1]return to caller
   }
/* --------------------------------------------------------------------------- */
.L_end:
      .size repstreamN_asm, .L_end-repstreamN_asm
