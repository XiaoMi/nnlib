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
  Memory
     CODESIZE 1408 bytes
     STACK     112 bytes

  Description
     Utilize the v65 vrmpy instructions. Common wiehgts with 2 inputs and 2 outputs. 2 data inputs
     are in a pair. Key is to feed each input with a different stream. Solution is to shuffle the
     stream with a delayed version of itself. This doubles the size of the activations so a 
     smaller circular buffer of size filt_height*input depth*width*2.

     Example depth = 16 shuffle blocks of 4 bytes together e.g. x00 =[x00.0,x00.1,x00.2,x00.3]

       x00 x01 x02 x03|x10 x11 x12 x13|x20 x21 x22 x23|x30 x31 x32 x33
       x40 x41 x42 x43|x50 x51 x52 x53|x60 x61 x62 x63|x70 x71 x72 x73
       x80 x81 x82 x83|x90 x91 x92 x93|xa0 xa1 xa2 xa3|xb0 xb1 xb2 xb3
       xc0 xc1 xc2 xc3|xd0 xd1 xd2 xd3|xe0 xe1 xe2 xe3|xf0 xf1 xf2 xf3
     to
       x00 x40 x01 x41 x02 x42 x03 x43|x10 x50 x11 x51 x12 x52 x13 x53|
       x20 x60 x21 x61 x22 x62 x23 x63|x30 x70 x31 x71 x32 x72 x33 x73|
       x40 x80 x41 x81 x42 x82 x43 x83|x50 x90 x51 x91 x52 x92 x53 x93|
       x60 xa0 x61 xa1 x62 xa2 x63 xa3|x70 xb0 x71 xb1 x72 xb2 x73 xb3|
       x80 xc0 x81 xc1 x82 xc2 x83 xc3|x90 xd0 x91 xd1 x92 xd2 x93 xd3|
       xa0 xe0 xa1 xe1 xa2 xe2 xa3 xe3|xb0 xf0 xb1 xf1 xb2 xf2 xb3 xf3|
       xc0     xc1     xc2     xc3    |xd0     xd1     xd2     xd3    |
       xe0     xe1     xe2     xe3    |xf0     xf1     xf2     xf3    |

     So each memd access into the buffer access two streams which are delayed from each other.
     While this is occuring the sequence can be aligned so that the extra computation on the 
     ends can be minimized.
     To further minimize memory the circular buffer is updated inside the kernel each 
     line.
     This version consumes 2 sets fo weights (64) each time and produces 2
     rows of the output at once.
 */
/*===============================================================================*/
        .text
        .file "gvconv2dbbb_circ_d64_v65_h.S"
        .global gvconv2dbbb_circ_d64_v65_asm
        .balign 32
        .type  gvconv2dbbb_circ_d64_v65_asm, @function
gvconv2dbbb_circ_d64_v65_asm:
/*===============================================================================*/
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug vec reg
/*===============================================================================*/
/* ---------------------------------- CALL REGS -------------------------------- */
#define ptr_xi                 r0     //12 activation data
#define ptr_wi                 r1     //13 weights
#define ptr_zi                 r2     //14 results
#define next_inbuf_width       r3     //(pad_l+in_width+pad_r)
#define out_width_depth        r4     //next line amount 
#define out_width              r5     //15 amount of work to be done
#define stride_h_w             r6     //30 stride_height, stride_width 
#define in_depth               r22    //31 input depth multiples of 32
#define filt_width             r23    //32 horizontal fuilter width
#define filt_height            r8     //33 filt_height lines per filter
#define out_height             r9     //34 number of vertical lines to perform
#define ptr_filtsum            r24    //35 includes the computation filt_sum * in_offset + biasvec
#define ptr_max                r31    //36 maximum and minum buffer
#define recip_level            r26    //37 255 / (MAX - MIN) - used to scale to bytes
#define out_width_32           r7     //actual out_width in depth32
#define ptr_cbufi              r16    //40 read buffer pointer
#define zshift                 r6     //41 extra shift on output before quantization
#define in_zero                r25    //42
#define cbuf_eob               r27    //18 end of cuirc buffer
#define cbuf_size              r28    //19 size in bytes of circ buf -1
#define weight_stride          r15    //22 distance to next set of weights
/* --------------------------------- SCALER REGS ------------------------------- */
#define cm4                    r2     //shuffle/deal ints
#define col_count              r2     //horizontal counter
#define in_width_32            r3     //total input width in bytes in buffer
#define x70_x30                r15:14 //4n+0 inputs
#define x60_x20                r17:16 //4n+0 inputs
#define x50_x10                r19:18 //4n+0 inputs
#define x40_x00                r15:14 //4n+0 inputs
#define x71_x31                r17:16 //4n+1 inputs
#define x61_x21                r15:14 //4n+1 inputs
#define x51_x11                r19:18 //4n+1 inputs
#define x41_x01                r17:16 //4n+1 inputs
#define ptr_wi_ptr_xi          r1:0   //
#define fetch_ptr_base         r1     //base pointer for l1 prefetch
#define fetch_ptr              r10    //current pointer for l1 prefetch
#define stride3                r11    //3*stride
#define stride                 r12    //current to next input
#define ptr_x0                 r26    //base input pointer
#define ptr_x1                 r13    //current input ptr      
#define ptr_w0                 r20    //13 even output depth 32 weights
#define ptr_w1                 r21    //20 odd output depth 32 weights
#define ptr_z0                 r0     //even output depth 32 outputs
#define ptr_z1                 r25    //21 write buffer sp position and odd output depth 32 outputs
#define adjust                 r10    //
#define delta                  r7     //difference between filt height and stride height
#define buf_fill               r17    //22 number of total lines in circ buffer to write
#define out_width_64           r22    //actual out_width in depth32 *2
/* ---------------------------------- VEC REGS -------------------------------- */
#define x3x2x1x0               v10    //input data
#define x7x6x5x4               v11    //next input data
#define y3y2y1y0               v14    //aligned input data
#define y7y6y5y4               v15    //delayed aligned inout data
#define y7y6y5y4_y3y2y1y0      v15:14 //aligned data
#define ybyay9y8               v16    //delayed by 2 aligned data
#define z73z62                 v13    //shuffled delayed input
#define z51z40                 v12    //shuffled delayed input
#define z73z62_z51z40          v13:12 //shuffled output data 
/* ---------------------------------------------------------------------------- */
#define s17_s13                v25:24 //odd  output accs 3,7
#define s16_s12                v23:22 //odd  output accs 2,6
#define s15_s11                v21:20 //odd  output accs 1,5
#define s14_s10                v19:18 //odd  output accs 0,4
#define s07_s03                v17:16 //even output accs 3,7
#define s06_s02                v15:14 //even output accs 2,6
#define s05_s01                v13:12 //even output accs 1,5
#define s04_s00                v11:10 //even output accs 0,4
#define s17                    v25    //odd acc 7
#define s16                    v23    //odd acc 6
#define s15                    v21    //odd acc 5
#define s14                    v19    //odd acc 4
#define s07                    v17    //even acc 7
#define s06                    v15    //even acc 6
#define s05                    v13    //even acc 5
#define s04                    v11    //even acc 4
#define s13                    v24    //odd acc 3
#define s12                    v22    //odd acc 2
#define s11                    v20    //odd acc 1
#define s10                    v18    //odd acc 0
#define s03                    v16    //even acc 3
#define s02                    v14    //even acc 2
#define s01                    v12    //even acc 1
#define s00                    v10    //even acc 0
#define w00                    v0     //weights even 0-31
#define w01                    v0     //weights even 32-63
#define w10                    v0     //weights odd 0-31
#define w11                    v0     //weights odd 32-63
#define vrecip                 v1     //reciprocal 255/MAx replicated
#define s0_sh                  v8     //shifted value
#define s1_sh                  v26    //shifted value
#define wsum0                  v2     //sum of weights column + bias add 0-31
#define wsum1                  v3     //sum of weights column + bias add 32-63
#define d010                   v27    //even lines upper 16bit packed accs 0,1
#define d032                   v28    //even lines upper 16bit packed accs 2,3
#define d03210                 v28    //8bit shifted, packed saturated 0-3
#define d054                   v29    //even lines upper 16bit packed accs 4,5
#define d076                   v31    //even lines upper 16bit packed accs 6,7
#define d07654                 v31    //8bit shifted, packed saturated 4-7
#define d110                   v27    //odd lines upper 16bit packed accs 0,1
#define d132                   v28    //odd lines upper 16bit packed accs 2,3
#define d13210                 v28    //8bit shifted, packed saturated 0-3
#define d154                   v29    //odd lines upper 16bit packed accs 4,5
#define d176                   v31    //odd lines upper 16bit packed accs 6,7
#define d17654                 v31    //8bit shifted, packed saturated 4-7
#define maxo_maxe              v5:4   //packed maxes
#define maxo                   v5     //odd maxes
#define maxe                   v4     //even maxes
#define mino_mine              v7:6   //packed mins 
#define mino                   v7     //odd mins
#define mine                   v6     //even mins
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/* --------------------------------------------------------------------------- */
   {   allocframe(#112)                                  //0th entry on stack is 112+8)/4=30 ints
       stride_h_w = memw(sp+#0<<2)                       //stride horizontl and vertical
   } {
       memd(sp+#4<<2) = r21:20                           //save 20,21
       memd(sp+#6<<2) = r23:22                           //save 22,23
   } {
       memd(sp+#0<<2)  = r17:16                          //save 16,17
       memd(sp+#2<<2)  = r19:18                          //save 18,19
   } {
       memd(sp+#8<<2) = r25:24                           //save 24,25
       memd(sp+#10<<2) = r27:26                          //save 26,27
   } {
       filt_height = memw(sp+#33<<2)                     //filter height
       memw(sp+#15<<2) = out_width                       //save output width
       stride = zxth(stride_h_w)                         //horizontal stride
   } {
       in_depth = memw(sp+#31<<2)                        //input depth
       filt_width = memw(sp+#32<<2)                      //filter width
   } {
       ptr_max = memw(sp+#36<<2)                         //get max/min ptr
       cbuf_size = mpyi(filt_height, in_depth)           //circular buffer size
       stride = asl(stride, #5)                          //32 * stride_w
       dcfetch(ptr_xi+#0<<6)                             //
   } {
       out_height = memw(sp+#34<<2)                      //height of output
       in_zero = memw(sp+#41<<2)                         //
       stride3 = addasl(stride, stride, #1)              //3Xstride
       cbuf_size = mpyi(cbuf_size, next_inbuf_width)     //circular buffer size
   } {
       cbuf_size = add(cbuf_size, cbuf_size)             //x2
       in_zero = vsplatb(in_zero)                        //
       memd(sp+#12<<2) = ptr_wi_ptr_xi                   //save weights:activation
       maxe = vmem(ptr_max+#0)                           //
   } {
       weight_stride=mpy(filt_width.L,filt_height.L)     //offset between filter rows
       filt_width = asl(filt_width, #2)                  //*32/8
       mine = vmem(ptr_max+#1)                           //
       memw(sp+#14<<2) = ptr_zi                          //save output ptr
   } {
       weight_stride=mpy(weight_stride.L,in_depth.L)     //distance between weight rows
       recip_level = memw(sp+#37<<2)                     //255/max
       ptr_cbufi = memw(sp+#39<<2)                       //circular buffer
   } {
       out_width_32 = memw(sp+#38<<2)                    //total width of output
       vrecip = vsplat(recip_level)                      //used to compress to 8bits 255/max
       cbuf_eob = add(ptr_cbufi, cbuf_size)              //end of circ buffer marker
       dcfetch(ptr_xi+#1<<6)                             //
   } {
       memw(sp+#21<<2) = ptr_cbufi                       //cbuf write ptr
       filt_width = add(filt_width, #-1)                 //account for epilog
       ptr_wi += asl(weight_stride,#5)                   //weights stride
       filt_height = mpyi(filt_height, in_depth)         //total number of depth32 filter rows
   } {
       weight_stride = asl(weight_stride, #6)            //
       col_count = memw(sp+#15<<2)                       //initialize width count
       cbuf_eob = add(cbuf_eob, #-4)                     //make so comparison is >= eob
       dcfetch(ptr_xi+#2<<6)                             //
   } {
       filt_height = lsr(filt_height, #5)                //num d32 rows in filter
       out_width_64 = add(out_width_32, out_width_32)    //
       ptr_filtsum = memw(sp+#35<<2)                     //ptr to the sum of filters+offset
       dcfetch(ptr_xi+#3<<6)                             //
   } {
       memw(sp+#20<<2) = ptr_wi                          //spill weight stride for later
       memw(sp+#24<<2) = weight_stride                   //
       in_width_32 = asl(next_inbuf_width, #6)           //next d32 line x 2
       filt_height = add(filt_height, #-1)               //
   }
/* -------------------------------------------------------------------------- */
   .balign 32
.L_height:
   {   
       ptr_x0 = memw(sp+#12<<2)                          //ptr_x0=ptr_cbufi read circ buffer
       wsum0 = vmem(ptr_filtsum++#1)                     //set 1st weight offset
       nop; nop
   } {
       ptr_z0 = memw(sp+#14<<2)                          //output ptr for even lines
       wsum1 = vmem(ptr_filtsum++#1)                     //set 2nd weight offset
       fetch_ptr_base = add(ptr_x0, in_width_32)         //fetch is next row ahead
       nop
   } { 
       p1 = cmp.gt(fetch_ptr_base, cbuf_eob)             //if prefetch >= circ buffer wrap around
       if(p1.new)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size) //wrap fetch ptr around independently
       s06_s02 = vcombine(wsum0,wsum0)                   //init sum2 and 6
       s07_s03 = vcombine(wsum0,wsum0)                   //init sum3 and 7
   } {
       loop1(.L_filt_height, filt_height)                //setup vertical filte rloop
       s04_s00 = vcombine(wsum0,wsum0)                   //init sum0 and 4
       s05_s01 = vcombine(wsum0,wsum0)                   //init sum1 and 5
       ptr_z1 = add(ptr_z0, out_width_32)                //next output depth32 (as outdepth 64)
   } {
       s16_s12 = vcombine(wsum1,wsum1)                   //init sum 2 and 6
       s17_s13 = vcombine(wsum1,wsum1)                   //init sum 3 and 7
       ptr_w0 = memw(sp+#13<<2)                          //access ptr weight
       p3 = sp1loop0(.L_filt_width, filt_width)          //setup inner filter loop
   } {
       memw(sp+#14<<2) += out_width_64                   //update output ptr
       s14_s10 = vcombine(wsum1,wsum1)                   //init sum 0 and 4
       s15_s11 = vcombine(wsum1,wsum1)                   //init sum 1 and 5
       ptr_w1 = memw(sp+#20<<2)                          //access weights stride
   }
   .balign 32
.L_width:
.L_filt_height:
   {   x70_x30 = memd(ptr_x0+stride3<<#1)                //[0, 0]load pt 3 and 7
       fetch_ptr = add(fetch_ptr_base, #0)               //initial fetch ptr
       p0 = cmp.eq(filt_height, #0)                      //
       if(p0.new) jump:nt .L_last1
   } {
       x60_x20 = memd(ptr_x0+stride<<#2)                 //[0, 2]load pt 2 and 6
       ptr_x1 = ptr_x0                                   //set up currne tinput ptr
       ptr_x0 = add(ptr_x0, in_width_32)                 //if >= buf_size -= buf_size
       fetch_ptr_base=add(fetch_ptr_base,in_width_32)    //if >= buf_size -= buf_size
   } 
   .balign 32
.L_filt_width:
   {   
       dcfetch(fetch_ptr+#0<<6)                          //[0, 2]fetch 64bytes-2 lots 8 x 4 bytes
       fetch_ptr = add(fetch_ptr, #64)                   //[0, 2]inc fetch by 32/64 bytes (1 line)
       nop; nop
   } {
       w00.tmp = vmem(ptr_w0+#0)                         //[0, 3]1st 32 weights of out depth
       s06_s02.w+= vrmpy(w00.b, x60_x20.ub)              //[0, 3]macc 2,6  out 0
       s07_s03.w+= vrmpy(w00.b, x70_x30.ub)              //[0, 3]macc 3,7 even row
       x50_x10 = memd(ptr_x1+stride<<#1)                 //[0, 3]load pt 1 5
   } {
       w10.tmp = vmem(ptr_w1+#0)                         //[0, 4]1st 32 weights stream 1
       s16_s12.w += vrmpy(w10.b, x60_x20.ub)             //[0, 4]acc 2,6 out 1
       s17_s13.w += vrmpy(w10.b, x70_x30.ub)             //[0, 4]acc 3,7 out 1
       x40_x00 = memd(ptr_x1++#1<<3)                     //[0, 4]load pts 0, 4
   } {
       w00.tmp = vmem(ptr_w0++#1)                        //[0, 5]same 1st 32weights stream 0
       s04_s00.w += vrmpy(w00.b, x40_x00.ub)             //[0, 5]acc 0,4,1,5 out 0
       s05_s01.w += vrmpy(w00.b, x50_x10.ub)             //[0, 5]
       x71_x31 = memd(ptr_x1+stride3<<#1)                //[0, 5]
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[0, 6]same 1st 32weight stream 1
       s14_s10.w += vrmpy(w10.b, x40_x00.ub)             //[0, 6]acc 0,4,1,5 stream 1
       s15_s11.w += vrmpy(w10.b, x50_x10.ub)             //[0, 6]
       x61_x21 = memd(ptr_x1+stride<<#2)                 //[0, 6]
   } {
       w01.tmp = vmem(ptr_w0+#0)                         //[0, 7]2nd 32weights stream 0
       s06_s02.w += vrmpy(w01.b, x61_x21.ub)             //[0, 7]acc 2,3,6,7
       s07_s03.w += vrmpy(w01.b, x71_x31.ub)             //[0, 7]
       x51_x11 = memd(ptr_x1+stride<<#1)                 //[0, 7]
   } {
       w11.tmp = vmem(ptr_w1+#0)                         //[0, 8]2nd 32weights of stream 1
       s16_s12.w += vrmpy(w11.b, x61_x21.ub)             //[0, 8]
       s17_s13.w += vrmpy(w11.b, x71_x31.ub)             //[0, 8]
       x41_x01 = memd(ptr_x1++#1<<3)                     //[0, 8]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[0, 9]same 2nd 32weights stream 0
       s04_s00.w += vrmpy(w01.b, x41_x01.ub)             //[0, 9]
       s05_s01.w += vrmpy(w01.b, x51_x11.ub)             //[0, 9]
       x70_x30 = memd(ptr_x1+stride3<<#1)                //[1, 0]
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[0,10]same 2nd 32weights stream 1
       s14_s10.w += vrmpy(w11.b, x41_x01.ub)             //[0,10]
       s15_s11.w += vrmpy(w11.b, x51_x11.ub)             //[0,10]
       x60_x20 = memd(ptr_x1+stride<<#2)                 //[1, 1]
   }:endloop0
   {   dcfetch(fetch_ptr+#0<<6)                          //[0, 0]fetch 64bytes-2 lots 8 x 4 bytes
       p3 = sp1loop0(.L_filt_width, filt_width)          //set up inne rloop for next time
       p1 = cmp.gt(fetch_ptr_base, cbuf_eob)             //[E,10]
       if(p1.new)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size)//[E,10]wrap around end fetch ptr
   } {
       w00.tmp = vmem(ptr_w0+#0)                         //[1, 3]1st 32 weights of out depth
       s06_s02.w+= vrmpy(w00.b, x60_x20.ub)              //[1, 3]acc 2,3,6,7  out 0
       s07_s03.w+= vrmpy(w00.b, x70_x30.ub)              //[1, 3]
       x50_x10 = memd(ptr_x1+stride<<#1)                 //[1, 3]
   } {
       w10.tmp = vmem(ptr_w1+#0)                         //[1, 4]1st 32 weights stream 1
       s16_s12.w += vrmpy(w10.b, x60_x20.ub)             //[1, 4]acc 2,3,6,7 out 1
       s17_s13.w += vrmpy(w10.b, x70_x30.ub)             //[1, 4]
       x40_x00 = memd(ptr_x1++#1<<3)                     //[1, 4]
   } {
       w00.tmp = vmem(ptr_w0++#1)                        //[1, 5]same 1st 32weights stream 0
       s04_s00.w += vrmpy(w00.b, x40_x00.ub)             //[1, 5]acc 0,4,1,5 out 0
       s05_s01.w += vrmpy(w00.b, x50_x10.ub)             //[1, 5]
       x71_x31 = memd(ptr_x1+stride3<<#1)                //[1, 5]
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[1, 6]same 1st 32weight stream 1
       s14_s10.w += vrmpy(w10.b, x40_x00.ub)             //[1, 6]acc 0,4,1,5 stream 1
       s15_s11.w += vrmpy(w10.b, x50_x10.ub)             //[1, 6]
       x61_x21 = memd(ptr_x1+stride<<#2)                 //[1, 6]
   } {
       w01.tmp = vmem(ptr_w0+#0)                         //[1, 7]2nd 32weights stream 0
       s06_s02.w += vrmpy(w01.b, x61_x21.ub)             //[1, 7]acc 2,3,6,7
       s07_s03.w += vrmpy(w01.b, x71_x31.ub)             //[1, 7]
       x51_x11 = memd(ptr_x1+stride<<#1)                 //[1, 7]
   } {
       w11.tmp = vmem(ptr_w1+#0)                         //[1, 8]2nd 32weights of stream 1
       s16_s12.w += vrmpy(w11.b, x61_x21.ub)             //[1, 8]
       s17_s13.w += vrmpy(w11.b, x71_x31.ub)             //[1, 8]
       x41_x01 = memd(ptr_x1++#1<<3)                     //[1, 8]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[1, 9]same 2nd 32weights stream 0
       s04_s00.w += vrmpy(w01.b, x41_x01.ub)             //[1, 9]
       s05_s01.w += vrmpy(w01.b, x51_x11.ub)             //[1, 9]
       p0 = cmp.gt(ptr_x0, cbuf_eob)                     //[E,10]
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[1,10]same 2nd 32weights stream 1
       s14_s10.w += vrmpy(w11.b, x41_x01.ub)             //[1,10]
       s15_s11.w += vrmpy(w11.b, x51_x11.ub)             //[1,10]
       if(p0)ptr_x0 = sub(ptr_x0, cbuf_size)             //[E,10]wrap around end of buffer
   }:endloop1
.L_last1:
   {   p3 = sp1loop0(.L_filt_width1, filt_width)         //set up inne rloop for next time
       ptr_x1 = ptr_x0                                   //set up currne tinput ptr
       ptr_x0 = add(ptr_x0, in_width_32)                 //if >= buf_size -= buf_size
   } {
       x70_x30 = memd(ptr_x1+stride3<<#1)                //[0, 0]load pt 3 and 7
       p0 = cmp.gt(ptr_x0, cbuf_eob)                     //[E,10]
       if(p0.new)ptr_x0 = sub(ptr_x0, cbuf_size)         //[E,10]wrap around end of buffer
   } {
       x60_x20 = memd(ptr_x1+stride<<#2)                 //[0, 2]load pt 2 and 6
       fetch_ptr = addasl(ptr_x0, stride, #4)            //initial fetch ptr
   } 
   .balign 32
.L_filt_width1:
   {   dcfetch(fetch_ptr+#0<<6)                          //[0, 2]fetch 64bytes-2 lots 8 x 4 bytes
       fetch_ptr = add(fetch_ptr, #64)                   //[0, 2]inc fetch by 32/64 bytes (1 line)
   } {
       w00.tmp = vmem(ptr_w0+#0)                         //[0, 3]1st 32 weights of out depth
       s06_s02.w+= vrmpy(w00.b, x60_x20.ub)              //[0, 3]macc 2,6  out 0
       s07_s03.w+= vrmpy(w00.b, x70_x30.ub)              //[0, 3]macc 3,7 even row
       x50_x10 = memd(ptr_x1+stride<<#1)                 //[0, 3]load pt 1 5
   } {
       w10.tmp = vmem(ptr_w1+#0)                         //[0, 4]1st 32 weights stream 1
       s16_s12.w += vrmpy(w10.b, x60_x20.ub)             //[0, 4]acc 2,6 out 1
       s17_s13.w += vrmpy(w10.b, x70_x30.ub)             //[0, 4]acc 3,7 out 1
       x40_x00 = memd(ptr_x1++#1<<3)                     //[0, 4]load pts 0, 4
   } {
       w00.tmp = vmem(ptr_w0++#1)                        //[0, 5]same 1st 32weights stream 0
       s04_s00.w += vrmpy(w00.b, x40_x00.ub)             //[0, 5]acc 0,4,1,5 out 0
       s05_s01.w += vrmpy(w00.b, x50_x10.ub)             //[0, 5]
       x71_x31 = memd(ptr_x1+stride3<<#1)                //[0, 5]
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[0, 6]same 1st 32weight stream 1
       s14_s10.w += vrmpy(w10.b, x40_x00.ub)             //[0, 6]acc 0,4,1,5 stream 1
       s15_s11.w += vrmpy(w10.b, x50_x10.ub)             //[0, 6]
       x61_x21 = memd(ptr_x1+stride<<#2)                 //[0, 6]
   } {
       w01.tmp = vmem(ptr_w0+#0)                         //[0, 7]2nd 32weights stream 0
       s06_s02.w += vrmpy(w01.b, x61_x21.ub)             //[0, 7]acc 2,3,6,7
       s07_s03.w += vrmpy(w01.b, x71_x31.ub)             //[0, 7]
       x51_x11 = memd(ptr_x1+stride<<#1)                 //[0, 7]
   } {
       w11.tmp = vmem(ptr_w1+#0)                         //[0, 8]2nd 32weights of stream 1
       s16_s12.w += vrmpy(w11.b, x61_x21.ub)             //[0, 8]
       s17_s13.w += vrmpy(w11.b, x71_x31.ub)             //[0, 8]
       x41_x01 = memd(ptr_x1++#1<<3)                     //[0, 8]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[0, 9]same 2nd 32weights stream 0
       s04_s00.w += vrmpy(w01.b, x41_x01.ub)             //[0, 9]
       s05_s01.w += vrmpy(w01.b, x51_x11.ub)             //[0, 9]
       x70_x30 = memd(ptr_x1+stride3<<#1)                //[1, 0]
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[0,10]same 2nd 32weights stream 1
       s14_s10.w += vrmpy(w11.b, x41_x01.ub)             //[0,10]
       s15_s11.w += vrmpy(w11.b, x51_x11.ub)             //[0,10]
       x60_x20 = memd(ptr_x1+stride<<#2)                 //[1, 1]
   }:endloop0
   {   w00.tmp = vmem(ptr_w0+#0)                         //[1, 3]1st 32 weights of out depth
       s06_s02.w+= vrmpy(w00.b, x60_x20.ub)              //[1, 3]acc 2,3,6,7  out 0
       s07_s03.w+= vrmpy(w00.b, x70_x30.ub)              //[1, 3]
       x50_x10 = memd(ptr_x1+stride<<#1)                 //[1, 3]
   } {
       w10.tmp = vmem(ptr_w1+#0)                         //[1, 4]1st 32 weights stream 1
       s16_s12.w += vrmpy(w10.b, x60_x20.ub)             //[1, 4]acc 2,3,6,7 out 1
       s17_s13.w += vrmpy(w10.b, x70_x30.ub)             //[1, 4]
       x40_x00 = memd(ptr_x1++#1<<3)                     //[1, 4]
   } {
       w00.tmp = vmem(ptr_w0++#1)                        //[1, 5]same 1st 32weights stream 0
       s04_s00.w += vrmpy(w00.b, x40_x00.ub)             //[1, 5]acc 0,4,1,5 out 0
       s05_s01.w += vrmpy(w00.b, x50_x10.ub)             //[1, 5]
       x71_x31 = memd(ptr_x1+stride3<<#1)                //[1, 5]
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[1, 6]same 1st 32weight stream 1
       s14_s10.w += vrmpy(w10.b, x40_x00.ub)             //[1, 6]acc 0,4,1,5 stream 1
       s15_s11.w += vrmpy(w10.b, x50_x10.ub)             //[1, 6]
       x61_x21 = memd(ptr_x1+stride<<#2)                 //[1, 6]
   } {
       w01.tmp = vmem(ptr_w0+#0)                         //[1, 7]2nd 32weights stream 0
       s06_s02.w += vrmpy(w01.b, x61_x21.ub)             //[1, 7]acc 2,3,6,7
       s07_s03.w += vrmpy(w01.b, x71_x31.ub)             //[1, 7]
       x51_x11 = memd(ptr_x1+stride<<#1)                 //[1, 7]
   } {
       w11.tmp = vmem(ptr_w1+#0)                         //[1, 8]2nd 32weights of stream 1
       s16_s12.w += vrmpy(w11.b, x61_x21.ub)             //[1, 8]
       s17_s13.w += vrmpy(w11.b, x71_x31.ub)             //[1, 8]
       x41_x01 = memd(ptr_x1++#1<<3)                     //[1, 8]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[1, 9]same 2nd 32weights stream 0
       s04_s00.w += vrmpy(w01.b, x41_x01.ub)             //[1, 9]
       s05_s01.w += vrmpy(w01.b, x51_x11.ub)             //[1, 9]
       dcfetch(fetch_ptr+#0<<6)                          //[0, 0]fetch 64bytes-2 lots 8 x 4 bytes
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[1,10]same 2nd 32weights stream 1
       s14_s10.w += vrmpy(w11.b, x41_x01.ub)             //[1,10]
       s15_s11.w += vrmpy(w11.b, x51_x11.ub)             //[1,10]
   }
/* ------------------------------------------------------------------------ */
   {   zshift = memw(sp+#40<<2)                          //final shift 7 + 16
       mine.w = vmin(mine.w, s00.w)                      //min accumulation
       maxe.w = vmax(maxe.w, s00.w)                      //max accumulation
   } { 
       s0_sh.w = vasl(s00.w, zshift)                     //
       maxe.w = vmax(maxe.w, s01.w)                      //max accumulation
       mine.w = vmin(mine.w, s01.w)                      //min accumulation
       col_count = add(col_count, #-8)                   //decrement width count by 8
   } {
       s1_sh.w = vasl(s01.w, zshift)                     //
       maxe.w = vmax(maxe.w, s02.w)                      //max accumulation
       mine.w = vmin(mine.w, s02.w)                      //min accumulation
//     adjust = memw(sp+#23<<2)                          //
   } {
       s00.w = vmpye(s0_sh.w, vrecip.uh)
       maxe.w = vmax(maxe.w, s03.w)                      //max accumulation
       mine.w = vmin(mine.w, s03.w)                      //min accumulation
       p0 = cmp.ge(col_count, #0)                        //will be back at same place just advance
   } {
       s00.w += vmpyo(s0_sh.w, vrecip.h):SSR
       mine.w = vmin(mine.w, s04.w)                      //min accumulation
       maxe.w = vmax(maxe.w, s04.w)                      //max accumulation
//     ptr_x0 = sub(ptr_x0, adjust)                      //-=filt_height if stride_height > filt_height
   } {
       s01.w = vmpye(s1_sh.w, vrecip.uh)
       s0_sh.w = vasl(s02.w, zshift)                     //
       mine.w = vmin(mine.w, s05.w)                      //min accumulation
   } {
       s01.w += vmpyo(s1_sh.w, vrecip.h):SSR
       maxe.w = vmax(maxe.w, s05.w)                      //max accumulation
       mine.w = vmin(mine.w, s06.w)                      //min accumulation
   } {
       s02.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s03.w, zshift)                     //
       maxe.w = vmax(maxe.w, s06.w)                      //max accumulation
   } {
       s02.w += vmpyo(s0_sh.w, vrecip.h):SSR
       mine.w = vmin(mine.w, s07.w)                      //min accumulation
       maxe.w = vmax(maxe.w, s07.w)                      //max accumulation
   } {
       d010.h = vpacke(s01.w, s00.w)                     //pack high 16bits of accs 
       s03.w = vmpye(s1_sh.w, vrecip.uh)
       s0_sh.w = vasl(s04.w, zshift)                     //
   } {
       s03.w += vmpyo(s1_sh.w, vrecip.h):SSR
       maxe.w = vmax(maxe.w, s10.w)                      //max accumulation
       mine.w = vmin(mine.w, s10.w)                      //min accumulation
       ptr_x0 += mpyi(stride, #16)                       //stride*2*4 advance buffer by 8 outputs
   } {
       s04.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s05.w, zshift)                     //
       maxe.w = vmax(maxe.w, s11.w)                      //max accumulation
   } {
       d032.h = vpacke(s03.w, s02.w)                     //pack high 16bits of accs 
       s04.w += vmpyo(s0_sh.w, vrecip.h):SSR
       mine.w = vmin(mine.w, s11.w)                      //min accumulation
   } {
       s05.w = vmpye(s1_sh.w, vrecip.uh)
       s0_sh.w = vasl(s06.w, zshift)                     //
       mine.w = vmin(mine.w, s12.w)                      //min accumulation
   } {
       maxe.w = vmax(maxe.w, s12.w)                      //max accumulation
       s05.w += vmpyo(s1_sh.w, vrecip.h):SSR
       d03210.ub = vpack(d032.h, d010.h):sat             //shift 16bits by zshift
       vmem(ptr_z0++#1) = d03210.new                     //store 0-3 even row
   } {
       s06.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s07.w, zshift)                     //
       mine.w = vmin(mine.w, s13.w)                      //min accumulation
   } {
       s06.w += vmpyo(s0_sh.w, vrecip.h):SSR
       d054.h = vpacke(s05.w, s04.w)                     //pack high 16bits of accs 
       maxe.w = vmax(maxe.w, s13.w)                      //max accumulation
   } {
       s07.w = vmpye(s1_sh.w, vrecip.uh)
       mine.w = vmin(mine.w, s14.w)                      //min accumulation
       s0_sh.w = vasl(s10.w, zshift)                     //
   } {
       s07.w += vmpyo(s1_sh.w, vrecip.h):SSR
       maxe.w = vmax(maxe.w, s14.w)                      //max accumulation
       mine.w = vmin(mine.w, s15.w)                      //min accumulation
   } {
       s10.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s11.w, zshift)                     //
       maxe.w = vmax(maxe.w, s15.w)                      //max accumulation
   } {
       d076.h = vpacke(s07.w, s06.w)                     //pack high 16bits of accs 
       s10.w += vmpyo(s0_sh.w, vrecip.h):SSR
       mine.w = vmin(mine.w, s16.w)                      //min accumulation
   } {
       s11.w = vmpye(s1_sh.w, vrecip.uh)
       s0_sh.w = vasl(s12.w, zshift)                     //
       maxe.w = vmax(maxe.w, s16.w)                      //max accumulation
   } {
       s11.w += vmpyo(s1_sh.w, vrecip.h):SSR
       maxe.w = vmax(maxe.w, s17.w)                      //max accumulation
       d07654.ub = vpack(d076.h, d054.h):sat             //shift 16bits by zshift
       if(p0) vmem(ptr_z0++#1):nt = d07654.new           //store 4-7 even row
   } {
       s12.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s13.w, zshift)                     //
       mine.w = vmin(mine.w, s17.w)                      //min accumulation
   } {
       d110.h = vpacke(s11.w, s10.w)                     //pack high 16bits of accs 
       s12.w += vmpyo(s0_sh.w, vrecip.h):SSR
   } {
       s13.w = vmpye(s1_sh.w, vrecip.uh)
       s0_sh.w = vasl(s14.w, zshift)                     //
   } {
       loop1(.L_filt_height, filt_height)                //setup vertical filte rloop
       s13.w += vmpyo(s1_sh.w, vrecip.h):SSR
       p1 = cmp.gt(col_count, #0)
   } {
       s14.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s15.w, zshift)                     //
       p3 = sp1loop0(.L_filt_width, filt_width)          //setup inner filter loop
   } {
       d132.h = vpacke(s13.w, s12.w)                     //pack high 16bits of accs 
       s14.w += vmpyo(s0_sh.w, vrecip.h):SSR
   } {
       s15.w = vmpye(s1_sh.w, vrecip.uh)
       s0_sh.w = vasl(s16.w, zshift)                     //
   } {
       s15.w += vmpyo(s1_sh.w, vrecip.h):SSR
       d13210.ub = vpack(d132.h, d110.h):sat             //shift 16bits by zshift
       vmem(ptr_z1++#1):nt = d13210.new                  //store 0-3 odd row
   } {
       s16.w = vmpye(s0_sh.w, vrecip.uh)
       s1_sh.w = vasl(s17.w, zshift)                     //
   } {
       s16.w += vmpyo(s0_sh.w, vrecip.h):SSR
       d154.h = vpacke(s15.w, s14.w)                     //pack high 16bits of accs 
       fetch_ptr_base = add(ptr_x0, in_width_32)         //fetch is next row ahead
   } {
       s17.w = vmpye(s1_sh.w, vrecip.uh)
       s04_s00 = vcombine(wsum0,wsum0)                   //init sum0 and 4
       p2 = cmp.gt(fetch_ptr_base, cbuf_eob)             //[E,10]
       if(p2.new)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size)//[E,10]wrap around end fetch ptr
   } {
       s17.w += vmpyo(s1_sh.w, vrecip.h):SSR
       s05_s01 = vcombine(wsum0,wsum0)                   //init sum1 and 5
   } {
       s06_s02 = vcombine(wsum0,wsum0)                   //init sum2 and 6
       s07_s03 = vcombine(wsum0,wsum0)                   //init sum3 and 7
       ptr_w0 = memw(sp+#13<<2)                          //access ptr weight
   } {
       d176.h = vpacke(s17.w, s16.w)                     //pack high 16bits of accs 
       s14_s10 = vcombine(wsum1,wsum1)                   //init sum 0 and 4
       ptr_w1 = memw(sp+#20<<2)                          //access weights stride
   } {
       s15_s11 = vcombine(wsum1,wsum1)                   //init sum 1 and 5
       s16_s12 = vcombine(wsum1,wsum1)                   //init sum 2 and 6
       weight_stride = memw(sp+#24<<2)                   //
   } {
       s17_s13 = vcombine(wsum1,wsum1)                   //init sum 3 and 7
       d17654.ub = vpack(d176.h, d154.h):sat             //shift 16bits by zshift
       if( p0) vmem(ptr_z1++#1):nt = d17654.new          //store 4-7 odd row
       if( p1) jump   .L_width                           //next 2 rows 8 points per row
   }//endloop width
/* --------------------------------------------------------------------------- */
   {   out_height = add(out_height, #-1)                 //decrement height count
       col_count = memw(sp+#15<<2)                       //initialize width count
       memw(sp+#13<<2) += weight_stride                  //access ptr weight
   } {
       memw(sp+#20<<2) += weight_stride                  //access weights stride
       p0 = cmp.eq(out_height, #0)                       //are vertical lines done?
       if(!p0.new) jump:t .L_height                            //then go again
   }
/* ------------------------------------------------------------------------ */
#if 0
.L_domax:
   {   ptr_max = memw(sp+#36<<2)                         //get max/min ptr
       cm4 = #-4                                         //define int based deal
   } {
       loop0(.L_peak, #4)                                //set up vec reduce
       maxo_maxe = vdeal(maxe, maxe, cm4)                //deal out odd and even
   }
.L_peak:
   {   maxe.w = vmax(maxe.w, maxo.w)                     //reduce
       mino_mine = vdeal(mine, mine, cm4)                //split out and and even min
   } {
       mine.w = vmin(mine.w, mino.w)                     //reduce mins by 2
   } {
       maxo_maxe = vdeal(maxe, maxe, cm4)                //split out odd and even max
   }:endloop0
   {   maxe.w = vmax(maxo.w, maxe.w)                     //reduce max
       vmem(ptr_max+#0) = maxe.new                       //store max
       mino_mine = vdeal(mine, mine, cm4)                //split out mins
   } {
       mine.w = vmin(mino.w, mine.w)                     //reduce mins to final 1
       vmem(ptr_max+#1) = mine.new                       //store min
   }
/* ------------------------------------------------------------------------ */
   {   r17:16 = memd(sp+#0)                              //restore stack
       r19:18 = memd(sp+#2<<2)                           //18,19
#else
   {   vmem(ptr_max+#0) = maxe                           //store max
       r17:16 = memd(sp+#0)                              //restore stack
   } {
       r19:18 = memd(sp+#2<<2)                           //18,19
       vmem(ptr_max+#1) = mine                           //store min
#endif
   } {
       r21:20 = memd(sp+#4<<2)                           //20,21
       r23:22 = memd(sp+#6<<2)                           //22,23
   } {
       r25:24 = memd(sp+#8<<2)                           //24,25
       r27:26 = memd(sp+#10<<2)                          //26,27
   } {
       dealloc_return                                    //
   }
/* ------------------------------------------------------------------------ */
.L_end:
/* ======================================================================== */
      .size gvconv2dbbb_circ_d64_v65_asm, .L_end-gvconv2dbbb_circ_d64_v65_asm
