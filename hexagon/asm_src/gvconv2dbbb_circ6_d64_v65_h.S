/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
        .file "gvconv2dbbb_circ6_d64_v65_h.S"
        .global gvconv2dbbb_circ6_d64_v65_asm
        .balign 32
        .type  gvconv2dbbb_circ6_d64_v65_asm, @function
gvconv2dbbb_circ6_d64_v65_asm:
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
#define out_width_32           r7     //38 actual out_width in depth32
#define ptr_cbufi              r16    //39 read buffer pointer
#define zshift                 r6     //40 extra shift on output before quantization
#define in_zero                r25    //41
#define store_cntrl            r11    //42   
//#define ptr_equalize           r17    //43
#define cbuf_eob               r27    //18 end of cuirc buffer
#define cbuf_size              r28    //19 size in bytes of circ buf -1
#define weight_stride          r15    //20 distance to next set of weights
/* --------------------------------- SCALER REGS ------------------------------- */
#define cm4                    r2     //shuffle/deal ints
#define col_count              r2     //horizontal counter
#define in_width_32            r3     //total input width in bytes in buffer
#define x50_x20                r17:16 //1-----1
#define x40_x10                r19:18 //111----
#define x30_x00                r17:16 //-111---
#define x51_x21                r15:14 //--11---
#define x41_x11                r19:18 //---1111
#define x31_x01                r15:14 //----111
#define ptr_wi_ptr_xi          r1:0   //
#define fetch_ptr_base         r1     //base pointer for l1 prefetch
#define fetch_ptr              r10    //current pointer for l1 prefetch
#define stride                 r12    //current to next input
#define ptr_x0                 r26    //base input pointer
#define ptr_x1                 r13    //current input ptr      
#define ptr_w0                 r20    //even output depth 32 weights
#define ptr_w1                 r21    //odd output depth 32 weights
#define ptr_z0                 r0     //even output depth 32 outputs
#define ptr_z1                 r25    //21 write buffer sp position and odd output depth 32 outputs
#define out_width_64           r22    //
#define adjust                 r10    //
#define delta                  r7     //difference between filt height and stride height
#define align                  r11    //temp alignment of output
#define vplut                  r18    //vector pred lookup
#define ones                   r19    //0x01010101
/* ---------------------------------- VEC REGS -------------------------------- */
#define tmax                   v18
#define tmin                   v19
//#define wscale0                v0     //
#define wscale1                v23    //
#define vin_zero               v9     //
#define vpred0                 v24    //
#define vpred1                 v25    //
#define vpred2                 v31    //
#define s15_s12                v23:22 //odd  output accs 2,6
#define s14_s11                v21:20 //odd  output accs 1,5
#define s13_s10                v19:18 //odd  output accs 0,4
#define s05_s02                v15:14 //even output accs 2,6
#define s04_s01                v13:12 //even output accs 1,5
#define s03_s00                v11:10 //even output accs 0,4
#define s15                    v23    //odd acc 5
#define s12                    v22    //odd acc 2
#define s14                    v21    //odd acc 4
#define s11                    v20    //odd acc 1
#define s13                    v19    //odd acc 3
#define s10                    v18    //odd acc 0
#define s05                    v15    //even acc 5
#define s02                    v14    //even acc 2
#define s04                    v13    //even acc 4
#define s01                    v12    //even acc 1
#define s03                    v11    //even acc 3
#define s00                    v10    //even acc 0
#define w00                    v27    //weights even 0-31
// note w01,w10,w11 are only used as .tmp destinations.
#define w01                    v0     //weights even 32-63
#define w10                    v0     //weights odd 0-31
#define w11                    v0     //weights odd 32-63
#define vrecip0                v1     //reciprocal 255/MAx replicated
#define vrecip1                v30    //reciprocal 255/MAx replicated
#define s0_sh                  v8     //shifted value
#define s1_sh                  v26    //shifted value
#define wsum0                  v8  //2 //sum of weights column + bias add 0-31
#define wsum1                  v26 //3 //sum of weights column + bias add 32-63
#define d010                   v27    //even lines upper 16bit packed accs 0,1
#define d032                   v28    //even lines upper 16bit packed accs 2,3
#define d03210                 v28    //8bit shifted, packed saturated 0-3
#define d054                   v29    //even lines upper 16bit packed accs 4,5
#define d110                   v27    //odd lines upper 16bit packed accs 0,1
#define d132                   v28    //odd lines upper 16bit packed accs 2,3
#define d13210                 v28    //8bit shifted, packed saturated 0-3
#define d154                   v29    //odd lines upper 16bit packed accs 4,5
#define maxo_maxe              v5:4   //packed maxes
#define maxo                   v5     //odd maxes
#define maxe                   v4     //even maxes
#define mino_mine              v7:6   //packed mins 
#define mino                   v7     //odd mins
#define mine                   v6     //even mins
#define gmax                   v2     //odd mins
#define gmin                   v3     //even mins
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/* --------------------------------------------------------------------------- */
   {   allocframe(#112)                                  //0th entry on stack is 112+8)/4=30 ints
       stride_h_w = memw(sp+#0<<2)                       //stride horizontl and vertical
   } {
       memd(sp+#4<<2) = r21:20                           //save 20,21
       memd(sp+#6<<2) = r23:22                           //save 22,23
       r23 = #0x80000001
   } {
       memd(sp+#0<<2)  = r17:16                          //save 16,17
       memd(sp+#2<<2)  = r19:18                          //save 18,19
       maxe = vsplat(r23)                                // maxe <- -0x7fffffff
   } {
       memd(sp+#8<<2) = r25:24                           //save 24,25
       memd(sp+#10<<2) = r27:26                          //save 26,27
       mine.w = vabs(maxe.w)                             // mine <- +0x7fffffff
   } {
       memd(sp+#12<<2) = ptr_wi_ptr_xi                   //save weights: activation 
       memw(sp+#14<<2) = ptr_zi                          //save output ptr
       maxo = maxe                                       //
       mino = mine
   } {
       filt_height = memw(sp+#33<<2)                     //filter height
       memw(sp+#15<<2) = out_width                       //save output width
       stride = zxth(stride_h_w)                         //horizontal stride
       //nop                                               //
   } {
       ptr_max = memw(sp+#36<<2)                         //get max/min ptr
       in_depth = memw(sp+#31<<2)                        //input depth
   } {
       filt_width = memw(sp+#32<<2)                      //filter width
       out_height = memw(sp+#34<<2)                      //height of output
       stride = asl(stride, #5)                          //32 * stride_w
   } {
       cbuf_size = mpyi(filt_height, in_depth)           //circular buffer size
       gmax = vmem(ptr_max+#0)                           //
       in_zero = memw(sp+#41<<2)                         //
   } {
       gmin = vmem(ptr_max+#1)                       //
       store_cntrl = memw(sp+#42<<2)                     //
   } {
       in_zero = vsplatb(in_zero)                        //
       cbuf_size = mpyi(cbuf_size, next_inbuf_width)     //circular buffer size
       ptr_cbufi = memw(sp+#39<<2)                       //circular buffer
       dcfetch(ptr_xi+#0<<6)
   } {
       cbuf_size = add(cbuf_size, cbuf_size)             //x2
       vin_zero = vsplat(in_zero)                        //
       recip_level = memw(sp+#37<<2)                     //255/max
       dcfetch(ptr_xi+#1<<6)
   } {
       weight_stride=mpy(filt_width.L,filt_height.L)     //offset between filter rows
       memw(sp+#21<<2) = ptr_cbufi                       //cbuf write ptr
       out_width_32 = memw(sp+#38<<2)                    //total width of output
       filt_width = asl(filt_width, #2)                  //*32/8
   } {
       weight_stride=mpy(weight_stride.L,in_depth.L)     //distance between weight rows
       vrecip0 = vmem(recip_level++#2)                   //used to compress to 8bits 255/max
       filt_width = add(filt_width, #-1)                 //account for epilog
       dcfetch(ptr_xi+#2<<6)
   } {
       ptr_wi += asl(weight_stride,#5)                   //weights stride
       cbuf_eob = add(ptr_cbufi, cbuf_size)              //end of circ buffer marker
       vpred1 = vmem(store_cntrl+#1)                     //
       dcfetch(ptr_xi+#3<<6)
   } {
       filt_height = mpyi(filt_height, in_depth)         //total number of depth32 filter rows
       vpred2 = vmem(store_cntrl+#2)                     //
       weight_stride = asl(weight_stride, #6)            //
       cbuf_eob = add(cbuf_eob, #-4)                     //make so comparison is >= eob
   } {
       filt_height = lsr(filt_height, #5)                //num d32 rows in filter
       out_width_64 = add(out_width_32, out_width_32)    //
       memw(sp+#20<<2) = ptr_wi                          //spill weight stride for later
       memw(sp+#24<<2) = weight_stride                   //
   } {
       in_width_32 = asl(next_inbuf_width, #6)           //next d32 line x 2
       filt_height = add(filt_height, #-1)
       ptr_filtsum = memw(sp+#35<<2)                     //ptr to the sum of filters+offset
       col_count = memw(sp+#15<<2)                       //initialize width count
   } {
       vrecip1 = vmem(recip_level+#-1)
       memw(sp+#37<<2) = recip_level                     //255/max
   }
/* -------------------------------------------------------------------------- */
   .balign 32
.L_height:
   {   wsum0 = vmem(ptr_filtsum+#0)                      //set 1st weight offset
       ptr_x0 = memw(sp+#12<<2)                          //ptr_x0=ptr_cbufi read circ buffer
       loop1(.L_filt_height, filt_height)                //setup vertical filte rloop
       out_height = add(out_height, #-1)                 //decrement height count
   } {                                                   //buffer read ptr if ptr_xi >= buf_size-=size
       wsum1 = vmem(ptr_filtsum+#1)                      //set 2nd weight offset
       fetch_ptr_base = add(ptr_x0, in_width_32)         //fetch is next row ahead
       ptr_z0 = memw(sp+#14<<2)                          //output ptr for even lines
       p3 = sp1loop0(.L_filt_width, filt_width)          //set up inne rloop for next time
   } {
       ptr_z1 = add(ptr_z0, out_width_32)                //
       s03_s00 = vcombine(wsum0,wsum0)                   //init sum0 and 4
       s04_s01 = vcombine(wsum0,wsum0)                   //init sum1 and 5
       p1 = cmp.gt(fetch_ptr_base, cbuf_eob)             //if prefetch >= circ buffer wrap around
   } {
       s15_s12 = vcombine(wsum1,wsum1)                   //init sum 2 and 6
       s05_s02 = vcombine(wsum0,wsum0)                   //init sum2 and 6
       ptr_w0 = memw(sp+#13<<2)                          //access ptr weight
       memw(sp+#14<<2) += out_width_64                   //update output ptr to next out_width_32
   } {
       s13_s10 = vcombine(wsum1,wsum1)                   //init sum 0 and 4
       s14_s11 = vcombine(wsum1,wsum1)                   //init sum 1 and 5
       ptr_w1 = memw(sp+#20<<2)                          //access weights stride
       if(p1)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size)//wrap fetch ptr around independently
   }
.L_width:
   {   x50_x20 = memd(ptr_x0+stride<<#2)                 //[0, 0]load pt 2 and 6
       w00 = vmem(ptr_w0++#1)                            //[0, 0]1st 32 weights of out depth
       p0 = cmp.eq(filt_height, #0) 
       if(p0.new) jump:nt .L_last1
   }
   .balign 32
.L_filt_height:
   {   ptr_x1 = ptr_x0                                   //set up currne tinput ptr
       ptr_x0 = add(ptr_x0, in_width_32)                 //if >= buf_size -= buf_size
       fetch_ptr_base=add(fetch_ptr_base,in_width_32)    //if >= buf_size -= buf_size
       x40_x10 = memd(ptr_x0+stride<<#1)                 //[0, 1]load pt 1 5
   } { 
       dcfetch(fetch_ptr_base+#0<<6)                     //[0, 6]fetch 64bytes-2 lots 8 x 4 bytes
       fetch_ptr = add(fetch_ptr_base, #64)              //initial fetch ptr
       p0 = cmp.gt(ptr_x0, cbuf_eob)                     //[E,10]
       if(p0.new)ptr_x0 = sub(ptr_x0, cbuf_size)         //[E,10]wrap around end of buffer
   }
   .balign 32
.L_filt_width:
   {   s05_s02.w += vrmpy(w00.b, x50_x20.ub)             //[0, 2]macc 2,6  out 0
       w10.tmp = vmem(ptr_w1+#0)                         //[0, 2]1st 32 weights stream 1
       s15_s12.w += vrmpy(w10.b, x50_x20.ub)             //[0, 2]acc 2,6 out 1
       x30_x00 = memd(ptr_x1++#1<<3)                     //[0, 2]load pts 0, 4
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[0, 3]same 1st 32weight stream 1
       s13_s10.w += vrmpy(w10.b, x30_x00.ub)             //[0, 3]acc 0,4,1,5 stream 1
       s14_s11.w += vrmpy(w10.b, x40_x10.ub)             //[0, 3]
       x51_x21 = memd(ptr_x1+stride<<#2)                 //[0, 3]
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[0, 4]
       w01.tmp = vmem(ptr_w0+#0)                         //[0, 4]2nd 32weights stream 0
       s05_s02.w += vrmpy(w01.b, x51_x21.ub)             //[0, 4]acc 2,3,6,7
       x41_x11 = memd(ptr_x1+stride<<#1)                 //[0, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[0, 5]acc 0,4,1,5 out 0
       w11.tmp = vmem(ptr_w1+#0)                         //[0, 5]2nd 32weights of stream 1
       s15_s12.w += vrmpy(w11.b, x51_x21.ub)             //[0, 5]
       x31_x01 = memd(ptr_x1++#1<<3)                     //[0, 5]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[0, 6]same 2nd 32weights stream 0
       s03_s00.w += vrmpy(w01.b, x31_x01.ub)             //[0, 6]
       s04_s01.w += vrmpy(w01.b, x41_x11.ub)             //[0, 6]
       dcfetch(fetch_ptr+#0<<6)                          //[0, 6]fetch 64bytes-2 lots 8 x 4 bytes
   } {
       fetch_ptr = add(fetch_ptr, #64)                   //[0, 7]inc fetch by 32/64 bytes (1 line)
       x50_x20 = memd(ptr_x1+stride<<#2)                 //[1, 0]load pt 2 and 6
       w00 = vmem(ptr_w0++#1)                            //[1, 0]1st 32 weights of out depth
       nop                                               //
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[0, 8]same 2nd 32weights stream 1
       s13_s10.w += vrmpy(w11.b, x31_x01.ub)             //[0, 8]
       s14_s11.w += vrmpy(w11.b, x41_x11.ub)             //[0, 8]
       x40_x10 = memd(ptr_x1+stride<<#1)                 //[1, 1]load pt 1 5
   }:endloop0
   {   s05_s02.w+= vrmpy(w00.b, x50_x20.ub)              //[1, 2]macc 2,6  out 0
       w10.tmp = vmem(ptr_w1+#0)                         //[1, 2]1st 32 weights stream 1
       s15_s12.w += vrmpy(w10.b, x50_x20.ub)             //[1, 2]acc 2,6 out 1
       x30_x00 = memd(ptr_x1++#1<<3)                     //[1, 2]load pts 0, 4
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[1, 3]same 1st 32weight stream 1
       s13_s10.w += vrmpy(w10.b, x30_x00.ub)             //[1, 3]acc 0,4,1,5 stream 1
       s14_s11.w += vrmpy(w10.b, x40_x10.ub)             //[1, 3]
       x51_x21 = memd(ptr_x1+stride<<#2)                 //[1, 3]
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[1, 4]
       w01.tmp = vmem(ptr_w0+#0)                         //[1, 4]2nd 32weights stream 0
       s05_s02.w += vrmpy(w01.b, x51_x21.ub)             //[1, 4]acc 2,3,6,7
       x41_x11 = memd(ptr_x1+stride<<#1)                 //[1, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[1, 5]acc 0,4,1,5 out 0
       w11.tmp = vmem(ptr_w1+#0)                         //[1, 5]2nd 32weights of stream 1
       s15_s12.w += vrmpy(w11.b, x51_x21.ub)             //[1, 5]
       x31_x01 = memd(ptr_x1++#1<<3)                     //[1, 5]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[1, 6]same 2nd 32weights stream 0
       s03_s00.w += vrmpy(w01.b, x31_x01.ub)             //[1, 6]
       s04_s01.w += vrmpy(w01.b, x41_x11.ub)             //[1, 6]
       //dcfetch(fetch_ptr+#0<<6)                          //[1, 6]fetch 64bytes-2 lots 8 x 4 bytes
   } {
       w00 = vmem(ptr_w0++#1)                            //[E, 0]1st 32 weights of out depth
       x50_x20 = memd(ptr_x0+stride<<#2)                 //[E, 0]load pt 2 and 6
       p1 = cmp.gt(fetch_ptr_base, cbuf_eob)             //[E,10]
       p3 = sp1loop0(.L_filt_width, filt_width)          //set up inne rloop for next time
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[1, 8]same 2nd 32weights stream 1
       s13_s10.w += vrmpy(w11.b, x31_x01.ub)             //[1, 8]
       s14_s11.w += vrmpy(w11.b, x41_x11.ub)             //[1, 8]
       if(p1)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size)//[E,10]wrap around end fetch ptr
   }:endloop1
.L_last1:
   {   ptr_x1 = ptr_x0                                   //set up currne tinput ptr
       ptr_x0 = add(ptr_x0, in_width_32)                 //if >= buf_size -= buf_size
   } {
       p3 = sp1loop0(.L_filt_width1, filt_width)         //set up inne rloop for next time
       p0 = cmp.gt(ptr_x0, cbuf_eob)                     //[E,10]
       if(p0.new)ptr_x0 = sub(ptr_x0, cbuf_size)         //[E,10]wrap around end of buffer
   } {
       fetch_ptr = addasl(ptr_x0, stride, #4)            //initial fetch ptr
       x40_x10 = memd(ptr_x1+stride<<#1)                 //[0, 1]load pt 1 5
       nop; nop
   }
   .balign 32
.L_filt_width1:
   {   s05_s02.w+= vrmpy(w00.b, x50_x20.ub)              //[0, 2]macc 2,6  out 0
       w10.tmp = vmem(ptr_w1+#0)                         //[0, 2]1st 32 weights stream 1
       s15_s12.w += vrmpy(w10.b, x50_x20.ub)             //[0, 2]acc 2,6 out 1
       x30_x00 = memd(ptr_x1++#1<<3)                     //[0, 2]load pts 0, 4
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[0, 3]same 1st 32weight stream 1
       s13_s10.w += vrmpy(w10.b, x30_x00.ub)             //[0, 3]acc 0,4,1,5 stream 1
       s14_s11.w += vrmpy(w10.b, x40_x10.ub)             //[0, 3]
       x51_x21 = memd(ptr_x1+stride<<#2)                 //[0, 3]
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[0, 4]
       w01.tmp = vmem(ptr_w0+#0)                         //[0, 4]2nd 32weights stream 0
       s05_s02.w += vrmpy(w01.b, x51_x21.ub)             //[0, 4]acc 2,3,6,7
       x41_x11 = memd(ptr_x1+stride<<#1)                 //[0, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[0, 5]acc 0,4,1,5 out 0
       w11.tmp = vmem(ptr_w1+#0)                         //[0, 5]2nd 32weights of stream 1
       s15_s12.w += vrmpy(w11.b, x51_x21.ub)             //[0, 5]
       x31_x01 = memd(ptr_x1++#1<<3)                     //[0, 5]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[0, 6]same 2nd 32weights stream 0
       s03_s00.w += vrmpy(w01.b, x31_x01.ub)             //[0, 6]
       s04_s01.w += vrmpy(w01.b, x41_x11.ub)             //[0, 6]
       dcfetch(fetch_ptr+#0<<6)                          //[0, 6]fetch 64bytes-2 lots 8 x 4 bytes
   } {
       fetch_ptr = add(fetch_ptr, #64)                   //[1, 7]inc fetch by 32/64 bytes (1 line)
       x50_x20 = memd(ptr_x1+stride<<#2)                 //[0, 0]load pt 2 and 6
       w00 = vmem(ptr_w0++#1)                            //[0, 0]1st 32 weights of out depth
   } {
       x40_x10 = memd(ptr_x1+stride<<#1)                 //[1, 1]load pt 1 5
       w11.tmp = vmem(ptr_w1++#1)                        //[0, 7]same 2nd 32weights stream 1
       s13_s10.w += vrmpy(w11.b, x31_x01.ub)             //[0, 7]
       s14_s11.w += vrmpy(w11.b, x41_x11.ub)             //[0, 7]
   }:endloop0
   {   s05_s02.w+= vrmpy(w00.b, x50_x20.ub)              //[1, 2]macc 2,6  out 0
       w10.tmp = vmem(ptr_w1+#0)                         //[1, 2]1st 32 weights stream 1
       s15_s12.w += vrmpy(w10.b, x50_x20.ub)             //[1, 2]acc 2,6 out 1
       x30_x00 = memd(ptr_x1++#1<<3)                     //[1, 2]load pts 0, 4
   } {
       w10.tmp = vmem(ptr_w1++#1)                        //[1, 3]same 1st 32weight stream 1
       s13_s10.w += vrmpy(w10.b, x30_x00.ub)             //[1, 3]acc 0,4,1,5 stream 1
       s14_s11.w += vrmpy(w10.b, x40_x10.ub)             //[1, 3]
       x51_x21 = memd(ptr_x1+stride<<#2)                 //[1, 3]
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[1, 4]
       w01.tmp = vmem(ptr_w0+#0)                         //[1, 4]2nd 32weights stream 0
       s05_s02.w += vrmpy(w01.b, x51_x21.ub)             //[1, 4]acc 2,3,6,7
       x41_x11 = memd(ptr_x1+stride<<#1)                 //[1, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[1, 5]acc 0,4,1,5 out 0
       w11.tmp = vmem(ptr_w1+#0)                         //[1, 5]2nd 32weights of stream 1
       s15_s12.w += vrmpy(w11.b, x51_x21.ub)             //[1, 5]
       x31_x01 = memd(ptr_x1++#1<<3)                     //[1, 5]
   } {
       w01.tmp = vmem(ptr_w0++#1)                        //[1, 6]same 2nd 32weights stream 0
       s03_s00.w += vrmpy(w01.b, x31_x01.ub)             //[1, 6]
       s04_s01.w += vrmpy(w01.b, x41_x11.ub)             //[1, 6]
       dcfetch(fetch_ptr+#0<<6)                          //[1, 6]fetch 64bytes-2 lots 8 x 4 bytes
   } {
       w11.tmp = vmem(ptr_w1++#1)                        //[1, 7]same 2nd 32weights stream 1
       s13_s10.w += vrmpy(w11.b, x31_x01.ub)             //[1, 7]
       s14_s11.w += vrmpy(w11.b, x41_x11.ub)             //[1, 7]
       p2 = cmp.ge(col_count,#6)                         // Do we have a full 6 cols?
   }
/* ------------------------------------------------------------------------ */
//     q2   q1   q0
// 0      --54 3210
//   0000 0011 1111                 3210 = if(q0) vmem =  3210   --54 =  vmux(q0, --54,3210)  if(q1) vmem = --54  if(q2) vmem = ----
// 1      -54- 2103
//   0000 0111 1110                 210- = if(q0) vmem =  2103   2543 =  vmux(q0, -54-,2103)  if(q1) vmem = 2543  if(q2) vmem = ----
// 2      54-- 1032
//   0000 1111 1100                 10-- = if(q0) vmem =  1032   5432 =  vmux(q0, 54--,1032)  if(q1) vmem = 5432  if(q2) vmem = ----
// 3      4--5 0321
//   0001 1111 1000                 0--- = if(q0) vmem =  0321   4321 =  vmux(q0, 4--5,0321)  if(q1) vmem = 4321  if(q2) vmem = ---5
/* ------------------------------------------------------------------------ */
   {   zshift = memw(sp+#40<<2)                          //final shift 7 + 16
       mine.w = vmin(mine.w, s00.w)                      //min accumulation
       maxe.w = vmax(maxe.w, s00.w)                      //max accumulation
       if (!p2) s05 = s00                                //
   } { 
       s0_sh.w = vasl(s00.w, zshift)                     //
       maxe.w = vmax(maxe.w, s01.w)                      //max accumulation
       mine.w = vmin(mine.w, s01.w)                      //min accumulation
       if (!p2) s15 = s00                                //
   } {
       s1_sh.w = vasl(s01.w, zshift)                     //
       maxe.w = vmax(maxe.w, s02.w)                      //max accumulation
       mine.w = vmin(mine.w, s02.w)                      //min accumulation
//     adjust = memw(sp+#23<<2)                          //
   } {
       s00.w = vmpye(s0_sh.w, vrecip0.uh)                 //
       maxe.w = vmax(maxe.w, s03.w)                      //max accumulation
       mine.w = vmin(mine.w, s03.w)                      //min accumulation
       col_count = add(col_count, #-6)                   //decrement width count by 8
   } {
       s00.w += vmpyo(s0_sh.w, vrecip0.h):SSR             //
       mine.w = vmin(mine.w, s04.w)                      //min accumulation
       maxe.w = vmax(maxe.w, s04.w)                      //max accumulation
//     ptr_x0 = sub(ptr_x0, adjust)                      //-=filt_height if stride_height > filt_height
   } {
       s01.w = vmpye(s1_sh.w, vrecip0.uh)                 //
       s0_sh.w = vasl(s02.w, zshift)                     //
       mine.w = vmin(mine.w, s05.w)                      //min accumulation
   } {
       s01.w += vmpyo(s1_sh.w, vrecip0.h):SSR             //
       maxe.w = vmax(maxe.w, s05.w)                      //max accumulation
       mino.w = vmin(mino.w, s11.w)                      //min accumulation
       ones.L = #0x0101                                  //
   } {
       s02.w = vmpye(s0_sh.w, vrecip0.uh)                 //
       s1_sh.w = vasl(s03.w, zshift)                     //
       mino.w = vmin(mino.w, s13.w)                      //min accumulation
       ones.H = #0x0101                                  //
   } {
       s02.w += vmpyo(s0_sh.w, vrecip0.h):SSR             //
       mino.w = vmin(mino.w, s10.w)                      //min accumulation
       maxo.w = vmax(maxo.w, s11.w)                      //max accumulation
       vplut = extractu(ptr_z0, #2, #5)                  //1100000 ->  0x08080808
   } {
       d010.h = vpack(s01.w, s00.w):sat                  //pack high 16bits of accs
       s03.w = vmpye(s1_sh.w, vrecip0.uh)                 //
       s0_sh.w = vasl(s04.w, zshift)                     //
       align = sub(#128, ptr_z0)                         //
   } {
       s03.w += vmpyo(s1_sh.w, vrecip0.h):SSR             //
       maxo.w = vmax(maxo.w, s10.w)                      //max accumulation
       mino.w = vmin(mino.w, s15.w)                      //min accumulation
       ptr_x0 += mpyi(stride, #12)                       //stride*2*3 advance buffer by 6 outputs
   } {
       s04.w = vmpye(s0_sh.w, vrecip0.uh)                 //
       s1_sh.w = vasl(s05.w, zshift)                     //
       mino.w = vmin(mino.w, s14.w)                      //min accumulation
       vplut = asl(ones, vplut)                          // 3,2,1,0
   } {
       d032.h = vpack(s03.w, s02.w):sat                  //pack high 16bits of accs
       s04.w += vmpyo(s0_sh.w, vrecip0.h):SSR             //
       s0_sh.w = vasl(s10.w, zshift)                     //
   } {
       s05.w = vmpye(s1_sh.w, vrecip0.uh)                 //
       mino.w = vmin(mino.w, s12.w)                      //min accumulation
       p3 = sp1loop0(.L_filt_width, filt_width)          //set up inne rloop for next time
   } {
       maxo.w = vmax(maxo.w, s12.w)                      //max accumulation
       s05.w += vmpyo(s1_sh.w, vrecip0.h):SSR             //
       d03210.ub = vpack(d032.h, d010.h):sat             //shift 16bits by zshift
   } {
       vmemu(ptr_z0+#0) = d03210                         //
       maxo.w = vmax(maxo.w, s13.w)                      //max accumulation
       s10.w = vmpye(s0_sh.w, vrecip1.uh)                 //
   } {
       d054.h = vpack(s05.w, s04.w):sat                  //pack high 16bits of accs
       s1_sh.w = vasl(s11.w, zshift)                     //
       s10.w += vmpyo(s0_sh.w, vrecip1.h):SSR             //
   } {
       s11.w = vmpye(s1_sh.w, vrecip1.uh)                 //
       s0_sh.w = vasl(s12.w, zshift)                     //
       maxo.w = vmax(maxo.w, s15.w)                      //max accumulation
   } {
       s11.w += vmpyo(s1_sh.w, vrecip1.h):SSR             //
       d054.ub = vpack(vin_zero.h, d054.h):sat           //shift 16bits by zshift
       s1_sh.w = vasl(s13.w, zshift)                     //
   } {
       maxo.w = vmax(maxo.w, s14.w)                      //max accumulation
       s12.w = vmpye(s0_sh.w, vrecip1.uh)                 //
   } {
       s12.w += vmpyo(s0_sh.w, vrecip1.h):SSR             //
       d054 = vror(d054, align)                          //
   } {
       s13.w = vmpye(s1_sh.w, vrecip1.uh)                 //
       loop1(.L_filt_height, filt_height)                //setup vertical filte rloop
   } {
       q1 = vand(vpred1, vplut)                          //
       q2 = vand(vpred2, vplut)                          //
       d110.h = vpack(s11.w, s10.w):sat                  //pack high 16bits of accs
   } {   
       if(q1)  vmem(ptr_z0+#1):nt = d054                 //
       s0_sh.w = vasl(s14.w, zshift)                     //
       p1 = cmp.gt(col_count, #0)                        //
   } {
       if(q2) vmem(ptr_z0+#2):nt = d054                  //
       s13.w += vmpyo(s1_sh.w, vrecip1.h):SSR             //
       ptr_z0 = add(ptr_z0, #192)                        //
   } {
       s14.w = vmpye(s0_sh.w, vrecip1.uh)                 //
       s1_sh.w = vasl(s15.w, zshift)                     //
       p3 = sp1loop0(.L_filt_width, filt_width)          //setup inner filter loop
   } {
       d132.h = vpack(s13.w, s12.w):sat                  //pack high 16bits of accs
       s14.w += vmpyo(s0_sh.w, vrecip1.h):SSR             //
       vplut = extractu(ptr_z1, #2, #5)                  //1100000 ->  0x08080808
       align = sub(#128, ptr_z1)                         //
   } {
       ones = ##0x01010101                               //
       s15.w = vmpye(s1_sh.w, vrecip1.uh)                 //
       wsum0 = vmem(ptr_filtsum+#0)                      //set 2nd weight offset
   } {
       s15.w += vmpyo(s1_sh.w, vrecip1.h):SSR             //
       d13210.ub = vpack(d132.h, d110.h):sat             //shift 16bits by zshift
       ptr_w0 = memw(sp+#13<<2)                          //access ptr weight
       ptr_w1 = memw(sp+#20<<2)                          //access weights stride
   } {
       s05_s02 = vcombine(wsum0,wsum0)                   //init sum1 and 5
       fetch_ptr_base = add(ptr_x0, in_width_32)         //fetch is next row ahead
       vplut = asl(ones, vplut)                          // 3,2,1,0
       wsum1 = vmem(ptr_filtsum+#1)                      //set 2nd weight offset
   } {
       s03_s00 = vcombine(wsum0,wsum0)                   //init sum0 and 4
       d154.h = vpack(s15.w, s14.w):sat                  //pack high 16bits of accs
       p2 = cmp.gt(fetch_ptr_base, cbuf_eob)             //[E,10]
       if(p2.new)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size)//[E,10]wrap around end fetch ptr
   } {
       if(p1) zshift = #0                                // clear zshift (except last loop)
       vmemu(ptr_z1+#0) = d13210                         //
       s10 = wsum1                                       //init sum 0 and 4
   } {
       maxe.w = vasl(maxe.w,zshift)                      // maxe <<= zshift (on last only)
       s14_s11 = vcombine(wsum1,wsum1)                   //init sum 1 and 5
       d154.ub = vpack(vin_zero.h, d154.h):sat           //shift 16bits by zshift
   } {
       s15_s12 = vcombine(wsum1,wsum1)                   //init sum 1 and 5
       s04_s01 = vcombine(wsum0,wsum0)                   //init sum1 and 5
       //ptr_equalize = memw(sp+#43<<2)                    //
   } {
       s13 = wsum1                                       //init sum 0 and 4
       d154 = vror(  d154, align)                        //
       q1 = vand(vpred1, vplut)                          //
     //  if(!p1) wscale0 = vrecip0          //
   } {
       if(q1) vmem(ptr_z1+#1):nt = d154                  //
       q2 = vand(vpred2, vplut)                          //
   } {
       if(q2) vmem(ptr_z1+#2):nt = d154                  //
       ptr_z1 = add(ptr_z1, #192)                        //
       if (p1) jump   .L_width                           //next 2 rows 8 points per row
   }//endloop width
/* --------------------------------------------------------------------------- */
 // need to apply << zshift  to mine, maxo, mino;
 // maxe has been done already
 // The below then scales these by the per-depth gains, applies to gmin/gmax,
 // and resets mine:maxe and mino:maxo  to +7fffffff : -7fffffff

   {   p0 = cmp.eq(out_height, #0)                       //are vertical lines done?
       tmax.w = vmpye(maxe.w, vrecip0.uh)                //
       mine.w = vasl(mine.w,zshift)                      //
   } {
       tmax.w+= vmpyo(maxe.w, vrecip0.h):SSR             //
       maxe = #0
       ptr_filtsum = add(ptr_filtsum, #256)              //
       wscale1 = vrecip1
   } {
       gmax.w = vmax(gmax.w, tmax.w)
       tmin.w = vmpye(mine.w, vrecip0.uh)                //
       maxo.w = vasl(maxo.w,zshift)                      //
   } {
     //  memw(sp+#43<<2) = ptr_equalize                  //
       tmin.w+= vmpyo(mine.w, vrecip0.h):SSR             //
       mine = vnot(maxe)                                 // 0xfffffff
       col_count = memw(sp+#15<<2)                       //initialize width count
   } {
       mino.w = vasl(mino.w,zshift)                      //
       gmin.w = vmin(gmin.w, tmin.w)
       tmax.w = vmpye(maxo.w, wscale1.uh)                //
       recip_level = memw(sp+#37<<2)                     //255/max
   } {
       tmax.w+= vmpyo(maxo.w, wscale1.h):SSR             //
       mine.uw = vavg(mine.uw,maxe.uw)                   // 0x7fffffff
       if(!p0)vrecip0 = vmem(recip_level++#1)
       weight_stride = memw(sp+#24<<2)                   //
   } {
       gmax.w = vmax(gmax.w, tmax.w)
       tmin.w = vmpye(mino.w, wscale1.uh)                //
       if(!p0)vrecip1 = vmem(recip_level++#1)
       memw(sp+#13<<2) += weight_stride                  //ptr_w0 access ptr weight
   } {
       mino = mine
       memw(sp+#37<<2) = recip_level                     //255/max
       tmin.w+= vmpyo(mino.w, wscale1.h):SSR             //
       maxe.w = vsub(maxe.w,mine.w)                   // -0x7fffffff
   } {
       maxo = maxe
       gmin.w = vmin(gmin.w, tmin.w)
       memw(sp+#20<<2) += weight_stride                  //ptr_w1 access weights stride
       if(!p0) jump .L_height                            //then go again
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
   {   vmem(ptr_max+#0) = gmax                           //store max
       r17:16 = memd(sp+#0)                              //restore stack
   } {
       vmem(ptr_max+#1) = gmin                           //store min
       r19:18 = memd(sp+#2<<2)                           //18,19
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
      .size gvconv2dbbb_circ6_d64_v65_asm, .L_end-gvconv2dbbb_circ6_d64_v65_asm
