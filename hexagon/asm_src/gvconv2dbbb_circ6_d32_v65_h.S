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
     CODESIZE 1120 bytes
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
     The code only processes 32 sets of weights at once inner loop is optimized
     to 6 packets.
 */
/*===============================================================================*/
        .text
        .file "gvconv2dbbb_circ6_d32_v65_h.S"
        .global gvconv2dbbb_circ6_d32_v65_asm
        .balign 32
        .type  gvconv2dbbb_circ6_d32_v65_asm, @function
gvconv2dbbb_circ6_d32_v65_asm:
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
#define ptr_max                r28    //36 maximum and minum buffer
#define recip_level            r26    //37 255 / (MAX - MIN) - used to scale to bytes
#define out_width_32           r7     //38 actual out_width in depth32
#define ptr_cbufi              r16    //39 read buffer pointer
#define zshift                 r21    //40 extra shift on output before quantization
#define in_zero                r25    //41
#define store_cntrl            r11    //42
//#define ptr_equalize           r17    //43
#define cbuf_eob               r18    //18 end of cuirc buffer
#define cbuf_size              r19    //19 size in bytes of circ buf -1
/* --------------------------------- SCALER REGS ------------------------------- */
#define delta                  r7     //difference ebetween stride height and filt_height
#define cm4                    r2     //shuffle/deal ints
#define col_count              r2     //horizontal counter
#define in_width_32            r3     //total input width in bytes in buffer
#define x41_x11                r15:14 //4n+1 inputs
#define x40_x10                r17:16 //4n+0 inputs
#define x30_x00                r15:14 //4n+0 inputs
#define x31_x01                r17:16 //4n+1 inputs
#define x50_x20                r15:14 //4n+0 inputs
#define x51_x21                r17:16 //4n+1 inputs
#define ptr_wi_ptr_xi          r1:0   //
#define fetch_ptr_base         r1     //base pointer for l1 prefetch
#define fetch_ptr              r10    //current pointer for l1 prefetch
#define ones                   r11    //3*stride
#define vplut                  r14    //lookup table for store contro`l
#define align                  r15
#define stride                 r12    //current to next input
#define ptr_x0                 r26    //base input pointer
#define ptr_x10                r13    //current even input ptr      
#define ptr_x11                r27    //current odd input ptr      
#define ptr_w0                 r20    //even output depth 32 weights
#define ptr_z0                 r0     //even output depth 32 outputs
#define adjust                 r10    //
/* ---------------------------------- VEC REGS -------------------------------- */
#define wscale                 v11    //
#define vin_zero               v9     //
#define s05_s02                v15:14 //even output accs 2,5
#define s04_s01                v13:12 //even output accs 1,4
#define s03_s00                v11:10 //even output accs 0,3
#define s05                    v15    //even acc 5
#define s02                    v14    //even acc 2
#define s04                    v13    //even acc 4
#define s01                    v12    //even acc 1
#define s03                    v11    //even acc 3
#define s00                    v10    //even acc 0
#define s15_s12                v23:22 //even output accs 2,5
#define s14_s11                v21:20 //even output accs 1,4
#define s13_s10                v19:18 //even output accs 0,3
#define s13                    v19    //even output accs 3
#define s10                    v18    //even output accs 0
#define s14                    v21    //even output accs 4
#define s11                    v20    //even output accs 1
#define s15                    v23    //even output accs 5
#define s12                    v22    //even output accs 2
#define vpred0                 v24    //
#define vpred1                 v25    //
#define vpred2                 v16    //
#define vzero                  v30    //
#define w00                    v0     //weights even 0-31
#define w01                    v3     //weights even 32-63
#define vrecip                 v1     //reciprocal 255/MAx replicated
#define s0_sh                  v8     //round value
#define s1_sh                  v26    //round value
#define wsum0                  v2     //sum of weights column + bias add 0-31
#define d010                   v27    //even lines upper 16bit packed accs 0,1
#define d032                   v28    //even lines upper 16bit packed accs 2,3
#define d03210                 v28    //8bit shifted, packed saturated 0-3
#define d054                   v29    //even lines upper 16bit packed accs 4,5
#define maxo_maxe              v5:4   //packed maxes
#define maxo                   v5     //odd maxes
#define maxe                   v4     //even maxes
//#define mino_mine              v7:6   //packed mins
#define mino                   v18    //odd mins
#define mine                   v17    //even mins
#define gmax                   v7   //packed mins 
#define gmin                   v6   //packed mins 
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/* --------------------------------------------------------------------------- */
   {   allocframe(#112)                                  //0th entry on stack is (112+8)/4=30 ints
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
       memd(sp+#12<<2) = ptr_wi_ptr_xi                   //save weights:activation
       memw(sp+#14<<2) = ptr_zi                          //save output ptr
   } {
       memw(sp+#15<<2) = out_width                       //save output width
       filt_height = memw(sp+#33<<2)                     //filter height
       stride = zxth(stride_h_w)                         //horizontal stride
   } {
       ptr_filtsum = memw(sp+#35<<2)                     //ptr to the sum of filters+offset
       filt_width = memw(sp+#32<<2)                      //filter width
   } {
       in_depth = memw(sp+#31<<2)                        //input depth
       wsum0 = vmem(ptr_filtsum+#0)                      //set 1st weight offset
   } {
       in_zero = memw(sp+#41<<2)                         //
       cbuf_size = mpyi(filt_height, in_depth)           //circular buffer size
       dcfetch(ptr_xi+#0<<6)                             //
   } {
       ptr_max = memw(sp+#36<<2)                         //get max/min ptr
       in_zero = vsplatb(in_zero)                        //
       stride = asl(stride, #5)                          //32 * stride_w
       dcfetch(ptr_xi+#1<<6)                             //
   } {
       zshift = memw(sp+#40<<2)                          //final shift 7 + 16
       store_cntrl = memw(sp+#42<<2)                     //
       vin_zero = vsplat(in_zero)                        //
       cbuf_size = mpyi(cbuf_size, next_inbuf_width)     //circular buffer size
   } {
       out_width_32 = memw(sp+#38<<2)                    //total width of output
       ptr_cbufi = memw(sp+#39<<2)                       //circular buffer
       cbuf_size = add(cbuf_size, cbuf_size)             //x2
       filt_width = asl(filt_width, #2)                  //*32/8
   } {
       cbuf_eob = add(ptr_cbufi, cbuf_size)              //end of circ buffer marker
       filt_height = mpyi(filt_height, in_depth)         //total number of depth32 filter rows
       recip_level = memw(sp+#37<<2)                     //255/max
       gmax = vmem(ptr_max+#0)                           //
   } {
       cbuf_eob = add(cbuf_eob, #-4)                     //make so comparison is >= eob
       gmin = vmem(ptr_max+#1)                           //
       dcfetch(ptr_xi+#2<<6)                             //
   } {
       filt_width = add(filt_width, #-1)                 //account for epilog
       filt_height = lsr(filt_height, #5)                //num d32 rows in filter
       vpred1 = vmem(store_cntrl+#1)                     //
       dcfetch(ptr_xi+#3<<6)                             //
   } {
       filt_height = add(filt_height, #-1)
       in_width_32 = asl(next_inbuf_width, #6)           //next d32 line x 2
       out_height = memw(sp+#34<<2)                      //height of output
       vrecip = vmem(recip_level++#1)                      //used to compress to 8bits 255/max
   } {
       vpred2 = vmem(store_cntrl+#2)                     //
       memw(sp+#37<<2) = recip_level                     //255/max
   }
/* -------------------------------------------------------------------------- */
   .balign 32
.L_height:
   {   ptr_z0 = memw(sp+#14<<2)                          //output ptr for even lines
       ptr_x0 = memw(sp+#12<<2)                          //ptr_x0=ptr_cbufi read circ buffer
       s03_s00 = vcombine(wsum0,wsum0)                   //init sum0 and 4
       s13_s10 = #0                                      //
   } {                                                   //buffer read ptr if ptr_xi >= buf_size-=size
       memw(sp+#14<<2) += out_width_depth                //update output ptr
       fetch_ptr_base = add(ptr_x0, in_width_32)         //fetch is next row ahead
       s05_s02 = vcombine(wsum0,wsum0)                   //init sum2 and 6
       out_height = add(out_height, #-1)                 //decrement height count
   } {
       p1 = cmp.gt(fetch_ptr_base, cbuf_eob)             //if prefetch >= circ buffer wrap around
       s04_s01 = vcombine(wsum0,wsum0)                   //init sum1 and 5
       s14_s11 = #0                                      //
       ptr_w0 = memw(sp+#13<<2)                          //access ptr weight
   } {
       s15_s12 = #0                                      //
       if(p1)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size) //wrap fetch ptr around independently
       col_count = memw(sp+#15<<2)                       //initialize width count
       nop                                               //
   }
.L_width:
   {   p2 = cmp.eq(filt_height, #0)
       loop1(.L_filt_height, filt_height)                //setup vertical filte rloop
       nop; nop;                                         //
   }{
       if(p2) jump .L_last1                              //
       nop; nop; nop                                     //
   }
   .balign 32
.L_filt_height:
   {   w00 = vmem(ptr_w0++#1)                            //[0, 0]1st 32 weights of out depth
       x50_x20 = memd(ptr_x0+stride<<#2)                 //[0, 0]load pt 3 and 7
       p3 = sp1loop0(.L_filt_width, filt_width)          //set up inne rloop for next time
       ptr_x11 = add(ptr_x0, #8)                         //set up currne tinput ptr
   } {
       fetch_ptr = add(fetch_ptr_base, #0)               //initial fetch ptr
       w01 = vmem(ptr_w0++#1)                            //[0, 1]2nd 32weights stream 0
       x51_x21 = memd(ptr_x11+stride<<#2)                //[0, 1]
       ptr_x10 = ptr_x0                                  //set up currne tinput ptr
   } 
   .balign 32
.L_filt_width:
   {   dcfetch(fetch_ptr+#0<<6)                          //[0, 2]fetch 64bytes-2 lots 8 x 4 bytes
       fetch_ptr = add(fetch_ptr, #64)                   //[0. 2]
       s05_s02.w += vrmpy(w00.b, x50_x20.ub)             //[0, 2]macc 2,6  out 0
       nop
   } {
       s15_s12.w += vrmpy(w01.b, x51_x21.ub)             //[0, 3]acc 2,3,6,7
       x40_x10 = memd(ptr_x10+stride<<#1)                //[0, 3]load pt 1 5
       x41_x11 = memd(ptr_x11+stride<<#1)                //[0, 3]
       nop
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[0, 4]
       s14_s11.w += vrmpy(w01.b, x41_x11.ub)             //[0, 4]
       x30_x00 = memd(ptr_x10++#2<<3)                    //[0, 4]load pts 0, 4
       x31_x01 = memd(ptr_x11++#2<<3)                    //[0, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[0, 5]acc 0,4,1,5 out 0
       w00 = vmem(ptr_w0++#1)                            //[1, 0]1st 32 weights of out depth
       x50_x20 = memd(ptr_x10+stride<<#2)                //[1, 0]load pt 3 and 7
       if(!p3) ptr_x0 = add(ptr_x0, in_width_32)         //if >= buf_size -= buf_size
   } {
       if(!p3) fetch_ptr_base=add(fetch_ptr_base,in_width_32)//if >= buf_size -= buf_size
       s13_s10.w += vrmpy(w01.b, x31_x01.ub)             //[0, 6]
       w01 = vmem(ptr_w0++#1)                            //[1, 1]2nd 32weights stream 0
       x51_x21 = memd(ptr_x11+stride<<#2)                //[1, 1]
   }:endloop0
   {   dcfetch(fetch_ptr+#0<<6)                          //[1, 2]fetch 64bytes-2 lots 8 x 4 bytes
       s05_s02.w += vrmpy(w00.b, x50_x20.ub)             //[1, 2]macc 2,6  out 0
       p1 = cmp.gt(fetch_ptr_base, cbuf_eob)             //[E,10]
       if(p1.new)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size)//[E,10]wrap around end fetch ptr
   } {
       s15_s12.w += vrmpy(w01.b, x51_x21.ub)             //[1, 3]acc 2,3,6,7
       x40_x10 = memd(ptr_x10+stride<<#1)                //[1, 3]load pt 1 5
       x41_x11 = memd(ptr_x11+stride<<#1)                //[1, 3]
       nop
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[1, 4]
       s14_s11.w += vrmpy(w01.b, x41_x11.ub)             //[1, 4]
       x30_x00 = memd(ptr_x10++#2<<3)                    //[1, 4]load pts 0, 4
       x31_x01 = memd(ptr_x11++#2<<3)                    //[1, 4]
   } {
       p0 = cmp.gt(ptr_x0, cbuf_eob)                     //[E,10]
       if(p0.new)ptr_x0 = sub(ptr_x0, cbuf_size)         //[E,10]wrap around end of buffer
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[1, 5]acc 0,4,1,5 out 0
       s13_s10.w += vrmpy(w01.b, x31_x01.ub)             //[1, 5]
   }:endloop1
.L_last1:
   {   ptr_x11 = add(ptr_x0, #8)                         //set up currne tinput ptr
       ptr_x10 = ptr_x0                                  //set up currne tinput ptr
       ptr_x0 = add(ptr_x0, in_width_32)                 //if >= buf_size -= buf_size
       w00 = vmem(ptr_w0++#1)                            //[0, 0]1st 32 weights of out depth
   } {
       x50_x20 = memd(ptr_x10+stride<<#2)                //[0, 0]load pt 3 and 7
       p0 = cmp.gt(ptr_x0, cbuf_eob)                     //[E,10]
       if(p0.new)ptr_x0 = sub(ptr_x0, cbuf_size)         //[E,10]wrap around end of buffer
       loop0(.L_filt_width1, filt_width)                 //set up inne rloop for next time
   } {
       w01 = vmem(ptr_w0++#1)                            //[0, 1]2nd 32weights stream 0
       x51_x21 = memd(ptr_x11+stride<<#2)                //[0, 1]
       fetch_ptr = add(ptr_x0, mpyi(stride, #12))        //stride*2*3 advance buffer by 6 outputs
       nop
   } 
   .balign 32
.L_filt_width1:
   {   dcfetch(fetch_ptr+#0<<6)                          //[0, 2]fetch 64bytes-2 lots 8 x 4 bytes
       fetch_ptr = add(fetch_ptr, #64)                   //[0. 2]
       s05_s02.w += vrmpy(w00.b, x50_x20.ub)             //[0, 2]macc 2,6  out 0
   } {
       s15_s12.w += vrmpy(w01.b, x51_x21.ub)             //[0, 3]acc 2,3,6,7
       x40_x10 = memd(ptr_x10+stride<<#1)                //[0, 3]load pt 1 5
       x41_x11 = memd(ptr_x11+stride<<#1)                //[0, 3]
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[0, 4]
       s14_s11.w += vrmpy(w01.b, x41_x11.ub)             //[0, 4]
       x30_x00 = memd(ptr_x10++#2<<3)                    //[0, 4]load pts 0, 4
       x31_x01 = memd(ptr_x11++#2<<3)                    //[0, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[0, 5]acc 0,4,1,5 out 0
       w00 = vmem(ptr_w0++#1)                            //[1, 0]1st 32 weights of out depth
       x50_x20 = memd(ptr_x10+stride<<#2)                //[1, 0]load pt 3 and 7
   } {
       s13_s10.w += vrmpy(w01.b, x31_x01.ub)             //[0, 6]
       w01 = vmem(ptr_w0++#1)                            //[1, 1]2nd 32weights stream 0
       x51_x21 = memd(ptr_x11+stride<<#2)                //[1, 1]
   }:endloop0
   {   dcfetch(fetch_ptr+#0<<6)                          //[1, 2]fetch 64bytes-2 lots 8 x 4 bytes
       s05_s02.w += vrmpy(w00.b, x50_x20.ub)             //[1, 2]macc 2,6  out 0
//     adjust = memw(sp+#23<<2) 
       col_count = add(col_count, #-6)                   //decrement width count by 8
   } {
       s15_s12.w += vrmpy(w01.b, x51_x21.ub)             //[1, 3]acc 2,3,6,7
       x40_x10 = memd(ptr_x10+stride<<#1)                //[1, 3]load pt 1 5
       x41_x11 = memd(ptr_x11+stride<<#1)                //[1, 3]
       p2 = cmp.ge(col_count,#0)                         //
   } {
       s04_s01.w += vrmpy(w00.b, x40_x10.ub)             //[1, 4]
       s14_s11.w += vrmpy(w01.b, x41_x11.ub)             //[1, 4]
       x30_x00 = memd(ptr_x10+#0<<3)                     //[1, 4]load pts 0, 4
       x31_x01 = memd(ptr_x11+#0<<3)                     //[1, 4]
   } {
       s03_s00.w += vrmpy(w00.b, x30_x00.ub)             //[1, 5]acc 0,4,1,5 out 0
       s13_s10.w += vrmpy(w01.b, x31_x01.ub)             //[1, 5]
       ones = ##0x01010101                               //
   }
/* ------------------------------------------------------------------------ */
   {   s00.w = vadd(s00.w, s10.w)                        //combine reducnent accs
       s01.w = vadd(s01.w, s11.w)                        //combine reducnent accs
       s02.w = vadd(s02.w, s12.w)                        //combine reducnent accs
       s03.w = vadd(s03.w, s13.w)                        //combine reducnent accs
   } {
       s04.w = vadd(s04.w, s14.w)                        //combine reducnent accs
       s05.w = vadd(s05.w, s15.w)                        //combine reducnent accs
       mine.w = vmin(mine.w, s00.w)                      //min accumulation
       s0_sh.w = vasl(s00.w, zshift)                     //
   } {
       maxe.w = vmax(maxe.w, s00.w)                      //max accumulation
       s1_sh.w = vasl(s01.w, zshift)                     //
//     ptr_x0 = sub(ptr_x0, adjust)                      //-=filt_height if stride_height > filt_height
       if (!p2) s05 = s00                                //
   } {   
       s00.w = vmpye(s0_sh.w, vrecip.uh)                 //
       maxe.w = vmax(maxe.w, s01.w)                      //max accumulation
       mine.w = vmin(mine.w, s01.w)                      //min accumulation
   } {
       s00.w += vmpyo(s0_sh.w, vrecip.h):SSR             //
       ptr_x0 += mpyi(stride, #12)                       //stride*2*4 advance buffer by 8 outputs
   } {
       s0_sh.w = vasl(s02.w, zshift)                     //o
       s01.w = vmpye(s1_sh.w, vrecip.uh)                 //
       align = sub(#128, ptr_z0)                         //
   } {
       maxe.w = vmax(maxe.w, s02.w)                      //max accumulation
       mine.w = vmin(mine.w, s02.w)                      //min accumulation
       s01.w += vmpyo(s1_sh.w, vrecip.h):SSR             //
       p0 = cmp.gt(col_count, #0)                        //will be back at saem place just advance
   } {
       maxe.w = vmax(maxe.w, s03.w)                      //max accumulation
       s1_sh.w = vasl(s03.w, zshift)                     //o
       s02.w = vmpye(s0_sh.w, vrecip.uh)                 //
   } {
       mine.w = vmin(mine.w, s03.w)                      //min accumulation
       d010.h = vpack(s01.w, s00.w):sat                  //pack high 16bits of accs
       s02.w += vmpyo(s0_sh.w, vrecip.h):SSR             //
       vplut = extractu(ptr_z0, #2, #5)                  //
   } {
       s0_sh.w = vasl(s04.w, zshift)                     //o
       s03.w = vmpye(s1_sh.w, vrecip.uh)                 //
       mine.w = vmin(mine.w, s04.w)                      //min accumulation
   } {
       s03.w += vmpyo(s1_sh.w, vrecip.h):SSR             //
       vplut = asl(ones, vplut)                          //
       maxe.w = vmax(maxe.w, s04.w)                      //max accumulation
   } {
       mine.w = vmin(mine.w, s05.w)                      //min accumulation
       s1_sh.w = vasl(s05.w, zshift)                     //o
       s04.w = vmpye(s0_sh.w, vrecip.uh)                 //
   } {
       maxe.w = vmax(maxe.w, s05.w)                      //max accumulation
       d032.h = vpack(s03.w, s02.w):sat                  //pack high 16bits of accs
       s04.w += vmpyo(s0_sh.w, vrecip.h):SSR             //
   } {
       s05.w = vmpye(s1_sh.w, vrecip.uh)                 //
       fetch_ptr_base = add(ptr_x0, in_width_32)         //fetch is next row ahead
       wsum0 = vmem(ptr_filtsum+#0)                      //set 1st weight offset
   } {
       d03210.ub = vpack(d032.h, d010.h):sat             //shift 16bits by zshift
       s05.w += vmpyo(s1_sh.w, vrecip.h):SSR             //
   } {
       p2 = cmp.gt(fetch_ptr_base, cbuf_eob)             //if prefetch >= circ buffer wrap around
       if(p2.new)fetch_ptr_base=sub(fetch_ptr_base,cbuf_size) //wrap fetch ptr around independently
       s03_s00 = vcombine(wsum0,wsum0)                   //init sum0 and 4
   } {
       q1 = vand(vpred1, vplut)                          //
       d054.h = vpack(s05.w, s04.w):sat                  //pack high 16bits of accs
   } {
       s04_s01 = vcombine(wsum0,wsum0)                   //init sum1 and 5
       s05_s02 = vcombine(wsum0,wsum0)                   //init sum1 and 5
   } {
       d054.ub = vpack(vin_zero.h, d054.h):sat           //shift 16bits by zshift
       ptr_w0 = memw(sp+#13<<2)                          //access ptr weight
       //ptr_equalize = memw(sp+#43<<2)                     //
   } {
       vmemu(ptr_z0+#0) = d03210
       s13_s10 = #0                                      //init sum0 and 4
//     loop1(.L_filt_height, filt_height)                //setup vertical filte rloop
       //lc1 = filt_height
       maxo.w = vasl(maxe.w,zshift)                      // speculative maxe <<= zshift
   } {
       d054 = vror(d054, align)                          //
       s14_s11 = #0                                      //init sum1 and 5
       if(!p0)wscale = vrecip                    //
   } {
       s15_s12 = #0                                      //init sum1 and 5
       if(q1) vmem(ptr_z0+#1):nt = d054                  //store 0-3 even row
       q2 = vand(vpred2, vplut)                          //
   } {
       if(q2) vmem(ptr_z0+#2):nt = d054                  //store 4-7 even row
       ptr_z0 = add(ptr_z0, #192)                        //
       if (p0) jump   .L_width                           //next 2 rows 8 points per row
   }//endloop width
/*
       ----|--54|0000 0011
       ----|-54-|0000 0110
       ----|54--|0000 1100
       ---5|4---|0001 1000
 */
/* --------------------------------------------------------------------------- */
   {   recip_level = memw(sp+#37<<2)                      //255/max
       mine.w = vasl(mine.w,zshift)
       maxe.w = vmpye(maxo.w, wscale.uh)                  //[1, 0]equalize max's
       p0 = cmp.eq(out_height, #0)                        //are vertical lines done?
   } {
       col_count = #0x80000001                            // init for maxe
       mino.w = vmpye(mine.w, wscale.uh)                  //[1, 1]equalize min's
       if(!p0)vrecip = vmem(recip_level++#1)              //used to compress to 8bits 255/max
   } {
       ptr_z0 = abs(col_count)                            // init for mine
       memw(sp+#37<<2) = recip_level                      //255/max
       maxe.w+= vmpyo(maxo.w, wscale.h):SSR               //[1, 2]equalize max's
   } {
       gmax.w = vmax(gmax.w, maxe.w)
       mino.w+= vmpyo(mine.w,wscale.h):SSR                //[1, 3]equalize min's
       ptr_filtsum = add(ptr_filtsum, #128)               //
   } {
       mine = vsplat(ptr_z0)
       maxe = vsplat(col_count)
       gmin.w = vmin(gmin.w, mino.w)
       if(!p0) jump:nt .L_height                          //then go again
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
   .size gvconv2dbbb_circ6_d32_v65_asm, .L_end-gvconv2dbbb_circ6_d32_v65_asm
