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

/*======================================================================*/
/*  FUNCTIONS      : gvconv2dbbbb_asm                                   */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform 2d convolution with input depth to otuput    */
/*                 max, min computed and output scaled to 8bits         */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date         Comments                           */
/*  -------------------------------------------------------------       */
/*  DJH                 04/21/17     created                            */
/*  DJH                 05/12/17     update api precomputed filt_offset */
/*  DJH                 05/16/17     Hoisted loop0 around to prolog and */
/*                                   epilog of loop1                    */
/*======================================================================*/
#if 0
#endif
/*=============================================================================*/
        .text
        .file "gvconv2db2b2b2_d32_h_v66.S"
        .global gvconv2db2b2b2_d32_asm
        .balign 32
        .type  gvconv2db2b2b2_d32_asm, @function
gvconv2db2b2b2_d32_asm:
/*=============================================================================*/
/*=============================================================================*/
#define ptr_xei               r0    //data    aligned 128
#define ptr_xoi               r1    //data    aligned 128
#define ptr_wi                r2    //weights aligned 128
#define ptr_zei               r3    //results aligned 128
#define ptr_zoi               r4    //results aligned 128
#define in_width              r5    //(pad_l+in_width+pad_r) => 4 %4
#define out_width             in_width

#define out_width_depth       r4    //0 value in bytes to get to next full out row
#define col_count             r25   //1 out_width_pad
#define stride_h_w            r26   //2 stride_height|stride_width
#define in_depth              r27   //3 %32
#define filt_width            r20   //4 >= 1
#define filt_height           r8    //5 >= 1filt_height lines per filter
#define out_height            r9    //6 >= 1 number of vertical lines to perform
#define ptr_biasadd           r10   //7 aligned 128
#define ptr_minmax            r12   //8 aligned 128
#define recip_level           r14   //9 recip is 31bit unsigned 0x7f800000000LL / max
#define recip_shift           r13   //10 recip is 31bit unsigned 0x7f800000000LL / max
#define out_align             r6    //11 0, 32, 64, 96
#define skip_col              r21   //12
/*=============================================================================*/
#define actvtn_stride         r1    //distance between odd and even activations
#define ptr_wi1               r10   //
#define c8                    r3    //
#define prod_cnt              r12   //used to count through the 3 products
#define filt_cnt              r11   //how many vertical filter rows there are
#define in_width_stride_h_depth r15 //in_width * stride_h * in_depth for next output
#define ptr_x0                r16   //
#define ptr_x1                r7    //
#define stride_w              r18   //stride width
#define next_outputs          r19   //jump to input ptr for next set of outputs
#define ptr_w                 r17   //
#define ptr_w_ptr_x0          r17:16//
#define in_width_32           r22   //
#define ptr_x2                r23   //
#define ptr_z0                r24   //
#define ptr_z1                r26   //
#define ptr_xij               r27   //ptr_xi + j
#define scratch               r0    //
#define scratch0              r1    //
#define prep_ptr              r2    //pointer to pre-coded read ptrs for each part of mpy
/*=============================================================================*/
#define PV32(VSRC) .word (0x1DFFE020+VSRC)
#define s0                    v0    //
#define s1                    v1    //
#define s1s0                  v1:0  //
#define s2                    v2    //
#define s3                    v3    //
#define s3s2                  v3:2  //
#define s3s2s1s0              v3:0  //
#define w0                    v21   //
#define x0                    v4    //
#define x1                    v5    //
#define x2                    v6    //
#define x3                    v7    //
#define x3210e                v6    //
#define x3210o                v21   //
#define x3210e_prev           v16   //previous value
#define x3210o_prev           v22   //previous value
#define xout                  v17   //realigned out
#define y10                   v8    //
#define y0                    v8    //
#define y1                    v9    //
#define y32                   v10   //
#define y2                    v10   //
#define y3                    v11   //
#define wsum                  v14   //initialzed to in_offsey*wsum + biasoffset
#define maxe                  v12   //
#define mine                  v18   //
#define biasvec               v16   //
#define recipvec              v15   //
#define vcrnd                 v20   //contain 0080 in all of the words
#define vzero                 v23   //
#define sk                    V24   //
#define RSS     <<1:rnd:sat:shift   //unverbose the insturction
#define SPVEC                 704
/*=============================================================================
(sp+#56) 0              # w0*x0
(sp+#60) ptr_wi         # w0*x0
(sp+#64) actvtn_stride  #0 w0*x1
(sp+#68) ptr_wi         #0 w0*x1
(sp+#72) 0              #1 w1*x0
(sp+#76) ptr_wi1        #1 w1*x0
(sp+#80) actvtn_stride  #2 w1*x1
(sp+#84) ptr_wi1        #2 w1*x1
(sp+#88) #0             # guard against oob load
(sp+#92) #0             # guard against oob load
(sp+#96) ptr_zi
(sp+#100) ptr_ze
===============================================================================*/
       {   allocframe(#(4*SPVEC+56))                  //0th entry on stack is (512+56+8)/4 =16 ints
       } {
           memd(sp+#0)  = r17:16                      //save 16,17
           memd(sp+#8)  = r19:18                      //save 18,19
           ptr_xei = and(ptr_xei, #-2)                //guarentee lsb is 0
           ptr_xoi = and(ptr_xoi, #-2)                //guarentee lsb is 0
       } {
           memd(sp+#16) = r21:20                      //save 20,21
           memd(sp+#24) = r23:22                      //save 22,23
#if defined(FAST_16B_CONV)
           r20 = ##0x00000080                         //rounding value for middle products +80)>>8
#else
           r20 = ##0x00008080                         //rounding value for middle products +80)>>8
#endif
       } {
           vcrnd = vsplat(r20)                        //rounding value for >>8
           ptr_minmax = memw(sp+#(SPVEC+24)<<2)       //ptr pre computed max value in output
           memw(sp+#68) = ptr_wi                      //save weights ptr
           vzero = #0                                 //vector 0
       } {
           memd(sp+#32) = r25:24                      //save 24,25
           mine = vmem(ptr_minmax+#1)                 //get running min
           y0 = vcrnd                                 //
       } {
           memd(sp+#40) = r27:26                      //save 26,27
           maxe = vmem(ptr_minmax+#0)                 //get running max
       } {
           stride_h_w = memw(sp+#(SPVEC+18)<<2)       //extract strides h + w
       } {
           stride_w = zxth(stride_h_w)                //extract stride width
           memw(sp+#60) = ptr_wi                      //save weights ptr
       } {
           stride_w = asl(stride_w, #2)               //4*stride_w
       } {
           p0 = cmp.eq(stride_w, #8)                  //if stride_w = 2 * 4 modify z_buf pointer lsb
           if(p0.new) ptr_xei = add(ptr_xei, #1)      //make lsb of ptr 1 for stride = 2
           if(p0.new) ptr_xoi = add(ptr_xoi, #1)      //make lsb of ptr 1 for stride = 2
       } {
           memw(sp+#48) = ptr_xei                     //save ptr to activations
           actvtn_stride = sub(ptr_xoi, ptr_xei)
           in_depth = memw(sp+#(SPVEC+19)<<2)         //get input depth
       } {
           filt_width = memw(sp+#(SPVEC+20)<<2)       //extract filt_width
           filt_height = memw(sp+#(SPVEC+21)<<2)      //extract filt_height
       } {
           filt_height = mpy(filt_height.L,in_depth.L)//filt_height*in_depth
           out_height = memw(sp+#(SPVEC+22)<<2)       //number of output lines
           ptr_biasadd = memw(sp+#(SPVEC+23)<<2)      //ptr pre computed weight sum
           filt_width = asl(filt_width, #1)           //x2 to account for loop of 16bytes
       } {
           filt_height = lsr(filt_height, #5)         //filt_height * in_depth / 32
           recip_level = memw(sp+#(SPVEC+25)<<2)      //get scalaer 32bit recip level
       } {
           recipvec = vsplat(recip_level)             //spread recip_val across 32words
           recip_shift = memw(sp+#(SPVEC+26)<<2)      //can we flush align to do last col
           skip_col = memw(sp+#(SPVEC+28)<<2)         //can we flush align to do last col
       } {
           in_width_32 = asl(in_width, #5)            //32 * in_width d32 line
           wsum = vmem(ptr_biasadd+#0)                //gemsumb + bias offsets
       } {
           ptr_wi1 = mpyi(filt_height, filt_width)    //offset for hi bytes of weights
           out_align = memw(sp+#(SPVEC+27)<<2)        //output alignment 0,32,64,96
           filt_width = add(filt_width, #-1)          //account for epilog
       } {
           next_outputs = asl(stride_w, #5)           //1,2 32*stride*4 i.e. 128 or 256
           in_width_stride_h_depth= mpy(stride_h_w.H, in_depth.L) //
           memw(sp+#96) = ptr_zei                     //save output ptr on stack
           p3 = cmp.eq(out_align, #0)                 //if no alignment enable store
       } {
           in_width_stride_h_depth=mpyi(in_width,in_width_stride_h_depth) //total vertical stride bytes
           stride_w = mpyi(stride_w, #24)             //offset for z buf 96 or 192
           memw(sp+#100) = ptr_zoi                    //save output ptr on stack
           c8 = #0                                    //temp 0
       } {
           ptr_wi1 = asl(ptr_wi1, #9)                 //* 512 = 2/32*16*32
           stride_w = add(stride_w, #4)               //preset offset for z buf
           memw(sp+#72) = c8                          //d0 x w1 activtn offset
           memw(sp+#88) = c8                          //
       } {
           ptr_wi1 = add(ptr_wi, ptr_wi1)             //ptr to odd bytes of weights
           memw(sp+#64) = actvtn_stride               //d1 x w0 activtn offset
           memw(sp+#56) = c8                          //
           if (p3) y0 = vzero                         //
       } {
           memw(sp+#76) = ptr_wi1                     //save d0 x w1 weight ptr
           memw(sp+#84) = ptr_wi1                     //save d1 x w1 weight ptr
           q1 = vcmp.eq(vzero.w,y0.w)                 //
       } {
           memw(sp+#80) = actvtn_stride               //d1 x w1 activations
           scratch0 = add(sp, #127)                   //align stack to next 128b
           q0 = or(q1, q1)                            //
       } {
           col_count = memw(sp+#(SPVEC+17)<<2)        //read width of activations
           scratch0 = and(scratch0, #-128)            //align stack to next 128b
#if !defined(FAST_16B_CONV)
           c8 = #8                                    //
#endif
       } {
           scratch0 = add(scratch0, #128)             //align stack to next 128b
           out_width_depth = memw(sp+#(SPVEC+16)<<2)  //read width of activations
#if !defined(SPLIT_OUTPUT)
           out_align = mux(p3,#0,#64)                 //
           out_width = add(col_count,#-4)             //
#endif
       }
/*=============================================================================*/
   .balign 64
/*=============================================================================*/
.L_height:
       {
#if defined(FAST_16B_CONV)
           ptr_w_ptr_x0 = memd(sp+#64)                //[Pre-Width]initialize filter pointer & activation offset
#else
           ptr_w_ptr_x0 = memd(sp+#56)                //[Pre-Width]initialize filter pointer & activation offset
#endif
           loop1(.L_filt_height, filt_height)         //[Pre-Width]for(filt_y=0;filt_y<height*in_depth/32;filt_y++){
           ptr_xij = memw(sp+#48)                     //initial main actvtn. ptr_xi
           filt_cnt = add(filt_height, #-1)           //pre-width]initialize filt height cntr
       } {
           loop0(.L_filt_width, filt_width)           //[Pre-Width], 0]ki is k1/32 - 0
           ptr_x0 = add(ptr_xij, ptr_x0)              //[Pre-Width]odd activations + in_Depth_32
#if defined(FAST_16B_CONV)
           prod_cnt = #3                              //[Pre-Width]total 3 partial products
#else
           prod_cnt = #4                              //[Pre-Width]total 4 partial products
#endif
           ptr_z0 = memw(sp+#96)                      //add(ptr_zi, #0)
       } {
           ptr_x2 = and(ptr_x0, #-128)                //[Pre-Width]make loads aligned to 128 zero out bits 0-6
           s1s0 = vcombine(vcrnd, vcrnd)              //[Pre-Width]accumulator 0,1
#if defined(FAST_16B_CONV)
           prep_ptr = add(sp, #(64+8))                //[Pre-Width]ptr to pre computed ptr list
#else
           prep_ptr = add(sp, #(56+8))                //[Pre-Width]ptr to pre computed ptr list
#endif
           p2 = !cmp.eq(r0,r0)                        //p2=0
       } {
           ptr_z1 = memw(sp+#100)                      //add(ptr_zi, #0)
           z = vmem(ptr_x2+#0)                        //[Pre-Width][Pheight]load 0-127
           s3s2 = vcombine(vcrnd,vcrnd)               //[Pre-Width]accumulator 2,3
           scratch = scratch0                         //[Pre-Width]temp accumuator buffer
       } {
           p3 = cmp.eq(out_align, #0)                 //if no alignment enable store
           z = vmem(ptr_x2+#1)                        //[Pre-Width]load 128-255
           ptr_x1 = add(ptr_x0, stride_w)             //[Pre-Width]setup initial pointer
           ptr_x0 = add(ptr_x0, in_width_32)          //[Pre-Width], 0]move to next even line of filter activations
       }
/*=============================================================================*/
   .balign 64
.L_width:
.L_products:                                          //d1 * w0,d0 * w1,d1 * w1
.L_filt_height:
.L_filt_width:
       {   w0.tmp         = vmem(ptr_w++#1)           //[0, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 0]perform mac across 4 streams with saem weights
       } {
           w0.tmp         = vmem(ptr_w++#1)           //[0, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 1]perform mac across 4 streams with saem weights
       } {
           w0.tmp         = vmem(ptr_w++#1)           //[0, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 2]perform mac across 4 streams with saem weights
       } {
           w0.tmp         = vmem(ptr_w++#1)           //[0, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 3]perform mac across 4 streams with saem weights
           z = vmem(ptr_x1+#0)                        //[0, 3]load next stride=1 128 or stride=2 64 bytes
       }:endloop0
/*=============================================================================*/
       {   w0.tmp     = vmem(ptr_w++#1)               //[0, 4]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 4]perform mac across 4 streams with saem weights
           p0 = cmp.eq(filt_cnt, #0)                  //[Kernel]count filt height itns.
           if(p0.new) ptr_x0 = memw(prep_ptr++#1<<3)  //[Width]initialize activation offset
       } {
           loop0(.L_filt_width, filt_width)           //[P, 0]ki is k1/32 - 0
           w0.tmp     = vmem(ptr_w++#1)               //[0, 5]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 5]perform mac across 4 streams with saem weights
           if(p0) ptr_x0 = add(ptr_xij, ptr_x0)       //[Width]create next activation ptr
       } {
           w0.tmp     = vmem(ptr_w++#1)               //[0, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)    //[0, 6]perform mac across 4 streams with saem weights
           ptr_x2 = and(ptr_x0, #-128)                //[P, 0]make loads aligned to 128 zero out bits 0-6
           filt_cnt = add(filt_cnt, #-1)              //decrement filt height cnt
       } {
           w0.tmp     = vmem(ptr_w++#1)               //[0, 7]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )    //[0, 7]perform mac across 4 streams with saem weights
           z = vmem(ptr_x2+#0)                        //[P, 3]load 0-127
           ptr_x1  = add(ptr_x0, stride_w)            //[P, 1]setup initial pointer
       } {
           ptr_x0 = add(ptr_x0, in_width_32)          //[P, 2]move to next even line of filter activations
           z = vmem(ptr_x2+#1)                        //[P, 4]load 128-255
           if(p0) ptr_w = memw(prep_ptr+#-1<<2)       //[Width]initialize filter pointer
       }:endloop1
/*============================================================================*/
       {   s0.w = vasr(s0.w, c8)                      //
           loop1(.L_filt_height, filt_height)         //[Width]for(filt_y=0;filt_y<height*in_depth/32;filt_y++){
           prod_cnt = add(prod_cnt, #-1)              //[Width]net partial product
           p0 = cmp.eq(prod_cnt, #3)                  //[Width]end of partial products?
       } {
           s1.w = vasr(s1.w, c8)                      //
           if (!p3) s0 = vzero                        //out_left_junk, so zero out the computation
       } {
           s2.w = vasr(s2.w, c8)                      //
           p1 = cmp.eq(prod_cnt, #0)                  //[Width]end of partial products?
           filt_cnt = add(filt_height, #-1)           //[Kernel]
       } {
           s3.w = vasr(s3.w, c8)                      //
           c8 = mux(p0,#8,#0)                         //
           if(!p1) jump:t .L_products                 //[Width]next product
       }
/*=============================================================================*/
   .balign 64
       {
#if !defined(FAST_16B_CONV)
           c8 = #8                                    //[Post-Width]8bit shift for lower products
#endif
           ptr_xij = add(ptr_xij, next_outputs)       //[Post-Width]reset data ptr to next 4
           sk = #0                                    //
#if defined(SPLIT_OUTPUT)
           y32.uh = vpack(y3.w, y2.w):sat             //[Post-Width-P]pack low 16bits together
#else
           x3210o.uh = vpack(y3.w, y2.w):sat          //[Post-Width]pack low 16bits together
#endif
       } {
#if defined(SPLIT_OUTPUT)
           x3210e.b = vpacke(y32.h, y10.h)            //[Post-Width-P]
#endif
           if (q0) s0.w += wsum.w                     //[Post-Width]
           s1.w = vadd(s1.w, wsum.w)                  //[Post-Width]
           s2.w = vadd(s2.w, wsum.w)                  //[Post-Width]
       } {
#if defined(SPLIT_OUTPUT)
           x3210o.b = vpacko(y32.h, y10.h)            //[Post-Width-P]
#endif
           p0 = cmp.ge(col_count,#2)                  //
           p1 = cmp.ge(col_count,#3)                  //
           col_count=add(col_count,#-4)               //[Post-Width]count -=4 ptr_z += 128
       } {
           s3.w = vadd(s3.w, wsum.w)                  //[Post-Width]
           maxe.w = vmax(maxe.w, s0.w)                //[Post-Width]see if s0 is max
           mine.w = vmin(mine.w, s0.w)                //[Post-Width]see if s0 is min
           s0.w = vasl(s0.w, recip_shift)             //[Post-Width]
       } {
           if (p0) sk = s1                            //
           p0 = cmp.ge(col_count,#(4-4))              //
           s1.w = vasl(s1.w, recip_shift)             //[Post-Width]
#if defined(SPLIT_OUTPUT)
           xout = vlalign(x3210e,x3210e_prev,out_align)//[Post-Width-P]
#else
           xout = vlalign(x3210e,x3210o_prev,out_align)//[Post-Width-P]
#endif
       } {
           maxe.w = vmax(maxe.w, sk.w)                //[Post-Width]
           mine.w = vmin(mine.w, sk.w)                //[Post-Width]see if z0 is max
           if (p1) sk = s2                            //
           s2.w = vasl(s2.w, recip_shift)             //[Post-Width]
       } {
           maxe.w = vmax(maxe.w, sk.w)                //[Post-Width]
           mine.w = vmin(mine.w, sk.w)                //[Post-Width]see if z0 is max
           if(p2)vmem(ptr_z0++#1):nt = xout           //[Post-Width-P]store 2nd 32bytes
           if (p0) sk = s3                            //
       } {
           maxe.w = vmax(maxe.w, sk.w)                //[Post-Width]
           mine.w = vmin(mine.w, sk.w)                //[Post-Width]see if z0 is max
           s3.w = vasl(s3.w, recip_shift)             //[Post-Width]
#if !defined(SPLIT_OUTPUT)
           p0 = cmp.eq(col_count,out_width)
#endif
       } {
#if defined(SPLIT_OUTPUT)
           xout = vlalign(x3210o,x3210o_prev,out_align)//[Post-Width-P]align old and new data
#else
           xout = vlalign(x3210o,x3210e,out_align)    //[Post-Width]
#endif
           y0.w = vmpye(s0.w, recipvec.uh)            //[Post-Width](s2 * recip + rnd)>>31
#if defined(FAST_16B_CONV)
           ptr_w_ptr_x0 = memd(sp+#64)                //[Pre-Width]initialize filter pointer & activation offset
           prep_ptr = add(sp, #(64+8))                //[Pre-Width]ptr to pre computed ptr list
#else
           ptr_w_ptr_x0 = memd(sp+#56)                //[Pre-Width]initialize filter pointer & activation offset
           prep_ptr = add(sp, #(56+8))                //[Pre-Width]ptr to pre computed ptr list
#endif
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):RSS         //[Post-Width<<1:rnd:sat:shift
           ptr_x0 = add(ptr_xij, ptr_x0)              //[Pre-Width]odd activations + in_Depth_32
#if defined(FAST_16B_CONV)
           prod_cnt = #3                              //[Pre-Width]total 3 partial products
#else
           prod_cnt = #4                              //[Pre-Width]total 4 partial products
#endif
           s0 = vcrnd                                 //[Pre-Width]accumulator 0
       } {
#if defined(SPLIT_OUTPUT)
           if(p2)vmem(ptr_z1++#1):nt = xout           //[Post-Width-P]store 2nd 32bytes
#else
           if(!p0)vmem(ptr_z0++#1):nt = xout          //[Post-Width-P]store 2nd 32bytes
#endif
           x3210e_prev = x3210e                       //[Post-Width-P]save data for next output align
           y1.w = vmpye(s1.w, recipvec.uh)            //[Post-Width](s2 * recip + rnd)>>31
           ptr_x2 = and(ptr_x0, #-128)                //[Pre-Width]make loads aligned to 128 zero out bits 0-6
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):RSS         //[Post-Width<<1:rnd:sat:shift
           s1 = vcrnd                                 //[Pre-Width]accumulator 1
           z = vmem(ptr_x2+#0)                        //[Pre-Width][Pheight]load 0-127
           p2 = p3                                    //[Post-Width]
       } {
           y2.w = vmpye(s2.w, recipvec.uh)            //[Post-Width](s2 * recip + rnd)>>31
           z = vmem(ptr_x2+#1)                        //[Pre-Width]load 128-255
           ptr_x1 = add(ptr_x0, stride_w)             //[Pre-Width]setup initial pointer
           scratch = scratch0                         //[Pre-Width]temp accumuator buffer
       } {
#if defined(SPLIT_OUTPUT)
           y10.uh = vpack(y1.w, y0.w):sat             //[Post-Width]pack low 16bits together
#else
           x3210e.uh = vpack(y1.w, y0.w):sat          //[Post-Width]pack low 16bits together
#endif
           y2.w+= vmpyo(s2.w, recipvec.h):RSS         //[Post-Width]<<1:rnd:sat:shift
           ptr_x0 = add(ptr_x0, in_width_32)          //[Pre-Width], 0]move to next even line of filter activations
           p1 = cmp.gt(col_count, #0)                 //[Post-Width]compare for branch
       } {
           x3210o_prev = x3210o                       //[Post-Width-P]save data for next output align
           y3.w = vmpye(s3.w, recipvec.uh)            //[Post-Width](s2 * recip + rnd)>>31
           q0 = vcmp.eq(x3210o.w,x3210o.w)            //
           p3 = cmp.eq(r0, r0)                        //[Post-Width]set to true
       } {
           y3.w+= vmpyo(s3.w, recipvec.h):RSS         //[Post-Width]<<1:rnd:sat:shift
           s2 = vcrnd                                 //[Pre-Width]accumulator 2]
           s3 = vcrnd                                 //[Pre-Width]accumulator 3]
           if(p1) jump:t .L_width                     //[Post-Width]
       } //cols per line kernel loop width
/*=============================================================================*/
       {   memw(sp+#48) += in_width_stride_h_depth    //[Height]ptr_x+=2*in_width*stride_h*in_depth)
           p0 = !cmp.eq(skip_col, #0)                 //[Height]
           p3 = cmp.gt(out_align, #0)                 //
           out_height = add(out_height, #-1)          //
       } {
#if defined(SPLIT_OUTPUT)
           y32.uh = vpack(y3.w, y2.w):sat             //[Post-Width]pack low 16bits together
#else
           x3210o.uh = vpack(y3.w, y2.w):sat          //[Post-Width]pack low 16bits together
#endif
           memw(sp+#96) += out_width_depth            //[Height]ptr_zi = add(ptr_zi, out_width_depth)
           p0 = or(p0, p3)                            //
       } {
#if defined(SPLIT_OUTPUT)
           memw(sp+#100) += out_width_depth           //[Height]ptr_zi = add(ptr_zi, out_width_depth)
#else
           memw(sp+#96) += out_width_depth            //[Height]ptr_zi = add(ptr_zi, out_width_depth)
#endif
           col_count = memw(sp+#(SPVEC+17)<<2)        //read width of activations
       } {
#if defined(SPLIT_OUTPUT)
           x3210e.b = vpacke(y32.h, y10.h)            //[Post-Width]
#endif
           q0 = or(q1, q1)                            //
       } {
#if defined(SPLIT_OUTPUT)
           x3210o.b = vpacko(y32.h, y10.h)            //[Post-Width]
#endif
           p1 = cmp.eq(out_height, #0)                //[Height]
       } {
#if defined(SPLIT_OUTPUT)
           xout = vlalign(x3210e,x3210e_prev,out_align)//[Post-Width]
#else
           xout = vlalign(x3210e,x3210o_prev,out_align)//[Post-Width]
#endif
           if (p2) vmem(ptr_z0++#1):nt = xout.new      //[Post-Width]store 2nd 32bytes
       } {
#if defined(SPLIT_OUTPUT)
           xout = vlalign(x3210o,x3210o_prev,out_align)//[Post-Width]
           if (p2) vmem(ptr_z1++#1):nt = xout.new      //[Post-Width]store 2nd 32bytes
#else
           xout = vlalign(x3210o,x3210e,out_align)     //[Post-Width]
           vmem(ptr_z0++#1):nt = xout.new              //[Post-Width]store 2nd 32bytes
           p0 = cmp.eq(out_align, #0)                 //if no alignment enable store
#endif
       } {
#if defined(SPLIT_OUTPUT)
           xout = vlalign(x3210e, x3210e, out_align)  //[Height]
           if( p0) vmem(ptr_z0+#0):nt = xout.new      //[Height]store 2nd 32bytes
       } {
           xout = vlalign(x3210o, x3210o, out_align)  //[Height]
           if( p0) vmem(ptr_z1+#0):nt = xout.new      //[Height]store 2nd 32bytes
#else
           xout = vlalign(x3210o,x3210o,out_align)    //[Post-Width]
           if(!p0) vmem(ptr_z0+#0):nt = xout.new      //[Height]store 2nd 32bytes
#endif
           if(!p1) jump:t .L_height                   //[Height]
       }//end lines per block
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)                       //restore r16, r17from stack
           ptr_minmax = memw(sp+#(SPVEC+24)<<2)       //ptr pre computed max value in output
       } {
           vmem(ptr_minmax+#0) = maxe                 //[E, 0]32max
           r19:18 = memd(sp+#8)                       //restore r18,r19
       } {
           vmem(ptr_minmax+#1) = mine                 //[E, 0]32min
           r21:20 = memd(sp+#16)                      //restore r20,r21
       } {
           r23:22 = memd(sp+#24)                      //restore r22,r23
           r25:24 = memd(sp+#32)                      //restore r24,r25
       } {
           r27:26 = memd(sp+#40)                      //restore r26,r27
           dealloc_return                             //restore fram and return
       }
.L_end:
/*=============================================================================*/
      .size gvconv2db2b2b2_d32_asm, .L_end-gvconv2db2b2b2_d32_asm
/*=============================================================================*/
