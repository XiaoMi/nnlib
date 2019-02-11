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
/*  DJ                  05/17/17     speciaized version with hstride = 1 */
/*======================================================================*/
#if 0
#endif
/*=============================================================================*/
        .text
        .file "gvconv2dbbb_d32_s1_h_v66.S"
        .global gvconv2dbbbs1_v66_asm
        .balign 32
        .type  gvconv2dbbbs1_v66_asm, @function
gvconv2dbbbs1_v66_asm:
/*=============================================================================*/
/*=============================================================================*/
#define ptr_xi                r0    //data    aligned 128
#define ptr_wi                r1    //weights aligned 128
#define ptr_zi                r2    //results aligned 128
#define in_width              r3    //(pad_l+in_width+pad_r) => 4 %4
#define out_width_stride_depth r4   //value in bytes to get to next full out row
#define out_width             r5    //out_width_pad 
#define stride_h_w            r26   //0 stride_height|stride_width 
#define in_depth              r27   //1 %32
#define filt_width            r17   //2 >= 1
#define filt_height           r8    //3 >= 1filt_height lines per filter
#define out_height            r9    //4 >= 1 number of vertical lines to perform
#define ptr_filtsum           r10   //5 aligned 128
#define ptr_suma              r11   //6 not used in this architeture
#define next_suma             r13   //7 not used in this architecture
#define ptr_max               r12   //8 aligned 128
#define recip_level           r14   //9  recip is 31bit unsigned 0x7f800000000LL / max
#define out_align             r1    //10 0, 32, 64, 96
/*=============================================================================*/
#define c4                    r3    //
#define stride_h              r26   //0 stride_height|stride_width 
#define in_width_stride_h_depth r28 //in_width * stride_h * in_depth for next output
#define ptr_x0                r6    //r6 spare
#define ptr_x1                r7    //
#define ptr_x1_ptr_x0         r7:6  //
#define stride_w              r18   //stride width =1
#define next_outputs          r19   //jump to input ptr for next set of outputs
#define ptr_w                 r20   //
#define in_width_32           r22   //
#define ptr_x2                r23   //
#define ptr_z                 r24   //
#define col_count             r25   //
#define col_count_ptr_z       r25:24//packed together to save a slot
#define cm4_c128              r11:10//combine constants -4, 128
#define filt_cnt              r18
#define ptr_x0_               r12
#define ptr_x1_               r13   //15 spare 14 spare
#define ptr_x1_ptr_x0_        r13:12
#define z_ptr                 r3 
#define c8_c96                r15:14
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
#define x3210                 v6    //
#define x3_prev               v16   //previous value
#define xout                  v17   //realigned out
#define y0                    v8    //
#define y1                    v9    //
#define y2                    v10   //
#define y3                    v11   //
#define wsum                  v14   //initialzed to in_offsey*wsum + biasoffset
#define maxomaxe              v13:12//
#define maxe                  v12   //
#define maxo                  v13   //
#define minomine              v19:18//
#define mine                  v18   //
#define mino                  v19   //
#define biasvec               v16   //
#define recipvec              v15   //
#define RSS     <<1:rnd:sat:shift   //unverbose the insturction
/*=============================================================================*/
       {   allocframe(#56)                            //0th entry on stack is (56+8)/4 =16 ints
       } {
           memd(sp+#0)  = r17:16                      //
           memd(sp+#8)  = r19:18                      //
       } {
           memd(sp+#16) = r21:20                      //
           memd(sp+#24) = r23:22                      //
           r20 = ##0x7fffffff                         //
       } {
           memd(sp+#32) = r25:24                      //
           memd(sp+#40) = r27:26                      //
           mine = vsplat(r20)                         //
           maxe = #0                                  //
       } {
           maxe.w = vsub(maxe.w, mine.w)              //
           stride_h_w = memw(sp+#16<<2)               //extract strides h + w 
           in_depth = memw(sp+#17<<2)                 //
           ptr_w = ptr_wi                             //[P,0]ptr_y=ptr_yi initialize filter pointer
       } {
           filt_height = memw(sp+#19<<2)              //extract filt_height 
           filt_width = memw(sp+#18<<2)               //extract filt_width
       } {   
           filt_height = mpy(filt_height.L,in_depth.L)//filt_height*in_depth
           out_height = memw(sp+#20<<2)               //number of output lines
           recip_level = memw(sp+#25<<2)              // 
       } {
           filt_height = lsr(filt_height, #5)         //filt_height * in_depth / 32
           ptr_filtsum = memw(sp+#21<<2)              //ptr pre computed weight sum
           recipvec = vsplat(recip_level)             //
           memw(sp+#52) = ptr_wi                      //save wi for someone else
       } {
           memw(sp+#48) = r28
           in_width_32 = asl(in_width, #5)            //32 * in_width d32 line
           in_width_stride_h_depth = mpy(stride_h_w.H, in_depth.L)
       } {
           next_outputs=mpyi(filt_height,in_width)    //filt_height*in_width*in_depth
           out_align = memw(sp+#26<<2)                //
           wsum = vmem(ptr_filtsum+#0)                //
           filt_height = mpyi(filt_height, filt_width)//2d filter
       } {
           next_outputs = add(next_outputs, #-4)      //1,2
           loop0(.L_filt, filt_height)                //[P,0]for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1)
           c8_c96 = combine(#8, #96)                  //
       } {
           cm4_c128 = combine(#-4, #128)              //combine 2 constants
           next_outputs = asl(next_outputs, #5)       //(flt_hight*in_width*in_depth/32-4*stride)*32
           in_width_stride_h_depth=mpyi(in_width,in_width_stride_h_depth) //total vertical stride bytes
       }
/*=============================================================================*/
       {   ptr_x0 = ptr_xi                             //ptr_xi 
           out_height = add(out_height, #-1)           //
           ptr_z = add(ptr_zi, #-128)                  //
           filt_cnt = filt_width
       } {
           p2 = !cmp.eq(r0, r0)                        //force p2 off
           col_count = out_width                       //out_width 
           ptr_xi= add(ptr_xi,in_width_stride_h_depth) //ptr_x+=in_width*stride_h*in_depth)
           p0 = cmp.eq(out_align, #0)                  //if no alignment enable store
       } {
           ptr_x1 = add(ptr_x0, #100)                    //[Pheight]setup initial pointer
           z = vmem(ptr_x0+#0)                         //[Pheight]load 0-127
           s3s2 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
           s1s0 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
       }
/*=============================================================================*/
        .balign 64
.L_height:
.L_width:
.L_filt:
       {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 6]perform mac across 4 streams with saem weights
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 6]perform mac across 4 streams with saem weights
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 6]perform mac across 4 streams with saem weights
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 7]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 7]perform mac across 4 streams with saem weights
           filt_cnt = add(filt_cnt, #-1)
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 0]perform mac across 4 streams with saem weights
           p3 = cmp.eq(filt_cnt, #0)                   //[0, 0]ki is k1/32 - 0
           if(p3.new) ptr_x0 = add(ptr_x0, in_width_32)//[0, 0]move to next line ptr_y keeps going
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 1]perform mac across 4 streams with saem weights
           ptr_x1_ptr_x0_= vaddw(ptr_x1_ptr_x0,c8_c96) //0, 1]ptr_x1_=add(ptr_x1,#8)||/ptr_x0_=add(ptr_x0, #96)
           if(p3) filt_cnt = filt_width                //[0, 1]
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 2]perform mac across 4 streams with saem weights
           z_ptr   = mux(p3, ptr_x0,  ptr_x1_)         //[0, 2]
           ptr_x1_ = mux(p3, ptr_x0_, ptr_x1_)         //[0, 2]
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )     //[0, 3]perform mac across 4 streams with saem weights
           z = vmem(z_ptr+#0)                          //[0, 3]load 0-127 bytes into z buffer
           ptr_x1 = add(ptr_x1_, #4)                   //[0, 3]reset ptr for next row of filter taps
       }:endloop0
/*=============================================================================*/
       {   maxe.w = vmax(maxe.w, s0.w)                 //[E0, 0]see if s0 is max
           mine.w = vmin(mine.w, s0.w)                 //[E0, 0]see if s0 is min
           loop0(.L_filt, filt_height)                 //[E0, 0]for(filt_y=0;filt_y<height*in_depth/32;filt_y++){
           x1.h = vpacke(y1.w, y0.w)                   //[E1, 0]packe low 16bits together
       } {
           ptr_x0 = sub(ptr_x0, next_outputs)          //[E0, 1]reset data ptr to next 4
           y0.w = vmpye(s0.w, recipvec.uh)             //[E0, 1](s2 * recip + rnd)>>31
           x3.h = vpacke(y3.w, y2.w)                   //[E1, 1]pack low 16bits together
           col_count_ptr_z=vaddw(col_count_ptr_z,cm4_c128)//count -=4 ptr_z += 128
       } {
           maxe.w = vmax(maxe.w, s3.w)                 //[E0, 2]
           mine.w = vmin(mine.w, s3.w)                 //[E0, 2]see if z0 is max
           y0.w+= vmpyo(s0.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 2]
           filt_cnt = filt_width
       } {
           y1.w  = vmpye(s1.w, recipvec.uh)            //[E0, 3](s2 * recip + rnd)>>31
           maxe.w = vmax(maxe.w, s1.w)                 //[E0, 3]
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255  and pack low 8bits
           ptr_x1 = add(ptr_x0, #100)  //4                   //setup initial pointer
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 4](s2 * recip + rnd)>>31
           mine.w = vmin(mine.w, s1.w)                 //[E0, 4]see if z0 is max
           maxe.w = vmax(maxe.w, s2.w)                 //[E0, 4]
           p1 = cmp.eq(col_count, #0)                  //[E0, 4]compare for branch
       } {
           y2.w  = vmpye(s2.w, recipvec.uh)            //[E0, 5](s2 * recip + rnd)>>31
           mine.w = vmin(mine.w, s2.w)                 //[E0, 5]see if z0 is max
           s0 = wsum                                   //[E0, 5]initialize accumulator 0
           ptr_w = memw(sp+#52)                        ////[E0, 5]ptr_w=ptr_wi init filter pointer
       } {
           y2.w+= vmpyo(s2.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 6](s2 * recip + rnd)>>31
           s1 = wsum                                   //[E0, 6]initialize accumulator 1
           xout = vlalign(x3210,x3_prev,out_align)     //[E1, 6]
           if(p2)vmem(ptr_z+#-1) = xout.new            //[E1, 6]store 2nd 32bytes
       } {
           y3.w  = vmpye(s3.w, recipvec.uh)            //[E0, 7]#(s2 * recip + rnd)>>31
           x3_prev = x3210                             //[E1, 7]save data for next output align
           p2 = p0                                     //[E1, 7]update predicate piplione
           p0 = cmp.eq(r0, r0)                         //[E1, 7]set to true
       } {
           y3.w+= vmpyo(s3.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 8](s2 * recip + rnd)>>31
           s3s2 = vcombine(wsum, wsum)                 //[E0, 8]initialize accumulator 2,3
           if(!p1) jump:t .L_width                     //[E1, 8]
           z = vmem(ptr_x0+#0)                         //pre load 0-127 for next row of filter
       }//cols per line kernel loop width
/*=============================================================================*/
       {   x1.h = vpacke(y1.w, y0.w)                   //[E1, 0]#>>16  
           p1 = cmp.eq(out_height, #0)                 //EE
           p2 = !cmp.eq(r0, r0)                        //[Pheight]force p2 off
       } {
           x3.h = vpacke(y3.w, y2.w)                   //[E1, 1]#sat8  <0, >255 
           out_height = add(out_height, #-1)           //Prolog width
           col_count = out_width                       //[Pheight]out_width 
       } {
           ptr_z = add(ptr_z, #128)                    //[E1, 3]unconditional ptr increment
           ptr_x0 = ptr_xi                             //Prolog width ptr_xi 
           p0 = cmp.eq(out_align, #0)                  //[Pheight]if no alignment enable store
       } {
           ptr_xi = add(ptr_xi,in_width_stride_h_depth)//[Pheight]ptr_x+=in_width*stride_h*in_depth)
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255 
           filt_cnt = filt_width
       } {
           ptr_zi = add(ptr_zi, out_width_stride_depth)//EEnext out line for this depth segment
           ptr_x1 = add(ptr_x0, #100) //4                   //[Pheight]setup initial pointer
           z = vmem(ptr_x0+#0)                         //[Pheight]load 0-127
       } {
           xout = vlalign(x3210, x3_prev, out_align)   //[E1, 6]
           vmem(ptr_z+#-1) = xout.new                  //[E1, 6]store 2nd 32bytes
           ptr_z = add(ptr_zi, #-128)                  //
           if(!p1) jump:t .L_height                    //EE
       }//end lines per block//last cols per line
/*=============================================================================*/
       {   c4 = #4                                     //constant
           ptr_max = memw(sp+#24<<2)                  //ptr pre computed max value in output
       } {
           loop0(.L_minmax, #4)                        //[P, 0]
           maxomaxe=vshuff(maxe,maxe,c4)               //[0, 0]find max among 32values
       } 
/*=============================================================================*/
.L_minmax:
       {   minomine=vshuff(mine,mine,c4)               //[0, 1]find min among 32values
           maxe.w = vmax(maxo.w, maxe.w)               //[0, 1]
           c4 = add(c4, c4)                            //[0, 1]
       } {
           mine.w = vmin(mino.w, mine.w)               //[0, 2]
           maxomaxe=vshuff(maxe,maxe,c4)               //[1, 0]
       }:endloop0
       {   minomine=vshuff(mine,mine,c4)               //[1, 1]
           maxe.w = vmax(maxo.w, maxe.w)               //[1, 1]
           vmem(ptr_max+#0) = maxe.new                 //[E, 0]32max
       } {
           mine.w = vmin(mino.w, mine.w)               //[1, 2]
           vmem(ptr_max+#1) = mine.new                 //[E, 0]32min
       }
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)                        //restore r16, r17from stack
           r19:18 = memd(sp+#8)                        //restore r18,r19
       } {
           r21:20 = memd(sp+#16)                       //restore r20,r11
           r23:22 = memd(sp+#24)                       //restore r22,r13
       } {    
           r25:24 = memd(sp+#32)                       //restore r24,r15
           r27:26 = memd(sp+#40)                       //restore r26,r17
       } {
           r28 = memw(sp+#48)
       } {
           dealloc_return                              //restore fram and return
       }
.L_end:
/*=============================================================================*/
      .size gvconv2dbbbs1_v66_asm, .L_end-gvconv2dbbbs1_v66_asm
/*=============================================================================*/
