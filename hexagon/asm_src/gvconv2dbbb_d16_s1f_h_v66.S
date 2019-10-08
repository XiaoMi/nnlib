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
/*  DJ                  05/17/17     speciaized version with hstride = 1 */
/*======================================================================*/
#if 0
#endif
/*=============================================================================*/
        .text
        .file "gvconv2dbbb_d16_s1f_h_v66.S"
        .global gvconv2dbbbs1_d16_v66_asm
        .balign 32
        .type  gvconv2dbbbs1_d16_v66_asm, @function
gvconv2dbbbs1_d16_v66_asm:
/*=============================================================================*/
/*=============================================================================*/
#define ptr_xi                r0    //data    aligned 128
#define ptr_wi                r1    //weights aligned 128
#define ptr_zi                r2    //results aligned 128
#define in_width              r3    //(pad_l+in_width+pad_r) => 4 %4
#define weight_batch_size     r3    //
#define weight_batch_size_ptr_zi r3:2  
#define out_width_stride_depth r4   //value in bytes to get to next full out row
#define out_width             r5    //out_width_pad 
#define stride_h_w            r26   //0 stride_height|stride_width 
#define in_depth              r27   //1 %32
#define in_depth_stride_h_w   r27:26
#define filt_width            r8    //2 >= 1
#define filt_height           r9    //3 >= 1filt_height lines per filter
#define filt_height_width     r9:8  //
#define out_height            r17   //4 >= 1 number of vertical lines to perform
#define ptr_filtsum           r27   //5 aligned 128
#define ptr_max               r21   //6 aligned 128
#define recip_level           r23   //7  recip is 31bit unsigned 0x7f800000000LL / max
#define out_align             r1    //8 0, 32, 64, 96
#define skip_col              r7    //9
#define out_next_d32          r7    //10
#define nslice                r12   //11
#define recip_shamt           r12   //12
#define STP                   r21   //
#define AEQ0                  r10   //
/*=============================================================================*/
#define filt_width_in         r26   //
#define filt_depth_width      r11   //total depth 32 iterations of each h filter
#define in_width_stride_h_depth r28 //in_width * stride_h * in_depth for next output 
#define ptr_x0                r6    //
#define ptr_x1                r7    //
#define ptr_x1_ptr_x0         r7:6  //
#define stride_w              r18   //stride width =1
#define next_outputs          r19   //jump to input ptr for next set of outputs
#define ptr_w                 r20   //
#define in_width_32           r22   //
#define ptr_z                 r24   //
#define col_count             r25   //
#define filt_cnt              r18
#define ptr_x0_               r12
#define ptr_x1_               r13   //
#define ptr_x1_ptr_x0_        r13:12
#define z_ptr                 r3 
#define c8_c96                r15:14
#define c8                    r15
#define filt_width512         r16   //corrected to use ptr_w as a counter for fused loop
#define c640                  r14
#define lmask                 r23   //
#define BASE                  72
#define PRED3_0               C4    //
/*=============================================================================*/
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug vec reg
#define PV(VSRC) .word (0x1DFFE020+VSRC)
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
#define maxe                  v12   //
#define mine                  v18   //
#define biasvec               v16   //
#define recipvec              v15   //
#define recip_sh_vec          v20   //
#define RSS     <<1:rnd:sat:shift   //unverbose the insturction
/*=============================================================================*/
       {   filt_height_width = memd(sp+#8)            //
           sp = add(sp,#-BASE)                          //
           out_width = lsr(out_width, #2)             //4 outputs at once
       } {
           memd(sp+#40) = r27:26                      //
           in_depth_stride_h_w = memd(sp+#(BASE+0))   //
           c640 = #640                                //
       } {
           memd(sp+#16) = r21:20                      //
           ptr_max = memw(sp+#(BASE+24))              //
           M1 = c640                                  //ptr_w ++ #5 equivalence
           filt_depth_width = lsr(in_depth, #5)       //filt_wiidth * in_depth / 32
       } {
           memd(sp+#0)  = r17:16                      //
           memd(sp+#8)  = r19:18                      //
           next_outputs = mpy(filt_height.L,in_depth.L)//filt_height*in_depth
           in_width_stride_h_depth = mpy(stride_h_w.H, in_depth.L)
       } {
           memd(sp+#24) = r23:22                      //
           recip_level = memw(sp+#(BASE+28))          //
           next_outputs = lsr(next_outputs, #5)        //filt_height*depth/32
           filt_depth_width = mpyi(filt_depth_width, filt_width)//2d filter
       } {
           memd(sp+#32) = r25:24                      //
           ptr_w = ptr_wi                             //[P,0]ptr_y=ptr_yi initialize filter pointer
           maxe = vmem(ptr_max+#0)                    //
       } {
           memw(sp+#48) = ptr_xi                      //
           in_width_stride_h_depth=mpyi(in_width,in_width_stride_h_depth) //total vertical stride bytes
           in_width_32 = asl(in_width, #5)            //32 * in_width d32 line
           mine = vmem(ptr_max+#1)                    //
       } {
           recipvec = vmem(recip_level++#1)           //
           next_outputs=mpyi(next_outputs,in_width)   //filt_height*in_width*in_depth
           weight_batch_size = mpyi(filt_depth_width,filt_height)
       } {
           memw(sp+#(BASE+28)) = recip_level          //
           r6 = #32                                   //
           recip_shamt = memw(sp+#(BASE+48))          //
       } {
           out_align = memw(sp+#(BASE+32))            //
           filt_depth_width = sub(filt_depth_width, filt_width)//-1 to allow for depth 16 pass
           weight_batch_size = asl(weight_batch_size,#10) // 32*filt_width*filt_height*in_depth
           filt_width_in = add(filt_width, #-1)       //account for peeling 1 iteration off
       } {
           recip_sh_vec = vsplat(recip_shamt)         //
           r6 -= lsr(out_align,#2)                    //
       } {
           r6 = and(r6, #0x1f)                        // 
           r7 = #-1                                   //
       } {
           r7 = asl(r7, r6)                           //
       } {
           memw(sp+#64) = r7                          //
           skip_col = memw(sp+#(BASE+36))             //
           r6 = #0x1f
       } {
           r6 &= asl(skip_col, #3)                    //
           r7 = #-1                                   //
       } {
           r7 = lsr(r7, r6)                           //
       } {
           memw(sp+#68) = r7                          //
       } {
           memw(sp+#60) = weight_batch_size           //
           filt_width = asl(filt_width, #10)          //convert for use in address counter sequence
           loop1(.L_filt_h, filt_height)              //for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1)
           AEQ0 = cmp.eq(out_align, #0)                 //if no alignment enable store
       } {
           next_outputs = add(next_outputs, #-4)      //1,2
           c8_c96 = combine(#8, #96)                  //
           filt_width = add(filt_width, #-128)        //1 cycle off from consumer to generator 
           filt_width512 = add(filt_width, #-512-512) //account for 512bytes per loop of w + precorrect for loop correction
       } {
           ptr_filtsum = memw(sp+#(BASE+20))            //ptr pre computed weight sum
           filt_depth_width +=add(filt_depth_width, #-1) //as 16 per loop so need 2x; peel 1 iteration off
           next_outputs = asl(next_outputs, #5)       //(flt_hight*in_width*in_depth/32-4*stride)*32
       }
/*=============================================================================*/
        .balign 64
.L_depth:
       {   
           ptr_xi = memw(sp+#48)                      //
           filt_cnt = add(filt_width512, ptr_w)       //add(filt_width, #-1) //ptr_w)
           STP = !cmp.eq(r0, r0)                      //force p2 off
           lmask = memw(sp+#64)                       //left mask
       } {
           col_count = out_width                      //setup first width loop
           memw(sp+#52) = ptr_w                       //save wi for someone else
           ptr_x0 = ptr_xi                             //ptr_xi 
           wsum = vmem(ptr_filtsum++#1)                //
       } {
           memw(sp+#56) = ptr_zi
           ptr_xi= add(ptr_xi,in_width_stride_h_depth) //ptr_x+=in_width*stride_h*in_depth)
           ptr_z = add(ptr_zi, #0)                     //
           ptr_x1 = add(ptr_x0, #100)                  //[Pheight]setup initial pointer
       } {
           out_height = memw(sp+#(BASE+16))              //number of output lines
           z = vmem(ptr_x0+#0)                         //[Pheight]load 0-127
           s3s2 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
           s1s0 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
       }
/*=============================================================================*/
        .balign 64
.L_height:
.L_width:
.L_filt_h:
       {   w0.tmp     = vmem(ptr_w++#1)                //[0, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 0]perform mac across 4 streams with saem weights
           loop0(.L_filt_w, filt_depth_width)          //for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1)
           nop                                         //
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 1]perform mac across 4 streams with saem weights
           z_ptr= add(ptr_x1,#8)                       //[0, 1]ptr_x1_=add(ptr_x1,#8)
           nop                                         //
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 2]perform mac across 4 streams with saem weights
           c8 = #8                                     //restore c8 to 8 for 1st loop
           filt_cnt = add(filt_cnt, #512)              //restore filt_cnt to this 1st loop
       } {
           w0.tmp    = vmem(ptr_w++#1)                 //[0, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )     //[0, 3]perform mac across 4 streams with saem weights
           z = vmem(z_ptr+#0)                          //[0, 3]load 0-127 bytes into z buffer
           ptr_x1 = add(z_ptr, #4)                     //[0, 3]reset ptr for next row of filter taps
       }
.L_filt_w:
       {   w0.tmp     = vmem(ptr_w++#1)                //[0, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 0]perform mac across 4 streams with saem weights
           p3 = cmp.eq(filt_cnt, ptr_w)                //[0, 0]ki is k1/32 - 0
           if(p3.new) ptr_x0 = add(ptr_x0, in_width_32)//[0, 0]move to next line ptr_y keeps going
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 1]perform mac across 4 streams with saem weights
           if(p3) filt_cnt = add(filt_width, ptr_w)    //[0, 1]
           ptr_x1_ptr_x0_= vaddw(ptr_x1_ptr_x0,c8_c96) //[0, 1]ptr_x1_=add(ptr_x1,#8)||/ptr_x0_=add(ptr_x0, #96)
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 2]perform mac across 4 streams with saem weights
           z_ptr   = mux(p3, ptr_x0,  ptr_x1_)         //[0, 2]
           ptr_x1_ = mux(p3, ptr_x0_, ptr_x1_)         //[0, 2]
       } {
           w0.tmp    = vmem(ptr_w++#1)                 //[0, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )     //[0, 3]perform mac across 4 streams with saem weights
           z = vmem(z_ptr+#0)                          //[0, 3]load 0-127 bytes into z buffer
           ptr_x1 = add(ptr_x1_, #4)                   //[0, 3]reset ptr for next row of filter taps
       }:endloop0
/*=============================================================================*/
       {   loop0(.L_filt_d16, filt_width_in)         //for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1)
           filt_cnt = add(filt_cnt, #-512)
           w0.tmp     = vmem(ptr_w++#1)                //[0, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 0]perform mac across 4 streams with saem weights
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 1]perform mac across 4 streams with saem weights
           z_ptr= add(ptr_x1,#24)                       //[0, 1]ptr_x1_=add(ptr_x1,#8)
           nop
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 2]perform mac across 4 streams with saem weights
           c8 = #24                                    //8 + 16 advance decision by 1/2 loop
           nop
       } {
           w0.tmp    = vmem(ptr_w++M1)                 //[0, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )     //[0, 3]perform mac across 4 streams with saem weights
           z = vmem(z_ptr+#0)                          //[0, 3]load 0-127 bytes into z buffer
           ptr_x1 = add(z_ptr, #4 )                     //[0, 3]reset ptr for next row of filter taps
       }
.L_filt_d16:
       {   w0.tmp     = vmem(ptr_w++#1)                //[0, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 0]perform mac across 4 streams with saem weights
           p3 = cmp.eq(filt_cnt, ptr_w)                //[0, 0]ki is k1/32 - 0
           if(p3.new) ptr_x0 = add(ptr_x0, in_width_32)//[0, 0]move to next line ptr_y keeps going
       } {
           if(p3) filt_cnt = add(filt_width, ptr_w)    //[0, 1]
           w0.tmp     = vmem(ptr_w++#1)                //[0, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 1]perform mac across 4 streams with saem weights
           ptr_x1_ptr_x0_= vaddw(ptr_x1_ptr_x0,c8_c96) //[0, 1]ptr_x1_=add(ptr_x1,#8)||/ptr_x0_=add(ptr_x0, #96)
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[0, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[0, 2]perform mac across 4 streams with saem weights
           z_ptr   = mux(p3, ptr_x0,  ptr_x1_)         //[0, 2]
           ptr_x1_ = mux(p3, ptr_x0_, ptr_x1_)         //[0, 2]
       } {
           w0.tmp    = vmem(ptr_w++M1)                 //[0, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )     //[0, 3]perform mac across 4 streams with saem weights
           z = vmem(z_ptr+#0)                          //[0, 3]load 0-127 bytes into z buffer
           ptr_x1 = add(ptr_x1_, #4 )                  //[0, 3]reset ptr for next row of filter taps
       }:endloop0:endloop1                             //
/*=============================================================================*/
           s0.w = vasl(s0.w, recip_sh_vec.w)           //
           s1.w = vasl(s1.w, recip_sh_vec.w)           //
           s2.w = vasl(s2.w, recip_sh_vec.w)           //
           s3.w = vasl(s3.w, recip_sh_vec.w)           //
       {   
           x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]packe low 16bits together
           loop1(.L_filt_h, filt_height)               //for(filt_y=0;filt_y<height*in_depth/32;filt_y++)
       } {
           PRED3_0 = lmask
           lmask = #-1                                 //
       } {
           ptr_x0 = sub(ptr_x0, next_outputs)          //[E0, 1]reset data ptr to next 4
           y0.w = vmpye(s0.w, recipvec.uh)             //[E0, 1](s2 * recip + rnd)>>31
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]pack low 16bits together
           col_count = add(col_count, #-1)             //
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 2]
           ptr_w = memw(sp+#52)                        ////[E0, 5]ptr_w=ptr_wi init filter pointer
       } {
           if(!p0) y0 = maxe                           //
       } {
           maxe.w = vmax(maxe.w, y0.w)                 //[E0, 0]see if s0 is max
           if(!p0) y0 = mine                           //
       } {
           mine.w = vmin(mine.w, y0.w)                 //[E0, 0]see if s0 is min
       } {
           y1.w  = vmpye(s1.w, recipvec.uh)            //[E0, 3](s2 * recip + rnd)>>31
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255  and pack low 8bits
           ptr_x1 = add(ptr_x0, #100)  //4                   //setup initial pointer
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 4](s2 * recip + rnd)>>31
       } {
           if(!p1) y1 = maxe                           //
       } {
           maxe.w = vmax(maxe.w, y1.w)                 //[E0, 3]
           if(!p1) y1 = mine                           //
       } {
           mine.w = vmin(mine.w, y1.w)                 //[E0, 4]see if z0 is max
           p1 = STP                                    //
       } {
           y2.w  = vmpye(s2.w, recipvec.uh)            //[E0, 5](s2 * recip + rnd)>>31
           s1s0 = vcombine(wsum, wsum)                 //[E0, 5]initialize accumulator 0
           filt_cnt = add(ptr_w,filt_width512)         //using the ptr_w as a counter
       } {
           y2.w+= vmpyo(s2.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 6](s2 * recip + rnd)>>31
           xout = vlalign(x3210,x3_prev,out_align)     //[E1, 6]
           if(p1)vmem(ptr_z++#1):nt = xout.new         //[E1, 6]store 2nd 32bytes
       } {
           if(!p2) y2 = maxe                           //
           p0 = cmp.eq(col_count, #1)                  //
       } {
           maxe.w = vmax(maxe.w, y2.w)                 //[E0, 4]
           if(!p2) y2 = mine                           //
       } {
           mine.w = vmin(mine.w, y2.w)                 //[E0, 5]see if z0 is max
           p1 = cmp.eq(col_count, #0)                  //
           if(p0) lmask = memw(sp+#68)                 //right mask
       } {
           y3.w  = vmpye(s3.w, recipvec.uh)            //[E0, 7]#(s2 * recip + rnd)>>31
           x3_prev = x3210                             //[E1, 7]save data for next output align
           STP = AEQ0                                  //[E1, 7]update predicate piplione
           AEQ0 = cmp.eq(r0, r0)                       //[E1, 7]set to true
       } {
           y3.w+= vmpyo(s3.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0,8](s2 * recip + rnd)>>31
           s3s2 = vcombine(wsum, wsum)                 //[E0, 8]initialize accumulator 2,3
           z = vmem(ptr_x0+#0)                         //pre load 0-127 for next row of filter
       } {
           if(!p3) y3 = maxe                           //
       } {
           maxe.w = vmax(maxe.w, y3.w)                 //[E0, 2]
           if(!p3) y3 = mine                           //
       } {
           mine.w = vmin(mine.w, y3.w)                 //[E0, 2]see if z0 is max
           if(!p1) jump .L_width                       //
       }//cols per line kernel loop width
/*=============================================================================*/
       {   x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]#>>16
           out_height = add(out_height, #-1)           //Prolog width
           STP = !cmp.eq(r0, r0)                       //[Pheight]force p2 off
       } {
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]#sat8  <0, >255
           p1 = !cmp.eq(out_height, #0)                //EE
           loop1(.L_filt_h, filt_height)               //for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1)
       } {
           ptr_x0 = ptr_xi                             //Prolog width ptr_xi 
           AEQ0 = cmp.eq(out_align, #0)                //[Pheight]if no alignment enable store
           col_count = out_width                       //outer loop set up width
       } {
           ptr_xi = add(ptr_xi,in_width_stride_h_depth)//[Pheight]ptr_x+=in_width*stride_h*in_depth)
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255 
           skip_col = memw(sp+#(BASE+36))              //
       } {
           p3 = tstbit(skip_col, #2)                  //[E1, 6] == 4
           ptr_zi = add(ptr_zi, out_width_stride_depth)//EEnext out line for this depth segment
           ptr_x1 = add(ptr_x0, #100)                  //[Pheight]setup initial pointer
           if (p1) z = vmem(ptr_x0+#0)                 //[Pheight]load 0-127
       } {
           xout = vlalign(x3210, x3_prev, out_align)   //[E1, 6]
           vmem(ptr_z+#0):nt = xout.new                //[E1, 6]store 2nd 32bytes
       } {
           xout = vlalign(x3210, x3210, out_align)     //[E1, 7]
           if(p3) vmem(ptr_z+#1):nt = xout.new         //[E1, 7]flush out last values
           ptr_z = add(ptr_zi, #0)                     //
           if (p1) jump:t .L_height                    //EE
       }//end lines per block//last cols per line
/*=============================================================================*/
       {   
           nslice = memw(sp+#(BASE+44))                 //
           weight_batch_size_ptr_zi = memd(sp+#56)     //
       } {
           nslice = add(nslice,#-1)                    //
           ptr_w = add(ptr_w,weight_batch_size)        //
       } {
           out_next_d32 = memw(sp+#(BASE+40))          //
           p1 = cmp.gt(nslice,#0)                      //
           recip_level = memw(sp+#(BASE+28))           //
       } {
           ptr_zi = add(ptr_zi,out_next_d32)           //
           if(p1) recipvec = vmem(recip_level++#1)     //
       } {
           memw(sp+#(BASE+28)) = recip_level           //
           memw(sp+#(BASE+44)) = nslice                //update nslice (weight chunks)
           if p1 jump .L_depth                         //
       }
/*=============================================================================*/
           ptr_max = memw(sp+#(BASE+24))              //
       {   r17:16 = memd(sp+#0)                        //restore r16,r17 from stack
           r19:18 = memd(sp+#8)                        //restore r18,r19
       } {
           vmem(ptr_max+#0) = maxe                     //[E, 0]32max
           r23:22 = memd(sp+#24)                       //restore r22,r23
       } {
           r25:24 = memd(sp+#32)                       //restore r24,r25
           vmem(ptr_max+#1) = mine                     //[E, 0]32min
       } {
           r21:20 = memd(sp+#16)                       //restore r20,r21
           r27:26 = memd(sp+#40)                       //restore r26,r27
           sp = add(sp,#BASE)                          //
           jumpr r31                                   //
       }
.L_end:
/*=============================================================================*/
      .size gvconv2dbbbs1_d16_v66_asm, .L_end-gvconv2dbbbs1_d16_v66_asm
/*=============================================================================*/
