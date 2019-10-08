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
#define out_next_row          r4    //value in bytes to get to next full out row
#define out_width             r5    //out_width_pad 
#define stride_h_w            r26   //0 stride_height|stride_width 
#define in_depth              r27   //1 %32
#define in_depth_stride_h_w   r27:26//
#define filt_width            r8    //2 >= 1
#define filt_height           r9    //3 >= 1filt_height lines per filter
#define filt_height_width     r9:8
#define out_height            r10   //4 >= 1 number of vertical lines to perform
#define ptr_filtsum           r11   //5 aligned 128
#define ptr_max               r21   //6 aligned 128 was 12
#define recip_level           r23   //7  recip is 31bit unsigned 0x7f800000000LL / max
#define out_align             r1    //8 0, 32, 64, 96
#define skip_col              r7    //21   
#define lmask                 r27
#define out_next_d32          r24
#define nslice                r25
#define nslice_out_next_d32   r25:24
#define recip_shift           r11
/*=============================================================================*/
#define stride_h              r26   //0 stride_height|stride_width 
#define in_next_rows          r28   //in_width * stride_h * in_depth for next output
#define ptr_x0                r6    //r6 spare
#define ptr_x1                r7    //
#define ptr_x1_ptr_x0         r7:6  //
#define stride_w              r18   //stride width =1
#define next_outputs          r19   //jump to input ptr for next set of outputs
#define ptr_w                 r20   //
#define in_width_32           r22   //
#define ptr_z                 r24   //
#define filt_cnt              r18
#define ptr_x0_               r12
#define ptr_x1_               r13   //
#define ptr_x1_ptr_x0_        r13:12
#define z_ptr                 r3 
#define c8_c96                r15:14
#define filt_width512         r16   //corrected to use ptr_w as a counter for fused loop
#define ptr_w_next            r17
#define ptr_z_next            r24
#define STQ                   r23   //shared with recip_level
#define AEQ0                  r21   //align equals 0  needs to be persistent               
#define PRED3_0               C4
#define w_count               r25
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
#define maxe                  v12   //
#define mine                  v13   //
#define wsum                  v14   //initialzed to in_offsey*wsum + biasoffset
#define recipvec              v15   //
#define recip_sh_vec          v19   //
#define RSS     <<1:rnd:sat:shift   //unverbose the insturction
#define STACK                 72
/*=============================================================================*/
       {   filt_height_width = memd(sp+#8)             // //recip_level_ptr_max = memd(sp+#24) 
           sp = add(sp,#-STACK)                        //
           out_width = lsr(out_width, #2)              //4 outputs at once
       } {
           memd(sp+#40) = r27:26                       //
           in_depth_stride_h_w = memd(sp+#(STACK+0))   // //recipvec = vsplat(recip_level)    
       } {
           memd(sp+#32) = r25:24                       //
           memd(sp+#16) = r21:20                       //
           c8_c96 = combine(#8, #96)                   //
       } {
           memd(sp+#24) = r23:22                       //
           filt_height = mpy(filt_height.L,in_depth.L) //filt_height*in_depth
           in_next_rows= mpy(stride_h_w.H, in_depth.L) //
           recip_level = memw(sp+#(STACK+28))          //
       } {
           recipvec = vmem(recip_level++#1)            //
           ptr_max = memw(sp+#(STACK+24))              //
           filt_height = lsr(filt_height, #5)          //filt_height * in_depth / 32
       } {
           memd(sp+#8)  = r19:18                       //
           memw(sp+#(STACK+28)) = recip_level          //
           loop1(.L_width, out_width)                  //
       } {
           memd(sp+#0)  = r17:16                       //
           ptr_w_next = ptr_wi                         //[P,0]ptr_y=ptr_yi initialize filter pointer
           out_align = memw(sp+#(STACK+32))            //
           r6 = #32
       } {
           recip_shift = memw(sp+#(STACK+48))          //
           r6 -= lsr(out_align,#2)                     //1/4 for bytes -> 0 -> 32, 32,-> 24, 64 -> 16, 96 -> 8
       } { 
           recip_sh_vec= vsplat(recip_shift)           //
           r6 = and(r6, #0x1f)                         //
           r7 = #-1
       } { 
           r7 = asl(r7, r6)                            // 
       } {
           memw(sp+#60) = r7                           //left mask
           skip_col = memw(sp+#(STACK+36))             //right mask - skip col
           r6 = #0x1f
       } {
           r6 &= asl(skip_col, #3)                     // (skip_col << 3) & 31
           r7 = #-1             
       } {  
           r7 = lsr(r7, r6)                            //
       } {
           memw(sp+#64) = r7                           //right mask
       } {
           ptr_filtsum = memw(sp+#(STACK+20))          //
           in_width_32 = asl(in_width, #5)             //32 * in_width d32 line
           in_next_rows=mpyi(in_width,in_next_rows)    //total vertical stride bytes
           maxe = vmem(ptr_max+#0)                     //
       } {
           out_next_d32 = memw(sp+#(STACK+40))         //
           next_outputs= mpyi(filt_height,in_width_32) //filt_height*in_width*in_depth
           filt_height = mpy(filt_height.L, filt_width.L):<<1 //2d filter
       } {
           memw(sp+#48) = ptr_xi                       //
           filt_width = asl(filt_width, #10)           //
           loop0(.L_filt, filt_height)                 //[P,0]for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1)
       } {
           next_outputs = add(next_outputs, #-128)     //(flt_hight*in_width*in_depth/32-4*stride)*32
           filt_width512 = add(filt_width, #-512)      //account for 512bytes per loop of w
           filt_width    = add(filt_width, #-128)      //1 cycle off from consumer to generator 
       } {
           mine = vmem(ptr_max+#1)                     //
           AEQ0 = cmp.eq(out_align, #0)                  //if no alignment enable store
       }
/*=============================================================================*/
        .balign 64
.L_depth:
       {   ptr_xi = memw(sp+#48)                       // restore ptr_xi
           out_height = memw(sp+#(STACK+16))           //number of output lines
           ptr_z_next = add(ptr_zi,out_next_d32)       //
           STQ = !cmp.eq(r0, r0)                        //force p2 off
       } {
           ptr_w = ptr_w_next                          // set ptr of weight
           ptr_x0 = ptr_xi                             //ptr_xi 
           wsum = vmem(ptr_filtsum++#1)                //
           lmask = memw(sp+#60)                        //left mask
       } {
           memw(sp+#52) = ptr_w_next                   //save wi for someone else
           memw(sp+#56) = ptr_z_next                   //
           ptr_x1 = add(ptr_xi, #100)                  //[Pheight]setup initial pointer
           filt_cnt = add(filt_width512, ptr_w)        //add(filt_width, #-1) //ptr_w)
       } {
           z = vmem(ptr_x0+#0)                         //[Pheight]load 0-127
           ptr_z = ptr_zi                              //
           s3s2 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
           s1s0 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
       }
           w_count = out_width
/*=============================================================================*/
        .balign 64
.L_height:
.L_width:
.L_filt:
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
       {
           PRED3_0 = lmask
           lmask = #-1                                 //next mask default is all on 0xffffffff
           x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]packe low 16bits together
       } { 
           loop0(.L_filt, filt_height)                 //[E0, 0]for(filt_y=0;filt_y<height*in_depth/32;filt_y++){
           s0.w = vasl(s0.w, recip_sh_vec.w)           //
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]pack low 16bits together
       } { 
           s1.w = vasl(s1.w, recip_sh_vec.w)           //
           ptr_x0 = sub(ptr_x0, next_outputs)          //[E0, 1]reset data ptr to next 4
           ptr_w_next = ptr_w                          //
           w_count = add(w_count, #-1)
       } {
           y0.w = vmpye(s0.w, recipvec.uh)             //[E0, 1](s2 * recip + rnd)>>31
           ptr_w = memw(sp+#52)                        ////[E0, 5]ptr_w=ptr_wi init filter pointer
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255  and pack low 8bits
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 2]
           s3.w = vasl(s3.w, recip_sh_vec.w)           //
           filt_cnt = add(ptr_w,filt_width512)         //using the ptr_w as a counter
       } {
           if(!p0) y0 = maxe
           y1.w  = vmpye(s1.w, recipvec.uh)            //[E0, 3](s2 * recip + rnd)>>31
           s2.w = vasl(s2.w, recip_sh_vec.w)           //
           ptr_x1 = add(ptr_x0, #100)                  //setup initial pointer
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):RSS          //[E0, 4](s2 * recip + rnd)>>31
           maxe.w = vmax(maxe.w, y0.w)                 //[E0, 0]see if s0 is max
           if(!p0) y0 = mine
           p0 = STQ
       } {	
           if(!p1) y1 = maxe
           mine.w = vmin(mine.w, y0.w)                 //[E0, 0]see if s0 is min
           y2.w  = vmpye(s2.w, recipvec.uh)            //[E0, 5](s2 * recip + rnd)>>31
           STQ = AEQ0 
       } {
           maxe.w = vmax(maxe.w, y1.w)                 //[E0, 3]
           if(!p1) y1 = mine
           y2.w+= vmpyo(s2.w, recipvec.h):RSS          //[E0, 6](s2 * recip + rnd)>>31
           AEQ0 = cmp.eq(r0, r0)  
       } {
           mine.w = vmin(mine.w, y1.w)                 //[E0, 4]see if z0 is max
           if(!p2) y2 = maxe
           y3.w  = vmpye(s3.w, recipvec.uh)            //[E0, 7]#(s2 * recip + rnd)>>31
       } {
           maxe.w = vmax(maxe.w, y2.w)                 //[E0, 4]
           if(!p2) y2 = mine
           y3.w+= vmpyo(s3.w, recipvec.h):RSS          //[E0,8](s2 * recip + rnd)>>31
           p2 = cmp.eq(w_count, #1)                    //
       } {
           mine.w = vmin(mine.w, y2.w)                 //[E0, 5]see if z0 is max
           if(!p3) y3 = maxe
           z = vmem(ptr_x0+#0)                         //pre load 0-127 for next row of filter
           s1s0 = vcombine(wsum, wsum)                 //[E0, 8]initialize accumulator 2,3
       } {
           maxe.w = vmax(maxe.w, y3.w)                 //[E0, 2]
           if(!p3) y3 = mine
           xout = vlalign(x3210,x3_prev,out_align)     //[E1, 6]
           if(p0)vmem(ptr_z++#1):nt = xout.new         //[E1, 6]store 2nd 32bytes
       } {
           x3_prev = x3210                             //[E1, 7]save data for next output align
           s3s2 = vcombine(wsum, wsum)                 //[E0, 8]initialize accumulator 2,3
           if(p2) lmask = memw(sp+#64)                 //right mask
           mine.w = vmin(mine.w, y3.w)                 //[E0, 2]see if z0 is max
       }:endloop1 //cols per line kernel loop width
/*=============================================================================*/
       {   x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]#>>16
           out_height = add(out_height, #-1)           //Prolog width
           STQ = !cmp.eq(r0, r0)                       //[Pheight]force p2 off
           ptr_xi= add(ptr_xi,in_next_rows)            //ptr_x+=in_width*stride_h*in_depth)
       } {
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]#sat8  <0, >255
           p1 = !cmp.eq(out_height, #0)                //EE
           //loop1(.L_width, out_width)                //[Pheight]out_width 
           lc1 = out_width                             //[Pheight]out_width 
       } {
           ptr_x0 = ptr_xi                             //Prolog width ptr_xi 
           AEQ0 = cmp.eq(out_align, #0)                //[Pheight]if no alignment enable store
           skip_col = memw(sp+#(STACK+36))             //
       } {
           p3 = tstbit(skip_col, #2)                   //[E1, 6] == 4
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255 
       } {
           ptr_zi = add(ptr_zi, out_next_row)          //EEnext out line for this depth segment
           ptr_x1 = add(ptr_x0, #100)                  //[Pheight]setup initial pointer
           if (p1) z = vmem(ptr_x0+#0)                 //[Pheight]load 0-127
       } {
           xout = vlalign(x3210, x3_prev, out_align)   //[E1, 6]
           vmem(ptr_z+#0):nt = xout.new                //[E1, 6]store 2nd 32bytes
           w_count = out_width
           lmask = memw(sp+#60)                        //left mask
       } {
           xout = vlalign(x3210, x3210, out_align)     //[E1, 7]
           if(p3) vmem(ptr_z+#1):nt = xout.new         //[E1, 7]flush out last values
           ptr_z = add(ptr_zi, #0)                     //
           if (p1) jump:t .L_height                    //EE
       }//end lines per block//last cols per line
/*=============================================================================*/
           nslice = memw(sp+#(STACK+44))               //
           nslice = add(nslice,#-1)                    //
           memw(sp+#(STACK+44)) = nslice               //
       {   
           p1 = cmp.gt(nslice,#0)                      //
           out_next_d32 = memw(sp+#(STACK+40))         //
           recip_level = memw(sp+#(STACK+28))          //
       } {
           ptr_zi = memw(sp+#56)                       //
           if(p1) recipvec = vmem(recip_level++#1)     //
       } {
           memw(sp+#(STACK+28)) = recip_level          //
           if p1 jump .L_depth                         //
       } 
/*=============================================================================*/
           ptr_max = memw(sp+#(STACK+24))              //
       {   vmem(ptr_max+#0) = maxe                     //[E, 0]32max
           r17:16 = memd(sp+#0)                        //restore r16, r17from stack
       } {
           vmem(ptr_max+#1) = mine                     //[E, 0]32min
           r19:18 = memd(sp+#8)                        //restore r18,r19
       } {
           r21:20 = memd(sp+#16)                       //restore r20,r21
           r23:22 = memd(sp+#24)                       //restore r22,r13
       } {
           r25:24 = memd(sp+#32)                       //restore r24,r15
           r27:26 = memd(sp+#40)                       //restore r26,r17
           sp = add(sp,#STACK)                            //
           jumpr r31                                   //
       }
.L_end:
/*=============================================================================*/
      .size gvconv2dbbbs1_v66_asm, .L_end-gvconv2dbbbs1_v66_asm
/*=============================================================================*/
