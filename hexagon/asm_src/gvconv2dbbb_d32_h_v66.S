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
        .file "gvconv2dbbb_h.S"
        .global gvconv2dbbb_v66_asm
        .balign 32
        .type  gvconv2dbbb_v66_asm, @function
gvconv2dbbb_v66_asm:
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
#define in_depth_stride_h_w   r27:26
#define filt_width            r8    //2 >= 1
#define filt_height           r9    //3 >= 1filt_height lines per filter
#define filt_height_width     r9:8
#define out_height            r10   //4 >= 1 number of vertical lines to perform
#define ptr_filtsum           r11   //5 aligned 128
#define ptr_max               r12   //6 aligned 128
#define recip_level           r13   //7 recip is 31bit unsigned 0x7f800000000LL / max
#define recip_level_ptr_max   r13:12//
#define out_align             r6    //8 0, 32, 64, 96
#define left_skip_tmp_out_align r7:6
#define left_skip_tmp         r7    //9
#define out_next_d32          r14   //10
#define nslice                r21   //11
#define recip_shamt           r27   //12
#define ptr_w_next            r17   //
#define AEQ0                  r27   // 
#define STQ                   r12   // 
#define ptr_x1                r7    //
#define skip_col              r13   //
#define lmask                 r21   //
/*=============================================================================*/
#define stride_h              r26   //0 stride_height|stride_width 
#define in_next_rows          r15   //in_width * stride_h * in_depth for next output
#define ptr_x0                r16   //
#define stride_w              r18   //stride width
#define next_outputs          r19   //jump to input ptr for next set of outputs
#define ptr_w                 r20   //
#define in_width_32           r22   //
#define ptr_x2                r23   //
#define ptr_z                 r24   //
#define col_count             r25   //
#define PRED3_0                C4   //used to boradside load all predicates
#define STACK                  72   //amount of stack to reserve
/*=============================================================================*/
#define PV32(VSRC) .word (0x1DFFE020+VSRC)
#define s0                    v0    //
#define s1                    v1    //
#define s1s0                  v1:0  //
#define s2                    v2    //
#define s3                    v3    //
#define s3s2                  v3:2  //
#define s3s2s1s0              v3:0  //
#define w0                    v19   //
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
#define recip_sh_vec          v13   //
#define RSS     <<1:rnd:sat:shift   //unverbose the insturction
/*=============================================================================*/
       {   filt_height_width = memd(sp+#8)            //
           recip_level_ptr_max = memd(sp+#24)         //ptr pre computed max value in output
           ptr_xi = and(ptr_xi, #-2)                  //guarentee lsb is 0
           sp = add(sp,#-STACK)                       //
       } {
           memd(sp+#40) = r27:26                      //
           in_depth_stride_h_w = memd(sp+#(STACK+0))     //extract strides h + w 
       } {
           memd(sp+#32) = r25:24                      //
           col_count = out_width                      //out_width 
           filt_width += add(filt_width, #-1)         //x2 to account for lop of 16bytes, -1 account for epilog
           maxe = vmem(ptr_max+#0)                    //
       } {
           memd(sp+#24) = r23:22                      //
           memd(sp+#16) = r21:20                      //
           filt_height = mpy(filt_height.L,in_depth.L)//filt_height*in_depth
           in_next_rows= mpy(stride_h_w.H, in_depth.L)//
       } {
           memd(sp+#8)  = r19:18                      //
           memd(sp+#0)  = r17:16                      //
           filt_height = lsr(filt_height, #5)         //filt_height * in_depth / 32
           in_width_32 = asl(in_width, #5)            //32 * in_width d32 line
       } {
           ptr_filtsum = memw(sp+#(STACK+20))            //ptr pre computed weight sum
           loop1(.L_filt_height, filt_height)         //[P,0]for(filt_y=0;filt_y<height*in_depth/32;filt_y+=1){
           stride_w = zxth(stride_h_w)                //extract width
           r21 = #32
       } {
           recipvec = vmem(recip_level++#1)           //
       } {
           next_outputs=mpyi(filt_height,in_width)    //filt_height*in_width*in_depth
           loop0(.L_filt_width, filt_width)           //[P, 0]ki is k1/32 - 0
           left_skip_tmp_out_align = memd(sp+#(STACK+32))// NOTE clobbers ptr_x1, but not in use yet
           mine = vmem(ptr_max+#1)                    //
       } {
           memw(sp+#(STACK+28)) =  recip_level           //ptr pre computed max value in output
           left_skip_tmp = and(left_skip_tmp, #0x7f)  //extract lower left skip value
           skip_col = extract(left_skip_tmp, #2, #16) //extract skip col bits
       } {
           ptr_xi = add(ptr_xi,left_skip_tmp)
           next_outputs -= asl(stride_w,#2)           //1,2
           r21 -= lsr(out_align,#2)                   //1/4 for bytes: 0->32, 32->24, 64-> 6, 96->8
           recip_shamt = memw(sp+#(STACK+48))         //
       } { 
           STQ = !cmp.eq(r0, r0)                      //force p2 off
           r21 = and(r21, #0x1f)                      //
           r7 = #-1                                   //
       } {
           recip_sh_vec = vsplat(recip_shamt)         //
           r7 = asl(r7, r21)                          //create left mask
           r21 = #0x1f                                //
       } {
           memw(sp+#60) = r7                          //
           r21 &= asl(skip_col, #3)                   //
           r7 = #-1                                   //
       } {
           r7 = lsr(r7, r21)                          //create rmask
       } {
           memw(sp+#64) = r7                          //
       } {
           memw(sp+#48) = ptr_xi                      //
           next_outputs = asl(next_outputs, #5)       //(flt_hight*in_width*in_depth/32-4*stride)*32
           in_next_rows =mpyi(in_width,in_next_rows)  //total vertical stride bytes
           AEQ0 = cmp.eq(out_align, #0)               //if no alignment enable store
       }
/*=============================================================================*/
	.balign 64
.L_depth:
       {   ptr_w = ptr_wi                              //[P,0]ptr_y=ptr_yi initialize filter pointer
           out_height = memw(sp+#(STACK+16))           //number of output lines
           ptr_x0 = ptr_xi                             //ptr_xi 
           lmask = memw(sp+#60)                        //
       } {
           memw(sp+#56) = ptr_zi                       //
           ptr_z = add(ptr_zi, #0)                     //
           wsum = vmem(ptr_filtsum++#1)                //
           ptr_x2 = and(ptr_xi, #-128)                 //[Pheight]make loads aligned to 128 zero out bits 0-6
       } {   
           ptr_x1 = add(ptr_x0, #4)                    //[Pheight]setup initial pointer
           z = vmem(ptr_x2+#0)                         //[Pheight]load 0-127
           s3s2 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
           s1s0 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
       }
/*=============================================================================*/
.L_height:
.L_width:
.L_filt_height:
       {   ptr_x0 = add(ptr_x0, in_width_32)           //[E, 0]move to next line ptr_y keeps going
           z = vmem(ptr_x2+#1)                         //load 128-255
           ptr_x1 += mpyi(stride_w, #24*4)             //4*stride_w*24 = 96*stride_w- canbe adusted 0,+32
           nop
       }
	.balign 64
/*=============================================================================*/
.L_filt_width:
       {   w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
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
           z = vmem(ptr_x1+#0)                         //load next stride=1 128 or stride=2 64 bytes
       }:endloop0
/*=============================================================================*/
       {   w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 6]perform mac across 4 streams with saem weights
           loop0(.L_filt_width, filt_width)            //[P, 0]ki is k1/32 - 0
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 6]perform mac across 4 streams with saem weights
       } {
           ptr_x2 = and(ptr_x0, #-128)                 // make loads aligned to 128 zero out bits 0-6
           w0.tmp     = vmem(ptr_w++#1)                //[1, 6]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub++)     //[1, 6]perform mac across 4 streams with saem weights
       } {
           w0.tmp = vmem(ptr_w++#1)                    //[1, 7]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x1.ub  )     //[1, 7]perform mac across 4 streams with saem weights
           z = vmem(ptr_x2+#0)                         //load 0-127 bytes into z buffer
           ptr_x1 = add(ptr_x0, #4)                    //reset ptr for next row of filter taps
       }:endloop1
/*=============================================================================*/
           s0.w = vasl(s0.w, recip_sh_vec.w)           //shift left before shifting right by 31
           s1.w = vasl(s1.w, recip_sh_vec.w)           //shift left before shifting right by 31
           s2.w = vasl(s2.w, recip_sh_vec.w)           //shift left before shifting right by 31
           s3.w = vasl(s3.w, recip_sh_vec.w)           //shift left before shifting right by 31
       {
           PRED3_0 = lmask                             //load all predicates with mask bits
           lmask = #-1                                 //default all enabled 
           x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]packe low 16bits together
       } {
           ptr_x0 = sub(ptr_x0, next_outputs)          //[E0, 1]reset data ptr to next 4
           y0.w = vmpye(s0.w, recipvec.uh)             //[E0, 1](s2 * recip + rnd)>>31
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]pack low 16bits together
           col_count=add(col_count,#-4)                //count -=4 ptr_z += 128
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):RSS          //[E0, 2]((s0<<rsh) * recip + rnd )>>31
           ptr_x2 = and(ptr_x0, #-128)                 //make loads aligned to 128 zero out bits 0-6
       } {
           loop1(.L_filt_height, filt_height)          //[E0, 0]for(filt_y=0;filt_y<height*in_depth/32;filt_y++){
           if(!p0) y0 = maxe                           //[E0,  ]
       } {
           maxe.w = vmax(maxe.w, y0.w)                 //[E0, 0]see if s0 is max
           if(!p0) y0 = mine                           //[E0,  ]
       } {
           mine.w = vmin(mine.w, y0.w)                 //[E0, 0]see if s0 is min
       } {
           y1.w  = vmpye(s1.w, recipvec.uh)            //[E0, 3](s2 * recip + rnd)>>31
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255  and pack low 8bits
           ptr_x1 = add(ptr_x0, #4)                    //setup initial pointer
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):RSS          //[E0, 4](s2 * recip + rnd)>>31
       } {
           if(!p1) y1 = maxe                           //[E0,  ]
       } {
           maxe.w = vmax(maxe.w, y1.w)                 //[E0, 3]
           if(!p1) y1 = mine                           //[E0,  ]
       } {
           mine.w = vmin(mine.w, y1.w)                 //[E0, 4]see if z0 is max
           p1 = STQ                                    //[E0,  ]
       } {
           y2.w  = vmpye(s2.w, recipvec.uh)            //[E0, 5](s2 * recip + rnd)>>31
           s1s0 = vcombine(wsum, wsum)                 //[E0, 5]initialize accumulator 0,1
           ptr_w_next = ptr_w                          //[E0,  ]
           ptr_w = ptr_wi                              //[E0, 5]ptr_y=ptr_yi init filter pointer
       } {
           y2.w+= vmpyo(s2.w, recipvec.h):RSS          //<<1:rnd:sat:shift //[E0, 6](s2 * recip + rnd)>>31
           xout = vlalign(x3210,x3_prev,out_align)     //[E1, 6]
           if(p1)vmem(ptr_z++#1):nt = xout.new         //[E1, 6]store 2nd 32bytes
       } {
           if(!p2) y2 = maxe                           //[E0,  ]
           p0 = cmp.eq(col_count, #4)                  //[E0, 4]compare for branch
       } {
           if(p0) lmask = memw(sp+#64)                 //[E0,  ]switch in the right hand mask on last iteration
           maxe.w = vmax(maxe.w, y2.w)                 //[E0, 4]
           if(!p2) y2 = mine                           //
       } {
           mine.w = vmin(mine.w, y2.w)                 //[E0, 5]see if z0 is max
       } {
           y3.w  = vmpye(s3.w, recipvec.uh)            //[E0, 7]#(s2 * recip + rnd)>>31
           x3_prev = x3210                             //[E1, 7]save data for next output align
           STQ = AEQ0                                  //[E1, 7]update predicate piplione
           AEQ0 = cmp.eq(r0, r0)                       //[E1, 7]set to true
       } {
           y3.w+= vmpyo(s3.w, recipvec.h):RSS          //[E0, 8](s2 * recip + rnd)>>31
           s3s2 = vcombine(wsum, wsum)                 //[E0, 8]initialize accumulator 2,3
           z = vmem(ptr_x2+#0)                         //[P, 0]pre load 0-127 for next row of filter
       } {
           if(!p3) y3 = maxe                           //[E0,  ]
           p0 = cmp.eq(col_count, #0)                  //[E0, 4]compare for branch
       } {
           maxe.w = vmax(maxe.w, y3.w)                 //[E0, 2]
           if(!p3) y3 = mine                           //[E0,  ]
       } {
           mine.w = vmin(mine.w, y3.w)                 //[E0, 2]see if z0 is max
           if(!p0) jump:t .L_width                     //[E1, 8]
       }//cols per line kernel loop width
/*=============================================================================*/
       {   x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]#>>16
           out_height = add(out_height, #-1)           //Prolog width
           STQ = !cmp.eq(r0, r0)                       //[Pheight]force p2 off
           ptr_xi= add(ptr_xi,in_next_rows)            //ptr_x+=in_width*stride_h*in_depth)
       } {
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]#sat8  <0, >255
           p1 = !cmp.eq(out_height, #0)                //EE
           col_count = out_width                       //[Pheight]out_width 
       } {
           ptr_x0 = ptr_xi                             //Prolog width ptr_xi 
           AEQ0 = cmp.eq(out_align, #0)                  //[Pheight]if no alignment enable store
           ptr_x2 = and(ptr_xi, #-128)                 //[Pheight]make loads aligned to 128 
       } {
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255 
       } {
           ptr_zi = add(ptr_zi, out_next_row)          //EEnext out line for this depth segment
           ptr_x1 = add(ptr_x0, #4)                    //[Pheight]setup initial pointer
           if (p1) z = vmem(ptr_x2+#0)                 //[Pheight]load 0-127
       } {
           xout = vlalign(x3210, x3_prev, out_align)   //[E1, 6]
           vmem(ptr_z+#0):nt = xout.new                //[E1, 6]store 2nd 32bytes
           ptr_z = add(ptr_zi, #0)                     //
           if (p1) jump:t .L_height                    //EE
       }//end lines per block//last cols per line
/*=============================================================================*/
           nslice = memw(sp+#(STACK+44))                 //
       {   
           nslice = add(nslice,#-1)                    //
       } {
           memw(sp+#(STACK+44)) = nslice                 //
       } {
           ptr_zi = memw(sp+#56)                       //
           p1 = cmp.gt(nslice,#0)                      //
           recip_level = memw(sp+#(STACK+28))                  //ptr pre computed max value in output
       } {
           out_next_d32 = memw(sp+#(STACK+40))            //
           ptr_wi = ptr_w_next                         // set ptr of weight
           ptr_xi = memw(sp+#48)                       // restore ptr_xi
       } {
           if(p1) recipvec = vmem(recip_level++#1)     //
       } {
           memw(sp+#(STACK+28)) = recip_level                  //ptr pre computed max value in output
           ptr_zi  = add(ptr_zi,out_next_d32)          //
           if p1 jump .L_depth                         //
       }
/*=============================================================================*/
           ptr_max = memw(sp+#(STACK+24))                 //ptr pre computed max value in output
       {   vmem(ptr_max+#0) = maxe                     //[E, 0]32max
           r17:16 = memd(sp+#0)                        //restore r16, r17from stack
       } {
           vmem(ptr_max+#1) = mine                     //[E, 0]32min
           r19:18 = memd(sp+#8)                        //restore r18,r19
       } {
           r21:20 = memd(sp+#16)                       //restore r20,r11
           r23:22 = memd(sp+#24)                       //restore r22,r13
       } {    
           r25:24 = memd(sp+#32)                       //restore r24,r15
           r27:26 = memd(sp+#40)                       //restore r26,r17
           sp = add(sp,#STACK)                            //
           jumpr r31                                   // return
       } 
.L_end:
/*=============================================================================*/
      .size gvconv2dbbb_v66_asm, .L_end-gvconv2dbbb_v66_asm
/*=============================================================================*/
