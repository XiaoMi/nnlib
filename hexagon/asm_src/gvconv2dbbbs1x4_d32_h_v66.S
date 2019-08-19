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
        .file "gvconv2dbbbs1x4_d32_v66_h.S"
        .global gvconv2dbbbs1x4_v66_asm
        .balign 64
        .type  gvconv2dbbbs1x4_v66_asm, @function
gvconv2dbbbs1x4_v66_asm:
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
#define filt_height_width     r9:8  //
#define out_height            r10   //4 >= 1 number of vertical lines to perform
#define ptr_filtsum           r11   //5 aligned 128
#define ptr_max               r12   //6 aligned 128
#define recip_level           r13   //7  recip is 31bit unsigned 0x7f800000000LL / max
#define recip_level_ptr_max   r13:12
#define out_next_d32          r19   //10
#define nslice                r21   //11
#define recip_shamt           r14   //12
/*=============================================================================*/
//#define stride_h              r26   //0 stride_height|stride_width 
#define ptr_x3                r7    //jump to input ptr for next set of outputs
#define filt_cnt0             r6    //
#define in_next_rows          r15   //in_width * stride_h * in_depth for next output
#define ptr_x0                r16   //
#define ptr_x1                r17   //
#define ptr_w_next            r18   //
#define filt_cnt              r19   //
#define ptr_w                 r20   //
#define in_width_32           r22   //
#define ptr_x2                r23   //
#define ptr_z                 r24   //
#define col_count             r25   //
#define next_ptr              r27   //
#define pred_true   cmp.eq(r0,r0)   //
#define pred_false !cmp.eq(r0,r0)   //
/*=============================================================================*/
#define PV32(VSRC) .word (0x1DFFE020+VSRC)
#define s0                    v0    //
#define s1                    v1    //
#define s1s0                  v1:0  //
#define s2                    v2    //
#define s3                    v3    //
#define s3s2                  v3:2  //
#define s3s2s1s0              v3:0  //
#define w0                    v22   //
#define x0                    v4    //
#define x1                    v5    //
#define x2                    v6    //
#define x3                    v7    //
#define x3210                 v6    //
#define x3_prev               v16   //previous value
#define y0                    v8    //
#define y1                    v9    //
#define y2                    v10   //
#define y3                    v11   //
#define maxe                  v12   //
#define mine                  v13   //
#define wsum                  v14   //initialzed to in_offsey*wsum + biasoffset
#define recipvec              v15   //
#define recip_sh_vec          v17   //
#define RSS     <<1:rnd:sat:shift   //unverbose the insturction
/*=============================================================================*/
       {   filt_height_width = memd(sp+#8)            //
           recip_level_ptr_max = memd(sp+#24)         //
           ptr_xi = and(ptr_xi, #-2)                  //make sure lsb is 0
           sp = add(sp,#-64)                          //
       } {
           memd(sp+#40) = r27:26                      //
           in_depth_stride_h_w = memd(sp+#(64+0))     //
           p2 = pred_false                            //
           nop                                        //
       } {
           memd(sp+#32) = r25:24                      //
           memd(sp+#24) = r23:22                      //
           col_count = out_width                      //out_width 
           in_next_rows= mpy(stride_h_w.H, in_depth.L) //
       } {
           memd(sp+#16) = r21:20                      //
           in_next_rows =mpyi(in_width,in_next_rows)  //total vertical stride bytes
           filt_height = mpy(filt_height.L,in_depth.L)//filt_height*in_depth
           maxe = vmem(ptr_max+#0)                    //
       } {
           memd(sp+#8)  = r19:18                      //
           filt_height = lsr(filt_height, #5)         //filt_height * in_depth / 32
           in_width_32 = asl(in_width, #5)            //32 * in_width d32 line
           mine = vmem(ptr_max+#1)                    //
       } {
           memd(sp+#0)  = r17:16                      //
           ptr_filtsum = memw(sp+#21<<2)              //ptr pre computed weight sum
           filt_width = asl(filt_width, #1)           //x4 to account for loop of 16 bytes
           loop1(.L_filt_height, filt_height)         //[P,0]for(filt_y=0;filt_y<height*in_depth/32;filt_y++){
       } {   
           nslice = memw(sp+#(64+44))                 //
           recip_shamt = memw(sp+#(64+48))            //
           filt_cnt0 = mpyi(filt_width, filt_height)  //
           filt_width = add(filt_width, #-2)          //account for epilog
       } {
           memw(sp+#48) = ptr_xi                      // store ptr_xi
           loop0(.L_filt_width, filt_width)           //[P, 0]ki is k1/32 - 0
           recip_sh_vec = vsplat(recip_shamt)         //
           nop                                        //
       }
        .balign 64
/*=============================================================================*/
.L_depth:
       {   ptr_w = ptr_wi                              //ptr_y=ptr_yi init filter pointer
           out_height = memw(sp+#20<<2)                //number of output lines
           nslice = add(nslice,#-1)                    //
           recipvec = vmem(recip_level++#1)            //
       } {
           memw(sp+#56) = ptr_zi                       //
           ptr_z = ptr_zi                              //
           ptr_x2 = and(ptr_xi, #-128)                 //aligned 
           wsum = vmem(ptr_filtsum++#1)                //
       } {
           s1s0 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
           s3s2 = vcombine(wsum, wsum)                 //[P, 0]initialize accumulators
           z = vmem(ptr_x2+#0)                         //load 0-127 bytes into z buffer
           ptr_x0 = ptr_xi                             //ptr_xi 
       }
/*=============================================================================*/
.L_height:
       {   p3 = pred_true                              //
           filt_cnt = sub(filt_cnt0, filt_height)
           ptr_x3 = add(ptr_xi, #100)                  //reset ptr for next row of filter taps
           z = vmem(ptr_x2+#1)                         //load 128-255
       }
/*=============================================================================*/
        .balign 64
.L_width:
.L_filt_height:
.L_filt_width:
       {   w0.tmp     = vmem(ptr_w++#1)                //[1, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 0]perform mac across 4 streams with saem weights
           if(p3)col_count=add(col_count,#-4)          //count -=4 ptr_z += 128
       } {
           w0.tmp   = vmem(ptr_w++#1)                  //[1, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 1]perform mac across 4 streams with saem weights
           if(p3) ptr_x1 = add(ptr_x0, in_width_32)    //[E, 0]move to next line ptr_y keeps going
           filt_cnt = add(filt_cnt, #-1)               //
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 2]perform mac across 4 streams with saem weights
           if(p3) ptr_x0 = add(ptr_x0, #128)           //
       } {
           p3 = pred_false                             //
           w0.tmp = vmem(ptr_w++#1)                    //[1, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 3]perform mac across 4 streams with saem weights
           z = vmem(ptr_x3+#0)                         //load next stride=1 128 or stride=2 64 bytes
       }:endloop0
/*=============================================================================*/
       {   w0.tmp     = vmem(ptr_w++#1)                //[1, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 0]perform mac across 4 streams with saem weights
           loop0(.L_filt_width, filt_width)            //[P, 0]ki is k1/32 - 0
       } {
           w0.tmp   = vmem(ptr_w++#1)                  //[1, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 1]perform mac across 4 streams with saem weights
           filt_cnt = add(filt_cnt, #-1)               //
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 2]perform mac across 4 streams with saem weights
       } {
           w0.tmp = vmem(ptr_w++#1)                    //[1, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 3]perform mac across 4 streams with saem weights
           z = vmem(ptr_x3+#0)                         //load next stride=1 128 or stride=2 64 bytes
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 0]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 0]perform mac across 4 streams with saem weights
           p0 = cmp.eq(filt_cnt, #0)                   //
       } {
           w0.tmp   = vmem(ptr_w++#1)                  //[1, 1]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 1]perform mac across 4 streams with saem weights
           p1 = cmp.eq(col_count, #0)                  //[E0, 4]compare for branch
           next_ptr = mux(p0, ptr_x0, ptr_x1)          //
       } {
           w0.tmp     = vmem(ptr_w++#1)                //[1, 2]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub++)     //[1, 2]perform mac across 4 streams with saem weights
           //ptr_x2 = and(ptr_x1, #-128)                 //Taligned 
           ptr_x2 = and(next_ptr, #-128)                 //Taligned 
       } {
           w0.tmp = vmem(ptr_w++#1)                    //[1, 3]load weights
           s3s2s1s0.w += vrmpyz(w0.b, ptr_x3.ub  )     //[1, 3]perform mac across 4 streams with saem weights
           z = vmem(ptr_x2+#0)                         //Tload 0-127 bytes into z buffer
       } {
           ptr_x1 = add(ptr_x1, in_width_32)           //[E, 0]move to next line ptr_y keeps going
           //ptr_x3 = add(ptr_x1, #100)                  //Treset ptr for next row of filter taps
           ptr_x3 = add(next_ptr, #100)                  //Treset ptr for next row of filter taps
           z = vmem(ptr_x2+#1)                         //Tload 128-255
       }:endloop1
/*=============================================================================*/
       {   s0.w = vasl(s0.w, recip_sh_vec.w)           //[E0, ]
           ptr_w_next = ptr_w                          //[E0, ]
       } {
           s1.w = vasl(s1.w, recip_sh_vec.w)           //[E0, ]
           loop0(.L_filt_width, filt_width)            //[P, 0]ki is k1/32 - 0
       } {
           y0.w = vmpye(s0.w, recipvec.uh)             //[E0, 1](s2 * recip + rnd)>>31
           loop1(.L_filt_height, filt_height)          //[P,0]for(filt_y=0;filt_y<height*in_depth/32;filt_y++)
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):RSS          //[E0, ]<<1:rnd:sat:shift //[E0, 2]
           s2.w = vasl(s2.w, recip_sh_vec.w)           //[E0, ]
           s0 = wsum                                   //[P, 0]initialize accumulators
       } {
           maxe.w = vmax(maxe.w, y0.w)                 //[E0, 0]see if s0 is max
           mine.w = vmin(mine.w, y0.w)                 //[E0, 0]see if s0 is min
           y1.w  = vmpye(s1.w, recipvec.uh)            //[E0, 3](s2 * recip + rnd)>>31
       } {
           s3.w = vasl(s3.w, recip_sh_vec.w)           //[E0,  ]
           y1.w+= vmpyo(s1.w, recipvec.h):RSS          //[E0, ]<<1:rnd:sat:shift (s2 * recip + rnd)>>31
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]#sat8  <0, >255
       } {
           maxe.w = vmax(maxe.w, y1.w)                 //[E0, 2]
           y2.w  = vmpye(s2.w, recipvec.uh)            //[E0, 5](s2 * recip + rnd)>>31
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, .]#sat8  <0, >255 
           if(p2) vmem(ptr_z++#1):nt = x3210.new       //[E1, .]store 2nd 32bytes
       } {
           mine.w = vmin(mine.w, y1.w)                 //[E0, 2]see if z0 is max
           ptr_w = ptr_wi                              //[E0, 5]ptr_y=ptr_yi init filter pointer
           y2.w+= vmpyo(s2.w, recipvec.h):RSS          //[E0. ]<<1:rnd:sat:shift s2 * recip + rnd)>>31
           x1.h = vpack(y1.w, y0.w):sat                //[E1, 0]#>>16
       } {
           maxe.w = vmax(maxe.w, y2.w)                 //[E0, 3]
           y3.w  = vmpye(s3.w, recipvec.uh)            //[E0, 7]#(s2 * recip + rnd)>>31
           p2 = pred_true                              //[P,  ]p2 = 1
           s1 = wsum                                   //[P, 0]initialize accumulators
       } {
           mine.w = vmin(mine.w, y2.w)                 //[E0, 4]see if z0 is max
           y3.w+= vmpyo(s3.w, recipvec.h):RSS          //[E0, 8](s2 * recip + rnd)>>31
           p3 = pred_true                              //[P,  ]
           filt_cnt = sub(filt_cnt0, filt_height)      //[P,  ]
       } {
           s3s2 = vcombine(wsum,wsum)                  //[P, 0]initialize accumulators
           mine.w = vmin(mine.w, y3.w)                 //[E0, 5]see if z0 is max
           maxe.w = vmax(maxe.w, y3.w)                 //[E0, 4]
           if(!p1) jump .L_width                       //[E1, 8]
       }//cols per line kernel loop width
/*=============================================================================*/
       {   out_height = add(out_height, #-1)           //Prolog width
           ptr_zi = add(ptr_zi, out_next_row)          //EEnext out line for this depth segment
           ptr_xi= add(ptr_xi,in_next_rows)            //ptr_x+=in_width*stride_h*in_depth)
           p1 = !cmp.eq(out_height, #1)                //EE
       } {
           x3.h = vpack(y3.w, y2.w):sat                //[E1, 1]#sat8  <0, >255
           ptr_x2 = and(ptr_xi, #-128)                 //aligned 
           ptr_x0 = ptr_xi                             //ptr_xi 
       } {
           col_count = out_width                       //out_width 
           if (p1) z = vmem(ptr_x2+#0)                 //load 0-127 bytes into z buffer
           p2 = pred_false
       } {
           x3210.ub = vpack(x3.h, x1.h):sat            //[E1, 3]#sat8  <0, >255 
           vmem(ptr_z+#0):nt = x3210.new               //[E1, 6]store 2nd 32bytes
           if (p1) jump:t .L_height                    //EE
           ptr_z = ptr_zi                              //
       }//end lines per block//last cols per line
/*=============================================================================*/
       {   
           ptr_wi = ptr_w_next                         // set ptr of weight
           ptr_zi = memw(sp+#56)                       //
           out_next_d32 = memw(sp+#(64+40))            //
           p1 = cmp.gt(nslice,#0)
       } {
           ptr_xi = memw(sp+#48)                       // restore ptr_xi
           ptr_zi  = add(ptr_zi,out_next_d32)          //
           if p1 jump .L_depth                         //
       }
/*=============================================================================*/
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
           sp = add(sp,#64)                            //pop stack
           jumpr r31                                   // return
       }
.L_end:
/*=============================================================================*/
      .size gvconv2dbbbs1x4_v66_asm, .L_end-gvconv2dbbbs1x4_v66_asm
/*=============================================================================*/
