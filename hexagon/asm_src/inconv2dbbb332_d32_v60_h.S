/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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
 *  FUNCTIONS      : gvconv2dbbb_v60_asm     
 *                                          
 *  DESCRIPTION                            
 *    Perform 2d convolution using elements of size in_depth < 32. Results are
 *    scaled and saturated to 8bits. Max and Min accumulations are kept.
 *                                       
 *  ARCHITECTURE   : QDSP6V60  + HVX     
 *
 *  REVISION HISTORY:                                    
 *  =================                                   
 *                                                     
 *  Author              Date           Comments       
 *  ------------------------------------------------------------- 
 *  DJH                 05/11/17       created                   
 *
 *  CYCLE-COUNT:                              
 *                                              
 *  MEMORY                                       
 *     CODESIZE = 928 bytes                       
 *     STACK    = 80 bytes                         
 *     ASSUMPTIONS                                  
 */
/*=============================================================================*/
        .text
        .file "inconv2dbbb332_d32_v60_h.S"
        .global inconv2dbbb332_v60_asm
        .balign 32
        .type  inconv2dbbb332_v60_asm, @function
/* parameters:
 *  r0      ( const uint8_t * input,
 *  r1       const uint8_t * weights,
 *  r2       uint8_t * output,
 *  r3       int in_width_pad,
 *  r4       int next_out_width_row,
 *  r5       int out_width,
 * PARMW(0)  int indepth,
 * PARMW(1)  int filt_width,
 * PARMW(2)  int filt_height,
 * PARMW(3)  int num_out_lines,
 * PARMW(4)  int32_t * minmax_buf,
 * PARMW(5)  int recip_level,
 * PARMW(6)  const int32_t *biasbuf,
 * PARMW(7)  const int32_t *ptr_suma,
 * PARMW(8)  int next_suma,
 * PARMW(9)  int stride_height_width,
 * PARMW(10) int recip_shamt);
 */
inconv2dbbb332_v60_asm:
/*=============================================================================*/
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define ptr_zi                 r2     //results
#define in_width               r3     //(pad_l+in_width+pad_r)
#define out_width_stride_depth r4     //next line amount 
#define out_width              r5     //is amount of work to be done
#define in_depth               r26    //0
#define filt_width             r17    //1 horizontal fuilter width
#define filt_height            r8     //2 filt_height lines per filter
#define out_height             r9     //3 number of vertical lines to perform
#define ptr_max                r13    //4 maximum and minum buffer
#define recip_level            r14    //5 255 / (MAX - MIN) - used to scale to bytes
#define filt_sum               r15    //6 gemsumb
#define active_sum             r18    //7 gemsuma activations
#define next_suma_buf          r1     //8 stride for suma buffer
#define stride_v_h             r10    //9 stride_vert | stride_horz ->M0
#define sum0                   r19
#define sum1                   r19
#define sum2                   r19
#define sum3                   r19
#define recip_shamt            r1
/*=============================================================================*/
#define fetch_ptr              r0
#define fetch_ptr0             r2
#define in_depth3              r21
#define in_width_stride_depth  r15    //in_width * stride * in_depth for next output
#define ptr_x0                 r16    //tmp pointer to activations
#define ptr_x1                 r23    //dynamic pointer to activations
#define next_outputs           r27    //jump to input ptr for next set of outputs
#define ptr_w                  r20    //pointer to weights
#define in_width_depth         r22    //width of input image in bytes
#define c4                     r2     //shuffle size in final max and min find
#define ptr_z                  r24    //pointer to outputs
#define col_count              r25    //column count, how much of width used
#define x07x04_x03x00          r11:10 //8 activations output 0
#define x07x04                 r11    //4 activations output 0
#define x03x00                 r10    //4 activations output 0
#define x17x14_x13x10          r13:12 //8 activations output 1
#define x17x14                 r13    //4 activations output 1
#define x13x10                 r12    //4 activations output 1
#define x27x24_x23x20          r11:10 //8 activations output 2
#define x27x24                 r11    //4 activations output 2
#define x23x20                 r10    //4 activations output 2
#define x37x34_x33x30          r7:6   //8 activations output 3
#define x37x34                 r7     //4 activations output 3
#define x33x30                 r6     //4 activations output 3
/*=============================================================================*/
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca reg
#define s0                     v0     //accumulator for output 0
#define s1                     v1     //accumulator for output 1
#define s1s0                   v1:0   //accumulator 
#define s2                     v2     //accumulator for output 2
#define s3                     v3     //accumulator for output 3
#define s3s2                   v3:2   //
#define d0                     v4     //
#define d1                     v4     //
#define d2                     v4     //
#define d3                     v5     //
#define y0                     v8     //
#define y1                     v9     //
#define y2                     v10    //
#define y3                     v11    //
#define yout                   v17    //
#define y3_prev                v16    //copy of previous value
#define wsum                   v14    //initialzed to in_offsey*wsum + biasoffset
#define maxomaxe               v13:12 //
#define maxe                   v12    //
#define maxo                   v13    //
#define minomine               v19:18 //
#define mine                   v18    //
#define mino                   v19    //
#define biasvec                v16    //
#define recipvec               v15    //
#define vzero                  v20
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/*=============================================================================*/
#define FRAMESIZE 72
#define PARMW(n) sp+#(8+FRAMESIZE+4*(n))
       {   allocframe(#FRAMESIZE)                       // 0th entry on stack is (72+8)/4 =20 ints
       } {  
           memd(sp+#0)  = r17:16                        //
           memd(sp+#8)  = r19:18                        //
       } {
           memd(sp+#16) = r21:20                        //
           memd(sp+#24) = r23:22                        //
           r23 = ##0x7fffffff                           //max pos
       } {
           memd(sp+#32) = r25:24                        //
           memd(sp+#40) = r27:26                        //
           mine = vsplat(r23)
       } {
           memw(sp+#48) = ptr_xi
           memw(sp+#52) = ptr_wi
           maxe = vnot(mine)                           // all -0x80000000
       } {
           active_sum = memw(PARMW(7))                  //
           in_depth = memw(PARMW(0))
       } {
           filt_height = memw(PARMW(2))                 //extract filt_height
           filt_width = memw(PARMW(1))                  //extract filt_width
       } {   
           memw(sp+#56) = ptr_zi
           filt_width = lsr(filt_width, #1)             //
           out_height = memw(PARMW(3))                  //number of output lines
       } {
           recip_level = memw(PARMW(5))                 //
           filt_sum = memw(PARMW(6))
           in_width_depth  = mpyi(in_width, in_depth)   //in_depth * in_width line
       } {
           wsum = vmem(filt_sum+#0)                     //
           recipvec = vsplat(recip_level)               //
           stride_v_h = memw(PARMW(9))
       } {
          // next_suma_buf = memw(PARMW(8))
           in_width_stride_depth=mpy(in_width_depth.L,stride_v_h.H)//
           stride_v_h = zxth(stride_v_h) 
       } {
           dcfetch(active_sum+#0<<6)
           in_depth = mpyi(in_depth, stride_v_h)
           stride_v_h = asl(stride_v_h, #2) 
       } {
           next_outputs=mpyi(filt_height,in_width_depth)//filt_height*in_width*in_depth
           //M0 = stride_v_h = #8
       } {
           next_outputs += mpyi(stride_v_h, #-4)        //
           memw(sp+#60) = active_sum                    //
           in_depth3 = addasl(in_depth, in_depth, #1)   //
       }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_height:
       {   active_sum = memw(sp+#60)
           ptr_x0 = memw(sp+#48)                        //ptr_xi                         
           loop0(.L_filt_height, filt_height)           //[P, 0]for(fil=0;fil<h*depth/32;fil+=1){
       } {
           ptr_z = memw(sp+#56)                         //ptr_zi
           col_count = out_width
           sum0 = memw(active_sum++#8)                  //stride = 8
           fetch_ptr = add(ptr_x0, in_width_depth)      //
       } {
           next_suma_buf = memw(PARMW(8))
           dcfetch(ptr_x0+#0<<6)                        //[0, 3]
           d0 = vsplat(sum0)
       } {
           memw(sp+#60) += next_suma_buf
           sum1 = memw(active_sum++#8)                  //stride = 8
           s0.w = vadd(wsum.w, d0.w)
       } {
           d1 = vsplat(sum1)
           sum2 = memw(active_sum++#8)                  //stride = 8
           dcfetch(fetch_ptr+#0<<6)                     //[0, 3]
       } {
           s1.w = vadd(wsum.w, d1.w)
           d2 = vsplat(sum2)
           sum3 = memw(active_sum++#8)                  //stride = 8
           memw(sp+#56) += out_width_stride_depth       //ptr_zi += out_width_stride_depth
       } {  // r1 is recip_shamt, except when it's used as next_suma_buf above.
           recip_shamt = memw(PARMW(10))
           memw(sp+#48) += in_width_stride_depth        //ptr_xi+=in_width_stride_depth  //ptr_x+=in_width*stride*in_depth)
           s2.w = vadd(wsum.w, d2.w)
           d3 = vsplat(sum3)
       } {
           out_height = add(out_height, #-1)            //
           s3.w = vadd(wsum.w, d3.w)
           ptr_w = memw(sp+#52)                         //[P, 0]ptr_wi initialize filter pointer
           fetch_ptr = add(ptr_x0, in_width_depth)      //
       } 
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_width:
.L_filt_height:
       {
           ptr_x1 = ptr_x0                              //[P, 5]
           ptr_x0 = add(ptr_x0, in_width_depth)         //[E, 7]next line ptr_y keeps going
           x27x24_x23x20 = memd(ptr_x0+in_depth<<#1)    //[0, 0]
           x37x34_x33x30 = memd(ptr_x0+in_depth3<<#0)   //[0, 0]
       } {
           y0.cur     = vmem(ptr_w++#1)                 //[0, 1]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[0, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[0, 1]
           x17x14_x13x10 = memd(ptr_x1+in_depth<<#0)    //[0, 1]
       } { 
           y1.cur     = vmem(ptr_w++#1)                 //[0, 2]
           s2.uw       += vrmpy(y1.ub, x27x24.ub)       //[0, 2]
           s3.uw       += vrmpy(y1.ub, x37x34.ub)       //[0, 2]
           x07x04_x03x00 = memd(ptr_x1++#1<<3)          //[0, 2]stride = 4
       } {
           s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[0, 3]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[0, 3]
           dcfetch(fetch_ptr+#0<<6)                     //[0, 3]
           x37x34 = memw(ptr_x1+#0<<3)                  //[1, 2]
       } {
           s0.uw       += vrmpy(y1.ub, x07x04.ub)       //[0, 4]
           s1.uw       += vrmpy(y1.ub, x17x14.ub)       //[0, 4]
           x27x24_x23x20 = memd(ptr_x1+in_depth<<#1)    //[1, 0]
                  x33x30 = memw(ptr_x1+in_depth3<<#0)   //[1, 0]
       } {   
           y0.cur     = vmem(ptr_w++#1)                 //[1, 1]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[1, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[1, 1]
           x13x10 = memw(ptr_x1+in_depth<<#0)           //[1, 1]
       } {
           s0.uw       += vrmpy(y0.ub, x37x34.ub)       //[1, 3]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[1, 3]
           fetch_ptr = add(fetch_ptr, in_width_depth)   //
       }:endloop0
/* ---------------------------------------------------------------------------- */
#if 1
// << by recip_shamt; scale, and pack. min/max are found on the results
// prior to packing.
      {
           s2.w = vasl(s2.w,recip_shamt)
           ptr_x0 = sub(ptr_x0, next_outputs)           //reset data ptr to next 4
           dcfetch(active_sum+#0<<6)
       } {
           s3.w = vasl(s3.w,recip_shamt)
           col_count = add(col_count, #-4)              //
           loop0(.L_filt_height, filt_height)           //[P, 1]for(fil=0;fil<h*depth/32;fil+=1){
       } {
           y2.w  = vmpye(s2.w, recipvec.uh)             //
           s0.w = vasl(s0.w,recip_shamt)
           p1 = cmp.eq(col_count, #0)                   //
       } {
           y2.w+= vmpyo(s2.w, recipvec.h):SSR           //
           s1.w = vasl(s1.w,recip_shamt)
           sum3 = memw(active_sum+#(3*8)) //#2<<2)         //stride = 4
       } {
           mine.w = vmin(mine.w, y2.w)                  //see if y2 is min
           maxe.w = vmax(maxe.w, y2.w)                  //see if y2 is max
           y3.w  = vmpye(s3.w, recipvec.uh)             //
       } {
           d3 =vsplat(sum3)
           dcfetch(ptr_x0+#0<<6)
           sum2 = memw(active_sum+#(2*8))                  //stride = 4
       } {
           s3.w = vadd(wsum.w,d3.w)
           fetch_ptr = add(ptr_x0, in_width_depth)      //
           y3.w+= vmpyo(s3.w, recipvec.h):SSR           //
       } {
           mine.w = vmin(mine.w, y3.w)                  //see if y3 is min
           maxe.w = vmax(maxe.w, y3.w)                  //see if y3 is max
           y0.w  = vmpye(s0.w, recipvec.uh)             //
           dcfetch(fetch_ptr+#0<<6)
       } {
           sum0 = memw(active_sum+#0)                  //stride = 4
           d2 = vsplat(sum2)
           fetch_ptr = add(ptr_x0, in_width_depth)      //
       } {
           s2.w = vadd(wsum.w,d2.w)
           y0.w+= vmpyo(s0.w, recipvec.h):SSR           //
           y3.h = vpack(y3.w, y2.w):sat                //#sat8  <0, >255
       } {
           mine.w = vmin(mine.w, y0.w)                  //see if y0 is min
           maxe.w = vmax(maxe.w, y0.w)                  //see if y0 is max
           y1.w  = vmpye(s1.w, recipvec.uh)             //
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):SSR           //
       } {
           d0 = vsplat(sum0)
           mine.w = vmin(mine.w, y1.w)                  //see if y1 is min
           maxe.w = vmax(maxe.w, y1.w)                  //see if y1 is max
       } {
           s0.w = vadd(wsum.w,d0.w)
           sum1 = memw(active_sum+#8)                  //stride = 4
           y1.h = vpack(y1.w, y0.w):sat                //#sat8  <0, >255
       } {
           ptr_w = memw(sp+#52)                         //[P, 0]ptr_wi initialize filter pointer
           d1 = vsplat(sum1)
           active_sum = add(active_sum,#8*4)
       } {
           s1.w = vadd(wsum.w, d1.w)
           y3.ub = vpack(y3.h, y1.h):sat                //#sat8  <0, >255
           vmem(ptr_z++#1) = y3.new                     //#[E,  ]store 2nd 32bytes
           if(!p1) jump:t .L_width                      //
       }//end cols per line

#else
       {   y2.w  = vmpye(s2.w, recipvec.uh)             //
           mine.w = vmin(mine.w, s2.w)                  //see if s2 is max
           ptr_x0 = sub(ptr_x0, next_outputs)           //reset data ptr to next 4
           dcfetch(active_sum+#0<<6)
       } {
           y2.w+= vmpyo(s2.w, recipvec.h):SSR           //
           col_count = add(col_count, #-4)              //
           loop0(.L_filt_height, filt_height)           //[P, 1]for(fil=0;fil<h*depth/32;fil+=1){
       } {
           y3.w  = vmpye(s3.w, recipvec.uh)             //#
           mine.w = vmin(mine.w, s3.w)                  //[E, 4]see if s3 is max
           maxe.w = vmax(maxe.w, s3.w)                  //[E, 3]
           p1 = cmp.eq(col_count, #0)                   //
       } {
           maxe.w = vmax(maxe.w, s2.w)                  //[E, 4]
           y3.w+= vmpyo(s3.w, recipvec.h):SSR           //
           sum0 = memw(active_sum++#8) //#2<<2)         //stride = 4
       } {
           maxe.w = vmax(maxe.w, s0.w)                  //see if s0 is max
           mine.w = vmin(mine.w, s0.w)                  //see if s0 is max
           y0.w = vmpye(s0.w, recipvec.uh)              //
       } {
           d0 =vsplat(sum0)
           maxe.w = vmax(maxe.w, s1.w)                  //
           mine.w = vmin(mine.w, s1.w)                  //see if s1 is max
           dcfetch(ptr_x0+#0<<6)                        //[1, 2]
       } {
           y3.h = vpack(y3.w, y2.w):sat                 //#sat8  <0, >255
           y0.w+= vmpyo(s0.w, recipvec.h):SSR           //
           sum1 = memw(active_sum++#8)                  //stride = 4
       } {
           s0.w = vadd(wsum.w, d0.w)
           y1.w  = vmpye(s1.w, recipvec.uh)             //
           fetch_ptr = add(ptr_x0, in_width_depth)      //
       } {
           d1 = vsplat(sum1)
           sum2 = memw(active_sum++#8)                  //stride = 4
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):SSR           //
           s1.w = vadd(wsum.w, d1.w)
           dcfetch(fetch_ptr+#0<<6)                     //[0, 3]
       } {
           d2 = vsplat(sum2)
           sum3 = memw(active_sum++#8)                  //stride = 4
           fetch_ptr = add(ptr_x0, in_width_depth)      //
       } {
           s2.w = vadd(wsum.w, d2.w)
           y1.h = vpack(y1.w, y0.w):sat                 //#>>16
           ptr_w = memw(sp+#52)                         //[P, 0]ptr_wi initialize filter pointer
           d3 = vsplat(sum3)
       } {
           s3.w = vadd(wsum.w, d3.w)
           y3.ub = vpack(y3.h, y1.h):sat                //#sat8  <0, >255 
           vmem(ptr_z++#1) = y3.new                     //#[E,  ]store 2nd 32bytes
           if(!p1) jump:t .L_width                      //
       }//end cols per line
#endif
/* ---------------------------------------------------------------------------- */
       {   p1 = cmp.eq(out_height, #0)                  //
           if(!p1.new) jump:t .L_height                 //
       }//end lines per block
/* ---------------------------------------------------------------------------- */
       {
           ptr_max = memw(PARMW(4))                     //ptr pre computed max value in output
           r17:16 = memd(sp+#0)
       } {
           r19:18 = memd(sp+#8)                         //Q
           maxo = vmem(ptr_max+#0)
       } {
           r21:20 = memd(sp+#16)                        //Q
          maxe.w = vmax(maxo.w,maxe.w)
          vmem(ptr_max+#0) = maxe.new
       } {
          r23:22 = memd(sp+#24)                        //Q
          mino = vmem(ptr_max+#1)
       } {
          mine.w = vmin(mino.w,mine.w)
          vmem(ptr_max+#1) = mine.new
       }

/* ---------------------------------------------------------------------------- */
       {
           r25:24 = memd(sp+#32)                        //Q
           r27:26 = memd(sp+#40)                        //Q
       } {
           dealloc_return                               //Q
       }
.L_end:
/*=============================================================================*/
      .size inconv2dbbb332_v60_asm, .L_end-inconv2dbbb332_v60_asm
/*=============================================================================*/
