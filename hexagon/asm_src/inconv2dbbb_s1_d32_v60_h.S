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
 *  FUNCTIONS      : gvconv2dbbb_s1_v60_asm     
 *                                          
 *  DESCRIPTION                            
 *    Perform 2d convolution using elements of size in_depth < 32. Results are
 *    scaled and saturated to 8bits. Max and Min accumulations are kept.
 *    Horizontal stride is 1 and depth = 4
 *                                       
 *  ARCHITECTURE   : QDSP6V60  + HVX     
 *
 *  REVISION HISTORY:                                    
 *  =================                                   
 *                                                     
 *  Author              Date           Comments       
 *  ------------------------------------------------------------- 
 *  DJH                 10/13/17       created                   
 *
 *  CYCLE-COUNT:                              
 *                                              
 *  MEMORY                                       
 *     CODESIZE = 848 bytes                       
 *     STACK    = 80 bytes                         
 *     ASSUMPTIONS                                  
 *      Input depth assumed 4  stride horz assumed 1
 */
/*=============================================================================*/
        .text
        .file "inconv2dbbb_s1_d32_v60_h.S"
        .global inconv2dbbb_s1_v60_asm
        .balign 32
        .type  inconv2dbbb_s1_v60_asm, @function
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
inconv2dbbb_s1_v60_asm:
/*=============================================================================*/
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define ptr_zi                 r2     //results
#define in_width               r3     //(pad_l+in_width+pad_r)
#define out_width_stride_depth r4     //next line amount 
#define out_width              r5     //is amount of work to be done
#define in_depth               r26    //0 is always 4 so not really needed
#define filt_width             r17    //1 horizontal fuilter width
#define filt_height            r8     //2 filt_height lines per filter
#define out_height             r9     //3 number of vertical lines to perform
#define ptr_max                r13    //4 maximum and minum buffer
#define recip_level            r14    //5 255 / (MAX - MIN) - used to scale to bytes
#define filt_sum               r15    //6 gemsumb
#define ptr_suma               r18    //7 gemsuma activations
#define next_suma_buf          r1     //8 stride for suma buffer
#define stride_v_h             r10    //9 stride_vert | stride_horz=1 ->M0
#define sum0                   r19    //temp sum of activations
#define sum1                   r19    //temp sum of activations
#define sum2                   r19    //temp sum of activations
#define sum3                   r19    //temp sum of activations
/*=============================================================================*/
#define fetch_ptr              r6     //horizontal fetch ptr
#define fetch_ptr0             r2     //vertical fetch ptr
#define in_width_stride_depth  r15    //in_width * stride * in_depth for next output
#define ptr_x0                 r16    //tmp pointer to activations
#define ptr_x1                 r23    //dynamic pointer to activations
#define next_outputs           r27    //jump to input ptr for next set of outputs
#define ptr_w                  r20    //pointer to weights
#define in_width_depth         r22    //width of input image in bytes
#define recip_shamt            r7     //left shift in result
#define ptr_z                  r24    //pointer to outputs
#define col_count              r25    //column count, how much of width used
#define x13x10_x03x00          r11:10 //8 activations output 1 0/4
#define x13x10                 r11    //4 activations output 1
#define x03x00                 r10    //4 activations output 0
#define x33x30_x23x20          r13:12 //8 activations output 2 3
#define x33x30                 r13    //4 activations output 3
#define x23x20                 r12    //4 activations output 2
#define x43x40                 r21
/*=============================================================================*/
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca reg
#define s0                     v0     //accumulator for output 0
#define s1                     v1     //accumulator for output 1
#define s1s0                   v1:0   //accumulator 
#define s2                     v2     //accumulator for output 2
#define s3                     v3     //accumulator for output 3
#define s3s2                   v3:2   //
#define y0                     v8     //weights/quantized output
#define y1                     v9     //weights/quantized output
#define y2                     v10    ///quantized output
#define y3                     v11    ///quantized output
#define wsum                   v14    //initialzed to in_offsey*wsum + biasoffset
#define maxomaxe               v13:12 //
#define maxe                   v12    //
#define maxo                   v13    //
#define minomine               v19:18 //
#define mine                   v18    //
#define mino                   v19    //
#define recipvec               v15    //7f80000000 / max
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
           memw(sp+#52) = ptr_wi                        //
           maxe = vnot(mine)                           // all -0x80000000
       } {
           in_depth = memw(PARMW(0))                    //
           filt_height = memw(PARMW(2))                 //extract filt_height
       } {
           filt_width = memw(PARMW(1))                  //extract filt_width
           memw(sp+#56) = ptr_zi                        //
           p3 = cmp.eq(filt_height, #1)                 //is filt height 1?
       } {   
           p2 = cmp.eq(filt_width, #1)                  //is filt width = 1?
           p0 = !tstbit(filt_width, #0)                 //is filt 1xN? - even?
           filt_width = lsr(filt_width, #1)             //
           out_height = memw(PARMW(3))                  //number of output lines
       } {
           recip_level = memw(PARMW(5))                 //
           filt_sum = memw(PARMW(6))                    //
           in_width_depth  = mpyi(in_width, in_depth)   //in_depth * in_width line
       } {
           wsum = vmem(filt_sum+#0)                     //
           recipvec = vsplat(recip_level)               //
           stride_v_h = memw(PARMW(9))                  //
       } {
           next_suma_buf = memw(PARMW(8))               //
           ptr_suma = memw(PARMW(7))                    //
           in_width_stride_depth=mpy(in_width_depth.L,stride_v_h.H)//
           stride_v_h = zxth(stride_v_h)                //
       } {
           in_depth = mpyi(in_depth, stride_v_h)        //
           stride_v_h = asl(stride_v_h, #2)             //v-h = 1 1
           memw(sp+#60) = ptr_suma                      //
           recip_shamt = memw(PARMW(10))
       } {
           next_outputs=mpyi(filt_height,in_width_depth)//filt_height*in_width*in_depth
           filt_height = add(filt_height, #-1)          //
           M0 = stride_v_h                              //
       } {
           next_outputs += mpyi(stride_v_h, #-4)        //
           memw(sp+#60) += next_suma_buf                //
       }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_height:
       {   sum0 = memw(ptr_suma++M0)                    //stride = 1
           loop1(.L_filt_height, filt_height)           //[P, 0]for(fil=0;fil<h*depth/32;fil+=1){
           out_height = add(out_height, #-1)            //
       } {
           s0 = vsplat(sum0)                            //
           sum1 = memw(ptr_suma++M0)                    //stride = 1
           ptr_w = memw(sp+#52)                         //[P, 0]ptr_wi initialize filter pointer
           col_count = out_width                        //end for height
       } {
           s0.w = vadd(wsum.w, s0.w)                    //
           s1 = vsplat(sum1)                            //
           sum2 = memw(ptr_suma++M0)                    //stride = 1
           ptr_x0 = ptr_xi                              //ptr_xi                         
       } {   
           s1.w = vadd(wsum.w, s1.w)                    //
           s2 = vsplat(sum2)                            //
           sum3 = memw(ptr_suma++M0)                    //stride = 1
           fetch_ptr0 = add(ptr_x0, in_width_depth)     //l1 fetch from next lin 
       } {
           ptr_z = memw(sp+#56)                         //ptr_zi
           memw(sp+#56) += out_width_stride_depth       //ptr_zi += out_width_stride_depth
           s2.w = vadd(wsum.w, s2.w)                    // 
           s3 = vsplat(sum3)                            //
       } {
           ptr_xi = add(ptr_xi, in_width_stride_depth)  //ptr_xi+=in_width_stride_depth  in_width*stride*in_depth)
           s3.w = vadd(wsum.w, s3.w)                    //
           y0         = vmem(ptr_w++#1)                 //[P,-1]
           fetch_ptr = fetch_ptr0                       //[P,-1]
       }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_width:
      {
           p1 = cmp.eq(recip_shamt,#0)
           if(p3) jump .L_filt_heighteq1                //[P, 0]
      }
.L_filt_height:
       {   loop0(.L_filt_widthN_1, filt_width)          //[P, 0]ki is k1/32 - 0
           x13x10_x03x00 = memd(ptr_x0+#0<<3)           //[P, 0]stride = 1
           ptr_x1 = add(ptr_x0, #8)                     //[P, 0]
           if(p2) jump .L_epi_onlyN_1                   //[P, 0]if width = 1 skip kernel
       }
        .balign 32
.L_filt_widthN_1:
       {   s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[0, 0]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[0, 0]
           x33x30_x23x20 = memd(ptr_x1++#1<<3)          //[0, 0]
       } {
           y1.cur     = vmem(ptr_w++#1)                 //[0, 1]
           s0.uw       += vrmpy(y1.ub, x13x10.ub)       //[0, 1]
           s1.uw       += vrmpy(y1.ub, x23x20.ub)       //[0, 1]
           x43x40 = memw(ptr_x1+#0<<3)                  //
       } {
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[0, 2]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[0, 2]
           y0         = vmem(ptr_w++#1)                 //[0, 2]
           x13x10_x03x00 = combine(x33x30,x23x20)       //[0, 2]x13x10_x03x00 = memd(ptr_x1++#1<<3)
       } {
           s2.uw       += vrmpy(y1.ub, x33x30.ub)       //[0, 3]
           s3.uw       += vrmpy(y1.ub, x43x40.ub)       //[0, 3]
           dcfetch(fetch_ptr+#0<<5)                     //[0, 3]
           fetch_ptr = add(fetch_ptr, #32)              //[0, 3]
       }:endloop0  
/* ---------------------------------------------------------------------------- */
           if(p0) jump .L_skipN_1
.L_epi_onlyN_1:
       {   s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[E, 0]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[E, 0]
           fetch_ptr  = add(fetch_ptr0, in_width_depth) //[E, 0]
           x33x30_x23x20 = memd(ptr_x1+#0<<3)           //[E, 1]
       } {
           fetch_ptr0 = add(fetch_ptr0, in_width_depth) //[E, 0]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[E, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[E, 1]
           y0         = vmem(ptr_w++#1)                 //[P, 1]
       }
.L_skipN_1:
       {
           ptr_x0 = add(ptr_x0, in_width_depth)         //[P, 0]
       }:endloop1
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_filt_heighteq1:
       {   loop0(.L_filt_width1, filt_width)            //[P, 0]ki is k1/32 - 0
           x13x10_x03x00 = memd(ptr_x0+#0<<3)           //[P, 0]stride = 1
           ptr_x1 = add(ptr_x0, #8)                     //[P, 0]
           if(p2) jump .L_epi_only1                     //[P, 0]if width = 1 skip kernel
       }
        .balign 32
.L_filt_width1:
       {   s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[0, 0]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[0, 0]
           x33x30_x23x20 = memd(ptr_x1++#1<<3)          //[0, 0]
       } {
           y1.cur     = vmem(ptr_w++#1)                 //[0, 1]
           s0.uw       += vrmpy(y1.ub, x13x10.ub)       //[0, 1]
           s1.uw       += vrmpy(y1.ub, x23x20.ub)       //[0, 1]
           x43x40 = memw(ptr_x1+#0<<3)                  //
       } {
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[0, 2]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[0, 2]
           y0         = vmem(ptr_w++#1)                 //[0, 2]
           x13x10_x03x00 = combine(x33x30,x23x20)       //[0, 2]x13x10_x03x00 = memd(ptr_x1++#1<<3)
       } {
           s2.uw       += vrmpy(y1.ub, x33x30.ub)       //[0, 3]
           s3.uw       += vrmpy(y1.ub, x43x40.ub)       //[0, 3]
           dcfetch(fetch_ptr+#0<<5)                     //[0, 3]
           fetch_ptr = add(fetch_ptr, #32)              //[0, 3]
       }:endloop0  
/* ---------------------------------------------------------------------------- */
.L_epi_only1:
       {
           ptr_x0 = add(ptr_x0, in_width_depth)         //[P, 0]
           if(!p1) jump .L_go_apply_shamt               /// alternate path if shamt !=0
           if(p0) jump .L_skip1
        }
       {   s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[E, 0]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[E, 0]
           x33x30_x23x20 = memd(ptr_x1+#0<<3)           //[E, 0]
      } {
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[E, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[E, 1]
       }
// *** NOTE: the code at L_go_apply_shamt needs to replicate effects of
// code above the ".L_back_from_apply_shamt" label.
/* ---------------------------------------------------------------------------- */
.L_skip1:
       {   y0.w = vmpye(s0.w, recipvec.uh)              //[E, 2]
           maxe.w = vmax(maxe.w, s0.w)                  //[E, 2]see if s0 is max
           mine.w = vmin(mine.w, s0.w)                  //[E, 2]see if s0 is max
           ptr_x0 = sub(ptr_x0, next_outputs)           //[E, 2]reset data ptr to next 4
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):SSR           //[E, 3]
           maxe.w = vmax(maxe.w, s1.w)                  //[E, 3]
           mine.w = vmin(mine.w, s1.w)                  //[E, 3]see if s1 is max
           fetch_ptr0 = sub(fetch_ptr0, next_outputs)   //[E, 0]
       }
.L_back_from_apply_shamt:
       {
           y1.w  = vmpye(s1.w, recipvec.uh)             //[E, 4]
           mine.w = vmin(mine.w, s2.w)                  //[E, 4]see if s2 is max
           maxe.w = vmax(maxe.w, s2.w)                  //[E, 4]
           sum0 = memw(ptr_suma++M0)                    //[P, 4]stride = 1
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):SSR           //[E, 5]
           mine.w = vmin(mine.w, s3.w)                  //[E, 5]see if s3 is max
           maxe.w = vmax(maxe.w, s3.w)                  //[E, 5]
           dcfetch(fetch_ptr0+#0<<5)                    //[E, 1]
       } {
           y2.w  = vmpye(s2.w, recipvec.uh)             //[E, 6]
           ptr_w = memw(sp+#52)                         //[P, 6]ptr_wi initialize filter pointer
       } {
           s0 = vsplat(sum0)                            //[P, 7]
           sum1 = memw(ptr_suma++M0)                    //[P, 7]stride = 1
           fetch_ptr = fetch_ptr0                       //[P, 7]
       } {
           s0.w = vadd(wsum.w, s0.w)                    //[P, 8]
           s1 = vsplat(sum1)                            //[P, 8]
           sum2 = memw(ptr_suma++M0)                    //[P, 8]stride = 1
       } {
           y1.h = vpack(y1.w, y0.w):sat                 //[E, 9]#>>16
           y2.w+= vmpyo(s2.w, recipvec.h):SSR           //[E, 9]
           loop1(.L_filt_height, filt_height)           //[P, 9]for(fil=0;fil<h*depth/32;fil+=1){
       } {
           y3.w  = vmpye(s3.w, recipvec.uh)             //[E,10]#
           y0         = vmem(ptr_w++#1)                 //[P,10]
           dcfetch(fetch_ptr0+#1<<5)                    //[E, 3]
       } {
           y3.w+= vmpyo(s3.w, recipvec.h):SSR           //[E,11]
           s1.w = vadd(wsum.w, s1.w)                    //[P,11]
           fetch_ptr0 = add(ptr_x0, in_width_depth)     //[P, 6]l1 fetch from next lin 
       } {
           s2 = vsplat(sum2)                            //[P,12]
           sum3 = memw(ptr_suma++M0)                    //[P,12]stride = 1
           col_count = add(col_count, #-4)              //[E,12]
           y3.h = vpack(y3.w, y2.w):sat                 //[E,12]#sat8  <0, >255
       } {
           p1 = cmp.eq(col_count, #0)                   //[E,13]
           s2.w = vadd(wsum.w, s2.w)                    //[P,13]
           s3 = vsplat(sum3)                            //[P,13]
       } {
           s3.w = vadd(wsum.w, s3.w)                    //[P,14]
           y3.ub = vpack(y3.h, y1.h):sat                //[E,14]#sat8  <0, >255 
           vmem(ptr_z++#1) = y3.new                     //[E,14]next weidthstore 4*32bytes
           if(!p1) jump:t .L_width                      //[E,14]
       }//end cols per line
/* ---------------------------------------------------------------------------- */
       {
           ptr_suma = memw(sp+#60)                      //next suma line
           memw(sp+#60) += next_suma_buf                //
           p1 = cmp.eq(out_height, #0)                  //
           if(!p1.new) jump:t .L_height                 //next height
       }//end lines per block
/* ---------------------------------------------------------------------------- */
 /* scale mine, maxe according to recipvec ; apply to overall range */
       {
           mino.w = vmpye(mine.w, recipvec.uh)
           ptr_max = memw(PARMW(4))                     //ptr pre computed max value in output
       } {
           mine = vmem(ptr_max+#1)
           mino.w+= vmpyo(mine.w, recipvec.h):SSR
       } {
           mine.w = vmin( mine.w, mino.w)
           vmem(ptr_max+#1) = mine.new                      //[E, 2]
           maxo.w = vmpye(maxe.w, recipvec.uh)
       } {
           maxo.w+= vmpyo(maxe.w, recipvec.h):SSR
           maxe = vmem(ptr_max+#0)
       } {
           maxe.w = vmax( maxe.w, maxo.w)
           vmem(ptr_max+#0) = maxe.new                      //[E, 1]
       }
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)                         //restore stack
           r19:18 = memd(sp+#8)                         //Q
       } {
           r21:20 = memd(sp+#16)                        //Q
           r23:22 = memd(sp+#24)                        //Q
       } {
           r25:24 = memd(sp+#32)                        //Q
           r27:26 = memd(sp+#40)                        //Q
       } {
           dealloc_return                               //Q
       }

/*=============================================================================*/
.L_go_apply_shamt:		// still need to process 'if(p0) jump skip'
       if(p0) jump .L_skip1x
       {   s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[E, 0]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[E, 0]
           x33x30_x23x20 = memd(ptr_x1+#0<<3)           //[E, 0]
      } {
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[E, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[E, 1]
       }
/* ---------------------------------------------------------------------------- */
.L_skip1x:
// shift s0,s1,s2,s3 << recip_shamt
//  and:
//   find y0 = prod s0*recip_shamt
//   apply s0,s1 to min/max
//   adjust ptr_x0 and fetch_ptr0
//
       {
          ptr_x0 = sub(ptr_x0, next_outputs)           //[E, 2]reset data ptr to next 4
          fetch_ptr0 = sub(fetch_ptr0, next_outputs)   //[E, 0]
          s0.w = vasl(s0.w,recip_shamt)
       } {
           maxe.w = vmax(maxe.w, s0.w)                  //[E, 2]see if s0 is max
           mine.w = vmin(mine.w, s0.w)                  //[E, 2]see if s0 is max
           s1.w  = vasl(s1.w,recip_shamt)
       } {
           maxe.w = vmax(maxe.w, s1.w)                  //[E, 2]see if s1 is max
          y0.w = vmpye(s0.w, recipvec.uh)              //[E, 2]
           s2.w = vasl(s2.w,recip_shamt)
       } {
           mine.w = vmin(mine.w, s1.w)                  //[E, 2]see if s1 is max
           y0.w+= vmpyo(s0.w, recipvec.h):SSR           //[E, 3]
           s3.w = vasl(s3.w,recip_shamt)
           jump  .L_back_from_apply_shamt
      }
.L_end:
/*=============================================================================*/
      .size inconv2dbbb_s1_v60_asm, .L_end-inconv2dbbb_s1_v60_asm
/*=============================================================================*/
