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
 *  FUNCTIONS      : gvconv2dbbb_v60_asm     
 *                                          
 *  DESCRIPTION                            
 *    Perform 2d convolution using elements of size in_depth. Results are
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
 *        
 *        
 *        
 *  C MODEL                                             
 */
#if 0
#endif
/*=============================================================================*/
        .text
        .file "gvconv2dbbb_d32_v60_h.S"
        .global gvconv2dbbb_v60_asm
        .balign 32
        .type  gvconv2dbbb_v60_asm, @function
gvconv2dbbb_v60_asm:
/*=============================================================================*/
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define ptr_zi                 r2     //results
#define in_width               r3     //(pad_l+in_width+pad_r)
#define out_width_stride_depth r4     //next line amount 
#define out_width              r5     //is amount of work to be done
#define stride_h_w             r6     //0 stride|in_depth 
#define in_depth               r27    //1 input depth multiples of 32
#define filt_width             r5     //2 horizontal fuilter width
#define filt_height            r8     //3 filt_height lines per filter
#define out_height             r9     //4 number of vertical lines to perform
#define ptr_filtsum            r10    //5 includes the computation filt_sum * in_offset + biasvec
#define ptr_sumabuf0           r1     //6 filt_offset*in_offset - in_offset*sum(in)
#define next_sumabuf           r26    //7 stride to get to next row of suma values 
#define ptr_max                r7     //8 maximum and minum buffer
#define recip_level_ptr        r14    //9  pointer to  32 of int32, 255 / (MAX - MIN) - used to scale to bytes
#define zshift                 r7     //10 amount to << results (re-use ptr_max)

/*=============================================================================*/
#define ptr_sumabuf            r27    //temp value of suma buffer ptr
#define in_width_stride_depth  r15    //in_width * stride * in_depth for next output
#define fetch_ptr0             r17    //base fetch pointer
#define ptr_x0                 r16    //tmp pointer to activations
#define ptr_x1                 r23    //dynamic pointer to activations
#define fetch_ptr              r14    //dynamic fetch pointer
#define sum_stride             r21    //step between suma values 4 or 8 bytes
#define stride_h               r6     //vertical stride
#define stride3_w              r6     //3*32*stride_w
#define stride_w               r18    //32*stride_w
#define next_outputs           r19    //jump to input ptr for next set of outputs
#define fetch_offset           r21
#define fetch_out              r28
#define ptr_w                  r20    //pointer to weights
#define in_width_32            r22    //width of input image in bytes
#define c4                     r2     //shuffle size in final max and min find
#define ptr_z                  r24    //pointer to outputs
#define col_count              r25    //column count, how much of width used
#define col_count_ptr_z        r25:24 //packed for double word read
#define x07x04_x03x00          r11:10 //8 activations output 0
#define x07x04                 r11    //4 activations output 0
#define x03x00                 r10    //4 activations output 0
#define x17x14_x13x10          r13:12 //8 activations output 1
#define x17x14                 r13    //4 activations output 1
#define x13x10                 r12    //4 activations output 1
#define x27x24_x23x20          r11:10 //8 activations output 2
#define x27x24                 r11    //4 activations output 2
#define x23x20                 r10    //4 activations output 2
#define x37x34_x33x30          r3:2   //8 activations output 3
#define x37x34                 r3     //4 activations output 3
#define x33x30                 r2     //4 activations output 3
#define sum0                   r2     //suma for output 0
#define sum1                   r3     //suma for output 1
#define sum2                   r2     //suma for output 2
#define sum3                   r3     //suma for output 3
/*=============================================================================*/
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define PS(VSRC) nop //.word (0x1DFFE100+VSRC) //debug vec reg
#define s0                     v0     //accumulator for output 0
#define s1                     v1     //accumulator for output 1
#define s1s0                   v1:0   //accumulator 
#define s2                     v2     //accumulator for output 2
#define s3                     v3     //accumulator for output 3
#define s3s2                   v3:2   //
#define d0                     v4     //
#define d1                     v5     //
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
#define vrnd                   v20    //
#define sk                     v21    //
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/*=============================================================================*/

       {   allocframe(#72)                              // 0th entry on stack is (72+8)/4 =20 ints
       } {  
           memd(sp+#0)  = r17:16                        //
           memd(sp+#24) = r23:22                        //
           r23 = ##0x80000001                           // init for max
       } {
           memd(sp+#8)  = r19:18                        //
           memd(sp+#16) = r21:20                        //
           maxe = vsplat(r23)                           //
       } {
           memw(sp+#52) = ptr_wi                        //
           memd(sp+#40) = r27:26                        //
           mine.w = vabs(maxe.w)                        // +0x7fffffff
       } {
           zshift = memw(sp+#30<<2)                     // amount to << the products
           ptr_sumabuf0 = memw(sp+#26<<2)               //read in the ptr to the suma buffer value
       } {
           p3 = cmp.gt(zshift,#0)                       // do we need to do << on the sums?
           memw(sp+#56) = ptr_zi                        //
           memw(sp+#60) = out_width                     //
       } {
           stride_h_w = memw(sp+#20<<2)                 //extract stride*depth 
           in_depth   = memw(sp+#21<<2)                 //
       } {
           filt_width = memw(sp+#22<<2)                 //extract filt_width
           filt_height = memw(sp+#23<<2)                //extract filt_height 
           stride_w = zxth(stride_h_w)                  //
           stride_h = lsr(stride_h_w, #16)              //
       } {
           filt_width = asl(filt_width, #2)             //
           filt_height=mpy(filt_height.L,in_depth.L)    //filt_height*in_depth
           fetch_out = sub(filt_width,stride_w)         // filt_width - stride_w
           dcfetch(ptr_sumabuf0+#0)                     //fetch the suma values
       } {
           filt_height = lsr(filt_height, #5)           //filt_height * in_depth / 32
           fetch_out = mpyi(fetch_out,#32*3)            //
           ptr_filtsum = memw(sp+#25<<2)                //ptr pre computed weight sum
           out_height  = memw(sp+#24<<2)                //number of output lines
       } {
           memd(sp+#32) = r25:24                        //
           recip_level_ptr = memw(sp+#29<<2)               //
           filt_width = add(filt_width, #-1)            //account for software pipeline
           in_width_32 = asl(in_width, #5)              //32 * in_width d32 line
       } {
           sum_stride = asl(stride_w, #2)               //stride for th sum data 1 or 2 words
           recipvec = vmem(recip_level_ptr+#0)               //
           next_sumabuf = memw(sp+#27<<2)               //stride of ptr pre computed input sum
       } {
           M0 = sum_stride                              //allow stride by 4bytes or 8bytes
           wsum = vmem(ptr_filtsum+#0)                  //
           in_width = mpyi(in_width, in_depth)          //
           ptr_sumabuf = ptr_sumabuf0                   //
       } {
           next_outputs=mpyi(filt_height,in_width_32)   //filt_height*in_width*in_depth
           stride_w = asl(stride_w, #7)                 //128 or 256
           filt_height = add(filt_height, #-1)          //peel off 1 itn of outer loop
           dcfetch(ptr_xi+#2<<5)                        //
       } {
           fetch_offset = #0                            //
           next_outputs = sub(next_outputs, stride_w)   //1*128,2*128
           sum0 = memw(ptr_sumabuf++M0)                 //[P, 0]sum 0
           dcfetch(ptr_xi+#3<<5)                        //
       } {
           in_width_stride_depth=mpyi(in_width,stride_h)//
           d0 = vsplat(sum0)                            //[P, 1]
           sum1 = memw(ptr_sumabuf++M0)                 //[P, 1]sum 1
       } {
           stride_w = lsr(stride_w, #2)                 //32 or 64
           s0.w = vadd(wsum.w, d0.w)                    //[P, 2]
           d1 = vsplat(sum1)                            //[P, 2]
           sum2 = memw(ptr_sumabuf++M0)                 //[P, 2]sum 2
       } {
           stride3_w = addasl(stride_w, stride_w, #1)   //3*stride, 32*3 or 64*3
           s1.w = vadd(wsum.w, d1.w)                    //[P, 3]
           d2 = vsplat(sum2)                            //[P, 3]
           sum3 = memw(ptr_sumabuf++M0)                 //[P, 3]sum 3
       }{
           fetch_out = max(fetch_out,fetch_offset)      //if negative, set to 0
           s2.w = vadd(wsum.w, d2.w)                    //[P, 4]
           d3 = vsplat(sum3)                            //[P, 4]
       } 
/* ---------------------------------------------------------------------------- */
//      .balign 32
.L_height:
       {   ptr_x0 = ptr_xi                              //ptr_xi
           col_count_ptr_z = memd(sp+#56<<0)            //out_width & ptr_zi on stack //
           out_height = add(out_height, #-1)            //
           fetch_ptr0 = add(ptr_xi, in_width_32)        //[P, 4]fetch from next line 
       } {
           memw(sp+#56<<0) += out_width_stride_depth    //ptr_zi next output line for depth segmnt
           p0 = cmp.eq(out_height,#0)                   // last iteration?
           ptr_x1 = ptr_x0                              //[P, 5]
           loop1(.L_filt_height, filt_height)           //[P, 0]for(fil=0;fil<h*depth/32;fil+=1){
       } {
           ptr_w = memw(sp+#52<<0)                      //[P, 0]ptr_wi initialize filter pointer
           if (!p0) ptr_sumabuf0=add(ptr_sumabuf0,next_sumabuf)  
                                                        // ptr_sumabuf += next_sumabuf
           if (!p0) ptr_xi = add(ptr_xi, in_width_stride_depth)  
                                                        //ptr_x+=in_width*stride*in_depth)    
           if ( p0) fetch_offset = fetch_out            //
       }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_width:
       {   s3.w = vadd(wsum.w, d3.w)                    //[P, 5]
           p0 = cmp.eq(filt_height, #0)                 //if in_depth = 32 and filt_height = 1
           p1 = cmp.gt(col_count,#4)                    //p1 = !(last iteration of L_width loop)
           if (!p1.new) fetch_ptr0 = sub(fetch_ptr0,fetch_offset)
                                                        //fetch_offset =0 except last iteration of L_height loop
       }{
           fetch_ptr = fetch_ptr0                       //
           col_count = add(col_count, #-4)              //
           x27x24_x23x20 = memd(ptr_x0+stride_w<<#1)    //[0, 0]
           if (p0) jump:nt .L_peel1                     //peel off 1 iteration
       }
/* ---------------------------------------------------------------------------- */
.L_filt_height:
       {   loop0(.L_filt_widthN_1, filt_width)          //[P, 0]ki is k1/32 - 0
           x37x34_x33x30 = memd(ptr_x0+stride3_w<<#0)   //[0, 0]
           ptr_x0 = add(ptr_x0, in_width_32)            //[E, 7]next line ptr_y keeps going
           dcfetch(fetch_ptr+#0<<5)                     //[1, 3]
       }{
           y0.cur     = vmem(ptr_w++#1)                 //[0, 1]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[0, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[0, 1]
           x17x14_x13x10 = memd(ptr_x1+stride_w<<#0)    //[0, 1]
       } 
/* ---------------------------------------------------------------------------- */
.L_filt_widthN_1:
       {   y1.cur     = vmem(ptr_w++#1)                 //[0, 2]
           s2.uw       += vrmpy(y1.ub, x27x24.ub)       //[0, 2]
           s3.uw       += vrmpy(y1.ub, x37x34.ub)       //[0, 2]
           x07x04_x03x00 = memd(ptr_x1++#1<<3)          //[0, 2]
       } {
           s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[0, 3]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[0, 3]
           dcfetch(fetch_ptr+#1<<5)                     //[0, 3]prefetch the right line
           x37x34_x33x30 = memd(ptr_x1+stride3_w<<#0)   //[1, 0]
       } {
           s0.uw       += vrmpy(y1.ub, x07x04.ub)       //[0, 4]
           s1.uw       += vrmpy(y1.ub, x17x14.ub)       //[0, 4]
           x27x24_x23x20 = memd(ptr_x1+stride_w<<#1)    //[1, 0]
           fetch_ptr = add(fetch_ptr, #32)              //[0, 4]advance fetch by 1 line 
       } { 
           y0.cur     = vmem(ptr_w++#1)                 //[1, 1]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[1, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[1, 1]
           x17x14_x13x10 = memd(ptr_x1+stride_w<<#0)    //[1, 1]
       }:endloop0  
/* ---------------------------------------------------------------------------- */
       {   y1.cur     = vmem(ptr_w++#1)                 //[1, 2]
           s2.uw       += vrmpy(y1.ub, x27x24.ub)       //[1, 2]
           s3.uw       += vrmpy(y1.ub, x37x34.ub)       //[1, 2]
           x07x04_x03x00 = memd(ptr_x1+#0<<3)           //[1, 2]
       } {
           s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[1, 3]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[1, 3]
           ptr_x1 = ptr_x0                              //[P, 2]
           fetch_ptr0 = add(fetch_ptr0, in_width_32)    //[P, 4]fetch from next line
       } {
           s0.uw       += vrmpy(y1.ub, x07x04.ub)       //[1, 4]
           s1.uw       += vrmpy(y1.ub, x17x14.ub)       //[1, 4]
           x27x24_x23x20 = memd(ptr_x0+stride_w<<#1)    //[P, 4]
           fetch_ptr = fetch_ptr0                       //
       }:endloop1
/* ----------------------------------------------------------------- */
.L_peel1:
       {   loop0(.L_filt_width1, filt_width)            //[P, 0]ki is k1/32 - 0
           x37x34_x33x30 = memd(ptr_x0+stride3_w<<#0)   //[0, 0]
           fetch_ptr = sub(fetch_ptr0, next_outputs)    //[E, 7]move to column +in_width_32 +next_outputs 
       }{
           if (!p1) fetch_ptr = ptr_xi                  //if last iteration, move to next row
       }
/* ---------------------------------------------------------------------------- */
        .balign 32
.L_filt_width1:
       {   y0.cur     = vmem(ptr_w++#1)                 //[0, 1]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[0, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[0, 1]
           x17x14_x13x10 = memd(ptr_x1+stride_w<<#0)    //[0, 1]
       } {
           y1.cur     = vmem(ptr_w++#1)                 //[0, 2]
           s2.uw       += vrmpy(y1.ub, x27x24.ub)       //[0, 2]
           s3.uw       += vrmpy(y1.ub, x37x34.ub)       //[0, 2]
           x07x04_x03x00 = memd(ptr_x1++#1<<3)          //[0, 2]
       } {
           s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[0, 3]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[0, 3]
           dcfetch(fetch_ptr+#0<<5)                     //[0, 3]final prefetch sequence for this column
           x37x34_x33x30 = memd(ptr_x1+stride3_w<<#0)   //[1, 0]
       } {
           s0.uw       += vrmpy(y1.ub, x07x04.ub)       //[0, 4]
           s1.uw       += vrmpy(y1.ub, x17x14.ub)       //[0, 4]
           x27x24_x23x20 = memd(ptr_x1+stride_w<<#1)    //[1, 0]
           fetch_ptr = add(fetch_ptr, #32)              //[0, 3]final fetch advance
       }:endloop0
/* ---------------------------------------------------------------------------- */
       {   y0.cur     = vmem(ptr_w++#1)                 //[1, 1]
           s2.uw       += vrmpy(y0.ub, x23x20.ub)       //[1, 1]
           s3.uw       += vrmpy(y0.ub, x33x30.ub)       //[1, 1]
           x17x14_x13x10 = memd(ptr_x1+stride_w<<#0)    //[1, 1]
       } {
           y1.cur     = vmem(ptr_w++#1)                 //[1, 2]
           s2.uw       += vrmpy(y1.ub, x27x24.ub)       //[1, 2]
           s3.uw       += vrmpy(y1.ub, x37x34.ub)       //[1, 2]
           x07x04_x03x00 = memd(ptr_x1+#0<<3)           //[1, 2]
       } {
           s0.uw       += vrmpy(y0.ub, x03x00.ub)       //[1, 3]
           s1.uw       += vrmpy(y0.ub, x13x10.ub)       //[1, 3]
           if (!p1) ptr_sumabuf = ptr_sumabuf0          // 
       } {
           y2.w  = vmpye(s2.w, recipvec.uh)             //
           ptr_x0 = add(ptr_x0, in_width_32)            //[E, 7]next line ptr_y keeps going
           dcfetch(ptr_sumabuf+#0<<5)                   //fetch next suma's
           if( p3 ) jump:nt .L_do_zshift                // go do (s0,s1,s2,s3) <<= zshift
       } { //** note ** effect of following packet needs to be duplicated at .L_do_zshift
           s0.uw       += vrmpy(y1.ub, x07x04.ub)       //[1, 4]
           s1.uw       += vrmpy(y1.ub, x17x14.ub)       //[1, 4]
           dcfetch(fetch_ptr+#0<<5)                     //fetch line 
           ptr_x0 = sub(ptr_x0, next_outputs)           //reset data ptr to next 4
       }
.L_done_zshift:
/* ---------------------------------------------------------------------------- */
       {   y2.w+= vmpyo(s2.w, recipvec.h):SSR 
           maxe.w = vmax(maxe.w, s0.w)                  //see if s0 is max
           p2 = cmp.gt(col_count,#-3)                   //should s1 be included ?
           sk = s0                                      //initialize sk with s0
       } {
           y0.w  = vmpye(s0.w, recipvec.uh)             //
           mine.w = vmin(mine.w, s0.w)                  //see if s0 is min
           if (p2) sk = s1                              //inclued s1 if needed
           p2 = cmp.gt(col_count,#-2)                   //should s2 be included ?
       } {
           y0.w+= vmpyo(s0.w, recipvec.h):SSR           //
           maxe.w = vmax(maxe.w, sk.w)                  //see if s1 is max
           mine.w = vmin(mine.w, sk.w)                  //see if s1 is min
           ptr_w = memw(sp+#52<<0)                      //[P, 0]ptr_wi initialize filter pointer
       } {
           y3.w  = vmpye(s3.w, recipvec.uh)             //
           if (p2) sk = s2                              //include s2 if needed
           ptr_x1 = ptr_x0                              //[P, 5]
           //loop1(.L_filt_height, filt_height)         //[P, 1]for(fil=0;fil<h*depth/32;fil+=1){
           lc1 = filt_height                            //allow future code movement
       } {
           y3.w+= vmpyo(s3.w, recipvec.h):SSR           //
           maxe.w = vmax(maxe.w, sk.w)                  //see if s2 is max
           mine.w = vmin(mine.w, sk.w)                  //see if s2 is min
           p2 = cmp.gt(col_count,#-1)                   //should s3 be included ?
       } {
           y1.w  = vmpye(s1.w, recipvec.uh)             //
           if (p2) sk = s3                              //include s3 if needed
           sum0 = memw(ptr_sumabuf++M0)                 //[P, 0]sum 0
       } {
           y1.w+= vmpyo(s1.w, recipvec.h):SSR           //
           maxe.w = vmax(maxe.w, sk.w)                  //see if s2 is max
           mine.w = vmin(mine.w, sk.w)                  //see if s3 is min
           sum1 = memw(ptr_sumabuf++M0)                 //[P, 1]sum 1
       } {
           y3.h = vpack(y3.w, y2.w):sat                 //#sat8  <0, >255
           d0 = vsplat(sum0)                            //[P, 1]
           d1 = vsplat(sum1)                            //[P, 2]
           sum2 = memw(ptr_sumabuf++M0)                 //[P, 2]sum 2
       } {
           y1.h = vpack(y1.w, y0.w):sat                    //#>>16
           fetch_ptr0 = add(ptr_x0, in_width_32)        //[P, 4]fetch from next line
           sum3 = memw(ptr_sumabuf++M0)                 //[P, 3]sum 3
       } {
           s0.w = vadd(wsum.w, d0.w)                    //[P, 2]
           s1.w = vadd(wsum.w, d1.w)                    //[P, 3]
           d2 = vsplat(sum2)                            //[P, 3]
           d3 = vsplat(sum3)                            //[P, 4]
       } {
           y3.ub = vpack(y3.h, y1.h):sat                //#sat8  <0, >255 
           vmem(ptr_z++#1):nt = y3.new                  //#[E,  ]store 2nd 32bytes
           s2.w = vadd(wsum.w, d2.w)                    //[P, 4]
           if (p1) jump:t .L_width                      //
       }//end cols per line
/* ---------------------------------------------------------------------------- */
       {   p0 = cmp.eq(out_height, #0)                  //
           if(!p0.new) jump:t .L_height                 //
       }//end lines per block
/* ---------------------------------------------------------------------------- */


// scale mine,maxe according to scales
// find min/max reduced over previous min/max;
// We don't need to << the min/max by zshift, since they were reduced
// from the values taken after the << by zshift.
//
       {
           y0.w  = vmpye(maxe.w, recipvec.uh)
           ptr_max = memw(sp+#28<<2)                    // ptr to existing max/min values
       } {
           s0  = vmem(ptr_max+#0)                      // previous max
           y0.w+= vmpyo(maxe.w, recipvec.h):SSR
       } {
           maxe.w = vmax(y0.w, s0.w)                   // new max
           s1  = vmem(ptr_max+#1)                      // previous min
           y1.w  = vmpye(mine.w, recipvec.uh)
       } {
           vmem(ptr_max+#0) = maxe
           y1.w+= vmpyo(mine.w, recipvec.h):SSR
       }

/*=============================================================================*/
       {
           vmem(ptr_max+#1) = mine.new                  //
           mine.w = vmin(s1.w, y1.w);                   // new min
       } {
           r17:16 = memd(sp+#0)                         //restore stack
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

// broken-out code to << s0,s1,s2,s3 all by zshift
// But, at time of branch, sums s0,s1 aren't yet completed;
// and there are some side effects from a skipped packet that
// we need to duplicate before going back.
// Also, y2=s2*scale product is partly done when the branch occurs, so that
// needs to be redone with the new s2.
//
       .balign 32
.L_do_zshift:
       {
           s2.w = vasl(s2.w, zshift )
           s0.uw       += vrmpy(y1.ub, x07x04.ub)       //[1, 4]
           s1.uw       += vrmpy(y1.ub, x17x14.ub)       //[1, 4]
       } {
           s3.w = vasl(s3.w, zshift )
           dcfetch(fetch_ptr+#0<<5)                     //fetch line
           ptr_x0 = sub(ptr_x0, next_outputs)           //reset data ptr to next 4
       } {
          s0.w = vasl(s0.w, zshift)
          y2.w  = vmpye(s2.w, recipvec.uh);             // redo the s2->y2 partial calc
       } {
          s1.w = vasl(s1.w, zshift)
          jump .L_done_zshift;
       }

.L_end:
/*=============================================================================*/
      .size gvconv2dbbb_v60_asm, .L_end-gvconv2dbbb_v60_asm
/*=============================================================================*/
