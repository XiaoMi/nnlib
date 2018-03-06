/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
 *  FUNCTIONS      : gvconv2dbbb_s2_v60_asm
 *
 *  DESCRIPTION
 *   Perform 2d convolution using elements along depth, do only simple
 *   convolution. Stride horizontally by 2.
 *   Sums are scaled and saturated to 8bits. Max and Min accumulations are kept.
 *
 *  ARCHITECTURE   : QDSP6V60  + HVX
 *
 *  REVISION HISTORY:
 *  =================
 *
 *  Author              Date           Comments
 *  -------------------------------------------------------------
 *  DJH                 10/10/17       created
 *
 *  CYCLE-COUNT:
 *
 *  MEMORY
 *     CODESIZE = 768 bytes
 *     STACK    = 80 bytes
 *     ASSUMPTIONS
 */
#if 0
void dwconv2d_cn(
   uint8_t *in_buf, int in_width, int in_height, int depth, 
   int stride_width, int stride_height, int in_offset,
   uint8_t  *filt,   int filt_width, int filt_height, int filt_offset,
   int *out_buf, int out_width, int out_height, int adj_x, int adj_y)
{
   int out_y, in_y_base, out_x, in_x_base;
   int out_z, filt_y, filt_x, in_element, filt_element, sum;
   int * outstripe;
   uint8_t * instripe;
   uint8_t * filtstripe;

   for (out_y = 0; out_y < out_height; out_y++) {
       in_y_base = out_y * stride_height - adj_y;
       for (out_x = 0; out_x < out_width; out_x++) {
           in_x_base = out_x * stride_width - adj_x;
           outstripe = out_buf+(depth*(out_x+ out_width*out_y));
           for (out_z = 0; out_z < depth; out_z++) {
               sum = 0;
               for (filt_y = 0; filt_y < filt_height; filt_y++) {
                   if ((in_y_base + filt_y) >= in_height) continue;
                   if ((in_y_base + filt_y) < 0) continue;
                   for (filt_x = 0; filt_x < filt_width; filt_x++) {
                      if ((in_x_base + filt_x) >= in_width) continue;
                      if ((in_x_base + filt_x) < 0) continue;

                      filtstripe = filt+(depth*(filt_x+ filt_width*filt_y));
                      filt_element = filtstripe[out_z] - filt_offset;

                      instripe = in_buf+(depth*(in_x_base + filt_x + in_width*(in_y_base + filt_y))) ;
                      in_element = instripe[out_z] - in_offset;

                      sum += in_element*filt_element;
                  }
               }
               outstripe[out_z] = sum;
           }
       }
   }
   return;
}
#endif
#if 0
/*=============================================================================*/
        .text
        .file "dwconv2dbbb_unsigned_s2_d32_v60_h.S"
        .global dwconv2dbbb_unsigned_s2_v60_asm
        .balign 32
        .type  dwconv2dbbb_unsigned_s2_v60_asm, @function
dwconv2dbbb_unsigned_s2_v60_asm:
/*=============================================================================*/
//stride assumed 2 filt width assumed 3 - horz stride 1 uses stride1  only 
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define ptr_zi                 r2     //results
#define next_in_width_depth    r3     //width*depth*(stride==2)
#define next_out_width_depth   r4     //next output line amount in bytes
#define next_in_width_32       r5     //width*32*(stride==2)
#define next_out_width_32      r16    //0 next output line amount in bytes
#define in_depth               r17    //1 total in depth split into rows of depth 32
#define out_width              r18    //2 is amount of work to be done
#define out_height             r19    //3 number of vertical lines to perform
#define filt_height            r20    //4 filt_height lines per filter
#define ptr_max                r21    //5 maximum and minum buffer
#define recip_level            r22    //6 255 / (MAX - MIN) - used to scale to bytes
#define filt_sumi              r23    //7 gemsumb
#define stride_height          r24    //8 vertical strideing any number
#define zshift                 r26    //9 shift correction for small accs
#define padding                r27    //10 padding = 1 then shift 8 pad else shift 0
#define filt_offset            r6     //11
//-----------------------------------------------------------------
#define c4                     r6     //integer max find
#define s16                    r9     //const = 16
#define in_wide_deep_high_256  r8     //width*depth*filt_height - 256
#define depth                  r10    //current depth used 
#define ptr_x0                 r11    //widthdata ptr
#define ptr_x1                 r12    //depth ptr data
#define ptr_z0                 r13    //depth otuput ptr
#define ptr_z1                 r14    //width output ptr
#define ptr_w                  r15    //ptr to weights
#define filt_sum               r22    //ptr to gemsumb
#define col_count              r25    //width count down
#define out_wide_deep_128      r7     //advance ptr 128 along and pack to current line start
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
//-----------------------------------------------------------------
#define vrecip                 v0     //32bit recprocal scale 1/max-min
#define s00                    v2     //
#define s01                    v3     //
#define s01_s00                v3:2   //
#define s10                    v4     //
#define s11                    v5     //
#define s00_s                  v7     //
#define s01_s                  v8     //
#define s10_s                  v9     //
#define s11_s                  v17    //
#define d1_d0                  v11:10 //
#define d3_d2                  v13:12 //
#define d0                     v10    //
#define d1                     v11    //
#define d2                     v12    //
#define d3                     v13    //
#define d1d0                   v10    //
#define d3d2                   v12    //
#define d3_d0                  v12    //
#define y0                     v21    //
#define y1                     v24    //
#define y2                     v16    //
#define x0                     v13    //
#define x1                     v10    //
#define x2                     v10    //
#define z3210                  v26    //
#define z5432                  v28    //
#define z7654                  v29    //
#define zba98                  v9     //
#define z9876                  v6     //
#define z54__                  v19    //
#define z__76                  v20    //
#define z5476                  v27    //
#define w_210                  v22    //
#define ww210                  v18    //
#define u_210                  v23    //
#define maxo_maxe              v31:30 //
#define mino_mine              v15:14 //
#define maxe                   v30    //
#define mine                   v14    //
#define maxo                   v31    //
#define mino                   v15    //
#define C0                     v1
#define C1                     v25
#define C2                     v31
#define C3                     v15
#define SSR        <<1:rnd:sat:shift  //simplify mpy instruction
/*=============================================================================*/
        {  allocframe(#72)                             // 0th entry on stack is (72+8)/4 =20 ints
           maxe = #0                                   //
        } {
           memd(sp+#0)  = r17:16                       //
           memd(sp+#8)  = r19:18                       //
        } {
           memd(sp+#16) = r21:20                       //
           memd(sp+#24) = r23:22                       //
           s16= #16                                    //
        } {
           memd(sp+#32) = r25:24                       //
           memd(sp+#40) = r27:26                       //  
           r23 = ##0x7fffffff                          //max pos
        } {
           mine = vsplat(r23)                          //
           zshift = memw(sp+#29<<2)                    //1 - #8 2 - 0
           filt_offset = memw(sp+#31<<2)               //
        } {
           maxe.w = vsub(maxe.w, mine.w)               //
           next_out_width_32 = memw(sp+#20<<2)         //
           in_depth = memw(sp+#21<<2)                  //
           filt_offset = vsplatb(filt_offset)          //
        } {
           out_width = memw(sp+#22<<2)                 //
           out_height = memw(sp+#23<<2)                //
           filt_offset = asl(filt_offset, #8)          //FFF0
        } {
           recip_level = memw(sp+#26<<2)               //
           padding = memw(sp+#30<<2)                   //1 - #8 2 - 0
           out_wide_deep_128=add(next_out_width_depth,#-128) //
           filt_offset = lsr(filt_offset, #8)          //0FFF
        } {
           vrecip = vsplat(recip_level)                //
           filt_sumi = memw(sp+#27<<2)                 //
           out_wide_deep_128=add(out_wide_deep_128,next_out_width_32) //
        } {
           filt_height = memw(sp+#24<<2)               //
           stride_height = memw(sp+#28<<2)             //skip n vert lines
           in_wide_deep_high_256=add(next_in_width_depth, #-256) //
           filt_offset = asl(filt_offset, padding)     //make FFF0 instead of 0FFF
        } {
           in_depth = lsr(in_depth, #5)                // 1/32
           filt_height = add(filt_height, #-1)         //
           in_wide_deep_high_256=add(in_wide_deep_high_256, next_in_width_32) //
           col_count = out_width                       //
        }
         .balign 32
.L_height:
        {  ptr_z0 = ptr_zi                             //
           ptr_x0 = ptr_xi                             //
           ptr_zi=add(ptr_zi,next_out_width_depth)     //
           ptr_xi+=mpyi(next_in_width_depth,stride_height) //
        }
         .balign 32
.L_width:
        {  x0.tmp = vmem(ptr_x0+#0)                    //[0,0]
           y0.b = vshuff(x0.b)                         //[0,0]
           ptr_x1 = ptr_x0                             //[P,0]
           loop1(.L_depth, in_depth)                   //[P,0]
        } {
           x1.tmp = vmem(ptr_x1+#1)                    //[0,1]
           y1.b = vshuff(x1.b)                         //[0,1]
           ptr_x0 = add(ptr_x0, next_in_width_32)      //[P,1]
           ptr_w = ptr_wi                              //restart filter stream
        } {
           x2.tmp = vmem(ptr_x1+#2)                    //[0,2]
           y2.b = vshuff(x2.b)                         //[0,2]
           ptr_z1 = ptr_z0                             //[P,2]
           ptr_z0 = add(ptr_z0, next_out_width_32)     //[P,2]
        } {
           z3210.b = vshuff(y0.b)                      //[0,3]x3210
           ptr_x1 =add(ptr_x1, next_in_width_depth)    //[0,3]move to next pt in same depth position
           s10 = vmem(filt_sumi+#0)                    //[P,3]
           loop0(.L_vloop, filt_height)                //[P,3]can have a filter of Nx3 stride = 1
        } {
           u_210.tmp = vmem(ptr_w++#1)                 //[0,4]
           w_210.w = vasl(u_210.w, padding)            //[0,4]
           z7654.b = vshuff(y1.b)                      //[0,4]x7654
           filt_sum = add(filt_sumi, #128)             //[P,4]
        } {
           zba98.b = vshuff(y2.b)                      //[0,5]
           s01_s00 = vcombine(s10, s10)                //[P,5]filter offset * xoffset and bias
           s11 = s10                                   //[P,5]filter offset * xoffset and bias
           col_count = add(col_count, #-4)             //[P,5]
        } {
           C0 = #0                                     //[P, 5]clear suma accumulators
           C1 = #0                                     //[P, 5]
           C2 = #0                                     //[P, 5]
           C3 = #0                                     //[P, 5]
        }
/* --------------------------------------------------------------------------- */
         .balign 32
.L_depth:
.L_vloop:
        {  s00.uw += vrmpy(z3210.ub, w_210.ub)         //[0,6]filter even output
           z54__.w = vasl(z7654.w, s16)                //[0,6]
           x0.tmp = vmem(ptr_x1+#0)                    //[1,0]
           y0.b = vshuff(x0.b)                         //[1,0]
        } {
           s10.uw += vrmpy(z7654.ub, w_210.ub)         //[0,7]z5432
           z__76.uw = vlsr(z7654.uw, s16)              //[0,7]
           x1.tmp = vmem(ptr_x1+#1)                    //[1,1]
           y1.b = vshuff(x1.b)                         //[1,1]
        } {
           ww210 = w_210                               //[0,8]
           z5432.h = vshuffo(z54__.h, z3210.h)         //[0,8]
           x2.tmp = vmem(ptr_x1+#2)                    //[1,2]
           y2.b = vshuff(x2.b)                         //[1,2]
        } {
           C0.uw += vrmpy(z3210.ub, filt_offset.ub)    //
           C1.uw += vrmpy(z7654.ub, filt_offset.ub)    //
           z9876.h = vshuffe(zba98.h, z__76.h)         //[0,9]
           z3210.b = vshuff(y0.b)                      //[1,3]x3210
        } {
           s01.uw += vrmpy(z5432.ub, ww210.ub)         //[0,10]filter even output
           u_210.tmp = vmem(ptr_w++#1)                 //[1,4]
           w_210.w = vasl(u_210.w, padding)            //[1,4]
           z7654.b = vshuff(y1.b)                      //[1,4]x7654
        } {
           s11.uw += vrmpy(z9876.ub, ww210.ub)         //[0,11]filter even output
           zba98.b = vshuff(y2.b)                      //[1,5]
           ptr_x1 =add(ptr_x1, next_in_width_depth)    //[1,3]move to next pt in same depth position
        } {
           C2.uw += vrmpy(z5432.ub, filt_offset.ub)    //
           C3.uw += vrmpy(z9876.ub, filt_offset.ub)    //
        }:endloop0
/* --------------------------------------------------------------------------- */
        {  s00.uw += vrmpy(z3210.ub, w_210.ub)         //[1,6]filter even output
           z54__.w = vasl(z7654.w, s16)                //[1,6]
        } {
           s10.uw += vrmpy(z7654.ub, w_210.ub)         //[1,7]z5432
           z__76.uw = vlsr(z7654.uw, s16)              //[1,7]
           z5432.h = vshuffo(z54__.h, z3210.h)         //[1,7]
        } {
           C0.uw += vrmpy(z3210.ub, filt_offset.ub)    //
           C1.uw += vrmpy(z7654.ub, filt_offset.ub)    //
           z9876.h = vshuffe(zba98.h, z__76.h)         //[1,8]
        } {
           s00.w = vsub(s00.w, C0.w)                   //[E,9]
           s10.w = vsub(s10.w, C1.w)                   //[E,9]
           s01.uw += vrmpy(z5432.ub, w_210.ub)         //[1,9]filter even output
        } {
           mine.w = vmin(mine.w, s00.w)                //[E,8]
           maxe.w = vmax(maxe.w, s00.w)                //[E,8]
           s11.uw += vrmpy(z9876.ub, w_210.ub)         //[1,10]filter even output
        } {
           s00_s.w = vasl(s00.w, zshift)               //[E,8]
           C2.uw += vrmpy(z5432.ub, filt_offset.ub)    //
           C3.uw += vrmpy(z9876.ub, filt_offset.ub)    //
           maxe.w = vmax(maxe.w, s10.w)                //[E,9]
        } {
           s10_s.w = vasl(s10.w, zshift)               //[W,9]
           s01.w = vsub(s01.w, C2.w)                   //[E,10]subtract suma
           s11.w = vsub(s11.w, C3.w)                   //[E,10]subtract suma
           mine.w = vmin(mine.w, s10.w)                //[E,10]
        } {
           d0.w = vmpye(s00_s.w, vrecip.uh)            //[E,11]
           maxe.w = vmax(maxe.w, s01.w)                //[E,10]
           mine.w = vmin(mine.w, s01.w)                //[E,11]
        } {
           d0.w += vmpyo(s00_s.w, vrecip.h):SSR        //[E,12]
           s01_s.w = vasl(s01.w, zshift)               //[E,11]
        } {
           s11_s.w = vasl(s11.w, zshift)               //[E,12]
           x0.tmp = vmem(ptr_x0+#0)                    //[0,0]
           y0.b = vshuff(x0.b)                         //[0,0]
        } {
           d1.w = vmpye(s01_s.w, vrecip.uh)            //[E,13]
           mine.w = vmin(mine.w, s11.w)                //[E,13]
           maxe.w = vmax(maxe.w, s11.w)                //[E,12]
           ptr_x1 = ptr_x0                             //[P,0]
        } {
           d1.w += vmpyo(s01_s.w, vrecip.h):SSR        //[E,14]
           x1.tmp = vmem(ptr_x1+#1)                    //[0,1]
           y1.b = vshuff(x1.b)                         //[0,1]
           ptr_x0 = add(ptr_x0, next_in_width_32)      //[P,1]
        } {
           d2.w = vmpye(s10_s.w, vrecip.uh)            //[E,15]
           x2.tmp = vmem(ptr_x1+#2)                    //[0,2]
           y2.b = vshuff(x2.b)                         //[0,2]
        } {
           d2.w += vmpyo(s10_s.w, vrecip.h):SSR        //[E,16]
           d1d0.h = vpacke(d1.w, d0.w)                 //[E,16]
        } {
           d3.w = vmpye(s11_s.w, vrecip.uh)            //[E,17]
           z3210.b = vshuff(y0.b)                      //[0,3]x3210
           ptr_x1 =add(ptr_x1, next_in_width_depth)    //[0,3]move to next pt in same depth position
        } {
           d3.w += vmpyo(s11_s.w, vrecip.h):SSR        //[E,18]
           s01 = vmem(filt_sum++#1)                    //[P,3]
           loop0(.L_vloop, filt_height)                //[P,3]can have a filter of Nx3 stride = 1
        } {
           zba98.b = vshuff(y2.b)                      //[0,5]
           s10 = s01                                   //[P,5]filter offset * xoffset and bias
           s11 = s01                                   //[P,5]filter offset * xoffset and bias
           C0 = #0                                     //[P,5] init suma accumulator
        } {
           d3d2.h = vpacke(d3.w, d2.w)                 //[E,20]
           C1 = #0                                     //[P, 6] init suma accumulator
           C2 = #0                                     //[P, 6] init suma accumulator
           C3 = #0                                     //[P, 6] init suma accumulator
        } {
           u_210.tmp = vmem(ptr_w++#1)                 //[0,4]
           w_210.w = vasl(u_210.w, padding)            //[0,4]
           z7654.b = vshuff(y1.b)                      //[0,4]x7654
           s00 = s01                                   //[P,4]filter offset * xoffset and bias
        } {
           d3_d0.ub = vpack(d3d2.h,d1d0.h):sat         //[E,22]
           vmem(ptr_z1+#0) = d3_d0.new                 //[E,22]
           ptr_z1 = ptr_z0                             //[P,2]
           ptr_z0 = add(ptr_z0, next_out_width_32)     //[P,2]
        }:endloop1                                     //end depth
/* --------------------------------------------------------------------------- */
        {  ptr_x0=sub(ptr_x0,in_wide_deep_high_256)    //next inputs
           ptr_z0=sub(ptr_z0,out_wide_deep_128)        //next output
           p0 = cmp.eq(col_count, #0)                  //
           if(!p0.new) jump:t .L_width                 //
        }
/* --------------------------------------------------------------------------- */
        {  out_height = add(out_height, #-1)           //
           col_count = out_width                       //
           p0 = cmp.eq(out_height, #1)                 //
           if(!p0.new) jump:t .L_height                //
        }
/* --------------------------------------------------------------------------- */
        {  loop0(.L_peak, #4)                          //
           c4 = #-4                                    //
        } {
           maxo_maxe = vdeal(maxe, maxe, c4)           //
           ptr_max = memw(sp+#25<<2)                   //
        }
.L_peak:
        {  maxe.w = vmax(maxe.w, maxo.w)               //
           mino_mine = vdeal(mine, mine, c4)           //
        } {
           mine.w = vmin(mine.w, mino.w)               //
        } {
           maxo_maxe = vdeal(maxe, maxe, c4)           //
        }:endloop0 
        {
           maxe.w = vmax(maxe.w, maxo.w)               //
           vmem(ptr_max+#0) = maxe.new                 //
           mino_mine = vdeal(mine, mine, c4)           //
        } {
           mine.w = vmin(mine.w, mino.w)               //
           vmem(ptr_max+#1) = mine.new                 //
        }
/* --------------------------------------------------------------------------- */
        {  r17:16 = memd(sp+#0)                        //restore stack
           r19:18 = memd(sp+#8)                        //Q
        } {
           r21:20 = memd(sp+#16)                       //Q
           r23:22 = memd(sp+#24)                       //Q
        } {
           r25:24 = memd(sp+#32)                       //Q
           r27:26 = memd(sp+#40)                       //Q
        } {
           dealloc_return                              //Q
        }
.L_end:
      .size dwconv2dbbb_unsigned_s2_v60_asm, .L_end-dwconv2dbbb_unsigned_s2_v60_asm
#else
/*=============================================================================*/
        .text
        .file "dwconv2dbbb_unsigned_s2_d32_v60_h.S"
        .global dwconv3x3bbb_unsigned_s2_v60_asm
        .balign 32
        .type  dwconv3x3bbb_unsigned_s2_v60_asm, @function
dwconv3x3bbb_unsigned_s2_v60_asm:
/* =========================================================================== */
//h stride assumed 1 vstride 1 or 2 filt width assumed 3 - hstride 2 requires new function
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define filt_sumi              r2     //gemsumb
#define ptr_zi                 r3     //results
#define next_in_row            r4     //width*depth*(stride_horz==1)
#define next_in_width_32       r5     //width*32*(stride_horz==1)
#define in_depth               r8     //1 total in depth 
#define out_width              r9     //2is amount of work to be done
#define next_out_row           r10    //3next output line amount in bytes
#define out_height             r11    //4 number of vertical lines to perform
#define recip_level            r12    //5 255 / (MAX - MIN) - used to scale to bytes
#define zshift                 r13    //6 spare input
#define ptr_max                r14    //7 maximum and minum buffer
#define stride_vert            r15    //8 vertical stride is an option to save ops
#define filt_offset            r16    //9
#define padding                r17    //10 padding=0 patter PPPx, else xPPP
#define filt_off210_           r16    //
#define filt_off_210           r17    //
#define out_width_in_depth     r9:8   //
#define out_height_next_out_row r11:10//
#define zshift_recip_level     r13:12 //    
#define stride_vert_ptr_max    r15:14 //
#define padding_filt_offset    r17:16 //
//-----------------------------------------------------------------
#define s8                     r7     //const = 8
#define nrot                   r6     //rotate amount
#define ptr_x                  r6     //
#define ptr_x0                 r18    //
#define ptr_x1                 r19    //
#define ptr_x2                 r20    //
#define ptr_z                  r21    //
#define ptr_w                  r22    //
#define filt_sum               r23    //
#define filt_sum_ptr_w         r23:22 //
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
//-----------------------------------------------------------------
#define vrecip                 v0     //
#define woffset                v1     //
#define s00                    v2     //
#define s01                    v3     //
#define s01s00                 v3:2   //
#define s10                    v4     //
#define s11                    v5     //
#define s11s10                 v5:4   //
#define sa0                    v6     //
#define sa1                    v7     //
#define sa1sa0                 v7:6   //
#define sa2                    v8     //
#define sa3                    v9     //
#define sa3sa2                 v9:8   //
#define l0w_210                v10    //
#define l1w_210                v11    //
#define l2w_210                v12    //
#define l0z3210                v13    //
#define l1z3210                v14    //
#define l2z3210                v15    //
#define l0z7654                v16    //
#define l1z7654                v16    //
#define l2z7654                v16    //
#define l0zba98                v17    //
#define l1zba98                v17    //
#define l2zba98                v17    //
#define l0z3232                v18    //
#define l1z3232                v18    //
#define l2z3232                v18    //
#define l0z5432                v18    //
#define l1z5432                v18    //
#define l2z5432                v18    //
#define l0z7676                v19    //
#define l1z7676                v19    //
#define l2z7676                v19    //
#define l0z9876                v19    //
#define l1z9876                v19    //
#define l2z9876                v19    //
#define d0                     v20    //
#define d1                     v21    //
#define d1_d0                  v21:20 //
#define d2                     v22    //
#define d3                     v23    //
#define d3_d2                  v23:22 //
#define d1d0                   v24    //
#define d3d2                   v25    //
#define d3210                  v24    //
#define maxe                   v26    //
#define mine                   v27    //
#define y00                    v16    //
#define y01                    v17    //
#define y10                    v16    //
#define y11                    v17    //
#define y20                    v16    //
#define y21                    v17    //
#define x00                    v30    //
#define x01                    v30    //
#define x10                    v30    //
#define x11                    v30    //
#define x20                    v30    //
#define x21                    v30    //
#define w0                     v30    //
#define w1                     v30    //
#define w2                     v30    //
#define maxt                   v2     //
#define mint                   v3     //
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/*=============================================================================*/
    {  out_width_in_depth      = memd(sp+#0)        //
       out_height_next_out_row = memd(sp+#8)        //
       sp = add(sp,#-4*8)                           //
    }{
       zshift_recip_level  = memd(sp+#(16+4*8))     //
       stride_vert_ptr_max = memd(sp+#(24+4*8))     //
       nop; nop                                     //
    }{
       memd(sp+#0)  = r17:16                        //
       padding_filt_offset = memd(sp+#(32+4*8))     //
       vrecip = vsplat(recip_level)                 //
    }{
       memd(sp+#8)  = r19:18                        //
       in_depth = lsr(in_depth, #5)                 //1/32
       p0 = cmp.eq(padding,#0)                      //
    }{
       memd(sp+#16) = r21:20                        //
       memd(sp+#24) = r23:22                        //
       filt_offset = vsplatb(filt_offset)           //
       out_width = lsr(out_width,#2)                //
    }{
       filt_off210_ = asl(filt_offset, #8)          //
       filt_off_210 = lsr(filt_offset, #8)          //
       filt_sum_ptr_w = combine(filt_sumi,ptr_wi)   //
       maxe = vmem(ptr_max+#0)                      //
    }
    {  s8 = mux(p0,#0,#8)                           //if padding = 1 then shift by 8
       filt_offset = mux(p0,filt_off_210,filt_off210_)
       mine = vmem(ptr_max+#1)                      //
    }
/* --------------------------------------------------------------------------- */
.L_height:
    {  ptr_x = ptr_xi                               //
       ptr_z = ptr_zi                               //
       p3 = xor(p3,p3)                              //    
       loop1(.L_depth,in_depth)                     //
    }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_depth:
    {  woffset = vmem(filt_sum++#1)                 //read in sum of taps
       ptr_x0 = ptr_x                               //
       ptr_x1 = add(ptr_x,next_in_row)              //
       nop                                          //
    }{ 
       x00.tmp = vmem(ptr_x0++#1):nt                //
       l0z3210.b = vshuff(x00.b)                    //
       ptr_x2 = add(ptr_x1,next_in_row)             //
       nop                                          //
    }{ 
       x10.tmp = vmem(ptr_x1++#1)                   //
       l1z3210.b = vshuff(x10.b)                    //
       loop0(.L_width,out_width)                    //
       nop                                          //
    }{ 
       x20.tmp = vmem(ptr_x2++#1)                   //
       l2z3210.b = vshuff(x20.b)                    //
       ptr_x = add(ptr_x,next_in_width_32)          //
    }{ 
       l0z3210.b = vshuff(l0z3210.b)                //
       w0.tmp = vmem(ptr_w++#1)                     //
       l0w_210.w = vasl(w0.w, s8)                   //
    }{
       l1z3210.b = vshuff(l1z3210.b)                //
       w1.tmp = vmem(ptr_w++#1)                     //
       l1w_210.w = vasl(w1.w, s8)                   //
    }{
       l2z3210.b = vshuff(l2z3210.b)                //
       w2.tmp = vmem(ptr_w++#1)                     //
       l2w_210.w = vasl(w2.w, s8)                   //
    }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
    {  x00.tmp = vmem(ptr_x0++#1):nt                //[0, 0]
       y00.b = vshuff(x00.b)                        //[0, 0]
       s00 = woffset                                //[0, 0]
       d3.w = vmpye(s11.w, vrecip.uh)               //[0,28]multiply by 1/max 
    }{
       x01.tmp = vmem(ptr_x0++#1):nt                //[0, 1]
       y01.b = vshuff(x01.b)                        //[0, 1]
       s10 = woffset                                //[0, 1]
       d2.w += vmpyo(s10.w, vrecip.h):SSR           //[1,29]multiply by 1/max 
    }{
       l0z7654.b = vshuff(y00.b)                    //[0, 2]
       l0z3232.h = vshuffo(l0z3210.h,l0z3210.h)     //[0, 2]
       d3.w += vmpyo(s11.w, vrecip.h):SSR           //[1,30]multiply by 1/max 
    }{
       l0zba98.b = vshuff(y01.b)                    //[0, 3]
       s00.uw += vrmpy(l0z3210.ub, l0w_210.ub)      //[0, 3]
       l0z5432.h = vshuffe(l0z7654.h, l0z3232.h)    //[0, 3]
    }{
       s10.uw += vrmpy(l0z7654.ub, l0w_210.ub)      //[0, 4]
       l0z7676.h = vshuffo(l0z7654.h,l0z7654.h)     //[0, 4]
       s01 = woffset                                //[0, 4]
    }{
       sa0.uw  = vrmpy(l0z3210.ub, filt_offset.ub)  //[0, 5]z3210filter oddn output
       sa2.uw  = vrmpy(l0z7654.ub, filt_offset.ub)  //[0, 5]z3210filter oddn output
       s11 = woffset                                //[0, 5]
       l0z9876.h = vshuffe(l0zba98.h, l0z7676.h)    //[0, 5]
    }{
       s01.uw += vrmpy(l0z5432.ub, l0w_210.ub)      //[0, 6]
       l0z3210 = l0zba98                            //[0, 6]
       x10.tmp = vmem(ptr_x1++#1)                   //[0, 6]
       y10.b = vshuff(x10.b)                        //[0, 6]
    }{
       s11.uw += vrmpy(l0z9876.ub, l0w_210.ub)      //[0, 7]
       x11.tmp = vmem(ptr_x1++#1)                   //[0, 7]
       y11.b = vshuff(x11.b)                        //[0, 7]
    }{
       sa1.uw  = vrmpy(l0z5432.ub, filt_offset.ub)  //[0, 8]z3210filter oddn output
       sa3.uw  = vrmpy(l0z9876.ub, filt_offset.ub)  //[0, 8]z3210filter oddn output
       l1z7654.b = vshuff(y10.b)                    //[0, 8]
       l1z3232.h = vshuffo(l1z3210.h,l1z3210.h)     //[0, 8]
    }{
       l1zba98.b = vshuff(y11.b)                    //[0, 9]
       s00.uw += vrmpy(l1z3210.ub, l1w_210.ub)      //[0, 9]
       l1z5432.h = vshuffe(l1z7654.h, l1z3232.h)    //[0, 9]
    }{
       s10.uw += vrmpy(l1z7654.ub, l1w_210.ub)      //[0,10]
       l1z7676.h = vshuffo(l1z7654.h,l1z7654.h)     //[0,10]
    }{
       sa0.uw += vrmpy(l1z3210.ub, filt_offset.ub)  //[0,11]z3210filter oddn output
       sa2.uw += vrmpy(l1z7654.ub, filt_offset.ub)  //[0,11]z3210filter oddn output
       l1z9876.h = vshuffe(l1zba98.h, l1z7676.h)    //[0,11]
    }{
       s01.uw += vrmpy(l1z5432.ub, l1w_210.ub)      //[0,12]
       l1z3210 = l1zba98                            //[0,12]
       x20.tmp = vmem(ptr_x2++#1)                   //[0,12]
       y20.b = vshuff(x20.b)                        //[0,12]
    }{
       s11.uw += vrmpy(l1z9876.ub, l1w_210.ub)      //[0,13]
       x21.tmp = vmem(ptr_x2++#1)                   //[0,13]
       y21.b = vshuff(x21.b)                        //[0,13]
    }{
       sa1.uw += vrmpy(l1z5432.ub, filt_offset.ub)  //[0,14]
       sa3.uw += vrmpy(l1z9876.ub, filt_offset.ub)  //[0,14]
       l2z7654.b = vshuff(y20.b)                    //[0,14]
       l2z3232.h = vshuffo(l2z3210.h,l2z3210.h)     //[0,14]
    }{
       l2zba98.b = vshuff(y21.b)                    //[0,15]
       s00.uw += vrmpy(l2z3210.ub, l2w_210.ub)      //[0,15]
       l2z5432.h = vshuffe(l2z7654.h, l2z3232.h)    //[0,15]
    }{
       s10.uw += vrmpy(l2z7654.ub, l2w_210.ub)      //[0,16]
       l2z7676.h = vshuffo(l2z7654.h,l2z7654.h)     //[0,16]
    }{
       sa0.uw += vrmpy(l2z3210.ub, filt_offset.ub)  //[0,17]
       sa2.uw += vrmpy(l2z7654.ub, filt_offset.ub)  //[0,17]
       l2z9876.h = vshuffe(l2zba98.h, l2z7676.h)    //[0,17]
    }{
       s01.uw += vrmpy(l2z5432.ub, l2w_210.ub)      //[0,18]
       l2z3210 = l2zba98                            //[0,18]
       d1d0.h = vpack(d1.w, d0.w):sat               //[1,31]take lower 16bits of rnded acc
    }{
       sa1.uw += vrmpy(l2z5432.ub, filt_offset.ub)  //[0,19]
       sa3.uw += vrmpy(l2z9876.ub, filt_offset.ub)  //[0,19]
       d3d2.h = vpack(d3.w, d2.w):sat               //[1,32]take lower 16bits of rnded acc
    }{
       s11.uw += vrmpy(l2z9876.ub, l2w_210.ub)      //[0,20]
       s01s00.w = vsub(s01s00.w, sa1sa0.w)          //[0,20]
    }{
       maxe.w = vmax(maxe.w, s00.w)                 //[0,21]find max
       mine.w = vmin(mine.w, s00.w)                 //[0,21]find min
       s00.w = vasl(s00.w,zshift)                   //[0,21]
       s10.w = vsub(s10.w, sa2.w)                   //[0,21]
    }{
       maxe.w = vmax(maxe.w, s01.w)                 //[0,22]find max
       mine.w = vmin(mine.w, s01.w)                 //[0,22]find min
       s01.w = vasl(s01.w,zshift)                   //[0,22]
       s11.w = vsub(s11.w, sa3.w)                   //[0,22]
    }{
       d0.w = vmpye(s00.w, vrecip.uh)               //[0,23]multiply by 1/max 
       maxe.w = vmax(maxe.w, s10.w)                 //[0,23]find max
       mine.w = vmin(mine.w, s10.w)                 //[0,23]find min
    }{
       d1.w = vmpye(s01.w, vrecip.uh)               //[0,24]multiply by 1/max 
       maxe.w = vmax(maxe.w, s11.w)                 //[0,24]find max
       mine.w = vmin(mine.w, s11.w)                 //[0,24]find min of acc
    }{
       d0.w += vmpyo(s00.w, vrecip.h):SSR           //[0,25]multiply by 1/max 
       s10.w = vasl(s10.w,zshift)                   //[0,25]
    }{
       d1.w += vmpyo(s01.w, vrecip.h):SSR           //[0,26]multiply by 1/max 
       s11.w = vasl(s11.w,zshift)                   //[0,26]
       d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[1,33]deal into sequence
       if p3 vmem(ptr_z++#1):nt = d3210.new         //[1,33]store quantized bytes
    }{
       d2.w = vmpye(s10.w, vrecip.uh)               //[0,27]multiply by 1/max 
       p3 = or(p3,!p3)                              //    
    }:endloop0:endloop1
    
/* --------------------------------------------------------------------------- */
    {  d3.w = vmpye(s11.w, vrecip.uh)               //[0,28]multiply by 1/max 
    }{
       d2.w += vmpyo(s10.w, vrecip.h):SSR           //[1,29]multiply by 1/max 
       out_height = add(out_height, #-1)            //
    }{
       d3.w += vmpyo(s11.w, vrecip.h):SSR           //[1,30]multiply by 1/max 
       p0 = cmp.eq(out_height, #0)                  //
    }{
       d1d0.h = vpack(d1.w, d0.w):sat               //[1,31]take lower 16bits of rnded acc
       ptr_xi+=mpyi(stride_vert,next_in_row)        //incrmeent input ptr
       ptr_zi=add(ptr_zi,next_out_row)              //incrmeent output ptr
    }{
       d3d2.h = vpack(d3.w, d2.w):sat               //[1,32]take lower 16bits of rnded acc
       filt_sum_ptr_w = combine(filt_sumi,ptr_wi)   //
    }{
       d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[1,33]deal into sequence
       vmem(ptr_z+#0):nt = d3210.new                //[1,33]store quantized bytes
       if(!p0) jump .L_height                       //next line
    }
/* --------------------------------------------------------------------------- */
    {  vmem(ptr_max+#0) = maxe                      //store max
       r17:16 = memd(sp+#0)                         //restore stack
    }{
       vmem(ptr_max+#1) = mine                      //store min
       r19:18 = memd(sp+#8)                         //restore
    }{
/* --------------------------------------------------------------------------- */
       r21:20 = memd(sp+#16)                        //restore
       r23:22 = memd(sp+#24)                        //restore
       sp = add(sp,#4*8)                             // pop stack
       jumpr r31                                     //return
    }
.L_end:
      .size dwconv3x3bbb_unsigned_s2_v60_asm, .L_end-dwconv3x3bbb_unsigned_s2_v60_asm
#endif
