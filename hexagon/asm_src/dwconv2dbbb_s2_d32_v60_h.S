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
 *  FUNCTIONS      : gvconv2dbbb_v60_asm
 *
 *  DESCRIPTION
 *   Perform 2d convolution using elements along depth, do only simple
 *   convolution.
 *   Sums are scaled and saturated to 8bits. Max and Min accumulations are kept.
 *
 *  ARCHITECTURE   : QDSP6V60  + HVX
 *
 *  REVISION HISTORY:
 *  =================
 *
 *  Author              Date           Comments
 *  -------------------------------------------------------------
 *  DJH                 07/ 6/17       created
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
        .file "dwconv2dbbb_s2_d32_v60_h.S"
        .global dwconv2dbbb_s2_v60_asm
        .balign 32
        .type  dwconv2dbbb_s2_v60_asm, @function
dwconv2dbbb_s2_v60_asm:
/*=============================================================================*/
//stride assumed 1 filt width assumed 3 - stride 2 requires new function
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define ptr_zi                 r2     //results
#define next_in_width_depth    r3     //width*depth*(stride==1)
#define next_out_width_depth   r4     //next output line amount in bytes
#define next_in_width_32       r5     //width*32*(stride==1)
#define next_out_width_32      r16    //0next output line amount in bytes
#define in_depth               r17    //1 total in depth split into rows of depth 32
#define out_width              r18    //2is amount of work to be done
#define out_height             r19    //3 number of vertical lines to perform
#define filt_height            r20    //4 filt_height lines per filter
#define ptr_max                r21    //5 maximum and minum buffer
#define recip_level            r22    //6 255 / (MAX - MIN) - used to scale to bytes
#define filt_sumi              r23    //7 gemsumb
#define stride_height          r24    //8 vertical strideing any number
#define zshift                 r26    //9 shift correction for small accs
#define padding                r27    //10 padding = 1 then shift 8 pad else shift 0
//-----------------------------------------------------------------
#define c4                     r6     //
#define s16                    r9     //const = 16
#define in_wide_deep_high_256  r8     //width*depth*filt_height - 256
#define depth                  r10    //current depth used 
#define ptr_x0                 r11    //
#define ptr_x1                 r12    //
#define ptr_z0                 r13    //
#define ptr_z1                 r14    //
#define ptr_w                  r15    //
#define filt_sum               r22    //
#define col_count              r25    //
#define out_wide_deep_128      r7     //advance ptr 128 along and pack to current line start
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
//-----------------------------------------------------------------
#define vrecip                 v0
#define woffset                v1
#define s00                    v2
#define s01                    v3
#define s10                    v4
#define s11                    v5
#define s00_s                  v7
#define s01_s                  v8
#define s10_s                  v9
#define s11_s                  v17
#define d1_d0                  v11:10
#define d3_d2                  v13:12
#define d0                     v10
#define d1                     v11
#define d2                     v12
#define d3                     v13
#define d1d0                   v10 
#define d3d2                   v12 
#define d3_d0                  v12
#define y0                     v21  //
#define y1                     v24  //
#define y2                     v16  //
#define x0                     v10 //
#define x1                     v10 //
#define x2                     v10 //
#define z3210                  v26  //
#define z5432                  v28  //
#define z7654                  v29 //
#define zba98                  v9  //
#define z9876                  v6  //
#define z54__                  v19  //
#define z__76                  v20  //
#define z5476                  v27  //
#define w_210                  v22  //
#define ww210                  v18  //
#define u_210                  v23  //
#define maxo_maxe              v31:30
#define mino_mine              v15:14
#define maxe                   v30
#define mine                   v14
#define maxo                   v31
#define mino                   v15
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
           c4 = #-4                                    //
        } {
           memd(sp+#32) = r25:24                       //
           memd(sp+#40) = r27:26                       //  
           r23 = ##0x7fffffff                          //max pos
        } {
           mine = vsplat(r23)                          //
           zshift = memw(sp+#29<<2)                    //1 - #8 2 - 0
        } {
           maxe.w = vsub(maxe.w, mine.w)               //
           next_out_width_32 = memw(sp+#20<<2)         //
           in_depth = memw(sp+#21<<2)                  //
        } {
           out_width = memw(sp+#22<<2)                 //
           out_height = memw(sp+#23<<2)                //
        } {
           recip_level = memw(sp+#26<<2)               //
           padding = memw(sp+#30<<2)                   //1 - #8 2 - 0
        } {
           vrecip = vsplat(recip_level)                //
           ptr_max = memw(sp+#25<<2)                   //
           filt_sumi = memw(sp+#27<<2)                 //
        } {
           out_wide_deep_128=add(next_out_width_depth,#-128) //
           filt_height = memw(sp+#24<<2)               //
           stride_height = memw(sp+#28<<2)             //skip n vert lines
        } {
           in_wide_deep_high_256=add(next_in_width_depth, #-256) //
           in_depth = lsr(in_depth, #5)                // 1/32
           filt_height = add(filt_height, #-1)         //
        }
         .balign 32
.L_height:
        {
           col_count = out_width                       //
           ptr_z0 = ptr_zi                             //
           ptr_zi=add(ptr_zi,next_out_width_depth)     //
        } {
           ptr_x0 = ptr_xi                             //
           ptr_xi+=mpyi(next_in_width_depth,stride_height) //
        }
         .balign 32
.L_width:
        {
           loop1(.L_depth, in_depth)                   //
           filt_sum = filt_sumi                        //
           ptr_w = ptr_wi                              //restart filter stream
           col_count = add(col_count, #-4)             //
        } {
           x0.tmp = vmem(ptr_x0+#0)                    //[0,0]
           y0.b = vshuff(x0.b)                         //[0,0]
           ptr_x1 = ptr_x0                             //[P,0]
        }
/* --------------------------------------------------------------------------- */
         .balign 32
.L_depth:
        {
           x1.tmp = vmem(ptr_x1+#1)                    //[0,1]
           y1.b = vshuff(x1.b)                         //[0,1]
           ptr_x0 = add(ptr_x0, next_in_width_32)      //[P,1]
        } {
           x2.tmp = vmem(ptr_x1+#2)                    //[0,2]
           y2.b = vshuff(x2.b)                         //[0,2]
           ptr_z1 = ptr_z0                             //[P,2]
           ptr_z0 = add(ptr_z0, next_out_width_32)     //[P,2]
        } {
           z3210.b = vshuff(y0.b)                      //[0,3]x3210
           ptr_x1 =add(ptr_x1, next_in_width_depth)    //[0,3]move to next pt in same depth position
           woffset = vmem(filt_sum++#1)                //[P,3]
           loop0(.L_vloop, filt_height)                //[P,3]can have a filter of Nx3 stride = 1
        } {
           u_210.tmp = vmem(ptr_w++#1)                 //[0,4]
           w_210.w = vasl(u_210.w, padding)            //[0,4]
           z7654.b = vshuff(y1.b)                      //[0,4]x7654
           s00 = woffset                               //[P,4]filter offset * xoffset and bias
        } {
           zba98.b = vshuff(y2.b)                      //[0,5]
           s01 = woffset                               //[P,5]filter offset * xoffset and bias
           s10 = woffset                               //[P,5]filter offset * xoffset and bias
           s11 = woffset                               //[P,5]filter offset * xoffset and bias
        }
/* --------------------------------------------------------------------------- */
         .balign 32
.L_vloop:
        {  s00.w += vrmpy(z3210.ub, w_210.b)           //[0,6]filter even output
           z54__.w = vasl(z7654.w, s16)                //[0,6]
           x0.tmp = vmem(ptr_x1+#0)                    //[1,0]
           y0.b = vshuff(x0.b)                         //[1,0]
        } {
           s10.w += vrmpy(z7654.ub, w_210.b)           //[0,7]z5432
           z__76.uw = vlsr(z7654.uw, s16)              //[0,7]
           x1.tmp = vmem(ptr_x1+#1)                    //[1,1]
           y1.b = vshuff(x1.b)                         //[1,1]
        } {
           ww210 = w_210                               //[0,8]
           z5432.h = vshuffo(z54__.h, z3210.h)         //[0,8]
           x2.tmp = vmem(ptr_x1+#2)                    //[1,2]
           y2.b = vshuff(x2.b)                         //[1,2]
        } {
           z9876.h = vshuffe(zba98.h, z__76.h)         //[0,9]
           z3210.b = vshuff(y0.b)                      //[1,3]x3210
           ptr_x1 =add(ptr_x1, next_in_width_depth)    //[1,3]move to next pt in same depth position
        } {
           s01.w += vrmpy(z5432.ub, ww210.b)           //[0,10]filter even output
           u_210.tmp = vmem(ptr_w++#1)                 //[1,4]
           w_210.w = vasl(u_210.w, padding)            //[1,4]
           z7654.b = vshuff(y1.b)                      //[1,4]x7654
        } {
           s11.w += vrmpy(z9876.ub, ww210.b)           //[0,11]filter even output
           zba98.b = vshuff(y2.b)                      //[1,5]
        }:endloop0
/* --------------------------------------------------------------------------- */
        {  s00.w += vrmpy(z3210.ub, w_210.b)           //[1,6]filter even output
           z54__.w = vasl(z7654.w, s16)                //[1,6]
        } {
           s10.w += vrmpy(z7654.ub, w_210.b)           //[1,7]z5432
           z__76.uw = vlsr(z7654.uw, s16)              //[1,7]
           z5432.h = vshuffo(z54__.h, z3210.h)         //[1,7]
        } {
           PV(2)
        } {
           PV(4)
        } {
           z9876.h = vshuffe(zba98.h, z__76.h)         //[1,8]
           s00_s.w = vasl(s00.w, zshift)               //[E,8]
           mine.w = vmin(mine.w, s00.w)                //[E,8]
           maxe.w = vmax(maxe.w, s00.w)                //[E,8]
        } {
           s01.w += vrmpy(z5432.ub, w_210.b)           //[1,9]filter even output
           s10_s.w = vasl(s10.w, zshift)               //[W,9]
           maxe.w = vmax(maxe.w, s10.w)                //[E,9]
        } {
           s11.w += vrmpy(z9876.ub, w_210.b)           //[1,10]filter even output
           mine.w = vmin(mine.w, s10.w)                //[E,10]
           maxe.w = vmax(maxe.w, s01.w)                //[E,10]
        } {
           PV(3)
        } {
           PV(5)
        } {
           s01_s.w = vasl(s01.w, zshift)               //[E,11]
           d0.w = vmpye(s00_s.w, vrecip.uh)            //[E,11]
           mine.w = vmin(mine.w, s01.w)                //[E,11]
        } 
/* --------------------------------------------------------------------------- */
        {  s11_s.w = vasl(s11.w, zshift)               //[E,12]
           d0.w += vmpyo(s00_s.w, vrecip.h):SSR        //[E,12]
           maxe.w = vmax(maxe.w, s11.w)                //[E,12]
        } {
           d1.w = vmpye(s01_s.w, vrecip.uh)            //[E,13]
           mine.w = vmin(mine.w, s11.w)                //[E,13]
        } {
           d1.w += vmpyo(s01_s.w, vrecip.h):SSR        //[E,14]
        } {
           d2.w = vmpye(s10_s.w, vrecip.uh)            //[E,15]
        } {
           d2.w += vmpyo(s10_s.w, vrecip.h):SSR        //[E,16]
           d1d0.h = vpacke(d1.w, d0.w)                 //[E,16]
        } {
           d3.w = vmpye(s11_s.w, vrecip.uh)            //[E,17]
        } 
        {
           d3.w += vmpyo(s11_s.w, vrecip.h):SSR        //[E,18]
        } {
           x0.tmp = vmem(ptr_x0+#0)                    //[0,0]
           y0.b = vshuff(x0.b)                         //[0,0]
           ptr_x1 = ptr_x0                             //[P,0]
        } {
           d3d2.h = vpacke(d3.w, d2.w)                 //[E,20]
        } {
        } {
           d3_d0.ub = vpack(d3d2.h,d1d0.h):sat         //[E,22]
           vmem(ptr_z1+#0) = d3_d0.new                 //[E,22]
        }:endloop1                                     //end depth
/* --------------------------------------------------------------------------- */
       {  ptr_x0=sub(ptr_x0,in_wide_deep_high_256)     //next inputs
          ptr_z0=sub(ptr_z0,out_wide_deep_128)         //next output
          p0 = cmp.eq(col_count, #0)                   //
          if(!p0.new) jump:t .L_width                  //
       }
/* --------------------------------------------------------------------------- */
       {  out_height = add(out_height, #-1)            //
          p0 = cmp.eq(out_height, #1)                  //
          if(!p0.new) jump:t .L_height                 //
       }
/* --------------------------------------------------------------------------- */
       {
          loop0(.L_peak, #4)                           //
          maxo_maxe = vdeal(maxe, maxe, c4)            //
       }
.L_peak:
       {
          maxe.w = vmax(maxe.w, maxo.w)                //
          mino_mine = vdeal(mine, mine, c4)            //
       } {
          mine.w = vmin(mine.w, mino.w)                //
       } {
          maxo_maxe = vdeal(maxe, maxe, c4)            //
       }:endloop0 
       {
          maxe.w = vmax(maxe.w, maxo.w)                //
          vmem(ptr_max+#0) = maxe.new                  //
          mino_mine = vdeal(mine, mine, c4)            //
       } {
          mine.w = vmin(mine.w, mino.w)                //
          vmem(ptr_max+#1) = mine.new                  //
       }
/* --------------------------------------------------------------------------- */
       {  r17:16 = memd(sp+#0)                         //restore stack
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
.L_end:
      .size dwconv2dbbb_s2_v60_asm, .L_end-dwconv2dbbb_s2_v60_asm
/*=============================================================================*/

