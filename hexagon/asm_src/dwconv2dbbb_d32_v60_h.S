/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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
 *  MEMORY
 *     CODESIZE = 640 bytes + 512 bytes of control tables
 *     STACK    = 80 bytes
 *     ASSUMPTIONS
 *      width multiple of 4 depth multiple of 32 aligned to 128bytes
 *   MODEL
#if 0
 * void dwconv2d_cn(
 *   uint8_t * in_buf, int in_width, int in_height, int depth, 
 *   int stride_width, int stride_height,
 *   int in_offset, int8_t  * filt,   int filt_width, int filt_height,
 *   int * out_buf, int out_width, int out_height, int adj_x, int adj_y)
 * {
 *   int out_y, in_y_base, out_x, in_x_base;
 *   int out_z, filt_z, filt_y, filt_x, in_element, filt_element, sum;
 *   int * outstripe;
 *   uint8_t * instripe;
 *   int8_t * filtstripe;
 *
 *   for (out_y = 0; out_y < out_height; out_y++) {
 *     in_y_base = out_y * stride_height - adj_y;
 *     for (out_x = 0; out_x < out_width; out_x++) {
 *       in_x_base = out_x * stride_width - adj_x;
 *       outstripe = out_buf+(depth*(out_x+ out_width*out_y));
 *       for (out_z = 0; out_z < depth; out_z++) {
 *         sum = 0;
 *         for (filt_y = 0; filt_y < filt_height; filt_y++) {
 *           if ((in_y_base + filt_y) >= in_height) continue;
 *           if ((in_y_base + filt_y) < 0) continue;
 *           for (filt_x = 0; filt_x < filt_width; filt_x++) {
 *             if ((in_x_base + filt_x) >= in_width) continue;
 *             if ((in_x_base + filt_x) < 0) continue;
 *
 *             filtstripe = filt+(depth*(filt_x+ filt_width*filt_y));
 *             filt_element = filtstripe[out_z];
 *
 *             instripe = in_buf+depth*(in_x_base+filt_x+in_width*(in_y_base+filt_y));
 *             in_element = instripe[out_z] - in_offset;
 *
 *             sum += in_element*filt_element;
 *           }
 *         }
 *         outstripe[out_z] = sum;
 *       }
 *     }
 *   }
 *   return;
 * }
#endif
 */
/* =========================================================================== */
        .text
        .file "dwconv2dbbb_d32_v60_h.S"
        .global dwconv2dbbb_v60_asm
        .balign 32
        .type  dwconv2dbbb_v60_asm, @function
dwconv2dbbb_v60_asm:
/* =========================================================================== */
//h stride assumed 1 vstride 1 or 2 filt width assumed 3 - hstride 2 requires new function
#define ptr_xi                 r0     //data
#define ptr_wi                 r1     //weights
#define ptr_zi                 r2     //results
#define next_in_width_depth    r3     //width*depth*(stride_horz==1)
#define next_out_width_depth   r4     //next output line amount in bytes
#define next_in_width_32       r5     //width*32*(stride_horz==1)
#define next_out_width_32      r16    //0next output line amount in bytes
#define in_depth               r17    //1 total in depth split into rows of depth 32
#define out_width              r18    //2is amount of work to be done
#define out_height             r19    //3 number of vertical lines to perform
#define filt_height            r20    //4 filt_height lines per filter
#define ptr_max                r21    //5 maximum and minum buffer
#define recip_level            r22    //6 255 / (MAX - MIN) - used to scale to bytes
#define filt_sumi              r23    //7 gemsumb
#define stride_vert            r24    //8 vertical stride is an option to save ops
#define zshift                 r6     //9 spare input
#define perm_ctrl              r25    //10 ptr to the fancy data shuffling controls
//-----------------------------------------------------------------
#define s8                     r7     //const = 8
#define c4                     r6     //deal words
#define out_in_wide_deep_128   r9:8   //
#define out_wide_deep_128      r9     //advance ptr 128 along and pack to current line start
#define in_wide_deep_high_128  r8     //width*depth*filt_height - 128
#define depth                  r10    //current depth used 
#define ptr_x0                 r12    //
#define ptr_z0                 r13    //
#define ptr_z0_ptr_x0          r13:12 //
#define ptr_x1                 r11    //
#define ptr_z1                 r14    //
#define ptr_w                  r15    //
#define filt_sum               r22    //
#define col_count              r26    //
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
#define d1_d0                  v11:10 //
#define d3_d2                  v13:12 //
#define d0                     v10    //
#define d1                     v11    //
#define d2                     v12    //
#define d3                     v13    //
#define d1d0                   v8     //
#define d3d2                   v16    //
#define d3_d0                  v17    //
#define perm2                  v6     //
#define perm3                  v7     //
#define y0                     v21    //
#define y1                     v24    //
#define x0                     v29    //
#define x1                     v9     //
#define z3210                  v26    //
#define z5432                  v28    //
#define z5476                  v27    //
#define w_210                  v22    //
#define u_210                  v2     //
#define w210_                  v23    //
#define maxo_maxe              v31:30 //
#define mino_mine              v15:14 //
#define maxe                   v30    //
#define mine                   v14    //
#define maxo                   v31    //
#define mino                   v15    //
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/*=============================================================================*/
   {  allocframe(#72)                             //0th entry on stack (72+8)/4=20 
      maxe = #0                                   //
      s8 = #8                                     //shift by 8
   } {
      memd(sp+#16) = r21:20                       //
      memd(sp+#24) = r23:22                       //
      r23 = ##0x7fffffff                          //max pos
   } {
      memd(sp+#0)  = r17:16                       //
      memd(sp+#8)  = r19:18                       //
      mine = vsplat(r23)                          //
   } {
      memd(sp+#32) = r25:24                       //
      memd(sp+#40) = r27:26                       //  
   } {
      perm_ctrl = memw(sp+#30<<2)                 //
      next_out_width_32 = memw(sp+#20<<2)         //
      maxe.w = vsub(maxe.w, mine.w)               //
   } {
      in_depth = memw(sp+#21<<2)                  //
      out_width = memw(sp+#22<<2)                 //
   } {
      out_height = memw(sp+#23<<2)                //
      recip_level = memw(sp+#26<<2)               //
   } {
      vrecip = vsplat(recip_level)                //
      perm2 = vmem(perm_ctrl+#0)                  //
      ptr_max = memw(sp+#25<<2)                   //
   } {
      filt_sumi = memw(sp+#27<<2)                 //
      perm3 = vmem(perm_ctrl+#1)                  //
      out_wide_deep_128=add(next_out_width_depth,#-128)//
      in_wide_deep_high_128=add(next_in_width_depth,#-128)//
   } {
      filt_height = memw(sp+#24<<2)               //
      in_depth = lsr(in_depth, #5)                //1/32
      col_count = out_width                       //
   } {
      filt_height = add(filt_height, #-1)         //correct for vertical loop
      ptr_z0 = ptr_zi                             //
      ptr_x0 = ptr_xi                             //
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_height:
.L_width:
   {  loop1(.L_depth, in_depth)                   //number of 32 depths
      woffset.cur = vmem(filt_sumi+#0)            //
      ptr_z1 = ptr_z0                             //
      s01s00 = vcombine(woffset,woffset)          //filter offset * xoffset and bias
   } {
      loop0(.L_vloop, filt_height)                //can have a filter of Nx3 stride=1
      ptr_x1 = ptr_x0                             //
      x0.tmp = vmem(ptr_x0+#0)                    //[0,0]
      y0.b = vshuff(x0.b)                         //[0,0]
   } {
      x1.tmp = vmem(ptr_x1+#1)                    //[0,1]
      y1 = vrdelta(x1, perm2)                     //[0,1]
      ptr_x1 =add(ptr_x1, next_in_width_depth)    //[0,1]move to next pt in same depth position
   } {
      z3210.b = vshuff(y0.b)                      //[0,2]x3210
      w_210 = vmem(ptr_wi+#0)                     //[0,2]
      s11s10 = vcombine(woffset,woffset)          //
      filt_sum = add(filt_sumi, #128)             //
   } {
      z5476 = vdelta(y1, perm3)                   //[0,3]x7654
      u_210.tmp = vmem(ptr_wi+#0)                 //[0,3]
      w210_.w = vasl(u_210.w, s8)                 //[0,3]
      ptr_w = add(ptr_wi, #128)                   //restart filter stream
   } 
/* --------------------------------------------------------------------------- */
     .balign 32
.L_depth:
.L_vloop:
   {  s00.w += vrmpy(z3210.ub, w_210.b)           //[0,4]filter even output
      z5432.h = vshuffo(z5476.h, z3210.h)         //[0,4]
      x0.tmp = vmem(ptr_x1+#0)                    //[1,0]
      y0.b = vshuff(x0.b)                         //[1,0]
   } {
      s01.w += vrmpy(z3210.ub, w210_.b)           //[0,5]z3210filter oddn output
      x1.tmp = vmem(ptr_x1+#1)                    //[1,1]
      y1 = vrdelta(x1, perm2)                     //[1,1]
      ptr_x1 =add(ptr_x1, next_in_width_depth)    //[1,1]move to next pt in same depth position
   } {
      s10.w += vrmpy(z5432.ub, w_210.b)           //[0,6]z5432
      z3210.b = vshuff(y0.b)                      //[1,2]x3210
      w_210 = vmem(ptr_w+#0)                      //[1,2]
   } {
      s11.w += vrmpy(z5432.ub, w210_.b)           //[0,7]z5432
      z5476 = vdelta(y1, perm3)                   //[1,3]x7654
      u_210.tmp = vmem(ptr_w++#1)                 //[1,3]
      w210_.w = vasl(u_210.w, s8)                 //[1,3]
   }:endloop0                                     //max accumulator=9*255=8f7=12bits-2^24 
/* --------------------------------------------------------------------------- */
   {  s00.w += vrmpy(z3210.ub, w_210.b)           //[1,4]z3210
      z5432.h = vshuffo(z5476.h, z3210.h)         //[1,4]
      ptr_x0 = add(ptr_x0, next_in_width_32)      //update input ptr to next depth position
      loop0(.L_vloop, filt_height)                //can have a filter of Nx3 stride = 1
   } {
      s01.w += vrmpy(z3210.ub, w210_.b)           //[1,5]z3210
      zshift = memw(r29+#29<<2)
      maxe.w = vmax(maxe.w, s00.w)                //find max
      mine.w = vmin(mine.w, s00.w)                //find min
   } {
      s10.w += vrmpy(z5432.ub, w_210.b)           //[1,6]z5432
      mine.w = vmin(mine.w, s01.w)                //find min
      s00.w = vasl(s00.w,zshift)
      ptr_z0 = add(ptr_z0, next_out_width_32)     //update output ptr to next depth
   } {
      maxe.w = vmax(maxe.w, s01.w)                //find max
   } {
      s11.w += vrmpy(z5432.ub, w210_.b)           //[1,7]z5432
      s01.w = vasl(s01.w,zshift)
      maxe.w = vmax(maxe.w, s10.w)                //find max
      ptr_x1 = ptr_x0                             //
   } {
      d0.w = vmpye(s00.w, vrecip.uh)              //multiply by 1/max 
      s10.w = vasl(s10.w,zshift)
      mine.w = vmin(mine.w, s10.w)                //find min
   } {
      s11.w = vasl(s11.w,zshift)
      d1.w = vmpye(s01.w, vrecip.uh)              //multiply by 1/max 
      x0.tmp = vmem(ptr_x1+#0)                    //[0,0]read first 4 depths
      y0.b = vshuff(x0.b)                         //[0,0]1st part of shuffle 4
   } {
      d0.w += vmpyo(s00.w, vrecip.h):SSR          //multiply by 1/max 
      maxe.w = vmax(maxe.w, s11.w)                //find max
      x1.tmp = vmem(ptr_x1+#1)                    //[0,1]load 2nd 4 depths
      y1 = vrdelta(x1, perm2)                     //[0,1]1st part of shuffle 4
   } {
      d1.w += vmpyo(s01.w, vrecip.h):SSR          //multiply by 1/max 
   } {
      d2.w = vmpye(s10.w, vrecip.uh)              //multiply by 1/max 
   } {
      woffset = vmem(filt_sum++#1)                //read in sum of taps
      ptr_x1 =add(ptr_x1, next_in_width_depth)    //[0,1]update ptr to next logical line
      z5476 = vdelta(y1, perm3)                   //[0,3]x7654
      d3.w = vmpye(s11.w, vrecip.uh)              //multiply by 1/max 
   } {
      d2.w += vmpyo(s10.w, vrecip.h):SSR          //multiply by 1/max 
      d1d0.h = vpacke(d1.w, d0.w)                 //take upp er 16bits of rnded acc
   } {
      d3.w += vmpyo(s11.w, vrecip.h):SSR          //multiply by 1/max 
   } {
      mine.w = vmin(mine.w, s11.w)                //find min of acc
      w_210 = vmem(ptr_w+#0)                      //[0,2]
      s00 = woffset                               //init accumulators
      s01 = woffset                               //init accumulators
   } {
      d3d2.h = vpacke(d3.w, d2.w)                 //take upp er 16bits of rnded acc
      s10 = woffset                               //init accumulators
      stride_vert = memw(sp+#28<<2)               //
      c4 = #-4                                    //deal words
   } {
      s11 = woffset                               //init accumulators
      u_210.tmp = vmem(ptr_w++#1)                 //[0,3]
      w210_.w = vasl(u_210.w, s8)                 //[0,3]adjust taps to odd locations
      z3210.b = vshuff(y0.b)                      //[0,2]x3210
   } {
      d3_d0.ub = vpack(d3d2.h, d1d0.h):sat                    //deal into sequence
      vmem(ptr_z1+#0) = d3_d0.new                 //store quantized bytes
      ptr_z1 = ptr_z0                             //next output depth 
   }:endloop1                                     //end depth
/* --------------------------------------------------------------------------- */
   {  ptr_z0_ptr_x0 = vsubw(ptr_z0_ptr_x0, out_in_wide_deep_128) //next inputs/outputs
      col_count = add(col_count, #-4)             //dec width count
      p0 = cmp.eq(col_count, #4)                  //next line
      if(!p0.new) jump:t .L_width                 //
   }
/* --------------------------------------------------------------------------- */
   {  ptr_zi=add(ptr_zi,next_out_width_depth)     //incrmeent output ptr
      ptr_xi+=mpyi(stride_vert,next_in_width_depth)//incrmeent input ptr
      col_count = out_width                       //reset row count
      out_height = add(out_height, #-1)           //
   } {
      ptr_z0 = ptr_zi                             //update to next output
      ptr_x0 = ptr_xi                             //update to next input
      p0 = cmp.eq(out_height, #0)                 //
      if(!p0.new) jump:t .L_height                //next line
   }
/* --------------------------------------------------------------------------- */
   {  loop0(.L_peak, #4)                          //
      maxo_maxe = vdeal(maxe, maxe, c4)           //deal out odd and even
      r17:16 = memd(sp+#0)                        //restore stack
      r19:18 = memd(sp+#8)                        //restore
   }
.L_peak:
   {  maxe.w = vmax(maxe.w, maxo.w)               //reduce
      mino_mine = vdeal(mine, mine, c4)           //
   } {
      mine.w = vmin(mine.w, mino.w)               //
   } {
      maxo_maxe = vdeal(maxe, maxe, c4)           //
   }:endloop0 
   {  maxe.w = vmax(maxe.w, maxo.w)               //
      vmem(ptr_max+#0) = maxe.new                 //store max
      mino_mine = vdeal(mine, mine, c4)           //
      r25:24 = memd(sp+#32)                       //restore
   } {
      mine.w = vmin(mine.w, mino.w)               //
      vmem(ptr_max+#1) = mine.new                 //store min
      r27:26 = memd(sp+#40)                       //restore
   }
/* --------------------------------------------------------------------------- */
   {  
      r21:20 = memd(sp+#16)                       //restore
      r23:22 = memd(sp+#24)                       //restore
   } {
      dealloc_return                              //return
   }
.L_end:
      .size dwconv2dbbb_v60_asm, .L_end-dwconv2dbbb_v60_asm
