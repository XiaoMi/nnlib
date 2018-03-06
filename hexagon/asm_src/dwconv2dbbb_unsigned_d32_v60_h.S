/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
 *   int in_offset, int8_t  * filt,   int filt_width, int filt_height, int filt_offset,
 *   int * out_buf, int out_width, int out_height, int adj_x, int adj_y)
 * {
 *   int out_y, in_y_base, out_x, in_x_base;
 *   int out_z, filt_z, filt_y, filt_x, in_element, filt_element, sum;
 *   int * outstripe;
 *   uint8_t * instripe;
 *   uint8_t * filtstripe;
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
 *             filt_element = filtstripe[out_z] - filt_offset;
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
#if 0
/* =========================================================================== */
        .text
        .file "dwconv2dbbb_unsigned_d32_v60_h.S"
        .global dwconv2dbbb_unsigned_v60_asm
        .balign 32
        .type  dwconv2dbbb_unsigned_v60_asm, @function
dwconv2dbbb_unsigned_v60_asm:
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
#define filt_offset            r25    //
#define filt_off210_           r25    //
#define filt_off_210           r21    //
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
#define s01_s                  v18
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
#define sa0                    v15    //
#define sa1                    v31    //
#define sa2                    v10    //
#define sa3                    v11    //
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
      sa0 = #0                                    //
   } {
      vrecip = vsplat(recip_level)                //
      perm2 = vmem(perm_ctrl+#0)                  //
      ptr_z0 = ptr_zi                             //
      sa1 = #0                                    //
   } {
      filt_sumi = memw(sp+#27<<2)                 //
      perm3 = vmem(perm_ctrl+#1)                  //
      out_wide_deep_128=add(next_out_width_depth,#-128)//
      in_wide_deep_high_128=add(next_in_width_depth,#-128)//
   } {
      zshift = memw(r29+#29<<2)                   //
      filt_offset = memw(sp+#31<<2)               //
      sa2 = #0                                    //
      sa3 = #0                                    //
   } {
      filt_height = memw(sp+#24<<2)               //
      in_depth = lsr(in_depth, #5)                //1/32
      col_count = out_width                       //
      filt_offset = vsplatb(filt_offset)          //
   } {
      filt_height = add(filt_height, #-1)         //correct for vertical loop
      ptr_x0 = ptr_xi                             //
      filt_off210_ = asl(filt_offset, #8)         //
      filt_off_210 = lsr(filt_offset, #8)         //
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
   {  s00.uw += vrmpy(z3210.ub, w_210.ub)         //[0,4]filter even output
      z5432.h = vshuffo(z5476.h, z3210.h)         //[0,4]
      x0.tmp = vmem(ptr_x1+#0)                    //[1,0]
      y0.b = vshuff(x0.b)                         //[1,0]
   } {
      s01.uw += vrmpy(z3210.ub, w210_.ub)         //[0,5]z3210filter oddn output
      x1.tmp = vmem(ptr_x1+#1)                    //[1,1]
      y1 = vrdelta(x1, perm2)                     //[1,1]
      ptr_x1 =add(ptr_x1, next_in_width_depth)    //[1,1]move to next pt in same depth position
   } {
      sa0.uw += vrmpy(z3210.ub, filt_off_210.ub)  //[1,2]z3210filter oddn output
      sa1.uw += vrmpy(z3210.ub, filt_off210_.ub)  //[1,2]z3210filter oddn output
   } {
      s10.uw += vrmpy(z5432.ub, w_210.ub)         //[0,7]z5432
      z3210.b = vshuff(y0.b)                      //[1,3]x3210
      w_210 = vmem(ptr_w+#0)                      //[1,3]
   } {
      s11.uw += vrmpy(z5432.ub, w210_.ub)         //[0,8]z5432
      z5476 = vdelta(y1, perm3)                   //[1,4]x7654
      u_210.tmp = vmem(ptr_w++#1)                 //[1,4]
      w210_.w = vasl(u_210.w, s8)                 //[1,4]
   } {
      sa2.uw += vrmpy(z5432.ub, filt_off_210.ub)  //[1,5]z3210filter oddn output
      sa3.uw += vrmpy(z5432.ub, filt_off210_.ub)  //[1,5]z3210filter oddn output
   }:endloop0                                     //max accumulator=9*255=8f7=12bits-2^24 
/* --------------------------------------------------------------------------- */
   {  s00.uw += vrmpy(z3210.ub, w_210.ub)         //[1,6]z3210
      z5432.h = vshuffo(z5476.h, z3210.h)         //[1,6]
      ptr_x0 = add(ptr_x0, next_in_width_32)      //update input ptr to next depth position
      loop0(.L_vloop, filt_height)                //can have a filter of Nx3 stride = 1
   } {
      sa0.uw += vrmpy(z3210.ub, filt_off_210.ub)  //[1,7]z3210filter oddn output
      sa1.uw += vrmpy(z3210.ub, filt_off210_.ub)  //[1,7]z3210filter oddn output
   } {
      s00.w = vsub(s00.w, sa0.w)                  //
      s01.uw += vrmpy(z3210.ub, w210_.ub)         //[1,8]z3210
   } {
      s01.w = vsub(s01.w, sa1.w)                  //
      s10.uw += vrmpy(z5432.ub, w_210.ub)         //[1,9]z5432
      maxe.w = vmax(maxe.w, s00.w)                //find max
   } {
      sa2.uw += vrmpy(z5432.ub, filt_off_210.ub)  //[1,10]z3210filter oddn output
      sa3.uw += vrmpy(z5432.ub, filt_off210_.ub)  //[1,10]z3210filter oddn output
      mine.w = vmin(mine.w, s00.w)                //find min
      maxe.w = vmax(maxe.w, s01.w)                //find max
   } {
      s01_s.w = vasl(s01.w,zshift)                //
      s10.w = vsub(s10.w, sa2.w)                  //
      s11.uw += vrmpy(z5432.ub, w210_.ub)         //[1,11]z5432
   } {
      s00.w = vasl(s00.w,zshift)                  //
      s11.w = vsub(s11.w, sa3.w)                  //
      mine.w = vmin(mine.w, s01.w)                //find min
      maxe.w = vmax(maxe.w, s10.w)                //find max
   } {
      mine.w = vmin(mine.w, s10.w)                //find min
      s10.w = vasl(s10.w,zshift)                  //
      ptr_z0 = add(ptr_z0, next_out_width_32)     //update output ptr to next depth
      d0.w = vmpye(s00.w, vrecip.uh)              //multiply by 1/max 
   } {
      maxe.w = vmax(maxe.w, s11.w)                //find max
      d1.w = vmpye(s01_s.w, vrecip.uh)            //multiply by 1/max 
      x0.tmp = vmem(ptr_x0+#0)                    //[0,0]read first 4 depths
      y0.b = vshuff(x0.b)                         //[0,0]1st part of shuffle 4
   } {
      mine.w = vmin(mine.w, s11.w)                //find min of acc
      d0.w += vmpyo(s00.w, vrecip.h):SSR          //multiply by 1/max 
      x1.tmp = vmem(ptr_x0+#1)                    //[0,1]load 2nd 4 depths
      y1 = vrdelta(x1, perm2)                     //[0,1]1st part of shuffle 4
   } {
      ptr_x1 = ptr_x0                             //
      d1.w += vmpyo(s01_s.w, vrecip.h):SSR        //multiply by 1/max 
      sa0 = #0                                    //
      s11.w = vasl(s11.w,zshift)                  //
   } {
      d2.w = vmpye(s10.w, vrecip.uh)              //multiply by 1/max 
      sa1 = #0                                    //
   } {
      woffset = vmem(filt_sum++#1)                //read in sum of taps
      ptr_x1 =add(ptr_x1, next_in_width_depth)    //[0,1]update ptr to next logical line
      z5476 = vdelta(y1, perm3)                   //[0,3]x7654
      d3.w = vmpye(s11.w, vrecip.uh)              //multiply by 1/max 
   } {
      d2.w += vmpyo(s10.w, vrecip.h):SSR          //multiply by 1/max 
      d1d0.h = vpacke(d1.w, d0.w)                 //take upp er 16bits of rnded acc
      sa2 = #0                                    //
   } {
      d3.w += vmpyo(s11.w, vrecip.h):SSR          //multiply by 1/max 
      sa3 = #0                                    //
   } {
      w_210 = vmem(ptr_w+#0)                      //[0,2]
      s00 = woffset                               //init accumulators
      s01 = woffset                               //init accumulators
   } {
      d3d2.h = vpacke(d3.w, d2.w)                 //take upp er 16bits of rnded acc
      s10 = woffset                               //init accumulators
      stride_vert = memw(sp+#28<<2)               //
   } {
      s11 = woffset                               //init accumulators
      u_210.tmp = vmem(ptr_w++#1)                 //[0,3]
      w210_.w = vasl(u_210.w, s8)                 //[0,3]adjust taps to odd locations
      z3210.b = vshuff(y0.b)                      //[0,2]x3210
   } {
      d3_d0.ub = vpack(d3d2.h, d1d0.h):sat        //deal into sequence
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
   {  c4 = #-4                                    //deal words
      ptr_max = memw(sp+#25<<2)                   //
      loop0(.L_peak, #4)                          //
   } {  
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
      .size dwconv2dbbb_unsigned_v60_asm, .L_end-dwconv2dbbb_unsigned_v60_asm
#else
/* =========================================================================== */
        .text
        .file "dwconv2dbbb_unsigned_d32_v60_h.S"
        .global dwconv3x3bbb_unsigned_v60_asm
        .balign 32
        .type  dwconv3x3bbb_unsigned_v60_asm, @function
dwconv3x3bbb_unsigned_v60_asm:
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
#define filt_off210_           r16    //
#define filt_off_210           r17    //
#define out_width_in_depth     r9:8   //
#define out_height_next_out_row r11:10//
#define zshift_recip_level     r13:12 //    
#define stride_vert_ptr_max    r15:14 //
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
#define l0w210_                v11    //
#define l1w_210                v12    //
#define l1w210_                v13    //
#define l2w_210                v14    //
#define l2w210_                v15    //
#define l0z3210                v16    //
#define l1z3210                v17    //
#define l2z3210                v18    //
#define l0z7654                v19    //
#define l1z7654                v19    //
#define l2z7654                v19    //
#define y0                     v19    //
#define y1                     v19    //
#define y2                     v19    //
#define l0z3232                v20    //
#define l1z3232                v20    //
#define l2z3232                v20    //
#define l0z5432                v20    //
#define l1z5432                v20    //
#define l2z5432                v20    //
#define d3210                  v21    //
#define d0                     v22    //
#define d1                     v23    //
#define d1_d0                  v23:22 //
#define d2                     v24    //
#define d3                     v25    //
#define d3_d2                  v25:24 //
#define maxe                   v26    //
#define mine                   v27    //
#define d1d0                   v21    //
#define d3d2                   v24    //
#define x0                     v30    //
#define x1                     v30    //
#define x2                     v30    //
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
       filt_offset = memw(sp+#(32+4*8))             //
       vrecip = vsplat(recip_level)                 //
       s8 = #8                                      //shift by 8
    }{
       memd(sp+#8)  = r19:18                        //
       in_depth = lsr(in_depth, #5)                 //1/32
       filt_offset = vsplatb(filt_offset)           //
       maxe = vmem(ptr_max+#0)                      //
    }{
       memd(sp+#16) = r21:20                        //
       memd(sp+#24) = r23:22                        //
       out_width = lsr(out_width,#2)                //
       ptr_w = ptr_wi                               //
    }{
       filt_off210_ = asl(filt_offset, #8)          //
       filt_off_210 = lsr(filt_offset, #8)          //
       filt_sum = filt_sumi                         //
       mine = vmem(ptr_max+#1)                      //
    }
/* --------------------------------------------------------------------------- */
.L_height:
    {  ptr_x = ptr_xi                               //
       ptr_z = ptr_zi                               //
       p3 = xor(p3,p3)                              //    
       loop1(.L_depth,in_depth)                     //
    }
     .balign 32
.L_depth:
    {  woffset = vmem(filt_sum++#1)                 //read in sum of taps
       ptr_x0 = ptr_x                               //
       ptr_x1 = add(ptr_x,next_in_row)              //
       nop                                          //
    }{ 
       x0.tmp = vmem(ptr_x0++#1):nt                 //
       l0z3210.b = vshuff(x0.b)                     //
       ptr_x2 = add(ptr_x1,next_in_row)             //
       nop                                          //
    }{ 
       x1.tmp = vmem(ptr_x1++#1)                    //
       l1z3210.b = vshuff(x1.b)                     //
       loop0(.L_width,out_width)                    //
       nop                                          //
    }{ 
       x2.tmp = vmem(ptr_x2++#1)                    //
       l2z3210.b = vshuff(x2.b)                     //
       ptr_x = add(ptr_x,next_in_width_32)          //
    }{ 
       l0z3210.b = vshuff(l0z3210.b)                //
       l0w_210.cur = vmem(ptr_w++#1)                //
       l0w210_.w = vasl(l0w_210.w, s8)              //
    }{
       l1z3210.b = vshuff(l1z3210.b)                //
       l1w_210.cur = vmem(ptr_w++#1)                //
       l1w210_.w = vasl(l1w_210.w, s8)              //
    }{
       l2z3210.b = vshuff(l2z3210.b)                //
       l2w_210.cur = vmem(ptr_w++#1)                //
       l2w210_.w = vasl(l2w_210.w, s8)              //
    }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
    {  x0.tmp = vmem(ptr_x0++#1):nt                 //[0, 0]
       y0.b = vshuff(x0.b)                          //[0, 0]
       s00 = woffset                                //[0, 0]
       d2.w += vmpyo(s10.w, vrecip.h):SSR           //[1,27]multiply by 1/max 
    }{
       l0z3232.h = vshuffo(l0z3210.h,l0z3210.h)     //[0, 1]
       s01 = woffset                                //[0, 1]
       d3.w += vmpyo(s11.w, vrecip.h):SSR           //[1,28]multiply by 1/max 
    }{
       l0z7654.b = vshuff(y0.b)                     //[0, 2]x3210
       s00.uw += vrmpy(l0z3210.ub, l0w_210.ub)      //[0, 2]filter even output
       s10 = woffset                                //[0, 0]
    }{
       l0z5432.h = vshuffe(l0z7654.h, l0z3232.h)    //[0, 3]
       s11 = woffset                                //[0, 3]
       s01.uw += vrmpy(l0z3210.ub, l0w210_.ub)      //[0, 3]z3210filter oddn output
    }{
       sa0.uw  = vrmpy(l0z3210.ub, filt_off_210.ub) //[0, 4]z3210filter oddn output
       sa1.uw  = vrmpy(l0z3210.ub, filt_off210_.ub) //[0, 4]z3210filter oddn output
       d1d0.h = vpack(d1.w, d0.w):sat               //[1,29]take lower 16bits of rnded acc
    }{
       s10.uw += vrmpy(l0z5432.ub, l0w_210.ub)      //[0, 5]z5432
       l0z3210 = l0z7654                            //[0, 5]
       d3d2.h = vpack(d3.w, d2.w):sat               //[1,30]take lower 16bits of rnded acc
    }{
       x1.tmp = vmem(ptr_x1++#1)                    //[0, 6]
       y1.b = vshuff(x1.b)                          //[0, 6]
       s11.uw += vrmpy(l0z5432.ub, l0w210_.ub)      //[0, 6]z5432
    }{
       l1z3232.h = vshuffo(l1z3210.h,l1z3210.h)     //[0, 7]
       sa2.uw  = vrmpy(l0z5432.ub, filt_off_210.ub) //[0, 7]z3210filter oddn output
       sa3.uw  = vrmpy(l0z5432.ub, filt_off210_.ub) //[0, 7]z3210filter oddn output
    }{
       l1z7654.b = vshuff(y1.b)                     //[0, 8]x3210
       s00.uw += vrmpy(l1z3210.ub, l1w_210.ub)      //[0, 8]filter even output
    }{
       l1z5432.h = vshuffe(l1z7654.h, l1z3232.h)    //[0, 9]
       s01.uw += vrmpy(l1z3210.ub, l1w210_.ub)      //[0, 9]z3210filter oddn output
    }{
       sa0.uw += vrmpy(l1z3210.ub, filt_off_210.ub) //[0,10]z3210filter oddn output
       sa1.uw += vrmpy(l1z3210.ub, filt_off210_.ub) //[0,10]z3210filter oddn output
    }{
       s10.uw += vrmpy(l1z5432.ub, l1w_210.ub)      //[0,11]z5432
       l1z3210 = l1z7654                            //[0,11]
    }{
       x2.tmp = vmem(ptr_x2++#1)                    //[0,12]
       y2.b = vshuff(x2.b)                          //[0,12]
       s11.uw += vrmpy(l1z5432.ub, l1w210_.ub)      //[0,12]z5432
    }{
       l2z3232.h = vshuffo(l2z3210.h,l2z3210.h)     //[0,13]
       sa2.uw += vrmpy(l1z5432.ub, filt_off_210.ub) //[0,13]z3210filter oddn output
       sa3.uw += vrmpy(l1z5432.ub, filt_off210_.ub) //[0,13]z3210filter oddn output
    }{
       l2z7654.b = vshuff(y2.b)                     //[0,14]x3210
       s00.uw += vrmpy(l2z3210.ub, l2w_210.ub)      //[0,14]filter even output
    }{
       l2z5432.h = vshuffe(l2z7654.h, l2z3232.h)    //[0,15]
       s01.uw += vrmpy(l2z3210.ub, l2w210_.ub)      //[0,15]z3210filter oddn output
    }{
       sa0.uw += vrmpy(l2z3210.ub, filt_off_210.ub) //[0,16]z3210filter oddn output
       sa1.uw += vrmpy(l2z3210.ub, filt_off210_.ub) //[0,16]z3210filter oddn output
       l2z3210 = l2z7654                            //[0,16]
    }{
       s10.uw += vrmpy(l2z5432.ub, l2w_210.ub)      //[0,17]z5432
       s01s00.w = vsub(s01s00.w, sa1sa0.w)          //[0,17]
    }{
       s11.uw += vrmpy(l2z5432.ub, l2w210_.ub)      //[0,18]z5432
       maxe.w = vmax(maxe.w, s00.w)                 //[0,18]find max
       mine.w = vmin(mine.w, s00.w)                 //[0,18]find min
    }{
       sa2.uw += vrmpy(l2z5432.ub, filt_off_210.ub) //[0,19]z3210filter oddn output
       sa3.uw += vrmpy(l2z5432.ub, filt_off210_.ub) //[0,19]z3210filter oddn output
       maxe.w = vmax(maxe.w, s01.w)                 //[0,19]find max
       s00.w = vasl(s00.w,zshift)                   //[0,19]
    }{
       s11s10.w = vsub(s11s10.w, sa3sa2.w)          //[0,20]
       mine.w = vmin(mine.w, s01.w)                 //[0,20]find min
       s01.w = vasl(s01.w,zshift)                   //[0,20]
    }{
       d0.w = vmpye(s00.w, vrecip.uh)               //[0,21]multiply by 1/max 
       maxe.w = vmax(maxe.w, s10.w)                 //[0,21]find max
       mine.w = vmin(mine.w, s10.w)                 //[0,21]find min
    }{
       d1.w = vmpye(s01.w, vrecip.uh)               //[0,22]multiply by 1/max 
       maxe.w = vmax(maxe.w, s11.w)                 //[0,22]find max
       mine.w = vmin(mine.w, s11.w)                 //[0,22]find min of acc
    }{
       d0.w += vmpyo(s00.w, vrecip.h):SSR           //[0,23]multiply by 1/max 
       s10.w = vasl(s10.w,zshift)                   //[0,23]
    }{
       d1.w += vmpyo(s01.w, vrecip.h):SSR           //[0,24]multiply by 1/max 
       s11.w = vasl(s11.w,zshift)                   //[0,24]
    }{
       d2.w = vmpye(s10.w, vrecip.uh)               //[0,25]multiply by 1/max 
       d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[1,31]deal into sequence
       if p3 vmem(ptr_z++#1):nt = d3210.new         //[1,31]store quantized bytes
    }{
       d3.w = vmpye(s11.w, vrecip.uh)               //[0,26]multiply by 1/max 
       p3 = or(p3,!p3)                              //    
    }:endloop0:endloop1
/* --------------------------------------------------------------------------- */
    {  d2.w += vmpyo(s10.w, vrecip.h):SSR           //[1,27]multiply by 1/max 
       out_height = add(out_height, #-1)            //
    }{
       d3.w += vmpyo(s11.w, vrecip.h):SSR           //[1,28]multiply by 1/max 
       p0 = cmp.eq(out_height, #0)                  //
    }{
       d1d0.h = vpack(d1.w, d0.w):sat               //[1,29]take lower 16bits of rnded acc
       ptr_xi+=mpyi(stride_vert,next_in_row)        //incrmeent input ptr
       ptr_zi=add(ptr_zi,next_out_row)              //incrmeent output ptr
    }{
       d3d2.h = vpack(d3.w, d2.w):sat               //[1,30]take lower 16bits of rnded acc
       ptr_w = ptr_wi                               //
       filt_sum = filt_sumi                         //
    }{
       d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[1,31]deal into sequence
       vmem(ptr_z+#0):nt = d3210.new                //[1,31]store quantized bytes
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
       sp = add(sp,#4*8)                            // pop stack
       jumpr r31                                    //return
    }
.L_end:
      .size dwconv3x3bbb_unsigned_v60_asm, .L_end-dwconv3x3bbb_unsigned_v60_asm
#endif
