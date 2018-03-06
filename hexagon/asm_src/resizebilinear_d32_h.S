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
 *  FUNCTIONS      : do_bilin_interp_x2_single_slice_HVX
 *
 *  DESCRIPTION
 *    2x bilinear interpolation resize
 *
 *  ARCHITECTURE   : QDSP6V60  + HVX
 *
 *  REVISION HISTORY:
 *  =================
 *
 *  Author              Date           Comments
 *  -------------------------------------------------------------
 *
 *  CYCLE-COUNT:
 *
 *  MEMORY
 *     CODESIZE =    bytes
 *     STACK    =    bytes
 *     ASSUMPTIONS
 *
 *
 *  C MODEL
 */
/*=============================================================================*/
        .text
        .global do_bilin_interp_x2_single_slice_HVX
        .balign 32
        .type  do_bilin_interp_x2_single_slice_HVX, @function
do_bilin_interp_x2_single_slice_HVX:
/*=============================================================================*/
#if __HEXAGON_ARCH__ == 60
#define COND(_a)
#elif __HEXAGON_ARCH__ >= 62
#define COND(_a) IF (_a)
#endif
/*=============================================================================*/
#define in_ptr                 r0
#define in_row_stride          r1
#define in_width               r2
#define in_height              r3
#define out_ptr                r4
#define out_row_stride         r5
#define c_32                   r6
#define c32                    r7
#define c32_32                 r7:6
#define wloops                 r8
#define in_heighteven          r9
#define vpin1                  in_heighteven
#define vpin0                  r10
#define vpout0                 r11
#define vpout0_vpin0           r11:10
#define vpin2                  r12
#define vpout1                 r13
#define vpout2                 r14
#define vpout3                 r15
#define vpout3_2               r15:14
#define vplastline             r28
/*=============================================================================*/
#define qright                 q0
#define vl0                    v0
#define vl1                    v1
#define vl2                    v2
#define vr0                    v3
#define vr1                    v4
#define vr2                    v5
#define v0_1                   v6
#define v1_1                   v0_1
#define v2_1                   v0_1
#define vl0s1                  v0_1
#define vl1s1                  v0_1
#define vl2s1                  v0_1
#define v0odd                  v7
#define v1odd                  v8
#define v2odd                  v9
#define vout0_l                v10
#define vout0_h                v11
#define dvout0                 v11:10
#define vout1_l                v12
#define vout1_h                v13
#define dvout1                 v13:12
#define vout2_l                v14
#define vout2_h                v15
#define dvout2                 v15:14
#define sv2_l                  v16
#define sv2_h                  v17
#define dv2                    v17:16
#define vout1a                 v18
#define vout1b                 v19
#define vout3a                 v20
#define vout3b                 v21
#define vodd0                  v22
#define vodd1                  v23
#define vodd2                  v24
#define vr                     v25
#define vl0t                   vr
/*=============================================================================*/
    {
      in_width = add(in_width,#-1)                  //
      in_heighteven = asr(in_height,#1)             //
      vplastline = mpyi(in_height,in_row_stride)    //
      c32_32 = combine(#32,#-32)                    //
    }{
      wloops = asr(in_width,#2)                     //
      vplastline = add(vplastline,in_ptr)           //
	  vpout3_2 = combine(out_ptr,out_ptr)           //
	  vpout1 = out_ptr                              //
    }{
      in_width = asl(in_width,#5)                   //
      p1 = cmp.gt(in_heighteven,#0)                 //
      vplastline = sub(vplastline,in_row_stride)    //
      p0 = cmp.gt(r0,r0)                            //
    }{
      qright = vsetq(in_width)                      //
      if !p1 jump .lpy_end                          //
      p1 = cmp.gt(wloops,#0)                        //
      loop1(.lpy,in_heighteven)                     //
    }
    .falign
.lpy:
    {
	  vpout0_vpin0 = combine(out_ptr,in_ptr)        //
      vpin1 = add(in_ptr,in_row_stride)             //
      vpin2 = addasl(in_ptr,in_row_stride,#1)       //
      if p0 vmem(vpout0+#0) = vout0_h               //[O2]
    }{
      vl0 = vmem(vpin0++#1)                         //
      vpout1 = add(out_ptr,out_row_stride)          //
      p3 = sp1loop0(.lpx,wloops)
      COND(p0) vmem(vpout1+#0) = vout1b             //[O2]
    }{
      vl1 = vmem(vpin1++#1)                         //
      vpin2 = min(vpin2,vplastline)                 //
      vpout2 = add(vpout1,out_row_stride)           //
      COND(p0) vmem(vpout2+#0) = vout1_h            //[O2]
    }{
      vl2 = vmem(vpin2++#1)                         //
      vpout3 = add(vpout2,out_row_stride)           //
      if !p1 jump .lpx_end                          //
      COND(p0) vmem(vpout3+#0) = vout3b             //[O2]
    }
    .falign
.lpx:
    {
      vr0.cur = vmem(vpin0++#1)                     //
      v0_1 = valign(vr0,vl0,c32)                    //
      COND(p3) vmem(vpout1+#0) = vout1b             //[2]
	  if p3 vpout1 = add(vpout1,#64)                //[2]
    }{
      vr1.cur = vmem(vpin1++#1)                     //
      v1_1 = valign(vr1,vl1,c32)                    //
      v0odd.ub = vavg(vl0.ub,v0_1.ub)               //
      COND(p3) vmem(vpout3+#0) = vout3a             //[2]
    }{
      vr2.cur = vmem(vpin2++#1)                     //
      v2_1 = valign(vr2,vl2,c32)                    //
      v1odd.ub = vavg(vl1.ub,v1_1.ub)               //
      COND(p3) vmem(vpout3+#1) = vout3b             //[2]
    }{
      v2odd.ub = vavg(vl2.ub,v2_1.ub)               //
      dvout0 = vshuff(v0odd,vl0,c_32)               //
      vmem(vpout0++#1) = vout0_l.new                //
      vl0 = vr0                                     //
    }{
      dvout2 = vshuff(v1odd,vl1,c_32)               //
      vmem(vpout2++#1) = vout2_l.new                //
      vl1 = vr1                                     //
	  if p3 vpout1 = add(vpout1,#64)                //[2]
    }{
      dv2 = vshuff(v2odd,vl2,c_32)                  //
      vmem(vpout0++#1) = vout0_h                    //
      vl2 = vr2                                     //
    }{
      vmem(vpout2++#1) = vout2_h                    //
      vout1b.ub = vavg(vout2_h.ub,vout0_h.ub)       //
      vout3a.ub = vavg(sv2_l.ub,vout2_l.ub)         //
      vout3b.ub = vavg(sv2_h.ub,vout2_h.ub)         //
    }{
      vout1a.ub = vavg(vout2_l.ub,vout0_l.ub)       //
      vmem(vpout1++#1) = vout1a.new                 //
	  if p3 vpout3 = add(vpout3,#256)               //[2]
    }:endloop0
.lpx_end:
	{
      if p3 vmem(vpout1++#1) = vout1b               //[e]
      vl0s1 = vror(vl0,c32)                         //
      p0 = tstbit(in_width,#1+5)                    //
    }{
      if p3 vmem(vpout3++#1) = vout3a               //[e]
      vr0 = vmux(qright,vl0s1,vl0)                  //
      vl1s1 = vror(vl1,c32)                         //
    }{
      if p3 vmem(vpout3++#1) = vout3b               //[e]
      vodd0.ub = vavg(vr0.ub,vl0.ub)                //
      vr1 = vmux(qright,vl1s1,vl1)                  //
      vl2s1 = vror(vl2,c32)                         //
    }{
      vodd1.ub = vavg(vr1.ub,vl1.ub)                //
      vr2 = vmux(qright,vl2s1,vl2)                  //
    }{
      dvout0 = vshuff(vodd0,vl0,c_32)               //
      vodd2.ub = vavg(vr2.ub,vl2.ub)                //
      vmem(vpout0++#1) = vout0_l.new                //
    }{
      dvout1 = vshuff(vodd1,vl1,c_32)               //
      vmem(vpout2++#1) = vout1_l.new                //
      in_ptr += mpyi(in_row_stride,#2)              //
      out_ptr += mpyi(out_row_stride,#4)            //
                                                    //
    }{
      dvout2 = vshuff(vodd2,vl2,c_32)               //
      vout1a.ub = vavg(vout1_l.ub,vout0_l.ub):rnd   //
      vout1b.ub = vavg(vout1_h.ub,vout0_h.ub):rnd   //
      vmem(vpout1++#1) = vout1a.new                 //
    }{
      vout3a.ub = vavg(vout2_l.ub,vout1_l.ub):rnd   //
      vout3b.ub = vavg(vout2_h.ub,vout1_h.ub):rnd   //
      vmem(vpout3++#1) = vout3a.new                 //
    }:endloop1
.lpy_end:
	{
      if p0 vmem(vpout0+#0) = vout0_h               //[e]
      vpin0 = in_ptr                                //
      vpout0 = out_ptr                              //
      p2 = tstbit(in_height,#0)                     //
    }{
      if p0 vmem(vpout1+#0) = vout1b                //[e]
      vpout1 = add(out_ptr,out_row_stride)          //
    }{
      if p0 vmem(vpout2+#0) = vout1_h               //[e]
    }{
      if p0 vmem(vpout3+#0) = vout3b                //[e]
      if (!p2) jumpr r31                            //
    }{
      if !p1 jump .loddx_end                        //
      p3 = sp1loop0(.lodd,wloops)                   //
      vl0 = vmem(vpin0++#1)                         //
    }
    .falign
.lodd:
    {
      vr0.cur = vmem(vpin0++#1)                     //
      vl0s1 = valign(vr0,vl0,c32)                   //
      vl0t = vl0                                    //
      COND(p3) vmem(vpout0+#0) = vout0_h            //[2]
    }{
      v0odd.ub = vavg(vl0.ub,vl0s1.ub)              //
      vl0 = vr0                                     //
      if p3 vmem(vpout1++#1) = vout0_l              //[2]
    }{
      if p3 vmem(vpout1++#1) = vout0_h              //[2]
	  if p3 vpout0 = add(vpout0,#128)               //[2]
    }{
      dvout0 = vshuff(v0odd,vl0t,c_32)              //
      vmem(vpout0++#1) = vout0_l.new                //
    }:endloop0
.loddx_end:
	{
      if p3 vmem(vpout0++#1) = vout0_h              //[e]
      vl0s1 = vror(vl0,c32)                         //
    }{
      if p3 vmem(vpout1++#1) = vout0_l              //[e]
      vr = vmux(qright,vl0s1,vl0)                   //
    }{
      if p3 vmem(vpout1++#1) = vout0_h              //[e]
      vodd0.ub = vavg(vr.ub,vl0.ub)                 //
    }{
      dvout0 = vshuff(vodd0,vl0,c_32)               //
      vmem(vpout0++#1) = vout0_l.new                //
      p0 = tstbit(in_width,#1+5)                    //
    }{
      vmem(vpout1++#1) = vout0_l                    //
    }{
      if p0 vmem(vpout0++#1) = vout0_h              //
    }{
      if p0 vmem(vpout1++#1) = vout0_h              //
      jumpr r31                                     //
    }
.do_bilin_interp_x2_single_slice_HVX_end:
/*=============================================================================*/
      .size do_bilin_interp_x2_single_slice_HVX, .do_bilin_interp_x2_single_slice_HVX_end-do_bilin_interp_x2_single_slice_HVX
/*=============================================================================*/

