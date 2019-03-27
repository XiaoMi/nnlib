
/*
 * Copyright (c) 2016,2017,2018 The Linux Foundation. All rights reserved.
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
/*  FUNCTIONS      : maxpool_slice_hvx_3x3_stride2                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Get maximun on 3x3 with d32 format                   */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->                                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =      bytes                                            */
/*     STACK    =      bytes                                            */
/*     ASSUMPTIONS                                                      */
/*        arrays are 128 byte aligned                                   */
/*  C MODEL                                                             */
/*======================================================================*/

/*
int maxpool_slice_hvx_3x3_stride2(
    uint8_t *out,
    const uint8_t *in,
    int32_t in_next_row,
    int32_t out_next_row,
    int32_t out_vectors_wide,
    int32_t out_lines,
    int32_t out_lalign);
*/
        .file "maxpool_3x3_s2_d32.S"
/*======================================================================*/
#define outptr                  r0
#define inptr                   r1
#define in_outptr               r1:0
#define in_next_row             r2
#define tmp_outptr              r3
#define r_valign                r4
#define out_lines               r5
#define const32                 r6
#define const_m32               r7
#define const_m32_32            r7:6
#define outer_next_outptr       r8
#define outer_next_inptr        r9
#define outer_next_in_outptr    r9:8
#define rtmp                    r10
#define out_align               r11
#define fr                      out_align
#define out_vectors_wide        r12
#define out_next_row            r13
#define l2param_l               r14
#define l2param_h               r15
#define l2param                 r15:14
#define pf_width                r14
#define pf_height               r15
#define c128                    r28
#define p_odd                   p1
#define pfptr                   r10
/*======================================================================*/
#define in00                    v0
#define in10                    v1
#define in20                    v2
#define in30                    v3
#define in40                    v4
#define h0_0246                 v8
#define h0_1357                 v9
#define h0_1357_0246            v9:8
#define h1_0246                 v10
#define h1_1357                 v11
#define h1_1357_0246            v11:10
#define vsum0L                  v12
#define vsum1L                  v13
#define vsum0M                  v14
#define vsum1M                  v15
#define acc0                    v18
#define acc1                    v19
#define maxtmp                  v22
#define vXsum0L                 v24
#define vXsum1L                 v25
#define vXsum0M                 v26
#define vXsum1M                 v27
/* NOTE CAREFULLY: alias INx8 to INx0 for stride2 */
#define vsum0R                  vsum0L
#define vsum1R                  vsum1L
#define vXsum0R                 vXsum0L
#define vXsum1R                 vXsum1L
/*======================================================================*/
#if __HEXAGON_ARCH__ == 60
#define COND(_a)
#elif __HEXAGON_ARCH__ >= 62
#define COND(_a) IF (_a)
#endif
/*======================================================================*/
//
// last parameter out_lalign:
//   = 0 means  output[0] is based on in[0,1,2], output [1] on in[2,3,4] etc
//   = 32 means output [0] is based on [1,2,3], output [1] on in[3,4,5] etc
//   = 64 means output[3] is based on in[0,1,2], output [4] on in[2,3,4] etc
//   = 96 means output[3] is based on in[1,2,3], output [4] on in[3,4,5] etc
//
// (currently this supports only 0 and 96; others are only needed when
// w_pad_before != 4, and the C code isn't set up to handle that yet.
/*======================================================================*/
    .text
    .global maxpool_slice_hvx_3x3_stride2
    .type maxpool_slice_hvx_3x3_stride2,@function
maxpool_slice_hvx_3x3_stride2:
    {
      out_lines = asrrnd(out_lines,#1)                  //
      m0 = in_next_row                                  //
      out_vectors_wide = r4                             // free r4 for r_valign
      out_next_row = r3                                 // free r3
    }{
      out_align=memw(sp+#0)                             //
      rtmp = asl(in_next_row,#2)                        //
      c128 = #128                                       //
      pf_width = asl(out_vectors_wide,#8)               // 128*2*out_vectors_wide
    }{
      rtmp = sub(#128, rtmp)                            //
      p_odd = tstbit(out_align,#5)                      //
      if !p_odd.new pf_width  = add(pf_width,c128)      // 128*2*out_vectors_wide (+ 128)
      pf_height = #4                                    //
    }{
      m1 = rtmp                                         //
      const_m32_32 = combine(#-32,#32)                  //
      l2param_l = combine(pf_width.l,pf_height.l)       //
      l2param_h = in_next_row                           //
    }{
      l2param_h.H = #1                                  //l2parm= 1|in_next_row|128*2*out_vectors_wide(+128)|4
      loop1(.Louter_3x3s2,out_lines)                    //
      outer_next_inptr  = addasl(inptr,in_next_row,#2)  //
      nop                                               //
    }{
      p0 = cmp.gt(out_lines,#1)                         //
      if !p0.new l2param_l = #0                         //
      pfptr = add(outer_next_inptr,in_next_row)         //
      if (p_odd) jump:nt .Ldo_inner96                   //
    }
    .balign 32
.Louter_3x3s2:
.wait_l2fetch:
    { fr = usr                                          // wait_for_l2fetch()
    }{
      p0 = cmp.gt(fr,#-1)                               //
      if (!p0.new) jump:nt .wait_l2fetch                //
    }{
      l2fetch(pfptr,l2param)                            //
    }{
      in00 = vmem(inptr++m0)                            //
      out_lines = add(out_lines,#-1)                    //
      p3 = sp1loop0(.Linner_3x3s2,out_vectors_wide)     //
    }{
      in10.tmp = vmem(inptr++m0)                        //
      vsum0L.ub = vmax(in00.ub,in10.ub)                 //
      nop                                               //
    }{
      in20.cur = vmem(inptr++m0)                        //
      vsum0L.ub = vmax(vsum0L.ub,in20.ub)               //
      outer_next_outptr = addasl(outptr,out_next_row,#1)//
      nop                                               //
    }{
      in30.tmp = vmem(inptr++m0)                        //
      vsum1L.ub = vmax(in20.ub,in30.ub)                 //
    }{
      in40.tmp = vmem(inptr++m1)                        //
      vsum1L.ub = vmax(vsum1L.ub,in40.ub)               //
    }
    .balign 32
.Linner_3x3s2:
    {
      in00 = vmem(inptr++m0)                            //
      acc0.ub = vmax(acc0.ub, maxtmp.ub)                //[2]row 0 outputs.
      tmp_outptr = add(outptr,out_next_row)             //[2]
      COND(p3) vmem(outptr+#0) = acc0.new               //[2]
    }{
      in10.tmp = vmem(inptr++m0)                        //
      vsum0M.ub = vmax(in00.ub,in10.ub)                 //
      maxtmp = valign( vsum1R, h1_0246, const32 )       //[2] {2,4,6,8 }
    }{
      in20.cur = vmem(inptr++m0)                        //
      vsum0M.ub = vmax(vsum0M.ub,in20.ub)               //
      acc1.ub = vmax(acc1.ub, maxtmp.ub)                //[2]row 1 outputs.
      COND(p3) vmem(tmp_outptr+#0) = acc1.new           //[2]
    }{
      in30.tmp = vmem(inptr++m0)                        //
      vsum1M.ub = vmax(in20.ub,in30.ub)                 //
      if p3 outptr = add(outptr,c128)                   //[2]
    }{
      h0_1357_0246 = vdeal( vsum0M, vsum0L,const_m32)   //
      in40.tmp = vmem(inptr++m1)                        //
      vsum1M.ub = vmax(vsum1M.ub,in40.ub)               //
    }{
                                                        // have {0,2,4,6} and {1,3,5,7}; need {2,4,6,8}
      acc0.ub = vmax(h0_1357.ub, h0_0246.ub )           //
      in00 = vmem( inptr++m0)                           //
    }{
      h1_1357_0246 = vdeal( vsum1M, vsum1L,const_m32)   //
      in10.tmp = vmem(inptr++m0)                        //
      vsum0R.ub = vmax(in00.ub,in10.ub)                 //
    }{
      in20.cur = vmem(inptr++m0)                        //
      vsum0R.ub = vmax(vsum0R.ub,in20.ub)               //
      acc1.ub = vmax(h1_1357.ub, h1_0246.ub )           //
    }{
      in30.tmp = vmem(inptr++m0)                        //
      vsum1R.ub = vmax(in20.ub,in30.ub)                 //
    }{
      maxtmp = valign( vsum0R, h0_0246, const32 )       // {2,4,6,8 }
      in40.tmp = vmem(inptr++m1)                        //
      vsum1R.ub = vmax(vsum1R.ub,in40.ub)               //
    }:endloop0

    {
      acc0.ub = vmax(acc0.ub, maxtmp.ub)                //[2]row 0 outputs.
      tmp_outptr = add(outptr,out_next_row)             //[2]
      vmem(outptr+#0) = acc0.new                        //[2]
      in_outptr = outer_next_in_outptr                  //
    }{
      maxtmp = valign( vsum1R, h1_0246, const32 )       //[2] {2,4,6,8 }
      p0 = cmp.gt(out_lines,#1)                         //
      outer_next_inptr += asl(in_next_row,#2)           //
    }{
      acc1.ub = vmax(acc1.ub, maxtmp.ub)                //[2]row 1 outputs.
      vmem(tmp_outptr+#0) = acc1.new                    //[2]
      if !p0 l2param_l = #0                             //
      pfptr = add(outer_next_inptr,in_next_row)         //
    }:endloop1
/*======================================================================*/
    {
      r0 = #0                                           //
      jumpr r31                                         //
    }
/*======================================================================*/
// out_lalign = 96
// this is like the align = 0 case, if we push 5 pixels into the start of each row first.
// this is done by finding vsumX[01][LMR] and then align M,L,
//
// before row loop:
//
//      Lx = L= [x,x,x,x]
//  in row loop:
//
//      Mx = [0,1,2,3]         + 8*i
//        ->  M = valign(Mx,Lx,96) = [x,0,1,2]
//     Rx = [4,5,6,7]       + 8*i
//       -> R = valign(Rx,Mx,96) = [3,4,5,6]
//      - use L,M,R as before;
//     Lx <= Rx; L<=R  (done by aliased regs)
//
//
//
/*======================================================================*/
.Ldo_inner96:
    {
      loop1(.Louter_3x3s2_x96,out_lines)                //
      r_valign = #96                                    //
    }
    .balign 32
.Louter_3x3s2_x96:
.wait_l2fetch_x96:
    { fr = usr                                          // wait_for_l2fetch()
    }{
      p0 = cmp.gt(fr,#-1)                               //
      if (!p0.new) jump:nt .wait_l2fetch_x96            //
    }{
      l2fetch(pfptr,l2param)                            //
    }{
      vXsum0L = #0                                      //
      vXsum1L = #0                                      //
      outer_next_outptr = addasl(outptr,out_next_row,#1)//
      p3 = sp1loop0(.Linner_3x3s2_x96,out_vectors_wide) //
    }{
      vsum0L = vXsum0L                                  //
      vsum1L = vXsum1L                                  //
      nop                                               //
    }
    .balign 32
.Linner_3x3s2_x96:
    {
      in00 = vmem(inptr++m0)                            //
      acc1.ub = vmax(h1_1357.ub, h1_0246.ub )           //[2]
      tmp_outptr = add(outptr,out_next_row)             //[2]
      maxtmp = valign( vsum0R, h0_0246, const32 )       //[2] {2,4,6,8 }
    }{
      in10 = vmem(inptr++m0)                            //
      vsum1R = valign(vXsum1R,vXsum1M,r_valign)         //[2]
      acc0.ub = vmax(acc0.ub, maxtmp.ub)                //[2]
#if __HEXAGON_ARCH__ == 60
      vmem(outptr+#0) = acc0.new                        //[2]
#else
      if p3 vmem(outptr++#1) = acc0.new                 //[2]
#endif
    }{
      in20 = vmem(inptr++m0)                            //
      vXsum0M.ub = vmax(in00.ub,in10.ub)                //
    }{
      vXsum0M.ub = vmax(vXsum0M.ub,in20.ub)             // done
      maxtmp = valign( vsum1R, h1_0246, const32 )       //[2] {2,4,6,8 }
    }{
      in30.cur = vmem(inptr++m0)                        //
      vXsum1M.ub = vmax(in20.ub,in30.ub)                //
      acc1.ub = vmax(acc1.ub, maxtmp.ub)                //[2]row 1 outputs.
#if __HEXAGON_ARCH__ == 60
      vmem(tmp_outptr+#0) = acc1.new                    //[2]
#else
      if p3 vmem(tmp_outptr+#0) = acc1.new              //[2]
#endif
    }{
      vsum0M = valign(vXsum0M,vXsum0L,r_valign)         //
      in40.cur = vmem(inptr++m1)                        //
      vXsum1M.ub = vmax(vXsum1M.ub,in40.ub)             // done
    }{
      in00 = vmem(inptr++m0)                            //
#if __HEXAGON_ARCH__ == 60
      if p3 outptr = add(outptr,c128)                   //[2]
#endif
    }{
      in10.cur = vmem(inptr++m0)                        //
      vXsum0R.ub = vmax(in00.ub,in10.ub)                //
      vsum1M = valign(vXsum1M,vXsum1L,r_valign);        //
    }{
      in20.cur = vmem(inptr++m0)                        //
      vXsum0R.ub = vmax(vXsum0R.ub,in20.ub)             // done
      h0_1357_0246 = vdeal( vsum0M, vsum0L,const_m32)   //
    }{
      in30.cur = vmem(inptr++m0)                        //
      vXsum1R.ub = vmax(in20.ub,in30.ub)                //
      h1_1357_0246 = vdeal( vsum1M, vsum1L,const_m32)   //
    }{
      in40.cur = vmem(inptr++m1)                        //
      vXsum1R.ub = vmax(vXsum1R.ub,in40.ub)             // done
      vsum0R = valign(vXsum0R,vXsum0M,r_valign);        //
      acc0.ub = vmax(h0_1357.ub, h0_0246.ub )           //
    }:endloop0

    {
      acc1.ub = vmax(h1_1357.ub, h1_0246.ub )           //
      tmp_outptr = add(outptr,out_next_row);            //
      maxtmp = valign( vsum0R, h0_0246, const32 )       // {2,4,6,8 }
      out_lines = add(out_lines,#-1)                    //
    }{
      acc0.ub = vmax(acc0.ub, maxtmp.ub);               // row 0 outputs.
      vmem(outptr+#0) = acc0.new                        //
      vsum1R = valign(vXsum1R,vXsum1M,r_valign);        //
      in_outptr = outer_next_in_outptr                  //
    }{
      maxtmp = valign( vsum1R, h1_0246, const32 )       // {2,4,6,8 }
      outer_next_inptr += asl(in_next_row,#2)           //
      p0 = cmp.gt(out_lines,#1)                         //
      if !p0.new l2param_l = #0                         //
    }{
      acc1.ub = vmax(acc1.ub, maxtmp.ub);               // row 1 outputs.
      vmem(tmp_outptr+#0) = acc1.new                    //
      pfptr = add(outer_next_inptr,in_next_row)         //
    }:endloop1
/*======================================================================*/
    {
      r0 = #0                                           //
      jumpr r31                                         //
    }
/*======================================================================*/
    .size maxpool_slice_hvx_3x3_stride2,.-maxpool_slice_hvx_3x3_stride2
/*======================================================================*/
