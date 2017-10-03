/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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

        .file "avgpool_d32.S"
/*======================================================================*/
/*  FUNCTIONS      : avgpool_slice_hvx_3x3_stride1                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform 3x3 avgerage on d32 format                   */
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

/*=============================================================================*/
#define optr            r0
#define iptr            r1
#define in_next_row     r2 
#define out_next_row    r3
#define out_vectors     r4
#define out_lines       r5
/*=============================================================================*/
#define c32             r6
#define c64             r7
#define c64_c32         r7:6
#define recip9          r8
#define c01             r9
#define iptr0           r10
#define optr0           r11
#define optr0_iptr0     r11:10
#define optr1           r12
#define scale           r13
#define l2param_l       r14
#define l2param_h       r15
#define l2param         r15:14
#define pfptr0          r16
#define pfptr1          r17
#define in_next_row_2   r28
#define out_alig        in_next_row
#define offset          scale
/*=============================================================================*/
#define sl0             v0
#define sl2             v1
#define dl2l0           v1:0
#define sl1             v2
#define sout0           v3
#define svsum00_e       v4
#define svsum00_o       v5
#define dvsum00         v5:4
#define svsum01_e       v6
#define svsum01_o       v7
#define dvsum01         v7:6
#define svsum10_e       v8
#define svsum10_o       v9
#define dvsum10         v9:8
#define svsum11_e       v10
#define svsum11_o       v11
#define dvsum11         v11:10
#define ssum0_e         v12
#define ssum0_o         v13
#define dsum0           v13:12
#define ssum1_e         v14
#define ssum1_o         v15
#define dsum1           v15:14
#define svs0d2_e        v16
#define svs0d2_o        v17
#define dvs0d2          v17:16
#define sout0_new       v18
#define sout0_old       v19
#define sout1_new       v20
#define sout1_old       v21
#define sl3             sl1
#define sout1           sout0
#define svs0d1_e        svs0d2_e 
#define svs0d1_o        svs0d2_o
#define svs1d1_e        svs0d2_e 
#define svs1d1_o        svs0d2_o
#define svs1d2_e        svs0d2_e 
#define svs1d2_o        svs0d2_o
/*=============================================================================*/
#define SRS             <<1:rnd:sat
/*=============================================================================*/
    .text
    .global avgpool_slice_hvx_3x3_stride1
    .balign 32
    .type  avgpool_slice_hvx_3x3_stride1, @function
avgpool_slice_hvx_3x3_stride1:
    {  p0 = cmp.gt(out_lines,#0)                    //
       if !p0.new jumpr:nt r31                      //
       recip9 = ##0x0e390e39                        //32768/9 (rounded) = 3641
        // note: (255*9* 3641 + (1<<14))>>15 = 255 (no overflow).
    }{
       out_lines = asrrnd(out_lines,#1)             //
       m0 = in_next_row                             //
       offset = #128                                //
       c64_c32 = combine(#64,#32)                   //
    }{
       offset -= mpyi(in_next_row,#3)               //
       in_next_row_2 = ASL(in_next_row,#1)          //
       out_alig = memw(sp+#0)                       //
       sp = add(sp,#-8)                             //
    }{
       memd(sp+#0) = r17:16                         //
       m1 = offset                                  //
       c01 = ##0x01010101                           //
    }{
       l2param_l = add(out_vectors,#1)              //
       l2param_h = #128                             //
       pfptr0 = addasl(iptr,in_next_row_2,#1)       //
       pfptr1 = addasl(iptr,in_next_row_2,#1)       //
    }{
       pfptr1 += lsr(in_next_row_2,#1)              //
       l2param_l = combine(l2param_h.l,l2param_l.l) // l2param=(0|128|128|out_vectors+1)
       loop1(.avgpool3x3s1_d32_outloop,out_lines)   //
       optr0_iptr0 = combine(optr,iptr)             //
    }
    .balign 32
/*============================================================================*/
.avgpool3x3s1_d32_outloop:
    {  optr1 = add(optr0,out_next_row)              //
       p0 = cmp.eq(out_lines,#1)                    // if (last iteration)
       if p0.new l2param_l = #0                     // then cancel l2fetch
       sl0 = vmem(iptr0++m0)                        //
    }{
       l2fetch(pfptr0,l2param)                      //
    }{
       l2fetch(pfptr1,l2param)                      //
    }{  
       sl1.tmp = vmem(iptr0++m0)                    //
       dvsum00.h = vmpy(sl1.ub,c01.b)               //
       p3=sp1loop0(.avgpool3x3s1_d32_rowloop,out_vectors)//
    }{
       sl2.tmp = vmem(iptr0++m0)                    //
       dvsum00.h += vmpa(dl2l0.ub,c01.b)            // vertical summation
    }{
       sl3.tmp = vmem(iptr0++m1)                    //
       dvsum11.h = vsub(sl3.ub,sl0.ub)              //
       scale = #0                                   // to set sout0/1_old = 0
    }{
       dvsum11.h = vadd(dvsum11.h,dvsum00.h)        // vertical summation
       ssum0_e = #0                                 // t0 set sout0_old = 0
    }
    .balign 32
/*============================================================================*/
.avgpool3x3s1_d32_rowloop:
    {  sl0 = vmem(iptr0++m0)                        //[1, 0]
       ssum0_o.h = vmpy(ssum0_o.h,scale.h):SRS      //[2, 0]
       ssum1_o.h = vadd(svsum10_o.h,svs1d1_o.h)     //[2, 0]
    }{
       sl1.tmp = vmem(iptr0++m0)                    //[1, 1]
       dvsum01.h = vmpy(sl1.ub,c01.b)               //[1, 1]
       svs1d2_e = valign(svsum11_e,svsum10_e,c64)   //[2, 1]
       sout0_new.b = vshuffe(ssum0_o.b,ssum0_e.b)   //[2, 1]
    }{
       sl2.tmp = vmem(iptr0++m0)                    //[1, 2]
       dvsum01.h += vmpa(dl2l0.ub,c01.b)            //[1, 2]vertical summation
       svs1d2_o = valign(svsum11_o,svsum10_o,c64)   //[2, 2]
       ssum1_e.h = vadd(ssum1_e.h,svs1d2_e.h)       //[2, 2]
    }{
       sout0 = vlalign(sout0_new,sout0_old,out_alig)//[2, 3]
       if p3 vmem(optr0++#1) = sout0.new            //[2, 3]
       ssum1_o.h = vadd(ssum1_o.h,svs1d2_o.h)       //[2, 3]
       dvsum10 = dvsum11                            //[2, 3]
    }{
       sl3.tmp = vmem(iptr0++m1)                    //[1, 4]
       dvsum11.h = vsub(sl3.ub,sl0.ub)              //[1, 4]
       svs0d1_e = valign(svsum01_e,svsum00_e,c32)   //[1, 4]
       sout0_old = sout0_new                        //[2, 4]
    }{
       dvsum11.h = vadd(dvsum11.h,dvsum01.h)        //[1, 5]vertical summation
       svs0d1_o = valign(svsum01_o,svsum00_o,c32)   //[1, 5]
       sout1_old = sout1_new                        //[2, 5]
    }{
       svs0d2_e = valign(svsum01_e,svsum00_e,c64)   //[1, 6]
       ssum0_e.h = vadd(svsum00_e.h,svs0d1_e.h)     //[1, 6]
       ssum1_e.h = vmpy(ssum1_e.h,scale.h):SRS      //[2, 6]
    }{
       svs0d2_o = valign(svsum01_o,svsum00_o,c64)   //[1, 7]
       ssum0_o.h = vadd(svsum00_o.h,svs0d1_o.h)     //[1, 7]
       ssum1_o.h = vmpy(ssum1_o.h,scale.h):SRS      //[2, 7]
    }{
       dsum0.h = vadd(dsum0.h,dvs0d2.h)             //[1, 8]
       svs1d1_e = valign(svsum11_e,svsum10_e,c32)   //[1, 8]
       sout1_new.b = vshuffe(ssum1_o.b,ssum1_e.b)   //[2, 8]
    }{
       svs1d1_o = valign(svsum11_o,svsum10_o,c32)   //[1, 9]
       dvsum00 = dvsum01                            //[1, 9]
       scale = recip9                               //[1, 9]
    }{
       ssum1_e.h = vadd(svsum10_e.h,svs1d1_e.h)     //[1,10]
       ssum0_e.h = vmpy(ssum0_e.h,scale.h):SRS      //[1,10]
       sout1 = vlalign(sout1_new,sout1_old,out_alig)//[2,10]
       if p3 vmem(optr1++#1) = sout1.new            //[2,10]
    }:endloop0
/*============================================================================*/
    {  ssum0_o.h = vmpy(ssum0_o.h,scale.h):SRS      //[2, 0]
       ssum1_o.h = vadd(svsum10_o.h,svs1d1_o.h)     //[2, 0]
       sout1_old = sout1_new                        //[2, 5]
    }{
       svs1d2_e = valign(svsum11_e,svsum10_e,c64)   //[2, 1]
       sout0_new.b = vshuffe(ssum0_o.b,ssum0_e.b)   //[2, 1]
    }{
       svs1d2_o = valign(svsum11_o,svsum10_o,c64)   //[2, 2]
       ssum1_e.h = vadd(ssum1_e.h,svs1d2_e.h)       //[2, 2]
    }{
       sout0 = vlalign(sout0_new,sout0_old,out_alig)//[2, 3]
       if p3 vmem(optr0++#1) = sout0.new            //[2, 3]
       ssum1_o.h = vadd(ssum1_o.h,svs1d2_o.h)       //[2, 3]
    }{
       ssum1_e.h = vmpy(ssum1_e.h,scale.h):SRS      //[2, 6]
       out_lines = add(out_lines,#-1)               //
    }{
       ssum1_o.h = vmpy(ssum1_o.h,scale.h):SRS      //[2, 7]
       pfptr0 = add(pfptr0,in_next_row_2)           //
       pfptr1 = add(pfptr1,in_next_row_2)           //
    }{
       sout1_new.b = vshuffe(ssum1_o.b,ssum1_e.b)   //[2, 8]
       iptr  = add(iptr,in_next_row_2)              //
       optr += asl(out_next_row,#1)                 //
    }{
       sout1 = vlalign(sout1_new,sout1_old,out_alig)//[2,10]
       if p3 vmem(optr1++#1) = sout1.new            //[2,10]
       optr0_iptr0 = combine(optr,iptr)             //
    }:endloop1
/*============================================================================*/
    {  r17:16 = memd(sp+#0)                         //
       sp = add(sp,#8)                              //
       r0 = #0                                      //
       jumpr r31                                    //
    }
.avgpool_slice_hvx_3x3_stride1_end:
/*=============================================================================*/
    .size avgpool_slice_hvx_3x3_stride1, .-avgpool_slice_hvx_3x3_stride1
/*=============================================================================*/

#define OUTPTR r0
#define INPTR r1
#define IN_ROW_STRIDE r2
#define OUT_STRIDE r3
#define OUT__IN_ROW_STRIDE r3:2
#define IN_STRIDE r4
#define ITERS r5
#define LALIGN_RECIP r7:6
#define LALIGN r7
#define RECIP r6
#define RONES r11
#define NEXT_INPTR r12
#define OUTER_NEXT_INPTR r13
#define WINDOW_W_H r15:14
#define WINDOW_W r15
#define WINDOW_H r14

#define LINE00_ACC v1:0
#define LINE00_ACC_H v1
#define LINE00_ACC_L v0
#define LINE04_ACC v3:2
#define LINE04_ACC_H v3
#define LINE04_ACC_L v2
#define RED_ACC v5:4
#define RED_ACC_H v5
#define RED_ACC_L v4
#define OUTER_ACC v7:6
#define OUTER_ACC_H v7
#define OUTER_ACC_L v6

#define PRODUCTH v9:8
#define PRODUCTH_H v9
#define PRODUCTH_L v8
#define PRODUCTL v11:10
#define PRODUCTL_H v11
#define PRODUCTL_L v10

#define OUT v29
#define LASTOUT v28
#define VZERO v30
#define TMP v31

	/* avgpool_hvx_d32(OUTPTR,INPTR,IN_ROW_STRIDE,OUT_STRIDE,IN_STRIDE,ITERS,WINDOW_H,WINDOW_W, RECIP,LALIGN) */
	/* Compute a vector (4x32) of WINDOW_HxWINDOW_W avgpooling, then go forward OUT/IN strides and iterate */
	.global avgpool_hvx_d32
	.type avgpool_hvx_d32,@function
	.p2align 6
avgpool_hvx_d32:
	{
		WINDOW_W_H = memd(r29+#0)
		c7:6 = OUT__IN_ROW_STRIDE
		LALIGN_RECIP = memd(r29+#8)
		VZERO = vxor(VZERO,VZERO)
	}
	{
		LASTOUT = vxor(VZERO,VZERO)
		RONES = ##0x01010101
		loop1(.Lavgpool_hvx_d32_outer,ITERS)
	}
#undef OUT_STRIDE
#undef IN_ROW_STRIDE
#undef OUT__IN_ROW_STRIDE
#undef ITERS
#define SHIFT r5
#define REDUC_AMT r3
	/* Accumulate down height, reduce to 1x4 (x2) */
.Lavgpool_hvx_d32_outer:
	{
		loop0(.LNx4_first_inner,WINDOW_H)
		OUTER_NEXT_INPTR = add(INPTR,IN_STRIDE)
		NEXT_INPTR = add(INPTR,#128)
		REDUC_AMT = add(WINDOW_W,#-1)
	}
	{
		LINE00_ACC_L = VZERO
		LINE00_ACC_H = VZERO
		LINE04_ACC_L = VZERO
		LINE04_ACC_H = VZERO
	}
	{
		OUTER_ACC_H = VZERO
		OUTER_ACC_L = VZERO
	}
.LNx4_first_inner:
	{
		TMP.tmp = vmem(INPTR++M0)
		LINE00_ACC.h += vmpy(TMP.ub,RONES.b)
	}:endloop0
.Lavgpool_hvx_d32_reduc:
	{
		loop0(.LNx4_main_inner,WINDOW_H)
		RECIP = combine(RECIP.L,RECIP.L)
		INPTR = NEXT_INPTR
		NEXT_INPTR = add(NEXT_INPTR,#128)
	}
.LNx4_main_inner:
	{
		TMP.tmp = vmem(INPTR++M0)
		LINE04_ACC.h += vmpy(TMP.ub,RONES.b)
	}:endloop0
	/* Reduce right up to 3 times */
	{
		RED_ACC = LINE00_ACC
		SHIFT = #32
		TMP = VZERO
		p0 = cmp.gt(REDUC_AMT,#0)
		if (!p0.new) jump:nt .Lred_done
	}

	{
		TMP = valign(LINE04_ACC_H,LINE00_ACC_H,SHIFT)
		RED_ACC_L.h = vadd(RED_ACC_L.h,TMP.h)
	}
	{
		RED_ACC_H.h = vadd(RED_ACC_H.h,TMP.h)
		TMP = valign(LINE04_ACC_L,LINE00_ACC_L,SHIFT)
		SHIFT = add(SHIFT,#32)
		p0 = cmp.gt(REDUC_AMT,#1)
		if (!p0.new) jump:nt .Lred_done
	}

	{
		TMP = valign(LINE04_ACC_H,LINE00_ACC_H,SHIFT)
		RED_ACC_L.h = vadd(RED_ACC_L.h,TMP.h)
	}
	{
		RED_ACC_H.h = vadd(RED_ACC_H.h,TMP.h)
		TMP = valign(LINE04_ACC_L,LINE00_ACC_L,SHIFT)
		SHIFT = add(SHIFT,#32)
		p0 = cmp.gt(REDUC_AMT,#2)
		if (!p0.new) jump:nt .Lred_done
	}

	{
		TMP = valign(LINE04_ACC_H,LINE00_ACC_H,SHIFT)
		RED_ACC_L.h = vadd(RED_ACC_L.h,TMP.h)
	}
	{
		RED_ACC_H.h = vadd(RED_ACC_H.h,TMP.h)
		TMP = valign(LINE04_ACC_L,LINE00_ACC_L,SHIFT)
		SHIFT = add(SHIFT,#32)
	}

.Lred_done:
	{
		RED_ACC_L.h = vadd(RED_ACC_L.h,TMP.h)
		LINE00_ACC = LINE04_ACC
		REDUC_AMT = add(REDUC_AMT,#-4)
	}
	{
		p0 = cmp.gt(REDUC_AMT,#0)
		if (p0.new) jump:t .Lavgpool_hvx_d32_reduc
		OUTER_ACC.h = vadd(OUTER_ACC.h,RED_ACC.h)
	}
	/* Multiply OUTER_ACC_HL by reciprocal and pack back for output */

	PRODUCTH.uw = vmpy(OUTER_ACC_H.uh,RECIP.uh)
	PRODUCTL.uw = vmpy(OUTER_ACC_L.uh,RECIP.uh)
	OUTER_ACC_H.h = vshuffo(PRODUCTH_H.h,PRODUCTH_L.h)
	OUTER_ACC_L.h = vshuffo(PRODUCTL_H.h,PRODUCTL_L.h)
	OUT.ub = vsat(OUTER_ACC_H.h,OUTER_ACC_L.h)
	{
		TMP = vlalign(OUT,LASTOUT,LALIGN)
		vmem(OUTPTR++M1) = TMP.new
		LASTOUT = OUT
		INPTR = OUTER_NEXT_INPTR
	}:endloop1
	{
		r0 = #0
		jumpr r31
	}
	.size avgpool_hvx_d32,.-avgpool_hvx_d32

#undef OUTPTR
#undef INPTR
#undef R32
#undef RECIP
#undef LALIGN
#undef LALIGN_RECIP
#undef WINDOW_H
#undef WINDOW_W
#undef WINDOW_W_H
#undef IN_STRIDE
#undef SHIFT
#undef REDUC_AMT

#undef LINE00_ACC
#undef LINE00_ACC_H
#undef LINE00_ACC_L
#undef LINE04_ACC
#undef LINE04_ACC_H
#undef LINE04_ACC_L
#undef RED_ACC
#undef RED_ACC_H
#undef RED_ACC_L
#undef OUTER_ACC
#undef OUTER_ACC_H
#undef OUTER_ACC_L

#undef PRODUCTH
#undef PRODUCTH_H
#undef PRODUCTH_L
#undef PRODUCTL
#undef PRODUCTL_H
#undef PRODUCTL_L

#undef OUT
#undef LASTOUT
#undef VZERO
#undef TMP

#define DST r0
#define SRC0 r1
#define NEXT_ROW r2
#define SRC1 r3
#define COUNT r4
#define R32 r5


	/*
	 * avgpool_zap_row(dstptr,srcptr,next_row_in_bytes)
	 */
	.global avgpool_zap_row
	.type avgpool_zap_row,@function
	.p2align 6
avgpool_zap_row:
	{
		SRC1 = add(SRC0,NEXT_ROW)
		COUNT = lsr(NEXT_ROW,#7)
	}
	{
		loop0(.Lzap_row_loop,COUNT)
		R32 = #32
	}
.Lzap_row_loop:
	v0 = vmem(SRC0++#1)
	{
		v1.tmp = vmem(SRC1++#1)
		v0.ub = vavg(v0.ub,v1.ub):rnd
		vmem(DST++#1) = v0.new
	}:endloop0
	/* Write one more set of 32 bytes off the end, just in case */
	{
		v0 = vmem(SRC0+#0)
		q0 = vsetq(R32)
	}
	{
		v1.tmp = vmem(SRC1+#0)
		v0.ub = vavg(v0.ub,v1.ub):rnd
	}
	{
		if (q0) vmem(DST+#0) = v0
	}
	{
		r0 = #0
		jumpr r31
	}
	.size avgpool_zap_row,.-avgpool_zap_row

#undef DST
#undef SRC0
#undef NEXT_ROW
#undef SRC1
#undef COUNT



#define PTR r0
#define HEIGHT r1
#define WIDTH r2
#define LEFT_PAD r3
#define NEXT_ROW r4
#define LEFT_PTR r5
#define R_64 r7
#define R_32 r6
#define R_64_32 r7:6
#define ITERS r12
#define R_ONE r13

#define LEFT_DATA v0
#define RIGHT_DATA v1
#define LEFT_32 v2
#define RIGHT_32 v3
#define LEFT_TMP0 v10
#define LEFT_TMP1 v11
#define RIGHT_TMP0 v20
#define RIGHT_TMP1 v21

#define LEFT_Q_MASK q0
#define RIGHT_Q_MASK q1
#define V_ONE v31
#define V_ZERO v30

	/*
 	 * avgpool_zap_lr(ptr,height,width,left_pad,next_row_in_bytes)
	 * There is no if (Q) vmemu, so we need to grab both aligned chunks and store aligned
	 */
	.global avgpool_zap_lr
	.type avgpool_zap_lr,@function
	.p2align 6
avgpool_zap_lr:
	{
		R_64_32 = combine(#64,#32)
		WIDTH = add(WIDTH,LEFT_PAD)
		LEFT_PAD = add(LEFT_PAD,#-1)
		R_ONE = #1
	}
	{
		V_ONE = vsplat(R_ONE)
		LEFT_PTR = addasl(PTR,LEFT_PAD,#5)
		V_ZERO = vxor(V_ZERO,V_ZERO)
	}
	{
		LEFT_DATA = vlalign(V_ONE,V_ZERO,R_32)
#define RIGHT_PTR r0
		RIGHT_PTR = addasl(PTR,WIDTH,#5)
#undef PTR
	}
	{
		RIGHT_DATA = vlalign(LEFT_DATA,V_ONE,RIGHT_PTR)
		//ITERS = lsr(NEXT_ROW,#7)
	}
	{
		LEFT_DATA = vlalign(LEFT_DATA,V_ONE,LEFT_PTR)
		M0 = NEXT_ROW
	}
	{
		LEFT_Q_MASK = vcmp.eq(LEFT_DATA.w,V_ZERO.w)
		RIGHT_Q_MASK = vcmp.eq(RIGHT_DATA.w,V_ZERO.w)
		loop0(.Lzap_lr_loop,HEIGHT)
	}
.Lzap_lr_loop:
	{
		LEFT_32 = vmem(LEFT_PTR+#1)
	}
	{
		RIGHT_32 = vmem(RIGHT_PTR+#-1)
	}
	{
		LEFT_DATA.cur = vmem(LEFT_PTR+#0)
		LEFT_TMP0 = valign(LEFT_32,LEFT_DATA,R_32)
	}
	{
		RIGHT_DATA.cur = vmem(RIGHT_PTR+#0)
		RIGHT_TMP0 = vlalign(RIGHT_DATA,RIGHT_32,R_32)
	}
	{
		LEFT_TMP1 = valign(LEFT_32,LEFT_DATA,R_64)
	}
	{
		RIGHT_TMP1 = vlalign(RIGHT_DATA,RIGHT_32,R_64)
		LEFT_TMP0.ub = vavg(LEFT_TMP0.ub,LEFT_TMP1.ub):rnd
	}
	{
		if (LEFT_Q_MASK) vmem(LEFT_PTR++M0) = LEFT_TMP0
		RIGHT_TMP0.ub = vavg(RIGHT_TMP0.ub,RIGHT_TMP1.ub):rnd
	}
	{
		if (RIGHT_Q_MASK) vmem(RIGHT_PTR++M0) = RIGHT_TMP0
	}:endloop0
	{
		jumpr r31
		r0 = #0
	}
	.size avgpool_zap_lr,.-avgpool_zap_lr

