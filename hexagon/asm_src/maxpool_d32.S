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

        .file "maxpool_d32.S"
/*======================================================================*/
/*  FUNCTIONS      : maxpool_slice_hvx_3x3_stride1                      */
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
#define iptr0           r8
#define optr0           r9
#define optr0_iptr0     r9:8
#define iptr1           r10
#define optr1           r11
#define pfptr0          r12
#define pfptr1          r13
#define l2param_l       r14
#define l2param_h       r15
#define l2param         r15:14
#define out_lines_d2    r28
#define out_alig        out_lines
#define offset          optr1
/*=============================================================================*/
#define sl0             v0
#define sl1             v1
#define smaxl012_0      v2
#define smaxl012_1      v3
#define smaxl123_0      v4
#define smaxl123_1      v5
#define sout0_new       v6
#define sout0_old       v7
#define sout1_new       v8
#define sout1_old       v9
#define smaxl012d1      v10
#define smaxl123d1      v11
#define sot0            v12
#define sot1            v13
#define sout0           v14
#define sout1           v15
#define smaxl12         v16
#define sl2             sl0
#define sl3             sl0
#define smaxl012d2      smaxl012d1
#define smaxl123d2      smaxl123d1
/*=============================================================================*/

    .text
    .global maxpool_slice_hvx_3x3_stride1
    .balign 32
    .type  maxpool_slice_hvx_3x3_stride1, @function
maxpool_slice_hvx_3x3_stride1:
    {  p0 = cmp.gt(out_lines,#0)                    //
       if !p0.new r0 = #0                           //
       if !p0.new jumpr:nt r31                      //
    }{
       out_lines_d2 = asrrnd(out_lines,#1)          //
       m0 = in_next_row                             //
       offset = #128                                //
       c64_c32 = combine(#64,#32)                   //
    }{
       out_alig = memw(sp+#0)                       //
       offset -= asl(in_next_row,#1)                //   
       loop1(.maxpool3x3s1_d32_outloop,out_lines_d2)//
    }{
       m1 = offset                                  //
       l2param_l = add(out_vectors,#1)              //
       l2param_h = #128                             //
       pfptr0 = addasl(iptr,in_next_row,#2)         //
    }{
       pfptr1 = add(pfptr0,in_next_row)             //
       l2param_l = combine(l2param_h.l,l2param_l.l) // l2param=(0|128|128|out_vectors+1)
       iptr1 = add(iptr,in_next_row)                //
    }
    .balign 32
/*============================================================================*/
.maxpool3x3s1_d32_outloop:
    {  optr0_iptr0 = combine(optr,iptr)             //
       p0 = cmp.eq(out_lines_d2,#1)                 // if (last iteration)
       if p0.new l2param_l = #0                     // then cancel l2fetch
       sl1 = vmem(iptr1++m0)                        //
    }{
       l2fetch(pfptr0,l2param)                      //
       optr1 = add(optr0,out_next_row)              //
    }{
       l2fetch(pfptr1,l2param)                      //
    }{
       sl2.tmp = vmem(iptr1++m0)                    //
       smaxl12.ub = vmax(sl1.ub,sl2.ub)             //
       p3=sp2loop0(.maxpool3x3s1_d32_rowloop,out_vectors)//
    }{
       sl0.tmp = vmem(iptr0++#1)                    //
       smaxl012_1.ub = vmax(sl0.ub,smaxl12.ub)      //
       sout0_old = #0                               //
       sout1_old = #0                               //
    }{
       sl3.tmp = vmem(iptr1++m1)                    //
       smaxl123_1.ub = vmax(smaxl12.ub,sl3.ub)      //
    }
    .balign 32
/*============================================================================*/
.maxpool3x3s1_d32_rowloop:
    {  sl1 = vmem(iptr1++m0)                        //[1, 0]
       sot1.ub = vmax(smaxl123_0.ub,smaxl123d1.ub)  //[2, 6]
       sout0 = vlalign(sout0_new,sout0_old,out_alig)//[3,11]
       if p3 sout0_old = sout0_new                  //[3,11]
    }{
       sl2.tmp = vmem(iptr1++m0)                    //[1, 1]
       smaxl12.ub = vmax(sl1.ub,sl2.ub)             //[1, 1]
       sout1 = vlalign(sout1_new,sout1_old,out_alig)//[3,12]
       if p3 sout1_old = sout1_new                  //[3,12]
    }{
       smaxl012_0 = smaxl012_1                      //[1, 2]
       sl0.tmp = vmem(iptr0++#1)                    //[1, 2]
       smaxl012_1.ub = vmax(sl0.ub,smaxl12.ub)      //[1, 2]
       smaxl012d2=valign(smaxl012_1,smaxl012_0,c64) //[2, 7]
    }{
       smaxl123_0 = smaxl123_1                      //[1, 3]
       sl3.tmp = vmem(iptr1++m1)                    //[1, 3]
       smaxl123_1.ub = vmax(smaxl12.ub,sl3.ub)      //[1, 3]
       smaxl123d2=valign(smaxl123_1,smaxl123_0,c64) //[2, 8]
    }{
       smaxl012d1=valign(smaxl012_1,smaxl012_0,c32) //[1, 4]
       sout0_new.ub = vmax(sot0.ub,smaxl012d2.ub)   //[2, 9]
       if p3 vmem(optr0++#1) = sout0                //[3,13]
    }{
       smaxl123d1=valign(smaxl123_1,smaxl123_0,c32) //[1, 5]
       sot0.ub = vmax(smaxl012_0.ub,smaxl012d1.ub)  //[1, 5]
       sout1_new.ub = vmax(sot1.ub,smaxl123d2.ub)   //[2,10]
       if p3 vmem(optr1++#1) = sout1                //[3,14]
    }:endloop0
/*============================================================================*/
    {  sot1.ub = vmax(smaxl123_0.ub,smaxl123d1.ub)  //[2, 6]
       sout0 = vlalign(sout0_new,sout0_old,out_alig)//[3,11]
       if p3 sout0_old = sout0_new                  //[3,11]
       if p3 vmem(optr0++#1) = sout0.new            //[3,13]
    }{
       sout1 = vlalign(sout1_new,sout1_old,out_alig)//[3,12]
       if p3 sout1_old = sout1_new                  //[3,12]
       if p3 vmem(optr1++#1) = sout1.new            //[3,14]
    }{
       smaxl012d2=valign(smaxl012_1,smaxl012_0,c64) //[2, 7]
       iptr += asl(in_next_row,#1)                  //
       optr += asl(out_next_row,#1)                 //
    }{
       smaxl123d2=valign(smaxl123_1,smaxl123_0,c64) //[2, 8]
       sout0_new.ub = vmax(sot0.ub,smaxl012d2.ub)   //[2, 9]
    }{
       sout1_new.ub = vmax(sot1.ub,smaxl123d2.ub)   //[2,10]
       pfptr0 += asl(in_next_row,#1)                //
       pfptr1 += asl(in_next_row,#1)                //
    }{
       sout0 = vlalign(sout0_new,sout0_old,out_alig)//[3,11]
       vmem(optr0+#0) = sout0.new                   //[3,13]
       out_lines_d2 = add(out_lines_d2,#-1)         //
       iptr1 = add(iptr,in_next_row)                //
    }{
       sout1 = vlalign(sout1_new,sout1_old,out_alig)//[3,12]
       vmem(optr1+#0) = sout1.new                   //[3,14]
    }:endloop1
/*============================================================================*/
    {  r0 = #0                                      //
       jumpr r31                                    //
    }
.maxpool_slice_hvx_3x3_stride1_end:
/*=============================================================================*/
    .size maxpool_slice_hvx_3x3_stride1, .-maxpool_slice_hvx_3x3_stride1
/*=============================================================================*/

#if 0 //////////////////////////// >>>> MOVED TO maxpool_3x3s2_d32.S >>>>>>>>>>>>
/*
 * Redo these.  EJP did horizontal max and then vertical.  That's dumb.  
 * We should do vertical first and then horizontal since the horizontal is more expensive.
 */


#define IN00 v0
#define IN04 v1
#define IN10 v2
#define IN14 v3
#define IN20 v4
#define IN24 v5
#define IN30 v6
#define IN34 v7
#define IN40 v8
#define IN44 v9
#define IN50 v10
#define IN54 v11
#define TMP0 v12
#define TMP1 v13
#define TMP2 v14
#define TMP3 v15
#define TMP4 v16
#define TMP5 v17
#define ACC0 v18
#define ACC1 v19
#define ACC2 v20
#define ACC3 v21
#define ACC4 v22
#define ACC5 v23
#define OLDOUT0 v24
#define OLDOUT1 v25
#define OLDOUT2 v26
#define OLDOUT3 v27
#define OUT0 v28
#define OUT1 v29
#define OUT2 v30
#define OUT3 v31

#define OUTPTR r0
#define INPTR r1
#define IN_OUTPTR r1:0
#define IN_NEXT_ROW r2
#define OUT_NEXT_ROW r3
#define OUT_VECTORS_WIDE r4
#define OUT_LINES_AND_LALIGN r5
#define R32 r6
#define R64 r7
#define R64_32 r7:6
#define NEXT_OUTPTR r8
#define NEXT_INPTR r9
#define NEXT_IN_OUTPTR r9:8
#define OUTER_NEXT_OUTPTR r10
#define OUTER_NEXT_INPTR r11
#define OUTER_NEXT_IN_OUTPTR r11:10
#define RNEG32 r15


/* NOTE CAREFULLY: alias INx8 to INx0 for stride2 */
#define IN08 IN00
#define IN18 IN10
#define IN28 IN20
#define IN38 IN30
#define IN48 IN40
#define P_ODD p3

	.text
	.global maxpool_slice_hvx_3x3_stride2
	.type maxpool_slice_hvx_3x3_stride2,@function
maxpool_slice_hvx_3x3_stride2:
	{
		OUT_LINES_AND_LALIGN = add(OUT_LINES_AND_LALIGN,#1)
		M0 = IN_NEXT_ROW
		R64_32 = combine(#64,#32)
	}
	{
		M1 = OUT_NEXT_ROW
		OUT_LINES_AND_LALIGN = lsr(OUT_LINES_AND_LALIGN,#1)
		OLDOUT0 = vxor(OLDOUT0,OLDOUT0)
		OLDOUT1 = vxor(OLDOUT1,OLDOUT1)
	}
	{
		RNEG32 = #-32
		loop1(.Louter_3x3s2,OUT_LINES_AND_LALIGN)
		OUT_LINES_AND_LALIGN=memw(sp+#0)
	}
	{
		P_ODD = tstbit(OUT_LINES_AND_LALIGN,#5)
	}
#if 0 // not sure about this for now...
	OUT_LINES_AND_LALIGN = clrbit(OUT_LINES_AND_LALIGN,#5)	// clear odd bit
	OUT_LINES_AND_LALIGN = lsr(OUT_LINES_AND_LALIGN,#1)	// shift right by 1
	OUT_LINES_AND_LALIGN = add(OUT_LINES_AND_LALIGN,#64)	// Add 64 to handle 2x reduction
#endif
	.falign
.Louter_3x3s2:
	{
		OUTER_NEXT_INPTR = addasl(INPTR,IN_NEXT_ROW,#2)
		OUTER_NEXT_OUTPTR = addasl(OUTPTR,OUT_NEXT_ROW,#1)
		IN00 = vmem(INPTR++M0)
		NEXT_INPTR = add(INPTR,#128)
	}
	IN10 = vmem(INPTR++M0)
	IN20 = vmem(INPTR++M0)
	{
		IN30 = vmem(INPTR++M0)
		loop0(.Linner_3x3s2,OUT_VECTORS_WIDE)
	}
	{
		IN40 = vmem(INPTR)
		INPTR = NEXT_INPTR
	}
.Linner_3x3s2:
	{
		NEXT_INPTR = add(INPTR,#128)
		IN04.cur = vmem(INPTR++M0)
		TMP0 = valign(IN04,IN00,R32)
	}
	{
		IN14.cur = vmem(INPTR++M0)
		TMP1 = valign(IN14,IN10,R32)
		ACC0.ub = vmax(IN00.ub,TMP0.ub)
	}
	{
		IN24.cur = vmem(INPTR++M0)
		TMP2 = valign(IN24,IN20,R32)
		ACC1.ub = vmax(IN10.ub,TMP1.ub)
	}
	{
		IN34.cur = vmem(INPTR++M0)
		TMP3 = valign(IN34,IN30,R32)
		ACC2.ub = vmax(IN20.ub,TMP2.ub)
	}
	{
		IN44.cur = vmem(INPTR+#0)
		TMP4 = valign(IN44,IN40,R32)
		ACC3.ub = vmax(IN30.ub,TMP3.ub)
		INPTR = NEXT_INPTR
	}
	{
		ACC4.ub = vmax(IN40.ub,TMP4.ub)
		TMP0 = valign(IN04,IN00,R64)
		NEXT_INPTR = add(INPTR,#128)
	}
	{
		TMP1 = valign(IN14,IN10,R64)
		ACC0.ub = vmax(ACC0.ub,TMP0.ub)
	}
	{
		TMP2 = valign(IN24,IN20,R64)
		ACC1.ub = vmax(ACC1.ub,TMP1.ub)
	}
	{
		TMP3 = valign(IN34,IN30,R64)
		ACC2.ub = vmax(ACC2.ub,TMP2.ub)
	}
	{
		TMP4 = valign(IN44,IN40,R64)
		ACC3.ub = vmax(ACC3.ub,TMP3.ub)
		ACC1.ub = vmax(ACC1.ub,ACC2.ub)
	}
	{
		ACC4.ub = vmax(ACC4.ub,TMP4.ub)
		OUT0.ub = vmax(ACC0.ub,ACC1.ub)
		IN08.cur = vmem(INPTR++M0)
		TMP0 = valign(IN08,IN04,R32)
	}
	{
		ACC3.ub = vmax(ACC3.ub,ACC4.ub)
		IN18.cur = vmem(INPTR++M0)
		TMP1 = valign(IN18,IN14,R32)
		ACC0.ub = vmax(IN04.ub,TMP0.ub)
	}
	{
		OUT1.ub = vmax(ACC2.ub,ACC3.ub)
		IN28.cur = vmem(INPTR++M0)
		TMP2 = valign(IN28,IN24,R32)
		ACC1.ub = vmax(IN14.ub,TMP1.ub)
	}
	{
		IN38.cur = vmem(INPTR++M0)
		TMP3 = valign(IN38,IN34,R32)
		ACC2.ub = vmax(IN24.ub,TMP2.ub)
	}
	{
		IN48.cur = vmem(INPTR++M0)
		TMP4 = valign(IN48,IN44,R32)
		ACC3.ub = vmax(IN34.ub,TMP3.ub)
	}
	{
		ACC4.ub = vmax(IN44.ub,TMP4.ub)
		TMP0 = valign(IN08,IN04,R64)
	}
	{
		TMP1 = valign(IN18,IN14,R64)
		ACC0.ub = vmax(ACC0.ub,TMP0.ub)
	}
	{
		TMP2 = valign(IN28,IN24,R64)
		ACC1.ub = vmax(ACC1.ub,TMP1.ub)
	}
	{
		TMP3 = valign(IN38,IN34,R64)
		ACC2.ub = vmax(ACC2.ub,TMP2.ub)
	}
	{
		TMP4 = valign(IN48,IN44,R64)
		ACC3.ub = vmax(ACC3.ub,TMP3.ub)
	}
	{
		ACC4.ub = vmax(ACC4.ub,TMP4.ub)
		ACC1.ub = vmax(ACC1.ub,ACC2.ub)
	}
	/* OUT0 and OUT2 have the upper output line */
	/* OUT1 and OUT3 have the lower output line */
	/* BUT we computed all the values across, we only need half of them. */
	/* Which half to take? With stride 2, depends on required_w_before! */
	/* So we can lalign first x4 and then deal, or we can deal first, select, and align */
	/* We choose to deal first, then select, and then align */
	{
		ACC3.ub = vmax(ACC3.ub,ACC4.ub)
		OUT2.ub = vmax(ACC0.ub,ACC1.ub)
	}
	{
		OUT3.ub = vmax(ACC2.ub,ACC3.ub)
		vdeal(OUT2,OUT0,RNEG32)
	}
	{
		vdeal(OUT3,OUT1,RNEG32)
		NEXT_OUTPTR = add(OUTPTR,#128)
		if (P_ODD) OUT0 = OUT2
	}
	/* Now we need to align these and store. */
	{
		if (P_ODD) OUT1 = OUT3
	}
	{
		TMP0 = VLALIGN(OUT0,OLDOUT0,OUT_LINES_AND_LALIGN)
		//TMP0 = OUT0
		VMEM(OUTPTR++M1) = TMP0.new
		OLDOUT0 = OUT0
	}
	{
		TMP1 = VLALIGN(OUT1,OLDOUT1,OUT_LINES_AND_LALIGN)
		//TMP1 = OUT1
		OLDOUT1 = OUT1
		VMEM(OUTPTR++M1) = TMP1.new
	}
	/* NOTE that IN08 and IN00 are aliased, so we're ready to go for the next iteration */
	{
		IN_OUTPTR = NEXT_IN_OUTPTR
	}:endloop0
	{
		IN_OUTPTR = OUTER_NEXT_IN_OUTPTR
	}:endloop1
	{
		r0 = #0
		jumpr r31
	}
	.size maxpool_slice_hvx_3x3_stride2,.-maxpool_slice_hvx_3x3_stride2

#endif //////////////////////////// <<<<< MOVED TO maxpool_3x3s2_d32.S <<<<<<<<<<<


#if 1
/*======================================================================*/
/*  FUNCTIONS      : maxpool_slice_hvx_2x2_stride2                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Get maximun on 2x2 with d32 format and stride_width=2*/
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
//#define optr            r0  
//#define iptr            r1 
//#define in_next_row     r2 
//#define out_next_row    r3
//#define out_vectors     r4
//#define out_lines       r5
/*=============================================================================*/
#define cn32            r6
#define adj             r7
//#define iptr0           r8
//#define optr0           r9
//#define optr0_iptr0     r9:8
//#define iptr1           r10
#define pfptr           r11
#define c128            r12
#define lines_out       r28
//#define l2param_l       r14
//#define l2param_h       r15
//#define l2param         r15:14
//#define out_alig        out_lines
/*=============================================================================*/
#define sl0x0           v0
#define sl0x1           v1
#define sl1x0           v2
#define sl1x1           v2
#define smaxv0          v3
#define smaxv1          v4
#define smaxv2          v5
#define smaxT_e         v6
#define smaxT_o         v7
#define dmaxT_oe        v7:6
#define sout_old        v8
#define sout_new        v9
#define sout            v10
#define smaxT0          smaxT_e
#define smaxT1          smaxT_o
/*=============================================================================*/
    .text
    .global maxpool_slice_hvx_2x2_stride2
    .balign 32
    .type  maxpool_slice_hvx_2x2_stride2, @function
maxpool_slice_hvx_2x2_stride2:
    {  p0 = cmp.gt(out_lines,#0)                    //
       if !p0.new r0 = #0                           //
       if !p0.new jumpr:nt r31                      //
    }{
       lines_out = out_lines                        //
       out_alig = memw(sp+#0)                       //
       l2param_l = asl(out_vectors,#1)              //
       cn32 = #-32                                  //
    }{
       p2 = tstbit(out_alig,#5)                     // ??
       if p2.new l2param_l = add(l2param_l,#1)      // need to read extra 128b
       l2param_h = #128                             //
       loop1(.maxpool2x2s2_d32_outloop,lines_out)   //
    }{
       adj = mux(p2,#96,#0)                         //
       c128 = #128                                  //
       pfptr = addasl(iptr,in_next_row,#2)          //
       l2param_l = combine(l2param_h.l,l2param_l.l) // l2param=(0|128|128|2*out_vectors+0/1)
    }

    .balign 32
/*============================================================================*/
.maxpool2x2s2_d32_outloop:
    {  optr0_iptr0 = combine(optr,iptr)             //
       p0 = cmp.gt(lines_out,#2)                    // if (last 2 iterations)
       if !p0.new l2param_l = #0                    // then cancel l2fetch
       p3=sp3loop0(.maxpool2x2s2_d32_rowloop,out_vectors)
                                                    //
    }{
       l2fetch(pfptr,l2param)                       //
       pfptr = add(pfptr,in_next_row)               //
    }{
       l2fetch(pfptr,l2param)                       //
       pfptr = add(pfptr,in_next_row)               //
    }{
       iptr1 = add(iptr,in_next_row)                //
       lines_out = add(lines_out,#-1)               //
       sout_old = #0                                //
       if !p2 jump .maxpool2x2s2_d32_rowloop        //
    }{
       sl0x0 = vmem(iptr0++#1)                      //
       nop                                          //
    }{
       sl1x0.tmp = vmem(iptr1++#1)                  //
       smaxv2.ub = vmax(sl0x0.ub,sl1x0.ub)          //
    }

    .balign 32
/*============================================================================*/
.maxpool2x2s2_d32_rowloop:
    {  sl0x0 = vmem(iptr0++#1)                      //[1, 0]
       smaxT0 = vlalign(smaxv1,smaxv0,adj)          //[2, 4]
       sout_new.ub = vmax(smaxT_e.ub,smaxT_o.ub)    //[3, 7]
       if p3 sout_old = sout_new                    //[4, 9]
    }{
       sl0x1 = vmem(iptr0++#1)                      //[1, 1]
       smaxv0 = smaxv2                              //[1, 1]
       smaxT1 = vlalign(smaxv2,smaxv1,adj)          //[2, 5]
       if p3 optr0 = add(optr0,c128)                //[4,10]
    }{
       sl1x0.tmp = vmem(iptr1++#1)                  //[1, 2]
       smaxv1.ub = vmax(sl0x0.ub,sl1x0.ub)          //[1, 2]
       sout = vlalign(sout_new,sout_old,out_alig)   //[3, 8]
       vmem(optr0+#0) = sout.new                    //[3, 8]for V65: use if P3 vmem(++#1)
    }{
       sl1x1.tmp = vmem(iptr1++#1)                  //[1, 3]
       smaxv2.ub = vmax(sl0x1.ub,sl1x1.ub)          //[1, 3]
       dmaxT_oe = vdeal(smaxT1,smaxT0,cn32)         //[2, 6]
    }:endloop0
/*============================================================================*/
    {  smaxT0 = vlalign(smaxv1,smaxv0,adj)          //[2, 4]
       sout_new.ub = vmax(smaxT_e.ub,smaxT_o.ub)    //[3, 7]
       if p3 sout_old = sout_new                    //[4, 9]
    }{
       smaxT1 = vlalign(smaxv2,smaxv1,adj)          //[2, 5]
       if p3 optr0 = add(optr0,c128)                //[4,10]
       p3 = cmp.gt(out_vectors,#1)                  //
    }{
       sout = vlalign(sout_new,sout_old,out_alig)   //[3, 8]
       if p3 vmem(optr0++#1) = sout.new             //[3, 8]
    }{
       dmaxT_oe = vdeal(smaxT1,smaxT0,cn32)         //[2, 6]
       if p3 sout_old = sout_new                    //[4, 9]
    }{
       sout_new.ub = vmax(smaxT_e.ub,smaxT_o.ub)    //[3, 7]
       iptr += asl(in_next_row,#1)                  //
       optr  = add(optr,out_next_row)               //
    }{
       sout = vlalign(sout_new,sout_old,out_alig)   //[3, 8]
       vmem(optr0+#0) = sout.new                    //[3, 8]
    }:endloop1
/*============================================================================*/
    {  r0 = #0                                      //
       jumpr r31                                    //
    }
.maxpool_slice_hvx_2x2_stride2_end:
/*=============================================================================*/
    .size maxpool_slice_hvx_2x2_stride2, .-maxpool_slice_hvx_2x2_stride2
/*=============================================================================*/
#else
	.text
	.global maxpool_slice_hvx_2x2_stride2
	.type maxpool_slice_hvx_2x2_stride2,@function
maxpool_slice_hvx_2x2_stride2:
	{
		OUT_LINES_AND_LALIGN = add(OUT_LINES_AND_LALIGN,#1)
		M0 = IN_NEXT_ROW
		R64_32 = combine(#64,#32)
	}
	{
		M1 = OUT_NEXT_ROW
		OUT_LINES_AND_LALIGN = lsr(OUT_LINES_AND_LALIGN,#1)
		OLDOUT0 = vxor(OLDOUT0,OLDOUT0)
		OLDOUT1 = vxor(OLDOUT1,OLDOUT1)
	}
	{
		RNEG32 = #-32
		loop1(.Louter_2x2s2,OUT_LINES_AND_LALIGN)
		OUT_LINES_AND_LALIGN=memw(sp+#0)
	}
	{
		P_ODD = tstbit(OUT_LINES_AND_LALIGN,#5)
	}
#if 0 // not sure about this for now...
	OUT_LINES_AND_LALIGN = clrbit(OUT_LINES_AND_LALIGN,#5)	// clear odd bit
	OUT_LINES_AND_LALIGN = lsr(OUT_LINES_AND_LALIGN,#1)	// shift right by 1
	OUT_LINES_AND_LALIGN = add(OUT_LINES_AND_LALIGN,#64)	// Add 64 to handle 2x reduction
#endif
	.falign
.Louter_2x2s2:
	{
		OUTER_NEXT_INPTR = addasl(INPTR,IN_NEXT_ROW,#2)
		OUTER_NEXT_OUTPTR = addasl(OUTPTR,OUT_NEXT_ROW,#1)
		IN00 = vmem(INPTR++M0)
		NEXT_INPTR = add(INPTR,#128)
	}
	IN10 = vmem(INPTR++M0)
	IN20 = vmem(INPTR++M0)
	{
		IN30 = vmem(INPTR+#0)
		loop0(.Linner_2x2s2,OUT_VECTORS_WIDE)
		INPTR = NEXT_INPTR
	}
.Linner_2x2s2:
	{
		NEXT_INPTR = add(INPTR,#128)
		IN04.cur = vmem(INPTR++M0)
		TMP0 = valign(IN04,IN00,R32)
	}
	{
		IN14.cur = vmem(INPTR++M0)
		TMP1 = valign(IN14,IN10,R32)
		ACC0.ub = vmax(IN00.ub,TMP0.ub)
	}
	{
		IN24.cur = vmem(INPTR++M0)
		TMP2 = valign(IN24,IN20,R32)
		ACC1.ub = vmax(IN10.ub,TMP1.ub)
	}
	{
		IN34.cur = vmem(INPTR+#0)
		TMP3 = valign(IN34,IN30,R32)
		ACC2.ub = vmax(IN20.ub,TMP2.ub)
		INPTR = NEXT_INPTR
	}
	{
		ACC3.ub = vmax(IN30.ub,TMP3.ub)
		NEXT_INPTR = add(INPTR,#128)
	}
	{
		OUT0.ub = vmax(ACC0.ub,ACC1.ub)
		OUT1.ub = vmax(ACC2.ub,ACC3.ub)
		IN08.cur = vmem(INPTR++M0)
		TMP0 = valign(IN08,IN04,R32)
	}
	{
		IN18.cur = vmem(INPTR++M0)
		TMP1 = valign(IN18,IN14,R32)
		ACC0.ub = vmax(IN04.ub,TMP0.ub)
	}
	{
		IN28.cur = vmem(INPTR++M0)
		TMP2 = valign(IN28,IN24,R32)
		ACC1.ub = vmax(IN14.ub,TMP1.ub)
	}
	{
		IN38.cur = vmem(INPTR++M0)
		TMP3 = valign(IN38,IN34,R32)
		ACC2.ub = vmax(IN24.ub,TMP2.ub)
		OUT2.ub = vmax(ACC0.ub,ACC1.ub)
	}
	{
		ACC3.ub = vmax(IN34.ub,TMP3.ub)
	}
	/* OUT0 and OUT2 have the upper output line */
	/* OUT1 and OUT3 have the lower output line */
	/* BUT we computed all the values across, we only need half of them. */
	/* Which half to take? With stride 2, depends on required_w_before! */
	/* So we can lalign first x4 and then deal, or we can deal first, select, and align */
	/* We choose to deal first, then select, and then align */
	{
		OUT3.ub = vmax(ACC2.ub,ACC3.ub)
		vdeal(OUT2,OUT0,RNEG32)
	}
	{
		vdeal(OUT3,OUT1,RNEG32)
		NEXT_OUTPTR = add(OUTPTR,#128)
		if (P_ODD) OUT0 = OUT2
	}
	/* Now we need to align these and store. */
	{
		if (P_ODD) OUT1 = OUT3
	}
	{
		TMP0 = VLALIGN(OUT0,OLDOUT0,OUT_LINES_AND_LALIGN)
		//TMP0 = OUT0
		VMEM(OUTPTR++M1) = TMP0.new
		OLDOUT0 = OUT0
	}
	{
		TMP1 = VLALIGN(OUT1,OLDOUT1,OUT_LINES_AND_LALIGN)
		//TMP1 = OUT1
		OLDOUT1 = OUT1
		VMEM(OUTPTR++M1) = TMP1.new
	}
	/* NOTE that IN08 and IN00 are aliased, so we're ready to go for the next iteration */
	{
		IN_OUTPTR = NEXT_IN_OUTPTR
	}:endloop0
	{
		IN_OUTPTR = OUTER_NEXT_IN_OUTPTR
	}:endloop1
	{
		r0 = #0
		jumpr r31
	}
	.size maxpool_slice_hvx_2x2_stride2,.-maxpool_slice_hvx_2x2_stride2
#endif


#undef IN00
#undef IN04
#undef IN10
#undef IN14
#undef IN20
#undef IN24
#undef IN30
#undef IN34
#undef IN40
#undef IN44
#undef IN50
#undef IN54
#undef TMP0
#undef TMP1
#undef TMP2
#undef TMP3
#undef TMP4
#undef TMP5
#undef ACC0
#undef ACC1
#undef ACC2
#undef ACC3
#undef ACC4
#undef ACC5
#undef OLDOUT0
#undef OLDOUT1
#undef OLDOUT2
#undef OLDOUT3
#undef OUT0
#undef OUT1
#undef OUT2
#undef OUT3

#undef OUTPTR
#undef INPTR
#undef IN_OUTPTR
#undef IN_NEXT_ROW
#undef OUT_NEXT_ROW
#undef OUT_VECTORS_WIDE
#undef OUT_LINES_AND_LALIGN
#undef R32
#undef R64
#undef R64_32
#undef NEXT_OUTPTR
#undef NEXT_INPTR
#undef NEXT_IN_OUTPTR
#undef OUTER_NEXT_OUTPTR
#undef OUTER_NEXT_INPTR
#undef OUTER_NEXT_IN_OUTPTR
#undef RNEG32


/* NOTE CAREFULLY: alias INx8 to INx0 for stride2 */
#undef IN08
#undef IN18
#undef IN28
#undef IN38
#undef IN48
#undef P_ODD
