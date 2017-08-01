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
/*======================================================================*/
/*  FUNCTIONS      : qmul_asm                                           */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform elementwise multiplication on input stream,  */
/*                 result at 32bits                                     */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  Om                  04/03/17       created                          */
/*                                                                      */
/*                                                                      */
/*                                                                      */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->  16*K*N/32+11*N/4+24                                          */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =    bytes                                              */
/*     STACK    =    bytes                                              */
/*     ASSUMPTIONS                                                      */
/*        a, b and out are 128 byte aligned                             */
/*        elem%128=0                                                    */
/*======================================================================*/



        .text
        .file "qelementwise_ops_h.S"
        .global qmul_asm
		.balign 32
		.type	qmul_asm,@function
qmul_asm:                               // @qmul_asm
// BB#0:                                // %entry
	{
		r6=asr(r4,#31)
		r7=r4
		memd(r29+#-16)=r17:16
		allocframe(#8)
	}
	{
		r7+=lsr(r6,#25)
		r16=#0
		r9=memw(r3+#0)
		r13=memw(r3+#4)
	}
	{
		r6=and(r7,#-128)
	}
	{
		r12=sub(r4,r6)
		if (cmp.gtu(r12.new,#31)) jump:t .L_qmul_BB0_2
	}
// BB#1:                                // %cond.end.thread
	{
		r3=r12
	}
	{
		r3=and(#124,asl(r3,#2))
	}
	{
		r16=r3 ; jump .L_qmul_BB0_3
	}
	.falign
.L_qmul_BB0_2:                                // %cond.end
	{
		r3=#128
		p1=cmp.gt(r12,#-1)
	}
	{
		if (p1) r16=r3
	}
	.falign
.L_qmul_BB0_3:                                // %cond.end9
	{
		r3=add(r12,#-32)
	}
	{
		r6=#32
		if (!cmp.gtu(r6.new,r3)) jump:t .L_qmul_BB0_5
	}
// BB#4:                                // %cond.end.thread.1
	{
		r3=and(#124,asl(r3,#2))
	}
	{
		r6=r3 ; jump .L_qmul_BB0_6
	}
	.falign
.L_qmul_BB0_5:                                // %cond.end.1
	{
		r3=#128
		r6=#0
		p0=cmp.gt(r12,#31)
	}
	{
		if (p0) r6=r3
	}
	.falign
.L_qmul_BB0_6:                                // %cond.end9.1
	{
		r3=add(r12,#-64)
	}
	{
		r7=#32
		if (!cmp.gtu(r7.new,r3)) jump:t .L_qmul_BB0_8
	}
// BB#7:                                // %cond.end.thread.2
	{
		r3=and(#124,asl(r3,#2))
	}
	{
		r7=r3 ; jump .L_qmul_BB0_9
	}
	.falign
.L_qmul_BB0_8:                                // %cond.end.2
	{
		r3=#128
		r7=#0
		p0=cmp.gt(r12,#63)
	}
	{
		if (p0) r7=r3
	}
	.falign
.L_qmul_BB0_9:                                // %cond.end9.2
	{
		p0=cmp.gt(r12,#95)
		r3=add(r12,#-96)
		r14=memw(r29+#16)
	}
	{
		r8=#32
		if (!cmp.gtu(r8.new,r3)) jump:t .L_qmul_BB0_11
	}
// BB#10:                               // %cond.end.thread.3
	{
		r3=and(#124,asl(r3,#2))
	}
	{
		jump .L_qmul_BB0_12
		r8=r3
	}
	.falign
.L_qmul_BB0_11:                               // %cond.end.3
	{
		r3=#128
		r8=#0
	}
	{
		if (p0) r8=r3
	}
	.falign
.L_qmul_BB0_12:                               // %cond.end9.3
	{
		r13|=asl(r13,#8)
		r5|=asl(r5,#8)
	}
	{
		r9|=asl(r9,#8)
		r14|=asl(r14,#8)
		r12=combine(r5.l,r5.l)
		r13=combine(r13.l,r13.l)
	}
	{
		v1=vsplat(r12)
		p1=cmp.eq(r12,#0)
		r3=combine(r9.l,r9.l)
		r9=combine(r14.l,r14.l)
	}
	{
		v0=vsplat(r3)
		v3=vsplat(r9)
		r5=r2
	}
	{
		r28=asr(r4,#7)
		v2=vsplat(r13)
		if (!cmp.gt(r28.new,#0)) jump:nt .L_qmul_BB0_27
	}
// BB#13:                               // %for.body23.lr.ph
	{
		r3=extractu(r4,#25,#7)
		r4=add(r1,#16384)
		r13=add(r0,#16384)
	}
	{
		r5+=asl(r3,#9)
		r14=addasl(r1,r3,#7)
		if (!p1) r3=#-4
	}
	{
		if (p1) jump:nt .L_qmul_BB0_17
	}
// BB#14:                               // %for.body23.lr.ph.split
	{
		if (p1.new) jump:nt .L_qmul_BB0_21
		v5:4.h=vsub(v1.ub,v0.ub)
		p1=cmp.eq(r9,#0)
		if (p1.new) r3=#-4
	}
// BB#15:
	{
		loop0(.L_qmul_BB0_16,r28)
		r15=#0x80;r14=##0x00800001 //EJP
	}
	{
		v21:20.h=vsub(v3.ub,v2.ub)
	}
	{
		v23:22.w=vmpy(v4.h,v20.h)
	}
	{
		v25:24.w=vmpy(v5.h,v21.h)
		v27:26=vshuff(v23,v22,r3)
	}
	{
		v29:28=vshuff(v25,v24,r3)
	}
	{
		v13:12=vshuff(v28,v26,r3)
	}
	{
		v15:14=vshuff(v29,v27,r3)
	}
	.falign
.L_qmul_tmp0:                                 // Block address taken
.L_qmul_BB0_16:                               // %for.body23
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r15:14)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r15:14)
	//# InlineAsm End
	{
		vmem(r2++#1)=v12
	}
	{
		vmem(r2++#1)=v13
	}
	{
		vmem(r2++#1)=v14
	}
	{
		nop
		vmem(r2++#1)=v15
	}:endloop0
	{
		jump .L_qmul_BB0_27
	}
	.falign
.L_qmul_BB0_17:                               // %for.body23.us.preheader
	{
		r15=asl(r3,#7)
		if (p0.new) jump:nt .L_qmul_BB0_24
		p0=cmp.eq(r9,#0)
		if (!p0.new) r3=#-4
	}
// BB#18:
	r10=##0x00800001 //EJP
	{
		loop0(.L_qmul_BB0_19,r28)
		v5:4.h=vsub(v3.ub,v2.ub)
		r11=#0x80;
	}
	.falign
.L_qmul_tmp1:                                 // Block address taken
.L_qmul_BB0_19:                               // %for.body23.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r14=add(r13,#-16384)
		r13=add(r13,#128)
	}
	{
		v6=vmem(r14+#0)
	}
	{
		v7:6.h=vsub(v6.ub,v0.ub)
	}
	{
		v9:8.w=vmpy(v6.h,v4.h)
	}
	{
		v11:10.w=vmpy(v7.h,v5.h)
		v13:12=vshuff(v9,v8,r3)
	}
	{
		v15:14=vshuff(v11,v10,r3)
	}
	{
		v17:16=vshuff(v14,v12,r3)
	}
	{
		v19:18=vshuff(v15,v13,r3)
		vmem(r2++#1)=v16
	}
	{
		vmem(r2++#1)=v17
	}
	{
		vmem(r2++#1)=v18
	}
	{
		nop
		vmem(r2++#1)=v19
	}:endloop0
// BB#20:
	{
		jump .L_qmul_BB0_26
		r14=r1
	}
	.falign
.L_qmul_BB0_21:                               // %for.body23.us125.preheader
	{
		loop0(.L_qmul_BB0_22,r28)
		v7:6.h=vsub(v1.ub,v0.ub)
		r11=#0x80;
	}
	r10=##0x00800001 //EJP
	.falign
.L_qmul_tmp2:                                 // Block address taken
.L_qmul_BB0_22:                               // %for.body23.us125
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r15=add(r4,#-16384)
		r4=add(r4,#128)
	}
	{
		v31:30.h=vsub(v8.ub,v2.ub)
		v8.cur=vmem(r15+#0)
	}
	{
		v9:8.w=vmpy(v6.h,v30.h)
	}
	{
		v11:10.w=vmpy(v7.h,v31.h)
		v13:12=vshuff(v9,v8,r3)
	}
	{
		v15:14=vshuff(v11,v10,r3)
	}
	{
		v17:16=vshuff(v14,v12,r3)
	}
	{
		v19:18=vshuff(v15,v13,r3)
		vmem(r2++#1)=v16
	}
	{
		vmem(r2++#1)=v17
	}
	{
		vmem(r2++#1)=v18
	}
	{
		nop
		vmem(r2++#1)=v19
	}:endloop0
// BB#23:
	{
		jump .L_qmul_BB0_27
		r1=r14
	}
	.falign
.L_qmul_BB0_24:                               // %for.body23.us.us.preheader
	{
		loop0(.L_qmul_BB0_25,r28)
		r1=#-4
		r11=#0x80;
	}
	r10=##0x00800001 //EJP
	.falign
.L_qmul_tmp3:                                 // Block address taken
.L_qmul_BB0_25:                               // %for.body23.us.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r3=add(r13,#-16384)
		r28=add(r4,#-16384)
		r4=add(r4,#128)
		r13=add(r13,#128)
	}
	{
		v17:16.h=vsub(v4.ub,v0.ub)
		v4.cur=vmem(r3+#0)
	}
	{
		v19:18.h=vsub(v5.ub,v2.ub)
		v5.cur=vmem(r28+#0)
	}
	{
		v21:20.w=vmpy(v16.h,v18.h)
	}
	{
		v23:22.w=vmpy(v17.h,v19.h)
		v25:24=vshuff(v21,v20,r1)
	}
	{
		v27:26=vshuff(v23,v22,r1)
	}
	{
		v29:28=vshuff(v26,v24,r1)
	}
	{
		v31:30=vshuff(v27,v25,r1)
		vmem(r2++#1)=v28
	}
	{
		vmem(r2++#1)=v29
	}
	{
		vmem(r2++#1)=v30
	}
	{
		nop
		vmem(r2++#1)=v31
	}:endloop0
	.falign
.L_qmul_BB0_26:                               // %for.end42.loopexit
	{
		r1=r14
		r0=add(r0,r15)
	}
	.falign
.L_qmul_BB0_27:                               // %for.end42
	{
		r2=add(r0,#16384)
		r15=#0x80;r14=##0x00800001 //EJP
	}
	//# InlineAsm Start
	 l2fetch(r2,r15:14)
	//# InlineAsm End
	{
		r2=add(r1,#16384)
	}
	//# InlineAsm Start
	 l2fetch(r2,r15:14)
	//# InlineAsm End
	{
		if (!p0.new) jump:nt .L_qmul_BB0_29
		p0=cmp.eq(r12,#0)
	}
// BB#28:                               // %cond.false47
	{
		v1=vmem(r0+#0)
	}
	.falign
.L_qmul_BB0_29:                               // %cond.end49
	{
		if (!p1.new) jump:nt .L_qmul_BB0_31
		p1=cmp.eq(r9,#0)
	}
// BB#30:                               // %cond.false53
	{
		v3=vmem(r1+#0)
	}
	.falign
.L_qmul_BB0_31:                               // %cond.end55
	{
		v1:0.h=vsub(v1.ub,v0.ub)
		r4=#-4
		p0=cmp.eq(r16,#128)
		q0=vsetq(r16)
	}
	{
		v3:2.h=vsub(v3.ub,v2.ub)
	}
	{
		v7:6.w=vmpy(v0.h,v2.h)
	}
	{
		v5:4.w=vmpy(v1.h,v3.h)
		v1:0=vshuff(v7,v6,r4)
	}
	{
		v3:2=vshuff(v5,v4,r4)
	}
	{
		if (!p0) jump:nt .L_qmul_BB0_33
		v11:10=vshuff(v2,v0,r4)
	}
// BB#32:                               // %if.else
	{
		jump .L_qmul_BB0_34
		vmem(r5+#0)=v10
	}
	.falign
.L_qmul_BB0_33:                               // %if.then
	{
		if (q0) vmem(r5+#0):nt=v10
	}
	.falign
.L_qmul_BB0_34:                               // %if.end
	{
		if (!p1.new) jump:nt .L_qmul_BB0_36
		r2=add(r5,#128)
		p1=cmp.eq(r6,#128)
		q1=vsetq(r6)
	}
// BB#35:                               // %if.else67
	{
		jump .L_qmul_BB0_37
		vmem(r2+#0)=v11
	}
	.falign
.L_qmul_BB0_36:                               // %if.then64
	{
		if (q1) vmem(r2+#0):nt=v11
	}
	.falign
.L_qmul_BB0_37:                               // %if.end69
	{
		if (!p0.new) jump:nt .L_qmul_BB0_39
		r2=add(r5,#256)
		p0=cmp.eq(r7,#128)
		v9:8=vshuff(v3,v1,r4)
	}
// BB#38:                               // %if.else75
	{
		jump .L_qmul_BB0_40
		vmem(r2+#0)=v8
	}
	.falign
.L_qmul_BB0_39:                               // %if.then72
	{
		q2=vsetq(r7)
	}
	{
		if (q2) vmem(r2+#0):nt=v8
	}
	.falign
.L_qmul_BB0_40:                               // %if.end77
	{
		if (!p1.new) jump:nt .L_qmul_BB0_42
		r2=add(r5,#384)
		p1=cmp.eq(r8,#128)
		q3=vsetq(r8)
	}
// BB#41:                               // %if.end85
	{
		r17:16=memd(r29+#0)
		vmem(r2+#0)=v9
	}
	{
		dealloc_return
	}
	.falign
.L_qmul_BB0_42:                               // %if.then80
	{
		r17:16=memd(r29+#0)
		if (q3) vmem(r2+#0):nt=v9
	}
	{
		dealloc_return
	}
.L_qmul_func_end0:
.L_qmul_tmp4:
	.size	qmul_asm, .L_qmul_tmp4-qmul_asm
//===============================================================================================================================================================
    .text
    .file "qelementwise_ops_h.S"
    .global qadd_asm
	.balign 32
	.type	qadd_asm,@function
qadd_asm:                               // @qadd_asm
// BB#0:                                // %entry
	{
		r6=asr(r4,#31)
		r7=r4
		memd(r29+#-16)=r17:16
		allocframe(#8)
	}
	{
		r7+=lsr(r6,#25)
		r13=memw(r3+#0)
		r14=memw(r3+#4)
	}
	{
		r6=and(r7,#-128)
		r9=memw(r3+#8)
		r12=memw(r3+#12)
	}
	{
		r3=#0
		r15=sub(r4,r6)
		if (cmp.gtu(r15.new,#31)) jump:t .L_qadd_BB0_2
	}
// BB#1:                                // %cond.end.thread
	{
		r6=r15
	}
	{
		r6=and(#124,asl(r6,#2))
	}
	{
		r3=r6 ; jump .L_qadd_BB0_3
	}
	.falign
.L_qadd_BB0_2:                                // %cond.end
	{
		r6=#128
		p1=cmp.gt(r15,#-1)
	}
	{
		if (p1) r3=r6
	}
	.falign
.L_qadd_BB0_3:                                // %cond.end11
	{
		r6=add(r15,#-32)
	}
	{
		r7=#32
		if (!cmp.gtu(r7.new,r6)) jump:t .L_qadd_BB0_5
	}
// BB#4:                                // %cond.end.thread.1
	{
		r6=and(#124,asl(r6,#2))
	}
	{
		r16=r6 ; jump .L_qadd_BB0_6
	}
	.falign
.L_qadd_BB0_5:                                // %cond.end.1
	{
		r6=#128
		r16=#0
		p0=cmp.gt(r15,#31)
	}
	{
		if (p0) r16=r6
	}
	.falign
.L_qadd_BB0_6:                                // %cond.end11.1
	{
		r6=add(r15,#-64)
	}
	{
		r7=#32
		if (!cmp.gtu(r7.new,r6)) jump:t .L_qadd_BB0_8
	}
// BB#7:                                // %cond.end.thread.2
	{
		r6=and(#124,asl(r6,#2))
	}
	{
		r7=r6 ; jump .L_qadd_BB0_9
	}
	.falign
.L_qadd_BB0_8:                                // %cond.end.2
	{
		r6=#128
		r7=#0
		p0=cmp.gt(r15,#63)
	}
	{
		if (p0) r7=r6
	}
	.falign
.L_qadd_BB0_9:                                // %cond.end11.2
	{
		p0=cmp.gt(r15,#95)
		r28=add(r15,#-96)
		r6=memw(r29+#16)
	}
	{
		r8=#32
		if (!cmp.gtu(r8.new,r28)) jump:t .L_qadd_BB0_11
	}
// BB#10:                               // %cond.end.thread.3
	{
		r28=and(#124,asl(r28,#2))
	}
	{
		jump .L_qadd_BB0_12
		r8=r28
	}
	.falign
.L_qadd_BB0_11:                               // %cond.end.3
	{
		r28=#128
		r8=#0
	}
	{
		if (p0) r8=r28
	}
	.falign
.L_qadd_BB0_12:                               // %cond.end11.3
	{
		r5|=asl(r5,#8)
		r6|=asl(r6,#8)
		r15=combine(r9.l,r9.l)
		r10=combine(r12.l,r12.l)
	}
	{
		r13|=asl(r13,#8)
		r14|=asl(r14,#8)
		r12=combine(r5.l,r5.l)
		r9=combine(r6.l,r6.l)
	}
	{
		v0=vsplat(r12)
		r6=combine(r13.l,r13.l)
		r13=combine(r14.l,r14.l)
		r5=r2
	}
	{
		v1=vsplat(r9)
		v3=vsplat(r15)
		p1=cmp.eq(r12,#0)
	}
	{
		v2=vsplat(r10)
		v5=vsplat(r6)
	}
	{
		r28=asr(r4,#7)
		v4=vsplat(r13)
		if (!cmp.gt(r28.new,#0)) jump:nt .L_qadd_BB0_28
	}
// BB#13:                               // %for.body25.lr.ph
	{
		r6=extractu(r4,#25,#7)
		r4=add(r1,#16384)
		r13=add(r0,#16384)
	}
	{
		r5+=asl(r6,#9)
		r14=addasl(r1,r6,#7)
		if (!p1) r6=#-4
	}
	{
		if (p1) jump:nt .L_qadd_BB0_18
	}
// BB#14:                               // %for.body25.preheader
	{
		if (p0.new) jump:nt .L_qadd_BB0_22
		v21:20.h=vsub(v0.ub,v5.ub)
		p0=cmp.eq(r9,#0)
		if (p0.new) r1=#-4
	}
// BB#15:
	{
		loop0(.L_qadd_BB0_16,r28)
		r15=#0x80;r14=##0x00800001 //EJP
	}
	{
		v23:22.h=vsub(v1.ub,v4.ub)
	}
	{
		v25:24.w=vmpy(v20.h,v3.h)
	}
	{
		v29:28.w=vmpy(v22.h,v2.h)
	}
	{
		v27:26.w=vmpy(v21.h,v3.h)
		v7:6.w=vadd(v25:24.w,v29:28.w):sat
	}
	{
		v31:30.w=vmpy(v23.h,v2.h)
	}
	{
		v11:10=vshuff(v7,v6,r6)
		v9:8.w=vadd(v27:26.w,v31:30.w):sat
	}
	{
		v13:12=vshuff(v9,v8,r6)
	}
	{
		v17:16=vshuff(v12,v10,r6)
	}
	{
		v19:18=vshuff(v13,v11,r6)
	}
	.falign
.L_qadd_tmp0:                                 // Block address taken
.L_qadd_BB0_16:                               // %for.body25
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r15:14)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r15:14)
	//# InlineAsm End
	{
		vmem(r2++#1)=v16
	}
	{
		vmem(r2++#1)=v17
	}
	{
		vmem(r2++#1)=v18
	}
	{
		nop
		vmem(r2++#1)=v19
	}:endloop0
// BB#17:
	{
		jump .L_qadd_BB0_24
		r14=r1
	}
	.falign
.L_qadd_BB0_18:                               // %for.body25.us.preheader
	{
		r15=asl(r6,#7)
		if (p1.new) jump:nt .L_qadd_BB0_25
		p1=cmp.eq(r9,#0)
		if (!p1.new) r6=#-4
	}
// BB#19:
	{
		loop0(.L_qadd_BB0_20,r28)
		v7:6.h=vsub(v1.ub,v4.ub)
		r11=#0x80;
	}
	r10=##0x00800001 //EJP
	{
		v9:8.w=vmpy(v6.h,v2.h)
	}
	{
		v13:12.w=vmpy(v7.h,v2.h)
	}
	.falign
.L_qadd_tmp1:                                 // Block address taken
.L_qadd_BB0_20:                               // %for.body25.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r14=add(r13,#-16384)
		r13=add(r13,#128)
	}
	{
		v7:6.h=vsub(v10.ub,v5.ub)
		v10.cur=vmem(r14+#0)
	}
	{
		v11:10.w=vmpy(v6.h,v3.h)
	}
	{
		v15:14.w=vmpy(v7.h,v3.h)
		v17:16.w=vadd(v11:10.w,v9:8.w):sat
	}
	{
		v21:20=vshuff(v17,v16,r6)
		v19:18.w=vadd(v15:14.w,v13:12.w):sat
	}
	{
		v23:22=vshuff(v19,v18,r6)
	}
	{
		v25:24=vshuff(v22,v20,r6)
	}
	{
		v27:26=vshuff(v23,v21,r6)
		vmem(r2++#1)=v24
	}
	{
		vmem(r2++#1)=v25
	}
	{
		vmem(r2++#1)=v26
	}
	{
		nop
		vmem(r2++#1)=v27
	}:endloop0
// BB#21:
	{
		jump .L_qadd_BB0_27
		r14=r1
	}
	.falign
.L_qadd_BB0_22:                               // %for.body25.us161.preheader
	{
		loop0(.L_qadd_BB0_23,r28)
		v15:14.h=vsub(v0.ub,v5.ub)
		r11=#0x80;
	}
	{
		v7:6.w=vmpy(v14.h,v3.h)
		r10=##0x00800001 //EJP
	}
	{
		v11:10.w=vmpy(v15.h,v3.h)
	}
	.falign
.L_qadd_tmp2:                                 // Block address taken
.L_qadd_BB0_23:                               // %for.body25.us161
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r6=add(r4,#-16384)
		r4=add(r4,#128)
	}
	{
		v17:16.h=vsub(v8.ub,v4.ub)
		v8.cur=vmem(r6+#0)
	}
	{
		v19:18.w=vmpy(v16.h,v2.h)
	}
	{
		v21:20.w=vmpy(v17.h,v2.h)
		v23:22.w=vadd(v7:6.w,v19:18.w):sat
	}
	{
		v27:26=vshuff(v23,v22,r1)
		v25:24.w=vadd(v11:10.w,v21:20.w):sat
	}
	{
		v29:28=vshuff(v25,v24,r1)
	}
	{
		v31:30=vshuff(v28,v26,r1)
	}
	{
		v9:8=vshuff(v29,v27,r1)
		vmem(r2++#1)=v30
	}
	{
		vmem(r2++#1)=v31
	}
	{
		vmem(r2++#1)=v8
	}
	{
		nop
		vmem(r2++#1)=v9
	}:endloop0
	.falign
.L_qadd_BB0_24:                               // %for.end44.loopexit149
	{
		jump .L_qadd_BB0_28
		r1=r14
	}
	.falign
.L_qadd_BB0_25:                               // %for.body25.us.us.preheader
	{
		loop0(.L_qadd_BB0_26,r28)
		r1=#-4
		r11=#0x80;
	}
	r10=##0x00800001 //EJP
	.falign
.L_qadd_tmp3:                                 // Block address taken
.L_qadd_BB0_26:                               // %for.body25.us.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r6=add(r13,#-16384)
		r28=add(r4,#-16384)
		r4=add(r4,#128)
		r13=add(r13,#128)
	}
	{
		v11:10.h=vsub(v6.ub,v5.ub)
		v6.cur=vmem(r6+#0)
	}
	{
		v13:12.h=vsub(v7.ub,v4.ub)
		v7.cur=vmem(r28+#0)
	}
	{
		v15:14.w=vmpy(v10.h,v3.h)
	}
	{
		v19:18.w=vmpy(v12.h,v2.h)
	}
	{
		v17:16.w=vmpy(v11.h,v3.h)
		v23:22.w=vadd(v15:14.w,v19:18.w):sat
	}
	{
		v21:20.w=vmpy(v13.h,v2.h)
	}
	{
		v27:26=vshuff(v23,v22,r1)
		v25:24.w=vadd(v17:16.w,v21:20.w):sat
	}
	{
		v29:28=vshuff(v25,v24,r1)
	}
	{
		v31:30=vshuff(v28,v26,r1)
	}
	{
		v9:8=vshuff(v29,v27,r1)
		vmem(r2++#1)=v30
	}
	{
		vmem(r2++#1)=v31
	}
	{
		vmem(r2++#1)=v8
	}
	{
		nop
		vmem(r2++#1)=v9
	}:endloop0
	.falign
.L_qadd_BB0_27:                               // %for.end44.loopexit
	{
		r1=r14
		r0=add(r0,r15)
	}
	.falign
.L_qadd_BB0_28:                               // %for.end44
	{
		r2=add(r0,#16384)
		r15=#0x80;r14=##0x00800001 //EJP
	}
	//# InlineAsm Start
	 l2fetch(r2,r15:14)
	//# InlineAsm End
	{
		r2=add(r1,#16384)
	}
	//# InlineAsm Start
	 l2fetch(r2,r15:14)
	//# InlineAsm End
	{
		if (!p0.new) jump:nt .L_qadd_BB0_30
		p0=cmp.eq(r12,#0)
	}
// BB#29:                               // %cond.false49
	{
		v0=vmem(r0+#0)
	}
	.falign
.L_qadd_BB0_30:                               // %cond.end51
	{
		if (!p1.new) jump:nt .L_qadd_BB0_32
		p1=cmp.eq(r9,#0)
	}
// BB#31:                               // %cond.false55
	{
		v1=vmem(r1+#0)
	}
	.falign
.L_qadd_BB0_32:                               // %cond.end57
	{
		v13:12.h=vsub(v0.ub,v5.ub)
		r4=#-4
		p0=cmp.eq(r3,#128)
		q0=vsetq(r3)
	}
	{
		v1:0.h=vsub(v1.ub,v4.ub)
	}
	{
		v5:4.w=vmpy(v12.h,v3.h)
	}
	{
		v31:30.w=vmpy(v0.h,v2.h)
	}
	{
		v29:28.w=vmpy(v13.h,v3.h)
		v9:8.w=vadd(v5:4.w,v31:30.w):sat
	}
	{
		v3:2.w=vmpy(v1.h,v2.h)
	}
	{
		v1:0=vshuff(v9,v8,r4)
		v7:6.w=vadd(v29:28.w,v3:2.w):sat
	}
	{
		v3:2=vshuff(v7,v6,r4)
	}
	{
		if (!p0) jump:nt .L_qadd_BB0_34
		v15:14=vshuff(v2,v0,r4)
	}
// BB#33:                               // %if.else
	{
		jump .L_qadd_BB0_35
		vmem(r5+#0)=v14
	}
	.falign
.L_qadd_BB0_34:                               // %if.then
	{
		if (q0) vmem(r5+#0):nt=v14
	}
	.falign
.L_qadd_BB0_35:                               // %if.end
	{
		if (!p1.new) jump:nt .L_qadd_BB0_37
		r2=add(r5,#128)
		p1=cmp.eq(r16,#128)
		q1=vsetq(r16)
	}
// BB#36:                               // %if.else69
	{
		jump .L_qadd_BB0_38
		vmem(r2+#0)=v15
	}
	.falign
.L_qadd_BB0_37:                               // %if.then66
	{
		if (q1) vmem(r2+#0):nt=v15
	}
	.falign
.L_qadd_BB0_38:                               // %if.end71
	{
		if (!p0.new) jump:nt .L_qadd_BB0_40
		r2=add(r5,#256)
		p0=cmp.eq(r7,#128)
		v5:4=vshuff(v3,v1,r4)
	}
// BB#39:                               // %if.else77
	{
		jump .L_qadd_BB0_41
		vmem(r2+#0)=v4
	}
	.falign
.L_qadd_BB0_40:                               // %if.then74
	{
		q2=vsetq(r7)
	}
	{
		if (q2) vmem(r2+#0):nt=v4
	}
	.falign
.L_qadd_BB0_41:                               // %if.end79
	{
		if (!p1.new) jump:nt .L_qadd_BB0_43
		r2=add(r5,#384)
		p1=cmp.eq(r8,#128)
		q3=vsetq(r8)
	}
// BB#42:                               // %if.end87
	{
		r17:16=memd(r29+#0)
		vmem(r2+#0)=v5
	}
	{
		dealloc_return
	}
	.falign
.L_qadd_BB0_43:                               // %if.then82
	{
		r17:16=memd(r29+#0)
		if (q3) vmem(r2+#0):nt=v5
	}
	{
		dealloc_return
	}
.L_qadd_func_end0:
.L_qadd_tmp4:
	.size	qadd_asm, .L_qadd_tmp4-qadd_asm

//==============================================================================================================
    .text
    .file "qelementwise_ops_h.S"
    .global qsub_asm
	.balign 32
	.type	qsub_asm,@function
qsub_asm:                               // @qsub_asm
// BB#0:                                // %entry
	{
		r6=asr(r4,#31)
		r7=r4
		memd(r29+#-16)=r17:16
		allocframe(#8)
	}
	{
		r7+=lsr(r6,#25)
		r13=memw(r3+#0)
		r14=memw(r3+#4)
	}
	{
		r6=and(r7,#-128)
		r9=memw(r3+#8)
		r12=memw(r3+#12)
	}
	{
		r3=#0
		r15=sub(r4,r6)
		if (cmp.gtu(r15.new,#31)) jump:t .L_qsub_BB0_2
	}
// BB#1:                                // %cond.end.thread
	{
		r6=r15
	}
	{
		r6=and(#124,asl(r6,#2))
	}
	{
		r3=r6 ; jump .L_qsub_BB0_3
	}
	.falign
.L_qsub_BB0_2:                                // %cond.end
	{
		r6=#128
		p1=cmp.gt(r15,#-1)
	}
	{
		if (p1) r3=r6
	}
	.falign
.L_qsub_BB0_3:                                // %cond.end11
	{
		r6=add(r15,#-32)
	}
	{
		r7=#32
		if (!cmp.gtu(r7.new,r6)) jump:t .L_qsub_BB0_5
	}
// BB#4:                                // %cond.end.thread.1
	{
		r6=and(#124,asl(r6,#2))
	}
	{
		r16=r6 ; jump .L_qsub_BB0_6
	}
	.falign
.L_qsub_BB0_5:                                // %cond.end.1
	{
		r6=#128
		r16=#0
		p0=cmp.gt(r15,#31)
	}
	{
		if (p0) r16=r6
	}
	.falign
.L_qsub_BB0_6:                                // %cond.end11.1
	{
		r6=add(r15,#-64)
	}
	{
		r7=#32
		if (!cmp.gtu(r7.new,r6)) jump:t .L_qsub_BB0_8
	}
// BB#7:                                // %cond.end.thread.2
	{
		r6=and(#124,asl(r6,#2))
	}
	{
		r7=r6 ; jump .L_qsub_BB0_9
	}
	.falign
.L_qsub_BB0_8:                                // %cond.end.2
	{
		r6=#128
		r7=#0
		p0=cmp.gt(r15,#63)
	}
	{
		if (p0) r7=r6
	}
	.falign
.L_qsub_BB0_9:                                // %cond.end11.2
	{
		p0=cmp.gt(r15,#95)
		r28=add(r15,#-96)
		r6=memw(r29+#16)
	}
	{
		r8=#32
		if (!cmp.gtu(r8.new,r28)) jump:t .L_qsub_BB0_11
	}
// BB#10:                               // %cond.end.thread.3
	{
		r28=and(#124,asl(r28,#2))
	}
	{
		jump .L_qsub_BB0_12
		r8=r28
	}
	.falign
.L_qsub_BB0_11:                               // %cond.end.3
	{
		r28=#128
		r8=#0
	}
	{
		if (p0) r8=r28
	}
	.falign
.L_qsub_BB0_12:                               // %cond.end11.3
	{
		r5|=asl(r5,#8)
		r6|=asl(r6,#8)
		r15=combine(r9.l,r9.l)
		r10=combine(r12.l,r12.l)
	}
	{
		r13|=asl(r13,#8)
		r14|=asl(r14,#8)
		r12=combine(r5.l,r5.l)
		r9=combine(r6.l,r6.l)
	}
	{
		v0=vsplat(r12)
		r6=combine(r13.l,r13.l)
		r13=combine(r14.l,r14.l)
		r5=r2
	}
	{
		v1=vsplat(r9)
		v3=vsplat(r15)
		p1=cmp.eq(r12,#0)
	}
	{
		v2=vsplat(r10)
		v5=vsplat(r6)
	}
	{
		r28=asr(r4,#7)
		v4=vsplat(r13)
		if (!cmp.gt(r28.new,#0)) jump:nt .L_qsub_BB0_28
	}
// BB#13:                               // %for.body25.lr.ph
	{
		r6=extractu(r4,#25,#7)
		r4=add(r1,#16384)
		r13=add(r0,#16384)
	}
	{
		r5+=asl(r6,#9)
		r14=addasl(r1,r6,#7)
		if (!p1) r6=#-4
	}
	{
		if (p1) jump:nt .L_qsub_BB0_18
	}
// BB#14:                               // %for.body25.preheader
	{
		if (p0.new) jump:nt .L_qsub_BB0_22
		v21:20.h=vsub(v0.ub,v5.ub)
		p0=cmp.eq(r9,#0)
		if (p0.new) r1=#-4
	}
// BB#15:
	{
		loop0(.L_qsub_BB0_16,r28)
		r15=#0x80;r14=##0x00800001 //EJP
	}
	{
		v23:22.h=vsub(v1.ub,v4.ub)
	}
	{
		v25:24.w=vmpy(v20.h,v3.h)
	}
	{
		v29:28.w=vmpy(v22.h,v2.h)
	}
	{
		v27:26.w=vmpy(v21.h,v3.h)
		v7:6.w=vsub(v25:24.w,v29:28.w):sat
	}
	{
		v31:30.w=vmpy(v23.h,v2.h)
	}
	{
		v11:10=vshuff(v7,v6,r6)
		v9:8.w=vsub(v27:26.w,v31:30.w):sat
	}
	{
		v13:12=vshuff(v9,v8,r6)
	}
	{
		v17:16=vshuff(v12,v10,r6)
	}
	{
		v19:18=vshuff(v13,v11,r6)
	}
	.falign
.L_qsub_tmp0:                                 // Block address taken
.L_qsub_BB0_16:                               // %for.body25
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r15:14)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r15:14)
	//# InlineAsm End
	{
		vmem(r2++#1)=v16
	}
	{
		vmem(r2++#1)=v17
	}
	{
		vmem(r2++#1)=v18
	}
	{
		nop
		vmem(r2++#1)=v19
	}:endloop0
// BB#17:
	{
		jump .L_qsub_BB0_24
		r14=r1
	}
	.falign
.L_qsub_BB0_18:                               // %for.body25.us.preheader
	{
		r15=asl(r6,#7)
		if (p1.new) jump:nt .L_qsub_BB0_25
		p1=cmp.eq(r9,#0)
		if (!p1.new) r6=#-4
	}
// BB#19:
	{
		loop0(.L_qsub_BB0_20,r28)
		v7:6.h=vsub(v1.ub,v4.ub)
		r11=#0x80;
	}
	{
		v9:8.w=vmpy(v6.h,v2.h)
		r10=##0x00800001 //EJP
	}
	{
		v13:12.w=vmpy(v7.h,v2.h)
	}
	.falign
.L_qsub_tmp1:                                 // Block address taken
.L_qsub_BB0_20:                               // %for.body25.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r14=add(r13,#-16384)
		r13=add(r13,#128)
	}
	{
		v7:6.h=vsub(v10.ub,v5.ub)
		v10.cur=vmem(r14+#0)
	}
	{
		v11:10.w=vmpy(v6.h,v3.h)
	}
	{
		v15:14.w=vmpy(v7.h,v3.h)
		v17:16.w=vsub(v11:10.w,v9:8.w):sat
	}
	{
		v21:20=vshuff(v17,v16,r6)
		v19:18.w=vsub(v15:14.w,v13:12.w):sat
	}
	{
		v23:22=vshuff(v19,v18,r6)
	}
	{
		v25:24=vshuff(v22,v20,r6)
	}
	{
		v27:26=vshuff(v23,v21,r6)
		vmem(r2++#1)=v24
	}
	{
		vmem(r2++#1)=v25
	}
	{
		vmem(r2++#1)=v26
	}
	{
		nop
		vmem(r2++#1)=v27
	}:endloop0
// BB#21:
	{
		jump .L_qsub_BB0_27
		r14=r1
	}
	.falign
.L_qsub_BB0_22:                               // %for.body25.us161.preheader
	{
		loop0(.L_qsub_BB0_23,r28)
		v15:14.h=vsub(v0.ub,v5.ub)
		r11=#0x80;
	}
	{
		v7:6.w=vmpy(v14.h,v3.h)
		r10=##0x00800001 //EJP
	}
	{
		v11:10.w=vmpy(v15.h,v3.h)
	}
	.falign
.L_qsub_tmp2:                                 // Block address taken
.L_qsub_BB0_23:                               // %for.body25.us161
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r6=add(r4,#-16384)
		r4=add(r4,#128)
	}
	{
		v17:16.h=vsub(v8.ub,v4.ub)
		v8.cur=vmem(r6+#0)
	}
	{
		v19:18.w=vmpy(v16.h,v2.h)
	}
	{
		v21:20.w=vmpy(v17.h,v2.h)
		v23:22.w=vsub(v7:6.w,v19:18.w):sat
	}
	{
		v27:26=vshuff(v23,v22,r1)
		v25:24.w=vsub(v11:10.w,v21:20.w):sat
	}
	{
		v29:28=vshuff(v25,v24,r1)
	}
	{
		v31:30=vshuff(v28,v26,r1)
	}
	{
		v9:8=vshuff(v29,v27,r1)
		vmem(r2++#1)=v30
	}
	{
		vmem(r2++#1)=v31
	}
	{
		vmem(r2++#1)=v8
	}
	{
		nop
		vmem(r2++#1)=v9
	}:endloop0
	.falign
.L_qsub_BB0_24:                               // %for.end44.loopexit149
	{
		jump .L_qsub_BB0_28
		r1=r14
	}
	.falign
.L_qsub_BB0_25:                               // %for.body25.us.us.preheader
	{
		loop0(.L_qsub_BB0_26,r28)
		r1=#-4
		r11=#0x80;
	}
	r10=##0x00800001 //EJP
	.falign
.L_qsub_tmp3:                                 // Block address taken
.L_qsub_BB0_26:                               // %for.body25.us.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r4,r11:10)
	//# InlineAsm End
	{
		r6=add(r13,#-16384)
		r28=add(r4,#-16384)
		r4=add(r4,#128)
		r13=add(r13,#128)
	}
	{
		v11:10.h=vsub(v6.ub,v5.ub)
		v6.cur=vmem(r6+#0)
	}
	{
		v13:12.h=vsub(v7.ub,v4.ub)
		v7.cur=vmem(r28+#0)
	}
	{
		v15:14.w=vmpy(v10.h,v3.h)
	}
	{
		v19:18.w=vmpy(v12.h,v2.h)
	}
	{
		v17:16.w=vmpy(v11.h,v3.h)
		v23:22.w=vsub(v15:14.w,v19:18.w):sat
	}
	{
		v21:20.w=vmpy(v13.h,v2.h)
	}
	{
		v27:26=vshuff(v23,v22,r1)
		v25:24.w=vsub(v17:16.w,v21:20.w):sat
	}
	{
		v29:28=vshuff(v25,v24,r1)
	}
	{
		v31:30=vshuff(v28,v26,r1)
	}
	{
		v9:8=vshuff(v29,v27,r1)
		vmem(r2++#1)=v30
	}
	{
		vmem(r2++#1)=v31
	}
	{
		vmem(r2++#1)=v8
	}
	{
		nop
		vmem(r2++#1)=v9
	}:endloop0
	.falign
.L_qsub_BB0_27:                               // %for.end44.loopexit
	{
		r1=r14
		r0=add(r0,r15)
	}
	.falign
.L_qsub_BB0_28:                               // %for.end44
	{
		r2=add(r0,#16384)
		r15=#0x80;r14=##0x00800001 //EJP
	}
	//# InlineAsm Start
	 l2fetch(r2,r15:14)
	//# InlineAsm End
	{
		r2=add(r1,#16384)
	}
	//# InlineAsm Start
	 l2fetch(r2,r15:14)
	//# InlineAsm End
	{
		if (!p0.new) jump:nt .L_qsub_BB0_30
		p0=cmp.eq(r12,#0)
	}
// BB#29:                               // %cond.false49
	{
		v0=vmem(r0+#0)
	}
	.falign
.L_qsub_BB0_30:                               // %cond.end51
	{
		if (!p1.new) jump:nt .L_qsub_BB0_32
		p1=cmp.eq(r9,#0)
	}
// BB#31:                               // %cond.false55
	{
		v1=vmem(r1+#0)
	}
	.falign
.L_qsub_BB0_32:                               // %cond.end57
	{
		v13:12.h=vsub(v0.ub,v5.ub)
		r4=#-4
		p0=cmp.eq(r3,#128)
		q0=vsetq(r3)
	}
	{
		v1:0.h=vsub(v1.ub,v4.ub)
	}
	{
		v5:4.w=vmpy(v12.h,v3.h)
	}
	{
		v31:30.w=vmpy(v0.h,v2.h)
	}
	{
		v29:28.w=vmpy(v13.h,v3.h)
		v9:8.w=vsub(v5:4.w,v31:30.w):sat
	}
	{
		v3:2.w=vmpy(v1.h,v2.h)
	}
	{
		v1:0=vshuff(v9,v8,r4)
		v7:6.w=vsub(v29:28.w,v3:2.w):sat
	}
	{
		v3:2=vshuff(v7,v6,r4)
	}
	{
		if (!p0) jump:nt .L_qsub_BB0_34
		v15:14=vshuff(v2,v0,r4)
	}
// BB#33:                               // %if.else
	{
		jump .L_qsub_BB0_35
		vmem(r5+#0)=v14
	}
	.falign
.L_qsub_BB0_34:                               // %if.then
	{
		if (q0) vmem(r5+#0):nt=v14
	}
	.falign
.L_qsub_BB0_35:                               // %if.end
	{
		if (!p1.new) jump:nt .L_qsub_BB0_37
		r2=add(r5,#128)
		p1=cmp.eq(r16,#128)
		q1=vsetq(r16)
	}
// BB#36:                               // %if.else69
	{
		jump .L_qsub_BB0_38
		vmem(r2+#0)=v15
	}
	.falign
.L_qsub_BB0_37:                               // %if.then66
	{
		if (q1) vmem(r2+#0):nt=v15
	}
	.falign
.L_qsub_BB0_38:                               // %if.end71
	{
		if (!p0.new) jump:nt .L_qsub_BB0_40
		r2=add(r5,#256)
		p0=cmp.eq(r7,#128)
		v5:4=vshuff(v3,v1,r4)
	}
// BB#39:                               // %if.else77
	{
		jump .L_qsub_BB0_41
		vmem(r2+#0)=v4
	}
	.falign
.L_qsub_BB0_40:                               // %if.then74
	{
		q2=vsetq(r7)
	}
	{
		if (q2) vmem(r2+#0):nt=v4
	}
	.falign
.L_qsub_BB0_41:                               // %if.end79
	{
		if (!p1.new) jump:nt .L_qsub_BB0_43
		r2=add(r5,#384)
		p1=cmp.eq(r8,#128)
		q3=vsetq(r8)
	}
// BB#42:                               // %if.end87
	{
		r17:16=memd(r29+#0)
		vmem(r2+#0)=v5
	}
	{
		dealloc_return
	}
	.falign
.L_qsub_BB0_43:                               // %if.then82
	{
		r17:16=memd(r29+#0)
		if (q3) vmem(r2+#0):nt=v5
	}
	{
		dealloc_return
	}
.L_qsub_func_end0:
.L_qsub_tmp4:
	.size	qsub_asm, .L_qsub_tmp4-qsub_asm
//===========================================================================================================================
    .text
    .file "qelementwise_ops_h.S"
	.global	qmaximum_asm
	.balign 32
	.type	qmaximum_asm,@function
qmaximum_asm:                           // @qmaximum_asm
// BB#0:                                // %entry
	{
		r5|=asl(r5,#8)
		r7=memw(r29+#0)
		r8=memw(r3+#0)
	}
	{
		r8|=asl(r8,#8)
		r7|=asl(r7,#8)
		r9=memw(r3+#4)
		r6=memw(r3+#8)
	}
	{
		r9|=asl(r9,#8)
		r6=combine(r6.l,r6.l)
		r12=memw(r3+#20)
		r13=memw(r3+#12)
	}
	{
		r12|=asl(r12,#8)
		v2=vsplat(r6)
		r13=combine(r13.l,r13.l)
		r6=combine(r5.l,r5.l)
	}
	{
		v1=vsplat(r13)
		r5=combine(r7.l,r7.l)
		r8=combine(r8.l,r8.l)
		r9=combine(r9.l,r9.l)
	}
	{
		v0=vsplat(r12)
		v4=vsplat(r6)
		r13=add(r1,#16384)
		r7=memw(r3+#16)
	}
	{
		v3=vsplat(r5)
		v6=vsplat(r8)
		p1=cmp.eq(r6,#0)
		r12=add(r0,#16384)
	}
	{
		r14=asr(r4,#7)
		v5=vsplat(r9)
		if (!cmp.gt(r14.new,#0)) jump:nt .L_qmaximum_BB9_6
	}
// BB#1:                                // %for.body.lr.ph
	{
		r3=extractu(r4,#25,#7)
	}
	{
		r15=addasl(r2,r3,#7)
		r8=addasl(r1,r3,#7)
		if (!p1) r3=#0
	}
	{
		if (p1) jump:nt .L_qmaximum_BB9_7
	}
// BB#2:                                // %for.body.lr.ph.split
	{
		p1=cmp.eq(r5,#0); if (p1.new) jump:nt .L_qmaximum_BB9_11
		v27:26.h=vsub(v4.ub,v6.ub)
		if (p1.new) r3=#0
	}
// BB#3:
	{
		loop0(.L_qmaximum_BB9_4,r14)
		r9=#0x80;r8=##0x00800001 //EJP
	}
	{
		v29:28.h=vsub(v3.ub,v5.ub)
	}
	{
		v31:30.w=vmpy(v26.h,v2.h)
	}
	{
		v9:8.w=vmpy(v27.h,v2.h)
	}
	{
		v11:10.w=vmpy(v28.h,v1.h)
		v20.h=vasr(v31.w,v30.w,r7)
	}
	{
		v13:12.w=vmpy(v29.h,v1.h)
		v21.h=vasr(v9.w,v8.w,r7)
	}
	{
		v22.h=vasr(v11.w,v10.w,r7)
	}
	{
		v23.h=vasr(v13.w,v12.w,r7)
	}
	{
		v24.ub=vasr(v21.h,v20.h,r3):sat
	}
	{
		v25.ub=vasr(v23.h,v22.h,r3):sat
		v23.ub=vadd(v24.ub,v0.ub):sat
	}
	{
		v27.ub=vadd(v25.ub,v0.ub):sat
	}
	{
		v13.ub=vmax(v23.ub,v27.ub)
	}
	.falign
.L_qmaximum_tmp25:                                // Block address taken
.L_qmaximum_BB9_4:                                // %for.body
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r12,r9:8)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r9:8)
	//# InlineAsm End
	{
		nop
		vmem(r2++#1)=v13
	}:endloop0
// BB#5:                                // %for.end.loopexit167
	{
		jump .L_qmaximum_BB9_18
		r8=r1
	}
	.falign
.L_qmaximum_BB9_6:
	{
		jump .L_qmaximum_BB9_18
		r8=r1
		r15=r2
	}
	.falign
.L_qmaximum_BB9_7:                                // %for.body.us.preheader
	{
		r9=asl(r3,#7)
		p0=cmp.eq(r5,#0); if (p0.new) jump:nt .L_qmaximum_BB9_14
		if (!p0.new) r3=#0
	}
// BB#8:
	{
		v9:8.h=vsub(v3.ub,v5.ub)
		r11=#0x80;r10=##0x00800001 //EJP
	}
	{
		v11:10.w=vmpy(v8.h,v1.h)
	}
	{
		v13:12.w=vmpy(v9.h,v1.h)
		v14.h=vasr(v11.w,v10.w,r7)
	}
	{
		v15.h=vasr(v13.w,v12.w,r7)
	}
	{
		v16.ub=vasr(v15.h,v14.h,r3):sat
	}
	{
		v7.ub=vadd(v16.ub,v0.ub):sat
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r8=add(r12,#-16384)
		r28=add(r14,#-1)
		p0=cmp.gtu(r14,#1)
	}
	{
		loop0(.L_qmaximum_BB9_9,r28)
		v15:14.h=vsub(v10.ub,v6.ub)
		r8=add(r12,#128)
		v10.cur=vmem(r8+#0)
	}
	{
		v17:16.w=vmpy(v14.h,v2.h)
	}
	{
		v19:18.w=vmpy(v15.h,v2.h)
		v11.h=vasr(v17.w,v16.w,r7)
	}
	{
		v12.h=vasr(v19.w,v18.w,r7)
	}
	{
		v13.ub=vasr(v12.h,v11.h,r3):sat
	}
	{
		if (!p0) jump:nt .L_qmaximum_BB9_10
		v11.ub=vadd(v13.ub,v0.ub):sat
	}
	.falign
.L_qmaximum_BB9_9:                                // %for.body.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r8,r11:10)
	//# InlineAsm End
	{
		r12=add(r8,#-16384)
		r8=add(r8,#128)
		v14.ub=vmax(v11.ub,v7.ub)
		vmem(r2++#1)=v14.new
	}
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		v21:20.h=vsub(v15.ub,v6.ub)
		v15.cur=vmem(r12+#0)
	}
	{
		v23:22.w=vmpy(v20.h,v2.h)
	}
	{
		v25:24.w=vmpy(v21.h,v2.h)
		v16.h=vasr(v23.w,v22.w,r7)
	}
	{
		v17.h=vasr(v25.w,v24.w,r7)
	}
	{
		v18.ub=vasr(v17.h,v16.h,r3):sat
	}
	{
		nop
		v11.ub=vadd(v18.ub,v0.ub):sat
	}:endloop0
	.falign
.L_qmaximum_BB9_10:
	{
		jump .L_qmaximum_BB9_17
		v19.ub=vmax(v11.ub,v7.ub)
		vmem(r2++#1)=v19.new
	}
	.falign
.L_qmaximum_BB9_11:                               // %for.body.us144.preheader
	{
		v15:14.h=vsub(v4.ub,v6.ub)
		r11=#0x80;r10=##0x00800001 //EJP
	}
	{
		v17:16.w=vmpy(v14.h,v2.h)
	}
	{
		v19:18.w=vmpy(v15.h,v2.h)
		v27.h=vasr(v17.w,v16.w,r7)
	}
	{
		v29.h=vasr(v19.w,v18.w,r7)
	}
	{
		v30.ub=vasr(v29.h,v27.h,r3):sat
	}
	{
		v8.ub=vadd(v30.ub,v0.ub):sat
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r9=add(r13,#-16384)
		r28=add(r14,#-1)
		p0=cmp.gtu(r14,#1)
	}
	{
		loop0(.L_qmaximum_BB9_12,r28)
		v21:20.h=vsub(v31.ub,v5.ub)
		r9=add(r13,#128)
		v31.cur=vmem(r9+#0)
	}
	{
		v23:22.w=vmpy(v20.h,v1.h)
	}
	{
		v25:24.w=vmpy(v21.h,v1.h)
		v7.h=vasr(v23.w,v22.w,r7)
	}
	{
		v9.h=vasr(v25.w,v24.w,r7)
	}
	{
		v10.ub=vasr(v9.h,v7.h,r3):sat
	}
	{
		if (!p0) jump:nt .L_qmaximum_BB9_13
		v12.ub=vadd(v10.ub,v0.ub):sat
	}
	.falign
.L_qmaximum_BB9_12:                               // %for.body.us144
                                        // =>This Inner Loop Header: Depth=1
	{
		r13=add(r9,#-16384)
		r14=add(r9,#128)
		v11.ub=vmax(v8.ub,v12.ub)
		vmem(r2++#1)=v11.new
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r9,r11:10)
	//# InlineAsm End
	{
		v27:26.h=vsub(v12.ub,v5.ub)
		r9=r14
		v12.cur=vmem(r13+#0)
	}
	{
		v29:28.w=vmpy(v26.h,v1.h)
	}
	{
		v31:30.w=vmpy(v27.h,v1.h)
		v13.h=vasr(v29.w,v28.w,r7)
	}
	{
		v14.h=vasr(v31.w,v30.w,r7)
	}
	{
		v15.ub=vasr(v14.h,v13.h,r3):sat
	}
	{
		nop
		v12.ub=vadd(v15.ub,v0.ub):sat
	}:endloop0
	.falign
.L_qmaximum_BB9_13:
	{
		jump .L_qmaximum_BB9_18
		v8.ub=vmax(v8.ub,v12.ub)
		vmem(r2++#1)=v8.new
	}
	.falign
.L_qmaximum_BB9_14:                               // %for.body.us.us.preheader
	{
		r1=#0
		r11=#0x80;r10=##0x00800001 //EJP
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r28=add(r13,#-16384)
		r3=add(r12,#-16384)
		r13=add(r13,#128)
		p1=cmp.gtu(r14,#1)
	}
	{
		v15:14.h=vsub(v17.ub,v6.ub)
		r3=add(r12,#128)
		r12=add(r14,#-1)
		v17.cur=vmem(r3+#0)
	}
	{
		loop0(.L_qmaximum_BB9_15,r12)
		v29:28.h=vsub(v18.ub,v5.ub)
		v18.cur=vmem(r28+#0)
	}
	{
		v21:20.w=vmpy(v15.h,v2.h)
	}
	{
		v31:30.w=vmpy(v14.h,v2.h)
	}
	{
		v23:22.w=vmpy(v28.h,v1.h)
		v20.h=vasr(v21.w,v20.w,r7)
	}
	{
		v25:24.w=vmpy(v29.h,v1.h)
		v19.h=vasr(v31.w,v30.w,r7)
	}
	{
		v29.h=vasr(v23.w,v22.w,r7)
	}
	{
		v30.h=vasr(v25.w,v24.w,r7)
	}
	{
		v31.ub=vasr(v20.h,v19.h,r1):sat
	}
	{
		v24.ub=vasr(v30.h,v29.h,r1):sat
		v10.ub=vadd(v31.ub,v0.ub):sat
	}
	{
		if (!p1) jump:nt .L_qmaximum_BB9_16
		v9.ub=vadd(v24.ub,v0.ub):sat
	}
	.falign
.L_qmaximum_BB9_15:                               // %for.body.us.us
                                        // =>This Inner Loop Header: Depth=1
	{
		r12=add(r13,#-16384)
		r14=add(r13,#128)
		v25.ub=vmax(v10.ub,v9.ub)
		vmem(r2++#1)=v25.new
	}
	//# InlineAsm Start
	 l2fetch(r3,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r13=r14
		r12=add(r3,#-16384)
		v26=vmem(r12+#0)
	}
	{
		v27:26.h=vsub(v26.ub,v5.ub)
		v28=vmem(r12+#0)
	}
	{
		v29:28.h=vsub(v28.ub,v6.ub)
		r3=add(r3,#128)
	}
	{
		v31:30.w=vmpy(v26.h,v1.h)
	}
	{
		v9:8.w=vmpy(v27.h,v1.h)
	}
	{
		v11:10.w=vmpy(v28.h,v2.h)
		v23.h=vasr(v31.w,v30.w,r7)
	}
	{
		v13:12.w=vmpy(v29.h,v2.h)
		v24.h=vasr(v9.w,v8.w,r7)
	}
	{
		v25.h=vasr(v11.w,v10.w,r7)
	}
	{
		v26.h=vasr(v13.w,v12.w,r7)
	}
	{
		v7.ub=vasr(v24.h,v23.h,r1):sat
	}
	{
		v8.ub=vasr(v26.h,v25.h,r1):sat
		v9.ub=vadd(v7.ub,v0.ub):sat
	}
	{
		nop
		v10.ub=vadd(v8.ub,v0.ub):sat
	}:endloop0
	.falign
.L_qmaximum_BB9_16:
	{
		r1=r8
		v9.ub=vmax(v10.ub,v9.ub)
		vmem(r2++#1)=v9.new
	}
	.falign
.L_qmaximum_BB9_17:                               // %for.end.loopexit
	{
		r8=r1
		r0=add(r0,r9)
	}
	.falign
.L_qmaximum_BB9_18:                               // %for.end
	{
		r4=and(#124,asl(r4,#2))
		r2=add(r0,#16384)
		r13=#0x80;
	}
	r12=##0x00800001 //EJP
	//# InlineAsm Start
	 l2fetch(r2,r13:12)
	//# InlineAsm End
	{
		r2=add(r8,#16384)
	}
	//# InlineAsm Start
	 l2fetch(r2,r13:12)
	//# InlineAsm End
	{
		p1=cmp.eq(r6,#0); if (!p1.new) jump:nt .L_qmaximum_BB9_20
	}
// BB#19:                               // %cond.false30
	{
		v4=vmem(r0+#0)
	}
	.falign
.L_qmaximum_BB9_20:                               // %cond.end32
	{
		p0=cmp.eq(r5,#0); if (!p0.new) jump:nt .L_qmaximum_BB9_22
		q0=vsetq(r4)
	}
// BB#21:                               // %cond.false36
	{
		v3=vmem(r8+#0)
	}
	.falign
.L_qmaximum_BB9_22:                               // %cond.end38
	{
		v7:6.h=vsub(v4.ub,v6.ub)
		r6=#0
	}
	{
		v29:28.h=vsub(v3.ub,v5.ub)
	}
	{
		v9:8.w=vmpy(v6.h,v2.h)
	}
	{
		v11:10.w=vmpy(v28.h,v1.h)
	}
	{
		v31:30.w=vmpy(v7.h,v2.h)
		v6.h=vasr(v9.w,v8.w,r7)
	}
	{
		v13:12.w=vmpy(v29.h,v1.h)
		v28.h=vasr(v11.w,v10.w,r7)
	}
	{
		v26.h=vasr(v31.w,v30.w,r7)
	}
	{
		v27.h=vasr(v13.w,v12.w,r7)
	}
	{
		v29.ub=vasr(v26.h,v6.h,r6):sat
	}
	{
		v30.ub=vasr(v27.h,v28.h,r6):sat
		v16.ub=vadd(v29.ub,v0.ub):sat
	}
	{
		v31.ub=vadd(v30.ub,v0.ub):sat
	}
	{
		v17.ub=vmax(v16.ub,v31.ub)
	}
	{
		if (q0) vmem(r15+#0):nt=v17
	}
	{
		jumpr r31
	}
.L_qmaximum_tmp26:                                // Address of block that was removed by CodeGen
.L_qmaximum_tmp27:                                // Address of block that was removed by CodeGen
.L_qmaximum_tmp28:                                // Address of block that was removed by CodeGen
.L_qmaximum_func_end9:
.L_qmaximum_tmp29:
	.size	qmaximum_asm, .L_qmaximum_tmp29-qmaximum_asm
//=================================================================================================================================
    .text
    .file "qelementwise_ops_h.S"
	.globl	qminimum_asm
	.balign 32
	.type	qminimum_asm,@function
qminimum_asm:                           // @qminimum_asm
// BB#0:                                // %entry
	{
		r5|=asl(r5,#8)
		r7=memw(r29+#0)
		r8=memw(r3+#0)
	}
	{
		r8|=asl(r8,#8)
		r7|=asl(r7,#8)
		r9=memw(r3+#4)
		r6=memw(r3+#8)
	}
	{
		r9|=asl(r9,#8)
		r6=combine(r6.l,r6.l)
		r12=memw(r3+#20)
		r13=memw(r3+#12)
	}
	{
		r12|=asl(r12,#8)
		v2=vsplat(r6)
		r13=combine(r13.l,r13.l)
		r6=combine(r5.l,r5.l)
	}
	{
		v1=vsplat(r13)
		r5=combine(r7.l,r7.l)
		r8=combine(r8.l,r8.l)
		r9=combine(r9.l,r9.l)
	}
	{
		v0=vsplat(r12)
		v4=vsplat(r6)
		r13=add(r1,#16384)
		r7=memw(r3+#16)
	}
	{
		v3=vsplat(r5)
		v6=vsplat(r8)
		p1=cmp.eq(r6,#0)
		r12=add(r0,#16384)
	}
	{
		r14=asr(r4,#7)
		v5=vsplat(r9)
		if (!cmp.gt(r14.new,#0)) jump:nt .L_qminimum_BB11_6
	}
// BB#1:                                // %for.body.lr.ph
	{
		r3=extractu(r4,#25,#7)
	}
	{
		r15=addasl(r2,r3,#7)
		r8=addasl(r1,r3,#7)
		if (!p1) r3=#0
	}
	{
		if (p1) jump:nt .L_qminimum_BB11_7
	}
// BB#2:                                // %for.body.lr.ph.split
	{
		p1=cmp.eq(r5,#0); if (p1.new) jump:nt .L_qminimum_BB11_11
		v27:26.h=vsub(v4.ub,v6.ub)
		if (p1.new) r3=#0
	}
// BB#3:
	{
		loop0(.L_qminimum_BB11_4,r14)
		r9=#0x80;r8=##0x00800001 //EJP
	}
	{
		v29:28.h=vsub(v3.ub,v5.ub)
	}
	{
		v31:30.w=vmpy(v26.h,v2.h)
	}
	{
		v9:8.w=vmpy(v27.h,v2.h)
	}
	{
		v11:10.w=vmpy(v28.h,v1.h)
		v20.h=vasr(v31.w,v30.w,r7)
	}
	{
		v13:12.w=vmpy(v29.h,v1.h)
		v21.h=vasr(v9.w,v8.w,r7)
	}
	{
		v22.h=vasr(v11.w,v10.w,r7)
	}
	{
		v23.h=vasr(v13.w,v12.w,r7)
	}
	{
		v24.ub=vasr(v21.h,v20.h,r3):sat
	}
	{
		v25.ub=vasr(v23.h,v22.h,r3):sat
		v23.ub=vadd(v24.ub,v0.ub):sat
	}
	{
		v27.ub=vadd(v25.ub,v0.ub):sat
	}
	{
		v13.ub=vmin(v23.ub,v27.ub)
	}
	.falign
.L_qminimum_tmp31:                                // Block address taken
.L_qminimum_BB11_4:                               // %for.body
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r12,r9:8)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r9:8)
	//# InlineAsm End
	{
		nop
		vmem(r2++#1)=v13
	}:endloop0
// BB#5:                                // %for.end.loopexit167
	{
		jump .L_qminimum_BB11_18
		r8=r1
	}
	.falign
.L_qminimum_BB11_6:
	{
		jump .L_qminimum_BB11_18
		r8=r1
		r15=r2
	}
	.falign
.L_qminimum_BB11_7:                               // %for.body.us.preheader
	{
		r9=asl(r3,#7)
		p0=cmp.eq(r5,#0); if (p0.new) jump:nt .L_qminimum_BB11_14
		if (!p0.new) r3=#0
	}
// BB#8:
	{
		v9:8.h=vsub(v3.ub,v5.ub)
		r11=#0x80;r10=##0x00800001 //EJP
	}
	{
		v11:10.w=vmpy(v8.h,v1.h)
	}
	{
		v13:12.w=vmpy(v9.h,v1.h)
		v14.h=vasr(v11.w,v10.w,r7)
	}
	{
		v15.h=vasr(v13.w,v12.w,r7)
	}
	{
		v16.ub=vasr(v15.h,v14.h,r3):sat
	}
	{
		v7.ub=vadd(v16.ub,v0.ub):sat
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r8=add(r12,#-16384)
		r28=add(r14,#-1)
		p0=cmp.gtu(r14,#1)
	}
	{
		loop0(.L_qminimum_BB11_9,r28)
		v15:14.h=vsub(v10.ub,v6.ub)
		r8=add(r12,#128)
		v10.cur=vmem(r8+#0)
	}
	{
		v17:16.w=vmpy(v14.h,v2.h)
	}
	{
		v19:18.w=vmpy(v15.h,v2.h)
		v11.h=vasr(v17.w,v16.w,r7)
	}
	{
		v12.h=vasr(v19.w,v18.w,r7)
	}
	{
		v13.ub=vasr(v12.h,v11.h,r3):sat
	}
	{
		if (!p0) jump:nt .L_qminimum_BB11_10
		v11.ub=vadd(v13.ub,v0.ub):sat
	}
	.falign
.L_qminimum_BB11_9:                               // %for.body.us
                                        // =>This Inner Loop Header: Depth=1
	//# InlineAsm Start
	 l2fetch(r8,r11:10)
	//# InlineAsm End
	{
		r12=add(r8,#-16384)
		r8=add(r8,#128)
		v14.ub=vmin(v11.ub,v7.ub)
		vmem(r2++#1)=v14.new
	}
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		v21:20.h=vsub(v15.ub,v6.ub)
		v15.cur=vmem(r12+#0)
	}
	{
		v23:22.w=vmpy(v20.h,v2.h)
	}
	{
		v25:24.w=vmpy(v21.h,v2.h)
		v16.h=vasr(v23.w,v22.w,r7)
	}
	{
		v17.h=vasr(v25.w,v24.w,r7)
	}
	{
		v18.ub=vasr(v17.h,v16.h,r3):sat
	}
	{
		nop
		v11.ub=vadd(v18.ub,v0.ub):sat
	}:endloop0
	.falign
.L_qminimum_BB11_10:
	{
		jump .L_qminimum_BB11_17
		v19.ub=vmin(v11.ub,v7.ub)
		vmem(r2++#1)=v19.new
	}
	.falign
.L_qminimum_BB11_11:                              // %for.body.us144.preheader
	{
		v15:14.h=vsub(v4.ub,v6.ub)
		r11=#0x80;r10=##0x00800001 //EJP
	}
	{
		v17:16.w=vmpy(v14.h,v2.h)
	}
	{
		v19:18.w=vmpy(v15.h,v2.h)
		v27.h=vasr(v17.w,v16.w,r7)
	}
	{
		v29.h=vasr(v19.w,v18.w,r7)
	}
	{
		v30.ub=vasr(v29.h,v27.h,r3):sat
	}
	{
		v8.ub=vadd(v30.ub,v0.ub):sat
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r9=add(r13,#-16384)
		r28=add(r14,#-1)
		p0=cmp.gtu(r14,#1)
	}
	{
		loop0(.L_qminimum_BB11_12,r28)
		v21:20.h=vsub(v31.ub,v5.ub)
		r9=add(r13,#128)
		v31.cur=vmem(r9+#0)
	}
	{
		v23:22.w=vmpy(v20.h,v1.h)
	}
	{
		v25:24.w=vmpy(v21.h,v1.h)
		v7.h=vasr(v23.w,v22.w,r7)
	}
	{
		v9.h=vasr(v25.w,v24.w,r7)
	}
	{
		v10.ub=vasr(v9.h,v7.h,r3):sat
	}
	{
		if (!p0) jump:nt .L_qminimum_BB11_13
		v12.ub=vadd(v10.ub,v0.ub):sat
	}
	.falign
.L_qminimum_BB11_12:                              // %for.body.us144
                                        // =>This Inner Loop Header: Depth=1
	{
		r13=add(r9,#-16384)
		r14=add(r9,#128)
		v11.ub=vmin(v8.ub,v12.ub)
		vmem(r2++#1)=v11.new
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r9,r11:10)
	//# InlineAsm End
	{
		v27:26.h=vsub(v12.ub,v5.ub)
		r9=r14
		v12.cur=vmem(r13+#0)
	}
	{
		v29:28.w=vmpy(v26.h,v1.h)
	}
	{
		v31:30.w=vmpy(v27.h,v1.h)
		v13.h=vasr(v29.w,v28.w,r7)
	}
	{
		v14.h=vasr(v31.w,v30.w,r7)
	}
	{
		v15.ub=vasr(v14.h,v13.h,r3):sat
	}
	{
		nop
		v12.ub=vadd(v15.ub,v0.ub):sat
	}:endloop0
	.falign
.L_qminimum_BB11_13:
	{
		jump .L_qminimum_BB11_18
		v8.ub=vmin(v8.ub,v12.ub)
		vmem(r2++#1)=v8.new
	}
	.falign
.L_qminimum_BB11_14:                              // %for.body.us.us.preheader
	{
		r1=#0
		r11=#0x80;r10=##0x00800001 //EJP
	}
	//# InlineAsm Start
	 l2fetch(r12,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r28=add(r13,#-16384)
		r3=add(r12,#-16384)
		r13=add(r13,#128)
		p1=cmp.gtu(r14,#1)
	}
	{
		v15:14.h=vsub(v17.ub,v6.ub)
		r3=add(r12,#128)
		r12=add(r14,#-1)
		v17.cur=vmem(r3+#0)
	}
	{
		loop0(.L_qminimum_BB11_15,r12)
		v29:28.h=vsub(v18.ub,v5.ub)
		v18.cur=vmem(r28+#0)
	}
	{
		v21:20.w=vmpy(v15.h,v2.h)
	}
	{
		v31:30.w=vmpy(v14.h,v2.h)
	}
	{
		v23:22.w=vmpy(v28.h,v1.h)
		v20.h=vasr(v21.w,v20.w,r7)
	}
	{
		v25:24.w=vmpy(v29.h,v1.h)
		v19.h=vasr(v31.w,v30.w,r7)
	}
	{
		v29.h=vasr(v23.w,v22.w,r7)
	}
	{
		v30.h=vasr(v25.w,v24.w,r7)
	}
	{
		v31.ub=vasr(v20.h,v19.h,r1):sat
	}
	{
		v24.ub=vasr(v30.h,v29.h,r1):sat
		v10.ub=vadd(v31.ub,v0.ub):sat
	}
	{
		if (!p1) jump:nt .L_qminimum_BB11_16
		v9.ub=vadd(v24.ub,v0.ub):sat
	}
	.falign
.L_qminimum_BB11_15:                              // %for.body.us.us
                                        // =>This Inner Loop Header: Depth=1
	{
		r12=add(r13,#-16384)
		r14=add(r13,#128)
		v25.ub=vmin(v10.ub,v9.ub)
		vmem(r2++#1)=v25.new
	}
	//# InlineAsm Start
	 l2fetch(r3,r11:10)
	//# InlineAsm End
	//# InlineAsm Start
	 l2fetch(r13,r11:10)
	//# InlineAsm End
	{
		r13=r14
		r12=add(r3,#-16384)
		v26=vmem(r12+#0)
	}
	{
		v27:26.h=vsub(v26.ub,v5.ub)
		v28=vmem(r12+#0)
	}
	{
		v29:28.h=vsub(v28.ub,v6.ub)
		r3=add(r3,#128)
	}
	{
		v31:30.w=vmpy(v26.h,v1.h)
	}
	{
		v9:8.w=vmpy(v27.h,v1.h)
	}
	{
		v11:10.w=vmpy(v28.h,v2.h)
		v23.h=vasr(v31.w,v30.w,r7)
	}
	{
		v13:12.w=vmpy(v29.h,v2.h)
		v24.h=vasr(v9.w,v8.w,r7)
	}
	{
		v25.h=vasr(v11.w,v10.w,r7)
	}
	{
		v26.h=vasr(v13.w,v12.w,r7)
	}
	{
		v7.ub=vasr(v24.h,v23.h,r1):sat
	}
	{
		v8.ub=vasr(v26.h,v25.h,r1):sat
		v9.ub=vadd(v7.ub,v0.ub):sat
	}
	{
		nop
		v10.ub=vadd(v8.ub,v0.ub):sat
	}:endloop0
	.falign
.L_qminimum_BB11_16:
	{
		r1=r8
		v9.ub=vmin(v10.ub,v9.ub)
		vmem(r2++#1)=v9.new
	}
	.falign
.L_qminimum_BB11_17:                              // %for.end.loopexit
	{
		r8=r1
		r0=add(r0,r9)
	}
	.falign
.L_qminimum_BB11_18:                              // %for.end
	{
		r4=and(#124,asl(r4,#2))
		r2=add(r0,#16384)
		r13=#0x80;
	}
	r12=##0x00800001 //EJP
	//# InlineAsm Start
	 l2fetch(r2,r13:12)
	//# InlineAsm End
	{
		r2=add(r8,#16384)
	}
	//# InlineAsm Start
	 l2fetch(r2,r13:12)
	//# InlineAsm End
	{
		p1=cmp.eq(r6,#0); if (!p1.new) jump:nt .L_qminimum_BB11_20
	}
// BB#19:                               // %cond.false30
	{
		v4=vmem(r0+#0)
	}
	.falign
.L_qminimum_BB11_20:                              // %cond.end32
	{
		p0=cmp.eq(r5,#0); if (!p0.new) jump:nt .L_qminimum_BB11_22
		q0=vsetq(r4)
	}
// BB#21:                               // %cond.false36
	{
		v3=vmem(r8+#0)
	}
	.falign
.L_qminimum_BB11_22:                              // %cond.end38
	{
		v7:6.h=vsub(v4.ub,v6.ub)
		r6=#0
	}
	{
		v29:28.h=vsub(v3.ub,v5.ub)
	}
	{
		v9:8.w=vmpy(v6.h,v2.h)
	}
	{
		v11:10.w=vmpy(v28.h,v1.h)
	}
	{
		v31:30.w=vmpy(v7.h,v2.h)
		v6.h=vasr(v9.w,v8.w,r7)
	}
	{
		v13:12.w=vmpy(v29.h,v1.h)
		v28.h=vasr(v11.w,v10.w,r7)
	}
	{
		v26.h=vasr(v31.w,v30.w,r7)
	}
	{
		v27.h=vasr(v13.w,v12.w,r7)
	}
	{
		v29.ub=vasr(v26.h,v6.h,r6):sat
	}
	{
		v30.ub=vasr(v27.h,v28.h,r6):sat
		v16.ub=vadd(v29.ub,v0.ub):sat
	}
	{
		v31.ub=vadd(v30.ub,v0.ub):sat
	}
	{
		v17.ub=vmin(v16.ub,v31.ub)
	}
	{
		if (q0) vmem(r15+#0):nt=v17
	}
	{
		jumpr r31
	}
.L_qminimum_tmp32:                                // Address of block that was removed by CodeGen
.L_qminimum_tmp33:                                // Address of block that was removed by CodeGen
.L_qminimum_tmp34:                                // Address of block that was removed by CodeGen
.L_qminimum_func_end11:
.L_qminimum_tmp35:
	.size	qminimum_asm, .L_qminimum_tmp35-qminimum_asm


