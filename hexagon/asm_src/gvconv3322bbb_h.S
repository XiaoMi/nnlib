    .text
    .file "gvconv3322bbb_h.S"
    .global conv3322bbb
    .balign 64
    .type  conv3322bbb, @function
conv3322bbb:
/* =========================================================================== */
#define ptr_xi                  r0       //data
#define ptr_wi                  r1       //weights
#define ptr_bias                r2       //bias
#define ptr_zi                  r3       //results
#define out_width               r4       //  amount of work to be done
#define out_height              r5       //  number of vertical lines to perform
#define recip_level             r6       //  
#define zshift                  r7       //  spare input
#define zshift_recip_level      r7:6
#define filt_offset             r8       // 
#define in_next_row             r9       //
#define in_next_row_filt_offset r9:8     //
//-----------------------------------------------------------------
#define f0d0_01                 r10
#define f0d0_x2                 r11
#define f0d0_x210               r11:10
#define f0d1_01                 r12
#define f0d1_x2                 r13
#define f0d1_x210               r13:12
#define f1d0_01                 r14
#define f1d0_x2                 r15
#define f1d0_x210               r15:14
#define f1d1_01                 r16
#define f1d1_x2                 r17
#define f1d1_x210               r17:16
#define f2d0_01                 r18
#define f2d0_x2                 r19
#define f2d0_x210               r19:18
#define f2d1_01                 r20
#define f2d1_x2                 r21
#define f2d1_x210               r21:20
#define f0d0_2x                 r22
#define f0d1_2x                 r23
#define f1d0_2x                 r24
#define f1d1_2x                 r25
#define f2d0_2x                 r26
#define f2d1_2x                 r27
#define lpcount0                r28
#define iptr0                   ptr_wi
#define optr                    ptr_bias
#define bias0                   f0d0_01
#define bias1                   f0d0_x2
#define bias1_bias0             f0d0_x210
#define inc                     f0d1_01
#define n_out_width             recip_level
//-----------------------------------------------------------------
#define x00                     v0
#define x02                     v1
#define x02x00                  v1:0
#define x10                     v2
#define x12                     v3
#define x12x10                  v3:2
#define x20                     v4
#define x22                     v5
#define x22x20                  v5:4
#define x0n                     v6
#define x1n                     v7
#define x2n                     v8
#define vrecip                  v9   
#define s00                     v10
#define s01                     v11
#define s01s00                  v11:10
#define s10                     v12
#define s11                     v13
#define s11s10                  v13:12
#define sa0                     v14
#define sa1                     v15
#define sa1sa0                  v15:14
#define sum00                   v16
#define sum01                   v17
#define sum01sum00              v17:16
#define sum10                   v18
#define sum11                   v19
#define sum11sum10              v19:18
#define d00                     v20
#define d01                     v21
#define d10                     v22
#define d11                     v23
#define vbias0                  v24
#define vbias1                  v25
#define zeros                   v26
#define ones                    v27
#define mask                    v27
#define mask0                   v28
#define mask1                   v29
#define yout_a                  d10
#define yout_b                  d11
#define y0                      d00
#define y1                      d01 
#define yout                    y0
//-----------------------------------------------------------------
#define SSR        <<1:rnd:sat:shift  //simplfy mpy instruction
/*=============================================================================*/
    {   zshift_recip_level = memd(sp+#0)                //
        in_next_row_filt_offset = memd(sp+#8)           //
        lpcount0 = add(out_width,#63)                   //
        out_width = asl(out_width,#1)                   //
    } { 
        bias1_bias0 = memd(ptr_bias+#0)                 //
        sp = add(sp,#-6*8)                              //
        inc = #128                                      //
        zeros = #0                                      //
    } {
        memd(sp+#0 ) = r17:16                           //
        memd(sp+#8 ) = r19:18                           //
        inc -= asl(in_next_row,#1)
        m0 = in_next_row
    } {
        memd(sp+#16) = r21:20                           //
        memd(sp+#24) = r23:22                           //
        lpcount0 = lsr(lpcount0,#6)
        m1 = inc
    } {
        memd(sp+#32) = r25:24                           //
        memd(sp+#40) = r27:26                           //
        filt_offset = vsplatb(filt_offset)              //
        vrecip = vsplat(recip_level)                    //
    } {
        n_out_width = neg(out_width)                    //
        vbias0 = vsplat(bias0)                          //
        vbias1 = vsplat(bias1)                          //
        ones = vnot(zeros)                              //
    } {
        f0d0_x210 = memd(ptr_wi+#0)                     //
        f0d1_x210 = memd(ptr_wi+#8)                     //
        loop1(.L_height,out_height)                     //
        mask = valign(zeros,ones,n_out_width)           //
    } {
        f1d1_x210 = memd(ptr_wi+#24)                    //
        f1d0_x210 = memd(ptr_wi+#16)                    //
        f0d0_2x = combine(f0d0_x2.l,f0d0_x2.h)          //
        f0d1_2x = combine(f0d1_x2.l,f0d1_x2.h)          //
    } {
        f2d0_x210 = memd(ptr_wi+#32)                    //
        f2d1_x210 = memd(ptr_wi+#40)                    //
        f1d0_2x = combine(f1d0_x2.l,f1d0_x2.h)          //
        f1d1_2x = combine(f1d1_x2.l,f1d1_x2.h)          //
    } {
        f2d0_2x = combine(f2d0_x2.l,f2d0_x2.h)          //
        f2d1_2x = combine(f2d1_x2.l,f2d1_x2.h)          //
        iptr0 = ptr_xi
    } {
        nop; nop; nop                                   //
    }

.L_height:
    {   x00 = vmem(iptr0++m0)                           //
        optr = ptr_zi                                   //
        p3 = sp1loop0(.L_width,lpcount0)                //
        nop                                             //
    } {
        x10 = vmem(iptr0++m0)                           //
        ptr_zi = add(ptr_zi,out_width)                  //
        ptr_xi = add(ptr_xi,in_next_row)                //
    } {
        x2n = vmem(iptr0++m1)                           //
    }
    .balign 64
.L_width:
    {   x0n.cur = vmem(iptr0++m0)                       //[1]
        x02 = valign(x0n,x00,#4)                        //[1]
        s00.uw += vrmpy(x22.ub,f2d0_x2.ub)              //[2]
        s01.uw += vrmpy(x22.ub,f2d0_2x.ub)              //[2]
    } {
        s10.uw += vrmpy(x22.ub,f2d1_x2.ub)              //[2]
        s11.uw += vrmpy(x22.ub,f2d1_2x.ub)              //[2]
        sum01sum00.w = vsub(s01s00.w,sa1sa0.w)          //[2]
    } {
        sa1sa0.uw  = vrmpy(x02x00.ub,filt_offset.ub,#0) //[1]
        sum11sum10.w = vsub(s11s10.w,sa1sa0.w)          //[2]
    } {
        x1n.cur = vmem(iptr0++m0)                       //[1]
        x12 = valign(x1n,x10,#4)                        //[1]
        s01s00.uw  = vrmpy(x02x00.ub,f0d0_01.ub,#0)     //[1]
    } {
        s11s10.uw  = vrmpy(x02x00.ub,f0d1_01.ub,#0)     //[1]
        x20 = x2n                                       //[1]
        sum00.w = vasl(sum00.w,zshift)                  //[2]
    } {
        sa0.uw += vrmpy(x02.ub,filt_offset.ub)          //[1]
        sa1.uw += vrmpy(x02.ub,filt_offset.ub)          //[1]
        sum01.w = vasl(sum01.w,zshift)                  //[2]
    } {
        sum10.w = vasl(sum10.w,zshift)                  //[2]
        d00.w = vmpye(sum00.w, vrecip.uh)               //[2]
    } {
        sum11.w = vasl(sum11.w,zshift)                  //[2]
        d01.w = vmpye(sum01.w, vrecip.uh)               //[2]
    } {
        x2n.cur = vmem(iptr0++m1)                       //[1]
        x22 = valign(x2n,x20,#4)                        //[1]
        d10.w = vmpye(sum10.w, vrecip.uh)               //[2]
    } {
        s00.w = vadd(s00.w,vbias0.w)                    //[1]
        s01.w = vadd(s01.w,vbias0.w)                    //[1]
        d11.w = vmpye(sum11.w, vrecip.uh)               //[2]
    } {
        s10.w = vadd(s10.w,vbias1.w)                    //[1]
        s11.w = vadd(s11.w,vbias1.w)                    //[1]
        d00.w += vmpyo(sum00.w, vrecip.h):SSR           //[2]
    } {
        d01.w += vmpyo(sum01.w, vrecip.h):SSR           //[2]
    } {
        d10.w += vmpyo(sum10.w, vrecip.h):SSR           //[2]
    } {
        d11.w += vmpyo(sum11.w, vrecip.h):SSR           //[2]
    } {
        s00.uw += vrmpy(x02.ub,f0d0_x2.ub)              //[1]
        s01.uw += vrmpy(x02.ub,f0d0_2x.ub)              //[1]
        y0.h = vsat(d01.w,d00.w)                        //[2]
    } {
        s10.uw += vrmpy(x02.ub,f0d1_x2.ub)              //[1]
        s11.uw += vrmpy(x02.ub,f0d1_2x.ub)              //[1]
        y1.h = vsat(d11.w,d10.w)                        //[2]
    } {
        sa1sa0.uw += vrmpy(x12x10.ub,filt_offset.ub,#0) //[1]
        x00 = x0n                                       //[1]
    } {
        s01s00.uw += vrmpy(x12x10.ub,f1d0_01.ub,#0)     //[1]
        yout.ub = vsat(y1.h,y0.h)                       //[2]
    } {
        s11s10.uw += vrmpy(x12x10.ub,f1d1_01.ub,#0)     //[1]
    } {
        sa0.uw += vrmpy(x12.ub,filt_offset.ub)          //[1]
        sa1.uw += vrmpy(x12.ub,filt_offset.ub)          //[1]
    } {
        s00.uw += vrmpy(x12.ub,f1d0_x2.ub)              //[1]
        s01.uw += vrmpy(x12.ub,f1d0_2x.ub)              //[1]
    } {
        s10.uw += vrmpy(x12.ub,f1d1_x2.ub)              //[1]
        s11.uw += vrmpy(x12.ub,f1d1_2x.ub)              //[1]
        x10 = x1n                                       //[1]
    } {
        sa1sa0.uw += vrmpy(x22x20.ub,filt_offset.ub,#0) //[1]
        if p3 vmemu(optr++#1) = yout                    //[2]
    } {
        s01s00.uw += vrmpy(x22x20.ub,f2d0_01.ub,#0)     //[1]
    } {
        s11s10.uw += vrmpy(x22x20.ub,f2d1_01.ub,#0)     //[1]
    } {
        sa0.uw += vrmpy(x22.ub,filt_offset.ub)          //[1]
        sa1.uw += vrmpy(x22.ub,filt_offset.ub)          //[1]
    }:endloop0
/* --------------------------------------------------------------------------- */
.L_width_end:
    {   s00.uw += vrmpy(x22.ub,f2d0_x2.ub)              //[2]
        s01.uw += vrmpy(x22.ub,f2d0_2x.ub)              //[2]
    } {
        s10.uw += vrmpy(x22.ub,f2d1_x2.ub)              //[2]
        s11.uw += vrmpy(x22.ub,f2d1_2x.ub)              //[2]
        sum01sum00.w = vsub(s01s00.w,sa1sa0.w)          //[2]
    } {
        sum11sum10.w = vsub(s11s10.w,sa1sa0.w)          //[2]
    } {
        sum00.w = vasl(sum00.w,zshift)                  //[2]
    } {
        sum01.w = vasl(sum01.w,zshift)                  //[2]
    } {
        sum10.w = vasl(sum10.w,zshift)                  //[2]
        d00.w = vmpye(sum00.w, vrecip.uh)               //[2]
    } {
        sum11.w = vasl(sum11.w,zshift)                  //[2]
        d01.w = vmpye(sum01.w, vrecip.uh)               //[2]
    } {
        d10.w = vmpye(sum10.w, vrecip.uh)               //[2]
    } {
        d11.w = vmpye(sum11.w, vrecip.uh)               //[2]
    } {
        d00.w += vmpyo(sum00.w, vrecip.h):SSR           //[2]
    } {
        d01.w += vmpyo(sum01.w, vrecip.h):SSR           //[2]
    } {
        d10.w += vmpyo(sum10.w, vrecip.h):SSR           //[2]
        mask0 = vlalign(mask,zeros,optr)                //
    } {
        d11.w += vmpyo(sum11.w, vrecip.h):SSR           //[2]
        mask1 = vlalign(zeros,mask,optr)                //
    } {
        y0.h = vsat(d01.w,d00.w)                        //[2]
        iptr0 = ptr_xi                                  //
    } {
        y1.h = vsat(d11.w,d10.w)                        //[2]
        q1 = vcmp.gt(mask0.ub,zeros.ub)                 //
    } {
        yout.ub = vsat(y1.h,y0.h)                       //[2]
        q2 = vcmp.gt(mask1.ub,zeros.ub)                 //
    } {
        yout_a = vlalign(yout,zeros,optr)               //[2]
    } {
        yout_b = vlalign(zeros,yout,optr)               //[2]
        if (q1) vmem(optr++#1) = yout_a                 //[2]
    } {
        if (q2) vmem(optr+#0)  = yout_b                 //[2]
    }:endloop1
/* --------------------------------------------------------------------------- */
    {   r27:26 = memd(sp+#40)                           //
        r25:24 = memd(sp+#32)                           //
    } {
        r23:22 = memd(sp+#24)                           //
        r21:20 = memd(sp+#16)                           //
    } {
        r19:18 = memd(sp+#8 )                           //
        r17:16 = memd(sp+#0 )                           //
        sp = add(sp,#6*8)                               // pop stack
        jumpr r31                                       //return
    }
/*=============================================================================*/
.L_end:
    .size conv3322bbb, .L_end-conv3322bbb
/*=============================================================================*/

     .text
     .global load_indata_d2
     .balign 64
     .type  load_indata_d2, @function
load_indata_d2:
/* =========================================================================== */
#define indata                  r0
#define in_width                r1
#define next_row                r2
#define left_pad                r3
#define right_pad               r4
#define pad_value               r5
#define out                     r6
#define nbytes                  r7
#define nbytes_out              r7:6
//-----------------------------------------------------------------
#define iteration               r8
#define rpad_pos                r9
#define out_r                   r10
#define width_roundup           r11
#define range                   r12
#define rpad_pos2               r13
#define outm                    r14
//-----------------------------------------------------------------
#define x0                      v0
#define x1                      v1
#define xout                    v2
#define vpad                    v3
/*=============================================================================*/
    {   nbytes_out = memd(sp+#0)                        //
        range = addasl(indata,in_width,#1)              //
        pad_value = vsplatb(pad_value)                  //
        rpad_pos = add(in_width,left_pad)               //
    } {
        range = add(range,#127)                         //
        vpad = vsplat(pad_value)                        //
        width_roundup = add(in_width,#63)               //
        p3 = cmp.gt(right_pad,#0)                       //
    } {
        range = and(range,#-128)                        //
        rpad_pos2 = asl(rpad_pos,#1)                    //
        width_roundup = and(width_roundup,#-64)         //
        x0 = vpad                                       //
    } {
        range = sub(range,indata)                       //
        q0 = vsetq(rpad_pos2)                           //
        out_r = add(out,rpad_pos2)                      //
        iteration = lsr(width_roundup,#6)               //
    } {
        p0 = cmp.gt(range,nbytes)                       //
        if p0.new iteration = add(iteration,#-1)        //
        if p0.new width_roundup= add(width_roundup,#-64)//
        left_pad = asl(left_pad,#1)                     //
    } {
        p1 = cmp.gt(rpad_pos,width_roundup)             //
        loop0(.L_loop,iteration)                        //
        outm = add(out,left_pad)                        //
        p2 = cmp.gt(iteration,#0)                       //
    } {
        if p0 in_width = sub(in_width,width_roundup)    //
        if !p2 jump .L_loopend                          //
        outm += asl(width_roundup,#1)                   //
    }
//-----------------------------------------------------------------
     .balign 64
.L_loop:
    {   x1 = vmemu(indata++#1)                          //
    } {
        xout = vlalign(x1,x0,left_pad)                  //
        vmem(out++#1) = xout.new                        //
        x0 = x1                                         //
    }:endloop0
//-----------------------------------------------------------------
.L_loopend:
    {   xout = vlalign(x1,x0,left_pad)                  //
        if (p1) vmem(out++#1) = xout.new                //
        if !p3 jump .L_continue                         //
    } {
        if (!q0) vmem(out_r+#0) = vpad                  //
    } 
.L_continue:
    {   if !p0 jumpr r31                                //
    } {
        r0 = outm                                       //
        r1 = indata                                     //
        r2 = asl(in_width,#1)                           //
        jump memcpy                                     //  
    } 
/*=============================================================================*/
    .size load_indata_d2, .-load_indata_d2
/*=============================================================================*/
