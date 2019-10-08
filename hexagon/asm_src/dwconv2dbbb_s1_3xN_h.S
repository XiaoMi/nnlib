/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dbbb_s1_3xN_h.S"
       .global dwconv2dbbb_s1_3xN_asm
       .balign 32
       .type  dwconv2dbbb_s1_3xN_asm, @function
dwconv2dbbb_s1_3xN_asm:
/* ----------------------------------------------------------------------------- */
#define   in_buf               r0
#define   filt                 r1
#define   out_buf              r2
#define   next_in_width_depth  r3
#define   next_out_width_depth r4     //currently unused
#define   next_in_width_32     r5
#define   next_out_width_32    r10
#define   depth                r11
#define   out_width            r12
#define   out_height           r13
#define   width_count          r27
#define   filt_height          r25
#define   filt_zero            r7 
#define   bias_sum             r14
#define   ptr_max              r15
#define   recip_level          r27
#define   recip_shift          r8
#define   stride_v             r28  
#define   ptr_w0               r16
#define   ptr_w1               r17
#define   c8                   r6
#define   bias_ptr             r18
#define   ptr_x0               r19
#define   ptr_x1               r20
#define   ptr_xin              r22
#define   ptr_y                r23
#define   depth_cnt            r26
#define   filt_size            r9
#define   next_in_width_depth_stride r28
#define   zzzz                 r7      //111111111111
#define   _zzz                 r21     //111111111111
#define   zzz_                 r4      //111111111111
#define   nort                 r24

#define   vrecip                v0
#define   vshamt_vec            v1
#define   maxv                  v2
#define   minv                  v3
#define   bias_val              v4 
#define   x0                    v8     //___11_______
#define   x1                   v29     //_111________
#define   w_210                 v9     //_______111__
#define   w210_                v28     //__________11
#define   x7x5x6x4             v29     //____11______
#define   x7x6x5x4             v10     //111111111111
#define   x3x1x2x0             v30     //_____11_____
#define   x3x2x1x0             v11     //111111111111
#define   x3x2x3x2             v6      //111111111111
#define   x5x4x3x2             v7      //111111111111
#define   s0                   v12     //111111111111
#define   s1                   v13     //111111111111
#define   s2                   v14     //111111111111
#define   s3                   v15     //111111111111
#define   z0                   v16     //111111111111
#define   z1                   v17     //111111111111
#define   z2                   v18     //111111111111
#define   z3                   v19     //111111111111
#define   d0                   v20     //
#define   d1                   v21     //
#define   d1d0                 v24     //
#define   d2                   v22     //
#define   d3                   v23     //
#define   d3d2                 v25     //
#define   d3210                v25     //
#define SSR        <<1:rnd:sat:shift   //simplfy mpy instruction
#define PV(VSRC) .word (0x1DFFE020+VSRC)
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca reg
/* =========================================================================== */
   {  allocframe(#56)                              //0th entry on stack (56+8)/4=20
      c8 = #8
   } {
      memd(sp+#0)  = r17:16
      memd(sp+#8)  = r19:18
   } {
      memd(sp+#16) = r21:20
      memd(sp+#24) = r23:22
   } {
      memd(sp+#32) = r25:24
      memd(sp+#40) = r27:26
   } {
      next_out_width_32 = memw(sp+#16<<2)  //
      depth  = memw(sp+#17<<2)             //
      nop;
      nop;
   } {
      nort = #0
      out_width = memw(sp+#18<<2)          //
      out_height = memw(sp+#19<<2)         //
      depth = lsr(depth, #5)               //depth/32
   } {
      filt_height = memw(sp+#20<<2)         //
      filt_zero = memw(sp+#21<<2)         //
      out_width = add(out_width, #3)
      nop;
   } {
      bias_sum = memw(sp+#22<<2)           //
      ptr_max = memw(sp+#23<<2)            //
      out_width = lsr(out_width, #2)
      filt_size = asl(filt_height, #7)
   } {
      recip_shift = memw(sp+#25<<2)        //
      zzzz  = vsplatb(filt_zero)
      p3 = cmp.eq(filt_height, #2)
   } {
      memw(sp+#17<<2) = depth              //
      vshamt_vec= vsplat(recip_shift)
      stride_v = memw(sp+#26<<2)         //
   } {
      next_in_width_depth_stride = mpyi(next_in_width_depth, stride_v)
      maxv = vmem(ptr_max+#0)
      depth_cnt = memw(sp+#17<<2)                  //depth
      filt_height = add(filt_height, #-2)
   } {
      minv = vmem(ptr_max+#1)
      zzz_  = asl(zzzz, c8)
      _zzz  = lsr(zzzz, c8)
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {
      recip_level = memw(sp+#24<<2)                //
   } {  
      bias_ptr = bias_sum                          //
      ptr_xin = in_buf
      ptr_w0 = filt
      filt_height = max(filt_height, nort)
   }
/* ----------------------------------------------------------------------------- */
.L_depth:

   {  x0 = vmemu(ptr_xin+#0)                       //[1, 0]
      ptr_x0 = ptr_xin                             //
   } {
      x1 = vmemu(ptr_x0+#1)                        //[0, 0]
      ptr_y = out_buf                              //
   } {
      vrecip = vmem(recip_level++#1)               //
   } {
      x3x1x2x0.b = vshuff(x0.b)                    //[0, 1]
      p2 = !cmp.eq(r0,r0)                          //[WIDTH]
      bias_val = vmem(bias_ptr++#1)                //
      memw(sp+#52) = recip_level                   //quantization level save to temp location
   } {
      ptr_x1 = add(ptr_x0, next_in_width_depth)    //[0, 1]
      x7x5x6x4.b = vshuff(x1.b)                    //[0, 2]
      loop1(.L_width, out_width)
      width_count = memw(sp+#18<<2)                //
   } {
      width_count = add(width_count, #-4)          //
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[0, 3]
      loop0(.L_vert, filt_height)                  //[WIDTH, P]
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
   {  
      w_210.cur = vmem(ptr_w0+#0)                  //[0, 2]
      w210_.w = vasl(w_210.w, c8)                  //[0, 5]
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[0, 5]
      ptr_w1 = add(ptr_w0, #128)                   //[WIDTH, P]
   } {
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[0, 4]
      x0 = vmemu(ptr_x1+#0)                        //[0, 4]
      z0.uw = vrmpy(x3x2x1x0.ub, _zzz.ub)          //[0, 9]filter even output
   } {
      s0.uw = vrmpy(x3x2x1x0.ub, w_210.ub)         //[0, 6]filter even output
      x5x4x3x2.h = vshuffe(x7x6x5x4.h,x3x2x3x2.h)  //[0, 6]
      x1 = vmemu(ptr_x1+#1)                        //[1, 0]
   } {
      s1.uw = vrmpy(x3x2x1x0.ub, w210_.ub)         //[0, 7]filter even output
      x3x1x2x0.b = vshuff(x0.b)                    //[1, 1]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[1, 1]
      s3 = bias_val                                //[WIDTH, P]
   } {
      s2.uw = vrmpy(x5x4x3x2.ub, w_210.ub)         //[0, 8]filter even output
      x7x5x6x4.b = vshuff(x1.b)                    //[1, 2]
      w_210 = vmem(ptr_w1++#1)                     //[1, 2]
      ptr_x0 = add(ptr_x0, #128)                   //[WIDTH,P]+4 32 depths for stride 1
   } {
      z1.uw = vrmpy(x3x2x1x0.ub, zzz_.ub)          //[0, 9]filter even output
      z2.uw = vrmpy(x5x4x3x2.ub, _zzz.ub)          //[0,11]filter even output
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[1, 3]
      s2.w = vadd(bias_val.w, s2.w)                //[WIDTH, P]
   } {
      s3.uw += vrmpy(x5x4x3x2.ub, w210_.ub)        //[0,10]filter even output
      x0 = vmemu(ptr_x1+#0)                        //[1, 4]
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[1, 4]
   } {
      z3.uw = vrmpy(x5x4x3x2.ub, zzz_.ub)          //[0,11]filter even output
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[1, 5]
      w210_.w = vasl(w_210.w, c8)                  //[1, 5]
      if(p3) jump .L_skip                          //
   }
     .balign 32
.L_vert:
   {  s0.uw += vrmpy(x3x2x1x0.ub, w_210.ub)        //[0, 6]filter even output
      x5x4x3x2.h = vshuffe(x7x6x5x4.h,x3x2x3x2.h)  //[0, 6]
      x1 = vmemu(ptr_x1+#1)                        //[1, 0]
   } {
      s1.uw += vrmpy(x3x2x1x0.ub, w210_.ub)        //[0, 7]filter even output
      x3x1x2x0.b = vshuff(x0.b)                    //[1, 1]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[1, 1]
      nop;
   } {
      s2.uw += vrmpy(x5x4x3x2.ub, w_210.ub)        //[0, 8]filter even output
      x7x5x6x4.b = vshuff(x1.b)                    //[1, 2]
      w_210 = vmem(ptr_w1++#1)                     //[1, 2]
      nop;
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, _zzz.ub)         //[0, 9]filter even output
      z1.uw += vrmpy(x3x2x1x0.ub, zzz_.ub)         //[0, 9]filter even output
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[1, 3]
      nop;
   } {
      s3.uw += vrmpy(x5x4x3x2.ub, w210_.ub)        //[0,10]filter even output
      x0 = vmemu(ptr_x1+#0)                        //[1, 4]
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[1, 4]
   } {
      z2.uw += vrmpy(x5x4x3x2.ub, _zzz.ub)         //[0,11]filter even output
      z3.uw += vrmpy(x5x4x3x2.ub, zzz_.ub)         //[0,11]filter even output
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[1, 5]
      w210_.w = vasl(w_210.w, c8)                  //[1, 5]
   }:endloop0

.L_skip:

   {  s0.uw += vrmpy(x3x2x1x0.ub, w_210.ub)        //[1, 6]filter even output
      x5x4x3x2.h = vshuffe(x7x6x5x4.h,x3x2x3x2.h)  //[1, 6]
      x0 = vmemu(ptr_x0+#0)                        //[1, 0]
   } {
      s1.w = vadd(bias_val.w, s1.w)                //[WIDTH, P]
      s0.w = vadd(s0.w, bias_val.w)                //[WIDTH, ]
      z0.uw += vrmpy(x3x2x1x0.ub, _zzz.ub)         //[1, 9]filter even output
      z1.uw += vrmpy(x3x2x1x0.ub, zzz_.ub)         //[1, 9]filter even output
   } {
      s1.uw += vrmpy(x3x2x1x0.ub, w210_.ub)        //[1, 7]filter even output
      s0.w = vsub(s0.w, z0.w)                      //[WIDTH, ]
      d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      p1 =!cmp.gt(width_count, #-3)                //[WIDTH]
   } {
      s2.uw += vrmpy(x5x4x3x2.ub, w_210.ub)        //[1, 8]filter even output
      s1.w = vsub(s1.w, z1.w)                      //[WIDTH, ]
      x3x1x2x0.b = vshuff(x0.b)                    //[0, 1]
      nop;
   } {
      s3.uw += vrmpy(x5x4x3x2.ub, w210_.ub)        //[1,10]filter even output
      s0.w = vasl(s0.w, vshamt_vec.w)              //[WIDTH, ]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
      loop0(.L_vert, filt_height)                  //[WIDTH, P]
   } {
      z2.uw += vrmpy(x5x4x3x2.ub, _zzz.ub)         //[1,11]filter even output
      z3.uw += vrmpy(x5x4x3x2.ub, zzz_.ub)         //[1,11]filter even output
      s1.w = vasl(s1.w, vshamt_vec.w)              //[WIDTH, ]
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[0, 3]
   } {
      d0.w = vmpye(s0.w, vrecip.uh)                //[WIDTH, ]multiply by 1/max
      s2.w = vsub(s2.w, z2.w)                      //[WIDTH, ]
      if(p1) s1 = s0                               //[WIDTH]
      nop;
   } {
      if(p2) vmemu(ptr_y++#1) = d3210              //[WIDTH, E]
      d0.w += vmpyo(s0.w, vrecip.h):SSR            //[WIDTH, ]
      s3.w = vsub(s3.w, z3.w)                      //[WIDTH, ]
   } {
      p1 =!cmp.gt(width_count, #-1)                //[WIDTH]
      minv.w = vmin(minv.w, d0.w)                    //[WIDTH, ]
      d1.w = vmpye(s1.w, vrecip.uh)                //[WIDTH, ]
      s2.w = vasl(s2.w, vshamt_vec.w)              //[WIDTH, ]
   } {
      d1.w += vmpyo(s1.w, vrecip.h):SSR            //[WIDTH, ]
      s3.w = vasl(s3.w, vshamt_vec.w)              //[WIDTH, ]
      maxv.w = vmax(maxv.w, d0.w)                    //[WIDTH, ]
      nop;
   } {
      d2.w = vmpye(s2.w, vrecip.uh)                //[WIDTH, ]
      if(p1) s3 = s0                               //[WIDTH]
      maxv.w = vmax(maxv.w, d1.w)                    //[WIDTH, ]
      p2 = cmp.eq(r0, r0)                          //[WIDTH]
   } {
      d1d0.h = vpack(d1.w, d0.w):sat               //[WIDTH, ]
      minv.w = vmin(minv.w, d1.w)                    //[WIDTH, ]
      d2.w += vmpyo(s2.w, vrecip.h):SSR            //[WIDTH, ]
      p1 =!cmp.gt(width_count, #-2)                //[WIDTH]
   } {
      if(p1) d2 = d0                               //[WIDTH]
      d3.w = vmpye(s3.w, vrecip.uh)                //[WIDTH, ]
      x1 = vmemu(ptr_x0+#1)                        //[0, 0]
   } {
      d3.w += vmpyo(s3.w, vrecip.h):SSR            //[WIDTH, ]
      maxv.w = vmax(maxv.w, d2.w)                    //[WIDTH, ]
      minv.w = vmin(minv.w, d2.w)                    //[WIDTH, ]
      width_count = add(width_count, #-4)          //[WIDTH]
   } 
   {
      ptr_x1 = add(ptr_x0, next_in_width_depth)    //[0, 1]
      maxv.w = vmax(maxv.w, d3.w)                    //[WIDTH, ]
      minv.w = vmin(minv.w, d3.w)                    //[WIDTH, ]
      x7x5x6x4.b = vshuff(x1.b)                    //[0, 2]
   }:endloop1 //end width
   {  d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      ptr_w0 = add(ptr_w0, filt_size)              //[DEPTH,E]filt_size = filt_height*256 //4*3*64
      ptr_xin = add(ptr_xin, next_in_width_32)     //[DEPTH]
      depth_cnt = add(depth_cnt, #-1)              //[DEPTH,E]
   } {
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
      recip_level = memw(sp+#52)                   //read quantization level from temp location
      out_buf = add(out_buf, next_out_width_32)    //[DEPTH]
      p0 = cmp.eq(depth_cnt, #0)                   //[DEPTH,E]
   } {
      vmemu(ptr_y+#0) = d3210                      //[WIDTH, E]
      if(!p0) jump .L_depth                        //[DEPTH,E]
      if( p0) out_height = add(out_height, #-1)    //
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                  //
      depth_cnt = memw(sp+#17<<2)                  //depth
      in_buf = add(in_buf, next_in_width_depth_stride) //stride
      if(!p0.new) jump:nt .L_height
   }
   //end height

/* ----------------------------------------------------------------------------- */
   {  r17:16 = memd(sp+#0)                         //restore
      vmem(ptr_max+#0) = maxv                       //
   } {
      r19:18 = memd(sp+#8)                         //restore
      vmem(ptr_max+#1) = minv                       //
   } {
      r21:20 = memd(sp+#16)                        //restore
      r23:22 = memd(sp+#24)                        //restore
   } {
      r25:24 = memd(sp+#32)                        //restore
      r27:26 = memd(sp+#40)                        //restore
   } {
      dealloc_return                               //return
   }
/* ----------------------------------------------------------------------------- */
.L_end:
      .size dwconv2dbbb_s1_3xN_asm, .L_end-dwconv2dbbb_s1_3xN_asm
/* ----------------------------------------------------------------------------- */



/* =========================================================================== */
