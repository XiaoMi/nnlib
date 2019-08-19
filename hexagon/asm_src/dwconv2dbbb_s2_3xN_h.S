/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dbbb_s2_3xN_list_h.S"
       .global dwconv2dbbb_s2_3xN_asm
       .balign 32
       .type  dwconv2dbbb_s2_3xN_asm, @function
dwconv2dbbb_s2_3xN_asm:
/* ----------------------------------------------------------------------------- */
//I/O registers
#define   in_buf               r0       //
#define   filt                 r1       //
#define   out_buf              r2       //
#define   next_in_width_depth  r3       //
#define   next_out_width_depth r4       //currently unused
#define   next_in_width_32     r5       //
#define   next_out_width_32    r10      //
#define   depth                r11      //
#define   out_width            r12      //
#define   out_height           r13      //
#define   filt_height          r25      //
#define   filt_zero            r7       //
#define   bias_sum             r14      //
#define   ptr_max              r15      //
#define   recip_level          r10      //
#define   recip_shift          r8       //
#define   stride_v             r28      //
#define   sbuf_base            r4       //
//scaler register
#define   ptr_w0               r16      //
#define   ptr_w1               r17      //
#define   width_cnt            r6       //
#define   bias_ptr             r18      //
#define   ptr_x0               r19      //
#define   ptr_x1               r20      //
#define   ptr_xin              r22      //
#define   ptr_y                r23      //
#define   depth_cnt            r26      //
#define   filt_size            r9       //
#define   next_in_width_depth_stride r28//
#define   zzzz                 r7       //
#define   _zzz                 r7       //
#define   sbuf                 r21      //
//vector registers
#define   vrecip                v0      //
#define   vshamt_vec            v1      //
#define   max                   v2      //
#define   min                   v3      //
#define   bias_val              v4      //
#define   x0                    v5      //
#define   x1                    v6      //
#define   x2                    v7      //
#define   xbxax9x8              v9      //
#define   x7x6x5x4             v10      //
#define   x3x2x1x0             v11      //
#define   w_210                 v8      //
#define   x3x2x3x2             v26      //
#define   x5x4x3x2             v27      //
#define   x7x6x7x6             v28      //
#define   x9x8x7x6             v29      //
#define   s0                   v12      //
#define   s1                   v13      //
#define   s2                   v14      //
#define   s3                   v15      //
#define   z0                   v16      //
#define   z1                   v17      //
#define   z2                   v18      //
#define   z3                   v19      //
#define   d0                   v20      //
#define   d1                   v21      //
#define   d1d0                 v24      //
#define   d2                   v22      //
#define   d3                   v23      //
#define   d3d2                 v25      //
#define   d3210                v25      //
#define SSR        <<1:rnd:sat:shift    //simplfy mpy instruction
#define PV(VSRC) .word (0x1DFFE020+VSRC)//debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC)//debug sca reg

/* =========================================================================== */
   {  allocframe(#56)                              //0th entry on sbuf (56+8)/4=20
   } {
      memd(sp+#0)  = r17:16                        //
      memd(sp+#8)  = r19:18                        //
   } {
      memd(sp+#16) = r21:20                        //
      memd(sp+#24) = r23:22                        //
   } {
      memd(sp+#32) = r25:24                        //
      memd(sp+#40) = r27:26                        //
   } {
      next_out_width_32 = memw(sp+#16<<2)          //
      depth  = memw(sp+#17<<2)                     //
   } {
      out_width = memw(sp+#18<<2)                  //
      out_height = memw(sp+#19<<2)                 //
      depth = lsr(depth, #5)                       //depth/32
   } {
      filt_height = memw(sp+#20<<2)                //
      filt_zero = memw(sp+#21<<2)                  //
      out_width = add(out_width, #3)               //
   } {
      bias_sum = memw(sp+#22<<2)                   //
      ptr_max = memw(sp+#23<<2)                    //
      out_width = lsr(out_width, #2)               //
   } {
      memw(sp+#17<<2) = depth                      //
      zzzz  = vsplatb(filt_zero)                   //
   } {
      recip_shift = memw(sp+#25<<2)                //
      stride_v = memw(sp+#26<<2)                   //
   } {
      sbuf_base = memw(sp+#27<<2)                  //scratchpad buffer 128*filt_height bytes
      vshamt_vec= vsplat(recip_shift)              //
      next_in_width_depth_stride = mpyi(next_in_width_depth,stride_v) //
   } {
      max = vmem(ptr_max+#0)                       //
      filt_size = filt_height                      //
      filt_height = add(filt_height, #-2)          //
   } {
      _zzz  = lsr(zzzz, #8)                        //
      min = vmem(ptr_max+#1)                       //
      depth_cnt = memw(sp+#17<<2)                  //depth
      p2 = !cmp.gt(filt_height, #0)                //<=0
   } 
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {
      recip_level = memw(sp+#24<<2)
   }
   {  bias_ptr = bias_sum                          //
      ptr_xin = in_buf
      ptr_w0 = filt
      out_height = add(out_height, #-1)            //
   }
/* ----------------------------------------------------------------------------- */
.L_depth:
   {
      vrecip = vmem(recip_level++#1)
   }
   {
      memw(sp+#52) = recip_level
   }
   {  bias_val = vmem(bias_ptr++#1)                //
      ptr_x0 = ptr_xin                             //
      ptr_y = out_buf                              //
      loop1(.L_width, out_width)
   } {
      x1 = vmemu(ptr_x0+#1)                        //[0, 0]
      loop0(.L_init, filt_size)
      ptr_w1 = ptr_w0                              //[WIDTH, P]
   } {
      sbuf = sbuf_base
      ptr_x1 = ptr_x0                              //[WIDTH, P]
      width_cnt = memw(sp+#18<<2)                  //
   }
/* --------------------------------------------------------------------------- */
.L_init:
   {  x3x2x1x0 = vmemu(ptr_x1+#0)                  //[0, 0]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[0, 0]
   } {
      x3x2x1x0.b = vshuff(x3x2x1x0.b)              //[0, 1]
   } {
      x3x2x1x0.b = vshuff(x3x2x1x0.b)              //[0, 2]
      vmem(sbuf++#1) = x3x2x1x0.new
   }:endloop0
   {  loop0(.L_vert, filt_height)                  //[WIDTH, P]
      p3 = !cmp.eq(r0, r0)                         //[WIDTH]
      x1.b = vshuff(x1.b)                          //[0, 3]
      ptr_x1 = ptr_x0                              //[WIDTH, P]
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
   {  x2 = vmemu(ptr_x1+#2)                        //[0, 4]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[0, 4]
      ptr_x0 = add(ptr_x0, #256)                   //[WIDTH,P]+4 32 depths for stride 2
   } {
      x7x6x5x4.b = vshuff(x1.b)                    //[0, 5]
      x3x2x1x0 = vmem(sbuf_base)                   //[0, 5]
      width_cnt = add(width_cnt, #-4)              //
      sbuf = sbuf_base                             //
   } {
      x2.b = vshuff(x2.b)                          //[0, 6]
      w_210 = vmem(ptr_w1++#1)                     //[0, 6]
      s1 = bias_val                                //[WIDTH, P]
      s0 = bias_val                                //[WIDTH, P]
   } 
/* --------------------------------------------------------------------------- */
   {  x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[0, 7]
      x1 = vmemu(ptr_x1+#1)                        //[1, 0]
      s2 = bias_val                                //[WIDTH, P]
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w_210.ub)        //[0, 8]filter even output
      xbxax9x8.b = vshuff(x2.b)                    //[0, 8]
      vmem(sbuf++#1) = xbxax9x8.new                //[0, 9]
      x5x4x3x2.h = vshuffe(x7x6x5x4.h,x3x2x3x2.h)  //[0, 8]
   } {
      z0.uw = vrmpy(x3x2x1x0.ub, _zzz.ub)          //[0, 9]filter even output
      z2.uw = vrmpy(x7x6x5x4.ub, _zzz.ub)          //[0, 9]filter even output
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[0, 9]
      z3 = #0                                      //  
   } {
      s1.uw += vrmpy(x5x4x3x2.ub, w_210.ub)        //[0,10]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h,x7x6x7x6.h)  //[0,10]
      x1.b = vshuff(x1.b)                          //[1, 3]
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w_210.ub)        //[0,11]filter even output
      x2 = vmemu(ptr_x1+#2)                        //[1, 4]
      s3 = bias_val                                //[WIDTH, P]
   } {
      s3.uw += vrmpy(x9x8x7x6.ub, w_210.ub)        //[0,12]filter even output
      x7x6x5x4.b = vshuff(x1.b)                    //[1, 5]
      x3x2x1x0 = vmem(sbuf)                        //[1, 5]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[1, 4]
   } {
      z1.uw = vrmpy(x5x4x3x2.ub, _zzz.ub)          //[0,13]filter even output
      x2.b = vshuff(x2.b)                          //[1, 6]
      w_210 = vmem(ptr_w1++#1)                     //[1, 6]
      if(p2) jump .L_skip                          //dont do <=0 iterations
   }
.L_vert:
   {  
      z3.uw += vrmpy(x9x8x7x6.ub, _zzz.ub)         //[0,13]filter even output
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[0, 7]
      x1 = vmemu(ptr_x1+#1)                        //[1, 0]
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w_210.ub)        //[0, 8]filter even output
      xbxax9x8.b = vshuff(x2.b)                    //[0, 8]
      x5x4x3x2.h = vshuffe(x7x6x5x4.h,x3x2x3x2.h)  //[0, 8]
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, _zzz.ub)         //[0, 9]filter even output
      z2.uw += vrmpy(x7x6x5x4.ub, _zzz.ub)         //[0, 9]filter even output
      vmem(sbuf++#1) = xbxax9x8                    //[0, 9]
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[0, 9]
   } {
      s1.uw += vrmpy(x5x4x3x2.ub, w_210.ub)        //[0,10]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h,x7x6x7x6.h)  //[0,10]
      x1.b = vshuff(x1.b)                          //[1, 3]
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w_210.ub)        //[0,11]filter even output
      x2 = vmemu(ptr_x1+#2)                        //[1, 4]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[1, 4]
   } {
      s3.uw += vrmpy(x9x8x7x6.ub, w_210.ub)        //[0,12]filter even output
      x7x6x5x4.b = vshuff(x1.b)                    //[1, 5]
      x3x2x1x0 = vmem(sbuf)                        //[1, 5]
   } {
      z1.uw += vrmpy(x5x4x3x2.ub, _zzz.ub)         //[0,13]filter even output
      x2.b = vshuff(x2.b)                          //[1, 6]
      w_210 = vmem(ptr_w1++#1)                     //[1, 6]
   }:endloop0
/* --------------------------------------------------------------------------- */
.L_skip:
   {  x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[1, 7]
      z3.uw += vrmpy(x9x8x7x6.ub, _zzz.ub)         //[0,13]filter even output
      d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w_210.ub)        //[1, 8]filter even output
      xbxax9x8.b = vshuff(x2.b)                    //[1, 8]
      x5x4x3x2.h = vshuffe(x7x6x5x4.h,x3x2x3x2.h)  //[1, 8]
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, _zzz.ub)         //[1, 9]filter even output
      z2.uw += vrmpy(x7x6x5x4.ub, _zzz.ub)         //[1, 9]filter even output
      vmem(sbuf++#1) = xbxax9x8                    //[1, 9]
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[1, 9]
   } {
      s0.w = vsub(s0.w, z0.w)                      //
      s1.uw += vrmpy(x5x4x3x2.ub, w_210.ub)        //[1,10]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h,x7x6x7x6.h)  //[1,10]
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w_210.ub)        //[1,11]filter even output
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
   } {
      s3.uw += vrmpy(x9x8x7x6.ub, w_210.ub)        //[1,12]filter even output
      s0.w = vasl(s0.w, vshamt_vec.w)              //
      s2.w = vsub(s2.w, z2.w)                      //
      p1 = !cmp.gt(width_cnt, #-2)                 //[WIDTH] test s2 oor
   } {
      if(p3) vmemu(ptr_y++#1) = d3210              //[WIDTH, E]
      z1.uw += vrmpy(x5x4x3x2.ub, _zzz.ub)         //[1,13]filter even output
      z3.uw += vrmpy(x9x8x7x6.ub, _zzz.ub)         //[1,13]filter even output
   } {
      s2.w = vasl(s2.w, vshamt_vec.w)              //
      s1.w = vsub(s1.w, z1.w)                      //
      d0.w = vmpye(s0.w, vrecip.uh)                //[0,15]multiply by 1/max
   } {
      s3.w = vsub(s3.w, z3.w)                      //
      d0.w += vmpyo(s0.w, vrecip.h):SSR            //[0,17]3
      x1 = vmemu(ptr_x0+#1)                        //[P, 0]
   } {
      s1.w = vasl(s1.w, vshamt_vec.w)              //
      d2.w = vmpye(s2.w, vrecip.uh)                //[0,15]multiply by 1/max
      min.w = vmin(min.w, d0.w)                    //[0,22]8               //0+2+1
      ptr_w1 = ptr_w0                              //[WIDTH, P]
   } {
      max.w = vmax(max.w, d0.w)                    //[0,18]5               //0+2+1
      s3.w = vasl(s3.w, vshamt_vec.w)              //
      d2.w += vmpyo(s2.w, vrecip.h):SSR            //[0,17]3
      p0 = !cmp.gt(width_cnt, #-1)                 //WIDTH] test s3 oor
   } {
      if(p1) d2 = d0                               //
      p1 = !cmp.gt(width_cnt, #-3)                 //[WIDTH] test s1 oor
      d1.w = vmpye(s1.w, vrecip.uh)                //[0,22]multiply by 1/max
      if(p0) s3 = s0                               //
   } {
      d1.w += vmpyo(s1.w, vrecip.h):SSR            //[0,23]9
      min.w = vmin(min.w, d2.w)                    //[0,22]8               //0+2+1
      max.w = vmax(max.w, d2.w)                    //[0,18]5               //0+2+1
      loop0(.L_vert, filt_height)                  //[WIDTH, P]
   } {
      if(p1) d1 = d0                               //
      d3.w = vmpye(s3.w, vrecip.uh)                //[0,22]multiply by 1/max
      x1.b = vshuff(x1.b)                          //[0, 3]
      p3 = cmp.eq(r0, r0)                          //[WIDTH]
   } {
      d3.w += vmpyo(s3.w, vrecip.h):SSR            //[0,23]9
      max.w = vmax(max.w, d1.w)                    //[0,26]12              //0+2+1
      min.w = vmin(min.w, d1.w)                    //[0,27]13              //0+2+1
      ptr_x1 = ptr_x0                              //[WIDTH, P]
   } {
      d1d0.h = vpack(d1.w, d0.w):sat               //[0,27]
      max.w = vmax(max.w, d3.w)                    //[0,26]12              //0+2+1
      min.w = vmin(min.w, d3.w)                    //[0,27]13              //0+2+1
   }:endloop1 //end width
   {  d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      ptr_w0 += asl(filt_size, #7)                 //[DEPTH,E]filt_size = filt_height*256 //4*3*64
      ptr_xin = add(ptr_xin, next_in_width_32)     //[DEPTH]
      next_out_width_32 = memw(sp+#16<<2)
   } {
      depth_cnt = add(depth_cnt, #-1)              //[DEPTH,E]
      out_buf = add(out_buf, next_out_width_32)    //[DEPTH]
      recip_level = memw(sp+#52)
   } {
      p0 = cmp.eq(depth_cnt, #0)                   //[DEPTH,E]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
   } {
      vmemu(ptr_y+#0) = d3210                      //[WIDTH, E]
      if(!p0) jump .L_depth                        //[DEPTH,E]
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                  //
      depth_cnt = memw(sp+#17<<2)                  //depth
      in_buf = add(in_buf, next_in_width_depth_stride) //stride 
      if(!p0.new) jump:nt .L_height                //
   }//end height
/* ----------------------------------------------------------------------------- */
      ptr_max = memw(sp+#23<<2)                    //
   {
      r17:16 = memd(sp+#0)                         //restore
      vmem(ptr_max+#0) = max                       //
   } {
      r19:18 = memd(sp+#8)                         //restore
      vmem(ptr_max+#1) = min                       //
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
      .size dwconv2dbbb_s2_3xN_asm, .L_end-dwconv2dbbb_s2_3xN_asm
/* ----------------------------------------------------------------------------- */
