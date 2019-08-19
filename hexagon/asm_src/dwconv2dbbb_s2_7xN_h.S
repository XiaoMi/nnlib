/*
 Behavioral C model
 */
#if 0
void dwconv2dbbb_MxN_cn(
   uint8_t *in_buf,
   uint8_t  *filt,
   uint8_t  *out_buf,
   int next_in_width,
   int next_out_width,
   int next_in_width_32,
   int next_out_width_32,
   int depth,
   int out_width,
   int out_height,
   int filt_width,
   int filt_height,
   int filt_zero,
   int32_t *bias_sum,
   int32_t *max,
   int recip_level,
   int recip_shift,
   int stride_width,
   int stride_height)
{
   int out_y, d, out_x, ur, in_val, filt_val;
   int out_z, filt_y, filt_x, cnt;
   int out_width_pad = (out_width+3)&(~3);
   int32_t sum, zum, sum0;
   int64_t lsum ;
   int o_filt_width = (filt_width+3)&(~3);
   int buf_offset;

    for (out_y = 0; out_y < out_height; out_y++) {
        cnt = out_width;
        for (out_x = 0; out_x < out_width_pad; out_x+=4) {
            cnt -= 4;
            for(d=0; d < depth/32; d++) {
               for (out_z = 0; out_z < 32; out_z++) {
                  for(ur=0; ur < 4; ur++)
                  {
                    sum = (int32_t)bias_sum[32*d+out_z];
                    zum = 0;
                    for (filt_y = 0; filt_y < filt_height; filt_y++) {
                       for (filt_x = 0; filt_x < o_filt_width; filt_x++) {
                          buf_offset = (out_y * stride_height +  filt_y) * next_in_width
                                      + d * next_in_width_32
                                      + (out_x*stride_width + ur*stride_width + filt_x) * 32
                                      + out_z;
                          in_val = in_buf[buf_offset];

                          filt_val = filt[32*d*filt_height*o_filt_width
                                           + (o_filt_width*filt_y)*32
                                           + out_z*4 + 128*(filt_x/4)
                                           + (filt_x % 4)] ;
                          sum += (uint32_t)in_val*(int32_t)filt_val;
                          if(filt_x < filt_width)
                             zum += (uint32_t)in_val*(int32_t)filt_zero;
                       }
                    }
                    sum = sum - zum;
                    if(ur==0) sum0 = sum;
                    if(ur == 1 && !(cnt > -3)) sum = sum0;
                    if(ur == 2 && !(cnt > -2)) sum = sum0;
                    if(ur == 3 && !(cnt > -1)) sum = sum0;

                    sum <<= recip_shift;
                    lsum = (int64_t)sum * ((int64_t)recip_level) + 0x40000000LL;
                    lsum = lsum >> 31;
                    sum = (int)lsum;
                    max[out_z]    = (sum > max[out_z]) ? sum : max[out_z];
                    max[out_z+32] = (sum < max[out_z+32]) ? sum : max[out_z+32];
                    if(lsum < 0) lsum = 0; if(lsum > 0xffll) lsum = 0xffll;
                    out_buf[out_y * next_out_width
                            + 32 * (out_x+ur)
                            + d * next_out_width_32
                            + out_z] = (uint8_t) lsum;
                  }//ur
               }//out_z
            }//d
        }//out_x
    }//out_y
    return;
}
#endif
/* 
   CODESIZE  912 bytes
   STACK     1856 bytes 
   PACKETS = height*(
                     (depth/32)*(
                                 (width/4)*(
                                            filt_height*12+11
                                           )+
                                 filt_height*6+7+
                                )+2
                    )+17
*/
/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dbbb_s2_7xN_h.S"
       .global dwconv2dbbb_s2_7xN_asm
       .balign 32
       .type  dwconv2dbbb_s2_7xN_asm, @function
dwconv2dbbb_s2_7xN_asm:
/* ----------------------------------------------------------------------------- */
//I/O registers
#define   in_buf               r0        //
#define   filt                 r1        //
#define   out_buf              r2        //
#define   next_in_width_depth  r3        //
#define   next_out_width_depth r4        //currently unused 
#define   next_in_width_32     r5        //
#define   next_out_width_32    r10       //
#define   depth                r11       //
#define   out_width            r12       //
#define   out_height           r13       //
#define   filt_height          r25       //
#define   filt_zero            r7        //
#define   bias_sum             r14       //
#define   ptr_max              r15       //
#define   recip_level          r4       //
#define   recip_shift          r8        //
#define   stride_v             r28       //
#define   sbuf                 r14       //
//Scaler registers                   
#define   ptr_w0               r16       //
#define   ptr_w1               r17       //
#define   col_count            r9        //
#define   bias_ptr             r18       //
#define   ptr_x0               r19       //
#define   ptr_x1               r20       //
#define   ptr_xin              r22       //
#define   ptr_y                r23       //
#define   depth_cnt            r26       //
#define   filt_size            r8        //
#define   next_in_width_depth_stride r28 //
#define   zzzz                 r7        //
#define   _zzz                 r27       //
#define   out_width4           r6        //
//Vector registers                  
#define   vrecip                v0       //
#define   vshamt_vec            v1       //
#define   max                   v2       //
#define   min                   v3       //
#define   bias_val              v4       //
#define   w_654                v8        //
#define   w3210                v28       //
#define   x0                   v30       //
#define   x1                   v29       // 
#define   x2                   v5        //
#define   x3                   v11       //
#define   x3x1x2x0             v30       //
#define   x3x2x1x0             v30       //
#define   x7x5x6x4             v29       //
#define   x7x6x5x4             v10       //
#define   xbxax9x8             v5        //
#define   xbx9xax8             v5        //
#define   xfxexdxc             v11       //
#define   x3x2x3x2             v6        //
#define   x5x4x3x2             v6        //
#define   x7x6x7x6             v9        //
#define   x9x8x7x6             v9        //
#define   xbxaxbxa             v7        //
#define   xdxcxbxa             v7        //
#define   s0                   v12       //
#define   s1                   v13       //
#define   s2                   v14       //
#define   s3                   v15       //
#define   z0                   v16       //
#define   z1                   v17       //
#define   z2                   v18       //
#define   z3                   v19       //
#define   d0                   v20       //
#define   d1                   v21       //
#define   d1d0                 v24       //
#define   d2                   v22       //
#define   d3                   v23       //
#define   d3d2                 v25       //
#define   d3210                v25       //
#define SSR        <<1:rnd:sat:shift     //simplfy mpy instruction    
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec regs 
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca regs 
/* =========================================================================== */
   {  allocframe(#56)                              //0th entry on stack (56+8)/4=20
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
      out_width4 = add(out_width, #3)              //
   } {
      ptr_max = memw(sp+#23<<2)                    //
      out_width4 = lsr(out_width4, #2)             //
      zzzz  = vsplatb(filt_zero)                   //
   } {
      _zzz  = lsr(zzzz, #8)                        //
      recip_shift = memw(sp+#25<<2)                //
      stride_v = memw(sp+#26<<2)                   //
   } {
      max = vmem(ptr_max+#0)                       //
      memw(sp+#17<<2) = depth                      //
      vshamt_vec= vsplat(recip_shift)              //
      next_in_width_depth_stride = mpyi(next_in_width_depth,stride_v) //
   } {
      min = vmem(ptr_max+#1)                       //
      depth_cnt = memw(sp+#17<<2)                  //depth
      filt_size = filt_height                      //
      filt_height = add(filt_height, #-1)          //
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {  bias_ptr = memw(sp+#22<<2)                   //
      ptr_xin = in_buf                             //
      ptr_w0 = filt                                //
      recip_level = memw(sp+#24<<2)
   }
/* ----------------------------------------------------------------------------- */
.L_depth:
    {  
      bias_val = vmem(bias_ptr++#1)                //
      loop0(.L_init, filt_size)                    //
      ptr_x1 = ptr_xin                             //[WIDTH, P]
      sbuf = memw(sp+#27<<2)                       //
   } {
      vrecip = vmem(recip_level++#1);
      loop1(.L_width, out_width4)                  //
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_init:
   {
      x0 = vmemu(ptr_x1+#0)                        //[0, 0]
   } {
      x1 = vmemu(ptr_x1+#1)                        //[0, 1]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[0, 1]
   } {
      x0.b = vshuff(x0.b)                          //[0, 2]
   } {
      x1.b = vshuff(x1.b)                          //[0, 3]
   } {
      x3x2x1x0.b = vshuff(x0.b)                    //[0, 4]
      vmem(sbuf++#1) = x3x2x1x0.new                //next xb-8
   } {
      x7x6x5x4.b = vshuff(x1.b)                    //[0, 5]
      vmem(sbuf++#1) = x7x6x5x4.new                //next xb-8
   }:endloop0
/* --------------------------------------------------------------------------- */
   {  sbuf = memw(sp+#27<<2)                       //[WIDTH]
      w3210 = vmem(ptr_w0+#0)                      //[0, 0]
      ptr_x0 = ptr_xin                             //
      p3 = !cmp.eq(r0, r0)                         //
   } {
      col_count = out_width                        //
      x3x2x1x0 = vmem(sbuf+#0)                     //[0, 1]
      ptr_y = out_buf                              //
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
   {  x7x6x5x4 = vmem(sbuf+#1)                     //[0, 2]
   } {
      x2 = vmemu(ptr_x0+#2)                        //[0, 3]
      ptr_w1 = add(ptr_w0, #128)                   //[WIDTH, P]
   } {
      s0.uw = vrmpy(x3x2x1x0.ub, w3210.ub)         //[0, 4]filter even output
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[0, 4]
      w_654 = vmem(ptr_w1++#1)                     //[0, 4]
      ptr_x1 = ptr_x0                              //[WIDTH, P]
   } {
      s2.uw  = vrmpy(x7x6x5x4.ub, w3210.ub)        //[0, 5]filter even output
      xbx9xax8.b = vshuff(x2.b)                    //[0, 5]
      x5x4x3x2.h = vshuffe(x7x6x5x4.h, x3x2x3x2.h) //[0, 5]
      ptr_x0 = add(ptr_x0, #256)                   //[WIDTH,P]+8 32 depths for stride 2
   } {
      z0.uw = vrmpy(x3x2x1x0.ub, zzzz.ub)          //[0, 6]filter even output
      z2.uw = vrmpy(x7x6x5x4.ub, zzzz.ub)          //[0, 6]filter even output
      x3 = vmemu(ptr_x1+#3)                        //[0, 6]
   } {
      s0.uw += vrmpy(x7x6x5x4.ub, w_654.ub)        //[0, 7]filter even output
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[0, 7]
      xbxax9x8.b = vshuff(xbx9xax8.b)              //[0, 7]
      vmem(sbuf++#1) = xbxax9x8.new                //[0, 7]next xb-8
   } {
      s1.uw = vrmpy(x5x4x3x2.ub, w3210.ub)         //[0, 8]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h, x7x6x7x6.h) //[0, 8]
      s0.w = vadd(s0.w, bias_val.w)                //[WIDTH, P]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[0, 8]
   } {
      z0.uw += vrmpy(x7x6x5x4.ub, _zzz.ub)         //[0, 9]filter even output
      z1.uw = vrmpy(x5x4x3x2.ub, zzzz.ub)          //[0, 9]filter even output
      x3.b = vshuff(x3.b)                          //[0, 9]
      s1.w = vadd(s1.w, bias_val.w)                //[WIDTH, P]
   } {
      s1.uw += vrmpy(x9x8x7x6.ub, w_654.ub)        //[0,10]filter even output
      xbxaxbxa.h = vshuffo(xbxax9x8.h, xbxax9x8.h) //[0,10] 
      s2.w = vadd(s2.w, bias_val.w)                //[WIDTH, P]
      loop0(.L_vert, filt_height)                  //[WIDTH, P]
   } {
      s3.uw = vrmpy(x9x8x7x6.ub, w3210.ub)         //[0,11]filter even output
      xfxexdxc.b = vshuff(x3.b)                    //[0,11]
      vmem(sbuf++#1) = xfxexdxc.new                //[0,11]next xb-8
      z3 = #0                                      //[WIDTH]
   } 
/* --------------------------------------------------------------------------- */
      .balign 32
.L_vert:
   {  z1.uw+= vrmpy(x9x8x7x6.ub, _zzz.ub)          //[0,12]filter even output
      z3.uw+= vrmpy(x9x8x7x6.ub, zzzz.ub)          //[0,12]filter even output
      xdxcxbxa.h = vshuffe(xfxexdxc.h, xbxaxbxa.h) //[0,12] 
      w3210 = vmem(ptr_w1++#1)                     //[1, 0]
   } {
      s2.uw += vrmpy(xbxax9x8.ub, w_654.ub)        //[0,13]filter even output
      x3x2x1x0 = vmem(sbuf+#0)                     //[1, 1]
   } {
      s3.uw += vrmpy(xdxcxbxa.ub, w_654.ub)        //[0,14]filter even output
      x7x6x5x4 = vmem(sbuf+#1)                     //[1, 2]
   } {
      z2.uw += vrmpy(xbxax9x8.ub, _zzz.ub)         //[0,15]filter even output
      z3.uw += vrmpy(xdxcxbxa.ub, _zzz.ub)         //[0,15]filter even output
      x2 = vmemu(ptr_x1+#2)                        //[1, 3]
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w3210.ub)        //[1, 4]filter even output
      x3x2x3x2.h = vshuffo(x3x2x1x0.h, x3x2x1x0.h) //[1, 4]
      w_654 = vmem(ptr_w1++#1)                     //[1, 4]
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w3210.ub)        //[1, 5]filter even output
      xbx9xax8.b = vshuff(x2.b)                    //[1, 5]
      x5x4x3x2.h = vshuffe(x7x6x5x4.h, x3x2x3x2.h) //[1, 5]
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, zzzz.ub)         //[1, 6]filter even output
      z2.uw += vrmpy(x7x6x5x4.ub, zzzz.ub)         //[1, 6]filter even output
      x3 = vmemu(ptr_x1+#3)                        //[1, 6]
   } {
      s0.uw += vrmpy(x7x6x5x4.ub, w_654.ub)        //[1, 7]filter even output
      x7x6x7x6.h = vshuffo(x7x6x5x4.h, x7x6x5x4.h) //[1, 7]
      xbxax9x8.b = vshuff(xbx9xax8.b)              //[1, 7]
      vmem(sbuf++#1) = xbxax9x8.new                //[1, 7]next xb-8
   } {
      s1.uw += vrmpy(x5x4x3x2.ub, w3210.ub)        //[1, 8]filter even output
      x9x8x7x6.h = vshuffe(xbxax9x8.h, x7x6x7x6.h) //[1, 8]
      ptr_x1 = add(ptr_x1, next_in_width_depth)    //[1, 8]
   } {
      z0.uw += vrmpy(x7x6x5x4.ub, _zzz.ub)         //[1, 9]filter even output
      z1.uw += vrmpy(x5x4x3x2.ub, zzzz.ub)         //[1, 9]filter even output
      x3.b = vshuff(x3.b)                          //[1, 9]
   } {
      s1.uw += vrmpy(x9x8x7x6.ub, w_654.ub)        //[1,10]filter even output
      xbxaxbxa.h = vshuffo(xbxax9x8.h, xbxax9x8.h) //[1,10] 
   } {
      s3.uw += vrmpy(x9x8x7x6.ub, w3210.ub)        //[1,11]filter even output
      xfxexdxc.b = vshuff(x3.b)                    //[1,11]
      vmem(sbuf++#1) = xfxexdxc.new                //[1,11]next xb-8
   }:endloop0  
/* --------------------------------------------------------------------------- */
   {  z2.uw += vrmpy(xbxax9x8.ub, _zzz.ub)         //[1,15]filter even output
      z3.uw+= vrmpy(x9x8x7x6.ub, zzzz.ub)          //[1,12]filter even output
      xdxcxbxa.h = vshuffe(xfxexdxc.h, xbxaxbxa.h) //[1,12] 
      s0.w = vsub(s0.w, z0.w)                      //[WIDTH]
   } {
      s2.uw += vrmpy(xbxax9x8.ub, w_654.ub)        //[1,13]filter even output
      s3.w = vadd(s3.w, bias_val.w)                //[WIDTH, P]
      d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      col_count = add(col_count, #-4)              //[WIDTH]
   } {
      z1.uw+= vrmpy(x9x8x7x6.ub, _zzz.ub)          //[1,12]filter even output
      z3.uw += vrmpy(xdxcxbxa.ub, _zzz.ub)         //[1,15]filter even output
      s2.w = vsub(s2.w, z2.w)                      //[WIDTH,E]
      s0.w = vasl(s0.w, vshamt_vec.w)              //[WIDTH,E]
   } {
      s1.w = vsub(s1.w, z1.w)                      //[WIDTH]
      s3.uw += vrmpy(xdxcxbxa.ub, w_654.ub)        //[1,14]filter even output
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
   } {
      s2.w = vasl(s2.w, vshamt_vec.w)              //[WIDTH]
      d0.w = vmpye(s0.w, vrecip.uh)                //[wIDTH]multiply by 1/max
      s3.w = vsub(s3.w, z3.w)                      //[WIDTH]
      p0 = !cmp.gt(col_count, #-3)                 //[WIDTH]
   } {
      d0.w += vmpyo(s0.w, vrecip.h):SSR            //[WIDTH]
      s1.w = vasl(s1.w, vshamt_vec.w)              //[WIDTH]
      if(p3) vmemu(ptr_y++#1) = d3210              //[WIDTH, E]
   } {
      d2.w = vmpye(s2.w, vrecip.uh)                //[0,15]multiply by 1/max
      s3.w = vasl(s3.w, vshamt_vec.w)              //[WIDTH]
      if(p0) s1 = s0                               //[WIDTH] if over gen'd write valid val
      p1 = !cmp.gt(col_count, #-1)                 //[WIDTH]
   } {
      if(p1) s3 = s0                               //[WIDTH]
      d2.w += vmpyo(s2.w, vrecip.h):SSR            //[0,17]3
      p0 = !cmp.gt(col_count, #-2)                 //[WIDTH]
      min.w = vmin(min.w, d0.w)                    //[0,22]
   } {
      d1.w = vmpye(s1.w, vrecip.uh)                //[0,22]multiply by 1/max
      max.w = vmax(max.w, d0.w)                    //[0,18]
      if(p0) d2 = d0                               //[WIDTH]
      p3 = cmp.eq(r0, r0)                          //[WIDTH]
   } {
      d1.w += vmpyo(s1.w, vrecip.h):SSR            //[0,23]
      min.w = vmin(min.w, d2.w)                    //[0,22]
      max.w = vmax(max.w, d2.w)                    //[0,18]
   } {
      d3.w = vmpye(s3.w, vrecip.uh)                //[0,22]multiply by 1/max
      max.w = vmax(max.w, d1.w)                    //[0,26]
      min.w = vmin(min.w, d1.w)                    //[0,27]
      sbuf = memw(sp+#27<<2)                       //[WIDTH]
   } {
      d1d0.h = vpack(d1.w, d0.w):sat               //[0,27]
      d3.w += vmpyo(s3.w, vrecip.h):SSR            //[0,23]
      w3210 = vmem(ptr_w0+#0)                      //[0, 0]
   } {
      max.w = vmax(max.w, d3.w)                    //[0,26]
      min.w = vmin(min.w, d3.w)                    //[0,27]
      x3x2x1x0 = vmem(sbuf+#0)                     //[0, 1]
   }:endloop1 //end width
/* --------------------------------------------------------------------------- */
   {  d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]
      ptr_w0 += asl(filt_size, #8)                 //[DEPTH,E]filt_size=filt_height*256
      ptr_xin = add(ptr_xin, next_in_width_32)     //[DEPTH]
   } {
      depth_cnt = add(depth_cnt, #-1)              //[DEPTH,E]
      out_buf = add(out_buf, next_out_width_32)    //[DEPTH]
      loop1(.L_width, out_width4)                  //[DEPTH]
   } {
      p0 = cmp.eq(depth_cnt, #0)                   //[DEPTH,E]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]
   } {
      vmemu(ptr_y+#0) = d3210                      //[WIDTH, E]
      if(!p0) jump .L_depth                        //[DEPTH,E]
      if(p0) out_height = add(out_height, #-1)     //
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                  //
      depth_cnt = memw(sp+#17<<2)                  //depth
      in_buf=add(in_buf,next_in_width_depth_stride)//stride 
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
      .size dwconv2dbbb_s2_7xN_asm, .L_end-dwconv2dbbb_s2_7xN_asm
/* ----------------------------------------------------------------------------- */
