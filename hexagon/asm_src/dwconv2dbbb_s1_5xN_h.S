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
/*  FUNCTIONS      : dwconv2dbbb_s1_5xN_asm                             */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Depthwise filter stride 1xM, filter size 5xN         */
/*                 input and output ptr non aligned output width        */
/*                 padded, max and min found only on valid range        */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 4.30.19                                         */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->                                                               */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE =   928 bytes                                           */
/*     STACK    =   48  bytes                                           */
/*     ASSUMPTIONS                                                      */
/*        none                                                          */
/*  C MODEL                                                             */
/*     exact match to assembly code                                     */
/*======================================================================*/
#if 0
void dwconv2dbbb_s1_cn(
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
   int filt_height,
   int filt_zero,
   int32_t *bias_sum,
   int32_t *max,
   int recip_level,
   int recip_shift,
   int stride_height)
{
   int out_y, d, out_x, ur, in_val, filt_val;
   int out_z, filt_y, filt_x, cnt;
   int out_width_pad = (out_width+3)&(~3);
   int32_t sum, zum, sum0;
   int64_t lsum ;
   int filt_width = 5;
   int o_filt_width = (filt_width+3)&(~3);
   int buf_offset;

    for (out_y = 0; out_y < out_height; out_y++) {
        for (out_x = 0; out_x < out_width_pad; out_x+=4) {
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
                                      + (out_x + ur + filt_x) * 32 
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
/*======================================================================*/
/* ----------------------------------------------------------------------------- */
       .text
       .file "dwconv2dbbb_s1_5xN_h.S"
       .global dwconv2dbbb_s1_5xN_asm
       .balign 32
       .type  dwconv2dbbb_s1_5xN_asm, @function
dwconv2dbbb_s1_5xN_asm:
/* ----------------------------------------------------------------------------- */
//calling values
#define   in_buf               r0       //ptr to start of activations
#define   filt                 r1       //ptr to filter array
#define   out_buf              r2       //ptr to output activations
#define   next_in_width_depth  r3       //logical input width
#define   next_out_width_depth r4       //currently unused
#define   next_in_width_32     r5       //physical input width
#define   next_out_width_32    r10      //logical output width
#define   depth                r11      //input and output depth
#define   out_width            r12      //output width
#define   out_height           r13      //output height
#define   filt_height          r25      //filter height (width is <=5 by definition)
#define   filt_zero            r7       //ilter zero point
#define   bias_sum             r14      //ptr to bias values
#define   ptr_max              r15      //ptr to max and mins
#define   recip_level          r3      //quantization output level
#define   recip_shift          r8       //shift for accumulator is outputs larger
#define   stride_v             r28      //vertical stride (horizontal stride assumed 1
//scaler values
#define   ptr_w0               r16      //ptr to weights for depth loop
#define   ptr_w1               r17      //ptr to weights for width loop
#define   c24                  r9       //const = 24
#define   c8                   r6       //const = 8
#define   bias_ptr             r18      //ptr to bias value for width loop
#define   ptr_xin              r19      //ptr to activations for depth loop
#define   ptr_xin_bias_ptr     r19:18   //ptrs to bias and activations
#define   ptr_x0               r22      //ptr to acts for width loop
#define   ptr_x1               r20      //ptr to acts for depth loop
#define   ptr_y                r23      //ptr to outputs for width loop
#define   depth_cnt            r26      //depth count 
#define   filt_size            r8       //size fo filters for each depth
#define   next_in_width_depth_stride r28//vertical input stride
#define   zzzz                 r7       //filter zero offsets 0,3
#define   ___z                 r27      //filter zero offsets 0
#define   zzz_                 r4       //filter zero offsets 1
#define   __zz                 r24      //filter zero offsets 1
#define   zz__                 r15      //filter zero offsets 2
#define   _zzz                 r21      //filter zero offsets 2
#define   z___                 r11      //filter zero offsets 3
#define   out_width4           r3       //rounded output width
//vector values
#define   vrecip                v0      //splat quantized level values
#define   vshamt_vec            v1      //splat quantized shift values
#define   max                   v2      //max vecs
#define   min                   v3      //min vecs
#define   bias_val              v4      //bias vector
#define   x0                   v30      //activations 0-3
#define   x1                   v29      //activations 4-7
#define   w4321                v5       //taps pos 3
#define   w_432                v6       //taps pos 2
#define   w__43                v27      //taps pos 1
#define   w___4                v29      //taps pos 0
#define   w3210                v28      //taps pos 0
#define   w210_                v7       //taps pos 1
#define   w10__                v8       //taps pos 2
#define   w0___                v9       //taps pos 3
#define   x7x5x6x4             v29      //input 4-7 1st shuffle
#define   x7x6x5x4             v10      //input 4-7 2nd shuffle
#define   x3x1x2x0             v30      //input 0-3 1st shuffle
#define   x3x2x1x0             v11      //input 0-3 2nd shuffle
#define   s0                   v12      //acumulator 0
#define   s1                   v13      //acumulator 1
#define   s2                   v14      //acumulator 2
#define   s3                   v15      //acumulator 3
#define   z0                   v16      //gemsuma 0
#define   z1                   v17      //gemsuma 1
#define   z2                   v18      //gemsuma 2
#define   z3                   v19      //gemsuma 3
#define   d0                   v20      //quantized output 0
#define   d1                   v21      //quantized output 1
#define   d1d0                 v24      //quantized output 0,1
#define   d2                   v22      //quantized output 2
#define   d3                   v23      //quantized output 3
#define   d3d2                 v25      //quantized output 2,3
#define   d3210                v25      //quantized and clipped outputs 0-3
#define SSR        <<1:rnd:sat:shift    //simplfy mpy instruction
#define PV(VSRC) .word (0x1DFFE020+VSRC)//debug vec reg
#define PS(SSRC) .word (0x1DFFE100+SSRC)//debug scaler reg
/* =========================================================================== */
   {  allocframe(#56)                              //0th entry on stack (56+8)/4=20
      c8 = #8                                      //constant value shift 8
      c24= #24                                     //constant value shift 24
      out_width = memw(sp+#2<<2)                   //load output width
   } {
      memd(sp+#0)  = r17:16                        //save on stack
      memd(sp+#8)  = r19:18                        //save on stack
      out_width = add(out_width, #3)               //round up to 4
   } {
      memd(sp+#16) = r21:20                        //save on stack
      memd(sp+#24) = r23:22                        //save on stack
      out_width = lsr(out_width, #2)               //divide by 4
   } {
      memd(sp+#32) = r25:24                        //save on stack
      memd(sp+#40) = r27:26                        //save on stack
   } {
      filt_zero = memw(sp+#21<<2)                  //load filt zero
      depth  = memw(sp+#17<<2)                     //depth
   } {
      M0 = next_in_width_depth                     //set vert filter stride
      out_height = memw(sp+#19<<2)                 //load output height count
      zzzz  = vsplatb(filt_zero)                   //splat filt zero
   } {
      next_out_width_32 = memw(sp+#16<<2)          //load activation width physical
      stride_v = memw(sp+#26<<2)                   //veticl stride
      depth = lsr(depth, #5)                       //depth/32
      _zzz  = lsr(zzzz, c8)                        //set pos 2 gemsuma mask
   } {
      filt_height = memw(sp+#20<<2)                //filter mask height
      memw(sp+#17<<2) = depth                      //save depth count for later
      __zz  = lsr(_zzz, c8)                        // gemsuma mask pos 1
      next_in_width_depth_stride = mpyi(next_in_width_depth,stride_v) //vertical stride
   } {
      memw(sp+#48) = out_width                     //
      recip_shift = memw(sp+#25<<2)                //qunatization shift
      zzz_  = asl(zzzz, c8)                        // gemsuma mask pos 1
   } {
      vshamt_vec= vsplat(recip_shift)              //splat shift
      ptr_max = memw(sp+#23<<2)                    //ptr to max and min
      filt_size = filt_height                      //tride for filter for each depth
   } {
      bias_sum = memw(sp+#22<<2)                   //ptr to bias values
      max = vmem(ptr_max+#0)                       //load current max vec
      ___z  = lsr(__zz, c8)                        // gemsuma mask pos 0
      filt_height = add(filt_height, #-1)          //decrement filt height for sw pipeline
   } {
      min = vmem(ptr_max+#1)                       //load current min vec
      depth_cnt = memw(sp+#17<<2)                  //depth count initialize
      zz__  = asl(zzz_, c8)                        // gemsuma mask pos 2
      z___  = asl(zzz_,#16)                        // gemsuma mask pos 3
   }
/* ----------------------------------------------------------------------------- */
     .balign 32
.L_height:
   {  out_height = add(out_height, #-1)            //decrement height count
      ptr_xin_bias_ptr = combine(in_buf, bias_sum) //init depth ptr and bias ptr
      ptr_w0 = filt                                //init filter ptr
      recip_level = memw(sp+#24<<2)                //quantization level read from origin 1st
   }
/* ----------------------------------------------------------------------------- */
.L_depth:
   {  x1 = vmemu(ptr_xin+#1)                       //[P, 0]load activations 4-7
      ptr_x0 = ptr_xin                             //init data ptr for depth loop
      ptr_x1 = ptr_xin                             //[WIDTH, P] data ptr for filt loop
   } {
      x0 = vmemu(ptr_x1++M0)                       //[P, 2]load activations 0-3
   } {
      x7x5x6x4.b = vshuff(x1.b)                    //[P, 3]shuffle inputs 4-7
      out_width = memw(sp+#18<<2)                  //initialize output width count
      vrecip = vmem(recip_level++#1)               //splat quant level VECTOR
   } {
      x3x1x2x0.b = vshuff(x0.b)                    //[P, 4]shuffle inputs 0-3
      bias_val = vmem(bias_ptr++#1)                //init bias offset vector
      memw(sp+#52) = recip_level                   //quantization level save to temp location
   } { 
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[P, 5]shuffle inputs 4-7
      ptr_y = out_buf                              //init output ptr
      p3 = !cmp.eq(r0, r0)                         //disable store for pipeline
      out_width4 = memw(sp+#48)                    //
   } {
      loop1(.L_width, out_width4)                  //init width loop
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[P, 6]shuffle inputs 0-3
      w3210     = vmem(ptr_w0+#0)                  //[P, 6]load first 4 taps
   }
/* --------------------------------------------------------------------------- */
     .balign 32
.L_width:
   {  w___4.cur = vmem(ptr_w0+#1)                  //[0, 7]taps output 0
      s0.uw = vrmpy(x7x6x5x4.ub, w___4.ub)         //[0, 7]filter even output
      loop0(.L_vert, filt_height)                  //[WIDTH, P]setup filter loop
      ptr_w1 = add(ptr_w0, #256)                   //[WIDTH, P]advance taps by 256
   } {
      z0.uw = vrmpy(x7x6x5x4.ub, ___z.ub)          //[0, 8]gemsuma for output 0
      z1.uw = vrmpy(x7x6x5x4.ub, __zz.ub)          //[0, 8]gemsuma for output 1
      w4321.uw = vlsr(w3210.uw, c8)                //[0, 8]create taps output 3
      s0.w = vadd(s0.w, bias_val.w)                //[WIDTH, P]add bias to acc 0
   } {
      s1 = bias_val                                //[WIDTH, P]add bias to acc 1
      s0.uw += vrmpy(x3x2x1x0.ub, w3210.ub)        //[0, 9]filter even output
      w210_.w = vasl(w3210.w, c8)                  //[0, 9]creat taps for output 1
      ptr_x0 = add(ptr_x0, #128)                   //[WIDTH,P]+4 32 depths for stride 1
   } {
      s2 = bias_val                                //[WIDTH, P]add bias to acc 2
      z2.uw = vrmpy(x3x2x1x0.ub, zz__.ub)          //[0,10]gemsuma for output 2
      z3.uw = vrmpy(x3x2x1x0.ub, z___.ub)          //[0,10]gemsuma for output 3
      w4321.w += vasl(w___4.w, c24)                //[0,10]creat taps for output 3
   } {
      s3 = bias_val                                //[WIDTH, P]add bias to acc 3
      w10__.w = vasl(w210_.w, c8)                  //[0,11]create taps for output 2
      s1.uw += vrmpy(x3x2x1x0.ub, w210_.ub)        //[0,11]filter even output
      out_width = add(out_width, #-4)              //[WIDTH]decremnt width count
   } 
/* --------------------------------------------------------------------------- */
      .balign 32
.L_vert:
   {  s3.uw += vrmpy(x7x6x5x4.ub, w4321.ub)        //[0,12]filter output 3
      x1 = vmemu(ptr_x1+#1)                        //[1, 0]load 4-7 activations
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, zzzz.ub)         //[0,13]gemsuma output 0
      z1.uw += vrmpy(x3x2x1x0.ub, zzz_.ub)         //[0,13]gemsuma output 1
      w_432.uw = vlsr(w4321.uw, c8)                //[0,13]create taps output 2
   } {
      s2.uw += vrmpy(x3x2x1x0.ub, w10__.ub)        //[0,14]filter output 2
      w0___.w = vasl(w10__.w, c8)                  //[0,14]create taps output 3
      x0 = vmemu(ptr_x1++M0)                       //[1, 2]load 0-3 activations
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w_432.ub)        //[0,15]filter output 2
      w__43.uw = vlsr(w_432.uw, c8)                //[0,15]create taps output 1
      x7x5x6x4.b = vshuff(x1.b)                    //[1, 3]shuffle inputs 4-7
   } {
      z2.uw += vrmpy(x7x6x5x4.ub, _zzz.ub)         //[0.16]gemsuma output 2
      z3.uw += vrmpy(x7x6x5x4.ub, zzzz.ub)         //[0,16]gemsuma output 3
      x3x1x2x0.b = vshuff(x0.b)                    //[1, 4]shuffle inputs 0-3
   } {
      s1.uw += vrmpy(x7x6x5x4.ub, w__43.ub)        //[0,17]filter output 1
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[1, 5]shuffle inputs 4-7
   } {
      s3.uw += vrmpy(x3x2x1x0.ub, w0___.ub)        //[0,18]filter output 3
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[1, 6]shuffle inputs 0-3
      w3210 = vmem(ptr_w1++#1)                     //[1, 6]taps output 0
   } {
      w___4.cur = vmem(ptr_w1++#1)                 //[1, 7]taps output 0
      s0.uw += vrmpy(x7x6x5x4.ub, w___4.ub)        //[1, 7]filter output 0
   } {
      z0.uw += vrmpy(x7x6x5x4.ub, ___z.ub)         //[1, 8]gemsuma output 0
      z1.uw += vrmpy(x7x6x5x4.ub, __zz.ub)         //[1, 8]gemsuma output 1
      w4321.uw = vlsr(w3210.uw, c8)                //[1, 8]creat taps output 3
   } {
      s0.uw += vrmpy(x3x2x1x0.ub, w3210.ub)        //[1, 9]filter output 0
      w210_.w = vasl(w3210.w, c8)                  //[1, 9]create taps output 1
   } {
      z2.uw += vrmpy(x3x2x1x0.ub, zz__.ub)         //[1,10]gemsuma output 2
      z3.uw += vrmpy(x3x2x1x0.ub, z___.ub)         //[1,10]gemsuma output 3
      w4321.w += vasl(w___4.w, c24)                //[1,10]create taps output 3
   } {
      w10__.w = vasl(w210_.w, c8)                  //[1,11]create taps output 2
      s1.uw += vrmpy(x3x2x1x0.ub, w210_.ub)        //[1,11]filter output 1
   }:endloop0                                      //  
/* --------------------------------------------------------------------------- */
   {  s3.uw += vrmpy(x7x6x5x4.ub, w4321.ub)        //[1,12]filter position 3
      d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]convert 32 to 16bit outputs
      w_432.uw = vlsr(w4321.uw, c8)                //[1,13]create taps for position 1
   } {
      z0.uw += vrmpy(x3x2x1x0.ub, zzzz.ub)         //[1,13]gemsuma for pos 0
      z1.uw += vrmpy(x3x2x1x0.ub, zzz_.ub)         //[1,13]gemsuma for pos 1
      w0___.w = vasl(w10__.w, c8)                  //[1,14]create taps for pos 2
   } {
      s0.w = vsub(s0.w, z0.w)                      //[WIDTH]subtract gemsuma from acc 0
      s2.uw += vrmpy(x3x2x1x0.ub, w10__.ub)        //[1,14]filter position 2
      w__43.uw = vlsr(w_432.uw, c8)                //[1,15]create filter for pos 1
   } {
      s2.uw += vrmpy(x7x6x5x4.ub, w_432.ub)        //[1,15]filter for pos 2
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]pack 4 outputs into 128b reg
   } {
      s0.w = vasl(s0.w, vshamt_vec.w)              //shift acc0 left
      s1.uw += vrmpy(x7x6x5x4.ub, w__43.ub)        //[1,17]filter position 1
   } {
      z2.uw += vrmpy(x7x6x5x4.ub, _zzz.ub)         //[1.16]gemsuma for pos 2
      z3.uw += vrmpy(x7x6x5x4.ub, zzzz.ub)         //[1,16]gemsuma for pos 3
      s1.w = vsub(s1.w, z1.w)                      //subtract suma from acc1
      ptr_x1 = ptr_x0                              //[WIDTH, P]init ptr x1 for next width
   } {
      if(p3) vmemu(ptr_y++#1) = d3210              //[WIDTH, E]store 4 outputs
      s2.w = vsub(s2.w, z2.w)                      //subtract gemsuma from acc2
      s3.uw += vrmpy(x3x2x1x0.ub, w0___.ub)        //[1,18]filter position 3
   } {
      s3.w = vsub(s3.w, z3.w)                      //subtract gemsuma
      d0.w = vmpye(s0.w, vrecip.uh)                //[0,15]multiply by 1/max
      s1.w = vasl(s1.w, vshamt_vec.w)              //shift acc 1 left
      p0 = !cmp.gt(out_width, #-3)                 //[WIDTH]see if acc1 is valid
   } {
      d0.w += vmpyo(s0.w, vrecip.h):SSR            //[0,17]quantize acc 0
      if(p0) s1 = s0                               //overwrite acc 1 if not valid
      s2.w = vasl(s2.w, vshamt_vec.w)              //shift acc 2 left
      p0 = !cmp.gt(out_width, #-2)                 //[WIDTH]see if acc 2 is valid
   } {
      s3.w = vasl(s3.w, vshamt_vec.w)              //do special shift left of acc
      min.w = vmin(min.w, d0.w)                    //[0,22]update min
      max.w = vmax(max.w, d0.w)                    //[0,18]update max
      p1 = !cmp.gt(out_width, #-1)                 //[WIDTH]check if acc 3 is valid
   } {
      d1.w = vmpye(s1.w, vrecip.uh)                //[0,22]multiply by 1/max
      x1 = vmemu(ptr_x0+#1)                        //[P, 0]load values 4-7
      if(p0) s2 = s0                               //overwrite 3rd acc. if not valid
   } {
      d1.w += vmpyo(s1.w, vrecip.h):SSR            //[0,23]quatize 2nd set of values
      x0 = vmemu(ptr_x1++M0)                       //load vals 0-3
      if(p1) s3 = s0                               //overwrite 3th acc with valid data
   } {
      max.w = vmax(max.w, d1.w)                    //[0,26]update maxes
      d2.w = vmpye(s2.w, vrecip.uh)                //[0,15]multiply by 1/max
      x7x5x6x4.b = vshuff(x1.b)                    //[P, 3]sguffle vlaues 4 to 7
      p3 = cmp.eq(r0, r0)                          //enable output store
   } {
      min.w = vmin(min.w, d1.w)                    //[0,27]update mins
      d1d0.h = vpack(d1.w, d0.w):sat               //[0,27]pack 1st and 2nd data
      d2.w += vmpyo(s2.w, vrecip.h):SSR            //[0,17]quantize 3rds block of data
   } {
      min.w = vmin(min.w, d2.w)                    //[0,22]update mins
      d3.w = vmpye(s3.w, vrecip.uh)                //[0,22]multiply by 1/max
      x3x1x2x0.b = vshuff(x0.b)                    //[P, 4]shuffle data 0 to 3
   } {
      max.w = vmax(max.w, d2.w)                    //[0,18]update max's
      d3.w += vmpyo(s3.w, vrecip.h):SSR            //[0,23]quantize 32 to 8bitd
      x7x6x5x4.b = vshuff(x7x5x6x4.b)              //[P, 5]2nd shuffl e of values 4 to 7
   } {
      max.w = vmax(max.w, d3.w)                    //[0,26]update max
      min.w = vmin(min.w, d3.w)                    //[0,27]update min
      x3x2x1x0.b = vshuff(x3x1x2x0.b)              //[P, 6]shuffle data again 
      w3210     = vmem(ptr_w0+#0)                  //[P, 6]load weights 0 to 3
   }:endloop1 //end width
/* --------------------------------------------------------------------------- */
   {  d3d2.h = vpack(d3.w, d2.w):sat               //[WIDTH, E]pack 3rd and 4th chunk together
      ptr_w0 += asl(filt_size, #8)                 //[DEPTH,E]filt_size = filt_height*256 //4*3*64
      ptr_xin = add(ptr_xin, next_in_width_32)     //[DEPTH]update data ptr to next sub line
   } {
      depth_cnt = add(depth_cnt, #-1)              //[DEPTH,E]decrement depth count by 32
      out_buf = add(out_buf, next_out_width_32)    //[DEPTH]update out pt to next sub line
      loop1(.L_width, out_width4)                  //[DEPTH]start next width loop
   } {
      p0 = cmp.eq(depth_cnt, #0)                   //[DEPTH,E]
      d3210.ub = vpack(d3d2.h, d1d0.h):sat         //[WIDTH, E]pack 64/64 together
      recip_level = memw(sp+#52)                   //read quantization level from temp location
   } {
      vmemu(ptr_y+#0) = d3210                      //[WIDTH, E]save last 128 actvtns.
      if(!p0) jump .L_depth                        //[DEPTH,E]next chunk of 32 depth
   }//end depth
/* ----------------------------------------------------------------------------- */
   {  p0 = cmp.eq(out_height, #0)                  //check if height used up 
      depth_cnt = memw(sp+#17<<2)                  //depth reload
      in_buf=add(in_buf,next_in_width_depth_stride)//stride update vertically
      if(!p0.new) jump:nt .L_height                //next row
   }//end height
/* ----------------------------------------------------------------------------- */
      ptr_max = memw(sp+#23<<2)                    //reload ptr to max/mins
   {
      r17:16 = memd(sp+#0)                         //restore
      vmem(ptr_max+#0) = max                       //save off max vals
   } {
      r19:18 = memd(sp+#8)                         //restore
      vmem(ptr_max+#1) = min                       //save off min vals
   } {
      r21:20 = memd(sp+#16)                        //restore
      r23:22 = memd(sp+#24)                        //restore
   } {
      r25:24 = memd(sp+#32)                        //restore
      r27:26 = memd(sp+#40)                        //restore
   } {
      dealloc_return                               //return
   }
/* =========================================================================== */
.L_end:
/* =========================================================================== */
      .size dwconv2dbbb_s1_5xN_asm, .L_end-dwconv2dbbb_s1_5xN_asm
/* =========================================================================== */
