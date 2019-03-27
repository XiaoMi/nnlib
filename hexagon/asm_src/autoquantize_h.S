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
   Desrciption
     Quantize float-point inputs to 8-bit fixed-point data. 
   Requirements
     Input/output must be aligned to 128 bytes. Final store will write only to amount needed. 
     
*/
/* ------------------------------------------------------------------------------------------ */
            .text
            .global quantize_floats_to_8b_asm
            .balign 32
            .type  quantize_floats_to_8b_asm, @function
quantize_floats_to_8b_asm:
/* ------------------------------------------------------------------------------------------ */
#define in_ptr                      r0     //ptr to input
#define out_ptr                     r1     //ptr to output 
#define elements                    r2     //
#define min_offset_in               r3     //
#define maxexp_in                   r4     //
#define scaling                     r5     //
#define mantbits                    r6     // 
#define c007f_ffff                  r7     //
#define c0080_0000                  r8     //
#define c0000_00ff                  r9     //
#define c31                         r10    //
#define partial                     r3     //
#define c128                        r8     //
#define lpcount                     r9     //
#define fr                          r7     //
#define numvec                      r10    //
#define maxblksize                  r11    //
#define block                       r12    //
#define fetch_adr                   r13    //
#define l2fparam_l                  r14    //
#define l2fparam_h                  r15    //
#define l2fparam                    r15:14 //
#define boundary                    r28
/* ------------------------------------------------------------------------------------------ */
#define vals(n)                     v0
#define s_exp(n)                    v1
#define exp(n)                      v2
#define mant(n)                     v3
#define shift(n)                    v4
#define smant(n)                    v5
#define mants(n)                    v6
#define neg_mants(n)                v7
#define word0                       v8
#define word1                       v9
#define word2                       v10
#define word3                       v11
#define halves0                     v12
#define halves1                     v13
#define bytes                       v12
#define maxexp                      v14
#define min_offset                  v15
#define const007f_ffff              v16
#define const0080_0000              v17
#define const0000_00ff              v18
#define const31                     v19
#define zero                        v20    //const zero 
#define mask                        v21
#define vscale                      v22
/* --------------------------------------------------------------------------------------- */
    { p0 = cmp.gt(elements,#0)                          //
      if (!p0.new) jumpr:nt r31                         //
      maxblksize = #4096                                //
    }{ 
      c007f_ffff = ##0x007FFFFF                         //
      c0080_0000 = ##0x00800000                         //
    }{
      const007f_ffff = vsplat(c007f_ffff)               //
      const0080_0000 = vsplat(c0080_0000)               //
      c31 = #31                                         //
      c128 = #128                                       //
    }{
      const31 = vsplat(c31)                             //
      c0000_00ff = #0x00FF                              //
      block = min(elements,maxblksize)                  //
    }{     
      const0000_00ff = vsplat(c0000_00ff)               //
      lpcount = add(block,#127)                         //
      numvec = lsr(block,#5)                            //
    }{
      boundary = addasl(in_ptr,elements,#2)             //
      l2fparam_h = #128                                 //
      l2fparam_l = combine(c128.l, numvec.l)            //
      lpcount = lsr(lpcount,#7)                         //
    }{
//    scaling = combine(scaling.l,scaling.l)            //
      maxexp = vsplat(maxexp_in)                        //
      min_offset = vsplat(min_offset_in)                //
      partial = neg(elements)                           //
    }{
      zero = #0                                         //
      vscale = vsplat(scaling)                          //
      mantbits = #23                                    //
      p3 = sp1loop0(.quantize_loop,lpcount)             //
    }{
      elements -= asl(lpcount,#7)                       //
      l2fetch(in_ptr,l2fparam)                          //
    } 
.quantize_wait_l2fetch:    
    { fr = usr                                          // wait_for_l2fetch()
    }{
      p0 = cmp.gt(fr,#-1)                               //
      if (!p0.new) jump:nt .quantize_wait_l2fetch       //
    }
.quantize_outerloop:
    { fetch_adr = in_ptr                                //
      block = min(elements,maxblksize)                  // block size for next iteration
      nop; nop                                          //
    }{
      fetch_adr += asl(lpcount,#9)                      //
      p2 = cmp.gt(block,#0)                             //
      numvec = lsr(block,#5)                            //
      lpcount = add(block,#127)                         //
    }{
      lpcount = lsr(lpcount,#7)                         //
      if !p2 jump .quantize_loop                        //
      l2fparam_l = combine(l2fparam_l.h,numvec.l)       //
      nop                                               //
    }{
      l2fetch(fetch_adr,l2fparam)                       //
      nop; nop; nop                                     //
    }
/* --------------------------------------------------------------------------------------- */
    .balign 32
.quantize_loop:
    { vals(0).cur = vmem(in_ptr++#1):nt                 //[0, 0]
      s_exp(0).w = vasr(vals(0).w,mantbits)             //[0, 0]
      smant(2).w = vadd(smant(2).w,min_offset.w)        //[1,16]
      Q0 = vcmp.gt(zero.w,s_exp(3).w)                   //[1,16]
    }{
      exp(0)  = vand(s_exp(0),const0000_00ff)           //[0, 1]
      mant(0) = vand(vals(0), const007f_ffff)           //[0, 1]
      mants(3).w = vlsr(mant(3).w,shift(3).w)           //[1,17]
      halves0.uh = vpack(word1.w,word0.w):sat           //[1,17]
    }{
      shift(0).w = vsub(maxexp.w,exp(0).w)              //[0, 2]
      p0 = cmp.gtu(boundary,in_ptr)                     //[0, 2]
      neg_mants(3).w = vsub(zero.w,mants(3).w)          //[1,18]
      word2.w = vmpye(smant(2).w,vscale.uh)             //[1,18]
    }{
      mant(0) = vor(mant(0),const0080_0000)             //[0, 3]
      shift(0).w = vmin(shift(0).w,const31.w)           //[0, 3]
      if (!p0) in_ptr = sub(in_ptr,c128)                //[0, 3]
      smant(3) = vmux(Q0,neg_mants(3),mants(3))         //[1,19]
    }{
      Q0 = vcmp.gt(zero.w,s_exp(0).w)                   //[0, 4]
      vals(1).cur = vmem(in_ptr++#1):nt                 //[0, 4]
      s_exp(1).w = vasr(vals(1).w,mantbits)             //[0, 4]
      smant(3).w = vadd(smant(3).w,min_offset.w)        //[1,20]
    }{
      mants(0).w = vlsr(mant(0).w,shift(0).w)           //[0, 5]
      exp(1)  = vand(s_exp(1),const0000_00ff)           //[0, 5]
      mant(1) = vand(vals(1), const007f_ffff)           //[0, 5]
      p0 = cmp.gtu(boundary,in_ptr)                     //[0, 5]
    }{
      neg_mants(0).w = vsub(zero.w,mants(0).w)          //[0, 6]
      shift(1).w = vsub(maxexp.w,exp(1).w)              //[0, 6]
      if (!p0) in_ptr = sub(in_ptr,c128)                //[0, 6]
      word3.w = vmpye(smant(3).w,vscale.uh)             //[1,21]
    }{
      smant(0) = vmux(Q0,neg_mants(0),mants(0))         //[0, 7]
      mant(1) = vor(mant(1),const0080_0000)             //[0, 7]
      shift(1).w = vmin(shift(1).w,const31.w)           //[0, 7]
    }{
      smant(0).w = vadd(smant(0).w,min_offset.w)        //[0, 8]
      Q0 = vcmp.gt(zero.w,s_exp(1).w)                   //[0, 8]
      vals(2).cur = vmem(in_ptr++#1):nt                 //[0, 8]
      s_exp(2).w = vasr(vals(2).w,mantbits)             //[0, 8]
    }{
      mants(1).w = vlsr(mant(1).w,shift(1).w)           //[0, 9]
      exp(2)  = vand(s_exp(2),const0000_00ff)           //[0, 9]
      mant(2) = vand(vals(2), const007f_ffff)           //[0, 9]
      halves1.uh = vpack(word3.w,word2.w):sat           //[1,22]
    }{
      word0.w = vmpye(smant(0).w,vscale.uh)             //[0,10]
      neg_mants(1).w = vsub(zero.w,mants(1).w)          //[0,10]
      shift(2).w = vsub(maxexp.w,exp(2).w)              //[0,10]
      p0 = cmp.gtu(boundary,in_ptr)                     //[0,10]
    }{
      smant(1) = vmux(Q0,neg_mants(1),mants(1))         //[0,11]
      mant(2) = vor(mant(2),const0080_0000)             //[0,11]
      shift(2).w = vmin(shift(2).w,const31.w)           //[0,11]
      if (!p0) in_ptr = sub(in_ptr,c128)                //[0,11]
    }{
      smant(1).w = vadd(smant(1).w,min_offset.w)        //[0,12]
      Q0 = vcmp.gt(zero.w,s_exp(2).w)                   //[0,12]
      vals(3).cur = vmem(in_ptr++#1):nt                 //[0,12]
      s_exp(3).w = vasr(vals(3).w,mantbits)             //[0,12]
    }{
      mants(2).w = vlsr(mant(2).w,shift(2).w)           //[0,13]
      exp(3)  = vand(s_exp(3),const0000_00ff)           //[0,13]
      mant(3) = vand(vals(3), const007f_ffff)           //[0,13]
      bytes.b = vpacko(halves1.h,halves0.h)             //[1,23]
    }{
      word1.w = vmpye(smant(1).w,vscale.uh)             //[0,14]
      neg_mants(2).w = vsub(zero.w,mants(2).w)          //[0,14]
      shift(3).w = vsub(maxexp.w,exp(3).w)              //[0,14]
    }{
      smant(2) = vmux(Q0,neg_mants(2),mants(2))         //[0,15]
      mant(3) = vor(mant(3),const0080_0000)             //[0,15]
      shift(3).w = vmin(shift(3).w,const31.w)           //[0,15]
      if p3 vmem(out_ptr++#1) = bytes                   //[1,24]
    }:endloop0
/* --------------------------------------------------------------------------------------- */
    { elements -= asl(lpcount,#7)                       //
      lc0 = lpcount                                     //
    }{
      if p2 jump .quantize_outerloop                    //
    }
.lp1end:
    { smant(2).w = vadd(smant(2).w,min_offset.w)        //[1,16]
      Q0 = vcmp.gt(zero.w,s_exp(3).w)                   //[1,16]
      mask = vnot(zero)                                 //
    }{
      mants(3).w = vlsr(mant(3).w,shift(3).w)           //[1,17]
      halves0.uh = vpack(word1.w,word0.w):sat           //[1,17]
    }{
      neg_mants(3).w = vsub(zero.w,mants(3).w)          //[1,18]
      word2.w = vmpye(smant(2).w,vscale.uh)             //[1,18]
    }{
      smant(3) = vmux(Q0,neg_mants(3),mants(3))         //[1,19]
      mask = valign(zero,mask,partial)                  //
    }{
      smant(3).w = vadd(smant(3).w,min_offset.w)        //[1,20]
    }{
      word3.w = vmpye(smant(3).w,vscale.uh)             //[1,21]
      Q1 = vcmp.gt(mask.ub,zero.ub)                     //
    }{
      halves1.uh = vpack(word3.w,word2.w):sat           //[1,22]
    }{
      bytes.b = vpacko(halves1.h,halves0.h)             //[1,23]
    }{
      if (Q1) vmem(out_ptr+#0) = bytes                  //[1,24]
    }
/* --------------------------------------------------------------------------------------- */
    { jumpr r31                                             //return
    }
.L_end:
/*=============================================================================*/
      .size quantize_floats_to_8b_asm, .L_end-quantize_floats_to_8b_asm
/*=============================================================================*/


/* ------------------------------------------------------------------------------------------ */
/*
   Desrciption
     find min/max of floats, and compare with values in min/max buffers.
   Requirements
     Input/output must be aligned to 128 bytes. 
*/
/* ------------------------------------------------------------------------------------------ */
            .global find_minmax_of_floats_asm
            .balign 32
            .type  find_minmax_of_floats_asm, @function
find_minmax_of_floats_asm:
/* ------------------------------------------------------------------------------------------ */
#define ptr_in                      r0     //ptr to input
#define length                      r1     //
#define ptr_minmax                  r2     //ptr to minmax
#define c80000000                   r3     //
#define c4                          r4     //
#define nrot                        r5     //
#define length_in_bytes             r5     //
/* ------------------------------------------------------------------------------------------ */
#define new_in                      v0     //
#define new1                        v1     //
#define fmax                        v2     //
#define fmin                        v3     //
#define new_t                       v4     //
#define const80000000               v5     //
#define const0                      v6
#define minmax                      v0     //
#define minmax_t                    v1     //
#define shuf                        v7:6   //
#define shuf_L                      v6     //
#define shuf_H                      v7     //
/* --------------------------------------------------------------------------------------- */
    { c80000000 = ##0x80000000                          //
      fmax = #0                                         //
      maxblksize = #4096                                //
    }{
      block = min(length,maxblksize)                    //
      const80000000 = vsplat(c80000000)                 //
      fmin = #0                                         //
      l2fparam_h = #128                                 //
    }{
      c4 = #4                                           //
      const0 = #0                                       //
      numvec = lsr(block,#5)                            //
      length_in_bytes = asl(length,#2)                  //
    }{
      p0 = cmp.gt(numvec,#0)                            //
      l2fparam_l = combine(l2fparam_h.l, numvec.l)      //
      Q1 = vsetq(length_in_bytes)                       //
      length -= asl(numvec,#5)                          //
    }{
      new1 = #0                                         //
      if !p0 jump .find_minmax_lp1end                   //
    }{
      l2fetch(ptr_in,l2fparam)                          //
    }
    
.find_minmax_outerloop:
    { loop0(.find_minmax_loop,numvec)                   //
      fetch_adr = ptr_in                                //
      block = min(length,maxblksize)                    // block size for next iteration
    }{
      fetch_adr += asl(numvec,#7)                       //
      numvec = lsr(block,#5)                            //
      p2 = cmp.gt(block,#31)                            //
    }
.find_minmax_wait_l2fetch:    
    { fr = usr                                          // wait_for_l2fetch()
      nop                                               //
    }{
      p0 = cmp.gt(fr,#-1)                               //
      if (!p0.new) jump:nt .find_minmax_wait_l2fetch    //
    }{
      if !p2 jump .find_minmax_loop                     //
      l2fparam_l = combine(l2fparam_l.h,numvec.l)       //
    }{
      l2fetch(fetch_adr,l2fparam)                       //
      nop                                               //
    }
/* --------------------------------------------------------------------------------------- */
    .balign 32
.find_minmax_loop:
    { new_in.tmp = vmem(ptr_in++#1)                     //[0, 0]
      fmax.w = vmax(fmax.w,new_in.w)                    //[0, 0]
      new1 = vxor(new_in,const80000000)                 //[0, 0]
      fmin.w = vmax(fmin.w,new1.w)                      //[1, 1]
    }:endloop0
/* --------------------------------------------------------------------------------------- */
.find_minmax_lp0end:
    { length -= asl(numvec,#5)                          //
      if p2 jump .find_minmax_outerloop                 //
    }
.find_minmax_lp1end:
    { fmin.w = vmax(fmin.w,new1.w)                      //[1, 1]
      p0 = bitsclr(length,#31)                          //
      if (p0.new) jump:nt .find_minmax_cont             //
    }{
      new_t.tmp = vmem(ptr_in+#0)                       //
      new_in = vmux(Q1,new_t,const0)                    //
    }{
      fmax.w = vmax(fmax.w,new_in.w)                    //
      new1 = vxor(new_in,const80000000)                 //
    }{
      fmin.w = vmax(fmin.w,new1.w)                      //
    }
/* --------------------------------------------------------------------------------------- */
.find_minmax_cont:                                               
    { shuf = vshuff(fmax,fmin,c4)                       //
      loop0(.L_reduce_loop,#4)                          //
    }{
      minmax.w = vmax(shuf_L.w,shuf_H.w)                //
      nrot = #8                                         //
    }
.L_reduce_loop:    
    { minmax_t = vror(minmax,nrot)                      //
    }{
      minmax.w = vmax(minmax.w,minmax_t.w)              //
      nrot = asl(nrot,#1)                               //
    }:endloop0
    
    { vmem(ptr_minmax+#0) = minmax                      //
    }{
      jumpr r31                                         //return
    } 
.L_end_1:
/*=============================================================================*/
      .size find_minmax_of_floats_asm, .L_end_1-find_minmax_of_floats_asm
/*=============================================================================*/

