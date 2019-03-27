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
 * FUNCTION
 *   to_d32 convert image of width,height,depth to d32 format each slice of depth 
 *   for each point on own line.
 *
 * input 0-6  0123456012345601234560123456 <-how depth is transformed
 * output     0000
 *            1111 
 *            2222 
 *            3333
 *            4444
 *            5555
 *            6666
 *
 *   *depth multiple of 32
 *   *output width multiple of 4 and output aligned to 128
 *
 * CODESIZE 
 *   240 bytes
 *    
 * C MODEL
 */
#if 0
void to_d32(
   uint8_t * data, int width, uint8_t * data_d32, int next_width, int height, int depth)
{
  int i,j,k;

  for(i=0; i < height; i++)
  {
    for(j=0; j < width; j++)
    {
      for(k=0; k < depth; k++)
      {
        data_d32[(i*depth+(k/32)*32)*next_width+j*32+(k%32)] = data[(i*width+j)*depth+k];
      }
    }
  }
  return;
}
#endif
/* ---------------------------------------------------------------- */
       .text
       .global to_d32_asm
       .balign 32
       .type to_d32_asm, @function 
to_d32_asm:
/* ---------------------------------------------------------------- */
#define ptr_ini        r0     //input pointer normal depth wise       
#define width          r1     //num depths wide
#define out_d32_i      r2     //output pointer into d32 array
#define next_width_d32 r3     //rnd(width,4)*32 - physical width of d32 array
#define height         r4     //number of group of widths
#define depth          r5     //depth of input
/* ---------------------------------------------------------------- */
#define count          r18    //temp coutn to index into depths
#define depth_count    r17    //number of 32s
#define depth_iters    r10    //round up to neaest 4
#define horz_iters     r13    //round horixontal upto mod 4
#define ptr_in0        r14    //temp input ptr
#define ptr_in1        r11    //temp input ptr
#define depth3         r12    //3 * depth
#define width_depth    r15    //width * depth of input
#define out_d32_0      r16    //temp output pointer
#define out_d32_1      r8     //temp output pointer
#define c32            r6     //const to shuffle 32 block
#define c64            r7     //const to shuffle 64 block
#define in_width_depth r9     //width*depth of output(inc. pad)
/* ---------------------------------------------------------------- */
#define x03210         v0     //1st 4 32 depth chunks
#define x13210         v1     //2nd depth chunks
#define x23210         v2     //3rd depth chunks
#define x33210         v3     //4td depth chunks
#define x03322_x01100  v5:4   //shuffle 32 together
#define x03322         v5     //shuffle 32 together
#define x01100         v4     //shuffle 32 together
#define x13322_x11100  v7:6   //shuffle 32 together
#define x13322         v7     //shuffle 32 together
#define x11100         v6     //shuffle 32 together
#define x3333_x2222    v11:10 //shuffle 64 together
#define x3333          v11    //shuffle 64 together
#define x2222          v10    //shuffle 64 together
#define x1111_x0000    v9:8   //shuffle 64 together
#define x1111          v9     //shuffle 64 together
#define x0000          v8     //shuffle 64 together
/* ---------------------------------------------------------------- */
    { allocframe(#24)                                 //
      c32 = #-32                                      //set up shuffle size
      c64 = #-64                                      //set up shuffle size
    } {
      memd(sp+#0)  = r17:16                           //stack
      memd(sp+#8)  = r19:18                           //stack
      in_width_depth = mpyi(width, depth)             //size of each physical input line
      depth_iters = lsr(depth, #5)                    //depth / 32
    } {
      width_depth=mpyi(next_width_d32,depth_iters)    //size of each physical output line
      depth_count = depth_iters                       //unrounded  depth cnt
      M0 = depth                                      //set up CR
    } {
      depth_iters = add(depth_iters, #3)              //round up to nearest 4
      depth3 = mpyi(depth, #3)                        //
      horz_iters = add(width, #3)                     //round horiontal to 4
    } {
      depth_iters = lsr(depth_iters, #2)              //depth /128
      depth3 = sub(#128, depth3)                      //retard point 128-3*depth
      horz_iters = lsr(horz_iters, #2)                //
    } {
      M1 = depth3                                     //
    }
/* ---------------------------------------------------------------- */
   .balign 32
.L_loop_height:
  {   height = add(height, #-1)                       //next height
      out_d32_0 = out_d32_i                           //tmp out ptr
      ptr_in0 = ptr_ini                               //tmp in ptr
      loop1(.L_loop_width, horz_iters)                //line loop
  }
/* ---------------------------------------------------------------- */
   .balign 32
.L_loop_width:
  {   out_d32_1 = out_d32_0                           //
      ptr_in1 = ptr_in0                               //
      loop0(.L_loop_depth4, depth_iters)              //set up depth loop
      count = add(depth_count, #4)                    //
  }
/* ---------------------------------------------------------------- */
   .balign 32
.L_loop_depth4:                                       //do multiples of 4 of depth
  {   x03210 = vmemu(ptr_in1++M0)                     //1st 128nyutes of depth 0
  } {
      x13210 = vmemu(ptr_in1++M0)                     //2nd 128nyutes of depth 0
  } {
      x23210 = vmemu(ptr_in1++M0)                     //3rd 128nyutes of depth 0
  } {
      x03322_x01100 = vshuff(x13210, x03210, c32)     //
  } {
      x33210 = vmemu(ptr_in1++M1)                     //4th 128nyutes of depth 0
  } {
      count = add(count, #-4)                         //decrement depth count
  } {
      x13322_x11100 = vshuff(x33210, x23210, c32)     //get even depths together
  } {
      p1 = cmp.gt(count, #1)                          //2 or mode?
  } {
      x1111_x0000 = vshuff(x11100, x01100, c64)       //get 0's together
      vmem(out_d32_1+#0) = x0000.new                  //
      out_d32_1 = add(out_d32_1, next_width_d32)      //
  } {
      if(p1) vmem(out_d32_1+#0) = x1111               //
      if(p1)out_d32_1 = add(out_d32_1,next_width_d32) //
      p1 = cmp.gt(count, #2)                          //3 or more
  } {
      x3333_x2222 = vshuff(x13322, x03322, c64)       //
      if(p1) vmem(out_d32_1+#0) = x2222.new           //
      if(p1)out_d32_1 = add(out_d32_1,next_width_d32) //
      p1 = cmp.gt(count, #3)                          //4 or more?
  } {
      if(p1) vmem(out_d32_1+#0) = x3333               //
      if(p1)out_d32_1 = add(out_d32_1,next_width_d32) //
  }:endloop0
/* ---------------------------------------------------------------- */
  {
      out_d32_0 = add(out_d32_0, #128)                //advance out depths by 128
      ptr_in0 = addasl(ptr_in0, depth, #2)            //advance input by 4 depths
  }:endloop1
/* ---------------------------------------------------------------- */
  {   out_d32_i = add(out_d32_i, width_depth)         //
      ptr_ini = add(ptr_ini, in_width_depth)          //
      p0 = cmp.eq(height, #0)                         //
      if(!p0.new) jump:t .L_loop_height               //new width
  }
/* ---------------------------------------------------------------- */
  {   r17:16 = memd(sp+ #0)                           //
      r19:18 = memd(sp+ #8)                           //
  } {
      dealloc_return                                  //
  }
.L_end:
/*=============================================================================*/
      .size to_d32_asm, .L_end-to_d32_asm
/*=============================================================================*/
