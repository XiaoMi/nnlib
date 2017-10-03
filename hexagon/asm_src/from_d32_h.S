/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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
 *   from_d32 convert d32 image of padded_rounded_width,height,depth to 
 *   regular non d32 format each slice of depth for each point on own line.
 *
 * input      0000
 *            1111
 *            2222
 *            3333
 *            4444
 *            5555
 *            6666
 * outut 0-6  0123456012345601234560123456 <-how depth is transformed
 *
 *   *depth multiple of 32
 *   *input width multiple of 4 and aligned to 128
 *
 * CODESIZE
 *   240 bytes
 *
 * C MODEL
 */

#if 0
void from_d32(
  uint8_t * data_d32, int in_width_pad, uint8_t * data, int width, int height, int depth)
{
  int i,j,k;
  for(i=0; i < height; i++)
  {
    for(j=0; j < width; j++)
    {
      for(k=0; k < depth; k++)
      {
        data[(i*width+j)*depth+k] = data_d32[(i*depth+(k/32)*32)*in_width_pad+j*32+(k%32)];
      }
    }
  }
  return;
}
#endif
/* ---------------------------------------------------------------- */
       .text
       .global from_d32_asm
       .balign 32
       .type from_d32_asm, @function 
from_d32_asm:
/* ---------------------------------------------------------------- */
#define ptr_ini         r0     //ptr to depth 32 data      
#define next_width_d32  r1     //(pad+rnd(width,4))*32
#define out_ptri        r2     //normal row
#define width           r3     //output width after conversion
#define height          r4     //number rows
#define depth           r5     //normal depth mult of 32
/* ---------------------------------------------------------------- */
#define depth_count     r18    // number of bytes left in depth
#define depth_iters     r8     //round up to neaest 4
#define horz_iters      r13    //width iteration round up to 4
#define ptr_in0         r17    //temp d32 in ptr
#define out_ptr0        r16    //normal temp output ptr
#define ptr_in0out_ptr0 r17:16 //packet data
#define ptr_in1         r11    //temp input ptr depth loop
#define out_ptr1        r10    //temp output ptr depth loop
#define out_width_depth r15    //size of normla output row
#define out_ptr2        r20    //tmp output ptr sub depth loop
#define c32             r6     //const for shuffling groups of 32
#define c64             r7     //const for shuffling groups of 64
#define in_width_depth  r9     //size of padded d32 row
#define mdst            r19    //tmp outpt rnded to 128
#define dalign          r14    //ditance from aligned ptr start to end of data
#define width_count     r12    //width counter pts left
#define scratch         r21    //temp buffer on stack
#define max_width       r22    //1,2,3,4 depths left
/* ---------------------------------------------------------------- */
#define x3333           v11    //dr32 row 3
#define x2222           v10    //dr32 row 2
#define x1111           v9     //dr32 row 1
#define x0000           v8     //dr32 row 0
#define x11010_x01010   v5:4   //0 and 1 shuffled
#define x11010          v5     //0 and 1 shuffled
#define x01010          v4     //0 and 1 shuffled
#define x13232_x03232   v7:6   //2 and 3 shuffled
#define x13232          v7     //2 and 3 shuffled
#define x03232          v6     //2 and 3 shuffled
#define x13210_x03210   v1:0   //0,1 and 2,3 shuffled
#define x13210          v1     //0,1 and 2,3 shuffled
#define x03210          v0     //0,1 and 2,3 shuffled
#define x33210_x23210   v3:2   //0,1 and 2,3 shuffled
#define x33210          v3     //0,1 and 2,3 shuffled
#define x23210          v2     //0,1 and 2,3 shuffled
#define qprolog         q0     //1st part of last depth fragment store
#define qepilog         q1     //2nd part of last depth fragment store
/* ---------------------------------------------------------------- */
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca reg
#define PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
/* ---------------------------------------------------------------- */
      {   allocframe(#640)                             //allocate scratch 
          c32 = #-32                                   //shuffle 32bytes
          c64 = #-64                                   //shuffle 64bytes
      } {
          sp = and(sp, #-128)                          //align stack to 128 bytes 
          out_width_depth = mpyi(width, depth)         //size of normal row
          depth_iters = lsr(depth, #5)                 //depth / 32
      } {
          memd(sp+#0)  = r17:16                        //save regs
          memd(sp+#8)  = r19:18                        //save regs
          in_width_depth=mpyi(next_width_d32,depth_iters)//size of padded row
          depth_iters = lsr(depth_iters, #2)           //depth /128
      } {
          memd(sp+#16)  = r21:20                       //save regs
          memd(sp+#24)  = r23:22                       //save regs
          horz_iters = add(width, #3)                  //round to 4
      } {
          horz_iters = lsr(horz_iters, #2)             //round up to mult of 4
          M0 = depth                                   //incrment ptr
      }
/* ---------------------------------------------------------------- */
       .balign 32
.L_loop_height:
      {   height = add(height, #-1)                    //decrement height count
          ptr_in0out_ptr0 = combine(ptr_ini, out_ptri) //next row ptrs
          loop1(.L_loop_width, horz_iters)             //loop for width
          width_count = add(width, #4)                 //set up width counter
      }
/* ---------------------------------------------------------------- */
       .balign 32
.L_loop_width:
      {   out_ptr1 = out_ptr0                          //next depth ptr 
          ptr_in1 = ptr_in0                            //next input depth ptr
          loop0(.L_loop_depth4, depth_iters)           //loop of depth / 128 itns
          depth_count = depth                          //loop counter for conditional store
      } {
          p0 = cmp.eq(depth_iters, #0)                 //depth < 4 ?
          if(p0.new) jump:nt .L_epi_only               //if depth < 4then skip to last chunk loop
          width_count = add(width_count, #-4)          //used up 4 more width elements
      }
/* ---------------------------------------------------------------- */
       .balign 32
.L_loop_depth4:                                        //do multiples of 4 of depth
      {   out_ptr2 = out_ptr1                          //set up 1st ptr
          x0000 = vmem(ptr_in1+#0)                     //load row 0 
          ptr_in1 = add(ptr_in1, next_width_d32)       //next d32 row
      } {
          x1111.tmp = vmem(ptr_in1+#0)                 //load row 1
          x11010_x01010 = vshuff(x1111, x0000, c32)    //shuffle rows 0 and 1
          ptr_in1 = add(ptr_in1, next_width_d32)       //next d32 row
      } {
          x2222 = vmem(ptr_in1+#0)                     //load row 2
          ptr_in1 = add(ptr_in1, next_width_d32)       //next d32 row
      } {
          x3333.tmp = vmem(ptr_in1+#0)                 //load row 3
          x13232_x03232 = vshuff(x3333, x2222, c32)    //shuffle rows 2 and 3
          ptr_in1 = add(ptr_in1, next_width_d32)       //next d32 row
      } {
          x13210_x03210 = vshuff(x03232,x01010,c64)    //shuffle 0,1,2,3
          out_ptr1 = add(out_ptr1, #128)               //increment 128 for next tiome
      } {
          x33210_x23210 = vshuff(x13232,x11010,c64)    //shuffle 0,1,2,3
          depth_count = add(depth_count, #-128)        //used up 128 worth of input
      } {
          vmemu(out_ptr2++M0) = x03210                 //save 128bytes of depth 0
          p0 = cmp.gt(width_count, #1)                 //are there >= 2?
      } {
          if(p0) vmemu(out_ptr2++M0) = x13210          //save 128bytes of depth 1
          p0 = cmp.gt(width_count, #2)                 //are there >= 3?
      } {
          if(p0) vmemu(out_ptr2++M0) = x23210          //save 128bytes of depth 2
          p0 = cmp.gt(width_count, #3)                 //are there >= 4?
      } {
          if(p0) vmemu(out_ptr2+#0)  = x33210          //save 128bytes of depth 3
      }:endloop0
      .balign 32
.L_epi_only:
      {   p0 = cmp.eq(depth_count, #0)                 //if out of depths skip 
          if(p0.new) jump:nt .L_skip_epi               //
      }
/* ----------------------------------------------------------- */
      {   x0000 = vmem(ptr_in1+#0)                     //get last block of depth
          ptr_in1 = add(ptr_in1, next_width_d32)       //inc input ptr
          scratch = add(sp, #128)                      //set up scratch buffer ptr
      } {
          x1111.tmp = vmem(ptr_in1+#0)                 //get last block of depth
          x11010_x01010 = vshuff(x1111, x0000, c32)    //shuffle 0 and 1
          ptr_in1 = add(ptr_in1, next_width_d32)       //inc input ptr
      } {
          x2222 = vmem(ptr_in1+#0)                     //get last block of depth
          ptr_in1 = add(ptr_in1, next_width_d32)       //inc input ptr
          max_width = width_count                      //copy width_count
          p0 = cmp.gt(width_count, #4)                 //if > 4 set to 4
      } {
          x3333.tmp = vmem(ptr_in1+#0)                 //get last block of depth
          x13232_x03232 = vshuff(x3333, x2222, c32)    //shuffle 2 and 3
          ptr_in1 = add(ptr_in1, next_width_d32)       //inc input ptr
      } {
          x13210_x03210 = vshuff(x03232, x01010, c64)  //shuffle 0,1,2,3
          vmem(scratch+#0) = x03210.new                //store 1st depth chunk
          if(p0) max_width = #4                        //4 until width < 4
      } {
          vmem(scratch+#1) = x13210                    //spec. store 2nd depth chunk
          loop0(.L_last_depths, max_width)             //excute 1,2,3 or 4
      } {
          x33210_x23210 = vshuff(x13232, x11010, c64)  //shuffle 0,1,2,3
          vmem(scratch+#2) = x23210.new                //spec. store 3rd depth chunk
      } {
          vmem(scratch+#3) = x33210                    //spec. store 4tt depth chunk
      }
      .balign 32
.L_last_depths:
      {   mdst = and(out_ptr1, #127)                   //ptr to mod 128
          x03210 = vmem(scratch++#1)                   //read last bit of depth from scratch 
      } {
          qprolog = vsetq(out_ptr1)                    //find dist to ptr
          dalign = add(mdst, depth_count)              //dist to end of data
          mdst = sub(#0, out_ptr1)                     //create left rotate
      } {
          x03210 = vror(x03210, mdst)                  //rotate left by ptr
      } {
          qepilog = vsetq(dalign)                      //do mask for 2nd store
          p1 = cmp.gt(dalign, #127)                    //is it a double store?
          if(p1.new) jump:nt .L_gt1280                 //skip over logic for 1 part store
      } {
          qprolog = or(qprolog, !qepilog)              //compound 2 masks
          qepilog = and(qprolog, !qprolog)             //cancel 2nd mask
      } 
.L_gt1280:
      {   if( qepilog) vmem(out_ptr1+#1) = x03210      //cond store 2nd part
      } {
          if(!qprolog) vmem(out_ptr1++M0) = x03210     //store 1st part, in ptr
      }:endloop0
/* ----------------------------------------------------------- */
      .balign 32
.L_skip_epi:
      {   ptr_in0 = add(ptr_in0, #128)                 //advace in d32 by 128
          out_ptr0 = addasl(out_ptr0, depth, #2)       //advance input by 4 depths
      }:endloop1
/* ---------------------------------------------------------------- */
      {   out_ptri = add(out_ptri, out_width_depth)    //next output line
          ptr_ini = add(ptr_ini, in_width_depth)       //next input d32 line block
          p0 = cmp.eq(height, #0)                      //next width
          if(!p0.new) jump:t .L_loop_height            //jump
      }
/* ---------------------------------------------------------------- */
      {  r17:16 = memd(sp+ #0)                         //retieve regs
         r19:18 = memd(sp+ #8)                         //retieve regs
      } {
         r21:20 = memd(sp+#16)                         //retieve regs
         r23:22 = memd(sp+#24)                         //retieve regs
      } {
         dealloc_return                                //return stack and out
      }
.L_end:
/*=============================================================================*/
      .size from_d32_asm, .L_end-from_d32_asm
/*=============================================================================*/
