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

/*
   DESCRIPTION
     Perform the sum of a square of activations using integral as input. The correction
     factor  const - sum( a(,)*filt_offset. 
 */
/* --------------------------------------------------------------------------------------- */
            .text
            .global gvsuma_asm
            .balign 32
            .type  gvsuma_asm, @function
gvsuma_asm:
/* --------------------------------------------------------------------------------------- */
#define ptr_xi               r0     //integral input ints
#define ptr_zi               r1     //filter output
#define integral_width       r2     //pad_l+in_width+pad_r
#define next_int_width       r3     //distance to next output > (in_width + 31)&~31
#define stride_h             r4     //vertical stride
#define filt_width           r5     //filter width 
#define filt_height          r6     //rows of filter
#define out_height           r7     //number of required output rows
#define offset               r8     //K*in_depth*filt_offset*in_offset
#define in_ptrT0             r0     //top row base of filter (1 above actual filter)
#define in_ptrT              r9     //top row of filter (1 above actual filter)
#define in_ptrB0             r10    //bottom row base of filter on actual filter)
#define in_ptrB              r11    //bottom row of filter on actual filter)
#define out_ptr              r13    //temp ptr for output
#define out_width            r12    //number of elements to compute on this row
/* --------------------------------------------------------------------------------------- */
#define topLeft              v0     //
#define botLeft              v1     //
#define Left                 v2     //
#define topRight             v4     //
#define botRight             v5     //
#define Right                v3     //
#define align_pw             v6     //
#define align_m1             v7     //
#define filt_out             v8     //
#define voffset              v9     //
#define vzero                v11    //
/* --------------------------------------------------------------------------------------- */
     {   filt_height = memw(sp+#0<<2)                  //
         out_width = lsr(integral_width, #5)           //1 / 32 
         out_height = memw(sp+#1<<2)                   //
     } {
         offset = memw(sp+#2<<2)                       //
         stride_h = mpyi(stride_h, next_int_width)     //
         filt_height = mpyi(filt_height,next_int_width)//
         out_width = add(out_width, #-1)               //
     } {
         topLeft = vmem(in_ptrT0+#0)                   //[P, 0] t31__t00
         in_ptrT = add(in_ptrT0, #128)                 //[P, 0]
         in_ptrB0  = add(ptr_xi, filt_height)          //
     } {
         botLeft.tmp = vmem(in_ptrB0+#0)               //[P, 1] b31__b00
         Left.w = vsub(botLeft.w, topLeft.w)           //[P, 1] t - b  01234567
         loop1(.L_height, out_height)                  //set up inner loop of horz sum
     } {
         voffset = vsplat(offset)                      //[P, 2]K*in_offset*filt_offset
         in_ptrT0 = add(in_ptrT0, stride_h)            //[P, 2]
         in_ptrB = add(in_ptrB0, #128)                 //[P, 1]
     }  {
         in_ptrB0 = add(in_ptrB0, stride_h)            //[P, 2]
         filt_width = asl(filt_width, #2)              //align to filt_w wirds
     }
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_height:
     {   topRight = vmem(in_ptrT++#1)                  //[0, 0]b63__b32
         loop0(.L_width, out_width)                    //set up inner loop of horz sum
     } {
         botRight.tmp = vmem(in_ptrB++#1)              //[0, 1]
         Right.w = vsub(botRight.w, topRight.w)        //[0, 1]t63__t32
     } {
         out_ptr = ptr_zi                              //
         p3 = cmp.eq(out_width, #0)                    //deal with xcase width <= 32
         if(p3.new) jump:nt .L_skip                    //deal with xcase width <= 32
     } 
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_width:
     {   align_pw = valign(Right, Left, filt_width)    //[0, 3]
         topRight = vmem(in_ptrT++#1)                  //[1, 0]b63__b32
     } {
         filt_out.w = vsub(align_pw.w, Left.w)         //[0, 4]
         Left = Right                                  //[0, 4]
         botRight.tmp = vmem(in_ptrB++#1)              //[1, 1]
         Right.w = vsub(botRight.w, topRight.w)        //[1, 1]t63__t32
     } {
         filt_out.w = vsub(voffset.w, filt_out.w)      //[0, 5]
         vmem(out_ptr++#1) = filt_out.new              //[0, 5]
     }:endloop0
/* --------------------------------------------------------------------------------------- */
.L_skip:
     {   align_pw = valign(Right, Left, filt_width)    //[1, 3]
         topLeft = vmem(in_ptrT0+#0)                   //[P, 0] t31__t00
         in_ptrT = add(in_ptrT0, #128)                 //[P, 0]
         ptr_zi = add(ptr_zi, next_int_width)          //update output pointer
     } {
         filt_out.w = vsub(align_pw.w, Left.w)         //[1, 4]
         botLeft.tmp = vmem(in_ptrB0+#0)               //[P, 1] b31__b00
         Left.w = vsub(botLeft.w, topLeft.w)           //[P, 1] t - b  01234567
         in_ptrT0 = add(in_ptrT0, stride_h)            //[P, 1]
     } {
	 vmem(out_ptr+#0) = filt_out.new               //[E, 6]
         filt_out.w = vsub(voffset.w, filt_out.w)      //[1, 5]
         in_ptrB = add(in_ptrB0, #128)                 //[P, 2]
         in_ptrB0 = add(in_ptrB0, stride_h)            //[P, 2]
     }:endloop1
/* --------------------------------------------------------------------------------------- */
     {   
         jumpr r31                                     //
     }
/* --------------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------------- */
.L_end:		
      .size gvsuma_asm, .L_end-gvsuma_asm