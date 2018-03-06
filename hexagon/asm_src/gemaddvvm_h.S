
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
 */
/*======================================================================*/
/*  FUNCTIONS      : gemaddvvm_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Add y row to each row of matrix z, add column values */
/*                 x to all columns of z.                               */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 07/09/16       created                          */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*                                                                      */
/*     ->  5*N/128+6                                                    */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 240 bytes                                             */
/*     ASSUMPTIONS                                                      */
/*        z output and input data is 128byte aligned and multiple of 32 */
/*        y input data is 128byte aligned and multiple of 32            */
/*        x is aligned to 8bytes                                        */
/*  C MODEL                                                             */
/*======================================================================*/
#if 0
void gemaddvmm_cn (int *x, int *y, int *z, int N, int M, int * maxmin)
{
    int i, j;
    for (i=0; i < N; i++) {
      for (j=0; j < 32; j++) {
            z[i*M+j] += x[i] + y[j] ;
            maxmin[1] = (z[i*M+j] > maxmin[1] ? z[i*M+j] : maxmin[1];
            maxmin[0] = (z[i*M+j] < maxmin[0] ? z[i*M+j] : maxmin[0];
      }
    }
    return;
}
#endif
/*======================================================================*/
        .text
        .file "gemaddvvm_h.S"
        .global gemaddvvm_asm
        .balign 32
        .type  gemaddvvm_asm, @function
gemaddvvm_asm:
/*======================================================================*/
#define ptr_x0     r0   //row sums
#define ptr_y0     r1   //column sums
#define ptr_z0     r2   //product accumulator
#define n          r3   //size of array
#define m          r4   //size of array
#define ptr_maxmin r5   //
#define reset      r9   //
#define sum1sum0   r11:10 //
#define sum1       r11  //
#define sum0       r10  //
#define ptr_z1     r8   //
#define c4         r6   //
#define mstride    M0   //
/*======================================================================*/
#define x0         v0   //
#define z0         v1   //
#define z1         v2   //
#define x2         v3   //
#define z2         v4   //
#define z3         v5   //
#define y0         v6   //
#define maxomaxe   v9:8   //
#define maxo       v9   //
#define maxe       v8   //
#define max        v8   //
#define minomine   v11:10 //
#define mino       v11 //
#define mine       v10 //
#define min        v10  //
/*======================================================================*/
       {   
           y0 = vmem(ptr_y0+#0)          //[P, 0]
           n  = lsr(n, #1)               //[P, 0]
           m = asl(m, #2)                //[P, 0]
       } {
           dcfetch(ptr_x0+#1<<5)         //[P, 1]
           mstride = m                   //[P, 1]
           n = add(n, #-1)               //[P, 1]
           max = vmem(ptr_maxmin+#1)     //[P, 1] 
       } {
           min = vmem(ptr_maxmin+#0)     //[P, 1] 
           ptr_z1 = ptr_z0               //[P, 2]
           sum1sum0 = memd(ptr_x0++#1<<3)//[0, 0]
       } {
           reset = memw(sp+#0)           //[P, 2]
           x0 = vsplat(sum0)             //[0, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[0, 1]
           z0.w = vadd(y0.w, z0.w)       //[0, 1]
       } {
           p2 = !cmp.eq(reset, #0)       //reset 0 accumulate
           x2 = vsplat(sum1)             //[0, 2]
           z1.w = vadd(x0.w, z0.w)       //[0, 2]
           vmem(ptr_z1++mstride)= z1.new //[0, 2]
       } {
           c4 = #4                       //
           if(p2) max = z1               //[1, 6]
           if(p2) min = z1               //[1, 6]
           loop0(.L_loopN, n)            //[P, 5]
       } {
           z2.tmp = vmem(ptr_z0++mstride)//[0, 3]
           z2.w = vadd(x2.w, z2.w)       //[0, 3]
           dcfetch(ptr_x0+#3<<5)         //[0, 3]
           min.w = vmin(min.w, z1.w)     //[0, 3]
       } 
#if 0
       {
           max.w = vmax(max.w, z1.w)     //[0, 4]
           z3.w = vadd(y0.w, z2.w)       //[0, 4]
           vmem(ptr_z1++mstride)= z3.new //[0, 4]
           sum1sum0 = memd(ptr_x0++#1<<3)//[1, 0]
       } {    
           max.w = vmax(max.w, z3.w)     //[0, 5]
           x0 = vsplat(sum0)             //[1, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[1, 1]
           z0.w = vadd(y0.w, z0.w)       //[1, 1]
       } {    
           min.w = vmin(min.w, z3.w)     //[0, 6]
           x2 = vsplat(sum1)             //[1, 2]
           z1.w = vadd(x0.w, z0.w)       //[1, 2]
           vmem(ptr_z1++mstride)= z1.new //[1, 2]
       } {    
           z2.tmp = vmem(ptr_z0++mstride)//[1, 3]
           z2.w = vadd(x2.w, z2.w)       //[1, 3]
           dcfetch(ptr_x0+#3<<5)         //[1, 3]
	   min.w = vmin(min.w, z1.w)     //[1, 3]
       } 
#endif
           .balign 32
.L_loopN:
       {    
	   max.w = vmax(max.w, z1.w)     //[1, 4]
           z3.w = vadd(y0.w, z2.w)       //[1, 4]
           vmem(ptr_z1++mstride)= z3.new //[1, 4]
           sum1sum0 = memd(ptr_x0++#1<<3)//[2, 0]
       } {    
	   min.w = vmin(min.w, z3.w)     //[1, 5]
           x0 = vsplat(sum0)             //[2, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[2, 1]
           z0.w = vadd(y0.w, z0.w)       //[2, 1]
       } {    
	   max.w = vmax(max.w, z3.w)     //[1, 6]
           x2 = vsplat(sum1)             //[2, 2]
           z1.w = vadd(x0.w, z0.w)       //[2, 2]
           vmem(ptr_z1++mstride)= z1.new //[2, 2]
       } {    
           z2.tmp = vmem(ptr_z0++mstride)//[2, 3]
           z2.w = vadd(x2.w, z2.w)       //[2, 3]
           dcfetch(ptr_x0+#3<<5)         //[2, 3]
	   min.w = vmin(min.w, z1.w)     //[2, 3]
       }:endloop0 
       {    
	   max.w = vmax(max.w, z1.w)     //[2, 4]
           z3.w = vadd(y0.w, z2.w)       //[2, 4]
           vmem(ptr_z1++mstride)= z3.new //[2, 4]
       } {    
           loop0(.L_peak, #5)   
	   min.w = vmin(min.w, z3.w)     //[2, 5]
       }
.L_peak:
       {  
           maxomaxe = vshuff(maxe, maxe, c4)
       } {
           maxe.w = vmax(maxo.w, maxe.w)
           minomine = vshuff(mine, mine, c4)
       } {
           c4 = add(c4, c4)
           mine.w = vmin(mino.w, mine.w)
       }:endloop0
       {   vmem(ptr_maxmin+#1) = max     //[E, 0]
       } {
           vmem(ptr_maxmin+#0) = min     //[E, 1]
	}{
           jumpr r31                     //[E, 1]
       }
/*======================================================================*/
.L_end:
      .size gemaddvvm_asm, .L_end-gemaddvvm_asm


/*======================================================================*/
        .global gemaddvvm_asm1
        .balign 32
        .type  gemaddvvm_asm1, @function
gemaddvvm_asm1:
/*======================================================================*/
       {   
           y0 = vmem(ptr_y0+#0)          //[P, 0]
           n  = lsr(n, #1)               //[P, 0]
           m = asl(m, #2)                //[P, 0]
       } {
           dcfetch(ptr_x0+#1<<5)         //[P, 1]
           mstride = m                   //[P, 1]
           n = add(n, #-1)               //[P, 1]
       } {
           ptr_z1 = ptr_z0               //[P, 2]
           sum1sum0 = memd(ptr_x0++#1<<3)//[0, 0]
       } {
           x0 = vsplat(sum0)             //[0, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[0, 1]
           z0.w = vadd(y0.w, z0.w)       //[0, 1]
       } {
           x2 = vsplat(sum1)             //[0, 2]
           z1.w = vadd(x0.w, z0.w)       //[0, 2]
           vmem(ptr_z1++mstride)= z1.new //[0, 2]
       } {
           c4 = #4                       //
           loop0(.L1_loopN, n)           //[P, 5]
       } {
           z2.tmp = vmem(ptr_z0++mstride)//[0, 3]
           z2.w = vadd(x2.w, z2.w)       //[0, 3]
           dcfetch(ptr_x0+#3<<5)         //[0, 3]
       } 
       .balign 32
.L1_loopN:
       {    
           z3.w = vadd(y0.w, z2.w)       //[1, 4]
           vmem(ptr_z1++mstride)= z3.new //[1, 4]
           sum1sum0 = memd(ptr_x0++#1<<3)//[2, 0]
       } {    
           x0 = vsplat(sum0)             //[2, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[2, 1]
           z0.w = vadd(y0.w, z0.w)       //[2, 1]
       } {    
           x2 = vsplat(sum1)             //[2, 2]
           z1.w = vadd(x0.w, z0.w)       //[2, 2]
           vmem(ptr_z1++mstride)= z1.new //[2, 2]
       } {    
           z2.tmp = vmem(ptr_z0++mstride)//[2, 3]
           z2.w = vadd(x2.w, z2.w)       //[2, 3]
           dcfetch(ptr_x0+#3<<5)         //[2, 3]
       }:endloop0 

       {   z3.w = vadd(y0.w, z2.w)       //[2, 4]
           vmem(ptr_z1++mstride)= z3.new //[2, 4]
       } {
           jumpr r31                     //[E, 1]
       }
/*======================================================================*/
.L1_end:
      .size gemaddvvm_asm1, .L1_end-gemaddvvm_asm1

