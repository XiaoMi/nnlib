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
/*======================================================================*/
/*  FUNCTIONS      : gvmaddvvm_asm                                      */
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
void gvmaddvmm_cn (int *x, int *y, int *z, int N, int M, int * maxmin)
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
        .file "gvmaddvvm_h.S"
        .global gvmaddvvm_asm
        .balign 32
        .type  gvmaddvvm_asm, @function
gvmaddvvm_asm:
/*======================================================================*/
#define ptr_x0     r0   //row sums
#define ptr_y0     r1   //column sums
#define ptr_z0     r2   //product accumulator
#define n          r3   //size of array
#define m          r4   //size of array
#define ptr_max    r5   //
#define reset      r6   //
#define ptr_z1     r8   //product accumulator
#define neg        r11  //
#define sum0       r10  //
#define c4         r7   //
#define mstride    M0   //
/*======================================================================*/
#define x0         v0   //
#define z0         v1   //
#define z1         v2   //
#define y0         v4   //
#define vneg       v3   //
#define maxomaxe   v7:6 //
#define maxo       v7   //
#define maxe       v6   //
#define max        v5   //
/*======================================================================*/
           .balign 32
       {   
           y0 = vmem(ptr_y0+#0)          //[P, 0]
           m = asl(m, #2)                //[P, 0]
           neg = ##0x80000000            //[P, 0]
       } {
           ptr_z1 = ptr_z0               //[P, 1]
           vneg = vsplat(neg)            //[P, 1]
           reset = memw(sp+#0)           //[P, 1]
           n = add(n, #-1)               //[P, 1]
       } {
           dcfetch(ptr_x0+#1<<5)         //[P, 2]
           max = vmem(ptr_max+#0)        //[P, 2] 
           p2 = !cmp.eq(reset, #0)       //[P, 2]reset 0 accumulate
           c4 = #4                       //[P, 2]
       } {
           mstride = m                   //[P, 3]
           z1 = vneg                     //[P, 3]
           if(p2) max = vneg             //[P, 3]most negative value
           sum0 = memw(ptr_x0++#1<<2)    //[0, 0]
       } {
           loop0(.L_loopN, n)            //[P, 4]
           x0 = vsplat(sum0)             //[0, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[0, 1]
           z0.w = vadd(y0.w, z0.w)       //[0, 1]
       } 
           .balign 32
.L_loopN:
       {
           max.w = vmax(max.w, z1.w)     //[0, 2]
           z1.w = vadd(x0.w, z0.w)       //[0, 2]
           vmem(ptr_z1++mstride):nt= z1.new //[0, 2]
           sum0 = memw(ptr_x0++#1<<2)    //[1, 0]
       } {
           dcfetch(ptr_x0+#2<<5)       //[0, 3]
           x0 = vsplat(sum0)             //[1, 1]
           z0.tmp = vmem(ptr_z0++mstride)//[1, 1]
           z0.w = vadd(y0.w, z0.w)       //[1, 1]
       }:endloop0
       {
           max.w = vmax(max.w, z1.w)     //[1, 2]
           z1.w = vadd(x0.w, z0.w)       //[1, 2]
           vmem(ptr_z1+#0):nt= z1.new       //[1, 2]
           loop0(.L_peak, #5)            //[P, 0]
       } {
           maxe.w = vmax(max.w, z1.w)    //[E, 0]
       }
.L_peak:
       {  
           maxomaxe=vshuff(maxe,maxe,c4) //[0, 0]
       } {
           maxe.w = vmax(maxo.w, maxe.w) //[0, 1]
           c4 = add(c4, c4)              //[0, 1]
       }:endloop0
       {   vmem(ptr_max+#0) = maxe       //[E, 0]
	}{
           jumpr r31                     //[E, 0]
       }
/*======================================================================*/
.L_end:
      .size gvmaddvvm_asm, .L_end-gvmaddvvm_asm
