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
#if 0
void copyNto4(uint8_t *out,uint8_t *in, int n, int in_offset, int in_depth)
{
   int i;
   for(i=0; i < n; i++) {
     if(in_depth == 1) {
       out4[4*i+0] = in3[1*i+0];
       out4[4*i+1] = in_offset
       out4[4*i+2] = in_offset
       out4[4*i+3] = in_offset;
     } else if(in_depth == 2) {
       out4[4*i+0] = in3[2*i+0];
       out4[4*i+1] = in3[2*i+1];
       out4[4*i+2] = in_offset;
       out4[4*i+3] = in_offset;
     } else if(in_depth == 3)
       out4[4*i+0] = in3[3*i+0];
       out4[4*i+1] = in3[3*i+1];
       out4[4*i+2] = in3[3*i+2];
       out4[4*i+3] = in_offset;
     } else { //in_depth == 4
       out4[4*i+0] = in3[4*i+0];
       out4[4*i+1] = in3[4*i+1];
       out4[4*i+2] = in3[4*i+2];
       out4[4*i+3] = in3[4*i+3];
     }
   }
   return;
}
#endif
/*======================================================================*/
            .text
            .global copyNto4_asm
            .balign 32
            .type  copyNto4_asm, @function
copyNto4_asm:
/* ------------------------------------------------------------------- */
#define  out4             r0   //aligned 128 out ptr
#define  inN              r1   //non-aligned input data
#define  elemns           r2   //num out elements
#define  in_offset        r3   //input "zero"
#define  in_depth         r4   //input depth 1-4
#define  cntrl_tab        r5   //cntrl table for 3 to 4 expansion
/* ------------------------------------------------------------------- */
#define  sizeb            r6   //shuffling bytes
#define  sizeh            r7   //shuffling shorts
/* ------------------------------------------------------------------- */
#define  vpred            v0   //vector predicate container
#define  vperm34          v1   //permute 3 to 4 control
#define  vin_offset       v2   //vectorized input zero
#define  b95b00           v3   //bytes 0-95
#define  b63b00           v3   //bytes 0-63
#define  b31b00           v3   //bytes 0-31
#define  b255b128_b127b00 v5:4 //output bytes and extra garbage
#define  b255b128         v5   //garbage
#define  b127b00          v4   //output data
#define  h127h64_h63h00   v7:6 //output hwords and garbage
#define  h127h64          v7   //garbage 
#define  h63h00           v6   //shuffled bytes to hwords
#define  PV(VSRC) .word (0x1DFFE020+VSRC) //debug vec reg
#define  PS(SSRC) .word (0x1DFFE100+SSRC) //debug sca reg
/* ------------------------------------------------------------------- */
      {  r6 = ##0x01000000                     //4th byte mux in filt_offset
         r10 = asl(in_depth, #5)               //input does 1-32,2-64,3-96,4-128 bytes
      } {
         M0 = r10                              //increment ptr 
         vperm34 = vmem(cntrl_tab+#0)          //permute for 3 to 4
         vpred = vsplat(r6)                    //vmem(cntrl_tab+#1)
      } {
         in_offset = vsplatb(in_offset)        //expenad zero for activations.
         loop1(.L_switch, #2)                  //set up switch loop
         sizeh = #-2                           //shuffle pairs of bytes
      } {
         vin_offset = vsplat(in_offset)        //mux in vector
         sizeb = #-1                           //shuffle pairs of bytes
         p0 = cmp.eq(in_depth, #1)             //is depth 1?
         if(p0.new) jump:nt .L_depth1          //
      } {
         r10= ##0x01010101                     //set up vpredicate
         p1 = cmp.eq(in_depth, #2)             //is depth 2?
         if(p1.new) jump:nt .L_depth2          //
      } {
         q0 = vand(vpred, r10)                 //setup merge predicate for 
         p2 = cmp.eq(in_depth, #3)             //is depth 3? 
         if(p2.new) jump:t .L_depth3           //
      }
/* ------------------------------------------------------------------- */
         .balign 32
.L_depth4:
      {  loop0(.L_loop44, elemns)                         //4byte loop default
      }
/* ------------------------------------------------------------------- */
.L_loop44:
      {  b127b00 =  vmemu(inN++M0)                        //increment by 128
      } {
         vmem(out4++#1) = b127b00                         //store 128bytes
      }:endloop0:endloop1
/* ------------------------------------------------------------------- */
      .balign 32
.L_depth3:
      {  b95b00 =  vmemu(inN++M0)                         //increment by 96
         loop0(.L_loop34, elemns)                         //3bytes loop
      }
/* ------------------------------------------------------------------- */
.L_loop34:
      {  b127b00 = vdelta(b95b00, vperm34)                //choose 4th byte
      } {
         b127b00 = vmux(q0,vin_offset,b127b00)            //merge in in_offset values
         b95b00 =  vmemu(inN++M0)                         //increment by 96
      } {
         vmem(out4++#1) = b127b00                         //store 128bytes
      }:endloop0:endloop1
/* ------------------------------------------------------------------- */
      .balign 32
.L_depth2:
      {  b63b00 =  vmemu(inN++M0)                         //increment by 64
         loop0(.L_loop24, elemns)                         //2byte depth
      }
/* ------------------------------------------------------------------- */
.L_loop24:
      {  h127h64_h63h00 = vshuff(vin_offset,b63b00,sizeh) //choose 2th bytes
      } {
         b63b00 = vmemu(inN++M0)                          //increment by 64
      } {
         vmem(out4++#1) = h63h00                          //store 128bytes
      }:endloop0:endloop1
/* ------------------------------------------------------------------- */
         .balign 32
.L_depth1:
      {  b31b00 =  vmemu(inN++M0)                         //increment by 32
         loop0(.L_loop14, elemns)                         //1byte loop
      }
/* ------------------------------------------------------------------- */
.L_loop14:
      {  h127h64_h63h00 = vshuff(vin_offset,b31b00,sizeb) //shuffle bytes P 1 P 0
      } {  
         b31b00 =  vmemu(inN++M0)                         //increment by 32
      } {
         b255b128_b127b00=vshuff(vin_offset,h63h00,sizeh) //shuffle in  hwords of in_offset
         vmem(out4++#1) = b127b00.new                     //store 128bytes
      }:endloop0:endloop1
/* ------------------------------------------------------------------- */
.L_switch:
      {  jumpr r31
      }
.L_end:
      .size copyNto4_asm, .L_end-copyNto4_asm
