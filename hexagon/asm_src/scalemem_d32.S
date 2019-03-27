/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
/*  FUNCTIONS      : scalemem_d32                                       */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 copy rows, scaling u8 via given scale/offset         */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  GLS                 13/09/17       created                          */
/*======================================================================*/
/*  IDEAL-CYCLE-COUNT:                                                  */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 348 bytes                                             */
/*     STACK    =  0 bytes                                              */
/*  ASSUMPTIONS                                                         */
/*        arrays rows are vector aligned                                */
/*  C MODEL                                                             */
/*======================================================================*/
#if 0
//
// input & output are both height x width x (32 bytes)
//  ptr_out, stride_out  are aligned
//  ptr_in, stride_in are aligned
//  'width', 'height' each >= 0.
// scl_off has a scale (<= 32767) in 16 lsbs and a signed offset in the 16 msbs.
// (it will also work for -ve scale).
//
void scalemem_d32_hvx(
		uint8_t * ptr_out, int32_t stride_out,
		uint8_t const * ptr_in, int32_t stride_in,
		int32_t height,
		int32_t width,
		int32_t scl_off)
{
	int i,j;
	int fullwid = width *32;
	int32_t offs = (slc_off >> 16) << 8;		// zero bias
	int32_t scl = (int16_t)slc_off

	for( i = 0; i < heignt; j++ ){
		for( int j = 0;  j < fullwid; j ++ ){
			int32_t inval = ptr_in[ stride_in * i + j ];
			int32_t prod = inval * scl  + offs;
			int32_t result = (prod + 16384) >> 15;
			ptr_out[ stride_out * i + j] = saturate_u8( result );
		}
	}
	// NOTE: if scl_off is in range 32704 .. 32767, the 'scaling' is a no-op and
	// it becomes a copy. i.e. if (scl_off>>6) == 511.
    // if scl_off is in the range 0x7f808000 .. 0x7f80803f, it becomes a 1's complement
    //operation.
}
// The scaling is actually dones as follows:
//   (1)  multiply input by 8 lsbs of scale  (u8*u8)
//   (2)  mul input by 8 msbs of scale (u8*s8); and add this to (1) result >>8 (lsr)
//        this is in (in*scale)>>8
//   (3)  add the 'offset' using saturated h add
//   (4)  >>7 , round, sat to u8.
//
#endif
/*=============================================================================*/
    .text
    .file "scalemem_d32.S"
    .global scalemem_d32_hvx
    .balign 32
    .type  scalemem_d32_hvx, @function
scalemem_d32_hvx:
/*=============================================================================*/
#define ptr_out         r0
#define stride_out      r1
#define ptr_in          r2
#define stride_in       r3
#define height          r4
#define width           r5
#define scl_off         r6
#define const7          r7
#define scale_lo        r8
#define scale_hi        r9
#define offs            r10
#define rowloops        r11
/*=============================================================================*/
#define vzero           v0
#define vinp            v1
#define vout            v2
#define voffs           v3
#define vprodA0         v10
#define vprodA1         v11
#define vvprodA         v11:10
#define vprodB0         v12
#define vprodB1         v13
#define vvprodB         v13:12
#define vvoffs          v5:4
#define vprodC0         v10
#define vprodC1         v11
#define vvprodC         v11:10
/*=============================================================================*/
    {
       scl_off = memw(sp+#0)                    // get scl_off
       r14 = add(width, #-1)                    // will find the loop count with this.
    }{
       offs = combine( scl_off.h, scl_off.h)    // upper 16 bits of offs
       scale_lo = vsplatb(scl_off)              // lo 8 bits of scale
       r15 = asr(scl_off, #8)
    }{
       scl_off  = asr(scl_off,#6)
       scale_hi = vsplatb(r15)                  // hi 8 bits of scale
    }{
       rowloops = lsr(r14,#2)                   // # of row loops (one less than vecs needed)
       voffs = vsplat(offs)
    }{
    // if scl_off == 511 now, it means we can use a copy instead.
    // if scl_off = 0x1fe0200, we use 2's complement.
    //
    // the row loops advance ptr_in by (rowloops+1)*128,
    // and ptr_out by rowloops*128; so deduct those amounts from stride_in and stride_out

       r14 = asl(rowloops,#7)                   // # find the row advance in h loop
       stride_in = add(stride_in, #-128)
       r15 = asl(width,#5)                      // for last-mask
       vvoffs = vcombine(voffs,voffs)           //
    }{
       stride_out = sub(stride_out,r14)
       stride_in  = sub(stride_in, r14)
       const7 = #7
       vzero = #0
    }{
#if __HEXAGON_ARCH__ >= 62
       q3 = vsetq2(r15)
#else
       q3 = vsetq(r15)
#endif
       p0 = cmp.eq(rowloops,#0)	                // true if no loops
       p1 = cmp.eq(scl_off,#511)	            // is just a copy if so.
    }
#if __HEXAGON_ARCH__ < 62
    // need to change q3 to all 1's if line width a multiple of 4.
    {
       p2 = bitsclr(width,#3)
       if( !p2.new ) jump:t .L_skip
    }{
       q3 = not(q3)
    }
.L_skip:
#endif

///////////////////////////////////////////
// the copy that needs scaling
///////////////////////////////////////////
    {
       if( p1 ) jump:nt .L_case_noS;
       p2 = cmp.eq(scl_off,#0x1fe0200)                      //2's complement if so
    }{
       if (p2) jump:nt .L_case_noS;
       loop1( .L_hloop_S, height );
    }

    .balign 32
 .L_hloop_S:
    {  vinp.cur = vmem(ptr_in++#1)                          //[1]load 
       vvprodA.uh = vmpy( vinp.ub, scale_lo.ub)             //[1]mul scale_lo by input bytes to get lo prod
       p3 = sp1loop0(.L_wloop_S, rowloops)                  //
       nop
    }{
       vprodB0.b = vshuffo(vzero.b, vprodA0.b)              //[1]>> 8 using vshuffo
       vprodB1.b = vshuffo(vzero.b, vprodA1.b)              //[1]
    }{
       vvprodB.h += vmpy( vinp.ub, scale_hi.b)              //[1]add the scale_hi product
       if( p0 ) jump:nt .L_wloop0_S                         // skip if loop count = 0
    }

    .balign 32
.L_wloop_S:
    {  vinp.cur = vmem(ptr_in++#1)                          //[1]load 
       vvprodA.uh = vmpy( vinp.ub, scale_lo.ub)             //[1]mul scale_lo by input bytes to get lo prod
       vout.ub = vasr(vprodC1.h, vprodC0.h, const7):rnd:sat //[3]
#if __HEXAGON_ARCH__ > 60
       if p3 vmem(ptr_out ++#1) = vout.new                  //[3] 
#endif
    }{
       vprodB0.b = vshuffo(vzero.b, vprodA0.b)              //[1]>> 8 using vshuffo
       vprodB1.b = vshuffo(vzero.b, vprodA1.b)              //[1]
       vvprodC.h = vadd(vvprodB.h, vvoffs.h):sat            //[2]add the offset
    }{
       vvprodB.h += vmpy( vinp.ub, scale_hi.b)              //[1]add the scale_hi product
#if __HEXAGON_ARCH__ == 60
       if p3 vmem(ptr_out ++#1) = vout                      //[3] 
#endif
    }:endloop0

.L_wloop0_S:
    {
       vout.ub = vasr(vprodC1.h, vprodC0.h, const7):rnd:sat //[3]
       if p3 vmem(ptr_out ++ #1) = vout.new                 //[3] 
       vvprodC.h = vadd(vvprodB.h, vvoffs.h):sat            //[2]add the offset
    }{ 
       vout.ub = vasr(vprodC1.h, vprodC0.h, const7):rnd:sat //[3]
       ptr_in = add(ptr_in, stride_in)
    }{
       if( q3 ) vmem(ptr_out+ #0) = vout                    // now store the last one
       ptr_out = add(ptr_out, stride_out)
    }: endloop1

    {
       jumpr r31
    }

    .balign 32
///////////////////////////////////////////
/// for when no scaling is needed - just a copy
/// This is also used for 1's complement:
///    p2    voffs         
//     0       0            copy
//     1      -1            invert
///////////////////////////////////////////
.L_case_noS:
    {
       voffs = vnot(vzero)                  // voffs = -1  (only if p2)
       loop1( .L_hloop_noS, height );
    }{
       if (!p2) voffs = vzero               // restore voffs = 0 if !p2
    }
    .balign 32
 .L_hloop_noS:
    {
       vinp.cur =vmem( ptr_in ++#1 )        // get first
       vout = vxor(vinp,voffs)
       loop0( .L_wloop_noS, rowloops )
       if( p0 ) jump:nt .L_wloop0_noS       // skip if loop count = 0
    }
     .balign 32
.L_wloop_noS:
    {
       vinp.cur = vmem(ptr_in++#1)          // load next
       vout = vxor(vinp,voffs)
       vmem( ptr_out ++ #1) = vout          // store current
    }: endloop0
.L_wloop0_noS:
    // now store the last one
    {
       ptr_in = add(ptr_in, stride_in)
       if( q3 ) vmem(ptr_out+ #0) = vout
       ptr_out = add(ptr_out, stride_out)
    }: endloop1

    {
       jumpr r31
    }
.L_end:
/*=============================================================================*/
    .size scalemem_d32_hvx, .L_end-scalemem_d32_hvx
/*=============================================================================*/
