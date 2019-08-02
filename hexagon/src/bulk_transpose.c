
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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

#include "nn_bulk_transpose.h"
#include "nn_graph.h"
#include "hvx_inlines.h"
#include "quantize.h"

static int __attribute__((cold))
find_workbuf_size(struct bulk_transpose_parms const * parms)
{
	int d = parms->in_dims[3];
	if( d < 1 || d > 16 || ((d-1)&d)!=0 ) return -1;
	int hwmax = max_i32( parms->in_dims[1], parms->in_dims[2] );
	if( hwmax <= 8) return 8*128;
	int maxn = 128 >> __builtin_ctz(d);
	if( hwmax >= maxn) return maxn*128;
	// round 'hwmax' up to a power of 2.
	hwmax = 1<< (32-__builtin_clz((unsigned)hwmax-1));
	return hwmax * 128;
}
//
// see nn_bulk_tranpose.h for description of perform_bulk_transpose and perform_bulk_transpose2
//

int
perform_bulk_transpose(
		void *outp,			// output buffer b * w * h * d bytes
		void const *inp,	// input  buffer b * h * w * d bytes
		void * work_area,	// vector aligned work area (see below).
		uint32_t b,			// batches >= 1
		uint32_t h,			// input height >= 1 (outpt width)
		uint32_t w,			// input width >= 1 (output height)
		uint32_t d,			// element size, bytes (power of 1, 1,2,4..16
		int flags)			// reserved for multi-threading tag.
{
	struct bulk_transpose_parms prms;
	prms.in_dims[0] = b;
	prms.in_dims[1] = h;
	prms.in_dims[2] = w;
	prms.in_dims[3] = d;
	prms.in_h_stride = d*w;
	prms.out_h_stride = d*h;
	prms.in_b_stride = prms.out_b_stride = h*w*b;
	return perform_bulk_transpose_2( outp, inp, work_area, &prms,flags);
}

int
perform_bulk_transpose_2(
		void *outp,			// output buffer b * w * h * d bytes
		void const *inp,	// input  buffer b * h * w * d bytes
		void * work_area,	// vector aligned work area (see below).
		struct bulk_transpose_parms const * parms,
		int flags)			// reserved for multi-threading tag. For now
{
	if( work_area == NULL)
		return find_workbuf_size( parms);
	int d = parms->in_dims[3];
	if( d < 1 || d > 16 || ((d-1)&d)!=0 ) return -1;
	int log2d = __builtin_ctz(d);
	int tileN = 128u>>log2d;			// biggest tile for this d.
	int log2_tilen = 7-log2d;

	int h_in = parms->in_dims[1];
	int w_in = parms->in_dims[2];
	// divide the input into tiles of  tileN x tileN x d; go down the cols.
	// in the input, and store across rows in the output.
	// 'full' tiles are 128 bytes wide, tileN high; but the last tile col
	// and the last tile row may not be full.

	int h_tiles = (h_in+(tileN-1))>>log2_tilen;
	int w_tiles = (w_in+(tileN-1))>>log2_tilen;
	if( h_tiles <= 0 || w_tiles <= 0) return 0;

	int in_h_stride = parms->in_h_stride;
	int out_h_stride = parms->out_h_stride;

	// batch loop
	for( int ibatch =0; ibatch < parms->in_dims[0]; ibatch++){
		uint8_t const* rdp0 = (uint8_t const *)inp + ibatch * parms->in_b_stride;
		uint8_t * wrp0 = (uint8_t *)outp + ibatch * parms->out_b_stride;

		for(int tilex = 0; tilex < w_tiles; tilex++){
			int colwid = min_i32( tileN, w_in-tilex*tileN);
			__builtin_assume(colwid >= 1);
			for( int tiley = 0; tiley < h_tiles; tiley++){
				int rowht = min_i32( tileN, h_in-tiley*tileN);
				__builtin_assume(rowht >= 1);
				uint8_t const* rdp = rdp0 + 128*tilex + in_h_stride*tileN * tiley;

				if( likely(colwid==tileN)){ //load full-wid cols
					HVX_Vector * wp = (HVX_Vector*)work_area;
					for(int i =0; i < rowht; i++){
						wp[i] = q6op_V_vldu_A((HVX_Vector const*)(rdp + i*in_h_stride));
					}
				}else{
					vmemcpy_2d_general_asm(
					      colwid*d,			// bytes wide
					      rowht,			//rows
					      work_area,		// destination address, any allowed
					      128,				// row pitch of dest; any allowed
					      rdp,				// source address, any allowed
					      in_h_stride);		// row pitch of source; any allowed
				}
				/// call the bulk transpose op
				transpose_rectangle( work_area, d, rowht,colwid);

				// now unload it
				uint8_t * wrp = wrp0 + 128*tiley + out_h_stride*tileN * tilex;
				if( likely(rowht==tileN)){ //copy full-wid vecs
					HVX_Vector const * rp = (HVX_Vector*)work_area;
					for(int i =0; i < colwid; i++){
						q6op_vstu_AV((HVX_Vector *)(wrp + i*out_h_stride), rp[i]);
					}
				}else{
					vmemcpy_2d_general_asm(
					      rowht*d,			// bytes wide
					      colwid,			//rows
					      wrp,				// destination address, any allowed
					      out_h_stride,		// row pitch of dest; any allowed
					      work_area,		// source address, any allowed
					      128);				// row pitch of source; any allowed
				}
			} //tilex
		} // tiley
	} // ibatch
	return 0;
}
//////////////////////////////////////////////////////////
//
// transpose_rectangle( uint8_t *buffer, int elementsize, int Nh, int Nw )
//
// This function is intended to form the core of a large transpose op on flat tensors e.g.
//   [ B, H W, D ] -> [ B, W, H, D ]
// .. where 'D' is one of 1,2,4,8,16
// and H and W are both large enough that you can't do shuffle/deal of rows.
//
// The core operation is given an array NhxNw of D-byte elements
//  nd performs a transpose on that rectangle, in-placing, the result being Nw rows x Nh cols.
// We require Nh,Nw to both be >= 2  and <= 128/D (and if one of them is 2,3, or 4 there are faster
// ways to do this).
//
// In the below, N = max(Nh,Nw).
//
// The supplied array is a work area; the output is written back to the same area, and in some
// cases it must be larger than 'N' rows. The work area must be vector aligned, and the row
// pitch is always 128 byes even if N*D is smaller than 128.
//
// Important: the N supplied will be rounded up to a power of two and to at least 8; this
//  determines the size of the work area. E.g. if you supply Nh=Nw=18, you must must supply a 32-row
// work area (4096 bytes); but the operation may go a bit faster than if Nh=Nw=32 is specified.
//
// The largest buffer case is where D=1 and N=65..128; in all such cases you need to supply
// a 128x128 buffer (16K bytes).
// In the case where D =16, Nh,Nw must be in range 2..8 and all are handled the same way; the buffer
// size is 1K bytes.

//======= method =====================
// In the below, N refers to the supplied N rounded up to a power of 2, at least 8.
// (and N <= 128/D; so if D=16, N can only be 8).
//
// Procedure is
/// (1) if N>8, transpose blocks of 8x8 elements. The 'superarray' of these blocks is (N/8) square;
//      this value can be one of 2,4,8,16 (the upper limit depending on D).
//  (2) perform N/8 operations, each of which does transposes within the 8x8 blocks in each row.
//      (this can actually be Nw/8 ops, rounded up).
//
//

void
transpose_rectangle(
		uint8_t * buffer,		//vector aligned work buffer; see above
		int elementsize,		// elementsize, must be 1,2,4,8 or 16,
		int Nh, int Nw)					// size of square; must be >=2  <= 128/elementsize
{
	// round n up to a power of 2, >= 8
	int n = max_i32(Nh,Nw);
	n = max_i32(n,8);
	int log2n8 = 29-Q6_R_cl0_R(n-1);	// ceil(log2(n/8)); 0...4)
	//int nrounded =  8 << log2n8;
	int blockwidth = 8*elementsize;		// width of 'block' element.

	HVX_Vector * ptr = (HVX_Vector *)buffer;

	switch(log2n8){
	  default:
	  case 0:			// Do nothing here.
		break;
	  case 1:
		// two rows of 8; blockwidth is 8 .. 64
		{
			HVX_VectorPair v01 = Q6_W_vshuff_VVR( ptr[8], ptr[0], blockwidth);
			for(int  i= 0; i < 7; i ++){
				ptr[i] = Q6_V_lo_W(v01);
				ptr[8+i] = Q6_V_hi_W(v01);
				v01 = Q6_W_vshuff_VVR( ptr[8+i+1], ptr[i+1], blockwidth);
			}
			ptr[7] = Q6_V_lo_W(v01);
			ptr[15] = Q6_V_hi_W(v01);
		}
		break;
	  case 2:
		// 4 rows of 8; blockwidth is 8 .. 32
		{
			HVX_VectorPair v01 = Q6_W_vshuff_VVR( ptr[8], ptr[0], blockwidth);
			HVX_VectorPair v23= Q6_W_vshuff_VVR( ptr[24], ptr[16], blockwidth);
			for(int  i= 0; i < 7; i ++){
				HVX_VectorPair v02 = Q6_W_vshuff_VVR( Q6_V_lo_W(v23), Q6_V_lo_W(v01),blockwidth*2 );
				HVX_VectorPair v13 = Q6_W_vshuff_VVR( Q6_V_hi_W(v23), Q6_V_hi_W(v01),blockwidth*2 );
				ptr[i] = Q6_V_lo_W(v02);
				ptr[8+i] = Q6_V_lo_W(v13);
				ptr[16+i] = Q6_V_hi_W(v02);
				ptr[24+i] = Q6_V_hi_W(v13);
				v01 = Q6_W_vshuff_VVR( ptr[8+i+1], ptr[i+1], blockwidth);
				v23 = Q6_W_vshuff_VVR( ptr[24+i+1], ptr[16+i+1], blockwidth);
			}
			HVX_VectorPair v02 = Q6_W_vshuff_VVR( Q6_V_lo_W(v23), Q6_V_lo_W(v01),blockwidth*2 );
			HVX_VectorPair v13 = Q6_W_vshuff_VVR( Q6_V_hi_W(v23), Q6_V_hi_W(v01),blockwidth*2 );
			ptr[7] = Q6_V_lo_W(v02);
			ptr[15] = Q6_V_lo_W(v13);
			ptr[23] = Q6_V_hi_W(v02);
			ptr[31] = Q6_V_hi_W(v13);
		}
		break;
	  case 3:
		// 8 rows of 8; blockwidth is 8 or 16
		{
			for(int i = 0; i < 8; i++){
				HVX_VectorPair t01 = Q6_W_vshuff_VVR( ptr[8+i],ptr[i],blockwidth);
				HVX_VectorPair t23 = Q6_W_vshuff_VVR( ptr[3*8+i],ptr[2*8+i],blockwidth);
				HVX_VectorPair t45 = Q6_W_vshuff_VVR( ptr[5*8+i],ptr[4*8+i],blockwidth);
				HVX_VectorPair t67 = Q6_W_vshuff_VVR( ptr[7*8+i],ptr[6*8+i],blockwidth);

				HVX_VectorPair t02 = Q6_W_vshuff_VVR( Q6_V_lo_W(t23), Q6_V_lo_W(t01),blockwidth*2 );
				HVX_VectorPair t13 = Q6_W_vshuff_VVR( Q6_V_hi_W(t23), Q6_V_hi_W(t01),blockwidth*2 );
				HVX_VectorPair t46 = Q6_W_vshuff_VVR( Q6_V_lo_W(t67), Q6_V_lo_W(t45),blockwidth*2 );
				HVX_VectorPair t57 = Q6_W_vshuff_VVR( Q6_V_hi_W(t67), Q6_V_hi_W(t45),blockwidth*2 );

				HVX_VectorPair t04 = Q6_W_vshuff_VVR( Q6_V_lo_W(t46), Q6_V_lo_W(t02),blockwidth*4 );
				HVX_VectorPair t26 = Q6_W_vshuff_VVR( Q6_V_hi_W(t46), Q6_V_hi_W(t02),blockwidth*4 );
				HVX_VectorPair t15 = Q6_W_vshuff_VVR( Q6_V_lo_W(t57), Q6_V_lo_W(t13),blockwidth*4 );
				HVX_VectorPair t37 = Q6_W_vshuff_VVR( Q6_V_hi_W(t57), Q6_V_hi_W(t13),blockwidth*4 );
				ptr[    i] = Q6_V_lo_W(t04);
				ptr[  8+i] = Q6_V_lo_W(t15);
				ptr[2*8+i] = Q6_V_lo_W(t26);
				ptr[3*8+i] = Q6_V_lo_W(t37);
				ptr[4*8+i] = Q6_V_hi_W(t04);
				ptr[5*8+i] = Q6_V_hi_W(t15);
				ptr[6*8+i] = Q6_V_hi_W(t26);
				ptr[7*8+i] = Q6_V_hi_W(t37);
			}
		}
		break;
	  case 4:
		// 16 rows of 8; blockwidth is 16 (elementsize must be 1 here)
		{
			for(int i = 0; i < 8; i++){
				// 16x16 transpose needs 4 passes of 8 transposes each.
				// first pass...
				HVX_VectorPair t01 = Q6_W_vshuff_VVR( ptr[8*1+i],ptr[i],blockwidth);
				HVX_VectorPair t23 = Q6_W_vshuff_VVR( ptr[8*3+i],ptr[8*2+i],blockwidth);
				HVX_VectorPair t45 = Q6_W_vshuff_VVR( ptr[8*5+i],ptr[8*4+i],blockwidth);
				HVX_VectorPair t67 = Q6_W_vshuff_VVR( ptr[8*7+i],ptr[8*6+i],blockwidth);
				HVX_VectorPair t89 = Q6_W_vshuff_VVR( ptr[8*9+i],ptr[8*8+i],blockwidth);
				HVX_VectorPair tab = Q6_W_vshuff_VVR( ptr[8*11+i],ptr[8*10+i],blockwidth);
				HVX_VectorPair tcd = Q6_W_vshuff_VVR( ptr[8*13+i],ptr[8*12+i],blockwidth);
				HVX_VectorPair tef = Q6_W_vshuff_VVR( ptr[8*15+i],ptr[8*14+i],blockwidth);

				// 'rotating labels' trick for the next 3 passes
				//    0,1 <- (0,2)
				//    2,3 <- (4,6)
				//... 6,7 <- (12,14)
				//    8,9 <- (1,3)
				//   10,11 <- (5,7)
				//...14,15 <- (13,15)
				int k = blockwidth*2;
				for( int j = 0; j < 3; j++){
					// don't overwrite something we haven't used yet...
					// note: compiler rearranges all this so the loop contains only
					// the 8 'vshuf' and two 'vcombine'
					HVX_Vector t1 = Q6_V_hi_W(t01);
					t01 = Q6_W_vshuff_VVR( Q6_V_lo_W(t23), Q6_V_lo_W(t01),k);
					HVX_Vector t3 = Q6_V_hi_W(t23);
					t23 = Q6_W_vshuff_VVR(  Q6_V_lo_W(t67), Q6_V_lo_W(t45),k);
					HVX_Vector t5 = Q6_V_hi_W(t45);
					t45 = Q6_W_vshuff_VVR( Q6_V_lo_W(tab), Q6_V_lo_W(t89) ,k);
					HVX_Vector t7 = Q6_V_hi_W(t67);
					t67 = Q6_W_vshuff_VVR(  Q6_V_lo_W(tef), Q6_V_lo_W(tcd),k);
					HVX_Vector t9 = Q6_V_hi_W(t89);
					t89 = Q6_W_vshuff_VVR( t3, t1,k);
					HVX_Vector t11 = Q6_V_hi_W(tab);
					tab = Q6_W_vshuff_VVR( t7, t5,k);
					HVX_Vector t13 = Q6_V_hi_W(tcd);
					tcd= Q6_W_vshuff_VVR( t11, t9,k);
					tef= Q6_W_vshuff_VVR( Q6_V_hi_W(tef), t13,k);
					k *= 2;
				}
				for( int j = 0; j < 4; j++){
					ptr[(j*2)*8+i] = Q6_V_lo_W(t01);		// # 0+2*j
					ptr[(j*2+1)*8+i] = Q6_V_lo_W(t23);		// # 1+2*j
					ptr[(j*2+8)*8+i] = Q6_V_hi_W(t01);		// # 8+2*j
					ptr[(j*2+9)*8+i] = Q6_V_hi_W(t23);		// # 9+2*j
					t01 = t45; t45 = t89; t89 = tcd;
					t23 = t67; t67 = tab; tab = tef;
				}
			}
		}
		break;
	}
	// do the 8x8 'intra-block' transposes. Only need to do nxpose8,  where nxpose8*8 >= Nw
	// (i.e. enough to cover Nw output rows)

	int nxpose8 = (Nw+7)/8u;
	for( int i = 0; i <nxpose8; i++){
		HVX_Vector *p4 = (HVX_Vector *)( buffer  + (8*i+4)*128);

		HVX_VectorPair t01 = Q6_W_vshuff_VVR( p4[-3],p4[-4],elementsize);
		HVX_VectorPair t23 = Q6_W_vshuff_VVR( p4[-1],p4[-2],elementsize);
		HVX_VectorPair t45 = Q6_W_vshuff_VVR( p4[ 1],p4[ 0],elementsize);
		HVX_VectorPair t67 = Q6_W_vshuff_VVR( p4[ 3],p4[ 2],elementsize);

		HVX_VectorPair t02 = Q6_W_vshuff_VVR( Q6_V_lo_W(t23), Q6_V_lo_W(t01),elementsize*2 );
		HVX_VectorPair t13 = Q6_W_vshuff_VVR( Q6_V_hi_W(t23), Q6_V_hi_W(t01),elementsize*2 );
		HVX_VectorPair t46 = Q6_W_vshuff_VVR( Q6_V_lo_W(t67), Q6_V_lo_W(t45),elementsize*2 );
		HVX_VectorPair t57 = Q6_W_vshuff_VVR( Q6_V_hi_W(t67), Q6_V_hi_W(t45),elementsize*2 );

		HVX_VectorPair t04 = Q6_W_vshuff_VVR( Q6_V_lo_W(t46), Q6_V_lo_W(t02),elementsize*4 );
		HVX_VectorPair t26 = Q6_W_vshuff_VVR( Q6_V_hi_W(t46), Q6_V_hi_W(t02),elementsize*4 );
		HVX_VectorPair t15 = Q6_W_vshuff_VVR( Q6_V_lo_W(t57), Q6_V_lo_W(t13),elementsize*4 );
		HVX_VectorPair t37 = Q6_W_vshuff_VVR( Q6_V_hi_W(t57), Q6_V_hi_W(t13),elementsize*4 );
		p4[-4] = Q6_V_lo_W(t04);
		p4[-3] = Q6_V_lo_W(t15);
		p4[-2] = Q6_V_lo_W(t26);
		p4[-1] = Q6_V_lo_W(t37);
		p4[ 0] = Q6_V_hi_W(t04);
		p4[ 1] = Q6_V_hi_W(t15);
		p4[ 2] = Q6_V_hi_W(t26);
		p4[ 3] = Q6_V_hi_W(t37);
	}
}
