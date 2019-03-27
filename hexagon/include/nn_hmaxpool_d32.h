
/*
 * Copyright (c) 2017,2018 The Linux Foundation. All rights reserved.
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

#ifndef NN_HMAXPOOL_D32_H_
#define NN_HMAXPOOL_D32_H_



// this implements a 'horizontal maxpool' on d32 data
// The 'top level API is':
//
//       struct hmaxpool_func_info hfinfo;
//       h_maxpool_func = hmaxpool_select_function( & hfinfo, win, stride, outlen, inoffset);
//
//     ... then you can call
//             h_maxpool_func(  &hfinfo, outrow0, outrow1, inrow0, inrow1)
//
// and it will process two input rows (inrow0, inrow1) to two output rows (outrow0, outrow1).
// To process a final 'odd' row, make both output pointers the same (and both input pointers).
// The row processing function doesn't modify 'hfinfo', so hfinfo can be shared among threads.
//
//
// The parameters to hmaxpool_select_function are:
//     win	  - the width of the maxpool window (>=1)
//     stride - the stride of the operation  (>=1)
//     outlen - the number of outputs needed (>=1)
//     inoffset - the offset of the first 'proper' width element in the first input vector.
//               (ranges 0..3). This includes any added 'padding' which is part of the operation.
//               For instance, if the width padding is 4, but you have win = 3, and need to pad one
//               on the left, you'd have a pointer to the 4 extra width units, and inoffset=3.
//
// The selection function will always write a full # of vectors, and in some cases will read more vectors
// than necessary to provide the specified # of outputs (this is especially true with larger strides).
//
// The commonest (and best supported) cases are:
//       win = 2  stride = 1 or 2
//       win = 3  stride = 1,2, or 3
//
// The value 'outvecs' in the hmaxpool_func_info is the size of the output row,
// in vectors (or, at least, an upper bound). This is at least (outlen+3)/4 but may
// be larger.
// All of the members of this struct *other* than outvecs, are internal information from 
// hmaxpool_select_function to the selected function, and may be unused in some situations
// (or may have different meanings).

struct hmaxpool_func_info {
	int nloops;
	uint8_t oshift;	// # of initial bytes to discard from output	 (where needed)
	uint8_t inoffs;	// # input offset (where needed; 0..3)
	int16_t win, stride;	// window and stride (where needed).
	uint16_t outvecs;
};



typedef void (*hmaxpool_funcp)(struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );

void hmaxpool_w0s0o0(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1);

void hmaxpool_wXsXoX( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );

void hmaxpool_w1234sXoX( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_wge5sXoX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1);

void hmaxpool_w1s1oX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1);

void hmaxpool_w2s1oX( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w2s2o0( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );

void hmaxpool_w3s1o0( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w3s1oX( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w3s2o0( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w3s2oX( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w3s3o0( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w3s3o12( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );
void hmaxpool_w3s3o3( struct hmaxpool_func_info const * ,
		uint8_t *, uint8_t *,  uint8_t const *, uint8_t const * );


hmaxpool_funcp nn_hmaxpool_select_function( struct hmaxpool_func_info * info,
		int win, int stride, int out_len, int inoffs );

static inline hmaxpool_funcp hmaxpool_select_function( struct hmaxpool_func_info * info,
		int win, int stride, int out_len, int inoffs ){
	return nn_hmaxpool_select_function( info, win, stride, out_len, inoffs);
}

static inline hmaxpool_funcp hmaxpool_select_function_inline( struct hmaxpool_func_info * info,
		int win, int stride, int out_len, int inoffs )
{
	if( win < 1 || stride < 1) return hmaxpool_w0s0o0;

	info->win = win;
	info->stride = stride;
	info->inoffs = inoffs;
	info->oshift = inoffs*32;
	info->outvecs = (out_len+3)>>2;
	info->nloops = (out_len+3)>>2;

	if( stride >= 1 &&  stride <= win && win <= 3){
		if( win == 1){
			if( stride == 1)
				return hmaxpool_w1s1oX;
		} else if( win == 2 ){
			if(stride==1){
				return hmaxpool_w2s1oX;
			}else {	// => stride == 2
				if( inoffs == 0) return hmaxpool_w2s2o0;
			}
		}else if( win == 3){
			if(stride==1){
				if( inoffs == 0)
					return hmaxpool_w3s1o0;
				else
					return hmaxpool_w3s1oX;
			}else if(stride ==2) {
				if( inoffs == 0)
					return hmaxpool_w3s2o0;
				else
					return hmaxpool_w3s2oX;
			}else {	// => stride == 3
				if( inoffs == 0)
					return hmaxpool_w3s3o0;
				else if(inoffs==3)
					return hmaxpool_w3s3o3;
				else
					return hmaxpool_w3s3o12;
			}
		}
	}
	info->nloops = out_len;

	if( win <= 4 ){
		// Handles any case, win <= 4
		return hmaxpool_w1234sXoX;
	}else{
		// Handles any case, win >= 5
		return hmaxpool_wge5sXoX;
	}

	/*
	// default: use fallback
	return hmaxpool_wXsXoX;
	*/
}



#endif /* NN_HMAXPOOL_D32_H_ */
