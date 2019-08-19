
/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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
#ifndef NN_GRAPH_TYPES_H
#define NN_GRAPH_TYPES_H 1
//
// to enable run-time checking of dims in tensor_prepare_xx
//#define NN_CHECK_TENSOR_DIMS 1
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This defines common types used.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "nn_graph_builtin.h"

typedef enum padding_type_enum {
	NN_PAD_NA = 0,
	NN_PAD_SAME,
	NN_PAD_VALID,
	NN_PAD_MIRROR_REFLECT,
	NN_PAD_MIRROR_SYMMETRIC,
	NN_PAD_SAME_CAFFE,
} padding_type;

typedef enum corner_type_enum {
   NN_CORNER_RELEASE = 0,
#if (__HEXAGON_ARCH__ >= 66)
   NN_CORNER_TURBOPLUS,
#endif
   NN_CORNER_TURBO,
   NN_CORNER_NOMPLUS,
   NN_CORNER_NOMINAL,
   NN_CORNER_SVSPLUS,
   NN_CORNER_SVS,
   NN_CORNER_SVS2
} hexagon_nn_corner_type;

typedef enum dcvs_type_enum {
   NN_DCVS_DEFAULT = 0,
   NN_DCVS_ENABLE,
   NN_DCVS_DISABLE
} hexagon_nn_dcvs_type;

// We must allocate storage for padded outputs before we
//   know how much padding will actually be applied by the
//   ops of the graph, so these are the largest paddings
//   that any op should presume.
#define MAX_PADDING_BATCHES 0
#define MAX_PADDING_HEIGHT 8
#define MAX_PADDING_WIDTH 7
#define MAX_PADDING_DEPTH 31

enum {
	NN_TYPE_VOID = 0,	// unknown type
	NN_TYPE_QINT8,
	NN_TYPE_QUINT8,
	NN_TYPE_INT8,
	NN_TYPE_UINT8,
	NN_TYPE_QINT16,
	NN_TYPE_QUINT16,
	NN_TYPE_INT32,
	NN_TYPE_FLOAT,
};
extern const char *TypeStrings[];


// Enum and Structs used for passing options into hexagon_nn_config()
//   (Designed so we can send a list of options across fastRPC,
//    without needing to change the interface every time we add an option)
enum {
	// WARNING - DANGER!  Do not alter existing enum-IDs.
	//                    These IDs should be stable across builds
	//                    So you should:
        //                                 Add new items to BOTTOM
	NN_OPTION_NOSUCHOPTION = 0,
	NN_OPTION_SCALAR_THREADS,
	NN_OPTION_HVX_THREADS,
	NN_OPTION_VTCM_REQ,
	NN_OPTION_ENABLE_GRAPH_PRINT,
	NN_OPTION_ENABLE_TENSOR_PRINT,
	NN_OPTION_TENSOR_PRINT_FILTER,
	NN_OPTION_HAP_MEM_GROW_SIZE,
	NN_OPTION_ENABLE_CONST_PRINT,
	//  /\.
	//   \___ Add NEW entries HERE, at BOTTOM
	NN_OPTION_LASTPLUSONE,
};
struct uint_option_t {
	uint32_t option_id;
	uint32_t uint_value;
};

enum {
        NN_EXECUTE_SUCCESS         = 0,
        NN_EXECUTE_ERROR,
        NN_EXECUTE_BUFFER_SIZE_ERROR
};


#define MAX_STRING_L 256
struct string_option_t {
	uint32_t option_id;
	char string_data[MAX_STRING_L];
};

static inline int tensor_type_size(unsigned int type) {
	switch (type) {
		case NN_TYPE_QUINT8:
		case NN_TYPE_QINT8:
		case NN_TYPE_UINT8:
		case NN_TYPE_INT8: return 1;
		case NN_TYPE_QUINT16:
		case NN_TYPE_QINT16: return 2;
		case NN_TYPE_VOID:
		case NN_TYPE_INT32:
		case NN_TYPE_FLOAT:
		default: return 4;
	};
}

enum {
	NN_LAYOUT_PLAIN = 0,	// normal BHWD layout
	NN_LAYOUT_D32,		// Depth-32 layout
	NN_LAYOUT_D32_LOVINGLY_PREPARED,		// Depth-32 layout
};

struct shape {
	union {
		struct {
			uint32_t batches;
			uint32_t height;
			uint32_t width;
			uint32_t depth;
		};
		struct {
			uint32_t filt_height;
			uint32_t filt_width;
			uint32_t filt_depth;
			uint32_t filt_batches;
		};
		// this is to force u64 alignment,
		// to allow faster copy & compare
		struct {
			uint64_t batches_height;
			uint64_t width_depth;
		};
		uint32_t dimension[4];
	};
};

struct tensor_format {
	union {
		unsigned short halves[8];
		struct {
			uint64_t raw0;
			uint64_t raw1;
		};
		struct {
			uint16_t depth_pad[2];
			uint16_t width_pad[2];
			uint16_t height_pad[2];
			uint16_t layout;
			uint16_t type;
		};
	};
};
// efficient comparison of shapes & format
static inline int shape_matches( struct shape const * a, struct shape const *b)
{
	return a->batches_height == b->batches_height
		&& a->width_depth == b->width_depth;
}
static inline int format_matches( struct tensor_format const * a, struct tensor_format const *b)
{
	return ((a->raw0 == b->raw0) && (a->raw1 == b->raw1));
}


struct nn_node;

struct tensor {
	struct shape shape;		// 16
	void *data;
	uint32_t max_size;
	uint32_t data_size;
	struct tensor *self;		// 32
	struct tensor_format format;
	struct nn_node *last_consumer;
	/* data type */
	/* Other flags / attributes */
};
static inline int tensor_is_plain( struct tensor const *t)
{
	return t->format.layout==NN_LAYOUT_PLAIN;
}
// is the tensor either of the D32 formats?
static inline int tensor_is_d32( struct tensor const *t)
{
	return ( t->format.layout==NN_LAYOUT_D32) || ( t->format.layout==NN_LAYOUT_D32_LOVINGLY_PREPARED);
}

//
// This struct contains 'addressing' info
// for locating elements in a d32 tensor. It is generated
// by tensor_addressing_d32  (or tensor_addressing_d32_func).
// Note that d0 = 0 in all cases.
//
// For a d32 tensor of qu8:
// The 'data' pointer is a multiple of 32 (it includes ht& width padding, but not depth padding)
// The 'stride' are all multiple of 128
//  An element [b,h,w,d] is located at data + b*batch_stride + h*height_stride + w*32 + d_offset
// where d_offset is obtained as:
//       dx = d + d0
//       d_offset = dx %32   + (dx/32) * d32_stride
//
// When used for a d32 tensor of qu16:
//    - the pointer is still a uint8_t pointer; it will be a multiple of 64.
//    - the strides are in *bytes*
//      An element [b,h,w,d] is located at data + b*batch_stride + h*height_stride + w*64 + d_offset
//
struct tensor_addressing {
	uint8_t * data;
	int32_t batch_stride;	// all strides are multiple of 128
	int32_t height_stride;
	int32_t d32_stride;
	uint16_t d0;			// depth_before padding
	uint16_t nd32;			// # of d32 slices containing data (d0+depth <= 32*nd32)
	uint16_t nd32_total;	// total d32 slices (usually same as nd32)
};
typedef unsigned long nn_id_t;


struct tensor *tensor_alloc(const struct shape *shape, size_t data_size);
struct tensor *tensor_dup(const struct tensor *src);
void tensor_free(struct tensor *tensor);

//
// overflow-proof unsigned multiply
// result is effectively min( 0xFFFFFFFF, a*b );
// will only be 0 if a or b is 0
// This form is pretty quick on hexagon
//
static inline uint32_t mulu32_sat( uint32_t a, uint32_t b){
	uint64_t p = (uint64_t)a * b;	// full product
	return ((p>>32)==0)? (uint32_t)p : (uint32_t)-1;
}
// multiply four #'s with saturation
static inline uint32_t mulu32_x4_sat( uint32_t a, uint32_t b, uint32_t c, uint32_t d){
	return mulu32_sat( mulu32_sat(a,b), mulu32_sat(c,d));
}

static inline int tensor_copy_inline(struct tensor *dst, const struct tensor *src)
{
	dst->shape = src->shape;
	dst->format = src->format;
	if (src->data_size > dst->max_size) return -1;
	dst->data_size = src->data_size;
	memcpy(dst->data,src->data,src->data_size);
	return 0;
}
int nn_tensor_copy(struct tensor *dst, const struct tensor *src);
static inline int tensor_copy(struct tensor *dst, const struct tensor *src)
{
	return nn_tensor_copy( dst, src);
}

static inline float tensor_get_float(const struct tensor *src, int index)
{
	float const *data = (float const *)src->data;
	// assert dst->data_size >= sizeof(float)?
	return data[index];
}

static inline void tensor_set_float(struct tensor *dst, int index, float val)
{
	float *data = (float *)dst->data;
	// assert dst->data_size >= sizeof(float)?
	data[index] = val;
}

static inline int32_t tensor_get_int32(const struct tensor *src, int index)
{
	const int32_t *data = (int32_t const*)src->data;
	// assert dst->data_size >= sizeof(float)?
	return data[index];
}

static inline void tensor_set_int32(struct tensor *dst, int index, int32_t val)
{
	int32_t *data = (int32_t *)dst->data;
	// assert dst->data_size >= sizeof(float)?
	data[index] = val;
}

static inline void tensor_set_shape(struct tensor *dst,
	uint32_t b, uint32_t h, uint32_t w, uint32_t d)
{
	dst->shape.depth = d;
	dst->shape.width = w;
	dst->shape.height = h;
	dst->shape.batches = b;
}

static inline void tensor_get_shape(const struct tensor *src,
	uint32_t *b, uint32_t *h, uint32_t *w, uint32_t *d)
{
	*d= src->shape.depth;
	*w = src->shape.width;
	*h = src->shape.height;
	*b = src->shape.batches;
}
static inline int32_t shape_element_count(const struct shape *src)
{
#ifdef NN_CHECK_TENSOR_DIMS
	uint32_t allsize = mulu32_x4_sat(src->batches, src->height, src->width, src->depth);
	return (allsize <(1u<<28))? allsize : -1;
#else
	return src->batches * src->height * src->width * src->depth;
#endif
}
static inline int32_t tensor_element_count(const struct tensor *src)
{
	return shape_element_count( &src->shape);
}

static inline int tensor_out_prepare_normal(
	struct tensor *dst,
	uint32_t b,
	uint32_t h,
	uint32_t w,
	uint32_t d,
	uint32_t type)
{
#ifdef NN_CHECK_TENSOR_DIMS
	// sanity check the dimensions
	// - dims all >= 1
	// - product of dims fits in u32 and is not absurd
	uint32_t allsize = mulu32_x4_sat(b,h,w,d);
	if( allsize ==0  || allsize >= (1u<<28)){ // allsize is enormous or 0
		return -4;
	}
	int32_t size = allsize * tensor_type_size(type);
#else
	int32_t size = b*h*w*d*tensor_type_size(type);
#endif
	tensor_set_shape(dst,b,h,w,d);
	dst->data_size = size;
	dst->format.raw0 = dst->format.raw1 = 0;
	dst->format.type = type;
	dst->format.layout = NN_LAYOUT_PLAIN;
	if (dst->max_size < size) return -1;
	return 0;
}
static inline int tensor_out_prepare_normal_fromshape_inline(
	struct tensor *dst,
	struct shape const * shp,
	uint32_t type)
{
	int32_t size = shape_element_count(shp)*tensor_type_size(type);
	dst->shape = *shp;
	dst->data_size = size;
	dst->format.raw0 = dst->format.raw1 = 0;
	dst->format.type = type;
	dst->format.layout = NN_LAYOUT_PLAIN;
	if (dst->max_size < size) return -1;
	return 0;
}
int nn_tensor_out_prepare_normal_fromshape(	struct tensor *dst,	struct shape const * shp,uint32_t type);
static inline int tensor_out_prepare_normal_fromshape(
		struct tensor *dst,
		struct shape const * shp,
		uint32_t type)
{
	return nn_tensor_out_prepare_normal_fromshape( dst, shp, type);
}
//
// note, as of 22-feb-2018 this function is not in use
//

static inline int tensor_out_prepare_d32(
	struct tensor *dst,
	uint32_t b,
	uint32_t h,
	uint32_t w,
	uint32_t d,
	uint32_t type)
{
#ifdef NN_CHECK_TENSOR_DIMS
	// sanity check the dimensions
	// -  dims all >= 1
	// - product of dims fits in u32 and is not absurd
	// w mult of 4, d mult of 32
	uint32_t allsize = mulu32_x4_sat(b,h,w,d);
	if( allsize ==0  || allsize >= (1u<<28)){ // allsize is enormous or 0
		return -4;
	}
	if(  (w&3)!= 0) return -6;
	if(  (d&31)!= 0) return -7;
	int32_t size = 128 + allsize * tensor_type_size(type);
#else
	int32_t size = 128+b*h*w*d*tensor_type_size(type);
#endif
	tensor_set_shape(dst,b,h,w,d);
	dst->data_size = size;
	dst->format.raw0 = dst->format.raw1 = 0;
	dst->format.type = type;
	dst->format.layout = NN_LAYOUT_D32;
	if (dst->max_size < size) return -1;
	return 0;
}

static inline int tensor_out_prepare_padded_d32(
	struct tensor *dst,
	uint32_t b,
	uint32_t h,
	uint32_t h_before,
	uint32_t h_after,
	uint32_t w,
	uint32_t w_before,
	uint32_t w_after,
	uint32_t d,
	uint32_t d_before,
	uint32_t d_after,
	uint32_t type)
{
	int32_t h_total = h+h_before+h_after;
	int32_t w_total = w+w_before+w_after;
	int32_t d_total = d+d_before+d_after;
#ifdef NN_CHECK_TENSOR_DIMS
	int elsize = tensor_type_size(type);
	// sanity check the dimensions and padding
	// - actual dims all >= 1
	// - product of padded dim fits in u32 and is not absurd
	// - padding amounts need to fit in bytes
	// - depth padding is at most 31;
	// - w_total must be multiple of 4 (or 2, when elsize =2)
	// - d_total must be multiple of 32.
	uint32_t allsize = mulu32_x4_sat(b,h_total,w_total,d_total);
	if( allsize ==0  || allsize >= (1u<<28)){ // allsize is enormous or 0
		return -4;
	}
	if( h < 1 || (unsigned)(h_before|h_after) >255u ) return -5;
	if( w < 1 || (unsigned)(w_before|w_after) >255u || ((w_total*elsize)&3)!= 0) return -6;
	if( d < 1 || (unsigned)(d_before|d_after) >31u  || (d_total&31)!= 0) return -7;
	int32_t size = allsize * elsize;
#else
	int32_t size = b*h_total*w_total*d_total*tensor_type_size(type);
#endif
	if (dst->format.layout == NN_LAYOUT_D32_LOVINGLY_PREPARED) {
		/* This tensor claims to have been prepared ahead of time */
		/* We might want to skip actually filling it out */
		if ((dst->shape.batches != b)
		||  (dst->shape.height != h)
		||  (dst->shape.width != w)
		||  (dst->shape.depth != d)) {
			/* Hmm, you promised me love but you don't respect my values. */
			/* I don't think this is going to work out. */
			return -8;
		} else {
			return 0;
		}
	}
	tensor_set_shape(dst,b,h,w,d);
	dst->data_size = size;
	dst->format.depth_pad[0] = d_before;
	dst->format.depth_pad[1] = d_after;
	dst->format.width_pad[0] = w_before;
	dst->format.width_pad[1] = w_after;
	dst->format.height_pad[0] = h_before;
	dst->format.height_pad[1] = h_after;
	dst->format.type = type;
	dst->format.layout = NN_LAYOUT_D32;
	if (dst->max_size < size) return -9;
	return 0;
}

static inline int tensor_out_prepare_d32_sameas(
	struct tensor *dst,
	const struct tensor *src)
{
	dst->shape = src->shape;
	dst->data_size = src->data_size;
	dst->format.raw0 = src->format.raw0;
	dst->format.raw1 = src->format.raw1;
	if (dst->max_size < dst->data_size) return -1;
	return 0;
}

// find the offset into a d32 tensor, in elements, accounting
// for all of the padding.

static inline size_t tensor_offset_for_d32(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w,
	int32_t d)
{
	int32_t h_before = src->format.height_pad[0];
	int32_t w_before = src->format.width_pad[0];
	int32_t d_before = src->format.depth_pad[0];
	int32_t h_total = h_before + src->shape.height + src->format.height_pad[1];
	int32_t w_total = w_before + src->shape.width + src->format.width_pad[1];
	int32_t d_total = d_before + src->shape.depth + src->format.depth_pad[1];
	uint32_t d_pos = d + d_before;

	return b*h_total*w_total*d_total
		+ (h+h_before)*w_total*d_total
		+ (w+w_before)*32
		+ (d_pos/32)*(w_total*32)
		+ d_pos%32;
}

static inline uint8_t *tensor_location_d32(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w,
	int32_t d)
{
	return  (uint8_t*)src->data + tensor_offset_for_d32(src, b,h,w,d);
}
// 16bit version
static inline int16_t *tensor_location_16b_d32(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w,
	int32_t d)
{
	return  (int16_t*)src->data + tensor_offset_for_d32(src, b,h,w,d);
}


static inline uint8_t *tensor_location_d32_aix(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w,
	int32_t d,
	const struct shape *in_shape
	)
{
	uint8_t *base = (uint8_t *)src->data;
	int32_t height = in_shape->height;
	int32_t width = in_shape->width;
	int32_t depth = in_shape->depth;
	int32_t h_before = 0;
	int32_t h_after = 0;
	int32_t w_before = 0;
	int32_t d_before = 0;
	int32_t d_after = (-(depth + d_before)) & 31;
	int32_t w_total = width;
	int32_t w_after = w_total - (w_before + width);
	int32_t h_total = h_before + height + h_after;
	w_total = w_before + width + w_after;
	int32_t d_total = d_before + depth + d_after;
	uint32_t d_pos = d + d_before;

	return base + b * h_total * w_total * d_total + (h + h_before) * w_total * d_total + (w + w_before) * 32 + (d_pos / 32) * (w_total * 32) + d_pos % 32;
}

static inline size_t tensor_offset_bhw_for_d32(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w)
{
	int32_t h_before = src->format.height_pad[0];
	int32_t w_before = src->format.width_pad[0];
	int32_t d_before = src->format.depth_pad[0];
	int32_t h_total = h_before + src->shape.height + src->format.height_pad[1];
	int32_t w_total = w_before + src->shape.width + src->format.width_pad[1];
	int32_t d_total = d_before + src->shape.depth + src->format.depth_pad[1];

	return b*h_total*w_total*d_total
		+ (h+h_before)*w_total*d_total
		+ (w+w_before)*32;
}

//
// this is like tensor_location_d32 except
// - there is no 'd' (assumed 0)
// - the address is not compensated for d_before padding.
//  So it's always 32 aligned.
// (64 aligned, in the 16b case).
//
static inline uint8_t *tensor_location_bhw_d32(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w)
{
	return (uint8_t*)src->data + tensor_offset_bhw_for_d32(src,b,h,w);
}

static inline int16_t *tensor_location_bhw_16b_d32(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w)
{
	return (int16_t*)src->data + tensor_offset_bhw_for_d32(src,b,h,w);
}

//
// generate tensor addressing struct,
// The 'data' pointer is the same as the value returned by tensor_location_bhw_d32(src,0,0,0).
// (i.e. it accounts for height * width padding, but not depth).
//
// NOTE:
//   d0  (depth_pad_before) is 0 in all real cases
//   nd32 is the smallest # of depth_slices needed to contain the data (depth_pad_before+depth+31)/32
//   nd32_total is the actual number: i.e. (depth_pad_before + depth + depth_pad_after)/32, division should
//   be exact. nd32_total is usually the same as nd32; some nodes may not be able to handle inputs
//   where nd32_total > nd32.
//
static inline struct tensor_addressing
tensor_addressing_d32__for_elsize( struct tensor const * src, unsigned elsize)
{
	int32_t h_before = src->format.height_pad[0];
	int32_t w_before = src->format.width_pad[0];
	int32_t d_before = src->format.depth_pad[0];
	int32_t h_total = h_before + src->shape.height + src->format.height_pad[1];
	int32_t w_total = w_before + src->shape.width + src->format.width_pad[1];
	unsigned d0_d = d_before + src->shape.depth;
	int32_t nd32_total = (unsigned)(d0_d + src->format.depth_pad[1])/32u;
	int w_stride = 32*elsize;
	int32_t d32_stride = w_total * w_stride;
	int32_t height_stride = d32_stride*nd32_total;

	struct tensor_addressing result;

	result.data = (uint8_t *)src->data + h_before * height_stride + w_before * w_stride;
	result.batch_stride = h_total * height_stride;
	result.height_stride = height_stride;
	result.d32_stride = d32_stride;
	result.d0 = d_before;
	result.nd32 = (d0_d+31)/32u;
	result.nd32_total = nd32_total;
	return result;
}
static inline struct tensor_addressing
tensor_addressing_d32_inline( struct tensor const * src)
{
	return tensor_addressing_d32__for_elsize(src, 1);
}
static inline struct tensor_addressing
tensor_addressing_d32_16b_inline( struct tensor const * src)
{
	return tensor_addressing_d32__for_elsize(src, 2);
}
// same thing but in a function
struct tensor_addressing nn_tensor_addressing_d32(struct tensor const * src) __attribute__((pure));
struct tensor_addressing nn_tensor_addressing_d32_16b(struct tensor const * src) __attribute__((pure));

static inline struct tensor_addressing
tensor_addressing_d32_func( struct tensor const * src){	return nn_tensor_addressing_d32(src); }
static inline struct tensor_addressing
tensor_addressing_d32_16b_func( struct tensor const * src){	return nn_tensor_addressing_d32_16b(src); }


// default is function
static inline struct tensor_addressing
tensor_addressing_d32( struct tensor const * src){	return nn_tensor_addressing_d32(src); }
static inline struct tensor_addressing
tensor_addressing_d32_16b( struct tensor const * src){	return nn_tensor_addressing_d32_16b(src); }




static inline void *tensor_location(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w,
	int32_t d)
{
        if ( tensor_is_plain(src)) {
                return (((uint8_t *) src->data) +
                        (tensor_type_size(src->format.type) *
                         (d +
                          (src->shape.depth *
                           (w +
                            (src->shape.width *
                             (h +
                              (src->shape.height *
                               (b)))))))));
        } else if ( tensor_is_d32(src)) {
		return ((uint8_t *) src->data) +
			( tensor_type_size(src->format.type) *
			  tensor_offset_for_d32(src, b,h,w,d));
        }
        return NULL;
}

static inline int32_t tensor_h_total_d32(const struct tensor *src)
{
	int32_t height = src->shape.height;
	int32_t h_before = src->format.height_pad[0];
	int32_t h_after = src->format.height_pad[1];
	return height+h_before+h_after;
}
static inline int32_t tensor_w_total_d32(const struct tensor *src)
{
	int32_t width = src->shape.width;
	int32_t w_before = src->format.width_pad[0];
	int32_t w_after = src->format.width_pad[1];
	return width+w_before+w_after;
}
static inline int32_t tensor_d_total_d32(const struct tensor *src)
{
	int32_t depth = src->shape.depth;
	int32_t d_before = src->format.depth_pad[0];
	int32_t d_after = src->format.depth_pad[1];
	return depth+d_before+d_after;
}
//
// stride between batch b and batch b+1
//
static inline int32_t tensor_batch_stride_d32(const struct tensor *src)
{
	return tensor_h_total_d32(src) * tensor_w_total_d32(src) * tensor_d_total_d32(src);
}

//
// stride from row r to row r+1
//

static inline int32_t tensor_row_stride_d32(const struct tensor *src)
{
	return tensor_w_total_d32(src) * tensor_d_total_d32(src);
}
// same as row stride
static inline int32_t tensor_height_stride_d32(const struct tensor *src)
{
	return tensor_w_total_d32(src) * tensor_d_total_d32(src);
}

//
// stride in depth slices: d  to d+32
//

static inline int32_t tensor_d32_stride_d32(const struct tensor *src)
{
	return tensor_w_total_d32(src)*32;
}

//
// set a tensor to a single float value (e.g. for min/max)
//
static inline int tensor_set_single_float( struct tensor * tens, float val)
{
	int k = tensor_out_prepare_normal( tens,1,1,1,1,NN_TYPE_FLOAT);
	if( k == 0)
		tensor_set_float(tens,0,val);
	return k;
}

///////////////////////////////////////////////
// descriptor for a d32 tensor 'slice' view.
// See support code in shape_util.c.
//
struct tensor_slice_d32 {
	struct shape shape;				// batches, height, width, depth
	uint16_t depth_total;			// depth total including padding (mult. of 32)
	uint16_t depth_pad_before;		// [0..31] add this to data to reach (0,0,0,0)
	uint8_t * data;					// this is 32 aligned
	int32_t batch_stride;			// stride between batches (multiple of vector)
	int32_t height_stride;				// stride between rows (multiple of vector)
	int32_t d32_stride;				// stride between chunks (multiple of vector)
};
/*
 * make a slice from a tensor, it encompasses the whole tensor
 */
int tensor_slice_from_tensor_d32( struct tensor_slice_d32  *slc, const struct tensor *tens);
//
// reduce a slice along a given dimension
// 0,1,2,3 = batches, height, width, depth
//
// NOTE: this function can make a new slice by slicing an existing
// slice, OR it can work in-place on a single slice (this is faster).
// 'in-place' slicing is selected by slc_out == slc_in, or by slc_in == NULL.

// returns:
//   0   ok
//  -1   bad 'dim_no'
//  -2  lo_index/newsize out of range.
//
int tensor_slice_on_dimension(
		struct tensor_slice_d32 * slc_out,			// output slice
		struct tensor_slice_d32 const * slc_in,		// input slice (can be same slice).
		int32_t dim_no,		// dimension to slice
		int32_t lo_index,	// start of slice (0..size-1)
		int32_t newsize );	// new size. 1 <= newsize <= size-lo_index
//
// This is used to progressively make slices of src_tensor, based on a series of
// supplied ref_shape, along a given dimension.
//  - src_tensor and ref_shape must agree in all dims but 'slice_dim'.
//  - the resulting slice will have the same shape as ref_shape, but will be a slice of src_tensor.
//  - along the sliced dimension, the slice will start at the supplied value of *slicepos_p
//  - on return, slicepos_p will be updated to the *next* position (i.e. the size of ref_shape along
//    the slice_dim will be added to *slicepos_p).
//
// *** IMPORTANT ***: if called with *slicepos_p > 0, it assumes that it has previously been
//  called with *slicepos_p = 0, and the same slc, src_tensor, and slice_dim (i.e. much of the slice
// is  filled in by that first call, and subsequent calls rely on that).
// (it's also ok to use different 'slc' if it's been copied from the result of a previous call).
// If you want to start slicing at a point other than 0, call first with *slice_pos= 0, and then change *slice_pos to something
// else and call again. This first call can be done with ref_shape = NULL, which will not affect *slice_pos.
//
// return values:
//    1 normal (and no more slices possible along the dim).
//    0 normal
//   -1 slice_dim out of range 0..3
//   -2 shapes don't match along non-slice dims.
//   -3 exceeded range of slice along slice_dim
//   -4 src_tensor is not d32
//
//
int tensor_slice_progressive_d32(
		struct tensor_slice_d32 * slc,			// output slice
		struct tensor const * src_tensor,		// tensor
		struct shape const * ref_shape,			// shape to match
		int slice_dim,							// dimension to slice along
		int *slicepos_p	);						// keeps track of slice.

///////////////////////////////////////////////

//
// This function looks at the shapes in an array of 1 or more 'struct_tensor'
// and, assuming they will be concatenated on dimension 'concat_dim', finds
// the overall shape.
// It also range-checks 'concat_dim' (must be 0..3) and
// ensures that all shapes match in all dims *other* than concat_dim.
//
// The output shape is the sum of the input shapes (on concat_dim) and
// matches them on all others.
//
// returns:
//   0:  ok
//   -1: concat_dim out of range
//   -2..-5: mismatch on dimension 0,1,2,3
//
// - No change is made to *allshape unless the function returns 0
// - caller to ensure that n_input >= 1.
//
int find_concat_shape(
		const struct tensor **input_tensors,
		int n_input,			// >= 1
		int concat_dim,			// 0..3
		struct shape *allshape );


// flags for check_compatible_elementwise_d32
//
enum tensor_compat_flags {
	compat_broadcast_B=1,	// indicates broadcast along dims..
	compat_broadcast_H=2,	// (the B,H,W,D bits must be in sequence)
	compat_broadcast_W=4,
	compat_broadcast_D=8,
	compat_misalign_W = 16,	// width_before padding is different by other than multiple of 4
	compat_misalign_D =32,	// depth padding is not the same.
	compat_skewed_D = 64,	// depth misalign, crossing d32 borders.

	compat_AtoB = 128,		// A broadcasts to B, rather than vice versa

	compat_broadcast_ALL = (compat_broadcast_B| compat_broadcast_H| compat_broadcast_W| compat_broadcast_D),
	compat_misalign_ALL = (compat_misalign_W | compat_misalign_D | compat_skewed_D),
	compat_ALL =  (compat_broadcast_ALL|compat_misalign_ALL)
};

//
// check to see if tensorA and tensorB
// (both d32) are compatible for an elementwise operation.
// This includes broadcasting B dims to A.
// It also checks for misalignments in W & D dimensions.
//
// Normally it will a value >= 0, an 'or' of the tensor_compat_flags.
// if a problem is found, it will report an error and then return -1;
// 'errnm' is a name for the error messages.
//
// If there are compatibility flags and the corresponding bits are
// *not* in allowed_compat, this is considered an error.
//  e.g. A.height = 20, B.height=1 is considered a mismatch if compat_broadcast_H
// is not in allowed_compat;  A.height=20, B.height=10 is always a mismatch.
//
// if compat_AtoB is in allowed_compat, situations where A broadcasts to B are
// accepted, and allowed_compat will be set in the return value when this is encountered
// (it will only be set if the A tensor is smaller than B).
// 'mixed' cases (e.g. (1,1,3,32), (1,5,3,1) are not accepted in any case.
//
// NOTE:
// - if dimensions are both 1, this tagged as 'broadcast' but only if the
//    broadcast is enabled in 'allowed_compat' ; otherwise it will not be tagged.
//
// - if you allow 'broadcast_W', then misalign_W is considered acceptable when B.width =1,
//    even if it's not in allowed_compat. It will be reported in the return result. This
//    may occur when both widths are 1
//
// - if you allow 'broadcast_D', then misalign_D is considered acceptable when B.depth=1,
//    even if it's not in allowed_compat.This will be reported in the return result. This
//    may occur when both depths are 1.
//
//  - skewed_D is a subset of misalign_D.
//    misaligned_D     skewed_D
//           0             0              both have the same depth_before padding
//           1             0              depth-before paddings different, but both fit in one d32
//           1             1              other cases.
//     skewed_D will never be flagged in the return value when tensB.depth =1; the 'A' side
//     might not fit in a d32, but that would be a broadcast_D case.

//
struct nn_graph;
int check_compatible_elementwise_d32(
		struct nn_graph *nn,
		char const * errnm,
		struct tensor const * tensA,
		struct tensor const * tensB,
		int allowed_compat);
//
// This function is given an input d32 tensor, and a reference d32 tensor
// (only used for its width,depth dimensions, and padding info), and optionally
// a work area; and it constructs a memory array formatted as if it were
// a (1,1,w,d) tensor (with w,d taken from tens_ref); this data also has
// the same depth and width padding as tens_ref.
//
// - the function provides a d32_stride for the constructed data. If broadcasting
//   on depth is being done (tens_in->depth = 1), this pitch is always 0, regardless
//   of the output depth (i.e. only one depth slice is constructed, all values are the
//   same across depth anyway.
//
// CALLER MUST ENSURE:
//   - tens_in and test_ref are both valid d32 tensors. b & h dimensions and padding are ignored.
//   - in the w and d dimensions, tens_in must either be 1 or must match tens_ref.
//   - if workbuf is not NULL, it must point to a vec-aligned work area of at least workbuf_len bytes
//   .. otherwise function may return NULL, or undefined behaviour may occur.
//
//
// The function returns the (vector-aligned) pointer to the data.
// it will return NULL if allocation fails or if a parameter problem is detected.
//
// Memory allocation:
//	 - In a fews cases the function may avoid a copy, and returns a pointer to the data referenced
// 	   by tens_in. So d32_stride_out will be from tens_in.
//   - Caller can supply a workbuf (and workbuf_len); this will be used if it is large enough,
//     otherwise memory will be allocated. (use workbuf = NULL, or workbuf_len = 0  if not supplying a buf)
// 'allocbuf_out' must be a pointer to a void * variable.
//    - this will be set to NULL if the routine allocated no memory.
//    - if not NULL, the value must be passed to nn_free to free the allocated memory.
//
// This function doesn't use any HVX ops (but it uses some hexagon 64-bit ops)
//
//
uint8_t const *
construct_broadcasted_data_d32(
		struct tensor const * tens_in,		// tensor containing data
		struct tensor const * tens_ref,		// tensor used as shape/alignment ref
		int32_t *d32_stride_out,				// used to return the d32 pitch
		void * workbuf,						// optional work area
		uint32_t workbuf_len,				// len in bytes of work area
		void **allocbuf_out );



//
// This utility examines a d32 tensor, which is assumed to have a shape (1,1,w,d),
// and finds the actual range of u8 values stored within.
// The range is returned as  (maxval<<8) | minval.
// No hvx instructions (it uses 64-bit hexagon vector ops).
//
uint32_t find_range_in_wd_tensor_d32( struct tensor const * tens_in );


//
// This copies one d32 tensor to another, with
// optional linear scaling out = in*a +b
//
//  Caller must:
//    - pre-allocate the output tensor
//    - must be the same size as input tensor
//    - must have the same depth_before padding
//   -  should have same width_before padding (modulo 4)  (else, use_hvx will be ignored).
//
//  NOTE: this currently does 32-bit aligned stores (i.e. it writes all of the depth extent,
//  including the padding).
//
//
//
//  The scaling is done as  out[i] = (in[i] *scale +   offset*256   +  16384) >> 15
//
//  In more detail:
//      (1)  p = (in[i] * scale ) >> 8;   [ fits in i16]
//      (2)  t  = add(p, offset)        [ saturate to i16]
//      (3)  out[i] = saturate_u8(   (t+64) >> 7 )
//
//  => offset = 0, scale in range 32704..32767 will be handled as a copy
//  => offset = 32640, scale = -32768 .. -32705  will result in a 1's complement operation.
//
// This is used for add_d32 and sub_d32 operator, when one of the inputs is a scalar, the
// operation can be done by a copy (with possible range extension).
// Also, for mul_d32 when one of the inputs is  a scalar, the op can be done by a copy (or
// inversion, when the scalar is negative).
//
// Normally returns 0; can return -1 if it logs an error.
//
int
tensor_copy_scaled_d32(
		struct nn_graph *nn,
		struct tensor const *tensor_src,
		struct tensor const *tensor_dst,
		int16_t scale,
		int16_t offset,
		int use_hvx,
		int max_threads );


//
// for running a generic unary float op.
// This could be enhanced to use threads,
// currently does not (and need_hvx is ignored)
//
int nn_generic_unary_float_op( struct nn_node *self, struct nn_graph *nn,
		void (*func)( float *, float const *, int n, void *info),
		void * info, int need_hvx);

uint32_t data_cksum(void *data, uint32_t bytes);
void print_tensor_to_file(struct nn_graph *nn, uint32_t id, uint32_t index, const struct tensor *t);
void print_tensor(const struct tensor *t, const char *str);
void print_tensors(const struct tensor *tensors, uint32_t n_tensors);

// 64-bit checksum of data in a memory area. If all bytes are the same value k,
// result will be k.
uint64_t nn_checksum_bytes( void const * p, int n);
// checksum a 'flat' tensor (or all of the memory in a d32 tensor, including padding)
uint64_t nn_checksum_tensor ( struct tensor const *tens);
// checksum all the data in a d32 tensor, in the proper order
uint64_t nn_checksum_tensor_d32 ( struct tensor const *tens);

// emit log msgs (at 'level') giving the shapes and  checksums of all outputs of a node. 4-byte
// tensors are displayed directly by value.
void nn_report_node_outputs( struct nn_graph * nn, int level, struct nn_node const *np);


//
// these are in argminmax.c
//
void hvx_argmin_or_max_in_rows( uint8_t const * data, int rows, int cols, int row_stride, int32_t * outp, int find_argmax );
void hvx_argmin_or_max_in_cols( uint8_t const * data, int rows, int cols, int row_stride, int32_t * outp, int find_argmax );
void hvx_argmin_or_max_d_8_d32( struct tensor const * data_tensor, int32_t * outp, int find_argmax );
void hvx_argmin_or_max_whb_8_d32( struct tensor const * data_tensor, int32_t * outp, int32_t  axis, int find_argmax);

static inline int shape_apparent_rank ( struct shape const *shp){
	return (shp->batches > 1)? 4 :
		 (shp->height >1)? 3:
		(shp->width > 1)? 2:
		(shp->depth>1)? 1: 0;
}
#endif
