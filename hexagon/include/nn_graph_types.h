
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
#ifndef NN_GRAPH_TYPES_H
#define NN_GRAPH_TYPES_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This defines common types used.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>


typedef enum padding_type_enum {
	NN_PAD_NA = 0,
	NN_PAD_SAME,
	NN_PAD_VALID,
	NN_PAD_MIRROR_REFLECT,
	NN_PAD_MIRROR_SYMMETRIC,
	NN_PAD_SAME_CAFFE,
} padding_type;

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
	};
};

struct tensor {
	struct shape shape;
	void *data;
	uint32_t max_size;
	uint32_t data_size;
	struct tensor *self;
};

typedef unsigned long nn_id_t;


struct tensor *tensor_alloc(const struct shape *shape, size_t data_size);
struct tensor *tensor_dup(const struct tensor *src);
void tensor_free(struct tensor *tensor);

static inline int tensor_copy(struct tensor *dst, const struct tensor *src)
{
	dst->shape = src->shape;
	if (src->data_size > dst->max_size) return -1;
	dst->data_size = src->data_size;
	memcpy(dst->data,src->data,src->data_size);
	return 0;
}

static inline float tensor_get_float(const struct tensor *src, int index)
{
	float *data = (float *)src->data;
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
	const int32_t *data = (int32_t *)src->data;
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

#endif
