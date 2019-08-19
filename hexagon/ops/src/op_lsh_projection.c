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
// Copied the original copyright notice after this line:
//
// Copyright (c) 2014 Google, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <nn_graph.h>
#include "hvx_inlines.h"

#define LSH_PROJECTION_TYPE_SPARSE_DEPRECATED 1
#define LSH_PROJECTION_TYPE_DENSE             2
#define LSH_PROJECTION_TYPE_SPARSE            3

//LSHProjection:
//    Locality Sensitive Hashing projection, bucketize input items.  Works on data as char array; known to support
//    bytes, halfs, words.
//    4 inputs:
//        0   hash functions     (float)             2D  i.e. 1,1,x,y
//                                                   x: number of hash functions,
//                                                   y: number of projected output bits
//        1   input tensor       (byte, half, word)  Flat buffer of any type, i.e. op works on data as char array,
//                                                   just need dimensions and buffer size.  First dimension describes
//                                                   the number of items in the input.
//        2   weights            (float)             1D  Either scalar or matching dimension to input 1
//                                                   scalar: must provide 1.0f, i.e. each input have same weight
//                                                   matching dimension to input 1: weight of each input item
//        3   type               (int32)             Scalar  Type of LSH projection to perform:
//                                                   type == 3: LSH_PROJECTION_TYPE_SPARSE
//                                                   type == 1: LSH_PROJECTION_TYPE_SPARSE_DEPRECATED
//                                                   type == 2: LSH_PROJECTION_TYPE_DENSE
//    1 output:
//        0   output tensor      (int32)             1D  Sparse type outputs 1,1,1,x.  Dense type outputs 1,1,1,x*y

//******** start of ported

//** start of farmhash::Fingerprint64 from "thirdparty/libtextclassifier/utils/hash/farmhash"

// assume 4 bytes aligned access, hexagon is small endian
// ported 64 bits version (Fetch64), need to fetch from 2 locations in case of 4 bytes rolling access
static inline uint64_t
Fetch(const char * x) {
	return ((((uint64_t) *((uint32_t *) (x+4))) << 32) + ((uint64_t) *((uint32_t *) (x))));
}
// fetch just reads
static inline uint64_t
Fetch32(const char * x) {
	return (*((uint32_t *)(x)));
}
// ported 64 bits version (BasicRotate64)
static inline uint64_t
Rotate(uint64_t val, int shift) {
	return (shift == 0 ? val : ((val >> shift) | (val << (64 - shift))));
}

static inline uint64_t
ShiftMix(uint64_t val) {
	return (val ^ (val >> 47));
}

static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

static uint64_t
HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
	uint64_t a = (u ^ v) * mul;
	a ^= (a >> 47);
	uint64_t b = (v ^ a) * mul;
	b ^= (b >> 47);
	b *= mul;
	return b;
}

static uint64_t
HashLen0to16(const char * s, const uint32_t len) {
	if (len >= 8) {
		uint64_t mul = k2 + len * 2;
		uint64_t a = Fetch(s) + k2;
		uint64_t b = Fetch(s + len - 8);
		uint64_t c = Rotate(b, 37) * mul + a;
		uint64_t d = (Rotate(a, 25) + b) * mul;
		return HashLen16(c, d, mul);
	}
	if (len >= 4) {
		uint64_t mul = k2 + len * 2;
		uint64_t a = Fetch32(s);
		return HashLen16(len + (a << 3), Fetch32(s + len - 4), mul);
	}
	if (len > 0) {
		uint8_t a = s[0];
		uint8_t b = s[len >> 1];
		uint8_t c = s[len - 1];
		uint32_t y = ((uint32_t) (a)) + ((uint32_t) (b) << 8);
		uint32_t z = len + ((uint32_t) (c) << 2);
		return ShiftMix(y * k2 ^ z * k0) * k2;
	}
	return k2;
}

static uint64_t
HashLen17to32(const char * s, const uint32_t len) {
	uint64_t mul = k2 + len * 2;
	uint64_t a = Fetch(s) * k1;
	uint64_t b = Fetch(s + 8);
	uint64_t c = Fetch(s + len - 8) * mul;
	uint64_t d = Fetch(s + len - 16) * k2;
	return HashLen16(Rotate((a + b), 43) + Rotate(c, 30) + d,
			a + Rotate((b + k2), 18) + c, mul);
}

static uint64_t
HashLen33to64(const char * s, const uint32_t len) {
	uint64_t mul = k2 + len * 2;
	uint64_t a = Fetch(s) * k2;
	uint64_t b = Fetch(s + 8);
	uint64_t c = Fetch(s + len - 8) * mul;
	uint64_t d = Fetch(s + len - 16) * k2;
	uint64_t y = Rotate((a + b), 43) + Rotate(c, 30) + d;
	uint64_t z = HashLen16(y, a + Rotate((b + k2), 18) + c, mul);
	uint64_t e = Fetch(s + 16) * mul;
	uint64_t f = Fetch(s + 24);
	uint64_t g = (y + Fetch(s + len - 32) ) * mul;
	uint64_t h = (z + Fetch(s + len - 24) ) * mul;
	return HashLen16(Rotate((e + f), 43) + Rotate(g, 30) + h,
			e + Rotate((f + a), 18) + g, mul);
}

void swap(uint64_t *a, uint64_t *b) {
	uint64_t tmp = *a;
	*b = *a;
	*a = tmp;
}

struct pair {
	uint64_t first;
	uint64_t second;
};

static struct pair
WeakHashLen32WithSeeds(const char* s, uint64_t a, uint64_t b) {
	uint64_t w = Fetch(s);
	uint64_t x = Fetch(s + 8);
	uint64_t y = Fetch(s + 16);
	uint64_t z = Fetch(s + 24);
	a += w;
	b = Rotate((b + a + z), 21);
	uint64_t c = a;
	a += x;
	a += y;
	b += Rotate(a, 44);
	return (struct pair ) { a + z, b + c } ;
}

static uint64_t
farmhash_fingerprint64(const char * s, const uint32_t len) {
	const uint64_t seed = 81;
	if (len <= 32) {
		if (len <= 16) {
			return HashLen0to16(s, len);
		} else {
			return HashLen17to32(s, len);
		}
	} else if (len <= 64) {
		return HashLen33to64(s, len);
	}

	// For strings over 64 bytes we loop.  Internal state consists of
	// 56 bytes: v, w, x, y, and z.
	uint64_t x = seed;
	uint64_t y = seed * k1 + 113;
	uint64_t z = ShiftMix((y * k2 + 113)) * k2;
	struct pair v = {0, 0};
	struct pair w = {0, 0};
	x = x * k2 + Fetch(s);

	// Set end so that after the loop we have 1 to 64 bytes left to process.
	const char* end = s + ((len - 1) / 64) * 64;
	const char* last64 = end + ((len - 1) & 63) - 63;
	//assert(s + len - 64 == last64);
	do {
		x = Rotate((x + y + v.first + Fetch(s + 8)), 37) * k1;
		y = Rotate((y + v.second + Fetch(s + 48)), 42) * k1;
		x ^= w.second;
		y += v.first + Fetch(s + 40);
		z = Rotate((z + w.first), 33) * k1;
		v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
		w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
		swap(&z, &x);
		s += 64;
	} while (s != end);
	uint64_t mul = k1 + ((z & 0xff) << 1);
	// Make s point to the last 64 bytes of input.
	s = last64;
	w.first += ((len - 1) & 63);
	v.first += w.first;
	w.first += v.first;
	x = Rotate((x + y + v.first + Fetch(s + 8)), 37) * mul;
	y = Rotate((y + v.second + Fetch(s + 48)), 42) * mul;
	x ^= w.second * 9;
	y += v.first * 9 + Fetch(s + 40);
	z = Rotate((z + w.first), 33) * mul;
	v = WeakHashLen32WithSeeds(s, v.second * mul, x + w.first);
	w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
	swap(&z, &x);
	return HashLen16(HashLen16(v.first, w.first, mul) + ShiftMix(y) * k0 + z,
	        HashLen16(v.second, w.second, mul) + x, mul);
}

//** lsh projection from "frameworks/ml/nn/common/operations/LSHProjection"
static int32_t
running_sign_bit(
		struct nn_graph *nn,
		const struct tensor * input,
		const struct tensor * weights,
		float seed
		)
{
	double score = 0.0;
	// items in last dim, all other dims' product as number of items
	uint32_t input_rank = shape_apparent_rank(&(input->shape));
	uint32_t items = (input_rank == 0) ? 1 : input->shape.dimension[4-input_rank];
	uint32_t input_item_bytes = input->data_size / items;

	char * input_ptr = (char *) input->data;

	const uint32_t seed_size = sizeof(float);
	const uint32_t key_bytes = sizeof(float) + input_item_bytes;
	if (nn_scratch_grow(nn, key_bytes)) {
		return errlog(nn, "Failed to get scratch buffer");
	}

	char * key = (char *) nn->scratch;

	uint32_t num_weights = weights->data_size / sizeof(float);
	uint32_t has_weights = (num_weights == items) ? 1 : 0;

	for (uint32_t i = 0; i < items; i++) {
		memcpy(key, &seed, seed_size);
		memcpy(key + seed_size, input_ptr, input_item_bytes);
		int64_t hash_signature = (int64_t) farmhash_fingerprint64(key, key_bytes);
		double running_value = (double)hash_signature;
		input_ptr += input_item_bytes;
		if (0 == has_weights) {
			score += running_value;
		} else {
			score += tensor_get_float(weights, i) * running_value;
		}
	}

	return (score > 0) ? 1 : 0;
}

static void
sparse_lsh_projection(
		struct nn_graph *nn,
		const struct tensor * hash,
		const struct tensor * input,
		const struct tensor * weights,
		struct tensor * output,
		uint32_t deprecated
		)
{
	uint32_t num_hash = hash->shape.width;
	uint32_t num_bits = hash->shape.depth;
	for (uint32_t i = 0; i < num_hash; i++) {
		int32_t hash_signature = 0;
		for (uint32_t j = 0; j < num_bits; j++) {
			float seed = tensor_get_float(hash, (i * num_bits + j));
			int32_t bit = running_sign_bit(nn, input, weights, seed);
			hash_signature = (hash_signature << 1) | bit;
		}
		if (deprecated) {
			tensor_set_int32(output, i, hash_signature);
		} else {
			tensor_set_int32(output, i, (hash_signature + i * (1 << num_bits)));
		}
	}
}

static void
dense_lsh_projection(
		struct nn_graph *nn,
		const struct tensor * hash,
		const struct tensor * input,
		const struct tensor * weights,
		struct tensor * output
		)
{
	uint32_t num_hash = hash->shape.width;
	uint32_t num_bits = hash->shape.depth;

	for (uint32_t i = 0; i < num_hash; i++) {
		for (uint32_t j = 0; j < num_bits; j++) {
			float seed = tensor_get_float(hash, (i * num_bits + j));
			int32_t bit = running_sign_bit(nn, input, weights, seed);
			tensor_set_int32(output, i * num_bits + j, bit);
		}
	}
}

static int
lsh_projection_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor * hash_tensor = self->inputs[0];
	const struct tensor * input_tensor = self->inputs[1];
	const struct tensor * weights_tensor = self->inputs[2];
	const struct tensor * type_tensor = self->inputs[3];

	struct tensor * output_tensor = self->outputs[0];

	uint32_t num_hash = hash_tensor->shape.width;
	uint32_t num_bits = hash_tensor->shape.depth;
	uint32_t input_rank = shape_apparent_rank(&(input_tensor->shape));
	uint32_t items = (input_rank == 0) ? 1 : input_tensor->shape.dimension[4-input_rank];
	uint32_t num_weights = weights_tensor->data_size / sizeof(float);
	int32_t type = tensor_get_int32(type_tensor, 0);

	if (num_bits > 32) return errlog(nn, "lsh projection only support up to 32 bits.");

	logmsg(nn,2,"lsh projection node %p start",self);
	switch (type) {
	case LSH_PROJECTION_TYPE_SPARSE:
	case LSH_PROJECTION_TYPE_SPARSE_DEPRECATED:
		if (num_weights != 1 || tensor_get_float(weights_tensor, 0) != 1.0f) {
			return errlog(nn, "Weights not supported for sparse lsh projection");
		}
		if (tensor_out_prepare_normal(output_tensor,1,1,1,num_hash,NN_TYPE_INT32)) {
			return errlog(nn, "lsh projection output too small");
		}
		sparse_lsh_projection(nn, hash_tensor, input_tensor, weights_tensor, output_tensor,
				(type == LSH_PROJECTION_TYPE_SPARSE_DEPRECATED) ? 1 : 0);
		break;
	case LSH_PROJECTION_TYPE_DENSE:
		if (num_weights != items) {
			return errlog(nn, "Weights mismatch number of input items for dense lsh projection");
		}
		if (tensor_out_prepare_normal(output_tensor,1,1,1,(num_hash * num_bits),NN_TYPE_INT32)) {
			return errlog(nn, "lsh projection output too small");
		}
		dense_lsh_projection(nn, hash_tensor, input_tensor, weights_tensor, output_tensor);
		break;
	default:
		return errlog(nn, "Unsupported lsh projection type %d", type);
	}
	logmsg(nn,2,"lsh projection node %p complete",self);
	return 0;
}
//******** end of ported
struct nn_node_ops nn_ops_for_LSHProjection = {
	.execute = lsh_projection_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(1),
};
