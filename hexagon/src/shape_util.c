/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
 * utilities for tensor shapes
 */
#include <stdint.h>
#include <nn_graph.h>
#include "nn_graph_types.h"
#include "quantize.h"
#include "nn_string_map.h"

int nn_tensor_copy(struct tensor *dst, const struct tensor *src)
{
	return tensor_copy_inline(dst, src);
}

//
// thread-parallel memcpy manager
//

void nn_mcmanager_init(struct nn_graph *nn, struct nn_memcpy_manager *mcm)
{
	nn_sem_init(&mcm->done_sem, 0);
	mcm->pending = 0;
	mcm->avglen = 32768;
	mcm->avail_set = (1 << NN_MEMCPY_MANAGER_SLOTS) - 1;
	mcm->ready_set = 0;
}
static void mcmanager_work_func(struct nn_graph *nn, void *mcmv);

// some common code for the mc_manager task buffer protocol
// returns a bit mask for the slot, and a pointer to the
// slot buffer to be filled in. If this fails, pointer is
// null, slot_bit = 0; this should never happen.
//
typedef struct
{
	unsigned slot_bit;
	struct nn_memcpy_manager_op *slotp;
} mcmanager_avail_slot;

static inline mcmanager_avail_slot
	__attribute__((always_inline))
	mcmanager_get_available_slot(struct nn_graph *nn, struct nn_memcpy_manager *mcm, int defer_flag)
{
	// find a slot to use: remove a '1' from avail_set.
	// pick the first available slot if defer_flag =0; last if defer_flag != 0
	int slotno;
	unsigned slot_bit;
	unsigned set_old0;
	unsigned set_old = mcm->avail_set;
	mcmanager_avail_slot result;
	do
	{
		unsigned set = set_old & (1 << NN_MEMCPY_MANAGER_SLOTS) - 1;
		if (!defer_flag)
		{
			slotno = Q6_R_ct0_R(set);
		}
		else
		{
			slotno = 31 - Q6_R_cl0_R(set);
		}
		if ((unsigned)slotno >= NN_MEMCPY_MANAGER_SLOTS)
		{
			errlog(nn, "??? invalid avail slot in mcmanager ");
			result.slotp = NULL;
			result.slot_bit = 0;
			return result;
		}
		slot_bit = 1 << slotno;
		unsigned set_new = set & ~slot_bit;
		set_old0 = set_old;
		set_old = __sync_val_compare_and_swap(&mcm->avail_set, set_old, set_new);
	} while (set_old != set_old0);
	result.slot_bit = slot_bit;
	result.slotp = &mcm->ops[slotno];
	return result;
}
// protocol to make the slot ready once it's been filled in.
// it's marked ready immediately, before we even check for "back-pressure".
// it may get picked up by an already-launched thread but
// that's OK; the thread we launch will then process an older slot.
static inline void
	__attribute__((always_inline))
	mcmanager_make_ready_slot(struct nn_graph *nn, struct nn_memcpy_manager *mcm, mcmanager_avail_slot slotinfo)
{
	// set the bit in 'ready_set', atomically
	__sync_or_and_fetch(&mcm->ready_set, slotinfo.slot_bit);

	// if too many pending, wait.
	while (mcm->pending >= NN_MEMCPY_MANAGER_MAX_THREADS)
	{
		nn_sem_wait(&mcm->done_sem);
		mcm->pending--;
	}
	mcm->pending++;
	nn_os_work_for_vector(nn, mcmanager_work_func, mcm);
}

// This is the # of l2fetch we do on vmemcpy in the
// 'launch' thread. The rest (if any) is done in the run thread.
//
#define NN_MCMANAGER_VMEMCPY_L2FETCH0 1024

// this executes memcpy or memset requests.
// This is *not* safe to call in multiple threads on the same nn_memcpy_manager
//  for copy: src != NULL
// for fill: src == NULL, and fillval is the *4 byte* pattern to fill with
// (it uses vmemset_32_2d_asm as the fill engine).
//
void nn_mcmanager_vmemcpy_or_set(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
								 void *dst, void const *src, unsigned len, unsigned fillval)
{
	unsigned copy_remain = len;
	uint8_t *dstp = (uint8_t *)dst;
	uint8_t const *srcp = (uint8_t const *)src;
	unsigned short_len = 1024; // short ops are < this
	if (src == NULL)
	{ // memset
		// different threshold for fills (and we can't do if it's not all the same byte)
		short_len = (Q6_R_vsplatb_R(fillval) == fillval) ? 512 : 0;
	}

	while (copy_remain > 0)
	{
		if (copy_remain < short_len)
		{
			if (src != NULL)
				memcpy(dstp, srcp, copy_remain);
			else
				memset(dstp, fillval, copy_remain);
			return;
		}
		// if >= 160K, break off min( 128K,n/2) and do the rest later
		//
		unsigned copynow = copy_remain;
		if (copynow > 160 * 1024)
		{
			copynow = min_u32(128 * 1024, copy_remain >> 1);
			// align the end of copy dest to multiple of 256
			uint8_t *dend = dstp + copynow;
			unsigned extra = (size_t)dend & 255;
			copynow -= extra;
		}
		// if memcpy, issue l2fetch for  at most NN_MCMANAGER_VMEMCPY_L2FETCH0
		// (the rest is done in the work thread)
		if (src != NULL)
		{
			l2fetch(srcp, 128, 128,
					min_i32((copynow + 127) >> 7, NN_MCMANAGER_VMEMCPY_L2FETCH0 / 128u));
		}
		//printf("!! %p <- %p of %u (%u left)\n", dstp, srcp, copynow, copy_remain);
		// find a slot to use:
		// pick the first available slot if the current
		// copy is large; and the last if it's small.
		unsigned avglen = mcm->avglen;
		mcm->avglen = (copynow + avglen) >> 1;
		mcmanager_avail_slot slot = mcmanager_get_available_slot(nn, mcm, copynow <= avglen);
		// we have allocated a slot. Fill it in and mark it ready.

		struct nn_memcpy_manager_op *opp = slot.slotp;
		if (opp == NULL) // failed to get slot
			return;
		opp->dst = dstp;
		opp->src = srcp;
		opp->len = copynow;
		opp->val = fillval;
		opp->rows = 0; // select '1-d' operation.

		// - protocol to make slot ready and launch work thread.
		mcmanager_make_ready_slot(nn, mcm, slot);

		dstp += copynow;
		if (src != NULL)
			srcp += copynow;
		copy_remain -= copynow;
	}
}
// this is like nn_mcmanager_vmemcpy_or_set but it deals with 2d memcpy and fill.
// it's simpler since it doesn't handle small ops directly, or break up large ones.
// if src is NULL, it's a fill and src_stride_or_fillval is the 32-bit fill value;
// if src is not NULL, it's a 2d copy.
void nn_manager_vmemcpy_or_set_2d(int width, int height, void *dst, void const *src,
								  unsigned dst_stride, unsigned src_stride_or_fillval, struct nn_graph *nn, struct nn_memcpy_manager *mcm)
{
	if (src != NULL && height < 2)
	{ // it's a 1-row 2d memcpy ?
		if (height <= 0)
			return; // it's no-row memcpy.
		height = 0; // this will get executed as a 1d memcpy.
		// do the early prefetch (work thread assumes this is done for 1d memcpy)
		l2fetch(src, 128, 128,
				min_i32((width + 127) >> 7, NN_MCMANAGER_VMEMCPY_L2FETCH0 / 128u));
	}
	mcmanager_avail_slot slot = mcmanager_get_available_slot(nn, mcm, 0);
	struct nn_memcpy_manager_op *opp = slot.slotp;
	if (opp == NULL) // failed to get slot
		return;
	opp->dst = dst;
	opp->src = src;
	opp->len = width;
	opp->val = src_stride_or_fillval;
	opp->rows = height;
	opp->dst_stride = dst_stride;
	mcmanager_make_ready_slot(nn, mcm, slot);
}
//
// this is wrapper for nn_mcmanager_vmemcpy, to handle tensors
// Returns -1 if the output is too small.
//
int nn_mcmanager_tensor_copy(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
							 struct tensor *dst, const struct tensor *src)
{
	unsigned copy_len = src->data_size;
	if (copy_len > dst->max_size)
		return -1;

	if (copy_len > 0 && dst->data != src->data)
		nn_mcmanager_vmemcpy(nn, mcm, dst->data, src->data, copy_len);
	dst->shape = src->shape;
	dst->format = src->format;
	dst->data_size = copy_len;
	return 0;
}

void nn_mcmanager_wait(struct nn_graph *nn, struct nn_memcpy_manager *mcm)
{
	while (mcm->pending > 0)
	{
		nn_sem_wait(&mcm->done_sem);
		--mcm->pending;
	}
}

static void
mcmanager_work_func(struct nn_graph *nn, void *mcmv)
{
	struct nn_memcpy_manager *mcm = (struct nn_memcpy_manager *)mcmv;
	//
	// grab one bit from 'ready_set' using compare_and_swap.
	// get the smallest bit # if more than one.
	unsigned set_old = mcm->ready_set;
	int slotno;
	unsigned set_old0;
	unsigned slot_bit = 0;
	do
	{
		slotno = Q6_R_ct0_R(set_old);
		if (slotno >= NN_MEMCPY_MANAGER_SLOTS)
			goto quit; // should not happen
		slot_bit = 1 << slotno;
		unsigned set_new = set_old & ~slot_bit;
		set_old0 = set_old;
		set_old = __sync_val_compare_and_swap(&mcm->ready_set, set_old, set_new);
	} while (set_old != set_old0);
	// remove data (and ownership) from the op slot before
	// starting the copy.
	struct nn_memcpy_manager_op const *opp = &mcm->ops[slotno];
	void *d = opp->dst;
	void const *s = opp->src;
	unsigned n = opp->len;
	unsigned fillval = opp->val;
	unsigned rows = opp->rows;
	unsigned dst_stride = opp->dst_stride;
	__sync_or_and_fetch(&mcm->avail_set, slot_bit);
	if (s != NULL)
	{
		if (rows == 0)
		{ // normal memcpy
			if (n > NN_MCMANAGER_VMEMCPY_L2FETCH0)
			{
				l2fetch((uint8_t const *)s + NN_MCMANAGER_VMEMCPY_L2FETCH0,
						128, 128, (n - NN_MCMANAGER_VMEMCPY_L2FETCH0 + 127) >> 7);
			}
			vmemcpy_asm(d, s, n);
		}
		else
		{
			// 2d memcpy of rows x n, with 'fillval' as src stride
			// use a variant with built-in prefetch strategy.
			vmemcpy_2d_general_with_prefetch(n, rows, d, dst_stride, s, fillval);
		}
	}
	else
	{
		if (rows == 0)
			vmemset_32_2d_asm(d, fillval, n, 1, 0);
		else
			vmemset_32_2d_general_asm(d, fillval, n, rows, dst_stride);
	}
quit:
	nn_sem_post(&mcm->done_sem);
}
// this is like memcpy_2d_general_asm except
// it does prefetch before each pass as needed, or once before
//     the whole thing.
void vmemcpy_2d_general_with_prefetch(int width, int height,
									  void *dst, int d_stride,
									  void const *src, unsigned s_stride)
{
	if (width < 1 || height < 1)
		return;
	// how much << do we need to get src & dst both multiples of 128?
	int Kshift = 7 - Q6_R_ct0_R((unsigned)d_stride | s_stride | 128);
	int K = 1 << Kshift;
	int npasses = min_i32(K, height); // this is the # of passes we need.
	uint8_t *dstp = (uint8_t *)dst;
	uint8_t const *srcp = (uint8_t *)src;
	unsigned src_aligned_stride = s_stride << Kshift;
	unsigned dst_aligned_stride = d_stride << Kshift;
	unsigned width_padded = (width + 127) & ~127u;
	// prefetch strategy:
	//  (1) if src_aligned_stride is equal to (width padded up to 128), then prefetch src as a single unit and we're done
	//  (2) otherwise issue a 2d prefetch  on first slice, and each slice which changes the base address to a new vector.
	//
	int need_prefetch_per_slice = 1;
	if (src_aligned_stride == width_padded)
	{
		l2fetch(srcp, 128, 128, (height * src_aligned_stride) >> 7);
		need_prefetch_per_slice = 0;
	}
	size_t srcp_prev = 0;
	int new_vec = need_prefetch_per_slice;

	for (int i = 0; i < npasses; i++)
	{
		int hx = (height + (K - 1) - i) >> Kshift;
		if (new_vec)
		{
			l2fetch(srcp, src_aligned_stride, width_padded, hx);
			srcp_prev = (size_t)srcp;
		}
		if (likely(hx > 1))
			vmemcpy_2d_asm(width, hx, dstp, dst_aligned_stride, srcp, src_aligned_stride);
		else
			vmemcpy_asm(dstp, srcp, width);
		srcp += s_stride;
		dstp += d_stride;
		new_vec = need_prefetch_per_slice ? (((size_t)srcp ^ srcp_prev) & ~127u) != 0 : 0;
	}
}

int nn_tensor_out_prepare_normal_fromshape(
	struct tensor *dst,
	struct shape const *shp,
	uint32_t type)
{
	return tensor_out_prepare_normal_fromshape_inline(dst, shp, type);
}

//
// 'actual function' version of this inline operation.
struct tensor_addressing __attribute__((pure))
nn_tensor_addressing_d32(struct tensor const *src)
{
	return tensor_addressing_d32_inline(src);
}
struct tensor_addressing __attribute__((pure))
nn_tensor_addressing_d32_16b(struct tensor const *src)
{
	return tensor_addressing_d32_16b_inline(src);
}
//
// check to see if tensorA and tensorB
// (both d32) are compatible for an elementwise operation.
// This includes broadcasting B dims to A.
// It also checks for misalignments in W & D dimensions.
//
// Normally it will a value >= 0, an 'or' of the tensor_compat_flags.
// if a problem is found, it will report an error and then return -1;
// 'ernm' is a name for the error messages.
//
// If there are compatibility flags and the corresponding bits are
// *not* in allowed_compat, this is considered an error.
//  e.g. A.height = 20, B.height=1 is considered a mismatch if compat_broadcast_H
// is not in allowed_compat.
//
// if compat_AtoB is in allowed_compat, situations where A broadcasts to B are
// accepted, and compat_AtoB will be set in the return value when this is encountered
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
//     skewed_D will never be flagged in the return value when tensB.depth =1.

//
int check_compatible_elementwise_d32(
	struct nn_graph *nn,
	char const *errnm,
	struct tensor const *tensA,
	struct tensor const *tensB,
	int allowed_compat)
{
	// check formats
	if (!tensor_is_d32(tensA))
	{
		return errlog(nn, "%s: 1st tensor not d32", errnm);
	}
	if (!tensor_is_d32(tensB))
	{
		return errlog(nn, "%s: 2nd tensor not d32", errnm);
	}
	int BtoA_ok = 1; // ok to broadcast B to A
	int AtoB_ok = (allowed_compat & compat_AtoB) ? 1 : 0;

	// check all the dims
	int compat_tags = 0;
	for (int i = 0; i < 4; i++)
	{
		int dimtag = compat_broadcast_B << i;
		int dima = tensA->shape.dimension[i];
		int dimb = tensB->shape.dimension[i];
		if (dima == dimb)
		{
			if (dimb == 1)
				compat_tags |= (allowed_compat & dimtag);
		}
		else
		{
			if (dimb == 1 && BtoA_ok && (allowed_compat & dimtag) != 0)
			{ // it's ok to broadcast B->A
				compat_tags |= dimtag;
				AtoB_ok = 0; // block the other option
			}
			else if (dima == 1 && AtoB_ok && (allowed_compat & dimtag) != 0)
			{ // A->B
				compat_tags |= dimtag | compat_AtoB;
				BtoA_ok = 0; // block the other option
			}
			else
			{
				return errlog(nn, "%s: incompatible %c: dims: %d vs %d", errnm, "BHWD"[i], dima, dimb);
			}
		}
	}
	// check width alignment
	if (((tensA->format.width_pad[0] ^ tensB->format.width_pad[0]) & 3) != 0)
	{
		compat_tags |= compat_misalign_W;
		if ((allowed_compat & compat_misalign_W) == 0 && (compat_tags & compat_broadcast_W) == 0)
		{
			return errlog(nn, "%s: width misalignment not supported", errnm);
		}
	}

	// check depth alignment
	int dpA = tensA->format.depth_pad[0];
	int dpB = tensB->format.depth_pad[0];

	if (dpA != dpB)
	{
		compat_tags |= compat_misalign_D;
		if ((compat_tags & compat_broadcast_D) == 0)
		{
			if (max_i32(dpA, dpB) + tensA->shape.depth > 32)
			{
				compat_tags |= compat_skewed_D;
			}
			if ((compat_tags & ~allowed_compat & (compat_misalign_D | compat_skewed_D)) != 0)
			{
				return errlog(nn, "%s: depth misalignment not supported", errnm);
			}
		}
	}
	return compat_tags;
}

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
//
int find_concat_shape(
	const struct tensor **input_tensors,
	int n_input,	// >= 1
	int concat_dim, // 0..3
	struct shape *allshape)
{
	// the first input tensor provides 'ref' dims

	uint32_t ref_batches = input_tensors[0]->shape.batches;
	uint32_t ref_height = input_tensors[0]->shape.height;
	uint32_t ref_width = input_tensors[0]->shape.width;
	uint32_t ref_depth = input_tensors[0]->shape.depth;

	uint32_t anydel_batches = 0, anydel_height = 0;
	uint32_t anydel_width = 0, anydel_depth = 0;
	uint32_t sum_del = 0;
	if (!(concat_dim >= 0 && concat_dim <= 3))
		return -1;

	// for all the others:
	//  find  del_XX = XX - ref_XX (mod uint32)
	//    - 'or' them all so that we can tell if any are different from ref
	//    - sum them all (across all dims) so we can figure the size of the
	//     concat dimension later.

	int i;
	for (i = 1; i < n_input; i++)
	{
		uint32_t del_batches = input_tensors[i]->shape.batches - ref_batches;
		anydel_batches |= del_batches;
		sum_del += del_batches;
		uint32_t del_height = input_tensors[i]->shape.height - ref_height;
		anydel_height |= del_height;
		sum_del += del_height;
		uint32_t del_width = input_tensors[i]->shape.width - ref_width;
		anydel_width |= del_width;
		sum_del += del_width;
		uint32_t del_depth = input_tensors[i]->shape.depth - ref_depth;
		anydel_depth |= del_depth;
		sum_del += del_depth;
	}
	// now:
	//  -all anydel_XX (except in the current dim) must be zero.
	//  -sum_del is the sum of the deltas XX - ref_XX in all dimensions on all inputs.
	//     This contains no contributions from the non-selected
	//     dimensions. So if we add n*ref_XX to this, the result is the sum
	//     of the selected dim across all inputs.

	if (anydel_batches != 0 && concat_dim != 0)
		return -2; // mismatch on batches
	if (anydel_height != 0 && concat_dim != 1)
		return -3; // mismatch on height
	if (anydel_width != 0 && concat_dim != 2)
		return -4; // mismatch on width
	if (anydel_depth != 0 && concat_dim != 3)
		return -5; // mismatch on depth

	// fill out the result...
	// one of these needs to be corrected later

	allshape->batches = ref_batches;
	allshape->height = ref_height;
	allshape->width = ref_width;
	allshape->depth = ref_depth;

	switch (concat_dim)
	{
	case 0:
		allshape->batches = n_input * ref_batches + sum_del;
		break;
	case 1:
		allshape->height = n_input * ref_height + sum_del;
		break;
	case 2:
		allshape->width = n_input * ref_width + sum_del;
		break;
	case 3:
		allshape->depth = n_input * ref_depth + sum_del;
		break;
	}
	return 0;
}

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
#if 0 ///// not currently used - add_d32 and mul_d32 don't need this now
uint8_t const *
construct_broadcasted_data_d32(
		struct tensor const * tens_in,		// tensor containing data
		struct tensor const * tens_ref,		// tensor used as shape/alignment ref
		int32_t *d32_stride_out,			// used to return the d32 stride of the result
		void * workbuf,						// optional work area
		uint32_t workbuf_len,				// len in bytes of work area
		void **allocbuf_out )
{
	int d_in = tens_in->shape.depth;
	int w_in = tens_in->shape.width;
	int wpad0_in = tens_in->format.width_pad[0];
	int dpad0_in = tens_in->format.depth_pad[0] & 31;

	int d_out = tens_ref->shape.depth;
	int w_out = tens_ref->shape.width;
	int wpad0_out = tens_ref->format.width_pad[0];
	int wpad1_out = tens_ref->format.width_pad[1];
	int dpad0_out = tens_ref->format.depth_pad[0] & 31;

	// get input data pointer. This is the start of the first depth group.
	uint8_t const *data_in = tensor_location_bhw_d32( tens_in, 0,0,0);
	int d32_stride_in = tensor_d32_stride_d32( tens_in);


	// determine output size
	// nd32_out is the nunber of deph slices, based on depth and output padding
	// we only need one when d_in = 1 (and when d_in != 1, d_in == d_out).

	int nd32_out = (unsigned)( dpad0_out + d_in + 31)/32;
	// output row bytes
	int wtotal_out = (wpad0_out + w_out + wpad1_out + 3)&~3;	// it should already be multiple of 4.

	*allocbuf_out = NULL;

	// (check for case where we can alias the input)
	// only possible when:
	//   d & w dimensions match
	//  dpad0 matches
	//   wpad0 matches mod 2, and output padding <= input padding.
	//  I.e. we are adjusting the pointer for (possibly) less left padding, nothing else.
	if(  w_in==w_out && d_in==d_out  && dpad0_in == dpad0_out && ((wpad0_in^wpad0_out)&3)==0
		&& wpad0_in >= wpad0_out  && tens_in->format.width_pad[1] >= wpad1_out ){
		*d32_stride_out = d32_stride_in;
		return data_in - wpad0_out*32;
	}

	int d32row_bytes = wtotal_out* 32;
	int allocsize = d32row_bytes * nd32_out;
	if( allocsize <= 0){
		return NULL;
	}
	void *mbuf;
	if( workbuf != NULL && workbuf_len >= (uint32_t)allocsize){
		mbuf = workbuf;
	}else{
		mbuf = nn_memalign(128,allocsize);
		*allocbuf_out = mbuf;
		if( mbuf == NULL) return NULL;
	}
	uint8_t * mbuf_u8 = (uint8_t *)mbuf;

	//
	if( d_in == 1){	// broadcast along depth
		uint8_t const * rp =  & data_in[dpad0_in];
		uint64_t * wp;
		int wcount;
		int dcount;		// this is in units if 16 bytes
		int i,j;

		if( w_in == 1){ // broadcast along width too
			wp = (uint64_t*)mbuf;
			wcount = 1;			// just fill the whole thing...
			dcount = wtotal_out*2;
		}else{
			if( w_in != w_out) goto do_error;
			wp = (uint64_t*)mbuf + wpad0_out * 4;	// start after output width padding
			wcount = w_out;
			dcount = 2;
		}
		for( i =0; i < wcount; i++ ){
			uint32_t d32 = Q6_R_vsplatb_R(rp[i*32]);	// get one byte from input
			uint64_t data = Q6_P_combine_RR( d32,d32);	// splat to 8 bytes
			for( j = 0; j < dcount; j++){
				*wp++ = data;
				*wp++ = data;
			}
		}
		// broadcast along depth, d32_stride is zero.
		*d32_stride_out = 0;
		return mbuf_u8;
	}else{
		if( d_in != d_out) goto do_error;
		if(w_in ==1){
			// broadcast along width only; and align on depth to output.
			// Start by constructing the depth slice (with dpad0_out) in the first
			// 32*nd32_out bytes of the buffer; and then copy to the rows.
			uint8_t *wp = mbuf_u8 + dpad0_out;
			int di_in = dpad0_in;
			int depth_remain = d_out;
			int i,j;
			while(1){
				int dcopy = min_i32( depth_remain, 32-di_in);	// most we can copy
				memcpy( wp, data_in+di_in, dcopy);
				depth_remain -= dcopy;
				if( depth_remain <= 0) break;
				wp += dcopy;		// writes are contiguous
				di_in = 0;			// move to next depth slice in input
				data_in += d32_stride_in;
			}
			// for each d32 slice, copy it to its allocated row. Start with the last
			// one (so we don't overwrite)
			//
			for( i = nd32_out-1; i >= 0; --i ){
				uint64_t const * rp = (uint64_t const *)( mbuf_u8 + 32 * i);
				uint64_t *wp = (uint64_t *)( mbuf_u8 + d32row_bytes * i);
				uint64_t x0 = rp[0];
				uint64_t x1 = rp[1];
				uint64_t x2 = rp[2];
				uint64_t x3 = rp[3];
				for( j = 0; j < wtotal_out; j++){
					wp[0] = x0;
					wp[1] = x1;
					wp[2] = x2;
					wp[3] = x3;
					wp += 4;
				}
			}
		}else{
			if( w_in != w_out) goto do_error;
			// broadcasting neither on width or depth, but there is some misalignment to take care of.
			// This can be done with one memcpy per d32 slice if
			//  (1) both source and dest fit in a single depth unit; *or*
			//  (2) if they have the same depth padding (i.e. only the w is misaligned).
			if( ( nd32_out == 1 && dpad0_in + d_in <=32)
			  || dpad0_in == dpad0_out ){
					int dx_in = dpad0_in;
			  	  	int dx_out = dpad0_out;
			  	  	int dremain = d_in;
			  	  	uint8_t *wp = mbuf_u8+ 32*wpad0_out;
			  	  	uint8_t const * rp = data_in;
			  	  	while(1){
			  	  		int dcopy = min_i32( dremain, 32-dx_out);	// number to copy (per width)
						memcpy( wp + dx_out,	// output address
								rp + dx_in,					// input address
								(w_in-1)*32 + dcopy);
						dremain -= dcopy;
						if(dremain <=0)
							break;
						// we only get here if there are multiple depth slices with
						// the same dpad0 on in & out.
						wp += d32row_bytes;
						rp += d32_stride_in;
						dx_in = dx_out = 0;
			  	  	}
			}else{
				// the ugly case...
				// do this within each width slot, using 64-bit aligned reads and writes.
				// There are 4 u64 elements within each slice.
				//  precalculated:
				//   - xin0 = offset for initial read (0..3)
				//   - xout0 = offset for initial write (0..3)
				//   - skew = byte skew for align
				//   - count = number of u64's to write
				// rd_first is a flag indicating we need a pre-read at the start.
				// rd_last is a flag which is true if the last output word needs its second input read.
				//
				int skewn = (dpad0_in&7)-(dpad0_out&7);
				int xin0 = dpad0_in >> 3;	// offset containing first input byte; 0..3
				int xout0 = dpad0_out >>3;	// offset containing first output byte; 0..3
				int xcount =  ((dpad0_out + d_out+7)>>3) - (xout0+1);	// total # of u64s to write (-1)
				int rd_first = skewn >= 0;
				// only need to read extra for last, if the last output word is less full than
				// the last input word (avoid possible 'wild read').
				int rd_last =  (( dpad0_out + d_out-1)&7)  >  (( dpad0_in + d_out-1)&7);
				int i,j;
				uint64_t * wp = (uint64_t *)( mbuf_u8+ 32*wpad0_out );
				uint64_t const * rp = (uint64_t const *) data_in;

				uint64_t dL = 0;
				uint8_t skew = (uint8_t)skewn;

				for( i = 0; i < w_in; i++){
					uint64_t const * rpx = rp + i*4;
					uint64_t  * wpx = wp + i*4;
					int xin = xin0;
					int xout = xout0;
					if( rd_first){		// preload dL (if rd_first){
						dL = rpx[xin];
						if(++xin>=4){
							rpx = (uint64_t const*)( (char const*)rpx + d32_stride_in);
							xin = 0;
						}
					}
					for( j = 0; j < xcount; j++){
						uint64_t dR = rpx[xin];
						wpx[xout] =  Q6_P_valignb_PPp( dR, dL, skew);
						dL = dR;
						if(++xin>=4){
							rpx  = (uint64_t const*)( (char const*)rpx + d32_stride_in);
							xin = 0;
						}
						if(++xout>=4){
							wpx  = (uint64_t*)( (char*)wpx + d32row_bytes);
							xout = 0;
						}
					}
					// we have one more to do .. but only do read if rd_last != 0.
					uint64_t dLast = dL;
					if( rd_last) dLast = rpx[xin];
					wpx[xout] =  Q6_P_valignb_PPp( dLast, dL, skew);

				}
			}
		}
		*d32_stride_out = d32row_bytes;
		return mbuf_u8;
	}
 /* unreachable*/

 do_error:
 	 if( *allocbuf_out!=NULL){
 		 nn_free(*allocbuf_out);
 		 *allocbuf_out = NULL;
 	 }
 	 return NULL;
}

#endif

//
// This utility examines a d32 tensor, which is assumed to have a shape (1,1,w,d),
// and finds the actual range of u8 values stored within.
// The range is returned as  (maxval<<8) | minval.
// No hvx instructions (it uses 64-bit hexagon vector ops).
//
//
// it works in depth chunks of size 8, finding the range in each chunk,
// and then masking out unused lanes in the chunk before proceeding.
//
//  all_min,all_max = {all_ff},{all_00}
//  for each depth chunk containing data:
//      dc_min,dc_max = all_min, all_max
//      for iw = 0 .. width-1:
//          update dc_min,dc_max with data at iwid
//      all_min, all_max = dc_min,dc_max (but only in 'active' byte lanes)
//  reduce all_min, all_max laterally
//

uint32_t find_range_in_wd_tensor_d32(struct tensor const *tens_in)
{
	int width = tens_in->shape.width;
	int depth = tens_in->shape.depth;

	int dpos = tens_in->format.depth_pad[0]; // where depth data starts
	int dend = dpos + depth;				 // where it ends
	int dpos0 = dpos & ~7;					 // round down to boundary

	uint8_t const *data_in = tensor_location_bhw_d32(tens_in, 0, 0, 0);
	int d32_stride_in = tensor_d32_stride_d32(tens_in);

	int xind = dpos >> 3; // get chunk index (0..3) within d32 slice

	uint64_t const *chunkp = (uint64_t const *)data_in + xind; // point to first containing data.

	uint64_t all_min = (uint64_t)-1;
	uint64_t all_max = 0;

	while (1)
	{ // while dpos < dend
		uint64_t dc_min = all_min;
		uint64_t dc_max = all_max;
		for (int i = 0; i < width; i++)
		{
			uint64_t d = chunkp[i * 4];
			dc_min = Q6_P_vminub_PP(dc_min, d);
			dc_max = Q6_P_vmaxub_PP(dc_max, d);
		}
		//
		// is chunk full?
		//
		int dpnext = dpos0 + 8; // start of next chunk
		int msk = 0xFF;
		if (dpos0 < dpos || dpnext > dend)
		{							 // chunk is not full
			msk = msk << (dpos & 7); // lo end mask
			if (dpnext > dend)
			{
				int m2 = (1 << (dend & 7)) - 1; // hi end mask
				msk &= m2;
			}
		}
		// where msk=1, replace all_min, all_max with 'dc_min', 'dc_max'.
		all_min = Q6_P_vmux_pPP(msk, dc_min, all_min);
		all_max = Q6_P_vmux_pPP(msk, dc_max, all_max);
		if (dpnext >= dend)
			break;
		dpos = dpos0 = dpnext;
		// advance to next chunk
		chunkp++;
		if (++xind >= 4)
		{
			xind = 0;
			chunkp = (uint64_t const *)((uint8_t const *)chunkp - 32 + d32_stride_in);
		}
	}
	// now reduce the 8 max and 8 min..

	all_max = Q6_P_vmaxub_PP(
		Q6_P_shuffeb_PP(all_max, ~all_min),
		Q6_P_shuffob_PP(all_max, ~all_min));
	// now we have 4 of { ~min, max}
	// 2 more reductions.
	all_max = Q6_P_vmaxub_PP(all_max, all_max >> 32);
	uint32_t x = Q6_P_vmaxub_PP(all_max, all_max >> 16);

	// truncate to 16 bits and ~min -> min
	return (uint16_t)x ^ 0xFF;
}

//
// When reducing selected dimensions on a linearly-addressed tensor,
// it is possible to 'munge' adjacent dims together if they are both
// reduced, or if both are not reduced. (where the input dim is 1,
// the dimension may be combined with either).
// for instance [2,3,5,6] -> [2,3,1,1] can be done as 6 1-d reductions [2*3,5*6]->[2*3,1]

// Thus with 4 dimensions, the worst case is to have 2 loops
// of reduction  (e.g [2,3,4,12] -> [2,1,4,1] we need reductions on 3->1, 12->1
// but  [2,3,1,12] -> [2,1,1,1] can be done as [2,3*1*12]->[2,1]
// Similarly, you never need more than 2 loops if iteration.
//
// General case can be done by mapping to a 5-dimensional case where
// the reduction is always done on the 2nd and 4th dims:
//    [p,q,r,s,t]->[p,1,r,1,t]
// (where at least one of p,r,t is 1; and s>1 (unless there is no reduction
// at all).
//
// Examples:
//  [2,3,5,6]->[1,1,1,1]     [1,1,1,2*3*5*2,1] -> [1,1,1,1,1]
//  [2,3,5,6]->[2,3,1,1]	 [1,1,2*3,5*6,1]	->[1,1,2*3,1,1]
//  [2,3,5,6]->[2,1,5,6]     [1,1,2,3,5*6] -> [1,1,2,1,5*6]
//
// This function finds the p,q,r,s,t parms to map a given
// reduction problem to [p,q,r,s,t]-> [p,1,r,1,t]
// It promises that:
//     -all are >= 1
//     -at least one of p,t is 1
//     -if s=1, then p=q=r=1 as well (this is a 'no reduction' case)
//     -if q=1, then p = 1  (1d reduction needs only r,s,t)
//     -if q >1, then r != 1  (2d reduction is only spec when needed)
//
// So there are four possible outcomes:
//     [1,1,1,1,t] :         no reduction, copy 't'
//     [1,1,r,s,t], s>1:     'r' of 1d-reduction(s) of t-vector
//                            r=t=1 is a full reduction.
//     [p,q,r,s,1], q,r,s>1  2-d reduction.
//     [1,q,r,s,t], q,r,s,t>1 :  2-d reduction
//
// IMPORTANT:
//  caller needs to ensure all dims >=1, and all dims
// match across in/out except where shape_out dim is 1.
//
void nn_find_generic_reduction_dims(
	struct shape const *shape_in,
	struct shape const *shape_out,
	int generic_reduction_dims[5])
{
	int i;
	int red_dims = 0;
	int optred_dims = 0; // 1->1 dims
	// make a tally of the dimensions that need reduction;
	// also find the input * output sizes
	int red_total = 1;
	int out_total = 1;
	for (i = 0; i < 4; i++)
	{
		int dout = shape_out->dimension[i];
		int din = shape_in->dimension[i];
		out_total *= dout;
		if (dout == 1)
		{
			int m = 1 << i;
			if (din > 1)
			{
				red_dims |= m;
				red_total *= din;
			}
			else
			{
				optred_dims |= m;
			}
		}
	}
	// red_dims : a->1  (a > 1)
	// optred_dims: 1->1
	// nored_dims : a->a
	int nored_dims = (~(red_dims | optred_dims)) & 0xF;

	// note: out_total * red_total = in_total
	// default the first 4 outputs to 1...
	for (i = 0; i < 4; i++)
		generic_reduction_dims[i] = 1;

	// check special cases:
	// red_total = 1:  no reduction at all
	// out_total = 1: full reduction
	if (red_total == 1 || out_total == 1)
	{
		generic_reduction_dims[3] = red_total; // 's'
		generic_reduction_dims[4] = out_total; // 't'
		return;
	}
	// now there is at least one dimension which is non-trivially
	// reduced, and at least one which is non-trivially maintained.
	// Starting with depth, and working to 'batch',
	// first, collect 't' from dims which are a->a (and including 1->1)
	// then 's' from dims which are a->1 (and including 1->1)
	// (the condition alternates, but 1->1 cases never force a change
	// of partition)
	//
	int idim = 3;
	int opos;
	for (opos = 4; opos >= 1; opos--)
	{ // t,s,r,q
		int k = 1;
		int exclude_set = ((opos & 1) == 0) ? red_dims : nored_dims;
		while (idim >= 0)
		{
			if (((exclude_set >> idim) & 1) != 0)
				break; // dim isn't in this partition.
			k *= shape_in->dimension[idim];
			idim--;
		}
		generic_reduction_dims[opos] = k;
		if (idim < 0)
			break; // all done
	}
	// we only finish that loop, with idim = 0,
	// in cases where we need p = in_batches = out_batches.
	//  e.g. [ 2, 4, 5, 16] -> [2, 1, 5, 1] needs pqrst = {2,4,5,16,1}
	if (idim >= 0)
	{
		// it should be only 0
		generic_reduction_dims[0] = shape_in->batches;
	}
}
//
// This function finds the output shape for reductions
// The 'generic_reduction_dims' is obtained by by passing
// the input and output shape to find_generic_reduction_dims,
// but in this case the 'output shape' is the shape *before* squeezing
// reduced dims (which can be different from the output shape
// when padding == NN_PAD_VALID).
//
void nn_find_reduction_shape(struct nn_node *self, struct nn_graph *nn,
							 struct shape *out_shape_p, int generic_reduction_dims[5])
{
	int i;
	if (self->n_inputs >= 2)
	{
		struct shape out_shape = self->inputs[0]->shape;
		const struct tensor *reduction_dims_tensor = self->inputs[1];
		const int32_t *dims = (const int32_t *)reduction_dims_tensor->data;
		int32_t dim;
		int repl = (self->padding == NN_PAD_VALID) ? 0 : 1;
		int32_t true_rank = 4;
		if (self->n_inputs >= 3)
			true_rank = tensor_get_int32(self->inputs[2], 0);
		int reduce_all = 0;

		for (i = 0; i < reduction_dims_tensor->shape.depth; i++)
		{
			dim = 4 - true_rank + dims[i]; // 0,1,2,3 -> b,h,w,d
			if (dims[i] < 0)
			{
				reduce_all = 1;
				break;
			}
			else if (0 <= dim && dim <= 3)
			{
				out_shape.dimension[dim] = repl;
			}
		}
		if (!reduce_all)
		{
			if (self->padding == NN_PAD_VALID)
			{
				/* Dimensions to be reduced have been set to 0 */
				// copy the dims to out_shape_p, starting at the D, but skipping
				// zero dims; then pad 1's after.
				// As we do this we also need to replace the 0's with 1 in out_shape.
				int ir, iw = 3;
				for (ir = 3; ir >= 0; ir--)
				{
					int dn = out_shape.dimension[ir];
					if (dn != 0)
					{
						out_shape_p->dimension[iw--] = dn;
					}
					else
					{
						// skip, and replace in out_shape.
						out_shape.dimension[ir] = 1;
					}
				}
				while (iw >= 0) // fill vacated dims with 1
					out_shape_p->dimension[iw--] = 1;
			}
			else
			{
				*out_shape_p = out_shape;
			}
			// find the 'generic reduction dims'
			nn_find_generic_reduction_dims(&self->inputs[0]->shape, &out_shape, generic_reduction_dims);
			/*
			{
				struct shape const *insh = &self->inputs[0]->shape;
				printf("[%d:%d:%d:%d] (%d,%d,%d,%d)->[%d:%d:%d:%d] (%s): [%d:%d:%d:%d:%d]\n",
					(int)insh->batches, (int)insh->height, (int)insh->width, (int)insh->depth,
					reduction_dims_tensor->shape.depth >= 1 ? dims[0] : 8,
					reduction_dims_tensor->shape.depth >= 2 ? dims[1] : 8,
					reduction_dims_tensor->shape.depth >= 3 ? dims[2] : 8,
					reduction_dims_tensor->shape.depth >= 4 ? dims[3] : 8,
						(int)out_shape_p->batches,(int)out_shape_p->height,(int)out_shape_p->width,(int)out_shape_p->depth,
						self->padding == NN_PAD_VALID? "VALID" : "SAME ",
						generic_reduction_dims[0],generic_reduction_dims[1],generic_reduction_dims[2],
						generic_reduction_dims[3],generic_reduction_dims[4]);
			}*/

			return;
		}
	}
	// set all to 1 (full reduction)
	//
	out_shape_p->batches = 1;
	out_shape_p->height = 1;
	out_shape_p->width = 1;
	out_shape_p->depth = 1;
	{
		struct shape const *insh = &self->inputs[0]->shape;
		generic_reduction_dims[0] = 1;
		generic_reduction_dims[1] = 1;
		generic_reduction_dims[2] = 1;
		generic_reduction_dims[3] = insh->batches * insh->height * insh->width * insh->depth;
		generic_reduction_dims[4] = 1;
	}
}

// for running a generic unary float op.
// This could be enhanced to use threads,
// currently does not (and need_hvx is ignored)
//
//
int nn_generic_unary_float_op(struct nn_node *self, struct nn_graph *nn,
							  void (*func)(float *, float const *, int n, void *info),
							  void *info, int need_hvx)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int elements = tensor_element_count(in_tensor);
	const float *in_data = in_tensor->data;
	float *out_data = out_tensor->data;

	if (tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_FLOAT) != 0)
	{
		return errlog(nn, "%s: output too small", hexagon_nn_op_names[self->node_type]);
	}
	if (elements > 0)
	{
		(*func)(out_data, in_data, elements, info);
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////

//=================================================
// Operations on "Tensor Slice" data structure
//=====================================================

//
// descriptor for a d32 tensor 'slice' view.
// This describes 'before'padding of the depth dimension (0..31)
// for height, width, no padding is described; batch_stride and height_stride are used instead,
// and the 'data' pointer is offset according to the 'left' padding in these dims.
// 'after' padding of the depth dimension is not given explicitly (but it's depth_total-depth_pad_before - shape.depth)
//
// to locate a byte at b,h,w,d:
//     dx = d+ depth_pad_before;
//     data + batch_stride * b + height_stride * h + d32_stride * (dx/32) + w*32 + (dx%32)
//
// Note, if a tensor is sliced in a depth dimension, in such a way that d32 chunks are eliminated, we trim
// depth_total to keep padding in range 0..31 at both ends
// Example:
//   original tensor has depth = 68, padded 0 and 28  to a total of 96
//   -slice extracts  34..45 in depth dimension.
//  Resulting slice will have depth = 12, depth_total = 32, depth_pad_before = 2
// So the original tensor has 3 chunks of 32, and the new one has one (the middle of the original);
//  - the first chunk of 32 is skipped by adding d32_stride to the data pointer.
//
// def is in nn_graph_types.h
// struct tensor_slice_d32 {
//	struct shape shape;				// batches, height, width, depth
//	uint16_t depth_total;			// depth total including padding (mult. of 32)
//	uint16_t depth_pad_before;		// [0..31] add this to data to reach (0,0,0,0)
//	uint8_t * data;					// this is 32 aligned
//	int32_t batch_stride;			// stride between batches (multiple of vector)
//	int32_t height_stride;				// stride between rows (multiple of vector)
//	int32_t d32_stride;				// stride between chunks (multiple of vector)
// };

/*
 * make a slice from a tensor, it encompasses the whole tensor
 */
int tensor_slice_from_tensor_d32(struct tensor_slice_d32 *slc, const struct tensor *tens)
{
	if (!tensor_is_d32(tens))
		return -1;
	slc->shape = tens->shape;
	int32_t depth_total = tensor_d_total_d32(tens);
	int32_t depth_pad_before = tens->format.depth_pad[0];
	int32_t batch_stride = tensor_batch_stride_d32(tens);
	int32_t height_stride = tensor_row_stride_d32(tens);
	int32_t d32_stride = tensor_d32_stride_d32(tens);
	slc->data = (uint8_t *)tens->data + height_stride * tens->format.height_pad[0] + 32 * tens->format.width_pad[0];
	slc->depth_total = depth_total;
	slc->depth_pad_before = depth_pad_before;
	slc->batch_stride = batch_stride;
	slc->height_stride = height_stride;
	slc->d32_stride = d32_stride;
	return 0;
}

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
	struct tensor_slice_d32 *slc_out,	  // output slice
	struct tensor_slice_d32 const *slc_in, // input slice (can be same slice).
	int32_t dim_no,						   // dimension to slice
	int32_t lo_index,					   // start of slice (0..size-1)
	int32_t newsize)					   // new size. 1 <= newsize <= size-lo_index
{
	if (slc_in != NULL && slc_in != slc_out)
	{
		*slc_out = *slc_in; // copy the source slice
	}
	// now we just work on slc_in...
	int min_curr_size;

	if (lo_index < 0 || newsize < 1 ||
		__builtin_sadd_overflow(lo_index, newsize, &min_curr_size))
		return -2;

	// min_curr_size guaranteed to be >=1, <= MAXINT)

	uint32_t *nsize_p;
	int32_t dim_stride;

	switch (dim_no)
	{
	case 0: // batch
		nsize_p = &slc_out->shape.batches;
		dim_stride = slc_out->batch_stride;
		break;
	case 1: // height
		nsize_p = &slc_out->shape.height;
		dim_stride = slc_out->height_stride;
		break;
	case 2: // width
		nsize_p = &slc_out->shape.width;
		dim_stride = 32;
		break;
	case 3: // depth
	{
		if (slc_out->shape.depth < (unsigned)min_curr_size)
			return -2;
		unsigned new_dpad = slc_out->depth_pad_before + lo_index;
		unsigned reduce_d32_before = new_dpad / 32; // # of groups to cut
		new_dpad %= 32;
		unsigned new_dtotal = (new_dpad + newsize + 31) & ~31; // new total size
		slc_out->data += reduce_d32_before * slc_out->d32_stride;
		slc_out->depth_pad_before = new_dpad;
		slc_out->depth_total = new_dtotal;
		slc_out->shape.depth = newsize;
	}
		return 0;

	default:
		return -1;
	}
	// generic slicing along batch, height, width dims.
	if (*nsize_p < (unsigned)min_curr_size)
		return -2;
	slc_out->data += lo_index * dim_stride;
	*nsize_p = newsize;
	return 0;
}

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
	struct tensor_slice_d32 *slc,	// output slice
	struct tensor const *src_tensor, // tensor
	struct shape const *ref_shape,   // shape to match
	int slice_dim,					 // dimension to slice along
	int *slicepos_p					 // keeps track of slice.
)
{
	int slicepos = *slicepos_p;
	if (slicepos <= 0)
	{
		if (slicepos != 0)
			return -3;
		int k = tensor_slice_from_tensor_d32(slc, src_tensor);
		if (k != 0)
			return -4;
		if (ref_shape == NULL)
			return 0; // a 'dummy' call to set up slice
	}
	// compare all the dims
	int dim_mismatch = (src_tensor->shape.depth != ref_shape->depth) ? 8 : 0;
	if (src_tensor->shape.width != ref_shape->width)
		dim_mismatch |= 4;
	if (src_tensor->shape.height != ref_shape->height)
		dim_mismatch |= 2;
	if (src_tensor->shape.batches != ref_shape->batches)
		dim_mismatch |= 1;
	int dim_sel = 1 << slice_dim;

	// dim_mismatch must be either 0 or dim_sel.
	if ((dim_mismatch | dim_sel) != dim_sel)
		return (slice_dim >= 0 && slice_dim <= 3) ? -2 : -1;

	int dimsize, new_dimsize, next_slicepos;
	// get '0' position of the tensor (excluding depth_pad_before; mult of 32).
	uint8_t *slc_data = tensor_location_bhw_d32(src_tensor, 0, 0, 0);

	switch (slice_dim)
	{
	case 0: // batches
		dimsize = src_tensor->shape.batches;
		new_dimsize = ref_shape->batches;
		slc->shape.batches = new_dimsize;
		slc_data += slc->batch_stride * slicepos;
		break;
	case 1: // height
		dimsize = src_tensor->shape.height;
		new_dimsize = ref_shape->height;
		slc->shape.height = new_dimsize;
		slc_data += slc->height_stride * slicepos;
		break;
	case 2: // width
		dimsize = src_tensor->shape.width;
		new_dimsize = ref_shape->width;
		slc->shape.width = new_dimsize;
		slc_data += 32 * slicepos;
		break;
	case 3: // depth slicing is messier...
	{
		dimsize = src_tensor->shape.depth;
		new_dimsize = ref_shape->depth;
		unsigned new_dpad_before = src_tensor->format.depth_pad[0] + slicepos;
		unsigned advance_d32 = new_dpad_before / 32;
		new_dpad_before %= 32;
		slc->depth_total = (new_dpad_before + new_dimsize + 31) & ~31;
		slc->depth_pad_before = new_dpad_before;
		slc->shape.depth = new_dimsize;
		// adjust for d32 sections skipped, if any
		slc_data += advance_d32 * slc->d32_stride;
	}
	break;
	default:
		return -1;
	}

	next_slicepos = slicepos + new_dimsize;
	if (new_dimsize <= 0 || next_slicepos > dimsize)
		return -3;
	slc->data = slc_data;
	*slicepos_p = next_slicepos;
	return (next_slicepos == dimsize) ? 1 : 0;
}

// array fill with int32.
void *
memset_32(void *dst, int val, unsigned num)
{
	int n = num;
	int32_t *ptr = (int32_t *)dst;
	if (n >= 5)
	{
		if (val == 0)
		{
			//  maybe memset has some zero-fill cache-line mojo
			return memset(dst, 0, n * sizeof(int32_t));
		}
		if (((size_t)ptr & 4) != 0)
		{
			*ptr++ = val;
			--n;
		}
		int n4 = n >> 2;
		__builtin_assume(n4 > 0);
		int64_t v64 = Q6_P_combine_RR(val, val);
		for (int i = 0; i < n4; i++)
		{
			((uint64_t *)ptr)[0] = v64;
			((uint64_t *)ptr)[1] = v64;
			ptr += 4;
		}
		n &= 3;
	}
	for (int i = 0; i < n; i++)
	{
		*ptr++ = val;
	}
	return dst;
}
// array fill with int16
void *
memset_16(void *dst, int val, unsigned num)
{
	int n = num;
	int16_t *ptr = (int16_t *)dst;
	if (n >= 11)
	{
		if (val == 0)
		{
			//  maybe memset has some zero-fill cache-line mojo
			return memset(dst, 0, n * sizeof(int16_t));
		}
		while (((size_t)ptr & 6) != 0)
		{
			*ptr++ = val;
			--n;
		}
		int n8 = n >> 3;
		__builtin_assume(n8 > 0);
		val = Q6_R_combine_RlRl(val, val);
		int64_t v64 = Q6_P_combine_RR(val, val);
		for (int i = 0; i < n8; i++)
		{
			((uint64_t *)ptr)[0] = v64;
			((uint64_t *)ptr)[1] = v64;
			ptr += 8;
		}
		if (n & 4)
		{
			((uint64_t *)ptr)[0] = v64;
			ptr += 4;
		}
		n &= 3;
	}
	for (int i = 0; i < n; i++)
	{
		*ptr++ = val;
	}
	return dst;
}
//
// PAD 4-d flat tensor
//
//
// using scalar ops, pad the tensor
//  'inpv' (b_in,h_in,w_in,d_in)
//  by the given 8 pad amounts, and store at 'outpv'.
//
// if 'padval' is 0, element_size can be any value >=1
// Otherwise, element_size must be 1,2 or 4; and the lower
//    8,16,32 bits of 'padval' are used accordingly.
//    For element_size = 2 or 4, the input and output pointers
//    must be a aligned to a multiple of that.
//

void do_pad(
	void *outpv,
	const void *inpv,
	const int32_t b_in,
	const int32_t h_in,
	const int32_t w_in,
	const int32_t d_in,
	const int32_t pre_b,
	const int32_t post_b,
	const int32_t pre_h,
	const int32_t post_h,
	const int32_t pre_w,
	const int32_t post_w,
	const int32_t pre_d,
	const int32_t post_d,
	const int32_t element_size,
	const int32_t padval)
{
	int filld = element_size; // multiplier for fills
	void *(*memset_fp)(void *, int, size_t) = memset;
	if (padval != 0)
	{
		if (element_size == 2)
		{
			filld = 1;
			memset_fp = memset_16;
		}
		else if (element_size == 4)
		{
			filld = 1;
			memset_fp = memset_32;
		}
	}

	const char *in = inpv;
	char *out = outpv;
	int h_out = h_in + pre_h + post_h;
	int w_out = w_in + pre_w + post_w;
	int d_out = d_in + pre_d + post_d;
	int out_wd = w_out * d_out;
	int out_hwd = h_out * out_wd;

	int pre_h_fill = out_wd * pre_h;
	int post_h_fill = out_wd * post_h;

	int pre_w_fill = d_out * pre_w;
	int post_w_fill = d_out * post_w;
	int in_d_size = d_in * element_size;
	int b, h, w;

	if (pre_b)
	{
		int pre_b_fill = out_hwd * pre_b;
		(*memset_fp)(out, padval, pre_b_fill * filld);
		out += pre_b_fill * element_size;
	}
	for (b = 0; b < b_in; b++)
	{
		if (pre_h_fill)
		{
			(*memset_fp)(out, padval, pre_h_fill * filld);
			out += pre_h_fill * element_size;
		}
		for (h = 0; h < h_in; h++)
		{
			if (pre_w_fill)
			{
				(*memset_fp)(out, padval, pre_w_fill * filld);
				out += pre_w_fill * element_size;
			}
			if (d_out == d_in)
			{
				memcpy(out, in, in_d_size * w_in);
				in += in_d_size * w_in;
				out += in_d_size * w_in;
			}
			else
			{
				for (w = 0; w < w_in; w++)
				{
					if (pre_d)
					{
						(*memset_fp)(out, padval, pre_d * filld);
						out += pre_d * element_size;
					}
					memcpy(out, in, in_d_size);
					in += in_d_size;
					out += in_d_size;
					if (post_d)
					{
						(*memset_fp)(out, padval, post_d * filld);
						out += post_d * element_size;
					}
				}
			}
			if (post_w_fill)
			{
				(*memset_fp)(out, padval, post_w_fill * filld);
				out += post_w_fill * element_size;
			}
		}
		if (post_h_fill)
		{
			(*memset_fp)(out, padval, post_h_fill * filld);
			out += post_h_fill * element_size;
		}
	}
	if (post_b)
	{
		int post_b_fill = out_hwd * post_b;
		(*memset_fp)(out, padval, post_b_fill * filld);
		out += post_b_fill * element_size;
	}
}

/////////////////////////////////////////////////////////////////////
//
// Checksumming
//  This is fast but reasonably strong checksum algo for debugging.
//
//  Based on misimplemented CRC64 (for speed, only two shifts are done between inserting
// each byte, which compromises error-burst detection, but all bits are still included in the
//  sum)
// Also - it's contrived so that if input consists entirely of the same byte 'k'; the checksum
// will be 'k'. (and as a special case, a checksum of any number of 00 byte, including none, is 0)
//
// This is done by:
//    - init check to 0;
//    - xor all incoming bytes with x0 (first byte) before applying to checksum
//    - xor final check with x0.
//

static inline uint64_t checksum_byte_add(uint64_t prev, uint8_t newbyte)
{
	// this should really be done 8 times per byte, to make a proper CRC
	uint64_t newchk = Q6_P_lfs_PP(prev, 0x000000000000001BuLL);
	newchk = Q6_P_lfs_PP(newchk, 0x000000000000001BuLL);
	return newchk ^ newbyte;
}

// checksum of an arbitrary area of memory
uint64_t nn_checksum_bytes(void const *p, int n)
{
	uint64_t chk = 0;
	uint8_t const *rp = (uint8_t const *)p;
	if (n > 0)
	{
		unsigned x0 = rp[0];
		for (int i = 0; i < n; i++)
		{
			chk = checksum_byte_add(chk, rp[i] ^ x0);
		}
		chk ^= x0;
	}
	return chk;
}

//
// checksum of a flat tensor (or, all of the data in a d32 tensor including padding
//
uint64_t
nn_checksum_tensor(struct tensor const *tens)
{
	uint64_t chk = nn_checksum_bytes(tens->data, tens->data_size);
	return chk;
}
//
// checksum of a d32 tensor - only include the non-padding bytes,
// and include them in the same order as the 'flat' checksum
//
uint64_t nn_checksum_tensor_d32(struct tensor const *tens)
{
	uint64_t check = 0;
	struct tensor_addressing addr = tensor_addressing_d32(tens);

	int b = tens->shape.batches;
	int h = tens->shape.height;
	int w = tens->shape.width;

	int d0 = addr.d0; // depth_before padding
	int dall = d0 + tens->shape.depth;
	int nd32 = addr.nd32;
	int d32_stride = addr.d32_stride;
	unsigned x0 = addr.data[d0]; // first byten

	for (int ib = 0; ib < b; ib++)
	{
		for (int ih = 0; ih < h; ih++)
		{
			uint8_t const *p0 = addr.data + ib * addr.batch_stride + ih * addr.height_stride;
			for (int iw = 0; iw < w; iw++)
			{
				uint8_t const *p = p0 + 32 * iw;
				int da = d0;
				for (int id32 = 0; id32 < nd32; id32++)
				{
					int db = min_i32(32, dall - 32 * id32);
					for (int id = da; id < db; id++)
					{
						check = checksum_byte_add(check, p[id] ^ x0);
					}
					p += d32_stride;
					da = 0;
				}
			}
		}
	}
	return check ^ x0;
}

// report all the outputs of a node, by checksum (where len = 4, report the  value instead)
static void __attribute__((unused)) dump_node_output(struct nn_graph *nn, struct nn_node const *np);
//#define DUMP_NODE_OUTPUTS

void nn_report_node_outputs(struct nn_graph *nn, int level, struct nn_node const *np)
{
	int n_out = np->n_outputs;
	logmsg(nn, level, "Outputs for node 0x%x, type %s", np->node_id, op_type_to_string_alt(np->node_type, "??"));
	if (np->outputs == NULL || nn->debug_level < level)
		return;

#ifdef DUMP_NODE_OUTPUTS
	if (np->node_type == OP_Supernode_8x8p8to8_d32 || np->node_type == OP_InputSupernode_8x8p8to8_outd32 || np->node_type == OP_QuantizedAvgPool_8_d32 || np->node_type == OP_QuantizedMaxPool_8_d32 || np->node_type == OP_QuantizedReshape)
	{
		dump_node_output(nn, np);
	}
#endif
	// skip checksum on tensors > max_checksum (they can be very time consuming in sims)
	uint32_t max_checksum = nn_option_get(nn, debug_max_show_checksum);

	for (int i = 0; i < n_out; i++)
	{
		struct tensor const *ot = np->outputs[i];
		// common prefix
		char tmpbuf[128];
		snprintf(tmpbuf, sizeof(tmpbuf), "  %2d : (%2d,%3d,%3d,%4d)",
				 i, (int)ot->shape.batches, (int)ot->shape.height, (int)ot->shape.width, (int)ot->shape.depth);

		if (ot->data_size == 4)
		{
			logmsg(nn, level, "%s len=4   int = %d  float=%.8g",
				   tmpbuf, (int)tensor_get_int32(ot, 0), tensor_get_float(ot, 0));
		}
		else
		{
			uint64_t chk = 0;
			int do_checksum = 0;
			unsigned elsize = 0;
			char const *strd32 = "      ";
			if (ot->data_size > 0)
			{
				unsigned n = tensor_element_count(ot);
				if (tensor_is_d32(ot))
				{
					do_checksum = (n <= max_checksum);
					strd32 = " (d32)";
					if (do_checksum)
						chk = nn_checksum_tensor_d32(ot);
					elsize = 1;
				}
				else
				{
					do_checksum = (ot->data_size <= max_checksum);
					elsize = ot->data_size / n;
					if (do_checksum)
						chk = nn_checksum_tensor(ot);
				}
			}
			if (do_checksum)
			{
				logmsg(nn, level, "%s elsize=%u  %s Chk=0x%016llX", tmpbuf, elsize, strd32, (unsigned long long)chk);
			}
			else
			{
				logmsg(nn, level, "%s elsize=%u  %s", tmpbuf, elsize, strd32);
			}
		}
	}
}

static void
dump_node_output(struct nn_graph *nn, struct nn_node const *np)
{
#ifdef DUMP_NODE_OUTPUTS
	unsigned node_id = np->node_id;
	char filename[256];
	snprintf(filename, sizeof(filename), "dsp_%08X.bin", node_id);

	FILE *fout = fopen(filename, "wb");
	if (fout == NULL)
	{
		perror(filename);
		return;
	}
	struct tensor const *ot = np->outputs[0];

	printf("writing %s ...\n", filename);
	if (!tensor_is_d32(ot))
	{
		fwrite(ot->data, 1, ot->max_size, fout);
	}
	else
	{
		struct tensor_addressing tin = tensor_addressing_d32(ot);
		struct shape oshape = ot->shape;

		uint8_t const *base = tin.data;

		for (int ib = 0; ib < oshape.batches; ib++)
		{
			for (int ih = 0; ih < oshape.height; ih++)
			{
				for (int iw = 0; iw < oshape.width; iw++)
				{
					uint8_t const *rp = base + ib * tin.batch_stride + ih * tin.height_stride + iw * 32;
					int dremain = oshape.depth;
					while (dremain > 0)
					{
						int dx = min_i32(32, dremain);
						fwrite(rp, 1, dx, fout);
						rp += tin.d32_stride;
						dremain -= dx;
					}
				}
			}
		}
	}
	fclose(fout);
	snprintf(filename, sizeof(filename), "dsp_%08X.range", node_id);

	fout = fopen(filename, "wb");
	if (fout == NULL)
	{
		perror(filename);
		return;
	}
	float out_min = tensor_get_float(np->outputs[1], 0);
	float out_max = tensor_get_float(np->outputs[2], 0);
	fprintf(fout, "%.12g, %.12g\n", out_min, out_max);
	fclose(fout);
#endif
}
