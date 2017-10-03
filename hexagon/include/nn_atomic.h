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
#ifndef NN_ATOMIC_H
#define NN_ATOMIC_H 1

#include <stdint.h>

#ifdef __hexagon__
static inline uint64_t nn_atomic_add64(uint64_t volatile *p, uint64_t v)
{
	uint64_t t;
	asm volatile (	"1: %0 = memd_locked(%3)\n"
		"   %0 = add(%0,%2)\n"
		"   memd_locked(%3,p0) = %0\n"
		"   if (!p0) jump:nt 1b\n"
		:"=&r"(t),"+m"(*p)
		:"r"(v),"r"(p)
		:"p0");
	return t;
}

/* Compare and swap
 * Locked version of:
 *    int32 val = *p;
 *     if ( val == oldv ){
 *         *p = newv;
 *     }
 *     return val;
 *  The operation succeeded if the return value is the same as parameter
 *  oldv. when the return value != oldv, the store was not done, and the
 *  return value can be used as the next 'oldv'.
 */
static inline int32_t nn_atomic_cas32(int32_t volatile *p, int32_t oldv, int32_t newv)
{
	int32_t t;
	asm volatile (	"1: %0 = memw_locked(%4)\n"
		"   { p0 = cmp.eq(%0,%2); if (!p0.new) jump:nt 2f} \n"
		"    memw_locked(%4,p0) = %3 \n"
		"   if (!p0) jump:nt 1b\n"
		"2:\n"
		:"=&r"(t),"+m"(*p)
		:"r"(oldv),"r"(newv),"r"(p)
		:"p0");
	return t;
}

static inline void nn_atomic_min(int32_t volatile *p, int32_t newmin)
{
	int32_t t;
	asm volatile (
		"1: %0 = memw_locked(%3)\n"
		" %0 = min(%0,%2)\n"
		" memw_locked(%3,p0) = %0\n"
		" if (!p0) jump:nt 1b\n"
		:"=&r"(t),"+m"(*p)
		:"r"(newmin),"r"(p)
		:"p0");
}

static inline void nn_atomic_max(int32_t volatile *p, int32_t newmax)
{
	int32_t t;
	asm volatile (
		"1: %0 = memw_locked(%3)\n"
		" %0 = max(%0,%2)\n"
		" memw_locked(%3,p0) = %0\n"
		" if (!p0) jump:nt 1b\n"
		:"=&r"(t),"+m"(*p)
		:"r"(newmax),"r"(p)
		:"p0");
}
#else // non-hexagon place holders
// these should work on hexagon,too, assuming __sync_xx  are correctly implemented
// in the compiler
static inline uint64_t nn_atomic_add64(uint64_t volatile *p, uint64_t v)
{
	return __sync_add_and_fetch( p, v);
}
static inline int32_t nn_atomic_cas32(int32_t volatile *p, int32_t oldv, int32_t newv)
{
	return __sync_val_compare_and_swap( p, oldv, newv);
}
static inline void nn_atomic_min(int32_t volatile *p, int32_t newmin)
{
	int32_t oldmin = *p;
	while (newmin < oldmin){
		int32_t oldmin2 = nn_atomic_cas32(p,oldmin,newmin);
		if( oldmin2 == oldmin) break;	// succeeded..
		oldmin = oldmin2;
	}
}
static inline void nn_atomic_max(int32_t volatile *p, int32_t newmax)
{
	int32_t oldmax = *p;
	while (newmax > oldmax){
		int32_t oldmax2 = nn_atomic_cas32(p,oldmax,newmax);
		if( oldmax2 == oldmax) break;	// succeeded..
		oldmax = oldmax2;
	}
}
#endif



#endif
