/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
#include "nn_graph_builtin.h"

#if defined(__hexagon__)
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

static inline uint64_t nn_atomic_add32(uint32_t volatile *p, uint32_t v)
{
	uint32_t t;
	asm volatile (	"1: %0 = memw_locked(%3)\n"
		"   %0 = add(%0,%2)\n"
		"   memw_locked(%3,p0) = %0\n"
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
static inline uint32_t nn_atomic_casu32(uint32_t volatile *p, uint32_t oldv, uint32_t newv)
{
	return (uint32_t)nn_atomic_cas32((int32_t volatile *)p,(int32_t)oldv,(int32_t)newv);
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

/////////////////////////////////////////////////////////
// This is a mechanism to efficiently convert 'jobno' to jobno % inner_count and jobno/inner_count
// in the case where jobno increases from 0 but not continuously. Eliminates use of integer-divide
// and associated function call.
//
typedef struct  {
	int inner_count;	// # of jobs/batch
	int ibatch;			//	current batch
	int ibat_x_inner;	// cache of -ibatch*inner_count
} batchslice_decode;

static inline void __attribute__((always_inline)) batchslice_decode_init( batchslice_decode * bsmp, int inner_count )
{
	bsmp->inner_count =  inner_count;
	bsmp->ibatch = bsmp->ibat_x_inner = 0;
}

// This returns i_in = jobno%inner_count, and sets ibatch = jobno/inner_count, without
// doing integer divide. It requires that 'jobno' will never decrease from call to call,
// and assumes (for performance) that ibatch is usually the same from call to call. Both
// the init and update functions should be inlined in the same function where the struct
// is declared, so the struct just becomes 3 local vars with full visibility to the optimizer.
//
static inline int __attribute__((always_inline)) batchslice_decode_update( batchslice_decode * bsmp, int jobno )
{
	int i_in = jobno + bsmp->ibat_x_inner;		// correct, if < inner_count.
	if( i_in >= bsmp->inner_count){
		do{
			bsmp->ibatch++;
			i_in -= bsmp->inner_count;
		}while( __builtin_expect( i_in >= bsmp->inner_count,0) );
		bsmp->ibat_x_inner = i_in-jobno;
	}
	return i_in;
}
//
// recommended use:
//  ===> before launching threads:
//          runstate.jobno = 0;		// volatile int
//          runstate.jobcount = inner_count * batches;
//          runstate.inner_count = inner_count;
//
//  ===> Worker thread:
//
//   worker_thread( ... ){
//      ...
//       batchslice_decode bsdecode;
//       batchslice_decode_init( &bsdecode, runstatep->inner_count );
//
//       int njobs = runstatep->njobs;	// doesn't hurt to hoist this; not mandatory
//       int ijob;			// this is the job # 0.. njobs-1
//		 // divide the jobs between this and all other threads using locked add;
//       // each loop obtains a new job #, which is decoded to ibatch and i_inner.
//
//       while(  (ijob = __sync_fetch_and_add( &rrunstatep->jobno,1)), ijob < njobs ){
//
//       	int i_inner = batchslice_decode_update( &bsdecode, ijob);
//			int ibatch = bsdecode.ibatch;
//
//          /*
//           * ***   process for i_inner of ibatch ***
//           */
//       }
//    }
//
// typical code for batchslice_decode_update
//  ( r2 = jobno, r26 = inner_count, r22 = ibatch, r27 = ibat_x_inner, r3 = result):
//
//  		{ r3 = add(r27,r2);               if (cmp.gt(r26,r3.new)) jump:t .L2  }
// .L1: 	{ r27 = sub(r27,r26);             r22 = add(r22,#1) }
//          { r3 = add(r2,r27);               if (!cmp.gt(r26,r3.new)) jump:nt .L1 }
// .L2:
//



#endif
