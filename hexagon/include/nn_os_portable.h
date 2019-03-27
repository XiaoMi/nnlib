
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
#ifndef NN_OS_PORTABLE_H
#define NN_OS_PORTABLE_H 1

#if 0
static inline uint32_t nn_atomic_cas(uint32_t *ptr, uint32_t old, uint32_t newval)
{
	return __sync_val_compare_and_swap(ptr,old,newval);
}

static inline uint32_t nn_atomic_swap(uint32_t *ptr, uint32_t newval)
{
	uint32_t old;
	do {
		old = *ptr;
		if (likely(nn_atomic_cas(ptr,old,newval) == old)) return old;
	} while (1);
}
#endif

#include <math.h>
/* trying to use isinf() standard C library call fails on SD820 with unhelpful message */
static inline int nn_isinf(double val)
{
	return ((val == INFINITY) || (val == -INFINITY));
}

typedef union {
	uint32_t raw;
	nn_futex_t as_futex;
	struct {
		uint16_t amt;
		uint16_t waiters;
	};
} nn_sem_t;


static inline void nn_sem_init(nn_sem_t *sem, int val) { (*sem).raw = val; }

/* Sem add and sem sub fastpath: nonblocking, nonwaiting case, just atomic update */

void nn_sem_add_slowpath(nn_sem_t *sem, int amt);
void nn_sem_sub_slowpath(nn_sem_t *sem, int amt);
void nn_sem_add_fastpath(nn_sem_t *sem, int amt);
void nn_sem_sub_fastpath(nn_sem_t *sem, int amt);

static void nn_sem_add(nn_sem_t *sem, int amt);
static void nn_sem_sub(nn_sem_t *sem, int amt);

static inline void nn_sem_post(nn_sem_t *sem) { nn_sem_add(sem,1); }
static inline void nn_sem_wait(nn_sem_t *sem) { nn_sem_sub(sem,1); }

typedef union {
	uint32_t raw;
	nn_futex_t as_futex;
} nn_mutex_t;

#define NN_MUTEX_INIT { .raw = 0, }

static inline void nn_mutex_init(nn_mutex_t *mutex) { mutex->raw = 0; };

/* mutex lock and unlock fastpath: 0-->1 and 1-->0 */

void nn_mutex_lock_slowpath(nn_mutex_t *mutex);
void nn_mutex_unlock_slowpath(nn_mutex_t *mutex);
void nn_mutex_lock_fastpath(nn_mutex_t *mutex);
void nn_mutex_unlock_fastpath(nn_mutex_t *mutex);

static void nn_mutex_lock(nn_mutex_t *mutex);
static void nn_mutex_unlock(nn_mutex_t *mutex);

#include "nn_graph_pipe.h"

#if 0
typedef nn_portable_pipe_t nn_pipe_t;
static inline nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements) { return nn_pipe_alloc_portable(nn,pipe_elements); }
static inline void nn_pipe_free(nn_pipe_t *pipe) { nn_pipe_free_portable(pipe); }
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val) { return nn_pipe_send_portable(pipe,val); }
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return nn_pipe_recv_portable(pipe); }
#endif

#if 0
static inline void nn_sem_add(nn_sem_t *sem, int amt) { return nn_sem_add_slowpath(sem,amt); }
static inline void nn_sem_sub(nn_sem_t *sem, int amt) { return nn_sem_sub_slowpath(sem,amt); }
static inline void nn_mutex_lock(nn_mutex_t *mutex) { return nn_mutex_lock_slowpath(mutex); }
static inline void nn_mutex_unlock(nn_mutex_t *mutex) { return nn_mutex_unlock_slowpath(mutex); }
#else
static inline void nn_sem_add(nn_sem_t *sem, int amt) { return nn_sem_add_fastpath(sem,amt); }
static inline void nn_sem_sub(nn_sem_t *sem, int amt) { return nn_sem_sub_fastpath(sem,amt); }
static inline void nn_mutex_lock(nn_mutex_t *mutex) { return nn_mutex_lock_fastpath(mutex); }
static inline void nn_mutex_unlock(nn_mutex_t *mutex) { return nn_mutex_unlock_fastpath(mutex); }
#endif

#endif
