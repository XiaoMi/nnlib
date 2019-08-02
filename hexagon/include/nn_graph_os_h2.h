
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
#ifndef NN_GRAPH_OS_H2_H
#define NN_GRAPH_OS_H2_H 1
/*
 */

#if defined(__hexagon__)
#define RESET_PMU() __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU() __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define ENABLE_PMU() __asm__ __volatile__ (" r0 = #0x41 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DISABLE_PMU() __asm__ __volatile__ (" r0 = #0x42 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define SET_APP_REPORTED_STAT(STAT) __asm__ __volatile__ (" r0 = #0x53 ; r1 = %0 ; dccleana(r1); r3:2=memd(r1); trap0(#0); \n" : : "r"(&STAT) : "r0","r1","r2","r3","r4","r5","r6","r7","memory")

#ifndef ARCHV
#define ARCHV __HEXAGON_ARCH__
#endif

#include <h2.h>
#endif

typedef int32_t nn_futex_t;

static inline void nn_futex_wait(nn_futex_t *ptr, nn_futex_t val) { h2_futex_wait(ptr,val); }
static inline void nn_futex_wake(nn_futex_t *ptr, int howmany) { h2_futex_wake(ptr,howmany); }

struct nn_graph;

typedef pthread_t nn_thread_t;
typedef pthread_attr_t nn_thread_attr_t;



static inline int nn_thread_join(nn_thread_t id, void **retval) { return pthread_join(id,retval); }
static inline int nn_thread_attr_init(nn_thread_attr_t *attrs) { return pthread_attr_init(attrs); }
static inline int nn_thread_create(
	struct nn_graph *nn,
	nn_thread_t *tid,
	const nn_thread_attr_t *attrs,
	void *(*f)(void *),
	void *arg) 
{
	return pthread_create(tid,(nn_thread_attr_t *)attrs,f,arg);
}
static inline int nn_thread_attr_setstack(nn_thread_attr_t *attrs, void *stackaddr, size_t stacksize)
{
	if (stackaddr == NULL) return pthread_attr_setstacksize(attrs,stacksize);
	return pthread_attr_setstack(attrs,stackaddr,stacksize);
}

static inline int nn_os_get_main_thread_priority(int nn_priority) { return 0; }
static inline int nn_os_get_current_thread_priority(int *priority) { return 0; }
static inline int nn_os_set_current_thread_priority(int priority) { return 0; }

#if 0
typedef h2_sem_t nn_sem_t;
typedef h2_mutex_t nn_mutex_t;
static inline void nn_mutex_init(nn_mutex_t *mutex) { h2_mutex_init_type(mutex,H2_MUTEX_PLAIN); }
static inline void nn_mutex_lock(nn_mutex_t *mutex) {h2_mutex_lock(mutex); }
static inline void nn_mutex_unlock(nn_mutex_t *mutex) {h2_mutex_unlock(mutex); }
#define NN_MUTEX_INIT H2_MUTEX_T_INIT
static inline void nn_sem_init(nn_sem_t *sem, int val) { h2_sem_init_val(sem,val); }
static inline void nn_sem_post(nn_sem_t *sem) { h2_sem_up(sem); }
static inline void nn_sem_wait(nn_sem_t *sem) { h2_sem_down(sem); }
static inline void nn_sem_add(nn_sem_t *sem, int val) { h2_sem_add(sem,val); }
#endif

#if 0
typedef h2_pipe_t nn_pipe_t;
static inline nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements) 
{
	return h2_pipe_alloc(sizeof(h2_pipe_t)+sizeof(h2_pipe_data_t)*pipe_elements);
}
static inline void nn_pipe_free(nn_pipe_t *pipe) { h2_pipe_free(pipe); }
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val)
{
	h2_pipe_send(pipe,val);
}
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return h2_pipe_recv(pipe); }
#else
#if 0
#include "nn_graph_pipe.h"
typedef nn_portable_pipe_t nn_pipe_t;
static inline nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements) { return nn_pipe_alloc_portable(nn,pipe_elements); }
static inline void nn_pipe_free(nn_pipe_t *pipe) { nn_pipe_free_portable(pipe); }
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val) { return nn_pipe_send_portable(pipe,val); }
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return nn_pipe_recv_portable(pipe); }
#endif
#endif

static inline unsigned long long int nn_os_get_guest_pmucnt10()
{
	unsigned long long int ret;
	asm volatile (" %0 = g27:26 " : "=r"(ret));
	return ret;
}

static inline void nn_os_hvx_power_on(struct nn_graph *nn) {};
static inline void nn_os_hvx_power_off(struct nn_graph *nn) {};
static inline uint64_t nn_os_get_cycles(struct nn_graph *nn) {
	uint64_t retval = 0;
	asm volatile ("");
	retval = h2_get_core_pcycles();
	asm volatile ("");
	return retval;
}
uint64_t nn_os_get_perfcount(struct nn_graph *nn);

int nn_os_vector_acquire();
void nn_os_vector_release(int idx);
void nn_os_vector_init();
#if __HEXAGON_ARCH__ == 68
int nn_os_hmx_acquire();
void nn_os_hmx_release(int idx);
void nn_os_hmx_init();
#endif

int nn_os_workers_spawn(struct nn_graph *nn);
void nn_os_workers_kill(struct nn_graph *nn);
void nn_os_work_for_vector(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);

static inline uint64_t nn_os_get_usecs(struct nn_graph *nn)
{
	return h2_vmtrap_timerop(H2K_TIMER_TRAP_GET_TIME,0) / 1024;
}

#endif
