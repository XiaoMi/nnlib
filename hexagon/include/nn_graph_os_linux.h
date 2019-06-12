
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
#ifndef NN_GRAPH_OS_LINUX_H
#define NN_GRAPH_OS_LINUX_H 1
/*
 */

#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <pmu_control_linux.h>

#ifndef DONT_REDEF_ALLOC
#define DONT_REDEF_ALLOC 1
#endif
#include <stdlib.h>

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>

struct nn_graph;
typedef pthread_t nn_thread_t;
typedef pthread_attr_t nn_thread_attr_t;
typedef int nn_futex_t;

static inline int nn_thread_join(nn_thread_t id, void **retval) { return pthread_join(id,retval); }
static inline int nn_thread_attr_init(nn_thread_attr_t *attrs) { return pthread_attr_init(attrs); }
static inline int nn_thread_create(
	struct nn_graph *nn,
	nn_thread_t *tid,
	const nn_thread_attr_t *attrs,
	void *(*f)(void *),
	void *arg) 
{
	return pthread_create(tid,attrs,f,arg);
}
static inline int nn_thread_attr_setstack(nn_thread_attr_t *attrs, void *stackaddr, size_t stacksize)
{
	if (stackaddr == NULL) return pthread_attr_setstacksize(attrs,stacksize);
	return pthread_attr_setstack(attrs,stackaddr,stacksize);
}

static inline int nn_os_get_main_thread_priority(int nn_priority) { return 0; }
static inline int nn_os_get_current_thread_priority(int *priority) { return 0; }
static inline int nn_os_set_current_thread_priority(int priority) { return 0; }

static inline void nn_futex_wait(nn_futex_t *ptr, nn_futex_t val) { 
	//futex(ptr,FUTEX_WAIT,val,NULL,NULL,0);
	syscall(SYS_futex,ptr,FUTEX_WAIT,val,NULL,NULL,0);
}
static inline void nn_futex_wake(nn_futex_t *ptr, int howmany) {
	//futex(ptr,FUTEX_WAKE,howmany,NULL,NULL,0);
	syscall(SYS_futex,ptr,FUTEX_WAKE,howmany,NULL,NULL,0);
}

#if 0
typedef sem_t nn_sem_t;
typedef pthread_mutex_t nn_mutex_t;
static inline void nn_mutex_init(nn_mutex_t *mutex) { pthread_mutex_init(mutex,NULL); }
static inline void nn_mutex_lock(nn_mutex_t *mutex) {pthread_mutex_lock(mutex); }
static inline void nn_mutex_unlock(nn_mutex_t *mutex) {pthread_mutex_unlock(mutex); }
#define NN_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
static inline void nn_sem_init(nn_sem_t *sem, int val) { sem_init(sem,0,val); }
static inline void nn_sem_post(nn_sem_t *sem) { sem_post(sem); }
static inline void nn_sem_add(nn_sem_t *sem, int val) { for (int i=0; i < val; i++) {sem_post(sem);} }
static inline void nn_sem_wait(nn_sem_t *sem) { sem_wait(sem); }
#endif

#if 0
typedef struct nn_pipe nn_pipe_t;
#include "nn_graph_pipe.h"
static inline nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements)
{
	return nn_pipe_alloc_portable(nn,pipe_elements);
}
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val)
{
	nn_pipe_send_portable(pipe,val);
}
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return nn_pipe_recv_portable(pipe); }
static inline void nn_pipe_free(nn_pipe_t *pipe) { nn_pipe_free_portable(pipe); }
#endif

static inline int nn_os_vector_acquire(){return 0;}
static inline void nn_os_vector_release(int idx){};
static inline void nn_os_vector_init() {};
static inline void nn_os_hvx_power_on(struct nn_graph *nn) {};
static inline void nn_os_hvx_power_off(struct nn_graph *nn) {};
#if 0
static inline uint64_t nn_os_get_cycles(struct nn_graph *nn)
{
	struct timespec ts;
	asm volatile ("");
	clock_gettime(CLOCK_REALTIME, &ts);
	asm volatile ("");
	return ts.tv_nsec;
}
#endif
static inline uint64_t nn_os_get_cycles(struct nn_graph *nn)
{
	uint64_t ret;
	asm volatile ( " %0 = c15:14 // READ UPCYCLES \n" : "=r"(ret));
	return ret;
}

unsigned long long int nn_os_get_perfcount(struct nn_graph *nn);

int nn_os_workers_spawn(struct nn_graph *nn);
void nn_os_workers_kill(struct nn_graph *nn);
void nn_os_work_for_vector(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);

static inline uint64_t nn_os_get_usecs(struct nn_graph *nn)
{
	uint64_t ret;
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME,&ts);
	ret = ts.tv_sec;
	ret *= 1000*1000*1000;
	ret += ts.tv_nsec;
	return ret;
}

#endif // NN_GRAPH_OS_H
