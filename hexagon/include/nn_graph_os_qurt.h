
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
#ifndef NN_GRAPH_OS_QURT_H
#define NN_GRAPH_OS_QURT_H 1

#include <qurt.h>
#include <stdlib.h>
struct nn_graph;
typedef qurt_thread_t nn_thread_t;
typedef qurt_thread_attr_t nn_thread_attr_t;
typedef uint32_t nn_futex_t;

static inline int nn_thread_join(nn_thread_t id, void **retval) {
	int tmp;
	int ret;
	ret = qurt_thread_join(id,&tmp);
	if (retval) *retval = (void *)(tmp);
	return ret;
}
static inline int nn_thread_attr_init(nn_thread_attr_t *attrs) { qurt_thread_attr_init(attrs); return 0; }
int nn_thread_create(
	struct nn_graph *nn,
	nn_thread_t *tid,
	const nn_thread_attr_t *attrs,
	void *(*f)(void *),
	void *arg);
static inline int nn_thread_attr_setstack(nn_thread_attr_t *attrs, void *stackaddr, size_t stacksize)
{
	qurt_thread_attr_set_stack_addr(attrs,stackaddr);
	qurt_thread_attr_set_stack_size(attrs,stacksize);
	return 0;
}

int nn_os_get_main_thread_priority(int nn_priority);
int nn_os_get_current_thread_priority(int *priority);
int nn_os_set_current_thread_priority(int priority);

#if 0
typedef qurt_mutex_t nn_mutex_t;
typedef qurt_sem_t nn_sem_t;
static inline void nn_mutex_init(nn_mutex_t *mutex) { memset(mutex,0,sizeof(*mutex)); qurt_mutex_init(mutex); }
static inline void nn_mutex_lock(nn_mutex_t *mutex) { qurt_mutex_lock(mutex); }
static inline void nn_mutex_unlock(nn_mutex_t *mutex) { qurt_mutex_unlock(mutex); }
#define NN_MUTEX_INIT QURT_MUTEX_INIT
static inline void nn_sem_init(nn_sem_t *sem, int val) { memset(sem,0,sizeof(*sem)); qurt_sem_init_val(sem,val); }
static inline void nn_sem_post(nn_sem_t *sem) { qurt_sem_up(sem); }
static inline void nn_sem_add(nn_sem_t *sem, int val) { qurt_sem_add(sem,val); }
static inline void nn_sem_wait(nn_sem_t *sem) { qurt_sem_down(sem); }
#endif

static inline void nn_futex_wait(nn_futex_t *ptr, nn_futex_t val) { qurt_futex_wait(ptr,val); }
static inline void nn_futex_wake(nn_futex_t *ptr, int howmany) { qurt_futex_wake(ptr,howmany); }

#if 0
typedef qurt_pipe_t nn_pipe_t;
nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements);
static inline void nn_pipe_free(nn_pipe_t *pipe) { qurt_pipe_delete(pipe); } 
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val) { qurt_pipe_send(pipe,val); }
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return qurt_pipe_receive(pipe); }
// #else
#include "nn_graph_pipe.h"
typedef nn_portable_pipe_t nn_pipe_t;
static inline nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements) { return nn_pipe_alloc_portable(nn,pipe_elements); }
static inline void nn_pipe_free(nn_pipe_t *pipe) { nn_pipe_free_portable(pipe); }
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val) { return nn_pipe_send_portable(pipe,val); }
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return nn_pipe_recv_portable(pipe); }
#endif

static inline uint64_t nn_os_get_cycles(struct nn_graph *nn) {
	uint64_t ret = 0;
	//retval = qurt_get_core_pcycles();
	asm volatile ( " %0 = c15:14 // READ UPCYCLES \n" : "=r"(ret));
	return ret;
}
int nn_os_vector_acquire();
void nn_os_vector_release(int idx);
static inline void nn_os_vector_init() {};
void nn_os_hvx_power_on(struct nn_graph *nn);
void nn_os_hvx_power_off(struct nn_graph *nn);
uint64_t nn_os_get_perfcount(struct nn_graph *nn);

int nn_os_workers_spawn(struct nn_graph *nn);
void nn_os_workers_kill(struct nn_graph *nn);
void nn_os_work_for_vector(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);

static inline uint64_t nn_os_get_usecs(struct nn_graph *nn)
{
	return qurt_timer_timetick_to_us(qurt_sysclock_get_hw_ticks());
}

#endif // NN_GRAPH_OS_H
