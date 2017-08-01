
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
#ifndef NN_GRAPH_OS_H2_H
#define NN_GRAPH_OS_H2_H 1
/*
 */

#if defined(__hexagon__)
#define RESET_PMU() __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU() __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")

#include <h2.h>
#endif
struct nn_graph;
typedef h2_sem_t nn_sem_t;
typedef h2_mutex_t nn_mutex_t;
typedef struct nn_pipe nn_pipe_t;

#include "nn_graph_pipe.h"

static inline void nn_mutex_init(nn_mutex_t *mutex) { h2_mutex_init_type(mutex,H2_MUTEX_PLAIN); }
static inline void nn_mutex_lock(nn_mutex_t *mutex) {h2_mutex_lock(mutex); }
static inline void nn_mutex_unlock(nn_mutex_t *mutex) {h2_mutex_unlock(mutex); }
#define NN_MUTEX_INIT H2_MUTEX_T_INIT
static inline void nn_sem_init(nn_sem_t *sem, int val) { h2_sem_init_val(sem,val); }
static inline void nn_sem_post(nn_sem_t *sem) { h2_sem_up(sem); }
static inline void nn_sem_wait(nn_sem_t *sem) { h2_sem_down(sem); }
static inline nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements) 
{
	return nn_pipe_alloc_portable(nn,pipe_elements);
}
static inline void nn_pipe_send(nn_pipe_t *pipe, unsigned long long int val)
{
	nn_pipe_send_portable(pipe,val);
}
static inline unsigned long long int nn_pipe_recv(nn_pipe_t *pipe) { return nn_pipe_recv_portable(pipe); }
static inline void nn_os_hvx_power_on(struct nn_graph *nn) {};
static inline void nn_os_hvx_power_off(struct nn_graph *nn) {};
static inline uint64_t nn_os_get_cycles(struct nn_graph *nn) { return h2_get_core_pcycles(); }
static inline uint64_t nn_os_get_perfcount(struct nn_graph *nn) { return nn_os_get_cycles(nn); }
int nn_os_vector_acquire();
void nn_os_vector_release(int idx);
void nn_os_vector_init();

int nn_os_workers_spawn(struct nn_graph *nn);
void nn_os_workers_kill(struct nn_graph *nn);
void nn_os_work_for_vector(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);
void nn_os_work_for_scalar(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);

static inline uint64_t nn_os_get_usecs(struct nn_graph *nn)
{
	return h2_vmtrap_timerop(H2K_TIMER_TRAP_GET_TIME,0) / 1024;
}

#endif
