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

#ifdef USE_OS_QURT

#include "dspCV_hvx.h"
#include <qurt.h>
#include <nn_graph.h>

#if !defined(HEXAGON_V65) && !defined(HEXAGON_V66)
#else
#include <HAP_vtcm_mgr.h>
#endif
int nn_os_vtcm_acquire(struct nn_graph *nn)
{
#if !defined(HEXAGON_V65) && !defined(HEXAGON_V66)
	/* Just return 0 */
#else
// FIXME: query how much VTCM
#define VTCM_AMT (256*1024)
	int use_single_page = 1;
	void *ptr = HAP_request_VTCM(VTCM_AMT,use_single_page);
	if (ptr != NULL) {
		nn->vtcm_size = VTCM_AMT;
		nn->vtcm_ptr = ptr;
	}
#endif
	return 0;
}

int nn_os_vtcm_release(struct nn_graph *nn)
{
#if !defined(HEXAGON_V65) && !defined(HEXAGON_V66)
	return 0;
#else
	HAP_release_VTCM(nn->vtcm_ptr);
	nn->vtcm_ptr = NULL;
	nn->vtcm_size = 0;
#endif
	return 0;
}


int nn_os_vector_acquire()
{
	int wait_for_context = 1;
        if (dspCV_hvx_lock(DSPCV_HVX_MODE_128B, wait_for_context) < 0) {
		return 0;
	}
	return 0;
}


void nn_os_vector_release(int idx)
{
	dspCV_hvx_unlock();
}

void nn_os_hvx_power_on(struct nn_graph *nn)
{
	if (dspCV_hvx_power_on() != 0) {
		errlog(nn,"couldn't power on hvx\n");
	}
}

void nn_os_hvx_power_off(struct nn_graph *nn)
{
	dspCV_hvx_power_off();
}

nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements)
{
	qurt_pipe_attr_t pattr;
	nn_pipe_t *ret;
	const unsigned int PIPESIZE_ELEMENTS = 4;
	const unsigned int PIPESIZE_BYTES = PIPESIZE_ELEMENTS * 8;
	qurt_pipe_attr_init(&pattr);
	qurt_pipe_attr_set_buffer(&pattr,nn_malloc(PIPESIZE_BYTES));
	qurt_pipe_attr_set_elements(&pattr,PIPESIZE_ELEMENTS);
	qurt_pipe_create(&ret,&pattr);
	return ret;
}

struct qurt_startup {
	void *(*f)(void *);
	void *arg;
	nn_sem_t sem;
};

static void __attribute__((unused)) qurt_wrap(void *p)
{
	struct qurt_startup *st = p;
	void *(*f)(void *) = st->f;
	void *arg = st->arg;
	nn_sem_post(&st->sem);
	f(arg);
	qurt_thread_exit(0);
}

int nn_thread_create(
	struct nn_graph *nn,
	nn_thread_t *tid,
	const nn_thread_attr_t *attrs,
	void *(*f)(void *),
	void *arg)
{
	int ret;
	char name[16];
	unsigned int cycles = nn_os_get_cycles(nn);
	struct qurt_startup st;
	nn_thread_attr_t myattrs = *attrs;
	nn_sem_init(&st.sem,0);
	st.f = f;
	st.arg = arg;
	snprintf(name,16,"nn_%x",cycles);
	qurt_thread_attr_set_name(&myattrs,name);
	qurt_thread_attr_set_priority(&myattrs,QURT_THREAD_ATTR_PRIORITY_DEFAULT/2);
	ret = qurt_thread_create(tid,&myattrs,qurt_wrap,&st);
	if (ret != 0) return errlog(nn,"Can't create qurt thread ret=%x",ret);
	nn_sem_wait(&st.sem);
	return ret;
}

/* depending on config, get pcycles or PMU events */
unsigned long long int nn_os_get_perfcount(struct nn_graph *nn) {
	uint32_t lo;
	uint32_t hi;
	uint64_t ret;
	uint64_t lo64;
	if (nn->perf_event < NN_GRAPH_PERFEVENT_HWPMU) {
		if (nn->perf_event == 0) return qurt_get_core_pcycles();
	}
	if (nn->perf_event == NN_GRAPH_PERFEVENT_UTIME) return nn_os_get_usecs(nn);
	lo = qurt_pmu_get(QURT_PMUCNT0);
	hi = qurt_pmu_get(QURT_PMUCNT1);
	ret = (uint64_t)hi;
	ret <<= 32;
	lo64 = lo;
	ret |= lo64;	// shut up klockwork
	return ret;
}

#endif

