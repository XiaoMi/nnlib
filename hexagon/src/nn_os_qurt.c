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

#ifdef USE_OS_QURT

#include <qurt.h>
#include <nn_graph.h>
#include <HAP_power.h>

#define DEFAULT_MAIN_PRIORITY   0xC0
#define MIN_MAIN_PRIORITY       0xBB
#define MAX_MAIN_PRIORITY       0xC5

// Set default worker thread priority to be lower (i.e. a higher number) than FastRPC threads, which are by default 0xC0
static const unsigned short DEFAULT_WORKER_PRIORITY = 0xD0;
#define MIN_WORKER_PRIORITY 0xC8
#define MAX_WORKER_PRIORITY 0xD8


#if !defined(HEXAGON_V65) && !defined(HEXAGON_V66)
// include nothing
#else
#include <HAP_vtcm_mgr.h>
#endif

#if !defined(V65) && !defined(V66)
static int hvxpowercontext = 0xFeedF00d;
#endif

#if defined(HEXAGON_V66)

// FIXME: when SDK updates, this will populate
void* HAP_request_async_VTCM(unsigned int size,
                             unsigned int single_page_flag,
                             unsigned int timeout_us);
#endif


#define VTCM_AMT (256*1024)
int nn_os_vtcm_fake_acquire(struct nn_graph *nn)
{
	logmsg(nn,1,"couldn't acquire VTCM");
	if (unlikely(nn->fake_vtcm_ptr == NULL)) {
		if ((nn->fake_vtcm_ptr = nn_memalign(128,VTCM_AMT)) == NULL) {
			return errlog(nn,"Oops, can't memalign fake VTCM");
		}
	}
	nn->vtcm_ptr = nn->fake_vtcm_ptr;
	nn->vtcm_size = VTCM_AMT;
	return 0;
}
extern int VTCM_User_Req;


// NOTE: This function sets nn->vtcm_size under the assumption that it will be later allocated before use
//  Violation of this assumption may cause bugs elsewhere.
int nn_os_vtcm_choose_size(struct nn_graph *nn)
{
	nn->vtcm_size = VTCM_User_Req;

#if defined(HEXAGON_V66) || defined(HEXAGON_V65)
	if (nn->vtcm_size==-1) {
		// Query available VTCM, and warn if we're getting less than expected.
		unsigned int avail_block_size = 0, max_page_size = 0, num_pages = 0, arch_page_size = 0, arch_page_count = 0;
		if (HAP_query_avail_VTCM(&avail_block_size, &max_page_size, &num_pages)) {
			// Should this be fatal?
			errlog(nn,"ERROR: Could not query available VTCM from Qurt");
		}
		if (HAP_query_total_VTCM(&arch_page_size, &arch_page_count)) {
			// Should this be fatal?
			errlog(nn,"ERROR: Could not query VTCM architecture from Qurt");
		}
		if (arch_page_count != 1) {
			logmsg(nn,1,"WARN: Architectural VTCM page-count %u!=1 (%u,%u,%u,%u)", arch_page_count, avail_block_size, max_page_size, num_pages, arch_page_size);
		}
		if (arch_page_size != max_page_size) {
			logmsg(nn,1, "WARN: Max VTCM page available is less than architectural.  Maybe other users? (%u < %u)", max_page_size, arch_page_size);
		}
		logmsg(nn,1,"VTCM request: %u of %u", max_page_size, arch_page_size*arch_page_count);
		nn->vtcm_size = max_page_size;
	}
#else  // V60
	if (nn->vtcm_size==-1) {
		nn->vtcm_size = 0;
	}
#endif
	return 0;
}

int nn_os_vtcm_query_page_count(struct nn_graph *nn)
{
	unsigned int arch_page_size, arch_page_count;
	arch_page_size = 0;
	arch_page_count = 0;

#if defined(HEXAGON_V66) || defined(HEXAGON_V65)
	// Query current VTCM.
	if (HAP_query_total_VTCM(&arch_page_size, &arch_page_count)) {
		// Should this be fatal?
		logmsg(nn,1,"ERROR: Could not query VTCM architecture from Qurt");
	}
#endif
	return arch_page_count;
}

int nn_os_vtcm_acquire(struct nn_graph *nn)
{
	int use_single_page = 1;

	nn_os_vtcm_choose_size(nn);

	if (nn->vtcm_size == 0) {
#if defined(HEXAGON_V66)
		return errlog(nn, "ERROR: NO VTCM. Required for V66");
#else
		return nn_os_vtcm_fake_acquire(nn);
#endif
	}

#if defined(HEXAGON_V66)
    void *ptr;
    qurt_arch_version_t av;
    qurt_sysenv_get_arch_version(&av);
    if(av.arch_version == 0x00002866) { //check if running on ViperTooth
        logmsg(nn, 2, "Runing on QCS405, will not use asynchronous VTCM request");
        ptr = HAP_request_VTCM(nn->vtcm_size, 0);
    }
    else {
	    unsigned int timeout_us = 500*1000;
	    ptr = HAP_request_async_VTCM(nn->vtcm_size, use_single_page, timeout_us);
    }
#elif defined(HEXAGON_V65)
	void *ptr = HAP_request_VTCM(nn->vtcm_size,use_single_page);
#else  // V60, etc.
	void *ptr = NULL;
	(void)(use_single_page);
#endif
	if (ptr != NULL) {
		nn->vtcm_ptr = ptr;
	} else {
 #if defined(HEXAGON_V66)
 		// fake vtcm is not allowed on V66
 		return errlog(nn, "Could not acquire VTCM. Required for V66");
 #else
		return nn_os_vtcm_fake_acquire(nn);
#endif
	}

	return 0;
}



int nn_os_vtcm_release(struct nn_graph *nn)
{
#if !defined(HEXAGON_V65) && !defined(HEXAGON_V66)
	return 0;
#else
	/* If we really have VTCM, release it. */
	if (nn->vtcm_ptr && (nn->vtcm_ptr != nn->fake_vtcm_ptr)) {
		HAP_release_VTCM(nn->vtcm_ptr);
	}
	nn->vtcm_ptr = NULL;
	nn->vtcm_size = 0;
#endif
	return 0;
}

int nn_os_vector_acquire()
{
    int ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (ret != 0) return errlog(NULL,"Can't lock HVX context ret=%x",ret);
    return 0;
}

void nn_os_vector_release(int idx)
{
    if (qurt_hvx_unlock() != 0) {
        errlog(NULL,"couldn't unlock hvx\n");
    }
}

/*
 * This, and the power off counterpart, MUST be called while the
 * graph mutex is locked.
 *
 * Both execute() and prepare() lock the mutex before doing work.
 *
 * This code will do NOTHING on V65 and V66 for now.  The assumption
 * is that on CDSPs, HVX is powered on for us by the RPC system.
 * V65 and V66 deployment is only on CDSPs right now.
 *
 * The assumption is also that V60 is MAINLY ADSP.  This is true except
 * for SDM660. SDM660 is a CDSP so the infrastructure will power on HVX for us,
 * and this *should* have no effect if we vote it on/off anyway.
 *
 * If V65 or V66 end up with us running on HVX on an ADSP, we may
 * need to do this differently if the infrastructure won't power it on by
 * default for power reasons (ADSPs are often in low power islands
 * and we shouldn't leave a potentially unused hw block turned on
 * in a low power island)
 *
 */
void nn_os_hvx_power_on(struct nn_graph *nn)
{
#if !defined(V65) && !defined(V66)
    HAP_power_request_t request;
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    int ret = HAP_power_set((void*)&hvxpowercontext, &request);
	if (ret != 0) {
		errlog(nn,"couldn't power on hvx ret=%x\n",ret);
	}
#endif
}

void nn_os_hvx_power_off(struct nn_graph *nn)
{
#if !defined(V65) && !defined(V66)
    HAP_power_request_t request;
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = FALSE;
    int ret = HAP_power_set((void*)&hvxpowercontext, &request);
    if (ret != 0) {
        errlog(nn,"Graph %d couldn't power off hvx ret=%x\n",nn->id, ret);
    }
#endif
}

#if 0
nn_pipe_t *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements)
{
	qurt_pipe_attr_t pattr;
	nn_pipe_t *ret;
	const unsigned int PIPESIZE_ELEMENTS = 128;
	const unsigned int PIPESIZE_BYTES = PIPESIZE_ELEMENTS * 8;
	qurt_pipe_attr_init(&pattr);
	qurt_pipe_attr_set_buffer(&pattr,nn_malloc(PIPESIZE_BYTES));
	qurt_pipe_attr_set_elements(&pattr,PIPESIZE_ELEMENTS);
	qurt_pipe_create(&ret,&pattr);
	return ret;
}
#endif

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

static unsigned short get_qurt_worker_priority(int nn_priority) {
	unsigned short os_priority = (unsigned short) (((int) DEFAULT_WORKER_PRIORITY) + nn_priority);
	// cap priority to safe ranges, lower value is higher priority
	if (os_priority < MIN_WORKER_PRIORITY) os_priority = MIN_WORKER_PRIORITY;
	else if (os_priority > MAX_WORKER_PRIORITY) os_priority = MAX_WORKER_PRIORITY;
	return os_priority;
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
	qurt_thread_attr_set_priority(&myattrs, get_qurt_worker_priority(nn->priority));
	ret = qurt_thread_create(tid,&myattrs,qurt_wrap,&st);
	if (ret != 0) return errlog(nn,"Can't create qurt thread ret=%x",ret);
	nn_sem_wait(&st.sem);
	return ret;
}

int nn_os_get_main_thread_priority(int nn_priority) {
	int os_priority = ((int) DEFAULT_MAIN_PRIORITY) + nn_priority;
	// cap priority to safe ranges, lower value is higher priority
	if (os_priority < MIN_MAIN_PRIORITY) os_priority = MIN_MAIN_PRIORITY;
	else if (os_priority > MAX_MAIN_PRIORITY) os_priority = MAX_MAIN_PRIORITY;
	return os_priority;
}

int nn_os_get_current_thread_priority(int *priority) {
	qurt_thread_t id = qurt_thread_get_id();
	*priority = qurt_thread_get_priority(id);
	return 0;
}
int nn_os_set_current_thread_priority(int priority) {
	qurt_thread_t id = qurt_thread_get_id();
	return qurt_thread_set_priority(id, (unsigned short) priority);
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

