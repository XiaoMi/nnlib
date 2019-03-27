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

#ifdef USE_OS_H2

#include <nn_graph.h>

#define PIPESIZE_ELEMENTS 4
#define PIPESIZE_BYTES ((PIPESIZE_ELEMENTS)*8)
#define STACK_SIZE 8192


#include <h2.h>
#include <h2_vmtraps.h>
#include <h2_common_linear.h>
#include <h2_common_pmap.h>
#define H2K_GUEST_START 0x00000000

h2_vecaccess_state_t vecstate;
int vec_initted = 0;

int nn_os_vector_acquire()
{
	h2_vecaccess_ret_t ret = h2_vecaccess_acquire(&vecstate);
	return ret.idx;
}

void nn_os_vector_release(int idx)
{
	h2_vecaccess_release(&vecstate,idx);
}

void nn_os_vector_init()
{
	if (!vec_initted) {
		vec_initted = 1;
		h2_vecaccess_init(&vecstate,H2_VECACCESS_HVX_128);
	}
	
}

#if __HEXAGON_ARCH__ == 68
h2_mxaccess_state_t hmxstate;
int hmx_initted = 0;
int nn_os_hmx_acquire()
{
	return h2_mxaccess_acquire(&hmxstate);
}

void nn_os_hmx_release(int idx)
{
	h2_mxaccess_release(&hmxstate,idx);
}

void nn_os_hmx_init()
{
	if (!hmx_initted) {
		hmx_initted = 1;
		h2_mxaccess_init(&hmxstate);
	}
}
#endif

uint64_t nn_os_get_perfcount(struct nn_graph *nn)
{ 
	if (nn->perf_event == 0) return nn_os_get_cycles(nn);
	if (nn->perf_event == NN_GRAPH_PERFEVENT_UTIME) return nn_os_get_usecs(nn);
	return nn_os_get_guest_pmucnt10();
}


#if __HEXAGON_ARCH__ == 68
#define VTCM_ADDRESS 0xD8400000LL
#else
#define VTCM_ADDRESS 0xD8200000LL
#endif

void *nn_os_get_vtcm(struct nn_graph *nn, uint32_t *size)
{
	//pthread_once_t memsetup_once = PTHREAD_ONCE_INIT;
	//pthread_once(&memsetup_once,h2_mem_setup);
	return (void *)VTCM_ADDRESS;
}

extern int VTCM_User_Req;

int nn_os_vtcm_choose_size(struct nn_graph *nn)
{
	nn->vtcm_size = VTCM_User_Req;

#if defined(HEXAGON_V66) || defined(HEXAGON_V65)
	if (nn->vtcm_size==-1) {
#if __HEXAGON_ARCH__ == 68
		nn->vtcm_size = 4096*1024;
#else
		nn->vtcm_size = 256*1024;
#endif
		logmsg(nn,1,"VTCM request: %u of %u", nn->vtcm_size);
	}
#else  // V60
	if (nn->vtcm_size==-1) {
		nn->vtcm_size = 0;
	}
#endif
	return 0;
}

int nn_os_vtcm_acquire(struct nn_graph *nn) {
	nn->vtcm_ptr = (void *)VTCM_ADDRESS;
	return 0;
}

int nn_os_vtcm_release(struct nn_graph *nn) {
	nn->vtcm_ptr = NULL;
	nn->vtcm_size = 0;
        return 0;
}

#endif
