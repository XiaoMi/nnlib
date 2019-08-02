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

#ifdef USE_OS_H2

#include <nn_graph.h>

#define PIPESIZE_ELEMENTS 4
#define PIPESIZE_BYTES ((PIPESIZE_ELEMENTS)*8)
#define STACK_SIZE 8192


#include <h2.h>
#include <h2_vmtraps.h>
#include <h2_common_linear.h>
#include <h2_common_pmap.h>
#include <h2_config.h>
#include <assert.h>
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
H2K_linear_fmt_t pmap[] = {
  MEMORY_MAP((((H2K_GUEST_START >> 24) + 0) << 12), URWX, L1WB_L2C, SIZE_16M, (((H2K_GUEST_START >> 24) + 0) << 12)) 
  MEMORY_MAP((((H2K_GUEST_START >> 24) + 1) << 12), URWX, L1WB_L2C, SIZE_16M, (((H2K_GUEST_START >> 24) + 1) << 12)) 
  MEMORY_MAP((((H2K_GUEST_START >> 24) + 2) << 12), URWX, L1WB_L2C, SIZE_16M, (((H2K_GUEST_START >> 24) + 2) << 12)) 
  MEMORY_MAP((((H2K_GUEST_START >> 24) + 3) << 12), URWX, L1WB_L2C, SIZE_16M, (((H2K_GUEST_START >> 24) + 3) << 12))
  MEMORY_MAP((((H2K_GUEST_START >> 24) + 4) << 12), URWX, L1WB_L2C, SIZE_16M, (((H2K_GUEST_START >> 24) + 4) << 12))
// 256KB Page
	MEMORY_MAP((((0 >> 18) + 0) << 6), URWX, L1WB_L2C, SIZE_4M, (((0 >> 18) + 0) << 6))
	{ .raw = 0 },
};
unsigned int vtcm_base;
void *nn_os_get_vtcm(struct nn_graph *nn)
{
	//pthread_once_t memsetup_once = PTHREAD_ONCE_INIT;
	//pthread_once(&memsetup_once,h2_mem_setup);
	vtcm_base = h2_info(INFO_VTCM_BASE);
#if 0        
	assert(vtcm_base != -1); // error
	assert(vtcm_base != 0);  // no vtcm

	pmap[5].ppn = vtcm_base >> 12;  // 4K page number
	pmap[5].vpn = vtcm_base >> 12;

	assert(h2_vmtrap_newmap(&pmap, H2K_ASID_TRANS_TYPE_LINEAR, 1) != -1);

	printf("vtcm_base 0x%08x\n", vtcm_base);
#endif        
        return (void *)vtcm_base;
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

int nn_os_vtcm_query_page_count(struct nn_graph *nn)
{
	unsigned int arch_page_count;
	arch_page_count = 0;

#if defined(HEXAGON_V66) || defined(HEXAGON_V65)
	arch_page_count = 1;
#endif
	return arch_page_count;
}

int nn_os_vtcm_acquire(struct nn_graph *nn) {
#if 0  
	logmsg(nn,0,"vtcm_base 0x%p\n",nn_os_get_vtcm(nn) );
        nn->vtcm_ptr = (void *)VTCM_ADDRESS;
#else
        nn->vtcm_ptr = (void *)nn_os_get_vtcm(nn);
	logmsg(nn,1,"vtcm_base 0x%p", nn->vtcm_ptr);
#endif
#if __HEXAGON_ARCH__ == 68
	nn->vtcm_size = 4096*1024;
#else
	nn->vtcm_size = 256*1024;
#endif
	return 0;
}

int nn_os_vtcm_release(struct nn_graph *nn) {
	nn->vtcm_ptr = NULL;
	nn->vtcm_size = 0;
        return 0;
}

#endif
