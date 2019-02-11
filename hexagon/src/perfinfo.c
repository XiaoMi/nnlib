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
#include <nn_graph.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code to get/reset performance info
 */

//#ifndef __hexagon__
#if defined(H2_H)
static inline void config_event_hw(struct nn_graph *nn, uint32_t event)
{
	uint32_t pmuevtcfg;
	pmuevtcfg = (event & 0xFF) | 0x0100;
	h2_pmu_setreg(H2_PMUEVTCFG,pmuevtcfg);
#if defined(HEXAGON_V65) || defined(HEXAGON_V66)
	h2_pmu_setreg(H2_PMUCFG,(event & 0x100) >> 8);
#endif
}

#elif !defined(USE_OS_QURT)
static inline void config_event_hw(struct nn_graph *nn, uint32_t event)
{
	uint32_t pmuevtcfg;
	pmuevtcfg = (event & 0xFF) | 0x0100;

	pmu_write_enable(0xf);
	pmu_write_pmuevtcfg(pmuevtcfg);
}
#else
#ifndef NO_PMU_CONFIG
#include <qurt.h>
#include <qurt_event.h>
#include "nn_graph_log.h"

static inline void config_event_hw(struct nn_graph *nn, uint32_t event)
{
    qurt_arch_version_t hvxver;
    if (QURT_EOK != qurt_sysenv_get_arch_version(&hvxver))
    {
        //FARF(HIGH, "config_event_hw error");
        // error message
        return;
    }

    hvxver.arch_version &= 0xff;
    if (hvxver.arch_version >= 0x65)
    {
        qurt_pmu_set(QURT_PMUCFG, (event & 0x100) >> 8);
    }

    uint32_t pmuevtcfg = (event & 0xFF) | 0x0100;
    qurt_pmu_set(QURT_PMUEVTCFG,pmuevtcfg);
}
#endif // NO_PMU_CONFIG
#endif

int do_perfinfo_reset(struct nn_graph *nn, uint32_t event)
{
	struct nn_node *node;
#ifdef NO_PMU_CONFIG
	// Event 5 is usecs.  Doesnt' need PMU. Other events need PMU.
	// Using PMU causes a problem and is not supposed to be used in
	// production code, so we don't allow it.
	// Using PMU interferes with sysmon and DCVS on DSP..
	if (event != 5) {
	    return -1;
	}
#endif // NO_PMU_CONFIG
	for (node = nn->head; node != NULL; node = node->next) {
		node->perfcounter = 0;
	}
	nn->perf_event = event;
#ifndef NO_PMU_CONFIG
	if (event != 0) config_event_hw(nn,event);
#endif // NO_PMU_CONFIG
	return 0;
}

int do_perfinfo_get(struct nn_graph *nn, struct perfinfo *info, uint32_t info_len)
{
	struct nn_node *node;
	uint32_t i = 0;
	for (node = nn->head; node != NULL; node = node->next) {
		if (i >= info_len) return -1;
		if( node->node_type != OP_Const){
			info[i].node_id = node->node_id;
			info[i].executions = node->executions;
			info[i].counter = node->perfcounter;
			i++;
		}
	}
	return i;
}

