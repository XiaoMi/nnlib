
/*
 * Copyright (c) 2016, The Linux Foundation. All rights reserved.
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
/*
 */
#include "hexagon_nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "funcs.h"
#include <assert.h>


#ifdef ANDROID
#include "adspmsgd.h"
#include "dspCV.h"
#include "AEEStdErr.h"
#include <sys/types.h>
#include <sys/time.h>
#include "rpcmem.h" // RCPMEM_HEAP_DEFAULT?
void fastrpc_setup()
{
	int MCPS = 1000;
	int MBPS = 12000;
	int DCVS_DISABLE = 1;
	int retVal;

	adspmsgd_start(0,RPCMEM_HEAP_DEFAULT,4096);
	dspCV_Attribute attrib[] = {
		{DSP_TOTAL_MCPS, MCPS},
		{DSP_MCPS_PER_THREAD, MCPS / 2},
		{PEAK_BUS_BANDWIDTH_MBPS, MBPS},
		{BUS_USAGE_PERCENT, 50},
	};
	if (DCVS_DISABLE) {
		retVal = hexagon_nn_disable_dcvs();
		if (retVal) printf("Failed to disable DSP DCVS: %x!\n",retVal);
	}
	retVal = dspCV_initQ6_with_attributes(attrib,
			 sizeof(attrib) / sizeof(attrib[0]));
	printf("return value from dspCV_initQ6() : %d \n", retVal);
}

void fastrpc_teardown()
{
	adspmsgd_stop();
	dspCV_deinitQ6();
}

unsigned long long GetTime(void)
{
	struct timeval tv;
	struct timezone tz;

	gettimeofday(&tv, &tz);

	return tv.tv_sec * 1000000ULL + tv.tv_usec;
}
#else

static inline void fastrpc_setup() {}
static inline void fastrpc_teardown() {}
static inline unsigned long long GetTime() {}

#endif

int main(int argc, char *argv[])
{
	int i;
	uint32_t graph_id;
	unsigned long long int usecs = 0;
	unsigned long long int start_time;
	int APP_LOOPS = 1;

	fastrpc_setup();
	hexagon_nn_config();

	graph_id = graph_setup();

	for (i = 0; i < APP_LOOPS; i++) {
		start_time = GetTime();
		graph_execute(graph_id);
		usecs += GetTime() - start_time;
	}
	printf("%lld usecs for %d iterations (%lld / iter)\n",
		usecs,APP_LOOPS,usecs/APP_LOOPS);

	graph_perfdump(graph_id);

	if ((argc == 2) && (strcmp(argv[1],"pmu")==0)) {
		graph_get_all_perf(graph_id);
	}

	graph_teardown(graph_id);
	fastrpc_teardown();
}
