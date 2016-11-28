
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
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <hexagon_nn.h>
#include <time.h>
#include "funcs.h"

#define OUTPUT_ELEMENTS 1024
#define BATCHES 1
//#define HEIGHT 224
#define HEIGHT 299
#define WIDTH HEIGHT
#define DEPTH 3

#ifdef H2_H
#define clock_gettime clock_gettime_haha
static inline clock_gettime_haha(int x, struct timespec *t)
{
	t->tv_sec = t->tv_nsec = 0;
}
#endif

float output_vals[OUTPUT_ELEMENTS] __attribute__((aligned(128)));
extern uint8_t test_int_data[];
extern float test_float_data[];


#define DEBUG_LEVEL 0

extern void init_graph(uint32_t id);

#define PRINT_BUFSIZE (2*1024*1024)

static inline void print_log(uint32_t id)
{
	unsigned char *buf;
	if ((buf = malloc(PRINT_BUFSIZE)) == NULL) return;
	hexagon_nn_getlog(id,buf,PRINT_BUFSIZE);
	puts((char *)buf);
	free(buf);
}

static inline void print_graph(uint32_t id)
{
	unsigned char *buf;
	if ((buf = malloc(PRINT_BUFSIZE)) == NULL) return;
	hexagon_nn_snpprint(id,buf,PRINT_BUFSIZE);
	puts((char *)buf);
	free(buf);
}

uint32_t graph_setup()
{
	uint32_t id;
	int err;
	id = hexagon_nn_init();
	hexagon_nn_set_debug_level(id,DEBUG_LEVEL);
	init_graph(id);
	print_graph(id);
	printf("Init graph done.");
	if ((err = hexagon_nn_prepare(id)) != 0) {
		printf("Prepare returned 0x%x\n",err);
	} else {
		printf("Prepare success!\n");
	}
	print_log(id);
	return id;
}


uint32_t find_max_idx(const float *data, uint32_t entries)
{
	int i;
	float maxval = data[0];
	int maxidx = 0; 
	for (i = 0; i < entries; i++) {
		if (maxval < data[i]) {
			maxval = data[i];
			maxidx = i;
		}
	}
	return maxidx;
}

void graph_execute(uint32_t id)
{
	uint32_t out_batches, out_height, out_width, out_depth;
	uint32_t out_data_size;
	int err;
	struct timespec start;
	struct timespec end;
	unsigned int secs;
	unsigned int nsecs;
	int idx;
	double msecs;
	printf("Preparing to execute...\n");
	clock_gettime(CLOCK_REALTIME,&start);
	if ((err = hexagon_nn_execute(id,
		BATCHES,
		HEIGHT,
		WIDTH,
		DEPTH,
		//(uint8_t *)test_int_data,
		(uint8_t *)test_float_data,
		HEIGHT*WIDTH*DEPTH,
		&out_batches,
		&out_height,
		&out_width,
		&out_depth,
		(uint8_t *)output_vals,
		sizeof(output_vals),
		&out_data_size)) != 0) {
		printf("execute got err: %d\n",err);
	} else {
		clock_gettime(CLOCK_REALTIME,&end);
		printf("%dx%dx%dx%d, size=%d\n",
			out_batches,
			out_height,
			out_width,
			out_depth,
			out_data_size);
		idx = find_max_idx( output_vals,
			out_batches*out_height*out_width*out_depth);
		printf("max idx: %d / %f\n", idx,output_vals[idx]);
	}
	secs = end.tv_sec - start.tv_sec;
	if (end.tv_nsec < start.tv_nsec) {
		secs -= 1;
		nsecs = 1000*1000*1000 + end.tv_nsec - start.tv_nsec;
	} else {
		nsecs = end.tv_nsec - start.tv_nsec;
	}
	msecs = 1000.0*secs;
	msecs += nsecs/1.0e6;
	printf("elapsed time: %f msecs\n",msecs);
	print_log(id);
}

#define MAX_NODES 2048

static inline unsigned long long int get_counter(hexagon_nn_perfinfo s)
{
	unsigned long long int ret;
	ret = s.counter_hi;
	ret <<= 32;
	ret |= s.counter_lo;
	return ret;
}

int cycle_sorter(const void *va, const void *vb)
{
	const hexagon_nn_perfinfo *a = va;
	const hexagon_nn_perfinfo *b = vb;
	unsigned long long int acount = get_counter(*a);
	unsigned long long int bcount = get_counter(*b);
	if (acount < bcount) return -1;
	if (acount > bcount) return 1;
	return 0;
}
int graph_get_all_perf(uint32_t id);

//void graph_perfdump(uint32_t id, const char *(*id2name)(uint32_t node_id))
void graph_perfdump(uint32_t id)
{
	hexagon_nn_perfinfo info[MAX_NODES];
	unsigned long long int total_cycles = 0;
	unsigned long long int cum_cycles = 0;
	unsigned long long int counter = 0;
	int n_nodes;
	int i;
	printf("Perf dump follows:\n");
	if (hexagon_nn_get_perfinfo(id,info,MAX_NODES,&n_nodes) != 0) {
		printf("perf info failure\n");
		return;
	}
	printf("Total %d nodes.\n",n_nodes);
	qsort(info,n_nodes,sizeof(info[0]),cycle_sorter);
	for (i = 0; i < n_nodes; i++) {
		total_cycles += get_counter(info[i]);
	}
	printf("Total %lld cycles.\n",total_cycles);
	for (i = 0; i < n_nodes; i++) {
		counter = get_counter(info[i]);
		cum_cycles += counter;
		printf("node,0x%x,%s,%s,executions,%d,cycles,%lld,%f %%,cum_cycles,%lld,%f %%\n",
			info[i].node_id,
			info_id2name(info[i].node_id),
			info_id2opname(info[i].node_id),
			info[i].executions,
			counter,
			100*((double)counter)/total_cycles,
			cum_cycles,
			100*((double)cum_cycles)/total_cycles);
	}
	//graph_get_all_perf(id);
}

#define N_EVENTS 256

const char *event_names[] = {
	"CYCLES",		//0x0
	"USERDEF1 (im2col)",
	"USERDEF2 (gemm)",
	"COMMITTED_PKT_ANY",
	"COMMITTED_PKT_BSB",
	"",
	"",
	"COMMITTED_PKT_B2B",
	"COMMITTED_PKT_SMT",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",		//0x10
	"ICACHE_DEMAND_MISS_CYCLES",
	"ICACHE_DEMAND_MISS",
	"DCACHE_DEMAND_MISS",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"ANY_IU_REPLAY",		//0x20
	"ANY_DU_REPLAY",
	"CU_EARLY_CANCEL",
	"",
	"",
	"1T_RUNNING_PKTS",
	"2T_RUNNING_PKTS",
	"3T_RUNNING_PKTS",
	"",
	"",
	"COMMITTED_INSTS",
	"COMMITTED_TC1_INSTS",
	"COMMITTED_PRIVATE_INSTS",
	"",
	"",
	"4T_RUNNING_PKTS",
	"COMMITTED_LOADS",		//0x30
	"COMMITTED_STORES",
	"COMMITTED_MEMOPS",
	"",
	"",
	"",
	"",
	"COMMITED_CONTROL_FLOW",
	"COMMITTED_PKT_CHANGED_FLOW",
	"COMMITTED_ENDLOOP",
	"",
	"1T_RUNNING_CYCLES",
	"2T_RUNNING_CYCLES",
	"3T_RUNNING_CYCLES",
	"4T_RUNNING_CYCLES",
	"",
	"AXI_READS",		//0x40
	"AXI_READ_32",
	"AXI_WRITE",
	"AXI_WRITE_32",
	"AHB_READ",
	"AHB_WRITE",
	"",
	"AXI_SLAVE_MULTI_BEAT",
	"AXI_SLAVE_SINGLE_BEAT",
	"AXI2_READ",
	"AXI2_READ_32",
	"AXI2_WRITE",
	"AXI2_WRITE_32",
	"AXI2_CONGESTION",
	"",
	"",
	"COMMITTED_FP_INSTS",		//0x50
	"BRANCH_MISPREDICT_DIRECTION",
	"BRANCH_MISPREDICT_TARGET",
	"",
	"",
	"",
	"",
	"",
	"JTLB_MISS",
	"",
	"COMMITTED_PKT_RETURN",
	"COMMITTED_PKT_INDIR_JUMP",
	"COMMITTED_BIMODAL_BRANCH",
	"BRANCH_QUEUE_FULL",
	"SMT_XU_CONFLICT",
	"",
	"",		//0x60
	"",
	"",
	"",
	"",
	"",
	"IU_LINE_FROM_LOOP_BUF",
	"",
	"IU_1_PKT_AVAIL",
	"",
	"IU_REQ_TO_L2_REPLAYED",
	"IU_PREFETCH_TO_L2",
	"ITLB_MISS",
	"IU_2_PKT_AVAIL",
	"IU_3_PKT_AVAIL",
	"IU_REQ_STALLED",
	"BIMODAL_UPDATE_DROPPED",	//0x70
	"IU_0_PKT_AVAIL",
	"IU_LINE_FALLTHROUGH",
	"IU_LEFTOVER_DROP",
	"L2FETCH_IU_ACCESS",
	"L2FETCH_IU_MISS",
	"L2_IU_ACCESS",
	"L2_IU_MISS",
	"L2_IU_PREFETCH_ACCESS",
	"L2_IU_PREFETCH_MISS",
	"",
	"",
	"L2_DU_READ_ACCESS",
	"L2_DU_READ_MISS",
	"L2FETCH_ACCESS",
	"L2FETCH_MISS",
	"L2_AXI_INTERLEAVE_DROP",		//0x80
	"L2_ACCESS",
	"L2_PIPE_CONFLICT",
	"L2_TAG_ARRAY_CONFLICT",
	"AXI_RD_CONGESTION",
	"AHB_CONGESTION",
	"",
	"TCM_DU_ACCESS",
	"TCM_DU_RD_ACCESS",
	"TCM_IU_ACCESS",
	"L2_CASTOUT",
	"L2_DU_STORE_ACCESS",
	"L2_DU_STORE_MISS",
	"L2_DU_PREFETCH_ACCESS",
	"L2_DU_PREFETCH_MISS",
	"L2_DU_RETURN_NOT_ACKED",
	"L2_DU_LOAD_2NDARY_MISS",		//0x90
	"L2FETCH_COMMAND",
	"L2FETCH_COMMAND_KIILLED",
	"L2FETCH_COMMAND_OVERWRITE",
	"",
	"",
	"",
	"L2_ACCESS_EVEN",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
	"ANY_DU_STALL",		//0xA0
	"DU_BANK_CONFLICT_REPLAY",
	"",
	"L2_FIFO_FULL_REPLAY",
	"DU_STOREBUF_FULL_REPLAY",
	"",
	"",
	"",
	"DU_FILL_REPLAY",
	"",
	"",
	"",
	"DU_READ_TO_L2",
	"DU_WRITE_TO_L2",
	"",
	"DCZERO_COMMITTED",
	"",		//0xB0
	"",
	"",
	"DTLB_MISS",
	"",
	"",
	"STOREBUF_HIT_REPLAY",
	"STOREBUF_FORCE_REPLAY",
	"",
	"SMT_BANK_CONFLICT",
	"SMT_PORT_CONFLICT",
	"",
	"",
	"PAGE_CROSS_REPLAY",
	"",
	"DU_DEMAND_2NDARY_MISS",
	"",		//0xC0
	"",
	"",
	"DCFETCH_COMMITTED",
	"DCFETCH_HIT",
	"DCFETCH_MISS",
	"",
	"",
	"DU_LOAD_UNCACHEABLE",
	"DU_DUAL_LOAD_UNCACHEABLE",
	"DU_STORE_UNCACHEABLE",
	"",
	"MISS_TO_PREFETCH",
	"",
	"AXI_LINE64_READ_REQ",
	"AXI_LINE64_WRITE_REQ",
	"AXI_WR_CONGESTION",		//0xD0
	"AHB_8_RD_REQ",
	"AXI_INCOMPLETE_WR_REQ",
	"L2FETCH_COMMAND_PAGE_TERMINATION",
	"REQ_STALL_WRITE_BUFFER_EXHAUSTION",
	"L2_DU_STORE_COALESCE",
	"",
	"",
	"",
	"",
	"",
	"",
	"REQ_STALL_EVICTION_BUFFER_EXHAUSTION",
	"AHB_MULTIBEAT_RD_REQ",
	"",
	"L2_DU_LOAD_2NDARY_MISS_ON_SW_PF",
	"",		//0xE0
	"",
	"",
	"",
	"",
	"",
	"ARCH_LOCK_PVIEW_CYCLES",
	"IU_NO_PKT_PVIEW_CYCLES",
	"IU_BRANCH_MISS_PVIEW_CYCLES",
	"DU_CACHE_MISS_PVIEW_CYCLES",
	"DU_BUSY_OTHER_PVIEW_CYCLES",
	"CU_BUSY_PVIEW_CYCLES",
	"SMT_DU_CONFLICT_PVIEW_CYCLES",
	"SMT_XU_CONFLICT_PVIEW_CYCLES",
	"",
	"COPROC_L2_STORE_TO_LOAD_MISS",
	"COPROC_PKT_THREAD",		//0xF0
	"COPROC_PKT_COUNT",
	"COPROC_POWER_THROTTLING_STALL_CYCLES",
	"COPROC_REGISTER_STALL_CYCLES",
	"COPROC_LOAD_STALL_CYCLES",
	"COPROC_STORE_STALL_CYCLES",
	"COPROC_BUSY_STALL_CYCLES",
	"COPROC_FREEZE_STALL_CYCLES",
	"COPROC_CORE_VFIFO_FULL_STALL",
	"COPROC_L2_STORE_ACCESS",
	"COPROC_L2_STORE_MISS",
	"COPROC_L2_LOAD_ACCESS",
	"COPROC_L2_LOAD_MISS",
	"COPROC_TCM_STORE_ACCESS",
	"COPROC_TCM_LOAD_ACCESS",
	"COPROC_L2_LOAD_2NDARY_MISS",
};

int graph_get_all_perf(uint32_t id)
{
	int32_t n_nodes;
	uint64_t *counters;
	hexagon_nn_perfinfo info[MAX_NODES];
	printf("Getting all performance counter information\n");
	int i,j;
	if ((counters = malloc(N_EVENTS*MAX_NODES*sizeof(uint64_t))) == NULL) {
		printf("malloc fail\n");
		return -1;
	}
	for (i = 0; i < N_EVENTS; i++) {
		hexagon_nn_reset_perfinfo(id,i);
		printf("executing for event 0x%02x...\n",i);
		graph_execute(id);
		if (hexagon_nn_get_perfinfo(id,info,MAX_NODES,&n_nodes) != 0) {
			printf("perf info failure\n");
			return -1;
		}
		for (j = 0; j < n_nodes; j++) {
			counters[i*MAX_NODES+j] = get_counter(info[j]);
		}
	}
	printf("OPNAME,NAME,");
	for (i = 0; i < N_EVENTS; i++) {
		printf("%s,",event_names[i]);
	}
	printf("\n");
	for (j = 0; j < n_nodes; j++) {
		uint32_t node_id = info[j].node_id;
		if (strcmp("?",info_id2name(node_id))==0) continue;
		printf("%s,%s,",info_id2opname(node_id),info_id2name(node_id));
		for (i = 0; i < N_EVENTS; i++) {
			printf("%lld,",counters[i*MAX_NODES+j]);
		}
		printf("\n");
	}
	free(counters);
	return 0;
}

void graph_teardown(uint32_t id)
{
	hexagon_nn_teardown(id);
}

