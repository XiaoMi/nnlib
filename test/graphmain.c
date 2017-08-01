
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
/*
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <hexagon_nn.h>
//#define DEBUG_TEST_NEW_EXECUTE
#ifdef DEBUG_TEST_NEW_EXECUTE
#include <nn_graph.h>
#endif // DEBUG_TEST_NEW_EXECUTE
#include <time.h>
#include "graph_app.h"

#define OUTPUT_ELEMENTS 1024
#define BATCHES 1
#ifdef SNPE_TEST
	#define HEIGHT 224
#else
	#define HEIGHT 299
#endif
#define WIDTH HEIGHT
#define DEPTH 3

#ifdef H2_H
#define clock_gettime clock_gettime_haha
static inline int clock_gettime_haha(int x, struct timespec *t)
{
	t->tv_sec = t->tv_nsec = 0;
	return 0;
}
#endif

float output_vals[OUTPUT_ELEMENTS] __attribute__((aligned(128)));
extern uint8_t test_int_data[];
extern float test_float_data[];


extern void init_graph(uint32_t id);

#define PRINT_BUFSIZE (2*1024*1024)

void print_log(uint32_t id)
{
#ifndef NO_VERBOSE
	unsigned char *buf;
	unsigned char *p;
	if ((buf = malloc(PRINT_BUFSIZE)) == NULL) return;
	hexagon_nn_getlog(id,buf,PRINT_BUFSIZE);
	for (p = buf; *p != 0; p++) putc(*p,stdout);
	free(buf);
#endif
}

void print_graph(uint32_t id)
{
	unsigned char *buf;
	unsigned char *p;
	if ((buf = malloc(PRINT_BUFSIZE)) == NULL) return;
	hexagon_nn_snpprint(id,buf,PRINT_BUFSIZE);
	for (p = buf; *p != 0; p++) putc(*p,stdout);
	free(buf);
}

#ifdef DEBUG_TEST_NEW_EXECUTE
static inline void tensordef_shape_set(hexagon_nn_tensordef * dst,
	unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, int dataLen)
{
	dst->batches = batches;
	dst->height = height;
	dst->width = width;
	dst->depth = depth;
	dst->dataLen = dataLen;
}
static inline void tensordef_dataptr_set(hexagon_nn_tensordef * dst, unsigned char * dataPtr)
{
	dst->data = dataPtr;
}
static inline void tensordef_shape_get(const hexagon_nn_tensordef * src,
	unsigned int * batches, unsigned int * height, unsigned int * width, unsigned int * depth, int * dataLen)
{
	*batches = src->batches;
	*height = src->height;
	*width = src->width;
	*depth = src->depth;
	*dataLen = src->dataLen;
}
#endif

uint32_t graph_setup(int debug_level)
{
	uint32_t id;
	int err;
	id = hexagon_nn_init();
	hexagon_nn_set_debug_level(id,debug_level);
	init_graph(id);
	print_graph(id);
	printf("Init graph done.");
	if ((err = hexagon_nn_prepare(id)) != 0) {
		printf("Prepare %x returned 0x%x\n",(unsigned int)id,(unsigned int)err);
	} else {
		printf("Prepare %x success!\n",(unsigned int)id);
	}
	print_log(id);
	print_graph(id);
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

int graph_execute(
	uint32_t id, 
	void *output, 
	uint32_t *output_size_ptr,
	const void *input,
	int elementsize,
	int depth,
	int width,
	int height,
	float *msecs_out,
	unsigned long long int *pcycles_out)
{
	uint32_t out_batches, out_height, out_width, out_depth;
	uint32_t out_data_size;
	/* EJP: for the short term, we'll verify HWD here and check instead of the graph failing */
	struct timespec start;
	struct timespec end;
	unsigned int secs;
	unsigned int nsecs;
	unsigned long long int pcycles;
	unsigned int cycleslo;
	unsigned int cycleshi;
	float msecs;
	int err;
	uint32_t output_size = *output_size_ptr;
	clock_gettime(CLOCK_REALTIME,&start);
	if ((err = hexagon_nn_execute(id,
		1,
		height,
		width,
		depth,
		(const uint8_t *)input,
		depth*width*height*elementsize,
		&out_batches,
		&out_height,
		&out_width,
		&out_depth,
		(uint8_t *)output,
		output_size,
		&out_data_size)) != 0) {
		clock_gettime(CLOCK_REALTIME,&end);
		printf("execute got err: %d\n",err);
		print_log(id);
		return err;
	} else {
		clock_gettime(CLOCK_REALTIME,&end);
	}
	print_log(id);
	secs = end.tv_sec - start.tv_sec;
	if (end.tv_nsec < start.tv_nsec) {
		secs -= 1;
		nsecs = 1000*1000*1000 + end.tv_nsec - start.tv_nsec;
	} else {
		nsecs = end.tv_nsec - start.tv_nsec;
	}
	hexagon_nn_last_execution_cycles(id,&cycleslo,&cycleshi);
	pcycles = cycleshi;
	pcycles <<= 32;
	pcycles |= cycleslo;
	msecs = 1000.0f*secs;
	msecs += nsecs/1.0e6f;
	printf("elapsed time: %f msecs\n",msecs);
	if (msecs_out) *msecs_out = msecs;
	if (pcycles_out) *pcycles_out = pcycles;
	*output_size_ptr = out_data_size;
	return 0;
}

#if 0
#define OPBUFSIZE 32
unsigned long long graph_execute(uint32_t id)
{
	uint32_t out_batches, out_height, out_width, out_depth;
	uint32_t out_data_size;
	int err;
	struct timespec start;
	struct timespec end;
	unsigned int secs;
	unsigned int nsecs;
	unsigned int nop_id;
	int version;
	char buf[OPBUFSIZE];
	int idx[5], i;
	double msecs;
	unsigned long long pcycles = 0;

	//printf("executing bad id: %d\n",hexagon_nn_execute(0,0,0,0,0,NULL,0,NULL,NULL,NULL,NULL,NULL,0,NULL));
	if (hexagon_nn_version(&version) != 0) {
		printf("oops: version failed?");
		return 0;
	}
	if (hexagon_nn_op_name_to_id("Nop",&nop_id) != 0) {
		printf("oops: nop name failed?");
		return 0;
	}
	if (hexagon_nn_op_id_to_name(nop_id,buf,OPBUFSIZE) != 0) {
		printf("oops: nop id failed?");
		return 0;
	}
	printf("version=%x nop_id=%x nop_name=<<%s>>\n",version,nop_id,buf);
	
	printf("Preparing to execute id=%x...\n",(unsigned int)id);
	clock_gettime(CLOCK_REALTIME,&start);
#ifdef H2_H
	RESET_PMU();
	pcycles = h2_get_core_pcycles();
#endif
#ifdef DEBUG_TEST_NEW_EXECUTE
	hexagon_nn_tensordef in_tensordef;
	hexagon_nn_tensordef out_tensordef;
	uint32_t n_inputs;
	uint32_t n_outputs;
	n_inputs = 1;
	n_outputs = 1;
	tensordef_shape_set(&in_tensordef, BATCHES, HEIGHT, WIDTH, DEPTH, HEIGHT*WIDTH*DEPTH*sizeof(uint8_t));
	tensordef_dataptr_set(&in_tensordef, (uint8_t *)test_int_data);
	tensordef_dataptr_set(&out_tensordef, (uint8_t *)output_vals);
	if ((err = hexagon_nn_execute_new(id,
		&in_tensordef,
		n_inputs,
		&out_tensordef,
		n_outputs)) != 0) {
		clock_gettime(CLOCK_REALTIME,&end);
		printf("execute got err: %d\n",err);
	} else {
		tensordef_shape_get(&out_tensordef, (unsigned int *)&out_batches, (unsigned int *)&out_height, (unsigned int *)&out_width, (unsigned int *)&out_depth, (int *)&out_data_size);
#else
	if ((err = hexagon_nn_execute(id,
		BATCHES,
		HEIGHT,
		WIDTH,
		DEPTH,
		(uint8_t *)test_int_data,
		HEIGHT*WIDTH*DEPTH*sizeof(uint8_t),
		//(uint8_t *)test_float_data,
		//HEIGHT*WIDTH*DEPTH*sizeof(float),
		&out_batches,
		&out_height,
		&out_width,
		&out_depth,
		(uint8_t *)output_vals,
		sizeof(output_vals),
		&out_data_size)) != 0) {
		clock_gettime(CLOCK_REALTIME,&end);
		printf("execute got err: %d\n",err);
	} else {
#endif
#ifdef H2_H
		pcycles = h2_get_core_pcycles() - pcycles;
#endif
		clock_gettime(CLOCK_REALTIME,&end);
		printf("%dx%dx%dx%d, size=%d\n",
			(int)out_batches,
			(int)out_height,
			(int)out_width,
			(int)out_depth,
			(int)out_data_size);

		// Top 5 indices and output_vals
		for(i=0;i<5;i++)
		{
			idx[i] = find_max_idx( output_vals,
				out_batches*out_height*out_width*out_depth);
			printf("max idx%d: %4d / %f\n", i, (int)idx[i],output_vals[idx[i]]);
			output_vals[idx[i]] = 0;
		}
		if (idx[0] == 169) {
			puts("Index 169, I think that's a");
puts("######     #    #     # ######     #");
puts("#     #   # #   ##    # #     #   # #");
puts("#     #  #   #  # #   # #     #  #   #");
puts("######  #     # #  #  # #     # #     #");
puts("#       ####### #   # # #     # #######");
puts("#       #     # #    ## #     # #     #");
puts("#       #     # #     # ######  #     #");
		} else {
			puts("Index != 169, That's not a panda, which is");
puts("#     # #     # ######  #######    #    ######     #    ######  #       #######");
puts("#     # ##    # #     # #         # #   #     #   # #   #     # #       #");
puts("#     # # #   # #     # #        #   #  #     #  #   #  #     # #       #");
puts("#     # #  #  # ######  #####   #     # ######  #     # ######  #       #####");
puts("#     # #   # # #     # #       ####### #   #   ####### #     # #       #");
puts("#     # #    ## #     # #       #     # #    #  #     # #     # #       #");
puts(" #####  #     # ######  ####### #     # #     # #     # ######  ####### #######");
		}
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

	return pcycles;
}
#endif

#define MAX_NODES (2048*8)

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

//void graph_perfdump(uint32_t id, const char *(*id2name)(uint32_t node_id))
void graph_perfdump(uint32_t id)
{
	hexagon_nn_perfinfo info[MAX_NODES];
	unsigned long long int total_cycles = 0;
	unsigned long long int cum_cycles = 0;
	unsigned long long int counter = 0;
	unsigned int n_nodes;
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
			(int)info[i].node_id,
			info_id2name(info[i].node_id),
			info_id2opname(info[i].node_id),
			(int)info[i].executions,
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

#define MAX_PERF_OUTPUT_SIZE 4096

int graph_get_all_perf(uint32_t id,
	int elementsize,
	int depth,
	int width,
	int height)
{
	unsigned int n_nodes;
	uint32_t out_b, out_h,out_w,out_d,out_data_size;
	uint64_t *counters;
	void *input;
	void *output;
	hexagon_nn_perfinfo info[MAX_NODES];
	int i,j;
	int err;
	printf("Getting all performance counter information\n");
	if ((input = calloc(1,height*width*depth*elementsize)) == NULL) {
		printf("malloc fail\n");
		return -1;
	}
	if ((counters = malloc(N_EVENTS*MAX_NODES*sizeof(uint64_t))) == NULL) {
		printf("malloc fail\n");
		return -1;
	}
	if ((output = malloc(MAX_PERF_OUTPUT_SIZE)) == NULL) {
		printf("malloc fail\n");
		return -1;
	}
	for (i = 0; i < N_EVENTS; i++) {
		hexagon_nn_reset_perfinfo(id,i);
		printf("executing for event 0x%02x...\n",i);
		if ((err=hexagon_nn_execute(id,
			1,
			height,
			width,
			depth,
			input,
			depth*width*height*elementsize,
			&out_b,
			&out_h,
			&out_w,
			&out_d,
			output,
			MAX_PERF_OUTPUT_SIZE,
			&out_data_size)) != 0) {
			printf("execute err %d\n",err);
			exit(1);
		}
		print_log(id);
		if (hexagon_nn_get_perfinfo(id,info,MAX_NODES,&n_nodes) != 0) {
			printf("perf info failure\n");
			exit(1);
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
	free(output);
	free(input);
	free(counters);
	return 0;
}

void graph_teardown(uint32_t id)
{
	hexagon_nn_teardown(id);
}

