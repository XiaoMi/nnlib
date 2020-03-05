
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

/*
 */
#define DONT_REDEF_ALLOC 1
#include "hexagon_nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include "graph_app.h"
#include <assert.h>

//#define RUN_UNIT_TEST_OP

#ifndef RESET_PMU
#define RESET_PMU() /* NOTHING */
#endif
#ifndef DUMP_PMU
#define DUMP_PMU() /* NOTHING */
#endif

#ifndef ENABLE_PMU
 #define ENABLE_PMU() /* NOTHING */
#endif

#ifndef DISABLE_PMU
#define DISABLE_PMU() /* NOTHING */
#endif

#ifndef SET_APP_REPORTED_STAT
#define SET_APP_REPORTED_STAT(STAT) /* NOTHING */
#endif

#ifdef __hexagon__
#ifdef USE_OS_QURT
int qtest_get_cmdline(char *, int);
#endif
#endif

#ifdef ANDROID
#if 0
#include "adspmsgd.h"
#else
#define adspmsgd_start(_a, _b, _c)
#define adspmsgd_stop()
#endif
#include "AEEStdErr.h"
#include <sys/types.h>
#include <sys/time.h>
#include "rpcmem.h" // RCPMEM_HEAP_DEFAULT?
#define ION_HEAP_ID_SYSTEM 25

void fastrpc_setup(int MCPS, int MBPS, int DCVS_DISABLE)
{
	int retVal;

	adspmsgd_start(0,RPCMEM_HEAP_DEFAULT,4096);
	rpcmem_init();
	
	if (DCVS_DISABLE) {
		retVal = hexagon_nn_disable_dcvs();
		if (retVal) printf("Failed to disable DSP DCVS (did you ever use SDK to generate a testsig?): %x!\n",retVal);
	}
	
    hexagon_nn_dcvs_type dcvs;
    if(DCVS_DISABLE)
    {
        dcvs=NN_DCVS_DISABLE;
    }
    else
    {
        dcvs=NN_DCVS_ENABLE;
    }
    retVal = hexagon_nn_set_powersave_details(NN_CORNER_TURBO, dcvs, 0);     
    printf("return value from hexagon_nn_set_powersave_details() : %d \n", retVal);  
	
}

void fastrpc_teardown()
{
	hexagon_nn_set_powersave_details(NN_CORNER_RELEASE, FALSE, 0);
	rpcmem_deinit();
	adspmsgd_stop();
}

unsigned long long GetTime(void)
{
	struct timeval tv;
	struct timezone tz;

	gettimeofday(&tv, &tz);

	return tv.tv_sec * 1000000ULL + tv.tv_usec;
}
#else

static inline void fastrpc_setup(int MCPS, int MBPS, int DCVS_DISABLE) {}
static inline void fastrpc_teardown() {}
//static inline unsigned long long GetTime() { return 0ULL; }
#define rpcmem_free(a) free((a))
#define rpcmem_alloc(a, b, c) malloc(c)

#endif


struct basicperf {
	unsigned int executions;
	double total_msecs;
	unsigned long long int total_pcycles;
};


//uint8_t test_int_data[224*224*3];
//extern float test_float_data[224*224*3];


#ifndef APP_LOOPS
#define APP_LOOPS 1
#endif

#define OUTPUT_SIZE (1024 * sizeof(float))

static void pprint_floats(float *values, int n_floats)
{
	int i;
	printf("output values:\n");
	for (i = 0; i < n_floats; i++) {
		printf("%f,",values[i]);
	}
	printf("\n");
}

static int run(uint32_t id, void *input, int elementsize, int width, int height, struct options *options, struct basicperf *basicperf, void *output, int is_last)
{
	/* TBD: output buffer */
	//void *output;
	float msecs;
	unsigned long long int pcycles = 0;
	uint32_t output_size = OUTPUT_SIZE;
	int ret;
	if (!options->benchmark) printf("Run!\n");
	/* execute */
	RESET_PMU();
        ENABLE_PMU();
	ret = graph_execute(id,output,&output_size,input,elementsize,options->depth,width,height,&msecs,&pcycles,options);
	if (is_last) {
        DISABLE_PMU();
		//DUMP_PMU();
	}
	/* Accumulate basic perf */
	basicperf->executions++;
	basicperf->total_msecs += msecs;
	basicperf->total_pcycles += pcycles;
#ifdef __hexagon__
	if (options->node_perf && !ret) print_node_perf(id);
#endif
	if (!options->benchmark) printf("output size=%d\n",(int)output_size);
	/* If option says so, pretty print float data in some way */
	if (options->pprint_floats) pprint_floats(output,output_size/sizeof(float));
	/* If option says so, pretty print imagenet data */
	if (options->pprint_labels) top5(output,output_size/sizeof(float));
	return ret;
}

static void do_a_reorder(char *data, int elementsize, int depth, const char *layer_reorder)
{
	char *tmp;
	int i;
	/* Should use alloca here, but not in stdlib? */
	if ((tmp = malloc(elementsize*depth)) == NULL) {
		printf("malloc failed?\n");
		exit(1);
	}
	for (i = 0; i < depth; i++) {
		memcpy(tmp+i*elementsize,data+elementsize*(layer_reorder[i]-'0'),elementsize);
	}
	memcpy(data,tmp,depth*elementsize);
	free(tmp);
}

/* Be able to specify something like "210" to reorder BGR into RGB */
static int do_layer_reorder(char *data, int elementsize, int depth, int area, const char *layer_reorder)
{
	int i;
	if (depth != strlen(layer_reorder)) {
		printf("bad layer reorder string length\n");
		exit(1);
	}
	for (i = 0; i < depth; i++) {
		if ((layer_reorder[i] - '0') >= depth) {
			printf("bad layer reorder string, bad depth index\n");
			exit(1);
		}
	}
	for (i = 0; i < area; i++) {
		/* reorder depth items elementsize wide */
		do_a_reorder(data,elementsize,depth,layer_reorder);
		data += depth * elementsize;
	}
	return 0;
}

// Convert a tensor of uchar into a tensor of floats
float *uint8_to_float(uint8_t *data, size_t length, float zero, float max) {

	// Allocate enough ION to hold all the new floats
	float *float_data;
	if ((float_data = rpcmem_alloc(ION_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, sizeof(float)*length)) == NULL) {
		printf("malloc failed\n");
		return NULL;
	}

	// Scale the uint8t data into floats
	for (int i=0; i<length; i++) {
		float_data[i] = (data[i] * (max-zero) / 255) + zero;
	}
	return float_data;
}

static int load_and_run(uint32_t id, const char *filename, struct options *options, struct options *assumed, struct basicperf *basicperf, unsigned long long int *appreported)
{
	FILE *f;
	int elementsize = options->elementsize;
	int depth = options->depth;
	int width = options->width;
	int height = options->height;
	const char *layer_reorder = options->layer_reorder;
	int iters = options->iters;
	int report_iters = options->report_iters;
	size_t filesize;
	int elements;
	int area;
	void *data;
	void *output;
	int i;
	int ret = -1;
	unsigned long long int lastreport = 0;
	if ((output = rpcmem_alloc(ION_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, OUTPUT_SIZE)) == NULL) {
		printf("error: malloc fail");
		return -1;
	}
	if ((f = fopen(filename,"rb")) == NULL) {
		printf("File not found: %s",filename);
		return -1;
	}
	fseek(f,0L,SEEK_END);
	filesize = ftell(f);
	fseek(f,0L,SEEK_SET);
	elements = filesize / elementsize;
	area = elements / depth;
	if (height * width == 0) {
		height = width = sqrt(area);
		assumed->height = height;
		assumed->width = width;
	}
	if ((filesize % elementsize != 0)
		|| (elements % depth != 0)
		|| (height * width != area)) {
		printf("image size %zu does not match "
			"element size %d, "
			"depth %d, "
			"width %d, "
			"height %d\n",
			filesize,
			elementsize,
			depth,
			width,
			height);
		return -1;
	}
	printf("filesize=%zu elementsize=%d height=%d width=%d depth=%d\n",
		filesize,elementsize,height,width,depth);
	if ((data = rpcmem_alloc(ION_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, filesize)) == NULL) {
		printf("malloc failed\n");
		return -1;
	}
	if (filesize != fread(data,1,filesize,f)) {
		printf("read fail\n");
		return -1;
	}
	fclose(f);
	if (layer_reorder) {
		if (do_layer_reorder(data,elementsize,depth,area,layer_reorder) != 0) {
			printf("reorder failure\n");
			return -1;
		}
	}
	if (options->input_to_float) {
		data = uint8_to_float(data,filesize,options->float_zero,options->float_max);
		elementsize = sizeof(float);
	}

	for (i = 0; i < iters; i++) {
		if ((ret = run(id,data,elementsize,width,height,options,basicperf,output,(i==iters-1))) != 0) {
			printf("run failed: %d\n",ret);
			break;
		}
		*appreported = basicperf->total_pcycles - lastreport; // only care about last ones
		if (i < iters - report_iters) {
			// Stop chopping last report_iters number of iterations from appreported
			lastreport = basicperf->total_pcycles;
		}
	}
	if (!options->benchmark) {
		rpcmem_free(output);
		rpcmem_free(data);
	}
	*appreported = *appreported / report_iters;
	return ret;
}

int main(int argc, const char **argv)
{
#ifdef __hexagon__
#ifdef USE_OS_QURT

    char *buf = malloc(1024);
    if (!buf) return -1;
    char *argvbuf = malloc(1024);
    if (!argvbuf)
    {
        free(buf);
        return -1;
    }
    // char buf[1024];  // for holding & parsing the command line
    // char argvbuf[1024];

    argv = (const char**) argvbuf;
    argc = 0;

    // system call to retrieve the command line, supported by q6 simulator.
    qtest_get_cmdline(buf, 1024);

    // 1st argv is the program being run (i.e. "fastcv_test.ext") and its path
    argv[0] = strtok(buf, " ");

    // loop to pick up the rest of the command line args from the command line
    while (NULL != (argv[++argc] = strtok(NULL, " "))) {};
#endif // #ifdef USE_OS_QURT
#endif

	int i;
	uint32_t graph_id;
	struct options options;
	struct options assumed;
	struct basicperf basicperf;
	unsigned long long int appreported = 0;

	memset(&basicperf,0,sizeof(basicperf));
	option_init(&options);
	option_init(&assumed);

	/* Give help if nothing is specified. */
	if (argc <= 1) {
		option_print_help();
		exit(1);
	}
	/* Process all flags first */
	for (i = 1; i < argc; ) {
		if (is_option_flag(argv[i])) {
			i += do_option(&options,argc,argv,i);
		} else {
			i++;
		}
	}

    // This will print to the STDOUT the shared object addresses for symbols in
    // the variable named libraries shown.  You must take those addresses and
    // subtract them from the symbols maps via objdump to yield values compatible
    // with pytem config_etm.py for the elfList=[]
	if (options.showaddress) {
            uint32_t libhexagon_addr;
            uint32_t fastrpc_shell_addr;
	    hexagon_nn_get_dsp_offset(&libhexagon_addr,&fastrpc_shell_addr);
            printf("libhexagon_nn_skel.so 0x%08x\nfastrpc_shell_3.so 0x%08x\n",
                 (unsigned int)libhexagon_addr,(unsigned int)fastrpc_shell_addr);
	}

	/* Load labels from file, if specified*/
	if (options.labels_filename) {
		if (load_labels(options.labels_filename) != 0) {
			printf("Error loading labels from file\n");
			return -1;
		}
	}

	/* Set up environment */
	fastrpc_setup(options.MCPS, options.MBPS, options.DCVS_DISABLE);
	hexagon_nn_config();
	if ((graph_id = graph_setup(options.debug)) == 0) {
		return 1;
	}

	/* Teardown and rebuild the graph as many times as desired */
	int rebuild = options.graph_rebuild;
	while (rebuild-->0) {
		printf("Teardown and rebuild graph\n");
		graph_teardown(graph_id);
		if ((graph_id = graph_setup(options.debug)) == 0) {
			return 1;
		}
	}

	for (i = 1; i < argc; ) {
		/* Skip flags */
		if (is_option_flag(argv[i])) {
			i+=2;
			continue;
		}
		/* Process each test */
		printf("Using <%s>\n",argv[i]);
		if (load_and_run(graph_id,argv[i],&options,&assumed,&basicperf, &appreported) != 0) {
			return -1;
		}
		i++;
	}

	/* Free memory allocated to labels, if specified */
	if (options.labels_filename) free_labels();
	if (!options.benchmark) {
            SET_APP_REPORTED_STAT(appreported);   
            printf("AppReported: %llu\n", appreported);
        }
	if (options.benchmark) return 0;

#ifdef BAIL_EARLY
	goto out;
#endif

	if (options.bus_bw) {
		if (options.height * options.width == 0) {
			printf("For PMU run, please specify height/width\n");
			return -1;
		}
		graph_get_a_perf(graph_id,
			options.elementsize,
			options.depth,
			options.width,
			options.height,
			0);
		graph_get_a_perf(graph_id,
			options.elementsize,
			options.depth,
			options.width,
			options.height,
			63);
		graph_get_a_perf(graph_id,
			options.elementsize,
			options.depth,
			options.width,
			options.height,
			70);
	}

	if (options.perfdump) {
		printf("%f msecs for %d iterations (%f / iter)\n",
			basicperf.total_msecs,basicperf.executions,
			basicperf.total_msecs/basicperf.executions);
		graph_perfdump(graph_id);
	}

	if (options.pmu) {
		if (options.height * options.width == 0) {
			options.height = assumed.height;
			options.width = assumed.width;
		}
		if (options.height * options.width == 0) {
			printf("For PMU run, please specify height/width\n");
			return -1;
		}
		graph_get_all_perf(graph_id,
			options.elementsize,
			options.depth,
			options.width,
			options.height);
	}

	graph_teardown(graph_id);
	fastrpc_teardown();

	goto out;  // avoid "unused label" error
 out:

	/* not all pthreads getting to H2K_thread_stop, so this hangs */
/* #ifdef H2_H */
/* 	h2_thread_stop(0); */
/* #else */
	return 0;
/* #endif */

}
