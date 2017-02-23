
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
//static inline unsigned long long GetTime() { return 0ULL; }

#endif


struct basicperf {
	unsigned int executions;
	double total_msecs;
	unsigned long long int total_pcycles;
};


//uint8_t test_int_data[224*224*3];
//extern float test_float_data[224*224*3];

#ifdef SNPE_TEST
char file_path[] = "/usr2/oguniyam/git_repo/cnn/setup/snpe-0.7.0_internal/models/googlenet/data/";
uint8_t test_int_data[224*224*3];
float test_float_data[224*224*3];

void remove_space(char *src, char *dst)
{
	// Removing space in string
	do while(isspace(*src)) src++; while((*dst++ = *src++));
}


void read_data_from_file(char *file_name)
{
	char full_path1[200], full_path[200];
	FILE * fp;
	int i, j, ret_size;
	float temp1, temp2;

	// Form full file path (relative) to read raw data
	strcpy(full_path1,file_path);
	strcat(full_path1,file_name);
	remove_space(full_path1, full_path);
	printf("File used: %s\n", full_path);

	fp = fopen(full_path, "rb");
	if (fp==NULL)
		printf("File does not exist!!!\n");

	// Read image (224x224x3) for inceptionv1 network - values are in RGBRGBRGB float format
	clearerr(fp);
	rewind(fp);
	ret_size = fread(test_float_data, 1, sizeof(test_float_data), fp);
	if(ret_size != sizeof(test_float_data))
		printf("Full data not read from file , ret_size=%d, size(array)=%d!!!\n", ret_size, sizeof(test_float_data));
	printf("fp=0x%p, Read size = %d, feof:%d, ferror:%d \n", fp, ret_size, feof(fp), ferror(fp));

	for(i=0,j=0;i<sizeof(test_int_data);i+=3,j++)
	{
		//test_float_data[i] = ((test_float_data[i]+104)/256)-1;
		//test_float_data[i+1] = ((test_float_data[i+1]+117)/256)-1;
		//test_float_data[i+2] = ((test_float_data[i+2]+123)/256)-1;

		// Input from the file is in BGR format
		// our network needs the input in RGB format
		// Convert BGR to RGB format
		// 104, 123 and 117 are the offsets needed to bring the input within +/-1
		temp1 = ((test_float_data[i]+104)/128)-1;
		temp2 = ((test_float_data[i+2]+123)/128)-1;
		test_float_data[i+2] = temp1;
		test_float_data[i+1] = ((test_float_data[i+1]+117)/128)-1;
		test_float_data[i] = temp2;

		test_int_data[i] = (uint8_t)(round((1+test_float_data[i])*128));
		test_int_data[i+1] = (uint8_t)(round((1+test_float_data[i+1])*128));
		test_int_data[i+2] = (uint8_t)(round((1+test_float_data[i+2])*128));

		//test_int_data[i] = (uint8_t) (round(test_float_data[i]));
		//test_int_data[i+1] = (uint8_t) (round(test_float_data[i+1]));
		//test_int_data[i+2] = (uint8_t) (round(test_float_data[i+2]));

		if(j<20)
			printf("Value[%d]=%f, %d\n",j, test_float_data[j], test_int_data[j]);
	}

	fclose(fp);
}
#endif

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

static int run(uint32_t id, void *input, int elementsize, int width, int height, struct options *options, struct basicperf *basicperf)
{
	/* TBD: output buffer */
	void *output;
	float msecs;
	unsigned long long int pcycles;
	uint32_t output_size = OUTPUT_SIZE;
	uint32_t sum = 0;
	const char *ctmp = input;
	int i;
	printf("Run!\n");
	for (i = 0; i < height*width*options->depth*elementsize; i++) {
		sum += *ctmp++;
	}
	printf("sum=%d\n",(int)sum);
	if ((output = malloc(OUTPUT_SIZE)) == NULL) {
		return -1;
	}
	printf("Executing!\n");
	/* execute */
	RESET_PMU();
	graph_execute(id,output,&output_size,input,elementsize,options->depth,width,height,&msecs,&pcycles);
	/* Accumulate basic perf */
	basicperf->executions++;
	basicperf->total_msecs += msecs;
	basicperf->total_pcycles += pcycles;
	printf("output size=%d\n",(int)output_size);
	/* If option says so, pretty print float data in some way */
	if (options->pprint_floats) pprint_floats(output,output_size/sizeof(float));
	/* If option says so, pretty print imagenet data */
	if (options->pprint_imagenet) imagenet_top5(output,output_size/sizeof(float));
	free(output);
	return 0;
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

static int load_and_run(uint32_t id, const char *filename, struct options *options, struct basicperf *basicperf, unsigned long long int *appreported)
{
	FILE *f;
	int elementsize = options->elementsize;
	int depth = options->depth;
	int width = options->width;
	int height = options->height;
	const char *layer_reorder = options->layer_reorder;
	int iters = options->iters;
	size_t filesize;
	int elements;
	int area;
	void *data;
	int i;
	int ret;
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
	}
	if ((filesize % elementsize != 0)
		|| (elements % depth != 0)
		|| (height * width != area)) {
		printf("image size %d does not match "
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
	printf("filesize=%d elementsize=%d height=%d width=%d depth=%d\n",
		filesize,elementsize,height,width,depth);
	if ((data = malloc(filesize)) == NULL) {
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
	for (i = 0; i < iters; i++) {
		if ((ret = run(id,data,elementsize,width,height,options,basicperf)) != 0) {
			printf("run failed: %d\n",ret);
			break;
		}
		*appreported = basicperf->total_pcycles - *appreported; // only care about last one
	}
	free(data);
	return ret;
}

int main(int argc, const char **argv)
{
	int i;
	uint32_t graph_id;
	struct options options;
	struct basicperf basicperf;
	unsigned long long int appreported = 0;

	memset(&basicperf,0,sizeof(basicperf));
	option_init(&options);

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

	/* Set up environment */
	fastrpc_setup();
	hexagon_nn_config();
	graph_id = graph_setup(options.debug);

	for (i = 1; i < argc; ) {
		/* Skip flags */
		if (is_option_flag(argv[i])) {
			i+=2;
			continue;
		}
		/* Process each test */
		printf("Using <%s>\n",argv[i]);
		if (load_and_run(graph_id,argv[i],&options,&basicperf, &appreported) != 0) {
			return -1;
		}
		i++;
	}

	DUMP_PMU();
	printf("AppReported: %lld\n", appreported);

#ifdef BAIL_EARLY
	goto out;
#endif

	if (options.perfdump) {
		printf("%f msecs for %d iterations (%f / iter)\n",
			basicperf.total_msecs,basicperf.executions,
			basicperf.total_msecs/basicperf.executions);
		graph_perfdump(graph_id);
	}

	if (options.pmu) {
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
