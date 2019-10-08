
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
#ifndef GRAPH_APP_H
#define GRAPH_APP_H 1

#include <stdint.h>
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for things used in our little app.
 *//* Put some common funcs here */


typedef const char * string;

struct options {
#define DEF_OPTION(NAME,TYPE,...) TYPE NAME;
#include "options.def"
#undef DEF_OPTION
};

void info_for_debug(unsigned int id, const char *name, const char *opname);
const char *info_id2name(unsigned int id);
const char *info_id2opname(unsigned int id);
uint32_t graph_setup(int debug_level);
void graph_perfdump(uint32_t nn_id);
void graph_teardown(uint32_t nn_id);
int graph_get_all_perf(
	uint32_t id,
	int elementsize,
	int depth,
	int width,
	int height);
int graph_get_a_perf(
	uint32_t id,
	int elementsize,
	int depth,
	int width,
	int height,
	int event);
int graph_execute(uint32_t id, 
	void *output, 
	uint32_t *output_size,
	const void *input,
	int elementsize,
	int depth,
	int width,
	int height,
	float *msecs_out,
	unsigned long long int *pcycles_out,
	const struct options *options);
void option_init(struct options *options);
int is_option_flag(const char *flag);
int do_option(struct options *options, int argc, const char **argv, int i);
void option_print_help();
void top5(float *data, int length);
int load_labels(const char *labels_filename);
void free_labels();

#endif
