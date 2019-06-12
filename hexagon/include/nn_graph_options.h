
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
#ifndef NN_GRAPH_OPTIONS_H
#define NN_GRAPH_OPTIONS_H 1
#include <stdint.h>

struct nn_graph;
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for graph 'options'; it must be included
 * at the top of nn_graph.h
 * Symbols defined in here are of the form NN_OPT_NameOfOption if they
 * identify a specific option, or NN_OPTIONS_XXXX if they are part of the
 * options mechanism.
 */
/////////// OPTIONS ///////////////////////////////
// These macros are expanded multiple time to construct different tables.
//   NN_OPTIONS_BOOLDESC( name,   "description string")
//   NN_OPTIONS_INTDESC( name,  default,  "description string")
// The 'description string" are not used.
//
// Notes:
// [1] these will only have effect if the support code is #ifdef'd in; some are only possible in
//     non-rpc environments
// [2] these are for temporary use; e.g. to compare two possible options in one build
//
#define NN_OPTIONDESCS\
		NN_OPTIONS_BOOLDESC(test_no_d32conv ,           "no d32 conversions (for e.g. unit tests)") \
		NN_OPTIONS_BOOLDESC(test_force_graph_check,     "force graph_check even when debug=0") \
		NN_OPTIONS_BOOLDESC(debug_show_output_tensors,  "log output tensor shapes after execute [1]")\
		NN_OPTIONS_BOOLDESC(debug_dump_to_binary,        "dump output tensors to binary [1]")\
		NN_OPTIONS_BOOLDESC(debug_skip_output,           "OUTPUT node is skipped")\
		NN_OPTIONS_BOOLDESC(debug_skip_check,            "Check and Close nodes are skipped")\
		NN_OPTIONS_BOOLDESC(dev_feature_A,               "generic feature switch A [2]")\
		NN_OPTIONS_BOOLDESC(dev_feature_B,               "generic feature switch B [2]")\
		NN_OPTIONS_BOOLDESC(dev_feature_C,               "generic feature switch C [2]")\
		NN_OPTIONS_BOOLDESC(dev_feature_D,               "generic feature switch D [2]")\
		NN_OPTIONS_INTDESC(debug_max_show_checksum,-1,    "don't log output checksums on tensors > this (<0 to disable)")\

//////////////////////////////////////////////////////

#define NN_OPTIONS_INT_BASE 0
#define NN_OPTIONS_BOOL_BASE 256
#define NN_OPTIONS_N_INT NN_OPTIONS__next_int
#define NN_OPTIONS_N_BOOL (NN_OPTIONS__next_bool-NN_OPTIONS_BOOL_BASE)

//
// enum which defines all the options that have INT values
// These are 0,1, ...
//
#define NN_OPTIONS_BOOLDESC(NM,...)
#define NN_OPTIONS_INTDESC(NM,...) NN_OPT_##NM,
enum {
	NN_OPTIONS__dummy_int=-1,
	NN_OPTIONDESCS
	NN_OPTIONS__next_int
};
#undef NN_OPTIONS_BOOLDESC
#undef NN_OPTIONS_INTDESC
// define all the 1-bit options

#define NN_OPTIONS_BOOLDESC(NM,...) NN_OPT_##NM,
#define NN_OPTIONS_INTDESC(NM,...)
enum {
	NN_OPTIONS__dummy_bool= NN_OPTIONS_BOOL_BASE-1,
	NN_OPTIONDESCS
	NN_OPTIONS__next_bool
};
#undef NN_OPTIONS_BOOLDESC
#undef NN_OPTIONS_INTDESC
//
// this struct is included in the nn_graph struct
//
struct nn_graph_graphopts {
	union {
	uint32_t bool_opts[ (NN_OPTIONS_N_BOOL+31)/32u];
#define NN_OPTIONS_BOOLDESC(NM,...) unsigned NM:1;
#define NN_OPTIONS_INTDESC(NM,...)
	   struct {
		NN_OPTIONDESCS
	   };
#undef NN_OPTIONS_BOOLDESC
#undef NN_OPTIONS_INTDESC
	};
	union {
	int int_opts[NN_OPTIONS_N_INT];
#define NN_OPTIONS_BOOLDESC(NM,...)
#define NN_OPTIONS_INTDESC(NM,...) int NM;
	   struct {
		NN_OPTIONDESCS
	   };
#undef NN_OPTIONS_BOOLDESC
#undef NN_OPTIONS_INTDESC
	};
};
enum {
	NN_OPTIONS_TYPE_BOOL=1,
	NN_OPTIONS_TYPE_INT,
};

// this function sets an option to an integer value
//
int nn_option_set_int(  struct nn_graph * nn, char const *name, int value );

// table of options

// setter functions
typedef int (*option_setter_fp)( struct nn_graph * nn, int code, int value );

int nn_opt_setter_bool( struct nn_graph * nn, int code, int value );
int nn_opt_setter_int( struct nn_graph * nn, int code, int value );

struct nn_option_descriptor {
	char const  *name;		// the name
	int typecode;			// NN_OPTIONS_TYPE_XX
	option_setter_fp setter_func;
	int settercode;
	int defval;
};
// this is initialized to a value containing all the defaults.
extern const struct nn_graph_graphopts Default_graphoptions;


// In the C code, the following should be used to read an option:
//
//   nn_option_get(nn,OptionName)
//
#define nn_option_get(nn,NM) ((int)((nn)->graph_options.NM))





#endif // NN_GRAPH_OPTIONS_H
