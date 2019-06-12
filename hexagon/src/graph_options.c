
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains inner workings for graph 'options'.
 */
#include <stdio.h>
#include <string.h>
#include "nn_graph.h"

static const struct nn_option_descriptor OptionDescTable[];

// Set an option via name
// If the name ends with '*', value is ignored and all options with the given
// prefix are set to default.
//
int nn_option_set_int(  struct nn_graph * nn, char const *name, int value )
{
	struct nn_option_descriptor const * descp = OptionDescTable;

	char const * star_pos = strchr(name,'*');
	if( star_pos !=NULL && star_pos[1] == '\0'){	// ends with *
		logmsg(nn,3,"setting to defaults: %s",name);
		if( star_pos == name){
			nn->graph_options = Default_graphoptions;		// set all to default
			return 0;
		}
		int any = 0;
		int preflen = star_pos - name;
		// go through the table...
		while( descp->name != 0){
			if( strncmp( descp->name, name, preflen)==0){
				(descp->setter_func)(nn, descp->settercode, descp->defval);
				any ++;
			}
			++descp;
		}
		if( any == 0) logmsg(nn,0,"no options match %s", name);
		else logmsg(nn,3, "... %d options set to default", any);
		return 0;
	}



	while( descp->name != 0 ){
		if( strcmp(descp->name,name)==0){
			logmsg(nn,3,"set %s = %d", name, value);
			return (descp->setter_func)( nn, descp->settercode, value);
		}
		++descp;
	}
	return errlog(nn, "no option %s", name);
}

int nn_opt_setter_int( struct nn_graph * nn, int code, int value )
{
	struct nn_graph_graphopts * optp = &nn->graph_options;
	code -= NN_OPTIONS_INT_BASE;
	if(code < 0 || code > NN_OPTIONS_N_INT){
		return -1;
	}
	optp->int_opts[code] = value;
	return 0;
}

int nn_opt_setter_bool( struct nn_graph * nn, int code, int value )
{
	struct nn_graph_graphopts * optp = &nn->graph_options;
	code -= NN_OPTIONS_BOOL_BASE;
	if(code < 0 || code > NN_OPTIONS_N_BOOL){
		return -1;
	}
	uint32_t *pos = &optp->bool_opts[code/32u];
	int bit = (1<< code %32u);
	if( value) *pos |= bit;
	else *pos &=  ~bit;
	return 0;
}


//////// The option descriptor table
#define NN_OPTIONS_BOOLDESC( NM, DESC) { #NM, NN_OPTIONS_TYPE_BOOL, nn_opt_setter_bool, NN_OPT_##NM, 0 },
#define NN_OPTIONS_INTDESC( NM, DEFLT,DESC) { #NM, NN_OPTIONS_TYPE_INT,  &nn_opt_setter_int, NN_OPT_##NM , DEFLT},

static const
struct nn_option_descriptor OptionDescTable[] = {
		NN_OPTIONDESCS
	{ NULL,}
};

#undef NN_OPTIONS_BOOLDESC
#undef NN_OPTIONS_INTDESC

// a struct nn_graph_graphopts
// which is init'd to the defaults
#define NN_OPTIONS_BOOLDESC(NM,DESC)
#define NN_OPTIONS_INTDESC(NM, DEFLT, DESC)[NN_OPT_##NM-NN_OPTIONS_INT_BASE] =  (DEFLT),

const struct nn_graph_graphopts
Default_graphoptions= {
		.int_opts = {
				NN_OPTIONDESCS
		}
};
#undef NN_OPTIONS_BOOLDESC
#undef NN_OPTIONS_INTDESC
