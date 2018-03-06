#define NO_PREBUILT_OPNAME_HASH 1
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * code to map 'op_type' and 'padding_type' to/from strings
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
// minimal #includes, since this is used in "graphcan"
#include "hexagon_nn_ops.h"
#include "nn_graph_ops.h"
#include "nn_string_map.h"

//
// map an op_type to a string (with no "OP_" prefix); return NULL 
// if the value is out of range
//
char const *
op_type_to_string( op_type op )
{
    int i = op;
    if( i < 0 || i >= NN_OPS_MAX) return NULL;
    return hexagon_nn_op_names[i];
}
// same, but return 'altstr' if the name is invalid
char const *
op_type_to_string_alt( op_type op, char const * altstr )
{
    int i = op;
    if( i < 0 || i >= NN_OPS_MAX) return altstr;
    return hexagon_nn_op_names[i];
}

#ifdef NO_PREBUILT_OPNAME_HASH
// 
// map string to an op_type; return -1 if not found.
// 
int op_type_from_string( char const * s )
{
    if( s== NULL || s[0] == 0 ) return -1;
    
    for(int i = 0; i  < NN_OPS_MAX; i++ )
        if( strcmp( hexagon_nn_op_names[i], s) == 0 ) return i;
    return -1;
}
#else
#include "optab_hash.i"
#endif

//
// keep in sync with defs in nn_graph_types.h
//
static const char * const padding_name_table[] = {
 "NA",
 "SAME",
 "VALID",
 "MIRROR_REFLECT",
 "MIRROR_SYMMETRIC",
 "SAME_CAFFE" 
 };
#define NUM_PAD_TYPES ((int)( sizeof(padding_name_table)/sizeof(padding_name_table[0])))


//
// map a padding_type to a string (with no "NN_PAD_" prefix); return NULL 
// if the value is out of range
//


char const *
padding_type_to_string( padding_type op )
{
    int i = op;
    if( i < 0 || i >= NUM_PAD_TYPES) return NULL;
    return padding_name_table[i];
}
// same, but return 'altstr' if the name is invalid
char const *
padding_type_to_string_alt(padding_type op, char const * altstr )
{
    int i = op;
    if( i < 0 || i >= NUM_PAD_TYPES) return altstr;
    return padding_name_table[i];
}


// map string to a padding_type; return -1 if not found.
int 
padding_type_from_string( char const * str )
{
    int n = strlen(str);    // hash by len
    switch(n){
     case 2:
        if (strcmp(str, "NA")==0) return NN_PAD_NA;
        break;
     case 4:
        if (strcmp(str, "SAME")==0) return NN_PAD_SAME;
        break;
     case 5:
        if (strcmp(str, "VALID")==0) return NN_PAD_VALID;
        break;
     case 10:
        if (strcmp(str, "SAME_CAFFE")==0) return NN_PAD_SAME_CAFFE;
        break;
     case 14:
        if (strcmp(str, "MIRROR_REFLECT")==0) return NN_PAD_MIRROR_REFLECT;
        break;
     case 16:
        if (strcmp(str, "MIRROR_SYMMETRIC")==0) return NN_PAD_MIRROR_SYMMETRIC;
        break;
     default:
        ;
    }
    return -1;
}
