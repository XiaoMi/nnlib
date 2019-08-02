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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * code to map 'op_type' and 'padding_type' to/from strings
 */

#define NO_PREBUILT_OPNAME_HASH 1

#include <stdio.h>
#include <stdint.h>
#include <string.h>
// minimal #includes, since this is used in "graphcan"
#include "hexagon_nn_ops.h"
#include "nn_graph_ops.h"
#include "nn_string_map.h"

#ifdef __hexagon__
#include "nn_graph_os.h"
#endif

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


#define HASHN 256

#ifdef __hexagon__
static nn_mutex_t Opname_hash_mutex = NN_MUTEX_INIT;
#define HASH_MUTEX_LOCK() nn_mutex_lock(&Opname_hash_mutex)
#define HASH_MUTEX_UNLOCK() nn_mutex_unlock(&Opname_hash_mutex)
#else
#define HASH_MUTEX_LOCK()
#define HASH_MUTEX_UNLOCK()
#endif

// 'build-on-demand' hashing for op name->id; when a miss occurs,
//  we do a linear lookup and add to the hash.
// hash table entries are 0 for 'none'
// or op_id+1 to indicate an op (0 <= op_id < NN_OPS_MAX).
//
//  first probe: index Opname_hash_table[ hash(name) ]
//    If this is zero, hash bucket is empty; otherwise it
//     is the opid+1 of one name which has that hash.
//  If that's not the name you're looking for, try
//      Opname_hash_links[opid], and repeat the process until
//      you find a 0 or a match.
// When a miss occurs, and the linear lookup succeeds, the id+1
// is stored wherever the terminating 0 was found.
//
static uint16_t Opname_hash_table[HASHN];
static uint16_t Opname_hash_links[NN_OPS_MAX];

static int find_hash_of( char const *s)
{
	unsigned h = 0;
	int c;
	while ( (c=*s++)!= 0){
		h = 97*h + c;
	}
	return h % HASHN;
}

//
// map string to an op_type; return -1 if not found.
//
int op_type_from_string( char const * s )
{
    if( s== NULL || s[0] == 0 ) return -1;
    int hashval = find_hash_of(s);
    uint16_t * ptr = &Opname_hash_table[hashval];
    HASH_MUTEX_LOCK();
    int probes = 0;
    int result = -1;
    while(1){
    	int opid = *ptr-1;
    	if( opid <0 ) break;	// end of chain
    	if( opid >= NN_OPS_MAX						// should not happen
    		  || probes > NN_OPS_MAX ){				// should not happen
    		// throw it out and start again
    		memset(Opname_hash_table, 0,sizeof(Opname_hash_table));
    		memset(Opname_hash_links, 0,sizeof(Opname_hash_links));
    		ptr  = &Opname_hash_table[hashval];
    		break;
    	}
    	if( strcmp( hexagon_nn_op_names[opid],s)==0){
            //printf("found %s[%d]=%d after %d probes\n",s,hashval,opid,probes+1);
    		result = opid;
    		break;
    	}
    	ptr = & Opname_hash_links[opid];
    	probes++;
    }
    if( result < 0){
        //printf("missed %s[%d] after %d probes\n",s,hashval,probes+1);
        // was not found in hash - linear lookup
        // and add at the point where the miss occurred.
		for(int i = 0; i  < NN_OPS_MAX; i++ ){
			if( strcmp( hexagon_nn_op_names[i], s) == 0 ){
				*ptr = i+1;
				result = i;
				break;
			}
		}
    }
    HASH_MUTEX_UNLOCK();
    return result;
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
