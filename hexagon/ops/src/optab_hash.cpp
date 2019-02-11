
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
#include <string.h>
#include <stdint.h>

static const int HASHN = 256;   // must be a power of 2

//
// this needs -std=c++14
// for clang it needs -fconstexpr-steps=10000000
// (currently, the actual requirement is somewhere around 700K,
//  and the default seems to be 1 million - but the requirement is vaguely proportional to
//  HASHN * num_of_symbols, so some margin is good).
//
// The 'constexpr' compile time code will pick
// a 'K' for hashing the strings, and find 'L' (the largest # of
// of hashes in any bucket) and generate a hash table.
// The K and L values are compiled into the lookup function.
// This should be completely C-compatible in terms of linking.
//
//          U hexagon_nn_op_names
// 00000000 T op_type_from_string
//          U strcmp
// 00000000 r _ZL9HashTable
//
//
extern "C" {
    /*
     * entry point
     */
    int op_type_from_string( char const * name );
    extern const char *hexagon_nn_op_names[];
};

/*
 * We need to find the # of strings as a constant
 */
#define DEF_OP(NM) OP_##NM,
enum string_tags { 
#include "ops.def"
    NUM_STRINGS };
#undef DEF_OP


// we can use unsigned char for the tables, only
// if the number of strings is <= 255
// (0 is used as a null; 1..NUM_STRINGS represent the strings)
//
template <bool FITS_IN_BYTE> struct elsize_selector {
    typedef uint16_t typ;
};
template <> struct elsize_selector<true> {
    typedef uint8_t typ;
};


/* this struct defines the hash table */

struct hashtable {
    typedef elsize_selector<(NUM_STRINGS<=255)>::typ table_t;
    int hashK;      // K value used in the hash
    int hashL;      // max # of probes
    table_t root_table[HASHN];      // 'root' values
    table_t link_table[NUM_STRINGS];    // 'link' values
};


//>>>>>>>>>>> "compile time" code to make the table >>>>>>>>>>>>>>>>>>.

// compile-time 'hash' of a string for a given k. Parameter s
// can be a string constant (will be 'array[N] of char')
template <class S>
static constexpr int 
find_hash_of( S const & s, int k)
{
    unsigned h = s[0];
    for( int i = 1 ; i < (int)sizeof(S)-1; i++ )
        h = k*h+s[i];
    return h % HASHN;
}

struct hash_metrics {
    int L;      // max bucket size =   max { nb[i] } for i = 0..HASHN-1
    int P;      // total # of probes for all symbols;
                // = sum{  (nb[i]*(nb[i]+1))/2 }  for i= 0..HASHN-1
};
// evaluate a given K for L and P. 'Lcurrent' is the current best L,
// so stop if that's exceeded
//
static constexpr hash_metrics find_hash_score( const int k ,int Lcurrent )
{
#define DEF_OP(NM) find_hash_of( #NM, k),
    const int stringhashes[NUM_STRINGS]={
#include "ops.def"
    };
#undef DEF_OP
    int Lmax = 0;
    int P = 0;
    int bcount[HASHN] = {0,};
    for( int i = 0; i < NUM_STRINGS; i++ ){
        int h = stringhashes[i];
        int c = bcount[h]+1;
        P += c;
        bcount[h] = c;
        if ( c > Lmax ){
            Lmax = c;
            if ( Lmax > Lcurrent ) break;
        }
    }
    hash_metrics res = {Lmax, P};
    return res;
}

// build the hash table:
//  find the best 'k', and then fill in the table
//  This function returns the table struct.
//
static constexpr hashtable 
build_hashtable()
{
    // first, find the best K
    //  - choose K with smallest L
    //  - among equal L, choose smallest P
    //  - if we find a case with L=1, it doesn't get any better.
    //
    int bestK = 0;
    int bestL = NUM_STRINGS+1;	// all are better than this
    int bestP = 99999;
    // try all the odd numbers
    for( int k = 1; k < HASHN && bestL>1; k+=2){
        hash_metrics hm = find_hash_score( k, bestL );
        if( hm.L < bestL || (hm.L == bestL && hm.P < bestP )){
            bestK = k;
            bestL = hm.L;
            bestP = hm.P;
        }
    }
    // ok, now make the hash table - first we need a table of
    // hashes, based on bestK

#define DEF_OP(NM) find_hash_of( #NM, bestK),
    const int stringhashes[NUM_STRINGS]={
#include "ops.def"
    };
#undef DEF_OP
    hashtable res = { bestK, bestL,  };
    for( int i =0; i < HASHN; i++ ) res.root_table[i] = 0;

    // now make all the entries
    // do this backwards, so that, in case of collisions, strings closer to
    // the start of the table will have shorter lookups; on the (weak) assumption that
    // these are used more often
    for( int i = (NUM_STRINGS-1); i >= 0; i-- ){
        int h = stringhashes[i];
        int prev = res.root_table[h];
        res.root_table[h] = i+1;
        res.link_table[i] = prev;
    }

    return res;
}
/// <<<<<<<<<<<<<< end of compile time code <<<<<
//
// The table (built at compile time)
//
constexpr hashtable HashTable = build_hashtable();

/*
 * the lookup function
 */

int op_type_from_string( char const *s )
{
    unsigned h = s[0];
    if( h == 0 ) return -1;
    int c;
    while( (c=*s++) != 0 ){
        h= h*HashTable.hashK + c;
    }
    int idx = HashTable.root_table[ h % HASHN] -1;
    if( idx >= 0) {
        if ( strcmp (hexagon_nn_op_names[idx],s) == 0 )
            return idx;
        // for small L, this loop will be completely unrolled
        for( int i = 1; i < HashTable.hashL; i++ ){
            idx = HashTable.link_table[idx]-1;
            if( idx < 0)
                break;
            if(  strcmp (hexagon_nn_op_names[idx],s) == 0 ) 
                return idx;
        }
    }
    return -1;
}

