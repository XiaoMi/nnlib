#
# Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the
# disclaimer below) provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#    * Neither the name of The Linux Foundation nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
# GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
# HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import re
import sys
import os
#
# Make hash structure and functions for converting op strings to numeric codes.
#
# input to this script is output of
#   gcc -E -I hexagon/include -DUSE_OS_H2 hexagon/ops/src/optab.c
#
HASHN = int(512/2)
FUNC_NAME= "op_type_from_string"

if len(sys.argv) != 2:
    print >> sys.stderr, "make_optab_hash.py outfile"
    sys.exit(0)


output_prefix = """
// code fragment for hashed lookup of op names -> op id
// requires string.h, stdint.h, nn_graph.h
// auto-generated from ops.def
//
// entry point:
int %(FUNC_NAME)s( char const * name );

/*
 * %(nsym)d symbols, %(unused)d empty buckets; longest chain = %(HASHL)d
 * mean probes = %(mean_probes).2f
 */
"""
output_code = """
static inline
int ophash_findhash( char const * p)
{
    const unsigned hashN= %(HASHN)d;
    const int hashK= %(HASHK)d;
    uint8_t const *pp = (uint8_t const*)p;
    unsigned h = *pp++;
    if( h != 0 ){
        unsigned c;
        while ( (c=*pp++) != 0 ){
            h = h*hashK + c;
        }
    }
    return h %% hashN;
}

//
// translate name to index; or -1 if not valid
//
int %(FUNC_NAME)s( char const * name )
{
    const int hashL = %(HASHL)d; /* longest path */

    // this is a trap to cause a compile error if the
    // size of the table isn't actually %(nsym)d, e.g. if
    // we incorrectly parsed the source
    //
    int trapvar[ (NN_OPS_MAX==%(nsym)d)?1: -1 ] __attribute__((unused));

    // hash string and look up in table.
    int hash = ophash_findhash( name );
    int k  = ophash_root_table[hash]-1;
    if( k >= 0){
        if( strcmp( hexagon_nn_op_names[k], name) == 0 )
            return k;
        for( int i = 0; i < hashL-1; i++ ){
            k = ophash_link_table[k]-1;
            if( k < 0) break;
            if( strcmp( hexagon_nn_op_names[k], name) == 0 )
                return k;
        }
    }
    return -1;
}
"""


def hashStr( s, k ):
    val = 0
    for c in s:
        val = val*k + ord(c)
    return val % HASHN

def find_bestK(names):
    bestK = -1
    bestN = 0   # of empty bins - maximize
    bestT = 1000000
    bestL = len(names)+1    # biggest bucket - minimimize
    # N = no. of non-empty buckets
    # L = max bucket occupancy
    # T = total # probes needed to lookup all 
    for k in range( 1, HASHN,2):
        counts = [0]*HASHN
        N,L,T = 0,0,0
        for nm in names:
            h = hashStr(nm,k)
            newcount = counts[h]+1
            if newcount == 1:
                N += 1
            T += newcount
            counts[h] = newcount
            L = max(L,newcount)
            if L > bestL:
                break
        if L < bestL or( L == bestL and T < bestT ):
            bestK,bestN,bestL,bestT = k,N,L,T
            if bestL == 1:
                break  # perfect...

    return bestK,bestL,bestN,bestT

def make_tables( names, k ):
    n = len(names)
    root_tab = [0]* HASHN
    link_tab = [0]* n
    for i in xrange(n-1,-1,-1):
        h = hashStr( names[i],k)
        prev = root_tab[h]
        root_tab[h] = i+1
        if prev != 0:
            link_tab[i] = prev
    return root_tab,link_tab

def format_table( varname, typename, vals, fout ):
    n = len(vals)
    print >>fout, "static const %s %s[%d]= {" % ( typename,varname, n)
    ipos = 0
    while ipos < n:
        nrow = min(n-ipos,16)
        vrow = vals[ipos:ipos+nrow]
        row =  " /* %3d */ "% ipos + ",".join("%3d"%x for x in vrow)
        ipos += nrow
        if ipos < n:
            row += ','
        print >>fout, row
    print >>fout, "};"

###############################################
indat = sys.stdin.read()
# locate all the [OP_XXX]
opnames = re.findall(r"\[\s*OP_([A-Z]\w+)\s*\]",indat)
outfile = open(sys.argv[1],"w")


hashK,hashL,usedN,totalprobe = find_bestK( opnames)
table_type = "uint8_t"
if len(opnames)> 254:
    table_type = "uint16_t"

expand_parms = { "nsym": len(opnames), 
   "mean_probes": float(totalprobe)/len(opnames),
   "HASHN": HASHN,
   "HASHK": hashK,
   "HASHL": hashL,
   "unused": HASHN-usedN,
    "FUNC_NAME": FUNC_NAME }

root_tab, link_tab  = make_tables( opnames, hashK )

print >>outfile, output_prefix % expand_parms
format_table("ophash_root_table", table_type, root_tab, outfile)
format_table("ophash_link_table", table_type, link_tab, outfile)
print >>outfile, output_code % expand_parms
