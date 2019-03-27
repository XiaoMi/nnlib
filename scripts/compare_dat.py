#!/usr/bin/python

import sys

if len(sys.argv) < 7:
    print("Compares .dat files")
    print()
    print("Currently, this script expects six .dat files, representing two quantized tensors")
    print("e.g. A,Amin,Amax,B,Bmin,Bmax")
    print("The min/max files are currently ignored")
    print("TODO - Use the min/max files to compare the true quantized value")
    print("TODO - Add floating-point mode of operation for non-quantized tensors")
    print("TODO - Consider other modes of operation, e.g. pass/fail based on threshold, and advanced statistics, e.g. variance, and advanced debug/display, e.g. heatmaps and clustering")
    print("TODO - Add ability to parse a cluster of tensors and a graph_print file")
    print("TODO - Add commandline options controlling the above")
    print()    
    print("ERROR: Expected six files for quantized comparison")
    exit(1)

a_dat = sys.argv[1]
a_min = sys.argv[2]
a_max = sys.argv[3]
b_dat = sys.argv[4]
b_min = sys.argv[5]
b_max = sys.argv[6]

index = 0
maxdiff = 0   # Highest difference observed on a single element
numdiff = 0   # Number of elements with any difference
totaldiff = 0 # Sum of all differences (sum of absolute values)
with open(a_dat, "rb") as f_a_dat:
    with open(b_dat, "rb") as f_b_dat:
        while True:
            a = f_a_dat.read(1)
            b = f_b_dat.read(1)
            if a==b'' or b == b'':
                if a!=b'' or b!=b'':
                    print("ERROR: Comparing %s and %s, they seem different-sized" % (a_dat,b_dat))
                    exit(1)
                break
            a = ord(a)
            b = ord(b)                
            if a!=b:
                print("WARN: %d != %d  (%d)" % (a,b,index))
                diff = abs(a-b)
                numdiff += 1
                totaldiff += diff
                if diff > maxdiff:
                    maxdiff = diff                    
            index+=1


print("Stats:  max=%d, num=%d, avg=%f" % (maxdiff,numdiff,totaldiff/numdiff))
