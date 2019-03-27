/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "pmu_control_linux.h"

FILE *pmuevtcfg_file;
FILE *pmuenable_file;

int pmu_init()
{
        pmuevtcfg_file = fopen("/sys/kernel/debug/pmu/pmuevtcfg", "w+");
        pmuenable_file = fopen("/sys/kernel/debug/pmu/enable", "w+");
        if (!pmuevtcfg_file || !pmuenable_file ) {
                printf("Cannot open PMU control files\n");
                //  probably need to bail out too.
                return 1;
        }
        return 0;
}

static inline uint64_t get_pcycle()
{
#if defined(__hexagon__)
        uint64_t reg;
        asm volatile ("%0=upcycle\n"
                      :"=r"(reg));
        return reg;
#endif
}


unsigned long pmu_read_file(const char *fn)
{
        FILE *file;
        unsigned long ret;

        file = fopen(fn, "r");
        if (!file) return 0llu;
        fscanf(file, "%lu", &ret);
        fclose(file);
        return ret;
}

unsigned long long pmu_read_file_llu(const char *fn)
{
#if 1
        return get_pcycle();  //--> AppReported: 49928340 inceptionv3 20180228
#else
        FILE *file;
        unsigned long long ret;

        file = fopen(fn, "r");
        if (!file) return get_pcycle();
        fscanf(file, "%llu", &ret);  //--> AppReported: 67561559 inceptionv3 20180228 (Extra HVX stall cycles???)
        fclose(file);
        return ret;
#endif
}

void pmu_write_pmuevtcfg(uint32_t val)
{
        fprintf(pmuevtcfg_file, "0x%08lx\n", (unsigned long) val);
        fflush(pmuevtcfg_file);
}

void pmu_write_enable(uint32_t val)
{
        fprintf(pmuenable_file, "0x%lx\n", (unsigned long) val);
        fflush(pmuenable_file);
}

//  This is the environment variable based version which apparently nobody likes

int pmu_start()
{
        fprintf(pmuenable_file, "0x0\n");
        fflush(pmuenable_file);
        // this implicitly resets all pmu registers
        fprintf(pmuevtcfg_file, "%s\n", getenv("PMUEVTCFG"));
        fflush(pmuevtcfg_file);
        fprintf(pmuenable_file, "0xf\n");
        fflush(pmuenable_file);
        return 0;
}

int pmu_stop()
{
        fprintf(pmuenable_file, "0x0\n");
        fflush(pmuenable_file);
        return 0;
}
