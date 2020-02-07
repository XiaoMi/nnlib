/*
 * Copyright (c) 2019-2020, The Linux Foundation. All rights reserved.
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

#ifndef HEXAGON_NN_HEXNN_SOC_DEFINES_H
#define HEXAGON_NN_HEXNN_SOC_DEFINES_H


#define V60 "libhexagon_nn_skel.so"
#define V65 "libhexagon_nn_skel_v65.so"
#define V66 "libhexagon_nn_skel_v66.so"
#define ADSP "adsp"
#define CDSP "cdsp"

typedef enum {
    UNSPECIFIED_MODE = 0,
    NON_DOMAINS,
    DOMAINS
} SkelMode;

struct SocSkelTable {
    unsigned int soc_id;
    SkelMode mode;
    const char* skel;
    const char* dsp_type;
};

const static struct SocSkelTable socSkelInfo [] = {
    {246, NON_DOMAINS, NULL, NULL},
    {291, NON_DOMAINS, NULL, NULL},
    {292, DOMAINS, V60, ADSP},
    {305, NON_DOMAINS, NULL, NULL},
    {310, NON_DOMAINS, NULL, NULL},
    {311, NON_DOMAINS, NULL, NULL},
    {312, NON_DOMAINS, NULL, NULL},
    {317, DOMAINS, V60, CDSP},
    {318, DOMAINS, V60, CDSP},
    {319, DOMAINS, V60, ADSP},
    {321, DOMAINS, V65, CDSP},
    {324, DOMAINS, V60, CDSP},
    {325, DOMAINS, V60, CDSP},
    {326, DOMAINS, V60, CDSP},
    {327, DOMAINS, V60, CDSP},
    {336, DOMAINS, V65, CDSP},
    {337, DOMAINS, V65, CDSP},
    {339, DOMAINS, V66, CDSP},
    {341, DOMAINS, V65, CDSP},
    {347, DOMAINS, V65, CDSP},
    {352, DOMAINS, V66, CDSP},
    {355, DOMAINS, V66, CDSP},
    {356, DOMAINS, V66, CDSP},
    {360, DOMAINS, V65, CDSP},
    {362, DOMAINS, V66, CDSP},
    {365, DOMAINS, V65, CDSP},
    {366, DOMAINS, V65, CDSP},
    {367, DOMAINS, V66, CDSP},
    {373, DOMAINS, V66, CDSP},
    {377, DOMAINS, V66, CDSP},
    {384, DOMAINS, V66, CDSP},
    {393, DOMAINS, V65, CDSP},
    {394, DOMAINS, V66, CDSP},
    {400, DOMAINS, V66, CDSP},
    {407, DOMAINS, V66, CDSP},
    {0, UNSPECIFIED_MODE, NULL, NULL}
};

#endif // HEXAGON_NN_HEXNN_SOC_DEFINES_H
