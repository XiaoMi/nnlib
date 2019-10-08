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

#ifndef HEXAGON_NN_HEXNN_SOC_DEFINES_H
#define HEXAGON_NN_HEXNN_SOC_DEFINES_H

typedef enum
{
    UNKNOWN_SOC = 0,
    SD855,
    SD845,
    SD835,
    SD820,
    SD710,
    SD670,
    SD660,
    SD6150,
    SD7150,
    QCS405,
    SD6125,
    QCS403
} soc_model;


#define IS_QTI_SOC_8996(soc) ((soc) == 246 || (soc) == 291 || \
                              (soc) == 310 || (soc) == 311 || \
                              (soc) == 305 || (soc) == 312)

#define IS_QTI_SOC_8998(soc) ((soc) == 292 || (soc) == 319)

#define IS_QTI_SOC_SDM660(soc) ((soc) == 317 || (soc) == 324 || \
                                       (soc) == 325 || (soc) == 326 || \
                                       (soc) == 318 || (soc) == 327)

#define IS_QTI_SOC_SDM845(soc) ((soc) == 321 || (soc) == 341)

#define IS_QTI_SOC_SDM670(soc) ((soc) == 336 || (soc) == 337 || (soc) == 347)

#define IS_QTI_SOC_SDM855(soc) ((soc) == 339 || (soc) == 362 || (soc) == 367)

#define IS_QTI_SOC_SDM710(soc) ((soc) == 360 || (soc) == 393)

#define IS_QTI_SOC_SM6150(soc) ((soc) == 355 || (soc) == 377 || (soc) == 384)

#define IS_QTI_SOC_SM7150(soc) ((soc) == 365 || (soc) == 366)

#define IS_QTI_SOC_QCS405(soc) ((soc) == 352)

#define IS_QTI_SOC_SM6125(soc) ((soc) == 394)

#define IS_QTI_SOC_QCS403(soc) ((soc) == 373)

#endif // HEXAGON_NN_HEXNN_SOC_DEFINES_H
