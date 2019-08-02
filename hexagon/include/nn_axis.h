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

#ifndef HEXAGON_NN_NN_AXIS_H
#define HEXAGON_NN_NN_AXIS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "nn_graph.h"

#define DIM_NUM 4   //number of dimensions

//Convert the negative axis to be positive
static inline int handle_negative_axes(struct nn_graph *nn, int32_t * axes, const int32_t num_axes) {

    for(int i = 0; i < num_axes; i++)
    {
        if (axes[i] < DIM_NUM*(-1) || axes[i] >= DIM_NUM)
            return errlog(nn, "Axis value %ld is out of range. Must be in the range -4 < axis < 4\n", axes[i]);
        if (axes[i] < 0)
            axes[i] = axes[i] + DIM_NUM;
    }
    return 0;
}

//return -1 if the axes are invalid
static inline int get_number_elements_between_axes(const struct shape in_shape, const int first_axis_inclusive, const int last_axis_exclusive) {

    if(0 <= first_axis_inclusive && first_axis_inclusive <= last_axis_exclusive && last_axis_exclusive <= DIM_NUM) {
        int count = 1;
        for (int i = first_axis_inclusive; i < last_axis_exclusive; ++i ) {
            count *= in_shape.dimension[i];
        }
        return count;
    }
    return -1;
}

#endif //HEXAGON_NN_NN_AXIS_H
