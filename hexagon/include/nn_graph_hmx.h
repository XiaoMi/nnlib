
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
#ifndef NN_GRAPH_HMX_H
#define NN_GRAPH_HMX_H 1
//
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This defines common defs used for hmx.
 */
#define HMX_ALIGNMENT_REQ   2048
#define X_TILE_SIZE_DEF     8
#define Y_TILE_SIZE_DEF     8
#define Z_TILE_SIZE_DEF     32
#define X_TILE_OFFSET       5
#define Y_TILE_OFFSET       8
#define Z_TILE_OFFSET       0
#define X_TILE_MASK         7
#define Y_TILE_MASK         7
#define Z_TILE_MASK         31
#define INTER_X_SHIFT       3
#define INTER_Y_SHIFT       3
#define INTER_Z_SHIFT       5
#define INTER_TILE_SHIFT    (INTER_X_SHIFT+INTER_Y_SHIFT+INTER_Z_SHIFT)
#define SPATIAL_OFFSET      2
#define SPATIAL_SIZE        6


typedef struct __attribute__((aligned(8))) hmx_parameter {
    uint32_t    start;
    uint32_t    range;
    uint32_t    next_z_tile;
    uint32_t    next_x_tile;
    uint32_t    next_y_tile;
} hmx_params_t;

static int32_t tensor_index_tile(int32_t x, int32_t y, int32_t z, int32_t X, int32_t Y, int32_t Z, int mode)
{
	int x_tiles = (X + X_TILE_SIZE_DEF -1) >> INTER_X_SHIFT;
	int z_tiles = (Z + Z_TILE_SIZE_DEF -1) >> INTER_Z_SHIFT;
//	int y_tiles = (Y + Y_TILE_SIZE_DEF -1) >> INTER_Y_SHIFT;
	int32_t tiled_idx = 0;
	int32_t intra_tile_z = (z & Z_TILE_MASK);
	int32_t intra_tile_x = (x & X_TILE_MASK);
	int32_t intra_tile_y = (y & Y_TILE_MASK); 
	int32_t inter_tile_x = x >> INTER_X_SHIFT; 
	int32_t inter_tile_y = y >> INTER_Y_SHIFT; 
	int32_t inter_tile_z = z >> INTER_Z_SHIFT; 
	tiled_idx += (intra_tile_x << X_TILE_OFFSET);
	tiled_idx += (intra_tile_y << Y_TILE_OFFSET);
	tiled_idx += (intra_tile_z << Z_TILE_OFFSET);
	tiled_idx += (inter_tile_z + (inter_tile_x + inter_tile_y*x_tiles)*z_tiles) << INTER_TILE_SHIFT;

	if (mode==1) {
		tiled_idx = (tiled_idx&~0x7F)|((tiled_idx&0x1F)<<2)|((tiled_idx>>5)&3);
	}
	 return tiled_idx;
}
    
static inline uint8_t *tensor_location_tile(
	const struct tensor *src,
	int32_t b,
	int32_t h,
	int32_t w,
	int32_t d,
	int32_t mode)
{
	int32_t h_before = src->format.height_pad[0];
	int32_t w_before = src->format.width_pad[0];
	int32_t d_before = src->format.depth_pad[0];
  	int32_t h_total = h_before + src->shape.height + src->format.height_pad[1];
	int32_t w_total = w_before + src->shape.width + src->format.width_pad[1];
	int32_t d_total = d_before + src->shape.depth + src->format.depth_pad[1];
	uint8_t *base = (uint8_t*)src->data;
	int32_t w_pos = w + w_before;
	int32_t h_pos = h + h_before;
	int32_t d_pos = d + d_before;

	int32_t tiled_idx = tensor_index_tile(w_pos, h_pos, d_pos, w_total, h_total, d_total, mode);

	return base + tiled_idx;
}
#endif
