
/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
 */
/*======================================================================*/
/*  FUNCTIONS      : im2col_o.c                                         */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*  The memory layout of the data is, from biggest stride to smallest:  */
/*  input_data=[input_batchesx input_heightx (input_width x input_depth)*/
/*  each row has the depth axis of data packed together to a row is     */
/*  width*depth in length.                                              */
/*  filter_data =[filter_height, filter_width, input_depth,filter_count]*/
/*  The number of filters does not affect the size of the matrix.       */
/*  output_data=[input_batches, output_height,output_width,filter_size] */
/*  filtersize is height x width x input_depth - a cube of data         */
/*  packing the patches of the input image into columns (im2col)        */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */

/*  -------------------------------------------------------------       */
/*  DJH                 08/16/16       created                          */
/*======================================================================*/
/*  C OPTIMIZATIONS                                                     */
/*  The loops are re-ordered doing for each x                           */
/*     then for each filter count                                       */
/*       then for each y i.e a full column                              */
/*  This moves the control code to outer loops, the central copy is a 2d*/
/*  to 2d memcpy in assembly.                                           */
/*  The final result is padded to multiples of 8 wide and 4 high.       */
/*======================================================================*/
/*                                                                      */
/*======================================================================*/
/*  FUNCTIONS      : matrix multiply                                    */
/*  ARCHITECTURE   : QDSP6V6 HVX                                        */
/*                                                                      */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*======================================================================*/
/*     ASSUMPTIONS                                                      */
/*        Data is 128byte aligned                                       */
/************************************************************************/

//#ifdef __hexagon__
#if 1
#include <nn_graph.h>

//void vmemcpy_asm(void *dst, void *src, int len);
//void vmemset_asm(void *dst, uint8_t val, int len);

void gemm_cn(const uint8_t * a, int a_offset,
             const uint8_t * b, int b_offset,
             int * c,
             int N, int M, int K
            ) {
    int i, j, k;
    int sum, a_val, b_val;

    for (i=0; i < N; i++) {
      for (j=0; j < M; j++) {
          sum = 0;
          for (k=0; k < K; k++) {
            a_val = a[i*K+k] + a_offset;
            b_val = b[k*M+j] + b_offset;
            sum += a_val * b_val;
          }
          c[i*M+j] = sum;
       }
    }
    return;
}

void gemm_co(const uint8_t * a, int a_offset, const uint8_t * b, int b_offset, int * sumc, 
             int N, int M, int K, int *suma, int *sumb)
{
    int i, j, k, m;
    int32_t sum, c_offset;
    uint8_t a_val, b_val;

    for(m=0; m < M; m+=32) {
       //printf(" %d\n",m);
       c_offset = K*a_offset*b_offset;
       if(m==0)
       for (i=0; i < N; i++) {
         suma[i] = c_offset;
         for (k=0; k < K; k++) {
           a_val = a[i*K+k];
           suma[i] += a_val * b_offset;
         }
       }
       for (j=0; j <32; j++) {
         sumb[m+j] = 0;
         for (k=0; k < K; k++) {
           b_val = b[m+k*M+j];
           sumb[m+j] += b_val * a_offset;
         }
       }
       for (i=0; i < N; i++) {
         for (j=0; j <32; j++) {
           sumc[m+i*M+j] = 0 ;
           for (k=0; k < K; k++) {
             a_val = a[i*K+k];
             b_val = b[m+k*M+j];
             sumc[m+i*M+j]  += a_val * b_val ;
           }
         }
       }
       for (i=0; i < N; i++) {
         for (j=0; j <32; j++) {
           sum = sumb[m+j] + suma[i] + sumc[m+i*M+j];
           sumc[m+i*M+j] = sum;
         }
       }
    }
    return;
}

void gemsuma_cn(uint8_t * x, int N, int K, int *xsum, int y_offset, int z_offset) {
    int i,k, a_val, sum;
    for (i=0; i < N; i++) {
      sum = 0;
      for (k=0; k < K; k++) {
        a_val = x[i*K+k];
        sum += a_val;
      }
      xsum[i] = (sum*y_offset) + z_offset;
    }
}
void gemsumb_cn(uint8_t * y, int *z, int M, int K, int x_offset) {
    int j,k, a_val,b_val=x_offset;
     for (j=0; j < 32; j+=1) {
       z[j] = 0 ;
       for (k=0; k < K; k+=4) {
         a_val = y[32*k+0+4*j];
         z[j]  += a_val * b_val ;

         a_val = y[32*k+1+4*j];
         z[j]  += a_val * b_val ;

         a_val = y[32*k+2+4*j];
         z[j]  += a_val * b_val ;

         a_val = y[32*k+3+4*j];
         z[j]  += a_val * b_val ;
       }
     }
}

void gemmpybbw_cn(uint8_t * x, uint8_t * y, int * z, int N, int M, int K) {
   int i,j,k, a_val, b_val;
   for (i=0; i < N; i++) {
     for (j=0; j < 32; j+=1) {
       z[i*M+j] = 0 ;
       for (k=0; k < K; k+=4) {
         a_val = x[i*K+k];
         b_val = y[32*k+0+4*j];
         z[i*M+j]  += a_val * b_val ;

         a_val = x[i*K+k+1];
         b_val = y[32*k+1+4*j];
         z[i*M+j]  += a_val * b_val ;

         a_val = x[i*K+k+2];
         b_val = y[32*k+2+4*j];
         z[i*M+j]  += a_val * b_val ;

         a_val = x[i*K+k+3];
         b_val = y[32*k+3+4*j];
         z[i*M+j]  += a_val * b_val ;
       }
     }
   }
}
void gemaddvvm_cn(int * xsum, int * ysum, int * z, int N, int M)
{
  int i,j,sum;
    for (i=0; i < N; i++) {
      for (j=0; j <32; j++) {
        sum = xsum[i] + ysum[j] + z[i*M+j];
        z[i*M+j] = sum;
      }
    }
}


#if 1
#include <hexagon_protos.h>
void gemm_asm(const uint8_t * x, int x_offset,
              const uint8_t * yopt, int y_offset,
              int * z, int N, int M, int K, 
              int NSTEP, int MSTEP, int KSTEP,
              int * xsum, int * ysum, int *minmax)
{

    //output tile is NSTEP*128 bytes   i.e NSTEP = 128 is 16K only
    int i;
      for(i=0; i < M; i+=MSTEP) {
          if(i==0)gemsuma_asm(&x[0], NSTEP, Q6_R_combine_RlRl(KSTEP,K), &xsum[0], y_offset, K*x_offset*y_offset);
          //if(i==0)gemsuma_cn(&x[0], NSTEP, K, &xsum[0], y_offset, K*x_offset*y_offset);

          gemsumb_asm(&yopt[i*K],  &ysum[i], KSTEP, x_offset);
          //gemsumb_cn(&yopt[i*K], &ysum[i], M, KSTEP, x_offset);

          gemmpybbw_asm(x,  &yopt[i*K],  &z[i], NSTEP, M, Q6_R_combine_RlRl(KSTEP,K));
          //gemmpybbw_cn(&x[0],   &yopt[i*K],        &z[i], NSTEP, M, K);

          gemaddvvm_asm(&xsum[0], &ysum[i], &z[i],NSTEP, M, minmax, (i==0));
          //gemaddvvm_cn(&xsum[0], &ysum[i], &z[i],NSTEP, M); //, minmax, (i==0 && j==0));
      }//end M

    return;
}
#else
void gemm_asm(uint8_t * x, int x_offset,
              uint8_t * yopt, int y_offset,
              int * z, int N, int M, int K, 
              int NSTEP, int MSTEP, int KSTEP,
              int * xsum, int * ysum, int *minmax)
{

    //output tile is NSTEP*128 bytes   i.e NSTEP = 128 is 16K only
    int i,j,k;
    printf(" M = %d N = %d K = %d xo = %d yo = %d\n",M,N,K, x_offset, y_offset);
    if(M > N) {
      for(i=0; i < M; i+=MSTEP) {
        for(j=0; j < N; j+=NSTEP) {
          for(k=0; k < K; k+=KSTEP) {

            if(k==0) {
              if(i==0)gemsuma_cn(&x[j*K], NSTEP, K, &xsum[j], y_offset, K*x_offset*y_offset);

              if(j==0)gemsumb_cn(&yopt[i*K],       &ysum[i], M, KSTEP, x_offset);

              gemmpybbw_asm(&x[j*K],   &yopt[i*K],        &z[j*M+i], NSTEP, M, Q6_R_combine_RlRl(KSTEP,K));
              gemmpybbw_cn(&x[0],   &yopt[i*K],        &z[i], NSTEP, M, K);

            } else {
              if(i==0)gemacca_asm(&x[j*K+k], NSTEP, Q6_R_combine_RlRl(KSTEP,K), &xsum[j], y_offset);

              if(j==0)gemaccb_asm(&yopt[i*K+MSTEP*k], &ysum[i], KSTEP, x_offset);

              gemmacbbw_asm(&x[j*K+k], &yopt[i*K+MSTEP*k], &z[j*M+i], NSTEP, M, Q6_R_combine_RlRl(KSTEP,K));
            }
          }
          gemaddvvm_cn(&xsum[j], &ysum[i], &z[j*M+i],NSTEP, M); //, minmax, (i==0 && j==0));
        }//end N
      }//end M
    } else { //N>M
      for(j=0; j < N; j+=NSTEP) {
        for(i=0; i < M; i+=MSTEP) {
          for(k=0; k < K; k+=KSTEP) {

            if(k==0) {
              if(i==0)gemsuma_asm(&x[j*K],   NSTEP, Q6_R_combine_RlRl(KSTEP,K), &xsum[0], y_offset, K*x_offset*y_offset);

              if(j==0)gemsumb_asm(&yopt[i*K],         &ysum[i], KSTEP, x_offset);

              gemmpybbw_asm(&x[j*K],   &yopt[i*K],        &z[j*M+i], NSTEP, M, Q6_R_combine_RlRl(KSTEP,K));

            } else {
              if(i==0)gemacca_asm(&x[j*K+k], NSTEP, Q6_R_combine_RlRl(KSTEP,K), &xsum[0], y_offset);

              if(j==0)gemaccb_asm(&yopt[i*K+MSTEP*k], &ysum[i], KSTEP, x_offset);

              gemmacbbw_asm(&x[j*K+k], &yopt[i*K+MSTEP*k], &z[j*M+i], NSTEP, M, Q6_R_combine_RlRl(KSTEP,K));
            }
          }
          gemaddvvm_asm(&xsum[0], &ysum[i], &z[j*M+i],NSTEP, M, minmax, (i==0 && j==0));
        }//end M
      }//end N
    }
    return;
}
#endif

#if defined(__hexagon__)
static int max(int a, int b) { return((a>b) ? a : b); }
static int min(int a, int b) { return((a<b) ? a : b); }
#endif

#define HPAD 16
#define VPAD 8

void im2col_co(
  uint8_t* input_data, int input_height, int input_width, int input_depth, int input_offset, 
  uint8_t* im2col_buffer, int filter_height, int filter_width, int stride,
  int output_height, int output_width, int filter_left_offset, int filter_top_offset)
{
    int out_y, out_x, filter_y ;

    int filter_area = filter_width * input_depth;
    int filter_value_count = filter_area * filter_height; //will need to be padded to 8
    int filter_value_count_pad = (filter_value_count + HPAD - 1) & ~(HPAD-1);
    filter_value_count_pad = max(filter_value_count_pad,32);
    int pad_x = filter_value_count_pad - filter_value_count;
    int patches_pad = (output_width * output_height + VPAD - 1) & ~(VPAD-1);
    int pad_y = patches_pad - output_width * output_height;
    int im2col_width = output_width * filter_value_count_pad ;

    uint8_t* input_batch_start = input_data ;
    uint8_t* im2col_row_start;
    for (out_x = 0; out_x < output_width; ++out_x)
    {
        int in_y, start, stop, offset;
        uint8_t* input_row_start;
        uint8_t* im2col_row_start_l;
        uint8_t* im2col_row_start_c;
        uint8_t* im2col_row_start_r;
        int in_x_origin = (out_x * stride) - filter_left_offset;
        int in_x_end = in_x_origin + filter_width;
        int left_zero_count = max(0, 0 - in_x_origin)*input_depth;
        int right_zero_count = max(0, in_x_end - input_width)*input_depth;
        int center_copy_count = filter_width*input_depth - (left_zero_count + right_zero_count);
        int patch_index, patch_base = out_x * filter_value_count_pad;
        //each rounder of filtering creates
        for (filter_y = 0; filter_y < filter_height; ++filter_y)
        {
          offset = filter_y - filter_top_offset;
          im2col_row_start = im2col_buffer + patch_base+ (filter_y * filter_area);
          for (in_y = offset; in_y < 0; in_y+=stride)
          {
            vmemset_asm(im2col_row_start, input_offset, filter_width * input_depth);
            //memset(im2col_row_start, input_offset, filter_width * input_depth);
            im2col_row_start += im2col_width;
          }
          start = max(0, offset);
          stop = min(input_height, output_height*stride+offset);
          out_y = start - offset;
          patch_index = patch_base + (out_y/stride) * im2col_width;
          im2col_row_start = im2col_buffer + patch_index + (filter_y * filter_area);

          if (left_zero_count > 0) {
            im2col_row_start_l = im2col_row_start ;
            for (in_y = start; in_y < stop; in_y+=stride)
            {
              //memset(im2col_row_start_l, input_offset, left_zero_count);
              vmemset_asm(im2col_row_start_l, input_offset, left_zero_count);
              im2col_row_start_l += im2col_width;
            }//in_y
          }
          if (center_copy_count > 0) {
            im2col_row_start_c = im2col_row_start ;
            input_row_start = input_batch_start + (start * input_width * input_depth) + (max(0, in_x_origin) * input_depth);
#if 1      
            for(in_y=0;in_y<(stop-start+stride-1)/stride; in_y+=1)//for(in_y=start;in_y<stop;in_y+=stride)
            {
              vmemcpy_asm(im2col_row_start_c+left_zero_count, input_row_start, center_copy_count);
              im2col_row_start_c += im2col_width;
              input_row_start += (stride * input_width * input_depth) ;
            }//in_y
#else
            vmemcpy2d_asm(im2col_row_start_c+left_zero_count, im2col_width, 
                          input_row_start, stride*input_width*input_depth,
                          center_copy_count, (stop-start+stride-1)/stride);
#endif
          }
          if (right_zero_count > 0) {
            im2col_row_start_r = im2col_row_start ;
            for (in_y = start; in_y < stop; in_y+=stride)
            {
              vmemset_asm(im2col_row_start_r+left_zero_count + center_copy_count, input_offset, right_zero_count);
              //memset(im2col_row_start_r+left_zero_count + center_copy_count, input_offset, right_zero_count);
              im2col_row_start_r += im2col_width;
            }//in_y
          }
          //BOTTOM
          out_y = (input_height+stride-offset-1)/stride;
          patch_index = patch_base + (out_y * im2col_width);
          im2col_row_start = im2col_buffer + patch_index + (filter_y * filter_area);
          for (in_y = input_height+stride-1; in_y < output_height*stride+offset; in_y+=stride)
          {
            vmemset_asm(im2col_row_start, input_offset, filter_width * input_depth);
            //memset(im2col_row_start, input_offset, filter_width * input_depth);
            im2col_row_start += im2col_width;
          }//in_y
        }//filter_y
#if 1
         //at end of each block of filters add in the pad of 1 to 7 bytes
        if(pad_x > 0) {
          patch_index = out_x * filter_value_count_pad + filter_value_count;
          im2col_row_start = im2col_buffer + patch_index;
          for (out_y = 0; out_y < output_height; out_y++)
          {
            vmemset_asm(im2col_row_start, input_offset, pad_x);
            im2col_row_start += im2col_width;
          }//out_y
        }
#endif
      }//out_x
#if 1
      //bottom of block pad
      im2col_row_start = im2col_buffer + output_width*output_height*filter_value_count_pad;
      for (out_y =0 ; out_y < pad_y; out_y++)
      {
        vmemset_asm(im2col_row_start, input_offset, filter_value_count_pad); 
        im2col_row_start += filter_value_count_pad;
      }//in_y
#endif
    return;
}


// packing the patches of the input image into columns (im2col) 
void im2col_cn(
  uint8_t* input_data, int input_height, int input_width, int input_depth, int input_offset, 
  uint8_t* im2col_buffer, int filter_height, int filter_width, int stride, 
  int output_height, int output_width, int filter_left_offset, int filter_top_offset)
{
    int out_y, out_x, filter_y ;

    //        < filter value count >
    //   ^   +---------------------+
    // patch |                     |
    // count |                     |
    //   v   +---------------------+
    int filter_area = filter_width * input_depth;
    int filter_value_count = filter_area * filter_height; //will need to be padded 
    int filter_value_count_pad = (filter_value_count + HPAD - 1) & ~(HPAD-1);
    filter_value_count_pad = max(filter_value_count_pad,32);
    int pad_x = filter_value_count_pad - filter_value_count;
    int patches_pad = (output_width * output_height + VPAD - 1) & ~(VPAD-1);
    int pad_y = patches_pad - output_width * output_height;
    int im2col_width = output_width * filter_value_count_pad ;

      uint8_t* input_batch_start = input_data ;
      for (out_y = 0; out_y < output_height; ++out_y)
      {
        int in_y_origin = (out_y * stride) - filter_top_offset;
        for (out_x = 0; out_x < output_width; ++out_x)
        {
          int in_x_origin = (out_x * stride) - filter_left_offset;
          int patch_index =  (out_y * im2col_width) + out_x*filter_value_count_pad;
          uint8_t* im2col_patch_start = im2col_buffer + patch_index ;
          for (filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            int in_y = in_y_origin + filter_y;
            uint8_t* im2col_row_start = im2col_patch_start + (filter_y * filter_width * input_depth);
            // If we're off the top or the bottom of the input, fill the whole
            // row with zeroes.
            if (in_y < 0 || in_y >= input_height) {
              // We'll be subtracting this offset during the calculations
              // so to get an actual zero after that bias we need to set
              // it to input_offset here.
         
              memset(im2col_row_start, input_offset, filter_width * input_depth);
            } else {
              // < left_zero_count > < center_copy_count > < right_zero_count >
              // +------------------+---------------------+--------------------+
              // |     (filter)     |       (image)       |      (filter)      |
              // +------------------+---------------------+--------------------+
              // in_x_origin        0                 input_width       in_x_end
              int in_x_end = in_x_origin + filter_width;
              int left_zero_count = max(0, 0 - in_x_origin);
              int right_zero_count = max(0, in_x_end - input_width);
              int center_copy_count = filter_width - (left_zero_count + right_zero_count);
              if (left_zero_count > 0) {
                memset(im2col_row_start, input_offset, left_zero_count * input_depth);
              }
              if (center_copy_count > 0) {
                uint8_t* input_row_start = input_batch_start + (in_y * input_width * input_depth) +
                    (max(0, in_x_origin) * input_depth);
                uint8_t* im2col_center_start = im2col_row_start + (left_zero_count * input_depth);
                vmemcpy_asm(im2col_center_start, input_row_start, center_copy_count*input_depth);
              }
              if (right_zero_count > 0) {
                uint8_t* im2col_right_start = im2col_row_start +
                    ((left_zero_count + center_copy_count) * input_depth);
                memset(im2col_right_start, input_offset, right_zero_count * input_depth);
              }
            }
          }//filter_y
          memset(im2col_patch_start + filter_value_count, input_offset, pad_x);
        }//out_x
      }//out_y
      //pad out bottom rows if needed
      int patch_index = (output_height * output_width * filter_value_count_pad);
      memset(im2col_buffer+patch_index, input_offset, filter_value_count_pad*pad_y);
    return;
}

//perform the im2col in strips horizontally

void im2col_slice_v0_co(
  const uint8_t* in_data, int in_height, int in_width, int in_depth, int in_offset, 
  uint8_t* im2col_buffer, int filt_height, int filt_width, int stride, 
  int out_height, int out_width, int filt_left_offset, int filt_top_offset, int start_patches, int num_patches)
{
    int out_y, out_x, filt_y;
    int filt_area = filt_width * in_depth;
    int filt_value_count = filt_area * filt_height; //will need to be padded 
    int filt_value_count_pad = (filt_value_count + HPAD - 1) & ~(HPAD-1);
    int pad_x = filt_value_count_pad - filt_value_count;
    int patches_pad = (out_width * out_height + VPAD - 1) & ~(VPAD-1);
    int pad_y = patches_pad - out_width * out_height;
    //int im2col_width = out_width * filt_value_count_pad ;
    uint8_t* im2col_patch_start = im2col_buffer ;

    //start_patches = start_y * out_width + start_x
    int end_patches, y_start, x_start, y_stop, x_stop;

    end_patches = start_patches + num_patches;
    y_start = start_patches / out_width;
    x_start = start_patches - y_start * out_width;
    //printf(" %d %d\n", end_patches, out_width*out_height);
    if(end_patches >= out_width*out_height) 
    {
        y_stop = out_height-1;
        x_stop = out_width;
    } else {
        y_stop = (end_patches / out_width);
        x_stop = end_patches - y_stop * out_width;
    }
    //printf(" %d %d -> %d %d\n", y_start, x_start, y_stop, x_stop);
       
    const uint8_t* in_batch_start = in_data ;
    for (out_y = y_start; out_y <= y_stop; ++out_y)
    {
        int x_0, x_1;
        int in_y_origin = (out_y * stride) - filt_top_offset;
        if(out_y == y_start) {x_0 = x_start; x_1 = out_width; }
        else if(out_y == y_stop) {x_0 = 0; x_1 = x_stop; }
        else 
        { x_0 = 0; x_1 = out_width; }

        for (out_x = x_0; out_x < x_1; ++out_x)
        {
          int in_x_origin = (out_x * stride) - filt_left_offset;
          for (filt_y = 0; filt_y < filt_height; ++filt_y)
          {
            int in_y = in_y_origin + filt_y;
            uint8_t* im2col_row_start = im2col_patch_start + (filt_y * filt_width * in_depth);
            if (in_y < 0 || in_y >= in_height) {
              vmemset_asm(im2col_row_start, in_offset, filt_width * in_depth);
            } else {
              int in_x_end = in_x_origin + filt_width;
              int left_zero_count = max(0, 0 - in_x_origin);
              int right_zero_count = max(0, in_x_end - in_width);
              int center_copy_count = filt_width - (left_zero_count + right_zero_count);
              if (left_zero_count > 0) {
                vmemset_asm(im2col_row_start, in_offset, left_zero_count * in_depth);
              }
              if (center_copy_count > 0) {
                const uint8_t* in_row_start = in_batch_start + (in_y * in_width * in_depth) +
                    (max(0, in_x_origin) * in_depth);
                uint8_t* im2col_center_start = im2col_row_start + (left_zero_count * in_depth);
                vmemcpy_asm(im2col_center_start, in_row_start, center_copy_count*in_depth);
                //if(p==1) {printf(" in = %08x \n", (int) in_row_start); p=0; }
              }
              if (right_zero_count > 0) {
                uint8_t* im2col_right_start = im2col_row_start +
                    ((left_zero_count + center_copy_count) * in_depth);
                vmemset_asm(im2col_right_start, in_offset, right_zero_count * in_depth);
              }
            }
          }//filt_y
          vmemset_asm(im2col_patch_start + filt_value_count, in_offset, pad_x);
          im2col_patch_start += filt_value_count_pad;
        }//out_x
    }//out_y
    if(end_patches >= out_height*out_width) {
        //pad out bottom rows if needed
        vmemset_asm(im2col_patch_start, in_offset, filt_value_count_pad*pad_y);
    }
    return;
}

void im2col_slice_co(
  const uint8_t* in_data, int in_height, int in_width, int in_depth, int in_offset, 
  uint8_t* im2col_patch_start, int filt_height, int filt_width, int stride, 
  int out_height, int out_width, int filt_left_offset, int filt_top_offset, int start_patches, int num_patches)
{
    int out_y, out_x, filt_y ;
    int filt_area = filt_width * in_depth;
    int filt_value_count = filt_area * filt_height; //will need to be padded 
    int filt_value_count_pad = (filt_value_count + HPAD - 1) & ~(HPAD-1);
    int end_patches, y_start, x_start, y_stop, x_stop;

    end_patches = start_patches + num_patches;
    y_start = start_patches / out_width;
    x_start = start_patches - y_start * out_width;
    if(end_patches >= out_width*out_height) 
    {
        y_stop = out_height-1;
        x_stop = out_width;
    } else {
        y_stop = (end_patches / out_width);
        x_stop = end_patches - y_stop * out_width;
    }
    vmemset_asm(im2col_patch_start, in_offset, num_patches*filt_value_count_pad);
    const uint8_t* in_batch_start = in_data ;
    for (out_y = y_start; out_y <= y_stop; ++out_y)
    {
        int x_0, x_1;
        int in_y_origin = (out_y * stride) - filt_top_offset;
        x_0 = 0; x_1 = out_width;
        if(out_y == y_stop)  {x_0 = 0;       x_1 = x_stop; }
        if(out_y == y_start) {x_0 = x_start; x_1 = out_width; }

        for (out_x = x_0; out_x < x_1; ++out_x)
        {
          int in_x_origin = (out_x * stride) - filt_left_offset;
          for (filt_y = 0; filt_y < filt_height; ++filt_y)
          {
            int in_y = in_y_origin + filt_y;
            uint8_t* im2col_row_start = im2col_patch_start + (filt_y * filt_width * in_depth);
            if (in_y >= 0 && in_y < in_height) {
              int in_x_end = in_x_origin + filt_width;
              int left_zero_count = Q6_R_max_RR(0, 0 - in_x_origin);
              int right_zero_count = Q6_R_max_RR(0, in_x_end - in_width);
              int center_copy_count = filt_width - (left_zero_count + right_zero_count);
              if (center_copy_count > 0) {
                const uint8_t* in_row_start = in_batch_start + (in_y * in_width + Q6_R_max_RR(0, in_x_origin)) * in_depth;
                uint8_t* im2col_center_start = im2col_row_start + (left_zero_count * in_depth);
                vmemcpy_asm(im2col_center_start, in_row_start, center_copy_count*in_depth);
              }
            }
          }//filt_y
          im2col_patch_start += filt_value_count_pad;
        }//out_x
    }//out_y
    return;
}

//new virtual im2col, only pads single stream, oriented around lines instead of patches
void fast_im2col_co(
  const uint8_t* in_data, int in_height, int in_width, int in_depth, int in_offset,
  uint8_t* im2col_buf, int filt_height, int filt_width, int stride,
  int start_line, int num_lines, int out_width, int pad_left, int pad_top, int skip_unpad_k)
{
    int out_y, in_y, in_x, j, pad_right ;
    pad_right = ((in_depth + 15) &  ~15) - in_depth;
    for (out_y = start_line; out_y <= (start_line+num_lines + filt_height-1); out_y++)
    {
      in_y = (out_y * stride) - pad_top;

      if(skip_unpad_k)
        for(j=0; j < stride && in_y <= in_height + pad_top; j++, in_y++) {
          if(in_y < 0 || in_y >= in_height) {
            vmemset_asm(im2col_buf, in_offset, in_depth*(in_width+pad_left));
            im2col_buf += in_depth*(in_width+pad_left);
          } else {
            vmemset_asm(im2col_buf, in_offset, in_depth*pad_left);
            im2col_buf += in_depth*pad_left;
            vmemcpy_asm(im2col_buf, in_data+in_y*in_width*in_depth, in_width*in_depth);
            im2col_buf += in_depth*in_width;
          }
        }
      else
        //special case for filts = 1 but depth not multiple of 16
        for(j=0, in_x=0; j < out_width; j++, in_x+=in_depth) {
            vmemcpy_asm(im2col_buf, in_data+in_y*in_width*in_depth+in_x, in_depth);
            im2col_buf += in_depth;
            vmemset_asm(im2col_buf, in_offset, pad_right);
            im2col_buf += pad_right;
        }
    }//out_y
    return;
}

#endif //hexagon
