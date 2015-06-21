#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


#define O_TILE_WIDTH 12
#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)

//@@ INSERT CODE HERE
__global__ void convolution_2D_kernel(float *inputImage, float *outputImage, int height, int width, int channels, const float * __restrict__ mask) 
{
   __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH];

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int row_o = blockIdx.y * O_TILE_WIDTH + ty;
   int col_o = blockIdx.x * O_TILE_WIDTH + tx;
   int row_i = row_o - MASK_RADIUS;
   int col_i = col_o - MASK_RADIUS; 

   for (int k = 0; k < channels; ++k)
   {
      if( (row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width) ) 
         tile[ty][tx] = inputImage[(row_i * width + col_i) * channels + k];
      else
         tile[ty][tx] = 0.0f;

      __syncthreads();

      if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
      {
         float output = 0.0f;
         for(int i = 0; i < MASK_WIDTH; ++i) 
         {
            for(int j = 0; j < MASK_WIDTH; ++j) 
            {
               output += mask[i * MASK_WIDTH + j] * tile[i + ty][j + tx];
            }
         }
         if( row_o < height && col_o < width)
            outputImage[(row_o * width + col_o) * channels + k] = output; 
      }
      
      __syncthreads();
   }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid( (imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
    convolution_2D_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, 
                                                 imageHeight, imageWidth, imageChannels,
                                                 deviceMaskData);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
