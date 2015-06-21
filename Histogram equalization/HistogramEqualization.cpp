#include <wb.h>
#include <sstream>
#include <algorithm>
#include <limits>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define BLOCK_WIDTH 16
#define RGB_CHANNELS 3
#define HISTOGRAM_LENGTH 256
#define SCAN_BLOCK_SIZE 256
#define MIN_CDF_BLOCK_SIZE 256 
#define HEF_BLOCK_SIZE 256 

__global__ void convertToUnsignedChar(float *inputImage, unsigned char *outputImage, int height, int width, int channels) 
{
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if( (y < height) && (x < width) ) 
   {
      int pixelIndex = ( y * width + x ) * channels;
      for (int k = 0; k < channels; ++k)
      {
          outputImage[pixelIndex + k] = (unsigned char) ( 255 * inputImage[pixelIndex + k] );
      }
   }
}

__global__ void convertToGrayScaleImage(unsigned char *ucharImage, unsigned char *grayScaleImage, int height, int width, int channels) 
{
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if( (y < height) && (x < width) ) 
   {
      int ucharPixelIndex = ( y * width + x ) * channels;
      int grayScalePixelIndex = ( y * width + x );
      if (RGB_CHANNELS == channels)
      {
         unsigned char r = ucharImage[ucharPixelIndex];
         unsigned char g = ucharImage[ucharPixelIndex + 1];
         unsigned char b = ucharImage[ucharPixelIndex + 2];
         grayScaleImage[grayScalePixelIndex] = (unsigned char) ( 0.21 * r + 0.71 * g + 0.07 * b );
      }
      else
      {
         // counting average
         unsigned int average = 0;
         for (int k = 0; k < channels; ++k)
         {
             average += ucharImage[ucharPixelIndex + k];
         }
         grayScaleImage[grayScalePixelIndex] = (unsigned char) ( average / channels );
      }
   }
}

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)
{
   __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

   if (threadIdx.x < HISTOGRAM_LENGTH) 
      histo_private[threadIdx.x] = 0;
   
   __syncthreads();

   int i = threadIdx.x + blockIdx.x * blockDim.x;
   
   int stride = blockDim.x * gridDim.x; // stride is total number of threads
   while (i < size) 
   {
      atomicAdd( &(histo_private[buffer[i]]), 1);
      i += stride;
   }

   __syncthreads();

   if (threadIdx.x < HISTOGRAM_LENGTH) 
      atomicAdd( &(histo[threadIdx.x]), histo_private[threadIdx.x] );
}

__global__ void scan_sumUp(float * output, int len)
{
    unsigned int indexInBlock = threadIdx.x;
    unsigned int index = indexInBlock + blockDim.x;

    for ( int indexOfAgregate = blockDim.x - 1; index < len; )
    {
       output[index] += output[indexOfAgregate];
       indexOfAgregate += blockDim.x;
       index += blockDim.x;
       __syncthreads();
    }
}

__global__ void scan(unsigned int * input, float * output, int len, int pixelsAmount) 
{
    __shared__ float XY[2 * SCAN_BLOCK_SIZE];
    
    unsigned int firstIndexInBlock = threadIdx.x;
    unsigned int secondIndexInBlock = threadIdx.x + blockDim.x;
    unsigned int firstIndexInArray = 2 * blockIdx.x * blockDim.x + firstIndexInBlock;
    unsigned int secondIndexInArray = 2 * blockIdx.x * blockDim.x + secondIndexInBlock;

    //@@ Load a segment of the input vector into shared memory
    if (firstIndexInArray < len)
       XY[firstIndexInBlock] = (float) input[firstIndexInArray] / pixelsAmount;
    else
       XY[firstIndexInBlock] = 0.0f;

    if (secondIndexInArray < len)
       XY[secondIndexInBlock] = (float) input[secondIndexInArray] / pixelsAmount;
    else
       XY[secondIndexInBlock] = 0.0f;

    __syncthreads();

    for (int stride = 1; stride <= SCAN_BLOCK_SIZE; stride *= 2) 
    {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if(index < 2 * SCAN_BLOCK_SIZE)
          XY[index] += XY[index - stride];
       
       __syncthreads();
    }

    for (int stride = SCAN_BLOCK_SIZE / 2; stride > 0; stride /= 2) 
    {
       __syncthreads();
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if(index + stride < 2 * SCAN_BLOCK_SIZE) 
          XY[index + stride] += XY[index];
    }

    __syncthreads();

    if (firstIndexInArray < len)
       output[firstIndexInArray] = XY[firstIndexInBlock];
    if (secondIndexInArray < len)
       output[secondIndexInArray] = XY[secondIndexInBlock];
}

__global__ void computeMinimumCDF_kernel(float * input, float * output, int len) 
{
    __shared__ float partialMin[2 * MIN_CDF_BLOCK_SIZE];
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    //@@ Load a segment of the input vector into shared memory
    if (start + t < len)
       partialMin[t] = input[start + t];
    else
       partialMin[t] = 1.0f;

    if (start + blockDim.x + t < len)
       partialMin[blockDim.x + t] = input[start + blockDim.x + t];
    else
       partialMin[blockDim.x + t] = 1.0f;

    __syncthreads();

    //@@ Traverse the reduction tree
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
       if (t < stride )
       {
          if (partialMin[t + stride] < partialMin[t]) 
             partialMin[t] = partialMin[t + stride];
       }

       __syncthreads();
    }

    //@@ Write the computed minimum of the block to the output vector at the correct index
    if (t == 0)
       output[blockIdx.x] = partialMin[0];
}

__global__ void correctImageColor(unsigned char *deviceUcharImage, float* deviceComulativeDistributionFunction, float minimumCDF, int height, int width, int channels) 
{
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if( (y < height) && (x < width) ) 
   {
      int pixelIndex = ( y * width + x ) * channels;
      for (int k = 0; k < channels; ++k)
      {
          unsigned char correctedValue = 255 * ( (deviceComulativeDistributionFunction[deviceUcharImage[pixelIndex + k]] - minimumCDF) / (1 - minimumCDF));
          //std::min(std::max(correctedValue, 0), 255);
          unsigned char maxCorrected = 0;
          if (maxCorrected < correctedValue)
             maxCorrected = correctedValue;
          unsigned char clampedValue = 255;
          if (clampedValue > maxCorrected)
             clampedValue = maxCorrected;
          deviceUcharImage[pixelIndex + k] = clampedValue;
      }
   }
}

__global__ void castBackToFloat_kernel(unsigned char *inputImage, float *outputImage, int height, int width, int channels) 
{
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if( (y < height) && (x < width) ) 
   {
      int pixelIndex = ( y * width + x ) * channels;
      for (int k = 0; k < channels; ++k)
      {
          outputImage[pixelIndex + k] = (float) ( inputImage[pixelIndex + k] ) / 255;
      }
   }
}

void checkHistoOutput(unsigned char* deviceGrayScaleImage, int imageWidth, int imageHeight, unsigned int* deviceHistogram);
void checkComulativeDistributionFunction(unsigned int* deviceHistogram, int imageWidth, int imageHeight, float* deviceComulativeDistributionFunction);
void checkMinimumCDF(float* deviceComulativeDistributionFunction, float computedMinimumCDF);
void checkCorrectedImage(unsigned char* deviceUcharImage, unsigned char* hostUcharImageCopy, int imageWidth, int imageHeight, int imageChannels, float* deviceComulativeDistributionFunction, float minimumCDF);

float* prepareDeviceInputImageMemory(float* hostInputImageData, int imageHeight, int imageWidth, int imageChannels);
unsigned char* convertImageToUnsignedChar(float * deviceInputImageData, int imageHeight, int imageWidth, int imageChannels);
unsigned char* convertImageToGrayScale(unsigned char* deviceUcharImage, int imageHeight, int imageWidth, int imageChannels);
unsigned int* computeHistogram(unsigned char* deviceGrayScaleImage, int imageHeight, int imageWidth);
float* computeComulativeDistributionFunction(unsigned int* deviceHistogram, int imageHeight, int imageWidth);
float computeMinimumCDF(float* deviceComulativeDistributionFunction);
void applyHistogramEqualizationFunction(unsigned char* deviceUcharImage, int imageHeight, int imageWidth, int imageChannels,
                                        float* deviceComulativeDistributionFunction, float minimumCDF);
float* castBackToFloat(unsigned char* deviceUcharImage, int imageHeight, int imageWidth, int imageChannels);

int main(int argc, char ** argv) 
{
    wbArg_t args = wbArg_read(argc, argv); /* parse the input arguments */
                 
    const char * inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    wbImage_t inputImage = wbImport(inputImageFile);
    int imageWidth = wbImage_getWidth(inputImage);
    int imageHeight = wbImage_getHeight(inputImage);
    int imageChannels = wbImage_getChannels(inputImage);
    wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    float* hostInputImageData  = wbImage_getData(inputImage);

    float* deviceInputImageData = prepareDeviceInputImageMemory(hostInputImageData, imageHeight, imageWidth, imageChannels);
                                      
    unsigned char* deviceUcharImage = convertImageToUnsignedChar(deviceInputImageData, imageHeight, imageWidth, imageChannels);

    unsigned char* deviceGrayScaleImage = convertImageToGrayScale(deviceUcharImage, imageHeight, imageWidth, imageChannels);

    unsigned int* deviceHistogram = computeHistogram(deviceGrayScaleImage, imageHeight, imageWidth);
    //checkHistoOutput(deviceGrayScaleImage, imageWidth, imageHeight, deviceHistogram);

    float* deviceComulativeDistributionFunction = computeComulativeDistributionFunction(deviceHistogram, imageHeight, imageWidth);
    //checkComulativeDistributionFunction(deviceHistogram, imageHeight, imageWidth, deviceComulativeDistributionFunction);

    float MinimumCDF = computeMinimumCDF(deviceComulativeDistributionFunction);
    //checkMinimumCDF(deviceComulativeDistributionFunction, MinimumCDF);

    //unsigned char* hostUcharImageCopy = (unsigned char*) malloc(imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    //cudaMemcpy(hostUcharImageCopy, deviceUcharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    applyHistogramEqualizationFunction(deviceUcharImage, imageHeight, imageWidth, imageChannels, deviceComulativeDistributionFunction, MinimumCDF);
    //checkCorrectedImage(deviceUcharImage, hostUcharImageCopy, imageHeight, imageWidth, imageChannels, deviceComulativeDistributionFunction, MinimumCDF);
    
    float* deviceOutputImageData = castBackToFloat(deviceUcharImage, imageHeight, imageWidth, imageChannels);
    float* hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(Copy, "Copying output image from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output image from the GPU");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceUcharImage);
    cudaFree(deviceGrayScaleImage);
    cudaFree(deviceHistogram);
    cudaFree(deviceComulativeDistributionFunction);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

void checkHistoOutput(unsigned char* deviceGrayScaleImage, int imageWidth, int imageHeight, unsigned int* deviceHistogram)
{
    wbTime_start(Copy, "checkHistoOutput: Copying data from the GPU");
    unsigned char* hostGrayScaleImage = (unsigned char*) malloc(imageWidth * imageHeight * sizeof(unsigned char));
    cudaMemcpy(hostGrayScaleImage, deviceGrayScaleImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    unsigned int* hostHistogram = (unsigned int*) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "checkHistoOutput: Copying data from the GPU");

    wbTime_start(Copy, "checkHistoOutput: Serial Computation");

    unsigned int hostHistogramExpected[HISTOGRAM_LENGTH];

    for (unsigned int i = 0; i < HISTOGRAM_LENGTH; ++i)
    {
       hostHistogramExpected [ i ] = 0; 
    }

    for (unsigned int i = 0; i < imageHeight; ++i)
    {
       for (unsigned int j = 0; j < imageWidth; ++j)
       {
           hostHistogramExpected [ hostGrayScaleImage [ i * imageWidth + j ] ]++; 
       }
    }
    wbTime_stop(Copy, "checkHistoOutput: Serial Computation");

    const unsigned int Part1 = 50;
    std::stringstream strExpected_Part1;
    for (int i = 0; i < Part1; ++i)
    {
       strExpected_Part1 << hostHistogram[i] << " ";
    }  
    wbLog(TRACE, "expected first part: ", strExpected_Part1.str().c_str());

    std::stringstream strActual_Part1;
    for (int i = 0; i < Part1; ++i)
    {
       strActual_Part1 << hostHistogramExpected[i] << " ";
    }  
    wbLog(TRACE, "actual first part: ", strActual_Part1.str().c_str());

    const unsigned int Part2 = 90;
    std::stringstream strExpected_Part2;
    for (int i = Part1; i < Part2; ++i)
    {
       strExpected_Part2 << hostHistogram[i] << " ";
    }  
    wbLog(TRACE, "expected next part: ", strExpected_Part2.str().c_str());

    std::stringstream strActual_Part2;
    for (int i = Part1; i < Part2; ++i)
    {
       strActual_Part2 << hostHistogramExpected[i] << " ";
    }  
    wbLog(TRACE, "actual next part: ", strActual_Part2.str().c_str());

    free(hostGrayScaleImage);
    free(hostHistogram);
}

void checkComulativeDistributionFunction(unsigned int* deviceHistogram, int imageWidth, int imageHeight, float* deviceComulativeDistributionFunction)
{
    wbTime_start(Copy, "checkComulativeDistributionFunction: Copying data from the GPU");
    unsigned int* hostHistogram = (unsigned int*) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    float* hostComulativeDistributionFunction = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));
    cudaMemcpy(hostComulativeDistributionFunction, deviceComulativeDistributionFunction, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "checkComulativeDistributionFunction: Copying data from the GPU");

    wbTime_start(Copy, "checkComulativeDistributionFunction: Serial Computation");

    float hostComulativeDistributionFunctionExpected[HISTOGRAM_LENGTH];

    hostComulativeDistributionFunctionExpected [ 0 ] = (float) hostHistogram [ 0 ] / (imageWidth * imageHeight); 
    for (unsigned int i = 1; i < HISTOGRAM_LENGTH; ++i)
    {
       hostComulativeDistributionFunctionExpected [ i ] = hostComulativeDistributionFunctionExpected [ i - 1 ] + (float) hostHistogram [ i ] / (imageWidth * imageHeight); 
    }

    wbTime_stop(Copy, "checkComulativeDistributionFunction: Serial Computation");

    const unsigned int Part1 = 50;
    std::stringstream strExpected_Part1;
    for (int i = 0; i < Part1; ++i)
    {
       strExpected_Part1 << hostComulativeDistributionFunctionExpected[i] << " ";
    }  
    wbLog(TRACE, "expected first part: ", strExpected_Part1.str().c_str());

    std::stringstream strActual_Part1;
    for (int i = 0; i < Part1; ++i)
    {
       strActual_Part1 << hostComulativeDistributionFunction[i] << " ";
    }  
    wbLog(TRACE, "actual first part: ", strActual_Part1.str().c_str());

    const unsigned int Part2 = 80;
    std::stringstream strExpected_Part2;
    for (int i = Part1; i < Part2; ++i)
    {
       strExpected_Part2 << hostComulativeDistributionFunctionExpected[i] << " ";
    }  
    wbLog(TRACE, "expected next part: ", strExpected_Part2.str().c_str());

    std::stringstream strActual_Part2;
    for (int i = Part1; i < Part2; ++i)
    {
       strActual_Part2 << hostComulativeDistributionFunction[i] << " ";
    }  
    wbLog(TRACE, "actual next part: ", strActual_Part2.str().c_str());

    free(hostHistogram);
    free(hostComulativeDistributionFunction);
}

void checkMinimumCDF(float* deviceComulativeDistributionFunction, float computedMinimumCDF)
{
    wbTime_start(Copy, "checkMinimumCDF: Copying data from the GPU");
    float* hostComulativeDistributionFunction = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));
    cudaMemcpy(hostComulativeDistributionFunction, deviceComulativeDistributionFunction, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "checkMinimumCDF: Copying data from the GPU");

    wbTime_start(Copy, "checkMinimumCDF: Serial Computation");

    float expectedMinimumCDF = 1.0f;
    for (unsigned int i = 0; i < HISTOGRAM_LENGTH; ++i)
    {
       if (hostComulativeDistributionFunction [ i ] < expectedMinimumCDF)
          expectedMinimumCDF = hostComulativeDistributionFunction [ i ]; 
    }
    
    wbTime_stop(Copy, "checkMinimumCDF: Serial Computation");

    wbLog(TRACE, "expected MinimumCDF: ", expectedMinimumCDF);
    wbLog(TRACE, "actual MinimumCDF: ", computedMinimumCDF);
}

unsigned char clamp(unsigned char x, unsigned char start, unsigned char end)
{
    return std::min(std::max(x, start), end);
}

unsigned char correct_color(float* hostComulativeDistributionFunction, float minimumCDF, unsigned char val) 
{
    return clamp( 255 * ( (hostComulativeDistributionFunction[val] - minimumCDF) / (1 - minimumCDF) ), 0, 255 );
}

void checkCorrectedImage(unsigned char* deviceUcharImage, unsigned char* hostUcharImageCopy, int imageWidth, int imageHeight, int imageChannels, float* deviceComulativeDistributionFunction, float minimumCDF)
{
    wbTime_start(Copy, "checkCorrectedImage: Copying data from the GPU");
    unsigned char* hostUcharImage = (unsigned char*) malloc(imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMemcpy(hostUcharImage, deviceUcharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    float* hostComulativeDistributionFunction = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));
    cudaMemcpy(hostComulativeDistributionFunction, deviceComulativeDistributionFunction, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "checkCorrectedImage: Copying data from the GPU");

    wbTime_start(Copy, "checkCorrectedImage: Serial Computation");

    unsigned char* hostUcharImageExpected = (unsigned char*) malloc(imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    
    for (unsigned int i = 0; i < imageWidth * imageHeight * imageChannels; ++i)
    {
       hostUcharImageExpected [ i ] = correct_color(hostComulativeDistributionFunction, minimumCDF, hostUcharImageCopy [ i ]);
    }

    wbTime_stop(Copy, "checkCorrectedImage: Serial Computation");

    const unsigned int Part1 = 50;
    std::stringstream strExpected_Part1;
    for (int i = 0; i < Part1; ++i)
    {
       strExpected_Part1 << (int) hostUcharImageExpected[i] << " ";
    }  
    wbLog(TRACE, "expected first part: ", strExpected_Part1.str().c_str());

    std::stringstream strActual_Part1;
    for (int i = 0; i < Part1; ++i)
    {
       strActual_Part1 << (int) hostUcharImage[i] << " ";
    }  
    wbLog(TRACE, "actual first part: ", strActual_Part1.str().c_str());

    free(hostUcharImage);
    free(hostComulativeDistributionFunction);
    free(hostUcharImageExpected);
}

float* prepareDeviceInputImageMemory(float* hostInputImageData, int imageHeight, int imageWidth, int imageChannels)
{
    wbTime_start(GPU, "Allocating memory for images in GPU");
    float * deviceInputImageData;
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Allocating memory for images in GPU");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    return deviceInputImageData;
}

unsigned char* convertImageToUnsignedChar(float * deviceInputImageData, int imageHeight, int imageWidth, int imageChannels) 
{
    wbTime_start(GPU, "Allocating memory in GPU to convert input image to unsigned char");
    unsigned char* deviceUcharImage = NULL;
    cudaMalloc((void **) &deviceUcharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    wbTime_stop(GPU, "Allocating memory in GPU to convert input image to unsigned char");

    wbTime_start(Compute, "Convert image to unsigned char");
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid( (imageWidth - 1) / BLOCK_WIDTH + 1, (imageHeight - 1) / BLOCK_WIDTH + 1, 1);
    convertToUnsignedChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUcharImage, imageHeight, imageWidth, imageChannels);
    wbTime_stop(Compute, "Convert image to unsigned char");

    return deviceUcharImage;
}

unsigned char* convertImageToGrayScale(unsigned char* deviceUcharImage, int imageHeight, int imageWidth, int imageChannels) 
{
    wbTime_start(GPU, "Allocating memory in GPU to convert input image to gray scale");
    unsigned char* deviceGrayScaleImage = NULL;
    cudaMalloc((void **) &deviceGrayScaleImage, imageWidth * imageHeight * sizeof(unsigned char));
    wbTime_stop(GPU, "Allocating memory in GPU to convert input image to gray scale");

    wbTime_start(Compute, "Convert image to unsigned char");
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid( (imageWidth - 1) / BLOCK_WIDTH + 1, (imageHeight - 1) / BLOCK_WIDTH + 1, 1);
    convertToGrayScaleImage<<<dimGrid, dimBlock>>>(deviceUcharImage, deviceGrayScaleImage, imageHeight, imageWidth, imageChannels);
    wbTime_stop(Compute, "Convert image to unsigned char");

    return deviceGrayScaleImage;
}

unsigned int* computeHistogram(unsigned char* deviceGrayScaleImage, int imageHeight, int imageWidth)
{
    wbTime_start(GPU, "Allocating memory in GPU for histogram");
    unsigned int* deviceHistogram = NULL;
    cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
    wbTime_stop(GPU, "Allocating memory in GPU for histogram");

    wbTime_start(Compute, "Compute histogram of the image");
    dim3 DimGrid(HISTOGRAM_LENGTH, 1, 1);
    dim3 DimBlock(HISTOGRAM_LENGTH, 1, 1);
    histo_kernel<<<DimGrid, DimBlock>>>(deviceGrayScaleImage, imageHeight * imageWidth, deviceHistogram);
    wbTime_stop(Compute, "Compute histogram of the image");

    return deviceHistogram;
}

float* computeComulativeDistributionFunction(unsigned int* deviceHistogram, int imageHeight, int imageWidth)
{
    wbTime_start(GPU, "Allocating memory in GPU for histogram");
    float* deviceComulativeDistributionFunction = NULL;
    cudaMalloc((void **) &deviceComulativeDistributionFunction, HISTOGRAM_LENGTH * sizeof(float));
    wbTime_stop(GPU, "Allocating memory in GPU for histogram");

    dim3 DimGrid_scan((HISTOGRAM_LENGTH - 1) / (2 * SCAN_BLOCK_SIZE) + 1, 1, 1);
    dim3 DimBlock_scan(SCAN_BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing scan computation");
    
    scan<<< DimGrid_scan, DimBlock_scan >>>(deviceHistogram, deviceComulativeDistributionFunction, HISTOGRAM_LENGTH, imageHeight * imageWidth );

    wbTime_stop(Compute, "Performing scan computation");

    dim3 DimGrid_scan_sumUp(1, 1, 1);
    dim3 DimBlock_scan_sumUp(2 * SCAN_BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing scan_sumUp computation");
    scan_sumUp<<< DimGrid_scan_sumUp, DimBlock_scan_sumUp >>>(deviceComulativeDistributionFunction, HISTOGRAM_LENGTH);
    wbTime_stop(Compute, "Performing scan_sumUp computation");

    cudaDeviceSynchronize();

    return deviceComulativeDistributionFunction;
}

float computeMinimumCDF(float* deviceComulativeDistributionFunction)
{
    wbTime_start(GPU, "Allocating GPU memory for computeMinimumCDF");
    float* deviceMinimumCDF = NULL;
    wbCheck(cudaMalloc((void **) &deviceMinimumCDF, sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory for computeMinimumCDF");

    dim3 DimGrid(HISTOGRAM_LENGTH, 1, 1);
    dim3 DimBlock(HISTOGRAM_LENGTH, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation of Minimum CDF");
    computeMinimumCDF_kernel<<< DimGrid, DimBlock >>>(deviceComulativeDistributionFunction, deviceMinimumCDF, HISTOGRAM_LENGTH);
    wbTime_stop(Compute, "Performing CUDA computation of Minimum CDF");

    cudaDeviceSynchronize();

    float hostMinimumCDF = 0.0f;
    wbCheck(cudaMemcpy(&hostMinimumCDF, deviceMinimumCDF, sizeof(float), cudaMemcpyDeviceToHost));
    return hostMinimumCDF;
}

void applyHistogramEqualizationFunction(unsigned char* deviceUcharImage, int imageHeight, int imageWidth, int imageChannels,
                                        float* deviceComulativeDistributionFunction, float minimumCDF)
{
    wbTime_start(Compute, "Correct color of input image");
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid( (imageWidth - 1) / BLOCK_WIDTH + 1, (imageHeight - 1) / BLOCK_WIDTH + 1, 1);
    correctImageColor<<<dimGrid, dimBlock>>>(deviceUcharImage, deviceComulativeDistributionFunction, minimumCDF, imageHeight, imageWidth, imageChannels);
    wbTime_stop(Compute, "Correct color of input image");
}

float* castBackToFloat(unsigned char* deviceUcharImage, int imageHeight, int imageWidth, int imageChannels)
{
    wbTime_start(GPU, "Allocating memory in GPU to convert unsigned char image back to float");
    float* deviceFloatImage = NULL;
    cudaMalloc((void **) &deviceFloatImage, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Allocating memory in GPU to convert unsigned char image back to float");

    wbTime_start(Compute, "Convert image back to float");
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid( (imageWidth - 1) / BLOCK_WIDTH + 1, (imageHeight - 1) / BLOCK_WIDTH + 1, 1);
    castBackToFloat_kernel<<<dimGrid, dimBlock>>>(deviceUcharImage, deviceFloatImage, imageHeight, imageWidth, imageChannels);
    wbTime_stop(Compute, "Convert image back to float");

    return deviceFloatImage;
}

/*
Kind    Location        Time (ms)       Message
Generic main.cu::254    15.340598       Importing data and creating memory on host
GPU     main.cu::499    0.30188 Allocating memory for images in GPU
Copy    main.cu::504    3.314866        Copying data to the GPU
GPU     main.cu::513    0.121315        Allocating memory in GPU to convert input image to unsigned char
Compute main.cu::518    0.182567        Convert image to unsigned char
GPU     main.cu::529    0.102401        Allocating memory in GPU to convert input image to gray scale
Compute main.cu::534    0.155168        Convert image to unsigned char
GPU     main.cu::545    0.097568        Allocating memory in GPU for histogram
Compute main.cu::550    0.192569        Compute histogram of the image
Copy    main.cu::308    0.476576        checkHistoOutput: Copying data from the GPU
Copy    main.cu::316    0.78233 checkHistoOutput: Serial Computation
GPU     main.cu::561    0.025537        Allocating memory in GPU for histogram
Compute main.cu::571    0.025929        Performing scan computation
Compute main.cu::580    0.014831        Performing scan_sumUp computation
Copy    main.cu::370    0.026425        checkComulativeDistributionFunction: Copying data from the GPU
Copy    main.cu::378    0.00365 checkComulativeDistributionFunction: Serial Computation
GPU     main.cu::591    0.023863        Allocating GPU memory for computeMinimumCDF
Compute main.cu::600    0.03032 Performing CUDA computation of Minimum CDF
Copy    main.cu::426    0.013234        checkMinimumCDF: Copying data from the GPU
Copy    main.cu::431    0.002648        checkMinimumCDF: Serial Computation
Compute main.cu::614    0.215295        Correct color of input image
Copy    main.cu::458    1.5704  checkCorrectedImage: Copying data from the GPU
Copy    main.cu::466    8.081278        checkCorrectedImage: Serial Computation
GPU     main.cu::623    0.198255        Allocating memory in GPU to convert unsigned char image back to float
Compute main.cu::628    0.208209        Convert image back to float
Copy    main.cu::288    3.180634        Copying output image from the GPU

GPU time:
Kind    Location        Time (ms)       Message
Copy    main.cu::504    3.314866        Copying data to the GPU
GPU     main.cu::545    0.097568        Allocating memory in GPU for histogram
Compute main.cu::550    0.192569        Compute histogram of the image
GPU     main.cu::561    0.025537        Allocating memory in GPU for histogram
Compute main.cu::571    0.025929        Performing scan computation
Compute main.cu::580    0.014831        Performing scan_sumUp computation
Compute main.cu::600    0.03032 Performing CUDA computation of Minimum CDF
Compute main.cu::614    0.215295        Correct color of input image
GPU     main.cu::623    0.198255        Allocating memory in GPU to convert unsigned char image back to float
Compute main.cu::628    0.208209        Convert image back to float
Copy    main.cu::288    3.180634        Copying output image from the GPU

CPU time:
Copy    main.cu::316    0.78233 checkHistoOutput: Serial Computation
Copy    main.cu::378    0.00365 checkComulativeDistributionFunction: Serial Computation
Copy    main.cu::431    0.002648        checkMinimumCDF: Serial Computation
Copy    main.cu::466    8.081278        checkCorrectedImage: Serial Computation
*/
