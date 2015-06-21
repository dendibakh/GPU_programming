#include <wb.h>

#define SEGMENT_SIZE 512

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void vecAdd(float *in1, float *in2, float *out, int len) 
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < len)
           out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) 
{
  wbArg_t args;
  int inputLength;
  float *hostInputA;
  float *hostInputB;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInputA = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInputB = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");

        int segSize = SEGMENT_SIZE * sizeof(float);
        float *device_A0, *device_B0, *device_C0; 
        float *device_A1, *device_B1, *device_C1;
        /*
        wbCheck(cudaMalloc((void**)&device_A0, segSize));
        wbCheck(cudaMalloc((void**)&device_A1, segSize));
        wbCheck(cudaMalloc((void**)&device_B0, segSize));
        wbCheck(cudaMalloc((void**)&device_B1, segSize));
        wbCheck(cudaMalloc((void**)&device_C0, segSize));
        wbCheck(cudaMalloc((void**)&device_C1, segSize));
        */
        
        wbCheck(cudaHostAlloc((void**) &device_A0, segSize, cudaHostAllocDefault));
        wbCheck(cudaHostAlloc((void**) &device_A1, segSize, cudaHostAllocDefault));
        wbCheck(cudaHostAlloc((void**) &device_B0, segSize, cudaHostAllocDefault));
        wbCheck(cudaHostAlloc((void**) &device_B1, segSize, cudaHostAllocDefault));
        wbCheck(cudaHostAlloc((void**) &device_C0, segSize, cudaHostAllocDefault));
        wbCheck(cudaHostAlloc((void**) &device_C1, segSize, cudaHostAllocDefault));
        
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Performing computation.");

        cudaStream_t stream0;
        cudaStream_t stream1;
        cudaStreamCreate(&stream0);
        cudaStreamCreate(&stream1);
        int ElementsNumber0 = 0;
        int ElementsNumber1 = 0;
        int ElementsSize0 = 0;
        int ElementsSize1 = 0;

        for (int i = 0; i < inputLength; i += SEGMENT_SIZE * 2) 
        {
             if (inputLength - i >= SEGMENT_SIZE)
                ElementsNumber0 = SEGMENT_SIZE;
             else
                ElementsNumber0 = inputLength - i;

             ElementsSize0 = ElementsNumber0 * sizeof(float);

             if (inputLength - i >= 2 * SEGMENT_SIZE)
                ElementsNumber1 = SEGMENT_SIZE;
             else
                ElementsNumber1 = inputLength - i - SEGMENT_SIZE;

             ElementsSize1 = ElementsNumber1 * sizeof(float);

             wbLog(TRACE, i);
             wbLog(TRACE, ElementsNumber0);
             wbLog(TRACE, ElementsNumber1);

             if (ElementsNumber0 > 0)
             {
                wbCheck(cudaMemcpyAsync(device_A0, hostInputA + i, ElementsSize0, cudaMemcpyHostToDevice, stream0));
                wbCheck(cudaMemcpyAsync(device_B0, hostInputB + i, ElementsSize0, cudaMemcpyHostToDevice, stream0));
             }
             if (ElementsNumber1 > 0)
             {
                wbCheck(cudaMemcpyAsync(device_A1, hostInputA + i + SEGMENT_SIZE, ElementsSize1, cudaMemcpyHostToDevice, stream1));
                wbCheck(cudaMemcpyAsync(device_B1, hostInputB + i + SEGMENT_SIZE, ElementsSize1, cudaMemcpyHostToDevice, stream1));
             }

             if (ElementsNumber0 > 0)
                vecAdd<<< 1, SEGMENT_SIZE, 0, stream0 >>>(device_A0, device_B0, device_C0, ElementsNumber0);
             if (ElementsNumber1 > 0)
                vecAdd<<< 1, SEGMENT_SIZE, 0, stream1 >>>(device_A1, device_B1, device_C1, ElementsNumber1);
             
             if (ElementsNumber0 > 0)
                wbCheck(cudaMemcpyAsync(hostOutput + i, device_C0, ElementsSize0, cudaMemcpyDeviceToHost, stream0));
             if (ElementsNumber1 > 0)
                wbCheck(cudaMemcpyAsync(hostOutput + i + SEGMENT_SIZE, device_C1, ElementsSize1, cudaMemcpyDeviceToHost, stream1));
        }

  cudaDeviceSynchronize();

  wbTime_stop(GPU, "Performing computation.");

  wbSolution(args, hostOutput, inputLength);

  wbTime_start(GPU, "Freeing GPU Memory");
    
        /*
        wbCheck(cudaFree(device_A0));
        wbCheck(cudaFree(device_A1));
        wbCheck(cudaFree(device_B0));
        wbCheck(cudaFree(device_B1));
        wbCheck(cudaFree(device_C0));
        wbCheck(cudaFree(device_C1));
        */
        wbCheck(cudaFreeHost(device_A0));
        wbCheck(cudaFreeHost(device_A1));
        wbCheck(cudaFreeHost(device_B0));
        wbCheck(cudaFreeHost(device_B1));
        wbCheck(cudaFreeHost(device_C0));
        wbCheck(cudaFreeHost(device_C1));
        

  wbTime_stop(GPU, "Freeing GPU Memory");

  free(hostInputA);
  free(hostInputB);
  free(hostOutput);

  return 0;
}

//Simple memory
/*
Kind    Location        Time (ms)       Message
Generic main.cu::31     20.73786        Importing data and creating memory on host
GPU     main.cu::39     0.173059        Allocating GPU memory.
GPU     main.cu::63     0.217131        Performing computation.
GPU     main.cu::122    0.104572        Freeing GPU Memory
*/

//Pinned memory
/*
Kind    Location        Time (ms)       Message
Generic main.cu::31     20.809407       Importing data and creating memory on host
GPU     main.cu::39     0.608948        Allocating GPU memory.
GPU     main.cu::63     0.156293        Performing computation.
GPU     main.cu::122    0.318201        Freeing GPU Memory
*/

