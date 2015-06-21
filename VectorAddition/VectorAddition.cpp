#include <wb.h>

void cudaLog(cudaError_t err)
{
   if (err != cudaSuccess) 
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));
}

__global__ void vecAdd(float *in1, float *in2, float *out, int len) 
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < len)
           out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");

        int size = inputLength * sizeof(hostInput1);
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void**) &deviceInput1, size);
        cudaLog(err);
        err = cudaMalloc((void**) &deviceInput2, size);
        cudaLog(err);
        err = cudaMalloc((void**) &deviceOutput, size);
        cudaLog(err);
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
    
        err = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
        cudaLog(err);
        err = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
        cudaLog(err);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

        dim3 DimGrid(ceil ((inputLength - 1) / 256 + 1), 1, 1);
        dim3 DimBlock(256, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
 
        vecAdd<<< DimGrid, DimBlock >>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
        
        err = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
        cudaLog(err);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
    
        err = cudaFree(deviceInput1);
        cudaLog(err);
        err = cudaFree(deviceInput2);
        cudaLog(err);
        err = cudaFree(deviceOutput);
        cudaLog(err);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

/*
Kind    Location        Time (ms)       Message
Generic main.cu::28     20.906594       Importing data and creating memory on host
GPU     main.cu::36     0.145519        Allocating GPU memory.
GPU     main.cu::49     0.049223        Copying input memory to the GPU.
Compute main.cu::61     0.035521        Performing CUDA computation
Copy    main.cu::68     0.029193        Copying output memory to the CPU
GPU     main.cu::75     0.093408        Freeing GPU Memory
*/
