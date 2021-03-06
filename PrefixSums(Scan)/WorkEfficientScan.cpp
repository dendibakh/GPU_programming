// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

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

__global__ void scan(float * input, float * output, int len) 
{
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    __shared__ float XY[2 * BLOCK_SIZE];
    
    unsigned int firstIndexInBlock = threadIdx.x;
    unsigned int secondIndexInBlock = threadIdx.x + blockDim.x;
    unsigned int firstIndexInArray = 2 * blockIdx.x * blockDim.x + firstIndexInBlock;
    unsigned int secondIndexInArray = 2 * blockIdx.x * blockDim.x + secondIndexInBlock;

    //@@ Load a segment of the input vector into shared memory
    if (firstIndexInArray < len)
       XY[firstIndexInBlock] = input[firstIndexInArray];
    else
       XY[firstIndexInBlock] = 0.0f;

    if (secondIndexInArray < len)
       XY[secondIndexInBlock] = input[secondIndexInArray];
    else
       XY[secondIndexInBlock] = 0.0f;

    __syncthreads();

    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) 
    {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if(index < 2 * BLOCK_SIZE)
          XY[index] += XY[index - stride];
       
       __syncthreads();
    }

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) 
    {
       __syncthreads();
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if(index + stride < 2 * BLOCK_SIZE) 
          XY[index + stride] += XY[index];
    }

    __syncthreads();

    if (firstIndexInArray < len)
       output[firstIndexInArray] = XY[firstIndexInBlock];
    if (secondIndexInArray < len)
       output[secondIndexInArray] = XY[secondIndexInBlock];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions
    dim3 DimGrid_scan((numElements - 1) / (2 * BLOCK_SIZE) + 1, 1, 1);
    dim3 DimBlock_scan(BLOCK_SIZE, 1, 1);

    //@@ Modify this to complete the functionality of the scan on the deivce
    wbTime_start(Compute, "Performing scan computation");
    
    scan<<< DimGrid_scan, DimBlock_scan >>>(deviceInput, deviceOutput, numElements);

    wbTime_stop(Compute, "Performing scan computation");

    dim3 DimGrid_scan_sumUp(1, 1, 1);
    dim3 DimBlock_scan_sumUp(2 * BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing scan_sumUp computation");
    scan_sumUp<<< DimGrid_scan_sumUp, DimBlock_scan_sumUp >>>(deviceOutput, numElements);
    wbTime_stop(Compute, "Performing scan_sumUp computation");

    cudaDeviceSynchronize();

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

/*
Kind    Location        Time (ms)       Message
Generic main.cu::94     43.588496       Importing data and creating memory on host
GPU     main.cu::101    0.136614        Allocating GPU memory.
GPU     main.cu::106    0.01427 Clearing output memory.
GPU     main.cu::110    0.043469        Copying input memory to the GPU.
Compute main.cu::119    0.045436        Performing scan computation
Compute main.cu::128    0.019011        Performing scan_sumUp computation
Copy    main.cu::134    0.03538 Copying output memory to the CPU
GPU     main.cu::138    0.086684        Freeing GPU Memory
*/
