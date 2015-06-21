// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>
#include <sstream>

void printList(float *list, int N)
{
   std::stringstream str;
   for (int i = 0; i < N; ++i)
   {
      str << list[i] << " ";
   }  
   wbLog(TRACE, str.str().c_str());
}

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void total(float * input, float * output, int len) 
{
    __shared__ float partialSum[2 * BLOCK_SIZE];
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    //@@ Load a segment of the input vector into shared memory
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0.0f;

    if (start + blockDim.x + t < len)
       partialSum[blockDim.x + t] = input[start + blockDim.x + t];
    else
       partialSum[blockDim.x + t] = 0.0f;

    __syncthreads();

    //@@ Traverse the reduction tree
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
       if (t % stride == 0)
          partialSum[2 * t] += partialSum[2 * t + stride];

       __syncthreads();
    }

    //@@ Write the computed sum of the block to the output vector at the correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}

int main(int argc, char ** argv) 
{
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    if (numInputElements % (BLOCK_SIZE << 1) ) 
        numOutputElements++;
    
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory
    wbCheck(cudaMalloc((void **) &deviceInput,  numInputElements *  sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU
    wbCheck(cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions
    dim3 DimGrid((numInputElements - 1) / (2 * BLOCK_SIZE) + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel
    total<<< DimGrid, DimBlock >>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    
        //@@ Copy the GPU memory back to the CPU
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (int ii = 1; ii < numOutputElements; ii++) 
    {
        hostOutput[0] += hostOutput[ii];
    }

    /*
    float result = 0.0f;
    for (int ii = 0; ii < numInputElements; ii++) 
    {
        result += hostInput[ii];
    }
    printList(hostOutput, numOutputElements);
     
    wbLog(TRACE, "Expected result: ", result);
    wbLog(TRACE, "Actual result: ", hostOutput[0]);
    */
        
    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceOutput));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}

/*
Attempt Summary Submit Attempt for Grading
Dataset Id:     5
Created:        2 minutes ago
Status: Correct solution for this dataset.
Timer Output
Kind    Location        Time (ms)       Message
Generic main.cu::75     62.75326        Importing data and creating memory on host
GPU     main.cu::89     0.332663        Allocating GPU memory.
GPU     main.cu::96     0.055505        Copying input memory to the GPU.
Compute main.cu::105    0.047216        Performing CUDA computation
Copy    main.cu::112    0.024024        Copying output memory to the CPU
GPU     main.cu::142    0.151397        Freeing GPU Memory
Logger Output
Level   Location        Message
Trace   main::86        The number of input elements in the input is 12670
Trace   main::87        The number of output elements in the input is 13
*/
