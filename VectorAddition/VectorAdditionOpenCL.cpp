#include <wb.h> //@@ wb include opencl.h for you
#include <sstream>

#define wbCheck(stmt) do {                                                    \
        if (stmt != CL_SUCCESS) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got error ...  ", stmt);    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

//@@ OpenCL Kernel
  const char* vaddsrc =
"__kernel void vadd(__global const float *a, __global const float *b, __global float *result, const int len)     \
{                                                                                                                \
   int id = get_global_id(0);                                                                                    \
   if (id < len)                                                                                                 \
      result[id] = a[id] + b[id];                                                                                \
} ";

int main(int argc, char **argv) 
{
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  cl_int clerr = CL_SUCCESS;
  cl_uint numPlatforms;
  clerr = clGetPlatformIDs(0, NULL, &numPlatforms); wbCheck(clerr);
  cl_platform_id platform = NULL;

  if(numPlatforms > 0)
  {
      cl_platform_id* platforms = (cl_platform_id *)malloc (numPlatforms * sizeof(cl_platform_id));
      clerr = clGetPlatformIDs(numPlatforms, platforms, NULL); wbCheck(clerr);
      for(unsigned int i=0; i < numPlatforms; ++i)
      {
          char pbuff[100];
          clerr = clGetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL); wbCheck(clerr);
          platform = platforms[i];
          if(!strcmp(pbuff, "Advanced Micro Devices, Inc."))
          {
              break;
          }
      }
      free(platforms);
  }

  if(NULL == platform)
  {
      wbLog(ERROR, "NULL platform found so Exiting Application.");
      return -1;
  }

  /*
   * If we could find our platform, use it. Otherwise use just available platform.
   */

  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

  cl_context clctx = clCreateContextFromType(cps, CL_DEVICE_TYPE_ALL, NULL, NULL, &clerr); wbCheck(clerr);
  
  size_t parmsz;
  clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz); wbCheck(clerr);
  
  cl_device_id* cldevs = (cl_device_id *) malloc(parmsz);
  clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL); wbCheck(clerr);
  cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr); wbCheck(clerr);

  cl_program clpgm;
  clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr); wbCheck(clerr);

  char clcompileflags[4096];
  sprintf(clcompileflags, "-cl-mad-enable");
  
  clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL); wbCheck(clerr);
  
  cl_kernel clkern = clCreateKernel(clpgm, "vadd", &clerr); wbCheck(clerr);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  cl_mem deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float), hostInput1, &clerr); wbCheck(clerr);
  cl_mem deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float), hostInput2, &clerr); wbCheck(clerr);
  cl_mem deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, inputLength * sizeof(float), NULL, &clerr); wbCheck(clerr);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  clkern = clCreateKernel(clpgm, "vadd", NULL); 

  wbCheck(clSetKernelArg(clkern, 0, sizeof(cl_mem),(void *)&deviceInput1));
  wbCheck(clSetKernelArg(clkern, 1, sizeof(cl_mem),(void *)&deviceInput2));
  wbCheck(clSetKernelArg(clkern, 2, sizeof(cl_mem),(void *)&deviceOutput));
  wbCheck(clSetKernelArg(clkern, 3, sizeof(int), &inputLength));
  
  cl_event event = NULL;
  size_t globalWorkSize[3] = { inputLength, 1, 1 };
  size_t localWorkSize[3]  = { 1, 1, 1 };
  clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
  clerr = clWaitForEvents(1, &event);
  clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0, inputLength * sizeof(float), hostOutput, 0, NULL, NULL);
  
  //cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  /*
  std::stringstream strActual;
  for (int i = 0; i < 10; ++i)
  {
     strActual << (float) hostOutput[i] << " ";
  }  
  wbLog(TRACE, "actual : ", strActual.str().c_str());

  std::stringstream strExpected;
  for (int i = 0; i < 10; ++i)
  {
     strExpected << (float) hostInput1[i] + hostInput2[i] << " ";
  }  
  wbLog(TRACE, "expected : ", strExpected.str().c_str());
  */

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

/*
Kind    Location        Time (ms)       Message
Generic main.cu::31     74.611745       Importing data and creating memory on host
GPU     main.cu::92     0.044992        Allocating GPU memory.
GPU     main.cu::101    0.002444        Copying input memory to the GPU.
Compute main.cu::108    0.172068        Performing CUDA computation
Copy    main.cu::127    0.002491        Copying output memory to the CPU
GPU     main.cu::132    0.016664        Freeing GPU Memory
*/