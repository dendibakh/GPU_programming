#include <wb.h> 

int main(int argc, char **argv) 
{
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(Compute, "Performing CUDA computation");

  int dummyInputLength = inputLength;
  #pragma acc parallel copyin(hostInput1[0:inputLength]) copyin(hostInput2[0:inputLength]) copyout(hostOutput[0:inputLength]) gangs(1024) workers(32)
  {
     #pragma acc loop
     for (int i = 0; i < dummyInputLength; i++) 
     {
        hostOutput[i] = hostInput1[i] + hostInput2[i];
     }
  }

  wbTime_stop(Compute, "Performing CUDA computation");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

/*
Kind    Location        Time (ms)       Message
Generic main.cu::13     74.538875       Importing data and creating memory on host
Compute main.cu::21     26.728582       Performing CUDA computation
*/