#include <wb.h>
#include <amp.h>

using namespace concurrency;

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  //@@ Insert C++AMP code here

  array_view<float,1> AV(inputLength, hostInput1);
  array_view<float,1> BV(inputLength, hostInput2);
  array_view<float,1> CV(inputLength, hostOutput);
  
  CV.discard_data();
  parallel_for_each(CV.get_extent(), [=](index<1> i)
                      restrict(amp) 
                      {
                         CV[i] = AV[i] + BV[i];
                      }
                   );
  completion_future done = CV.synchronize_async();
  done.get();

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
