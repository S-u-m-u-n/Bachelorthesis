#include <iostream>
using namespace std;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      printf("FAIL: call='%s'. Reason:%s\n", #call,                            \
             cudaGetErrorString(status));                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

int *pArgc = NULL;
char **pArgv = NULL;

int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;
  cout << "Querying Device Info...\n\n";

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id)
         << "\n-> " << cudaGetErrorString(error_id) << "\nResult = FAIL\n";
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (0 == deviceCount) {
    cout << "There are no available device(s) that support CUDA\n";
  } else {
    cout << "Detected " << deviceCount << " CUDA Capable device(s)\n";
  }

  int devId = 0;
  CUDA_CHECK(cudaGetDevice(&devId));

  int driverVersion = 0, runtimeVersion = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devId);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  cout << "Info for device " << devId << "\n  === Device attributes === "
       << "\n  | Device Name: " << deviceProp.name
       << "\n  | CUDA Driver Version / Runtime Version: "
       << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / "
       << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10
       << "\n  | CUDA Capability Major/Minor version number: "
       << deviceProp.major << "." << deviceProp.minor
       << "\n  | Number of SMs: " << deviceProp.multiProcessorCount
       << "\n  | Threads per Warp: " << deviceProp.warpSize
       << "\n  === SM attributes === "
       << "\n  | Maximum Number of Threads per SM: "
       << deviceProp.maxThreadsPerMultiProcessor
       << "\n  === Threadblock attributes === "
       << "\n  | Maximum Shared Memory per Block: "
       << deviceProp.sharedMemPerBlock
       << "\n  | Maximum Number of Threads per Block: "
       << deviceProp.maxThreadsPerBlock
       << "\n  | Maximum Number of Registers per Block: "
       << deviceProp.regsPerBlock << "\n  ===" << endl;
}