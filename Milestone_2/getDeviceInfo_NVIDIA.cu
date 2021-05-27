#include <fstream>
#include <iostream>
using namespace std;

// compile this with: nvcc getDeviceInfo.cu -o getDeviceInfo

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
  cout << "Querying NVIDIA Device Info...\n\n";

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

  ofstream out;
  out.open("device_data.py");
  if (!out) {
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  int cuda_cores_per_SM;
  switch (deviceProp.major) {
  case 2: // Fermi
    if (deviceProp.minor == 1)
      cuda_cores_per_SM = 48;
    else
      cuda_cores_per_SM = 32;
    break;
  case 3: // Kepler
    cuda_cores_per_SM = 192;
    break;
  case 5: // Maxwell
    cuda_cores_per_SM = 128;
    break;
  case 6: // Pascal
    if ((deviceProp.minor == 1) || (deviceProp.minor == 2))
      cuda_cores_per_SM = 128;
    else if (deviceProp.minor == 0)
      cuda_cores_per_SM = 64;
    else
      cout << "Unknown device type\n";
    break;
  case 7: // Volta & Turing
    if ((deviceProp.minor == 0) || (deviceProp.minor == 5))
      cuda_cores_per_SM = 64;
    else
      cout << "Unknown device type\n";
    break;
  case 8: // Ampere
    if (deviceProp.minor == 0)
      cuda_cores_per_SM = 64;
    else if (deviceProp.minor == 6)
      cuda_cores_per_SM = 128;
    else
      cout << "Unknown device type\n";
    break;
  default:
    cout << "Unknown device type\n";
    break;
  }

  int warps_per_SM = (int)(deviceProp.maxThreadsPerMultiProcessor /
                           deviceProp.maxThreadsPerBlock);
  out << "Name = \"" << deviceProp.name << "\""
      << "\nSMs = " << deviceProp.multiProcessorCount
      << "\nwarps_per_SM = " << warps_per_SM
      << "\nthreads_per_warp = " << deviceProp.warpSize
      << "\nregisters_per_thread_block = " << deviceProp.regsPerBlock
      << "\nregisters_per_warp = " << deviceProp.regsPerBlock
      << "\ntotal_cuda_cores = "
      << cuda_cores_per_SM * deviceProp.multiProcessorCount
      << "\ncuda_capability_version = " << deviceProp.major << "."
      << deviceProp.minor;

  out.close();
  return 0;
}