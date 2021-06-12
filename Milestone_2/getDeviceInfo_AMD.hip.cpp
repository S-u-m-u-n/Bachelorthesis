// https://github.com/ROCm-Developer-Tools/HIP/blob/main/samples/1_Utils/hipInfo/hipInfo.cpp
#include "hip/hip_runtime.h"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace std;

int HIP_CHECK(hipError_t status) {
  if (status != hipSuccess) {
    printf("FAIL: call='%s'. Reason:%s\n", status, hipGetErrorString(status));
    exit(EXIT_FAILURE);
  }
}

void printCompilerInfo() {
#ifdef __HCC__
  printf("compiler: hcc version=%s, workweek (YYWWD) = %u\n", __hcc_version__,
         __hcc_workweek__);
#endif
#ifdef __NVCC__
  printf("compiler: nvcc\n");
#endif
}

int main(int argc, char *argv[]) {
  cout << "Querying AMD device info...\n\n";
  printCompilerInfo();
  int deviceCount = 0;
  hipError_t error_id = hipGetDeviceCount(&deviceCount);

  if (error_id != hipSuccess) {
    cout << "hipGetDeviceCount returned " << static_cast<int>(error_id)
         << "\n-> " << hipGetErrorString(error_id) << "\nResult = FAIL\n";
    exit(EXIT_FAILURE);
  }

  if (0 == deviceCount) {
    cout << "There are no available device(s) that support HIP\n";
  } else {
    cout << "Detected " << deviceCount << " HIP Capable device(s)\n";
  }

  int deviceId = 0;
  HIP_CHECK(hipGetDevice(&deviceId));

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, deviceId);

  int driverVersion = 0, runtimeVersion = 0;
  hipDriverGetVersion(&driverVersion);
  hipRuntimeGetVersion(&runtimeVersion);

  cout << "Info for device " << deviceId << "\n  === Device attributes === "
       << "\n  | Device Name: " << deviceProp.name
       << "\n  | HIP Driver Version / Runtime Version: " << driverVersion / 1000
       << "." << (driverVersion % 100) / 10 << " / " << runtimeVersion / 1000
       << "." << (runtimeVersion % 100) / 10
       << "\n  | HIP Capability Major/Minor version number: "
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

  int warps_per_SM = (int)(deviceProp.maxThreadsPerMultiProcessor /
                           deviceProp.maxThreadsPerBlock);
  // TODO: figure out how to get this number!!
  // number of compute cores per SM = warps_per_SM * warpSize?
  int compute_cores_per_SM = warps_per_SM * deviceProp.warpSize;
  out << "Name = \"" << deviceProp.name << "\""
      << "\nSMs = " << deviceProp.multiProcessorCount
      << "\nwarps_per_SM = " << warps_per_SM
      << "\nthreads_per_warp = " << deviceProp.warpSize
      << "\nregisters_per_thread_block = " << deviceProp.regsPerBlock
      << "\nregisters_per_warp = " << deviceProp.regsPerBlock
      << "\ntotal_compute_cores = "
      << compute_cores_per_SM * deviceProp.multiProcessorCount
      << "\ncapability_version = " << deviceProp.major << "."
      << deviceProp.minor;

  out.close();
  return 0;
}