#include </usr/local/cuda/include/CL/cl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

// compile this with: nvcc getDeviceInfo_AMD.cpp -o getDeviceInfo_AMD -lOpenCL

void check_cl_error(cl_int err_num, char *msg) {
  if (err_num != CL_SUCCESS) {
    printf("[Error] OpenCL error code: %d in %s \n", err_num, msg);
    exit(EXIT_FAILURE);
  }
}

int main() {
  cout << "Querying AMD Device Info...\n\n";

  /* Host/device data structures */
  cl_int err;

  char name_data[48];

  /* Identify a platform */
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err < 0) {
    perror("Couldn't find any platforms");
    exit(1);
  }

  /* Determine number of connected devices */
  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, NULL, &num_devices);
  if (err < 0) {
    perror("Couldn't find any devices");
    exit(1);
  }

  /* Access connected device */
  cl_device_id *device = (cl_device_id *)malloc(sizeof(cl_device_id));
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);

  /* Obtain data for the connected device */

  err = clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(name_data), name_data,
                        NULL);
  if (err < 0) {
    perror("Couldn't read extension data");
    exit(1);
  }

  char str_buffer[1024];
  err = clGetDeviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(str_buffer), &str_buffer, NULL);
  check_cl_error(err, "clGetDeviceInfo: Getting device name");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_COMPUTE_UNITS: %s\n",
         platform, device, str_buffer);

  printf("\t\t [Platform %d] Device ID: %d\n", platform, device);
  printf("\t\t ---------------------------\n\n");

  err = clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(str_buffer),
                        &str_buffer, NULL);
  check_cl_error(err, "clGetDeviceInfo: Getting device name");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_NAME: %s\n", platform,
         device, str_buffer);

  // Get device hardware version
  err = clGetDeviceInfo(*device, CL_DEVICE_VERSION, sizeof(str_buffer),
                        &str_buffer, NULL);
  check_cl_error(err, "clGetDeviceInfo: Getting device hardware version");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_VERSION: %s\n", platform,
         device, str_buffer);

  // Get device software version
  err = clGetDeviceInfo(*device, CL_DRIVER_VERSION, sizeof(str_buffer),
                        &str_buffer, NULL);
  check_cl_error(err, "clGetDeviceInfo: Getting device software version");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DRIVER_VERSION: %s\n", platform,
         device, str_buffer);

  // Get device OpenCL C version
  err = clGetDeviceInfo(*device, CL_DEVICE_OPENCL_C_VERSION, sizeof(str_buffer),
                        &str_buffer, NULL);
  check_cl_error(err, "clGetDeviceInfo: Getting device OpenCL C version");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_OPENCL_C_VERSION: %s\n",
         platform, device, str_buffer);

  // Get device max compute units available
  cl_uint max_compute_units_available;
  err = clGetDeviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(max_compute_units_available),
                        &max_compute_units_available, NULL);
  check_cl_error(err,
                 "clGetDeviceInfo: Getting device max compute units available");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_MAX_COMPUTE_UNITS: %d\n",
         platform, device, max_compute_units_available);

  // Get device local mem size
  cl_ulong local_mem_size;
  err = clGetDeviceInfo(*device, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(local_mem_size), &local_mem_size, NULL);
  check_cl_error(err, "clGetDeviceInfo: Getting device local mem size");
  printf("\t\t\t\t [Platform %d] [Device %d] CL_DEVICE_LOCAL_MEM_SIZE: %llu "
         "KB\n",
         platform, device, (unsigned long long)local_mem_size / 1024);

  free(device);

  // ofstream out;
  // out.open("device_data_AMD.py");
  // if (!out) {
  //   cerr << "Error: file could not be opened" << endl;
  //   exit(1);
  // }

  // int warps_per_SM = (int)(deviceProp.maxThreadsPerMultiProcessor /
  //                          deviceProp.maxThreadsPerBlock);
  // out << "Name = \"" << deviceProp.name << "\""
  //     << "\nSMs = " << deviceProp.multiProcessorCount
  //     << "\nwarps_per_SM = " << warps_per_SM
  //     << "\nthreads_per_warp = " << deviceProp.warpSize
  //     << "\nregisters_per_thread_block = " << deviceProp.regsPerBlock
  //     << "\nregisters_per_warp = " << deviceProp.regsPerBlock
  //     << "\ntotal_cuda_cores = "
  //     << cuda_cores_per_SM * deviceProp.multiProcessorCount
  //     << "\ncuda_capability_version = " << deviceProp.major << "."
  //     << deviceProp.minor;

  return 0;
}