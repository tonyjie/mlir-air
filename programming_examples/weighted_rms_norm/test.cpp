//===- test.cpp - Weighted RMS Norm profiling harness ----------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Loads compiled xclbin, runs weighted RMS norm kernel with timing,
// reports latency and bandwidth.
//
// Usage:
//   ./test.exe -x air.xclbin -k MLIR_AIE -i air.insts.bin -M 2048 -N 2048
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATATYPE = std::bfloat16_t;

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i", "path of file containing userspace instructions",
      cxxopts::value<std::string>())("size_m,M", "Number of rows",
                                     cxxopts::value<int>())(
      "size_n,N", "Number of columns (norm dimension)",
      cxxopts::value<int>())("iterations", "Number of timed iterations",
                             cxxopts::value<int>()->default_value("20"))(
      "warmup", "Number of warmup iterations",
      cxxopts::value<int>()->default_value("10"));
}

static inline std::bfloat16_t random_bfloat16_t() {
  return std::bfloat16_t(2.0f * (float)rand() / (float)(RAND_MAX)-1.0f);
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Weighted RMS Norm Profiler");
  cxxopts::ParseResult vm;
  add_default_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  int M = vm["size_m"].as<int>();
  int N = vm["size_n"].as<int>();

  int INPUT_VOLUME = M * N;
  int WEIGHT_VOLUME = N;
  int OUTPUT_VOLUME = M * N;

  int INPUT_SIZE = INPUT_VOLUME * sizeof(DATATYPE);
  int WEIGHT_SIZE = WEIGHT_VOLUME * sizeof(DATATYPE);
  int OUTPUT_SIZE = OUTPUT_VOLUME * sizeof(DATATYPE);

  srand(42);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // XRT setup
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // Buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_weight =
      xrt::bo(device, WEIGHT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_output =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Fill data using bo.map() — zero-copy
  DATATYPE *bufInput = bo_input.map<DATATYPE *>();
  for (int i = 0; i < INPUT_VOLUME; i++)
    bufInput[i] = random_bfloat16_t();

  DATATYPE *bufWeight = bo_weight.map<DATATYPE *>();
  for (int i = 0; i < WEIGHT_VOLUME; i++)
    bufWeight[i] = random_bfloat16_t();

  DATATYPE *bufOutput = bo_output.map<DATATYPE *>();
  memset(bufOutput, 0, OUTPUT_SIZE);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Sync to device
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned n_iterations = vm["iterations"].as<int>();
  unsigned n_warmup_iterations = vm["warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  // Data volume: input(M*N) + weight(N) + output(M*N) in bytes
  float data_bytes = (float)(INPUT_SIZE + WEIGHT_SIZE + OUTPUT_SIZE);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_weight,
                      bo_output);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    if (iter < n_warmup_iterations)
      continue;

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  float avg_time = npu_time_total / n_iterations;

  std::cout << std::endl;
  std::cout << "Weighted RMS Norm: M=" << M << " N=" << N << std::endl;
  std::cout << "Data volume: " << std::fixed << std::setprecision(1)
            << data_bytes / 1e6 << " MB" << std::endl;
  std::cout << std::endl;
  std::cout << "Avg NPU time: " << avg_time << " us" << std::endl;
  std::cout << "Avg bandwidth: " << std::setprecision(2)
            << data_bytes / (avg_time * 1000) << " GB/s" << std::endl;
  std::cout << std::endl;
  std::cout << "Min NPU time: " << npu_time_min << " us" << std::endl;
  std::cout << "Max bandwidth: " << data_bytes / (npu_time_min * 1000)
            << " GB/s" << std::endl;
  std::cout << std::endl;
  std::cout << "Max NPU time: " << npu_time_max << " us" << std::endl;
  std::cout << "Min bandwidth: " << data_bytes / (npu_time_max * 1000)
            << " GB/s" << std::endl;

  return 0;
}
