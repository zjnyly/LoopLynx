// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

#ifndef X_HLS_FENCE_H
#define X_HLS_FENCE_H
#ifdef __SYNTHESIS__
namespace hls {
// one-direction fence
template<unsigned BEFORE_NUM, unsigned AFTER_NUM, class... Args>  
void fence(Args&&... args) {
#pragma HLS inline
   __fpga_fence(BEFORE_NUM, AFTER_NUM, args...);  
}   
 
// bi-direction fence
template <class... Args>
void fence(Args&&... args) {
#pragma HLS inline
  __fpga_fence((int)sizeof...(args), (int)sizeof...(args), args..., args...);
}
}
#else
#include <atomic>
namespace hls {
template<unsigned BEFORE_NUM, unsigned AFTER_NUM, class... Args>
void fence(Args&&... args) {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

template <class... Args>
void fence(Args&&... args) {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}
}
#endif
#endif
