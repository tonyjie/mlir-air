#map = affine_map<()[s0] -> (0)>
#map1 = affine_map<()[s0] -> (s0 * 65536)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 512)>
#map3 = affine_map<()[s0] -> (s0)>
#map4 = affine_map<()[s0] -> (s0 * 128)>
module {
  func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<512xbf16, 2 : i32>, memref<512xbf16, 2 : i32>, memref<128xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
  func.func private @linalg_fill_bf16(bf16, memref<128xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
  func.func @matvec_multi_col(%arg0: memref<262144xbf16>, %arg1: memref<512xbf16>, %arg2: memref<512xbf16>) {
    air.launch () in () args(%arg3=%arg0, %arg4=%arg1, %arg5=%arg2) : memref<262144xbf16>, memref<512xbf16>, memref<512xbf16> {
      air.segment @gemv_seg  args(%arg6=%arg3, %arg7=%arg4, %arg8=%arg5) : memref<262144xbf16>, memref<512xbf16>, memref<512xbf16> {
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        air.herd @gemv_herd  tile (%arg9, %arg10) in (%arg11=%c1, %arg12=%c4) args(%arg13=%arg6, %arg14=%arg7, %arg15=%arg8) : memref<262144xbf16>, memref<512xbf16>, memref<512xbf16> {
          %alloc = memref.alloc() : memref<512xbf16, 2 : i32>
          %alloc_0 = memref.alloc() : memref<512xbf16, 2 : i32>
          %alloc_1 = memref.alloc() : memref<128xbf16, 2 : i32>
          %0 = affine.apply #map()[%arg10]
          %c512 = arith.constant 512 : index
          %c1_2 = arith.constant 1 : index
          air.dma_memcpy_nd (%alloc_0[] [] [], %arg14[%0] [%c512] [%c1_2]) : (memref<512xbf16, 2 : i32>, memref<512xbf16>)
          %cst = arith.constant 0.000000e+00 : bf16
          func.call @linalg_fill_bf16(%cst, %alloc_1) : (bf16, memref<128xbf16, 2 : i32>) -> ()
          %1 = affine.apply #map1()[%arg10]
          %c0 = arith.constant 0 : index
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg16 = %c0 to %c128 step %c1_3 {
            %3 = affine.apply #map2()[%1, %arg16]
            %c512_6 = arith.constant 512 : index
            %c1_7 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc[] [] [], %arg13[%3] [%c512_6] [%c1_7]) : (memref<512xbf16, 2 : i32>, memref<262144xbf16>)
            %4 = affine.apply #map3()[%arg16]
            %5 = arith.index_cast %4 : index to i32
            %c1_i32 = arith.constant 1 : i32
            %c512_i32 = arith.constant 512 : i32
            func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c512_i32, %5, %alloc, %alloc_0, %alloc_1) : (i32, i32, i32, memref<512xbf16, 2 : i32>, memref<512xbf16, 2 : i32>, memref<128xbf16, 2 : i32>) -> ()
          }
          %2 = affine.apply #map4()[%arg10]
          %c128_4 = arith.constant 128 : index
          %c1_5 = arith.constant 1 : index
          air.dma_memcpy_nd (%arg15[%2] [%c128_4] [%c1_5], %alloc_1[] [] []) : (memref<512xbf16>, memref<128xbf16, 2 : i32>)
          memref.dealloc %alloc : memref<512xbf16, 2 : i32>
          memref.dealloc %alloc_0 : memref<512xbf16, 2 : i32>
          memref.dealloc %alloc_1 : memref<128xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
