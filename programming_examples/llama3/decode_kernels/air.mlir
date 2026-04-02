#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
#map2 = affine_map<()[s0] -> (s0)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
module {
  func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
  func.func private @linalg_fill_bf16(bf16, memref<2xbf16, 2 : i32>) attributes {link_with = "mv.o", llvm.emit_c_interface}
  func.func @matvec_no_l2(%arg0: memref<2048x8192xbf16>, %arg1: memref<8192xbf16>, %arg2: memref<2048xbf16>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c128, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<2048x8192xbf16>, memref<8192xbf16>, memref<2048xbf16> {
      air.segment @seg  args(%arg10=%arg3, %arg11=%arg7, %arg12=%arg8, %arg13=%arg9) : index, memref<2048x8192xbf16>, memref<8192xbf16>, memref<2048xbf16> {
        %0 = affine.apply #map()[%arg10]
        %alloc = memref.alloc() : memref<1x8192xbf16, 2 : i32>
        %alloc_0 = memref.alloc() : memref<8192xbf16, 2 : i32>
        %alloc_1 = memref.alloc() : memref<2xbf16, 2 : i32>
        %c8 = arith.constant 8 : index
        %c1_2 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c8, %arg17=%c1_2) args(%arg18=%alloc, %arg19=%alloc_0, %arg20=%alloc_1, %arg21=%arg11, %arg22=%arg12, %arg23=%arg13, %arg24=%0) : memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>, memref<2048x8192xbf16>, memref<8192xbf16>, memref<2048xbf16>, index attributes {link_with = "mv.o"} {
          %cst = arith.constant 0.000000e+00 : bf16
          func.call @linalg_fill_bf16(%cst, %arg20) : (bf16, memref<2xbf16, 2 : i32>) -> ()
          %1 = affine.apply #map1()[%arg24, %arg14]
          %c0 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg25 = %c0 to %c2 step %c1_3 {
            %2 = affine.apply #map2()[%arg25]
            %3 = affine.apply #map3()[%1, %2]
            %c0_6 = arith.constant 0 : index
            %c1_7 = arith.constant 1 : index
            %c8192 = arith.constant 8192 : index
            %c8192_8 = arith.constant 8192 : index
            %c1_9 = arith.constant 1 : index
            air.dma_memcpy_nd (%arg18[] [] [], %arg21[%3, %c0_6] [%c1_7, %c8192] [%c8192_8, %c1_9]) : (memref<1x8192xbf16, 2 : i32>, memref<2048x8192xbf16>)
            %c8192_10 = arith.constant 8192 : index
            %c1_11 = arith.constant 1 : index
            air.dma_memcpy_nd (%arg19[] [] [], %arg22[] [%c8192_10] [%c1_11]) : (memref<8192xbf16, 2 : i32>, memref<8192xbf16>)
            %4 = arith.index_cast %2 : index to i32
            %c1_i32 = arith.constant 1 : i32
            %c8192_i32 = arith.constant 8192 : i32
            func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %4, %arg18, %arg19, %arg20) : (i32, i32, i32, memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2xbf16, 2 : i32>) -> ()
          }
          %c2_4 = arith.constant 2 : index
          %c1_5 = arith.constant 1 : index
          air.dma_memcpy_nd (%arg23[%1] [%c2_4] [%c1_5], %arg20[] [] []) : (memref<2048xbf16>, memref<2xbf16, 2 : i32>)
        }
        memref.dealloc %alloc : memref<1x8192xbf16, 2 : i32>
        memref.dealloc %alloc_0 : memref<8192xbf16, 2 : i32>
        memref.dealloc %alloc_1 : memref<2xbf16, 2 : i32>
      }
    }
    return
  }
}
