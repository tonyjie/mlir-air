// Bit-preserving replay of one chunk from a resident normalized activation.
#include <aie_api/aie.hpp>
#include <cstdint>
extern "C" void normalized_cache_chunk(bfloat16 *restrict out,
                                        const bfloat16 *restrict slab,
                                        int32_t chunk) {
  const bfloat16 *src = slab + chunk * 4096;
  for (int i = 0; i < 4096; i += 64) {
    aie::store_v(out + i, aie::load_v<64>(src + i));
  }
}
