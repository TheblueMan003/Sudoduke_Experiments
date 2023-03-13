#include <immintrin.h>
#include <iostream>
#include <cstring>

int main(){
    __m256i a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i b = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
    __m256i c = _mm256_add_epi32(a, b);

    uint32_t val_a[8];
    memcpy(val_a, &a, sizeof(val_a));
    std::cout << val_a[0] << "," <<  val_a[1] << "," <<  val_a[2] << "," <<  val_a[3] << "," 
              << val_a[4] << "," <<  val_a[5] << "," <<  val_a[6] << "," <<  val_a[7]  << std::endl;

    uint32_t val_b[8];
    memcpy(val_b, &b, sizeof(val_b));
    std::cout << val_b[0] << "," <<  val_b[1] << "," <<  val_b[2] << "," <<  val_b[3] << "," 
              << val_b[4] << "," <<  val_b[5] << "," <<  val_b[6] << "," <<  val_b[7]  << std::endl;

    uint32_t val_c[8];
    memcpy(val_c, &c, sizeof(val_c));
    std::cout << val_c[0] << "," <<  val_c[1] << "," <<  val_c[2] << "," <<  val_c[3] << "," 
              << val_c[4] << "," <<  val_c[5] << "," <<  val_c[6] << "," <<  val_c[7]  << std::endl;
}