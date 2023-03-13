#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cstring>
#include <fstream>

int main(){
    int j, k;
    uint64_t m, t;
    m = 0x00000000FFFFFFFFuLL;
    for (j = 32; j != 0; j = j >> 1, m = m ^ (m << j)) {
        for (k = 0; k < 64; k = (k + j + 1) & ~j) {
            std::cout << "t = (A[" << k << "] ^ (A["<< k + j << "] >> "<<j <<")) & "<<m<<";" << std::endl;
            std::cout << "A["<< k << "] = A["<<k<<"] ^ t;" << std::endl;
            std::cout << "A["<<k + j<<"] = A["<<k + j<<"] ^ (t << "<<j<<");" << std::endl;
        }
    }
}