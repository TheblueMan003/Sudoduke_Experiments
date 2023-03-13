#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cstring>


int32_t mask = 0b11110011110111101111;
size_t size_source = 8000000;
int32_t* source = new int32_t[size_source];


// Parent class
class Projector {
public:
    virtual int32_t project(size_t source_size, int32_t* source, int32_t* dest, int32_t mask) = 0;
    virtual void init(int32_t mask) = 0;
};

// Base class
class ProjectorBase : public Projector {
public:
    int32_t project(size_t source_size, int32_t* source, int32_t* dest, int32_t mask) override {
        for (size_t key = 0; key < source_size; key++)
        {
            int32_t result = 0;
            int j = 0;
            for(int32_t i = 0; i < 32; i++){
                if(mask & (1 << i)){
                    result |= ((key >> i) & 1) << j;
                    j ++;
                }
            }
            dest[result] += source[key];
        }
        return 0;
    }

    void init(int32_t mask) override {}
};

// Class with optimized init
class ProjectorV1 : public Projector {
private:
    size_t masksum;
    size_t* maskpos;
public:
    int32_t project(size_t source_size, int32_t* source, int32_t* dest, int32_t mask) override {
        for (size_t key = 0; key < source_size; key++)
        {
            size_t to = 0;
            //start at lsb of destination
            for (int wpos = 0; wpos < masksum; wpos++) {
                //get index of source from mask
                int rpos = maskpos[wpos];
                size_t bit = 1L << rpos;
                //read rpos^th bit from source and if 1, set the wpos^th bit of destination
                if (key & bit)
                    to |= 1 << wpos;

            }
            dest[to] += source[key];
        }
        return 0;
    }

    void init(int32_t mask) override {
        masksum = 0;
        maskpos = new size_t[32];
        for (int i = 0; i < 32; i++) {
            if (mask & (1L << i)) {
                maskpos[masksum] = i;
                masksum++;
            }
        }
    }
};

// Class with optimized init and project
class ProjectorV2 : public Projector {
protected:
    size_t masksum;
    size_t* maskpos;
    uint32_t m = 0, mv0, mv1, mv2, mv3, mv4, mv5, morig, mk, mp;
public:
    int32_t project(size_t source_size, int32_t* source, int32_t* dest, int32_t mask) override {
        for (size_t key = 0; key < source_size; key++)
        {
            uint32_t x = key & morig;
            uint32_t t;
            t = x & mv0;
            x = x ^ t | (t >> 1);
            t = x & mv1;
            x = x ^ t | (t >> 2);
            t = x & mv2;
            x = x ^ t | (t >> 4);
            t = x & mv3;
            x = x ^ t | (t >> 8);
            t = x & mv4;
            x = x ^ t | (t >> 16);
            //t = x & mv5;
            //x = x ^ t | (t >> 32);
            dest[x] += source[key];
        }
        return 0;
    }
    void init(int32_t mask) override {
        masksum = 0;
        maskpos = new size_t[64];
        for (int i = 0; i < 64; i++) {
            if (mask & (1L << i)) {
                maskpos[masksum] = i;
                masksum++;
                auto w = i >> 6;
                auto o = i & 63;
                m |= 1uLL << o; //set the 0th mask for word w
            }
        }

        mk, mp, morig = m; //m gets changed here but we want the original m later
        mk = ~m << 1;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        //mp = mp ^ (mp << 32);
        mv0 = mp & m;
        m = m ^ mv0 | (mv0 >> 1);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        //mp = mp ^ (mp << 32);
        mv1 = mp & m;
        m = m ^ mv1 | (mv1 >> 2);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        //mp = mp ^ (mp << 32);
        mv2 = mp & m;
        m = m ^ mv2 | (mv2 >> 4);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
       //mp = mp ^ (mp << 32);
        mv3 = mp & m;
        m = m ^ mv3 | (mv3 >> 8);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        //mp = mp ^ (mp << 32);
        mv4 = mp & m;
        m = m ^ mv4 | (mv4 >> 16);
        mk = mk & ~mp;

        //mp = mk ^ (mk << 1);
        //mp = mp ^ (mp << 2);
        //mp = mp ^ (mp << 4);
        //mp = mp ^ (mp << 8);
        //mp = mp ^ (mp << 16);
        ////mp = mp ^ (mp << 32);
        //mv5 = mp & m;
        //m = m ^ mv5 | (mv5 >> 32);
        //mk = mk & ~mp;
    }
};

// Class with optimized init and project with vecotrized project
class ProjectorV4 : public ProjectorV2 {
public:
    int32_t project(size_t source_size, int32_t* source, int32_t* dest, int32_t mask) override {
        __m256i vi = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i v1 = _mm256_set_epi32(8, 8, 8, 8, 8, 8, 8, 8);
        __m256i vmorign = _mm256_set_epi32(morig, morig, morig, morig, morig, morig, morig, morig);

        __m256i vmv0 = _mm256_set_epi32(mv0, mv0, mv0, mv0, mv0, mv0, mv0, mv0);
        __m256i vmv1 = _mm256_set_epi32(mv1, mv1, mv1, mv1, mv1, mv1, mv1, mv1);
        __m256i vmv2 = _mm256_set_epi32(mv2, mv2, mv2, mv2, mv2, mv2, mv2, mv2);
        __m256i vmv3 = _mm256_set_epi32(mv3, mv3, mv3, mv3, mv3, mv3, mv3, mv3);
        __m256i vmv4 = _mm256_set_epi32(mv4, mv4, mv4, mv4, mv4, mv4, mv4, mv4);
        //__m256i vmv5 = _mm256_set_epi32(mv5, mv5, mv5, mv5, mv5, mv5, mv5, mv5);

        for (size_t i = 0; i < source_size; i += 8)
        {
            __m256i x = _mm256_and_si256(vi, vmorign);
            __m256i t;

            t = _mm256_and_si256(x, vmv0);
            x = _mm256_xor_si256(x, t);
            x = _mm256_or_si256(x, _mm256_srli_epi32(t, 1));

            t = _mm256_and_si256(x, vmv1);
            x = _mm256_xor_si256(x, t);
            x = _mm256_or_si256(x, _mm256_srli_epi32(t, 2));

            t = _mm256_and_si256(x, vmv2);
            x = _mm256_xor_si256(x, t);
            x = _mm256_or_si256(x, _mm256_srli_epi32(t, 4));

            /*t = _mm256_and_si256(x, vmv3);
            x = _mm256_xor_si256(x, t);
            x = _mm256_or_si256(x, _mm256_srli_epi32(t, 8));

            t = _mm256_and_si256(x, vmv4);
            x = _mm256_xor_si256(x, t);
            x = _mm256_or_si256(x, _mm256_srli_epi32(t, 16));*/

            /*t = _mm256_and_si256(x, vmv5);
            x = _mm256_xor_si256(x, t);
            x = _mm256_or_si256(x, _mm256_srli_epi32(t, 32));*/

            uint32_t* val_b = (uint32_t*)&x;
            uint32_t* val_c = (uint32_t*)&vi;

            dest[val_b[0]] += source[val_c[0]];
            dest[val_b[1]] += source[val_c[1]];
            dest[val_b[2]] += source[val_c[2]];
            dest[val_b[3]] += source[val_c[3]];
            dest[val_b[4]] += source[val_c[4]];
            dest[val_b[5]] += source[val_c[5]];
            dest[val_b[6]] += source[val_c[6]];
            dest[val_b[7]] += source[val_c[7]];

            vi = _mm256_add_epi32(vi, v1);
        }
        return 0;
    }
    void init(int32_t mask) override {
        ProjectorV2::init(mask);
    };
};

size_t nb_bit(int32_t mask){
    size_t result = 1;
    for(size_t i = 0; i < 32; i++){
        if(mask & (1 << i)){
            result*=2;
        }
    }
    return result;
}

int32_t* run(Projector &projector, size_t size_source, int32_t* source, int32_t mask){
    size_t size = nb_bit(mask);
    int32_t* result = new int32_t[size];

    projector.init(mask);
    projector.project(size_source, source, result, mask);

    return result;
}

static void BM_V1() {
    ProjectorV1 projector = ProjectorV1();
    int32_t* result_base = run(projector, size_source, source, mask);
    delete result_base;
}

static void BM_V2() {
    ProjectorV2 projector = ProjectorV2();
    int32_t* result_base = run(projector, size_source, source, mask);
    delete result_base;
}

static void BM_V4() {
    ProjectorV4 projector = ProjectorV4();
    int32_t* result_base = run(projector, size_source, source, mask);
    delete result_base;
}

#include <chrono>
#include <ctime>

void Benchmark(const char* name, void(* fct)(void)) {
    // Get Time
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; i++)
    {
        fct();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << name << " - Time: " << diff.count() << " s" << std::endl;
}

void CheckResult(int32_t* result1, int32_t* result2, size_t size){
    for (size_t i = 0; i < size; i++)
    {
        if(result1[i] != result2[i]){
            std::cout << "Error: " << i << " - " << result1[i] << " instead of " << result2[i] << std::endl;
        }
        else{
            std::cout << "Correct: " << i << " - " << result1[i] << " / " << result2[i] << std::endl;
        }
    }
}

void CheckAll(){
    auto size = nb_bit(mask);
    ProjectorBase projectorSelf = ProjectorBase();
    int32_t* vs = run(projectorSelf, size_source, source, mask);

    ProjectorV1 projectorv1 = ProjectorV1();
    int32_t* v1 = run(projectorv1, size_source, source, mask);

    ProjectorV2 projectorv2 = ProjectorV2();
    int32_t* v2 = run(projectorv2, size_source, source, mask);

    ProjectorV4 projectorv4 = ProjectorV4();
    int32_t* v4 = run(projectorv4, size_source, source, mask);

    std::cout << size << std::endl;

    std::cout << "Check vs" << std::endl;
    CheckResult(vs, v1, size);

    std::cout << "Check v2" << std::endl;
    CheckResult(v2, v1, size);

    std::cout << "Check v4" << std::endl;
    CheckResult(v4, v1, size);
}

int main(){
    for(size_t i = 0; i < size_source; i++){
        source[i] = std::rand();
    }
    CheckAll();
    std::cout << "Mask: " << mask << std::endl;
    Benchmark("V1 (Bit by bit)", BM_V1);
    Benchmark("V2 (Smart)", BM_V2);
    Benchmark("V4 (Vectorized Smart)", BM_V4);
}