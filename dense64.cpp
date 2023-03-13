#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cstring>


int64_t mask = 0b01100;
size_t size_source = 800000;
int64_t* source = new int64_t[size_source];


// Parent class
class Projector {
public:
    virtual int64_t project(size_t source_size, int64_t* source, int64_t* dest, int64_t mask) = 0;
    virtual void init(int64_t mask) = 0;
};

// Base class
class ProjectorBase : public Projector {
public:
    int64_t project(size_t source_size, int64_t* source, int64_t* dest, int64_t mask) override {
        for (size_t key = 0; key < source_size; key++)
        {
            int64_t result = 0;
            int j = 0;
            for(int64_t i = 0; i < 64; i++){
                if(mask & (1 << i)){
                    result |= ((key >> i) & 1) << j;
                    j ++;
                }
            }
            dest[result] += source[key];
        }
        return 0;
    }

    void init(int64_t mask) override {}
};

// Class with optimized init
class ProjectorV1 : public Projector {
private:
    size_t masksum;
    size_t* maskpos;
public:
    int64_t project(size_t source_size, int64_t* source, int64_t* dest, int64_t mask) override {
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

    void init(int64_t mask) override {
        masksum = 0;
        maskpos = new size_t[64];
        for (int i = 0; i < 64; i++) {
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
    uint64_t m = 0, mv0, mv1, mv2, mv3, mv4, mv5, morig, mk, mp;
public:
    int64_t project(size_t source_size, int64_t* source, int64_t* dest, int64_t mask) override {
        for (size_t key = 0; key < source_size; key++)
        {
            uint64_t x = key & morig;
            uint64_t t;
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
            t = x & mv5;
            x = x ^ t | (t >> 32);
            dest[x] += source[key];
        }
        return 0;
    }
    void init(int64_t mask) override {
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
        mp = mp ^ (mp << 32);
        mv0 = mp & m;
        m = m ^ mv0 | (mv0 >> 1);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        mp = mp ^ (mp << 32);
        mv1 = mp & m;
        m = m ^ mv1 | (mv1 >> 2);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        mp = mp ^ (mp << 32);
        mv2 = mp & m;
        m = m ^ mv2 | (mv2 >> 4);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        mp = mp ^ (mp << 32);
        mv3 = mp & m;
        m = m ^ mv3 | (mv3 >> 8);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        mp = mp ^ (mp << 32);
        mv4 = mp & m;
        m = m ^ mv4 | (mv4 >> 16);
        mk = mk & ~mp;

        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        //mp = mp ^ (mp << 32);
        mv5 = mp & m;
        m = m ^ mv5 | (mv5 >> 32);
        mk = mk & ~mp;
    }
};

// Class with optimized init and project with variable number of masks
class ProjectorV3 : public ProjectorV2 {
protected:
    int mask_count;
    int* mask_value;
    int* mask_shift;
public:
    int64_t project(size_t source_size, int64_t* source, int64_t* dest, int64_t mask) override {
        for (size_t key = 0; key < source_size; key++)
        {
            uint64_t x = key & morig;
            uint64_t t;
            for (int i = 0; i < mask_count; i++) {
                t = x & mask_value[i];
                x = x ^ t | (t >> (mask_shift[i]));
            }
            dest[x] += source[key];
        }
        return 0;
    }
    void init(int64_t mask) override {
        ProjectorV2::init(mask);
        mask_count = 0;
        mask_value = new int[6];
        mask_shift = new int[6];
        if (mv0 != 0) {
            mask_value[mask_count] = mv0;
            mask_shift[mask_count] = 1;
            mask_count++;
        }
        if (mv1 != 0) {
            mask_value[mask_count] = mv1;
            mask_shift[mask_count] = 2;
            mask_count++;
        }
        if (mv2 != 0) {
            mask_value[mask_count] = mv2;
            mask_shift[mask_count] = 4;
            mask_count++;
        }
        if (mv3 != 0) {
            mask_value[mask_count] = mv3;
            mask_shift[mask_count] = 8;
            mask_count++;
        }
        if (mv4 != 0) {
            mask_value[mask_count] = mv4;
            mask_shift[mask_count] = 16;
            mask_count++;
        }
        if (mv5 != 0) {
            mask_value[mask_count] = mv5;
            mask_shift[mask_count] = 32;
            mask_count++;
        }
    }
};

// Class with optimized init and project with vecotrized project
class ProjectorV4 : public ProjectorV2 {
public:
    int64_t project(size_t source_size, int64_t* source, int64_t* dest, int64_t mask) override {
        __m512i vi = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512i v1 = _mm512_set_epi64(8, 8, 8, 8, 8, 8, 8, 8);
        __m512i vmorign = _mm512_set_epi64(morig, morig, morig, morig, morig, morig, morig, morig);

        __m512i vmv0 = _mm512_set_epi64(mv0, mv0, mv0, mv0, mv0, mv0, mv0, mv0);
        __m512i vmv1 = _mm512_set_epi64(mv1, mv1, mv1, mv1, mv1, mv1, mv1, mv1);
        __m512i vmv2 = _mm512_set_epi64(mv2, mv2, mv2, mv2, mv2, mv2, mv2, mv2);
        __m512i vmv3 = _mm512_set_epi64(mv3, mv3, mv3, mv3, mv3, mv3, mv3, mv3);
        __m512i vmv4 = _mm512_set_epi64(mv4, mv4, mv4, mv4, mv4, mv4, mv4, mv4);
        __m512i vmv5 = _mm512_set_epi64(mv5, mv5, mv5, mv5, mv5, mv5, mv5, mv5);

        for (size_t i = 0; i < source_size; i += 8)
        {
            __m512i x = _mm512_and_si512(vi, vmorign);
            __m512i t;

            t = _mm512_and_si512(x, vmv0);
            x = _mm512_xor_si512(x, t);
            x = _mm512_or_si512(x, _mm512_srli_epi64(t, 1));

            t = _mm512_and_si512(x, vmv1);
            x = _mm512_xor_si512(x, t);
            x = _mm512_or_si512(x, _mm512_srli_epi64(t, 2));

            t = _mm512_and_si512(x, vmv2);
            x = _mm512_xor_si512(x, t);
            x = _mm512_or_si512(x, _mm512_srli_epi64(t, 4));

            t = _mm512_and_si512(x, vmv3);
            x = _mm512_xor_si512(x, t);
            x = _mm512_or_si512(x, _mm512_srli_epi64(t, 8));

            t = _mm512_and_si512(x, vmv4);
            x = _mm512_xor_si512(x, t);
            x = _mm512_or_si512(x, _mm512_srli_epi64(t, 16));

            t = _mm512_and_si512(x, vmv5);
            x = _mm512_xor_si512(x, t);
            x = _mm512_or_si512(x, _mm512_srli_epi64(t, 32));

            uint64_t* val_b = (uint64_t*)&x;
            uint64_t* val_c = (uint64_t*)&vi;

            dest[val_b[0]] += source[val_c[0]];
            dest[val_b[1]] += source[val_c[1]];
            dest[val_b[2]] += source[val_c[2]];
            dest[val_b[3]] += source[val_c[3]];
            dest[val_b[4]] += source[val_c[4]];
            dest[val_b[5]] += source[val_c[5]];
            dest[val_b[6]] += source[val_c[6]];
            dest[val_b[7]] += source[val_c[7]];

            vi = _mm512_add_epi64(vi, v1);
        }
        return 0;
    }
    void init(int64_t mask) override {
        ProjectorV2::init(mask);
    };
};

size_t nb_bit(int64_t mask){
    size_t result = 1;
    for(size_t i = 0; i < 64; i++){
        if(mask & (1 << i)){
            result*=2;
        }
    }
    return result;
}

int64_t* run(Projector &projector, size_t size_source, int64_t* source, int64_t mask){
    size_t size = nb_bit(mask);
    int64_t* result = new int64_t[size];

    projector.init(mask);
    projector.project(size_source, source, result, mask);

    return result;
}


static void BM_V1() {
    ProjectorV1 projector = ProjectorV1();
    int64_t* result_base = run(projector, size_source, source, mask);
    delete result_base;
}

static void BM_V2() {
    ProjectorV2 projector = ProjectorV2();
    int64_t* result_base = run(projector, size_source, source, mask);
    delete result_base;
}

static void BM_V3() {
    ProjectorV3 projector = ProjectorV3();
    int64_t* result_base = run(projector, size_source, source, mask);
    delete result_base;
}

static void BM_V4() {
    ProjectorV4 projector = ProjectorV4();
    int64_t* result_base = run(projector, size_source, source, mask);
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

void CheckResult(int64_t* result1, int64_t* result2, size_t size){
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
    int64_t* vs = run(projectorSelf, size_source, source, mask);

    ProjectorV1 projectorv1 = ProjectorV1();
    int64_t* v1 = run(projectorv1, size_source, source, mask);

    ProjectorV2 projectorv2 = ProjectorV2();
    int64_t* v2 = run(projectorv2, size_source, source, mask);

    ProjectorV3 projectorv3 = ProjectorV3();
    int64_t* v3 = run(projectorv3, size_source, source, mask);

    ProjectorV4 projectorv4 = ProjectorV4();
    int64_t* v4 = run(projectorv4, size_source, source, mask);

    std::cout << size << std::endl;

    std::cout << "Check vs" << std::endl;
    CheckResult(vs, v1, size);

    std::cout << "Check v2" << std::endl;
    CheckResult(v2, v1, size);

    std::cout << "Check v3" << std::endl;
    CheckResult(v3, v1, size);

    std::cout << "Check v4" << std::endl;
    CheckResult(v4, v1, size);
}

int main(){
    for(size_t i = 0; i < size_source; i++){
        source[i] = 1;
    }
    CheckAll();
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << "Mask: " << i << std::endl;
        Benchmark("V1 (Bit by bit)", BM_V1);
        Benchmark("V2 (Smart)", BM_V2);
        Benchmark("V3 (Smart Selected)", BM_V3);
        Benchmark("V4 (Vectorized)", BM_V4);
        mask = (mask << 1) + 1;
    }
}