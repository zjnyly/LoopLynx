#pragma once

#include "ap_fixed.h"
#include "hls_half.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "hls_math.h"
#include "hls_task.h"
#include <stdint.h>

#include <cmath>

using namespace std;

const int EMBEDDING_SIZE = 768;
const int NUM_TOKEN = 512;
const int NUM_HEAD = 12;
const int EMBEDDING_HEAD = EMBEDDING_SIZE / NUM_HEAD;
const int PROCESSING_NUM = 8;
const int PROCESSING_ITER = EMBEDDING_SIZE / PROCESSING_NUM;
const int PROCESSING_ITER_HEAD = EMBEDDING_HEAD / PROCESSING_NUM;
const int PROCESSING_ITER_PACK = EMBEDDING_SIZE / PROCESSING_NUM / 2;
const int SA_SIZE = 8;

typedef ap_int<48> int48_t;
typedef ap_int<24> int24_t;
typedef ap_int<12> int12_t;

template <typename T, unsigned N> struct Array {
    typedef T value_type;
    T data[N];
    T &operator[](size_t pos) { return data[pos]; }
    const T &operator[](size_t pos) const { return data[pos]; }
};

typedef Array<int48_t, PROCESSING_NUM> int48_pack;
typedef Array<int8_t, PROCESSING_NUM> int8_pack;
typedef Array<float, PROCESSING_NUM> float_pack;

// typedef array<model_default_t, EMBEDDING_SIZE> token_t;
// typedef nnet::array<model_default_t, PROCESSING_NUM> splited_token_t;
// typedef nnet::array<model_default_t, TOKEN_NUM> attention_t;



// void attention_wrapper(
//     hls::stream<typename CONFIG::token_t> &input_token,
//     hls::stream<typename CONFIG::token_t> &output_token

// );


void LoopLynx(
    int8_pack * token_buffer_0,
    int8_pack * attention_buffer_0,
    int8_pack * q_weight_buffer_0,
    int8_pack * k_weight_buffer_0,
    int8_pack * v_weight_buffer_0,
    int8_pack * o_weight_buffer_0,
    int8_pack * mlp_1_weight_buffer_0,
    int8_pack * mlp_2_weight_buffer_0,
    int8_pack * token_buffer_1,
    int8_pack * attention_buffer_1,
    int8_pack * q_weight_buffer_1,
    int8_pack * k_weight_buffer_1,
    int8_pack * v_weight_buffer_1,
    int8_pack * o_weight_buffer_1,
    int8_pack * mlp_1_weight_buffer_1,
    int8_pack * mlp_2_weight_buffer_1
);
