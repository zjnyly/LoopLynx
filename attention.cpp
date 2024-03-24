#include "attention.h"
#include "systolic_array.cpp"

// template <typename CONFIG>
// void gelu(hls::stream<typename CONFIG::mlp_token_t> &input, hls::stream<typename CONFIG::mlp_token_t> &output)
// {
// GELU:
//     for (int i_token = 0; i_token < CONFIG::TOKEN_NUM; i_token++)
//     {
//         auto data_pack = input.read();
//         typename CONFIG::mlp_token_t res;

//         // GELU(x)=0.5∗x∗(1+Tanh((2/π)∗(x+0.044715∗x^3)))
//     COMPUTE_GELU:
//         for (int i_emb = 0; i_emb < CONFIG::MLP_EMBEDDING_SIZE; i_emb++)
//         {
// #pragma HLS PIPELINE
//             float ele = (float)data_pack[i_emb];
//             float ans = ele * 0.5 * (1 + tanh(0.7978845608028654 * (ele + 0.044715 * ele * ele * ele)));
//             res[i_emb] = (model_default_t)ans;
//         }
//         output.write(res);
//     }
// }

void layerNorm_Atten(
    hls::stream<float_pack> &input,
    hls::stream<float_pack> &output)
{
    const float gamma[EMBEDDING_SIZE] = {1};
    const float beta[EMBEDDING_SIZE] = {0};

    float mean[PROCESSING_NUM] = {0};
    float var[PROCESSING_NUM] = {0};

    float_pack buffer[EMBEDDING_SIZE];

    float_pack data_pack_in;
    float_pack data_pack_out;

layerNorm_Attn_Prepare:
    for (int i_emb = 0; i_emb < EMBEDDING_SIZE; i_emb++)
    {
        data_pack_in = input.read();
        for (int i_pack = 0; i_pack < PROCESSING_NUM; i_pack++)
        {
#pragma HLS PIPELINE II = 1
            auto data = data_pack_in[i_pack];
            buffer[i_emb][i_pack] = data;
            mean[i_pack] += data;
            var[i_pack] += data * data;
        }
    }

layerNorm_Attn_Compute:
    for (int i_pack = 0; i_pack < PROCESSING_NUM; i_pack++)
    {
#pragma HLS PIPELINE II = 1
        float real_mean = mean[i_pack] * (1 / EMBEDDING_SIZE);
        float real_mean_double = real_mean * real_mean;
        mean[i_pack] = real_mean;
        var[i_pack] = var[i_pack] * (1 / EMBEDDING_SIZE) - real_mean_double;
    }

    layerNorm_Attn_Apply:
    for (int i_emb = 0; i_emb < EMBEDDING_SIZE; i_emb++)
    {
        for (int i_pack = 0; i_pack < PROCESSING_NUM; i_pack++)
        {
#pragma HLS PIPELINE II = 1
            float data = buffer[i_emb][i_pack];
            float normalize_mean = (data - mean[i_pack]);
            float sqrt_var = sqrt(var[i_pack])+(float)0.00001;
            float normalize_var = normalize_mean / sqrt_var;
            float affine = normalize_var * gamma[i_emb] + beta[i_emb];
            data_pack_out[i_pack] = affine;
        }
        output.write(data_pack_out);
    }
}

void streamCopyQKV(
    hls::stream<int8_pack> &input,
    hls::stream<int8_pack> &output_q,
    hls::stream<int8_pack> &output_k,
    hls::stream<int8_pack> &output_v
)
{
    layerNorm_Attn_Prepare:
    for (int i_emb = 0; i_emb < EMBEDDING_SIZE; i_emb++)
    {
        auto data_pack = input.read();
        output_q.write(data_pack);
        output_k.write(data_pack);
        output_v.write(data_pack);
    }
}



void compute_Q(
  hls::stream<int8_pack>& inp,
  int8_pack * W,
  hls::stream<int8_pack>& outp,
  int pack_seq_offset,
  int layer
) { 
#pragma HLS INTERFACE mode=m_axi port=W offset=slave bundle=gmem0 max_widen_bitwidth=512  latency = 1
	int8_pack inp_buf[EMBEDDING_SIZE];
	const float s[EMBEDDING_SIZE] = {1.0};
	const int32_t B[EMBEDDING_SIZE] = {1};

  l_j1: for (int j1 = 0; j1 < EMBEDDING_SIZE; j1++) {    // L19
    #pragma HLS pipeline II=1
    inp_buf[j1] = inp.read();
  }
  gemm_systolic_array_qkv(inp_buf, W, B, s, outp, pack_seq_offset);
}


void Vortex(
    int8_pack * token_buffer,
    int8_pack * attention_buffer,
    int8_pack * q_weight_buffer,
    int8_pack * k_weight_buffer,
    int8_pack * v_weight_buffer,
    int8_pack * o_weight_buffer,
    int8_pack * mlp_1_weight_buffer,
    int8_pack * mlp_2_weight_buffer,
    hls::stream<int8_pack> & ring_in,
    hls::stream<int8_pack> & ring_out
)
{
    token_buffer[0] = attention_buffer[0];
    q_weight_buffer[0] = k_weight_buffer[0];
    v_weight_buffer[0] = o_weight_buffer[0];
    mlp_1_weight_buffer[0] = mlp_2_weight_buffer[0];
    ring_out.write(ring_in.read());
}


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
)
{
#pragma HLS INTERFACE mode=m_axi port=token_buffer_0 offset=slave bundle=gmem0 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=attention_buffer_0 offset=slave bundle=gmem1 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=q_weight_buffer_0 offset=slave bundle=gmem2 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=k_weight_buffer_0 offset=slave bundle=gmem3 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=v_weight_buffer_0 offset=slave bundle=gmem4 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=o_weight_buffer_0 offset=slave bundle=gmem5 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=mlp_1_weight_buffer_0 offset=slave bundle=gmem6 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=mlp_2_weight_buffer_0 offset=slave bundle=gmem7 max_widen_bitwidth=512 latency=1 depth=8

#pragma HLS INTERFACE mode=m_axi port=token_buffer_1 offset=slave bundle=gmem8 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=attention_buffer_1 offset=slave bundle=gmem9 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=q_weight_buffer_1 offset=slave bundle=gmem10 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=k_weight_buffer_1 offset=slave bundle=gmem11 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=v_weight_buffer_1 offset=slave bundle=gmem12 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=o_weight_buffer_1 offset=slave bundle=gmem13 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=mlp_1_weight_buffer_1 offset=slave bundle=gmem14 max_widen_bitwidth=512 latency=1 depth=8
#pragma HLS INTERFACE mode=m_axi port=mlp_2_weight_buffer_1 offset=slave bundle=gmem15 max_widen_bitwidth=512 latency=1 depth=8


    hls_thread_local hls::stream<int8_pack> ring_0_1;
    hls_thread_local hls::stream<int8_pack> ring_1_0;

    hls_thread_local hls::task Vortex_0(
        Vortex, 
        token_buffer_0, attention_buffer_0, 
        q_weight_buffer_0, k_weight_buffer_0, v_weight_buffer_0, o_weight_buffer_0,
        mlp_1_weight_buffer_0, mlp_2_weight_buffer_0, ring_1_0, ring_0_1);
    
    hls_thread_local hls::task Vortex_1(
        Vortex, 
        token_buffer_1, attention_buffer_1, 
        q_weight_buffer_1, k_weight_buffer_1, v_weight_buffer_1, o_weight_buffer_1,
        mlp_1_weight_buffer_1, mlp_2_weight_buffer_1, ring_0_1, ring_1_0); 
    
}






// void head_merger(
//     hls::stream<int8_pack>& input_blocks,
//     hls::stream<int8_pack>& out_
// )
// {
//     for (int i_emb = 0; i_emb < EMBEDDING_SIZE; i_emb++)
//     {

//     }
// }




// void attention()
// {
//     layerNorm_Attn
// }





// template <class data_T, class res_T, typename CONFIG, typename CONFIG_QKV, typename CONFIG_PROJ, typename CONFIG_MLP_1, typename CONFIG_MLP_2, typename CONFIG_SOFTMAX>
// void attention(hls::stream<typename CONFIG::token_t> &input_token_stream,
//                hls::stream<typename CONFIG::token_t> &output_token_stream)
// {

//     hls::stream<typename CONFIG::token_t> attentin_input("attentin_input");
// #pragma HLS STREAM variable = attentin_input depth = 1
//     hls::stream<typename CONFIG::token_t> attentin_input_bypass("attentin_input_bypass");
// #pragma HLS STREAM variable = attentin_input_bypass depth = 1

//     hls::stream<typename CONFIG::token_t> mlp_input("mlp_input");
// #pragma HLS STREAM variable = mlp_input depth = 1
//     hls::stream<typename CONFIG::token_t> mlp_input_bypass("mlp_input_bypass");
// #pragma HLS STREAM variable = mlp_input_bypass depth = 1

//     hls::stream<typename CONFIG::splited_token_t> splited_token_in_1("splited_token_in_1");
// #pragma HLS STREAM variable = splited_token_in_1 depth = 1
//     hls::stream<typename CONFIG::splited_token_t> splited_token_out_1("splited_token_out_1");
// #pragma HLS STREAM variable = splited_token_out_1 depth = 1

//     hls::stream<typename CONFIG::splited_token_t> splited_token_in_2("splited_token_in_2");
// #pragma HLS STREAM variable = splited_token_in_2 depth = 1

//     hls::stream<typename CONFIG::splited_token_t> splited_token_out_2("splited_token_out_2");
// #pragma HLS STREAM variable = splited_token_out_2 depth = 1

//     hls::stream<typename CONFIG::token_t> layernorm_in_2;
// #pragma HLS STREAM variable = layernorm_in_2 depth = 1

//     hls::stream<typename CONFIG::token_t> layernorm_out_1;
// #pragma HLS STREAM variable = layernorm_out_1 depth = 1

//     hls::stream<typename CONFIG::token_t> layernorm_out_2;
// #pragma HLS STREAM variable = layernorm_out_2 depth = 1

//     hls::stream<typename CONFIG::token_t> layernorm_pipe;
// #pragma HLS STREAM variable = layernorm_pipe depth = 1

//     hls::stream<typename CONFIG::token_t> q_in_pipe;
// #pragma HLS STREAM variable = q_in_pipe depth = 1

//     hls::stream<typename CONFIG::token_t> k_in_pipe;
// #pragma HLS STREAM variable = k_in_pipe depth = 1

//     hls::stream<typename CONFIG::token_t> v_in_pipe;
// #pragma HLS STREAM variable = v_in_pipe depth = 1

//     // hls::stream<typename CONFIG::splited_token_t> q_out_pipe[CONFIG::HEADS];
//     hls::stream<typename CONFIG::token_t> q_out_pipe;
// #pragma HLS STREAM variable = q_out_pipe depth = 1

//     // hls::stream<typename CONFIG::splited_token_t> k_out_pipe[1];
//     hls::stream<typename CONFIG::token_t> k_out_pipe;
// #pragma HLS STREAM variable = k_out_pipe depth = 1

//     // hls::stream<typename CONFIG::splited_token_t> v_out_pipe[1];
//     hls::stream<typename CONFIG::token_t> v_out_pipe;
// #pragma HLS STREAM variable = v_out_pipe depth = 1

//     hls::stream<typename CONFIG::attention_t> attention_pipe;
// #pragma HLS STREAM variable = attention_pipe depth = 1

//     hls::stream<typename CONFIG::token_t> proj_in_pipe;
// #pragma HLS STREAM variable = proj_in_pipe depth = 1

//     hls::stream<typename CONFIG::token_t> proj_out_pipe;
// #pragma HLS STREAM variable = proj_out_pipe depth = 1

//     // hls::stream<typename CONFIG::token_t> proj_in_pipe;
//     hls::stream<typename CONFIG::mlp_token_t> mlp_1_out_pipe("mlp_1_out_pipe");
// #pragma HLS STREAM variable = mlp_1_out_pipe depth = 1

//     hls::stream<typename CONFIG::mlp_token_t> gelu_mlp2_pipe("gelu_mlp2_pipe");
// #pragma HLS STREAM variable = gelu_mlp2_pipe depth = 1

//     hls::stream<typename CONFIG::token_t> mlp2_add_pipe("mlp2_add_pipe");
// #pragma HLS STREAM variable = mlp2_add_pipe depth = 1

// #pragma HLS DATAFLOW
//     stream_copy<CONFIG>(input_token_stream, attentin_input, attentin_input_bypass);

//     token_spliter<CONFIG>(attentin_input, splited_token_in_1);
//     layerNorm<CONFIG>(splited_token_in_1, splited_token_out_1, LN_weight_1, LN_bias_1);

//     token_merger<CONFIG>(splited_token_out_1, layernorm_out_1);

//     stream_copy_qkv<CONFIG>(layernorm_out_1, q_in_pipe, k_in_pipe, v_in_pipe);

//     compute_Q<CONFIG>(q_in_pipe, q_out_pipe);
//     compute_K<CONFIG>(k_in_pipe, k_out_pipe);
//     compute_V<CONFIG>(v_in_pipe, v_out_pipe);

//     flash_attention_softmax_wrapper<CONFIG>(q_out_pipe, k_out_pipe, v_out_pipe, attention_pipe);

//     compute_AV<CONFIG>(attention_pipe, proj_in_pipe);

//     compute_O<CONFIG>(proj_in_pipe, proj_out_pipe);

//     add<CONFIG>(proj_out_pipe, attentin_input_bypass, layernorm_in_2);

//     stream_copy<CONFIG>(layernorm_in_2, mlp_input, mlp_input_bypass);

//     token_spliter<CONFIG>(mlp_input, splited_token_in_2);
//     layerNorm<CONFIG>(splited_token_in_2, splited_token_out_2, LN_weight_2, LN_bias_2);

//     token_merger<CONFIG>(splited_token_out_2, layernorm_out_2);

//     compute_MLP_1<CONFIG>(layernorm_out_2, mlp_1_out_pipe);

//     gelu<CONFIG>(mlp_1_out_pipe, gelu_mlp2_pipe);

//     compute_MLP_2<CONFIG>(gelu_mlp2_pipe, mlp2_add_pipe);

//     add<CONFIG>(mlp2_add_pipe, mlp_input_bypass, output_token_stream);
// }

// void attention_wrapper(
//     hls::stream<typename CONFIG::token_t> &input_token,
//     hls::stream<typename CONFIG::token_t> &output_token

// )
// {
// }


