#include "attention.h"

int main()
{
    int8_pack token_buffer_0[10];
    int8_pack attention_buffer_0[10];
    int8_pack q_weight_buffer_0[10];
    int8_pack k_weight_buffer_0[10];
    int8_pack v_weight_buffer_0[10];
    int8_pack o_weight_buffer_0[10];
    int8_pack mlp_1_weight_buffer_0[10];
    int8_pack mlp_2_weight_buffer_0[10];

    int8_pack token_buffer_1[10];
    int8_pack attention_buffer_1[10];
    int8_pack q_weight_buffer_1[10];
    int8_pack k_weight_buffer_1[10];
    int8_pack v_weight_buffer_1[10];
    int8_pack o_weight_buffer_1[10];
    int8_pack mlp_1_weight_buffer_1[10];
    int8_pack mlp_2_weight_buffer_1[10];

    LoopLynx(
        token_buffer_0, attention_buffer_0, 
        q_weight_buffer_0, k_weight_buffer_0, v_weight_buffer_0, o_weight_buffer_0,
        mlp_1_weight_buffer_0, mlp_2_weight_buffer_0,
        token_buffer_1, attention_buffer_1, 
        q_weight_buffer_1, k_weight_buffer_1, v_weight_buffer_1, o_weight_buffer_1,
        mlp_1_weight_buffer_1, mlp_2_weight_buffer_1);

    return 0;
}