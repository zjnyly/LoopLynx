#include "attention.h"

void PE_8_4_pack(
	hls::stream<int8_t>& A_in, hls::stream<int8_t>& A_out,
    hls::stream<int8_t>& B_in, hls::stream<int8_t>& B_out,
	int48_t& C_out
) {
	PE_LOOP:
		ap_int<8> a = A_in.read();
		ap_int<8> b = B_in.read();
		ap_int<17> pack_b = (b(7, 4), ap_uint<13>(0)) + ap_int<17>(b(3, 0));
		ap_int<25> pack_c = a * pack_b;
		C_out += int48_t((pack_c(24, 13), ap_uint<24>(0)) + ap_int<36>(pack_c(11, 0)));
		A_out.write(a);
		B_out.write(b);
}


void systolic_array_k_768(
	hls::stream<int8_pack>& A_loader,
	hls::stream<int8_pack>& B_loader,
	hls::stream<int48_pack>& C_drainer
) {
	hls::stream<int8_t> A_in_buf[SA_SIZE];
    hls::stream<int8_t> B_in_buf[SA_SIZE];
	#pragma HLS STREAM variable=A_in_buf depth=SA_SIZE
    #pragma HLS STREAM variable=B_in_buf depth=SA_SIZE

    hls::stream<int8_t> A_fifo[SA_SIZE][SA_SIZE + 1];
    hls::stream<int8_t> B_fifo[SA_SIZE][SA_SIZE + 1];
    #pragma HLS STREAM variable=A_fifo depth=2
    #pragma HLS STREAM variable=B_fifo depth=2

    int48_t C[SA_SIZE][SA_SIZE] = {0};
    #pragma HLS ARRAY_PARTITION variable = C cyclic factor = SA_SIZE dim = 1
    #pragma HLS ARRAY_PARTITION variable = C cyclic factor = SA_SIZE dim = 2

	#pragma HLS DATAFLOW
	data_in_buffer:for (int k = 0; k < EMBEDDING_SIZE; k++) {
	#pragma HLS PIPELINE II=1
		int8_pack A_temp = A_loader.read();
		for (int m = 0; m < SA_SIZE; m++) {
			A_in_buf[m].write(A_temp.data[m]);
		}
		int8_pack B_temp = B_loader.read();
		for (int n = 0; n < SA_SIZE; n++) {
			B_in_buf[n].write(B_temp.data[n]);
		}
		for (int m = 0; m < SA_SIZE; m++) {
			A_fifo[m][0].write(A_in_buf[m].read());
		}
		for (int n = 0; n < SA_SIZE; n++) {
			B_fifo[n][0].write(B_in_buf[n].read());
		}

	for (int m = 0; m < SA_SIZE; m++) {
	#pragma HLS UNROLL
		for (int n = 0; n < SA_SIZE; n++) {
		#pragma HLS UNROLL
			PE_8_4_pack(A_fifo[m][n], A_fifo[m][n+1], B_fifo[n][m], B_fifo[n][m+1], C[m][n]);
		}
	}

		for (int m = 0; m < SA_SIZE; m++) {
			A_fifo[m][SA_SIZE].read();
		}
		for (int n = 0; n < SA_SIZE; n++) {
			B_fifo[n][SA_SIZE].read();
		}
	}

	data_drain_C: for (int n = 0; n < SA_SIZE; n++) {
	#pragma HLS PIPELINE II=1
		int48_pack C_temp;
		for (int m = 0; m < SA_SIZE; m++) {
			C_temp.data[m] = C[m][n];
		}
		C_drainer.write(C_temp);
	}
}



void gemm_systolic_array_qkv(
	int8_pack token[PROCESSING_NUM],
	int8_pack * W,
	const int32_t bias[EMBEDDING_SIZE],
  	const float s[EMBEDDING_SIZE],
	hls::stream<int8_pack>& outp,
  	int pack_seq_offset
){
#pragma HLS INTERFACE mode=m_axi port=W offset=slave bundle=gmem0 max_widen_bitwidth=512  latency = 1
#pragma HLS INTERFACE mode=ap_memory port=token
#pragma HLS INTERFACE mode=ap_memory port=bias
#pragma HLS INTERFACE mode=ap_memory port=s

	//#pragma HLS INTERFACE mode=m_axi port=W offset=slave max_widen_bitwidth=512  bundle=gmem0 latency = 1
	hls::stream<int8_pack> block_A_loader;
	hls::stream<int8_pack> block_B_loader;
	#pragma HLS STREAM variable=block_A_loader depth=4
    #pragma HLS STREAM variable=block_B_loader depth=4

	hls::stream<int48_pack> block_C_drainer;
	#pragma HLS STREAM variable=block_C_drainer depth=4

	block_gemm:
	for(int jj = 0; jj < PROCESSING_ITER_PACK; jj++){
	#pragma HLS LOOP_FLATTEN
	#pragma HLS DATAFLOW

		init_block_AB:
		for(int k = 0; k < EMBEDDING_SIZE; k++){
		#pragma HLS PIPELINE II=1
			block_A_loader.write(token[k]);
			block_B_loader.write(W[jj * EMBEDDING_SIZE + k]);
//			block_B_loader.write(W[k]);
		}

		systolic_array_k_768(block_A_loader, block_B_loader, block_C_drainer);

		int8_pack outp_data_pack_0;
		int8_pack outp_data_pack_1;
		l_bias_scale_j: for (int j = 0; j < SA_SIZE; j++) {    // L41
		#pragma HLS pipeline II=2
			int48_pack acc_temp = block_C_drainer.read();
			l_bias_scale_i: for (int i = 0; i < SA_SIZE; i++) {    // L40
				int24_t outp0_dp = acc_temp.data[i](47, 24) + int32_t(bias[(jj * SA_SIZE + j) * 2]);
				int24_t outp1_dp = acc_temp.data[i](23, 0) + int32_t(bias[(jj * SA_SIZE + j) * 2 + 1]);
				int8_t outp0 = outp0_dp * s[pack_seq_offset + i];
				int8_t outp1 = outp1_dp * s[pack_seq_offset + i];
				outp_data_pack_0.data[i] = outp0;
				outp_data_pack_1.data[i] = outp1;
			}
			outp.write(outp_data_pack_0);
			outp.write(outp_data_pack_1);
		}
	}
}

