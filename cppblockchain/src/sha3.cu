// src/sha3.cu
#include "sha3.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STATE_SIZE 25
#define BLOCK_SIZE_BYTES 136
#define BLOCK_WORDS 17
#define HASH_WORDS 8
#define ROUNDS 24
#define THREADS_PER_BLOCK 128

__constant__ uint64_t d_RC[ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x0000000000008002ULL, 0x0000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ inline uint64_t ROTL64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f(uint64_t *state) {
    const int rho_offsets[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
    for (int round = 0; round < ROUNDS; round++) {
        uint64_t C[5], D[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            asm("xor.b64 %0, %1, %2;\n\t"
                "xor.b64 %0, %0, %3;\n\t"
                "xor.b64 %0, %0, %4;\n\t"
                "xor.b64 %0, %0, %5;"
                : "=l"(C[x])
                : "l"(state[x]), "l"(state[x + 5]), "l"(state[x + 10]), "l"(state[x + 15]), "l"(state[x + 20]));
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ROTL64(C[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }
        uint64_t B[25];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = x + 5 * y;
                int new_x = y;
                int new_y = (2 * x + 3 * y) % 5;
                B[new_x + 5 * new_y] = ROTL64(state[idx], rho_offsets[idx]);
            }
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = x + 5 * y;
                state[idx] = B[idx] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= d_RC[round];
    }
}

__global__ void sha3_kernel(const uint64_t *d_input, uint64_t *d_output, size_t num_messages, size_t num_blocks) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_messages) return;

    uint64_t state[STATE_SIZE] = {0};
    size_t offset = tid * BLOCK_WORDS * num_blocks;

    #pragma unroll
    for (size_t i = 0; i < num_blocks; i++) {
        for (int j = 0; j < BLOCK_WORDS; j++) {
            state[j] ^= d_input[offset + i * BLOCK_WORDS + j];
        }
        keccak_f(state);
    }

    #pragma unroll
    for (int j = 0; j < HASH_WORDS; j++) {
        d_output[tid * HASH_WORDS + j] = state[j];
    }
}

__global__ void sha3_mining_kernel(const uint64_t *d_input, uint64_t *d_output, int difficulty, uint64_t *d_found_nonce, int *d_found_flag, size_t input_words) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = tid;
    uint64_t state[STATE_SIZE] = {0};

    for (size_t j = 0; j < input_words && j < BLOCK_WORDS; j++) {
        state[j] = d_input[j];
    }
    state[input_words] = nonce;
    if (input_words < BLOCK_WORDS - 2) {
        state[input_words + 1] = 0x06ULL;
        state[BLOCK_WORDS - 1] = 0x8000000000000000ULL;
    }

    keccak_f(state);

    bool valid = true;
    for (int i = 0; i < difficulty / 4; i++) {
        if ((state[i / 8] >> (56 - (i % 8) * 8)) & 0xFF) {
            valid = false;
            break;
        }
    }

    if (valid && atomicCAS(d_found_flag, 0, 1) == 0) {
        *d_found_nonce = nonce;
        for (int j = 0; j < HASH_WORDS; j++) {
            d_output[j] = state[j];
        }
    }
}

extern "C" void compute_sha3(const unsigned char *input, size_t input_len, unsigned char *output) {
    size_t num_blocks = (input_len + BLOCK_SIZE_BYTES - 1) / BLOCK_SIZE_BYTES;
    size_t padded_len = num_blocks * BLOCK_SIZE_BYTES;
    uint64_t *h_input;
    cudaMallocHost(&h_input, padded_len / 8 * sizeof(uint64_t));

    memcpy(h_input, input, input_len);
    if (input_len % BLOCK_SIZE_BYTES != 0) {
        ((unsigned char*)h_input)[input_len] = 0x06;
        memset(((unsigned char*)h_input) + input_len + 1, 0, padded_len - input_len - 1);
        ((unsigned char*)h_input)[padded_len - 1] |= 0x80;
    }

    uint64_t *d_input, *d_output;
    cudaMalloc(&d_input, num_blocks * BLOCK_WORDS * sizeof(uint64_t));
    cudaMalloc(&d_output, HASH_WORDS * sizeof(uint64_t));

    cudaMemcpy(d_input, h_input, num_blocks * BLOCK_WORDS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    sha3_kernel<<<1, THREADS_PER_BLOCK>>>(d_input, d_output, 1, num_blocks);
    cudaMemcpy(output, d_output, HASH_SIZE_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
}

extern "C" void mine_sha3(const unsigned char *input, size_t input_len, unsigned char *output, uint64_t *nonce, int difficulty) {
    size_t num_blocks = 1024;
    size_t padded_len = ((input_len + 7) / 8 + BLOCK_WORDS - 1) / BLOCK_WORDS * BLOCK_WORDS * 8;
    uint64_t *h_input;
    cudaMallocHost(&h_input, padded_len);
    memcpy(h_input, input, input_len);
    size_t input_words = input_len / 8;

    uint64_t *d_input, *d_output, *d_found_nonce;
    int *d_found_flag;
    cudaMalloc(&d_input, padded_len);
    cudaMalloc(&d_output, HASH_WORDS * sizeof(uint64_t));
    cudaMalloc(&d_found_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMemset(d_found_flag, 0, sizeof(int));

    cudaMemcpy(d_input, h_input, padded_len, cudaMemcpyHostToDevice);

    sha3_mining_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, difficulty, d_found_nonce, d_found_flag, input_words);
    cudaDeviceSynchronize();

    int h_found_flag;
    cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found_flag) {
        cudaMemcpy(nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(output, d_output, HASH_SIZE_BYTES, cudaMemcpyDeviceToHost);
    } else {
        printf("No valid nonce found in range. Increase num_blocks.\n");
        *nonce = 0;
        memset(output, 0, HASH_SIZE_BYTES);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
    cudaFreeHost(h_input);
}