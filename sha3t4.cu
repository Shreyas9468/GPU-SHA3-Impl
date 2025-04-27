#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// SHA-3 constants
#define STATE_SIZE 25
#define ROUNDS 24

// Round constants—SHA-3 standard
__constant__ uint64_t d_RC[ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x0000000000008002ULL, 0x0000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotate left for ρ step
__device__ inline uint64_t ROTL64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600]—core SHA-3 transform
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

// Kernel—processes one message per thread
__global__ void sha3_kernel(const uint64_t *d_input, uint64_t *d_output, size_t num_messages, size_t num_blocks, int block_words, int hash_words) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_messages) return;

    uint64_t state[STATE_SIZE] = {0};
    size_t offset = tid * block_words * num_blocks; // Coalesced access

    #pragma unroll
    for (size_t i = 0; i < num_blocks; i++) {
        for (int j = 0; j < block_words; j++) {
            state[j] ^= d_input[offset + i * block_words + j];
        }
        keccak_f(state);
    }

    #pragma unroll
    for (int j = 0; j < hash_words; j++) {
        d_output[tid * hash_words + j] = state[j];
    }
}

// Generate input with coalesced memory access—column-wise storage
void generate_input(uint64_t *input, size_t num_messages, size_t num_blocks, int block_words) {
    srand(time(NULL));
    for (size_t i = 0; i < num_blocks; i++) {
        for (size_t m = 0; m < num_messages; m++) {
            size_t idx = m * block_words * num_blocks + i * block_words; // Coalesced
            for (size_t j = 0; j < block_words - 2; j++) {
                input[idx + j] = ((uint64_t)rand() << 32) | rand();
            }
            if (i == num_blocks - 1) {
                input[idx + block_words - 2] = 0x06ULL; // SHA-3 padding
                input[idx + block_words - 1] = 0x8000000000000000ULL;
            } else {
                input[idx + block_words - 2] = 0;
                input[idx + block_words - 1] = 0;
            }
        }
    }
}

// Configuration struct for Tables 4 and 6
struct Config {
    size_t plaintext_bytes;
    size_t num_blocks;
    size_t num_threads_per_block;
};

// Main function
int main() {
    // Table 4 configurations (SHA-3-256, no streams)
    Config table4_configs[] = {
        {65536, 1024, 64},
        {131072, 1024, 128},
        {262144, 2048, 128},
        {524288, 4096, 128},
        {1048576, 8192, 128},
        {2097152, 16384, 128},
        {4194304, 16384, 256},
        {8388608, 16384, 512}
    };
    size_t table4_num_configs = sizeof(table4_configs) / sizeof(table4_configs[0]);

    // Table 6 configurations (SHA-3-512, three streams)
    Config table6_configs[] = {
        {65536, 1024, 128},
        {131072, 1024, 128},
        {262144, 2048, 128},
        {524288, 4096, 128},
        {1048576, 8192, 128}
    };
    size_t table6_num_configs = sizeof(table6_configs) / sizeof(table6_configs[0]);

    // Test Table 4 (SHA-3-256, no streams)
    {
        int block_size_bytes = 144; // SHA-3-256: 1152 bits
        int block_words = block_size_bytes / 8; // 18 words
        int hash_size_bytes = 32; // SHA-3-256: 256 bits
        int hash_words = hash_size_bytes / 8; // 4 words

        printf("\nTesting SHA-3-256 (Table 4, No Streams)\n");
        printf("Plaintext (bytes) | Blocks | Threads/Block | Throughput (Gb/s)\n");
        printf("----------------------------------------------------------\n");

        for (size_t c = 0; c < table4_num_configs; c++) {
            Config cfg = table4_configs[c];
            size_t num_messages = cfg.plaintext_bytes / block_size_bytes;
            size_t input_size = num_messages * cfg.num_blocks * block_words * sizeof(uint64_t);
            size_t output_size = num_messages * hash_words * sizeof(uint64_t);

            // Allocate pinned host memory
            uint64_t *h_input, *h_output;
            cudaMallocHost(&h_input, input_size);
            cudaMallocHost(&h_output, output_size);

            // Allocate device memory
            uint64_t *d_input, *d_output;
            cudaMalloc(&d_input, input_size);
            cudaMalloc(&d_output, output_size);

            // Generate coalesced input
            generate_input(h_input, num_messages, cfg.num_blocks, block_words);

            // Timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // Process without streams
            cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
            int grid_blocks = (num_messages + cfg.num_threads_per_block - 1) / cfg.num_threads_per_block;
            sha3_kernel<<<grid_blocks, cfg.num_threads_per_block>>>(d_input, d_output, num_messages, cfg.num_blocks, block_words, hash_words);
            cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop, 0);
            cudaDeviceSynchronize();

            float total_time_ms;
            cudaEventElapsedTime(&total_time_ms, start, stop);
            double total_time = total_time_ms / 1000.0;
            double throughput = (input_size * 8.0) / (total_time * 1e9); // Gb/s

            // Print results
            printf("%-17zu | %-6zu | %-12zu | %.2f\n", cfg.plaintext_bytes, cfg.num_blocks, cfg.num_threads_per_block, throughput);

            // Cleanup
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFreeHost(h_input);
            cudaFreeHost(h_output);
        }
    }

    // Test Table 6 (SHA-3-512, three streams)
    {
        int block_size_bytes = 136; // SHA-3-512: 1088 bits
        int block_words = block_size_bytes / 8; // 17 words
        int hash_size_bytes = 64; // SHA-3-512: 512 bits
        int hash_words = hash_size_bytes / 8; // 8 words

        printf("\nTesting SHA-3-512 (Table 6, Three Streams)\n");
        printf("Plaintext (bytes) | Blocks | Threads/Block | Throughput (Gb/s)\n");
        printf("----------------------------------------------------------\n");

        for (size_t c = 0; c < table6_num_configs; c++) {
            Config cfg = table6_configs[c];
            size_t num_messages = cfg.plaintext_bytes / block_size_bytes;
            size_t input_size = num_messages * cfg.num_blocks * block_words * sizeof(uint64_t);
            size_t output_size = num_messages * hash_words * sizeof(uint64_t);

            // Allocate pinned host memory
            uint64_t *h_input, *h_output;
            cudaMallocHost(&h_input, input_size);
            cudaMallocHost(&h_output, output_size);

            // Allocate device memory for 3 streams
            uint64_t *d_input[3], *d_output[3];
            for (int s = 0; s < 3; s++) {
                cudaMalloc(&d_input[s], input_size / 3);
                cudaMalloc(&d_output[s], output_size / 3);
            }

            // Generate coalesced input
            generate_input(h_input, num_messages, cfg.num_blocks, block_words);

            // Timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // Process with 3 CUDA streams
            cudaStream_t streams[3];
            for (int s = 0; s < 3; s++) {
                cudaStreamCreate(&streams[s]);
            }
            size_t messages_per_stream = num_messages / 3;
            size_t input_size_per_stream = input_size / 3;
            size_t output_size_per_stream = output_size / 3;
            int grid_blocks = (messages_per_stream + cfg.num_threads_per_block - 1) / cfg.num_threads_per_block;

            for (int s = 0; s < 3; s++) {
                size_t offset = s * input_size_per_stream / sizeof(uint64_t);
                cudaMemcpyAsync(d_input[s], h_input + offset, input_size_per_stream, cudaMemcpyHostToDevice, streams[s]);
                sha3_kernel<<<grid_blocks, cfg.num_threads_per_block, 0, streams[s]>>>(d_input[s], d_output[s], messages_per_stream, cfg.num_blocks, block_words, hash_words);
                cudaMemcpyAsync(h_output + s * output_size_per_stream / sizeof(uint64_t), d_output[s], output_size_per_stream, cudaMemcpyDeviceToHost, streams[s]);
            }
            for (int s = 0; s < 3; s++) {
                cudaStreamDestroy(streams[s]);
            }

            cudaEventRecord(stop, 0);
            cudaDeviceSynchronize();

            float total_time_ms;
            cudaEventElapsedTime(&total_time_ms, start, stop);
            double total_time = total_time_ms / 1000.0;
            double throughput = (input_size * 8.0) / (total_time * 1e9); // Gb/s

            // Print results
            printf("%-17zu | %-6zu | %-12zu | %.2f\n", cfg.plaintext_bytes, cfg.num_blocks, cfg.num_threads_per_block, throughput);

            // Cleanup
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            for (int s = 0; s < 3; s++) {
                cudaFree(d_input[s]);
                cudaFree(d_output[s]);
            }
            cudaFreeHost(h_input);
            cudaFreeHost(h_output);
        }
    }

    return 0;
}