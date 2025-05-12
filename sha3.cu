#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Configuration parameters based on the paper's performance analysis
struct SHA3Config {
    size_t num_messages;       // Total number of messages to process
    size_t num_blocks;         // Number of blocks per message
    size_t threads_per_block;  // Threads per block (from paper's optimization)
    size_t num_streams;        // Number of CUDA streams
};

// SHA-3 Constants - generalized to support different configurations
#define STATE_SIZE 25         // 25 x 64-bit words (1600 bits)
#define BLOCK_SIZE_BYTES 136  // 1088 bits = 136 bytes
#define BLOCK_WORDS 17        // 136 bytes = 17 x 64-bit words
#define HASH_SIZE_BYTES 64    // 512 bits = 64 bytes
#define HASH_WORDS 8          // 64 bytes = 8 x 64-bit words
#define ROUNDS 24             // Keccak-f rounds

// Round constants for ι step—SHA-3 standard
__constant__ uint64_t d_RC[ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x0000000000008002ULL, 0x0000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotate left—used in ρ and θ
__device__ inline uint64_t ROTL64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600]—optimized with direct ρ offsets and precomputed θ
__device__ void keccak_f(uint64_t *state) {
    for (int round = 0; round < ROUNDS; round++) {
        uint64_t C[5], D[5];

        // θ step—precompute C in registers—paper's optimization
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            asm("xor.b64 %0, %1, %2;\n\t"
                "xor.b64 %0, %0, %3;\n\t"
                "xor.b64 %0, %0, %4;\n\t"
                "xor.b64 %0, %0, %5;"
                : "=l"(C[x])
                : "l"(state[x]), "l"(state[x + 5]), "l"(state[x + 10]), "l"(state[x + 15]), "l"(state[x + 20]));
        }

        // θ diffusion—compute D
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ROTL64(C[(x + 1) % 5], 1);
        }

        // Apply θ
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        uint64_t B[25];

        // ρ+π—direct offset computation
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = x + 5 * y;
                int new_x = y;
                int new_y = (2 * x + 3 * y) % 5;
                // Compute ρ offset directly
                int t = (x + 3 * y) % 5;
                int offset = ((t + 1) * (t + 2) / 2) % 64;
                B[new_x + 5 * new_y] = ROTL64(state[idx], offset);
            }
        }

        // χ step
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = x + 5 * y;
                state[idx] = B[idx] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
            }
        }

        // ι step
        state[0] ^= d_RC[round];
    }
}

// SHA-3 kernel—one thread per message
__global__ void sha3_kernel(const uint64_t *d_input, uint64_t *d_output, 
                             size_t num_messages, size_t num_blocks) {
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

// Generate input—column-wise for coalesced access
void generate_input(uint64_t *input, size_t num_messages, size_t num_blocks) {
    srand(time(NULL));
    for (size_t m = 0; m < num_messages; m++) {
        for (size_t i = 0; i < num_blocks; i++) {
            uint64_t *block = input + (m * num_blocks + i) * BLOCK_WORDS;
            for (size_t j = 0; j < BLOCK_WORDS - 2; j++) {
                block[j] = ((uint64_t)rand() << 32) | rand();
            }
            if (i == num_blocks - 1) {
                block[BLOCK_WORDS - 2] = 0x06ULL;
                block[BLOCK_WORDS - 1] = 0x8000000000000000ULL;
            } else {
                block[BLOCK_WORDS - 2] = 0;
                block[BLOCK_WORDS - 1] = 0;
            }
        }
    }
}

// Configurable main function
int run_sha3_benchmark(const SHA3Config& config) {
    size_t input_size = config.num_messages * config.num_blocks * BLOCK_WORDS * sizeof(uint64_t);
    size_t output_size = config.num_messages * HASH_WORDS * sizeof(uint64_t);

    // Pinned host memory
    uint64_t *h_input, *h_output;
    cudaMallocHost(&h_input, input_size);
    cudaMallocHost(&h_output, output_size);

    generate_input(h_input, config.num_messages, config.num_blocks);

    // CUDA streams
    cudaStream_t streams[config.num_streams];
    uint64_t *d_input[config.num_streams], *d_output[config.num_streams];
    size_t messages_per_stream = config.num_messages / config.num_streams;

    for (size_t i = 0; i < config.num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_input[i], messages_per_stream * config.num_blocks * BLOCK_WORDS * sizeof(uint64_t));
        cudaMalloc(&d_output[i], messages_per_stream * HASH_WORDS * sizeof(uint64_t));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Launch streams
    for (size_t i = 0; i < config.num_streams; i++) {
        size_t offset = i * messages_per_stream;
        size_t stream_messages = (i == config.num_streams - 1) ? 
                                  (config.num_messages - offset) : messages_per_stream;
        size_t stream_input_size = stream_messages * config.num_blocks * BLOCK_WORDS * sizeof(uint64_t);
        size_t stream_output_size = stream_messages * HASH_WORDS * sizeof(uint64_t);

        cudaMemcpyAsync(d_input[i], 
                        h_input + offset * config.num_blocks * BLOCK_WORDS, 
                        stream_input_size, 
                        cudaMemcpyHostToDevice, 
                        streams[i]);
        
        int blocks = (stream_messages + config.threads_per_block - 1) / config.threads_per_block;
        sha3_kernel<<<blocks, config.threads_per_block, 0, streams[i]>>>(
            d_input[i], d_output[i], stream_messages, config.num_blocks);
        
        cudaMemcpyAsync(h_output + offset * HASH_WORDS, 
                        d_output[i], 
                        stream_output_size, 
                        cudaMemcpyDeviceToHost, 
                        streams[i]);
    }

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    float total_time_ms;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    double total_time = total_time_ms / 1000.0;
    double throughput = (input_size * 8.0) / (total_time * 1e9);

    printf("Processed %zu messages (%zu bytes each) in %.3f seconds\n",
           config.num_messages, config.num_blocks * BLOCK_SIZE_BYTES, total_time);
    printf("Throughput: %.2f Gb/s\n", throughput);
    printf("First hash: ");
    for (int i = 0; i < HASH_WORDS; i++) {
        printf("%016lx ", h_output[i]);
    }
    printf("\n");

    // Cleanup
    for (size_t i = 0; i < config.num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    return 0;
}

// Main function with example configurations
int main() {
    // Example configurations based on the paper's performance analysis
    // GTX 1070 Configuration
    SHA3Config abcd = {
        1048576,  // num_messages
        32,        // num_blocks
        64,      // threads_per_block (from paper's optimization)
        3         // num_streams
    };

   
    // Run benchmark with configuration
    printf("Running SHA-3 Benchmark :\n");
    run_sha3_benchmark(abcd);

  

    return 0;
}