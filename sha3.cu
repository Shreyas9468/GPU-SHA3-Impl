#include <cuda_runtime.h> // CUDA runtime library for GPU functions
#include <cstdint>        // For uint64_t type (64-bit unsigned integer)
#include <stdio.h>        // For printf
#include <stdlib.h>       // For malloc, free, srand, rand
#include <string.h>       // For string/memory operations (not heavily used here)

// Define constants for SHA-3 structure—matches paper’s Keccak-f[1600]
#define STATE_SIZE 25         // Keccak-f state: 25 x 64-bit words (1600 bits)
#define BLOCK_SIZE_BYTES 72   // SHA-3-512 rate: 72 bytes per block (1088 bits)
#define BLOCK_WORDS 9         // 72 bytes = 9 x 64-bit words (8 bytes each)
#define HASH_SIZE_BYTES 64    // SHA-3-512 output: 64 bytes (512 bits)
#define HASH_WORDS 8          // 64 bytes = 8 x 64-bit words
#define ROUNDS 24             // Keccak-f rounds—24 iterations per block

#define THREADS_PER_BLOCK 128 // Threads per block—128 threads/team—paper’s typical CUDA config

// Round constants for ι step—precomputed for 24 rounds—SHA-3 standard
__constant__ uint64_t d_RC[ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x0000000000008002ULL, 0x0000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotate left—shifts bits left by n, wraps around—used in ρ step
__device__ inline uint64_t ROTL64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n)); // Shift left n bits, OR with right-shifted remainder—64-bit rotation
}

// Keccak-f[1600]—SHA-3’s core—runs on GPU per thread—transforms 1600-bit state
__device__ void keccak_f(uint64_t *state) {
    // Rotation offsets for ρ step—predefined—SHA-3 standard
    const int rho_offsets[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
    // 24 rounds—each round applies θ, ρ+π, χ, ι steps
    for (int round = 0; round < ROUNDS; round++) {
        uint64_t C[5], D[5]; // Temp arrays for θ step—C (column XORs), D (diffusion)

        // θ step—XOR each column (5 words)—paper’s PTX optimization (Algorithm 2)
        #pragma unroll // Unroll loop—fewer branches—GPU speedup
        for (int x = 0; x < 5; x++) {
            // PTX inline assembly—XORs 5 words in column x—faster than C—paper’s key trick
            asm("xor.b64 %0, %1, %2;\n\t"  // XOR state[x] and state[x+5] into C[x]
                "xor.b64 %0, %0, %3;\n\t"  // XOR result with state[x+10]
                "xor.b64 %0, %0, %4;\n\t"  // XOR result with state[x+15]
                "xor.b64 %0, %0, %5;"      // XOR result with state[x+20]
                : "=l"(C[x])               // Output to C[x]—64-bit register
                : "l"(state[x]), "l"(state[x + 5]), "l"(state[x + 10]), "l"(state[x + 15]), "l"(state[x + 20])); // Inputs—5 words
        }

        // θ diffusion—combine C values—spread changes across columns
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ROTL64(C[(x + 1) % 5], 1); // D[x] = C[x-1] ^ (C[x+1] rotated left 1)—e.g., D[0] = C[4] ^ (C[1] << 1)
        }

        // Apply θ—update state with D—each column gets same D value
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5]; // XOR state with D—e.g., state[0] ^= D[0], state[5] ^= D[0]—5 lanes per column
        }

        uint64_t B[25]; // Temp array for ρ+π—holds rotated/permuted values

        // ρ+π—rotate and permute—paper’s direct indexing (Section III.D)
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = x + 5 * y;           // Original position—e.g., state[6] = (x=1, y=1)
                int new_x = y;                 // New x = old y—e.g., 1
                int new_y = (2 * x + 3 * y) % 5; // New y = (2x + 3y) mod 5—e.g., (2*1 + 3*1) % 5 = 0
                B[new_x + 5 * new_y] = ROTL64(state[idx], rho_offsets[idx]); // Rotate—e.g., B[5] = state[6] << 44—no π table
            }
        }

        // χ—non-linear step—mixes bits within rows
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = x + 5 * y; // Position—e.g., state[0] = (x=0, y=0)
                state[idx] = B[idx] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]); // e.g., state[0] = B[0] ^ ((~B[1]) & B[2])
            }
        }

        // ι—add round constant—tweaks state[0]—SHA-3 standard
        state[0] ^= d_RC[round]; // e.g., state[0] ^= 0x0000000000000001ULL—unique per round
    }
}

// GPU kernel—each thread hashes one message—paper’s parallel design
__global__ void sha3_kernel(const uint64_t *d_input, uint64_t *d_output, size_t num_messages, size_t num_blocks) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID—e.g., block 100, thread 50 = 100 * 128 + 50 = 12,850
    if (tid >= num_messages) return; // Safety—extra threads exit—8192 * 128 > 1M

    uint64_t state[STATE_SIZE] = {0}; // Local 1600-bit state—25 x 64-bit words—each thread’s workspace—starts at zero
    size_t offset = tid * BLOCK_WORDS * num_blocks; // Start of this thread’s data—e.g., tid=12,850 * 288 = 3,700,800—coalesced

    #pragma unroll // Unroll—faster GPU execution—32 blocks per message
    for (size_t i = 0; i < num_blocks; i++) {
        for (int j = 0; j < BLOCK_WORDS; j++) { // 9 words/block—72 bytes
            state[j] ^= d_input[offset + i * BLOCK_WORDS + j]; // XOR block into state—e.g., state[0] ^= d_input[3,700,800]—absorb
        }
        keccak_f(state); // Transform state—32 times—paper’s multi-block method
    }

    #pragma unroll // Unroll—write 8-word hash
    for (int j = 0; j < HASH_WORDS; j++) {
        d_output[tid * HASH_WORDS + j] = state[j]; // Write hash—e.g., tid=12,850, d_output[102,800-102,807]—64 bytes—coalesced
    }
}

// CPU function—fills input data—runs on host
void generate_input(uint64_t *input, size_t num_messages, size_t num_blocks) {
    srand(time(NULL)); // Seed random—different data each run—uses current time
    for (size_t i = 0; i < num_blocks; i++) { // 32 blocks/message
        for (size_t m = 0; m < num_messages; m++) { // 1M messages
            uint64_t *block = input + i * num_messages * BLOCK_WORDS + m * BLOCK_WORDS; // Point to block—e.g., m=100, i=2—18,875,268—coalesced
            for (size_t j = 0; j < BLOCK_WORDS - 1; j++) { // 8 words—64 bytes
                block[j] = ((uint64_t)rand() << 32) | rand(); // Random 64-bit—e.g., 12345600000789—fill block
            }
            block[BLOCK_WORDS - 1] = (i == num_blocks - 1) ? 0x8000000000000006ULL : 0; // Last word—padding on block 31—SHA-3 standard
        }
    }
}

// Main—CPU orchestrates GPU hashing—paper’s workflow
int main() {
    // Define workload—1M messages, 32 blocks each—~2.41 GB—paper’s scale (Table 5 implied)
    size_t num_messages = 1048576; // 1M messages—each thread hashes one
    size_t num_blocks = 32;        // 32 blocks/message—~2,304 bytes
    size_t input_size = num_messages * num_blocks * BLOCK_WORDS * sizeof(uint64_t); // ~2.41 GB—1M * 32 * 9 * 8 bytes
    size_t output_size = num_messages * HASH_WORDS * sizeof(uint64_t); // ~67 MB—1M * 8 * 8 bytes

    // Allocate pinned CPU memory—faster H2D/D2H—WSL-compatible—paper assumes efficient transfer (Section IV)
    uint64_t *h_input, *h_output; // Host pointers—input data, output hashes
    cudaMallocHost(&h_input, input_size); // Pinned memory—~2.41 GB—fast GPU transfer
    cudaMallocHost(&h_output, output_size); // Pinned memory—~67 MB—for hashes

    // Allocate GPU memory—fits RTX 4060’s 8 GB VRAM—paper’s GPU focus
    uint64_t *d_input, *d_output; // Device pointers—GPU input/output
    cudaMalloc(&d_input, input_size); // ~2.41 GB on GPU—input data
    cudaMalloc(&d_output, output_size); // ~67 MB on GPU—hashes

    generate_input(h_input, num_messages, num_blocks); // Fill ~2.41 GB—coalesced—paper’s input prep (Section III.E)

    int blocks = (num_messages + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // 8192 blocks—1M threads—paper’s parallel setup

    // Timing—measure throughput—paper reports 88.51 Gb/s (Table 5)
    cudaEvent_t start, stop; // Events—start/stop stopwatch—GPU timing
    cudaEventCreate(&start); // Create start event—initialize
    cudaEventCreate(&stop);  // Create stop event—initialize

    cudaEventRecord(start, 0); // Start timing—default stream—before H2D
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice); // H2D—~2.41 GB—CPU to GPU—paper’s data prep
    sha3_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, num_messages, num_blocks); // Launch GPU—1M threads—paper’s core
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost); // D2H—~67 MB—GPU to CPU—get hashes
    cudaEventRecord(stop, 0); // Stop timing—after D2H—full process

    cudaDeviceSynchronize(); // Wait for GPU—ensure all done—paper assumes completion

    float total_time_ms; // Time in milliseconds—event result
    cudaEventElapsedTime(&total_time_ms, start, stop); // Get time—ms—start to stop
    double total_time = total_time_ms / 1000.0; // Convert to seconds—e.g., 294 ms = 0.294 s
    double throughput = (input_size * 8.0) / (total_time * 1e9); // Gb/s—~2.41 GB * 8 / time—paper’s metric

    // Print results—verify and show performance—paper implies reporting
    printf("Processed %zu messages (%zu bytes each) in %.3f seconds\n",
           num_messages, num_blocks * BLOCK_SIZE_BYTES, total_time); // 1M messages, ~2,304 bytes, time—e.g., 0.294 s
    printf("Throughput: %.2f Gb/s\n", throughput); // Throughput—e.g., 65.83 Gb/s—key result
    printf("First hash: "); // Show first hash—verify correctness
    for (int i = 0; i < HASH_WORDS; i++) { // 8 words—64 bytes—SHA-3-512
        printf("%016lx ", h_output[i]); // Print each 64-bit word—e.g., 24656faf62f2f6bb
    }
    printf("\n"); // Newline—clean output

    // Cleanup—free resources—no leaks—good practice
    cudaEventDestroy(start); // Free start event—GPU memory
    cudaEventDestroy(stop);  // Free stop event—GPU memory
    cudaFree(d_input);       // Free GPU input—~2.41 GB
    cudaFree(d_output);      // Free GPU output—~67 MB
    cudaFreeHost(h_input);   // Free pinned CPU input—~2.41 GB
    cudaFreeHost(h_output);  // Free pinned CPU output—~67 MB

    return 0; // Exit—success—paper’s end
}