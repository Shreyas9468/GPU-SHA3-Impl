// src/sha3.h
#ifndef SHA3_H
#define SHA3_H

#include <cstdint>  // Added for uint64_t

#define HASH_SIZE_BYTES 64

extern "C" void compute_sha3(const unsigned char *input, size_t input_len, unsigned char *output);
extern "C" void mine_sha3(const unsigned char *input, size_t input_len, unsigned char *output, uint64_t *nonce, int difficulty);

#endif