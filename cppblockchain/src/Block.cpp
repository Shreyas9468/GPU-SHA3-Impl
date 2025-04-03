// src/Block.cpp
#include "Block.h"
#include "sha3.h"
#include <sstream>
#include <iomanip>

Block::Block(std::vector<Transaction> transactions, std::string prevHash, int difficulty) {
    this->transactions = transactions;
    this->prevHash = prevHash;
    this->timestamp = std::time(nullptr);
    this->difficulty = difficulty;
    this->nonce = 0;
    this->blockHash = mineBlock();
}

std::string Block::mineBlock() {
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&timestamp), "%Y-%m-%dT%H:%M:%S");  // Fixed: ×tamp -> &timestamp
    for (const auto& tx : transactions) {
        ss << tx.sender << tx.receiver << tx.amount;
    }
    ss << prevHash;
    std::string input = ss.str();

    unsigned char hash[HASH_SIZE_BYTES];
    mine_sha3((const unsigned char*)input.c_str(), input.size(), hash, &nonce, difficulty);

    std::stringstream hash_ss;
    for (int i = 0; i < HASH_SIZE_BYTES; i++) {
        hash_ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return hash_ss.str();
}

std::string Block::generateHash() const {
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&timestamp), "%Y-%m-%dT%H:%M:%S");  // Fixed: ×tamp -> &timestamp
    for (const auto& tx : transactions) {
        ss << tx.sender << tx.receiver << tx.amount;
    }
    ss << prevHash << nonce;
    std::string input = ss.str();

    unsigned char hash[HASH_SIZE_BYTES];
    compute_sha3((const unsigned char*)input.c_str(), input.size(), hash);

    std::stringstream hash_ss;
    for (int i = 0; i < HASH_SIZE_BYTES; i++) {
        hash_ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return hash_ss.str();
}