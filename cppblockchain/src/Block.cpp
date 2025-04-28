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

// std::string Block::mineBlock() {
//     // std::stringstream ss;
//     // ss << std::put_time(std::gmtime(&timestamp), "%Y-%m-%dT%H:%M:%S");
//     // for (const auto& tx : transactions) {
//     //     ss << tx.sender << tx.receiver << std::fixed << std::setprecision(6) << tx.amount;
//     // }
//     // ss << prevHash << nonce;
//     // std::string input = ss.str();

//     // unsigned char hash[HASH_SIZE_BYTES];
//     // mine_sha3((const unsigned char*)input.c_str(), input.size(), hash, &nonce, difficulty);

//     // if (nonce == 0) {
//     //     throw std::runtime_error("Mining failed: no valid nonce found");
//     // }

//     // std::stringstream hash_ss;
//     // for (int i = 0; i < HASH_SIZE_BYTES; i++) {
//     //     hash_ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
//     // }
//     // return hash_ss.str();


// }
std::string Block::mineBlock() {
    // Base input for mining (excluding nonce)
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&timestamp), "%Y-%m-%dT%H:%M:%S");
    for (const auto& tx : transactions) {
        ss << tx.sender << tx.receiver << std::fixed << std::setprecision(6) << tx.amount;
    }
    ss << prevHash;
    std::string base_input = ss.str();

    unsigned char hash[HASH_SIZE_BYTES];
    mine_sha3((const unsigned char*)base_input.c_str(), base_input.size(), hash, &nonce, difficulty);

    if (nonce == 0) {
        throw std::runtime_error("Mining failed: no valid nonce found");
    }

    // Final input with updated nonce
    std::stringstream final_ss;
    final_ss << std::put_time(std::gmtime(&timestamp), "%Y-%m-%dT%H:%M:%S");
    for (const auto& tx : transactions) {
        final_ss << tx.sender << tx.receiver << std::fixed << std::setprecision(6) << tx.amount;
    }
    final_ss << prevHash << nonce;
    std::string final_input = final_ss.str();
    compute_sha3((const unsigned char*)final_input.c_str(), final_input.size(), hash);

    std::stringstream hash_ss;
    for (int i = 0; i < HASH_SIZE_BYTES; i++) {
        hash_ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return hash_ss.str();
}

std::string Block::generateHash() const {
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&timestamp), "%Y-%m-%dT%H:%M:%S");
    for (const auto& tx : transactions) {
        ss << tx.sender << tx.receiver << std::fixed << std::setprecision(6) << tx.amount;
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