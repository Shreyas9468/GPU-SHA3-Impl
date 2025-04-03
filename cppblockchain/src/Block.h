// src/Block.h
#pragma once
#include <string>
#include <vector>
#include <ctime>
#include <cstdint>
#include "Transaction.h"

class Block {
public:
    std::string prevHash;           // Hash of the previous block
    std::string blockHash;          // Hash of the current block
    std::vector<Transaction> transactions; // List of transactions in this block
    std::time_t timestamp;          // Timestamp for when this block was created
    uint64_t nonce;                 // Nonce used for mining (64-bit unsigned integer)
    int difficulty;                 // The difficulty level for mining this block

    // Constructor to initialize a block with transactions, previous hash, and difficulty
    Block(std::vector<Transaction> transactions, std::string prevHash, int difficulty);

    // Method to mine the block by finding a nonce that satisfies the difficulty
    std::string mineBlock();

    // Method to generate the hash of the block's contents (for validation)
    std::string generateHash() const;

    // Accessor methods
    std::string getHash() const { return blockHash; }
    std::string getPreviousHash() const { return prevHash; }
    std::vector<Transaction> getTransactions() const { return transactions; }
    uint64_t getNonce() const { return nonce; }
    std::time_t getTimestamp() const { return timestamp; }
};