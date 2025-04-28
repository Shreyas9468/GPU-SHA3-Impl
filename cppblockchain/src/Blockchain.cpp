#include "Blockchain.h"
#include "Wallet.h"
#include <iostream>
#include <iomanip>

Blockchain::Blockchain() {
    std::vector<Transaction> emptyTransactions;
    chain.emplace_back(emptyTransactions, "0", 2);
}

void Blockchain::createTransaction(Transaction transaction, const std::vector<Wallet*>& wallets) {
    if (isTransactionValid(transaction, wallets)) {
        pendingTransactions.push_back(transaction);
        std::cout << "Transaction from " << transaction.sender << " to " << transaction.receiver << " is valid.\n";
    } else {
        std::cout << "Transaction from " << transaction.sender << " to " << transaction.receiver << " is invalid!\n";
    }
}

void Blockchain::minePendingTransactions() {
    Block newBlock(pendingTransactions, chain.back().blockHash, 2);
    chain.push_back(newBlock);
    pendingTransactions.clear();
}

bool Blockchain::isBlockHashValid(const Block& block) {
    std::string computed_hash = block.generateHash();
    bool valid = block.blockHash == computed_hash;
    if (!valid) {
        std::cerr << "Hash mismatch in block:\n";
        std::cerr << "Stored Hash: " << block.blockHash << "\n";
        std::cerr << "Computed Hash: " << computed_hash << "\n";
        std::cerr << "Nonce: " << block.getNonce() << "\n";
        std::time_t block_time = block.getTimestamp();
        struct tm* tm_info = std::gmtime(&block_time);
        if (tm_info) {
            std::cerr << "Timestamp: " << std::put_time(tm_info, "%Y-%m-%dT%H:%M:%S") << "\n";
        } else {
            std::cerr << "Timestamp: Invalid (" << block_time << ")\n";
        }
        std::cerr << "Previous Hash: " << block.getPreviousHash() << "\n";
        std::cerr << "Transactions:\n";
        for (const auto& tx : block.getTransactions()) {
            std::cerr << "  Sender: " << tx.sender << ", Receiver: " << tx.receiver 
                      << ", Amount: " << std::fixed << std::setprecision(6) << tx.amount << "\n";
        }
        std::stringstream ss;
        if (tm_info) {
            ss << std::put_time(tm_info, "%Y-%m-%dT%H:%M:%S");
        } else {
            ss << block_time;
        }
        for (const auto& tx : block.getTransactions()) {
            ss << tx.sender << tx.receiver << std::fixed << std::setprecision(6) << tx.amount;
        }
        ss << block.getPreviousHash() << block.getNonce();
        std::cerr << "Input String: " << ss.str() << "\n";
    }
    return valid;
}

bool Blockchain::isTransactionValid(const Transaction& tx, const std::vector<Wallet*>& wallets) {
    if (tx.amount <= 0) {
        return false;
    }
    for (const auto& wallet : wallets) {
        if (wallet->id == tx.sender && wallet->balance < tx.amount) {
            return false;
        }
    }
    return true;
}

bool Blockchain::isChainValid() {
    for (int i = 1; i < chain.size(); ++i) {
        Block currBlock = chain[i];
        Block prevBlock = chain[i - 1];
        std::cout << "This encountered 1 " << std::endl;
        if (!isBlockHashValid(currBlock)) {
            return false;
        }
        std::cout << "This passed 1" << std::endl;
        if (currBlock.prevHash != prevBlock.blockHash) {
            return false;
        }
        std::cout << "This passed 2" << std::endl;
        for (const auto& tx : currBlock.transactions) {
            RSA* publicKey = publicKeyMap[tx.sender];
            if (!tx.isValid(publicKey)) {
                return false;
            }
        }
        std::cout << "This passed 3" << std::endl;
    }
    return true;
}

void Blockchain::printChain() {
    for (const auto& block : chain) {
        std::time_t block_time = block.getTimestamp();
        struct tm* tm_info = std::gmtime(&block_time);
        if (tm_info == nullptr) {
            std::cerr << "Invalid timestamp: " << block_time << "\n";
            continue;
        }
        char buffer[32];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", tm_info);
        std::cout << "Block Timestamp: " << buffer << "\n";
        std::cout << "Previous Hash: " << block.getPreviousHash() << "\n";
        std::cout << "Block Hash: " << block.getHash() << "\n";
        std::cout << "Transactions: " << "\n";
        for (const auto& tx : block.getTransactions()) {
            std::cout << "  Sender: " << tx.sender 
                      << " Receiver: " << tx.receiver 
                      << " Amount: " << tx.amount << "\n";
        }
        std::cout << "Nonce: " << block.getNonce() << "\n\n";
    }
}

void Blockchain::notifyWallets(std::vector<Wallet*>& wallets) {
    // Update public key map
    for (auto& wallet : wallets) {
        publicKeyMap[wallet->id] = wallet->publicKey;
    }
    // Update balances only for the latest block
    if (!chain.empty()) {
        const auto& latestBlock = chain.back();
        for (auto& wallet : wallets) {
            wallet->updateBalance(latestBlock.transactions);
        }
    }
}