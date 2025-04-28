#pragma once
#include <vector>
#include "Block.h"
#include "Transaction.h"
#include "Wallet.h"
#include <unordered_map>

class Blockchain {
private:
    std::vector<Block> chain;
    std::vector<Transaction> pendingTransactions;
    std::unordered_map<std::string, RSA*> publicKeyMap;

public:
    Blockchain();
    void createTransaction(Transaction transaction, const std::vector<Wallet*>& wallets);
    void minePendingTransactions();
    bool isBlockHashValid(const Block& block);
    bool isTransactionValid(const Transaction& tx, const std::vector<Wallet*>& wallets);
    bool isChainValid();
    void printChain();
    void printEntireChain(); // New method
    void notifyWallets(std::vector<Wallet*>& wallets);
};