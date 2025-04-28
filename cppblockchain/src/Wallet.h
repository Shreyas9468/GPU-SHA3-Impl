#ifndef WALLET_H
#define WALLET_H

#include <string>
#include <vector>
#include <openssl/rsa.h>
#include "Transaction.h"

class Wallet {
public:
    Wallet(std::string id);
    ~Wallet();
    Transaction sendFunds(Wallet& receiver, float amount);
    void updateBalance(const std::vector<Transaction>& transactions);
    void printWalletData() const;

    std::string id;
    float balance;
    RSA* publicKey;
    int transactionNonce; // Added for incrementing nonce

private:
    RSA* privateKey;
    void generateKeys();
};

#endif // WALLET_H