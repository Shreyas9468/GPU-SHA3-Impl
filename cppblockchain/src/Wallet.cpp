#include "Wallet.h"
#include <iostream>
#include <openssl/rand.h>
#include <openssl/pem.h>

Wallet::Wallet(std::string id) : id(id), balance(0.0f), publicKey(nullptr), privateKey(nullptr), transactionNonce(0) {
    generateKeys();
}

Wallet::~Wallet() {
    if (privateKey) {
        RSA_free(privateKey);
        privateKey = nullptr;
    }
    if (publicKey) {
        RSA_free(publicKey);
        publicKey = nullptr;
    }
}

void Wallet::generateKeys() {
    privateKey = RSA_new();
    BIGNUM* exponent = BN_new();
    BN_set_word(exponent, RSA_F4);
    RSA_generate_key_ex(privateKey, 2048, exponent, nullptr);
    
    publicKey = RSA_new();
    RSA_set0_key(publicKey, BN_dup(RSA_get0_n(privateKey)), BN_dup(RSA_get0_e(privateKey)), nullptr);
    
    BN_free(exponent);
}

Transaction Wallet::sendFunds(Wallet& receiver, float amount) {
    Transaction tx(id, receiver.id, amount, transactionNonce++);
    tx.sign(privateKey);
    std::cout << privateKey << "\n";
    return tx;
}

void Wallet::updateBalance(const std::vector<Transaction>& transactions) {
    for (const auto& tx : transactions) {
        if (tx.sender == id) {
            balance -= tx.amount;
        }
        if (tx.receiver == id) {
            balance += tx.amount;
        }
    }
}

void Wallet::printWalletData() const {
    std::cout << "Wallet ID: " << id << "\n";
    std::cout << "Balance: " << balance << "\n";
    std::cout << "Public Key: " << publicKey << "\n";
}