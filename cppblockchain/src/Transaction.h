#ifndef TRANSACTION_H
#define TRANSACTION_H

#include <string>
#include <openssl/rsa.h>

class Transaction {
public:
    std::string sender;
    std::string receiver;
    float amount;
    std::string signature;
    int nonce;
    unsigned int signatureLength;

    Transaction(std::string sender, std::string receiver, float amount, int nonce);
    void sign(RSA* privateKey);
    bool verify(RSA* publicKey) const;
    bool isValid(RSA* publicKey) const;
};

#endif // TRANSACTION_H