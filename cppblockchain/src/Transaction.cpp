#include "Transaction.h"
#include <openssl/sha.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <iostream>
#include <iomanip>

void debugPrintHash(unsigned char* hash, size_t length) {
    for (size_t i = 0; i < length; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    std::cout << "\n";
}

Transaction::Transaction(std::string sender, std::string receiver, float amount, int nonce)
    : sender(sender), receiver(receiver), amount(amount), nonce(nonce) {}

void Transaction::sign(RSA* privateKey) {
    std::string dataToSign = sender + receiver + std::to_string(amount) + std::to_string(nonce);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(dataToSign.c_str()), dataToSign.size(), hash);

    unsigned char sign[RSA_size(privateKey)];
    unsigned int sLen;

    if (RSA_sign(NID_sha256, hash, SHA256_DIGEST_LENGTH, sign, &sLen, privateKey) != 1) {
        std::cout << "Signing failed.\n";
        return;
    }

    signature.assign(reinterpret_cast<char*>(sign), sLen);
    signatureLength = sLen;
    std::cout << "Signed successfully. Signature Length: " << sLen << "\n";
}

bool Transaction::verify(RSA* publicKey) const {
    ERR_clear_error();
    std::string dataToVerify = sender + receiver + std::to_string(amount) + std::to_string(nonce);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(dataToVerify.c_str()), dataToVerify.size(), hash);

    int ret = RSA_verify(NID_sha256, hash, SHA256_DIGEST_LENGTH, 
                         reinterpret_cast<const unsigned char*>(signature.data()), signatureLength, publicKey);
    
    if (ret != 1) {
        unsigned long err = ERR_get_error();
        char *errStr = ERR_error_string(err, NULL);
        std::cout << "Error occurred during verification: " << errStr << "\n";
        return false;
    }
    std::cout << "Verification successful.\n";
    return true;
}

bool Transaction::isValid(RSA* publicKey) const {
    if (nonce < 0 || !(amount > 0)) {
        return false;
    }
    return verify(publicKey);
}