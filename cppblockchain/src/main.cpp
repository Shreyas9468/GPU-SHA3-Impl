#include <iostream>
#include <vector>
#include "Blockchain.h"
#include "Wallet.h"
#include <iomanip>

int main() {
    Blockchain myBlockchain;
    std::vector<Wallet*> wallets;

    // Initialize wallets
    Wallet user1("User1");
    Wallet user2("User2");
    Wallet user3("User3");

    user1.balance = 100;
    user2.balance = 100;
    user3.balance = 10;

    wallets.push_back(&user1);
    wallets.push_back(&user2);
    wallets.push_back(&user3);

    // Transaction 1: User1 sends 50 to User2
    Transaction tx1 = user1.sendFunds(user2, 50);
    myBlockchain.createTransaction(tx1, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Transaction 2: User2 sends 30 to User3
    Transaction tx2 = user2.sendFunds(user3, 30);
    myBlockchain.createTransaction(tx2, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Transaction 3: User3 sends 10 to User1
    Transaction tx3 = user3.sendFunds(user1, 10);
    myBlockchain.createTransaction(tx3, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Transaction 4: User1 sends 20 to User3
    Transaction tx4 = user1.sendFunds(user3, 20);
    myBlockchain.createTransaction(tx4, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Validate and print the blockchain
    if (myBlockchain.isChainValid()) {
        std::cout << "Blockchain is valid.\n";
    } else {
        std::cout << "Blockchain is not valid!\n";
    }

    myBlockchain.printChain(); // Keep existing block printing
    myBlockchain.printEntireChain(); // Add entire blockchain printing

    // Print wallet balances
    std::cout << "\nWallet Balances:\n";
    for (const auto& wallet : wallets) {
        std::cout << "Wallet " << wallet->id << " has balance: " << std::fixed << std::setprecision(2) << wallet->balance << "\n";
    }

    return 0;
}