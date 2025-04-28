#include <iostream>
#include <vector>
#include "Blockchain.h"
#include "Wallet.h"

int main() {
    Blockchain myBlockchain;
    std::vector<Wallet*> wallets;

    // Initialize wallets
    Wallet alice("Alice");
    Wallet bob("Bob");
    Wallet charlie("Charlie");

    alice.balance = 100;
    bob.balance = 100;
    charlie.balance = 10;

    wallets.push_back(&alice);
    wallets.push_back(&bob);
    wallets.push_back(&charlie);

    // Transaction 1: Alice sends 50 to Bob
    Transaction tx1 = alice.sendFunds(bob, 50);
    myBlockchain.createTransaction(tx1, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Transaction 2: Bob sends 30 to Charlie
    Transaction tx2 = bob.sendFunds(charlie, 30);
    myBlockchain.createTransaction(tx2, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Transaction 3: Charlie sends 10 to Alice
    Transaction tx3 = charlie.sendFunds(alice, 10);
    myBlockchain.createTransaction(tx3, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Transaction 4: Alice sends 20 to Charlie
    Transaction tx4 = alice.sendFunds(charlie, 20);
    myBlockchain.createTransaction(tx4, wallets);
    myBlockchain.minePendingTransactions();
    myBlockchain.notifyWallets(wallets);

    // Validate and print the blockchain
    if (myBlockchain.isChainValid()) {
        std::cout << "Blockchain is valid main .\n";
    } else {
        std::cout << "Blockchain is not valid main !\n";
    }

    myBlockchain.printChain();

    // Print wallet balances
    std::cout << "\nWallet Balances:\n";
    for (const auto& wallet : wallets) {
        std::cout << "Wallet " << wallet->id << " has balance: " << wallet->balance << "\n";
    }

    return 0;
}