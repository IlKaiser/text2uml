## Classes
- Bank — manages customers, consists of branches
- Customer — firstName (String), lastName (String)
- Branch — (branch of a bank)
- Account — accountNumber (int), balance (double)
- CurrentAccount — overdraftLimit (double), subtype of it (Account)
- SavingsAccount — subtype of it (Account)

## Relationships
- Bank and its Customers (1 -> 0..*, any number of them).
- It and its Branches (1 -> 0..*, any number of them).
- Customer and Account (1 -> 1..5), its owner being one of them (a customer of the bank).
- Account and its subtypes (CurrentAccount, SavingsAccount, two basic types, generalization).
