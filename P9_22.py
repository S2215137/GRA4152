# Created a Portfolion class that creates two instance variables, 'S' for saving account and 'C' for checking account
class Portfolio():
    def __init__(self, check_bal=0, sav_acc=0):
        self._check_acc = check_bal
        self._sav_acc = sav_acc
        # Created a dictionary for cleaner code
        self.dicty = {'S': self._sav_acc, 'C': self._check_acc} 
    
    # Deposits the amount into the given bank account 
    def deposit(self, amount, account):
        self.dicty[account] += amount
    
    # Accessor method that returns the balance for the given account     
    def getBalance(self, account):
        return self.dicty[account]
    
    # Withdraws the amount from the given bank account
    def withdraw(self, amount, account):
        if self.getBalance(account) >= amount:
            self.dicty[account] -= amount
        else:
            # Value output is in USD instead of NOK, since I thought it looked cleaner with the $ symbol.
            print(f'Insufficient Funds, the requested amount was {amount}$, and your balance is {self.getBalance(account)}$ in this account')
            
    # Transfers the amount from the given account to the other account as mentioned in the book.
    def transfer(self, amount, account):
        if self.getBalance(account) >= amount:
            if account == 'S':
                tran_to_acc = 'C'
            else:
                tran_to_acc = 'S'
            self.withdraw(amount, account)
            self.deposit(amount, tran_to_acc)
        else:
            print(f'Insufficient Funds, unable to transfer the amount into the other account.')

# Creating a simple test case, without the bank_account transfer.
def test_portfolio_acc_method():
    my_portf = Portfolio(100, 400)
    assert my_portf.getBalance('C') == 100
    assert my_portf.getBalance('S') == 400

# A test case to check if the transfer method works properly
def test_transfer():
    my_portf = Portfolio(100, 700)
    my_portf.transfer(200, 'S')
    assert my_portf.getBalance('C') == 300
    assert my_portf.getBalance('S') == 500
