## A bank account that represents depositing and withdrawing from a bank in USD$.
class BankAccount():
    ## Constructs a bank account with zero balance.
    # Monthly withdrawals instance variable is for the 
    def __init__(self):
        self._balance = 0
        self._monthWithdrawals = 0
        self._withdrawals = 0
    
    def deposit(self, amount):
        self._balance = self._balance + amount
        
    def withdraw(self, amount):
        self._balance = self._balance - amount
        
    def getBalance(self):
        return self._balance

## A checking account that has adjusted deposit and withdrawal methods.     
class CheckingsAccount(BankAccount):
    ## Class variables are implemented to make it easily visible if the rules change.
    FREE_WITHDRAWALS = 3
    WITHDRAWAL_FEE = 1
    
    def __init__(self):
        # Instantiates the super class variables.
        super().__init__()
    
    def withdraw(self, amount):
        # Adds a counter on the monthly and regular total withdrawals.
        self._withdrawals = self._withdrawals + 1
        self._monthWithdrawals = self._monthWithdrawals + 1
        super().withdraw(amount + self.over_exceed_fee())
        
    ## A class method that checks if the user has exceeded the monthly limit of withdrawals and deposits.
    # Returns the Withdrawal fee if it's been exceeded and if not returns 0.
    def over_exceed_fee(self):
        if self._monthWithdrawals > CheckingsAccount.FREE_WITHDRAWALS:
            return CheckingsAccount.WITHDRAWAL_FEE
        else:
            return 0
        
    def deposit(self, amount):
        self._withdrawals = self._withdrawals + 1
        self._monthWithdrawals = self._monthWithdrawals + 1
        balance = super().getBalance()
        super().deposit(amount - self.over_exceed_fee())
        
## A test class to check if the over exceeder method is working when exceeded.
#
def test_exc_withdrawals():
    checkings = CheckingsAccount()
    checkings.deposit(300)
    checkings.withdraw(100)
    checkings.withdraw(100)
    checkings.withdraw(50)
    # Should return 300 - 100 - 100 + (50-1), which is 49 due to the transaction fee.
    assert checkings.getBalance() == 49

## Testing the class when it's not exceeded 3 transactions.
def test_not_exceeded_withdrawals():
    checkings = CheckingsAccount()
    checkings.deposit(300)
    checkings.deposit(100)
    checkings.withdraw(20)
    # Should return 300 + 100 - 20, which is 380
    assert checkings.getBalance() == 380