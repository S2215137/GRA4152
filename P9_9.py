## A class that represents a combination lock. In retroperspective, this code could be much more simplificated.
class ComboLock():
    ## Constructor takes in the parameters if they are all within 0-39, otherwise it raises a ValueError.
    def __init__(self, secret1, secret2, secret3):
        if 0 <= secret1 <=39 and 0 <= secret2 <=39 and 0 <= secret3 <=39:
            self._right1 = secret1
            self._left = secret2
            self._right2 = secret3
            
        else:
            raise ValueError('All values are not within 0 and 39')
            
        # Instantiates the rotate instance methods to 0, these simulates the rotations of the lock
        self._leftRotate = 0
        self._rightRotate = 0
        
        # the instance variables are the locks that represents if they are open or not
        self._rightLock1 = 'closed'
        self._leftLock = 'closed'
        self._rightLock2 = 'closed'
        
        # The main function that checks whether the rotation instance variable are the same as the lock combination
        self._lockChecker1()
    
    # Each of lockChecker function is running the next one as well, in case the locks are multiple value
    def _lockChecker1(self):
        if self._rightRotate == self._right1:
            self._rightLock1 = 'open'
            self._lockChecker2()
        else:
            self._rightLock1 = 'closed'
            self._leftLock = 'closed'
            self._rightLock2 = 'closed'
    
    # LockChecker 2 checks if the left rotation instance variable is the same as the lock combinations second value
    def _lockChecker2(self):
        if self._left == self._leftRotate and self._rightLock1 == 'open':
            # If criteria is met, the second lock is opened and lockChecker 3 is runned
            self._leftLock = 'open'
            self._lockChecker3()
        else:
            # Locks the 2nd lock and the 3rd lock if it's not satisfied 
            self._leftLock = 'closed'
            self._rightLock2 = 'closed'
    
    # LockChecker 3 sees if the last right rotation variable is equivalent to the last lock combination value
    def _lockChecker3(self):
        # Both previous locks have to be opened for this code to run
        if self._rightRotate == self._right2 and self._rightLock1 == 'open' and self._leftLock == 'open':
            self._rightLock2 = 'open'
        else:
            self._rightLock2 = 'closed'
    
    # Whenever a right or left rotation occurs, the opposite directions value is set to zero, to make it easier for the user.
    def turnRight(self, num):
        self._leftRotate = 0
        self._rightRotate += num
        
        # checks to see if the second combination is opened, if it is, it means that the first lock is also opened,
        # so lockChecker3 function is activated.
        
        if self._leftLock == 'open':
            self._lockChecker3()
            
        # If criteria is not met, it means that the user still tries to lock up the first value, so first lock is running.
        else:
            self._lockChecker1()
    
    def turnLeft(self, num):
        self._rightRotate = 0
        self._leftRotate += num
        # LockChecker2 is activated, since if the user is using this instance method, it means that the user tries to 
        # lock up number 2
        self._lockChecker2()
    
    def reset(self):
        self._rightRotate = 0
        self._leftRotate = 0

        # In the case that the combinations are all zero, the lockChecker function runs
        self._lockChecker1() 
    
    # The function checks if all the locks are opened, and if that's the case it prints out that it is.
    def open(self):
        if self._rightLock1 == 'open' and self._leftLock == 'open' and self._rightLock2 == 'open':
            print('Lock is Open')
        else:
            print('Lock is closed')

new_lock = ComboLock(4, 5, 10)
print('Trying to open up the combination lock after instantiating the new object:')
new_lock.open()
new_lock.turnRight(4)
new_lock.turnLeft(5)
print('Second try after opening the two first combinations:')
new_lock.open()

new_lock.turnRight(10)
print('After opening up the third lock:')
new_lock.open()

# Generating a combination with only zero's as the lock combination:
zero_lock = ComboLock(0, 0, 0)
print('Trying to open up the lock combination without changing the variables on the newly generated object:')
zero_lock.open()
