# A superclass that requires two parameters, the individuals name and year of birth respectively.
class Person():
    def __init__(self, name, year_of_birth):
        self._name = name
        self._year = year_of_birth  

# A subclass that represents a student, and inherits from the Person class.
# Additional required argument is which major the student is taking. The default is undeclared.
class Student(Person):
    def __init__(self, name, year_of_birth, major='Undeclared'):
        super().__init__(name, year_of_birth)
        self._major = major
    
    # Changing the __repr__ instance method, to a more user friendly output.
    def __repr__(self):
        return f"This is {self._name}, and is a student majoring in {self._major}"

# A sublcass that inherits from the Person class, and should represent a teacher.
# Additional required parameter is the salary which should be a discrete integer.
class Instructor(Person):
    def __init__(self, name, year_of_birth, salary: int):
        super().__init__(name, year_of_birth)
        self._salary = salary
    
    # Changed the __repr__ instance method to make it more user friendly.
    def __repr__(self):
        return f"This is {self._name}, and is a teachor with a salary of: {self._salary}"

# Test to check whether the student sub-class was inherited properly.
def test_student_inher():
    anders = Student('Anders', 1999, 'Data Science in Business')
    assert anders._year == 1999
    assert anders._major == 'Data Science in Business'

# Test to see whether the instructor sub-class was inherited properly.
def test_instr_inher():
    berit = Instructor('Berit', 1980, 800000)
    assert berit._name == 'Berit'
    assert berit._salary == 800000
