# Imported the class from P10_3, it's essentially the same code as in section 10.1, except for the
# checkAnswer method, which is modified in this subclass either way.
from P10_3 import Question

# Inherited class from Question class that contains several answers to the given question.  
class MultiChoiceQuestion(Question):
    def __init__(self):
        # Initializes the super class instance variables.
        super().__init__()
    
    # Instance method setAnswer has been modified to use answers in a list datatype. 
    def setAnswer(self, correctResponse: list):
        # Making sure the values in the list are strings, so that numbers can be checked as well.
        lister = [str(i) for i in correctResponse]
        # Sorting the response, so there won't be any false negatives.
        correctResponse.sort()
        self._answer = lister
    
    # Modified the checkAnswer method to input several answers in the string, only separated by a single white space.
    def checkAnswer(self, response: str):
        lister = response.split()
        lister.sort()
        return lister == self._answer
    
quiz = MultiChoiceQuestion() 
quiz.setText("Name three names that I'm thinking of. Note: Answers has to be inserted on the same line only separated by a space.")
quiz.setAnswer(['Tina', 'Bob', 'Anders'])
quiz.display()
quiz.checkAnswer('Bob Tina Anders')

def test_multiple_correct():
    quiz = MultiChoiceQuestion() 
    quiz.setAnswer([4, 8])
    assert quiz.checkAnswer('8 4') == True
    
def test_assert_wrong():
    quiz = MultiChoiceQuestion()
    quiz.setAnswer(['Paris', 'Madrid'])
    assert quiz.checkAnswer('Madrid London') == False