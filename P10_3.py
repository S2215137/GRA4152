# Modified the checkAnswer method in the Question class from section 10.1 to disregard-
# spaces and lower or uppercase characters.
class Question():
    def __init__(self):
        self._text = ''
        self._answer = ''
    
    def setText(self, questionText):
        self._text = questionText
        
    def setAnswer(self, correctResponse):
        self._answer = correctResponse
    
    # Modified checkAnswer instance method that removes all spaces and is not case sensitive if the input is string type.
    def checkAnswer(self, response):
        # Making a placeholder variable for the answer to not change the original answer.
        # Modified the variable to lowercase and without spaces, so that it's just the characters remaining 
        if type(response) == str and type(self._answer) == str:
            pl_answer = self._answer.lower()
            pl_answer = pl_answer.replace(' ', '')
            response = response.lower()
            response = response.replace(' ', '')
            return response == pl_answer
        else:
            return response == self._answer
        
    def display(self):
        print(self._text)
    
# Test to see whether the method modified version is still valid for non-string inputs. 
def test_non_string_ans():
    quiz_lst = Question()
    quiz_lst.setAnswer([0, 5])
    assert quiz_lst.checkAnswer([0, 5]) == True 
    
# A test to see whether the modified instance method removes spaces and is not case sensitive.

def test_check_answer_spaced_cap():
    quiz = Question()
    quiz.setAnswer('Abraham Lincoln')
    assert quiz.checkAnswer(' A B R AHAM lincoln') == True

# Checking whether wrong answer's will still remain False.
def test_wrong_answer():
    quiz = Question()
    quiz.setAnswer('Max Planck')
    assert quiz.checkAnswer('Abraham Lincoln') == False
