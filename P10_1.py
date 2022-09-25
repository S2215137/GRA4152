# The class Question is copy pasted from the book.
class Question():
    def __init__(self):
        self._text = ''
        self._answer = ''
    
    def setText(self, questionText):
        self._text = questionText
        
    def setAnswer(self, correctResponse):
        self._answer = correctResponse
    
    def checkAnswer(self, response):
        return response == self._answer
    
    def display(self):
        print(self._text)

# A subclass that inherits from the Question class. Only for numerical discrete numbers.     
class NumericQuestion(Question):
    def checkAnswer(self, num_response: float):
        # Takes the absolute value of the difference to check if it's lower than 0.01. 
        # Due to some rounding error on Python's part, I have to round the values.
        return round(abs(float(self._answer) - num_response), 6) <= 0.01


# Testing whether the assigned number will pass a test where the difference is 0.01, should return True.
def test_num_answer_correct():
    num_question = NumericQuestion()
    num_question.setAnswer(9.87)
    assert num_question.checkAnswer(9.86) == True

# Test where the answer is widely different, and should return False.
def test_num_answer_false():
    num_question = NumericQuestion()
    num_question.setAnswer(0)
    assert num_question.checkAnswer(1) == False
