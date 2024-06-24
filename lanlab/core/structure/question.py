from lanlab.core.structure.text import Text
from lanlab.core.structure.segment import Segment

class Question(Text):
    def __init__(self,question_text):
        super().__init__(question_text)

        self[0]['format']['chat'] = '[text]'
        self[0]['format']['completion'] = 'Q:[text]A:'

        self[0]['info']['question'] = question_text