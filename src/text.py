from nltk.corpus import stopwords
import re

class Text:
    text = None
    entities = None
    def __init__(self, text_path, entities_path):
        self.text_path = text_path
        self.entities_path = entities_path
        self.__read_text()
        self.__remove_stopwords()
        self.entities = self.__read_entities()

    def __read_text(self):
        with open(self.text_path, 'r') as file:
            self.text = file.read().lower().split()

    def __read_entities(self):
        with open(self.entities_path, 'r') as file:
            self.entities = [line.strip().lower().split() for line in file.readlines()]

    def __remove_stopwords(self): 
        stop_words = set(stopwords.words('english'))
        self.text = [word for word in self.text if word not in stop_words]
        ### not including punctuation

    def count_co_occurrences(self, window_size):
        import CoOccurances
        return CoOccurances.CoOcCount(self.entities, self.text, window_size)