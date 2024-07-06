from LLM import Bot
import random

class Parse_Prompt(Bot):
    def __init__(self):
        super().__init__()
        self.change = True
        self.model1 = None
        self.model2 = None
        self.chat_history_1 = []
        self.chat_history_2 = []

    def model_init(self):
        return random.sample(self.models, 2)

    def clear_history(self):
        self.chat_history_1 = []
        self.chat_history_2 = []

    def change_models(self):
        self.clear_history()
        self.change = True

    def current_model1(self):
        return self.model1
    def current_model2(self):
        return self.model2

    def gen_output(self, temp, prompt):
        if self.change:
            [self.model1, self.model2] = self.model_init()
            self.change = False
        self.chat_history_1.append([prompt, self.response(self.model1, prompt, temp)])
        self.chat_history_2.append([prompt, self.response(self.model2, prompt, temp)])
        return self.chat_history_1, self.chat_history_2