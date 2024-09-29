import numpy as np
from prompt_parser import Parse_Prompt
import pandas as pd

class Score(Parse_Prompt):
    def __init__(self):
        super().__init__()
        self.file_path = 'scoreboard.csv'
        self.init_scores()

    def init_scores(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            data = {
                'Models': self.models,
                'Fights Won': np.zeros(10, dtype = int)
            }
            self.df = pd.DataFrame(data)
            self.df.to_csv(self.file_path, index=False)

    def update(self, model, df):
        df.loc[self.df["Models"] == model, 'Fights Won'] += 1
        df.to_csv(self.file_path, index=False)
        self.clear_history()

    def df_show(self):
        return pd.read_csv(self.file_path)