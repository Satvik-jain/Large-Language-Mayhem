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
            # Update model names in existing CSV if they've changed
            model_mapping = {
                'gemini-1.0-pro': 'gemini-pro',
                'gemini-1.5-pro': 'gemini-1.5-flash'  # Map to available model
            }
            self.df['Models'] = self.df['Models'].replace(model_mapping)
            # Ensure all current models are in the dataframe
            current_models = set(self.models)
            existing_models = set(self.df['Models'].values)
            missing_models = current_models - existing_models
            if missing_models:
                new_rows = pd.DataFrame({
                    'Models': list(missing_models),
                    'Fights Won': [0] * len(missing_models)
                })
                self.df = pd.concat([self.df, new_rows], ignore_index=True)
            # Remove models that no longer exist
            self.df = self.df[self.df['Models'].isin(self.models)]
            self.df.to_csv(self.file_path, index=False)
        except FileNotFoundError:
            data = {
                'Models': self.models,
                'Fights Won': np.zeros(len(self.models), dtype = int)
            }
            self.df = pd.DataFrame(data)
            self.df.to_csv(self.file_path, index=False)

    def update(self, model, df):
        self.df.loc[self.df["Models"] == model, 'Fights Won'] += 1
        self.df.to_csv(self.file_path, index=False)
        self.clear_history()

    def df_show(self):
        return pd.read_csv(self.file_path)