import numpy as np
import pandas as pd


class DataSupplier(object):
    def __init__(self, config):
        self.config = config
        self.alternatives_df = None

    def load_alternatives(self, number=None, columns=None, sample_seed=1):
        alternatives_filename = self.config.get('filenames', 'alternatives')
        self.alternatives_df = pd.read_table(alternatives_filename, header=None)
        if columns:
            self.alternatives_df = pd.DataFrame(self.alternatives_df, columns=columns)
        if number:
            self.alternatives_df = self.alternatives_df.sample(number, random_state=sample_seed)
        return self.alternatives_df.values

    def get_criteria_weights(self):
        pass
