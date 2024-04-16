from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, DBSCAN


class Sequencer:
    """
    A class to sequence data into sequences of data based on a clustering algorithm.
    These sequences can be used to train a feature selection model on those sequences,
    under the assumption that each sequence may have a different distribution and thus
    different features may be important in each sequence.
    """

    def __init__(self, data: pd.DataFrame, targets: pd.Series, cols: List[str],
                 sequencing_method: str = 'MeanShift', as_type: np.dtype = np.float32):
        data = data.astype(as_type)
        targets = targets.astype(as_type)
        self.cols = cols
        self.clustering_method = MeanShift if sequencing_method == 'MeanShift' else DBSCAN
        self.sequences, self.targets = self.sequence_data(data, targets)
        self.index = 0
        self.previous_index = 0

    def sequence_data(self, data: pd.DataFrame, targets: pd.Series) -> (List[pd.DataFrame], List[pd.Series]):
        """
        Performs analysis on the data to break it into sequences by the given columns.
        :return: A list of sequences of data and a list of sequences of targets
        """
        sequences = []
        sequence_targets = []
        clustering = self.clustering_method()
        # Fit the clustering algorithm to the data to identify sequences in the data
        clusters = clustering.fit_predict(data[self.cols])
        for cluster in set(clusters):
            sequence = data[clusters == cluster]
            # Sort the sequence by its column values
            sequence = sequence.sort_values(by=self.cols)
            # Get the targets for the sequence
            current_targets = targets[clusters == cluster]
            # Sort the targets by the sequence
            current_targets = current_targets[sequence.index]
            sequences.append(sequence)
            sequence_targets.append(current_targets)
        return sequences, sequence_targets

    def normalize(self):
        """
        Normalize the data in the sequences
        :return:
        """
        for i in range(len(self.sequences)):
            std = self.sequences[i].std()
            # Locate zeroes in the standard deviation and replace them with 1, to avoid division by zero
            std[std == 0] = 1
            self.sequences[i] = (self.sequences[i] - self.sequences[i].mean()) / std


    def pad_sequences(self, max_sequence_length: int):
        """
        Pad the sequences to the same length
        :param max_sequence_length:
        :return:
        """
        for i in range(len(self.sequences)):
            sequence = self.sequences[i]
            if len(sequence) < max_sequence_length:
                # Pad the sequence with zeros
                padding = pd.DataFrame(np.zeros((max_sequence_length - len(sequence), len(sequence.columns))),
                                       columns=sequence.columns)
                self.sequences[i] = pd.concat([sequence, padding])

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next sequence in the list of sequences
        :return:
        """
        current_sequence = self.sequences[self.index]
        self.previous_index = self.index
        self.index += 1
        self.index = self.index % len(self.sequences)
        return current_sequence.to_numpy().reshape(-1,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item], self.targets[item]

    def get_current(self):
        """
        Get the current sequence and target, without moving to the next sequence
        :return:
        """
        return self.sequences[self.index], self.targets[self.index]

    def get_max_sequence_length(self):
        """
        Get the length of the longest sequence
        :return:
        """
        return max([len(sequence) for sequence in self.sequences])

    def reset(self):
        """
        Reset the index of the sequences
        :return:
        """
        self.index = 0
        self.previous_index = 0
        return self.sequences[self.index].to_numpy().reshape(-1,)

    def is_done(self):
        """
        Check if the sequence is done
        :return:
        """
        return self.index <= self.previous_index



