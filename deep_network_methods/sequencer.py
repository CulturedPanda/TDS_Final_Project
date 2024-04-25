from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, DBSCAN
from .util_funcs import is_boolean_column


class Sequencer:
    """
    A class to sequence data into sequences of data based on a clustering algorithm.
    These sequences can be used to train a feature selection model on those sequences,
    under the assumption that each sequence may have a different distribution and thus
    different features may be important in each sequence.
    """

    def __init__(self, data: pd.DataFrame, targets: pd.Series, col: str, flatten: bool,
                 clustering_method: str = 'MeanShift', as_type: np.dtype = np.float32):
        """
        :param data: The data to sequence
        :param targets: The targets for the data
        :param col: The column to sequence the data by
        :param flatten: Whether to flatten the data or not
        :param clustering_method: The method to use for sequencing the data. Either 'MeanShift' or 'DBSCAN'
        :param as_type: The data type to cast the data to
        """
        data = data.astype(as_type)
        self.max_index = max(data.index * 10)
        targets = targets.astype(as_type)
        self.col = col

        # The clustering method to use
        self.clustering_method = MeanShift if clustering_method == 'MeanShift' else DBSCAN

        # Sequence the data
        self.sequences, self.targets = self.sequence_data(data, targets)
        self.index = 0
        self.previous_index = 0
        # The ranges of each sequence found in the training data
        self.ranges = [(sequence[self.col].min(), sequence[self.col].max()) for sequence in self.sequences]
        # Sort the ranges by the start of the range, so that they are in order
        self.ranges.sort(key=lambda x: x[0])
        self.flatten = flatten

    def sequence_data(self, data: pd.DataFrame, targets: pd.Series) -> (List[pd.DataFrame], List[pd.Series]):
        """
        Performs analysis on the data to break it into sequences by the given columns.
        :return: A list of sequences of data and a list of sequences of targets
        """
        sequences = []
        sequence_targets = []
        clustering = self.clustering_method()

        # Fit the clustering algorithm to the data to identify sequences in the data
        clustering_cols = pd.DataFrame(data[self.col])
        clusters = clustering.fit_predict(clustering_cols)

        for cluster in set(clusters):
            sequence = data[clusters == cluster]

            # Sort the sequence by its column values
            sequence = sequence.sort_values(by=self.col)

            # Get the targets for the sequence
            current_targets = targets[clusters == cluster]

            # Sort the targets by the sequence
            current_targets = current_targets[sequence.index]

            # Edge case - targets only contain one value. In this case, add dummy data and a dummy target
            if len(pd.unique(current_targets)) == 1:
                dummy_data = pd.DataFrame(np.zeros((1, len(sequence.columns))), columns=sequence.columns,
                                          index=range(self.max_index, self.max_index + 1))
                unique_target = pd.unique(current_targets)[0]

                # Add a dummy target that is the opposite of the unique target
                dummy_target = pd.Series([(unique_target + 1) % 2], index=[self.max_index])
                sequence = pd.concat([sequence, dummy_data])
                current_targets = pd.concat([current_targets, dummy_target])

            sequences.append(sequence)
            sequence_targets.append(current_targets)

        return sequences, sequence_targets

    def normalize(self, sequences: List[pd.DataFrame] = None):
        """
        Normalize the data in the sequences.
        Either normalize the data in the sequences provided as input to this method,
        or normalize the data in the stored sequences if no input is provided.
        :return: The normalized sequences, if sequences are provided.
        """
        if sequences is None:
            for i in range(len(self.sequences)):
                std = self.sequences[i].std()
                # Locate zeroes in the standard deviation and replace them with 1, to avoid division by zero
                std[std == 0] = 1
                # Avoid normalizing boolean columns
                for col in self.sequences[i].columns:
                    if not is_boolean_column(self.sequences[i][col]):
                        self.sequences[i][col] = (self.sequences[i][col] - self.sequences[i][col].mean()) / std[col]
        else:
            for i in range(len(sequences)):
                std = sequences[i].std()
                std[std == 0] = 1
                # Avoid normalizing boolean columns
                for col in sequences[i].columns:
                    if not is_boolean_column(sequences[i][col]):
                        sequences[i][col] = (sequences[i][col] - sequences[i][col].mean()) / std[col]
            return sequences


    def pad_sequences(self, max_sequence_length: int, sequences: List[pd.DataFrame] = None):
        """
        Pad the sequences to the same length.
        Either pad the sequences provided as input to this method to the same length,
        or pad the stored sequences to the same length if no input is provided.
        :param max_sequence_length: The maximum length of the sequences.
        :return: The padded sequences, if sequences are provided.
        """
        if sequences is None:
            for i in range(len(self.sequences)):
                sequence = self.sequences[i]
                if len(sequence) < max_sequence_length:

                    # Pad the sequence with zeros
                    padding = pd.DataFrame(np.zeros((max_sequence_length - len(sequence), len(sequence.columns))),
                                           columns=sequence.columns,
                                           index=range(self.max_index, self.max_index + max_sequence_length - len(sequence)))
                    self.sequences[i] = pd.concat([sequence, padding])
        else:
            for i in range(len(sequences)):
                sequence = sequences[i]
                if len(sequence) < max_sequence_length:
                    padding = pd.DataFrame(np.zeros((max_sequence_length - len(sequence), len(sequence.columns))),
                                           columns=sequence.columns,
                                           index=range(self.max_index, self.max_index + max_sequence_length - len(sequence)))
                    sequences[i] = pd.concat([sequence, padding])
                # If the sequence is longer than the maximum sequence length, split it into segments
                # and pad the last segment
                elif len(sequence) > max_sequence_length:
                    num_segments = len(sequence) // max_sequence_length
                    segments = []
                    for j in range(num_segments):
                        segment = sequence[j * max_sequence_length:(j + 1) * max_sequence_length]
                        segments.append(segment)

                    # Pad the last segment, if necessary
                    final_segment = segments[-1]
                    padding = pd.DataFrame(np.zeros((max_sequence_length - len(final_segment), len(sequence.columns))),
                                             columns=sequence.columns,
                                             index=range(self.max_index, self.max_index + max_sequence_length - len(final_segment)))
                    final_segment = pd.concat([final_segment, padding])
                    segments[-1] = final_segment
                    # Delete the original sequence and append the segments to the list of sequences
                    del sequences[i]
                    sequences.extend(segments)
            return sequences

    def __iter__(self):
        return self

    def __next__(self):
        self.previous_index = self.index
        self.index += 1
        self.index = self.index % len(self.sequences)
        current_sequence = self.sequences[self.index]
        # If the data is to be flattened, reshape the data to a 1D array
        if self.flatten:
            return current_sequence.to_numpy().reshape(-1,)
        return current_sequence.to_numpy()

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
        :return: The first sequence
        """
        self.index = 0
        self.previous_index = 0
        if self.flatten:
            return self.sequences[self.index].to_numpy().reshape(-1,)
        return self.sequences[self.index].to_numpy()

    def is_done(self):
        """
        Check if the sequence is done
        :return:
        """
        return self.index <= self.previous_index

    def sequence_by_ranges(self, data, targets = None):
        """
        Sequences the data by the ranges computed on the training data
        :return:
        """
        data = data.sort_values(by=self.col)
        sequences = []
        sequences_targets = []
        for i in range(len(self.ranges)):
            # For the lowest range, include all values less than the maximum value
            if i == 0:
                sequence = data[data[self.col] <= self.ranges[i][1]]
            # For all other ranges, include values between the maximum value of the previous range and the maximum value
            # of the current range.
            # This is done to account for the fact that the ranges may not be continuous, and there may be gaps between
            # the ranges, where some data points may not be included in any range.
            else:
                sequence = data[(data[self.col] <= self.ranges[i][1]) & (data[self.col] > self.ranges[i - 1][1])]
            if targets is not None:
                sequence_targets = targets[sequence.index]

                # Edge case - targets only contain one value. In this case, add dummy data and a dummy target
                if len(pd.unique(sequence_targets)) == 1:
                    dummy_data = pd.DataFrame(np.zeros((1, len(sequence.columns))), columns=sequence.columns,
                                              index=range(self.max_index, self.max_index + 1))
                    unique_target = pd.unique(sequence_targets)[0]

                    # Add a dummy target that is the opposite of the unique target
                    dummy_target = pd.Series([(unique_target + 1) % 2], index=[self.max_index])
                    sequence = pd.concat([sequence, dummy_data])
                    sequence_targets = pd.concat([sequence_targets, dummy_target])
            else:
                sequence_targets = None
            sequences.append(sequence)
            sequences_targets.append(sequence_targets)
        return sequences, sequences_targets



