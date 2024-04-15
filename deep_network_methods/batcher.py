import pandas as pd


class Batcher:
    """
    A batcher for the reinforcement learning algorithms.
    Cycle through the data and return a batch of data, always returning the same batch size.
    """

    def __init__(self, X, batch_size, dtype=None):
        self.X = X
        self.index = 0
        self.prev_index = 0
        self.batch_size = batch_size
        if dtype is not None:
            self.X = self.X.astype(dtype)

    def next_batch(self):
        """
        :return: The next batch of data
        """
        # Account for cases where the batch size does not divide the number of samples
        # In this case, we loop back to the beginning of the data and append those samples to the batch
        if self.index + self.batch_size > len(self.X):
            batch_len = len(self.X) - self.index
        else:
            batch_len = self.batch_size
        batch_X = self.X[self.index:self.index + batch_len]
        # Append the beginning of the data to the batch if necessary
        if batch_len < self.batch_size:
            batch_X = pd.concat([batch_X, self.X[:self.batch_size - batch_len]])
        # Update the index, ensuring that it wraps around when necessary
        self.prev_index = self.index
        self.index = (self.index + self.batch_size) % len(self.X)
        return batch_X.to_numpy().reshape(-1, )

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def reset(self):
        self.index = 0
        self.prev_index = 0
        return self.next_batch()

    def is_done(self):
        return self.index <= self.prev_index
