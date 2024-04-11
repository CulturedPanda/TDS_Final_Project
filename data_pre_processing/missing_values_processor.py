class MissingValuesProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        self.data = self.data.fillna(self.data.mean())
        return self.data