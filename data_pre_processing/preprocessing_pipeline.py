from categorical_feature_processor import CategoricalFeatureProcessor
from collinearity_processor import CollinearityProcessor
from missing_values_processor import MissingValuesProcessor


class PreprocessingPipeline:

    def __init__(self, df, cat_cols, threshold):
        self.df = df
        self.cat_cols = cat_cols
        self.threshold = threshold

    def preprocess(self):
        cat_processor = CategoricalFeatureProcessor(self.df, self.cat_cols)
        self.df = cat_processor.preprocess()
        missing_processor = MissingValuesProcessor(self.df)
        self.df = missing_processor.process()
        coll_processor = CollinearityProcessor(self.threshold)
        coll_processor.fit(self.df)
        self.df = coll_processor.transform(self.df)
        return self.df
