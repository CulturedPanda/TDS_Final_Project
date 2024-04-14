from .categorical_feature_processor import CategoricalFeatureProcessor
from .collinearity_processor import CollinearityProcessor
from .missing_values_processor import MissingValuesProcessor


class PreprocessingPipeline:

    def __init__(self, df, cat_cols, threshold):
        self.df = df
        self.cat_cols = cat_cols
        self.threshold = threshold

    def preprocess(self):
        cat_processor = CategoricalFeatureProcessor(self.df, self.cat_cols)
        self.df, continuous_cols, cat_cols = cat_processor.preprocess()
        missing_processor = MissingValuesProcessor(self.df)
        self.df = missing_processor.process()
        coll_processor = CollinearityProcessor(0.5)
        coll_processor.fit(self.df)
        self.df = coll_processor.transform(self.df)
        # Remove the names of the columns that were removed by the collinearity processor
        continuous_cols = [col for col in continuous_cols if col in self.df.columns]
        cat_cols = [col for col in cat_cols if col in self.df.columns]
        return self.df, continuous_cols, cat_cols
