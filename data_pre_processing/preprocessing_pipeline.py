from .categorical_feature_processor import CategoricalFeatureProcessor
from .collinearity_processor import CollinearityProcessor
from .missing_values_processor import MissingValuesProcessor


class PreprocessingPipeline:

    def __init__(self, df, cat_cols, threshold, target_name):
        self.df = df
        self.cat_cols = cat_cols
        self.threshold = threshold
        self.target_name = target_name

    def preprocess(self):
        cat_processor = CategoricalFeatureProcessor(self.df, self.cat_cols)
        self.df, continuous_cols, cat_cols = cat_processor.preprocess()
        missing_processor = MissingValuesProcessor(self.df)
        self.df = missing_processor.process()
        coll_processor = CollinearityProcessor(target_name=self.target_name, threshold=self.threshold)
        coll_processor.fit(self.df)
        self.df = coll_processor.transform(self.df)
        # Remove the names of the columns that were removed by the collinearity processor
        continuous_cols = [col for col in continuous_cols if (col in self.df.columns and col != self.target_name)]
        if cat_cols is not None:
            cat_cols = [col for col in cat_cols if col in self.df.columns]
        return self.df, continuous_cols, cat_cols
