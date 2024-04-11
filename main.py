from sklearn.datasets import fetch_california_housing

from filter_methods import WeightedCombination
from data_pre_processing import PreprocessingPipeline

data = fetch_california_housing()
X, y = data.data, data.target

preprocessor = PreprocessingPipeline(X, ['ocean_proximity'], 0.9)
X = preprocessor.preprocess()

filter_method = WeightedCombination()
filter_method.fit(X, y)
selected_features = filter_method.transform(X, num_features=5)
print(selected_features)