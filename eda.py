import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train_pr = ProfileReport(train)
test_pr = ProfileReport(test)

train_pr.to_file("train.html")
test_pr.to_file("test.html")