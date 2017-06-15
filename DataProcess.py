import pandas as pd

import Classifier

origin_data = pd.read_csv('./origin_data/DIABETIC_DATA.csv')
diabetic_label = origin_data['READMITTED'].replace(['NO', '<30', '>30'], [0, 1, 1]).as_matrix()
origin_data = origin_data.drop('READMITTED', 1)
one_hot_data = pd.get_dummies(origin_data[0:-1]).as_matrix()
Classifier.random_forest(one_hot_data[0:int(len(origin_data)*0.8)],diabetic_label[0:int(len(origin_data)*0.8)],
                         one_hot_data[int(len(origin_data)*0.8):],diabetic_label[int(len(origin_data)*0.8):])
# for i in range(1,100):
#     accuracy, importance = Classifier.random_forest([[1, 2, 3], [1, 2, 1]], [1, 0], [[1, 2, 2], [1, 2, 1]], [0, 0], n_trees=i)
#     print(i, accuracy, importance)
