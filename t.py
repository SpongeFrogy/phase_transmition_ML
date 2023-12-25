from model.classification_model import ClassifierModel
import numpy as np



y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0 ,0 ,0 ,0 ,0 ,0])

y_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
          0, 0, 0 ,0 ,0 ,0 ,0 ,1])

y_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
          0, 0, 0 ,0 ,0 ,0 ,0 ,0])

y_3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
          0, 0, 0 ,0 ,0 ,0 ,0 ,0])


print(ClassifierModel.score(y_true, y_1))
print(ClassifierModel.score(y_true, y_2))
print(ClassifierModel.score(y_true, y_3))


