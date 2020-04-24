import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

y_score = np.array([107.4,106.5,105.4,98.0])
y_true = np.array([1,1,1,0]) # 異常データを１

print(f"y_score => {y_score}")
print(f"y_true => {y_true}")

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
print(f"thresholds => {thresholds}")
auc = metrics.auc(fpr, tpr)
print(f"AUC => {auc}")
plt.plot(fpr, tpr)
plt.legend()
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.show()