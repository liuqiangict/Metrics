
import numpy as np
import random

from sklearn import metrics

import matplotlib.pyplot as plt

pred = np.array([0.61202417, 0.41762708, 0.21843196, 0.63177703, 0.07735603, 0.27251836, 0.56929917, 0.84448728, 0.93489042, 0.04836425])
y = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 1])
print(pred)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)

print(thresholds)
print(fpr)
print(tpr)

score = metrics.roc_auc_score(y, pred)
print(score)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show(block=True)

