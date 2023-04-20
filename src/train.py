import cnn
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split as tts
import numpy as np
import json
import tensorflow as tf

if os.path.exists("saved_model_CNN") == False:
        os.system('mkdir saved_model_CNN')

dirpath = os.getcwd()

filepath = dirpath + "/source/dataset30000_32.csv"

dataset = pd.read_csv(filepath, delimiter = ',')

x_feature = "lattice_configs"

y_feature = "T"
# y_feature = "magnetization_records"
# y_feature = "Jij"

for name in dataset.columns:
    if name == x_feature:
        X = dataset[name].values

    if name == y_feature:
        Y = dataset[name].values

YY = Y
# YY = []
# for tmp in Y:
#     tmp = json.loads(tmp)
#     YY.append(tmp)
#     # print(tmp)

# YYY = []
# for label in YY:
#     y = sum(label)/len(label)
#     # print(sum(label))
#     # print(len(label))
#     if abs(y) < 0.5:
#         YYY.append(0)
#     else:
#         YYY.append(1)
# YY = YYY

XX = []
for tmp in X:
    tmp = tmp.strip('"')
    tmp = json.loads(tmp)
    XX.append(tmp)

YY = np.array(YY)
XX = np.array(XX)
print(XX.shape)
print(YY.shape)
x, xt, y, yt = tts(XX,YY)
print(y)
print(x[0].shape)

model = cnn.build_model(x[0])
model.summary()

fitting = model.fit(x ,y , epochs=500, batch_size=50 ,validation_split=0.1)

score = model.evaluate(xt, yt, verbose = 1)

y_pre = model.predict(xt)


y_pre.tolist()
index = 0
for i in range(len(y_pre)):
if abs(yt[i] - y_pre[i]) < 0.05:
    index += 1
acc = index/len(y_pre)
print("accurate = " + str(acc))

model.save('saved_model_CNN/my_model_1')

print("score = " + str(score))

# print(score)
import matplotlib.pyplot as plt
line = np.linspace(0,10)
plt.plot(line,line, color = 'red',label = 'predicted = exact')
plt.scatter(yt,y_pre,s=5.0)
plt.xlabel("Exact",fontsize=12)
plt.ylabel(r"$predict$",fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("train_1_"+ str(acc) +".png")
