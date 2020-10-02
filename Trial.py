import numpy as np


str = "263,211,324,339,8 165,264,253,372,8 241,194,295,299,8"
line = str.split()
bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[0:]])
print(bboxes)
print(bboxes.shape)

list = []
a = [1, 2]
b = [3, 4]
list.append(a)
list.append(b)
list = np.array(list)
print(list)
print(list.shape)
