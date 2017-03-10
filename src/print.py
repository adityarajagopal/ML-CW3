#!/usr/local/env

import json
import numpy as np
import matplotlib.pyplot as plt

with open('files/graphs.json') as data_file:    
    data = json.load(data_file)

x = data["axes"]["x"]
y = data["axes"]["y"]
orig = data["labels"]["original"]
learnt = data["labels"]["learned"]
colour = ["b" if i is 1 else "r" for i in data["labels"]["colour"]]

plt.figure()
plt.title(data["labels"]["title"])
plt.xlabel(data["labels"]["xlabel"])
plt.ylabel(data["labels"]["ylabel"])
plt.scatter(x, y, s=1, c=colour, label=data["labels"]["legend"])
plt.scatter(x, orig, s=1, c="black")
plt.scatter(x, learnt, s=1, c="green")
plt.ylim((-1,2))
plt.legend()
plt.show()


