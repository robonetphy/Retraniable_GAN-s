# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:19:32 2019

@author: DELL
"""
import matplotlib.pyplot as plt
import pickle



loss=[1,2,3,4,5,]
with open("./loss/list.loss", "wb") as fp: 
          pickle.dump(loss, fp)
with open("./loss/list.loss", "rb") as fp:
            loss2  = pickle.load(fp)
            print (loss2)
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(loss2,label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()