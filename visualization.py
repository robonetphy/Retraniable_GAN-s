# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 02:11:02 2019

@author: DELL
"""


import matplotlib.pyplot as plt
import os
import pickle


G_losses = []
D_losses = []

def loss_load():
     if os.path.isfile('loss/G_list.loss') and os.path.isfile('loss/D_list.loss'):
        with open("./loss/G_list.loss", "rb") as fp:
            global G_losses
            G_losses  = pickle.load(fp)
        with open("./loss/D_list.loss", "rb") as fp:
            global D_losses
            D_losses  = pickle.load(fp)
            
loss_load()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()