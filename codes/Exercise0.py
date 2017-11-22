#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 00:05:42 2017

@author: andrewwork
"""

# Exercise 0

import matplotlib.pyplot as plt
import numpy as np

NPOINTS = 100

div_points = np.random.rand(2,2)
w1 = (div_points[1,1] - div_points[0,1])/(div_points[1,0] - div_points[0,0])
w0 = div_points[0,1]- div_points[0,0]*w1
w = np.array([w0,w1,-1])

test_points = np.ones([NPOINTS,4])
test_points[:,1:3] = np.random.rand(NPOINTS,2)

test_points[:,-1] = np.dot(test_points[:,0:3],w)

positives = test_points[test_points[:,-1]>0,:]
negatives = test_points[test_points[:,-1]<0,:]

x = np.linspace(0,1)
xpos = positives[:,1]
ypos = positives[:,2]
xneg = negatives[:,1]
yneg = negatives[:,2]

plt.plot(xneg,yneg,'.g')
plt.plot(xpos,ypos,'.r')
plt.plot(x,w1*x+w0)
plt.axis([0,1,0,1])
plt.legend(['Negative','Positive'])
plt.show()
