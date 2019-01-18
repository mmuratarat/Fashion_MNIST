import os
import pandas as pd
import numpy as np
import math

def one_hot(y, num_classes):
    return (np.squeeze(np.eye(num_classes)[y.reshape(-1)])).astype(np.int32)

def miniBatch(x, y, batchSize):
    numObs  = x.shape[0]
    batches = [] 
    batchNum = math.floor(numObs / batchSize)
    
    if numObs % batchSize == 0:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
    else:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
        xBatch = x[batchNum * batchSize:, :]
        yBatch = y[batchNum * batchSize:, :]
        batches.append((xBatch, yBatch))
    return batches