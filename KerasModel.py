#!/usr/bin/env python
# coding: utf-8
# %%
"""
Uncomment the following for CPU not GPU
"""
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv1DTranspose
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


# %%


# updatable plot
# a minimal example (sort of)

class BestEpoch(Callback):
  def __init__(self, metric='val_loss', mode='min'):
    super().__init__()
    self.metric = metric
    self.mode = mode
    

  def on_train_begin(self, logs={}):
    self.bestEpoch = 0
    if self.mode == 'min':
      self.bestLoss = 1e8
    else:
      self.bestLoss = -1e8

  def on_epoch_end(self, epoch, logs={}):
    valLoss = logs.get(self.metric)
    if self.mode == 'min' and valLoss < self.bestLoss:
      self.bestLoss = valLoss
      self.bestEpoch = epoch+1
    elif valLoss > self.bestLoss:
      self.bestLoss = valLoss
      self.bestEpoch = epoch+1

  def get_best_epoch(self):
    return self.bestEpoch



# %%


class PlotLoss(Callback):
  def __init__(self, metric='val_loss'):
    super().__init__()
    self.metric = metric
    matplotlib.interactive(True)
    
#  def __del__(self):
#    plt.close('all')

  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.loss_y = []
    self.metric_y = []
    self.logs = []
    self.fig = plt.figure()


#  def on_train_batch_end(self, batch, logs={}):
#    self.logs.append(logs)
#    self.x.append(self.i)
#    self.loss_y.append(logs.get('loss'))
#    self.metric_y.append(logs.get(self.metric))
#    self.i += 1
#
#    #clear_output(wait=True)
#    plt.clf()
#    plt.plot(self.x, self.loss_y, label="loss")
#    plt.plot(self.x, self.metric_y, label=self.metric)
#    plt.legend()
#    plt.show(block=False);
#    plt.draw();
#    plt.pause(0.05)
#    #print(logs)

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    self.loss_y.append(logs.get('loss'))
    self.metric_y.append(logs.get(self.metric))
    self.i += 1

    #clear_output(wait=True)
    plt.clf()
    plt.plot(self.x, self.loss_y, label="loss")
    plt.plot(self.x, self.metric_y, label=self.metric)
    plt.legend()
    plt.show(block=False);
    plt.draw();
    plt.pause(0.05)
    print(logs)




# %%


""" 
Keras base neural network model
"""

class KerasModel():
  def __init__(self):

    self.optimizer = Adam()
    self.loss = 'mse'
    self.metrics = ['mse']
    self.monitor = 'val_loss'
    self.saveBestOnly = True
    self.saveWeightsOnly = False
    self.isTrained = False

    #m = Sequential()
    #m.add(Dense(100, activation='relu'))
    #m.add(Dense(1, activation='sigmoid'))

    #self.Compile()
    self.epochs = 100
    self.batchSize = 512
    self.bestEpoch = 0

  def SetModel(self, model):
    self.model = model

  def SetModelPath(self, path):
    self.modelPath = path

  def SetModelName(self, name):
    self.modelName = name

  def GetModelName(self):
    return self.modelName

  def SetModelVersion(self, version):
    self.modelVersion = version

  def SetModelRevision(self, revision):
    self.modelRevision = revision

  def SetModelTrial(self, trial):
    self.modelTrial = trial

  def SetLoss(self, loss):
    self.loss = loss

  def SetMetrics(self, metrics):
    self.metrics = metrics

  def SetMonitor(self, monitor):
    self.monitor = monitor

  def SaveBestOnly(self):
    self.saveBestOnly = True

  def SaveEveryEpoch(self):
    self.saveBestOnly = False

  def SaveWeightsOnly(self):
    self.saveWeightsOnly = True

  def SaveFullModel(self):
    self.saveWeightsOnly = False
    
  def GetModelFullName(self):
    if self.modelStep:
      return "%sv%dr%ds%d" %(self.sModelName, self.modelVersion, self.modelRevision, self.modelStep)
    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

  def GetModelVersion(self):
    return self.modelVersion

  def GetModelRevision(self):
    return self.modelRevision

  def GetModelTrial(self):
    return self.modelTrial

  def SetTensorboard(self, useTensorboard = True):
    self.useTensorboard = useTensorboard

  def SetLearningRate(self, rate):
    self.learningRate = rate

  def SetOptimizer(self, optimizer):
    self.optimizer = optimizer(learning_rate=self.learningRate)

  def Compile(self):
    self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

  def ResetWeights(self):
    if self.isTrained:
      weights = []
      initializers = []
      for layer in self.model.layers:
        #if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
        if isinstance(layer, (Dense, Conv1D, Conv2D, Conv1DTranspose)):
          weights += [layer.kernel, layer.bias]
          initializers += [layer.kernel_initializer, layer.bias_initializer]
        elif isinstance(layer, BatchNormalization):
          weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
          initializers += [layer.gamma_initializer,
                       layer.beta_initializer,
                       layer.moving_mean_initializer,
                       layer.moving_variance_initializer]
      for w, init in zip(weights, initializers):
        w.assign(init(w.shape, dtype=w.dtype))

    self.isTrained = False

  def SetEpochs(self, epochs):
    self.epochs = epochs

  def SetBatchSize(self, batch_size):
    self.batchSize = batch_size

  def SetModel(self, model):
    self.model = model

  def LoadModel(self):
    modelFile = self.GetModelFile()
    print("Loading model %s ..."%(modelFile))
    self.model = tf.keras.models.load_model(modelFile)

  def SaveModel(self):
    self.model.save(self.GetModelFile())

  def GetBestEpoch(self):
    return self.bestEpoch

  def SetCallbacks(self, cb):
    self.callbacks = cb

  def Fit(self):
    # Fit the model
    #self.model.fit(self.dfTrain[self.GetFeatureKeys()], self.dfTrain[self.keyTarget], validation_split=0.10, epochs=self.epochs, batch_size=self.batchSize, callbacks=self.callbacks, verbose=0, shuffle=1)
    xValid = self.dfValid[self.GetFeatureKeys()]
    yValid = self.dfValid[self.keyTarget]
    self.model.fit(self.dfTrain[self.GetFeatureKeys()], self.dfTrain[self.keyTarget], validation_data=(xValid, yValid), epochs=self.epochs, batch_size=self.batchSize, callbacks=self.callbacks, verbose=0, shuffle=1)
    self.isTrained = True

  def Train(self):
    # Set the modei file name
    filepath="%s/%st%d-e{epoch:d}%s" %(self.sModelPath, self.GetModelFullName(), self.GetModelTrial(), self.sModelSuffix)
    # default checkpoint settings
    checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1, save_best_only=self.saveBestOnly, save_weights_only=self.saveWeightsOnly, mode='min')
    # plot loss after each epoch
    plotloss = PlotLoss()
    bestepoch = BestEpoch()
    if self.earlyStopPatience > 0:
      earlystop = EarlyStopping(monitor=self.monitor, mode='min', patience=self.earlyStopPatience, min_delta=self.earlyStopThreshold)
      self.callbacks = [checkpoint, plotloss, bestepoch, earlystop]
    else:
      self.callbacks = [checkpoint, plotloss, bestepoch]
    self.bestEpoch = 0
    # Fit the model
    self.Fit()
    self.bestEpoch = bestepoch.get_best_epoch()


  def Predict(self):
    predictBatchSize = 32
    #self.dfTrain[self.keyPrediction] = self.model.predict(self.dfTrain[self.GetFeatureKeys()], batch_size=predictBatchSize)
    self.dfValid[self.keyPrediction] = self.model.predict(self.dfValid[self.GetFeatureKeys()], batch_size=predictBatchSize)

    X = self.GetTournament()[self.GetFeatureKeys()].to_numpy()
    yPredSplit = int(self.GetNTournament() / 2)
    yPred1 = self.model.predict(X[:yPredSplit], batch_size=self.batchSize)
    yPred2 = self.model.predict(X[yPredSplit:], batch_size=self.batchSize)
    self.dfTournament[self.keyPrediction] = np.concatenate((yPred1, yPred2))
    #self.dfTournament[self.keyPrediction] = self.model.predict(self.dfTournament[self.GetFeatureKeys()], batch_size=predictBatchSize)

  def PredictY(self, X, batch_size=32, splits=2):
    m = int(len(X))
    split = int(m / 2)
    y = np.zeros((m,1))
    for s in range(splits-1):
      y[split*s:split*s+split] = self.model.predict(X[split*s:split*s+split], batch_size=batch_size)
    s = splits - 1
    y[split*s:] = self.model.predict(X[split*s:], batch_size=batch_size)
    return y

