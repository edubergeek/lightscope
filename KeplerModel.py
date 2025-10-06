#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import sys
import os
import numpy as np
import argparse
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, AdamW
from keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Concatenate, BatchNormalization, Conv1DTranspose
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, Conv3D

from KerasModel import KerasModel, BestEpoch, PlotLoss

def reshape_Pad(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Pad':
        fromPos = int(t['arg1'])
        toPos = int(t['arg2'])
        featureVal = float(t['arg3'])
        # initialize an array of "featureVal" values in the correct shape
        #val = np.ones((toPos-fromPos)) * featureVal
        #pad = tf.constant(val, dtype=tf.float32)
        pad = tf.slice(features, [0], [toPos-fromPos])
        features = tf.concat([features, pad], 0)
        
    return features, targets

def replace_NaN(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'NaN':
        featureVal = float(t['arg1'])
        targetVal = float(t['arg2'])
        mask = tf.math.is_nan(features)
        maskVal = tf.cast(tf.ones_like(mask), tf.float32) * tf.constant(featureVal, dtype=tf.float32)
        features = tf.where(mask, maskVal, features)
        mask = tf.math.is_nan(targets)
        maskVal = tf.cast(tf.ones_like(mask), tf.float32) * tf.constant(targetVal, dtype=tf.float32)
        targets = tf.where(mask, maskVal, targets)
    return features, targets

def decode_tfr(record_bytes):
    schema =  {
      "id": tf.io.FixedLenFeature([], dtype=tf.string),
      "name": tf.io.FixedLenFeature([], dtype=tf.string),
      "p": tf.io.FixedLenFeature([], dtype=tf.float32),
      "x":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
      "y":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
    }    
    example = tf.io.parse_single_example(record_bytes, schema)
    x = tf.stack([example['x'], example['y']], axis=0)
    y = example['p']
    return x, y

class KeplerModel(KerasModel):
  def __init__(self, modelname, hyperParam):
    super().__init__()
    
    self.sModelName = modelname
    self.sModelPath = 'model'
    self.modelStep = 0
    self.modelVersion = hyperParam["version"]
    self.modelRevision = hyperParam["revision"]
    self.modelTrial = hyperParam["trial"]
    self.modelEpoch = hyperParam["epoch"]
    self.sModelSuffix = ""
    self.earlyStopPatience = hyperParam["patience"]
    self.earlyStopThreshold = hyperParam["threshold"]
    self.batchSize = hyperParam["batch_size"]
    self.epochs = hyperParam["epochs"]
    self.begin = hyperParam["begin"]
    self.monitor = 'val_loss'
    self.loss = 'mse'
    self.mode = 'min'
    self.isTrained = False
    self.useTensorboard = False
    self.hparam = hyperParam

  def GetModelFile(self):
    return "%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, self.sModelName, self.modelVersion, self.modelRevision, self.modelTrial, self.modelEpoch, self.sModelSuffix)

  def LoadModel(self):
    self.modelFile = self.GetModelFile()
    print("Loading ", self.modelFile)
    self.model = tf.keras.models.load_model(self.modelFile)
    if self.hparam['arch'] == 'NC' or self.hparam['arch'] == 'CC':
      self.Compile(self.model, loss='sparse_categorical_crossentropy', metric=['accuracy'])
    else:
      self.Compile(self.model)

  def Compile(self, model, loss=None, metric=None, loss_weight=None):
    # compile the model  
    self.optimizer = Adam(learning_rate=self.hparam['lr'])
    if self.hparam['arch'] == 'AE':
      losses = {
        "target_output": self.loss,
        "input_output": self.loss,
      }
      lossWeights = {"target_output": 1.0, "input_output": self.hparam['epsilon']}
      model.compile(optimizer=self.optimizer, loss=losses, loss_weights=lossWeights, metrics=metric)
    else:
      if loss is None:
        loss = self.loss
      if metric is None:
        metric = [loss]
      model.compile(optimizer=self.optimizer, loss=loss, metrics=metric, loss_weights=loss_weight)
    return model

  def Train(self, ds, dsv):
    # Set the model file name
    filepath="%s/%st%d-e{epoch:d}%s" %(self.sModelPath, self.GetModelFullName(), self.GetModelTrial(), self.sModelSuffix)
    # default checkpoint settings
    checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode=self.mode)
    # plot loss after each epoch
    bestepoch = BestEpoch(metric=self.monitor, mode=self.mode)
    self.bestEpoch = 0

    self.callbacks = [checkpoint, bestepoch]

    if self.earlyStopPatience > 0:
      earlystop = EarlyStopping(monitor=self.monitor, mode=self.mode, patience=self.earlyStopPatience, min_delta=self.earlyStopThreshold)
      self.callbacks.append(earlystop)

#    if self.useTensorboard:
#      tensorboard_callback = TensorBoard(log_dir="./logs")
#      self.callbacks.append(tensorboard_callback)
#    else:
#      plotloss = PlotLoss(metric=self.monitor)
#      self.callbacks.append(plotloss)
    plotloss = PlotLoss(metric=self.monitor)
    self.callbacks.append(plotloss)
          
    self.model.fit(ds, validation_data=dsv, initial_epoch=self.begin, epochs=self.epochs, batch_size=self.batchSize, callbacks=self.callbacks, verbose=1, shuffle=1)
    
    self.isTrained = True  
    self.bestEpoch = bestepoch.get_best_epoch()
    return self.bestEpoch

  def SetMonitor(self, monitor):
    self.monitor = monitor
    if monitor == 'val_accuracy':
      self.mode = 'max'

  def GetDataSet(self, filenames, transform):   
    at = tf.data.AUTOTUNE
    
    dataset = (
      tf.data.TFRecordDataset(filenames, num_parallel_reads=at)
      .map(decode_tfr, num_parallel_calls=at)
    )
    
    if not transform == '-':
      for t in self.hparam[transform]:
        if t['name'] == 'NaN':
          dataset = dataset.map(replace_NaN, num_parallel_calls=at)
        if t['name'] == 'Pad':
          dataset = dataset.map(reshape_Pad, num_parallel_calls=at)

    if self.hparam['arch'] == 'AE':
      dataset = dataset.map(remap_autoencoder, num_parallel_calls=at)
    
    dataset = dataset.batch(self.batchSize).prefetch(at).repeat(count=1)

    return dataset
       
  def DataSet(self, path, pattern):
    pattern_list = pattern.split()
    filenames = []
    for pat in pattern_list:
      filenames += tf.io.gfile.glob(os.path.join(path, pat))
    TRANSFORM='transform'
    return self.GetDataSet(filenames, TRANSFORM)



