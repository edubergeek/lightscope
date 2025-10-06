import os
from enum import IntEnum
from datetime import datetime
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, FloatList, Int64List, BytesList
import pandas as pd
import numpy as np
from astropy.io import fits
from sklearn.model_selection import train_test_split


class TrainingSet(IntEnum):
  DONOTUSE = 0
  TRAIN=1
  VALID=2
  TEST=3

# general purpose wrappers to convert Python types to flattened lists for Tensorflow
def _floatvector_feature(value):
  return Feature(float_list=FloatList(value=value))

def _float_feature(value):
  return Feature(float_list=FloatList(value=[value]))

def _int64_feature(value):
  return Feature(int64_list=Int64List(value=[value]))

def _bytes_feature(value):
  return Feature(bytes_list=BytesList(value=[value]))

def _dtype_feature(ndarray):
  """match appropriate tf.train.Feature class with dtype of ndarray. """
  assert isinstance(ndarray, np.ndarray)
  dtype_ = ndarray.dtype
  if dtype_ == np.float64 or dtype_ == np.float32:
    return lambda array: Feature(float_list=FloatList(value=array))
  elif dtype_ == np.int64:
    return lambda array: Feature(int64_list=Int64List(value=array))
  else:
    raise ValueError("The input should be numpy ndarray: got {}".format(ndarray.dtype))

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))



def load_fits(filnam, idx=0):
  hdulist = fits.open(filnam)
#  meta = {}
#  # Need to check for idx in range or throw an Exception
#  h = list(chunkstring(hdulist[idx].header, 80))
#  for index, item in enumerate(h):
#    m = str(item)
#    mh = list(chunkstring(m, 80))
#    #print(mh)
#    for ix, im in enumerate(mh):
#      #print(index, ix, im)
#      mm = im.split('/')[0].split('=')
#      if len(mm) == 2:
#        #print(index, ix, mm[0], mm[1])
#        meta[mm[0].strip()] = mm[1].strip()
  #nAxes = int(meta['NAXIS'])
  # check this logic in MOPS FITS files
  data = hdulist[idx].data
  meta = hdulist[idx].header
  hdulist.close
  return data, meta


# Inherit and overload the TrainingExample method at a minimum
# Optionally overload the Load and init as needed
# create an array of integers in the range 0 to nExamples - 1
# self.example is a list of training examples
# self.nExamples is the number of items in the list
# SaveDataset will index self.example and write it to storage

class ETL():
  def __init__(self, root_dir, output_dir, shard_size = 5000, train_split = 0.9, valid_split=0.05):
    self.rootDir = root_dir
    self.outputDir = output_dir
    self.shardSize = shard_size
    self.outputPart = [None, "train", "valid", "test"]

    self.partitionTag = None
    self.trainPercent = train_split
    self.validPercent = valid_split
    self.testPercent = 1.0 - (train_split + valid_split)
    if self.testPercent < 0:
      raise Exception("train + valid splits must be <= 1.0")

  def SetPartitionTag(self, tag):
    self.partitionTag = tag

  def Load(self):
    self.nExamples = len(self.manifest)
    x_data = np.arange(self.nExamples)
    # x_data are the indices so y_data must match x_data
    y_data = np.copy(x_data)
    self.trainIdx, x_rem, _, y_rem = train_test_split(x_data, y_data, train_size=self.trainPercent)
    self.validIdx, self.testIdx, _, _ = train_test_split(x_rem, y_rem, train_size=self.validPercent/(self.validPercent+self.testPercent))

    self.nTrain = self.trainIdx.shape[0]
    self.nValid = self.validIdx.shape[0]
    self.nTest = self.testIdx.shape[0]

  def LoadExample(self, idx):
    # return a pair of x and y examples
    img, meta = load_fits(self.manifest['kepfits'][idx])
    img = np.asarray(img.T)
    img = img.flatten()
    x = {'image': img, 
         'tel': self.manifest['tel'],
         'filt': self.manifest['filt']=='o',
         'seeing': meta['SEEING'],
         'airmass': meta['AIRMASS'],
         'skymag': meta['SKYMAG']
        }
    y = {'class': self.manifest['class']}
    return x, y

  def OutputPath(self, training_set, n):
    # the TFRecord file containing the training set
    shard = int(n / self.shardSize)
    if self.partitionTag is None:
      path = '%s/%s_%d.tfr' % (self.outputDir, self.outputPart[training_set], shard)
    else:
      path = '%s/%s_%s_%d.tfr' % (self.outputDir, self.outputPart[training_set], self.partitionTag, shard)
    print(path, n)
    return path

  # self.example[m] could be a file path to load
  # the end result should be an x and y training example
  def Examples(self, m):
    x, y = LoadExample(m)
    return x, y

  def TrainingExample(self, m):
    x, y, id = self.Examples(m)
    feature = { 'x': _floatvector_feature(x['image']), 'y': _int64_feature(y['class']) }
    example = []
    # Create an example protocol buffer
    example.append(Example(features=Features(feature=feature)))
    return example, 1

  def SaveDataset(self):
    writer = [ None,
               tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TRAIN, 0)),
               tf.io.TFRecordWriter(self.OutputPath(TrainingSet.VALID, 0)),
               tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TEST, 0)),
             ]
    counter = [0,0,0,0]
    for m in range(self.nExamples):
      if m in self.trainIdx:
        cursor = TrainingSet.TRAIN
      if m in self.validIdx:
        cursor = TrainingSet.VALID
      if m in self.testIdx:
        cursor = TrainingSet.TEST

      example, examples = self.TrainingExample(m)
      for e in range(examples):
        if counter[cursor] % self.shardSize == 0:
          if counter[cursor] > 0:
            writer[cursor].close()
            writer[cursor] = tf.io.TFRecordWriter(self.OutputPath(cursor, counter[cursor]))
        writer[cursor].write(example[e].SerializeToString())
        counter[cursor] += 1

    if counter[TrainingSet.TRAIN] > 0:
      writer[TrainingSet.TRAIN].close()
    if counter[TrainingSet.VALID] > 0:
      writer[TrainingSet.VALID].close()
    if counter[TrainingSet.TEST] > 0:
      writer[TrainingSet.TEST].close()

  def Examples(self):
    return self.nExamples


class KeplerETL(ETL):
  def __init__(self, root_dir, output_dir, koidf, manidf, shard_size = 1000, train_split=0.9, valid_split=0.05):
    super().__init__(root_dir, output_dir, shard_size, train_split, valid_split)
    dt = datetime.now()

    self.manifest = self.JoinKeplerData(koidf, manidf)
    self.writer = None
    self.counter = [0,0,0,0]
    self.examples = [0,0,0,0]
    self.mask = [np.ones((1)),np.ones((1)),np.ones((1)),np.ones((1))]

  def JoinKeplerData(self, kepdf, manidf):
    """
    Joins two pandas DataFrames on the Kepler ID.

    Args:
        kepdf (pd.DataFrame): DataFrame with 'kepid', 'kepler_name', and 'koi_period' columns.
        manidf (pd.DataFrame): DataFrame with 'id', 'obstime', and 'fits_file.' columns.

    Returns:
        pd.DataFrame: Joined DataFrame with 'kepid', 'kepler_name', 'koi_period', 'obstime', and 'fits_file.' columns.
    """
    
    # Merge the dataframes on the 'kepid' column
    # Use inner join to keep only matching Kepler IDs
    df = pd.merge(kepdf[['kepid', 'kepler_name', 'koi_period']], manidf[['kepid', 'kepobs', 'kepfits']], on='kepid', how='inner')
    df.sort_values(by='kepfits', inplace=True)
    df[['kepler_name']] = df[['kepler_name']].fillna('')

    return df

  def Writer(self):
    if self.writer is None:
      self.writer = [ None,
        tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TRAIN, 0)),
        tf.io.TFRecordWriter(self.OutputPath(TrainingSet.VALID, 0)),
        tf.io.TFRecordWriter(self.OutputPath(TrainingSet.TEST, 0)),
      ]
    return self.writer

  def OpenDataset(self):
    self.Writer()

  def Load(self, mode, reload=True):
    # load train/valid/test data
    if reload:
      super().Load()
      self.mask = [np.ones((1)),np.ones((self.nTrain)),np.ones((self.nValid)),np.ones((self.nTest))]
      self.examples = [0,self.nTrain,self.nValid,self.nTest]


    if mode == TrainingSet.TRAIN:
      self.idx = self.trainIdx
    elif mode == TrainingSet.VALID:
      self.idx = self.validIdx
    else:
      self.idx = self.testIdx
    print('Loaded %d examples' % (len(self.idx)))
       
  def TimeSeries(self, fitsheader, fitsdata):
    # Read in the "BJDREF" which is the time offset of the time array.
    bjdrefi = fitsheader['BJDREFI']
    bjdreff = fitsheader['BJDREFF']

    # Read in the columns of data.
    times = fitsdata['time']
    uncorr_flux = fitsdata['SAP_FLUX']
    corr_flux = fitsdata['PDCSAP_FLUX']
    # Convert the time array to full BJD by adding the offset back in.
    obstime = times + bjdrefi + bjdreff
    return np.nan_to_num(obstime), np.nan_to_num(uncorr_flux), np.nan_to_num(corr_flux)

  def Chunkify(self, t, flux, chunklen):
    chunks = []
    series_len = len(t)
    nchunk = series_len // chunklen
    #print(f'Chunkify: {nchunk} chunks of len {chunklen}')
    offset = 0
    for c in range(nchunk):
      chunk = {'t': t[offset:offset+chunklen], 'flux': flux[offset:offset+chunklen]}
      chunks.append(chunk)
      offset += chunklen
    return chunks

  def LoadExample(self, m):
    # return pairs of x and y examples
    # note that m must be a true index into the full manifest
    fits_path = os.path.join(self.rootDir, self.manifest['kepfits'][m])
    tbl, meta = load_fits(fits_path, idx=1)
    #img = np.asarray(img.T)
    # split table tbl into chunks of x (time) dimension 256
    t, _, flux = self.TimeSeries(meta, tbl)
    x = []
    y = []
    chunks = self.Chunkify(t, flux, 256)
    for chunk in chunks:
      chunk_x = {'t': chunk['t']}
      x.append(chunk_x)
      chunk_y = {'flux': chunk['flux'], 'period': self.manifest['koi_period'][m], 'name': self.manifest['kepler_name'][m]}
      y.append(chunk_y)
    return x, y

  def Examples(self, m):
    # Get the mth index from the current index
    m = self.idx[m]
    x, y = self.LoadExample(m)
    return x, y, self.manifest['kepfits'][m]

  def TrainingExample(self, m):
    # Note that x and y are arrays of chunks
    x, y, id = self.Examples(m)
    example = []
    n = len(x)
    print(f'{n} TrainingExamples for {id}')
    for idx in range(n):
      #print(f'Serializing example {m} chunk {idx}')
      feature = { 
          'x': _floatvector_feature(x[idx]['t']), 
          'y': _floatvector_feature(y[idx]['flux']), 
          'p': _float_feature(y[idx]['period']),
          'id': _bytes_feature(id.encode('utf-8')), 
          'name': _bytes_feature(y[idx]['name'].encode('utf-8'))
      }
      #print(feature)
      # Create an example protocol buffer
      example.append(Example(features=Features(feature=feature)))
    return example, n

  def SaveDataset(self, dataset, mode):
    cursor = mode
    # Make sure we have initialized a writer
    self.Writer()
    manilen = len(self.idx)
    print(f'Saving {manilen} manifest entries')
    for m in range(manilen):
      # Take an example if selected for the current mode
      #if mode == TrainingSet.TEST or self.mask[dataset][m] == (mode == TrainingSet.TRAIN):
      example, examples = self.TrainingExample(m) 
      #print(f'Saving training example {m} with {examples} chunks')
      # write one (usually) or a set of correlated examples to the TFRecord file via the writer object
      for exidx in range(examples):
        #print(e)
        # Check for reaching shard partition size and if so, close the shard and start a new one
        # for multiple examples should we really do this?
        if self.counter[cursor] % self.shardSize == 0:
          if self.counter[cursor] > 0:
            self.writer[cursor].close()
            self.writer[cursor] = tf.io.TFRecordWriter(self.OutputPath(cursor, self.counter[cursor]))
        #print(f'Saving training example chunk {exidx}')
        #print(example[exidx])
        self.writer[cursor].write(example[exidx].SerializeToString())
        self.counter[cursor] += 1

  def CloseDatasets(self):
    if self.counter[TrainingSet.TRAIN] > 0:
      self.writer[TrainingSet.TRAIN].close()
    if self.counter[TrainingSet.VALID] > 0:
      self.writer[TrainingSet.VALID].close()
    if self.counter[TrainingSet.TEST] > 0:
      self.writer[TrainingSet.TEST].close()



