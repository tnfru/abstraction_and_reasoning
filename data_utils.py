import os, json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import colors
from tensorflow import expand_dims


# Read in data

def read_tasks(path):
  files = sorted(os.listdir(path))
  tasks = []

  for file in files:
    task_file = str(path / file)

    with open(task_file, 'r') as f:
      task = json.load(f)

    tasks.append(task)
   
  return tasks

def get_task_sets():
    PATH = Path('./data/')
    TRAIN_PATH = PATH / 'training'
    VAL_PATH = PATH / 'evaluation'
    TEST_PATH = PATH / 'test'

    train_tasks = read_tasks(TRAIN_PATH)
    val_tasks = read_tasks(VAL_PATH)
    test_tasks = read_tasks(TEST_PATH)

    return train_tasks, val_tasks, test_tasks

# Data Vizualization
# from kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks

def plot_pictures(pictures, labels):
  cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
  norm = colors.Normalize(vmin=0, vmax=9)

  fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
  for i, (pict, label) in enumerate(zip(pictures, labels)):
    axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
    axs[i].set_title(label)
  plt.show()
    
def plot_sample(sample, predict=None):
  if predict is None:
    plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
  else:
    plot_pictures([sample['input'], sample['output'], predict], ['Input', 'Output', 'Predict'])
       
def plot_task(task):
  for sample in task['train']:
    plot_sample(sample)
  for sample in task['test']:
    plot_sample(sample)

def plot_prediction(predictions, task):
  for i in range(len(task)):
      plot_sample(task[i], predictions[i])

# Data transformation

def input_dim_equals_output_dim(task, isTrain=True):
  for example in task['train']:
    if np.array(example['input']).shape != np.array(example['output']).shape:
      return False
  if isTrain:
    for example in task['test']:
      if np.array(example['input']).shape != np.array(example['output']).shape:
        return False  
  
  return True

def count_equal_dims(tasks, isTrain=True):
  equal_dims = 0
  for task in tasks:
    equal_dims += input_dim_equals_output_dim(task, isTrain)
  return equal_dims


# from kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks

def inp2img(inp):
    inp = np.array(inp)
    img = np.full((inp.shape[0], inp.shape[1], 10), 0, dtype=np.float32)
    for i in range(10):
        img[:,:,i] = (inp==i)
    return expand_dims(img, 0)

def calk_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]
