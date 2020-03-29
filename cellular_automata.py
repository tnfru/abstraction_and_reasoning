import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Dropout
from tensorflow.keras.activations import softmax
from data_utils import inp2img, plot_prediction
import numpy as np
from matplotlib import pyplot as plt


def get_default_model():
  inputs = Input(shape=(None, None, 10))
  x = Conv2D(128, 3, padding='same')(inputs)
  x = Activation('relu')(x)
  x = Conv2D(10, 1)(x)
  outputs = softmax(x, axis=3)

  return tf.keras.Model(inputs=inputs, outputs=outputs)

def solve_task(task, model_fn, max_steps=10, epochs=100):
  epochs = 100
  model = model_fn()
  losses = []
  
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  for num_steps in range(1, max_steps):
    optimizer = tf.keras.optimizers.Adam(learning_rate=(np.abs(0.1/np.exp(num_steps))))

    for epoch in range(epochs):
      loss = 0.0

      for sample in task:
        with tf.GradientTape() as tape:
          # predict output from input

          x = inp2img(sample['input'])
          y = tf.convert_to_tensor(sample['output'], dtype=tf.int8)

          y_pred = model(softmax(x, axis=3))
          
          for _ in range(num_steps):
            y_pred = model(y_pred)

          loss += loss_fn(y, y_pred)

          # predict output from output to force stabilization

          y_pred = model(inp2img(sample['output']))

          loss += 5 * loss_fn(y, y_pred)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
      
      losses.append(loss)
  
  return model, np.array(losses)


def predict(model, task, num_steps=100):
  predictions = []

  for sample in task:
    x = inp2img(sample['input'])
    pred = model(softmax(x, axis=3))
        
    for _ in range(num_steps):
      pred = model(pred)
        
    predictions.append(tf.argmax(pred, axis=3).numpy().squeeze())
  return predictions

def train_and_plot(task_set, task_ids, model_fn=get_default_model, max_steps=2):
  for i in task_ids:
    model, losses = solve_task(task_set[i]['train'], model_fn, max_steps=max_steps)
    plt.plot(losses)
    predictions = predict(model, task_set[i]['train'])
    plot_prediction(predictions, task_set[i]['train'])
    predictions = predict(model, task_set[i]['test'])
    plot_prediction(predictions, task_set[i]['test'])
