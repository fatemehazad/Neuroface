#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[ ]:


import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# # Preprocess (Load - Resize - Scale)

# In[ ]:


def load_pairs(file_path):
    with open(file_path, "rb") as f:
        pairs, labels = pkl.load(f)
    return pairs, labels


# In[ ]:


# file_path = 'pairs/train_pairs_mixed.pkl'
# pairs , labels = load_pairs(file_path)
# print(f'pairs shape : {pairs.shape} , labels shape : {labels.shape} ')
# first_pair = pairs[0]
# first_label = labels[0]
# print(f'first pair : {first_pair}')
# print(f'first label : {first_label}')


# In[ ]:


def resize_scale(img_path):
    byte_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img


# In[ ]:


# img_path = first_pair[0]
# img = resize_scale(img_path)
# print(img)
# plt.imshow(img)


# In[ ]:


def preprocess_data(file_path):
    pairs, labels = load_pairs(file_path)
    imgs_left = []
    imgs_right = []
    for i in range(len(pairs)):
        imgs_left.append(resize_scale(pairs[i][0]))
        imgs_right.append(resize_scale(pairs[i][1]))
    imgs_left = tf.convert_to_tensor(imgs_left)
    imgs_right = tf.convert_to_tensor(imgs_right)
    labels = tf.convert_to_tensor(labels)
    data = tf.data.Dataset.from_tensor_slices((imgs_left, imgs_right, labels))
    data = data.cache().shuffle(buffer_size=10000, seed=42)
    return data


# In[ ]:


train_data = preprocess_data('pairs/train_pairs_mixed.pkl')


# In[ ]:


# print(f' train_data length : {len(train_data)}')
# samples = train_data.as_numpy_iterator()
# example = samples.next()
# print(f'first data : {example}')
# plt.imshow(example[0])


# In[ ]:


validation_data = preprocess_data('pairs/validation_pairs_mixed.pkl')


# In[ ]:


test_data = preprocess_data('pairs/test_pairs_mixed.pkl')
test_data_men = preprocess_data('pairs/test_pairs_men.pkl')
test_data_women = preprocess_data('pairs/test_pairs_women.pkl')


# # Build Model

# In[ ]:


def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=inp, outputs=d1, name='embedding')


# In[ ]:


embedding = make_embedding()
# embedding.summary()


# In[ ]:


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, embedding_left, embedding_right):
        return tf.math.abs(embedding_left - embedding_right)


# In[ ]:


def make_siamese_model(): 
    
    image_left = Input(name='image_left', shape=(100,100,3))
    image_right = Input(name='image_right', shape=(100,100,3))

    embedding_left = embedding(image_left)
    embedding_right = embedding(image_right)

    dist_layer = L1Dist(name='distance')
    distances = dist_layer(embedding_left, embedding_right)
    
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[image_left, image_right], outputs=classifier, name='siamese_model')


# In[ ]:


siamese_model = make_siamese_model()
# siamese_model.summary()


# # Train and Validate

# In[ ]:


train_data = train_data.batch(16).prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.batch(16).prefetch(tf.data.AUTOTUNE)


# In[ ]:


binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


# In[ ]:


checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# In[ ]:


with open('training_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)      
    writer.writerow(['Epoch', 'Train Recall', 'Train Precision', 'Train Accuracy', 'Validation Recall', 'Validation Precision', 'Validation Accuracy'])


# In[ ]:


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:     
        X = batch[:2]
        y = batch[2] 
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss


# In[ ]:


def validate(data):
    r = Recall()
    p = Precision()
    a = BinaryAccuracy()
    progbar = tf.keras.utils.Progbar(len(data), unit_name='batch')
    for idx, batch in enumerate(data):
        yhat = siamese_model.predict(batch[:2], verbose=0)
        r.update_state(batch[2], yhat)
        p.update_state(batch[2], yhat)
        a.update_state(batch[2], yhat)
        progbar.update(idx + 1)
    return r.result().numpy(), p.result().numpy(), a.result().numpy()


# In[ ]:


def train(train_data, validation_data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(train_data), unit_name='batch')
        r = Recall()
        p = Precision()
        a = BinaryAccuracy()
        for idx, batch in enumerate(train_data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2], verbose=0)
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            a.update_state(batch[2], yhat)
            progbar.update(idx + 1)
        train_recall = r.result().numpy()
        train_precision = p.result().numpy()
        train_accuracy = a.result().numpy()
        print(f'[Train Results] recall: {train_recall} precision: {train_precision} accuracy: {train_accuracy}')
        
        val_recall, val_precision, val_accuracy = validate(validation_data)
        print(f'[Validation Results] recall: {val_recall} precision: {val_precision} accuracy: {val_accuracy}')
        
        with open('training_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_recall, train_precision, train_accuracy, val_recall, val_precision, val_accuracy])
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# In[ ]:


train(train_data, validation_data, EPOCHS=80)


# In[ ]:


siamese_model.save('siamesemodel_v1.h5')


# # Test

# In[ ]:


test_data = test_data.batch(16).prefetch(tf.data.AUTOTUNE)
test_data_men = test_data_men.batch(16).prefetch(tf.data.AUTOTUNE)
test_data_women = test_data_women.batch(16).prefetch(tf.data.AUTOTUNE)


# In[ ]:


siamese_model = tf.keras.models.load_model('siamesemodel_v1.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[ ]:


with open('test_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)      
    writer.writerow(['Gender', 'Test Recall', 'Test Precision', 'Test Accuracy'])


# In[ ]:


def test(data, gender):
    r = Recall()
    p = Precision()
    a = BinaryAccuracy()
    for x_left, x_right, y_true in data.as_numpy_iterator():
        yhat = siamese_model.predict([x_left, x_right], verbose=0)
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
        a.update_state(y_true, yhat)
    test_recall = r.result().numpy()
    test_precision = p.result().numpy()
    test_accuracy = a.result().numpy()
    print(f'|Test Results - {gender}| recall: {test_recall} precision: {test_precision} accuracy: {test_accuracy}')
    with open('test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)      
        writer.writerow([gender ,test_recall, test_precision, test_accuracy])


# In[ ]:


test(test_data, "mixed")
test(test_data_men, "men")
test(test_data_women, "women")

