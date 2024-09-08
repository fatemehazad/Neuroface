#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[ ]:


import os
import shutil
import random
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split


# # Split Data

# In[ ]:


dataset_path = 'NeuroFaceDataSet'

train_dir = 'dataset_split/train'
validation_dir = 'dataset_split/validation'
test_dir = 'dataset_split/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# In[ ]:


def split_folders(folders, train_size=0.7, validation_size=0.15, test_size=0.15):
    train, validation_test = train_test_split(folders, test_size=(validation_size + test_size), random_state=42)
    validation, test = train_test_split(validation_test, test_size=test_size / (validation_size + test_size), random_state=42)
    return train, validation, test


# In[ ]:


def move_folders(folders, source_dir, destination_dir):
    for folder in folders:
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(destination_dir, folder))


# In[ ]:


for gender_dir in [os.path.join(dataset_path, 'men'),os.path.join(dataset_path, 'women')]:
    people_dirs = [d for d in os.listdir(gender_dir) if os.path.isdir(os.path.join(gender_dir, d))]
    
    train_dirs, validation_dirs, test_dirs = split_folders(people_dirs)
    
    move_folders(train_dirs, gender_dir, os.path.join(train_dir, os.path.basename(gender_dir)))
    move_folders(validation_dirs, gender_dir, os.path.join(validation_dir, os.path.basename(gender_dir)))
    move_folders(test_dirs, gender_dir, os.path.join(test_dir, os.path.basename(gender_dir)))
print('Split Data :  done!')


# # Build Pairs

# In[ ]:


pairs_dir = 'pairs'

os.makedirs(pairs_dir, exist_ok=True)


# In[ ]:


def load_image_paths(base_dir):
    image_paths = {'men': {}, 'women': {}}
    valid_extensions = ('.jpg', '.jpeg', '.png') 
    for gender in ['men', 'women']:
        gender_dir = os.path.join(base_dir, gender)
        people_dirs = [os.path.join(gender_dir, d) for d in os.listdir(gender_dir) if os.path.isdir(os.path.join(gender_dir, d))]
        for person_dir in people_dirs:
            person_name = os.path.basename(person_dir)
            image_paths[gender][person_name] = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.lower().endswith(valid_extensions)]
    return image_paths


# In[ ]:


def add_similar_pairs(image_paths, pairs, labels):
    for imgs in image_paths.values():
        if len(imgs) > 1:
            similar_pairs = [(img1, img2) for i, img1 in enumerate(imgs) for img2 in imgs[i + 1:]]
            pairs.extend(similar_pairs)
            labels.extend([1] * len(similar_pairs))


# In[ ]:


def add_dissimilar_pairs(image_paths, pairs, labels, num_similar_pairs):
    all_people = list(image_paths.keys())
    random.seed(42)
    while len(labels) < num_similar_pairs * 2:
        person1 = random.choice(all_people)
        person2 = random.choice(all_people)
        if person1 != person2:
            img1 = random.choice(image_paths[person1])
            img2 = random.choice(image_paths[person2])
            pairs.append((img1, img2))
            labels.append(0)


# In[ ]:


def generate_pairs(image_paths, condition):
    pairs = []
    labels = []
    if condition == 'women':
        add_similar_pairs(image_paths['women'], pairs, labels)
        add_dissimilar_pairs(image_paths['women'], pairs, labels, len(labels))
    elif condition == 'men':
        add_similar_pairs(image_paths['men'], pairs, labels)
        add_dissimilar_pairs(image_paths['men'], pairs, labels, len(labels))
    elif condition == 'mixed':
        all_people = {**image_paths['men'], **image_paths['women']}
        add_similar_pairs(all_people, pairs, labels)
        add_dissimilar_pairs(all_people, pairs, labels, len(labels))
    return np.array(pairs), np.array(labels)


# In[ ]:


def save_pairs(base_dir, file_name, condition):
    image_paths = load_image_paths(base_dir)
    pairs, labels = generate_pairs(image_paths, condition)
    with open(os.path.join(pairs_dir, file_name), "wb") as f:
        pkl.dump((pairs, labels), f)


# In[ ]:


# train pairs
save_pairs(train_dir, 'train_pairs_mixed.pkl', 'mixed')
# validation pairs
save_pairs(validation_dir, 'validation_pairs_mixed.pkl', 'mixed')
# test pairs
for condition in ['women', 'men', 'mixed']:
    save_pairs(test_dir, f'test_pairs_{condition}.pkl', condition)
print('Build Pairs :  done!')

