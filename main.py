import yaml, os, itertools, pandas as pd
import numpy as np
from munch import DefaultMunch

import logging
logging.basicConfig(filename='changes.log', encoding='utf-8', level=logging.DEBUG)

config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml")))['main']

def get_all_combinations(*arrays):
    all_combinations = []
    for combination in itertools.product(*arrays):
        all_combinations.append(combination)
    return all_combinations

def update_config(file_path, keys, value):
    def update_nested_key(data, keys, value):
        if len(keys) == 1:
            if keys[0] in data:
                data[keys[0]] = value
        else:
            key = keys.pop(0)
            if key in data:
                update_nested_key(data[key], keys, value)

    with open(file_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.Loader)

    update_nested_key(yaml_data, keys, value)

    with open(file_path, 'w') as f:
        yaml.dump(yaml_data, f, Dumper=yaml.Dumper, default_flow_style=False, sort_keys=False)

dataset = {
    'n_triplets': [10000, 20000, 30000, 40000],
    #'mode': ['mfcc', 'melbank'],
    #'mic': [1,2],
    'length': [3, 3.5, 4.0, 4,5, 5.0],
    }

if(config.dataset):
    if(config.dataset_search):
        for key, values in dataset.items():
            for value in values:
                if(key == 'length'):
                    os.system(f'python3 src/dataset.py --length {value} --min_duration {value}')
                else:
                    os.system(f'python3 src/dataset.py --{key} {value}')

                os.system(f'python3 src/siamese.py --param {key}')
                os.system('python3 src/classifier.py')

            data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
            data = data.loc[data['param'] == key] #rows with tested param
            data = data.loc[data['ACC'].idxmax()] #row with highest acc
            best_value = data[key]
            if(type(best_value).__module__ == np.__name__):
                best_value = best_value.item()
            logging.info(f'{key} => {best_value}' )
            update_config("config.yml", ['dataset',key], best_value) #change best value in yaml
            if(key == 'length'):
                update_config("config.yml", ['dataset','min_duration'], best_value) #change best value in yaml
    else:
        os.system('python3 src/dataset.py')

# siamese
""" siamese = {
    'v_cnn': [1, 2, 3],
    'activation': ['ReLU', 'LeakyReLU', 'PReLU'],
    'padding': ['same', 'valid'],
    'embeding': [128, 256, 512, 1024],
    'kernel': [(1,3), (3,1), (3,3), (3,5), (5,3), (5,5), (5,7), (7,7)],
    'strides': [(1,1), (2,2), (2,1), (1,2)],
    }

optimizer = ['SGD', 'Adam', 'AdamW', 'Nadam']
lr = [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]

margin = [0.1, 0.5, 1.0, 1.5, 2.0] """

siamese = {
    'v_cnn': [1, 2, 3],
    'embeding': [128, 256, 512, 1024, 2048],
    'activation': ['ReLU', 'LeakyReLU', 'PReLU'],
    'padding': ['same', 'valid'],
    'kernel': ['[1,3]','[3,1]','[3,3]', '[3,5]', '[5,3]','[5,5]','[5,7]','[7,7]'],
    'strides': ['[1,1]', '[2,2]', '[2,1]', '[1,2]'],
    }

optimizer = ['SGD', 'Adam', 'Nadam']
lr = [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]

margin = [0.1, 0.5, 1.0, 1.5, 2.0]

if(config.siamese):
    if(config.siamese_search):
        for key, values in siamese.items():
            for value in values:
                os.system(f'python3 src/siamese.py --param {key} --{key} {value}')
                os.system('python3 src/classifier.py')

            data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
            data = data.loc[data['param'] == key] #rows with tested param
            data = data.loc[data['ACC'].idxmax()] #row with highest acc
            best_value = data[key]
            if(type(best_value).__module__ == np.__name__):
                best_value = best_value.item()

            logging.info(f'{key} => {best_value}' )
            update_config("config.yml", ['siamese',key], best_value) #change best value in yaml 

        for item in get_all_combinations(optimizer, lr):
            param = 'optimizer+lr'
            os.system(f'python3 src/siamese.py --param {param} --optimizer {item[0]} --lr {item[1]}')
            os.system('python3 src/classifier.py')
        if(get_all_combinations(optimizer, lr)):
            data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
            data = data.loc[data['param'] == param] #rows with tested param
            data = data.loc[data['ACC'].idxmax()] #row with highest acc
            best_optimizer = str(data['optimizer'])
            best_lr = int(data['lr'])
            logging.info(f'{param} => optimizer{best_optimizer}, lr{best_lr}' )
            update_config("config.yml", ['siamese','optimizer'], best_optimizer) #change best value in yaml 
            update_config("config.yml", ['siamese','learning_rate'], best_lr) #change best value in yaml 

        for value in margin:
            os.system(f'python3 src/siamese.py --param margin --margin {value}')
            os.system('python3 src/classifier.py')
        if(margin):
            key = 'margin'
            data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
            data = data.loc[data['param'] == key] #rows with tested param
            data = data.loc[data['ACC'].idxmax()] #row with highest acc
            best_value = float(data[key])

            logging.info(f'{key} => {best_value}' )
            update_config("config.yml", ['siamese',key], best_value)
    else:
        os.system('python3 src/siamese.py')


# classifier
classifier = {
    'n_pairs': [10000, 20000, 30000],
    'vectors_join': ['concatenate', 'euclidian', 'difference'],
    }

if(config.classifier):
    if(config.classifier_search):
        for key, values in classifier.items():
            for value in values:
                os.system(f'python3 src/classifier.py --param {key} --{key} {value}')

            data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
            data = data.loc[data['param'] == key] #rows with tested param
            data = data.loc[data['ACC'].idxmax()] #row with highest acc
            best_value = data[key]
            if(type(best_value).__module__ == np.__name__):
                best_value = best_value.item()

            logging.info(f'{key} => {best_value}' )
            update_config("config.yml", ['classifier',key], best_value) #change best value in yaml 
    else:
        os.system('python3 src/classifier.py')
    