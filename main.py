import yaml, os, itertools, pandas as pd
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

""" data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
key='fmax'
data = data.loc[data['param'] == key] #rows with tested param
print(data)
print(data['ACC'].idxmax())
data = data.loc[data['ACC'].idxmax()] #row with highest acc
print(data[key])
update_config("config.yml", ['dataset',key], int(data[key])) #change best value in yaml 
exit() """


# dataset
mic = [1, 2]
length = [1.5,2,2.5,3,3.5,4.5,5,5.5,6]
n_mfcc = [15,20,25,30,40,45,50]
n_mels = [64,128,256,512]
n_fft = [512,1024,2048,5096]
min_duration = [2.7, 3, 3.3]
fmin = [0, 100, 200, 300, 400, 500, 600, 700]
fmax = [5000, 6000, 7000, 8000, 9000, 10000]
sr = [22050, 48000, 22050, 48000]

# siamese
lr = [0.01,0.001,0.0001]
batch = [64,128,256,512]

tests = {
    'fmin': [0, 100, 200],
    'fmax': [5000, 6000, 7000],
    }

if(config.dataset):
    if(config.dataset_search):
        """ for item in get_all_combinations(sr):
            os.system(f'python3 src/dataset.py --sr {item[0]} --fmax {int(item[0]/2)}')
            os.system('python3 src/siamese.py') """
        for key, values in tests.items():
            for value in values:
                os.system(f'python3 src/dataset.py --{key} {value}')
                os.system(f'python3 src/siamese.py --param {key}')
                os.system('python3 src/classifier.py')

            data = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
            data = data.loc[data['param'] == key] #rows with tested param
            data = data.loc[data['ACC'].idxmax()] #row with highest acc
            best_value = int(data[key])
            print(key, best_value)
            logging.info(f'{key} => {best_value}' )
            update_config("config.yml", ['dataset',key], best_value) #change best value in yaml 

    else:
        os.system('python3 src/dataset.py')


if(config.siamese and not config.dataset_search):
    if(config.siamese_search):
        for item in get_all_combinations(lr, batch):
            os.system(f'python3 src/siamese.py --lr {item[0]} --batch {item[1]}')
    else:
        os.system('python3 src/siamese.py')

if(config.classifier and not config.dataset_search):
    os.system('python3 src/classifier.py')
    