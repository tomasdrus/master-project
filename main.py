import yaml, os, itertools
from munch import DefaultMunch

config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml"))['main'])

def get_all_combinations(*arrays):
    all_combinations = []
    for combination in itertools.product(*arrays):
        all_combinations.append(combination)
    return all_combinations

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

if(config.dataset):
    if(config.dataset_search):
        for item in get_all_combinations(sr):
            os.system(f'python src/dataset.py --sr {item[0]} --fmax {int(item[0]/2)}')
            os.system('python src/siamese.py')
    else:
        os.system('python src/dataset.py')


if(config.siamese and not config.dataset_search):
    if(config.siamese_search):
        for item in get_all_combinations(lr, batch):
            os.system(f'python src/siamese.py --lr {item[0]} --batch {item[1]}')
    else:
        os.system('python src/siamese.py')

if(config.classifier):
    os.system('python src/classifier.py')
    