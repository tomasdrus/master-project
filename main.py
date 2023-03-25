import yaml, os
from munch import DefaultMunch

config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml"))['main'])

if(config.dataset):
    os.system('python src/dataset.py')

if(config.siamese):
    os.system(f'python src/siamese.py')

if(config.classifier):
    os.system('python src/classifier.py')
    