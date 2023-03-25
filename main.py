




if __name__ == '__main__':
    exec(open("src/prepare-dataset.py").read())
    exec(open("src/train-dataset.py").read())
    exec(open("src/train-mlp.py").read())