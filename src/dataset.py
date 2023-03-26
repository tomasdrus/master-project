#basic imports
import os, glob, re, yaml, random, argparse
from munch import DefaultMunch

from helpers import * # custom helpers functions

#data processing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from alive_progress import alive_bar, alive_it, config_handler
config_handler.set_global(theme='classic')

config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml"))['dataset'])

# arguments (for grid search)
parser = argparse.ArgumentParser()
parser.add_argument('--length', dest='length', type=float)
parser.add_argument('--mic', dest='mic', type=int)
parser.add_argument('--min_duration', dest='min_duration', type=float)
parser.add_argument('--n_triplets', dest='n_triplets', type=int)
parser.add_argument('--sr', dest='sr', type=int)
parser.add_argument('--mode', dest='mode')
parser.add_argument('--n_mfcc', dest='n_mfcc', type=int)
parser.add_argument('--n_mels', dest='n_mels', type=int)
parser.add_argument('--n_fft', dest='n_fft', type=int)
parser.add_argument('--fmin', dest='fmin', type=int)
parser.add_argument('--fmax', dest='fmax', type=int)
parser.add_argument('--power', dest='power', type=int)

args = parser.parse_args()

def args_conf(name):
    if(hasattr(args, name) and getattr(args, name) is not None):
        return getattr(args, name)
    return getattr(config, name)

# extract features
def extract_features(file_path, length=args_conf('length'), overlap=args_conf('overlap'), max_count=args_conf('max_count'), min_duration=args_conf('min_duration'), mode='spectogram'):
    # length - in seconds to create audio cuts
    # overlap - between audio cuts in percent, default no overlap
    # max_count - max number of blocks per audio
    y, sr = librosa.load(file_path, sr=args_conf('sr'), mono=True)
    y, _ = librosa.effects.trim(y, top_db=args_conf('top_db'), ref=np.max)
    duration = librosa.get_duration(y=y, sr=sr)

    if(duration <= min_duration):
        return [], None

    buffer = size = int(length * sr)
    samples_total = len(y)
    samples_wrote = 0
    count = 0

    features = []
    while (samples_wrote < samples_total and count < max_count):
        #check if the buffer is not exceeding total samples
        if buffer > samples_total - samples_wrote:
            buffer = samples_total - samples_wrote

        block = y[samples_wrote: (samples_wrote + buffer)]
        block_duration = librosa.get_duration(y=block, sr=sr)
        #print(f'full duration {round(duration, 2)}, block {count}, duration {round(block_duration, 2)}')

        if(block_duration < min_duration):
            break

        # short blocks pad with zeros to fit size
        block = librosa.util.fix_length(block, size=(size))

        if args_conf('mode') == 'spectogram':
            feature = librosa.feature.melspectrogram(y=block, sr=sr, n_mels=args_conf('n_mels'), n_fft=args_conf('n_fft'), fmin=args_conf('fmin'), fmax=args_conf('fmax'), power=args_conf('power'))
        else:
            feature = librosa.feature.mfcc(y=block, sr=sr, n_mfcc=args_conf('n_mfcc'), n_mels=args_conf('n_mels'), n_fft=args_conf('n_fft'), fmin=args_conf('fmin'), fmax=args_conf('fmax'), power=args_conf('power')) 

        features.append(feature)
        
        if buffer >= samples_total - samples_wrote:
            samples_wrote += buffer
        else:
            samples_wrote += int(buffer * (1 - overlap))
        count += 1
    #print(f'+++ {len(features)} +++' if len(features) > 0 else '--------')
    return features, duration

# create speakers dict
def create_speakers_dict(directory, mic=1):
    speaker_dict = {}
   
    for speaker_dir in sorted(glob.glob(os.path.join(directory, "*/"))):
        speaker_id = int(re.search(r'\d+',os.path.basename(os.path.dirname(speaker_dir))).group())
        speaker_audio = glob.glob(os.path.join(speaker_dir, f"*mic{mic}.flac"))

        speaker_dict[speaker_id] = []
        for audio_path in speaker_audio:
            speaker_dict[speaker_id].append(audio_path)
                
    return speaker_dict

# create data
def create_data(dict, max_speakers, max_recordings, mode='mfcc'):
    data, labels, duration = [], [], []

    speakers = list(dict.keys())[:max_speakers]
    speakers_len = len(speakers)
    print(f'\nProcessing recordings for {speakers_len} speakers ')
    for id, speaker_id in enumerate(speakers):
        recordings = dict[speaker_id][:max_recordings]

        for recording in alive_it(recordings, title=f'Speaker {speaker_id} {id}/{speakers_len}'):
            # multiple features per recording
            features, d = extract_features(recording, mode=mode)
            for feature in features:
                duration.append(d)
                data.append(feature)
                labels.append(id)

    return [np.array(data), np.array(labels), np.array(duration)]

def get_feature(X, y, label):
    idx = np.random.randint(len(y))
    while y[idx] != label:
        idx = np.random.randint(len(y))
    return X[idx]

""" def get_triplet(X, y):
    keys = list(set(y))
    key = random.choice(keys)
    indicies = list(np.where(y == key)[0])

    anch_i, pos_i = random.sample(indicies, 2)

    neg_key = random.choice([id for id in keys if id != key])
    neg_i = random.choice(list(np.where(y == neg_key)[0]))
    anch, pos, neg = X[anch_i], X[pos_i], X[neg_i]

    ap_ssim = ssim(anch, pos, data_range=pos.max() - pos.min())
    an_ssim = ssim(anch, neg, data_range=neg.max() - neg.min())
    print('AP: ',round(ap_ssim,3),'AN: ',round(an_ssim,3), 'DIFF: ', ap_ssim - an_ssim)

    return anch, pos, neg """


def get_triplet(X, y):
    while True:
        keys = list(set(y))
        key = random.choice(keys)
        indicies = list(np.where(y == key)[0])

        anch_i, pos_i = random.sample(indicies, 2)

        neg_key = random.choice([id for id in keys if id != key])
        neg_i = random.choice(list(np.where(y == neg_key)[0]))
        anch, pos, neg = X[anch_i], X[pos_i], X[neg_i]
        return anch, pos, neg


def generate_triplets(X, y, n):
    anchors, positives, negatives = [], [], []
    labels = np.ones(n)
    for i in range(n):
        anchor, positive, negative = get_triplet(X, y)
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    
    return [np.array(anchors), np.array(positives), np.array(negatives)], labels

def generate_triplets_unique(X, y, n):
    triplets = set()
    shape = None
    print(f'\nGenerating {n} unique triplets')
    with alive_bar(n) as bar:
        while(len(triplets) < n):
            anchor, positive, negative = get_triplet(X, y)
            if(shape == None):
                shape = anchor.shape

            triplet = (anchor.tobytes(), positive.tobytes(), negative.tobytes())
            if(set_add(triplets, triplet)):
                bar()

    anchors = [np.frombuffer(triplet[0], dtype='float32').reshape(shape) for triplet in triplets]
    positives = [np.frombuffer(triplet[1], dtype='float32').reshape(shape) for triplet in triplets]
    negatives = [np.frombuffer(triplet[2], dtype='float32').reshape(shape) for triplet in triplets]

    labels = np.ones(len(triplets))
    
    return [np.array(anchors), np.array(positives), np.array(negatives)], labels

def print_audio_lengths(d):
    print(f'\nLength: {len(d)}, Mean: {np.mean(d)}, Median: {np.median(d)}, Max: {np.max(d)}, Min: {np.min(d)}')
    print(f'unde 1s: {percent(len(np.where(d<=1)[0]), len(d))}, 1s - 2s: {percent(len(np.where((d >= 1) & (d <= 2))[0]), len(d))}')
    print(f'2s - 3s: {percent(len(np.where((d >= 2) & (d <= 3))[0]), len(d))}, 3s - 4s: {percent(len(np.where((d >= 3) & (d <= 4))[0]), len(d))}')
    print(f'4s - 5s: {percent(len(np.where((d >= 4) & (d <= 5))[0]), len(d))}, over 5s: {percent(len(np.where(d>=5)[0]), len(d))}\n')

# create dictionary from directory
speakers_dict = create_speakers_dict(config.directory, mic=args_conf('mic'))

# create dataset with extracted features and labels
X, y, d = create_data(speakers_dict, config.n_speakers, config.n_recordings, config.mode)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args_conf('test_size'), random_state=42)
X_trip, y_trip = generate_triplets_unique(X_train, y_train, args_conf('n_triplets'))

print(f'\nData shape: {X.shape}, Train Test split: {y_train.shape[0]} / {y_test.shape[0]}, Triplets shape: {X_trip[0].shape}\n')
#print_audio_lengths(d)

np.savez('data/data.npz', X=X, y=y, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
np.savez('data/triplets.npz', X_trip=X_trip, y_trip=y_trip)

np.save('data/settings.npy', {
    'n_triplets':args_conf('n_triplets'),
    'mode':args_conf('mode'),
    'mic':args_conf('mic'),
    'sr':args_conf('sr'),
    'length':args_conf('length'),
    'min_duration':args_conf('min_duration'),
    'n_mfcc':args_conf('n_mfcc'),
    'n_mels':args_conf('n_mels'),
    'n_fft':args_conf('n_fft'),
    'fmin':args_conf('fmin'),
    'fmax':args_conf('fmax'),
    'power':args_conf('power'),
    }) 