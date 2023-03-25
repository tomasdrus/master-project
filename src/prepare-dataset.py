#basic imports
import os, glob, re, yaml, random

#data processing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from alive_progress import alive_bar, alive_it, config_handler
config_handler.set_global(theme='classic')
from tqdm import tqdm

CONFIG = yaml.safe_load(open("config.yml"))['prepare']

def extract_features(file_path, length=3, overlap=0.5, max_count=1, min_duration=2.7, mode='spectogram'):
    # length - in seconds to create audio cuts
    # overlap - between audio cuts in percent, default no overlap
    # max_count - max number of blocks per audio
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y)
    duration = librosa.get_duration(y=y, sr=sr)

    if(duration <= min_duration):
        return [], None

    buffer = length * sr
    samples_total = len(y)
    samples_wrote = 0
    count = 0

    features = []
    while (samples_wrote < samples_total and count < max_count):
        #check if the buffer is not exceeding total samples
        if buffer > samples_total - samples_wrote:
            buffer = samples_total - samples_wrote

        block = y[samples_wrote: (samples_wrote + buffer)]

        # short blocks pad with zeros to fit size
        block = librosa.util.fix_length(block, size=(length * sr))

        if mode == 'spectogram':
            feature = librosa.feature.melspectrogram(y=block, sr = sr, n_mels=128, n_fft = 1024, fmax = None)
        else:
            feature = librosa.feature.mfcc(y=block, sr=sr, n_mfcc=20) 

        features.append(feature)
        
        if buffer >= samples_total - samples_wrote:
            samples_wrote += buffer
        else:
            samples_wrote += int(buffer * (1 - overlap))
        count += 1
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
def create_data(dict, max_speakers, max_recordings, mode=2):
    data, labels, duration = [], [], []

    speakers = list(dict.keys())[:max_speakers]
    for id, speaker_id in enumerate(speakers):
        recordings = dict[speaker_id][:max_recordings]

        for recording in alive_it(recordings, title=f'P{speaker_id}'):
            #print(speaker_id, recording)

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

        ap_ssim = ssim(anch, pos, data_range=pos.max() - pos.min())
        an_ssim = ssim(anch, neg, data_range=neg.max() - neg.min())
        diff = ap_ssim - an_ssim

        #if(diff < 0.2 and diff > 0.05):
        #if(diff > 0):
            #print('AP: ',round(ap_ssim,3),'AN: ',round(an_ssim,3), 'DIFF: ', diff)
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
    with alive_bar(n) as bar:
        while(len(triplets) < n):
            anchor, positive, negative = get_triplet(X, y)
            if(shape == None):
                shape = anchor.shape

            triplet = (anchor.tobytes(), positive.tobytes(), negative.tobytes())
            triplets.add(triplet)
            bar()

    anchors = [np.frombuffer(triplet[0], dtype='float32').reshape(shape) for triplet in triplets]
    positives = [np.frombuffer(triplet[1], dtype='float32').reshape(shape) for triplet in triplets]
    negatives = [np.frombuffer(triplet[2], dtype='float32').reshape(shape) for triplet in triplets]

    labels = np.ones(len(triplets))
    
    return [np.array(anchors), np.array(positives), np.array(negatives)], labels

# create dictionary from directory
speakers_dict = create_speakers_dict(CONFIG['directory'], mic=CONFIG['mic'])
# create dataset with extracted features and labels
X, y, d = create_data(speakers_dict, 10, 50, CONFIG['features'])
print(f'\nLength: {len(d)}, Mean: {np.mean(d)}, Median: {np.median(d)}, Max: {np.max(d)}, Min: {np.min(d)}')
print(f'unde 1s: {len(np.where(d<=1)[0])}, 1s - 2s: {len(np.where((d >= 1) & (d <= 2))[0])}')
print(f'2s - 3s: {len(np.where((d >= 2) & (d <= 3))[0])}, 3s - 4s: {len(np.where((d >= 3) & (d <= 4))[0])}')
print(f'4s - 5s: {len(np.where((d >= 4) & (d <= 5))[0])}, over 5s: {len(np.where(d>=5)[0])}\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X.shape)

X_trip, y_trip = generate_triplets_unique(X_train, y_train, 5000)
print(X_trip[0].shape)

np.savez('data/data.npz', X=X, y=y, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
np.savez('data/triplets.npz', X_trip=X_trip, y_trip=y_trip)