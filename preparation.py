# Copyright 2024 Boring Task AI
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from encodec.utils import convert_audio
import torch
import torchaudio
import numpy as np
import gc

dataset_path = 'datasets/id/'
hubert_path = 'data/models/hubert/hubert.pt'
hubert_tokenizer_path = 'data/models/hubert/tokenizer.pth'

max_duration_sec = 15.12 # the maximum allowed duration in seconds

path = dataset_path
device = "cuda"

SAMPLE_RATE = 24_000
CHANNELS = 1

# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
from hubert.hubert_manager import HuBERTManager
hubert_manager = HuBERTManager()

from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path=hubert_path).to(device)
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False

# Load the CustomTokenizer model
hubert_tokenizer = CustomTokenizer.load_from_checkpoint(hubert_tokenizer_path).to(device)  # Automatically uses the right layers

from bark.generation import load_codec_model
codec_model = load_codec_model(use_gpu=True)
codec_model.eval()

for param in codec_model.parameters():
    param.requires_grad = False


def get_duration(wav, sr):
    return wav.shape[1] / sr

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8', errors='ignore') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        base = os.path.dirname(filename)
        for j in range(len(filepaths_and_text)):
            filepaths_and_text[j][0] = os.path.join(base, "wav", filepaths_and_text[j][0])
    return filepaths_and_text

valid_lines_train = []
# convert wavs to semantic tokens
for wav_path, txt in load_filepaths_and_text(path + 'train.txt'):
    wav, sr = torchaudio.load(wav_path, format="mp3")
    if not get_duration(wav, sr) > max_duration_sec:
        valid_lines_train.append((wav_path, txt))
    wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS).to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=SAMPLE_RATE)
    semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)

    # save semantic tokens
    os.makedirs(os.path.join(path, 'tokens'), exist_ok=True)
    semantic_tokens = semantic_tokens.cpu().numpy()

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()

    # save tokens
    np.savez_compressed(os.path.join(path, 'tokens', os.path.basename(wav_path).replace('.mp3', '.npz')), fine=codes, coarse=codes[:2, :], semantic=semantic_tokens)

# rewrite train.txt with valid lines
with open(path + 'train_valid.txt', 'w', encoding='utf-8') as f:
    for wav_path, txt in valid_lines_train:
        wav_path = os.path.relpath(wav_path, dataset_path).replace('\\', '/')
        f.write(f'{wav_path}|{txt}\n')

valid_lines_valid = []
for wav_path, txt in load_filepaths_and_text(path + 'valid.txt'):
    wav, sr = torchaudio.load(wav_path, format="mp3")
    if not get_duration(wav, sr) > max_duration_sec:
        valid_lines_valid.append((wav_path, txt))
    wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS).to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=SAMPLE_RATE)
    semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)

    # save semantic tokens
    os.makedirs(os.path.join(path, 'tokens'), exist_ok=True)
    semantic_tokens = semantic_tokens.cpu().numpy()
    
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()

    # save tokens
    np.savez_compressed(os.path.join(path, 'tokens', os.path.basename(wav_path).replace('.mp3', '.npz')), fine=codes, coarse=codes[:2, :], semantic=semantic_tokens)

# rewrite valid.txt with valid lines
with open(path + 'test_valid.txt', 'w', encoding='utf-8') as f:
    for wav_path, txt in valid_lines_valid:
        wav_path = os.path.relpath(wav_path, dataset_path).replace('\\', '/')
        f.write(f'{wav_path}|{txt}\n')


del hubert_model
del hubert_tokenizer
del codec_model
gc.collect()
torch.cuda.empty_cache()