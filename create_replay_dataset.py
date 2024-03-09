import csv
import numpy as np 
import pandas as pd
import sys 
import os 
#replay_dataset = sys.argv[1]
#train_dataset = sys.argv[2]
#replay_dataset = "/gpfsstore/rech/nou/uzn19yk/speechbrain/recipes/LibriSpeech/CTC/results/titouan_checkpoint/1986/train-clean-360.csv"
#train_dataset = "/gpfsscratch/rech/nou/uzn19yk/download/cv-corpus-12.0-2022-12-07/apostrophed_en_prep/new_train.csv"
#out_set = "double_train.csv"
def create_double_dataset(replay_dataset, train_dataset, out_set) : 
    replay = pd.read_csv(replay_dataset)
    train = pd.read_csv(train_dataset)
    if replay.shape[0] < train.shape[0] : 
        raise Exception("Replay dataset should be bigger than training one")
    train = train.sort_values('duration')
    replay = replay.sort_values('duration')


    replay_start = np.random.randint(0, replay.shape[0] -train.shape[0]+1)
    replay_considered = replay[replay_start : replay_start + train.shape[0]]
    train["wav2"] = list(replay_considered["wav"])
    train["duration2"] = list(replay_considered["duration"])
    train.to_csv(out_set, index=False)

if __name__=="__main__" : 
    replay_dataset = sys.argv[1]
    train_dataset = sys.argv[2]
    out_set = sys.argv[3]
    create_double_dataset(replay_dataset, train_dataset, out_set)

