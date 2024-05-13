# モデルのインスタンス作成に必要なモジュールをインポート
from src.language_modeling_with_boxes.models import Word2Box, Word2Vec, Word2VecPooled, Word2BoxConjunction, Word2Gauss
from src.language_modeling_with_boxes.datasets.utils import get_iter_on_device
from src.language_modeling_with_boxes.__main__ import main

import torch
from torch import LongTensor, BoolTensor, Tensor, IntTensor
import pickle, json


# 保存してあるモデルと同じパラメータを設定 (すごい無理矢理ですが)
config = {
    "batch_size": 4096,
    "box_type": "BoxTensor",
    "data_device": "gpu",
    "dataset": "ptb",
    "embedding_dim": 64,
    "eos_mask": True,
    "eval_file": "../data/similarity_datasets/",
    "int_temp": 1.9678289474987882,
    "log_frequency": 10,
    "loss_fn": "max_margin",
    "lr": 0.004204091643267762,
    "margin": 5,
    "model_type": "Word2BoxConjunction",
    "n_gram": 5,
    "negative_samples": 2,
    "num_epochs": 10,
    "subsample_thresh": 0.001,
    "vol_temp": 0.33243242379830407,
    "save_model": "",
    "add_pad": "",
    "save_dir": "results",
}

# 語彙やデータローダーを作成
TEXT, train_iter, val_iter, test_iter, subsampling_prob = get_iter_on_device(
    config["batch_size"],
    config["dataset"],
    config["model_type"],
    config["n_gram"],
    config["subsample_thresh"],
    config["data_device"],
    config["add_pad"],
    config["eos_mask"],
)

# モデルのインスタンスを作成
model = Word2BoxConjunction(
    TEXT=TEXT,
    embedding_dim=config["embedding_dim"],
    batch_size=config["batch_size"],
    n_gram=config["n_gram"],
    intersection_temp=config["int_temp"],
    volume_temp=config["vol_temp"],
    box_type=config["box_type"],
)

# 作成したインスタンスに保存してあるパラメータを読み込む
model.load_state_dict(torch.load('results/best_model.ckpt'))
print(model)

word_1 = "dog"
word_2 = "cat"

# 文字列 を ID に変換
word_1_id = TEXT.stoi["dog"]
word_2_id = TEXT.stoi["cat"]

# INT の ID を LongTensorにキャストした後で類似度を計算
similarity = model.word_similarity(LongTensor([word_1_id]), LongTensor([word_2_id]))
print(similarity)
