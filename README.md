# gpt-in-context-learning-on-google-bert-api
Google Bertを転移学習させた自作モデルでドキュメント検索した結果をIn-context LearningしたChat GPTで回答させるAPIの実装

## 環境

TODO

## 学習

input/datasets.ymlを記載して以下3つのうちどれかを実行。下に行けば行くほど性能良いが、推論は遅い。

[nnlm](https://tfhub.dev/google/nnlm-ja-dim128-with-normalization/2)を利用した転移学習モデルを作成

```sh
$ python train_nnlm.py
```

[universal-sentence-encoder-multilingual](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)を利用した転移学習モデルを作成

```sh
$ python train_usem.py
```

[bert](https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3)を利用した転移学習モデルを作成。

```sh
$ python train_bert.py
```
