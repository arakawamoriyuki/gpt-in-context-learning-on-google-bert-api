# gpt-in-context-learning-on-google-bert-api
Google Bertを転移学習させた自作モデルでドキュメント検索した結果をIn-context LearningしたChat GPTで回答させるAPIの実装

## 環境

TODO

## 学習

[nnlm](https://tfhub.dev/google/nnlm-ja-dim128-with-normalization/2)を利用した転移学習モデルを作成

```sh
$ python train_nnlm.py
```

