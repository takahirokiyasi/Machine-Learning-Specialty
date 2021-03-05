# ベイスの定理
https://ai-trend.jp/basic-study/bayes/bayes-theorem/

# 勾配ブースティング
テーブルデータの教師あり学習において幅広いデータセットで高い精度を出すモデルとして知られている。弱い学習器を逐次的に学習するモデルである。
弱学習器としては具体的なモデルを仮定していない。
勾配降下法を使う

## 勾配ブースティング木（GBDT）
弱学習器に決定木を使用したもの
xgboost・lightGBMとかこれ

# ランダムフォレスト
バギングと呼ばれるサンプルの重複ありランダムサンプリングと各決定木で使用する特徴量のランダムサンプリングを行っている。

# 決定木における特徴量重要度
## 分割回数(frequency)
それぞれの特徴量の分割回数
## 不純度

# 正規化
最小値を0，最大値を1とする0-1スケーリング手法
異常に小さかったり、大きかったりする外れ値がある場合はデータが偏ってしまうので注意が必要
# 標準化
平均を0，分散を1とするスケーリング手法
最大値や最小値が決まっていないため特定のアルゴリズムで問題になったりする
一般的には標準化の方がよく使う

# バイアスとバリアンス
訓練をすればするほどバイアスは低くなるが、一方でバリアンスは高くなりがち（トレードオフの関係）
## バイアス
簡単にいうと実際値と予測値の差
## バリアンス
予測値の散らばってる度合い

# アンサンブル学習
## バギング
それぞれ学習した多数の弱学習器の多数決（分類の場合　回帰の場合は平均値）を取ることによって汎化性能を高めたアンサンブル学習のこと
並列的
## ブートストラップサンプリング（重複ありランダムサンプリング）
重複を許して無作為同数リサンプリングを反復する方法
ちなみに重複を許さないのがジャックナイフ法
## ブースティング
直列的に弱学習器を用いる。学習器を連続的に学習させて、より精度が向上するように修正していく
バイアスは下がりやすく、バリアンスが高くなりやすい、過学習に注意
## スタッキング
第一段階として学習器にランダムフォレストや勾配ブースティングなどのさまざまな計算法を使って複数のモデルを用意し学習し、予測値を出力（データセットは分けてあるモデルで学習に使ったデータセットと予測に使うデータセットはそれぞれ別にしないと過学習になる）
第一段階の予測値を取りまとめるモデルをメタモデルという。メタモデルは第一段階の予測値を特徴量として学習して、最終の予測値を出す。
メタモデルには回帰ならよく線形回帰モデルが使われるが特にアルゴリズムの決まりはない
[アンサンブル学習の種類](https://toukei-lab.com/ensemble#i-2)