# SageMaker
## パイプモード
パイプ入力モードを使うとデータセットが最初にダウンロードされるのではなく、トレーニングインスタンスに直接ストリーミングされます。
これにより、トレーニングジョブが直ぐに始まり、早く完了し、必要なディスク容量も少なくて済む

## バッチ変換
以下のユースケースで使用される。
- データセットを前処理して、トレーニングや推論を妨げるノイズやバイアスをデータセットから取り除く場合。
- 大規模なデータセットから推論を取得する場合。
- 永続的なエンドポイントが不要なときに推論を実行する場合。
- 入力レコードを推論に関連付けて結果の解釈に役立てる場合。

## Edge Manager

## 推論パイプライン（inferece pipeline）
数行のscikit-learnまたはSparkコードのみを使用して、独自の変換ロジックを実装できる

## 組み込みアルゴリズム
### 分類・回帰
- Linear Learner
- Factorization Machines
- XGBoost Algorithm
- K-Nearest Neighbors
### 画像分類
- Image Classification Algorithm
### 物体検出
- Object Detection Algorithm
### 機械翻訳
- Sequence2Sequence
### クラスタリング
- K-Means Algorithm
### 特徴量の抽出
- Principal Component Analysis(PCA)
### トピックモデリング
- Latent Dirichlet Allocation(LDA)
- Neural Topic Model(NTM)
### 時系列予測
- DeepAR Forecasting
### Word2Vec
- BlazingText
### 異常検知
- Randam Cut Forest


