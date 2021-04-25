# SageMaker

## モデルの作成
### SageMaker Studio
#### SageMaker Studio Notebooks

### SageMaker Autopilot
自動で、未加工データを検証、機能プロセッサーを適用し、最適なアルゴリズムセットを選出、複数のモデルをトレーニング及び調整し、パフォーマンスの追跡とパフォーマンスに基づくモデルのランク付けを行う

### SageMaker Ground Truth
ラベリングサービス

## モデルのトレーニング
### Amazon SageMaker Experiments
機械学習モデルへの繰り返し処理を追跡および調整するのに使う。
入力パラメータや構成、結果等を自動的に補足し、これらを「実験結果（experiments）」として保存することで、繰り返し作業を管理しやす区する。
### Amazon SageMaker Debugger
モデル精度の改善のために、トレーニングや評価、混同行列、学習勾配などのリアルタイムメトリクスをトレーニング中に自動取得し、トレーニング処理を明確化する。
取得したメトリクスは、Amazon SageMaker Studioで確認できる

## モデルのデプロイ
### Amazon SageMaker Model Monitor
機械学習モデルの品質についてリアルタイムの継続的な監視を行う。
エンドポイントから予測リクエストと応答を簡単に収集し、本番稼働用環境で収集されたデータを分析し、トレーニングもしくは検証データと比較して偏差を検出。
これにより、コンセプトドリフトによる機械学習モデルのパフォーマンスの低下に対応できる。
### Amazon Augmented AI (Amazon A2I)
人による機械学習予測のレビューに必要なワークフローを簡単に構築。
Amazon SageMakerで構築した機械学習モデル用に独自のワークフローを作成することができ、モデルが信頼性の高い予測を作成できない場合に、人を介したレビューを行うことができる。

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
数行の`scikit-learn`または`Spark`コードのみを使用して、独自の変換ロジックを実装できる

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

# Rekognition
AWS CLI または Rekognition APIを使用
画像から人物特定とかテキスト抽出とか人の装備とかをチェックするのが得意

## Rekognition DetectFaces
検出された各顔を含むJSON応答を返すことができる。  
Amazon Rekognitionは画像検品ソリューションに特化したサービスではない（代わりにG2インスタンスに
ソフトウェアを導入する方が良い）
※画像をバイトコードに変換してデータを受け渡す方法はサポート外

## Amazon Rekognition Video
Amazon Rekognition Video ではライブビデオストリーミングをリアルタイムで解析して、顔を検出し、判別できます。Amazon Kinesis Video Streams のストリーミングデータを Rekognition Video に入力し、最大数千万もの顔データと照らし合わせて、超低レイテンシーでの顔認識を行います。バッチ処理のユースケースとして、Amazon Rekognition Video では Amazon S3 に保存した録画データを解析することもできる。
Amazon Rekognition Videoは、ストリーミングビデオの分析を開始および管理するために使用できるストリームプロセッサ（CreateStreamProcessor）を提供します。

# Polly
テキストの音声読み上げサービス

# Transcribe
音声からテキスト変換サービス

# Lex
チャットボットサービス

# Forcast
時系列予測サービス
需要予測など

# Personalize
レコメンデーションサービス
おすすめ商品とか

# Comprehend
テキストからそのテキストの人がどういう感情なのか判定する

# Translate
翻訳サービス


# Glue
