# SageMaker

## モデルの作成
### SageMaker Studio
#### SageMaker Studio Notebooks

### SageMaker Autopilot
自動で、未加工データを検証、機能プロセッサーを適用し、最適なアルゴリズムセットを選出、複数のモデルをトレーニング及び調整し、パフォーマンスの追跡とパフォーマンスに基づくモデルのランク付けを行う

### SageMaker Ground Truth
ラベリングサービス

## SageMaker Clearify
ブラックボックスになりがちな機械学習のモデルの解釈可能性・公平性を明らかにする際の支援をする機能

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

## 運用
### パイプモード
パイプ入力モードを使うとデータセットが最初にダウンロードされるのではなく、トレーニングインスタンスに直接ストリーミングされる。

- s3からストリーミング
- EBSの消費が少ない
- 読み込みが高速

### ファイルモード
EBSにデータをダウンロードしてから処理
- 容量を消費する
- 読み込みが遅い

### パイプラインモード
前処理を含めた形でモデル作成まで行う
処理A用のコンテナ、処理B用のコンテナ、モデル作成用のコンテナと順番に実行する
パイプラインは同じEC2上で処理するのでローカルデータを利用するため、処理が早くなる
record io形式だとさらに高速

### デプロイ方式
#### 永続エンドポイント
リアルタイム推論。deployメソッド。APIで取得。
#### バッチ変換
データセット全体を推論。create_transform_jobメソッド。結果はS3に保存。
以下のユースケースで使用される。
- データセットを前処理して、トレーニングや推論を妨げるノイズやバイアスをデータセットから取り除く場合。
- 大規模なデータセットから推論を取得する場合。
- 永続的なエンドポイントが不要なときに推論を実行する場合。
- 入力レコードを推論に関連付けて結果の解釈に役立てる場合。

#### ローカルモード
オンプレミスでコンテナを起動し、学習や推論を行う

## Edge Manager

## 推論パイプライン（inferece pipeline）
パイプラインは線形に並べられたコンテナで構成されていて、コンテナの数は2~5
前処理・推論・後処理を行うことができる
### 特徴量の前処理
数行の`scikit-learn`または`Spark`コードのみを使用して、独自の変換ロジックを実装できる

## Dockerレジストリパス
:1を選択するとアルゴリズムの安定バージョンが使用できる
:latestだと最新だけど後方互換性がない場合がある

## 組み込みアルゴリズム
### 分類・回帰
- Linear Learner
- Factorization Machines（因数分解機）
- XGBoost Algorithm(Tenserflowのアルゴとしても使える)
- K-Nearest Neighbors
サンプリング・次元削減・インデックス作成の3ステップがある
### 画像分類
- Image Classification Algorithm
### 物体検出
- Object Detection Algorithm
### 機械翻訳
- Sequence2Sequence(seq2seq)
翻訳したり対話モデルを作ったり
EncoderとDecoderを使う
### クラスタリング
- K-Means Algorithm
kはクラスタリングの数を決めるハイパーパラメータ

レコメンドなどで使用

初期クラスターセンターを決める方法がランダムアプローチとK-means ++という方法がある
https://docs.aws.amazon.com/sagemaker/latest/dg/algo-kmeans-tech-notes.html

### トピックモデリング
トピックモデリングはクラスタリングとは違い複数のクラスタに属することができる
- Latent Dirichlet Allocation(LDA・潜在的ディレクレ配分)
観察は文書と呼ばれます。機能セットはトピックと呼ばれます。特徴は単語と呼ばれます。そして、結果のカテゴリはトピックと呼ばれます。

LDAはbag of words なので単語の順序は重要ではない

- Neural Topic Model(NTM・ニューラルトピックモデル)
### 時系列予測
- DeepAR Forecasting
### Word2Vec・テキスト分類
- BlazingText
教師ありモードと教師なしモード両方提供されている
word2Vecは感情分析、名前付きエンティティ認識、機械翻訳など、多くの下流の自然言語処理（NLP）のタスク
テキスト分類は、ウェブ検索、情報検索、ランキング、文書分類などのタスク
### 異常検知
- Randam Cut Forest
時系列データのスパイク、周期性の中断、分類できないデータポイントなどを検出
- ip-insight
IPアドレス,エンティティの情報をベクトル化し、そのベクトルからIPアドレス-エンティティの関連性を計算し、異常な挙動を見つける
- K-means
### 次元削減・特徴量の抽出
- PCA
通常モード：データがまばらで特徴量もそんなにない時
ランダムモード：多数の観測値と特徴をもつデータセットの場合
### セマンティックセグメンテーション
画像内をピクセル単位でどこが何を示しているかラベルづけしてくれる。
https://jp.mathworks.com/solutions/image-video-processing/semantic-segmentation.html
- Semantic Segmentation

### その他・注意点
#### 教師なし学習アルゴリズムを実行する場合
ターゲットを持たない教師なし学習アルゴリズムを実行するには、コンテンツタイプのラベル列の数を指定します。text / csv; label_size = 0
#### RecordIO形式のデータ
組み込みアルゴリズムはRecordIO形式の対応していることがあり、この形式を使うことによって学習時間を大幅に改善することができる。S3への保存コストは上がる
#### 制約
SageMakerの教師あり学習アルゴリズムの場合、ターゲット変数は最初の列にあり、ヘッダーレコードを持たないようにする必要がある

#### トレーニングデータとテストデータでの精度がどちらも悪い時
正則化をやめる
特徴量を増やす

## IAM関連
### IAM IDベースのポリシー
IAMユーザーとかIAMロールのポリシーにsagemakerへのアクセスの許可や制限を入れることでアクセスコントロールできる
### リソースタグに基づく認証
上の進化版で特定タグがついてるノートブックへアクセスの許可や制限をすることがポリシーでできる。

https://dev.classmethod.jp/articles/sagemaker-restrict-access/

### S3との接続
デフォルトのSagemakerIAMロールは、名前にsagemakerが含まれるバケットにアクセスするためのアクセス許可を取得する。
SageMakerサービスプリンシパルにS3FullAccessアクセス許可を付与するロールにポリシーを追加する場合、バケットの名前にsagemakerを含める必要はない。

## モニタリング
### CloudTrail
InvokeEndpointを除いた全てのAPIコールをキャプチャする
ユーザーやロールやAWSサービスによって実行されたアクションを記録し90日保持する。
### CloudWatch
1分間隔でメトリクスを取得できる

## Sagemeker Neo
Amazon SageMaker Neoを使うことで機械学習モデルをコンパイルすることが出来る。モデルのコンパイルを行うことで特定のデバイスに最適化を行うことが出来、推論速度の高速化と省メモリ化を実現できる
エッジデバイスで推論できるモデルを生成するため、通信環境がない場合にも使える。

よく画像分類モデルを最適化するのに使う

対応モデル:
MXNet および TensorFlow でトレーニングされた AlexNet、ResNet、VGG、Inception、MobileNet、SqueezeNet、DenseNetなどのモデル
XGBoostでトレーニングされたランダムカットフォレストモデルなど

## Production Variant
ホストしたいモデルと、それをホストするために配置するリソースを特定します。複数のモデルをデプロイする場合は、バリアントウェイトを指定することで、モデル間のトラフィックをどのように分配するかをAmazon SageMakerに伝えます。
インスタンスタイプ選べたり

## boto3 command
### describe_training_job
create_training_job（）メソッドを呼び出してトレーニングジョブを開始した後、トレーニングジョブの進行状況に関するステータスを取得できる

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
Glueは、RecordIO-Protobuf形式で出力を書き込むことはできない。Apach Spark　EMRならできる
## Glue ジョブ
### Python シェル
numpy、pandas、sklearnなどのライブラリに依存するGlueジョブをサポート
### PySpark
主にApacheSparkのPythonAPIを使用して記述されたGlueジョブをサポート
## ML transform ジョブ
サーバーレス方式で重複排除を実行できる
ラベル生成できる
## Glue Crawler（クローラー））
データストアを調べて、データカタログに登録してくれる。定期実行をする事で、スキーマやパーティションの定期的な自動更新も可能
クローラー作成してs3のデータに対してクローラーを走らせてからAthenaAthenaでクエリ実行がよくあるパターン
https://dev.classmethod.jp/articles/glue-crawler-athena-tutorial/

# Redshift
## S3からのロードのコマンド
COPYを使用すると早い
INSERTとか使うより速い

# その他
## データウェアハウスとデータレイクの違い
データウェアハウスは構造化データのみを格納できますが、データレイクは構造化データ、半構造化データ、および非構造化データを格納できる