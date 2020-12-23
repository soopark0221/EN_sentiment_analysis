# EN_sentiment_analysis

## 설명
BERT는 양방향으로 사전학습을 진행한 언어 모델이다. BERT의 사전학습은 MLM, NSP 두가지 방식으로 진행되며, 사전학습된 모델을 Fine-tuning을 통해 다양한 Task를 수행할 수 있다.
본 프로젝트에서는 BERT를 사용하여 Friends dialog에 대한 감정 분석을 진행한다. Friends diaglog 데이터는 speaker, utterance, annotation, emotion으로 구성되며 emotion은 8가지로 구분된다. Utterance 입력에 대하여 emotion을 분류하는 Multiclass classification 학습 모델을 만든다. 

## Installation
  - Python 3.7
  - PyTorch 1.7.0
  - transformers 4.1.1
  - json
  - pandas
  - seaborn
  - matplotlib
  - numpy
  - sklearn

## Input data
Friends : http://doraemon.iis.sinica.edu.tw/emotionlines/download.html
- friends_train.txt : 10.6K dialogs -> Train 데이터로 사용
- friends_dev.txt : 1.2K dialogs -> Validation 데이터로 사용
- format : speaker, utterance, annotation, emotion (8 class)

## 특징
- Emotion label은 총 8개의 class로 구분
  - [neutral, non-neutral, joy, surprise, anger, sadness, disgust, fear]
- Class Imbalance
![](pics/class_balance.png)

## 환경
- colab 개발환경 사용
- gpu 사용으로 학습속도 향상

## Training 실행
1. Import Modules
  - Installation의 모듈을 설치한다.
 
2. Data Load
  - 올바른 train data의 Path를 입력하여 train data를 로딩한다.
  - json 파일을 dataframe 형식을 변환한다.
  - Data의 label 분포를 통해 Imbalance 여부를 확인한다.
  - Data의 document의 최대, 평균 길이를 확인한다. Friends 데이터의 평균 길이는 39.7이다. 이후 단어 임베딩 시 max_length를 결정할 때 참고한다.
 
3. Preprocessing
  - bert-base-uncased tokenizer를 사용해 utterance 데이터를 토큰화하고 임베딩한다. 
  - max_sequence_length = 40~60으로 설정하며, Padding하여 문장 길이를 매칭한다.
  - 임베딩된 문장에서 padding이 아닌 부분은 1, padding 부분은 0으로 입력하는 attention mask를 만들어 0인 부분은 attention을 수행하지 않도록 한다.
  - str 형식의 emotion 데이터를 int로 변환한다.
 
4. Modeling
  - torch.utils.data.dataloader를 사용해 문장 입력, attention mask, label 데이터를 묶고 설정한 배치사이즈 만큼 데이터를 가져온다.
  - 배치사이즈는 16으로 설정한다.
  - Training 모델은 bert-base-uncased 모델을 사용한다.
  - Linear classifier로 Label 분류를 수행한다. 

5. Training 환경 및 파라미터
  - gpu (cuda library)
  - Optimizer : AdamW
  - Loss 함수 : Cross entropy
  - Epoch : 4,5
  - Learning rate : 1e-4~ 매 epoch마다 감소하도록 step 부여
  - Train 데이터로 학습, validation 데이터로 평가
  - 평가 항목 : accuracy (multiclass data에서는 accuracy, micro f1-score, micro recall, micro precision이 동일한 값을 가진다)

## Model 저장 
   - 학습한 모델을 torch.save를 사용해 저장한다.
   
## Test 실행 
  - Model를 테스트하기 위해 csv 형식의 unlabeled된 test data를 준비한다. 
  - Train 데이터와 동일한 폴더에 Test할 데이터를 저장하거나 path를 정확히 명시한다.
  - 단어임베딩과 attention mask를 만들고 학습 모델에 입력한다.
  - 출력값을 실제 데이터 label과 비교하여 모델의 성능을 테스트한다. 

## Reference
  - https://mccormickml.com/2019/07/22/BERT-fine-tuning/
  - https://medium.com/@inmoonlight/pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%95%98%EA%B8%B0-cnn-62a9326111ae
  - 실습파일 : https://colab.research.google.com/drive/1EMzEfTYjYLgEHjCCP1vEr9oOZLXMocGh?usp=sharing
