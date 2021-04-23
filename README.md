# P2-KLUE

# 전체 개요 설명
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.
이번 대회에서는 문장, 엔티티, 관계에 대한 정보를 통해, 문장과 엔티티 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 엔티티들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 model이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

- 예시
    - sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
    - entity 1: 썬 마이크로시스템즈
    - entity 2: 오라클
    - relation: 단체:별칭
- input: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.
- output: relation 42개 classes 중 1개의 class를 예측한 값입니다.
- 위 예시문에서 단체:별칭의 label은 6번(아래 label_type.pkl 참고)이며, 즉 모델이 sentence, entity 1과 entity 2의 정보를 사용해 label 6을 맞추는 분류 문제입니다.

# 평가 방법
- 모델 제출은 하루 5회로 제한됩니다.
- 평가는 테스트 데이터셋의 Accuracy 로 평가 합니다. 테스트 데이터셋으로 부터 관계를 예측한 classes를 csv 파일로 변환한 후, 정답과 비교합니다.

# 데이터 개요
전체 데이터에 대한 통계는 다음과 같습니다. 학습에 사용될 수 있는 데이터는 train.tsv 한 가지 입니다. 주어진 데이터의 범위 내 혹은 사용할 수 있는 외부 데이터를 적극적으로 활용하세요!
- train.tsv: 총 9000개
- test.tsv: 총 1000개 (정답 라벨 blind)
- answer: 정답 라벨 (비공개)

학습을 위한 데이터는 총 9000개 이며, 1000개의 test 데이터를 통해 리더보드 순위를 갱신합니다. private 리더보드는 운영하지 않는 점 참고해 주시기바랍니다.
- label_type.pkl: 총 42개 classes (class는 아래와 같이 정의 되어 있며, 평가를 위해 일치 시켜주시길 바랍니다.) pickle로 load하게 되면, 딕셔너리 형태의 정보를 얻을 수 있습니다.
- {'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9, '단체:구성원': 10, '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19, '단체:본사_도시': 20, '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29, '인물:거주_주(도)': 30, '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39, '인물:사망_국가': 40, '인물:나이': 41}

- Data 예시
![24c50b71-20c4-4200-8c29-53d1af36617e](https://user-images.githubusercontent.com/55614265/114416836-0bf31800-9bec-11eb-98f7-8e4ce1a20958.png)
    - column 1: 데이터가 수집된 정보.
    - column 2: sentence.
    - column 3: entity 1
    - column 4: entity 1의 시작 지점.
    - column 5: entity 1의 끝 지점.
    - column 6: entity 2
    - column 7: entity 2의 시작 지점.
    - column 8: entity 2의 끝 지점.
    - column 9: entity 1과 entity 2의 관계를 나타내며, 총 42개의 classes가 존재함.
    - class에 대한 정보는 위 label_type.pkl를 따라 주시기 바랍니다.

# 코드 설명
```
$> tree -d
.
├── /Baseline Code
│     ├── evaluation.py : public leaderboard와의 정답을 비교해 정확도를 알려줍니다.
│     ├── inference.py : 학습된 model을 통해 prediction하며, 예측한 결과를 csv 파일로 저장해줍니다.
│     ├── load_data.py : baseline code의 전처리와 데이터셋 구성을 위한 함수들이 있는 코드입니다.
│     ├── requirements.txt : 라이브러리 설치
│     └── train.py : baseline code를 학습시키기 위한 파일입니다.
├── EDA.ipynb : 간단한 EDA 코드입니다.
├── [Try_1] KoBERT.ipynb : KoBERT 기반의 Classification 코드입니다.
├── [Try_2] KoBERT_LSTM.ipynb : KoBERT 기반의 Classification에 LSTM을 추가한 코드입니다.
└── [Try_3] CNN.ipynb : CNN 기반의 Classification 코드입니다.
```

# Wrap up Report

## 기술적인 도전

### 본인의 점수 및 순위

- LB 점수 accuracy: 0.0000%, 136등(꼴등)

    1. 데이터 부족 (Train : 9,000 + Test : 1,000 = Total : 10,000)
    2. 부족한 데이터에 비해 많은 클래스 (Class : 42)
    3. 데이터 불균형
    - ![](https://images.velog.io/images/skaurl/post/c0fdfd9e-d6f3-4009-b0e1-6279d655e872/image.png)
    4. 데이터 오류 (공식적으로 9개, 비공식은…)
    5. Test dataset에 없는 클래스가 존재 (3개로 추정)
    6. Private Leaderboard의 부재
    7. 베이스라인 코드만으로도 상위권(80% 대) 가능

    - 세상에 완벽한 데이터가 없다는 것은 알고 있습니다.
    - 하나씩 따로 보면 1~7 모두 극복해야 하고 극복할 수 있는 문제라고 생각합니다.
    - 하지만 이렇게 많은 문제가 동시다발적으로 존재한다면, 이는 Competition과 데이터에 어느 정도 문제가 있다고 생각합니다.
    - 그리고 데이터에 문제가 있다 보니, 성능 향상에 도움이 된다고 알려진 일반적인 방법들 또는 SOTA 논문에서 제시된 방법들을 사용해보더라도 성능에는 큰 영향을 미치지 못했습니다.
    - 성능과는 별개로 시도해보고 도전해보는 것도 중요하다고 하지만, 시도와 도전을 통해서 Competition에서 가장 중요한 성능 향상을 하지 못한다면 의미가 많이 퇴색된다고 생각합니다.
    - 따라서, 이번 Competition은 리더보드에 큰 의미가 없다고 판단되어 accuracy: 0.0000%로 제출하였습니다. (~~물론, 그렇다고 논 것은 아닙니다.~~)

### 검증(Validation)전략

1. Train : Vali= 4 : 1 (random_seed = 42)
2. 1번의 Vali_acc와 Test_acc가 유사한 것을 확인
3. Split 비율과 random_seed에 따라 성능 차이가 있는 것을 확인
4. 2, 3번을 이유로 5-fold cross-validation을 따로 사용하지 않았습니다.

### 사용한 모델 아키텍처 및 하이퍼 파라미터

- Baseline Code : xlm-roberta-large
    1. http://boostcamp.stages.ai/competitions/4/discussion/post/144
    2. LB 점수 accuracy : 78 ~ 79%
    3. output_dir='./results',
    4. save_total_limit=3,
    5. save_steps=100,
    6. num_train_epochs=10,
    7. learning_rate=1e-5,
    8. per_device_train_batch_size=32,
    9. per_device_eval_batch_size=32,
    10. warmup_steps=300,
    11. weight_decay=0.01,
    12. logging_dir='./logs',
    13. logging_steps=100,
    14. evaluation_strategy='steps',
    15. eval_steps = 100,
    16. dataloader_num_workers=4,
    17. label_smoothing_factor=0.5
    18. 해당 모델에 ensemble을 하면 accuracy : 80%이 가능합니다.
    19. ~~해당 결과로 이게 무슨 의미가 있는지 허탈함을 느꼈습니다.~~

### 앙상블 방법

1. Hard Voting
2. Soft Voting

### 시도했으나 잘 되지 않았던 것들

1. KoBERT 기반의 Classification
    1. 계기 : transformers의 Trainer는 편리한 만큼 코드를 수정하기가 어렵다고 생각합니다. 그래서 가장 기본적인 형태로 구현하는 것이 여러 가지로 실험해볼 수 있다고 생각되어 KoBERT 기반의 Classification을 구현하였습니다.
    2. 참고 자료 : https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb
    3. Code : http://boostcamp.stages.ai/competitions/4/discussion/post/144
    4. 해당 Competition에서는 데이터가 부족하기 때문에, Pretrain한 모델이 더 적합했다고 생각합니다. (베이스라인 코드의 결과가 이를 뒷받침합니다.)
    5. 그래서 처음부터 학습시키는 해당 모델은 한계가 있다고 생각했고 첫 주차에만 다뤄봤습니다.
    6. 그래도 조금 더 개선할 여지는 있었다고 생각합니다.
        1. KoBERT에서 xlm-roberta-large로 변경
        2. dataset['sentence'] = dataset['entity_01'] + ' [SEP] ' + dataset['entity_02'] + ' [SEP] ' + dataset['sentence']가 아니라 tokenizer로 dataset['entity_01'] + ' [SEP] ' + dataset['entity_02']와 dataset['sentence']를 각각 문장으로 봤어야 한다고 생각합니다.
        - Ex) tokenized_sentences = tokenizer(concat_entity, list(dataset['sentence']), return_tensors="pt", padding=True, truncation='only_second', max_length=100, add_special_tokens=True)
        3. i, ii를 했다면 충분히 베이스라인 코드는 따라잡았을 것이라고 생각합니다.

2. KoBERT(+LSTM) 기반의 Classification
    - 1번의 KoBERT 기반의 Classification에서 LSTM Layer를 추가했으나 생각보다 성공적이지 않아서 아쉬웠습니다.

3. CNN 기반의 Classification
    1. 계기 : 이전에 NLP에 CNN 기반의 Classification을 구현해본 적이 있습니다. 당연히 성능은 높지 않을 것이라고 예상했으나, 실험 차원에서 해당 Competition에 적용해보았습니다.
    2. 참고 자료 : https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/
    3. 다음과 같이 임베딩을 해주었습니다. NLP에서 CNN을 사용할 때에는, 캐릭터 단위로 임베딩해주는 것이 좋다고 알고 있습니다.
    - ![](https://images.velog.io/images/skaurl/post/6c05a687-1632-4ba3-99a7-1baf9bb27c12/image.png)
    4. 예전에 사용했던 코드로 손 쉽게 구현하기는 했지만, 예전 기술인 만큼 성능이 좋지 못했습니다.
    5. 새삼스럽지만 BERT 이후 기술들의 힘을 느낄 수 있었습니다.

4. 그 외
    1. Entity에 NER을 추가해보았으나, 생각보다 성공적이지 않아서 아쉬웠습니다.
        - 누구는 성능이 올랐다고 하고, 누구는 성능에 변화가 없다고 하니, 사람마다 케이스 바이 케이스가 있는 것 같습니다.
    2. 외부 데이터를 추가해보았으나, 생각보다 성공적이지 않아서 아쉬웠습니다.
        - 과도한 일반화로 인해 실패했다고 생각합니다.
        - 해당 Competition에서는 일반화보다는 train과 test dataset에 overfitting 되는 것이 중요했다고 생각합니다.

### 대회와 관계없이 시도한 것들

- 현재 부스트캠프에서 만난 팀원들과 함께 KBO 챗봇 프로젝트를 하고 있습니다.
- 위에서도 언급했지만, 해당 Competition을 진행하는 것 보다는 해당 프로젝트를 진행하는 것이 더 좋을 것 같다고 판단했습니다.
- 그래서 이번 Competition과 병행하면서 다음과 같은 일을 처리했습니다.
- ~~Competition에 열심히 참여하지는 않았지만 놀지 않았다는 것을 보여주고자...~~

1. KBO Record Crawler : https://github.com/baseballChatbot7/KBO-Record-Crawler
    1. KBO 기록 크롤러를 구현 후, 데이터를 수집했습니다.
    2. 해당 데이터는 KBO 챗봇에 사용될 예정입니다.

2. KBOBERT : https://github.com/baseballChatbot7/KBOBERT
    1. 실습에 사용했던 BERT 학습 코드를 사용하면, KBO 도메인에 특화된 BERT를 만들 수 있지 않을까 생각했습니다. (GPU도 자유롭게 사용할 수 있으니까요.)
    2. 그래서 실습 코드를 받자마자, KBO 관련 뉴스 크롤러를 구현 후 약 46,000건의 뉴스를 수집했습니다.
    3. 테스트 차원에서 해당 데이터를 가지고 BERT를 학습시켜본 결과(10에포크, 약 20시간 학습), 야구 도메인에 대한 [MASK]를 잘 예측하는 것을 확인할 수 있었습니다.
    4. 최종적으로는 다른 팀원이 수집 중인 나무위키 야구 관련 문서를 합쳐서 BERT를 학습시킬 예정입니다.

3. MRC : 현재 테스트 중
    1. 저희가 만들고자 하는 챗봇은 MRC 기반입니다.
    2. MRC를 제대로 만들 수 있을까 하는 걱정이 있었지만, MRC 실습 코드를 통해서 이러한 부담을 줄일 수 있었습니다.
    3. 다음 STAGE 3에서 MRC를 제대로 구현해볼 생각이지만, 여의치 않다면 해당 코드로 MRC를 만들어보려고 합니다.

- 이번 STAGE의 가장 큰 재산은 마스터 님의 실습 코드와 직접 시도해본 결과들입니다. 이대로라면 목표로 하고 있는 네트워킹 데이까지 프로토타입 개발이 가능 할 것 같습니다…

## 학습과정에서의 교훈

### 학습과 관련하여 개인과 동료로서 얻은 교훈

- 이번 Competition에서는 리더보드 보다는 토론에 더 많은 집중을 했습니다.
- 많은 분들이 읽어 주시고, 좋아해 주시고, 사용해 주셔서 뿌듯했습니다.
- 다음 Competition에서도 리더보드에 집중하지 않을 것이라면, 토론에 집중할 생각입니다.

### 피어세션을 진행하며 좋았던 부분과 동료로부터 배운 부분

- 이번 피어세션을 통해서 팀원으로 함께하고 싶은 사람을 만날 수 있었고, 이를 계기로 STAGE 3의 팀원으로 이어져서 좋았습니다.

## <마주한 한계와 도전숙제>

### 아쉬웠던 점들

- Competition의 완성도가 낮아 몰입하기 어려웠던 점이 많이 아쉬웠습니다.
- Competition과는 별개로 개인적으로 시도와 도전을 해볼 수 있지 않았을까…? 하는 아쉬움이 남습니다.

### 한계/교훈을 바탕으로 다음 스테이지에서 새롭게 시도해볼 것

- 위에서 언급한 프로젝트를 위해 다음 스테이지에서는 성능이 좋은 MRC 개발에 힘을 쏟을 계획입니다.
