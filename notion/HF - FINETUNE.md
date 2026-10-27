# HF - FINETUNE

공통 규칙
: 어떤 과제든 Dataset이 돌려주는 딕셔너리 키 이름이 모델 forward( )의 인자 이름과 같아야하고,
 정답 키는 labels입니다. 
→ 이 한가지만 지키면 Trainer가 loss 계산부터 역전파 까지 처리합니다.

 

graph LR
    classDef step fill:#f0f4f8,stroke:#333,stroke-width:1.5px,rx:10,ry:10;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px,rx:10,ry:10;

    S1["<b>STEP 1</b><br/>모델 선택<br/>(num_labels=6)"]:::step --> S2
    S2["<b>STEP 2</b><br/>토크나이저<br/>(AutoTokenizer)"]:::step --> S3
    S3["<b>STEP 3</b><br/>데이터셋<br/>(SPDataSet)"]:::step --> S4
    S4["<b>STEP 4</b><br/>모델 로드<br/>(AutoModel)"]:::step --> S5
    S5["<b>STEP 5</b><br/>학습 인자<br/>(TrainingArgs)"]:::step --> S6
    S6["<b>STEP 6</b><br/>Trainer 학습<br/>(train)"]:::core --> S7
    S7["<b>STEP 7</b><br/>평가·저장<br/>(pipeline)"]:::core

    linkStyle default stroke:#666,stroke-width:1.5px;















