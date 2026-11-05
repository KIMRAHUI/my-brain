# Common_data

# ──공통:실습용 토이 말뭉치 (4문장짜리 짧은 이야기 600개) ─────────────────────────
#실제 과제(AIHub말뭉치 + SentencePiece/BPE)로 바꾸기 전에, CPU에서 몇 초 만에 돌려볼 수 있는 작은 데이터입니다.##🧭 이 셀의 큰 흐름 (신경망은 글자를 못 읽고 '숫자'만 계산하므로,글 →숫자 변환 준비를 합니다)#이야기 600개 생성 →단어 사전(vocab)만들기 → encode(문장→숫자) → pad_batch(길이 맞추기) →모델 입력##🏷️ 타입 힌트란?  `x: int = 3`은 "x는 정수", `def f(a: str) -> list[int]`는 "문자열을 받아 정수 리스트를 돌려준다"는 표시입니다.#실행 결과에는 영향이 없고, '이 값이 어떤 모양인지'읽는 사람이 헷갈리지 않게 돕는 메모입니다.importrandom, math, time, copy

fromtyping importIterator
importnumpy asnp
importtorch, torch.nn asnn, torch.nn.functional asF
importmatplotlib.pyplot asplt

SEED: int= 42# SEED에는 int(정수)를 넣겠다random.seed(SEED)                   #파이썬 random용 난수 고정 (매번 같은 '무작위')np.random.seed(SEED)                # NumPy난수 고정torch.manual_seed(SEED)             # PyTorch난수 고정# device에는 torch.device객체를 넣겠다: GPU(cuda)가 있으면 GPU,없으면 CPUdevice: torch.device= torch.device('cuda'iftorch.cuda.is_available() else'cpu')

# ──데이터 정의 ──NAMES: list[str] = ['하나', '지수', '서아', '도하']        #문자열들로 이루어진 리스트PLACES: list[str] = ['도서관', '카페', '학교', '공원']
TOPICS: dict[str, list[str]] = {   #토픽 이름(str) →문장 템플릿 4개(list[str]).  {s}=사람, {p}=장소 가 들어갈 빈칸'공부': ['{s}는 {p}에서 영어를 공부했다', '{s}는 단어를 외우고 문제를 풀었다', '{s}는 공부를 마치고 커피를 마셨다', '{s}는 집에 돌아가 일찍 잠들었다'],
    '운동': ['{s}는 {p}에서 달리기를 시작했다', '{s}는 숨이 차서 잠시 물을 마셨다', '{s}는 다시 힘을 내어 열 바퀴를 달렸다', '{s}는 운동을 마치고 샤워를 했다'],
    '요리': ['{s}는 시장에서 채소를 샀다', '{s}는 집에서 채소를 썰고 국을 끓였다', '{s}는 완성된 국을 식탁에 놓았다', '{s}는 가족과 함께 맛있게 먹었다'],
    '여행': ['{s}는 아침 일찍 기차를 탔다', '{s}는 창밖으로 바다를 구경했다', '{s}는 도착해서 사진을 많이 찍었다', '{s}는 저녁에 숙소로 돌아왔다'],
}

TOPIC_NAMES: list[str] = list(TOPICS)                      # ['공부', '운동', '요리', '여행']defmake_story(topic: str) -> list[str]:
    """토픽 이름을 받아 이름·장소를 채운 4문장 이야기를 돌려준다."""s: str= random.choice(NAMES)                          #예: '서아'p: str= random.choice(PLACES)                         #예: '학교'return[t.format(s=s, p=p) fortinTOPICS[topic]]     # .format이 {s},{p}빈칸을 채운다#이야기 한 개 = (문장 4개,토픽 번호).  tuple[...]에 별명(Story)을 붙여 두면 아래 타입 힌트가 짧아진다Story = tuple[list[str], int]

#아래 한 줄 =이중 반복문:  150번 × (토픽 4개) =이야기 600개STORIES: list[Story] = [(make_story(t), ti) for_ inrange(150) forti, t inenumerate(TOPICS)]
random.shuffle(STORIES) #섞기n_tr: int= int(len(STORIES) * 0.8)   
# 80% = 480 →학습용TRAIN_STORIES: list[Story] = STORIES[:n_tr] 
#학습 480개VALID_STORIES: list[Story] = STORIES[n_tr:]                
#검증 120개 (학습에 안 쓰고 성능 확인용)
# ──토크나이저 (단어 단위) ──글(str) ↔숫자(id)를 서로 바꿔 주는 도구PAD, UNK, BOS, EOS, CLS, SEP, MASK = 0, 1, 2, 3, 4, 5, 6
#특수 토큰 번호: PAD=빈칸채움, UNK=모르는단어, BOS/EOS=문장시작/끝, CLS/SEP/MASK=BERT용SPECIALS: list[str] = ['<pad>', '<unk>', '<bos>', '<eos>', '[CLS]', '[SEP]', '[MASK]']
words: list[str] = sorted({w forst, _ inSTORIES forsent inst forw insent.split()})   #모든 단어를 중복 없이 가나다순 (64개)itos: list[str] = SPECIALS + words                         # itos = id →단어   (itos[7] == '가족과')stoi: dict[str, int] = {w: i fori, w inenumerate(itos)}  # stoi =단어 → idV: int= len(itos)                                         #사전 크기 71 (=모델 출력 차원)defencode(sent: str) -> list[int]:
    """문장(str) →토큰 id리스트.예) '하나는 도서관에서 영어를 공부했다' → [66, 21, 47, 9].사전에 없으면 UNK(1)."""return[stoi.get(w, UNK) forwinsent.split()]        # ⚠️ split()으로 '단어'단위로 쪼개야 한다 (빠지면 글자 단위가 되어 전부 UNK)defdecode(ids: list[int]) -> str:
    """토큰 id리스트 →문장(str).특수 토큰은 뺀다.  encode의 반대 방향 (모델 출력을 사람이 읽을 때 사용)."""return' '.join(itos[i] foriinidsifinot in(PAD, BOS, EOS, CLS, SEP))

defpad_batch(seqs: list[list[int]], pad: int= PAD) -> torch.Tensor:
    """길이가 다른 시퀀스들을 가장 긴 길이에 맞춰 PAD(0)로 채워 (B, L)텐서로 만든다.예) [[66,21,47,9],[66,56,24]] → tensor([[66,21,47,9],[66,56,24,0]])   (B=문장 수, L=가장 긴 길이)"""L: int= max(len(s) forsinseqs)
    returntorch.tensor([list(s) + [pad] * (L- len(s)) forsinseqs])

print(f'device={device} | train/valid이야기 ={len(TRAIN_STORIES)}/{len(VALID_STORIES)} | vocab={V}')
print(TRAIN_STORIES[0])
"""device=cpu | train/valid이야기 = 480/120 | vocab=71(['서아는 학교에서 영어를 공부했다', '서아는 단어를 외우고 문제를 풀었다', '서아는 공부를 마치고 커피를 마셨다', '서아는 집에 돌아가 일찍 잠들었다'], 0)"""



