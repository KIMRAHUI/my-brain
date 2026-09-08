# Tensor-Creation-Functions

특수한 텐서 생성 함수

텐서를 처음부터 만들 때, 값을 하나하나 직접 입력하는 대신 PyTorch가 제공하는 생성 함수를 활용하면 훨씬 편합니다. 특히 딥러닝에서는 모델의 가중치를 "무작위 값"으로 초기화하거나, 마스크(mask)나 초기 상태를 "0 또는 1로 채운 텐서"로 만들어야 하는 경우가 매우 많습니다.

1) 랜덤한 값으로 만들기

torch.rand() — 균등 분포(Uniform Distribution)

0과 1 사이의 균등 분포를 따르는 난수 텐서를 생성합니다. "균등 분포"란 0~1 사이의 모든 값이 나올 확률이 동일하다는 뜻입니다.

import torch

t = torch.rand(2, 3)
print(t)

tensor([[0.5734, 0.1298, 0.8821],
        [0.4456, 0.9012, 0.2233]])

범위 확장 팁: 0~1이 아닌 다른 범위(예: -5 ~ 5)의 난수가 필요하면 아래 공식을 사용합니다.

min_val, max_val = -5, 5
t_ranged = min_val + (max_val - min_val) * torch.rand(2, 3)

원리: torch.rand()가 만드는 값은 항상 [0, 1) 구간이므로, 여기에 (max_val - min_val)을 곱해서 구간의 "폭"을 늘리고, min_val을 더해서 시작점을 옮기는 것입니다.

torch.randn() — 정규 분포(Normal / Gaussian Distribution)

평균 0, 표준편차 1인 표준 정규 분포를 따르는 난수를 생성합니다. rand와 달리 값이 0 근처에 몰려 있고, 아주 드물게 큰 음수/양수 값도 나올 수 있습니다. (종 모양 분포)

t = torch.randn(2, 3)
print(t)

tensor([[-0.3211,  1.4523, -0.8821],
        [ 0.0456, -1.2012,  0.7233]])

왜 가중치 초기화에 필수적인가?

딥러닝 모델의 초기 가중치를 모두 0이나 동일한 값으로 설정하면, 모든 뉴런이 동일하게 학습되어버리는 "대칭성 문제(symmetry problem)"가 발생합니다. torch.randn()처럼 평균 0 근처에 값이 몰려있는 무작위 분포로 초기화하면, 각 뉴런이 서로 다른 방향으로 학습을 시작할 수 있어 이 문제를 방지합니다. 그래서 nn.Linear, nn.Conv2d 등 PyTorch의 레이어 내부에서도 기본적으로 이와 유사한 원리(정규분포 또는 균등분포 기반)로 가중치가 초기화됩니다.

rand vs randn 비교

특정한 값으로 채우기

torch.zeros() — 0으로 채우기

t = torch.zeros(2, 3)
print(t)

tensor([[0., 0., 0.],
        [0., 0., 0.]])

주로 마스크(mask) 초기값, 손실(loss) 누적용 변수 초기화, 신경망의 편향(bias) 초기값 등에 사용됩니다.

torch.ones() — 1로 채우기

t = torch.ones(2, 3)
print(t)

tensor([[1., 1., 1.],
        [1., 1., 1.]])

마스킹 연산이나 "전체를 그대로 유지"하는 초기값이 필요할 때 자주 사용됩니다.

torch.full((shape), value) — 사용자 지정 상수로 채우기

t = torch.full((2, 3), 7)
print(t)

tensor([[7, 7, 7],
        [7, 7, 7]])

zeros나 ones가 아닌 다른 상수(예: 패딩값으로 자주 쓰이는 -1이나 특정 토큰 ID)로 텐서를 채워야 할 때 사용합니다.

자주 하는 실수

torch.rand()와 torch.randn()을 이름이 비슷해서 혼동하는 경우가 많습니다. 이름 끝에 n이 붙으면 "Normal(정규분포)"라고 기억하면 헷갈리지 않습니다.

torch.full((2, 3), 7)에서 shape 인자는 튜플 (2, 3)로 감싸서 넣어야 합니다. torch.zeros(2, 3)처럼 괄호 없이 숫자만 나열하는 다른 함수들과 문법이 다르니 주의가 필요합니다.

매번 실행할 때마다 rand/randn의 결과가 달라지는 것이 당연합니다. 실험을 재현 가능하게 하려면 torch.manual_seed(42)처럼 시드를 고정해야 합니다.

torch.manual_seed(42)
t = torch.rand(2, 3)  # 항상 동일한 값이 나옴

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Tensor-Shape-Interpretation

Tensor-Operations-Broadcasting

