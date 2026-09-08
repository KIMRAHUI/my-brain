# PyTorch Tensor Manipulation and Operations Deep Dive

개요

본 문서는 PyTorch의 핵심 데이터 구조인 텐서(Tensor)의 행렬 구조 해석, 생성 함수, CPU/GPU 장치(Device) 관리 및 디바이스 불일치 에러 해결법, 데이터 타입 변환, NumPy 배열과의 연동, 그리고 브로드캐스팅(Broadcasting) 원리를 심층적으로 다룹니다.

텐서의 행과 열 (Shape 해석 방법)

행렬이나 텐서의 모양을 나타낼 때 (행, 열) 또는 (세로, 가로) 순서로 적습니다. 헷갈릴 때는 다음 기준을 적용합니다.

순서 기억하기: 우리말 "행렬"을 떠올리세요. 행(세로) 먼저, 열(나중)입니다.

코드와 출력 매칭 팁 (torch.rand(2, 3) 예시):

텐서 기본 속성 및 메서드 치트시트

CPU와 GPU 장치(Device) 관리 및 에러 해결

딥러닝에서 CPU와 GPU의 역할

CPU: 복잡한 제어 로직, 데이터 전처리, 모델 객체 생성 초기 단계 등에 주로 사용됩니다.

GPU: 대규모 병렬 연산(행렬 곱셈, 역전파 등)을 초고속으로 처리하여 딥러닝 학습 시간을 극적으로 단축시킵니다.

장치 지정 및 이동 방법 (.to() 또는 .cuda())

장치 가용성 확인: device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 형태로 현재 환경에서 GPU 사용 가능 여부를 동적으로 체크합니다.

텐서를 GPU로 보내기: tensor_gpu = tensor.to("cuda") 또는 tensor.to(device)

텐서를 CPU로 가져오기: tensor_cpu = tensor_gpu.to("cpu") (GPU 텐서는 바로 NumPy로 변환할 수 없으므로, 반드시 .to("cpu")를 거쳐야 numpy() 변환이 가능합니다.)

주의: 디바이스 불일치 에러(RuntimeError)와 해결법

에러 발생 원인: 연산을 수행하려는 두 개 이상의 텐서(예: tensor_a와 tensor_b) 중 하나는 CPU에 있고 다른 하나는 GPU(cuda:0)에 있을 때 사칙연산이나 행렬 곱셈을 시도하면 발생합니다.

대표 에러 메시지:

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

해결 방법: 연산 전 두 텐서가 동일한 장치에 위치하도록 .to() 메서드를 사용하여 장치를 일치시켜야 합니다.

# 예시: GPU에 있는 텐서와 CPU에 있는 텐서를 더할 때
a = torch.tensor([1.0, 2.0]).to("cuda")
b = torch.tensor([3.0, 4.0]).to("cpu")  # 또는 기본값 cpu

# b를 cuda로 일치시키거나, a를 cpu로 내려야 에러가 안 남
b = b.to(a.device)
result = a + b

특수한 텐서 생성 함수

1) 랜덤한 값으로 만들기

torch.rand() (균등 분포): 0과 1 사이의 균등 분포를 따르는 난수 텐서를 생성합니다.

torch.randn() (정규 분포): 평균 0, 표준편차 1인 표준 정규 분포를 따르는 난수를 생성합니다. 주로 가중치 초기화(Weight Initialization)에 필수적으로 사용됩니다.

특정한 값으로 채우기

torch.zeros(): 모든 값이 0인 텐서를 생성합니다.

torch.ones(): 모든 값이 1인 텐서를 생성합니다.

torch.full((shape), value): 사용자가 지정한 상수로 텐서를 채웁니다.

데이터 타입 변환 및 NumPy 연동

타입 변환 방법

권장 방식 (.to()): tensor.to(torch.float32) 또는 tensor.to(dtype=torch.float32, device="cuda")

간편 메서드: .float(), .int(), .long(), .double() 등

주의 사항: 행렬 곱셈(torch.matmul) 시 연산하려는 두 텐서의 데이터 타입(dtype)이 다르면 RuntimeError가 발생하므로 반드시 .to()로 타입을 일치시켜야 합니다.

PyTorch Tensor ⇄ NumPy 배열 변환

tensor.numpy(): 텐서와 메모리를 공유합니다. 원본 텐서가 바뀌면 NumPy 배열도 함께 바뀝니다. (단, CPU 텐서에서만 작동하며, GPU 텐서는 반드시 .to("cpu").numpy()를 거쳐야 함)

np.array(tensor): 데이터를 복사(Copy)하여 독립된 새로운 NumPy 배열을 생성합니다.

텐서 연산 및 브로드캐스팅 (Broadcasting)

텐서 연산 종류

요소별 연산 (Element-wise): a + b, torch.add(a, b), a * b (같은 위치의 원소끼리 연산)

행렬 곱셈 (Matrix Multiplication): torch.matmul(a, b) 또는 a @ b (앞 행렬의 열 개수와 뒤 행렬의 행 개수가 일치해야 함)

브로드캐스팅(Broadcasting) 원리

크기가 다른 두 텐서 간의 연산이 필요할 때, 작은 쪽의 차원을 자동으로 확장해 셰이프를 맞춘 뒤 연산하는 기능입니다.

작동 규칙: 뒤쪽(오른쪽) 차원에서부터 차례대로 비교했을 때,

두 차원의 크기가 완전히 같거나

둘 중 하나의 크기가 1인 경우

자동으로 크기를 늘려 연산을 수행합니다.

7. 공식 문서 및 참고 링크

PyTorch Official Documentation - torch.Tensor

PyTorch Official Documentation - CUDA semantics (Device Management)

NumPy Official Documentation - Broadcasting

