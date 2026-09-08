# understanding_and_types_of_data

개요

데이터 분석의 출발점이 되는 데이터의 구성 요소(개체와 속성), 구조적 형태(정형/반정형/비정형), 그리고 통계 분석의 기초가 되는 속성별 유형(데이터 척도)을 명확하게 이해합니다.

데이터의 구성 요소 (Data Elements)

데이터는 크게 관찰 대상이 되는 개체와, 그 대상이 가지고 있는 특성인 속성으로 구성됩니다.

개체 (Entity): 분석하려는 관찰 대상 하나하나를 의미합니다.

속성 (Attribute): 개체가 가지고 있는 측정 가능한 특성이나 변수입니다.

# Pandas DataFrame 표현 예시
import pandas as pd

# 행(Index) = 개체(Entity) / 열(Columns) = 속성(Attribute)
data = {
    'Customer_ID': ['C001', 'C002', 'C003'],  # 개체 식별자
    'Age': [25, 34, 41],                    # 속성 1 (수치형)
    'Gender': ['F', 'M', 'F'],              # 속성 2 (범주형)
    'Purchase_Amount': [15000, 32000, 8900] # 속성 3 (수치형)
}
df = pd.DataFrame(data)

데이터의 구조적 유형 (Structural Data Types)

데이터가 저장되고 표현되는 방식과 스키마(Schema)의 유무에 따른 분류입니다.

① 정형 데이터 (Structured Data)

특징: 고정된 행과 열을 가진 테이블 구조로, 엄격한 데이터 타입과 스키마가 정의되어 있어 RDBMS(SQL) 및 데이터프레임 형태로 바로 활용이 가능합니다.

예시: Relational Database(MySQL, MariaDB), Excel, CSV 파일

② 반정형 데이터 (Semi-structured Data)

특징: 데이터 내부에 구조(메타데이터, 태그, 키-값)가 포함되어 있으나, 정형 데이터처럼 고정된 테이블 형태나 스키마를 강제하지 않는 데이터입니다.

예시: JSON, XML, HTML, Log 파일

③ 비정형 데이터 (Unstructured Data)

특징: 정해진 구조나 규칙이 없는 데이터로, 전체 데이터의 80% 이상을 차지합니다. 전통적인 SQL로는 조회하기 어려우며 AI/머신러닝 전처리가 필수적입니다.

예시: 이미지, 동영상, 음성 파일, 자유 형식 텍스트(이메일, SNS 게시글)

데이터의 속성별 유형 (Data Measurement Scales)

데이터 분석 및 통계 모델링 시 어떤 연산(사칙연산, 평균 산출 등)을 적용할 수 있는지 결정짓는 4단계 척도 분류법입니다.

정성적 데이터 / 질적 데이터 (Qualitative / Categorical Data)

숫자로 표현되어 있더라도 크기나 양을 나타내지 않고 상태나 범주를 구분하는 데이터입니다.

명목형 (Nominal Scale)

순서형 (Ordinal Scale)

정량적 데이터 / 양적 데이터 (Quantitative / Numerical Data)

수치로 측정 및 계산이 가능한 데이터입니다.

구간형 / 등간형 (Interval Scale)

비율형 (Ratio Scale)

공식 문서 및 참고 링크

Pandas Data Structures (Series & DataFrame Official Guide)

Scikit-Learn Preprocessing & Categorical Features

Khan Academy - Quantitative & Categorical Data Types

