# 2025 1-2 어프렌티스 프로젝트 실습 포트폴리오

## 개요

본 저장소는 2025년 1-2 어프렌티스 프로젝트 실습 과정에서 수행한 머신러닝 기초 실습 결과물을 정리한 포트폴리오임. NumPy를 활용한 수치 연산부터 경사하강법, 선형회귀, 모델 평가, 정규화에 이르기까지 머신러닝 학습의 기본 흐름을 단계적으로 다룸. 모든 실습은 Jupyter Notebook 환경에서 진행하였으며, scikit-learn 및 scipy 라이브러리와 직접 구현한 코드를 비교하는 방식으로 학습 효과를 높임.

## 사용 기술

- **언어**: Python 3
- **라이브러리**: NumPy, pandas, matplotlib, scikit-learn, scipy
- **개발 환경**: Jupyter Notebook

## 실습 구성

### 1. [lec02_numpy_for_student.ipynb](lec02_numpy_for_student.ipynb) — NumPy 기초

- 배열 생성(`array`, `arange`, `linspace`, `zeros`, `ones`, `eye`, 시드 기반 난수 생성)과 reshape, 기본 연산(elementwise/matrix), 스택(`hstack`, `vstack`)을 실습함.
- 얕은 복사(view)와 깊은 복사(copy)의 메모리 공유 차이를 `np.shares_memory`로 검증함.
- Boolean mask, `np.where`를 활용한 조건부 인덱싱 및 값 치환을 다룸.
- Python 리스트 반복문과 NumPy `dot` 연산의 성능 차이를 1e1~1e6 데이터 규모로 비교한 그래프를 작성하여, NumPy의 벡터화 연산 이점을 정량적으로 확인함.

### 2. [lec05_1_gradient_decent_for_student.ipynb](lec05_1_gradient_decent_for_student.ipynb) — 경사하강법

- 2차 목적함수 `f(x) = 6x² + 9x + 3`의 최솟값을 `scipy.optimize`와 직접 구현한 경사하강법으로 각각 탐색함.
- 학습률(`alpha`), 허용오차(`tol`), 최대 반복 횟수(`max_iter`)를 인자로 받는 범용 `minimize` 함수를 구현함.
- 6차 다항함수 `x⁶ - 5x⁴ + 7x² - 3`을 통해 초기값에 따라 지역 최솟값(local minima)에 빠지는 문제를 시각적으로 확인함.

### 3. [lec05_2_linear_regression_univariate_for_student.ipynb](lec05_2_linear_regression_univariate_for_student.ipynb) — 단변량 선형회귀

- 10개 샘플 데이터에 대해 단변량 선형회귀를 세 가지 방식으로 구현하고 결과를 비교함.
  - scikit-learn `LinearRegression` 활용
  - `scipy.optimize.minimize`로 MSE 손실 최소화
  - MSE 손실에 대한 기울기(`grad_w`, `grad_b`)를 직접 유도하여 경사하강법으로 학습
- 회귀 직선을 산점도와 함께 시각화하여 각 방식의 수렴 결과가 일치함을 확인함.

### 4. [lec05_3_linear_regression_multivariate_for_student.ipynb](lec05_3_linear_regression_multivariate_for_student.ipynb) — 다변량 선형회귀

- Boston House 데이터셋을 pickle로 로드하고 pandas DataFrame으로 변환하여 특성을 분석함.
- `train_test_split`으로 학습/테스트 데이터를 분할하고, z-score 정규화로 전처리를 수행함.
- scikit-learn `LinearRegression`과 `scipy.optimize.minimize` 두 가지 방식으로 다변량 회귀를 학습하고, 테스트셋 MSE를 비교함.

### 5. [lec07_1_r2_for_student.ipynb](lec07_1_r2_for_student.ipynb) — 결정계수(R²) 구현

- R² = 1 − SS_res / SS_tot 공식을 직접 구현한 `my_r2_score` 함수와 `sklearn.metrics.r2_score`의 결과가 일치함을 검증함.
- 평가지표를 단순히 사용하는 단계를 넘어 내부 동작을 이해하는 것을 목표로 함.

### 6. [lec07_regularization_v2_for_student.ipynb](lec07_regularization_v2_for_student.ipynb) — 정규화(Ridge/Lasso)

- `0.5x² + x + 2`에 가우시안 노이즈를 추가한 합성 데이터셋(150개)을 생성함.
- `PolynomialFeatures`와 `StandardScaler`를 `make_pipeline`으로 결합하여 고차 다항회귀 모델을 구성함.
- MLE(일반 선형회귀), Ridge(L2), Lasso(L1) 세 가지 모델의 예측 곡선과 계수를 비교하여, 정규화가 과적합을 억제하고 계수를 축소(특히 Lasso는 0으로 만듦)하는 효과를 확인함.

## 학습 성과

- 라이브러리에 의존하지 않고 핵심 알고리즘(경사하강법, MSE 손실, R² 계산)을 직접 구현함으로써 머신러닝의 수학적 기반을 체계적으로 이해함.
- scikit-learn / scipy의 결과와 자체 구현 결과를 교차 검증하는 습관을 들여, 모델 동작을 신뢰성 있게 검증하는 방법을 익힘.
- 데이터 시각화를 통해 모델 학습 과정과 결과를 직관적으로 분석하는 역량을 함양함.
