---
layout : single
title : "KNN Hyperparamter Tuning" 
---
## Chapter 02 데이터 다루기

### 1 | 훈련 세트와 테스트 세트

#### 지도학습 vs. 비지도학습

- 지도학습 : 타겟값(정답)을 가지고 알고리즘이 정답을 맞추는 것
  * 훈련데이터 : 훈련에 사용되는 데이터 -  fit 함수에 사용 
  = 입력(이전 챕터 fish_data) + 타겟(이전 챕터 fish_target)
    + 입력 : 여러 개의 샘플로 구성
    + 타겟 : 조건 부합여부로 이뤄진 리스트
  * 테스트세트 : 평가에 사용되는 데이터 - score 함수에 사용
    + 다른 데이터 사용 or 데이터 중 일부 사용
- 비지도학습 : 타겟값 없이 알고리즘이 정답을 맞추는 것

이 중 지금은 지도학습 중!

##### 데이터 세트 만들기

Chapter 1과 동일한 방식이니 **자세한 설명은 생략한다.**

```
# 길이
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

# 무게
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 무게와 길이를 대응시킨 리스트
fish_data = [[l,w] for l, w in zip(fish_length, fish_weight)]

# 타겟값
fish_target = [1] * 35 + [0] * 14
```
##### 훈련 세트 및 테스트 세트 만들기 - 기존 데이터에서 일부 떼옴 (feat. 슬라이싱)

```
# 훈련 세트

train_input = fish_data[:35]

train_target = fish_target[:35] # 도미만 떼올 것임

# 테스트 세트

test_input = fish_data[35:]

test_target = fish_target[35:] # 빙어만 떼올 것임
```

##### 모델 객체 생성

```
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(train_input,train_target)

print(kn.score(test_input, test_target))
```

결과가 0이 나와서 충격적이긴 한데 이유가 너무 명확하다.  

**이거 학부모 컴플레인감이다.**   

도미만 가르쳐놓고 학생한테 빙어 맞추라고 하면 맞추겠냐고.  


이게 바로 **샘플링 편향**이 주는 문제점이다.  

해결하려면 어떻게 하지?  

어떡하긴 둘 다 가르치면 되지.  


둘 다 가르치려면 샘플들을 섞으면 된다.   

넘파이를 이용해서 섞어보쟈.  


##### 넘파이 불러와서 랜덤으로 섞은 훈련 세트 만들기


```
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr.shape)
```

49개니까 인덱스도 49개로 만들면 된다

인덱스를 만들어 보자.

```
np.random.seed(42)

index = np.arange(49)

np.random.shuffle(index) # 0~48까지 랜덤으로 뽑음

```

인덱스를 이용하여 랜덤으로 섞은 훈련세트와 테스트세트를 만들자.

```
train_input = input_arr[index[:35]]

train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]

test_target = target_arr[index[35:]]
```

그림 그려서 제대로 했나 보자.

```
import matplotlib.pyplot as plt

plt.scatter(train_input[:,0],train_input[:,1],label = 'train')
plt.scatter(test_input[:,0],test_input[:,1], label = 'test')

plt.legend()
plt.xlabel('length')
plt.ylabel('weight')

plt.show()
```

##### 다시 적합 해보기

```
kn.fit(train_input,train_target)

print(kn.score(test_input, test_target))

kn.predict(test_input)
```

그래도 아까보다는 합리적이다.

### 2 | 데이터 전처리

#### 데이터 세트 만들기 (feat. numpy)



```
# 길이
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

# 무게
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


import numpy as np

np.column_stack((fish_length,fish_weight))
# stack : 튜플의 원소를 하나씩 꺼내서 대응시켜줌

fish_target = np.concatenate((np.ones(35),np.zeros(14))) # 얘도 튜플 써야 함
```

#### 훈련세트와 테스트세트 나누기 (feat.사이킷런)

##### train_test_split : 알잘딱깔센

```
from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target = train_test_split(fish_data, fish_target, random_state = 42, stratify = fish_target )
# random_state로 시드 지정 가능
# starify로 클래스 비율 지정 가능(편향 방지)
```

제대로 됐나 보자.


```

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(train_input,train_target)

print(kn.score(test_input, test_target))

```
