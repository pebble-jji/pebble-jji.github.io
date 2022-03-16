---
layout : single
title : "3/15 (화) ML day 1" 
---

###  **Chapter 01 - 3 ~ 02 - 1**
# 생선 분류 모델 (feat. k-최근접 이웃)
기능 : 데이터를 넣으면 도미인지 아닌지 판단해줌

### 도미 데이터는 어떻게 생겼나

```
# 도미 길이
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

# 도미 무게
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

import matplotlib.pyplot as plt

plt.scatter(bream_length,bream_weight)

plt.title('Bream')

plt.xlabel('length')
plt.ylabel('weight')

plt.show()
```

### 빙어 데이터는 어떻게 생겼나

```
# 빙어 길이
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

# 빙어 무게
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(smelt_length,smelt_weight)

plt.title('Smelt')

plt.xlabel('length')
plt.ylabel('weight')

plt.show()
```

### 두 데이터 같이 보기

```
plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)

plt.title('Fish')

plt.xlabel('length')
plt.ylabel('weight')

plt.show()
```

## k-Nearest Neighbors : 도미와 빙어 구분하기

도미 데이터와 빙어 데이터를 합쳐주자!

```
# 도미 + 빙어 = 전체 리스트
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# Comprehension으로 각 물고기의 길이와 무게 대응시키기
fish_data = [[l,w] for l, w in zip(length,weight)]
```

얘는 답지보고 공부하는 애니까 답지 만들어 주자 - 도미는 1 빙어는 0으로 !

```
fish_target = [1] * 35 + [0] * 14
```

사이킷런에서 kNeighborsClassifier 불러오쟈

```
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier() # KN 뭐시기 객체 가져오라

kn.fit(fish_data,fish_target) # 적합!

print(kn.score(fish_data,fish_target)) # 이 때 1이 나오는 이유는 ?
```

내일 또 공부하기로 해
