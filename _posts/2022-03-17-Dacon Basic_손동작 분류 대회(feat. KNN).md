---
layout : single
title : "2022-03-17-Dacon Basic_손동작 분류 대회(feat. KNN)" 
categories : 
  -dacon_b
---

# KNN을 활용한 손동작 분류모델

머신러닝 3일차인 저는 그동안 배운거라곤 KNN 밖에 없는데  데이콘에 코드 제출을 의의로 하고 일단 배운대로 해보기로 했습니당

다행히도 배운 게 분류모델이라서 다른 것을 알아보거나 하지 않아도 돼서 다행이었어욤..

## Import Modules

필요한 라이브러리들을 불러옵니다.


```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import csv
```

## Load Data

데이터를 불러옵니다. 저는 구글 드라이브에 자료를 올려두고 로드했습니다.

```python
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train.csv')

train = train.drop('id', axis = 1)

test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/test.csv')

test = test.drop('id', axis = 1)
```

## EDA
간단한 EDA를 진행합니다.  

EDA의 다수의 코드는 데이콘 내의 코드를 인용했습니다.  

출처 : 데이터 분석 입문자를 위한 데이터 살펴보기  
[https://dacon.io/competitions/official/235876/codeshare/4608?page=1&dtype=recent]

### 결측치 확인

```python
def check_missing_col(dataframe):
    missing_col = []
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 컬럼은: {col}입니다')
            print(f'해당 컬럼에 총 {missing_values}개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if counted_missing_col == 0:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)
```

결측치가 없습니다.

### describe 함수를 이용하여 기초통계량 보기

```python
train.describe()
```

거의 모든 센서들의 평균값은 0에 가까운 것을 볼 수 있고, 제1사분위수와 제3사분위수에 비해 최대값과 최소값이 차이가 많이 난다는 것을 볼 수 있습니다.  특이값이 많을 것으로 예상되네요. 또, 센서들이 너무 많아서 target값에 영향을 주지 않는 센서는 제외할 필요가 있어보입니다.

```python
feature = train.columns

plt.figure(figsize=(20,60))

for i in range(len(feature)):
    plt.subplot(11,3,i+1)
    plt.title(feature[i])
    plt.boxplot(train[feature[i]])

plt.show()
```

IQR 외부에 값들이 많은 것을 보아 특이값이 많습니다.  

따라서 특이값을 95분위수와 5분위수로 윈저라이징하고, PCA를 실시하겠습니다.

### 윈저라이징

```python
for i in range(1,len(train.columns)) :

  for j in range(0,len(train[f'sensor_{i}'])) : 

    if train[f'sensor_{i}'][j] > float(train[f'sensor_{i}'].quantile(0.95,interpolation='nearest')) :
      train[f'sensor_{i}'][j] = float(train[f'sensor_{i}'].quantile(0.95,interpolation='nearest'))
    
    elif train[f'sensor_{i}'][j] < float(train[f'sensor_{i}'].quantile(0.05,interpolation='nearest')) :
      train[f'sensor_{i}'][j] = float(train[f'sensor_{i}'].quantile(0.05,interpolation='nearest'))

    else : 
      continue
```

#### describe 함수를 이용하여 기초통계량 보기

```python
train.describe()
```

최대 최소 값이 분위수들로 잘 바뀐 것을 볼 수 있습니다.


### PCA : 주성분 분석

#### 표준화

사이킷런의 StandardScaler를 이용해서 스케일링을 해보겠습니다.

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()


ss.fit(train.drop('target', axis = 1))

train_scaled = ss.transform(train.drop('target', axis = 1))

train_scaled = pd.DataFrame(train_scaled, columns=train.columns[0:32])

ssf = StandardScaler()

ssf.fit(test)

test_scaled = ssf.transform(test)

test_scaled = pd.DataFrame(test_scaled, columns=test.columns)
```

### 설명력 시각화 
주성분별 설명력을 보고 주성분의 개수를 결정하겠습니다.

```python
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(train_scaled)

exp_var = pca.explained_variance_ratio_

px.histogram(x = range(1,exp_var.shape[0]+1) , y = exp_var,labels={"x": "# Components", "y": "Explained Variance"})

pca = PCA(n_components=24)

printcipalComponents = pca.fit_transform(train_scaled)

principalDf = pd.DataFrame(data=printcipalComponents, columns = train.columns[0:24])

print(f'24개 : {sum(pca.explained_variance_ratio_)}')

pca = PCA(n_components=26)

printcipalComponents = pca.fit_transform(train_scaled)

principalDf = pd.DataFrame(data=printcipalComponents, columns = train.columns[0:26])

print(f'26개 : {sum(pca.explained_variance_ratio_)}')

pca = PCA(n_components=29)

printcipalComponents = pca.fit_transform(train_scaled)

principalDf = pd.DataFrame(data=printcipalComponents, columns = train.columns[0:29])

print(f'29개 : {sum(pca.explained_variance_ratio_)}')
```

주성분을 24개로 줄이면 약 90%의 분산이, 26개는 약 94% , 29개로 줄이면 약 98% 설명 가능합니다.  

주성분 개수를 26개로 분석을 진행하겠습니다.

```python
pca = PCA(n_components=26)

principalComponents = pca.fit_transform(train_scaled)

principalDf = pd.DataFrame(data=principalComponents, columns = train.columns[0:26])

principalDf
```

train 세트에서 센서를 26개로 줄이겠습니다.

```python
for i in range(26,32) :
  del train_scaled[f'sensor_{i}']
  del test_scaled[f'sensor_{i}']
```

KNN 모델을 적용시킵니다.

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(train_scaled,train['target'])

print(kn.predict(test_scaled))
```

  예측값을 csv 파일로 저장한다.

```python
import csv

id = list(range(1,len(test)+1))
target = list(kn.predict(test))

with open('/content/sample_data/answer.csv','w',newline='') as f :
  writer = csv.writer(f)
  writer.writerow(id)
  writer.writerow(target)
```

다시 코드를 보면서 생각해보니까 26개가 무조건 1~26번째 센서로만 생각해가지고 잘못된 결과를 도출했던 것 같습니다.
다음에는 이런 점을 조심하는 것이 좋겠습니다ㅠ
