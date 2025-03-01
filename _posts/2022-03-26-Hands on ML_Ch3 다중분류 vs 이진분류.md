---
layout : single
title : "2022-03-26-Hands on ML_Ch3 다중분류 vs 이진분류"
categories : 
  - homl 
---
## 4. 다중 분류
이진 분류기 vs 다항 분류기
- 이진 분류기 : 2개의 클래스 구별  
  + OvR(One vs Rest) : 제일 결정점수 높은 것(OvA)
  + OvO(One vs One) : 2개 중 결정점수 높은 것 택함.
- 다항 분류
  + 다중 분류기  
  ex) SGD , Naive, Random Forest
  + 이진 분류기 여러 개를 결합  
    영향력 : 훈련세트 개수 작은 것 > 모델 여러 개  
     * OvR :  레이블 개수만큼
     * OvO : 작은 훈련세트의 모델 여러 개 -> 빠름

그림들의 출처는 오른쪽 블로그에 있다. [아무튼 워라밸](https://hleecaster.com/ml-svm-concept/)

Support Vector Machine(SVM) : 서포트 벡터 머신

<img src="https://hleecaster.com/wp-content/uploads/2020/01/svm01.png" style="zoom:33%;" >

SVM 분류는 **두 집단의 경계선을 알려주는 모델**이다. Feature가 2개라면 위와 같고, Feature가 3개라면 3개의 축이 생기므로 입체공간에서 분류가 이뤄지게 된다. 따라서 그림은 아래와 같다.

<img src="https://i0.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm02.png?fit=1024%2C852" alt="img" style="zoom:40%;" />

또. 이 때의 결정 경계는 **선이 아닌 평면**이 된다. 이 이후는 사람이 생각할 수 없는 평면, 즉, **초평면**이라고 할 수 있다.

이 때 샘플과 결정 경계가 멀면 멀수록 분류를 명확하게 하는 것이다. 따라서 결정 경계와의 최근접한 샘플과의 거리를 **마진(Margin)**이라고 한다.

<img src="https://i0.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm04.png?fit=1024%2C768" alt="img" style="zoom:40%;" />

따라서 **최적의 결정 경계**는 **마진이 최대화되는 지점**이다.

<img src="https://i1.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm06.png?fit=1024%2C768" alt="img" style="zoom:40%;" />

마진 내에 존재하는 이상치가 많으면 소프트 마진, 이상치가 들어오는 것을 허용하지 않을수록 하드 마진이라고 한다.

*(대법관인데 좀 말랑말랑하면 소프트 마진, 대법관이 포청천이면 하드마진인거임)*

이 때 적정선을 잘 찾아야 한다. 너무 소프트하면 과소적합, 너무 하드하면 과대적합이 될 수 있다.

```python
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X_train,y_train)
    svc.predict([some_digit])
```

```
array([5], dtype=uint8)
```

결정점수가 5에서 가장 큰 것을 볼 수 있다.

```python
some_digit_scores = svc.decision_function([some_digit])

np.argmax(some_digit_scores)

svc.classes_

print(some_digit_scores,svc.classes_[5])
```

```python
[[ 1.72501977  2.72809088  7.2510018   8.3076379  -0.31087254  9.3132482
   1.70975103  2.76765202  6.23049537  4.84771048]]
5
```

SVC를 10개 만들어 0~9 중 어떤 숫자를 나타내는지 말함.

```python
from sklearn.multiclass import OneVsRestClassifier
ovc = OneVsRestClassifier(SVC())
ovc.fit(X_train,y_train)
ovc.predict([some_digit])
len(ovc.estimators_)
```

```python
10
```

그냥 다중분류기도 잘 예측하는게 보임

```python
from sklearn.linear_model import SGDClassifier
sgc.fit(X_train,y_train)
sgc.predict([some_digit])

sgc.decision_function([some_digit])
```

```python
array([[-31893.03095419, -34419.69069632,  -9530.63950739,
          1823.73154031, -22320.14822878,  -1385.80478895,
        -26188.91070951, -16147.51323997,  -4604.35491274,
        -12050.767298  ]])
```

```
cross_val_score(sgc,X_train,y_train,cv=3,scoring = 'accuracy')
```

```python
array([0.87365, 0.85835, 0.8689 ])
```

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgc,X_train_scaled,y_train,cv = 3, scoring = 'accuracy')
```

```
array([0.8983, 0.891 , 0.9018])
```

표준화까지 하니까 꽤 정확도 높아짐.

## 5. 에러분석 for 성능향상

1. 오차행렬

   ```python
   y_train_pred = cross_val_predict(sgc,X_train_scaled,y_train,cv=3)
   conf_mx = confusion_matrix(y_train,y_train_pred)
   conf_mx
   ```

   ```python
   array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
          [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
          [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
          [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
          [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
          [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
          [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
          [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
          [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
          [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]])
   ```

   ```python
   plt.matshow(conf_mx,cmap = plt.cm.gray)
   plt.show()
   ```

   ![img](data:image/png;base64,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)

   
   
   ```python
   row_sums = conf_mx.sum(axis = 1, keepdims = True)
   norm_conf_mx = conf_mx / row_sums
   ```
   
   ```python
   np.fill_diagonal(norm_conf_mx,0)
   plt.matshow(norm_conf_mx,cmap = plt.cm.gray)
   plt.show()
   ```
   
   

![img](data:image/png;base64,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)

```python
cl_a, cl_b = 3,5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
```

```python
# 숫자 그림을 위한 추가 함수
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # 
    n_rows = (len(instances) - 1) // images_per_row + 1

    # 필요하면 그리드 끝을 채우기 위해 빈 이미지를 추가합니다:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # 배열의 크기를 바꾸어 28×28 이미지를 담은 그리드로 구성합니다:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # 축 0(이미지 그리드의 수직축)과 2(이미지의 수직축)를 합치고 축 1과 3(두 수평축)을 합칩니다. 
    # 먼저 transpose()를 사용해 결합하려는 축을 옆으로 이동한 다음 합칩니다:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # 하나의 큰 이미지를 얻었으므로 출력하면 됩니다:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
```

```python
plt.figure(figsize = (8,8))
plt.subplot(221); plot_digits(X_aa[:25],images_per_row = 5)
plt.subplot(222); plot_digits(X_ab[:25],images_per_row = 5)
plt.subplot(223); plot_digits(X_ba[:25],images_per_row = 5)
plt.subplot(224); plot_digits(X_bb[:25],images_per_row = 5)
plt.show()
```

![img](https://lh3.googleusercontent.com/tuh0WGPuLbAPk2RvrWdJr32QK5mBQpP1GWHBhG0exzApvBqMV2UQBV6RY2RDl-Fn-nG6W3pPrxEvfZ0BFgSXkGI2Tl3OZQehP8wEd5jbJF528J515V_Feh6E-5-3O3F8IsQPvHRSLaX0YkV0_evidU0hGTTAsGhM9K5rEHMoLoYDzwVDy5PM0ZL724ecVCr6rLy2ktLMm0ajyZk-vrweGZVsn_ZVjCRQ53M9H4He4G50hHbom5IE7VbhIVTQvMH4ZFw1bhJrS2_JRZBbu5VEBXUDMmgpe241EF1oGYHM2lbAy6Ddy9GhlGA9b5IU2icC6aTS1OD5izrtmC06mhIrnCgId_I5gVQG4l9gJlWETkB9LB3_qBRBslWcpFOk9IIQl46pIicvnK-EAUmR3vaEN0ozE4Te15XgJYKgvD6Grsdq3yT7dJ05EqnN8O6XzsDomCvKddDoJ4Tcn8qVSVQKsgBzBOU4yfwCXo0Uaa67tmUcQ-q4ARiKQjF4LZudAxWMmSzbeF73jRjuAqqF50M10nt0dt3GbSND8JplbFH4RT-b8N4Jr7eowIuPtzJEfSjrOMingsmWsxvOl2zn__-rretd9fSQ3hLQutnXRJf7N29Fnksl6Dt0MmN-yfKicmsdYzz1gIOkT5BL0OLYEtYgtDVgPsN5U3tyThcnSmSTNweuibPEfuclE4GZVzaBfokkWKLJphbBAw-6W_w3MXAOs0tlN-bg9ViWHA2E0W71Wuxa_5hWYxOMlpysLDsjSA=w454-h448-no?authuser=0)
