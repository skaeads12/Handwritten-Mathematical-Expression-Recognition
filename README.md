# Handwritten-Mathematical-Expression-Recognition

|Version|Description|
|:---:|:---:|
|1.0.0|README.md 작성 및 업로드|

이 저장소는 [Improving Attention-Based Handwritten Mathematical Expression Recognition with Scale Augmentation and Drop Attention](https://arxiv.org/abs/2007.10092) 논문을 바탕으로 구현되었습니다.

# 요약

해당 모델은 손으로 작성된 수식을 인식하기 위한 모델입니다. Attention 기반의 Encoder-Decoder 모델을 사용하였고, 인코더와 디코더는 각각 ResNet18과 LSTM이 사용되었습니다. CROHME 2014, CROHME 2016 데이터셋을 통하여 학습되었습니다.

# Usage

##### 학습

```
mkdir trained_model/

python main.py \
  --task "train" \
  --data_dir "data/" \
  --image_dir "data/data_png_Training/" \
  --batch_size 32 \
  --sequence_length 64 \
  --save_dir "trained_model/" \
  --patience 5
```

##### 검증

```
python main.py \
  --task "eval" \
  --data_dir "data/" \
  --image_dir "data/data_png_Training/" \
  --batch_size 32 \
  --sequence_length 64 \
  --save_dir "./predictions.txt"
```

# 모델 구조

![math_expression](https://user-images.githubusercontent.com/45366231/174712910-a337d6d6-220b-44c9-8c63-132dd94a1d63.jpg)

> 각 모듈 아래에 모델 차원 변화의 이해를 돕기 위한 차원 수를 작성함  
> 논문에서 RNN과 LSTM 용어를 구분하지 않고 혼용하기 때문에 해당 README.md에서도 두 용어를 구분하지 않음 (RNN = LSTM)  

##### ENCODER: ResNet18 사용
##### DECODER: Attention Based Decoder 사용

# Decoder 설명

##### Attention Based Decoder는 아래 4개의 벡터에 의해 연산됨
1. Encoder(ResNet18)를 거쳐 나온 Feature F
2. Feature F의 절대 위치 정보를 담은 Positional Embedding Vector Q (Sin, Cos 사용)
3. LSTM의 Layer를 거쳐 나온 Hidden State H
4. Weighted Sum $\alpha$의 누적 합 S

##### 아래 수식을 통해 Weighted Sum을 구하고 RNN과 Linear G를 통해 토큰 하나를 예측 (\[EOS\] 토큰이 예측될 때까지 time step t에 대하여 반복)

$$
e = W_e(tanh(W_f(f) + W_q(q) + W_h(h) + W_s(s)))
$$

$$
\alpha = softmax(e)
$$

$$
c = \sum^{L-1}_{l=0}\alpha_l * f_l
$$

$$
p(y_t) = g(c_t, h_t)
$$

> \*: Element-wise product  
> L: Height * Width

##### 다음 Step t+1을 위해 아래 연산

$$
s = s + \alpha
$$

$$
c^{\prime} = \sum^{L-1}_{l=0}\alpha_l * (f_l + q_l)
$$

$$
h_{t+1} = RNN([E^d(y_{t}), c^{\prime}], h_{t})
$$

# EXPERIMENTS

TODO: Write this
