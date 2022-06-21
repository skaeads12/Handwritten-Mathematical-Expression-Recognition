# Handwritten-Mathematical-Expression-Recognition

|Version|Description|
|:---:|:---:|
|1.0.0|Write and upload README.md|

This repository is implemented based on the [Improving Attention-Based Handwritten Mathematical Expression Recognition with Scale Augmentation and Drop Attention](https://arxiv.org/abs/2007.10092) paper.

# Abstract

This model is for recognizing handwritten formulas. The Encoder-Decoder model based on Attention was used, and ResNet18 and LSTM were used for the encoder and decoder,
respectively. It was learned through the CROHME 2014 and CROHME 2016 datasets.

# Usage

##### Training

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

##### Validation

```
python main.py \
  --task "eval" \
  --data_dir "data/" \
  --image_dir "data/data_png_Training/" \
  --batch_size 32 \
  --sequence_length 64 \
  --save_dir "./predictions.txt"
```

# Model Architecture

![math_expression](https://user-images.githubusercontent.com/45366231/174712910-a337d6d6-220b-44c9-8c63-132dd94a1d63.jpg)

> The number below each module is the number of dimensions to help you understand the model dimension changes  
> Since the paper uses RNN and LSTM terms without distinction, the corresponding README.md also does not distinguish between the two terms (RNN = LSTM)  

##### ENCODER: ResNet18 사용
##### DECODER: Attention Based Decoder 사용

# Decoder 설명

##### Attention Based Decoder is calculated by the following four vectors
1. Feature F via Encoder (ResNet18)
2. Position Embedding Vector Q containing the absolute position information of Feature F (using Sin, Cos)
3. Hidden State H through LSTM's Layer
4. Cumulative sum of Weighted Sum $\alpha$, S

##### Obtain Weighted Sum using the formula below and predict one token via RNN and Linear G (repeat for time step until the \[EOS\] token is predicted)

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

##### Operate below for next step

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
