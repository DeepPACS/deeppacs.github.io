# Sequence-to-Sequence Model (Seq2Seq)

Given an input sequence:

$$
\mathbf{X} = (x_1, x_2, \dots, x_T)
$$

and target output sequence:

$$
\mathbf{Y} = (y_1, y_2, \dots, y_{T'})
$$

---

## 1. Encoder

The encoder computes hidden states:

$$
h_t = f_{\text{enc}}(x_t, h_{t-1})
$$

At the end of encoding, it produces:

- Fixed vector: $$ \mathbf{c} = h_T \quad \text{(basic)} $$
- Or sequence: $$ (h_1, \dots, h_T) \quad \text{(with attention)} $$

---

## 2. Decoder

The decoder computes:

$$
s_t = f_{\text{dec}}(y_{t-1}, s_{t-1}, \mathbf{c})
$$

$$
\hat{y}_t = \text{softmax}(W_o s_t + b_o)
$$

---

## 3. Training Objective

The training maximizes the log-likelihood of the output sequence:

$$
\mathcal{L} = \sum_{t=1}^{T'} \log p(y_t \mid y_{<t}, \mathbf{X})
$$

With:

$$
p(y_t \mid y_{<t}, \mathbf{X}) = \text{softmax}(W_o s_t + b_o)
$$

---

## 4. Attention Mechanism (Optional)

The context vector at each decoding step is:

$$
\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i} h_i
$$

where attention weights are:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}, \quad e_{t,i} = \text{score}(s_{t-1}, h_i)
$$

Then the decoder becomes:

$$
s_t = f_{\text{dec}}(y_{t-1}, s_{t-1}, \mathbf{c}_t)
$$
