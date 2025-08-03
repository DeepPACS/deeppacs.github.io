\section*{I. Input: Chuỗi Phoneme}

Giả sử chuỗi đầu vào là một dãy các phoneme:

\[
\mathbf{x} = (x_1, x_2, \dots, x_T)
\]

Mỗi $x_i$ là một phoneme, được ánh xạ thành vector nhúng thông qua hàm embedding:

\[
\mathbf{e}_i = \text{Embed}(x_i) \in \mathbb{R}^d
\]

\section*{II. Output: Mel-Spectrogram}

Chuỗi đầu ra là dãy các vector đặc trưng mel-spectrogram:

\[
\mathbf{y} = (y_1, y_2, \dots, y_{T'}), \quad y_j \in \mathbb{R}^m
\]

\section*{III. Mô hình ánh xạ}

\subsection*{1. Mô hình Tacotron (Autoregressive, Attention-based)}

Hàm ánh xạ tổng thể:

\[
\mathbf{y} = f_\theta(\mathbf{x})
\]

\textbf{Encoder:}

\[
\mathbf{h} = (\mathbf{h}_1, \dots, \mathbf{h}_T) = \text{Encoder}(\mathbf{e}_1, \dots, \mathbf{e}_T)
\]

\textbf{Attention và Decoder:}

Tại mỗi bước thời gian $j$:

\[
\alpha_{j,i} = \text{Attention}(s_{j-1}, h_i), \quad c_j = \sum_{i=1}^T \alpha_{j,i} h_i
\]

Decoder tạo ra frame mel:

\[
y_j = \text{Decoder}(y_{j-1}, s_{j-1}, c_j)
\]

\subsection*{2. Mô hình FastSpeech (Non-autoregressive)}

\textbf{Bước 1:} Encode chuỗi phoneme:

\[
\mathbf{h} = \text{Encoder}(\mathbf{e}_1, \dots, \mathbf{e}_T)
\]

\textbf{Bước 2:} Dự đoán thời lượng của từng phoneme:

\[
d_i = \text{DurationPredictor}(h_i)
\]

\textbf{Bước 3:} Điều chỉnh độ dài chuỗi bằng Length Regulator:

\[
\tilde{h} = (\underbrace{h_1, \dots, h_1}_{d_1}, \dots, \underbrace{h_T, \dots, h_T}_{d_T})
\]

\textbf{Bước 4:} Decoder sinh chuỗi mel-spectrogram:

\[
y_j = \text{Decoder}(\tilde{h}_j)
\]

\section*{IV. Hàm mất mát}

Hàm mất mát chính là sai số giữa mel thật và mel dự đoán:

\[
\mathcal{L}_{\text{mel}} = \sum_{j=1}^{T'} \left\| y_j - \hat{y}_j \right\|_1
\]

Trong FastSpeech, tổng hàm mất mát thường là:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mel}} + \lambda_1 \mathcal{L}_{\text{dur}} + \lambda_2 \mathcal{L}_{\text{pitch}} + \lambda_3 \mathcal{L}_{\text{energy}}
\]

\section*{V. Tóm tắt pipeline}

\[
\mathbf{x} \xrightarrow{\text{Embedding}} \mathbf{e} \xrightarrow{\text{Encoder}} \mathbf{h} 
\xrightarrow{\text{Length Regulator}} \tilde{h} 
\xrightarrow{\text{Decoder}} \hat{\mathbf{y}} \approx \mathbf{y}
\]