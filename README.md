# HRAGRU Experiment Results

Below is the table of results for the baseline and ablation experiments.

```latex
\begin{table}[t]
    \centering
    \caption{The index of the baseline and ablation experiments}
    \label{tab:my_label}
    \begin{threeparttable}
    \setlength{\tabcolsep}{4.8pt}
    \begin{tabular}{lllll}
    \hline
    \makebox[0.14\textwidth][l]{\textbf{model}}        & \textbf{MSE} & \textbf{ACC} & \textbf{Recall} & \textbf{F1}\\
    \hline
    CNN     & 0.1618 &  0.5073 & 0.0961 & 0.1001\\
    LSTM & 0.0850 & 0.6378 & 0.5479 & 0.6137\\
    GRU & 0.0775 &  0.6450 & 0.6001 & 0.6225\\
    NGCU & 0.1361 &  0.4023 & 0.6193 & 0.5102\\
    TRA & 0.1166 &  0.4350 & 0.2287 & 0.2669\\ 
    Transformer & 0.1193  & 0.4672 & 0.1425 & 0.1968\\
    \hline
    AlphaStock & 0.1122 &  0.4753 & 0.0556 & 0.2120\\ 
    DeepTrader & 0.1350 &  0.5048 & 0.3494 & 0.3992\\ 
    CTTS & 0.0746 &  0.6597 & 0.6912 & 0.6839\\ 
    FactorVAE & 0.0738 &  0.6771 & 0.6802 & 0.6793\\
    Mamba & 0.0755 &  0.6730 & 0.5425 & 0.6728\\
    xLSTM & 0.0760 &  0.6714 & 0.5601 & 0.6711\\
    PatchTST & 0.0734 &  0.6766 & 0.6941 & 0.6842\\
    Logistic-CNN-BiLSTM-att & 0.0798 &  0.6652 & 0.6810 & 0.6695\\
    \hline
    HRAGRU-(b)+BERT & 0.0728 &  0.6775 & 0.6959 & 0.6864\\
    HRAGRU-(c) & 0.0730 &  0.6782 & 0.6943 & 0.6850\\
    HRAGRU-(*) & 0.0749 &  0.6716 & 0.6895 & 0.6803\\
    HRAGRU-(d)+GRU & 0.0743 &  0.6735 & 0.6901 & 0.6821\\
    HRAGRU & \pmb{0.0726} &  \pmb{0.6798} & \pmb{0.7028} & \pmb{0.6876}\\
    \hline
    \end{tabular}
    \end{threeparttable}
\end{table}
