import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.size'] = 14

from project_directories import raw_data_dir


if __name__ == '__main__':

    # Load or download the dataset as a dataframe
    df_url = 'https://huggingface.co/datasets/inria-soda/tabular-benchmark/raw/main/reg_num/cpu_act.csv'

    if os.path.isfile(raw_data_dir+'cpu_act.csv'):
        df = pd.read_csv(raw_data_dir+'cpu_act.csv', index_col=0)
    else:
        df = pd.read_csv(df_url, index_col=0)
        df.to_csv(raw_data_dir+'cpu_act.csv', index=True)

    # print(df)

    # Obtain data matrix and values
    data = df.to_numpy(dtype=np.float32)
    # print(data)
    A = data[:, :-1]
    b = data[:, -1].reshape(-1, 1)

    # Prepend a column of ones to A to act as a bias
    m, n = A.shape
    A = np.concatenate((np.ones((m, 1), dtype=np.float32), A), axis=1)
    # print(A)

    # Solve Ax = b using the SVD to compute the pseud-inverse
    U, Sig, VT = np.linalg.svd(A, full_matrices=False)
    x = VT.T @ np.linalg.inv(np.diag(Sig)) @ U.T @ b

    # Plot the actual cpu times in user mode vs the ones predicted with linear regression
    fig, ax = plt.subplots()

    ax.plot(b, 'k-', lw=2, label='% CPU time in user mode')
    ax.plot(A@x, 'b--', lw=1.5, ms=6, alpha=.8, label='Regression values')
    ax.set(xlabel='user', ylabel='cpu user time')
    ax.legend()
    plt.show()

    # We can also sort the cpu times and re-plot, to see if the general trend is captured
    sort_idxs = np.argsort(b[:,0])
    b = b[sort_idxs]

    fig, ax = plt.subplots()

    ax.plot(b, 'k-', lw=2, label='% CPU time in user mode')
    ax.plot(A[sort_idxs, :]@x, 'b--', lw=1.5, ms=6, alpha=.8, label='Regression values')
    ax.set(xlabel='user', ylabel='cpu user time')
    ax.legend()
    plt.show()
