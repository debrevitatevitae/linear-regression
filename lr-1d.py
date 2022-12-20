import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    np.random.seed(0)

    x = 2.
    a = np.linspace(0., 10., 20).reshape(-1, 1)  # reshape for svd later
    b = a * x

    # include noise in both a and b
    a_noisy = a + 1. * np.random.randn(*a.shape)
    b_noisy = a_noisy * x + 1. * np.random.randn(*b.shape)

    # plot the generating line and the noisy datapoints
    fig, ax = plt.subplots()
    ax.plot(a, b, 'b-', label='True line')
    ax.scatter(a_noisy, b_noisy, edgecolor='orange',
               facecolor='orange', alpha=.6, label='Data')
    ax.set(xlabel='measurements', ylabel='values', title='Data and true curve')
    ax.grid()
    ax.legend()
    plt.show(block=False)

    # compute the best least square approximant with the SVD
    u, sig, vt = np.linalg.svd(a, full_matrices=False)
    pseudo_inverse = vt.T @ np.diag(1 / sig) @ u.T
    x_mod = pseudo_inverse @ b_noisy

    # add the linear regression prediction
    ax.plot(a, x_mod * a, 'k--', label='LR model')
    ax.set(xlabel='measurements', ylabel='values',
           title='Data, true and LR curves')
    ax.legend()
    plt.show()
