import numpy as np
import matplotlib.pyplot as plt

x = 0.5
alpha = np.linspace(-1, 3, 100)
f = lambda x : np.cos(np.pi * x)
grad = lambda x: -np.sin(np.pi * x)

eta_ls = [0.3, 0.5, 0.7, 0.9]

def plotter(x, alpha, f, grad, eta_ls):
    f_alpha = f(x - alpha * grad(x))
    f_x = f(x)
    plt.plot(alpha * np.pi, f_alpha, label='f(x - alpha * grad(x))')
    for eta in eta_ls:
        Armijo_sufficient_descent = f_x - eta * alpha * grad(x)**2
        plt.plot(alpha * np.pi, Armijo_sufficient_descent, label=f'eta={eta}')

    plt.title('Armijo Sufficient Descent')
    plt.legend()
    plt.xlabel('alpha')
    plt.grid()
    plt.savefig(f'Armijo_sufficient_descent.png')


plotter(x, alpha, f, grad, eta_ls)