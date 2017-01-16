import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.array(np.linspace(start=-3, stop=3, num=1001, dtype=np.float))
    y_logit = np.log(1 + np.exp(-x))/math.log(2)
    y_01 = x < 0
    y_hinge = 1.0 -x
    y_hinge[y_hinge<0] = 0
    y_square = (x-1)**2
    y_exponential = np.exp(-x)

    plt.plot(x, y_logit, 'y-', label='Logistic Loss', linewidth=2)
    plt.plot(x, y_01, 'b-', label='0/1 Loss', linewidth=2)
    plt.plot(x, y_hinge, 'r-', label='Hinge Loss', linewidth=2)
    plt.plot(x, y_square, 'k-', label='Square Loss', linewidth=2)
    plt.plot(x, y_exponential, 'g-', label='Exponential Loss', linewidth=2)
    # plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    # plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
    # plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    # plt.plot(x, y_square, 'k-', label='Square Loss', linewidth=2)
    # plt.plot(x, y_exponential, 'm-', label='Exponential Loss', linewidth=2)
    plt.grid()
    plt.xlim((-2, 2))
    plt.ylim((0, 9))
    plt.legend(loc='upper right')
    plt.savefig('1.png')
    plt.show()