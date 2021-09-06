"""Implementação simplificada da descida do gradiente."""

from typing import List
from typing import Tuple

import numpy as np
import numpy.typing as npt

from sklearn import metrics


# mais detalhes sobre a notação de tipos numpy:
# https://numpy.org/devdocs/reference/typing.html


def descida_gradiente(
    X: npt.NDArray, y: npt.NDArray, lr: float = 0.05, epoch: int = 10
) -> Tuple[float, float, List[Tuple[float, float]], List[float]]:
    """
    Descida do gradiente com regressão linear para feature única.

    Parameters
    ----------
    X : npt.NDArray (numpy ndarray)
        amostras de entrada.
    y : npt.NDArray (numpy ndarray)
        valores alvo.
    lr : float, optional
        taxa de aprendizado, by default 0.05.
    epoch : int, optional
        número limite de iterações da otimização, by default 10.

    Returns
    -------
    Tuple[float, float, List[Tuple[float, float]], float]
        intercepto, coeficiente angular, histórico da otimização e métrica de erro.
    """
    # parâmetros iniciais
    m = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)

    log, mse = [], []  # armazena o processo de aprendizagem
    N = len(X)  # número de amostras

    for _ in range(epoch):

        f = y - (m * X + b)

        # atualiza m and b
        m -= lr * (-2 * X.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)

        log.append((m, b))
        mse.append(metrics.mean_squared_error(y, (m * X + b)))

    return m, b, log, mse
