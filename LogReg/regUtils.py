from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression

#  define varios tipos de datos personalizados en Python utilizando la biblioteca NumPy
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]] # puede ser o un XY o un solo array
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Función recibe un modelo de regresión logística como entrada y devuelve los parámetros del modelo: coeficientes e intercepción"""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Actualiza el modelo con los nuevos parámetros. los coeficientes b1,b2... y la intercep b0"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model



def set_initial_params(model: LogisticRegression):
    """Establece los parámetros iniciales del modelo de regresión logística como ceros,
     necesarios ya que los parámetros del modelo no se inicializan hasta que se llama a la función fit
    """
    n_classes = 2
    n_features = 39  # Número de características en el dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))





    