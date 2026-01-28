"""
Utilidades para análisis de riesgos financieros.

Este módulo contiene funciones para calcular estadísticos descriptivos,
VaR (Value at Risk) usando diferentes metodologías, y análisis de riesgo sistémico.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import t
from arch import arch_model

# Constantes
DIAS_ANUALES = 252


def descriptive_stats(returns, rf_daily):
    """
    Calcula estadísticos descriptivos anualizados de una serie de retornos.
    
    Parámetros:
    -----------
    returns : pd.DataFrame
        DataFrame con series de retornos logarítmicos diarios
    rf_daily : float
        Tasa libre de riesgo diaria
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame con estadísticos: media anual, volatilidad anual, 
        asimetría, curtosis y Sharpe ratio
    """
    # Calcular estadísticos anualizados
    media_anual = returns.mean() * DIAS_ANUALES
    volatilidad_anual = returns.std() * np.sqrt(DIAS_ANUALES)
    rf_anual = rf_daily * DIAS_ANUALES
    
    # Calcular Sharpe ratio
    sharpe_ratio = (media_anual - rf_anual) / volatilidad_anual
    
    # Crear DataFrame con resultados
    stats = pd.DataFrame({
        "Media anual": media_anual,
        "Volatilidad anual": volatilidad_anual,
        "Asimetría": returns.skew(),
        "Curtosis": returns.kurtosis(),
        "Sharpe Ratio": sharpe_ratio
    })
    
    return stats


def _calcular_var_caviar(returns_array, beta, tau):
    """
    Función auxiliar para calcular VaR usando parámetros CAViaR.
    
    Parámetros:
    -----------
    returns_array : np.ndarray
        Array de retornos
    beta : np.ndarray
        Parámetros del modelo [beta0, beta1, beta2, beta3]
    tau : float
        Nivel de confianza (ej: 0.01 para VaR al 1%)
        
    Retorna:
    --------
    np.ndarray
        Serie de VaR estimado
    """
    num_observaciones = len(returns_array)
    var_series = np.zeros(num_observaciones)
    var_series[0] = np.quantile(returns_array, tau)
    
    for t in range(1, num_observaciones):
        retorno_anterior = returns_array[t - 1]
        var_series[t] = (
            beta[0]
            + beta[1] * var_series[t - 1]
            + beta[2] * max(retorno_anterior, 0)
            - beta[3] * max(-retorno_anterior, 0)
        )
    
    return var_series


def caviar_loss(beta, returns, tau):
    """
    Función de pérdida para el modelo CAViaR (Conditional Autoregressive VaR).
    
    Parámetros:
    -----------
    beta : np.ndarray
        Parámetros del modelo a optimizar
    returns : np.ndarray
        Serie de retornos
    tau : float
        Nivel de confianza
        
    Retorna:
    --------
    float
        Valor de la función de pérdida
    """
    var_estimado = _calcular_var_caviar(returns, beta, tau)
    
    # Calcular hits (violaciones del VaR)
    hits = returns - var_estimado
    
    # Función de pérdida cuantílica
    loss = np.sum((tau - (hits < 0)) * hits)
    
    return loss


def caviar_asymmetric(returns, tau=0.01):
    """
    Estima VaR usando el modelo CAViaR asimétrico.
    
    El modelo CAViaR permite que el VaR responda asimétricamente a 
    retornos positivos y negativos.
    
    Parámetros:
    -----------
    returns : pd.Series
        Serie de retornos logarítmicos
    tau : float, opcional
        Nivel de confianza (default: 0.01 para VaR al 99%)
        
    Retorna:
    --------
    tuple
        (pd.Series con VaR estimado, np.ndarray con parámetros estimados)
    """
    # Limpiar valores faltantes
    returns_clean = returns.dropna()
    returns_array = returns_clean.values
    
    # Valores iniciales para optimización
    beta_inicial = np.array([-0.02, 0.9, 0.1, 0.1])
    
    # Restricciones para los parámetros
    bounds = [
        (None, None),  # beta0: sin restricción
        (0, 1),        # beta1: persistencia entre 0 y 1
        (0, None),     # beta2: impacto positivo >= 0
        (0, None)      # beta3: impacto negativo >= 0
    ]
    
    # Optimización
    resultado = minimize(
        caviar_loss,
        beta_inicial,
        args=(returns_array, tau),
        method="L-BFGS-B",
        bounds=bounds
    )
    
    beta_estimado = resultado.x
    
    # Calcular VaR con parámetros estimados
    var_estimado = _calcular_var_caviar(returns_array, beta_estimado, tau)
    
    return pd.Series(var_estimado, index=returns_clean.index), beta_estimado


def parametric_var_gjr(returns, tau=0.01):
    """
    Calcula VaR paramétrico usando modelo GJR-GARCH con distribución t-Student.
    
    El modelo GJR-GARCH captura la asimetría en la volatilidad (efecto leverage).
    
    Parámetros:
    -----------
    returns : pd.Series
        Serie de retornos logarítmicos
    tau : float, opcional
        Nivel de confianza (default: 0.01 para VaR al 99%)
        
    Retorna:
    --------
    pd.Series
        Serie de VaR estimado
    """
    # Eliminar valores faltantes
    returns_clean = returns.dropna()
    
    # Ajustar modelo GJR-GARCH(1,1,1) con distribución t
    modelo = arch_model(
        returns_clean * 100,  # Escalar para estabilidad numérica
        mean="Constant",
        vol="GARCH",
        p=1,  # Orden GARCH
        o=1,  # Orden asimétrico (GJR)
        q=1,  # Orden ARCH
        dist="t"  # Distribución t-Student
    )
    
    resultado = modelo.fit(disp="off")
    
    # Extraer parámetros estimados
    media_condicional = resultado.params["mu"] / 100
    volatilidad_condicional = resultado.conditional_volatility / 100
    grados_libertad = resultado.params["nu"]
    
    # Calcular cuantil de la distribución t-Student
    cuantil_t = t.ppf(tau, grados_libertad)
    factor_escala = np.sqrt((grados_libertad - 2) / grados_libertad)
    
    # Calcular VaR
    var_estimado = media_condicional + volatilidad_condicional * factor_escala * cuantil_t
    
    return pd.Series(var_estimado, index=returns_clean.index)


def clean_desc(delta_covar_dict, label):
    """
    Crea tabla de estadísticos descriptivos para ΔCoVaR.
    
    Parámetros:
    -----------
    delta_covar_dict : dict
        Diccionario con empresa como clave y serie ΔCoVaR como valor
    label : str
        Etiqueta para identificar el tipo de análisis
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame con estadísticos descriptivos por empresa
    """
    filas = []
    
    for empresa, serie in delta_covar_dict.items():
        filas.append([
            empresa,
            serie.mean(),
            serie.median(),
            serie.std()
        ])
    
    df = pd.DataFrame(
        filas,
        columns=["Empresa", "Media", "Mediana", "Desv. típica"]
    )
    
    df = df.set_index("Empresa").round(3)
    
    print(f"\nDESCRIPTIVOS ΔCoVaR – {label}\n")
    print(df)
    
    return df


def obtener_estadisticas(tickers):
    """
    Obtiene estadísticas fundamentales de empresas desde Yahoo Finance.
    
    Parámetros:
    -----------
    tickers : list
        Lista de símbolos ticker
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame con información fundamental de cada empresa
    """
    datos = []
    
    for ticker in tickers:
        accion = yf.Ticker(ticker)
        info = accion.info
        
        datos.append({
            "Ticker": ticker,
            "Nombre": info.get("shortName"),
            "Market Cap": info.get("marketCap"),
            "Beta": info.get("beta"),
            "P/E": info.get("trailingPE"),
            "Dividend Yield": info.get("dividendYield"),
            "Total Debt": info.get("totalDebt"),
            "Avg_Daily_Volume": info.get("averageVolume")
        })
    
    return pd.DataFrame(datos)