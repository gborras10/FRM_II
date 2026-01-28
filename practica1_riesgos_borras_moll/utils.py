# ================================================================
#  Quantitative Finance Utilities — BARRA Model & Portfolio Analysis
# ================================================================
#  Authors : Guillem Borràs Espert (github:guiillem10) & Gonzalo Moll Acha 
#  Project: Quantitative Risk & Factor Modelling Toolkit
#  File   : utils.py
#  Date   : 2025-10-27
#  Version: ---
# ---------------------------------------------------------------
#  Description:
#  Comprehensive utility module for multi-factor risk decomposition
#  and portfolio analytics, including:
#
#    • BARRA-style factor model estimation via GLS (ajusta_barra_mcg)
#    • Portfolio variance decomposition (systematic / idiosyncratic)
#    • OLS / GLS regression diagnostics and AIC computation
#    • Risk contribution and block-level visualization
#    • Long-short portfolio construction and factor exposure mapping
#    • Clean plotting utilities for time series and factor results
#
#  Main Functions:
#    - plot(df, activos="All", stock=True, savepath=None)
#    - ajusta_barra_mcg(B, R)
#    - analiza_cartera_barra(w, B, R, out_barra)
#    - analiza_cartera_w(w, B, Sigma_f, D, Sigma)
#    - calcular_aic_cartera(w, F, ret_df)
#    - estimar_beta(r, F)
#    - pesos_longshort(long_idx, short_idx, universe)
#    - grafico_contrib(resultado, nombre_cartera, top_k=23)
#
#  Notes:
#    • All matrix operations follow NumPy broadcasting conventions.
#    • Compatible with pandas DataFrames for financial time series.
#    • Outputs formatted for LaTeX / academic reporting.
#
#  Dependencies:
#    numpy, pandas, matplotlib, statsmodels
#
#  License:
#    MIT License — © 2025 Guillem Borràs Espert
# ---------------------------------------------------------------
#  Example:
#    from utils import ajusta_barra_mcg, analiza_cartera_barra
#    out = ajusta_barra_mcg(B, R)
#    res = analiza_cartera_barra(w, B, R, out)
#    grafico_contrib(res, "Cartera_PC1")
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%
def plot(df, activos="All", stock=True, savepath=None):
    """
    df        : DataFrame con índice de fechas y columnas = activos
                (puede ser precios o rendimientos logarítmicos)
    activos   : "All" o lista de columnas a representar
    stock     : True -> precios, False -> rendimientos logarítmicos
    savepath  : ruta para guardar (png). Si None, solo muestra
    """

    # --- selección de activos ---
    if activos == "All":
        data = df.copy()
    else:
        if isinstance(activos, str):
            activos = [activos]
        missing = [s for s in activos if s not in df.columns]
        if missing:
            print(f"Aviso: no encontrados {missing}. Se omiten.")
        keep = [s for s in activos if s in df.columns]
        if not keep:
            print("Nada que representar.")
            return
        data = df[keep].copy()

    # --- estilo general ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    # --- representación ---
    for col in data.columns:
        ax.plot(
            data.index,
            data[col],
            label=col,
            linewidth=1.3,
            alpha=0.8
        )

    # --- estética ---
    ax.set_xlabel("Fecha", fontsize=10, labelpad=6)
    ax.set_ylabel("Precio" if stock else "Rendimiento logarítmico", fontsize=10, labelpad=6)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.spines["top"].set_alpha(0.0)
    ax.spines["right"].set_alpha(0.0)

    if data.shape[1] <= 15:
        ax.legend(frameon=False, fontsize=8, ncols=4)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
#%%
def ajusta_modelo(X, Y):
    """
    X: (n, k) incluye constante en la primera columna
    Y: (n, m) una o varias series (activos) en columnas
    """
    n, k = X.shape
    # Coeficientes OLS (k x m)
    G = np.linalg.solve(X.T @ X, X.T @ Y)
    # Alternativa robusta:
    # G, *_ = np.linalg.lstsq(X, Y, rcond=None)

    # Residuos (n x m)
    E = Y - X @ G

    # Varianza residual por activo (vector de longitud m)
    diagE = np.diag(E.T @ E)                 # SSE por columna
    var_res = diagE / (n - k)                # σ̂²_j
    desvTip = np.sqrt(var_res)               # σ̂_j

    # SST por activo (vector de longitud m)
    sumSquares = np.sum((Y - Y.mean(axis=0))**2, axis=0)

    # AIC por activo
    AIC = n * np.log(diagE/n) + 2 * k

    # Betas sin la constante (quita la primera fila)
    beta = G[1:, :]                          # (k-1, m)

    return {
        "beta": beta,
        "desvTip": desvTip,
        "AIC": AIC,
        "residuos": E,
        "G": G,
    }
#%%
def to_pct(x):
    return float(np.nan if x is None else x * 100.0)
#%%
def calcular_aic_cartera(w, F, ret_df):
    """Ajusta la cartera con factores y devuelve AIC del modelo OLS."""
    r = ret_df.dot(w)  # retornos de la cartera
    common_idx = F.index.intersection(r.index)
    X = sm.add_constant(F.loc[common_idx])
    y = r.loc[common_idx]
    model = sm.OLS(y, X).fit()
    return model.aic 
#%%
def estimar_beta(r, F):
    """Ajusta r ~ F y devuelve betas, residuos y modelo OLS."""
    common_idx = F.index.intersection(r.index)
    X = sm.add_constant(F.loc[common_idx])
    y = r.loc[common_idx]
    model = sm.OLS(y, X).fit()
    beta = model.params.drop('const').values
    residuos = model.resid
    return beta, residuos, model
#%%
def analiza_cartera_w(w, B, Sigma_f, D, Sigma, asset_index=None, r_p=None, k_params=None):
    """
    Análisis de cartera con descomposición de varianzas y cálculo opcional del AIC.

    Parámetros
    ----------
    w : array-like | pd.Series
        Pesos (longitud N). Si es Series, se reordena a 'asset_index' (faltantes -> 0).
    B : (N x K) np.ndarray
        Matriz de cargas del modelo BARRA por activo.
    Sigma_f : (K x K) np.ndarray
        Covarianza de factores.
    D : (N,) o (N x N)
        Varianzas idiosincráticas (vector) o matriz diagonal.
    Sigma : (N x N) np.ndarray
        Covarianza total de activos.
    asset_index : list-like, opcional
        Orden esperado de activos para reindexar w si viene como Series.
    r_p : array-like, opcional
        Serie temporal de rendimientos de la cartera (para calcular AIC).
    k_params : int, opcional
        Nº de parámetros del modelo para el AIC. Por defecto: K + 2 (media + varianza idio/efectiva).

    Retorna
    -------
    dict con:
        var_total, var_sist, var_idio, pct_sist, pct_idio, w, (opcional) aic, n_obs, k_params
    """
    # ---- Convertir y alinear pesos
    if isinstance(w, (pd.Series, pd.DataFrame)):
        w_ser = pd.Series(w).astype(float).squeeze()
        if asset_index is not None:
            w_ser = w_ser.reindex(asset_index).fillna(0.0)
        w = w_ser.values
    else:
        w = np.asarray(w, dtype=float).reshape(-1,)

    # ---- Chequeo dimensiones
    if w.shape[0] != Sigma.shape[0]:
        raise ValueError(f"w tiene longitud {w.shape[0]} pero Sigma espera {Sigma.shape[0]} activos.")

    w_col = w.reshape(-1, 1)

    # ---- Varianzas (modelo)
    sigma2_total = float(w_col.T @ Sigma @ w_col)
    sigma2_sist  = float(w_col.T @ B @ Sigma_f @ B.T @ w_col)

    Dm = np.asarray(D)
    if Dm.ndim == 1:
        Dm = np.diag(Dm)
    sigma2_idio  = float(w_col.T @ Dm @ w_col)

    # ---- Arreglos numéricos
    tol = 1e-16
    if -tol < sigma2_total < 0: sigma2_total = 0.0
    if -tol < sigma2_sist  < 0: sigma2_sist  = 0.0
    if -tol < sigma2_idio  < 0: sigma2_idio  = 0.0

    if sigma2_total <= 0:
        pct_sist = np.nan
        pct_idio = np.nan
    else:
        pct_sist = sigma2_sist / sigma2_total
        pct_idio = sigma2_idio / sigma2_total

    out = {
        "var_total": sigma2_total,
        "var_sist":  sigma2_sist,
        "var_idio":  sigma2_idio,
        "pct_sist":  pct_sist,
        "pct_idio":  pct_idio,
        "w":         w,
    }
    return out
#%%
def ajusta_barra_mcg(B, R, round_dec=4):
    # --- nombres de factres ---
    factor_names = list(B.columns)

    # --- orientar R y tomar nombres de activos DESDE R ---
    if isinstance(R, pd.DataFrame):
        if R.shape[0] != B.shape[0] and R.shape[1] == B.shape[0]:
            # R es T×N (fechas x activos)
            asset_names = list(R.columns)
            time_index  = list(R.index)
            Rm = R.values.T     # N×T
        elif R.shape[0] == B.shape[0]:
            # R es N×T (activos x fechas)
            asset_names = list(R.index)
            time_index  = list(R.columns)
            Rm = R.values       # N×T
        else:
            raise ValueError("Dimensiones de R incompatibles con B.")
    elif isinstance(R, pd.Series):
        if len(R) == B.shape[0]:
            # Serie con misma longitud que filas de B
            asset_names = [R.name if R.name is not None else "cartera"]
            time_index  = list(R.index)
            Rm = R.values.reshape(1, -1)  # 1×T (un activo por T observaciones)
        else:
            raise ValueError("Longitud de la Serie R incompatible con B.")
    else:
        Rm = np.asarray(R)
        if Rm.ndim == 1:
            # Vector 1D, convertir a matriz 1×T
            Rm = Rm.reshape(1, -1)
            asset_names = ["cartera"]
            time_index = [f"t{i+1}" for i in range(Rm.shape[1])]
        else:
            asset_names = [f"a{i+1}" for i in range(Rm.shape[0])]
            time_index  = [f"t{i+1}" for i in range(Rm.shape[1])]

    Bm = B.values
    N_factors, K = Bm.shape
    N_assets, T = Rm.shape

    # Verificar compatibilidad
    if N_factors != N_assets:
        raise ValueError(f"B tiene {N_factors} filas pero R tiene {N_assets} activos. Deben coincidir.")

    # --- MCO inicial ---
    BtB   = Bm.T @ Bm
    BtR   = Bm.T @ Rm
    F_ols = np.linalg.solve(BtB, BtR)
    E_ols = Rm - Bm @ F_ols

    # --- var idiosincrática diagonal (Psi) ---
    e_bar    = E_ols.mean(axis=1, keepdims=True)
    psi_diag = ((E_ols - e_bar)**2).sum(axis=1) / max(T - 1, 1)
    psi_diag = np.where(psi_diag <= 0, 1e-12, psi_diag)
    W        = 1.0 / psi_diag

    # --- GLS ---
    BW  = (Bm.T * W) @ Bm
    BWR = (Bm.T * W) @ Rm
    Fm  = np.linalg.solve(BW, BWR)        # (K,T)
    Em  = Rm - Bm @ Fm                    # (N,T)

    # --- cov factores y varianzas ---
    f_bar   = Fm.mean(axis=1, keepdims=True)
    Omega_f = ((Fm - f_bar) @ (Fm - f_bar).T) / max(T - 1, 1)
    var_factor    = np.einsum('ij,jk,ik->i', Bm, Omega_f, Bm)
    var_idio      = psi_diag
    var_total_hat = var_factor + var_idio
    pct_explica_factor = 100.0 * var_factor / var_total_hat
    pct_idiosincratica = 100.0 * var_idio   / var_total_hat

    # --- R^2 temporal (referencia) ---
    r_bar = Rm.mean(axis=1, keepdims=True)
    sst   = ((Rm - r_bar)**2).sum(axis=1)
    sse   = ((Em - Em.mean(axis=1, keepdims=True))**2).sum(axis=1)
    k_params = K + 1
    # Evitar log(0): si sse_i <= 0, ponemos NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        AIC_vec = T * np.log(np.where(sse > 0, sse / T, np.nan)) + 2 * k_params


    # --- DataFrames de salida mínimos ---
    factores_df = pd.DataFrame(Fm.T, index=time_index, columns=factor_names).round(round_dec)
    Omega_df    = pd.DataFrame(Omega_f, index=factor_names, columns=factor_names).round(round_dec)
    residuos_df = pd.DataFrame(Em.T, index=time_index, columns=asset_names).round(round_dec)

    resumen_pct = pd.DataFrame({
        'var_factor'         : var_factor,
        'var_idio'           : var_idio,
        'var_total_hat'      : var_total_hat,
        'pct_explica_factor' : pct_explica_factor,
        'pct_idiosincratica' : pct_idiosincratica,
        'AIC': AIC_vec
    }, index=asset_names).round(round_dec)

    # --- versión para mostrar: primera columna = activo, sin índice numérico ---
    resumen = resumen_pct.copy()
    resumen.insert(0, 'activo', resumen.index.astype(str))
    resumen.reset_index(drop=True, inplace=True)

    return {
        'factores'   : factores_df,
        'Omega_f'    : Omega_df,
        'resumen_pct': resumen_pct,
        'resumen'    : resumen,
        'residuos'   : residuos_df,
    }
#%%
def analiza_cartera_barra(w, B, R, out_barra, nombre='cartera', longshort=False):
    """
    Analiza cartera por agregación lineal (sin re-estimar).
    
    Parámetros:
        w: pesos (N,)
        B: exposiciones (N x K)
        R: retornos (T x N o N x T)
        out_barra: dict con 'Omega_f', 'residuos', 'factores'
        nombre: str
        longshort: bool, si True permite var negativas
    
    Retorna: dict con 'betas', 'residuos', 'resumen', 'contrib', 'factores_medios'
    """
    Omega = out_barra['Omega_f'].values
    resid = out_barra['residuos']
    factores = out_barra['factores']  # T x K
    
    w = np.asarray(w).ravel()
    facs = list(B.columns)
    
    # Betas
    beta = w @ B.values
    
    # Retornos y residuos de cartera
    if isinstance(R, pd.DataFrame):
        R_cart = R.values @ w if R.shape[1] == len(w) else w @ R.values
        t_idx = R.index if R.shape[1] == len(w) else R.columns
    else:
        R = np.asarray(R)
        R_cart = R @ w if R.shape[1] == len(w) else w @ R
        t_idx = range(len(R_cart))
    
    e_cart = resid.values @ w

    # AIC de la cartera
    Tn = len(e_cart)
    k_params = len(facs) + 1
    sse_cart = float(np.sum((e_cart - np.mean(e_cart))**2))
    AIC = (Tn * np.log(sse_cart / Tn) + 2 * k_params) if sse_cart > 0 else np.nan

    # Varianzas
    var_tot = np.var(R_cart, ddof=1)
    var_sys = float(beta @ Omega @ beta)
    var_idio = var_tot - var_sys
    
    if not longshort and var_idio < 0:
        var_idio = 0.0
    
    pct_sys = 100 * var_sys / var_tot if var_tot > 0 else np.nan
    pct_idio = 100 * var_idio / var_tot if var_tot > 0 else np.nan
    
    # Contribuciones
    rc = beta * (Omega @ beta)
    rc_sum = rc.sum()
    contrib = pd.DataFrame({
        'cartera': nombre,
        'factor': facs,
        'rc': rc,
        'pct_rc': 100 * rc / rc_sum if abs(rc_sum) > 1e-10 else 0
    })
    
    # Media temporal de factores (para esta cartera son los mismos del modelo global)
    factores_medios = factores.mean().to_frame(name='media_temporal')
    
    # Output
    betas_df = pd.DataFrame({nombre: beta}, index=facs).T
    resid_df = pd.Series(e_cart, index=t_idx, name=nombre)
    
    resumen = pd.DataFrame({
        'var_total': var_tot,
        'var_sist': var_sys,
        'var_idio': var_idio,
        'pct_sist': pct_sys,
        'pct_idio': pct_idio,
        'T': len(R_cart),
        'net_exp': w.sum(),
        'AIC': AIC
    }, index=[nombre])
    
    return {
        'betas': betas_df,
        'residuos': resid_df,
        'resumen': resumen,
        'contrib': contrib,
        'factores_medios': factores_medios
    }
#%%
def pesos_longshort(long_idx, short_idx, universe):
    """Crea pesos long-short: +1/n_long, -1/n_short, resto 0."""
    w = pd.Series(0.0, index=universe)
    
    long = universe.intersection(long_idx)
    if len(long) > 0:
        w[long] = 1.0 / len(long)
    
    short = universe.intersection(short_idx)
    if len(short) > 0:
        w[short] = -1.0 / len(short)
    
    return w
#%%
def grafico_contrib(resultado, nombre_cartera, top_k=23):
    """Gráfico de contribuciones: top factores + grupos."""
    contrib = resultado['contrib'].copy()
    
    # Clasificar en bloques
    def _bloque(f):
        f = f.lower()
        if 'industry' in f or 'ind_' in f: return 'Industry'
        if 'country' in f or 'pais' in f: return 'Country'
        if 'size' in f: return 'Size'
        if 'book' in f or 'valor' in f: return 'Book'
        if 'beta' in f: return 'Beta'
        return 'Other'
    
    contrib['bloque'] = contrib['factor'].apply(_bloque)
    contrib_sort = contrib.sort_values('rc', key=abs, ascending=False)
    
    # Top-K factores
    top = contrib_sort.head(top_k).iloc[::-1]
    
    # Contribución por bloques
    bloques = contrib.groupby('bloque')['rc'].sum()
    total_rc = contrib['rc'].sum()
    pct_bloques = 100 * bloques / total_rc
    pct_bloques = pct_bloques.sort_values(ascending=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Riesgo — {nombre_cartera}", fontsize=13, fontweight='bold')
    
    # (A) Top factores
    ax = axes[0]
    colors = ['#d62728' if x < 0 else '#1f77b4' for x in top['pct_rc']]
    ax.barh(top['factor'], top['pct_rc'], color=colors, alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('% contribución')
    ax.set_title(f'Top-{len(top)} factores')
    ax.grid(axis='x', alpha=0.3)
    
    # Etiquetas
    for i, (idx, row) in enumerate(top.iterrows()):
        v = row['pct_rc']
        ax.text(v, i, f" {v:.1f}%", va='center', fontsize=8, 
                ha='left' if v > 0 else 'right')
    
    # (B) Bloques
    ax = axes[1]
    colors_blk = ['#d62728' if x < 0 else '#1f77b4' for x in pct_bloques]
    ax.bar(range(len(pct_bloques)), pct_bloques.values, color=colors_blk, alpha=0.7)
    ax.set_xticks(range(len(pct_bloques)))
    ax.set_xticklabels(pct_bloques.index, rotation=45, ha='right')
    ax.set_ylabel('% varianza factores')
    ax.set_title('Contribución por bloques')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    
    # Etiquetas
    for i, v in enumerate(pct_bloques.values):
        ax.text(i, v, f"{v:.1f}%", ha='center', 
                va='bottom' if v > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.show()