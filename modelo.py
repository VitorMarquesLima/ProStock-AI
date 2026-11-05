"""Motor de previsão do ProStock-AI.

O modelo prevê o RETORNO dos próximos N pregões (não o preço), usando apenas
features relativas (não dependem do nível de preço). A qualidade é medida com
backtest walk-forward fora da amostra e comparada com baselines ingênuos.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from ta.momentum import RSIIndicator

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = [
    "ret_5", "ret_20", "ret_60",
    "dist_ma50", "dist_ma200",
    "rsi", "vol_20", "volume_rel",
]
MIN_PREGOES = 300  # histórico mínimo (MA200 + janela de treino)
N_SPLITS = 5


@dataclass
class Backtest:
    """Métricas fora da amostra (walk-forward)."""
    n_previsoes: int
    acerto_direcao: float        # % de vezes que acertou se sobe ou cai
    acerto_baseline: float       # % de acerto chutando sempre a direção mais comum no treino
    mae_modelo: float            # erro médio absoluto do retorno previsto
    mae_baseline: float          # erro médio absoluto prevendo retorno 0 (preço não muda)
    erro_q10: float              # quantis dos erros, usados para o intervalo
    erro_q90: float

    @property
    def supera_baseline(self) -> bool:
        return self.acerto_direcao > self.acerto_baseline and self.mae_modelo < self.mae_baseline


@dataclass
class Resultado:
    ticker: str
    atual: float
    retorno_previsto: float
    alvo: float
    intervalo: tuple[float, float]
    prazo_pregoes: int
    rsi: float
    ma200: float
    pvp: float | None
    dy: float | None              # dividend yield dos últimos 12 meses (fração)
    backtest: Backtest
    historico: pd.Series = field(repr=False)


def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe OHLCV (colunas Close e Volume) e devolve as features."""
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["Close"] = close
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["ret_60"] = close.pct_change(60)
    out["dist_ma50"] = close / close.rolling(50).mean() - 1
    ma200 = close.rolling(200).mean()
    out["MA200"] = ma200
    out["dist_ma200"] = close / ma200 - 1
    out["rsi"] = RSIIndicator(close=close, window=14).rsi()
    out["vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    if "Volume" in df and df["Volume"].fillna(0).sum() > 0:
        vol = df["Volume"].replace(0, np.nan)
        out["volume_rel"] = (vol / vol.rolling(20).mean()).fillna(1.0)
    else:
        out["volume_rel"] = 1.0
    return out


def _novo_modelo() -> RandomForestRegressor:
    # min_samples_leaf alto reduz overfitting em séries financeiras (muito ruído)
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=20, max_features=0.5,
        n_jobs=-1, random_state=42,
    )


def backtest_walk_forward(X: pd.DataFrame, y: pd.Series, prazo: int) -> Backtest:
    """Treina só com o passado e testa no futuro, em N_SPLITS janelas.

    O gap = prazo evita vazamento: o alvo de uma linha de treino usa preços de
    até `prazo` pregões à frente, que não podem cair dentro do período de teste.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=prazo)
    preds, reais, base_dir = [], [], []
    for idx_tr, idx_te in tscv.split(X):
        m = _novo_modelo().fit(X.iloc[idx_tr], y.iloc[idx_tr])
        preds.append(m.predict(X.iloc[idx_te]))
        reais.append(y.iloc[idx_te].values)
        # baseline de direção: a direção mais frequente no treino
        sobe = (y.iloc[idx_tr] > 0).mean() >= 0.5
        base_dir.append(np.full(len(idx_te), 1.0 if sobe else -1.0))

    p, r, b = np.concatenate(preds), np.concatenate(reais), np.concatenate(base_dir)
    erros = r - p
    return Backtest(
        n_previsoes=len(r),
        acerto_direcao=float(np.mean(np.sign(p) == np.sign(r))),
        acerto_baseline=float(np.mean(b == np.sign(r))),
        mae_modelo=float(np.mean(np.abs(erros))),
        mae_baseline=float(np.mean(np.abs(r))),
        erro_q10=float(np.quantile(erros, 0.10)),
        erro_q90=float(np.quantile(erros, 0.90)),
    )


def dividend_yield_12m(dividendos: pd.Series | None, preco: float) -> float | None:
    if dividendos is None or dividendos.empty or not preco:
        return None
    ultimo = dividendos.index.max()
    soma = dividendos[dividendos.index > ultimo - pd.Timedelta(days=365)].sum()
    return float(soma / preco) if soma > 0 else 0.0


def analisar(ticker: str, df: pd.DataFrame, prazo: int, pvp: float | None = None) -> Resultado | None:
    """Análise completa a partir de um DataFrame de histórico (testável sem internet)."""
    if df is None or len(df) < MIN_PREGOES:
        return None
    feats = montar_features(df)
    feats["alvo"] = feats["Close"].shift(-prazo) / feats["Close"] - 1
    base = feats.dropna(subset=FEATURES)
    treino = base.dropna(subset=["alvo"])
    if len(treino) < 200:
        return None

    X, y = treino[FEATURES], treino["alvo"]
    bt = backtest_walk_forward(X, y, prazo)

    modelo = _novo_modelo().fit(X, y)
    ultimo = base[FEATURES].iloc[[-1]]
    ret = float(modelo.predict(ultimo)[0])
    atual = float(base["Close"].iloc[-1])

    return Resultado(
        ticker=ticker,
        atual=atual,
        retorno_previsto=ret,
        alvo=atual * (1 + ret),
        intervalo=(atual * (1 + ret + bt.erro_q10), atual * (1 + ret + bt.erro_q90)),
        prazo_pregoes=prazo,
        rsi=float(base["rsi"].iloc[-1]),
        ma200=float(base["MA200"].iloc[-1]),
        pvp=pvp,
        dy=dividend_yield_12m(df.get("Dividends"), atual),
        backtest=bt,
        historico=df["Close"].tail(250),
    )


def classificar(r: Resultado) -> tuple[str, str]:
    """Regras de sinal. Só usa o P/VP quando ele existe de verdade."""
    var = r.retorno_previsto * 100
    tendencia_alta = r.atual > r.ma200
    pvp_ok = r.pvp is not None
    barato = r.rsi < 35 or (pvp_ok and r.pvp < 0.95)

    if not r.backtest.supera_baseline:
        aviso = " (modelo não supera o baseline neste ativo: sinal fraco)"
    else:
        aviso = ""

    if var > 1.5 and tendencia_alta and (not pvp_ok or r.pvp < 1.1):
        return "COMPRA (Trend)", "Tendência de alta e projeção positiva." + aviso
    if barato and var > -1.0:
        return "COMPRA (Value)", f"Ativo descontado (RSI {r.rsi:.0f})." + aviso
    if var < -2.5 or (not tendencia_alta and not barato):
        return "VENDA/ALERTA", "Risco de queda ou fora de tendência." + aviso
    return "NEUTRO", "Aguardar melhor posição." + aviso


def baixar_e_analisar(ticker: str, prazo: int) -> Resultado | None:
    """Busca os dados no Yahoo Finance e roda a análise."""
    import yfinance as yf

    ativo = yf.Ticker(ticker)
    df = ativo.history(period="5y")
    pvp = None
    try:
        v = ativo.info.get("priceToBook")
        pvp = float(v) if v not in (None, 0) else None
    except Exception:
        pass  # .info é instável; seguimos sem P/VP
    return analisar(ticker, df, prazo, pvp)
