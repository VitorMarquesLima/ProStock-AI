"""Testes do motor de previsão com dados sintéticos (rodam sem internet)."""
import numpy as np
import pandas as pd
import pytest

from modelo import FEATURES, analisar, classificar, dividend_yield_12m, montar_features


def serie(n=1200, drift=0.0003, vol=0.015, seed=0, inicio=10.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = inicio * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.DataFrame({"Close": close, "Volume": rng.integers(1e5, 1e6, n).astype(float),
                         "Dividends": 0.0}, index=idx)


def test_features_nao_dependem_do_nivel_de_preco():
    df = serie()
    a = montar_features(df)[FEATURES].dropna()
    df2 = df.copy()
    df2["Close"] *= 50  # mesmo ativo, preço 50x maior
    b = montar_features(df2)[FEATURES].dropna()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_historico_curto_retorna_none():
    assert analisar("X", serie(n=250), 10) is None


@pytest.mark.parametrize("prazo", [5, 21])
def test_resultado_coerente(prazo):
    r = analisar("X", serie(), prazo, pvp=None)
    assert r is not None
    assert r.prazo_pregoes == prazo
    assert r.intervalo[0] < r.alvo < r.intervalo[1]
    assert np.isclose(r.alvo, r.atual * (1 + r.retorno_previsto))
    bt = r.backtest
    assert 0 <= bt.acerto_direcao <= 1 and 0 <= bt.acerto_baseline <= 1
    assert bt.n_previsoes > 300


def test_na_maxima_historica_o_alvo_pode_passar_do_topo():
    """O modelo antigo (preço como alvo) ficava preso abaixo da máxima do treino.
    Prevendo retorno, uma tendência forte projeta acima do topo."""
    df = serie(drift=0.002, vol=0.005, seed=1)
    r = analisar("X", df, 21)
    assert r.alvo > df["Close"].max()


def test_ruido_puro_nao_supera_baseline_com_folga():
    """Passeio aleatório: o backtest não deve mostrar acerto alto (sem vazamento)."""
    r = analisar("X", serie(drift=0.0, seed=7), 10)
    assert r.backtest.acerto_direcao < 0.60


def test_pvp_ausente_nao_vira_1():
    r = analisar("X", serie(), 10, pvp=None)
    assert r.pvp is None
    status, _ = classificar(r)
    assert status in {"COMPRA (Trend)", "COMPRA (Value)", "VENDA/ALERTA", "NEUTRO"}


def test_dividend_yield():
    idx = pd.bdate_range("2024-01-01", periods=300)
    div = pd.Series(0.0, index=idx)
    div.iloc[[50, 150, 250]] = 0.10
    assert dividend_yield_12m(div, 10.0) == pytest.approx(0.03)
    assert dividend_yield_12m(None, 10.0) is None
