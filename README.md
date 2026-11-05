# ProStock-AI 📈🤖

**ProStock-AI** é um dashboard de análise financeira desenvolvido em Python que usa Machine Learning para projetar o retorno de ativos (B3 e bolsas internacionais). O sistema combina indicadores técnicos clássicos com um modelo *Random Forest* e, principalmente, **mede o quanto o modelo acerta** com um backtest fora da amostra antes de mostrar qualquer alvo.

## 🚀 Funcionalidades

* **Previsão de retorno com IA:** um *Random Forest* prevê o retorno dos próximos 5, 10, 21 ou 42 pregões (1 semana a 2 meses). O modelo usa apenas features relativas (retornos de 5/20/60 pregões, distância das médias de 50 e 200, RSI, volatilidade e volume relativo), então não fica preso à faixa de preço que viu no treino.
* **Backtest walk-forward:** para cada ativo, o modelo é treinado só com o passado e testado no futuro em 5 janelas (`TimeSeriesSplit` com *gap* igual ao prazo, para não vazar informação). O app mostra o **% de acerto de direção** comparado com um baseline ingênuo e avisa quando o modelo não supera o baseline.
* **Faixa de incerteza honesta:** o intervalo de 80% vem dos erros reais do backtest (quantis 10% e 90%), não da dispersão das árvores no próprio treino.
* **Indicadores:** RSI, MA200, P/VP (exibido como "N/D" quando o Yahoo não informa) e Dividend Yield dos últimos 12 meses calculado a partir dos proventos pagos.
* **Gestão de Carteira:** lista de favoritos por usuário em SQLite.
* **Gráficos:** histórico de 60 pregões e projeção no mesmo eixo, com a faixa de incerteza.
* **Interface responsiva:** a análise roda em segundo plano, então a janela não congela.

## 🛠️ Tecnologias

* **Linguagem:** Python 3.10+
* **Interface Gráfica:** Tkinter
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Dados Financeiros:** yfinance (Yahoo Finance)
* **Análise Técnica:** ta
* **Base de Dados:** SQLite3
* **Gráficos:** Matplotlib
* **Testes:** pytest

## 📂 Estrutura

```
app_analise_fii.py   # interface (login, dashboard, gráfico)
modelo.py            # features, treino, backtest e regras de sinal
tests/               # testes do modelo com dados sintéticos (rodam sem internet)
```

## ▶️ Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python app_analise_fii.py
```

Na tela de login, use **Criar Conta** para criar seu usuário.

Testes:

```bash
python -m pytest
```

## ⚠️ Aviso

Projeto educacional. As projeções não são recomendação de investimento: veja sempre a taxa de acerto do backtest antes de levar um sinal a sério.
