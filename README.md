# ProStock-AI 📈🤖

**ProStock-AI** é um dashboard de análise financeira desenvolvido em Python que utiliza algoritmos de Machine Learning para prever tendências de preços de ativos (B3 e bolsas internacionais). O sistema combina indicadores técnicos clássicos com a robustez do modelo *Random Forest* para oferecer uma visão preditiva ao investidor.

## 🚀 Funcionalidades

* **Previsão Baseada em IA:** Utiliza o modelo regressor *Random Forest* para projetar preços em diferentes prazos (7, 10, 30 ou 60 dias).
* **Análise de Indicadores:** Integra dados de RSI (Índice de Força Relativa), Médias Móveis (MA200), P/VP e **Dividend Yield**.
* **Backtesting de Precisão:** Realiza uma simulação histórica para calcular a taxa de acerto (%) da IA antes de exibir o alvo.
* **Gestão de Carteira:** Sistema de "Favoritos" persistente em base de dados SQLite.
* **Gráficos Dinâmicos:** Visualização do histórico de preços vs. projeção futura com cone de incerteza.
* **Sistema de Licenciamento:** Controle de acesso por login com contador de dias de licença restantes.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Interface Gráfica:** Tkinter
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Dados Financeiros:** YFinance (API Yahoo Finance)
* **Análise Técnica:** TA-Lib / TA (Technical Analysis Library)
* **Base de Dados:** SQLite3
* **Gráficos:** Matplotlib

## 📋 Pré-requisitos

Para rodar este projeto, deve-se instalar as dependências necessárias:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib ta