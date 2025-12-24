import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Silenciar avisos técnicos do Scikit-Learn
warnings.filterwarnings("ignore", category=UserWarning)

# --- BANCO DE DADOS E SEGURANÇA ---
def gerar_hash(senha):
    return hashlib.sha256(senha.encode()).hexdigest()

def iniciar_db():
    """Garante que as tabelas existam com a estrutura correta."""
    conn = sqlite3.connect('investimentos.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (user TEXT PRIMARY KEY, senha TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS favoritos (user TEXT, ticker TEXT, PRIMARY KEY (user, ticker))''')
    conn.commit()
    conn.close()

# --- MOTOR DE INTELIGÊNCIA ARTIFICIAL ---
def calcular_ia_real(ticker, dias_previsao):
    """Treina o modelo com alvo dinâmico e indicadores de valor."""
    try:
        ativo = yf.Ticker(ticker)
        df = ativo.history(period="5y")
        if df.empty or len(df) < 200: return None

        # Cálculo de indicadores técnicos
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['Volatilidade'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df = df.dropna()

        # Define o alvo com base na escolha do usuário
        df['Target'] = df['Close'].shift(-dias_previsao)
        df_ml = df.dropna()
        
        features = ['Close', 'RSI', 'Volatilidade']
        X = df_ml[features]
        y = df_ml['Target']

        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        
        # Predição
        ultimos_dados = pd.DataFrame([df[features].iloc[-1].values], columns=features)
        predicao = modelo.predict(ultimos_dados)[0]

        # Intervalo de confiança (Desvio padrão das árvores)
        preds_trees = [tree.predict(ultimos_dados)[0] for tree in modelo.estimators_]
        desvio = np.std(preds_trees)

        info = ativo.info
        return {
            "atual": df['Close'].iloc[-1],
            "alvo": predicao,
            "intervalo": (predicao - desvio, predicao + desvio),
            "pvp": info.get('priceToBook', 1.0),
            "ma200": df['MA200'].iloc[-1],
            "rsi": df['RSI'].iloc[-1],
            "historico": df['Close'].tail(250),
            "prazo_dias": dias_previsao # <--- ADICIONADO: Retorna o prazo para usar no gráfico
        }
    except Exception as e:
        print(f"Erro IA em {ticker}: {e}")
        return None

# --- TELAS DO SISTEMA ---
class TelaLogin:
    def __init__(self, root):
        self.root = root
        self.root.title("Login - Investidor IA.PRO")
        self.root.geometry("350x400")
        self.root.configure(bg="#1B1919")
        self.root.resizable(False, False)
        iniciar_db()

        tk.Label(root, text="INVESTIDOR IA.PRO", fg="white", bg="#1B1919", font=("Aptos", 16, "bold")).pack(pady=20)
        
        tk.Label(root, text="Usuário:", fg="white", bg="#1B1919").pack()
        self.ent_user = tk.Entry(root, justify='center')
        self.ent_user.pack(pady=5)
        
        tk.Label(root, text="Senha:", fg="white", bg="#1B1919").pack()
        self.ent_pass = tk.Entry(root, show="*", justify='center')
        self.ent_pass.pack(pady=5)
        
        tk.Button(root, text="Entrar", bg="#27ae60", fg="white", width=15, command=self.validar).pack(pady=10)
        tk.Button(root, text="Criar Conta", width=15, command=self.cadastrar).pack()

    def validar(self):
        user, senha = self.ent_user.get(), gerar_hash(self.ent_pass.get())
        conn = sqlite3.connect('investimentos.db')
        res = conn.execute("SELECT * FROM usuarios WHERE user=? AND senha=?", (user, senha)).fetchone()
        conn.close()
        if res:
            self.root.destroy()
            main_root = tk.Tk()
            AppPrincipal(main_root, user)
            main_root.mainloop()
        else: messagebox.showerror("Erro", "Login Inválido.")

    def cadastrar(self):
        user, senha = self.ent_user.get(), self.ent_pass.get()
        try:
            conn = sqlite3.connect('investimentos.db')
            conn.execute("INSERT INTO usuarios VALUES (?, ?)", (user, gerar_hash(senha)))
            conn.commit()
            conn.close()
            messagebox.showinfo("Sucesso", "Usuário criado!")
        except: messagebox.showerror("Erro", "Usuário já existe.")

class AppPrincipal:
    def __init__(self, root, user):
        self.root = root
        self.user = user
        self.root.title(f"Dashboard - {user}")
        self.root.geometry("1150x900")
        
        # Cache para permitir trocar o gráfico ao clicar
        self.cache_dados = {}
        self.prazos = {"7 Dias": 7, "10 Dias": 10, "1 Mês": 30, "2 Meses": 60}

        # Header
        header = tk.Frame(root, bg="#34495e", height=40)
        header.pack(fill=tk.X)
        tk.Label(header, text=f"Logado como: {user}", fg="white", bg="#34495e").pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="Logout", command=self.logout).pack(side=tk.RIGHT, padx=10)

        # Controles
        ctrl = tk.Frame(root, pady=10)
        ctrl.pack()
        
        self.ent_ticker = tk.Entry(ctrl, font=("Arial", 12), width=15)
        self.ent_ticker.pack(side=tk.LEFT, padx=5)
        self.ent_ticker.insert(0, "ITSA4.SA")

        tk.Label(ctrl, text="Previsão:").pack(side=tk.LEFT, padx=2)
        self.cb_prazo = ttk.Combobox(ctrl, values=list(self.prazos.keys()), width=10, state="readonly")
        self.cb_prazo.set("10 Dias")
        self.cb_prazo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(ctrl, text="Add", command=self.add_fav).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Remover", bg="#e74c3c", fg="white", command=self.remover_fav).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="ANALISAR LISTA", bg="navy", fg="white", font=("Arial", 10, "bold"), command=self.processar_lote).pack(side=tk.LEFT, padx=10)

        tk.Label(root, text="(Clique em uma linha para ver o gráfico específico)", font=("Arial", 8, "italic"), fg="gray").pack()

        # Área de Tabela - Adicionado bind de clique
        self.output = scrolledtext.ScrolledText(root, width=145, height=20, font=("Consolas", 9))
        self.output.pack(pady=5, padx=10)
        self.output.bind("<ButtonRelease-1>", self.clique_na_tabela)

        # Gráfico
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=5)

    def logout(self):
        self.root.destroy()
        r = tk.Tk()
        TelaLogin(r)
        r.mainloop()

    def add_fav(self):
        t = self.ent_ticker.get().upper().strip()
        if not t: return
        try:
            conn = sqlite3.connect('investimentos.db')
            conn.execute("INSERT INTO favoritos VALUES (?, ?)", (self.user, t))
            conn.commit()
            conn.close()
            messagebox.showinfo("OK", f"{t} adicionado!")
        except: messagebox.showwarning("Aviso", "Já existe na lista.")

    def remover_fav(self):
        t = self.ent_ticker.get().upper().strip()
        if not t: return
        if messagebox.askyesno("Confirmar", f"Excluir {t} da lista?"):
            conn = sqlite3.connect('investimentos.db')
            conn.execute("DELETE FROM favoritos WHERE user=? AND ticker=?", (self.user, t))
            conn.commit()
            conn.close()
            messagebox.showinfo("OK", f"{t} removido.")
            self.processar_lote()

    def clique_na_tabela(self, event):
        """Identifica o ticker clicado e atualiza o gráfico."""
        try:
            linha_texto = self.output.get("insert linestart", "insert lineend")
            ticker_clicado = linha_texto.split('|')[0].strip()
            if ticker_clicado in self.cache_dados:
                self.atualizar_grafico(ticker_clicado)
        except: pass

    def atualizar_grafico(self, ticker):
        """Renderiza o gráfico com foco na projeção futura."""
        res = self.cache_dados[ticker]
        self.ax.clear()
        
        # 1. Preparar dados: Usar apenas os ultimos 50 dias para dar "zoom" no presente
        hist_recente = res['historico'].tail(50)
        y_hist = hist_recente.values
        x_hist = np.arange(len(y_hist)) # Eixo X numérico para facilitar projeção
        
        # 2. Coordenadas da Projeção
        ultimo_x = x_hist[-1]
        ultimo_preco = y_hist[-1]
        
        dias_futuro = res['prazo_dias']
        alvo_preco = res['alvo']
        
        # X e Y do futuro
        x_futuro = [ultimo_x, ultimo_x + dias_futuro]
        y_futuro = [ultimo_preco, alvo_preco]
        
        # 3. Plotar Histórico (Linha Azul Sólida)
        self.ax.plot(x_hist, y_hist, label='Histórico (50d)', color='#2980b9', linewidth=2)
        
        # 4. Plotar Projeção (Linha Verde Tracejada) conectando Hoje -> Alvo
        self.ax.plot(x_futuro, y_futuro, linestyle='--', color='#27ae60', linewidth=2, label=f'Projeção ({dias_futuro}d)')
        
        # 5. Ponto final (Alvo)
        self.ax.scatter([x_futuro[1]], [alvo_preco], color='#27ae60', s=50, zorder=5)
        
        # 6. Cone de Incerteza (Intervalo de confiança)
        # O intervalo começa no preço atual (incerteza 0) e abre até o intervalo previsto
        conf_min, conf_max = res['intervalo']
        self.ax.fill_between(x_futuro, [ultimo_preco, conf_min], [ultimo_preco, conf_max], color='gray', alpha=0.15, label='Incerteza')

        # Estilização
        self.ax.set_title(f"Projeção IA: {ticker} -> Alvo: {alvo_preco:.2f}", fontsize=10)
        self.ax.legend(loc='upper left', fontsize=8)
        self.ax.grid(True, alpha=0.3, linestyle=':')
        
        # Remove rotulos X numéricos confusos e poe apenas texto indicativo
        self.ax.set_xticks([ultimo_x, ultimo_x + dias_futuro])
        self.ax.set_xticklabels(["Hoje", f"+{dias_futuro} Dias"])

        self.canvas.draw()

    def processar_lote(self):
        self.output.delete(1.0, tk.END)
        self.cache_dados = {} # Limpa cache anterior
        prazo_dias = self.prazos[self.cb_prazo.get()]
        
        conn = sqlite3.connect('investimentos.db')
        tickers = conn.execute("SELECT ticker FROM favoritos WHERE user=?", (self.user,)).fetchall()
        conn.close()

        if not tickers: return
        
        # Header mantendo o campo de intervalo solicitado
        header = f"{'TICKER':<10} | {'PRECO':<8} | {'ALVO ('+str(prazo_dias)+'d)':<22} | {'VARIA %':<8} | {'P/VP':<6} | {'STATUS'}\n"
        self.output.insert(tk.END, header + "-"*115 + "\n")

        for (t,) in tickers:
            self.output.insert(tk.END, f"Processando {t}...")
            self.root.update()
            res = calcular_ia_real(t, prazo_dias)
            self.output.delete("insert linestart", "insert lineend")

            if res:
                self.cache_dados[t] = res # Salva para o clique
                var = ((res['alvo'] / res['atual']) - 1) * 100
                tendencia_alta = res['atual'] > res['ma200']
                esta_barato = res['rsi'] < 35 or res['pvp'] < 0.95

                if var > 1.5 and res['pvp'] < 1.1 and tendencia_alta:
                    status = "COMPRA (Trend)"
                    rec = "Trend de alta confirmada."
                elif esta_barato and var > -1.0:
                    status = "COMPRA (Value)"
                    rec = f"Ativo descontado (RSI {res['rsi']:.0f})."
                elif var < -2.5 or (not tendencia_alta and not esta_barato):
                    status = "VENDA/ALERTA"
                    rec = "Risco de queda ou fora de tendência."
                else:
                    status = "NEUTRO"
                    rec = "Aguardar melhor posição."

                # Formatação do intervalo de variação
                intervalo_str = f"{res['alvo']:.2f} ({res['intervalo'][0]:.2f}-{res['intervalo'][1]:.2f})"
                
                linha = f"{t:<10} | {res['atual']:>8.2f} | {intervalo_str:<22} | {var:>7.1f}% | {res['pvp']:>6.2f} | {status}\n"
                self.output.insert(tk.END, linha + f" > {rec}\n\n")

                self.atualizar_grafico(t)
            else:
                self.output.insert(tk.END, f"{t:<10} | Dados insuficientes.\n")
            self.root.update()

if __name__ == "__main__":
    iniciar_db()
    root = tk.Tk()
    TelaLogin(root)
    root.mainloop()