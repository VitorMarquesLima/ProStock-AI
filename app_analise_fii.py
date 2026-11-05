import hashlib
import queue
import sqlite3
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from modelo import baixar_e_analisar, classificar

# Caminho absoluto: o banco é sempre o que está ao lado do script,
# não importa de qual pasta o app foi aberto.
DB_PATH = Path(__file__).resolve().parent / "investimentos.db"
BG = "#1B1919"


# --- BANCO DE DADOS E SEGURANÇA ---
def conectar():
    return sqlite3.connect(DB_PATH)


def gerar_hash(senha):
    return hashlib.sha256(senha.encode()).hexdigest()


def iniciar_db():
    """Garante que as tabelas existam e migra bancos antigos."""
    with conectar() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS usuarios (user TEXT PRIMARY KEY, senha TEXT, validade TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS favoritos (user TEXT, ticker TEXT, PRIMARY KEY (user, ticker))")
        cols = [c[1] for c in conn.execute("PRAGMA table_info(usuarios)")]
        if "validade" not in cols:
            conn.execute("ALTER TABLE usuarios ADD COLUMN validade TEXT")
    conn.close()


# --- TELAS DO SISTEMA ---
class TelaLogin(tk.Frame):
    def __init__(self, app):
        super().__init__(app.root, bg=BG)
        self.app = app
        tk.Label(self, text="INVESTIDOR IA.PRO", fg="white", bg=BG, font=("Aptos", 16, "bold")).pack(pady=20)

        tk.Label(self, text="Usuário:", fg="white", bg=BG).pack()
        self.ent_user = tk.Entry(self, justify="center")
        self.ent_user.pack(pady=5)

        tk.Label(self, text="Senha:", fg="white", bg=BG).pack()
        self.ent_pass = tk.Entry(self, show="*", justify="center")
        self.ent_pass.pack(pady=5)
        self.ent_pass.bind("<Return>", lambda e: self.validar())

        tk.Button(self, text="Entrar", bg="#27ae60", fg="white", width=15, command=self.validar).pack(pady=10)
        tk.Button(self, text="Criar Conta", width=15, command=self.cadastrar).pack()
        self.ent_user.focus_set()

    def validar(self):
        user, senha = self.ent_user.get().strip(), gerar_hash(self.ent_pass.get())
        with conectar() as conn:
            res = conn.execute("SELECT 1 FROM usuarios WHERE user=? AND senha=?", (user, senha)).fetchone()
        conn.close()
        if res:
            self.app.mostrar_dashboard(user)
        else:
            messagebox.showerror("Erro", "Login Inválido.")

    def cadastrar(self):
        user, senha = self.ent_user.get().strip(), self.ent_pass.get()
        if not user or not senha:
            messagebox.showwarning("Aviso", "Preencha usuário e senha.")
            return
        try:
            with conectar() as conn:
                conn.execute("INSERT INTO usuarios (user, senha) VALUES (?, ?)", (user, gerar_hash(senha)))
            conn.close()
            messagebox.showinfo("Sucesso", "Usuário criado!")
        except sqlite3.IntegrityError:
            messagebox.showerror("Erro", "Usuário já existe.")


class Dashboard(tk.Frame):
    # Prazos em PREGÕES (dias úteis), o mesmo eixo do histórico no gráfico
    PRAZOS = {"1 Semana (5 pregões)": 5, "2 Semanas (10 pregões)": 10,
              "1 Mês (21 pregões)": 21, "2 Meses (42 pregões)": 42}
    COLUNAS = [("ticker", "Ticker", 80), ("preco", "Preço", 70), ("alvo", "Alvo", 70),
               ("faixa", "Faixa 80%", 120), ("var", "Var %", 60), ("acerto", "Acerto (base)", 100),
               ("pvp", "P/VP", 55), ("dy", "DY 12m", 60), ("status", "Status", 120), ("obs", "Observação", 330)]

    def __init__(self, app, user):
        super().__init__(app.root)
        self.app, self.user = app, user
        self.cache_dados = {}
        self.fila = queue.Queue()
        self.processando = False

        header = tk.Frame(self, bg="#34495e", height=40)
        header.pack(fill=tk.X)
        tk.Label(header, text=f"Logado como: {user}", fg="white", bg="#34495e").pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="Logout", command=self.app.mostrar_login).pack(side=tk.RIGHT, padx=10)

        ctrl = tk.Frame(self, pady=10)
        ctrl.pack()
        self.ent_ticker = tk.Entry(ctrl, font=("Arial", 12), width=15)
        self.ent_ticker.pack(side=tk.LEFT, padx=5)
        self.ent_ticker.insert(0, "ITSA4.SA")

        tk.Label(ctrl, text="Previsão:").pack(side=tk.LEFT, padx=2)
        self.cb_prazo = ttk.Combobox(ctrl, values=list(self.PRAZOS), width=22, state="readonly")
        self.cb_prazo.set("2 Semanas (10 pregões)")
        self.cb_prazo.pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl, text="Add", command=self.add_fav).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Remover", bg="#e74c3c", fg="white", command=self.remover_fav).pack(side=tk.LEFT, padx=2)
        self.btn_analisar = tk.Button(ctrl, text="ANALISAR LISTA", bg="navy", fg="white",
                                      font=("Arial", 10, "bold"), command=self.processar_lote)
        self.btn_analisar.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(self, text="Clique em uma linha para ver o gráfico. "
                                   "Acerto = % de vezes que o modelo acertou a direção no backtest (baseline entre parênteses).",
                                   font=("Arial", 8, "italic"), fg="gray")
        self.lbl_status.pack()

        # Tabela
        self.tabela = ttk.Treeview(self, columns=[c[0] for c in self.COLUNAS], show="headings", height=12)
        for cid, titulo, larg in self.COLUNAS:
            self.tabela.heading(cid, text=titulo)
            self.tabela.column(cid, width=larg, anchor="w" if cid in ("ticker", "obs", "status") else "e")
        self.tabela.tag_configure("compra", foreground="#1e8449")
        self.tabela.tag_configure("venda", foreground="#c0392b")
        self.tabela.tag_configure("erro", foreground="gray")
        self.tabela.pack(fill=tk.X, padx=10, pady=5)
        self.tabela.bind("<<TreeviewSelect>>", self.clique_na_tabela)

        # Gráfico
        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(pady=5, fill=tk.BOTH, expand=True)

    # --- favoritos ---
    def add_fav(self):
        t = self.ent_ticker.get().upper().strip()
        if not t:
            return
        try:
            with conectar() as conn:
                conn.execute("INSERT INTO favoritos VALUES (?, ?)", (self.user, t))
            conn.close()
            messagebox.showinfo("OK", f"{t} adicionado!")
        except sqlite3.IntegrityError:
            messagebox.showwarning("Aviso", "Já existe na lista.")

    def remover_fav(self):
        t = self.ent_ticker.get().upper().strip()
        if t and messagebox.askyesno("Confirmar", f"Excluir {t} da lista?"):
            with conectar() as conn:
                conn.execute("DELETE FROM favoritos WHERE user=? AND ticker=?", (self.user, t))
            conn.close()
            if self.tabela.exists(t):
                self.tabela.delete(t)
            self.cache_dados.pop(t, None)

    # --- análise em segundo plano (a janela não congela) ---
    def processar_lote(self):
        if self.processando:
            return
        with conectar() as conn:
            tickers = [r[0] for r in conn.execute("SELECT ticker FROM favoritos WHERE user=?", (self.user,))]
        conn.close()
        if not tickers:
            messagebox.showinfo("Lista vazia", "Adicione ativos com o botão Add.")
            return

        self.tabela.delete(*self.tabela.get_children())
        self.cache_dados = {}
        self.processando = True
        self.btn_analisar.config(state=tk.DISABLED, text="ANALISANDO...")
        prazo = self.PRAZOS[self.cb_prazo.get()]
        threading.Thread(target=self._trabalho, args=(tickers, prazo), daemon=True).start()
        self.after(100, self._consumir_fila)

    def _trabalho(self, tickers, prazo):
        for i, t in enumerate(tickers, 1):
            self.fila.put(("progresso", t, i, len(tickers)))
            try:
                res = baixar_e_analisar(t, prazo)
            except Exception as e:  # erro de rede, ticker inválido etc.
                res = e
            self.fila.put(("resultado", t, res))
        self.fila.put(("fim",))

    def _consumir_fila(self):
        try:
            while True:
                msg = self.fila.get_nowait()
                if msg[0] == "progresso":
                    _, t, i, n = msg
                    self.lbl_status.config(text=f"Analisando {t} ({i}/{n})... treinando e fazendo backtest.")
                elif msg[0] == "resultado":
                    self._mostrar_linha(msg[1], msg[2])
                else:
                    self.processando = False
                    self.btn_analisar.config(state=tk.NORMAL, text="ANALISAR LISTA")
                    self.lbl_status.config(text="Pronto. Clique em uma linha para ver o gráfico.")
                    return
        except queue.Empty:
            pass
        self.after(100, self._consumir_fila)

    def _mostrar_linha(self, t, res):
        if res is None or isinstance(res, Exception):
            motivo = "Dados insuficientes (mín. ~300 pregões)." if res is None else f"Erro: {res}"
            self.tabela.insert("", tk.END, iid=t, values=(t, "", "", "", "", "", "", "", "", motivo), tags=("erro",))
            return
        self.cache_dados[t] = res
        status, rec = classificar(res)
        bt = res.backtest
        valores = (
            t, f"{res.atual:.2f}", f"{res.alvo:.2f}",
            f"{res.intervalo[0]:.2f} – {res.intervalo[1]:.2f}",
            f"{res.retorno_previsto * 100:+.1f}%",
            f"{bt.acerto_direcao:.0%} ({bt.acerto_baseline:.0%})",
            f"{res.pvp:.2f}" if res.pvp is not None else "N/D",
            f"{res.dy:.1%}" if res.dy is not None else "N/D",
            status, rec,
        )
        tag = "compra" if status.startswith("COMPRA") else "venda" if status.startswith("VENDA") else ""
        self.tabela.insert("", tk.END, iid=t, values=valores, tags=(tag,))
        if len(self.cache_dados) == 1:
            self.atualizar_grafico(t)

    def clique_na_tabela(self, _event=None):
        sel = self.tabela.selection()
        if sel and sel[0] in self.cache_dados:
            self.atualizar_grafico(sel[0])

    def atualizar_grafico(self, ticker):
        """Histórico e projeção no MESMO eixo: pregões."""
        res = self.cache_dados[ticker]
        self.ax.clear()

        y_hist = res.historico.tail(60).values
        x_hist = np.arange(-len(y_hist) + 1, 1)  # ... -2, -1, 0 (hoje)
        n = res.prazo_pregoes
        lo, hi = res.intervalo

        self.ax.plot(x_hist, y_hist, label="Histórico (60 pregões)", color="#2980b9", linewidth=2)
        self.ax.plot([0, n], [res.atual, res.alvo], linestyle="--", color="#27ae60", linewidth=2,
                     label=f"Projeção ({n} pregões)")
        self.ax.scatter([n], [res.alvo], color="#27ae60", s=50, zorder=5)
        self.ax.fill_between([0, n], [res.atual, lo], [res.atual, hi], color="gray", alpha=0.2,
                             label="Faixa 80% (erros do backtest)")

        bt = res.backtest
        self.ax.set_title(f"{ticker}: alvo {res.alvo:.2f} ({res.retorno_previsto * 100:+.1f}%)  |  "
                          f"acerto de direção no backtest {bt.acerto_direcao:.0%} vs baseline {bt.acerto_baseline:.0%}",
                          fontsize=9)
        self.ax.legend(loc="upper left", fontsize=8)
        self.ax.grid(True, alpha=0.3, linestyle=":")
        self.ax.set_xlabel("Pregões (0 = hoje)")
        self.canvas.draw()


class App:
    """Uma única janela Tk; login e dashboard são telas trocadas dentro dela."""
    def __init__(self, root):
        self.root = root
        self.tela = None
        iniciar_db()
        self.mostrar_login()

    def _trocar(self, nova, titulo, geometria, redimensionar):
        if self.tela is not None:
            self.tela.destroy()
        self.root.title(titulo)
        self.root.geometry(geometria)
        self.root.resizable(redimensionar, redimensionar)
        self.tela = nova
        self.tela.pack(fill=tk.BOTH, expand=True)

    def mostrar_login(self):
        self._trocar(TelaLogin(self), "Login - Investidor IA.PRO", "350x400", False)

    def mostrar_dashboard(self, user):
        self._trocar(Dashboard(self, user), f"Dashboard - {user}", "1250x900", True)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
