import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go

st.set_page_config(page_title="Painel IA — Bolsa EUA", page_icon="📈", layout="wide")

st.title("📈 Painel IA — Bolsa Americana")
st.caption("Protótipo educativo: análise histórica e simulação fictícia. Não envia ordens nem movimenta dinheiro real.")

with st.sidebar:
    st.header("Configurações")
    ticker = st.text_input("Símbolo da ação/ETF", value="AAPL").strip().upper()
    period = st.selectbox("Histórico para análise", ["2y", "5y", "10y"], index=1)
    horizon = st.selectbox("Horizonte da previsão", [5, 10, 20], index=0)
    initial_cash = st.number_input("Saldo virtual inicial (US$)", min_value=100.0, max_value=1_000_000.0, value=10_000.0, step=500.0)
    st.caption("Exemplos de símbolos: AAPL, MSFT, NVDA, SPY. Os dados dependem do provedor e podem ter atraso.")

@st.cache_data(ttl=900)
def load_data(symbol, hist_period):
    df = yf.download(symbol, period=hist_period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = {"Close", "High", "Low", "Volume"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    return df.dropna().copy()

def make_features(df):
    x = pd.DataFrame(index=df.index)
    close = df["Close"]
    x["ret_1d"] = close.pct_change()
    x["ret_5d"] = close.pct_change(5)
    x["ret_20d"] = close.pct_change(20)
    x["volatility_20d"] = x["ret_1d"].rolling(20).std()
    x["sma_ratio_20"] = close / close.rolling(20).mean() - 1
    x["sma_ratio_50"] = close / close.rolling(50).mean() - 1
    x["volume_change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return x.replace([np.inf, -np.inf], np.nan)

if not ticker:
    st.info("Digite um símbolo para começar.")
    st.stop()

with st.spinner("Carregando dados históricos..."):
    raw = load_data(ticker, period)

if raw.empty or len(raw) < 100:
    st.error("Não foi possível obter dados suficientes para esse símbolo. Confira o código e tente novamente.")
    st.stop()

close = raw["Close"].astype(float)
features = make_features(raw)
future_return = close.shift(-horizon) / close - 1
dataset = features.copy()
dataset["target"] = (future_return > 0).astype(int)
dataset.loc[future_return.isna(), "target"] = np.nan
dataset = dataset.dropna()

if len(dataset) < 80:
    st.error("Há poucos dados válidos para treinar o protótipo.")
    st.stop()

X = dataset.drop(columns=["target"])
y = dataset["target"].astype(int)
split = int(len(dataset) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

latest_features = features.dropna().iloc[[-1]]
if latest_features.empty:
    st.error("Não há dados suficientes para calcular os indicadores.")
    st.stop()

prob_up = float(model.predict_proba(latest_features[X.columns])[0][1])
last_price = float(close.iloc[-1])
sma20 = float(close.rolling(20).mean().iloc[-1])
sma50 = float(close.rolling(50).mean().iloc[-1])
trend = "Tendência positiva" if last_price > sma20 > sma50 else ("Tendência negativa" if last_price < sma20 < sma50 else "Tendência mista")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Último fechamento", f"${last_price:,.2f}", help=f"Data: {close.index[-1].date()}")
c2.metric("Estimativa do modelo", f"{prob_up:.0%}", help=f"Probabilidade estimada pelo classificador de retorno positivo em {horizon} pregões; não é garantia.")
c3.metric("Leitura técnica simples", trend)
c4.metric("Acurácia no período de teste", f"{test_accuracy:.1%}", help="Métrica histórica simples; não inclui custos de negociação e não prevê desempenho futuro.")

st.warning("A probabilidade e a acurácia são experimentais. Acurácia isolada não mede rentabilidade; este painel não é aconselhamento financeiro.")

tab1, tab2, tab3 = st.tabs(["Mercado", "Simulação fictícia", "Como o modelo foi avaliado"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=close, mode="lines", name="Fechamento"))
    fig.add_trace(go.Scatter(x=raw.index, y=close.rolling(20).mean(), mode="lines", name="Média 20 pregões"))
    fig.add_trace(go.Scatter(x=raw.index, y=close.rolling(50).mean(), mode="lines", name="Média 50 pregões"))
    fig.update_layout(title=f"{ticker} — preço ajustado", xaxis_title="Data", yaxis_title="US$", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Últimos dados")
    st.dataframe(raw.tail(10).sort_index(ascending=False), use_container_width=True)

with tab2:
    st.subheader("Simulação de estratégia didática")
    st.write("Regra ilustrativa: mantém uma posição virtual quando o preço está acima da média de 20 e 50 pregões; caso contrário, fica em caixa. Não representa recomendação de compra ou venda.")
    sim = raw[["Close"]].copy()
    sim["SMA20"] = sim["Close"].rolling(20).mean()
    sim["SMA50"] = sim["Close"].rolling(50).mean()
    sim["position"] = ((sim["Close"] > sim["SMA20"]) & (sim["SMA20"] > sim["SMA50"])).astype(int)
    sim["market_return"] = sim["Close"].pct_change().fillna(0)
    sim["strategy_return"] = sim["position"].shift(1).fillna(0) * sim["market_return"]
    sim["equity"] = initial_cash * (1 + sim["strategy_return"]).cumprod()
    sim["buy_hold"] = initial_cash * (1 + sim["market_return"]).cumprod()
    sim["drawdown"] = sim["equity"] / sim["equity"].cummax() - 1
    total_return = sim["equity"].iloc[-1] / initial_cash - 1
    bh_return = sim["buy_hold"].iloc[-1] / initial_cash - 1
    max_dd = sim["drawdown"].min()
    a, b, c = st.columns(3)
    a.metric("Saldo virtual da estratégia", f"${sim['equity'].iloc[-1]:,.2f}", f"{total_return:+.1%}")
    b.metric("Referência comprar e manter", f"${sim['buy_hold'].iloc[-1]:,.2f}", f"{bh_return:+.1%}")
    c.metric("Maior queda do saldo virtual", f"{max_dd:.1%}")
    eqfig = go.Figure()
    eqfig.add_trace(go.Scatter(x=sim.index, y=sim["equity"], name="Estratégia simulada"))
    eqfig.add_trace(go.Scatter(x=sim.index, y=sim["buy_hold"], name="Referência comprar e manter"))
    eqfig.update_layout(title="Evolução do saldo virtual", xaxis_title="Data", yaxis_title="US$", hovermode="x unified")
    st.plotly_chart(eqfig, use_container_width=True)
    st.caption("Simulação simplificada: não inclui impostos, taxas, spread, slippage, dividendos reinvestidos ou execução realista. Resultados passados não garantem resultados futuros.")

with tab3:
    st.subheader("Avaliação temporal")
    st.write(f"O modelo foi treinado nos primeiros 80% dos exemplos cronológicos e avaliado nos 20% finais. O alvo é saber se o preço ajustado estará acima do nível atual após {horizon} pregões.")
    st.write(f"Exemplos de treino: {len(X_train):,} · exemplos de teste: {len(X_test):,}")
    st.write("Para uma avaliação mais rigorosa, o próximo passo é usar validação walk-forward, incluir custos de negociação e comparar com uma referência simples.")
    st.caption("A saída do modelo é uma estimativa estatística baseada em dados históricos, não uma certeza nem uma recomendação personalizada.")

st.divider()
st.caption("Protótipo para estudo. Dados fornecidos por Yahoo Finance via yfinance; disponibilidade, cobertura, precisão e permissões de uso podem variar. Não execute operações reais a partir deste protótipo.")
