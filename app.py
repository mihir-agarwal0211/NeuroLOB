import streamlit as st
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp, norm, t as student_t, wasserstein_distance
from statsmodels.tsa.stattools import acf
import os
import io

# ==========================================
# 1. APP CONFIG & STYLE
# ==========================================
st.set_page_config(
    page_title="NeuroLOB Simulation Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2e3140;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================
class Config:
    SEQ_LEN = 64
    EMBED_DIM = 128
    N_HEADS = 4
    N_LAYERS = 2
    DIFFUSION_STEPS = 50
    NUM_EVENT_TYPES = 6
    DEVICE = "cpu" # Force CPU for streamlit compatibility

config = Config()

# ==========================================
# 3. NEURAL ARCHITECTURE
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x, times):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)).to(x.device)
        sinusoid_inp = torch.einsum("bi,j->bij", times, inv_freq)
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class TimeAwareEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, config.EMBED_DIM)
        self.rope = RotaryEmbedding(config.EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.EMBED_DIM, nhead=config.N_HEADS, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)
    def forward(self, x):
        times = torch.cumsum(torch.abs(x[:, :, 0]), dim=1)
        h = self.input_proj(x)
        h = self.rope(h, times)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        h = self.transformer(h, mask=mask)
        return h[:, -1, :]

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        combined_dim = input_dim + 32 + context_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x, t, context):
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb, context], dim=1)
        return self.net(inp)

class FullModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TimeAwareEncoder(input_dim)
        self.diffusion = DiffusionModel(input_dim, config.EMBED_DIM)

class NPPHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cont_head = nn.Sequential(nn.Linear(config.EMBED_DIM, 64), nn.ReLU(), nn.Linear(64, 3))
        self.type_head = nn.Sequential(nn.Linear(config.EMBED_DIM, 64), nn.ReLU(), nn.Linear(64, config.NUM_EVENT_TYPES))
    def forward(self, context):
        return self.cont_head(context), self.type_head(context)

class NPPBaseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TimeAwareEncoder(input_dim)
        self.head = NPPHead(input_dim)
    def forward(self, history):
        return self.head(self.encoder(history))

# ==========================================
# 4. UTILITIES & GENERATION LOGIC
# ==========================================
@st.cache_resource
def load_model(uploaded_file, model_type, local_path=None):
    input_dim = 9 
    model = FullModel(input_dim).to(config.DEVICE) if model_type == "Diffusion (SDE)" else NPPBaseline(input_dim).to(config.DEVICE)
    
    loaded_source = None
    if uploaded_file is not None:
        try:
            state_dict = torch.load(uploaded_file, map_location=config.DEVICE)
            model.load_state_dict(state_dict)
            loaded_source = "Uploaded File"
        except: pass
    elif local_path and os.path.exists(local_path):
        try:
            state_dict = torch.load(local_path, map_location=config.DEVICE)
            model.load_state_dict(state_dict)
            loaded_source = f"Local: {local_path}"
        except: pass
        
    return model, loaded_source

@st.cache_data
def load_real_data(file_input):
    """
    Robust CSV loader that handles both plain text and GZIP compressed files
    automatically, solving the '0x8b' error.
    """
    if file_input is None: return None
    
    df = None
    try:
        # 1. Try reading normally
        if isinstance(file_input, str):
            try:
                df = pd.read_csv(file_input)
            except UnicodeDecodeError:
                # If standard read fails, try GZIP
                df = pd.read_csv(file_input, compression='gzip')
        else:
            # For uploaded file objects
            try:
                df = pd.read_csv(file_input)
            except UnicodeDecodeError:
                file_input.seek(0)
                df = pd.read_csv(file_input, compression='gzip')
                
        # 2. Normalize columns
        if df is not None:
            if 'price_ret' not in df.columns and 'price' in df.columns:
                df['price_ret'] = df['price'].pct_change().fillna(0)
            return df
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def generate_enhanced_data(_model, model_type, n_steps, vol_mult, drift, start_price, duration_days, tick_size, kurtosis_boost):
    _model.eval()
    input_dim = 9
    current_seq = torch.randn(1, config.SEQ_LEN, input_dim).to(config.DEVICE)
    raw_outputs = []
    
    with torch.no_grad():
        for _ in range(n_steps):
            if model_type == "Diffusion (SDE)":
                context = _model.encoder(current_seq)
                x = torch.randn(1, input_dim).to(config.DEVICE)
                for i in reversed(range(10)):
                    t_tensor = (torch.ones(1, 1) * i).to(config.DEVICE) * (config.DIFFUSION_STEPS // 10)
                    pred_noise = _model.diffusion(x, t_tensor, context)
                    z = torch.randn_like(x) if i > 0 else 0
                    x = x - (0.1 * pred_noise) + (0.05 * z) 
                final_step = x
            else:
                pred_cont, pred_logits = _model(current_seq)
                noise = torch.randn_like(pred_cont) * 0.1 
                next_cont = pred_cont + noise
                next_type = torch.zeros(1, config.NUM_EVENT_TYPES).to(config.DEVICE)
                next_type[0, 0] = 1.0 
                final_step = torch.cat([next_cont, next_type], dim=1)

            raw_outputs.append(final_step.cpu().numpy().flatten())
            current_seq = torch.cat([current_seq[:, 1:, :], final_step.unsqueeze(1)], dim=1)

    # Physics Injection Layer
    cols = ['norm_log_dt', 'norm_price_ret', 'norm_log_vol'] + [f'type_{i}' for i in range(6)]
    df = pd.DataFrame(raw_outputs, columns=cols)
    
    raw_nn_rets = df['norm_price_ret'].values
    
    # A. Fat Tail Injection
    target_df = 30.0 - (kurtosis_boost * 27.0) 
    if target_df < 2.1: target_df = 2.1 
    
    z_scores = (raw_nn_rets - np.mean(raw_nn_rets)) / (np.std(raw_nn_rets) + 1e-8)
    u_vals = norm.cdf(z_scores)
    fat_tailed_rets = student_t.ppf(u_vals, df=target_df)
    
    # B. Volatility Clustering (GARCH)
    garch_rets = []
    h = 1.0 
    omega, alpha, beta = 0.05, 0.15, 0.80
    
    for r in fat_tailed_rets:
        h = omega + alpha * (r**2) + beta * h
        sigma = np.sqrt(h)
        garch_rets.append(r * sigma)
        
    garch_rets = np.array(garch_rets)
    
    # C. Final Scaling
    target_step_std = (vol_mult * 0.01) * np.sqrt(duration_days / n_steps)
    current_std = np.std(garch_rets)
    scaled_rets = (garch_rets / current_std) * target_step_std
    
    step_drift = (drift * 0.0001) 
    df['price_ret'] = scaled_rets + step_drift
    
    raw_price = start_price * (1 + df['price_ret']).cumprod()
    df['price'] = np.round(raw_price / tick_size) * tick_size 
    df['final_ret'] = df['price'].pct_change().fillna(0)
    
    total_sec = duration_days * 24 * 3600
    df['timestamp'] = [pd.Timestamp.now() + pd.Timedelta(seconds=i*(total_sec/n_steps)) for i in range(len(df))]
    df['amount'] = np.exp(df['norm_log_vol']) * 1000
    
    return df

def calculate_metrics(df):
    total_ret = (df['price'].iloc[-1] / df['price'].iloc[0]) - 1
    steps_per_year = len(df) * (365 / ((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400))
    vol = df['final_ret'].std() * np.sqrt(steps_per_year) * 100 
    drawdown = (df['price'] - df['price'].cummax()) / df['price'].cummax()
    return total_ret, vol, drawdown.min(), df['amount'].sum() * df['price'].mean()

# ==========================================
# 5. UI & EXECUTION FLOW
# ==========================================
st.sidebar.title("🎛 NeuroLOB Config")

model_type = st.sidebar.selectbox("Architecture", ["Diffusion (SDE)", "NPP (Point Process)"])
default_path = "models_diffusion_lob_final.pth" if model_type == "Diffusion (SDE)" else "models_npp_baseline_final.pth"
uploaded_model = st.sidebar.file_uploader("Load Weights (Optional)", type=['pth'])
model, source = load_model(uploaded_model, model_type, default_path)

if source: st.sidebar.success(f"Weights: {source}")
else: st.sidebar.warning("Using Random Weights (Simulation Mode)")

st.sidebar.subheader("Benchmarking")
local_data = "bitmex_incremental_book_L2.csv"
use_local = False
# if os.path.exists(local_data):
#     use_local = st.sidebar.checkbox("Use local 'bitmex_incremental_book_L2.csv'", value=False)

df_real = None
if use_local:
    df_real = load_real_data(local_data)
else:
    uploaded_csv = st.sidebar.file_uploader("Upload Real CSV", type=['csv', 'gz'])
    if uploaded_csv:
        df_real = load_real_data(uploaded_csv)

if df_real is not None: st.sidebar.success(f"Benchmark: {len(df_real):,} rows")

st.sidebar.subheader("Scenario")
days = st.sidebar.slider("Days to Generate", 1, 30, 7)
price = st.sidebar.number_input("Start Price", value=95000.0)
tick_size = st.sidebar.number_input("Tick Size", value=0.5)

st.sidebar.subheader("Market Regime")
vol = st.sidebar.slider("Volatility (Annual %)", 10.0, 200.0, 60.0)
drift = st.sidebar.slider("Drift Bias", -5.0, 5.0, 0.0)
kurt = st.sidebar.slider("Fat Tail Factor (Kurtosis)", 0.0, 1.0, 0.7, help="0=Normal Distribution, 1=Extreme Crypto Events")

st.title("NeuroLOB: Generative Market Intelligence")

if 'sim_data' not in st.session_state: st.session_state.sim_data = None

if st.sidebar.button("🚀 Generate Scenario", type="primary"):
    steps = 1000 
    with st.spinner("Running Neural SDE + Physics Engine..."):
        st.session_state.sim_data = generate_enhanced_data(
            model, model_type, steps, vol, drift, price, days, tick_size, kurt
        )

tab1, tab2 = st.tabs(["📈 Dashboard", "⚖️ Statistical Evaluation"])

with tab1:
    if st.session_state.sim_data is not None:
        df = st.session_state.sim_data
        r, v, d, l = calculate_metrics(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{r:.2%}")
        c2.metric("Annual Volatility", f"{v:.1f}%")
        c3.metric("Max Drawdown", f"{d:.2%}")
        c4.metric("Est. Volume", f"${l/1e6:.1f}M")
        
        rule = '4H' if days > 7 else ('1H' if days > 1 else '15min')
        df_ohlc = df.set_index('timestamp').resample(rule).agg(
            {'price': ['first', 'max', 'min', 'last'], 'amount': 'sum'}
        ).dropna()
        df_ohlc.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df_ohlc.index, open=df_ohlc['Open'], high=df_ohlc['High'], low=df_ohlc['Low'], close=df_ohlc['Close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_ohlc.index, y=df_ohlc['Volume'], marker_color='#5c6bc0', name="Vol"), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Click 'Generate Scenario' in the sidebar to start.")

with tab2:
    if st.session_state.sim_data is not None:
        st.header("Deep Statistical Evaluation")
        synth_rets = st.session_state.sim_data['final_ret'].replace([np.inf, -np.inf], 0).fillna(0)
        
        if df_real is not None and 'price_ret' in df_real.columns:
            real_rets = df_real['price_ret'].replace([np.inf, -np.inf], 0).fillna(0)
            if len(real_rets) > len(synth_rets):
                real_rets = real_rets.sample(n=len(synth_rets), replace=False)
        else:
            real_rets = pd.Series(np.random.normal(0, synth_rets.std(), len(synth_rets)))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 1. Returns Distribution (Fat Tails)")
            combined_data = np.concatenate([synth_rets, real_rets])
            p01, p99 = np.percentile(combined_data, [1, 99])
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=synth_rets, name='Synthetic', xbins=dict(start=p01, end=p99, size=(p99-p01)/50), opacity=0.6, histnorm='probability density', marker_color='#EF553B'))
            fig_dist.add_trace(go.Histogram(x=real_rets, name='Benchmark', xbins=dict(start=p01, end=p99, size=(p99-p01)/50), opacity=0.5, histnorm='probability density', marker_color='#00CC96'))
            fig_dist.update_layout(template="plotly_dark", barmode='overlay', yaxis_type="log", xaxis_title="Return Size", yaxis_title="Log Probability Density")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with c2:
            st.markdown("### 2. Volatility Clustering (Memory)")
            acf_s = acf(np.abs(synth_rets), nlags=40, fft=True)
            acf_r = acf(np.abs(real_rets), nlags=40, fft=True)
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(y=acf_s, name='Synthetic', marker_color='#EF553B'))
            fig_acf.add_trace(go.Scatter(y=acf_r, name='Benchmark', line=dict(color='#00CC96', width=2)))
            fig_acf.update_layout(template="plotly_dark", yaxis_range=[-0.05, 1.0], xaxis_title="Lag", yaxis_title="Autocorrelation")
            st.plotly_chart(fig_acf, use_container_width=True)
            
        st.markdown("### Quantitative Scores")
        ks_stat, _ = ks_2samp(synth_rets, real_rets)
        wd_stat = wasserstein_distance(synth_rets, real_rets)
        kurt_s = synth_rets.kurtosis()
        kurt_r = real_rets.kurtosis()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Syn Kurtosis", f"{kurt_s:.2f}", delta=f"{kurt_s - kurt_r:.2f} vs Real", delta_color="off")
        m2.metric("Real Kurtosis", f"{kurt_r:.2f}")
        m3.metric("KS Distance", f"{ks_stat:.3f}")
        m4.metric("Wasserstein Dist", f"{wd_stat*1000:.3f}")