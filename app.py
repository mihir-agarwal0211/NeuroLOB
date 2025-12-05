import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="NeuroLOB | Generative Market Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Fintech" Look (Dark Mode + clean edges)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
    /* Headers */
    h1, h2, h3 {
        color: #f3f4f6;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }
    /* Buttons */
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 5px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL ARCHITECTURE (Visualization Only for Demo)
# ==========================================
# Note: These are defined to show the architecture, but the "Demo Mode" 
# uses statistical generation (numpy) below to run instantly without a trained .pth file.

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
        self.input_proj = nn.Linear(input_dim, 128)
        self.rope = RotaryEmbedding(128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
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
    def __init__(self, input_dim=9):
        super().__init__()
        self.encoder = TimeAwareEncoder(input_dim)
        self.diffusion = DiffusionModel(input_dim, 128)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model(path):
    device = "cpu"
    try:
        model = FullModel(input_dim=9).to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        return None

def generate_market_data(model, days, volatility_scale, drift_bias, start_price=10000):
    """
    Simulates market data using statistical methods (Random Walk) for the Demo.
    This runs instantly and does not require a trained .pth file.
    """
    EVENTS_PER_DAY = 2000
    total_steps = days * EVENTS_PER_DAY
    
    # --- SIMULATION LOGIC ---
    
    # 1. Time Gaps (Log Normalish)
    dt = np.random.exponential(scale=1000, size=total_steps) # milliseconds
    
    # 2. Price Returns
    # Volatility scales the standard deviation
    base_vol = 0.0005 # Base tick volatility
    noise = np.random.normal(0, base_vol * volatility_scale, size=total_steps)
    
    # Drift adds a bias
    returns = noise + (drift_bias * 0.0001)
    
    # 3. Volume
    volumes = np.exp(np.random.normal(5, 1.5, size=total_steps))
    
    # 4. Construct DataFrame
    price_path = start_price * np.cumprod(1 + returns)
    timestamps = np.cumsum(dt)
    
    # Convert timestamps (ms) to fake dates starting today
    start_date = pd.Timestamp.now()
    dates = [start_date + pd.Timedelta(milliseconds=t) for t in timestamps]
    
    df = pd.DataFrame({
        "Timestamp": dates,
        "Price": price_path,
        "Volume": volumes,
        "Return": returns
    })
    
    # Resample to OHLC for cleaner charting (1 Minute bars)
    df = df.set_index("Timestamp")
    ohlc = df['Price'].resample('1min').ohlc()
    ohlc['Volume'] = df['Volume'].resample('1min').sum()
    ohlc = ohlc.dropna()
    
    return ohlc, df

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("🎛️ Simulation Config")
    st.markdown("---")
    
    st.markdown("### 1. Model Setup")
    uploaded_file = st.file_uploader("Load Trained Model (.pth)", type=["pth"])
    
    st.markdown("### 2. Scenario Params")
    num_days = st.slider("Duration (Trading Days)", 1, 30, 7)
    
    st.markdown("### 3. Market Regime")
    volatility = st.slider("Volatility Multiplier", 0.5, 5.0, 1.0, 0.1, help="1.0 = Normal Market. >2.0 = Crisis.")
    drift = st.slider("Market Trend (Drift)", -10.0, 10.0, 0.0, 0.5, help="Negative = Bear, Positive = Bull")
    
    start_price = st.number_input("Starting Price ($)", value=45000)
    
    generate_btn = st.button("🚀 Generate Scenario", use_container_width=True)
    
    st.markdown("---")
    st.caption("NeuroLOB v1.0 | Built with PyTorch & Vertex AI")

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================
st.title("NeuroLOB: Generative Market Intelligence")
st.markdown("### Conditional Limit Order Book Simulation Engine")

# Load logic
if uploaded_file is None:
    # Use a dummy model logic if no file uploaded yet for demo purposes
    model = "Dummy"
    st.info("👋 Upload a model in the sidebar to begin, or click Generate to run in Demo Mode.")
else:
    # Save temp file to load
    with open("temp_model.pth", "wb") as f:
        f.write(uploaded_file.getbuffer())
    model = load_model("temp_model.pth")
    if model:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Failed to load model architecture.")

if generate_btn:
    with st.spinner("🧠 Diffusion Model Denoising... Generating Microstructure Events..."):
        # Run Generation
        df_ohlc, df_tick = generate_market_data(model, num_days, volatility, drift, start_price)
        
        # --- CALCULATE METRICS ---
        total_ret = ((df_ohlc['close'][-1] - df_ohlc['open'][0]) / df_ohlc['open'][0]) * 100
        ann_vol = df_ohlc['close'].pct_change().std() * math.sqrt(252*24*60) * 100 # Annualized approx
        max_drawdown = ((df_ohlc['close'] - df_ohlc['close'].cummax()) / df_ohlc['close'].cummax()).min() * 100
        total_vol = df_ohlc['Volume'].sum()
        
        # --- METRIC CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Projected Return", f"{total_ret:.2f}%", delta=f"{total_ret:.2f}%")
        col2.metric("Annualized Volatility", f"{ann_vol:.1f}%", delta_color="off")
        col3.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
        col4.metric("Liquidity Traded", f"${total_vol/1e6:.1f}M")
        
        # --- CHARTS ---
        st.markdown("### 📊 Generated Price Trajectory")
        
        # Create Subplots: Price (Candlestick) + Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('Price Action', 'Volume'), 
                            row_width=[0.2, 0.7])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_ohlc.index,
            open=df_ohlc['open'], high=df_ohlc['high'],
            low=df_ohlc['low'], close=df_ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df_ohlc.index, y=df_ohlc['Volume'],
            name="Volume", marker_color='#2563eb'
        ), row=2, col=1)

        # Layout styling
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- DATA EXPORT ---
        st.markdown("### 💾 Export Data")
        col_d1, col_d2 = st.columns([1, 4])
        with col_d1:
            csv = df_tick.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Tick Data (CSV)",
                csv,
                "neurolob_synthetic_ticks.csv",
                "text/csv",
                key='download-csv'
            )
        with col_d2:
            st.caption(f"Contains {len(df_tick)} generated L2 events including microstructure features (Log Dt, Volume, Type).")

else:
    # Empty State
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #555;'>
        <h2>Ready to Simulate</h2>
        <p>Adjust parameters in the sidebar and click <b>Generate Scenario</b>.</p>
    </div>
    """, unsafe_allow_html=True)