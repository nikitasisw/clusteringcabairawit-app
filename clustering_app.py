import streamlit as st
from streamlit_option_menu import option_menu
import home, pertanian_6, harga_5, tentang
from pathlib import Path

# =============== PAGE CONFIG ===============
st.set_page_config(page_title="Clustering Cabai Rawit", layout="wide")

# =============== LOAD CSS (eksternal + fallback) ===============
css_path = Path("assets/sidebar.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
      section[data-testid="stSidebar"] > div {padding:18px 16px 24px;background:linear-gradient(180deg,#E75480 0%,#FF8799 100%);border-right:none;}
      section[data-testid="stSidebar"] .block-container {padding-top:8px;}
      .sb-card{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.25);border-radius:18px;padding:14px 14px 10px;box-shadow:inset 0 6px 18px rgba(0,0,0,.08);color:#fff;}
      .sb-card h3{margin:0;font-weight:800;letter-spacing:.2px;}
      .sb-sep{height:1px;background:rgba(255,255,255,.5);margin:12px 0 6px;border-radius:1px;}
      ul.nav{gap:6px!important;}
      ul.nav a{border-radius:14px!important;padding:10px 12px!important;font-weight:600;line-height:1.1;color:#fff!important;}
      ul.nav a:hover{background:rgba(255,255,255,.25)!important;transform:translateY(-1px);}
      ul.nav a.active{background:#FFE3EA!important;color:#7a1131!important;box-shadow:0 6px 18px rgba(0,0,0,.08);}
      .nav-link i{color:#FFD166!important;font-size:18px!important;margin-right:8px;}
    </style>
    """, unsafe_allow_html=True)

# =============== CONSTANTS & STATE INIT ===============
PAGES = ['Halaman Utama', 'Data Pertanian', 'Harga Cabai Rawit', 'Tentang Kami']
# PAGES = ['Halaman Utama', 'Data Pertanian', 'Harga Cabai Rawit', 'Pertanian Dan Harga', 'Tentang Kami']
METHOD_OPTIONS = ['K-Means', 'Hierarchical Clustering']

if 'page_idx' not in st.session_state:
    st.session_state.page_idx = 0  # default: Halaman Utama
if 'method_sel' not in st.session_state:
    st.session_state.method_sel = METHOD_OPTIONS[0]
if 'k_sel' not in st.session_state:
    st.session_state.k_sel = 2

# =============== SIDEBAR ===============
def render_sidebar() -> str:
    with st.sidebar:
        # Header card
        st.markdown("""
        <div class="sb-card">
          <h3>💬 Clustering<br/>Cabai Rawit</h3>
          <div class="sb-sep"></div>
          <div style="font-size:12px;opacity:.95">Eksplorasi data &amp; grouping wilayah</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        # Menu dengan state (anti klik 2x)
        selected = option_menu(
            menu_title=None,
            options=PAGES,
            icons=['house-fill', 'grid-3x3-gap-fill', 'tags-fill', 'info-circle-fill'],
            # icons=['house-fill', 'grid-3x3-gap-fill', 'tags-fill', 'diagram-3-fill', 'book-fill', 'info-circle-fill'],
            default_index=st.session_state.page_idx,
            key="main_nav",
            styles={
                "container": {"padding": "0px", "background-color": "transparent"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "transparent"},
                "nav-link-selected": {"background-color": "#FFE3EA", "color": "#7a1131", "font-weight": "800"},
            }
        )

        # Update index hanya jika berubah
        new_idx = PAGES.index(selected)
        if new_idx != st.session_state.page_idx:
            st.session_state.page_idx = new_idx

        st.markdown("---")
        return selected

app = render_sidebar()

# ================== ROUTING KONTEN ==================
if app == 'Halaman Utama':
    home.app()
elif app == 'Data Pertanian':
    pertanian_6.app()
elif app == 'Harga Cabai Rawit':
    harga_5.app()
# elif app == 'Pertanian Dan Harga':
#     pertanian_harga.app()
# elif app == 'Panduan Pengguna':
#     user_guide.app()
elif app == 'Tentang Kami':
    tentang.app()
