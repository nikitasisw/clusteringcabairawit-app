import io, re, json, unicodedata, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time
import matplotlib.patheffects as pe

# Peta
import folium
from streamlit_folium import st_folium

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import hashlib


# ==== Export helpers (PNG & Excel) ====
def _fig_to_png_bytes(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def _df_to_excel_bytes(df: pd.DataFrame, sheet_name="data"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        # jika MultiIndex kolom, pipihkan agar rapi di Excel
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            df_export = df.copy()
            df_export.columns = ["_".join(map(str, c)).strip() for c in cols.to_list()]
        else:
            df_export = df
        if isinstance(df_export.index, pd.MultiIndex):
            df_export = df_export.reset_index()
        df_export.to_excel(w, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.read()

def download_png_button(fig, filename_base: str, help_txt=None, key: str=None):
    try:
        png = _fig_to_png_bytes(fig)
        st.download_button(
            "⬇️ Download PNG",
            data=png,
            file_name=f"{filename_base}.png",
            mime="image/png",
            help=help_txt or "Unduh gambar visualisasi sebagai PNG",
            use_container_width=True,
            key=key or f"png_{filename_base}"
        )
    except Exception as e:
        st.caption(f"Gagal menyiapkan unduhan PNG: {e}")

def download_excel_button(df: pd.DataFrame, filename_base: str, sheet_name="data", help_txt=None, key: str=None):
    try:
        xls = _df_to_excel_bytes(df, sheet_name=sheet_name)
        st.download_button(
            "⬇️ Download Excel",
            data=xls,
            file_name=f"{filename_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=help_txt or "Unduh tabel sebagai Excel",
            use_container_width=True,
            key=key or f"xlsx_{filename_base}"
        )
    except Exception as e:
        st.caption(f"Gagal menyiapkan unduhan Excel: {e}")

def show_fig_and_download(fig, name_base, key=None):
    """Tampilkan matplotlib fig + tombol unduh PNG, sekaligus tetap terekam ke PDF report."""
    st.pyplot(fig)
    download_png_button(fig, name_base, key=key)



# Dendrogram (opsional)
try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import openpyxl
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False


from pathlib import Path

# Path template & dataset contoh (sesuaikan nama file jika beda)
TEMPLATE_PATHS = {
    "Provinsi":       str(Path("dataset") / "template_dataset_pertanian_provinsi.xlsx"),
    "Kabupaten/Kota": str(Path("dataset") / "template_dataset_pertanian_kabupaten.xlsx"),
}
DATASET_PATHS = {
    "Provinsi":       str(Path("dataset") / "dataset_pertanian_provinsi.xlsx"),
    "Kabupaten/Kota": str(Path("dataset") / "dataset_pertanian_kabupaten.xlsx"),
}

def _load_template_from_path(level: str) -> bytes:
    """Baca template .xlsx dari path di atas dan kembalikan bytes-nya."""
    p = Path(TEMPLATE_PATHS.get(level, ""))
    if not p.exists():
        raise FileNotFoundError(f"Template tidak ditemukan: {p}")
    return p.read_bytes()

def _load_dataset_from_path(level: str) -> pd.DataFrame:
    """(Tetap) baca dataset contoh dari path DATASET_PATHS."""
    p = Path(DATASET_PATHS.get(level, ""))
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    return pd.read_excel(p, engine="openpyxl")


# ------------------------ Konstanta UI/Kategori ------------------------
PALETTE = {
    "sangat rendah": "#d73027",
    "rendah": "#fc8d59",
    "cukup rendah": "#fee090",
    "sedang": "#ffffbf",
    "cukup tinggi": "#e0f3f8",
    "tinggi": "#91bfdb",
    "sangat tinggi": "#4575b4",
}
CAT_ORDER = ["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi","Sangat Tinggi"]
CAT_RANK  = {name:i for i,name in enumerate(CAT_ORDER)}
LEVEL_OPTIONS = ["Provinsi", "Kabupaten/Kota"]

# Bobot (untuk ranking cluster)
FEATURE_WEIGHTS = {"luas":0.15, "produksi":0.35, "produkt":0.50}
def _feature_weight_for_col(col: str, weights: dict[str, float]) -> float:
    cl = col.lower()
    for k,w in weights.items():
        if cl.startswith(k): return float(w)
    return 1.0


# ============================== Utils ==============================
def _detect_feature_cols(df: pd.DataFrame):
    luas  = [c for c in df.columns if re.search(r"^luas|_luas|luas_", c, flags=re.I)]
    prod  = [c for c in df.columns if re.search(r"^produksi|_produksi|produksi_", c, flags=re.I)]
    prdx  = [c for c in df.columns if re.search(r"^produkt|_produkt|produkt_", c, flags=re.I)]
    return luas, prod, prdx

def detect_granularity(df: pd.DataFrame) -> str:
    """Deteksi kasar granularitas waktu agar tidak error saat dipanggil."""
    cols = [str(c).lower() for c in df.columns]
    # Ada kolom bertahun di nama kolom
    if any(re.search(r'(19|20)\d{2}', c) for c in cols):
        return "tahunan"
    # Ada kolom bulan
    if any(("bulan" in c) or re.search(r"\bmonth\b", c) for c in cols):
        return "bulanan"
    # Ada kolom tanggal/waktu
    if any(("tanggal" in c) or ("date" in c) or ("waktu" in c) or ("time" in c) for c in cols):
        return "harian"
    return "tahunan"


def render_pca_scatter_visual(X_scaled: np.ndarray, df_out: pd.DataFrame, labels: np.ndarray):
    """PCA 2D hanya untuk visualisasi komposisi cluster (tidak memengaruhi hasil)."""
    if X_scaled is None or len(X_scaled) == 0 or df_out is None or df_out.empty:
        st.info("Belum ada data untuk visualisasi PCA."); 
        return

    try:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X_scaled)
    except Exception as e:
        st.warning(f"Gagal menghitung PCA untuk visualisasi: {e}")
        return

    # siapkan warna per kategori
    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    colors = {
        "Sangat Rendah":"#d73027","Rendah":"#fc8d59","Cukup Rendah":"#fee090",
        "Sedang":"#ffffbf","Cukup Tinggi":"#e0f3f8","Tinggi":"#91bfdb","Sangat Tinggi":"#4575b4"
    }

    df_plot = pd.DataFrame({
        "PC1": X2[:,0], "PC2": X2[:,1],
        "Kategori": df_out["Kategori"].astype(str).values
    })

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for cat in cat_order:
        sub = df_plot[df_plot["Kategori"] == cat]
        if sub.empty: 
            continue
        ax.scatter(sub["PC1"], sub["PC2"], s=26, alpha=0.85, 
                   color=colors.get(cat, "#888"), label=cat, edgecolor="white", linewidth=0.4)

    ax.set_title("Visualisasi PCA 2D (Warna = Kategori Cluster)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax.grid(True, linestyle="--", alpha=.35)
    ax.legend(title="Kategori", ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0,0,0.82,1])
    # st.pyplot(fig)
    show_fig_and_download(
        fig,
        f"pca2d_{st.session_state.get('method_used','unknown')}_k{st.session_state.get('k_used','?')}"
    )



def _composite_score(df: pd.DataFrame, cols: list[str] | None = None) -> pd.Series:
    if cols:
        Z=[]
        for c in cols:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                mu, sd = v.mean(), v.std(ddof=0) or 1.0
                Z.append((v-mu)/sd)
        if not Z: raise ValueError("Tidak ada kolom valid untuk skor.")
        return pd.concat(Z, axis=1).mean(axis=1)
    luas, prod, prdx = _detect_feature_cols(df); parts=[]
    for fam in [luas, prod, prdx]:
        if fam:
            s = pd.to_numeric(df[fam].mean(axis=1, skipna=True), errors="coerce")
            mu, sd = s.mean(), s.std(ddof=0) or 1.0
            parts.append((s-mu)/sd)
    if not parts: raise ValueError("Tidak ada kolom luas/produksi/produkt untuk skor.")
    return pd.concat(parts, axis=1).mean(axis=1)

# =================== Prepare Numeric (tanpa PCA otomatis) ===================
def _prepare_numeric(df: pd.DataFrame):
    """
    - Pilih kolom domain (luas/produksi/produkt) termasuk varian bertahun.
    - Imputasi mean + StandardScaler.
    - TIDAK melakukan PCA otomatis.
    """
    cols_year  = [c for c in df.columns if re.search(r"(luas|produksi|produkt).*?(19|20)\d{2}", c, flags=re.I)]
    cols_plain = [c for c in df.columns if re.search(r"^(luas|produksi|produkt)", c, flags=re.I) and not re.search(r"(19|20)\d{2}", c, flags=re.I)]
    feature_cols = cols_year if cols_year else cols_plain
    if not feature_cols:
        raise ValueError("Tidak ditemukan kolom fitur untuk clustering.")

    X = df[feature_cols].copy()
    imp = SimpleImputer(strategy="mean"); X_imp = imp.fit_transform(X.values)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, feature_cols, df.index

# ==================== Ranking Weighted Z-Score ====================
def ranking_weighted_zscore(df: pd.DataFrame, labels: np.ndarray, feat_cols: list[str],
                            weights: dict[str, float] | None = None):
    if weights is None: weights = FEATURE_WEIGHTS
    used = [c for c in feat_cols if c in df.columns]
    if not used: raise ValueError("feat_cols kosong / tidak cocok.")

    Z={}
    for c in used:
        v = pd.to_numeric(df[c], errors="coerce"); mu, sd = v.mean(), v.std(ddof=0) or 1.0
        Z[c] = (v-mu)/sd
    Z = pd.DataFrame(Z, index=df.index)

    col_w = {c:_feature_weight_for_col(c, weights) for c in used}
    denom = sum(abs(col_w[c]) for c in used) or 1.0
    row_score = (Z[used]*pd.Series(col_w)).sum(axis=1)/denom

    tmp = pd.DataFrame({"Cluster":labels, "_score":row_score})
    cluster_scores = tmp.groupby("Cluster")["_score"].mean()
    ranked = cluster_scores.sort_values(ascending=True).index.tolist()  # rendah → tinggi
    return ranked, cluster_scores

# ========================= Normalisasi Urutan =========================
def normalize_cluster_order(df_out: pd.DataFrame, fitur_list):
    if "Kategori" not in df_out.columns: raise ValueError("Kolom 'Kategori' tidak ada.")
    fitur_list = list(fitur_list or [])
    num_cols = [c for c in fitur_list if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])]
    if not num_cols:
        patt = re.compile(r"(luas|produksi|produkt)", flags=re.I)
        num_cols = [c for c in df_out.columns if patt.search(str(c)) and pd.api.types.is_numeric_dtype(df_out[c])]

    try:
        score_row = _composite_score(df_out, cols=num_cols)
        temp = df_out.copy(); temp["_score_combo__"] = score_row
        cat_score = temp.groupby("Kategori")["_score_combo__"].mean()
        ordered = cat_score.sort_values(ascending=True).index.tolist()
    except Exception:
        ordered = sorted(df_out["Kategori"].dropna().unique().tolist(), key=lambda x: CAT_RANK.get(str(x), 999))

    st.session_state["ordered_categories"] = ordered
    out = df_out.copy()
    out["Kategori"] = pd.Categorical(out["Kategori"], categories=ordered, ordered=True)
    return out

def get_ordered_categories(desc: bool=False):
    base = st.session_state.get("ordered_categories", CAT_ORDER)
    use  = [c for c in base if c in CAT_ORDER]
    return list(reversed(use)) if desc else use

# ======================== Clustering Utama ========================
def _fit_model(X, method: str, k: int):
    if method == "K-Means":
        return (MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=5).fit(X)
                if X.shape[0]>10000 else KMeans(n_clusters=k, random_state=42, n_init=10).fit(X))
    if X.shape[0] > 2000: raise RuntimeError("Data terlalu besar untuk Hierarchical (>2000).")
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)



def run_clustering(df: pd.DataFrame, method: str, k: int):
    if not isinstance(k,(int,np.integer)): k=int(np.squeeze(k))
    if k<2 or k>7: raise ValueError("Jumlah cluster (k) harus 2–7.")

    # Siapkan fitur (scaled)
    X_scaled, used_feature_cols, _ = _prepare_numeric(df)

    # (Opsional) PCA + K-Means
    pca_info = {"used": False, "n_components": 0, "explained": None}
    X_clust = X_scaled
    method_for_fit = method
    if method == "PCA + K-Means":
        max_comp = min(6, X_scaled.shape[1])
        if max_comp < 2: max_comp = min(2, X_scaled.shape[1])
        pca_probe = PCA(n_components=max_comp).fit(X_scaled)
        csum = np.cumsum(pca_probe.explained_variance_ratio_)
        n_opt = np.searchsorted(csum, 0.90) + 1
        n_opt = int(np.clip(n_opt, 2, max_comp))
        pca = PCA(n_components=n_opt, random_state=42)
        X_clust = pca.fit_transform(X_scaled)
        pca_info = {"used": True, "n_components": n_opt, "explained": float(np.sum(pca.explained_variance_ratio_))}
        method_for_fit = "K-Means"

    # Fit
    model  = _fit_model(X_clust, method_for_fit, k)
    labels = np.asarray(model.labels_ if hasattr(model,"labels_") else model.fit_predict(X_clust), dtype=int)

    # Metrik
    try: sil = round(silhouette_score(X_clust, labels, sample_size=min(3000, X_clust.shape[0]), random_state=42),4)
    except Exception: sil = float("nan")
    try: dbi = round(davies_bouldin_score(X_clust, labels),4)
    except Exception: dbi = float("nan")
    results_df = pd.DataFrame([{"Metode":method, "Jumlah Cluster":len(np.unique(labels)), "Silhouette":sil, "Davies-Bouldin":dbi}])

    # Ranking cluster (skala asli)
    ranked, cluster_scores = ranking_weighted_zscore(df, labels, used_feature_cols, FEATURE_WEIGHTS)

    LABEL_SETS = {
        2:["Rendah","Tinggi"],
        3:["Rendah","Sedang","Tinggi"],
        4:["Sangat Rendah","Rendah","Tinggi","Sangat Tinggi"],
        5:["Sangat Rendah","Rendah","Sedang","Tinggi","Sangat Tinggi"],
        6:["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi"],
        7:["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi","Sangat Tinggi"],
    }
    chosen_labels = LABEL_SETS[len(np.unique(labels))]
    cluster_label_map = {int(c): chosen_labels[i] for i,c in enumerate(ranked)}

    df_out = df.copy()
    df_out["Cluster"]  = labels
    df_out["Kategori"] = df_out["Cluster"].map(cluster_label_map)
    df_out = normalize_cluster_order(df_out, fitur_list=used_feature_cols)

    # simpan state
    st.session_state.update({
        "ordered_categories": list(df_out["Kategori"].cat.categories),
        "cluster_label_map": cluster_label_map,
        "method_used": method,
        "k_used": k,
        "diag_used_cols": used_feature_cols,
        "diag_counts": {fam: sum(c.lower().startswith(fam) for c in used_feature_cols) for fam in ["luas","produksi","produkt"]},
        "diag_nan": {fam: float(np.isnan(pd.to_numeric(df[[c for c in used_feature_cols if c.lower().startswith(fam)]].values.ravel(), errors="coerce")).mean()) if any(c.lower().startswith(fam) for c in used_feature_cols) else np.nan for fam in ["luas","produksi","produkt"]},
        "cluster_scores": cluster_scores.to_dict(),
        "pca_info": pca_info
    })
    return results_df, df_out, X_clust, labels, cluster_label_map


# ===================== Evaluasi & Visualisasi =====================
def render_cluster_performance(X_scaled, method_sel, k_sel=None):
    """
    Menampilkan evaluasi performa clustering (Silhouette & Davies–Bouldin) untuk k=2..7.
    - Angka nilai ditulis di atas setiap titik (dengan stroke putih agar tidak 'hilang').
    - Sumbu-Y diberi padding agar label tidak terpotong oleh batas grafik.
    """

    st.markdown("## 📊 Evaluasi Performa Clustering")
    st.caption("Silhouette lebih tinggi lebih baik; Davies–Bouldin lebih rendah lebih baik.")

    if X_scaled is None or len(X_scaled) == 0:
        st.info("Tidak ada data untuk evaluasi.")
        return

    # --- parameter k
    k_sel = int(k_sel or st.session_state.get("k_used", 2))
    k_values = list(range(2, 7 + 1))

    # --- hitung skor untuk semua k (agar bisa diplot sekaligus)
    silhouette_scores, dbi_scores = [], []
    t0 = time.time()
    for k in k_values:
        try:
            if method_sel in ["K-Means", "KMeans"]:
                model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            else:
                model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_scaled)
            labels = getattr(model, "labels_", None)
            if labels is None:
                labels = model.fit_predict(X_scaled)

            sil = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
        except Exception:
            sil, dbi = np.nan, np.nan
        silhouette_scores.append(sil)
        dbi_scores.append(dbi)
    elapsed = time.time() - t0

    # --- k optimum indikatif (berdasarkan Silhouette)
    sil_arr = np.asarray(silhouette_scores, dtype=float)
    best_k = None if np.isnan(sil_arr).all() else int(k_values[int(np.nanargmax(sil_arr))])

    # --- ambil nilai untuk k yang dipilih
    idx = k_sel - 2
    sil_k = silhouette_scores[idx] if 0 <= idx < len(silhouette_scores) else np.nan
    dbi_k = dbi_scores[idx] if 0 <= idx < len(dbi_scores) else np.nan

    # === METRIK UTAMA ===
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Metode", method_sel.replace("+", "").replace("-", ""))
    with c2:
        st.metric(
            "Silhouette Score",
            f"{sil_k:.4f}" if not np.isnan(sil_k) else "—",
            help="Semakin tinggi semakin baik; menunjukkan seberapa jelas pemisahan antar cluster."
        )
    with c3:
        st.metric("Waktu Proses", f"{elapsed:.4f} detik")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Jumlah Cluster", f"{k_sel}")
    with c5:
        st.metric(
            "Davies–Bouldin Index",
            f"{dbi_k:.4f}" if not np.isnan(dbi_k) else "—",
            help="Semakin rendah semakin baik; mengukur seberapa kompak cluster."
        )
    with c6:
        pinfo = st.session_state.get("pca_info", {"used": False})
        st.metric("PCA", "Tidak dipakai" if not pinfo.get("used") else f"{pinfo.get('n_components')} komponen")

    # === GRAFIK ===
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Kurva
    ax[0].plot(k_values, silhouette_scores, "o-", lw=2)
    ax[1].plot(k_values, dbi_scores, "o-", lw=2, color="orange")

    # Garis k terpilih
    ax[0].axvline(k_sel, color="red", ls="--", lw=2)
    ax[1].axvline(k_sel, color="red", ls="--", lw=2)

    # --- Padding sumbu-Y supaya label angka tidak terpotong
    def _pad_ylim(ax_, values, top_pad=0.18, bottom_pad=0.06):
        vals = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            span = max(hi - lo, 0.05)
            ax_.set_ylim(lo - span * bottom_pad, hi + span * top_pad)

    _pad_ylim(ax[0], silhouette_scores)
    _pad_ylim(ax[1], dbi_scores)

    # --- Tulis angka dengan offset & stroke putih
    stroke = [pe.withStroke(linewidth=2.5, foreground="white")]
    for k, s in zip(k_values, silhouette_scores):
        if not np.isnan(s):
            ax[0].annotate(f"{s:.3f}", (k, s), xytext=(0, 8), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9, zorder=5, path_effects=stroke)

    for k, d in zip(k_values, dbi_scores):
        if not np.isnan(d):
            ax[1].annotate(f"{d:.3f}", (k, d), xytext=(0, 8), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9, zorder=5, path_effects=stroke)

    # Label & grid
    ax[0].set_title("Silhouette Score");     ax[0].set_xlabel("Jumlah Cluster"); ax[0].set_ylabel("Score"); ax[0].grid(alpha=.3)
    ax[1].set_title("Davies–Bouldin Index"); ax[1].set_xlabel("Jumlah Cluster"); ax[1].set_ylabel("Index"); ax[1].grid(alpha=.3)

    plt.tight_layout()
    # st.pyplot(fig)
    show_fig_and_download(fig, f"cluster_performance_{method_sel.replace(' ','_')}_k{k_sel}")

    # === Keterangan tambahan
    st.caption(
        f"Menampilkan hasil untuk **k = {k_sel}** (garis merah putus–putus). "
        "Silhouette lebih tinggi → cluster makin terpisah; Davies–Bouldin lebih rendah → cluster makin kompak."
    )
    if best_k is not None:
        st.caption(f"K optimum indikatif (berdasar Silhouette): **k = {best_k}**.")



def reshape_long_format(df_out: pd.DataFrame):
    luas = [c for c in df_out.columns if re.search(r"luas.*?(19|20)\d{2}", c, flags=re.I)]
    prod = [c for c in df_out.columns if re.search(r"produksi.*?(19|20)\d{2}", c, flags=re.I)]
    prdx = [c for c in df_out.columns if re.search(r"produkt.*?(19|20)\d{2}", c, flags=re.I)]
    frames=[]
    for subset, fitur in [(luas,"luas_areal"),(prod,"produksi"),(prdx,"produktivitas")]:
        if not subset: continue
        t = df_out.melt(id_vars=["Kategori"], value_vars=subset, var_name="Tahun", value_name="Nilai")
        t["Fitur"]=fitur; t["Tahun"]=t["Tahun"].astype(str).str.extract(r"(\d{4})"); frames.append(t)
    if not frames: return pd.DataFrame(columns=["Kategori","Tahun","Fitur","Nilai"])
    df_long = pd.concat(frames, ignore_index=True)
    df_long["Tahun"]=pd.to_numeric(df_long["Tahun"], errors="coerce")
    return df_long.dropna(subset=["Nilai"])


def render_boxplot(df_out: pd.DataFrame):

    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data numerik untuk boxplot.")
        return

    # urutan kategori
    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    df_long["Kategori"] = pd.Categorical(df_long["Kategori"], categories=cat_order, ordered=True)
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]

    color_map = {"luas_areal": "#F5B7B1", "produksi": "#82E0AA", "produktivitas": "#85C1E9"}
    fitur_unik = ["luas_areal", "produksi", "produktivitas"]
    tahun_list = sorted(df_long["tahun"].dropna().unique())

    st.markdown("## 📈 Analisis Distribusi Boxplot Fitur per Tahun")
    tahun_pilihan = st.selectbox("Pilih Tahun:", ["Seluruh Tahun"] + [str(t) for t in tahun_list], index=0)
    fitur_pilihan = st.multiselect("Pilih Fitur:", options=fitur_unik, default=fitur_unik)
    use_log = st.toggle("Gunakan skala logaritmik", value=False)

    # 👉 guard: kalau kosong, jangan lanjut (mencegah ncol=0)
    if not fitur_pilihan:
        st.warning("Pilih minimal satu fitur.")
        return

    # standarisasi per (fitur, tahun)
    df_filtered = df_long.copy()
    if tahun_pilihan != "Seluruh Tahun":
        df_filtered = df_filtered[df_filtered["tahun"] == int(tahun_pilihan)]

    df_filtered["nilai_standar"] = np.nan
    for fitur in fitur_pilihan:
        for th in df_filtered["tahun"].dropna().unique():
            m = (df_filtered["tahun"] == th) & (df_filtered["fitur"] == fitur)
            if m.sum() > 0:
                vals = df_filtered.loc[m, "nilai"].values.reshape(-1, 1)
                df_filtered.loc[m, "nilai_standar"] = StandardScaler().fit_transform(vals).ravel()

    # ================= Seluruh Tahun (grid) =================
    if tahun_pilihan == "Seluruh Tahun":
        n_years = len(tahun_list)
        n_cols = 3
        n_rows = int(np.ceil(n_years / n_cols)) if n_years else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = np.atleast_1d(axes).flatten()

        width = 0.25
        offsets = np.linspace(-width, width, len(fitur_pilihan)) if len(fitur_pilihan) > 1 else [0.0]
        ax_last = None

        for idx, year in enumerate(tahun_list):
            ax = axes[idx]
            data_y = df_filtered[df_filtered["tahun"] == year]
            if data_y.empty:
                ax.axis("off")
                continue

            cluster_order = data_y["kategori"].cat.categories
            for i, fitur in enumerate(fitur_pilihan):
                sub = data_y[data_y["fitur"] == fitur]
                vals = [sub[sub["kategori"] == c]["nilai_standar"].dropna().values for c in cluster_order]
                pos = np.arange(len(cluster_order)) + offsets[i]
                ax.boxplot(
                    vals, positions=pos, widths=0.18, patch_artist=True,
                    boxprops=dict(facecolor=color_map.get(fitur, "#ccc"), alpha=.85),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=2.5, alpha=.3),
                )

            ax.set_title(f"Tahun {int(year)}")
            ax.set_ylabel("Nilai Standarisasi")
            ax.grid(axis="y", linestyle="--", alpha=.35)

            counts = data_y.groupby("kategori").size().reindex(cluster_order, fill_value=0)
            ax.set_xticks(range(len(cluster_order)))
            ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cluster_order], fontsize=8)

            if use_log and (data_y["nilai_standar"].dropna() > 0).any():
                ax.set_yscale("log")

            ax_last = ax

        # matikan axes kosong
        for j in range(len(tahun_list), len(axes)):
            axes[j].axis("off")

        # Legend di dalam plot (kanan-atas) pada axis terakhir
        if ax_last is None:
            ax_last = axes[0]
        handles = [plt.Line2D([0], [0], lw=8, label=f.replace("_", " ").title(),
                              color=color_map.get(f, "#999")) for f in fitur_pilihan]
        ax_last.legend(handles=handles, title="Fitur", loc="upper right",
                       frameon=True, fancybox=True, borderpad=0.6, labelspacing=0.5)

        fig.tight_layout()
        # st.pyplot(fig)
        show_fig_and_download(fig, f"boxplot_all_years_{'_'.join(fitur_pilihan)}")
        return

    # ================= Satu Tahun =================
    year = int(tahun_pilihan)
    data_y = df_filtered[df_filtered["tahun"] == year]
    if data_y.empty:
        st.warning("Tidak ada data untuk tahun itu.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_order = data_y["kategori"].cat.categories

    width = 0.25
    offsets = np.linspace(-width, width, len(fitur_pilihan)) if len(fitur_pilihan) > 1 else [0.0]
    for i, fitur in enumerate(fitur_pilihan):
        sub = data_y[data_y["fitur"] == fitur]
        vals = [sub[sub["kategori"] == c]["nilai_standar"].dropna().values for c in cluster_order]
        pos = np.arange(len(cluster_order)) + offsets[i]
        ax.boxplot(
            vals, positions=pos, widths=0.18, patch_artist=True,
            boxprops=dict(facecolor=color_map.get(fitur, "#ccc"), alpha=.85),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=3, alpha=.3),
        )

    ax.set_title(f"Tahun {year}")
    ax.set_ylabel("Nilai Standarisasi")
    ax.grid(axis="y", linestyle="--", alpha=.35)

    counts = data_y.groupby("kategori").size().reindex(cluster_order, fill_value=0)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cluster_order], fontsize=9)

    if use_log and (data_y["nilai_standar"].dropna() > 0).any():
        ax.set_yscale("log")

    # Legend di dalam plot (kanan-atas)
    handles = [plt.Line2D([0], [0], lw=8, label=f.replace("_", " ").title(),
                          color=color_map.get(f, "#999")) for f in fitur_pilihan]
    ax.legend(handles=handles, title="Fitur", loc="upper right",
              frameon=True, fancybox=True, borderpad=0.6, labelspacing=0.5)

    plt.tight_layout()
    # st.pyplot(fig)
    show_fig_and_download(fig, f"boxplot_tahun_{year}_{'_'.join(fitur_pilihan)}")


def render_boxplot_combined(df_out: pd.DataFrame):

    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data numerik.")
        return

    # Normalisasi kolom + urutan kategori
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]
    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    df_long["kategori"] = pd.Categorical(df_long["kategori"], categories=cat_order, ordered=True)

    fitur_unik = ["luas_areal", "produksi", "produktivitas"]
    st.markdown("## 📈 Analisis Distribusi Boxplot Fitur Seluruh Tahun")
    fitur_pilihan = st.multiselect(
        "Pilih Fitur:", options=fitur_unik, default=fitur_unik, key="fitur_boxplot_combined"
    )
    use_log = st.toggle("Gunakan skala logaritmik", value=False, key="log_boxplot_combined")

    # Guard: wajib pilih minimal 1 fitur
    if not fitur_pilihan:
        st.warning("Pilih minimal satu fitur.")
        return

    # Standarisasi per fitur (lintas tahun)
    df_long["nilai_standar"] = np.nan
    for fitur in fitur_pilihan:
        m = (df_long["fitur"] == fitur)
        if m.sum() > 0:
            vals = df_long.loc[m, "nilai"].values.reshape(-1, 1)
            df_long.loc[m, "nilai_standar"] = StandardScaler().fit_transform(vals).ravel()

    color_map = {"luas_areal": "#F5B7B1", "produksi": "#82E0AA", "produktivitas": "#85C1E9"}

    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.25
    # Offset rapi: jika hanya 1 fitur, tepat di tengah
    offsets = np.linspace(-width, width, len(fitur_pilihan)) if len(fitur_pilihan) > 1 else [0.0]

    for i, fitur in enumerate(fitur_pilihan):
        sub = df_long[df_long["fitur"] == fitur]
        vals = [sub[sub["kategori"] == c]["nilai_standar"].dropna().values for c in cat_order]
        pos = np.arange(len(cat_order)) + offsets[i]
        ax.boxplot(
            vals, positions=pos, widths=0.18, patch_artist=True,
            boxprops=dict(facecolor=color_map.get(fitur, "#ccc"), alpha=.85),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=2.5, alpha=.3),
        )

    ax.set_title("Distribusi Fitur vs Kategori (All Years)")
    ax.set_ylabel("Nilai Standarisasi")
    ax.grid(axis="y", linestyle="--", alpha=.35)

    counts = df_long.groupby("kategori").size().reindex(cat_order, fill_value=0)
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cat_order], fontsize=9)

    if use_log and (df_long["nilai_standar"].dropna() > 0).any():
        ax.set_yscale("log")

    # Legend di dalam plot (kanan atas) → hindari fig.legend/ncol=0
    handles = [
        plt.Line2D([0], [0], lw=8, label=f.replace("_", " ").title(),
                   color=color_map.get(f, "#999"))
        for f in fitur_pilihan
    ]
    ax.legend(handles=handles, title="Fitur", loc="upper right",
              frameon=True, fancybox=True, borderpad=0.6, labelspacing=0.5)

    plt.tight_layout()
    # st.pyplot(fig)
    show_fig_and_download(fig, f"boxplot_combined_{'_'.join(fitur_pilihan)}")



def render_dendrogram_adaptive(X_scaled, level_sel, title_suffix="", k_sel=None,
                               max_visible_labels=60, max_label_len=18):
    """
    Dendrogram adaptif:
    - Level 'Provinsi'  -> tampilkan label wilayah (ringkas + tampil tiap-N).
    - Level 'Kabupaten/Kota' -> sembunyikan label (lebih bersih).
    """
    if not SCIPY_OK:
        st.info("scipy belum terpasang."); return
    if X_scaled is None or len(X_scaled)==0:
        st.warning("Tidak ada data."); return

    
    n = X_scaled.shape[0]

    # ---------- siapkan label bila perlu (hanya untuk Provinsi) ----------
    labels_all = None
    if str(level_sel).lower().startswith("prov"):
        df_auto = st.session_state.get("df_out")
        if isinstance(df_auto, pd.DataFrame) and len(df_auto)==n:
            lokasi_col = next((c for c in df_auto.columns
                               if str(c).lower() in ["provinsi","kabupaten/kota","kabupaten","kota","lokasi"]), None)
            if lokasi_col is not None:
                def _shorten(s, L=18):
                    s = (str(s).replace("Daerah Khusus Ibukota","DKI")
                                   .replace("Provinsi","")
                                   .replace("Kabupaten","Kab.")
                                   .replace("Kabupaten.","Kab.")
                                   .strip())
                    s = re.sub(r"\s+"," ",s)
                    return s if len(s)<=L else s[:L-1]+"…"
                labels_all = df_auto[lokasi_col].astype(str).map(lambda x: _shorten(x, max_label_len)).tolist()

    # ---------- sampling maks 300 ----------
    if n>300:
        rng = np.random.default_rng(42)
        sel = rng.choice(n, size=300, replace=False)
        Xs = X_scaled[sel]
        labels_sel = [labels_all[i] for i in sel] if labels_all is not None else None
        subset_info = f"(ditampilkan 300 sampel acak dari {n} data)"
        m = 300
    else:
        Xs = X_scaled
        labels_sel = labels_all
        subset_info = f"({n} data)"
        m = n

    # ---------- linkage & ambang potong ----------
    Z = linkage(Xs, method="ward", metric="euclidean")
    cut_height = (np.sort(Z[:,2])[-(k_sel-1)] if (k_sel and k_sel>1) else 0.7*np.max(Z[:,2]))

    # ---------- ukuran figur dinamis ----------
    # fig_w = min(30, 6 + 0.06*m)
    # fig_h = 6.5
    fig_w = 12
    fig_h = 6.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Jika level Kabupaten/Kota -> sembunyikan label dengan memberi labels=None
    show_labels = (labels_sel is not None) and str(level_sel).lower().startswith("prov")
    D = dendrogram(
        Z,
        labels=(labels_sel if show_labels else None),
        leaf_rotation=90,
        leaf_font_size=8 if show_labels else 0,
        color_threshold=cut_height,
        above_threshold_color="#999999"
    )

    # Jika label tampil (Provinsi), kurangi kepadatan dengan interval
    if show_labels:
        interval = max(1, math.ceil(m / max_visible_labels))
        xt = ax.get_xticklabels()
        for i, t in enumerate(xt):
            if (i % interval) != 0:
                t.set_text("")
        ax.set_xticklabels(xt, rotation=90, fontsize=8)
        pad_bottom = 0.12
    else:
        # tanpa label -> ruang bawah kecil
        pad_bottom = 0.05

    # Garis potong + dekorasi
    ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.2, label=f"k={k_sel or '?'}")
    ax.set_title(f"Dendrogram Ward {title_suffix}")
    ax.set_ylabel("Jarak (euclidean)")
    ax.grid(axis="y", linestyle="--", alpha=.4)
    ax.text(0.98, 0.02, subset_info, transform=ax.transAxes, fontsize=9, color="gray",
            ha="right", va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout(rect=[0, pad_bottom, 1, 1])
    # st.pyplot(fig)
    show_fig_and_download(fig, f"dendrogram_{str(level_sel).lower()}_k{k_sel or st.session_state.get('k_used','?')}")


    # Verifikasi jumlah komponen pada garis potong
    try:
        eps = 1e-9
        k_found = len(np.unique(fcluster(Z, t=cut_height - eps, criterion="distance")))
        st.caption(f"Verifikasi komponen pada garis: **k = {k_found}**.")
    except Exception:
        pass


# -------------------- Tren & Top Lokasi --------------------
def render_tren_hasil_panen(df_out: pd.DataFrame):
    """
    Tren Hasil Panen (pertanian) dengan 3 mode tampilan:
      • Garis (Top-N)
      • Facet per Kategori
      • Heatmap (Lokasi × Tahun)

    Fitur:
      - Pencarian / bandingkan lokasi (comma-separated). Bila ada hasil pencarian,
        hanya lokasi itu yang diplot (kecuali user centang 'Tambahkan Top-N lainnya').
      - Top-N, urutan besar/kecil, smoothing median-3, marker, direct label di kanan.
      - Sumbu-Y otomatis dengan sedikit padding agar tidak terpotong.
    """
    
    st.markdown("## 📈 Tren Hasil Panen")

    if df_out is None or df_out.empty:
        st.warning("Dataset kosong.")
        return

    # ---------- kolom kunci ----------
    lokasi_col = next((c for c in df_out.columns
                       if c.lower() in ["provinsi", "kabupaten", "kabupaten/kota", "lokasi"]), None)
    if lokasi_col is None:
        st.warning("Kolom lokasi tidak ditemukan.")
        return
    if "Kategori" not in df_out.columns:
        df_out = df_out.copy()
        df_out["Kategori"] = "-"

    # ---------- helpers ----------
    def _year_cols(prefix: str) -> list[str]:
        patt = re.compile(rf"^{prefix}.*?(19|20)\d{{2}}", flags=re.I)
        cols = [c for c in df_out.columns if patt.search(str(c))]
        # urutkan berdasar tahun yang ditemukan
        def _key(c):
            m = re.search(r"(19|20)\d{2}", str(c))
            return int(m.group()) if m else -1
        return sorted(cols, key=_key)

    def _smooth_med3(y: np.ndarray) -> np.ndarray:
        s = pd.Series(y, dtype=float)
        return s.rolling(3, center=True, min_periods=1).median().to_numpy()

    def _fit_ylim_full(vals, pad_top=0.08, pad_bot=0.06):
        v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy()
        if v.size == 0:
            return None, None
        lo, hi = float(np.min(v)), float(np.max(v))
        span = max(hi - lo, 1e-9)
        return lo - span * pad_bot, hi + span * pad_top

    # ---------- UI atas ----------
    fitur_opsi  = ["luas_areal", "produksi", "produktivitas"]
    fitur_label = {"luas_areal":"Luas Areal", "produksi":"Produksi", "produktivitas":"Produktivitas"}
    fitur_unit  = {"luas_areal":"(ha)", "produksi":"(ton)", "produktivitas":"(ton/ha)"}

    c1, c2, c3, c4 = st.columns([1.5, 1.4, 1.2, 1.3])
    with c1:
        fitur_sel = st.selectbox("Pilih Fitur:", fitur_opsi,
                                 format_func=lambda k: fitur_label[k], key="tren_fitur_nat")
    with c2:
        mode_view = st.selectbox("Mode Tampilan:", ["Garis (Top-N)", "Facet per Kategori", "Heatmap"],
                                key="tren_mode_nat")

    with c3:
        topn = st.slider("Lokasi Tertinggi:", 1, 20, 5, key="tren_topn_nat")
    with c4:
        urutkan = st.radio("Urutan:", ["Terbesar", "Terkecil"], horizontal=True, key="tren_urut_nat")

    c5, c6, c7 = st.columns([1.6, 1.3, 1.8])
    with c5:
        smooth = st.checkbox("Haluskan garis (median 3)", value=True, key="tren_smooth_nat")
    with c6:
        show_markers = st.checkbox("Tampilkan marker", value=True, key="tren_marker_nat")
    with c7:
        direct_label = st.checkbox("Label langsung di kanan (tanpa legend)", value=False,
                                   key="tren_labelkanan_nat")

    # --- pencarian / bandingkan lokasi
    q = st.text_input("🔎 Bandingkan Lokasi (pisahkan dengan koma):", value="",
                      help="Contoh: polewali mandar, kolaka utara, manokwari, buru")
    include_topn = st.checkbox("Tambahkan Top-N lainnya", value=False, key="tren_inc_topn")

    ascending_flag = (urutkan == "Terkecil")
    rank_label = "Tertinggi" if urutkan == "Terbesar" else "Terendah"
    colors = plt.cm.tab20.colors
    ycols = _year_cols(fitur_sel)

    if not ycols:
        st.warning(f"Tidak ada kolom '{fitur_sel}' yang bertahun (mis. {fitur_sel}_2019).")
        return

    # ---------- siapkan data long: Lokasi, Kategori, Tahun, Nilai, Tanggal ----------
    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    d = df_num.melt(id_vars=[lokasi_col, "Kategori"], value_vars=ycols,
                    var_name="Kolom", value_name="Nilai")
    d["Tahun"] = d["Kolom"].str.extract(r"((?:19|20)\d{2})").astype(float)
    d.drop(columns=["Kolom"], inplace=True)
    d = d.dropna(subset=["Tahun", "Nilai"])
    d = d[d["Tahun"].between(2015, 2024)]
    d["Tanggal"] = pd.to_datetime(d["Tahun"].astype(int).astype(str) + "-01-01")

    # ranking mean → Top-N default
    mean_by_loc = d.groupby(lokasi_col)["Nilai"].mean().sort_values(ascending=ascending_flag)
    top_lokasi = mean_by_loc.index.tolist()[:topn]

    # ---------- proses query bandingkan ----------
    names = df_out[lokasi_col].astype(str)
    matched_locs: list[str] = []
    if q.strip():
        # split dan cari yang mengandung (case-insensitive)
        tokens = [t.strip().lower() for t in q.split(",") if t.strip()]
        for tok in tokens:
            if not tok:
                continue
            # izinkan variasi "kab"/"kabupaten", "kota"
            mask = names.str.lower().str.contains(tok)
            matched_locs.extend(names[mask].unique().tolist())
        # unik & pertahankan urutan kemunculan
        matched_locs = list(dict.fromkeys(matched_locs))

        if matched_locs:
            st.caption("Lokasi cocok: " + ", ".join(matched_locs))
        else:
            st.caption("Tidak ada lokasi yang cocok dengan kueri.")

    # --- tentukan baris yang akan diplot
    if matched_locs:
        plot_locs = matched_locs if not include_topn else list(dict.fromkeys(matched_locs + top_lokasi))
    else:
        plot_locs = top_lokasi

    d_plot = d[d[lokasi_col].isin(plot_locs)]

    years = sorted(d_plot["Tahun"].dropna().unique())
    unit_txt = fitur_unit.get(fitur_sel, "")
    ylabel = f"{fitur_label[fitur_sel]} {unit_txt}".strip()

    # ---- MODE 1: Garis (Top-N) ----
    if mode_view == "Garis (Top-N)":
        fig, ax = plt.subplots(figsize=(10.8, 6.0))

        # urutan garis: jika user cari → pakai urutan input; kalau tidak → urutan mean_by_loc (TopN)
        if matched_locs and not include_topn:
            ordered = plot_locs[:]  # sesuai input user
        else:
            ordered = [loc for loc in mean_by_loc.index if loc in plot_locs]

        handles, labels_leg = [], []
        for i, lok in enumerate(ordered):
            sub = d_plot[d_plot[lokasi_col] == lok].sort_values("Tanggal")
            if sub.empty:
                continue
            y = sub["Nilai"].to_numpy(dtype=float)
            if smooth:
                y = _smooth_med3(y)

            (line,) = ax.plot(sub["Tanggal"], y,
                              linewidth=2.2,
                              marker="o" if show_markers else None, markersize=4,
                              alpha=0.95, color=colors[i % len(colors)])
            # label + kategori dominan
            cat = sub["Kategori"].dropna().mode().iat[0] if not sub["Kategori"].dropna().empty else "-"
            label = f"{lok} ({cat})"
            handles.append(line); labels_leg.append(label)

            if direct_label:
                ax.text(sub["Tanggal"].iloc[-1] + pd.Timedelta(days=25), y[-1], s=lok,
                        va="center", fontsize=9, color=colors[i % len(colors)])

        ax.set_title(f"TREN PERBANDINGAN: {', '.join(ordered).upper() if matched_locs else f'{rank_label}-{topn}'}")
        ax.set_xlabel("Tahun"); ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=.35)

        if years:
            xticks = pd.to_datetime([f"{int(y)}-01-01" for y in years])
            ax.set_xticks(xticks); ax.set_xticklabels([str(int(y)) for y in years])

        ylo, yhi = _fit_ylim_full(d_plot["Nilai"])
        if ylo is not None:
            ax.set_ylim(ylo, yhi)

        # Tambah ruang kanan bila direct label
        if direct_label:
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.06)

        if not direct_label and handles:
            ax.legend(handles=handles, labels=labels_leg,
                      title="Lokasi", fontsize=8,
                      ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False)
            plt.tight_layout(rect=[0, 0, 0.78, 1])
        else:
            plt.tight_layout()

        # st.pyplot(fig)
        nm = ("tren_garis_" + fitur_sel + "_" +
              ("vs_" + "_".join([re.sub(r'\W+','_', s) for s in ordered]) if matched_locs else f"top{topn}"))
        show_fig_and_download(fig, nm)

    # ---- MODE 2: Facet per Kategori ----
    elif mode_view == "Facet per Kategori":
        # bila user mencari → facet hanya dari lokasi yang dicari
        base = d_plot if matched_locs else d
        cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
        cats = [c for c in cat_order if c in base["Kategori"].dropna().unique().tolist()] \
               or sorted(base["Kategori"].dropna().unique())

        n = max(len(cats), 1)
        ncols = 2 if n >= 2 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.2 * nrows),
                                 sharex=True, squeeze=False)
        axes = axes.flatten()

        for idx, cat in enumerate(cats):
            ax = axes[idx]
            sub_cat = base[base["Kategori"] == cat].copy()
            if sub_cat.empty:
                ax.set_axis_off(); continue

            # jika user mencari → tampilkan semua lokasi hasil cari pada kategori tsb
            if matched_locs:
                locs_cat = [loc for loc in plot_locs if loc in sub_cat[lokasi_col].unique()]
            else:
                mean_by_loc_cat = sub_cat.groupby(lokasi_col)["Nilai"].mean().sort_values(ascending=ascending_flag)
                locs_cat = mean_by_loc_cat.head(min(topn, 6)).index.tolist()

            for i, lok in enumerate(locs_cat):
                sub = sub_cat[sub_cat[lokasi_col] == lok].sort_values("Tanggal")
                y = sub["Nilai"].to_numpy(dtype=float)
                if smooth:
                    y = _smooth_med3(y)
                ax.plot(sub["Tanggal"], y,
                        linewidth=2.0, marker="o" if show_markers else None, markersize=3.5,
                        color=colors[i % len(colors)], alpha=.95, label=lok)

            ylo, yhi = _fit_ylim_full(sub_cat["Nilai"])
            if ylo is not None:
                ax.set_ylim(ylo, yhi)

            ax.set_title(cat); ax.grid(True, linestyle="--", alpha=.35)
            if years:
                xticks = pd.to_datetime([f"{int(y)}-01-01" for y in years])
                ax.set_xticks(xticks); ax.set_xticklabels([str(int(y)) for y in years])
            ax.legend(fontsize=8, frameon=False)

        # matikan sisa axes bila ada
        for j in range(len(cats), len(axes)):
            axes[j].set_axis_off()

        fig.suptitle(
            f"{fitur_label[fitur_sel]} - Facet per Kategori"
            + (
                f" (Lokasi dicari: {', '.join(matched_locs)})"
                if matched_locs
                else f" ({rank_label}-{min(topn,6)} per facet • {urutkan})"
            ),
            y=0.995, fontsize=12, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        # st.pyplot(fig)
        show_fig_and_download(fig, f"tren_facet_{fitur_sel}")

    # ---- MODE 3: Heatmap (Lokasi × Tahun) ----
    elif mode_view == "Heatmap":
        pvt_all = d.groupby([lokasi_col, "Tahun"])["Nilai"].mean().unstack("Tahun")

        if matched_locs:
            # gunakan urutan sesuai input pengguna
            # rows = [loc for loc in plot_locs if loc in pvt_all.index]
            rows = [loc for loc in plot_locs if loc in pvt_all.index]
        else:
            order = pvt_all.mean(axis=1).sort_values(ascending=ascending_flag)
            # rows = pvt_all.mean(axis=1).sort_values(ascending=ascending_flag).head(topn).index.tolist()
            rows = order.head(topn).index.tolist()
            # Jika 'Terkecil', tampilkan sebagai Bottom → letakkan di bagian bawah
            if ascending_flag:
                rows = rows[::-1]

        # pvt = pvt_all.loc[rows]
        pvt = pvt_all.loc[rows]

        fig, ax = plt.subplots(figsize=(1.2 * len(pvt.columns) + 4, 0.45 * len(pvt.index) + 2.6))
        im = ax.imshow(pvt.values, aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label(fitur_label[fitur_sel])

        ax.set_xticks(range(len(pvt.columns)))
        ax.set_xticklabels([str(int(c)) for c in pvt.columns], rotation=0)
        ax.set_yticks(range(len(pvt.index))); ax.set_yticklabels(pvt.index)

        ax.set_title(
            f"{fitur_label[fitur_sel]} - Heatmap "
            # + (f"(Lokasi dicari)" if matched_locs else f"(Top-{topn} Lokasi • {urutkan})")
            + (f"(Lokasi dicari)" if matched_locs else f"({rank_label}-{topn} Lokasi • {urutkan})")
        )

        mean_val = np.nanmean(pvt.values)
        for (i, j), val in np.ndenumerate(pvt.values):
            if not np.isnan(val):
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center",
                        fontsize=8, color=("white" if val > mean_val else "black"))

        plt.tight_layout()
        # st.pyplot(fig)
        plt.tight_layout()
        show_fig_and_download(fig, f"heatmap_{fitur_sel}_{'search' if matched_locs else f'top{topn}'}")


def render_top_lokasi(df_out: pd.DataFrame):
    """
    TOP Lokasi Hasil Panen (luas_areal / produksi / produktivitas)
    Periode: Per Tahun atau Semua Tahun (agregat 2015–2024 atau sesuai kolom tersedia)
    """
    
    st.markdown("## 🏆 Top Lokasi Harga")

    # ===== guard =====
    if df_out is None or df_out.empty:
        st.warning("Dataset kosong."); return

    lokasi_col = next((c for c in df_out.columns
                       if str(c).lower() in ["provinsi", "kabupaten", "kabupaten/kota", "lokasi"]), None)
    if lokasi_col is None:
        st.warning("Kolom lokasi tidak ditemukan."); return

    # ===== UI: baris 1 =====
    fitur_opsi = ["luas_areal", "produksi", "produktivitas"]
    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c1:
        fitur_sel = st.selectbox("Pilih Fitur:", fitur_opsi, key="tp_fitur")
    with c2:
        urutkan = st.radio("Urutan:", ["Terbesar", "Terkecil"], horizontal=True, key="tp_urut")
    with c3:
        topn = st.slider("Jumlah Lokasi:", 1, 30, 10, key="tp_topn")

    # ===== cari kolom tahunan utk fitur =====
    def _year_cols(prefix: str) -> list[str]:
        cols = [c for c in df_out.columns if re.search(rf"^{prefix}", str(c), flags=re.I)]
        cols = [c for c in cols if re.search(r"(19|20)\d{2}", str(c))]
        def _y(c):
            m = re.search(r"(19|20)\d{2}", str(c))
            return int(m.group()) if m else -1
        return sorted(cols, key=_y)

    fitur_cols = _year_cols(fitur_sel)
    if not fitur_cols:
        st.warning(f"Tidak ada kolom bertahun untuk '{fitur_sel}'."); return

    # map tahun->kolom
    year_map = {}
    for c in fitur_cols:
        m = re.search(r"(19|20)\d{2}", str(c))
        if m: year_map[int(m.group())] = c
    tahun_list = sorted(year_map.keys())
    if not tahun_list:
        st.warning("Tidak ada informasi tahun pada kolom fitur."); return

    # ===== UI: baris 2 (Periode di satu baris) =====
    r1, r2 = st.columns([0.32, 2.68])
    with r1:
        st.markdown("**Periode:**")
    with r2:
        periode = st.radio("", ["Per Tahun", "Semua Tahun"],
                           horizontal=True, label_visibility="collapsed",
                           key="tp_periode")

    # ===== UI: baris 3 (checkbox bulan) =====
    if periode == "Per Tahun":
        per_bulan = st.checkbox("Lihat per bulan (Jan–Des)", value=False, key="tp_per_bulan")
    else:
        per_bulan = False

    # ===== UI: baris 4 (tahun/bulan) + subset data =====
    if periode == "Per Tahun":
        th_sel = st.selectbox("Pilih Tahun:", tahun_list, index=0, key="tp_th")
        if per_bulan:
            bulan_list = list(range(1, 13))
            bl_sel = st.selectbox("Pilih Bulan:", bulan_list, index=0,
                                  format_func=lambda m: ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][m-1],
                                  key="tp_bl")
        col_th = year_map[th_sel]
        df_sel = df_out[[lokasi_col, "Kategori", col_th]].copy()
        df_sel.rename(columns={col_th: "Nilai"}, inplace=True)
        # jika per_bulan, gunakan nilai bulan jika ada kolom bulanan spesifik; jika tidak, tetap pakai tahunan
        subtitle = f"Tahun {th_sel}"
        if per_bulan:
            subtitle = f"Bulan {['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'][bl_sel-1]} {th_sel}"
    else:
        df_sel = df_out[[lokasi_col, "Kategori"] + fitur_cols].copy()
        for c in fitur_cols:
            df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
        df_sel["Nilai"] = df_sel[fitur_cols].mean(axis=1, skipna=True)
        y_min, y_max = min(tahun_list), max(tahun_list)
        subtitle = f"Semua Tahun {y_min}–{y_max}"

    if df_sel.empty:
        st.warning("Tidak ada data pada periode yang dipilih."); return

    # ===== ranking =====
    agg = (df_sel.groupby([lokasi_col, "Kategori"], dropna=False)["Nilai"]
                 .mean().reset_index())

    ascending = (urutan := urutkan) == "Terkecil"
    rank_word = "TERTINGGI" if not ascending else "TERENDAH"

    subset  = agg.sort_values("Nilai", ascending=ascending).head(topn).reset_index(drop=True)
    plot_df = subset.sort_values("Nilai", ascending=True).reset_index(drop=True)

    # ===== warna & label =====
    cat_to_color = {
        "Sangat Rendah":"#d73027","Rendah":"#fc8d59","Cukup Rendah":"#fee090",
        "Sedang":"#ffffbf","Cukup Tinggi":"#e0f3f8","Tinggi":"#91bfdb","Sangat Tinggi":"#4575b4"
    }
    bar_colors = [cat_to_color.get(k, "#999999") for k in plot_df["Kategori"].astype(str)]
    satuan = "(ha)" if "luas" in fitur_sel else "(ton)" if "produksi" in fitur_sel else "(ton/ha)"
    label = fitur_sel.replace("_", " ").title()

    # ===== plot vertikal =====
    fig_w = max(10, min(22, 0.5 * len(plot_df) + 6))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bars = ax.bar(range(len(plot_df)), plot_df["Nilai"],
                  color=bar_colors, edgecolor="#333", linewidth=.6)
    for i, b in enumerate(bars):
        h = float(plot_df["Nilai"].iat[i])
        ax.text(b.get_x() + b.get_width()/2, h + (abs(h) * .01 + 1e-9), f"{h:,.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1c2833")

    ax.set_title(f"{rank_word} {len(plot_df)} LOKASI — {label.upper()} • {subtitle}")
    ax.set_ylabel(f"Rata-Rata {satuan}"); ax.set_xlabel("Lokasi")
    ax.grid(axis="y", linestyle="--", alpha=.5); ax.set_axisbelow(True)

    xlabels = [f"{row[lokasi_col]} ({row['Kategori']})" for _, row in plot_df.iterrows()]
    ax.set_xticks(range(len(plot_df))); ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)

    # CAT_ORDER = ["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi","Sangat Tinggi"]
    # present = [k for k in CAT_ORDER if k in plot_df["Kategori"].dropna().astype(str).unique().tolist()]
    # if present:
    #     handles = [Patch(facecolor=cat_to_color.get(k, "#999999"), edgecolor="#333", label=k) for k in present]
    #     ax.legend(handles=handles, title="Kategori",
    #               loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    # LEGEND: hanya kategori yang tampil
    present = plot_df["Kategori"].dropna().astype(str).unique().tolist()
    legend_handles = [Patch(facecolor=cat_to_color.get(k, "#999999"), edgecolor="#333", label=k) for k in present]
    ax.legend(handles=legend_handles, title="Kategori",
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    # st.pyplot(fig)
    show_fig_and_download(fig, f"top_lokasi_{fitur_sel}_{periode.replace(' ','_').lower()}")


def render_korelasi_fitur_panen(df_out: pd.DataFrame):
    """
    Heatmap korelasi antar fitur: luas_areal, produksi, produktivitas.
    Memakai sampel per (Lokasi, Tahun) pada rentang tahun yang dipilih.
    """
   
    st.markdown("### 🔗 Korelasi Antar Fitur")

    if df_out is None or df_out.empty:
        st.info("Dataset kosong."); return

    # kolom lokasi
    lokasi_col = next((c for c in df_out.columns
                       if str(c).lower() in ["provinsi","kabupaten","kabupaten/kota","lokasi"]), None)
    if lokasi_col is None:
        st.warning("Kolom lokasi tidak ditemukan."); return

    # helper cari kolom bertahun per prefix
    def _year_cols(prefix: str) -> list[str]:
        cols = [c for c in df_out.columns if re.search(rf"^{prefix}", str(c), flags=re.I)]
        cols = [c for c in cols if re.search(r"(19|20)\d{2}", str(c))]
        def _y(c):
            m = re.search(r"(19|20)\d{2}", str(c))
            return int(m.group()) if m else -1
        return sorted(cols, key=_y)

    feats_all = ["luas_areal","produksi","produktivitas"]

    # UI
    c1, c2, c3 = st.columns([1.2, 1.1, 1.1])
    with c1:
        feats_sel = st.multiselect("Pilih Fitur:", feats_all, default=feats_all, key="kor_feat")
    with c2:
        metode = st.selectbox("Metode:", ["pearson","spearman","kendall"], index=0, key="kor_method")
    with c3:
        agreg = st.radio("Periode:", ["Semua Tahun", "Rentang Tahun"], horizontal=True, key="kor_period")

    # tahun tersedia gabungan dari semua fitur terpilih
    years = set()
    for f in feats_sel:
        for c in _year_cols(f):
            m = re.search(r"(19|20)\d{2}", str(c))
            if m: years.add(int(m.group()))
    if not years:
        st.info("Tidak ada kolom bertahun untuk fitur yang dipilih."); return
    y_min, y_max = min(years), max(years)

    if agreg == "Rentang Tahun":
        yr = st.slider("Pilih Rentang Tahun:", y_min, y_max, (max(y_min, 2015), y_max), key="kor_years")
        y_lo, y_hi = yr
    else:
        y_lo, y_hi = y_min, y_max

    # filter lokasi opsional
    q = st.text_input("Filter Lokasi (opsional, pisahkan koma)", "", key="kor_q")
    tokens = [t.strip().lower() for t in q.split(",") if t.strip()]

    # bangun matrix fitur: gabung per (Lokasi, Tahun)
    mats = []
    for feat in feats_sel:
        yc = _year_cols(feat)
        if not yc: 
            continue
        tmp = df_out[[lokasi_col] + yc].copy()
        for c in yc: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        longf = tmp.melt(id_vars=[lokasi_col], value_vars=yc, var_name="Kolom", value_name=feat)
        longf["Tahun"] = longf["Kolom"].str.extract(r"((?:19|20)\d{2})").astype(float)
        longf = longf.dropna(subset=["Tahun"])
        longf["Tahun"] = longf["Tahun"].astype(int)
        longf = longf[(longf["Tahun"] >= y_lo) & (longf["Tahun"] <= y_hi)]
        mats.append(longf[[lokasi_col, "Tahun", feat]])

    if len(mats) < 2:
        st.info("Pilih minimal dua fitur untuk korelasi."); return

    feat_df = mats[0]
    for dfm in mats[1:]:
        feat_df = feat_df.merge(dfm, on=[lokasi_col, "Tahun"], how="inner")

    if tokens:
        names = feat_df[lokasi_col].astype(str).str.lower()
        mask = np.column_stack([names.str.contains(t, regex=False) for t in tokens]).any(axis=1)
        feat_df = feat_df[mask]

    # minimal dua kolom numerik
    corr_cols = [c for c in feats_sel if c in feat_df.columns]
    if len(corr_cols) < 2 or feat_df.empty:
        st.info("Data tidak cukup untuk menghitung korelasi pada pilihan ini."); return

    corr_mat = feat_df[corr_cols].astype(float).corr(method=metode, min_periods=3)

    # plot heatmap
    fig_w = 1.2 * len(corr_cols) + 4
    fig_h = 1.0 * len(corr_cols) + 3.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr_mat.values, norm=norm, cmap="coolwarm", aspect="equal")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label(f"Korelasi ({metode.title()})")

    label_map = {"luas_areal":"Luas Areal","produksi":"Produksi","produktivitas":"Produktivitas"}
    ticks = [label_map.get(c, c) for c in corr_cols]
    ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(ticks)
    ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(ticks)

    for (i, j), v in np.ndenumerate(corr_mat.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=("white" if abs(v) > 0.5 else "black"), fontsize=10)

    ttl = f"{y_lo}–{y_hi}"
    if tokens:
        ttl += " • filter lokasi aktif"
    ax.set_title(f"Korelasi Antar Fitur • {ttl}")

    plt.tight_layout()
    show_fig_and_download(fig, f"korelasi_fitur_{y_lo}_{y_hi}")



# ----------------------------- Map -----------------------------
def _coerce_coord(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().replace(" ","")
    if "," in s and "." in s: s=s.replace(",","")
    elif "," in s: s=s.replace(",",".")
    s=re.sub(r"[^0-9.\-]","",s)
    try: return float(s)
    except Exception: return np.nan

def _clean_latlon(df):
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    if not lat_col or not lon_col: return None, None, df
    d=df.copy(); d[lat_col]=d[lat_col].apply(_coerce_coord); d[lon_col]=d[lon_col].apply(_coerce_coord)
    swap=(d[lat_col].abs()>90)&(d[lon_col].abs()<90)
    d.loc[swap,[lat_col,lon_col]]=d.loc[swap,[lon_col,lat_col]].values
    d=d[d[lat_col].between(-11,7) & d[lon_col].between(95,142)]
    return lat_col, lon_col, d

def render_points_map(df_out, title="Pemetaan Titik Lokasi (Latitude/Longitude)"):
    st.subheader(f"📍 {title}")
    try:
        lat_col, lon_col, df_pts = _clean_latlon(df_out)
        if not lat_col or not lon_col or df_pts.empty:
            st.info("Kolom Latitude/Longitude tidak ditemukan/valid."); return

        # Peta dasar
        m = folium.Map(location=[-2.5, 118], zoom_start=4.6, tiles="cartodbpositron")

        # Mapping kategori (Title-case) -> HEX dari PALETTE (lowercase key)
        ordered = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
        cat_to_hex = {c: PALETTE.get(c.lower(), "#666666") for c in ordered}

        # Rapiin kolom kategori agar sesuai urutan
        if "Kategori" in df_pts.columns:
            df_pts["Kategori"] = df_pts["Kategori"].astype(str).str.strip()
            df_pts["Kategori"] = pd.Categorical(df_pts["Kategori"], categories=ordered, ordered=True)

        # Helper rata-rata lintas tahun untuk popup
        def _mean_across_years(row, key):
            year_cols  = [c for c in df_pts.columns if re.search(fr"{key}.*?(19|20)\d{{2}}", c, flags=re.I)]
            plain_cols = [c for c in df_pts.columns if re.search(fr"^{key}", c, flags=re.I) and not re.search(r"(19|20)\d{2}", c, flags=re.I)]
            use = year_cols if year_cols else plain_cols
            if not use: return np.nan
            vals = pd.to_numeric(row[use], errors="coerce")
            return float(vals.mean())

        # Titik lokasi: gunakan CircleMarker dengan HEX (match legend)
        for _, r in df_pts.iterrows():
            kategori = str(r.get("Kategori", "Sedang"))
            hex_color = cat_to_hex.get(kategori, "#666666")

            nama = next((r.get(c) for c in ["Kabupaten/Kota","Kabupaten","Kota","Provinsi","Lokasi"] if c in df_pts.columns), "-")
            luas    = _mean_across_years(r, "luas")
            prod    = _mean_across_years(r, "produksi")
            produkt = _mean_across_years(r, "produkt")

            html = f"""<b>{nama}</b><br>
            Kategori: <b>{kategori}</b><br>Cluster: {r.get('Cluster','-')}
            <hr style='margin:4px 0;'>
            <b>Rata-Rata Lintas Tahun:</b><br>
            • Luas Areal: {luas:,.0f} ha<br>
            • Produksi: {prod:,.0f} ton<br>
            • Produktivitas: {produkt:,.2f} ton/ha"""

            folium.CircleMarker(
                location=[r[lat_col], r[lon_col]],
                radius=6,
                color=hex_color,          # outline
                weight=1,
                fill=True,
                fill_color=hex_color,     # isi titik
                fill_opacity=0.9,
                popup=folium.Popup(html, max_width=260),
            ).add_to(m)

        # Legend (tetap sama, tapi kini identik warnanya dengan titik)
        present = set(str(x) for x in df_pts.get("Kategori", pd.Series([], dtype=object)).dropna().unique())
        active = [c for c in ordered if c in present] or ordered
        legend = ""
        for name in active:
            hex_color = cat_to_hex.get(name, "#666666")
            legend += (
                "<div style='margin:2px 0;'>"
                f"<span style='display:inline-block;width:10px;height:10px;background:{hex_color};"
                "border-radius:50%;margin-right:6px;border:1px solid #333;'></span>"
                f"{name}</div>"
            )
        legend_html = f"""<div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
            background: rgba(255,255,255,.95); padding: 10px 12px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,.2);
            font-size: 13px; line-height: 1.2; min-width: 180px;"><div style="font-weight:700; margin-bottom:6px;">🗺️ Keterangan</div>{legend}</div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=None, height=560)
        try:
            html_bytes = m.get_root().render().encode("utf-8")
            st.download_button(
                "⬇️ Unduh Peta (HTML)",
                data=html_bytes,
                file_name="peta_lokasi.html",
                mime="text/html",
                use_container_width=True,
                key="dl_map_html"
            )
        except Exception as _e:
            st.caption("Tidak dapat mengekspor peta ke HTML.")

    except Exception as e:
        st.warning(f"Gagal membuat peta: {e}")


# ============================ Session ============================
def _ensure_session_state():
    defaults = {
        "clustering_done": False, "results_df": None, "df_out": None, "X_scaled": None,
        "labels": None, "cluster_label_map": {}, "method_used": None, "k_used": None,
        "ordered_categories": CAT_ORDER[:], "pca_info": {"used":False}
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v


def render_cluster_profile(df_out: pd.DataFrame, feat_cols: list[str] | None = None):
    """Profil ringkas setiap Kategori: mean / median / IQR untuk fitur-fitur numerik (skala asli)."""
    if df_out is None or df_out.empty or "Kategori" not in df_out.columns:
        st.info("Data belum tersedia untuk profil."); 
        return

    # Pilih fitur yang dipakai (prioritas dari feat_cols; kalau kosong, auto-detect)
    used = [c for c in (feat_cols or []) if c in df_out.columns]
    if not used:
        patt = re.compile(r"^(luas|produksi|produkt)", flags=re.I)
        used = [c for c in df_out.columns if patt.search(str(c)) and pd.api.types.is_numeric_dtype(df_out[c])]
    if not used:
        st.info("Tidak ada kolom fitur numerik yang cocok untuk profil."); 
        return

    num = df_out[["Kategori"] + used].copy()
    for c in used:
        num[c] = pd.to_numeric(num[c], errors="coerce")

    def _iqr(x: pd.Series) -> float:
        q1, q3 = np.nanpercentile(x, [25, 75])
        return float(q3 - q1)

    agg_mean   = num.groupby("Kategori", dropna=True)[used].mean(numeric_only=True)
    agg_median = num.groupby("Kategori", dropna=True)[used].median(numeric_only=True)
    agg_iqr    = num.groupby("Kategori", dropna=True)[used].agg(_iqr)

    prof = pd.concat({"Mean": agg_mean, "Median": agg_median, "IQR": agg_iqr}, axis=1)

    st.markdown("#### 📑 Profil Cluster (skala asli)")
    st.dataframe(prof, use_container_width=True)
    # ekspor rapi ke Excel
    prof_export = prof.copy()
    if isinstance(prof_export.columns, pd.MultiIndex):
        prof_export.columns = ["_".join(map(str, c)).strip() for c in prof_export.columns.to_list()]
    prof_export = prof_export.reset_index()
    download_excel_button(prof_export, "profil_cluster", sheet_name="profil")


def render_location_feature_means(df_out: pd.DataFrame, feat_cols: list[str] | None = None):
    """Rata-rata gabungan fitur per lokasi (lintas tahun), dipecah per Kategori."""
    if df_out is None or df_out.empty:
        return

    lokasi_col = next((c for c in df_out.columns 
                       if c.lower() in ["kabupaten/kota","kabupaten","kota","provinsi","lokasi"]), None)
    if not lokasi_col:
        return

    used = [c for c in (feat_cols or []) if c in df_out.columns]
    if not used:
        patt = re.compile(r"^(luas|produksi|produkt)", flags=re.I)
        used = [c for c in df_out.columns if patt.search(str(c))]
    if not used:
        return

    tmp = df_out[[lokasi_col, "Kategori"] + used].copy()
    for c in used:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp["Rata_Rata"] = tmp[used].mean(axis=1, skipna=True)

    out = (tmp.groupby([lokasi_col, "Kategori"])["Rata_Rata"]
             .mean()
             .reset_index()
             .sort_values("Rata_Rata", ascending=False))

    st.markdown("#### 📍 Rata-rata Fitur per Lokasi (lintas tahun)")
    st.dataframe(out, use_container_width=True)
    download_excel_button(out, "rata_rata_fitur_per_lokasi", sheet_name="rata2")


# ==== PDF helpers ====

def _add_report_fig(fig, key="report_figs"):
    # simpan figure ke session agar bisa dikompilasi jadi PDF
    lst = st.session_state.get(key, [])
    lst.append(fig)
    st.session_state[key] = lst

def _text_page_fig(title: str, lines: list[str], figsize=(8.27, 11.69)):
    """Buat halaman teks sebagai figure A4 portrait."""
    
    fig, ax = plt.subplots(figsize=figsize)  # inci ≈ A4
    ax.axis("off")
    y = 0.95
    ax.text(0.5, y, title, ha="center", va="top", fontsize=16, fontweight="bold")
    y -= 0.05
    for line in lines:
        ax.text(0.06, y, str(line), ha="left", va="top", fontsize=11)
        y -= 0.035
        if y < 0.05:
            break
    return fig

def _df_to_table_fig(df: pd.DataFrame, title="Tabel", max_rows=30, max_cols=12, figsize=(11.69, 8.27)):
    """Render cuplikan DataFrame ke gambar tabel agar bisa masuk PDF."""
    
    df_disp = df.copy()
    if df_disp.shape[0] > max_rows:
        df_disp = df_disp.head(max_rows)
    if df_disp.shape[1] > max_cols:
        keep = df_disp.columns[:max_cols]
        df_disp = df_disp[keep]

    fig, ax = plt.subplots(figsize=figsize)  # A4 landscape
    ax.set_title(title, pad=18, fontsize=14, fontweight="bold")
    ax.axis("off")

    tbl = ax.table(
        cellText=df_disp.astype(str).values,
        colLabels=df_disp.columns.astype(str).tolist(),
        loc="center",
        cellLoc="left",
        colLoc="left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.2)
    return fig

def build_pdf_report(level_sel: str) -> bytes:
    """
    Susun laporan PDF berisi:
      1) Halaman sampul
      2) Ringkasan metrik clustering (jika ada)
      3) Cuplikan hasil tabel (lokasi + Cluster + Kategori)
      4) Seluruh figure yang sudah dirender di UI dan ditangkap ke st.session_state["report_figs"]
         dengan deduplikasi agar tidak berulang

    Return: bytes PDF untuk dipakai pada st.download_button
    """
    
    # --------- helper: buat halaman teks sederhana sebagai figure ---------
    def _text_page_fig(title: str, lines: list[str], figsize=(8.27, 11.69)):
        # ukuran default ~ A4 portrait dalam inch
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        y = 0.95
        ax.text(0.5, y, title, ha="center", va="top", fontsize=16, fontweight="bold")
        y -= 0.06
        for line in lines:
            ax.text(0.06, y, str(line), ha="left", va="top", fontsize=11)
            y -= 0.035
            if y < 0.06:
                break
        return fig

    # --------- helper: render cuplikan DataFrame jadi tabel di figure ---------
    def _df_to_table_fig(df: pd.DataFrame, title="Tabel", max_rows=30, max_cols=12, figsize=(11.69, 8.27)):
        # ukuran default ~ A4 landscape dalam inch
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, pad=18, fontsize=14, fontweight="bold")
        ax.axis("off")

        df_disp = df.copy()
        if df_disp.shape[0] > max_rows:
            df_disp = df_disp.head(max_rows)
        if df_disp.shape[1] > max_cols:
            df_disp = df_disp.iloc[:, :max_cols]

        tbl = ax.table(
            cellText=df_disp.astype(str).values,
            colLabels=df_disp.columns.astype(str).tolist(),
            loc="center",
            cellLoc="left",
            colLoc="left"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.2)
        return fig

    # --------- helper: hash figure berbasis piksel PNG untuk dedup ---------
    def _fig_hash(fig) -> str:
        buf_img = io.BytesIO()
        fig.savefig(buf_img, format="png", dpi=120, bbox_inches="tight")
        return hashlib.sha1(buf_img.getvalue()).hexdigest()

    # --------- ambil meta dari session ---------
    method = st.session_state.get("method_used")
    k_used = st.session_state.get("k_used")
    resdf = st.session_state.get("results_df")
    df_out = st.session_state.get("df_out")
    used_cols = st.session_state.get("diag_used_cols", [])
    cat_order = st.session_state.get("ordered_categories", [])
    diag_counts = st.session_state.get("diag_counts")
    diag_nan = st.session_state.get("diag_nan")
    pinfo = st.session_state.get("pca_info", {"used": False})

    # --------- ambil semua figure yang sudah dikumpulkan dan dedup ---------
    figs = st.session_state.get("report_figs", [])
    uniq_figs, seen = [], set()
    for fg in figs:
        try:
            h = _fig_hash(fg)
        except Exception:
            continue
        if h in seen:
            continue
        seen.add(h)
        uniq_figs.append(fg)
    figs = uniq_figs

    # --------- halaman sampul ---------
    ts = datetime.now().strftime("%d-%m-%Y %H:%M")
    cover_lines = [
        f"Waktu pembuatan: {ts}",
        f"Level data: {level_sel}",
        f"Metode: {method if method else '-'}   |   k: {k_used if k_used is not None else '-'}",
        f"Fitur yang dipakai: {len(used_cols)} kolom" if used_cols else "Fitur yang dipakai: -",
        f"Kategori urut: {', '.join([str(c) for c in cat_order])}" if cat_order else "Kategori urut: -",
        "Peta Folium tidak dibekukan ke PDF. Untuk peta gunakan ekspor HTML terpisah.",
        f"PCA dipakai: {'Ya' if pinfo.get('used') else 'Tidak'}"
    ]
    cover_fig = _text_page_fig("Laporan Klastering Harga Cabai Rawit", cover_lines)

    # --------- siapkan tabel ringkas ---------
    table_figs = []
    try:
        if isinstance(resdf, pd.DataFrame) and not resdf.empty:
            table_figs.append(_df_to_table_fig(resdf, title="Ringkasan Metrik Clustering"))
    except Exception:
        pass

    try:
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            # pilih kolom identitas utama
            id_cols = [c for c in df_out.columns if str(c).lower() in ["provinsi", "kabupaten/kota", "kabupaten", "kota", "lokasi"]]
            show_cols = (id_cols[:1] if id_cols else []) + [c for c in ["Cluster", "Kategori"] if c in df_out.columns]
            if not show_cols:
                show_cols = df_out.columns[:6].tolist()
            table_figs.append(_df_to_table_fig(df_out[show_cols], title="Cuplikan Hasil Tabel"))
    except Exception:
        pass

    # Tambah tabel kecil untuk kualitas kolom harga (opsional)
    try:
        if isinstance(diag_counts, dict) or isinstance(diag_nan, dict):
            meta_rows = []
            if isinstance(diag_counts, dict):
                meta_rows.append({
                    "Info": "Jumlah kolom harga merah", "Nilai": diag_counts.get("harga_merah", "-")
                })
                meta_rows.append({
                    "Info": "Jumlah kolom harga hijau", "Nilai": diag_counts.get("harga_hijau", "-")
                })
            if isinstance(diag_nan, dict):
                meta_rows.append({
                    "Info": "Proporsi NaN harga merah", "Nilai": f"{diag_nan.get('harga_merah'):.3f}" if isinstance(diag_nan.get("harga_merah"), float) else str(diag_nan.get("harga_merah"))
                })
                meta_rows.append({
                    "Info": "Proporsi NaN harga hijau", "Nilai": f"{diag_nan.get('harga_hijau'):.3f}" if isinstance(diag_nan.get("harga_hijau"), float) else str(diag_nan.get("harga_hijau"))
                })
            if meta_rows:
                meta_df = pd.DataFrame(meta_rows)
                table_figs.append(_df_to_table_fig(meta_df, title="Kualitas Fitur"))
    except Exception:
        pass

    # --------- tulis ke PDF ---------
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # halaman sampul
        pdf.savefig(cover_fig, bbox_inches="tight")

        # tabel ringkas
        for tf in table_figs:
            try:
                pdf.savefig(tf, bbox_inches="tight")
            except Exception:
                continue

        # semua figure hasil visualisasi
        for fg in figs:
            try:
                pdf.savefig(fg, bbox_inches="tight")
            except Exception:
                continue

    buf.seek(0)
    return buf.read()



# ============================== APP (PERTANIAN – UI selaras) ==============================
def app():
    if not OPENPYXL_OK:
        st.error("Package **openpyxl** belum terpasang.\nJalankan: `pip install openpyxl` lalu refresh.")
        return
    _ensure_session_state()

    # === aktifkan penangkap figure sekali saja ===
    if not st.session_state.get("_pyplot_patched", False):
        _orig_pyplot = st.pyplot
        def _pyplot_capture(fig=None, **kwargs):
            out = _orig_pyplot(fig, **kwargs)
            try:
                if fig is not None:
                    _add_report_fig(fig)
            except Exception:
                pass
            return out
        st.pyplot = _pyplot_capture
        st.session_state["_pyplot_patched"] = True

    if "report_figs" not in st.session_state:
        st.session_state["report_figs"] = []


    st.markdown("<h1 style='text-align:center; font-weight:800;'>Data Pertanian Cabai Rawit di Indonesia</h1>", unsafe_allow_html=True)
    st.write("Pilih sumber data, tentukan level, pilih metode dan nilai k, lalu jalankan clustering.")
    st.markdown("---")
    # st.sidebar.toggle("Urutkan kategori dari Tinggi → Rendah", value=False, key="cat_desc")

    # Pilih level data
    prev_level = st.session_state.get("level_sel")
    level_sel  = st.radio("Pilih level data:", LEVEL_OPTIONS, horizontal=True, index=0, key="level_sel")

    # Pilih sumber data
    prev_mode = st.session_state.get("data_mode_sel")
    data_mode = st.radio("Sumber data:", ["Upload data sendiri", "Gunakan dataset contoh"], horizontal=True, key="data_mode_sel")

    # Reset state bila ganti level atau mode
    if (prev_level is not None and prev_level != level_sel) or (prev_mode is not None and prev_mode != data_mode):
        for k in ["clustering_done","results_df","df_out","X_scaled","labels","method_used","k_used","ordered_categories","cluster_label_map","pca_info"]:
            st.session_state.pop(k, None)

    df_raw = None

    # ========================= MODE: UPLOAD SENDIRI =========================
    if data_mode == "Upload data sendiri":
        # Tombol unduh template sesuai level (diarahkan ke path template)
        st.subheader("📚 Template Dataset")
        try:
            tmpl_bytes = _load_template_from_path(level_sel)
            st.download_button(
                "📄 Unduh Template Dataset",
                data=tmpl_bytes,
                file_name=Path(TEMPLATE_PATHS[level_sel]).name,  # mis. template_dataset_pertanian_provinsi.xlsx
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption(f"Sumber file: `{TEMPLATE_PATHS[level_sel]}`")
        except Exception as e:
            st.warning(f"Template tidak bisa dibaca dari path: {TEMPLATE_PATHS.get(level_sel, '-')}. {e}")
        st.subheader(f"📤 Upload Data {level_sel} (Excel .xlsx)")
        up = st.file_uploader(
            f"Tarik & lepas atau klik Browse (level {level_sel})",
            type=["xlsx"], accept_multiple_files=False,
            help="Maks 200MB • .xlsx (membutuhkan openpyxl)", key="uploader_by_level"
        )
        if up is None:
            st.info("Belum ada berkas yang diunggah.")
            return

        try:
            df_raw = pd.read_excel(up, header=0, engine="openpyxl").dropna(how="all")
        except Exception as e:
            st.error(f"Gagal memproses Excel: {e}")
            return

        # Validasi kolom identitas
        if level_sel == "Provinsi":
            if "Provinsi" not in map(str, df_raw.columns):
                st.error("Kolom **Provinsi** wajib ada.")
                return
        else:
            if not any(c.strip().lower() in ("kabupaten/kota","kabupaten","kota") for c in map(str, df_raw.columns)):
                st.error("Kolom **Kabupaten/Kota** (atau 'Kabupaten'/'Kota') wajib ada.")
                return

    # ========================= MODE: DATASET CONTOH (dari PATH) =========================
    else:
        st.subheader(f"🗂️ Gunakan Dataset dari Path ({level_sel})")
        try:
            df_raw = _load_dataset_from_path(level_sel)
            st.success(f"Membaca: {DATASET_PATHS[level_sel]}")
            st.dataframe(df_raw.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membaca file dari path: {e}")
            return

        # Optional: unduh dataset contoh
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df_raw.to_excel(w, index=False, sheet_name=f"dataset_{'provinsi' if level_sel=='Provinsi' else 'kabupaten'}")
        buf.seek(0)
        st.download_button(
            "⬇️ Unduh dataset contoh (.xlsx)",
            data=buf.read(),
            file_name=f"dataset_contoh_{'provinsi' if level_sel=='Provinsi' else 'kabupaten'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # ======================= Info granularitas & Pengaturan =======================
    try:
        gran = detect_granularity(df_raw)
        st.info(f"Granularitas terdeteksi: **{gran.upper()}**. Backend menyesuaikan otomatis.")
    except Exception:
        pass

    st.markdown("### ⚙️ Pengaturan Clustering")
    method_sel = st.selectbox(
        "Pilih metode clustering:",
        ["K-Means", "Hierarchical Clustering"]  # samakan UI dengan versi harga
    )
    k_sel = st.slider("Jumlah cluster (k)", 2, 7, 2)

    
    # ============================== Jalankan ==============================
    if st.button("🚀 Jalankan Clustering"):
        try:
            results_df, df_out, X_used, labels, cluster_label_map = run_clustering(df_raw, method_sel, k_sel)
            st.session_state.update({
                "results_df": results_df, "df_out": df_out, "X_scaled": X_used, "labels": labels,
                "cluster_label_map": cluster_label_map, "clustering_done": True,
                "method_used": method_sel, "k_used": k_sel
            })
            # reset koleksi figure untuk laporan
            st.session_state["report_figs"] = []
            st.success("Clustering berhasil dijalankan.")
        except Exception as e:
            st.error(f"Gagal menjalankan clustering: {e}")
            return


    # ============================== Output & Visual ==============================
    with st.expander("🔎 Kolom fitur yang dipakai & kualitasnya", expanded=False):
        st.write("Dipakai untuk clustering:", st.session_state.get("diag_used_cols","—"))
        st.write("Banyaknya kolom per keluarga:", st.session_state.get("diag_counts","—"))
        st.write("Proporsi NaN per keluarga:", st.session_state.get("diag_nan","—"))
        pinfo = st.session_state.get("pca_info", {"used":False})
        st.write("**PCA**:", "tidak dipakai" if not pinfo.get("used") else f"{pinfo['n_components']} komponen, explained ≈ {pinfo.get('explained',0):.2%}")

    if st.session_state.get("clustering_done", False):
        df_out = st.session_state["df_out"]

        # Panel evaluasi performa
        render_cluster_performance(st.session_state["X_scaled"], st.session_state["method_used"])

        # Tabel & ringkasan jumlah kategori
        col1, col2 = st.columns([1.3, 1])
        with col1:
            # st.markdown("### 📌 Hasil Tabel")
            # show_cols = [c for c in df_out.columns if c.lower() in ["lokasi","provinsi","kabupaten","kabupaten/kota"]] + ["Cluster","Kategori"]
            # show_cols = [c for c in show_cols if c in df_out.columns]
            # st.dataframe(df_out[show_cols] if show_cols else df_out.head(20), use_container_width=True)
            st.markdown("### 📌 Hasil Tabel")
            show_cols = [c for c in df_out.columns if c.lower() in ["lokasi","provinsi","kabupaten","kabupaten/kota"]] + ["Cluster","Kategori"]
            show_cols = [c for c in show_cols if c in df_out.columns]
            df_show = df_out[show_cols] if show_cols else df_out.head(20)
            st.dataframe(df_show, use_container_width=True)
            download_excel_button(df_show, f"hasil_tabel_{level_sel.lower()}", sheet_name="hasil_tabel")
        with col2:
            st.markdown("### 📊 Jumlah Data per Kategori")
            fig, ax = plt.subplots(figsize=(6,4))
            order_labels = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
            count_df = df_out["Kategori"].value_counts().reindex(order_labels, fill_value=0)
            warna = ["#C0392B","#D35400","#F1C40F","#27AE60","#2980B9","#8E44AD","#7F8C8D"][:len(count_df)]
            ax.bar(count_df.index, count_df.values, color=warna)
            for i, v in enumerate(count_df.values):
                ax.text(i, v+1, str(v), ha="center", fontsize=9, fontweight="bold")
            ax.set_title("Distribusi Kategori"); ax.set_xlabel("Kategori Cluster"); ax.set_ylabel("Jumlah")
            ax.set_xticklabels(count_df.index, rotation=15, ha="right"); ax.grid(axis="y", linestyle="--", alpha=.4)
            # st.pyplot(fig)
            show_fig_and_download(fig, "jumlah_data_per_kategori")

        # PCA 2D (komposisi cluster)
        st.markdown("### 🧭 Visualisasi Komposisi Cluster (PCA 2D)")
        render_pca_scatter_visual(st.session_state["X_scaled"], st.session_state["df_out"], st.session_state["labels"])

        # Profil cluster & rata-rata per lokasi
        with st.expander("📑 Profil Cluster (skala asli) + Rata-rata per Lokasi", expanded=False):
            render_cluster_profile(df_out, st.session_state.get("diag_used_cols", []))
            render_location_feature_means(df_out, st.session_state.get("diag_used_cols", []))

        # Anggota tiap cluster
        st.markdown("### 📋 Anggota Setiap Cluster")
        lokasi_cols = [c for c in df_out.columns if c.lower() in ["provinsi","kabupaten","kabupaten/kota","lokasi"]]
        lokasi_col = lokasi_cols[0] if lokasi_cols else df_out.columns[0]
        rows = []
        for cid, label in st.session_state.get("cluster_label_map", {}).items():
            anggota = df_out[df_out["Cluster"] == cid][lokasi_col].astype(str).tolist()
            rows.append({"Cluster": f"{label} (Cluster {cid})", "Jumlah Anggota": len(anggota), "Anggota": ", ".join(anggota) if anggota else "-"})
        # st.dataframe(pd.DataFrame(rows), use_container_width=True)
        rows_df = pd.DataFrame(rows)
        st.dataframe(rows_df, use_container_width=True)
        download_excel_button(rows_df, "anggota_cluster", sheet_name="anggota")


        # Dendrogram (jika hierarchical)
        # if st.session_state["method_used"] == "Hierarchical Clustering":
        #     render_dendrogram_from_X(st.session_state["X_scaled"], title_suffix=level_sel, k_sel=st.session_state["k_used"])
        
        if st.session_state["method_used"] == "Hierarchical Clustering":
            render_dendrogram_adaptive(st.session_state["X_scaled"], level_sel=level_sel, title_suffix=level_sel, k_sel=st.session_state["k_used"], max_visible_labels=60, max_label_len=18)

        # Peta titik
        render_points_map(df_out, title="Pemetaan Titik Lokasi (Latitude/Longitude)")

        # Tabs analisis lanjutan (struktur sama dengan UI versi harga)
        st.markdown("### 📋 Analisis Lanjutan")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Gabungan Seluruh Tahun","📆 Per Tahun","📈 Tren Hasil Panen","🏆 Lokasi Tertinggi", "🔗 Korelasi Fitur"])
        with tab1: render_boxplot_combined(df_out)
        with tab2: render_boxplot(df_out)
        with tab3: render_tren_hasil_panen(df_out)
        with tab4: render_top_lokasi(df_out)
        with tab5: render_korelasi_fitur_panen(df_out)
            # ============================== Unduh Laporan PDF ==============================
    st.markdown("### 📥 Unduh Laporan PDF")
    if st.session_state.get("clustering_done", False):
        try:
            pdf_bytes = build_pdf_report(level_sel)  # pastikan fungsi ini sudah didefinisikan
            fname = f"laporan_harga_cabai_{level_sel.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            clicked = st.download_button(
                "⬇️ Unduh Laporan PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True
            )
            if clicked:
                # kosongkan koleksi figure agar unduhan berikutnya tidak dobel
                st.session_state["report_figs"] = []
        except Exception as e:
            st.warning(f"Gagal membangun PDF: {e}")
    else:
        st.info("Jalankan clustering terlebih dahulu agar laporan dapat dibuat.")




# if __name__ == "__main__":
#     app()
