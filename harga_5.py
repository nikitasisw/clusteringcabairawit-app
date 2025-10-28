import io, re, json, unicodedata, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import hashlib

# Peta
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

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


# Dendrogram
try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import openpyxl
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False


# Path template & dataset contoh (sesuaikan nama file jika beda)
TEMPLATE_PATHS = {
    "Provinsi":       str(Path("dataset") / "template_dataset_harga_provinsi.xlsx"),
    "Kabupaten/Kota": str(Path("dataset") / "template_dataset_harga_kabupaten.xlsx"),
}
DATASET_PATHS = {
    "Provinsi":       str(Path("dataset") / "dataset_harga_provinsi.xlsx"),
    "Kabupaten/Kota": str(Path("dataset") / "dataset_harga_kabupaten.xlsx"),
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

# ------------------------ Bobot & util keluarga fitur HARGA ------------------------
FEATURE_WEIGHTS = {
    "harga_cabe_rawit_merah": 0.50,
    "harga_cabe_rawit_hijau": 0.50,
}
def _feature_weight_for_col(col: str, weights: dict[str, float]) -> float:
    cl = str(col).lower().replace(" ", "_")
    for k, w in weights.items():
        if cl.startswith(k):
            return float(w)
    return 1.0


# ------------------------ Lokasi & Fitur Harga ------------------------
def _find_lokasi_col(df: pd.DataFrame) -> str | None:
    """Cari nama kolom lokasi yang umum dipakai, case/space-insensitive."""
    if df is None or df.empty:
        return None
    candidates = [c for c in df.columns if isinstance(c, str)]
    targets = [
        "kabupaten/kota", "kabupaten", "kota", "provinsi",
        "lokasi", "nama_wilayah", "nama_kabupaten", "nama_provinsi"
    ]
    for t in targets:
        for c in candidates:
            if c.strip().lower() == t:
                return c
    return None

def _detect_feature_cols(df: pd.DataFrame):
    """Keluarga fitur = harga merah & hijau, multi-bulan/tahun."""
    merah = [c for c in df.columns if isinstance(c, str) and re.search(r"harga.*merah", c, flags=re.I)]
    hijau = [c for c in df.columns if isinstance(c, str) and re.search(r"harga.*hijau", c, flags=re.I)]
    return merah, hijau, []  # slot ketiga dibiarkan kosong demi kompatibilitas lama

# ==================== Helpers: Bulan & Granularitas ====================

# Tangkap nama bulan versi panjang/pendek & Inggris (buat jaga-jaga)
_BULAN_MAP = {
    "jan": 1, "januari": 1,
    "feb": 2, "februari": 2,
    "mar": 3, "maret": 3,
    "apr": 4, "april": 4,
    "mei": 5, "may": 5,
    "jun": 6, "juni": 6,
    "jul": 7, "juli": 7,
    "agu": 8, "agust": 8, "agustus": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oct": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "des": 12, "dec": 12, "desember": 12
}

def _has_month_token(name: str) -> bool:
    """Cek apakah string kolom mengandung token bulan (jan–des)."""
    s = str(name).lower()
    # longgar: cocokkan '_' atau boundary kata biar nangkap variasi
    for key in _BULAN_MAP:
        if re.search(rf"(?:^|[_\W]){re.escape(key)}(?:[_\W]|$)", s):
            return True
    return False

def _parse_month_year(colname: str):
    """
    Ekstrak (bulan, tahun) dari nama kolom.
    Mengembalikan: (month:int|None, year:int|None)
    """
    s = str(colname).lower().replace("-", " ").replace(".", " ").replace("__", "_")
    # Tahun
    m_year = re.search(r"(19|20)\d{2}", s)
    year = int(m_year.group(0)) if m_year else None
    # Bulan
    month = None
    for key, idx in _BULAN_MAP.items():
        if re.search(rf"(?:^|[_\W]){re.escape(key)}(?:[_\W]|$)", s):
            month = idx
            break
    return month, year

def detect_granularity(df: pd.DataFrame) -> str:
    """
    Deteksi granularity dataset harga:
    - 'monthly' jika nama kolom mengandung token bulan,
    - 'yearly' kalau tidak ada bulan tapi ada tahun,
    - 'unknown' kalau tidak ditemukan kolom harga.
    """
    if df is None or df.empty:
        return "unknown"
    harga_cols = [c for c in df.columns
                  if isinstance(c, str) and re.search(r"harga.*(merah|hijau)", c, flags=re.I)]
    if not harga_cols:
        return "unknown"
    if any(_has_month_token(c) for c in harga_cols):
        return "monthly"
    # cek ada tahun?
    if any(re.search(r"(19|20)\d{2}", str(c)) for c in harga_cols):
        return "yearly"
    return "unknown"



def _composite_score(df: pd.DataFrame, cols: list[str] | None = None) -> pd.Series:
    if cols:
        Z=[]
        for c in cols:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                mu, sd = v.mean(), v.std(ddof=0) or 1.0
                Z.append((v-mu)/sd)
        if not Z:
            raise ValueError("Tidak ada kolom valid untuk skor.")
        return pd.concat(Z, axis=1).mean(axis=1)

    merah, hijau, _ = _detect_feature_cols(df); parts=[]
    for fam in [merah, hijau]:
        if fam:
            s = pd.to_numeric(df[fam].mean(axis=1, skipna=True), errors="coerce")
            mu, sd = s.mean(), s.std(ddof=0) or 1.0
            parts.append((s-mu)/sd)
    if not parts:
        raise ValueError("Tidak ada kolom harga merah/hijau untuk skor.")
    return pd.concat(parts, axis=1).mean(axis=1)

# =================== Prepare Numeric (tanpa PCA otomatis) ===================
def _prepare_numeric(df: pd.DataFrame):
    """
    Pilih kolom harga (merah/hijau) bulanan/tahunan, lalu Impute mean + StandardScaler.
    TIDAK melakukan PCA otomatis. (PCA dilakukan hanya bila user memilih "PCA + K-Means")
    """
    gran = detect_granularity(df)

    patt_year = re.compile(r"(19|20)\d{2}")
    harga_cols = [c for c in df.columns
                  if isinstance(c, str) and re.search(r"harga.*(merah|hijau)", c, flags=re.I)]

    if gran == "monthly":
        feature_cols = [c for c in harga_cols if patt_year.search(c) and _has_month_token(c)]
        if not feature_cols:
            feature_cols = [c for c in harga_cols if patt_year.search(c)]
    elif gran == "yearly":
        feature_cols = [c for c in harga_cols if patt_year.search(c)]
        if not feature_cols:
            feature_cols = harga_cols
    else:
        feature_cols = harga_cols

    if not feature_cols:
        raise ValueError("Tidak ditemukan kolom fitur harga (merah/hijau).")

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
        patt = re.compile(r"(harga)", flags=re.I)
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
    if not isinstance(k, (int, np.integer)):
        k = int(np.squeeze(k))
    if k < 2 or k > 7:
        raise ValueError("Jumlah cluster (k) harus 2–7.")

    # 1) Siapkan fitur (scaled, TANPA PCA)
    X_scaled, used_feature_cols, _ = _prepare_numeric(df)

    # 2) Fit model langsung pada X_scaled
    model = _fit_model(X_scaled, method, k)
    labels = np.asarray(
        model.labels_ if hasattr(model, "labels_") else model.fit_predict(X_scaled),
        dtype=int
    )

    # 3) Metrik evaluasi pada ruang yang dipakai untuk fit
    try:
        sil = round(
            silhouette_score(
                X_scaled, labels,
                sample_size=min(3000, X_scaled.shape[0]),
                random_state=42
            ),
            4
        )
    except Exception:
        sil = float("nan")

    try:
        dbi = round(davies_bouldin_score(X_scaled, labels), 4)
    except Exception:
        dbi = float("nan")

    results_df = pd.DataFrame([{
        "Metode": method,
        "Jumlah Cluster": int(len(np.unique(labels))),
        "Silhouette": sil,
        "Davies-Bouldin": dbi
    }])

    # 4) Ranking cluster (skala asli) dengan weighted z-score
    ranked, cluster_scores = ranking_weighted_zscore(
        df, labels, used_feature_cols, FEATURE_WEIGHTS
    )

    LABEL_SETS = {
        2: ["Rendah", "Tinggi"],
        3: ["Rendah", "Sedang", "Tinggi"],
        4: ["Sangat Rendah", "Rendah", "Tinggi", "Sangat Tinggi"],
        5: ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
        6: ["Sangat Rendah", "Rendah", "Cukup Rendah", "Sedang", "Cukup Tinggi", "Tinggi"],
        7: ["Sangat Rendah", "Rendah", "Cukup Rendah", "Sedang", "Cukup Tinggi", "Tinggi", "Sangat Tinggi"],
    }
    chosen_labels = LABEL_SETS[len(np.unique(labels))]
    cluster_label_map = {int(c): chosen_labels[i] for i, c in enumerate(ranked)}

    # 5) Output dataframe + normalisasi urutan kategori
    df_out = df.copy()
    df_out["Cluster"] = labels
    df_out["Kategori"] = df_out["Cluster"].map(cluster_label_map)
    df_out = normalize_cluster_order(df_out, fitur_list=used_feature_cols)

    # 6) Diagnostik spesifik harga (merah/hijau)
    def _cols_like(pattern):
        rx = re.compile(pattern, flags=re.I)
        return [c for c in df.columns if rx.search(str(c))]
    merah_cols = _cols_like(r"harga.*(cabe|cabai).*rawit.*merah")
    hijau_cols = _cols_like(r"harga.*(cabe|cabai).*rawit.*hijau")

    def _prop_nan(cols):
        if not cols:
            return np.nan
        vals = pd.to_numeric(df[cols].values.ravel(), errors="coerce")
        return float(np.isnan(vals).mean())

    # 7) Simpan state (PCA di-nonaktifkan; visual PCA bisa dibuat terpisah)
    st.session_state.update({
        "ordered_categories": list(df_out["Kategori"].cat.categories),
        "cluster_label_map": cluster_label_map,
        "method_used": method,
        "k_used": k,
        "diag_used_cols": used_feature_cols,
        "diag_counts": {
            "harga_merah": sum("merah" in str(c).lower() for c in used_feature_cols),
            "harga_hijau": sum("hijau" in str(c).lower() for c in used_feature_cols),
        },
        "diag_nan": {
            "harga_merah": _prop_nan(merah_cols),
            "harga_hijau": _prop_nan(hijau_cols),
        },
        "cluster_scores": cluster_scores.to_dict(),
        "pca_info": {"used": False}  # hanya flag info; PCA tidak dipakai untuk fit
    })

    # 8) Kembalikan hasil
    return results_df, df_out, X_scaled, labels, cluster_label_map


# ===================== Evaluasi & Visualisasi =====================
def render_cluster_performance(X_scaled, method_sel, k_sel=None):
    import time
    import numpy as np
    st.markdown("## 📊 Evaluasi Performa Clustering")
    st.caption("Silhouette lebih tinggi lebih baik; Davies–Bouldin lebih rendah lebih baik.")

    if X_scaled is None or len(X_scaled) == 0:
        st.info("Tidak ada data untuk evaluasi."); return

    # ambil k dari session atau parameter
    k_sel = int(k_sel or st.session_state.get("k_used", 2))

    # ===== hitung semua metrik 2..7 (supaya bisa diplot) =====
    silhouette_scores, dbi_scores, k_values = [], [], list(range(2, 8))
    t0 = time.time()
    for k in k_values:
        try:
            if method_sel in ["K-Means", "KMeans"]:
                model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            else:
                model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_scaled)
            labels = model.labels_
            sil = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
        except Exception:
            sil, dbi = np.nan, np.nan
        silhouette_scores.append(sil)
        dbi_scores.append(dbi)
    elapsed = time.time() - t0

    # ===== ambil nilai sesuai k terpilih =====
    idx = k_sel - 2
    sil_k = silhouette_scores[idx] if 0 <= idx < len(silhouette_scores) else np.nan
    dbi_k = dbi_scores[idx] if 0 <= idx < len(dbi_scores) else np.nan

    # ===== hitung k optimum indikatif dari Silhouette (abaikan NaN) =====
    sil_arr = np.asarray(silhouette_scores, dtype=float)
    if np.isnan(sil_arr).all():
        best_k = None
    else:
        best_idx = int(np.nanargmax(sil_arr))
        best_k = int(k_values[best_idx])

    # ===== tampilkan metrik header =====
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Metode", method_sel.replace("+", "").replace("-", ""))
        st.metric("Jumlah Cluster", f"{k_sel}")
    with c2:
        st.metric("Silhouette Score", f"{sil_k:.4f}" if not np.isnan(sil_k) else "—",
                  help="Semakin tinggi semakin baik (cluster makin terpisah).")
        st.metric("Davies–Bouldin Index", f"{dbi_k:.4f}" if not np.isnan(dbi_k) else "—",
                  help="Semakin rendah semakin baik (cluster makin kompak).")
    with c3:
        st.metric("Waktu Proses", f"{elapsed:.4f} detik")
        pinfo = st.session_state.get("pca_info", {"used": False})
        st.metric("PCA", "Tidak dipakai" if not pinfo.get("used") else f"{pinfo.get('n_components')} komponen")

    # ===== visualisasi =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Silhouette
    axes[0].plot(k_values, silhouette_scores, "o-", color="royalblue", lw=2)
    axes[0].axvline(k_sel, ls="--", color="red", lw=2, alpha=0.8)
    for i, s in zip(k_values, silhouette_scores):
        if not np.isnan(s):
            axes[0].text(i, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
    axes[0].set_title("Silhouette Score")
    axes[0].set_xlabel("Jumlah Cluster")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=.3)

    # Davies–Bouldin
    axes[1].plot(k_values, dbi_scores, "o-", color="orange", lw=2)
    axes[1].axvline(k_sel, ls="--", color="red", lw=2, alpha=0.8)
    for i, d in zip(k_values, dbi_scores):
        if not np.isnan(d):
            axes[1].text(i, d + 0.01, f"{d:.3f}", ha="center", fontsize=8)
    axes[1].set_title("Davies–Bouldin Index")
    axes[1].set_xlabel("Jumlah Cluster")
    axes[1].set_ylabel("Index")
    axes[1].grid(alpha=.3)

    plt.tight_layout()
    # st.pyplot(fig)
    show_fig_and_download(fig, f"cluster_performance_{method_sel.replace(' ','_')}_k{k_sel}")

    # caption akhir
    st.caption(
        f"Menampilkan hasil untuk **k = {k_sel}** (garis merah putus–putus). "
        "Silhouette lebih tinggi → cluster makin terpisah; Davies–Bouldin lebih rendah → cluster makin kompak."
    )
    if best_k is not None:
        st.caption(f"K optimum indikatif (berdasar Silhouette): **k = {best_k}**.")



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


def reshape_long_format(df_out: pd.DataFrame):
    # deteksi kolom lokasi & siapkan kolom hasil
    lokasi_col = _find_lokasi_col(df_out)

    harga_cols = [c for c in df_out.columns if isinstance(c, str) and re.search(r"^harga.*(merah|hijau)", c, flags=re.I)]
    if not harga_cols:
        cols = ["Kategori","Tahun","Bulan","Fitur","Nilai"] + ([lokasi_col] if lokasi_col else [])
        return pd.DataFrame(columns=cols)

    frames=[]
    for c in harga_cols:
        m, y = _parse_month_year(c)  # m bisa None untuk data tahunan
        if y is None: 
            continue
        fitur = "harga_cabe_rawit_merah" if re.search(r"merah", c, flags=re.I) else "harga_cabe_rawit_hijau"
        s = pd.to_numeric(df_out[c], errors="coerce")
        row = {
            "Kategori": df_out.get("Kategori"),
            "Tahun": y, "Bulan": m, "Fitur": fitur, "Nilai": s
        }
        if lokasi_col:
            row[lokasi_col] = df_out[lokasi_col].values
        frames.append(pd.DataFrame(row))

    if not frames:
        cols = ["Kategori","Tahun","Bulan","Fitur","Nilai"] + ([lokasi_col] if lokasi_col else [])
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["Nilai"])
    if out["Bulan"].isna().all():
        out["Bulan"] = 1  # agar plotting dengan datetime konsisten
    return out


def render_boxplot(df_out: pd.DataFrame):
    """
    Boxplot:
    - 'Seluruh Tahun' (default): facet per tahun (per kategori) + legend warna fitur.
    - 'Per Tahun':
        * DEFAULT: boxplot tahunan per kategori + legend warna fitur.
        * Checkbox 'Lihat per bulan (Jan–Des)'  -> 12 panel (tiap panel=1 bulan), box per kategori.
        * Checkbox 'Lihat per kuartal (Q1–Q4)' -> 4 panel (tiap panel=1 kuartal), box per kategori.
        * Jika dua-duanya dicentang, mode BULANAN diprioritaskan.
    Warna:
        - harga_cabe_rawit_merah : pink  (#F78FB3)
        - harga_cabe_rawit_hijau : hijau (#78E08F)
    """

    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data numerik untuk boxplot.")
        return

    # Urutan kategori konsisten
    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))

    # Rapikan nama kolom
    df_long = df_long.copy()
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]
    if "kategori" in df_long.columns:
        df_long["kategori"] = pd.Categorical(df_long["kategori"], categories=cat_order, ordered=True)

    fitur_unik = ["harga_cabe_rawit_merah", "harga_cabe_rawit_hijau"]
    tahun_list = sorted(df_long["tahun"].dropna().unique())

    # Warna & legend untuk fitur
    color_map = {
        "harga_cabe_rawit_merah": "#F78FB3",  # pink
        "harga_cabe_rawit_hijau": "#78E08F",  # hijau
    }
    legend_handles_master = [
        Patch(facecolor=color_map["harga_cabe_rawit_merah"], edgecolor="#333", label="Cabe Rawit Merah"),
        Patch(facecolor=color_map["harga_cabe_rawit_hijau"], edgecolor="#333", label="Cabe Rawit Hijau"),
    ]

    st.markdown("## 📈 Analisis Distribusi Boxplot Fitur per Tahun")
    tahun_pilihan = st.selectbox("Pilih Tahun:", ["Seluruh Tahun"] + [str(int(t)) for t in tahun_list], index=0)
    fitur_pilihan = st.multiselect("Pilih Fitur:", options=fitur_unik, default=fitur_unik)

    # ⛑️ Guard: cegah kosong → potensial ncol=0 pada legend
    if not fitur_pilihan:
        st.warning("Pilih minimal satu fitur.")
        return

    use_log = st.toggle("Gunakan skala logaritmik", value=False)

    # ---------- MODE 'SELURUH TAHUN' (per kategori) ----------
    if tahun_pilihan == "Seluruh Tahun":
        d = df_long.copy()
        d["nilai_standar"] = np.nan
        for fitur in fitur_pilihan:
            m = (d["fitur"] == fitur)
            if m.sum() > 0:
                vals = d.loc[m, "nilai"].values.reshape(-1, 1)
                d.loc[m, "nilai_standar"] = StandardScaler().fit_transform(vals).ravel()

        n_years = len(tahun_list)
        n_cols = 3
        n_rows = int(np.ceil(n_years / n_cols)) if n_years else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), squeeze=False)
        axes = axes.flatten()

        idx = -1
        for idx, year in enumerate(tahun_list):
            ax = axes[idx]
            data_y = d[d["tahun"] == year]
            if data_y.empty:
                ax.axis("off"); 
                continue

            cluster_order = (list(data_y["kategori"].cat.categories)
                             if hasattr(data_y["kategori"], "cat")
                             else sorted(data_y["kategori"].dropna().unique()))
            width = 0.25
            offsets = np.linspace(-width, width, len(fitur_pilihan))

            for i, fitur in enumerate(fitur_pilihan):
                sub = data_y[data_y["fitur"] == fitur]
                vals = [sub[sub["kategori"] == c]["nilai_standar"].dropna().values for c in cluster_order]
                pos = np.arange(len(cluster_order)) + offsets[i]
                ax.boxplot(
                    vals, positions=pos, widths=0.18, patch_artist=True,
                    boxprops=dict(facecolor=color_map.get(fitur, "#cccccc"), alpha=.9, edgecolor="#333"),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=2.5, alpha=.3)
                )

            ax.set_title(f"Tahun {int(year)}")
            ax.set_ylabel("Nilai Standarisasi")
            ax.set_xticks(range(len(cluster_order)))
            counts = data_y.groupby("kategori").size().reindex(cluster_order, fill_value=0)
            ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cluster_order], fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=.35)
            if use_log and (data_y["nilai_standar"].dropna() > 0).any():
                ax.set_yscale("log")

        # Matikan panel kosong
        for j in range((idx + 1) if idx >= 0 else 0, len(axes)):
            axes[j].axis("off")

        # Legend di level-figure (AMAN)
        leg_handles = []
        if "harga_cabe_rawit_merah" in fitur_pilihan:
            leg_handles.append(legend_handles_master[0])
        if "harga_cabe_rawit_hijau" in fitur_pilihan:
            leg_handles.append(legend_handles_master[1])
        if leg_handles:
            fig.legend(handles=leg_handles,
                       loc="upper center",
                       ncol=max(1, len(leg_handles)),
                       frameon=False, bbox_to_anchor=(0.5, 1.02))

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        # st.pyplot(fig)
        show_fig_and_download(fig, f"boxplot_all_years_{'_'.join(fitur_pilihan)}")
        return

    # ---------- MODE 'PER TAHUN' ----------
    year = int(tahun_pilihan)
    data_y = df_long[df_long["tahun"] == year]
    if data_y.empty:
        st.warning("Tidak ada data untuk tahun itu."); 
        return

    has_months = data_y["bulan"].notna().any()
    colA, colB = st.columns(2)
    with colA:
        per_bulan = st.checkbox("Lihat per bulan (Jan–Des)", value=False, disabled=not has_months,
                                help="Tampilkan 12 panel (Jan..Des), tiap panel berisi boxplot per kategori.")
    with colB:
        per_quartal = st.checkbox("Lihat per kuartal (Q1–Q4)", value=False, disabled=not has_months,
                                  help="Tampilkan 4 panel (Q1..Q4), tiap panel berisi boxplot per kategori.")
    if per_bulan and per_quartal:
        st.info("Mode bulanan & kuartal sama-sama dicentang. Menampilkan **bulanan** (prioritas).")

    # ---------- PER BULAN: 12 panel ----------
    if per_bulan and has_months:
        d = data_y.copy()
        bulan_order = list(range(1, 13))
        label_bulan = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]

        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True, squeeze=False)
        axes = axes.flatten()

        width = 0.22
        offsets = np.linspace(-width, width, len(fitur_pilihan))

        for idx, m in enumerate(bulan_order):
            ax = axes[idx]
            sub_m = d[d["bulan"] == m]
            if sub_m.empty:
                ax.set_axis_off(); 
                continue

            cluster_order = (list(sub_m["kategori"].cat.categories)
                             if hasattr(sub_m["kategori"], "cat")
                             else sorted(sub_m["kategori"].dropna().unique()))
            if not cluster_order:
                ax.set_axis_off(); 
                continue

            for i, fitur in enumerate(fitur_pilihan):
                subf = sub_m[sub_m["fitur"] == fitur]
                vals = [subf[subf["kategori"] == c]["nilai"].dropna().values for c in cluster_order]
                pos = np.arange(len(cluster_order)) + offsets[i]
                ax.boxplot(
                    vals, positions=pos, widths=0.18, patch_artist=True,
                    boxprops=dict(facecolor=color_map.get(fitur, "#cccccc"), alpha=.9, edgecolor="#333"),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=2.5, alpha=.3)
                )

            counts = sub_m.groupby("kategori").size().reindex(cluster_order, fill_value=0)
            ax.set_xticks(range(len(cluster_order)))
            ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cluster_order], fontsize=8)
            ax.set_title(f"{label_bulan[m-1]} — {year}", fontsize=11)
            if idx % 4 == 0:
                ax.set_ylabel("Harga (Rp/kg)")
            ax.grid(axis="y", linestyle="--", alpha=.35)

        # Legend (AMAN)
        leg_handles = []
        if "harga_cabe_rawit_merah" in fitur_pilihan:
            leg_handles.append(legend_handles_master[0])
        if "harga_cabe_rawit_hijau" in fitur_pilihan:
            leg_handles.append(legend_handles_master[1])
        if leg_handles:
            fig.legend(handles=leg_handles,
                       loc="upper center",
                       ncol=max(1, len(leg_handles)),
                       frameon=False, bbox_to_anchor=(0.5, 1.02))

        if use_log and (pd.to_numeric(d["nilai"], errors="coerce") > 0).any():
            for ax in axes:
                if ax.has_data(): 
                    ax.set_yscale("log")

        fig.suptitle(f"Tahun {year}", y=0.99, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        # st.pyplot(fig)
        show_fig_and_download(fig, f"boxplot_tahun_{year}_{'_'.join(fitur_pilihan)}")
        return

    # ---------- PER KUARTAL: 4 panel ----------
    if per_quartal and has_months:
        d = data_y.copy()
        quarter_map = {"Q1":[1,2,3], "Q2":[4,5,6], "Q3":[7,8,9], "Q4":[10,11,12]}

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, squeeze=False)
        axes = axes.flatten()

        width = 0.22
        offsets = np.linspace(-width, width, len(fitur_pilihan))

        for idx, (qname, months) in enumerate(quarter_map.items()):
            ax = axes[idx]
            sub_q = d[d["bulan"].isin(months)]
            if sub_q.empty:
                ax.set_axis_off(); 
                continue

            cluster_order = (list(sub_q["kategori"].cat.categories)
                             if hasattr(sub_q["kategori"], "cat")
                             else sorted(sub_q["kategori"].dropna().unique()))
            if not cluster_order:
                ax.set_axis_off(); 
                continue

            for i, fitur in enumerate(fitur_pilihan):
                subf = sub_q[sub_q["fitur"] == fitur]
                vals = [subf[subf["kategori"] == c]["nilai"].dropna().values for c in cluster_order]
                pos = np.arange(len(cluster_order)) + offsets[i]
                ax.boxplot(
                    vals, positions=pos, widths=0.18, patch_artist=True,
                    boxprops=dict(facecolor=color_map.get(fitur, "#cccccc"), alpha=.9, edgecolor="#333"),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=2.5, alpha=.3)
                )

            counts = sub_q.groupby("kategori").size().reindex(cluster_order, fill_value=0)
            ax.set_xticks(range(len(cluster_order)))
            ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cluster_order], fontsize=8)
            ax.set_title(f"{qname} — {year}", fontsize=11)
            if idx in (0, 2):
                ax.set_ylabel("Harga (Rp/kg)")
            ax.grid(axis="y", linestyle="--", alpha=.35)

        # Legend (AMAN)
        leg_handles = []
        if "harga_cabe_rawit_merah" in fitur_pilihan:
            leg_handles.append(legend_handles_master[0])
        if "harga_cabe_rawit_hijau" in fitur_pilihan:
            leg_handles.append(legend_handles_master[1])
        if leg_handles:
            fig.legend(handles=leg_handles,
                       loc="upper center",
                       ncol=max(1, len(leg_handles)),
                       frameon=False, bbox_to_anchor=(0.5, 1.02))

        if use_log and (pd.to_numeric(d["nilai"], errors="coerce") > 0).any():
            for ax in axes:
                if ax.has_data(): 
                    ax.set_yscale("log")

        fig.suptitle(f"Tahun {year}", y=0.99, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        # st.pyplot(fig)
        show_fig_and_download(fig, f"boxplot_quartal_{year}_{'_'.join(fitur_pilihan)}")
        return

    # ---------- DEFAULT: BOX PLOT SATU TAHUN (AGREGASI PER KATEGORI) ----------
    d = data_y.copy()
    fig, ax = plt.subplots(figsize=(10, 5))

    cluster_order = (list(d["kategori"].cat.categories)
                     if hasattr(d["kategori"], "cat")
                     else sorted(d["kategori"].dropna().unique()))

    width = 0.25
    offsets = np.linspace(-width, width, len(fitur_pilihan))

    vals_std_all = []
    for i, fitur in enumerate(fitur_pilihan):
        sub = d[d["fitur"] == fitur]
        vals_std = []
        for c in cluster_order:
            v = sub[sub["kategori"] == c]["nilai"].dropna().values.reshape(-1, 1)
            vals_std.append(StandardScaler().fit_transform(v).ravel() if len(v) else np.array([]))
        vals_std_all.extend(vals_std)  # untuk cek log > 0 nanti
        pos = np.arange(len(cluster_order)) + offsets[i]
        ax.boxplot(
            vals_std, positions=pos, widths=0.18, patch_artist=True,
            boxprops=dict(facecolor=color_map.get(fitur, "#cccccc"), alpha=.9, edgecolor="#333"),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=3, alpha=.3)
        )

    ax.set_title(f"Tahun {year}")
    ax.set_ylabel("Nilai Standarisasi")
    ax.grid(axis="y", linestyle="--", alpha=.35)
    counts = d.groupby("kategori").size().reindex(cluster_order, fill_value=0)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cluster_order], fontsize=9)

    # Legend (AMAN)
    leg_handles = []
    if "harga_cabe_rawit_merah" in fitur_pilihan:
        leg_handles.append(legend_handles_master[0])
    if "harga_cabe_rawit_hijau" in fitur_pilihan:
        leg_handles.append(legend_handles_master[1])
    if leg_handles:
        ax.legend(handles=leg_handles,
                  loc="upper left",
                  frameon=False,
                  ncol=max(1, len(leg_handles)),
                  title="Fitur")

    if use_log and any((arr > 0).any() for arr in vals_std_all if len(arr)):
        ax.set_yscale("log")

    plt.tight_layout()
    # st.pyplot(fig)
    show_fig_and_download(fig, f"boxplot_tahun_{year}_{'_'.join(fitur_pilihan)}_std")


def render_boxplot_combined(df_out: pd.DataFrame):
    
    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data numerik.")
        return

    # Normalisasi nama kolom & kategori terurut
    df_long = df_long.copy()
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]

    st.markdown("## 📈 Analisis Distribusi Boxplot Fitur Seluruh Tahun")
    fitur_unik = ["harga_cabe_rawit_merah", "harga_cabe_rawit_hijau"]
    fitur_pilihan = st.multiselect(
        "Pilih Fitur:", options=fitur_unik, default=fitur_unik, key="fitur_boxplot_combined"
    )
    # ⛑️ Guard biar gak bisa kosong (mencegah legend ncol=0)
    if not fitur_pilihan:
        st.warning("Pilih minimal satu fitur.")
        return

    use_log = st.toggle("Gunakan skala logaritmik", value=False, key="log_boxplot_combined")

    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    if "kategori" in df_long.columns:
        df_long["kategori"] = pd.Categorical(df_long["kategori"], categories=cat_order, ordered=True)

    # Standarisasi per fitur (supaya sebanding)
    df_long["nilai_standar"] = np.nan
    for fitur in fitur_pilihan:
        m = (df_long["fitur"] == fitur)
        if m.sum() > 0:
            vals = df_long.loc[m, "nilai"].values.reshape(-1, 1)
            df_long.loc[m, "nilai_standar"] = StandardScaler().fit_transform(vals).ravel()

    # Warna & legend master
    color_map = {
        "harga_cabe_rawit_merah": "#F78FB3",  # pink
        "harga_cabe_rawit_hijau": "#78E08F",  # hijau
    }
    legend_master = {
        "harga_cabe_rawit_merah": Patch(facecolor=color_map["harga_cabe_rawit_merah"], edgecolor="#333", label="Cabe Rawit Merah"),
        "harga_cabe_rawit_hijau": Patch(facecolor=color_map["harga_cabe_rawit_hijau"], edgecolor="#333", label="Cabe Rawit Hijau"),
    }

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    offsets = np.linspace(-width, width, len(fitur_pilihan))

    for i, fitur in enumerate(fitur_pilihan):
        sub = df_long[df_long["fitur"] == fitur]
        vals = [sub[sub["kategori"] == c]["nilai_standar"].dropna().values for c in cat_order]
        pos = np.arange(len(cat_order)) + offsets[i]
        ax.boxplot(
            vals, positions=pos, widths=0.18, patch_artist=True,
            boxprops=dict(facecolor=color_map.get(fitur, "#cccccc"), alpha=.9, edgecolor="#333"),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=2.5, alpha=.3)
        )

    ax.set_title("Distribusi Fitur vs Kategori (All Years)")
    ax.set_ylabel("Nilai Standarisasi")

    counts = df_long.groupby("kategori").size().reindex(cat_order, fill_value=0)
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in cat_order], fontsize=9)

    ax.grid(axis="y", linestyle="--", alpha=.4)

    # Skala log (hanya jika ada >0)
    if use_log and (df_long["nilai_standar"].dropna() > 0).any():
        ax.set_yscale("log")

    # Legend AMAN: hanya jika ada handle, ncol >= 1
    leg_handles = [legend_master[k] for k in fitur_pilihan if k in legend_master]
    if leg_handles:
        ax.legend(handles=leg_handles, loc="upper left", frameon=False,
                  ncol=max(1, len(leg_handles)), title="Fitur")

    plt.tight_layout()
    # st.pyplot(fig)
    show_fig_and_download(fig, f"boxplot_all_years_combined_{'_'.join(fitur_pilihan)}")


# ---------- Silhouette per cluster & Profil (skala asli) ----------
def render_silhouette_per_cluster(X_scaled, labels):
    try:
        from sklearn.metrics import silhouette_samples
    except Exception:
        st.info("scikit-learn metrics tidak tersedia."); return
    if X_scaled is None or len(X_scaled)==0: st.info("Tidak ada data."); return
    s = silhouette_samples(X_scaled, labels)
    df_s = pd.DataFrame({"Cluster":labels, "sil":s})
    avg = df_s.groupby("Cluster")["sil"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6,3.8))
    ax.bar([str(i) for i in avg.index], avg.values)
    for i,v in enumerate(avg.values): ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title("Silhouette rata-rata per Cluster"); ax.set_xlabel("Cluster"); ax.set_ylabel("Silhouette avg")
    ax.grid(axis="y", alpha=.3, linestyle="--")
    # st.pyplot(fig)
    show_fig_and_download(fig, "silhouette_per_cluster")
    st.caption("Semakin tinggi semakin baik. Cluster bernilai rendah/negatif layak ditinjau ulang.")

def render_cluster_profile(df_out: pd.DataFrame, feat_cols: list[str]):
    if df_out is None or df_out.empty or "Kategori" not in df_out.columns:
        st.info("Data belum tersedia untuk profil."); return
    used=[c for c in (feat_cols or []) if c in df_out.columns]
    if not used: st.info("Tidak ada kolom fitur skala asli yang cocok."); return

    num = df_out[["Kategori"]+used].copy()
    for c in used: num[c]=pd.to_numeric(num[c], errors="coerce")
    def iqr(x): q=np.nanpercentile(x,[25,75]); return q[1]-q[0]
    agg_mean=num.groupby("Kategori")[used].mean(numeric_only=True)
    agg_median=num.groupby("Kategori")[used].median(numeric_only=True)
    agg_iqr=num.groupby("Kategori")[used].agg(iqr)
    prof=pd.concat({"Mean":agg_mean,"Median":agg_median,"IQR":agg_iqr}, axis=1)
    st.markdown("#### 📑 Profil Cluster (skala asli)")
    st.dataframe(prof, use_container_width=True)
    download_excel_button(prof, "profil_cluster_skala_asli", sheet_name="profil_cluster")

def render_location_feature_means(df_out: pd.DataFrame, feat_cols: list[str]):
    """Tambahan: rata-rata fitur per lokasi (lintas tahun) di bawah profil cluster."""
    if df_out is None or df_out.empty: return
    lokasi_col = _find_lokasi_col(df_out)
    if not lokasi_col: return
    used=[c for c in (feat_cols or []) if c in df_out.columns]
    if not used: return
    tmp = df_out[[lokasi_col,"Kategori"]+used].copy()
    for c in used: tmp[c]=pd.to_numeric(tmp[c], errors="coerce")
    tmp["Rata_Rata"]=tmp[used].mean(axis=1, skipna=True)
    out = (tmp.groupby([lokasi_col,"Kategori"])["Rata_Rata"].mean()
           .reset_index().sort_values("Rata_Rata", ascending=False))
    st.markdown("#### 📍 Rata-rata Fitur per Lokasi (lintas tahun)")
    st.dataframe(out, use_container_width=True)
    download_excel_button(out, "rata_rata_fitur_per_lokasi", sheet_name="rata_rata")

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

        m = folium.Map(location=[-2.5, 118], zoom_start=4.6, tiles="cartodbpositron")

        ordered = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
        cat_to_hex = {c: PALETTE.get(c.lower(), "#666666") for c in ordered}

        if "Kategori" in df_pts.columns:
            df_pts["Kategori"] = df_pts["Kategori"].astype(str).str.strip()
            df_pts["Kategori"] = pd.Categorical(df_pts["Kategori"], categories=ordered, ordered=True)

        def _mean_across_years(row, key):
            patt = re.compile(rf"^{key}.*(19|20)\d{{2}}", flags=re.I)
            idxs = [c for c in row.index if isinstance(c, str) and patt.search(c)]
            if not idxs:
                idxs = [c for c in row.index if isinstance(c, str) and str(c).lower().replace(" ","_").startswith(key)]
            if not idxs:
                return np.nan
            vals = pd.to_numeric(row[idxs], errors="coerce")
            return float(vals.mean())

        for _, r in df_pts.iterrows():
            kategori = str(r.get("Kategori", "Sedang"))
            hex_color = cat_to_hex.get(kategori, "#666666")

            nama = next((r.get(c) for c in ["Kabupaten/Kota","Kabupaten","Kota","Provinsi","Lokasi"] if c in df_pts.columns), "-")
            merah   = _mean_across_years(r, "harga_cabe_rawit_merah")
            hijau   = _mean_across_years(r, "harga_cabe_rawit_hijau")

            html = f"""<b>{nama}</b><br>
            Kategori: <b>{kategori}</b><br>Cluster: {r.get('Cluster','-')}
            <hr style='margin:4px 0;'>
            <b>Rata-Rata Harga:</b><br>
            • Rawit Merah: {merah:,.0f} Rp/kg<br>
            • Rawit Hijau: {hijau:,.0f} Rp/kg"""

            folium.CircleMarker(
                location=[r[lat_col], r[lon_col]],
                radius=6,
                color=hex_color,
                weight=1,
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.9,
                popup=folium.Popup(html, max_width=260),
            ).add_to(m)

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
        # ⬇️ Tambah tombol download HTML peta
        try:
            html_data = m.get_root().render().encode("utf-8")
            st.download_button(
                "⬇️ Download Peta (HTML)",
                data=html_data,
                file_name="peta_hasil_klaster.html",
                mime="text/html",
                use_container_width=True,
                key="dl_map_html"
            )
        except Exception as e:
            st.caption(f"Gagal menyiapkan unduhan peta: {e}")
    except Exception as e:
        st.warning(f"Gagal membuat peta: {e}")



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

    # ---------- jumlah data ----------
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
    show_fig_and_download(fig, f"dendrogram_{title_suffix}_k{k_sel or '?'}")

    # Verifikasi jumlah komponen pada garis potong
    try:
        eps = 1e-9
        k_found = len(np.unique(fcluster(Z, t=cut_height - eps, criterion="distance")))
        st.caption(f"Verifikasi komponen pada garis: **k = {k_found}**.")
    except Exception:
        pass

# ====== Tambahkan HELPER di bagian util (di atas render_tren_hasil_panen) ======
def _robust_ylim(values: np.ndarray, pad_ratio: float = 0.08):
    """Batas Y pakai quantile agar outlier tidak merusak skala."""
    vals = pd.to_numeric(pd.Series(values).dropna(), errors="coerce")
    if vals.empty:
        return None, None
    lo, hi = np.quantile(vals, [0.02, 0.98])
    pad = (hi - lo) * pad_ratio if hi > lo else 1.0
    return max(0, lo - pad), hi + pad

def _smooth_series(x: np.ndarray, y: np.ndarray, window: int = 3):
    """Smoothing median bergeser agar kurva lebih halus (tanpa menggeser puncak)."""
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def _fit_ylim_full(arr_like, pad=0.06, clamp_zero=False):
    """Hitung batas y dari data penuh + padding; tidak memangkas puncak/cekung.
    pad=0.06 berarti tambah 6% span di atas & bawah."""
    
    a = pd.to_numeric(pd.Series(arr_like), errors="coerce").astype(float).to_numpy()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None, None
    lo, hi = float(np.min(a)), float(np.max(a))
    if clamp_zero and lo > 0:
        lo = 0.0
    span = max(1.0, hi - lo)
    return lo - pad * span, hi + pad * span


# =============== GANTI FUNGSI INI DENGAN VERSI YANG BARU ===============
def render_tren_harga(df_out: pd.DataFrame):
    """
    Tren Harga Cabai dengan:
      • Garis (Top-N), Facet per Kategori, Heatmap
      • Smoothing median-3, marker, direct label
      • PENCARIAN lokasi: 'Bandingkan Lokasi (pisahkan dengan koma)'
        + opsi 'Tampilkan hanya yang dicari'
    Prasyarat helper: reshape_long_format, _find_lokasi_col, _smooth_series,
                      _fit_ylim_full, get_ordered_categories
    """
    
    st.markdown("## 📈 Tren Harga Cabai")

    if df_out is None or df_out.empty:
        st.warning("Dataset kosong."); return

    dlong = reshape_long_format(df_out)  # -> kolom: Fitur, Nilai, Tahun, Bulan, Kategori, ...lokasi
    if dlong.empty:
        st.warning("Tidak ada data harga untuk dianalisis."); return

    # Siapkan tanggal
    if "Bulan" not in dlong.columns:
        dlong["Bulan"] = 1
    dlong["Tanggal"] = pd.to_datetime(dict(
        year=pd.to_numeric(dlong["Tahun"], errors="coerce").fillna(0).astype(int),
        month=pd.to_numeric(dlong["Bulan"], errors="coerce").fillna(1).astype(int).clip(1, 12),
        day=1
    ), errors="coerce")

    # Kolom lokasi
    lokasi_col = _find_lokasi_col(dlong)
    if not lokasi_col:
        st.warning("Kolom lokasi (Provinsi/Kabupaten/Kota) tidak ditemukan."); return

    # Daftar fitur (auto)
    fitur_opsi = sorted([str(x) for x in dlong.get("Fitur", pd.Series([])).dropna().unique().tolist()]) \
                 or ["harga_cabe_rawit_merah", "harga_cabe_rawit_hijau"]
    fitur_label = {f: f.replace("_", " ").title() for f in fitur_opsi}

    # ====================== UI utama ======================
    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.1, 1.2])
    with c1:
        fitur_sel = st.selectbox("Pilih Fitur:", fitur_opsi,
                                 format_func=lambda k: fitur_label.get(k, k),
                                 key="tren_harga_fitur")
    with c2:
        mode_view = st.selectbox("Mode Tampilan:", ["Garis (Top-N)", "Facet per Kategori", "Heatmap"],
                                 key="tren_harga_mode")
    with c3:
        topn = st.slider("Top Lokasi:", 1, 20, 8, key="tren_harga_topn")
    with c4:
        urutkan = st.radio("Urutan:", ["Terbesar", "Terkecil"], horizontal=True, index=0, key="tren_harga_urut")

    c5, c6, c7 = st.columns([1.3, 1.2, 1.6])
    with c5:
        smooth = st.checkbox("Haluskan garis (median 3)", value=True, key="tren_harga_smooth")
    with c6:
        show_markers = st.checkbox("Tampilkan marker", value=True, key="tren_harga_marker")
    with c7:
        direct_label = st.checkbox("Label langsung di kanan (tanpa legend)", value=False,
                                   key="tren_harga_direct")

    # ====================== PENCARIAN lokasi ======================
    q = st.text_input(
        "🔎 Bandingkan Lokasi (pisahkan dengan koma):",
        key="tren_harga_query", help="Contoh: blitar, tuban, kediri"
    )
    only_q = st.checkbox("Tampilkan hanya yang dicari", value=False, key="tren_harga_onlyq")

    def _norm(s: str) -> str:
        s = str(s).lower()
        s = re.sub(r'\bkabupaten\b|\bkab\.\b', 'kab', s)
        s = re.sub(r'\bkota\b|\bkota\.\b', 'kota', s)
        s = re.sub(r'\bprovinsi\b|\bprov\.\b', 'prov', s)
        s = re.sub(r'[^a-z0-9 ]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _parse_query(qstr: str) -> list[str]:
        if not qstr: return []
        return [_norm(t) for t in re.split(r'[;,]', qstr) if t.strip()]

    # Filter fitur dulu
    d = dlong[dlong["Fitur"] == fitur_sel].copy()
    d = d.dropna(subset=["Tanggal", "Nilai"])
    if d.empty:
        st.warning("Tidak ada data untuk fitur terpilih."); return

    # Cari kecocokan
    terms = _parse_query(q)
    all_locs = d[lokasi_col].astype(str).unique().tolist()
    matches = [loc for loc in all_locs if terms and any(_norm(loc).find(t) >= 0 for t in terms)]
    if q:
        st.caption("Lokasi cocok: " + (", ".join(matches) if matches else "— tidak ada —"))

    # Jika “hanya yang dicari” → pakai itu saja
    if only_q and matches:
        d = d[d[lokasi_col].isin(matches)]

    ascending_flag = (urutkan == "Terkecil")
    rank_label = "Tertinggi" if not ascending_flag else "Terendah"
    mean_by_loc = d.groupby(lokasi_col)["Nilai"].mean().sort_values(ascending=ascending_flag)

    # Jadikan Top-N, tapi kalau ada hasil pencarian, pastikan semuanya ikut
    topn_eff = max(topn, len(matches)) if matches and not only_q else topn
    top_lokasi = mean_by_loc.index.tolist()[:topn_eff]
    if matches and not only_q:
        # union: hasil pencarian + top-N
        top_lokasi = list(dict.fromkeys(matches + top_lokasi))[:max(topn_eff, len(matches))]

    d_top = d[d[lokasi_col].isin(top_lokasi)].copy()

    # padding Y
    ylo, yhi = _fit_ylim_full(d_top["Nilai"])

    # palet
    colors = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)

    # ================= MODE 1: Garis (Top-N) =================
    if mode_view == "Garis (Top-N)":
        fig, ax = plt.subplots(figsize=(10.6, 6.0))
        ordered_for_plot = mean_by_loc.loc[top_lokasi].index.tolist()

        handles, labels_leg = [], []
        for i, lok in enumerate(ordered_for_plot):
            sub = d_top[d_top[lokasi_col] == lok].sort_values("Tanggal")
            y = sub["Nilai"].to_numpy()
            if smooth:
                y = _smooth_series(sub["Tanggal"].to_numpy(), y, window=3)

            (line,) = ax.plot(
                sub["Tanggal"], y,
                linewidth=2.2,
                marker="o" if show_markers else None, markersize=4,
                alpha=0.95, color=colors[i % len(colors)]
            )
            lab_cat = sub["Kategori"].dropna().mode().iat[0] if not sub["Kategori"].dropna().empty else "-"
            handles.append(line); labels_leg.append(f"{lok} ({lab_cat})")

            if direct_label and len(sub):
                x_last = sub["Tanggal"].iloc[-1]; y_last = y[-1]
                ax.text(x_last + pd.Timedelta(days=25), y_last, s=lok,
                        va="center", fontsize=9, color=colors[i % len(colors)])

        years = sorted(pd.to_datetime(d_top["Tanggal"]).dt.year.unique())
        if years:
            xticks = pd.to_datetime([f"{int(y)}-01-01" for y in years])
            ax.set_xticks(xticks); ax.set_xticklabels([str(int(y)) for y in years])

        if ylo is not None: ax.set_ylim(ylo, yhi)
        if direct_label:
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.06)

        ax.grid(True, linestyle="--", alpha=.35)
        # ttl = f"{fitur_label.get(fitur_sel, fitur_sel)} — Tren (Top-{len(ordered_for_plot)})"
        ttl = f"{fitur_label.get(fitur_sel, fitur_sel)} — Tren ({rank_label}-{len(ordered_for_plot)})"
        ax.set_title(ttl); ax.set_xlabel("Tahun"); ax.set_ylabel("Harga (Rp/kg)")

        if not direct_label:
            ax.legend(handles=handles, labels=labels_leg, title="Lokasi", fontsize=8,
                      ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, handlelength=3.0,
                      handler_map={mpl.lines.Line2D: HandlerLine2D(numpoints=2)})
            plt.tight_layout(rect=[0, 0, 0.78, 1])
        else:
            plt.tight_layout()

        # st.pyplot(fig)
        show_fig_and_download(fig, f"tren_{fitur_sel}_topN")
    

    # ================= MODE 2: Facet per Kategori =================
    elif mode_view == "Facet per Kategori":
        cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
        cats = [c for c in cat_order if c in d_top["Kategori"].dropna().unique().tolist()] \
               or sorted(d_top["Kategori"].dropna().unique())

        n = max(len(cats), 1)
        ncols = 2 if n >= 2 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.6, 4.3 * nrows),
                                 sharex=True, sharey=False, squeeze=False)
        axes = axes.flatten()

        global_handles, global_labels = [], []

        for idx, cat in enumerate(cats):
            ax = axes[idx]
            sub_cat = d_top[d_top["Kategori"] == cat].copy()
            if sub_cat.empty:
                ax.set_axis_off(); continue

            mean_by_loc_cat = sub_cat.groupby(lokasi_col)["Nilai"].mean().sort_values(ascending=ascending_flag)
            locs_cat = mean_by_loc_cat.head(min(topn_eff, 6)).index.tolist()

            local_handles, local_labels = [], []
            for i, lok in enumerate(locs_cat):
                sc = sub_cat[sub_cat[lokasi_col] == lok].sort_values("Tanggal")
                y = sc["Nilai"].to_numpy()
                if smooth: y = _smooth_series(sc["Tanggal"].to_numpy(), y, window=3)
                (line,) = ax.plot(sc["Tanggal"], y, linewidth=2.0,
                                  marker="o" if show_markers else None, markersize=3.5,
                                  color=colors[i % len(colors)], alpha=.95)
                local_handles.append(line); local_labels.append(lok)

            yy_lo, yy_hi = _fit_ylim_full(sub_cat["Nilai"])
            if yy_lo is not None: ax.set_ylim(yy_lo, yy_hi)

            for h, lab in zip(local_handles[:2], local_labels[:2]):
                global_handles.append(h); global_labels.append(lab)

            ax.set_title(cat); ax.grid(True, linestyle="--", alpha=.35)
            years = sorted(pd.to_datetime(sub_cat["Tanggal"]).dt.year.unique())
            if years:
                xticks = pd.to_datetime([f"{int(y)}-01-01" for y in years])
                ax.set_xticks(xticks); ax.set_xticklabels([str(int(y)) for y in years])

        for j in range(len(cats), len(axes)): axes[j].set_axis_off()

        if global_handles:
            fig.legend(global_handles, global_labels, title="Contoh Lokasi", ncol=1, frameon=False,
                       loc="center right", bbox_to_anchor=(1.02, 0.5),
                       handlelength=3.0, handler_map={mpl.lines.Line2D: HandlerLine2D(numpoints=2)})
            plt.tight_layout(rect=[0, 0, 0.85, 0.98])
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.98])

        fig.suptitle(
            f"{fitur_label.get(fitur_sel, fitur_sel)} — Facet per Kategori "
            # f"(Top-{min(topn_eff,6)} per facet • {urutkan})",
            f"({rank_label}-{min(topn_eff,6)} per facet • {urutkan})",
            y=0.995, fontsize=12, fontweight="bold"
        )
        # st.pyplot(fig)
        show_fig_and_download(fig, f"tren_{fitur_sel}_facet_kategori")

    # ================= MODE 3: Heatmap (Lokasi × Tahun) =================
    else:
        pvt = d.groupby([lokasi_col, "Tahun"])["Nilai"].mean().unstack("Tahun")
        # Urutan dasar berdasarkan rata-rata antar tahun
        order = pvt.mean(axis=1).sort_values(ascending=ascending_flag)
        # Top-N dengan memastikan hasil query ikut
        if only_q and matches:
            # rows = matches
            rows = [idx for idx in order.index if idx in matches]
        else:
            # rows = pvt.mean(axis=1).sort_values(ascending=ascending_flag).head(topn_eff).index.tolist()
            base_rows = order.head(topn_eff).index.tolist()
            if matches:
                # rows = list(dict.fromkeys(matches + rows))[:max(topn_eff, len(matches))]
                rows = list(dict.fromkeys(matches + base_rows))[:max(topn_eff, len(matches))]
            else:
                rows = base_rows
        
        # Jika 'Terkecil', tampilkan sebagai Bottom dengan membalik urutan baris
        if ascending_flag:
            rows = rows[::-1]
        # pvt = pvt.loc[[r for r in rows if r in pvt.index]]
        pvt = pvt.loc[[r for r in rows if r in pvt.index]]

        fig_w = 1.2 * (len(pvt.columns) or 1) + 4
        fig_h = 0.45 * (len(pvt.index) or 1) + 2.8
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(pvt.values, aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Harga (Rp/kg)")

        ax.set_xticks(range(len(pvt.columns)))
        ax.set_xticklabels([str(int(c)) for c in pvt.columns], rotation=0)
        ax.set_yticks(range(len(pvt.index))); ax.set_yticklabels(pvt.index)

        ax.set_title(f"{fitur_label.get(fitur_sel, fitur_sel)} — Heatmap ({rank_label}-{len(pvt.index)} Lokasi • {urutkan})")

        mean_val = np.nanmean(pvt.values)
        for (i, j), val in np.ndenumerate(pvt.values):
            if not np.isnan(val):
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center",
                        fontsize=8, color=("white" if val > mean_val else "black"))

        plt.tight_layout()
        # st.pyplot(fig)
        show_fig_and_download(fig, f"tren_{fitur_sel}_heatmap") 



def render_top_lokasi(df_out: pd.DataFrame):
    """Bar chart Top-N lokasi.
    Default = agregasi TAHUNAN (pilih tahun). Centang 'Lihat per bulan' untuk filter per-bulan."""
    # ===== helpers =====
    def _find_year_month_options(d):
        th = sorted(d["Tahun"].dropna().unique().astype(int).tolist())
        bl = sorted(d["Bulan"].dropna().unique().astype(int).tolist())
        return th, bl

    # ===== guard =====
    st.markdown("## 🏆 Top Lokasi Harga")
    if df_out is None or df_out.empty:
        st.warning("Dataset kosong."); return

    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data harga untuk ranking."); return

    lokasi_col = _find_lokasi_col(df_long)
    if not lokasi_col:
        st.warning("Kolom lokasi (Provinsi/Kabupaten/Kota) tidak ditemukan."); return

    # ===== UI =====
    fitur_opsi = ["harga_cabe_rawit_merah", "harga_cabe_rawit_hijau"]
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        fitur_sel = st.selectbox("Pilih Fitur:", fitur_opsi, key="top_fitur_yearmonth")
    with c2:
        urutkan = st.radio("Urutan:", ["Terbesar", "Terkecil"], horizontal=True)
    with c3:
        topn = st.slider("Jumlah Lokasi:", 3, 30, 10)

    # per_bulan = st.checkbox("Lihat per bulan (Jan–Des)", value=False)
    periode = st.radio("Periode:", ["Per Tahun", "Semua Tahun"], horizontal=True, key="top_periode")

    # Checkbox per-bulan hanya relevan untuk Per Tahun
    if periode == "Per Tahun":
        per_bulan = st.checkbox("Lihat per bulan (Jan–Des)", value=False, key="top_per_bulan")
    else:
        per_bulan = False

    # filter fitur
    d = df_long[df_long["Fitur"] == fitur_sel].copy()
    if d.empty:
        st.warning("Tidak ada data untuk fitur terpilih."); return

    tahun_list, bulan_list = _find_year_month_options(d)
    if not tahun_list:
        st.warning("Tidak ada informasi tahun."); return

    # pilih waktu
    if periode == "Per Tahun":
        if per_bulan and bulan_list:
            c1, c2 = st.columns(2)
            with c1:
                th_sel = st.selectbox("Pilih Tahun:", tahun_list, index=0, key="top_th_tahun")
            with c2:
                bl_sel = st.selectbox(
                    "Pilih Bulan:", bulan_list, index=0,
                    format_func=lambda m: ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][m-1],
                    key="top_bl_bulan"
                )
            d_sel = d[(d["Tahun"] == th_sel) & (d["Bulan"] == bl_sel)].copy()
            subtitle = f"Bulan {['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'][bl_sel-1]} {th_sel}"
        else:
            th_sel = st.selectbox("Pilih Tahun:", tahun_list, index=0, key="top_th_tahun_only")
            d_sel = d[d["Tahun"] == th_sel].copy()
            subtitle = f"Tahun {th_sel}"
    else:
        # Keseluruhan semua tahun yang tersedia
        y_min, y_max = min(tahun_list), max(tahun_list)
        d_sel = d.copy()
        subtitle = f"Semua Tahun {y_min}–{y_max}"

    if d_sel.empty:
        st.warning("Tidak ada data pada periode yang dipilih."); return

    # ranking
    agg = d_sel.groupby([lokasi_col, "Kategori"])["Nilai"].mean().reset_index()
    ascending = (urutkan == "Terkecil")
    rank_label = "Tertinggi" if urutkan == "Terbesar" else "Terendah"
    # subset = agg.sort_values("Nilai", ascending=ascending).head(topn).reset_index(drop=True)
    # agg = agg.sort_values("Nilai", ascending=ascending).head(topn).reset_index(drop=True)
    subset = agg.sort_values("Nilai", ascending=ascending).head(topn).reset_index(drop=True)

    # Urutan untuk tampilan: selalu ASC agar yang kecil berada di bawah dan yang besar di atas
    plot_df = subset.sort_values("Nilai", ascending=True).reset_index(drop=True)

    # warna kategori
    cat_to_color = {
        "Sangat Rendah":"#d73027","Rendah":"#fc8d59","Cukup Rendah":"#fee090",
        "Sedang":"#ffffbf","Cukup Tinggi":"#e0f3f8","Tinggi":"#91bfdb","Sangat Tinggi":"#4575b4"
    }
    # warna sesuai data yang TAMPIL
    bar_colors = [cat_to_color.get(k, "#999999") for k in plot_df["Kategori"]]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.bar(range(len(plot_df)), plot_df["Nilai"],
              color=bar_colors, edgecolor="#333", linewidth=.6)

    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + (abs(h)*.01 + 1e-6), f"{h:,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1c2833")

    label = fitur_sel.replace("_"," ").title()
    ax.set_title(f"{rank_label.upper()} {len(plot_df)} LOKASI — {label.upper()} • {subtitle}")
    ax.set_ylabel("Harga (Rp/kg)"); ax.set_xlabel("Lokasi")
    ax.grid(axis="y", linestyle="--", alpha=.45); ax.set_axisbelow(True)

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels([f"{row[lokasi_col]} ({row['Kategori']})" for _, row in plot_df.iterrows()],
                   rotation=45, ha="right", fontsize=9)

    # LEGEND: hanya kategori yang tampil
    present = plot_df["Kategori"].dropna().astype(str).unique().tolist()
    legend_handles = [Patch(facecolor=cat_to_color.get(k, "#999999"), edgecolor="#333", label=k) for k in present]
    ax.legend(handles=legend_handles, title="Kategori",
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    # st.pyplot(fig)
    show_fig_and_download(fig, f"top_lokasi_{fitur_sel}_{subtitle.replace(' ','_')}")


def render_korelasi_harga(df_out: pd.DataFrame):
    """
    Heatmap korelasi antar fitur harga:
      - harga_cabe_rawit_merah
      - harga_cabe_rawit_hijau
    Sampel per (Lokasi, Tahun) pada rentang tahun yang dipilih.
    """
    
    st.markdown("### 🔗 Korelasi Antar Fitur Harga")

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

    # daftar fitur harga yang didukung
    feats_all = ["harga_cabe_rawit_merah", "harga_cabe_rawit_hijau"]
    # keep only yang benar-benar tersedia kolom tahunnya
    feats_avail = [f for f in feats_all if _year_cols(f)]

    if not feats_avail:
        st.info("Tidak ditemukan kolom harga bertahun di dataset."); return

    # UI
    c1, c2, c3 = st.columns([1.4, 1.1, 1.2])
    with c1:
        feats_sel = st.multiselect("Pilih Fitur Harga:", feats_avail, default=feats_avail, key="kor_harga_feat")
    with c2:
        metode = st.selectbox("Metode:", ["pearson","spearman","kendall"], index=0, key="kor_harga_method")
    with c3:
        agreg = st.radio("Periode:", ["Semua Tahun", "Rentang Tahun"], horizontal=True, key="kor_harga_period")

    if len(feats_sel) < 2:
        st.info("Pilih minimal dua fitur harga untuk korelasi."); return

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
        yr = st.slider("Pilih Rentang Tahun:", y_min, y_max, (max(y_min, 2015), y_max), key="kor_harga_years")
        y_lo, y_hi = yr
    else:
        y_lo, y_hi = y_min, y_max

    # filter lokasi opsional sederhana
    q = st.text_input("Filter Lokasi (opsional, pisahkan koma)", "", key="kor_harga_q")
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
        st.info("Kolom fitur harga valid kurang dari dua pada periode yang dipilih."); return

    # inner join supaya observasi align di semua fitur
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

    label_map = {
        "harga_cabe_rawit_merah": "Harga Rawit Merah",
        "harga_cabe_rawit_hijau": "Harga Rawit Hijau",
    }
    ticks = [label_map.get(c, c) for c in corr_cols]
    ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(ticks)
    ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(ticks)

    for (i, j), v in np.ndenumerate(corr_mat.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=("white" if abs(v) > 0.5 else "black"), fontsize=10)

    ttl = f"{y_lo}–{y_hi}"
    if tokens:
        ttl += " • filter lokasi aktif"
    ax.set_title(f"Korelasi Antar Fitur Harga • {ttl}")

    plt.tight_layout()
    show_fig_and_download(fig, f"korelasi_harga_{y_lo}_{y_hi}")



# ============================ Session ============================
def _ensure_session_state():
    defaults = {
        "clustering_done": False, "results_df": None, "df_out": None, "X_scaled": None,
        "labels": None, "cluster_label_map": {}, "method_used": None, "k_used": None,
        "ordered_categories": CAT_ORDER[:], "pca_info": {"used":False}
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v


# ==== PDF helpers ====
def _add_report_fig(fig, key="report_figs"):
    # simpan figure ke session agar bisa dikompilasi jadi PDF
    lst = st.session_state.get(key, [])
    lst.append(fig)
    st.session_state[key] = lst


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



# ============================== APP ==============================
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


    st.markdown("<h1 style='text-align:center; font-weight:800;'>Data Harga Cabai Rawit di Indonesia</h1>", unsafe_allow_html=True)
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
        # Resource Center dengan template sesuai level
        
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

        # Uploader
        # st.subheader(f"📤 Upload Data {level_sel} (Excel .xlsx)")
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
        if level_sel=="Provinsi":
            if "Provinsi" not in map(str, df_raw.columns):
                st.error("Kolom **Provinsi** wajib ada.")
                return
        else:
            if not any(c.strip().lower() in ("kabupaten/kota","kabupaten","kota") for c in map(str, df_raw.columns)):
                st.error("Kolom **Kabupaten/Kota** (atau 'Kabupaten'/'Kota') wajib ada.")
                return

    # ========================= MODE: DATASET CONTOH =========================
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
        ["K-Means", "Hierarchical Clustering"]
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
    # Diagnostics selalu tampil bila ada
    with st.expander("🔎 Kolom fitur yang dipakai & kualitasnya", expanded=False):
        st.write("Dipakai untuk clustering:", st.session_state.get("diag_used_cols","—"))
        st.write("Banyaknya kolom per keluarga:", st.session_state.get("diag_counts","—"))
        st.write("Proporsi NaN per keluarga:", st.session_state.get("diag_nan","—"))
        pinfo = st.session_state.get("pca_info", {"used":False})
        st.write("**PCA**:", "tidak dipakai" if not pinfo.get("used") else f"{pinfo['n_components']} komponen, explained ≈ {pinfo.get('explained',0):.2%}")

    if st.session_state.get("clustering_done", False):
        df_out = st.session_state["df_out"]
        render_cluster_performance(st.session_state["X_scaled"], st.session_state["method_used"])

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.markdown("### 📌 Hasil Tabel")
            show_cols = [c for c in df_out.columns if c.lower() in ["lokasi","provinsi","kabupaten","kabupaten/kota"]] + ["Cluster","Kategori"]
            show_cols = [c for c in show_cols if c in df_out.columns]
            st.dataframe(df_out[show_cols] if show_cols else df_out.head(20), use_container_width=True)
            download_excel_button(
                df_out[show_cols] if show_cols else df_out,
                "hasil_tabel_cluster",
                sheet_name="hasil_cluster"
            )

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
            show_fig_and_download(fig, "distribusi_kategori")

        st.markdown("### 🧭 Visualisasi Komposisi Cluster (PCA 2D)")
        render_pca_scatter_visual(st.session_state["X_scaled"], st.session_state["df_out"], st.session_state["labels"])

        with st.expander("📑 Profil Cluster (skala asli) + Rata-rata per Lokasi", expanded=False):
            render_cluster_profile(df_out, st.session_state.get("diag_used_cols", []))
            render_location_feature_means(df_out, st.session_state.get("diag_used_cols", []))

        st.markdown("### 📋 Anggota Setiap Cluster")
        lokasi_cols = [c for c in df_out.columns if c.lower() in ["provinsi","kabupaten","kabupaten/kota","lokasi"]]
        lokasi_col = lokasi_cols[0] if lokasi_cols else df_out.columns[0]
        rows = []
        for cid, label in st.session_state.get("cluster_label_map", {}).items():
            anggota = df_out[df_out["Cluster"] == cid][lokasi_col].astype(str).tolist()
            rows.append({
                "Cluster": f"{label} (Cluster {cid})",
                "Jumlah Anggota": len(anggota),
                "Anggota": ", ".join(anggota) if anggota else "-"
            })

        anggota_df = pd.DataFrame(rows)
        st.dataframe(anggota_df, use_container_width=True)
        download_excel_button(anggota_df, "anggota_per_cluster", sheet_name="anggota_cluster")


        # if st.session_state["method_used"] == "Hierarchical Clustering":
        #     render_dendrogram_from_X(st.session_state["X_scaled"], title_suffix=level_sel, k_sel=st.session_state["k_used"])

        if st.session_state["method_used"] == "Hierarchical Clustering":
            render_dendrogram_adaptive(st.session_state["X_scaled"], level_sel=level_sel, title_suffix=level_sel, k_sel=st.session_state["k_used"], max_visible_labels=60, max_label_len=18)
        
        render_points_map(df_out, title="Pemetaan Titik Lokasi (Latitude/Longitude)")

        st.markdown("### 📋 Analisis Lanjutan")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Gabungan Seluruh Tahun","📆 Per Tahun","📈 Tren Harga","🏆 Lokasi Tertinggi", "🔗 Korelasi Fitur"])
        with tab1: render_boxplot_combined(df_out)
        with tab2: render_boxplot(df_out)
        with tab3: render_tren_harga(df_out)
        with tab4: render_top_lokasi(df_out)
        with tab5: render_korelasi_harga(df_out)
        
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




# Jalankan app() saat file dieksekusi oleh streamlit
# if __name__ == "__main__":
#     app()
