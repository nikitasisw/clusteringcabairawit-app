# Streamlit: Pertanian & Harga Cabai Rawit — Clustering, Peta, Analitik, PDF
# Versi ini sudah dipatch agar aman saat kolom 'Kategori' belum ada.
# Dendrogram dihapus karena tidak dipanggil di UI (pembersihan fungsi tak terpakai).

import io, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Peta
import folium
from streamlit_folium import st_folium

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import openpyxl
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False


# ======================= Konfigurasi Paths =======================
TEMPLATE_PATHS = {
    "Provinsi":       str(Path("dataset") / "template_dataset_pertanian_provinsi.xlsx"),
}
DATASET_PATHS = {
    "Provinsi":       str(Path("dataset") / "dataset_pertanian_harga.xlsx"),
}


def _load_template_from_path(level: str) -> bytes:
    p = Path(TEMPLATE_PATHS.get(level, ""))
    if not p.exists():
        raise FileNotFoundError(f"Template tidak ditemukan: {p}")
    return p.read_bytes()


def _load_dataset_from_path(level: str) -> pd.DataFrame:
    p = Path(DATASET_PATHS.get(level, ""))
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    return pd.read_excel(p, engine="openpyxl")


# ------------------------ Konstanta UI/Kategori ------------------------
CAT_ORDER = ["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi","Sangat Tinggi"]
CAT_RANK  = {name:i for i,name in enumerate(CAT_ORDER)}
FEATURE_WEIGHTS = {"luas":0.15, "produksi":0.35, "produkt":0.50}
PRICE_FEATURE_WEIGHTS = {"harga_merah": 0.5, "harga_hijau": 0.5}

def _feature_weight_for_col(col: str, weights: dict[str, float]) -> float:
    cl = col.lower()
    for k,w in weights.items():
        if cl.startswith(k): return float(w)
    return 1.0


# ============================== Utils ==============================
def detect_granularity(df: pd.DataFrame) -> str:
    cols = [str(c).lower() for c in df.columns]
    if any(re.search(r'(19|20)\d{2}', c) for c in cols):
        return "tahunan"
    if any(("bulan" in c) or re.search(r"\bmonth\b", c) for c in cols):
        return "bulanan"
    if any(("tanggal" in c) or ("date" in c) or ("waktu" in c) or ("time" in c) for c in cols):
        return "harian"
    return "tahunan"


def _ensure_kategori_col(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan selalu ada kolom 'Kategori' di df (fallback ke Agri/Harga atau '-')."""
    if "Kategori" in df.columns:
        return df
    df2 = df.copy()
    if "Kategori_Agri" in df2.columns:
        df2["Kategori"] = df2["Kategori_Agri"]
    elif "Kategori_Harga" in df2.columns:
        df2["Kategori"] = df2["Kategori_Harga"]
    else:
        df2["Kategori"] = "-"
    return df2


def render_pca_scatter_visual(X_scaled: np.ndarray, df_out: pd.DataFrame, labels: np.ndarray):
    if X_scaled is None or len(X_scaled) == 0 or df_out is None or df_out.empty:
        st.info("Belum ada data untuk visualisasi PCA.")
        return

    try:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X_scaled)
    except Exception as e:
        st.warning(f"Gagal menghitung PCA untuk visualisasi: {e}")
        return

    def _extract_kategori(df: pd.DataFrame) -> pd.Series:
        for cand in ["Kategori", "Kategori_Agri", "Kategori_Harga"]:
            if cand in df.columns:
                obj = df.loc[:, cand]
                if isinstance(obj, pd.DataFrame):
                    ser = obj.iloc[:, 0]
                else:
                    ser = obj
                return ser.astype(str)
        return pd.Series(["-"] * len(df), index=df.index, dtype="object")

    kat_ser = _extract_kategori(df_out)
    kat = np.asarray(kat_ser.values).ravel()

    n = X2.shape[0]
    if len(kat) != n:
        if len(kat) > n:
            kat = kat[:n]
        else:
            pad = np.array(["-"] * (n - len(kat)), dtype=object)
            kat = np.concatenate([kat, pad], axis=0)

    df_plot = pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1], "Kategori": kat})

    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    colors = {
        "Sangat Rendah": "#d73027",
        "Rendah": "#fc8d59",
        "Cukup Rendah": "#fee090",
        "Sedang": "#ffffbf",
        "Cukup Tinggi": "#e0f3f8",
        "Tinggi": "#91bfdb",
        "Sangat Tinggi": "#4575b4",
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    present = [c for c in cat_order if c in df_plot["Kategori"].unique()]
    if not present:
        present = list(df_plot["Kategori"].dropna().unique())

    for cat in present:
        sub = df_plot[df_plot["Kategori"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=26, alpha=0.85,
            color=colors.get(cat, "#888"),
            label=cat, edgecolor="white", linewidth=0.4
        )

    ax.set_title("Visualisasi PCA 2D (Warna = Kategori Cluster)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax.grid(True, linestyle="--", alpha=.35)
    ax.legend(title="Kategori", ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    st.pyplot(fig)


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

    luas  = [c for c in df.columns if re.search(r"^luas|_luas|luas_", c, flags=re.I)]
    prod  = [c for c in df.columns if re.search(r"^produksi|_produksi|produksi_", c, flags=re.I)]
    prdx  = [c for c in df.columns if re.search(r"^produkt|_produkt|produkt_", c, flags=re.I)]
    parts=[]
    for fam in [luas, prod, prdx]:
        if fam:
            s = pd.to_numeric(df[fam].mean(axis=1, skipna=True), errors="coerce")
            mu, sd = s.mean(), s.std(ddof=0) or 1.0
            parts.append((s-mu)/sd)
    if not parts: raise ValueError("Tidak ada kolom luas/produksi/produkt untuk skor.")
    return pd.concat(parts, axis=1).mean(axis=1)


# =================== Prepare Numeric ===================
def _prepare_numeric(df: pd.DataFrame):
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


# ===== Deteksi kolom HARGA (regex fleksibel) =====
def _detect_price_cols(df: pd.DataFrame):
    pat_merah = re.compile(r"(harga|price).*?(cabe|cabai)?\s*merah.*?(19|20)\d{2}", re.I)
    pat_hijau = re.compile(r"(harga|price).*?(cabe|cabai)?\s*hijau.*?(19|20)\d{2}", re.I)
    merah = [c for c in df.columns if pat_merah.search(str(c))]
    hijau = [c for c in df.columns if pat_hijau.search(str(c))]
    def _yearkey(c):
        m = re.search(r"(19|20)\d{2}", str(c))
        return int(m.group()) if m else -1
    return sorted(merah, key=_yearkey), sorted(hijau, key=_yearkey)


def _prepare_numeric_by_family(df: pd.DataFrame, family: str):
    """
    family: 'agri' untuk luas/produksi/produkt, 'price' untuk harga merah/hijau.
    """
    if family == "agri":
        cols_year  = [c for c in df.columns if re.search(r"(luas|produksi|produkt).*?(19|20)\d{2}", str(c), flags=re.I)]
        cols_plain = [c for c in df.columns if re.search(r"^(luas|produksi|produkt)", str(c), flags=re.I) and not re.search(r"(19|20)\d{2}", str(c), flags=re.I)]
        feature_cols = cols_year if cols_year else cols_plain
        if not feature_cols:
            raise ValueError("Tidak ditemukan kolom fitur pertanian.")
    else:
        merah, hijau = _detect_price_cols(df)
        nonyear = [c for c in df.columns if re.search(r"^harga.*(merah|hijau)$", str(c), flags=re.I)]
        feature_cols = merah + hijau + nonyear
        if not feature_cols:
            raise ValueError("Tidak ditemukan kolom fitur harga (merah/hijau).")

    X = df[feature_cols].copy()
    imp = SimpleImputer(strategy="mean")
    X_imp = imp.fit_transform(pd.to_numeric(X.stack(), errors="coerce").unstack())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, feature_cols, df.index


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


# ======================== Clustering ========================
def _fit_model(X, method: str, k: int):
    if method == "K-Means":
        return (MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=5).fit(X)
                if X.shape[0]>10000 else KMeans(n_clusters=k, random_state=42, n_init=10).fit(X))
    if X.shape[0] > 2000: raise RuntimeError("Data terlalu besar untuk Hierarchical (>2000).")
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)


def run_clustering_family(df: pd.DataFrame, method: str, k: int, family: str):
    """family: 'agri' atau 'price'."""
    if not isinstance(k,(int,np.integer)): 
        k = int(np.squeeze(k))
    if k < 2 or k > 7:
        raise ValueError("Jumlah cluster (k) harus 2–7.")

    if family == "agri":
        X_scaled, used_feature_cols, _ = _prepare_numeric_by_family(df, "agri")
        weights = FEATURE_WEIGHTS
    else:
        X_scaled, used_feature_cols, _ = _prepare_numeric_by_family(df, "price")
        weights = PRICE_FEATURE_WEIGHTS

    pca_info = {"used": False, "n_components": 0, "explained": None}
    X_clust = X_scaled
    method_for_fit = method
    if method == "PCA + K-Means":
        max_comp = min(6, X_scaled.shape[1]); max_comp = max(2, max_comp)
        pca_probe = PCA(n_components=max_comp).fit(X_scaled)
        csum = np.cumsum(pca_probe.explained_variance_ratio_)
        n_opt = int(np.clip(np.searchsorted(csum, 0.90) + 1, 2, max_comp))
        pca = PCA(n_components=n_opt, random_state=42)
        X_clust = pca.fit_transform(X_scaled)
        pca_info = {"used": True, "n_components": n_opt, "explained": float(np.sum(pca.explained_variance_ratio_))}
        method_for_fit = "K-Means"

    model = _fit_model(X_clust, "K-Means" if method_for_fit.startswith("K-Means") else method_for_fit, k)
    labels = np.asarray(getattr(model, "labels_", None) if hasattr(model, "labels_") else model.fit_predict(X_clust), dtype=int)

    try: sil = round(silhouette_score(X_clust, labels, sample_size=min(3000, X_clust.shape[0]), random_state=42),4)
    except Exception: sil = float("nan")
    try: dbi = round(davies_bouldin_score(X_clust, labels),4)
    except Exception: dbi = float("nan")
    results_df = pd.DataFrame([{
        "Keluarga": ("Pertanian" if family=="agri" else "Harga"),
        "Metode": method,
        "Jumlah Cluster": len(np.unique(labels)),
        "Silhouette": sil,
        "Davies-Bouldin": dbi
    }])

    ranked, cluster_scores = ranking_weighted_zscore(df, labels, used_feature_cols, weights)

    LABEL_SETS = {
        2:["Rendah","Tinggi"], 3:["Rendah","Sedang","Tinggi"],
        4:["Sangat Rendah","Rendah","Tinggi","Sangat Tinggi"],
        5:["Sangat Rendah","Rendah","Sedang","Tinggi","Sangat Tinggi"],
        6:["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi"],
        7:["Sangat Rendah","Rendah","Cukup Rendah","Sedang","Cukup Tinggi","Tinggi","Sangat Tinggi"],
    }
    chosen_labels = LABEL_SETS[len(np.unique(labels))]
    cluster_label_map = {int(c): chosen_labels[i] for i, c in enumerate(ranked)}

    col_cluster  = "Cluster_Agri" if family=="agri" else "Cluster_Harga"
    col_kategori = "Kategori_Agri" if family=="agri" else "Kategori_Harga"

    out = pd.DataFrame(index=df.index)
    out[col_cluster]  = labels
    out[col_kategori] = pd.Series(labels, index=df.index).map(cluster_label_map)

    diag = {"used_cols": used_feature_cols, "cluster_scores": cluster_scores.to_dict(), "pca_info": pca_info}
    return results_df, out, X_clust, labels, cluster_label_map, diag


def run_both_clusterings(df: pd.DataFrame, method: str, k: int):
    """
    Jalankan 2 clustering (pertanian & harga) bila kolom memadai.
    Mengembalikan df_out gabungan + dua ringkasan hasil.
    """
    # Pertanian (wajib)
    res_agri, out_agri, X_agri, lab_agri, map_agri, diag_agri = run_clustering_family(df, method, k, "agri")

    # Harga (opsional: kalau kolom ada)
    try:
        res_price, out_price, X_price, lab_price, map_price, diag_price = run_clustering_family(df, method, k, "price")
    except Exception:
        res_price = pd.DataFrame([{"Keluarga": "Harga", "Metode": method, "Jumlah Cluster": np.nan, "Silhouette": np.nan, "Davies-Bouldin": np.nan}])
        out_price = pd.DataFrame(index=df.index, data={"Cluster_Harga": np.nan, "Kategori_Harga": np.nan})
        X_price, lab_price, map_price, diag_price = None, None, {}, {"used_cols": [], "pca_info": {"used": False}}

    df_out = df.copy()
    df_out = df_out.join(out_agri).join(out_price)

    # kompat: set Kategori/Cluster generik = versi Agri (kalau ada)
    if "Cluster_Agri" in df_out.columns:
        df_out["Cluster"] = df_out["Cluster_Agri"]
    if "Kategori_Agri" in df_out.columns:
        df_out = _ensure_kategori_col(df_out)  # pastikan 'Kategori' ada
        df_out["Kategori"] = df_out["Kategori_Agri"]

    # normalisasi urutan kategori (pakai fitur pertanian)
    df_out = normalize_cluster_order(df_out, fitur_list=diag_agri["used_cols"])

    # fallback antipeluru
    if "Kategori_Agri" not in df_out.columns and "Kategori" in df_out.columns:
        df_out["Kategori_Agri"] = df_out["Kategori"]
    if "Cluster_Agri" not in df_out.columns and "Cluster" in df_out.columns:
        df_out["Cluster_Agri"] = df_out["Cluster"]

    st.session_state.update({
        # umum
        "df_out": df_out,
        "clustering_done": True,
        "method_used": method,
        "k_used": k,
        # pertanian
        "results_df_agri": res_agri,
        "X_scaled_agri": X_agri,
        "labels_agri": lab_agri,
        "cluster_label_map_agri": map_agri,
        "diag_used_cols_agri": diag_agri["used_cols"],
        "pca_info_agri": diag_agri["pca_info"],
        # harga
        "results_df_harga": res_price,
        "X_scaled_harga": X_price,
        "labels_harga": lab_price,
        "cluster_label_map_harga": map_price,
        "diag_used_cols_harga": diag_price["used_cols"],
        "pca_info_harga": diag_price.get("pca_info", {"used": False}),
    })

    results_both = pd.concat([res_agri, res_price], ignore_index=True)
    st.session_state["results_df"] = results_both
    return df_out, results_both



# ===================== Evaluasi & Visualisasi =====================
def render_cluster_performance(X_scaled, method_sel, k_sel=None):
    st.markdown("## 📊 Evaluasi Performa Clustering")
    st.caption("Silhouette lebih tinggi lebih baik; Davies–Bouldin lebih rendah lebih baik.")

    if X_scaled is None or len(X_scaled) == 0:
        st.info("Tidak ada data untuk evaluasi.")
        return

    k_sel = int(k_sel or st.session_state.get("k_used", 2))
    k_values = list(range(2, 8))
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

    sil_arr = np.asarray(silhouette_scores, dtype=float)
    best_k = None if np.isnan(sil_arr).all() else int(k_values[int(np.nanargmax(sil_arr))])

    idx = k_sel - 2
    sil_k = silhouette_scores[idx] if 0 <= idx < len(silhouette_scores) else np.nan
    dbi_k = dbi_scores[idx] if 0 <= idx < len(dbi_scores) else np.nan

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Metode", method_sel.replace("+", "").replace("-", ""))
    with c2: st.metric("Silhouette Score", f"{sil_k:.4f}" if not np.isnan(sil_k) else "—")
    with c3: st.metric("Waktu Proses", f"{elapsed:.4f} detik")

    c4, c5, c6 = st.columns(3)
    with c4: st.metric("Jumlah Cluster", f"{k_sel}")
    with c5: st.metric("Davies–Bouldin Index", f"{dbi_k:.4f}" if not np.isnan(dbi_k) else "—")
    with c6:
        pinfo = st.session_state.get("pca_info", {"used": False})
        st.metric("PCA", "Tidak dipakai" if not pinfo.get("used") else f"{pinfo.get('n_components')} komponen")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(k_values, silhouette_scores, "o-", lw=2)
    ax[1].plot(k_values, dbi_scores, "o-", lw=2)
    ax[0].axvline(k_sel, color="red", ls="--", lw=2)
    ax[1].axvline(k_sel, color="red", ls="--", lw=2)

    def _pad_ylim(ax_, values, top_pad=0.18, bottom_pad=0.06):
        vals = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            span = max(hi - lo, 0.05)
            ax_.set_ylim(lo - span * bottom_pad, hi + span * top_pad)

    _pad_ylim(ax[0], silhouette_scores)
    _pad_ylim(ax[1], dbi_scores)

    for k, s in zip(k_values, silhouette_scores):
        if not np.isnan(s):
            ax[0].annotate(f"{s:.3f}", (k, s), xytext=(0, 8), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9)
    for k, d in zip(k_values, dbi_scores):
        if not np.isnan(d):
            ax[1].annotate(f"{d:.3f}", (k, d), xytext=(0, 8), textcoords="offset points",
                           ha="center", va="bottom", fontsize=9)

    ax[0].set_title("Silhouette Score");     ax[0].set_xlabel("Jumlah Cluster"); ax[0].set_ylabel("Score"); ax[0].grid(alpha=.3)
    ax[1].set_title("Davies–Bouldin Index"); ax[1].set_xlabel("Jumlah Cluster"); ax[1].set_ylabel("Index"); ax[1].grid(alpha=.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        f"Menampilkan hasil untuk **k = {k_sel}** (garis merah putus–putus). "
        "Silhouette lebih tinggi → cluster makin terpisah; Davies–Bouldin lebih rendah → cluster makin kompak."
    )
    if best_k is not None:
        st.caption(f"K optimum indikatif (berdasar Silhouette): **k = {best_k}**.")


# ----------------- Reshape Pertanian (patched aman Kategori) -----------------
def reshape_long_format(df_out: pd.DataFrame):
    df_out = _ensure_kategori_col(df_out)

    luas = [c for c in df_out.columns if re.search(r"luas.*?(19|20)\d{2}", c, flags=re.I)]
    prod = [c for c in df_out.columns if re.search(r"produksi.*?(19|20)\d{2}", c, flags=re.I)]
    prdx = [c for c in df_out.columns if re.search(r"produkt.*?(19|20)\d{2}", c, flags=re.I)]

    frames=[]
    for subset, fitur in [(luas,"luas_areal"),(prod,"produksi"),(prdx,"produktivitas")]:
        if not subset: continue
        id_vars = [col for col in ["Kategori"] if col in df_out.columns]
        t = df_out.melt(id_vars=id_vars, value_vars=subset, var_name="Tahun", value_name="Nilai")
        t["Fitur"]=fitur
        t["Tahun"]=t["Tahun"].astype(str).str.extract(r"(\d{4})")
        frames.append(t)

    if not frames:
        return pd.DataFrame(columns=["Kategori","Tahun","Fitur","Nilai"])
    df_long = pd.concat(frames, ignore_index=True)
    df_long["Tahun"]=pd.to_numeric(df_long["Tahun"], errors="coerce")
    if "Kategori" not in df_long.columns:
        df_long["Kategori"] = "-"
    return df_long.dropna(subset=["Nilai"])


def render_boxplot(df_out: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data numerik untuk boxplot.")
        return

    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    df_long["Kategori"] = pd.Categorical(df_long["Kategori"], categories=cat_order, ordered=True)
    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]

    color_map = {"luas_areal": "#F5B7B1", "produksi": "#82E0AA", "produktivitas": "#85C1E9"}
    fitur_unik = ["luas_areal", "produksi", "produktivitas"]
    tahun_list = sorted(df_long["tahun"].dropna().unique())

    st.markdown("## 📈 Analisis Distribusi Boxplot Fitur per Tahun")
    tahun_pilihan = st.selectbox("Pilih Tahun:", ["Seluruh Tahun"] + [str(t) for t in tahun_list],
                                 index=0, key="boxplot_tahun_pertanian")
    fitur_pilihan = st.multiselect("Pilih Fitur:", options=fitur_unik, default=fitur_unik, key="fitur_boxplot_tahun")
    use_log = st.toggle("Gunakan skala logaritmik", value=False, key="boxplot_use_log")

    if not fitur_pilihan:
        st.warning("Pilih minimal satu fitur.")
        return

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

        for j in range(len(tahun_list), len(axes)):
            axes[j].axis("off")

        if ax_last is None:
            ax_last = axes[0]
        handles = [plt.Line2D([0], [0], lw=8, label=f.replace("_", " ").title(),
                              color=color_map.get(f, "#999")) for f in fitur_pilihan]
        ax_last.legend(handles=handles, title="Fitur", loc="upper right",
                       frameon=True, fancybox=True, borderpad=0.6, labelspacing=0.5)

        fig.tight_layout()
        st.pyplot(fig)
        return

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

    handles = [plt.Line2D([0], [0], lw=8, label=f.replace("_", " ").title(),
                          color=color_map.get(f, "#999")) for f in fitur_pilihan]
    ax.legend(handles=handles, title="Fitur", loc="upper right",
              frameon=True, fancybox=True, borderpad=0.6, labelspacing=0.5)

    plt.tight_layout()
    st.pyplot(fig)


def render_boxplot_combined(df_out: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    df_long = reshape_long_format(df_out)
    if df_long.empty:
        st.warning("Tidak ada data numerik.")
        return

    df_long.columns = [c.lower().replace(" ", "_") for c in df_long.columns]
    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    df_long["kategori"] = pd.Categorical(df_long["kategori"], categories=cat_order, ordered=True)

    fitur_unik = ["luas_areal", "produksi", "produktivitas"]
    st.markdown("## 📈 Analisis Distribusi Boxplot Fitur Seluruh Tahun")
    fitur_pilihan = st.multiselect(
        "Pilih Fitur:", options=fitur_unik, default=fitur_unik, key="fitur_boxplot_combined"
    )
    use_log = st.toggle("Gunakan skala logaritmik", value=False, key="log_boxplot_combined")

    if not fitur_pilihan:
        st.warning("Pilih minimal satu fitur.")
        return

    df_long["nilai_standar"] = np.nan
    for fitur in fitur_pilihan:
        m = (df_long["fitur"] == fitur)
        if m.sum() > 0:
            vals = df_long.loc[m, "nilai"].values.reshape(-1, 1)
            df_long.loc[m, "nilai_standar"] = StandardScaler().fit_transform(vals).ravel()

    color_map = {"luas_areal": "#F5B7B1", "produksi": "#82E0AA", "produktivitas": "#85C1E9"}

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
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

    handles = [
        plt.Line2D([0], [0], lw=8, label=f.replace("_", " ").title(),
                   color=color_map.get(f, "#999"))
    for f in fitur_pilihan]
    ax.legend(handles=handles, title="Fitur", loc="upper right",
              frameon=True, fancybox=True, borderpad=0.6, labelspacing=0.5)

    plt.tight_layout()
    st.pyplot(fig)


# -------------------- Tren & Top Lokasi (Pertanian) --------------------
def render_tren_hasil_panen(df_out: pd.DataFrame):
    st.markdown("## 📈 Tren Hasil Panen")
    if df_out is None or df_out.empty:
        st.warning("Dataset kosong."); return

    lokasi_col = next((c for c in df_out.columns
                       if c.lower() in ["provinsi", "kabupaten", "kabupaten/kota", "lokasi"]), None)
    if lokasi_col is None:
        st.warning("Kolom lokasi tidak ditemukan."); return
    if "Kategori" not in df_out.columns:
        df_out = df_out.copy(); df_out["Kategori"] = "-"

    def _year_cols(prefix: str) -> list[str]:
        patt = re.compile(rf"^{prefix}.*?(19|20)\d{{2}}", flags=re.I)
        cols = [c for c in df_out.columns if patt.search(str(c))]
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
        topn = st.slider("Top Lokasi:", 1, 20, 5, key="tren_topn_nat")
    with c4:
        urutkan = st.radio("Urutan:", ["Terbesar", "Terkecil"], horizontal=True, key="tren_urut_nat")

    c5, c6, c7 = st.columns([1.6, 1.3, 1.8])
    with c5:
        smooth = st.checkbox("Haluskan garis (median 3)", value=True, key="tren_smooth_nat")
    with c6:
        show_markers = st.checkbox("Tampilkan marker", value=True, key="tren_marker_nat")
    with c7:
        direct_label = st.checkbox("Label langsung di kanan (tanpa legend)", value=False, key="tren_labelkanan_nat")

    q = st.text_input("🔎 Bandingkan Lokasi (pisahkan dengan koma):", value="",
                      help="Contoh: polewali mandar, kolaka utara, manokwari, buru", key="tren_lokasi_q")
    include_topn = st.checkbox("Tambahkan Top-N lainnya", value=False, key="tren_inc_topn")

    ascending_flag = (urutkan == "Terkecil")
    colors = plt.cm.tab20.colors
    ycols = _year_cols(fitur_sel)
    if not ycols:
        st.warning(f"Tidak ada kolom '{fitur_sel}' yang bertahun."); return

    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols: df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    d = df_num.melt(id_vars=[lokasi_col, "Kategori"], value_vars=ycols,
                    var_name="Kolom", value_name="Nilai")
    d["Tahun"] = d["Kolom"].str.extract(r"((?:19|20)\d{2})").astype(float)
    d.drop(columns=["Kolom"], inplace=True)
    d = d.dropna(subset=["Tahun", "Nilai"])
    d = d[d["Tahun"].between(2015, 2024)]
    d["Tanggal"] = pd.to_datetime(d["Tahun"].astype(int).astype(str) + "-01-01")

    mean_by_loc = d.groupby(lokasi_col)["Nilai"].mean().sort_values(ascending=ascending_flag)
    top_lokasi = mean_by_loc.index.tolist()[:topn]

    names = df_out[lokasi_col].astype(str)
    matched_locs: list[str] = []
    if q.strip():
        tokens = [t.strip().lower() for t in q.split(",") if t.strip()]
        for tok in tokens:
            if not tok: continue
            mask = names.str.lower().str.contains(tok)
            matched_locs.extend(names[mask].unique().tolist())
        matched_locs = list(dict.fromkeys(matched_locs))
        st.caption("Lokasi cocok: " + (", ".join(matched_locs) if matched_locs else "—"))

    plot_locs = matched_locs if (matched_locs and not include_topn) else list(dict.fromkeys(matched_locs + top_lokasi)) if matched_locs else top_lokasi
    d_plot = d[d[lokasi_col].isin(plot_locs)]
    years = sorted(d_plot["Tahun"].dropna().unique())
    unit_txt = fitur_unit.get(fitur_sel, "")
    ylabel = f"{fitur_label[fitur_sel]} {unit_txt}".strip()

    if mode_view == "Garis (Top-N)":
        fig, ax = plt.subplots(figsize=(10.8, 6.0))
        ordered = plot_locs[:] if (matched_locs and not include_topn) else [loc for loc in mean_by_loc.index if loc in plot_locs]
        handles, labels_leg = [], []
        for i, lok in enumerate(ordered):
            sub = d_plot[d_plot[lokasi_col] == lok].sort_values("Tanggal")
            if sub.empty: continue
            y = sub["Nilai"].to_numpy(dtype=float)
            if smooth: y = _smooth_med3(y)
            (line,) = ax.plot(sub["Tanggal"], y, linewidth=2.2,
                              marker="o" if show_markers else None, markersize=4,
                              alpha=0.95, color=colors[i % len(colors)])
            cat = sub["Kategori"].dropna().mode().iat[0] if not sub["Kategori"].dropna().empty else "-"
            label = f"{lok} ({cat})"
            handles.append(line); labels_leg.append(label)
            if direct_label:
                ax.text(sub["Tanggal"].iloc[-1] + pd.Timedelta(days=25), y[-1], s=lok,
                        va="center", fontsize=9, color=colors[i % len(colors)])
        ax.set_title(f"TREN PERBANDINGAN: {', '.join(ordered).upper() if matched_locs else f'TOP-{topn}'}")
        ax.set_xlabel("Tahun"); ax.set_ylabel(ylabel); ax.grid(True, linestyle="--", alpha=.35)
        if years:
            xticks = pd.to_datetime([f"{int(y)}-01-01" for y in years])
            ax.set_xticks(xticks); ax.set_xticklabels([str(int(y)) for y in years])
        ylo, yhi = _fit_ylim_full(d_plot["Nilai"]); 
        if ylo is not None: ax.set_ylim(ylo, yhi)
        if direct_label:
            xmin, xmax = ax.get_xlim(); ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.06)
        if not direct_label and handles:
            ax.legend(handles=handles, labels=labels_leg, title="Lokasi", fontsize=8,
                      ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            plt.tight_layout(rect=[0, 0, 0.78, 1])
        else:
            plt.tight_layout()
        st.pyplot(fig)

    elif mode_view == "Facet per Kategori":
        base = d_plot if matched_locs else d
        cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
        cats = [c for c in cat_order if c in base["Kategori"].dropna().unique().tolist()] or sorted(base["Kategori"].dropna().unique())
        n = max(len(cats), 1); ncols = 2 if n >= 2 else 1; nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.2 * nrows), sharex=True, squeeze=False)
        axes = axes.flatten()
        for idx, cat in enumerate(cats):
            ax = axes[idx]; sub_cat = base[base["Kategori"] == cat].copy()
            if sub_cat.empty: ax.set_axis_off(); continue
            if matched_locs:
                locs_cat = [loc for loc in plot_locs if loc in sub_cat[lokasi_col].unique()]
            else:
                mean_by_loc_cat = sub_cat.groupby(lokasi_col)["Nilai"].mean().sort_values(ascending=ascending_flag)
                locs_cat = mean_by_loc_cat.head(min(topn, 6)).index.tolist()
            for i, lok in enumerate(locs_cat):
                sub = sub_cat[sub_cat[lokasi_col] == lok].sort_values("Tanggal")
                y = sub["Nilai"].to_numpy(dtype=float)
                if smooth: y = _smooth_med3(y)
                ax.plot(sub["Tanggal"], y, linewidth=2.0, marker="o" if show_markers else None, markersize=3.5,
                        color=colors[i % len(colors)], alpha=.95, label=lok)
            ylo, yhi = _fit_ylim_full(sub_cat["Nilai"])
            if ylo is not None: ax.set_ylim(ylo, yhi)
            ax.set_title(cat); ax.grid(True, linestyle="--", alpha=.35)
            if years:
                xticks = pd.to_datetime([f"{int(y)}-01-01" for y in years])
                ax.set_xticks(xticks); ax.set_xticklabels([str(int(y)) for y in years])
            ax.legend(fontsize=8, frameon=False)
        for j in range(len(cats), len(axes)): axes[j].set_axis_off()
        fig.suptitle(f"{fitur_label[fitur_sel]} — Facet per Kategori", y=0.995, fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98]); st.pyplot(fig)

    else:
        pvt_all = d.groupby([lokasi_col, "Tahun"])["Nilai"].mean().unstack("Tahun")
        rows = [loc for loc in pvt_all.mean(axis=1).sort_values(ascending=ascending_flag).head(topn).index.tolist()]
        pvt = pvt_all.loc[rows]
        fig, ax = plt.subplots(figsize=(1.2 * len(pvt.columns) + 4, 0.45 * len(pvt.index) + 2.6))
        im = ax.imshow(pvt.values, aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax); cbar.set_label(fitur_label[fitur_sel])
        ax.set_xticks(range(len(pvt.columns))); ax.set_xticklabels([str(int(c)) for c in pvt.columns], rotation=0)
        ax.set_yticks(range(len(pvt.index))); ax.set_yticklabels(pvt.index)
        ax.set_title(f"{fitur_label[fitur_sel]} — Heatmap (Top-{topn})")
        mean_val = np.nanmean(pvt.values)
        for (i, j), val in np.ndenumerate(pvt.values):
            if not np.isnan(val):
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=8, color=("white" if val > mean_val else "black"))
        plt.tight_layout(); st.pyplot(fig)


def render_top_lokasi(df_out: pd.DataFrame):
    import textwrap
    lokasi_col = next((c for c in df_out.columns
                       if c.lower() in ["provinsi","kabupaten","kabupaten/kota","lokasi"]), None)
    if lokasi_col is None:
        st.warning("Kolom lokasi tidak ditemukan."); return

    def _year_cols(prefix: str) -> list[str]:
        patt = re.compile(rf"^{prefix}.*?(19|20)\d{{2}}", flags=re.I)
        cols = [c for c in df_out.columns if patt.search(str(c))]
        def _y(c):
            m = re.search(r"(19|20)\d{2}", str(c))
            return int(m.group()) if m else -1
        cols = sorted(cols, key=_y)
        cols = [c for c in cols if 2015 <= _y(c) <= 2024] or cols
        return cols

    def _wrap(s: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(s), width=width)) if width and width>0 else s

    fitur_opsi  = ["luas_areal","produksi","produktivitas"]
    fitur_label = {"luas_areal":"Luas Areal","produksi":"Produksi","produktivitas":"Produktivitas"}
    fitur_unit  = {"luas_areal":"(ha)","produksi":"(ton)","produktivitas":"(ton/ha)"}

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        fitur_sel = st.selectbox("Pilih Fitur:", fitur_opsi, format_func=lambda k: fitur_label[k], key="toploc_fitur")
    with c2:
        mode = st.radio("Mode:", ["Top-N", "Pencarian"], horizontal=True, key="toploc_mode")
    with c3:
        urutkan = st.radio("Urutan:", ["Terbesar","Terkecil"], horizontal=True, key="toploc_order")

    if mode == "Top-N":
        topn = st.slider("Jumlah Lokasi:", 3, 30, 10, key="toploc_topn")
    else:
        q = st.text_input("🔎 Bandingkan Lokasi (pisahkan dengan koma):", "",
                          help="Contoh: kolaka utara, polewali mandar, luwu, sigi, mamuju",
                          key="toploc_find_q")
        only_matches = st.checkbox("Tampilkan hanya yang dicari", value=True, key="toploc_only")
        wrap_len = st.slider("Panjang label (wrap):", 10, 40, 18, key="toploc_wrap")

    ycols = _year_cols(fitur_sel)
    if not ycols:
        st.warning(f"Tidak ada kolom bertahun untuk '{fitur_sel}'."); return

    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols: df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    df_num["Rata_Rata"] = df_num[ycols].mean(axis=1, skipna=True)

    agg = (df_num.groupby([lokasi_col, "Kategori"], dropna=False)["Rata_Rata"]
           .mean().reset_index())

    ascending = (urutkan == "Terkecil")

    if mode == "Pencarian":
        matched = []
        tokens = [t.strip().lower() for t in q.split(",") if t.strip()]
        if tokens:
            names = agg[lokasi_col].astype(str).str.lower()
            def _match_mask(token):
                t = re.sub(r"\b(kab(\.|upaten)?|kota)\b", "", token).strip()
                if not t: return np.zeros(len(names), dtype=bool)
                return names.str.contains(t, regex=False)
            m = np.column_stack([_match_mask(t) for t in tokens]) if tokens else np.zeros((len(names),0), bool)
            mask = m.any(axis=1) if m.size else np.zeros(len(names), dtype=bool)
            matched = agg.loc[mask, lokasi_col].astype(str).tolist()
        st.caption("Lokasi cocok: " + (", ".join(matched) if matched else "— tidak ada —"))
        if st.session_state.get("toploc_only", True):
            agg = agg[agg[lokasi_col].astype(str).isin(matched)]
            if agg.empty:
                st.info("Tidak ada data untuk lokasi yang Anda ketik."); return
    else:
        agg = agg.sort_values("Rata_Rata", ascending=ascending).head(st.session_state.get("toploc_topn", 10))

    cat_to_color = {"Sangat Rendah":"#d73027","Rendah":"#fc8d59","Cukup Rendah":"#fee090",
                    "Sedang":"#ffffbf","Cukup Tinggi":"#e0f3f8","Tinggi":"#91bfdb","Sangat Tinggi":"#4575b4"}
    N = len(agg); fig_w = max(8, min(22, 0.35 * N + 6))
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    bar_colors = [cat_to_color.get(k, "#999999") for k in agg["Kategori"].astype(str)]
    bars = ax.bar(range(N), agg["Rata_Rata"].values, color=bar_colors, edgecolor="#333", linewidth=.5)
    for i, b in enumerate(bars):
        val = agg["Rata_Rata"].iat[i]
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(1e-9, abs(val)*0.01),
                f"{val:,.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1c2833")
    ylab = f"Rata-Rata {fitur_unit.get(fitur_sel,'')}".strip()
    ax.set_title(f"TOP {N} LOKASI — {fitur_label[fitur_sel].upper()} (2015–2024)")
    ax.set_ylabel(ylab); ax.set_xlabel("Lokasi")
    ax.grid(axis="y", linestyle="--", alpha=.5); ax.set_axisbelow(True)
    labels = [f"{row[lokasi_col]} ({row['Kategori']})" for _, row in agg.iterrows()]
    rot = 45 if st.session_state.get("toploc_mode") == "Top-N" else 0
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=9)
    plt.tight_layout(); st.pyplot(fig)


# ========================= Patch Korelasi Pertanian-Harga =========================
def _find_lokasi_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"provinsi", "lokasi", "kabupaten", "kabupaten/kota"}:
            return str(c)
    return None


def reshape_prices_long(df_out: pd.DataFrame):
    df_out = _ensure_kategori_col(df_out)
    lokasi_col = _find_lokasi_col(df_out)
    if lokasi_col is None:
        return pd.DataFrame(columns=["Lokasi", "Tahun", "Harga_Merah", "Harga_Hijau", "Kategori"])

    merah, hijau = _detect_price_cols(df_out)
    frames = []

    base_id = [lokasi_col]
    if "Kategori" in df_out.columns:
        base_id.append("Kategori")

    if merah:
        t = df_out.melt(id_vars=base_id, value_vars=merah,
                        var_name="Kolom", value_name="Harga_Merah")
        t["Tahun"] = t["Kolom"].astype(str).str.extract(r"((?:19|20)\d{2})").astype(float)
        t = t.drop(columns=["Kolom"]).dropna(subset=["Tahun"]).rename(columns={lokasi_col: "Lokasi"})
        frames.append(t)

    if hijau:
        t2 = df_out.melt(id_vars=base_id, value_vars=hijau,
                         var_name="Kolom", value_name="Harga_Hijau")
        t2["Tahun"] = t2["Kolom"].astype(str).str.extract(r"((?:19|20)\d{2})").astype(float)
        t2 = t2.drop(columns=["Kolom"]).dropna(subset=["Tahun"]).rename(columns={lokasi_col: "Lokasi"})
        frames.append(t2)

    if not frames:
        return pd.DataFrame(columns=["Lokasi", "Tahun", "Harga_Merah", "Harga_Hijau", "Kategori"])

    out = frames[0]
    for f in frames[1:]:
        keys = [c for c in ["Lokasi", "Tahun", "Kategori"] if c in out.columns and c in f.columns]
        out = pd.merge(out, f, on=keys, how="outer")
    if "Kategori" not in out.columns:
        out["Kategori"] = "-"
    return out


def _build_corr_dataset(df_out: pd.DataFrame) -> pd.DataFrame:
    df_out = _ensure_kategori_col(df_out)
    lokasi_col = _find_lokasi_col(df_out)
    if lokasi_col is None:
        return pd.DataFrame(columns=["Lokasi","Tahun","luas_areal","produksi","produktivitas","Harga_Merah","Harga_Hijau","Kategori"])

    patt = re.compile(r"(luas|produksi|produkt).*?(19|20)\d{2}", flags=re.I)
    ag_cols = [c for c in df_out.columns if patt.search(str(c))]
    if not ag_cols:
        return pd.DataFrame(columns=["Lokasi","Tahun","luas_areal","produksi","produktivitas","Harga_Merah","Harga_Hijau","Kategori"])

    ag_src = df_out[[lokasi_col, "Kategori"] + ag_cols].copy()
    parts = []
    for prefix, nama in [(r"luas", "luas_areal"), (r"produksi", "produksi"), (r"produkt", "produktivitas")]:
        cols = [c for c in ag_cols if re.search(rf"{prefix}.*?(19|20)\d{{2}}", str(c), flags=re.I)]
        if not cols: continue
        t = ag_src.melt(id_vars=[lokasi_col, "Kategori"], value_vars=cols,
                        var_name="Kolom", value_name="Nilai")
        t["Tahun"] = t["Kolom"].astype(str).str.extract(r"((?:19|20)\d{2})").astype(float)
        t = t.drop(columns=["Kolom"]).rename(columns={lokasi_col:"Lokasi"})
        t["Fitur"] = nama
        parts.append(t)

    if not parts:
        return pd.DataFrame(columns=["Lokasi","Tahun","luas_areal","produksi","produktivitas","Harga_Merah","Harga_Hijau","Kategori"])

    ag_all = pd.concat(parts, ignore_index=True)
    idx_cols = ["Lokasi", "Tahun"]
    if "Kategori" in ag_all.columns:
        idx_cols.insert(1, "Kategori")
    ag_pvt = ag_all.pivot_table(index=idx_cols, columns="Fitur", values="Nilai", aggfunc="mean").reset_index()

    pr_all = reshape_prices_long(df_out)
    if pr_all.empty:
        pr_all = pd.DataFrame(columns=["Lokasi","Tahun","Harga_Merah","Harga_Hijau","Kategori"])

    keys = [c for c in ["Lokasi","Tahun","Kategori"] if c in ag_pvt.columns and c in pr_all.columns]
    if not keys:
        keys = [c for c in ["Lokasi","Tahun"] if c in ag_pvt.columns and c in pr_all.columns]

    merged = pd.merge(ag_pvt, pr_all, on=keys, how="inner")

    for c in ["luas_areal","produksi","produktivitas","Harga_Merah","Harga_Hijau"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if "Kategori" not in merged.columns:
        merged["Kategori"] = "-"
    merged = merged.dropna(subset=["luas_areal","produksi","produktivitas"], how="any")
    return merged


def render_korelasi_pertanian_vs_harga(df_out: pd.DataFrame):
    st.markdown("## 🔗 Korelasi Pertanian ↔ Harga")
    data = _build_corr_dataset(df_out)
    if data.empty:
        st.info("Belum ada kombinasi data pertanian dan harga yang cocok per tahun.")
        return

    tahun_tersedia = sorted(data["Tahun"].dropna().unique().astype(int))
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        pilih_tahun = st.selectbox("Pilih Tahun", ["Seluruh Tahun"] + [str(t) for t in tahun_tersedia], index=0, key="corr_tahun")
    with c2:
        method = st.selectbox("Metode Korelasi", ["pearson", "spearman"], index=0, key="corr_method")
    with c3:
        pilih_harga = st.multiselect("Harga yang dianalisis", ["Harga_Merah","Harga_Hijau"], default=["Harga_Merah","Harga_Hijau"], key="corr_harga")

    dfc = data.copy()
    if pilih_tahun != "Seluruh Tahun":
        dfc = dfc[dfc["Tahun"] == float(pilih_tahun)]
    if dfc.empty:
        st.warning("Tidak ada data pada pilihan tersebut."); return

    left = [c for c in ["luas_areal","produksi","produktivitas"] if c in dfc.columns]
    right = [c for c in pilih_harga if c in dfc.columns]
    if not left or not right:
        st.info("Kolom fitur pertanian atau harga belum lengkap."); return

    corr_mat = pd.DataFrame(index=left, columns=right, dtype=float)
    for lx in left:
        for rx in right:
            x = pd.to_numeric(dfc[lx], errors="coerce")
            y = pd.to_numeric(dfc[rx], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() >= 3:
                if method == "spearman":
                    corr = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                else:
                    corr = np.corrcoef(x[m], y[m])[0,1]
                corr_mat.at[lx, rx] = float(corr)
            else:
                corr_mat.at[lx, rx] = np.nan

    fig, ax = plt.subplots(figsize=(4 + 1.2*len(right), 3.6))
    im = ax.imshow(corr_mat.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Koefisien Korelasi")
    ax.set_xticks(range(len(right))); ax.set_xticklabels([c.replace("_", " ") for c in right])
    ax.set_yticks(range(len(left))); ax.set_yticklabels([c.replace("_", " ").title() for c in left])
    ax.set_title("Matriks Korelasi Pertanian ↔ Harga" + (f" • Tahun {pilih_tahun}" if pilih_tahun != "Seluruh Tahun" else ""))

    for i in range(len(left)):
        for j in range(len(right)):
            v = corr_mat.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black")

    plt.tight_layout(); st.pyplot(fig)

    st.markdown("#### Scatter dan Tren Linear")
    c4, c5 = st.columns([1.1, 1.1])
    with c4:
        x_sel = st.selectbox("Fitur Pertanian (X)", left, format_func=lambda s: s.replace("_"," ").title(), key="corr_x")
    with c5:
        y_sel = st.selectbox("Fitur Harga (Y)", right, format_func=lambda s: s.replace("_"," ").title(), key="corr_y")

    sub = dfc[["Lokasi","Kategori","Tahun", x_sel, y_sel]].dropna()
    if sub.empty:
        st.info("Tidak ada pasangan data untuk scatter."); return

    Xv = sub[x_sel].to_numpy(dtype=float)
    Yv = sub[y_sel].to_numpy(dtype=float)
    if Xv.size >= 2:
        b1, b0 = np.polyfit(Xv, Yv, deg=1)
        r = np.corrcoef(Xv, Yv)[0,1] if Xv.size > 1 else np.nan
    else:
        b1, b0, r = 0.0, float(np.nan), float(np.nan)

    fig2, ax2 = plt.subplots(figsize=(7.6, 5.2))
    ax2.scatter(Xv, Yv, s=26, alpha=0.85)
    if np.isfinite(b1) and np.isfinite(b0):
        xs = np.linspace(np.nanmin(Xv), np.nanmax(Xv), 100)
        ax2.plot(xs, b1*xs + b0, lw=2)
    ax2.set_xlabel(x_sel.replace("_"," ").title())
    ax2.set_ylabel(y_sel.replace("_"," ").title())
    ttl = f"Scatter {x_sel.replace('_',' ').title()} vs {y_sel.replace('_',' ').title()}"
    if pilih_tahun != "Seluruh Tahun": ttl += f" • {pilih_tahun}"
    ax2.set_title(ttl); ax2.grid(True, linestyle="--", alpha=.35)
    if np.isfinite(r): ax2.text(0.02, 0.98, f"r = {r:.2f}", transform=ax2.transAxes, va="top", ha="left")
    plt.tight_layout(); st.pyplot(fig2)


# ----------------------- Visual Harga ----------------------
def _year_cols_price(df_out: pd.DataFrame, kind: str):
    patt = re.compile(rf"(harga|price).*?(cabe|cabai)?\s*{kind}.*?(19|20)\d{{2}}", re.I)
    cols = [c for c in df_out.columns if patt.search(str(c))]
    def _y(c):
        m = re.search(r"(19|20)\d{2}", str(c))
        return int(m.group()) if m else -1
    return sorted(cols, key=_y)


def render_tren_harga(df_out: pd.DataFrame):
    ascending_flag = (urutkan == "Terkecil")
    rank_label = "Top" if not ascending_flag else "Bottom"

    st.markdown("## 📈 Tren Harga Cabai Rawit")
    lokasi_col = _find_lokasi_col(df_out)
    if lokasi_col is None: st.warning("Kolom lokasi tidak ditemukan."); return

    c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1])
    with c1:
        jenis = st.selectbox("Jenis Harga", ["merah","hijau"], format_func=lambda s: f"Harga {s.title()}",
                             key="tren_harga_jenis")
    with c2:
        mode_view = st.selectbox("Mode", ["Garis (Top-N)", "Heatmap"], index=0, key="tren_harga_mode")
    with c3:
        topn = st.slider("Top Lokasi", 1, 20, 5, key="tren_harga_topn")
    with c4:
        urutkan = st.radio("Urutan", ["Terbesar","Terkecil"], horizontal=True, key="tren_harga_urut")

    ycols = _year_cols_price(df_out, jenis)
    if not ycols:
        st.info("Kolom harga tidak ditemukan."); return

    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols: df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    long = df_num.melt(id_vars=[lokasi_col, "Kategori"], value_vars=ycols, var_name="Kolom", value_name="Nilai")
    long["Tahun"] = long["Kolom"].astype(str).str.extract(r"((?:19|20)\d{2})").astype(float)
    long = long.dropna(subset=["Nilai","Tahun"]).rename(columns={lokasi_col:"Lokasi"})

    mean_by_loc = long.groupby("Lokasi")["Nilai"].mean().sort_values(ascending=(urutkan=="Terkecil"))
    top_locs = mean_by_loc.index.tolist()[:topn]
    years = sorted(long["Tahun"].dropna().unique())

    if mode_view == "Garis (Top-N)":
        fig, ax = plt.subplots(figsize=(10.5, 6))
        colors = plt.cm.tab20.colors
        for i, lok in enumerate(top_locs):
            sub = long[long["Lokasi"] == lok].sort_values("Tahun")
            ax.plot(sub["Tahun"], sub["Nilai"], marker="o", lw=2, alpha=.95, label=lok, color=colors[i % len(colors)])
        # ax.set_title(f"TREN TOP-{topn} Harga {jenis.title()}")
        ax.set_title(f"TREN {rank_label}-{topn} Harga {jenis.title()}")
        ax.set_xlabel("Tahun"); ax.set_ylabel("Harga")
        ax.grid(True, linestyle="--", alpha=.35)
        ax.legend(fontsize=8, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout(rect=[0,0,0.78,1]); st.pyplot(fig)
    else:
        # pvt = long.pivot_table(index="Lokasi", columns="Tahun", values="Nilai", aggfunc="mean")
        # rows = pvt.mean(axis=1).sort_values(ascending=(urutkan=="Terkecil")).head(topn).index.tolist()
        # pvt = pvt.loc[rows]
        # fig, ax = plt.subplots(figsize=(1.2*len(pvt.columns)+4, 0.45*len(pvt.index)+2.6))
        # im = ax.imshow(pvt.values, aspect="auto", interpolation="nearest")
        # fig.colorbar(im, ax=ax).set_label("Harga")
        # ax.set_xticks(range(len(pvt.columns))); ax.set_xticklabels([str(int(c)) for c in pvt.columns])
        # ax.set_yticks(range(len(pvt.index))); ax.set_yticklabels(pvt.index)
        # ax.set_title(f"Heatmap Harga {jenis.title()} — Top-{topn}")
        # mean_val = np.nanmean(pvt.values)
        # for (i,j), v in np.ndenumerate(pvt.values):
        #     if not np.isnan(v):
        #         ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=8, color=("white" if v>mean_val else "black"))
        # plt.tight_layout(); st.pyplot(fig)

        pvt = long.pivot_table(index="Lokasi", columns="Tahun", values="Nilai", aggfunc="mean")

        order = pvt.mean(axis=1).sort_values(ascending=ascending_flag)
        rows = order.head(topn).index.tolist()

        # Jika 'Terkecil', letakkan yang paling kecil di baris paling bawah
        if ascending_flag:
            rows = rows[::-1]

        pvt = pvt.loc[rows]

        fig, ax = plt.subplots(figsize=(1.2*len(pvt.columns)+4, 0.45*len(pvt.index)+2.6))
        im = ax.imshow(pvt.values, aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax).set_label("Harga")

        ax.set_xticks(range(len(pvt.columns)))
        ax.set_xticklabels([str(int(c)) for c in pvt.columns])
        ax.set_yticks(range(len(pvt.index)))
        ax.set_yticklabels(pvt.index)

        ax.set_title(f"Heatmap Harga {jenis.title()} — {rank_label}-{topn}")

        mean_val = np.nanmean(pvt.values)
        for (i, j), v in np.ndenumerate(pvt.values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=8,
                        color=("white" if v > mean_val else "black"))

        plt.tight_layout()
        st.pyplot(fig)



def render_top_harga(df_out: pd.DataFrame):
    st.markdown("## 🏆 Lokasi Harga Tertinggi")
    lokasi_col = _find_lokasi_col(df_out)
    if lokasi_col is None: st.warning("Kolom lokasi tidak ditemukan."); return
    c1, c2 = st.columns([1.2, 1])
    with c1:
        jenis = st.selectbox("Jenis Harga", ["merah","hijau"], format_func=lambda s: f"Harga {s.title()}",
                             key="topharga_jenis")
    with c2:
        urutkan = st.radio("Urutan", ["Terbesar","Terkecil"], horizontal=True, key="topharga_urut")
    ycols = _year_cols_price(df_out, jenis)
    if not ycols:
        st.info("Kolom harga tidak ditemukan."); return
    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols: df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    df_num["Rata_Rata"] = df_num[ycols].mean(axis=1, skipna=True)
    agg = df_num.groupby([lokasi_col, "Kategori"], dropna=False)["Rata_Rata"].mean().reset_index()
    agg = agg.sort_values("Rata_Rata", ascending=(urutkan=="Terkecil")).head(15)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.bar(range(len(agg)), agg["Rata_Rata"].values)
    for i,b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{agg['Rata_Rata'].iat[i]:,.0f}", ha="center", va="bottom", fontsize=9)
    labels = [f"{row[lokasi_col]} ({row['Kategori']})" for _, row in agg.iterrows()]
    ax.set_xticks(range(len(agg))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Rata-rata Harga"); ax.set_title(f"Top 15 Lokasi — Harga {jenis.title()} (Rata-rata  per Tahun)")
    ax.grid(axis="y", linestyle="--", alpha=.4)
    plt.tight_layout(); st.pyplot(fig)


def render_boxplot_harga(df_out: pd.DataFrame):
    st.markdown("## 📈 Distribusi Harga per Tahun")
    lokasi_col = _find_lokasi_col(df_out)
    if lokasi_col is None: st.warning("Kolom lokasi tidak ditemukan."); return

    jenis = st.selectbox("Jenis Harga", ["merah","hijau"], format_func=lambda s: f"Harga {s.title()}",
                         key="boxharga_jenis" )
    ycols = _year_cols_price(df_out, jenis)
    if not ycols:
        st.info("Kolom harga tidak ditemukan."); return
    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols: df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    long = df_num.melt(id_vars=[lokasi_col, "Kategori"], value_vars=ycols, var_name="Kolom", value_name="Nilai")
    long["Tahun"] = long["Kolom"].astype(str).str.extract(r"((?:19|20)\d{2})").astype(int)
    long = long.dropna(subset=["Nilai"]).rename(columns={lokasi_col:"Lokasi"})

    tahun_list = sorted(long["Tahun"].unique())
    tahun_pilihan = st.selectbox("Pilih Tahun", tahun_list, index=len(tahun_list)-1, key="boxharga_tahun")
    data_y = long[long["Tahun"] == tahun_pilihan]
    if data_y.empty: st.info("Tidak ada data untuk tahun itu."); return

    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    data_y["Kategori"] = pd.Categorical(data_y["Kategori"], categories=cat_order, ordered=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [data_y[data_y["Kategori"] == c]["Nilai"].dropna().values for c in cat_order]
    ax.boxplot(vals, patch_artist=True)
    ax.set_xticks(range(1, len(cat_order)+1)); ax.set_xticklabels(cat_order, rotation=0)
    ax.set_ylabel("Harga"); ax.set_title(f"Distribusi Harga {jenis.title()} • {tahun_pilihan}")
    ax.grid(axis="y", linestyle="--", alpha=.35)
    plt.tight_layout(); st.pyplot(fig)


def render_boxplot_harga_combined(df_out: pd.DataFrame):
    st.markdown("## 📈 Distribusi Harga Seluruh Tahun")
    lokasi_col = _find_lokasi_col(df_out)
    if lokasi_col is None: st.warning("Kolom lokasi tidak ditemukan."); return
    jenis = st.selectbox("Jenis Harga", ["merah","hijau"], index=0,
                         format_func=lambda s: f"Harga {s.title()}",
                         key="boxharga_combined_jenis" )
    ycols = _year_cols_price(df_out, jenis)
    if not ycols:
        st.info("Kolom harga tidak ditemukan."); return
    df_num = df_out[[lokasi_col, "Kategori"] + ycols].copy()
    for c in ycols: df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    long = df_num.melt(id_vars=[lokasi_col, "Kategori"], value_vars=ycols, var_name="Kolom", value_name="Nilai")
    long["Tahun"] = long["Kolom"].astype(str).str.extract(r"((?:19|20)\d{2})").astype(int)
    long = long.dropna(subset=["Nilai"]).rename(columns={lokasi_col:"Lokasi"})

    tahun_list = sorted(long["Tahun"].unique())
    n_years = len(tahun_list); n_cols = 3; n_rows = int(np.ceil(n_years / n_cols)) if n_years else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
    axes = np.atleast_1d(axes).flatten()

    cat_order = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
    for idx, year in enumerate(tahun_list):
        ax = axes[idx]; sub = long[long["Tahun"] == year].copy()
        if sub.empty: ax.axis("off"); continue
        sub["Kategori"] = pd.Categorical(sub["Kategori"], categories=cat_order, ordered=True)
        vals = [sub[sub["Kategori"] == c]["Nilai"].dropna().values for c in cat_order]
        ax.boxplot(vals, patch_artist=True); ax.set_title(f"{year}")
        ax.set_xticks(range(1, len(cat_order)+1)); ax.set_xticklabels(cat_order, rotation=0, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=.35)
    for j in range(len(tahun_list), len(axes)): axes[j].axis("off")
    fig.suptitle(f"Distribusi Harga {jenis.title()} per Kategori — All Years", y=0.995, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.98]); st.pyplot(fig)


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


def render_points_map_dual(df_out, title="Pemetaan Titik Lokasi (Latitude/Longitude)", basis="Pertanian"):
    st.subheader(f"📍 {title}")
    try:
        lat_col, lon_col, df_pts = _clean_latlon(df_out)
        if not lat_col or not lon_col or df_pts.empty:
            st.info("Kolom Latitude/Longitude tidak ditemukan/valid."); return

        basis = basis or "Pertanian"
        cat_col = "Kategori_Agri" if basis == "Pertanian" else "Kategori_Harga"
        clu_col = "Cluster_Agri"  if basis == "Pertanian" else "Cluster_Harga"

        m = folium.Map(location=[-2.5, 118], zoom_start=4.6, tiles="cartodbpositron")

        ordered = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
        cat_to_hex = {
            "Sangat Rendah":"#d73027","Rendah":"#fc8d59","Cukup Rendah":"#fee090",
            "Sedang":"#ffffbf","Cukup Tinggi":"#e0f3f8","Tinggi":"#91bfdb","Sangat Tinggi":"#4575b4"
        }

        def _mean_years(row, key_regex):
            year_cols  = [c for c in df_pts.columns if re.search(fr"{key_regex}.*?(19|20)\d{{2}}", str(c), flags=re.I)]
            plain_cols = [c for c in df_pts.columns if re.search(fr"^{key_regex}", str(c), flags=re.I) and not re.search(r"(19|20)\d{2}", str(c), flags=re.I)]
            use = year_cols if year_cols else plain_cols
            if not use: return np.nan
            v = pd.to_numeric(row[use], errors="coerce")
            return float(v.mean())

        def _mean_price(row, warna):
            ycols = [c for c in df_pts.columns
                     if re.search(rf"(harga|price).*?(cabe|cabai)?\s*{warna}.*?(19|20)\d{{2}}", str(c), re.I)]
            if not ycols: return np.nan
            v = pd.to_numeric(row[ycols], errors="coerce")
            return float(v.mean())

        name_cols = [c for c in ["Kabupaten/Kota","Kabupaten","Kota","Provinsi","Lokasi"] if c in df_pts.columns]

        for _, r in df_pts.iterrows():
            kategori_agri  = r.get("Kategori_Agri", "-")
            cluster_agri   = r.get("Cluster_Agri", "-")
            kategori_harga = r.get("Kategori_Harga", "-")
            cluster_harga  = r.get("Cluster_Harga", "-")

            cat_now = str(r.get(cat_col, kategori_agri))
            hex_color = cat_to_hex.get(cat_now, "#666666")
            nama = next((r.get(c) for c in name_cols if pd.notna(r.get(c))), "-")

            luas    = _mean_years(r, "luas")
            prod    = _mean_years(r, "produksi")
            produkt = _mean_years(r, "produkt")
            h_merah = _mean_price(r, "merah")
            h_hijau = _mean_price(r, "hijau")

            html = f"""<b>{nama}</b><br>
            <div style='margin-top:2px'>
              <u>Kategori Pertanian</u>: <b>{kategori_agri}</b> (Cluster: {cluster_agri})<br>
              <u>Kategori Harga</u>: <b>{kategori_harga}</b> (Cluster: {cluster_harga})
            </div>
            <hr style='margin:6px 0;'>
            <b>Rata-Rata Lintas Tahun:</b><br>
            • Luas Areal: {luas:,.0f} ha<br>
            • Produksi: {prod:,.0f} ton<br>
            • Produktivitas: {produkt:,.2f} ton/ha<br>
            • Harga Merah: {h_merah:,.0f}<br>
            • Harga Hijau: {h_hijau:,.0f}
            """

            folium.CircleMarker(
                location=[r[lat_col], r[lon_col]],
                radius=6, color=hex_color, weight=1, fill=True,
                fill_color=hex_color, fill_opacity=0.9,
                popup=folium.Popup(html, max_width=280),
            ).add_to(m)

        legend = ""
        present = set(str(x) for x in df_pts.get(cat_col, pd.Series([], dtype=object)).dropna().unique())
        active = [c for c in ordered if c in present] or ordered
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
            font-size: 13px; line-height: 1.2; min-width: 180px;">
            <div style="font-weight:700; margin-bottom:6px;">🗺️ Keterangan ({basis})</div>{legend}</div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=None, height=560)
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


def render_location_feature_means(df_out: pd.DataFrame, feat_cols: list[str] | None = None):
    if df_out is None or df_out.empty: return
    lokasi_col = next((c for c in df_out.columns 
                       if c.lower() in ["kabupaten/kota","kabupaten","kota","provinsi","lokasi"]), None)
    if not lokasi_col: return
    used = [c for c in (feat_cols or []) if c in df_out.columns]
    if not used:
        patt = re.compile(r"^(luas|produksi|produkt)", flags=re.I)
        used = [c for c in df_out.columns if patt.search(str(c))]
    if not used: return
    tmp = df_out[[lokasi_col, "Kategori"] + used].copy()
    for c in used: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp["Rata_Rata"] = tmp[used].mean(axis=1, skipna=True)
    out = (tmp.groupby([lokasi_col, "Kategori"])["Rata_Rata"].mean()
             .reset_index().sort_values("Rata_Rata", ascending=False))
    st.markdown("#### 📍 Rata-rata Fitur per Lokasi (lintas tahun)")
    st.dataframe(out, use_container_width=True)


# ==== PDF helpers ====
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def _add_report_fig(fig, key="report_figs"):
    lst = st.session_state.get(key, []); lst.append(fig); st.session_state[key] = lst


def build_pdf_report(level_sel: str) -> bytes:
    def _text_page_fig(title: str, lines: list[str], figsize=(8.27, 11.69)):
        fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
        y = 0.95; ax.text(0.5, y, title, ha="center", va="top", fontsize=16, fontweight="bold")
        y -= 0.06
        for line in lines:
            ax.text(0.06, y, str(line), ha="left", va="top", fontsize=11)
            y -= 0.035
            if y < 0.06: break
        return fig

    def _df_to_table_fig(df: pd.DataFrame, title="Tabel", max_rows=30, max_cols=12, figsize=(11.69, 8.27)):
        fig, ax = plt.subplots(figsize=figsize); ax.set_title(title, pad=18, fontsize=14, fontweight="bold"); ax.axis("off")
        df_disp = df.copy()
        if df_disp.shape[0] > max_rows: df_disp = df_disp.head(max_rows)
        if df_disp.shape[1] > max_cols: df_disp = df_disp.iloc[:, :max_cols]
        tbl = ax.table(cellText=df_disp.astype(str).values, colLabels=df_disp.columns.astype(str).tolist(),
                       loc="center", cellLoc="left", colLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.2)
        return fig

    method = st.session_state.get("method_used"); k_used = st.session_state.get("k_used")
    resdf = st.session_state.get("results_df"); df_out = st.session_state.get("df_out")
    used_cols = st.session_state.get("diag_used_cols_agri", []); cat_order = st.session_state.get("ordered_categories", [])
    pinfo = st.session_state.get("pca_info_agri", {"used": False})

    figs = st.session_state.get("report_figs", [])

    ts = datetime.now().strftime("%d-%m-%Y %H:%M")
    cover_lines = [
        f"Waktu pembuatan: {ts}",
        f"Level data: {level_sel}",
        f"Metode: {method if method else '-'}   |   k: {k_used if k_used is not None else '-'}",
        f"Fitur pertanian yang dipakai: {len(used_cols)} kolom" if used_cols else "Fitur pertanian: -",
        f"Kategori urut: {', '.join([str(c) for c in cat_order])}" if cat_order else "Kategori urut: -",
        "Peta Folium tidak dibekukan ke PDF. Untuk peta gunakan ekspor HTML terpisah.",
        f"PCA dipakai (agri): {'Ya' if pinfo.get('used') else 'Tidak'}"
    ]
    cover_fig = _text_page_fig("Laporan Klastering Pertanian & Harga Cabai Rawit", cover_lines)

    table_figs = []
    if isinstance(resdf, pd.DataFrame) and not resdf.empty:
        table_figs.append(_df_to_table_fig(resdf, title="Ringkasan Metrik Clustering"))
    if isinstance(df_out, pd.DataFrame) and not df_out.empty:
        id_cols = [c for c in df_out.columns if str(c).lower() in ["provinsi", "kabupaten/kota", "kabupaten", "kota", "lokasi"]]
        show_cols = (id_cols[:1] if id_cols else []) + [c for c in ["Cluster_Agri","Kategori_Agri","Cluster_Harga","Kategori_Harga"] if c in df_out.columns]
        if not show_cols: show_cols = df_out.columns[:6].tolist()
        table_figs.append(_df_to_table_fig(df_out[show_cols], title="Cuplikan Hasil Tabel"))

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(cover_fig, bbox_inches="tight")
        for tf in table_figs:
            try: pdf.savefig(tf, bbox_inches="tight")
            except Exception: pass
        for fg in figs:
            try: pdf.savefig(fg, bbox_inches="tight")
            except Exception: pass
    buf.seek(0); return buf.read()


# ============================== APP ==============================
def app():
    # ====== Prasyarat ======
    if not OPENPYXL_OK:
        st.error("Package **openpyxl** belum terpasang.\nJalankan: `pip install openpyxl` lalu refresh.")
        return
    _ensure_session_state()

    # --- patch penangkap figure agar bisa dikompilasi ke PDF ---
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

    # ====== Header ======
    st.markdown(
        "<h1 style='text-align:center; font-weight:800;'>Data Pertanian & Harga Cabai Rawit di Indonesia</h1>",
        unsafe_allow_html=True,
    )
    st.write("Pilih sumber data, tetapkan metode serta jumlah cluster, lalu jalankan clustering **Pertanian & Harga**.")
    st.markdown("---")
    st.sidebar.toggle("Urutkan kategori dari Tinggi → Rendah", value=False, key="cat_desc")

    # ====== Level data (fix Provinsi) ======
    prev_level = st.session_state.get("level_sel")
    level_sel = "Provinsi"
    st.session_state["level_sel"] = level_sel

    # ====== Sumber data ======
    prev_mode = st.session_state.get("data_mode_sel")
    data_mode = st.radio(
        "Sumber data:",
        ["Upload data sendiri", "Gunakan dataset contoh"],
        horizontal=True,
        key="data_mode_sel",
    )

    # Reset state bila ganti level atau mode
    if (prev_level is not None and prev_level != level_sel) or (prev_mode is not None and prev_mode != data_mode):
        for k in [
            "clustering_done",
            "results_df",
            "df_out",
            "X_scaled_agri",
            "labels_agri",
            "cluster_label_map_agri",
            "pca_info_agri",
            "diag_used_cols_agri",
            "X_scaled_harga",
            "labels_harga",
            "cluster_label_map_harga",
            "pca_info_harga",
            "diag_used_cols_harga",
            "method_used",
            "k_used",
            "ordered_categories",
            "report_figs",
        ]:
            st.session_state.pop(k, None)

    df_raw = None

    # ====== MODE: Upload ======
    if data_mode == "Upload data sendiri":
        st.subheader("📚 Template Dataset")
        try:
            tmpl_bytes = _load_template_from_path(level_sel)
            st.download_button(
                "📄 Unduh Template Dataset",
                data=tmpl_bytes,
                file_name=Path(TEMPLATE_PATHS[level_sel]).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="btn_dl_template",
            )
            st.caption(f"Sumber file: `{TEMPLATE_PATHS[level_sel]}`")
        except Exception as e:
            st.warning(f"Template tidak bisa dibaca dari path: {TEMPLATE_PATHS.get(level_sel, '-')}. {e}")

        st.subheader(f"📤 Upload Data {level_sel} (Excel .xlsx)")
        up = st.file_uploader(
            f"Tarik & lepas atau klik Browse (level {level_sel})",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Maks 200MB • .xlsx (membutuhkan openpyxl)",
            key="uploader_provinsi",
        )
        if up is None:
            st.info("Belum ada berkas yang diunggah.")
            return

        try:
            df_raw = pd.read_excel(up, header=0, engine="openpyxl").dropna(how="all")
        except Exception as e:
            st.error(f"Gagal memproses Excel: {e}")
            return

        # Validasi kolom identitas minimal
        if "Provinsi" not in map(str, df_raw.columns):
            st.error("Kolom **Provinsi** wajib ada.")
            return

    # ====== MODE: Dataset Contoh ======
    else:
        st.subheader(f"🗂️ Gunakan Dataset dari Path ({level_sel})")
        try:
            df_raw = _load_dataset_from_path(level_sel)
            st.success(f"Membaca: {DATASET_PATHS[level_sel]}")
            st.dataframe(df_raw.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membaca file dari path: {e}")
            return

        # Optional unduh dataset contoh
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df_raw.to_excel(w, index=False, sheet_name="dataset_provinsi")
        buf.seek(0)
        st.download_button(
            "⬇️ Unduh dataset contoh (.xlsx)",
            data=buf.read(),
            file_name="dataset_contoh_provinsi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="btn_dl_sample",
        )

    # ====== Info granularitas ======
    try:
        gran = detect_granularity(df_raw)
        st.info(f"Granularitas terdeteksi: **{gran.upper()}**. Backend menyesuaikan otomatis.")
    except Exception:
        pass

    # ====== Pengaturan Clustering ======
    st.markdown("### ⚙️ Pengaturan Clustering")
    method_sel = st.selectbox(
        "Pilih metode clustering:",
        ["K-Means", "Hierarchical Clustering"],  # bisa tambah "PCA + K-Means" bila ingin
        key="clust_method_sel",
    )
    k_sel = st.slider("Jumlah cluster (k)", 2, 7, 3, key="clust_k_sel")

    # ====== Jalankan ======
    if st.button("🚀 Jalankan Clustering (Pertanian & Harga)", key="btn_run_both"):
        try:
            df_out, results_both = run_both_clusterings(df_raw, method_sel, k_sel)
            st.session_state["report_figs"] = []  # reset koleksi figure untuk laporan
            st.success("Clustering pertanian & harga berhasil dijalankan.")
        except Exception as e:
            st.error(f"Gagal menjalankan clustering: {e}")
            return

    # ====== Panel Info Kolom ======
    with st.expander("🔎 Kolom fitur yang dipakai", expanded=False):
        st.write("**Pertanian**:", st.session_state.get("diag_used_cols_agri", "—"))
        st.write("**Harga**:", st.session_state.get("diag_used_cols_harga", "—"))
        st.write("**Metode**:", st.session_state.get("method_used", "—"), " | **k**:", st.session_state.get("k_used", "—"))

    # ====== Output & Visual ======
    if st.session_state.get("clustering_done", False):
        df_out = st.session_state["df_out"]

        # --- evaluasi performa ---
        st.markdown("### 📊 Evaluasi Performa Clustering")
        tabA, tabB = st.tabs(["Pertanian", "Harga"])
        with tabA:
            render_cluster_performance(
                st.session_state.get("X_scaled_agri"),
                st.session_state.get("method_used"),
                st.session_state.get("k_used"),
            )
        with tabB:
            if st.session_state.get("X_scaled_harga") is None:
                st.info("Kolom harga belum tersedia di dataset.")
            else:
                render_cluster_performance(
                    st.session_state.get("X_scaled_harga"),
                    st.session_state.get("method_used"),
                    st.session_state.get("k_used"),
                )

        # --- tabel hasil ringkas + distribusi ---
        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.markdown("### 📌 Hasil Tabel")
            lokasi_cols = [c for c in df_out.columns if c.lower() in ["lokasi", "provinsi", "kabupaten", "kabupaten/kota"]]
            show_cols = (lokasi_cols[:1] if lokasi_cols else []) + [
                "Cluster_Agri",
                "Kategori_Agri",
                "Cluster_Harga",
                "Kategori_Harga",
                "Cluster",
                "Kategori",
            ]
            show_cols = [c for c in show_cols if c in df_out.columns]
            st.dataframe(df_out[show_cols], use_container_width=True)

        with col2:
            st.markdown("### 📊 Jumlah Data per Kategori (Pertanian)")
            order_labels = get_ordered_categories(desc=st.session_state.get("cat_desc", False))
            cat_col = "Kategori_Agri" if "Kategori_Agri" in df_out.columns else ("Kategori" if "Kategori" in df_out.columns else None)
            if not cat_col:
                st.info("Kolom kategori pertanian tidak ditemukan. Jalankan clustering dulu.")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                count_df = df_out[cat_col].value_counts().reindex(order_labels, fill_value=0)
                warna = ["#C0392B", "#D35400", "#F1C40F", "#27AE60", "#2980B9", "#8E44AD", "#7F8C8D"][: len(count_df)]
                ax.bar(count_df.index, count_df.values, color=warna)
                for i, v in enumerate(count_df.values):
                    ax.text(i, v + 1, str(v), ha="center", fontsize=9, fontweight="bold")
                ax.set_title("Distribusi Kategori (Pertanian)")
                ax.set_xlabel("Kategori")
                ax.set_ylabel("Jumlah")
                ax.set_xticklabels(count_df.index, rotation=15, ha="right")
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                st.pyplot(fig)

        # --- PCA 2D ---
        st.markdown("### 🧭 Visualisasi Komposisi Cluster (PCA 2D)")
        t1, t2 = st.tabs(["Pertanian", "Harga"])
        with t1:
            cat_agri = "Kategori_Agri" if "Kategori_Agri" in df_out.columns else ("Kategori" if "Kategori" in df_out.columns else None)
            if cat_agri and st.session_state.get("X_scaled_agri") is not None:
                render_pca_scatter_visual(
                    st.session_state.get("X_scaled_agri"),
                    (df_out.assign(Kategori=df_out[cat_agri]) if cat_agri in df_out.columns else df_out),
                    st.session_state.get("labels_agri"),
                )
            else:
                st.info("Kolom kategori pertanian belum tersedia untuk PCA.")
        with t2:
            if st.session_state.get("X_scaled_harga") is None:
                st.info("Kolom harga belum tersedia.")
            else:
                cat_price = "Kategori_Harga" if "Kategori_Harga" in df_out.columns else ("Kategori" if "Kategori" in df_out.columns else None)
                if cat_price:
                    render_pca_scatter_visual(
                        st.session_state.get("X_scaled_harga"),
                        (df_out.assign(Kategori=df_out[cat_price]) if cat_price in df_out.columns else df_out),
                        st.session_state.get("labels_harga"),
                    )
                else:
                    st.info("Kolom kategori harga belum tersedia untuk PCA.")

        # --- Peta ---
        st.markdown("### 🗺️ Peta")
        basis_peta = st.selectbox(
            "Warna/legenda berdasarkan:",
            ["Pertanian", "Harga"],
            index=0,
            key="map_basis_select",
        )
        render_points_map_dual(
            df_out,
            title="Pemetaan Titik Lokasi (Latitude/Longitude)",
            basis=basis_peta,
        )

        # --- Analisis lanjutan: tab Pertanian & Harga ---
        st.markdown("### 📋 Analisis Lanjutan")
        T1, T2, T3, T4, T5 = st.tabs(
            [
                "📊 Gabungan Seluruh Tahun",
                "📆 Per Tahun",
                "📈 Tren",
                "🏆 Lokasi Tertinggi",
                "🔗 Korelasi P↔H",
            ]
        )
        with T1:
            A, B = st.tabs(["Pertanian", "Harga"])
            with A:
                render_boxplot_combined(df_out)
            with B:
                render_boxplot_harga_combined(df_out)
        with T2:
            A, B = st.tabs(["Pertanian", "Harga"])
            with A:
                render_boxplot(df_out)
            with B:
                render_boxplot_harga(df_out)
        with T3:
            A, B = st.tabs(["Pertanian", "Harga"])
            with A:
                render_tren_hasil_panen(df_out)
            with B:
                render_tren_harga(df_out)
        with T4:
            A, B = st.tabs(["Pertanian", "Harga"])
            with A:
                render_top_lokasi(df_out)
            with B:
                render_top_harga(df_out)
        with T5:
            render_korelasi_pertanian_vs_harga(df_out)

    # ====== Unduh Laporan PDF ======
    st.markdown("### 📥 Unduh Laporan PDF")
    if st.session_state.get("clustering_done", False):
        try:
            pdf_bytes = build_pdf_report(level_sel)
            fname = f"laporan_cabai_{level_sel.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            clicked = st.download_button(
                "⬇️ Unduh Laporan PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                key="btn_dl_pdf",
            )
            if clicked:
                st.session_state["report_figs"] = []
        except Exception as e:
            st.warning(f"Gagal membangun PDF: {e}")
    else:
        st.info("Jalankan clustering terlebih dahulu agar laporan dapat dibuat.")
