import os
import streamlit as st

# Opsional: cache baca file agar tidak berulang saat rerun
@st.cache_data
def load_pdf_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def app():
    st.title("Halaman Utama")
    st.markdown("---")

    # ====== Judul utama (center) ======
    st.markdown(
        "<h1 style='text-align:center; font-weight:800;'>Implementasi Algoritma K-Means dan Hierarchical Clustering "
        "untuk Pengelompokkan Wilayah Berdasarkan Produksi dan Harga Cabai di Indonesia</h1>",
        unsafe_allow_html=True
    )

    # ====== Gambar utama (center) ======
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.image(
            "assets/cabai rawit.jpg",
            caption="Cabai Rawit",
            use_container_width=False,
            width=620,
        )


    st.write("""
    Selamat datang di aplikasi Clustering Cabai Rawit. Aplikasi ini mendukung skripsi yang berfokus pada pengelompokan wilayah berdasarkan **produksi** dan **harga** cabai rawit di Indonesia.

    **Judul Skripsi:** Implementasi Algoritma K-Means Dan Hierarchical Clustering Untuk Pengelompokkan Wilayah Berdasarkan Produksi Dan Harga Cabai Di Indonesia

    **Tentang Aplikasi Skripsi:** Aplikasi web untuk mengelompokkan wilayah pada dua domain data, yaitu pertanian/produksi dan harga, agar pola serupa mudah diidentifikasi dan dianalisis secara visual.

    **Apa yang dilakukan aplikasi ini (ringkas):**
    - Memuat data pertanian/produksi dari BDSP Kementan dan data harga dari PIHPS
    - Melakukan pra-pemrosesan dan normalisasi variabel
    - Menjalankan K-Means dan Agglomerative (Ward)
    - Mengevaluasi hasil menggunakan Silhouette Coefficient dan Davies–Bouldin Index
    - Menyajikan hasil dalam tabel, grafik, dendrogram, serta peta tematik interaktif
    - Menyediakan perbandingan klaster produksi dengan klaster harga

    **Fitur Utama:**
    - **Halaman Utama:** Ringkasan aplikasi dan unduh manual book (PDF).
    - **Data Pertanian:** Analisis data produksi cabai rawit dari berbagai daerah di Indonesia.
    - **Harga Cabai Rawit:** Visualisasi tren harga cabai rawit dari waktu ke waktu.
    - **Tentang Kami:** Informasi tentang tim pengembang aplikasi ini.

    _Catatan: Versi ini tidak menggunakan data iklim. Analisis hanya mencakup data pertanian/produksi dan data harga._
    """)


    # ====== Tombol Unduh Manual Book (PDF) ======
    st.markdown("### Unduh Manual Book")
    pdf_path = "assets/manual_book.pdf"  # ganti sesuai lokasi PDF kamu

    if os.path.exists(pdf_path):
        pdf_bytes = load_pdf_bytes(pdf_path)
        st.download_button(
            label="⬇️ Unduh Manual Book (PDF)",
            data=pdf_bytes,                  # bisa juga: data=open(pdf_path, "rb")
            file_name="Manual_Book_Aplikasi_Cabai_Rawit.pdf",
            mime="application/pdf",
            key="download_manual_book"
        )
        st.caption("Klik tombol di atas untuk mengunduh file panduan aplikasi.")
    else:
        st.warning(f"File manual tidak ditemukan di: {pdf_path}. Pastikan path dan nama file sudah benar.")

    st.caption("Gunakan menu di sebelah kiri untuk menavigasi antara halaman-halaman yang berbeda.")
