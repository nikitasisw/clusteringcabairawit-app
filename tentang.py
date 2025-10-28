import streamlit as st

def app():
    st.title("Tentang Kami")
    st.write("""
    Aplikasi ini dirancang untuk membantu menganalisis pengelompokan produksi, harga, dan kondisi iklim cabai rawit di Indonesia. Dalam menghadapi permintaan yang tinggi, fluktuasi harga yang tajam, serta ketimpangan distribusi antarwilayah, aplikasi ini hadir sebagai solusi berbasis data untuk mendukung pengambilan keputusan yang lebih baik.

    Menggunakan data yang berasal dari Kementerian Pertanian (luas lahan, produksi, produktivitas), PIHPS Nasional (harga cabai rawit, cabai rawit hijau, cabai rawit merah), aplikasi ini menerapkan metode analisis pengelompokan (clustering) dengan teknik seperti K-Means dan Agglomerative Hierarchical Clustering (Ward’s Method).

    Pengguna dapat dengan mudah mengeksplorasi hasil analisis berupa label klaster per wilayah, grafik evaluasi, dendrogram, serta peta tematik interaktif yang memberikan wawasan tentang pola distribusi dan wilayah prioritas. Semua temuan ini dipetakan secara spasial, yang memudahkan pemahaman tentang keterkaitan antara produktivitas, harga, dan kondisi iklim di berbagai wilayah Indonesia.

    Dikembangkan dengan teknologi Python (Pandas, NumPy, Scikit-learn), aplikasi ini juga mengintegrasikan pemodelan terpisah per domain dan teknik evaluasi seperti Silhouette Coefficient dan Davies-Bouldin Index. Dengan antarmuka yang interaktif melalui Streamlit, aplikasi ini cocok digunakan oleh mahasiswa, peneliti, dan praktisi di bidang data science maupun pertanian.

    Secara keseluruhan, aplikasi ini memberikan alat yang berguna untuk perencanaan produksi, stabilisasi pasokan, serta strategi pemasaran, sekaligus menjadi sarana pembelajaran yang terstruktur dan dapat direplikasi untuk komoditas pertanian lainnya.
    """)