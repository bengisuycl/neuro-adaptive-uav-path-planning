import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IEEE Makale/Tez Grafik Standartları
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# Çıktı klasörünüzün yolu (main.py'nin kaydettiği yer)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# Hangi metriklerin analiz edileceği
METRICS = {
    "Calc": "Comp. Time (s)",
    "Path": "Path Length (km)",
    "Track": "Tracking Error (m)",
    "DynSat": "Kinematic Violations",
    "Risk": "Threat Exposure"
}


def load_all_results():
    """outputs klasöründeki tüm CSV dosyalarını okuyup tek bir DataFrame'de birleştirir."""
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*_runs.csv"))
    if not csv_files:
        print("❌ outputs klasöründe hiç CSV dosyası bulunamadı!")
        return pd.DataFrame()

    all_data = []
    for file in csv_files:
        filename = os.path.basename(file)
        # Dosya adı formatı: YYYY-MM-DD_SenaryoID_AlgoritmaAdı_runs.csv
        # Örnek: 2026-03-05_S1_Base_Neuro-Adaptive_runs.csv
        parts = filename.replace("_runs.csv", "").split("_")

        # Parçalama mantığı
        date_str = parts[0]
        scen_id = parts[1] + "_" + parts[2]  # S1_Base
        alg_name = "_".join(parts[3:])  # Neuro-Adaptive, K-GNP vb.

        try:
            df = pd.read_csv(file)
            df["Algorithm"] = alg_name
            df["Scenario"] = scen_id
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Dosya okunamadı {filename}: {e}")

    return pd.concat(all_data, ignore_index=True)


def generate_thesis_table(df):
    """Tez için Ortalama ± Standart Sapma tablosu üretir."""
    print("\n" + "=" * 80)
    print(" 🏆 THESIS BENCHMARK RESULTS (Mean ± Std) ".center(80))
    print("=" * 80)

    # Sadece başarılı olan uçuşların metriklerini alalım (Adil bir kıyaslama için)
    # Başarısızların Calc süreleri hesaba katılabilir ama yol metrikleri anlamsızdır.
    scenarios = df["Scenario"].unique()

    for scen in sorted(scenarios):
        print(f"\n--- SCENARIO: {scen} ---")
        scen_df = df[df["Scenario"] == scen]
        algs = scen_df["Algorithm"].unique()

        # Başlık satırı
        header = f"{'Algorithm':<18} | {'Success':<8}"
        for m in METRICS.keys():
            header += f" | {METRICS[m]:<18}"
        print(header)
        print("-" * len(header))

        for alg in sorted(algs):
            alg_df = scen_df[scen_df["Algorithm"] == alg]
            total_runs = len(alg_df)
            success_runs = len(alg_df[alg_df["sim_status"] == "SUCCESS"])
            success_rate = f"{success_runs}/{total_runs}"

            # Sadece başarılı uçuşlarda (veya kısmi başarılılarda) metrik hesapla
            valid_df = alg_df[alg_df["sim_status"] == "SUCCESS"]

            row_str = f"{alg:<18} | {success_rate:<8}"

            for m in METRICS.keys():
                if len(valid_df) > 0 and m in valid_df.columns:
                    mean_val = valid_df[m].mean()
                    std_val = valid_df[m].std()

                    # Eğer değer sıfırsa veya sapma yoksa temiz göster (DynSat gibi)
                    if pd.isna(std_val) or std_val == 0.0:
                        val_str = f"{mean_val:.2f}"
                    else:
                        val_str = f"{mean_val:.2f} ± {std_val:.2f}"
                else:
                    val_str = "N/A"

                row_str += f" | {val_str:<18}"
            print(row_str)


def plot_benchmark_charts(df):
    """Her senaryo için bar grafikleri çizer."""
    scenarios = df["Scenario"].unique()

    for scen in scenarios:
        scen_df = df[(df["Scenario"] == scen) & (df["sim_status"] == "SUCCESS")]
        if scen_df.empty:
            continue

        # Algoritmaların metrik ortalamalarını alalım
        grouped = scen_df.groupby("Algorithm").mean(numeric_only=True)
        algs = grouped.index.tolist()

        # DynSat (Kinematik Uyum), Track (Takip Hatası), Calc (Süre) için grafikler çizelim
        target_metrics = ["DynSat", "Track", "Calc"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Algorithm Performance Comparison - {scen}", weight="bold", y=1.05)

        for i, metric in enumerate(target_metrics):
            if metric not in grouped.columns: continue

            ax = axes[i]
            values = grouped[metric].values

            # Kendi algoritmalarınızı (K-GNP, T-GnP) farklı renkte çizmek için:
            colors = ["#d95f02" if alg in ["K-GNP", "T-GnP"] else "#377eb8" for alg in algs]

            bars = ax.bar(algs, values, color=colors, edgecolor="black")
            ax.set_title(METRICS[metric])
            ax.set_ylabel("Lower is Better" if metric != "Success" else "Higher is Better")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Değerleri barların üzerine yaz
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Benchmark_Plot_{scen}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Çizim kaydedildi: {save_path}")
        plt.close()


if __name__ == "__main__":
    df_results = load_all_results()
    if not df_results.empty:
        generate_thesis_table(df_results)
        plot_benchmark_charts(df_results)