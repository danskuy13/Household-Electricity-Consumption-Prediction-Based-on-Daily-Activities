import os
import pandas as pd
import numpy as np

# Fungsi utama untuk membuat dataset dummy
def main():
    np.random.seed(42)
    n_samples = 500

    aktivitas = ["Menonton TV", "Memasak", "Mencuci", "Menyetrika", "Bermain Komputer", "Tidak Ada Aktivitas"]
    perangkat_listrik = ["AC", "Kulkas", "Mesin Cuci", "Kompor Listrik", "TV", "Komputer"]
    waktu_hari = ["Pagi", "Siang", "Sore", "Malam"]

    data = {
        "aktivitas": np.random.choice(aktivitas, n_samples),
        "perangkat_listrik": np.random.choice(perangkat_listrik, n_samples),
        "waktu_hari": np.random.choice(waktu_hari, n_samples),
        "durasi_penggunaan_jam": np.round(np.random.uniform(0.5, 8, n_samples), 1)
    }

    # Simulasikan konsumsi listrik berdasarkan durasi + bobot perangkat
    bobot_perangkat = {
        "AC": 0.8,
        "Kulkas": 0.3,
        "Mesin Cuci": 0.6,
        "Kompor Listrik": 0.9,
        "TV": 0.2,
        "Komputer": 0.4
    }

    konsumsi = []
    for i in range(n_samples):
        durasi = data["durasi_penggunaan_jam"][i]
        perangkat = data["perangkat_listrik"][i]
        noise = np.random.normal(0, 0.05)
        konsumsi.append(round(durasi * bobot_perangkat[perangkat] + noise, 2))

    data["konsumsi_listrik_kwh"] = konsumsi

    df = pd.DataFrame(data)

    # Simpan ke folder
    os.makedirs("data/generated", exist_ok=True)
    path = "data/generated/listrik_rumah.csv"
    df.to_csv(path, index=False)
    print(f"Dataset dummy disimpan di {path}")

if __name__ == "__main__":
    main()
