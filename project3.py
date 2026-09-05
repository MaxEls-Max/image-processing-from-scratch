import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

FOLDER_HASIL = "hasil_morfologi"
os.makedirs(FOLDER_HASIL, exist_ok=True)


def simpan_perbandingan(sebelum, sesudah, judul_sebelum, judul_sesudah, judul_besar, nama_file):
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(judul_besar, fontsize=13, fontweight="bold")

    axes[0].imshow(sebelum, cmap="gray")
    axes[0].set_title(judul_sebelum)
    axes[0].axis("off")

    axes[1].imshow(sesudah, cmap="gray")
    axes[1].set_title(judul_sesudah)
    axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(FOLDER_HASIL, nama_file)
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Tersimpan: {path}")

def kasus_erosi():
    print("\n=== 1. EROSI ===")
    print("Definisi : Erosi mengikis/mengecilkan bagian tepi objek pada citra biner.")
    print("           Piksel dipertahankan HANYA JIKA seluruh area kernel di")
    print("           sekitarnya masih berupa objek (piksel putih).")
    print("Manfaat  : Cocok untuk kasus dua objek yang saling menempel/bertumpuk,")
    print("           misalnya dua koin yang posisinya berdekatan pada citra")
    print("           hasil threshold. Dengan mengecilkan kedua objek, celah")
    print("           di antara keduanya akan muncul sehingga bisa dipisahkan.")

    citra = np.zeros((200, 320), dtype=np.uint8)
    cv2.circle(citra, (95, 100), 65, 255, -1)   # koin 1
    cv2.circle(citra, (225, 100), 65, 255, -1)  # koin 2 (menempel tipis dengan koin 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    hasil = cv2.erode(citra, kernel, iterations=2)

    simpan_perbandingan(
        citra, hasil,
        "Sebelum: 2 koin menempel", "Sesudah: erosi (sudah terpisah)",
        "1. EROSI - Memisahkan Objek yang Menempel",
        "1_erosi_pisahkan_koin.png"
    )

def kasus_dilasi():
    print("\n=== 2. DILASI ===")
    print("Definisi : Dilasi menebalkan/memperbesar bagian objek pada citra biner.")
    print("           Piksel dijadikan objek JIKA ADA MINIMAL SATU piksel objek")
    print("           di dalam area kernel di sekitarnya.")
    print("Manfaat  : Cocok untuk kasus garis atau goresan tulisan yang terputus-")
    print("           putus (misalnya akibat scan dokumen kurang bagus atau hasil")
    print("           deteksi tepi yang terpotong-potong). Dilasi akan menebalkan")
    print("           garis sehingga bagian yang terputus bisa tersambung kembali.")

    # --- buat citra uji: garis putus-putus ---
    citra = np.zeros((150, 300), dtype=np.uint8)
    for x_awal in range(20, 280, 40):     # bikin beberapa segmen garis dengan celah
        cv2.line(citra, (x_awal, 75), (x_awal + 25, 75), 255, 3)

    # --- kernel / structuring element ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

    # --- proses dilasi ---
    hasil = cv2.dilate(citra, kernel, iterations=1)

    simpan_perbandingan(
        citra, hasil,
        "Sebelum: garis terputus-putus", "Sesudah: dilasi (tersambung)",
        "2. DILASI - Menyambung Garis yang Terputus",
        "2_dilasi_sambung_garis.png"
    )

def kasus_opening():
    print("\n=== 3. OPENING ===")
    print("Definisi : Opening = EROSI dilanjutkan DILASI. Objek besar akan")
    print("           kembali ke ukuran semula setelah didilasi, tetapi objek")
    print("           kecil (noise) yang sudah 'hilang' saat erosi tidak akan")
    print("           muncul kembali.")
    print("Manfaat  : Cocok untuk membersihkan noise berupa bintik-bintik kecil")
    print("           pada citra biner (misalnya hasil thresholding yang kurang")
    print("           bersih), tanpa merusak bentuk objek utama yang berukuran")
    print("           lebih besar.")

    # --- buat citra uji: 1 objek utama + banyak noise bintik kecil ---
    citra = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(citra, (60, 60), (140, 140), 255, -1)  # objek utama

    rng = np.random.default_rng(42)
    for _ in range(150):                                  # taburkan noise bintik kecil
        x, y = rng.integers(0, 200), rng.integers(0, 200)
        cv2.circle(citra, (x, y), 2, 255, -1)

    # --- kernel / structuring element ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # --- proses opening ---
    hasil = cv2.morphologyEx(citra, cv2.MORPH_OPEN, kernel)

    simpan_perbandingan(
        citra, hasil,
        "Sebelum: banyak noise bintik", "Sesudah: opening (noise hilang)",
        "3. OPENING - Menghilangkan Noise",
        "3_opening_hilangkan_noise.png"
    )

def kasus_closing():
    print("\n=== 4. CLOSING ===")
    print("Definisi : Closing = DILASI dilanjutkan EROSI. Kebalikan dari opening.")
    print("           Lubang kecil di dalam objek akan tertutup saat didilasi,")
    print("           dan ukuran objek akan kembali ke ukuran semula setelah")
    print("           dierosi kembali.")
    print("Manfaat  : Cocok untuk menutup lubang-lubang kecil di dalam objek atau")
    print("           menyambung celah kecil di tepi objek, misalnya pada citra")
    print("           hasil thresholding yang bolong-bolong padahal objek aslinya")
    print("           solid/utuh.")

    # --- buat citra uji: objek dengan beberapa lubang kecil di dalamnya ---
    citra = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(citra, (40, 40), (160, 160), 255, -1)   # objek utama (solid)
    for (x, y) in [(70, 70), (100, 100), (130, 130), (70, 130), (130, 70)]:
        cv2.circle(citra, (x, y), 6, 0, -1)               # lubang-lubang kecil (dibuat hitam)

    # --- kernel / structuring element ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # --- proses closing ---
    hasil = cv2.morphologyEx(citra, cv2.MORPH_CLOSE, kernel)

    simpan_perbandingan(
        citra, hasil,
        "Sebelum: ada lubang kecil", "Sesudah: closing (lubang tertutup)",
        "4. CLOSING - Menutup Lubang pada Objek",
        "4_closing_tutup_lubang.png"
    )
    
def kasus_thinning():
    print("\n=== 5. THINNING ===")
    print("Definisi : Thinning menipiskan objek secara berulang dari bagian")
    print("           tepinya sedikit demi sedikit, tetapi tetap menjaga bentuk")
    print("           topologi objek (tidak sampai terputus), sampai tersisa")
    print("           kerangka (skeleton) setebal 1 piksel.")
    print("Manfaat  : Cocok untuk analisis bentuk objek yang tebal, misalnya")
    print("           mengambil kerangka dari huruf/tulisan tangan yang tebal")
    print("           untuk keperluan pengenalan pola (pattern recognition),")
    print("           karena bentuk dasarnya jadi lebih sederhana untuk dianalisis.")

    # --- buat citra uji: huruf tebal ---
    citra = np.zeros((200, 300), dtype=np.uint8)
    cv2.putText(citra, "CV", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, thickness=25)

    # --- proses thinning (pakai skeletonize dari scikit-image) ---
    citra_biner = citra > 0                      # ubah ke True/False dulu
    hasil_biner = skeletonize(citra_biner)        # hasilnya juga True/False
    hasil = (hasil_biner * 255).astype(np.uint8)  # ubah balik ke 0-255 buat ditampilkan

    simpan_perbandingan(
        citra, hasil,
        "Sebelum: huruf tebal", "Sesudah: thinning (jadi skeleton)",
        "5. THINNING - Mengambil Kerangka/Skeleton Objek",
        "5_thinning_skeleton.png"
    )


# =====================================================================
# PROGRAM UTAMA
# =====================================================================
if __name__ == "__main__":
    kasus_erosi()
    kasus_dilasi()
    kasus_opening()
    kasus_closing()
    kasus_thinning()

    print(f"\nSemua selesai! Cek folder '{FOLDER_HASIL}/' untuk melihat hasilnya.")
