"""
TUGAS VISI KOMPUTER - PROJECT 2 - KELOMPOK 05
====================================================
"""

import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


PATH_GAMBAR = "gambar.jpg"

MAKS_DIMENSI = 200

FAKTOR_ZOOM = 2


def baca_citra_rgb(path):
    img = Image.open(path).convert('RGB')  # RGB = 3 kanal, masing-masing 8-bit (0-255)

    w, h = img.size
    if max(w, h) > MAKS_DIMENSI:
        skala = MAKS_DIMENSI / max(w, h)
        img = img.resize((int(w * skala), int(h * skala)))

    return np.array(img)   # shape: (tinggi, lebar, 3)


def brightness_manual(array, C):
    tinggi, lebar, kanal = array.shape
    hasil = np.zeros((tinggi, lebar, kanal), dtype=np.uint8)

    for y in range(tinggi):
        for x in range(lebar):
            for c in range(kanal):          # c = 0 (R), 1 (G), 2 (B)
                nilai = int(array[y, x, c]) + C
                nilai = max(0, min(255, nilai))
                hasil[y, x, c] = nilai

    return hasil


def contrast_manual(array, G, P):
    tinggi, lebar, kanal = array.shape
    hasil = np.zeros((tinggi, lebar, kanal), dtype=np.uint8)

    for y in range(tinggi):
        for x in range(lebar):
            for c in range(kanal):
                nilai = G * (int(array[y, x, c]) - P) + P
                nilai = int(round(nilai))
                nilai = max(0, min(255, nilai))
                hasil[y, x, c] = nilai

    return hasil


def pencerminan_manual(array, mode):
    tinggi, lebar, kanal = array.shape
    hasil = np.zeros((tinggi, lebar, kanal), dtype=np.uint8)

    for y in range(tinggi):
        for x in range(lebar):

            if mode == "horizontal":
                x_baru = lebar - 1 - x
                y_baru = y

            elif mode == "vertical":
                x_baru = x
                y_baru = tinggi - 1 - y

            else:  # kombinasi
                x_baru = lebar - 1 - x
                y_baru = tinggi - 1 - y

            hasil[y_baru, x_baru] = array[y, x]   # pindahkan (R,G,B) sekaligus

    return hasil


# ============================================================
# ROTASI 90 DERAJAT CLOCKWISE
# ============================================================
def rotasi_manual(array):
    tinggi, lebar, kanal = array.shape

    # Setelah diputar 90° clockwise:
    # tinggi menjadi lebar
    # lebar menjadi tinggi
    hasil = np.zeros((lebar, tinggi, kanal), dtype=np.uint8)

    for y in range(tinggi):
        for x in range(lebar):

            # Rumus rotasi 90° clockwise:
            #
            # x' = H - 1 - y
            # y' = x
            #
            # H = tinggi citra

            x_baru = tinggi - 1 - y
            y_baru = x

            # Memindahkan seluruh nilai RGB
            hasil[y_baru, x_baru] = array[y, x]

    return hasil


def deteksi_tepi_robert_manual(array):
    tinggi, lebar, kanal = array.shape
    hasil = np.zeros((tinggi, lebar, kanal), dtype=np.uint8)

    for y in range(tinggi - 1):
        for x in range(lebar - 1):
            for c in range(kanal):

                k1 = int(array[y, x, c]) - int(array[y + 1, x + 1, c])
                k2 = int(array[y, x + 1, c]) - int(array[y + 1, x, c])

                nilai = abs(k1) + abs(k2)
                nilai = min(255, nilai)

                hasil[y, x, c] = nilai

    return hasil


def noise_reduction_median_manual(array):
    tinggi, lebar, kanal = array.shape
    hasil = np.zeros((tinggi, lebar, kanal), dtype=np.uint8)

    for y in range(tinggi):
        for x in range(lebar):
            for c in range(kanal):

                tetangga = []

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):

                        yy = max(0, min(tinggi - 1, y + dy))
                        xx = max(0, min(lebar - 1, x + dx))

                        tetangga.append(int(array[yy, xx, c]))

                tetangga.sort()

                median = tetangga[len(tetangga) // 2]

                hasil[y, x, c] = median

    return hasil


# ============================================================
# GUI / APLIKASI
# ============================================================

class AplikasiCitra:

    def __init__(self, root, array_asli):

        self.root = root

        self.root.title(
            "Project 2 - Operasi Citra Warna / RGB (Visi Komputer)"
        )

        self.array_asli = array_asli
        self.zoom = FAKTOR_ZOOM

        tinggi, lebar, _ = array_asli.shape

        self.label_status = tk.Label(
            root,
            text="Gerakkan mouse di atas gambar untuk melihat koordinat & nilai RGB piksel",
            font=("Consolas", 10)
        )

        self.label_status.pack(pady=6)

        img_tampil = Image.fromarray(array_asli).resize(
            (lebar * self.zoom, tinggi * self.zoom),
            Image.NEAREST
        )

        self.foto = ImageTk.PhotoImage(img_tampil)

        self.canvas = tk.Canvas(
            root,
            width=lebar * self.zoom,
            height=tinggi * self.zoom
        )

        self.canvas.pack()

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.foto
        )

        self.canvas.bind(
            "<Motion>",
            self.saat_mouse_bergerak
        )

        frame_tombol = tk.Frame(
            root,
            pady=12
        )

        frame_tombol.pack()

        daftar_tombol = [

            ("1. Contrast", self.jalankan_contrast),

            ("2. Brightness", self.jalankan_brightness),

            ("3. Pencerminan", self.jalankan_pencerminan),

            ("4. Rotasi 90°", self.jalankan_rotasi),

            ("5. Deteksi Tepi (Robert)", self.jalankan_deteksi_tepi),

            ("6. Noise Reduction", self.jalankan_noise_reduction),

        ]

        for i, (teks, fungsi) in enumerate(daftar_tombol):

            tk.Button(
                frame_tombol,
                text=teks,
                width=22,
                command=fungsi
            ).grid(
                row=i // 3,
                column=i % 3,
                padx=5,
                pady=5
            )


    # ========================================================
    # MENAMPILKAN KOORDINAT DAN NILAI RGB
    # ========================================================

    def saat_mouse_bergerak(self, event):

        x = event.x // self.zoom
        y = event.y // self.zoom

        tinggi, lebar, _ = self.array_asli.shape

        if 0 <= x < lebar and 0 <= y < tinggi:

            r, g, b = self.array_asli[y, x]

            self.label_status.config(
                text=f"Pixel (x={x}, y={y})   R={r}  G={g}  B={b}"
            )


    # ========================================================
    # MENYIMPAN HASIL
    # ========================================================

    def simpan_hasil(self, array_hasil, nama_default):

        img_hasil = Image.fromarray(
            array_hasil,
            mode="RGB"
        )

        path_simpan = filedialog.asksaveasfilename(

            defaultextension=".jpg",

            initialfile=nama_default,

            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Semua file", "*.*")
            ]
        )

        if path_simpan:

            img_hasil.save(path_simpan)

            messagebox.showinfo(
                "Selesai",
                f"Hasil berhasil disimpan sebagai:\n{path_simpan}"
            )


    # ========================================================
    # 1. CONTRAST
    # ========================================================

    def jalankan_contrast(self):

        G = simpledialog.askfloat(
            "Contrast",
            "Masukkan nilai G (koefisien penguatan kontras):",
            initialvalue=1.5
        )

        if G is None:
            return

        P = simpledialog.askfloat(
            "Contrast",
            "Masukkan nilai P (pusat pengontrasan, 0-255):",
            initialvalue=128
        )

        if P is None:
            return

        hasil = contrast_manual(
            self.array_asli,
            G,
            P
        )

        self.simpan_hasil(
            hasil,
            "hasil_contrast.jpg"
        )


    # ========================================================
    # 2. BRIGHTNESS
    # ========================================================

    def jalankan_brightness(self):

        C = simpledialog.askinteger(
            "Brightness",
            "Masukkan nilai C (boleh negatif untuk menggelapkan):",
            initialvalue=30
        )

        if C is None:
            return

        hasil = brightness_manual(
            self.array_asli,
            C
        )

        self.simpan_hasil(
            hasil,
            "hasil_brightness.jpg"
        )


    # ========================================================
    # 3. PENCERMINAN
    # ========================================================

    def jalankan_pencerminan(self):

        pilihan = simpledialog.askstring(
            "Pencerminan",
            "Ketik salah satu: horizontal / vertical / kombinasi"
        )

        if not pilihan:
            return

        pilihan = pilihan.strip().lower()

        if pilihan not in (
            "horizontal",
            "vertical",
            "kombinasi"
        ):

            messagebox.showerror(
                "Error",
                "Pilihan harus: horizontal, vertical, atau kombinasi"
            )

            return

        hasil = pencerminan_manual(
            self.array_asli,
            pilihan
        )

        self.simpan_hasil(
            hasil,
            f"hasil_pencerminan_{pilihan}.jpg"
        )


    # ========================================================
    # 4. ROTASI 90° CLOCKWISE
    # ========================================================

    def jalankan_rotasi(self):

        hasil = rotasi_manual(
            self.array_asli
        )

        self.simpan_hasil(
            hasil,
            "hasil_rotasi_90_clockwise.jpg"
        )


    # ========================================================
    # 5. DETEKSI TEPI ROBERT
    # ========================================================

    def jalankan_deteksi_tepi(self):

        hasil = deteksi_tepi_robert_manual(
            self.array_asli
        )

        self.simpan_hasil(
            hasil,
            "hasil_deteksi_tepi_robert.jpg"
        )


    # ========================================================
    # 6. NOISE REDUCTION
    # ========================================================

    def jalankan_noise_reduction(self):

        hasil = noise_reduction_median_manual(
            self.array_asli
        )

        self.simpan_hasil(
            hasil,
            "hasil_noise_reduction.jpg"
        )


# ============================================================
# PROGRAM UTAMA
# ============================================================

if __name__ == "__main__":

    citra = baca_citra_rgb(
        PATH_GAMBAR
    )

    root = tk.Tk()

    app = AplikasiCitra(
        root,
        citra
    )

    root.mainloop()
