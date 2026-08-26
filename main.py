"""
TUGAS VISI KOMPUTER - PROJECT 2 - KELOMPOK 04
====================================================
"""

import math
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


def rotasi_manual(array, sudut_derajat):
    tinggi, lebar, kanal = array.shape
    theta = math.radians(sudut_derajat)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    lebar_baru = max(1, int(round(abs(lebar * cos_t) + abs(tinggi * sin_t))))
    tinggi_baru = max(1, int(round(abs(lebar * sin_t) + abs(tinggi * cos_t))))

    pojok = [(0, 0), (lebar - 1, 0), (0, tinggi - 1), (lebar - 1, tinggi - 1)]
    x_aksen_semua = [px * cos_t + py * sin_t for px, py in pojok]
    y_aksen_semua = [-px * sin_t + py * cos_t for px, py in pojok]
    x_min = min(x_aksen_semua)
    y_min = min(y_aksen_semua)

    hasil = np.zeros((tinggi_baru, lebar_baru, kanal), dtype=np.uint8)

    # INVERSE MAPPING: untuk tiap piksel pada citra HASIL, jadi cari posisi asalnya
    # di citra INPUT (supaya tidak ada 'lubang' pada hasil rotasi)
    for Y in range(tinggi_baru):
        for X in range(lebar_baru):
            x_aksen = X + x_min
            y_aksen = Y + y_min

            x_asal = x_aksen * cos_t - y_aksen * sin_t
            y_asal = x_aksen * sin_t + y_aksen * cos_t

            x_asal = int(round(x_asal))
            y_asal = int(round(y_asal))

            if 0 <= x_asal < lebar and 0 <= y_asal < tinggi:
                hasil[Y, X] = array[y_asal, x_asal]   # copy (R,G,B) sekaligus
            else:
                hasil[Y, X] = (0, 0, 0)   # di luar citra asli -> hitam

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
                        yy = max(0, min(tinggi - 1, y + dy))   # clamp ke tepi citra
                        xx = max(0, min(lebar - 1, x + dx))
                        tetangga.append(int(array[yy, xx, c]))

                tetangga.sort()
                median = tetangga[len(tetangga) // 2]
                hasil[y, x, c] = median
    return hasil


class AplikasiCitra:
    def __init__(self, root, array_asli):
        self.root = root
        self.root.title("Project 2 - Operasi Citra Warna / RGB (Visi Komputer)")

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
            (lebar * self.zoom, tinggi * self.zoom), Image.NEAREST
        )
        self.foto = ImageTk.PhotoImage(img_tampil)

        self.canvas = tk.Canvas(root, width=lebar * self.zoom, height=tinggi * self.zoom)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.foto)
        self.canvas.bind("<Motion>", self.saat_mouse_bergerak)

        frame_tombol = tk.Frame(root, pady=12)
        frame_tombol.pack()

        daftar_tombol = [
            ("1. Contrast", self.jalankan_contrast),
            ("2. Brightness", self.jalankan_brightness),
            ("3. Pencerminan", self.jalankan_pencerminan),
            ("4. Rotasi", self.jalankan_rotasi),
            ("5. Deteksi Tepi (Robert)", self.jalankan_deteksi_tepi),
            ("6. Noise Reduction", self.jalankan_noise_reduction),
        ]

        for i, (teks, fungsi) in enumerate(daftar_tombol):
            tk.Button(frame_tombol, text=teks, width=22, command=fungsi).grid(
                row=i // 3, column=i % 3, padx=5, pady=5
            )

    # -------------------------------------------------------------
    def saat_mouse_bergerak(self, event):
        x = event.x // self.zoom
        y = event.y // self.zoom
        tinggi, lebar, _ = self.array_asli.shape
        if 0 <= x < lebar and 0 <= y < tinggi:
            r, g, b = self.array_asli[y, x]
            self.label_status.config(text=f"Pixel (x={x}, y={y})   R={r}  G={g}  B={b}")

    # -------------------------------------------------------------
    def simpan_hasil(self, array_hasil, nama_default):
        img_hasil = Image.fromarray(array_hasil, mode="RGB")
        path_simpan = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=nama_default,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Semua file", "*.*")]
        )
        if path_simpan:
            img_hasil.save(path_simpan)
            messagebox.showinfo("Selesai", f"Hasil berhasil disimpan sebagai:\n{path_simpan}")

    # ------------------
    def jalankan_contrast(self):
        G = simpledialog.askfloat("Contrast", "Masukkan nilai G (koefisien penguatan kontras):", initialvalue=1.5)
        if G is None:
            return
        P = simpledialog.askfloat("Contrast", "Masukkan nilai P (pusat pengontrasan, 0-255):", initialvalue=128)
        if P is None:
            return

        hasil = contrast_manual(self.array_asli, G, P)
        self.simpan_hasil(hasil, "hasil_contrast.jpg")

    def jalankan_brightness(self):
        C = simpledialog.askinteger("Brightness", "Masukkan nilai C (boleh negatif untuk menggelapkan):", initialvalue=30)
        if C is None:
            return

        hasil = brightness_manual(self.array_asli, C)
        self.simpan_hasil(hasil, "hasil_brightness.jpg")

    def jalankan_pencerminan(self):
        pilihan = simpledialog.askstring(
            "Pencerminan",
            "Ketik salah satu: horizontal / vertical / kombinasi"
        )
        if not pilihan:
            return
        pilihan = pilihan.strip().lower()
        if pilihan not in ("horizontal", "vertical", "kombinasi"):
            messagebox.showerror("Error", "Pilihan harus: horizontal, vertical, atau kombinasi")
            return

        hasil = pencerminan_manual(self.array_asli, pilihan)
        self.simpan_hasil(hasil, f"hasil_pencerminan_{pilihan}.jpg")

    def jalankan_rotasi(self):
        sudut = simpledialog.askfloat("Rotasi", "Masukkan sudut rotasi (derajat, berlawanan arah jarum jam):", initialvalue=45)
        if sudut is None:
            return

        hasil = rotasi_manual(self.array_asli, sudut)
        self.simpan_hasil(hasil, "hasil_rotasi.jpg")

    def jalankan_deteksi_tepi(self):
        hasil = deteksi_tepi_robert_manual(self.array_asli)
        self.simpan_hasil(hasil, "hasil_deteksi_tepi_robert.jpg")

    def jalankan_noise_reduction(self):
        hasil = noise_reduction_median_manual(self.array_asli)
        self.simpan_hasil(hasil, "hasil_noise_reduction.jpg")


if __name__ == "__main__":
    citra = baca_citra_rgb(PATH_GAMBAR)

    root = tk.Tk()
    app = AplikasiCitra(root, citra)
    root.mainloop()
