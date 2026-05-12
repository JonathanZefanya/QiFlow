# QiFlow

QiFlow adalah aplikasi computer vision realtime bertema cultivation / magic spell casting menggunakan webcam. Fokusnya adalah hand tracking, spell system, efek visual, dan GUI modern.

## Screenshot
Tambahkan screenshot hasil aplikasi di sini setelah run pertama:
- Simpan screenshot ke folder [assets/](assets/) dengan nama `screenshot.png`.
- Update README agar menampilkan gambar.

## Fitur Utama
- Webcam realtime detection + hand tracking MediaPipe
- Deteksi posisi pergelangan, arah telapak, gesture, jarak antar jari, rotasi tangan
- Spell system dengan animasi overlay, efek suara, cooldown, nama skill, glow/particle
- FPS counter, fullscreen, toggle sound, camera selector, demo mode
- Screenshot dan record video
- Gesture training mode dan konfigurasi JSON
- Combo spell detection + aura animation + loading screen
- Debug log panel dan calibration mode

## Struktur Folder
- [main.py](main.py)
- [camera/](camera/)
- [gestures/](gestures/)
- [effects/](effects/)
- [audio/](audio/)
- [ui/](ui/)
- [config/](config/)
- [utils/](utils/)
- [assets/](assets/)
- [training/](training/)
- [recordings/](recordings/)
- [screenshots/](screenshots/)

## Setup
1. Pastikan Python 3.12+ terinstall
2. Buat virtual environment (opsional)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

Catatan: MediaPipe belum menyediakan wheel resmi untuk Python 3.13/3.14 di Windows. Gunakan Python 3.12 jika install MediaPipe gagal.

## Cara Menjalankan
```bash
python main.py
```

## Shortcut Keyboard
- F11: Toggle fullscreen
- R: Start/stop record
- P: Screenshot
- M: Mute/unmute

## Calibration Mode
Klik tombol `Calibrate` lalu tampilkan tangan stabil selama beberapa detik. Sistem akan menyesuaikan skala gesture agar lebih akurat dan menyimpan ke config.

## Custom Spell Config
Gunakan tombol `Save Spell Config` untuk menulis konfigurasi ke `config/spells_custom.json`.
Gunakan `Load Spell Config` untuk memuat ulang cooldown dan combo rule.

## Build Executable (.exe) Windows
Gunakan PyInstaller:

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --add-data "assets;assets" --add-data "config;config" main.py
```

Atau gunakan script build otomatis:

```powershell
scripts\build_release.ps1
```

Release akan muncul di folder [release/](release/).

Hasil build ada di folder `dist/`.

## Konfigurasi
- Edit [config/config.json](config/config.json) untuk mengubah sensitivity, cooldown, dan opsi UI.

## Troubleshooting
- Jika kamera tidak terdeteksi, ganti `camera.index` di config atau gunakan camera selector.
- Jika MediaPipe error di GPU, coba jalankan tanpa aplikasi lain dan kurangi resolusi.
- Jika audio tidak keluar, pastikan `audio.enabled` true dan device audio aktif.
- Jika ada warning dependency opsional, fitur terkait akan dinonaktifkan sampai dependency terpasang.

## Catatan
- Asset sample ada di folder `assets/`. Jika ingin membuat ulang, jalankan `python assets/generate_assets.py`.
- Gesture training mode menyimpan data ke folder `training/`.

## Bonus (opsional)
- Voice activation spell
- AI combo system
- Gesture combo chain
- Multiplayer LAN webcam duel
- Export training data
- Custom spell creator
