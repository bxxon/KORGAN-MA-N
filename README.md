# KORGAN

KORGAN, Windows uzerinde calisan kamera tabanli bir nesne tespit ve takip uygulamasidir. Sistem, goruntu alma ve cizim tarafinda C++/OpenCV kullanir; ONNX model cikarimini ise ayri bir Python worker surecine vererek ana uygulamayi daha temiz ve kontrol edilebilir tutar.

Bu repo, projeyi tekrar derlemek, calistirmak, gelistirmek ve GitHub uzerinden duzenli sekilde surdurmek icin gereken temel dosyalari ve aciklamalari bir arada tutar.

## Projenin yaptigi is

- Kameradan goruntu alir.
- Her frame'i Python tabanli ONNX Runtime worker surecine yollar.
- YOLOv8 ONNX modeli ile nesne tespiti yapar.
- Sonuclari tekrar C++ tarafina alir.
- Kalman tabanli basit bir takip mekanizmasi ile nesneleri track eder.
- Oncelikli hedef secerek ekranda odak durumunu gosterir.
- Telemetri bilgisini hem terminalde hem `telemetri_log.txt` dosyasinda yazar.

Varsayilan takip edilen siniflar:

- `0` -> `person`
- `32` -> `sports ball`
- `67` -> `cell phone`

Referans sinif:

- `39` -> `bottle`

Bu ayarlar [main.cpp](/C:/Users/askan/Desktop/KORGAN/main.cpp:16) tarafinda tanimlidir.

## Mimari ozet

Projede iki ana katman vardir:

1. `main.cpp`
   Windows, OpenCV, kamera akisi, cizim, takip, telemetry ve Python worker yonetimi burada yapilir.
2. `scripts/ort_worker.py`
   ONNX Runtime ile modeli yukler, frame preprocess eder, inference yapar ve NMS sonrasi JSON benzeri sonuc dondurur.

Calisma akisi kisaca soyledir:

1. C++ uygulamasi model paketini (`deliverables/onnxruntime_cpu_package`) yukler.
2. `scripts/ort_worker.py` ayri bir process olarak baslatilir.
3. Ana uygulama kameradan frame alir.
4. Frame JPEG olarak worker'a pipe uzerinden aktarilir.
5. Worker, ONNX modelini calistirir.
6. Tespitler geri doner.
7. C++ tarafinda tracking, hedef secimi ve overlay cizimi yapilir.
8. Sonuc pencereye ve log dosyasina yazilir.

## Klasor yapisi

```text
KORGAN/
|- main.cpp
|- scripts/
|  |- build_opencv.bat
|  |- run_opencv.bat
|  `- ort_worker.py
|- deliverables/
|  |- model.onnx
|  |- labels.txt
|  |- model-manifest.json
|  `- onnxruntime_cpu_package/
|     |- model.onnx
|     |- labels.txt
|     `- model-manifest.json
|- GIT_WORKFLOW.md
`- .gitignore
```

Dosyalarin ne oldugu:

- [main.cpp](/C:/Users/askan/Desktop/KORGAN/main.cpp:1): Ana uygulama. Kamera, takip, cizim, telemetry, worker haberlesmesi burada.
- [scripts/ort_worker.py](/C:/Users/askan/Desktop/KORGAN/scripts/ort_worker.py:1): ONNX Runtime worker. Model yukleme, preprocess, inference ve NMS burada.
- [scripts/build_opencv.bat](/C:/Users/askan/Desktop/KORGAN/scripts/build_opencv.bat:1): MSVC + OpenCV ile `main.cpp` dosyasini `hss_sistem.exe` olarak derler.
- [scripts/run_opencv.bat](/C:/Users/askan/Desktop/KORGAN/scripts/run_opencv.bat:1): Gerekli PATH ayarlarini yapip uygulamayi baslatir.
- [deliverables/model-manifest.json](/C:/Users/askan/Desktop/KORGAN/deliverables/model-manifest.json:1): Model paketi tanimi.
- [deliverables/labels.txt](/C:/Users/askan/Desktop/KORGAN/deliverables/labels.txt:1): Sinif isimleri.
- [deliverables/onnxruntime_cpu_package/model-manifest.json](/C:/Users/askan/Desktop/KORGAN/deliverables/onnxruntime_cpu_package/model-manifest.json:1): Uygulamanin varsayilan olarak kullandigi paket manifesti.
- [GIT_WORKFLOW.md](/C:/Users/askan/Desktop/KORGAN/GIT_WORKFLOW.md:1): Branch, commit ve GitHub calisma duzeni.

## Gereksinimler

Bu proje mevcut haliyle Windows odakli yazilmistir.

- Windows 10 veya Windows 11
- Visual Studio 2022 Build Tools
- MSVC toolchain
- Windows SDK
- OpenCV
- Python 3.12
- Python paketleri:
  - `opencv-python`
  - `numpy`
  - `onnxruntime`
- Calisan bir kamera

## Ortam beklentileri

Mevcut script'ler asagidaki dizin varsayimlariyla yazilmistir:

- OpenCV DLL ve kutuphaneleri: `C:\opencv\build\...`
- Python: `C:\Users\askan\AppData\Local\Programs\Python\Python312\python.exe`

Eger Python farkli bir yerdeyse su ortam degiskeni kullanilabilir:

```powershell
$env:KORGAN_PYTHON="C:\tam\yol\python.exe"
```

Kod, bu degiskeni once kontrol eder. Ayrica `python.exe` PATH uzerindeyse onu da kullanabilir.

## Python bagimliliklari

Python tarafini bir kere hazirlamak icin:

```powershell
pip install opencv-python numpy onnxruntime
```

## Derleme

Projeyi derlemek icin:

```powershell
scripts\build_opencv.bat
```

Bu script varsayilan olarak:

- `main.cpp` dosyasini derler
- OpenCV include ve lib yollarini kullanir
- cikti olarak repo kokunde `hss_sistem.exe` uretir

Eger farkli bir kaynak dosya vermek istersen:

```powershell
scripts\build_opencv.bat main.cpp
```

## Calistirma

En kolay calistirma:

```powershell
scripts\run_opencv.bat
```

Bu script:

- OpenCV DLL yolunu PATH'e ekler
- Gerekirse `KORGAN_PYTHON` degiskenini set eder
- `hss_sistem.exe` dosyasini baslatir

Alternatif olarak exe'yi dogrudan da calistirabilirsin:

```powershell
.\hss_sistem.exe
```

Varsayilan paket yolu:

```text
deliverables/onnxruntime_cpu_package
```

Farkli bir paket klasoru ile calistirmak icin:

```powershell
.\hss_sistem.exe deliverables\onnxruntime_cpu_package
```

Uygulamadan cikmak icin pencere aktifken `ESC` tusuna bas.

## Model paketi mantigi

Uygulama dogrudan tek bir `.onnx` dosyasina degil, bir paket klasorune baglanir. Bu paket icinde en az su dosyalar bulunmalidir:

- `model.onnx`
- `labels.txt`
- `model-manifest.json`

Manifest dosyasi, modelin hangi dosya adiyla yuklenecegini ve input/output beklentilerini tanimlar. Bu sayede kod, paket formatini okuyarak daha kontrollu sekilde calisir.

Not:

- `.gitignore` dosyasi geregi `*.onnx`, `*.pt` ve bazi buyuk cikti dosyalari Git tarafinda ignore edilir.
- Repo'yu baska bir makinede klonlarsan model dosyalarini manuel olarak tekrar koyman gerekebilir.
- Bunun icin `deliverables/onnxruntime_cpu_package/` klasoru uygulamanin bekledigi temel yerdir.

## Ekranda ne goruyorsun

Calisma sirasinda sistem:

- takip edilen hedeflerin kutularini cizer
- aktif odak hedefini farkli renkle gosterir
- hedef merkezi icin `X`, `Y` bilgisi uretir
- kutu genisligine gore yaklasik `Z` mesafesi hesaplar
- FPS ve gecikme bilgilerini ekranda/terminalde verir

Terminal satirinda gorulen alanlar genel olarak sunlari ifade eder:

- `DURUM`: Takip durumu (`BEKLENIYOR`, `IZLENIYOR`, `BELIRSIZ`)
- `X`, `Y`: Hedef merkezi
- `Z`: Yaklasik mesafe
- `T`: Track ID
- `CONF`: Guven skoru
- `HEALTH`: Takip sagligi
- `RT`, `DEC`, `PRE`, `INF`, `POST`, `TRK`, `DRW`, `FRM`: pipeline gecikme metrikleri

## Telemetri log dosyasi

Loglar su dosyaya append edilir:

```text
telemetri_log.txt
```

Bu dosyada ozellikle su bilgiler yer alir:

- zaman damgasi
- hedef koordinatlari
- yaklasik mesafe
- sinif ve etiket
- confidence
- track ID
- worker ve inference gecikmeleri

## Degistirilebilecek temel ayarlar

Asagidaki sabitler [main.cpp](/C:/Users/askan/Desktop/KORGAN/main.cpp:16) icinde kolayca degistirilebilir:

- `IZLENECEK_SINIFLAR`
- `REFERANS_SINIFLAR`
- `HEDEF_GERCEK_GENISLIK_CM`
- `KAMERA_ODAK_UZAKLIGI`
- `CONFIDENCE_THRESHOLD`
- `NMS_THRESHOLD`
- `TRACK_MATCH_DISTANCE_PX`
- `TRACK_MAX_MISSED_FRAMES`

Bu ayarlar sirasiyla:

- hangi siniflarin takip edilecegini
- hangi siniflarin referans sayilacagini
- yaklasik mesafe hesabini
- tespit filtrelemesini
- tracking toleransini

etkiler.

## Sik karsilasilan sorunlar

### 1. `Python bulunamadi`

Muhtemel nedenler:

- Python kurulu degil
- Python PATH'te degil
- `KORGAN_PYTHON` ayarlanmadi

Cozum:

```powershell
$env:KORGAN_PYTHON="C:\tam\yol\python.exe"
scripts\run_opencv.bat
```

### 2. `model-manifest.json acilamadi` veya `Model dosyasi bulunamadi`

Muhtemel neden:

- paket klasoru eksik
- `model.onnx` GitHub clone sonrasi manuel geri konmadi

Cozum:

- `deliverables\onnxruntime_cpu_package\model.onnx` dosyasinin varligini kontrol et
- paket icinde `labels.txt` ve `model-manifest.json` oldugunu dogrula

### 3. `Kamera acilamadi`

Muhtemel nedenler:

- kamera baska uygulama tarafindan kullaniliyor
- cihazda kamera yok
- Windows kamera izinleri kapali

### 4. OpenCV DLL hatasi

Muhtemel neden:

- `C:\opencv\build\x64\vc16\bin` yolu sistemde yok

Cozum:

- `scripts\run_opencv.bat` ile calistir
- OpenCV kurulum yolunun script ile uyustugunu kontrol et

## GitHub uzerinde ekipce nasil calisacagiz

Bu repo icin temel Git duzeni [GIT_WORKFLOW.md](/C:/Users/askan/Desktop/KORGAN/GIT_WORKFLOW.md:1) icinde ayrica bulunuyor. Kisa ozet:

- `main` her zaman calisan ve daha stabil dal olmali
- yeni isler icin `feature/...`
- hata duzeltmeleri icin `fix/...`
- Jetson ozel denemeler icin `jetson/...`

Onerilen akis:

```powershell
git checkout main
git pull
git checkout -b feature/yeni-degisiklik
```

Calisma bitince:

```powershell
git add .
git commit -m "feat: kisa aciklama"
git push -u origin feature/yeni-degisiklik
```

## Bu repoda bir degisiklik yapacaksan once buralara bak

Hangi konuya gore hangi dosyaya bakilacagi:

- kamera, overlay, telemetry, tracking: `main.cpp`
- model yukleme ve inference: `scripts/ort_worker.py`
- derleme sorunu: `scripts/build_opencv.bat`
- calistirma ortami/PATH sorunu: `scripts/run_opencv.bat`
- model paket yapisi: `deliverables/onnxruntime_cpu_package/`
- git akisi: `GIT_WORKFLOW.md`

## Son not

Bu README'nin amaci sadece projeyi tanitmak degil; yeni gelen birinin repo'yu klonlayip "bu ne, nereden baslayacagim?" sorusuna tek dosyada cevap verebilmektir. Projede yeni klasorler, yeni modeller veya farkli hedef siniflari eklenirse bu dosyanin da ayni anda guncellenmesi iyi bir ekip aliskanligi olur.
