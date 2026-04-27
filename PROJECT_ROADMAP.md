# KORGAN Project Roadmap

Bu dosya, projede nerede oldugumuzu, neleri tamamladigimizi ve sirada nelerin oldugunu tek yerde tutmak icin vardir.

Benimle ileride `yapilacaklar listesi` diye konustugunda bu dosya referans alinacak. O durumda sana su uc baslikla net ozet verecegim:

- `Ne yaptik`
- `Nerede kaldik`
- `Ne yapmamiz gerek`

## Guncel Durum

Tarih: `2026-04-27`
Durum: `Profesyonellesme asamasi basladi, runtime altyapisi guclendirildi`

## Tamamlananlar

- [x] README profesyonel ve teknik sekilde genisletildi.
- [x] Runtime config yapisi eklendi.
- [x] CLI arguman destegi eklendi.
- [x] ByteTrack mantigina yakin iki asamali tracking akisi eklendi.
- [x] Worker threshold parametreleri disaridan yonetilebilir hale getirildi.
- [x] Worker hata verdiginde sinirli otomatik restart eklendi.
- [x] Kamera tekrar acma denemesi eklendi.
- [x] JSONL telemetry log yapisi eklendi.
- [x] Opsiyonel video recording destegi eklendi.
- [x] Temel config ve manifest testleri eklendi.
- [x] GitHub Actions CI iskeleti eklendi.
- [x] `run_opencv.bat` arguman iletebilir hale getirildi.

## Nerede Kaldik

Su an kod tabani prototipten daha duzenli bir runtime yapisina gecti, fakat saha kullanimi icin henuz son nokta degil.

Aktif olarak geldigimiz nokta:

- masaustu Windows gelistirme akisi calisiyor
- config ile runtime ayarlanabiliyor
- tracker daha kararlı hale getirildi
- log ve recording altyapisi var
- test ve CI temeli atildi

Ama henuz yapilmayan kritik konular:

- gercek kamera ile saha testi ve tuning
- DeepSORT seviyesinde appearance/re-id tabanli takip
- kamera kalibrasyonu
- olay bazli kayit mantigi
- Jetson oncesi platform bagimliliklarini azaltma

## Hemen Yapilacaklar

- [ ] Mevcut degisiklikleri commit ve push etmek
- [ ] Gercek kamera ile canli test yapmak
- [ ] `config/runtime_config.json` threshold tuning yapmak
- [ ] JSONL loglardan performans ve takip davranisini incelemek
- [ ] Recording modunda ornek oturumlar toplamak

## Orta Vade

- [ ] DeepSORT veya benzeri appearance embedding tabanli takip eklemek
- [ ] Kamera kalibrasyonu eklemek
- [ ] Olay bazli kayit sistemi eklemek
- [ ] Log seviyeleri ve daha guclu health-check yapisi eklemek
- [ ] CI tarafina daha fazla dogrulama eklemek

## Jetson Oncesi Zorunlular

- [ ] Windows'a bagli kisimlari ayristirmak
- [ ] Process ve worker yapisini Linux/Jetson dostu hale getirmek
- [ ] Encode/decode maliyetini azaltmak
- [ ] Runtime pipeline icin performans baseline cikarmak
- [ ] Model paket sozlesmesini sabitlemek

## Jetson Gecis Plani

- [ ] Jetson uzerinde mevcut modelin temel calismasini dogrulamak
- [ ] ONNX Runtime CPU yerine TensorRT hattina gecmek
- [ ] GPU odakli veri akisina gecmek
- [ ] Jetson uzerinde FPS ve latency olcumleri almak
- [ ] Saha testi ile masaustu ve Jetson sonuclarini karsilastirmak

## Sonraki Onerilen Sira

1. Mevcut degisiklikleri commit et
2. Gercek kamera testi yap
3. Threshold tuning yap
4. DeepSORT veya re-id katmanini planla
5. Jetson oncesi platform temizliklerini yap
6. TensorRT gecisine basla

## Kullanim Notu

Bu dosya sabit bir belge degil. Her buyuk adimdan sonra guncellenmeli.

Benimle tekrar calisirken:

- `yapilacaklar listesi` dersen bu dosyayi ozetleyecegim
- `nerede kaldik` dersen son durumu buradan anlatacagim
- `bitenleri guncelle` dersen checklist'i ilerletecegim
- `siradaki isi yap` dersen en yakin mantikli adimdan devam edecegim
