# KORGAN Git / GitHub Duzeni

Bu projede hedefimiz `main` branch'ini her zaman calisan ve geri donulebilir bir taban olarak tutmak.

## Branch duzeni

- `main`
  - Her zaman en stabil surum.
- `feature/<ozellik-adi>`
  - Yeni fikirler, denemeler ve eklemeler.
- `fix/<sorun-adi>`
  - Hata duzeltmeleri.
- `jetson/<konu>`
  - Jetson'a ozel denemeler ve entegrasyonlar.

## Commit baslik formati

Kisa ve net commit basliklari kullan:

- `feat: terminal telemetry eklendi`
- `fix: tracking durum metni duzeltildi`
- `refactor: takip etiketleri sadeletildi`
- `perf: worker gecikme olcumu iyilestirildi`
- `docs: git workflow dosyasi eklendi`

## Onerilen calisma akisi

Yeni bir ozellik eklemeden once:

```powershell
git checkout main
git status
git checkout -b feature/ozellik-adi
```

Calisma bittiginde:

```powershell
git add .
git commit -m "feat: ozellik aciklamasi"
```

Branch'i GitHub'a gondermek icin:

```powershell
git push -u origin feature/ozellik-adi
```

## Geri donus senaryolari

Son degisiklik hosuna gitmezse:

```powershell
git checkout main
```

Sadece branch'i silmek istersen:

```powershell
git branch -D feature/ozellik-adi
```

Eski commit'e bakmak istersen:

```powershell
git log --oneline --graph --decorate
git checkout <commit-id>
```

## Onerilen ilk uzak depo akisi

GitHub'da bos repo actiktan sonra:

```powershell
git remote add origin <github-repo-url>
git add .
git commit -m "chore: ilk proje kaydi"
git push -u origin main
```

## Notlar

- Buyuk model dosyalari (`.onnx`, `.pt`) varsayilan olarak ignore edilir.
- Gerekirse bunlar icin ileride `Git LFS` kullanilabilir.
- `main` branch'e dogrudan deneysel degisiklik yapma.
- Her buyuk adimi ayri commit ile kaydetmek geri donusu cok kolaylastirir.
- Bu repoda `post-commit` hook kullanilarak her yeni commit'ten sonra otomatik `git push` denenir.
