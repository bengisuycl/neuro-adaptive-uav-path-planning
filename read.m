# Proje Devir Özeti

Bu dosya, `wcci_conference_project` projesinin mevcut durumunu yeni bir
sohbete veya geliştiriciye aktarmak için hazırlanmıştır.

## 1. Projenin Amacı

Proje, DEM tabanlı arazi ve statik/dinamik tehditler altında sabit kanatlı
bir hava aracı için uçulabilir, tehdit farkındalıklı ve görev koşullarına
uygun rota üretmeyi amaçlamaktadır.

Genel mimari:

1. Perception:
   - DEM arazi modeli
   - SAM, radar, EW ve dinamik tehdit modelleri
   - DNN-TRE neural tactical risk estimator
2. Decision:
   - Neuro-Adaptive advisor
   - Capability, Opportunity ve Pressure skorları
   - Fused tactical index ile planner seçimi
3. Action:
   - RL-Pilot
   - K-GNP
   - T-GnP
   - Ortak feasibility engine ve uçuş simülasyonu

Ana giriş noktaları:

- `main.py`: toplu benchmark ve temel çalışma akışı
- `gui_app.py`: masaüstü kontrol arayüzü
- `gui_runner.py`: GUI üzerinden benchmark yürütme

## 2. Temel Planner Yapısı

### RL-Pilot

- Dueling DQN tabanlı reaktif lokal policy'dir.
- Sekiz durum girdisi kullanır:
  `distance/guide geometry`, `heading error`, placeholder altitude error ve
  beş yerel yönsel risk gözlemi.
- Beş ayrık heading-change action üretir:
  `-30`, `-15`, `0`, `+15`, `+30` derece.
- Düşük riskli, kısa menzilli ve statik tehditli görevler için seçilir.

### K-GNP

- Heading-augmented Dijkstra tabanlı deterministik graph-search planner'dır.
- Durum temsili `(i, j, k_psi)` biçimindedir.
- Grid çözünürlüğü `200 m`, heading çözünürlüğü `10 deg` olarak kullanılır.
- Motion primitive'ler minimum dönüş yarıçapı ve heading değişimi açısından
  arama sırasında kontrol edilir.
- Orta düzey taktik karmaşıklık için fallback planner'dır.

### T-GnP

- K-GNP'nin heading-aware graph-search yapısını temel alır.
- Adaptive segment speed, two-stage turn feasibility, heading-change cost,
  time-integrated threat cost ve low-speed stability penalty ekler.
- DNN-TRE risk maliyeti isteğe bağlı olarak primitive değerlendirmesine
  dahil edilebilir.
- Yüksek riskli veya dinamik tehditli görevlerde seçilir.

## 3. Güncel Risk Fusion Formülleri

Kodun referans dosyası:

- `ai_modules/rl_advisor.py`

Normalizasyonlar:

```text
peak_norm      = rho_peak / (rho_peak + 1.1)
mean_norm      = rho_mean / (rho_mean + 0.35)
corridor_ratio = rho_corr / max(rho_peak, 1e-6)
corridor_norm  = rho_corr / (rho_corr + 0.28)
dynamic_norm   = min(n_dyn / 3, 1)
distance_norm  = min(d_mission / 120000, 1)
threat_density = min(n_threat / 5, 1)
urgency_norm   = 1 if urgency == HIGH else 0
```

Ara skorlar:

```text
C = 0.58 * peak_norm
  + 0.28 * mean_norm
  + 0.14 * threat_density

O = 0.50 * corridor_ratio
  + 0.30 * f_high
  + 0.20 * corridor_norm

P = 0.38 * dynamic_norm
  + 0.22 * urgency_norm
  + 0.25 * distance_norm
  + 0.15 * threat_density

F = 0.34 * C + 0.41 * O + 0.25 * P
```

Bu katsayılar teorik optimum olarak sunulmamalıdır. Bunlar corridor exposure
bilgisini önceliklendirmek üzere deneysel/design-tuned olarak seçilmiş,
yorumlanabilir planner allocation katsayılarıdır.

## 4. Güncel Planner Seçim Politikası

Kodla uyumlu `if / elif / else` sırası:

```text
IF n_dyn >= 2 AND (corridor_ratio >= 0.38 OR corridor_norm >= 0.42)
    -> T-GnP

ELSE IF n_dyn == 0
    AND n_threat <= 3
    AND d_mission < 45000 m
    AND F < 0.27
    AND corridor_norm < 0.22
    AND f_high < 0.10
    -> RL-Pilot

ELSE IF F >= 0.56
    OR corridor_ratio >= 0.50
    OR corridor_norm >= 0.46
    OR f_high >= 0.28
    OR (n_dyn >= 1 AND corridor_norm >= 0.34)
    -> T-GnP

ELSE
    -> K-GNP
```

Eski sunumlarda görülen `F < 0.33`, `F >= 0.62`, `xi >= 0.55` gibi eşikler
güncel kodla uyumlu değildir ve kullanılmamalıdır.

## 5. Ortak Model ve Benchmark Ayarları

Projede kullanılan başlıca ortak değerler:

```text
Cruise speed             = 220 m/s
Minimum speed            = 120 m/s
Maximum speed            = 280 m/s
Maximum bank angle       = 60 deg
Simulation time step     = 0.25 s
Segment check resolution = 50 m
Goal tolerance radius    = 3000 m
Threat weight            = 20
Threat cost scale        = 5000
Monte Carlo runs         = 30
```

DEM:

```text
CRS    = EPSG:32636 (WGS 84 / UTM zone 36N)
Region = yaklaşık Orta Anadolu, Türkiye
Safe altitude example = terrain maximum + 200 m
```

## 6. Yapılan Kod Değişiklikleri

### Benchmark eksen yazıları

`visualization/benchmark_outputs.py` içinde:

- X ekseni etiketleri büyütüldü.
- Etiket dönüşü `30 deg` yapıldı.
- X etiketleri sağa hizalandı.
- Y ekseni etiket boyutu artırıldı.

Bu değişiklik, çok algoritmalı benchmark grafiklerinin sunumda okunabilir
olması için yapıldı.

### Trajectory karşılaştırma yaklaşımı

Trajectory karşılaştırma figürlerinde bütün algoritmaların aynı kalınlık ve
kontrastla çizilmemesi önerildi:

- A-Star, Dijkstra, RRT-Star ve PSO: ince, kesikli, yarı saydam
- K-GNP, T-GnP, RL-Pilot ve Neuro-Adaptive: kalın, solid, yüksek kontrast

Sunum için en temiz çözüm:

1. Ana slaytta yalnızca önerilen planner'ları göstermek.
2. Classical/stochastic baseline'ları tabloda göstermek.
3. Tüm algoritmalar zorunluysa baseline çizgilerini görsel olarak geri plana
   almak.

Not: `visualization/plot_riskmap.py` güncel durumu tekrar kontrol edilmelidir;
trajectory styling değişikliğinin çalışma ağacında kalıcı olup olmadığı
doğrulanmadan figür yeniden üretilmemelidir.

### Silinen kullanılmayan dosyalar

Referans ve import taraması sonucunda aşağıdaki eski/demo dosyaları silindi:

- `environment/dem.py`
- `environment/check_dem.py`
- `visualization/ex.py`
- `visualization/visualize_dem_3d.py`
- `visualization/plot_tactical_3d.py`
- `visualization/ieee_fig_template.py`

Aktif `environment/terrain.py` ve `visualization/plot3d.py` korunmuştur.

## 7. Sunum ve Akademik Metin Durumu

Hazırlanan içerikler:

- Motivation and problem definition
- Environment modeling
- Common feasibility and benchmark conditions
- Classical/stochastic planner baselines
- K-GNP iki/üç bölümlü anlatım
- T-GnP iki bölümlü anlatım
- RL-Pilot policy ve Dueling DQN mimarisi
- DNN-TRE risk estimator
- Neuro-Adaptive risk fusion ve planner selection
- Benchmark, ablation, conclusion ve future work içerikleri

Akademik yazımda dikkat edilmesi gerekenler:

- "Theoretically optimal" veya "guarantees optimality" denmemeli.
- Fused-risk ağırlıkları empirical/design-tuned olarak tanımlanmalı.
- "Tactically optimal" yerine "tactically favorable" veya
  "selected according to multiple tactical objectives" kullanılmalı.
- DNN-TRE için kanıtlanmadıkça kesin `O(1)` iddiasından kaçınılmalı;
  "fixed-cost neural inference per queried point" daha güvenlidir.
- Framework, yeni genel amaçlı planning teorisi olarak değil; fixed-wing
  feasibility, tactical risk perception ve adaptive planner allocation
  entegrasyonu olarak konumlandırılmalıdır.

## 8. Mevcut Teknik Riskler

### Python ortamı

Projedeki `.venv`, erişilemeyen Microsoft Store Python yoluna bağlıdır:

```text
C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\...
```

Bu nedenle `compileall` doğrulaması çalıştırılamamıştır. Ortam yeniden
oluşturulmadan tam test sonucu verilmemelidir.

### Git çalışma ağacı

Git deposunun kökü `wcci_conference_project` üstündeki
`thesis_project` dizinidir. Çalışma ağacı çok sayıda:

- değiştirilmiş,
- takip edilmeyen,
- çıktı/model/veri dosyası

içermektedir. Kullanıcı değişiklikleri geri alınmamalıdır. Temizlik veya commit
öncesinde `.gitignore`, dosya sahipliği ve gerçek proje sınırı ayrıca
netleştirilmelidir.

### Test eksikliği

Projede görünür bir otomatik test paketi bulunmamaktadır. Özellikle aşağıdaki
alanlar için test eklenmesi önerilir:

- risk normalizasyonları,
- fused tactical index,
- planner selection threshold sınırları,
- terrain/threat feasibility,
- minimum-turn-radius kontrolü,
- trajectory post-processing.

## 9. Önerilen Sonraki Adımlar

1. Geçerli bir Python sanal ortamı oluştur:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Kullanılan bağımlılıkları belirleyip `requirements.txt` oluştur.
3. Tüm Python dosyalarında `compileall` ve import smoke test çalıştır.
4. `rl_advisor.py` için threshold boundary unit testleri yaz.
5. `plot_riskmap.py` trajectory stilini doğrula ve S1--S4 figürlerini yeniden
   üret.
6. Sunum ve tezdeki bütün risk-fusion formüllerini bu dosyadaki güncel kod
   formülleriyle karşılaştır.
7. Monte Carlo sonuçlarına confidence interval veya standard deviation ekle.
8. Git kapsamını temizle; model, DEM, output, cache ve geçici sunum dosyaları
   için `.gitignore` düzenle.
9. Temizlikten sonra kontrollü bir commit oluştur.

## 10. Yeni Sohbet İçin Kısa Başlangıç Metni

```text
Bu proje, DEM tabanlı arazi ve tehdit ortamlarında sabit kanatlı hava aracı
için Neuro-Adaptive perception-decision-action mission planning framework'üdür.
Ana planner'lar RL-Pilot, K-GNP ve T-GnP'dir. Güncel risk fusion ve planner
selection formülleri read.m dosyasında verilmiştir. Öncelikle mevcut kodu ve
read.m dosyasını incele; kullanıcı değişikliklerini geri alma. Sonraki hedef,
Python ortamını düzeltmek, seçim eşikleri için test yazmak ve trajectory
comparison figürlerini okunabilir biçimde yeniden üretmektir.
```
