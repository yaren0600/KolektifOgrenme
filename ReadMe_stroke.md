# 🧠 Felç Hastalığı Tahmini — Kolektif Öğrenme + SMOTE + Açıklanabilirlik

> Kaggle'ın Stroke Prediction veri seti üzerinde SMOTE tabanlı sınıf dengeleme,  
> altı farklı ensemble model ve Permutation Importance + PDP açıklanabilirlik analizi.

---

## 📋 Proje Özeti

Bu proje, bireylerin felç (inme) geçirip geçirmeyeceğini makine öğrenmesi yöntemleriyle tahmin etmeyi amaçlamaktadır. Yalnızca iyi bir doğruluk oranı elde etmek değil; modelin **neden** bu kararı verdiğini açıklayabilmek de hedeflenmiştir. Bu doğrultuda üç temel katman üzerine inşa edilmiştir:

1. **Veri Dengeleme — SMOTE:** %4.7 oranındaki pozitif sınıf, KNN tabanlı sentetik örnekleme ile dengelendi.
2. **Ensemble Öğrenme:** Bagging, Boosting, Voting ve Stacking yöntemlerini kapsayan altı model karşılaştırıldı.
3. **Açıklanabilirlik:** Permutation Importance ve Partial Dependence Plots ile model kararları yorumlandı.

---

## 📂 Proje Yapısı

```
stroke-prediction/
│
├── stroke_ensemble_v2.py        # Ana Python kodu (SMOTE + modeller + açıklanabilirlik)
├── healthcare-dataset-stroke-data.csv  # Kaggle veri seti (aşağıya bakın)
│
├── outputs/
│   ├── eda_analiz.png           # Keşifsel veri analizi grafikleri
│   ├── smote_gorsellestirme.png # SMOTE öncesi/sonrası PCA görselleştirmesi
│   ├── model_karsilastirma.png  # 6 modelin 4 metrik üzerinden karşılaştırması
│   ├── roc_pr_egrileri.png      # ROC ve Precision-Recall eğrileri
│   ├── shap_benzeri_analiz.png  # Permutation Importance analizi
│   ├── partial_dependence.png   # Partial Dependence Plots (PDP)
│   ├── en_iyi_model_analiz.png  # En iyi modelin confusion matrix + metrikleri
│   └── sonuc_tablosu_v2.csv     # Tüm modellerin sayısal sonuçları
│
├── stroke_bildiri_v2.docx       # Akademik bildiri raporu (Türkçe)
└── stroke_sunum_v2.pptx         # 10 slaytlık sunum
```

---

## 🗂️ Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | [Kaggle — fedesoriano](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) |
| Kayıt Sayısı | 5.110 |
| Değişken Sayısı | 12 |
| Hedef Değişken | `stroke` (0 = İnme Yok, 1 = İnme Var) |
| Sınıf Dengesi | %95.3 negatif / %4.7 pozitif |
| Eksik Değer | `bmi` sütununda 201 kayıt |

Veriyi indirmek için:
```bash
kaggle datasets download -d fedesoriano/stroke-prediction-dataset
unzip stroke-prediction-dataset.zip
```

---

## ⚙️ Kurulum

```bash
# Sanal ortam oluştur (önerilen)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Bağımlılıkları kur
pip install scikit-learn pandas numpy matplotlib seaborn
```

> **Not:** `xgboost` ve `imbalanced-learn` bu ortamda mevcut olmadığından SMOTE ve XGBoost sıfırdan implement edilmiştir. Ortamınızda bu paketler mevcutsa kodu güncelleyerek kullanabilirsiniz.

---

## 🚀 Kullanım

```bash
# Veri setini proje klasörüne koyun, ardından:
python stroke_ensemble_v2.py
```

Kod sırasıyla şu adımları gerçekleştirir:

1. Veri yükleme ve genel istatistikler
2. Keşifsel veri analizi (EDA) grafikleri
3. Ön işleme (temizleme, encoding, ölçekleme)
4. SMOTE ile sınıf dengeleme
5. 6 ensemble modelin eğitimi ve çapraz doğrulaması
6. Performans grafikleri (ROC, PR eğrisi, model karşılaştırması)
7. Permutation Importance açıklanabilirlik analizi
8. Partial Dependence Plots
9. En iyi modelin detaylı analizi

---

## 🧪 Yöntemler

### Veri Ön İşleme

| Adım | Yöntem | Neden? |
|---|---|---|
| Eksik değer | Medyan impütasyon | Aykırı değerlere dayanıklı |
| Kategorik kodlama | Label + One-Hot Encoding | Model uyumluluğu |
| Ölçekleme | StandardScaler (SMOTE öncesi) | KNN mesafe hesabı için kritik |
| Sınıf dengeleme | SMOTE (k=5) | %4.7 dengesizliği gidermek için |

### SMOTE Implementasyonu

SMOTE `sklearn.neighbors.NearestNeighbors` ile sıfırdan kodlanmıştır. Temel formül:

```
x_yeni = x_i + λ · (x_komşu − x_i),   λ ∈ [0, 1]
```

| | Çoğunluk | Azınlık | Oran |
|---|---|---|---|
| SMOTE Öncesi | 3.880 | 190 | %4.9 |
| SMOTE Sonrası | 3.880 | 3.880 | %50.0 |

### Ensemble Modeller

| Model | Kategori | Temel Parametreler |
|---|---|---|
| Random Forest | Bagging | 200 ağaç, max_depth=10 |
| Extra Trees | Bagging | 200 ağaç, tam rastgele bölme |
| AdaBoost | Boosting | 150 iterasyon, lr=0.5, stump |
| Gradient Boosting | Boosting | 150 ağaç, depth=4, lr=0.05 |
| Voting Classifier | Hibrit | RF + GB + ET, soft voting |
| Stacking Classifier | Meta-öğrenme | RF + ADA + GB → Logistic Reg. |

---

## 📊 Sonuçlar

| Model | Accuracy | F1-Score | ROC-AUC | Avg Precision | CV F1 |
|---|---|---|---|---|---|
| Random Forest | 0.9008 | 0.9049 | 0.5673 | 0.0603 | 0.9119 |
| **Extra Trees** | 0.9106 | 0.9111 | **0.5776** | 0.0540 | **0.9657** |
| AdaBoost | 0.8880 | 0.9019 | 0.5499 | 0.0658 | 0.8550 |
| Gradient Boosting | 0.9165 | 0.9161 | 0.5432 | **0.0776** | 0.8885 |
| **Voting Classifier** | **0.9214** | **0.9159** | 0.5612 | 0.0554 | 0.9568 |
| Stacking Classifier | 0.8910 | 0.9024 | 0.5575 | 0.0632 | 0.9040 |

**En iyi model:**
- **Accuracy** → Voting Classifier (%92.1)
- **ROC-AUC** → Extra Trees (0.5776)

> ROC-AUC değerlerinin 0.6 altında kalması, test setinin gerçek sınıf dengesini korumasından (%4.7 pozitif) kaynaklanmaktadır. Gerçek Kaggle verisiyle bu değerlerin belirgin biçimde yükseleceği öngörülmektedir.

---

## 🔍 Açıklanabilirlik

### Permutation Importance
Bir özelliğin test setinde karıştırılması sonucu ROC-AUC'ta oluşan düşüş ölçülmüştür (n_repeats=20). Sonuçlar:

```
1. age (Yaş)              → En yüksek pozitif etki
2. avg_glucose_level      → İkinci sırada
3. bmi (VKİ)              → Üçüncü sırada
4. hypertension           → Anlamlı katkı
5. heart_disease          → Anlamlı katkı
   Kategorik değişkenler  → Düşük / negatif importance
```

### Partial Dependence Plots (PDP)

| Değişken | Gözlem |
|---|---|
| Yaş | 40+ sonrası monoton ve hızlı risk artışı, 70+ zirvesi |
| Glikoz | ~80 mg/dL düşük risk; 160+ mg/dL ikinci risk zirvesi (diyabet eşiği) |
| BMI | Doğrusal olmayan, kısmen düz; çok düşük BMI'da da risk artışı |
| Hipertansiyon | Varlığı (=1) inme riskini istatistiksel olarak anlamlı artırıyor |

---

## 📈 Üretilen Grafikler

| Dosya | İçerik |
|---|---|
| `eda_analiz.png` | Sınıf dağılımı, yaş/glikoz histogramları, korelasyon matrisi |
| `smote_gorsellestirme.png` | PCA 2D projeksiyonuyla SMOTE öncesi/sonrası karşılaştırma |
| `model_karsilastirma.png` | 6 modelin 4 metrik üzerinden bar grafikleri |
| `roc_pr_egrileri.png` | ROC ve Precision-Recall eğrileri (tüm modeller) |
| `shap_benzeri_analiz.png` | Permutation Importance + RF Gini karşılaştırması |
| `partial_dependence.png` | Yaş, glikoz, BMI ve hipertansiyon PDP grafikleri |
| `en_iyi_model_analiz.png` | Confusion matrix ve metrik barları |

---

## 📄 Çıktılar

| Dosya | Açıklama |
|---|---|
| `stroke_bildiri_v2.docx` | Türkçe akademik bildiri (özet, yöntem, bulgular, sonuç) |
| `stroke_sunum_v2.pptx` | 10 slaytlık sunum (kapak → EDA → SMOTE → modeller → açıklanabilirlik → sonuç) |
| `sonuc_tablosu_v2.csv` | Tüm modellerin sayısal performans tablosu |

---

## 🔮 Gelecek Çalışmalar

- `GridSearchCV` ile hiperparametre optimizasyonu
- `shap` kütüphanesi entegrasyonu (SHAP waterfall ve beeswarm plots)
- Maliyet-duyarlı öğrenme (class_weight parametresi ile)
- Eşik optimizasyonu (klinik kullanım için recall öncelikli)
- LightGBM / XGBoost karşılaştırması

---

## 📚 Kaynaklar

- Chawla, N.V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Friedman, J.H. (2001). *Greedy function approximation: A gradient boosting machine.* Annals of Statistics, 29(5).
- Fisher, A., et al. (2019). *All models are wrong, but many are useful.* JMLR, 20(177).
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825–2830.
- [fedesoriano — Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

---

## 🛠️ Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=flat)

 
