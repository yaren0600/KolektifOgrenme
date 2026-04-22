🧠 Parkinson's Disease Classification via Ensemble Learning
Bu proje, Parkinson hastalarından alınan ses kayıtlarından elde edilen öznitelikleri kullanarak, kolektif öğrenme (Ensemble Learning) yöntemleriyle teşhis tahmini yapmaktadır. Projenin ana odak noktası, yüksek boyutlu verilerde (750+ özellik) veri önişleme ve özellik seçiminin model başarısı ve açıklanabilirliği üzerindeki etkisini incelemektir.

📌 Proje Motivasyonu ve Mühendislik Yaklaşımı
Bu veri setinde karşılaşılan en büyük zorluk "Boyut Laneti" (Curse of Dimensionality) problemidir. Veri setinde 756 örneklem varken 750'den fazla özellik bulunmaktadır. Bu durum modelin gürültüyü (noise) öğrenmesine ve ezber yapmasına (overfitting) neden olur.

🛠 Uygulanan Metodoloji
1. Veri Önişleme (Data Preprocessing)
ID Temizliği: id sütunu, modelin hastayı özelliklerinden değil sırasından tanımasını engellemek için kaldırılmıştır.

Ölçeklendirme (Standardization): Ses öznitelikleri (TQWT, MFCC, PPE) çok farklı sayısal aralıklardadır. Gradient tabanlı modellerin (Boosting gibi) kararlı çalışması için StandardScaler ile tüm veriler ortalaması 0, standart sapması 1 olacak şekilde normalize edilmiştir.

2. Özellik Seçimi (Feature Selection)
Neden Yapıldı? Gereksiz ve birbirini tekrar eden (redundant) özellikleri eleyerek modelin genelleme yeteneğini artırmak amaçlanmıştır.

Yöntem: Kolektif modellerin (Random Forest / Gradient Boosting) sunduğu "Feature Importance" skorları kullanılarak en etkili 100 özellik seçilmiştir.

Etki Analizi: Yapılan testlerde, özellik sayısı %86 oranında azaltılmasına rağmen doğruluk (accuracy) kaybı yaşanmamış, aksine modelin gürültüden arınması sağlanmıştır.

3. Kolektif Öğrenme (Ensemble Learning)
Projede Bagging (Random Forest) ve Boosting (Gradient Boosting) yaklaşımları kıyaslanmıştır:

Gradient Boosting (GBM): Hatalardan ders çıkaran ardışıl yapısı sayesinde Parkinson teşhisindeki ince detayları yakalamada en yüksek başarıyı (%86.8) sergilemiştir.

📊 Sonuçlar ve Değerlendirme
<img width="832" height="249" alt="image" src="https://github.com/user-attachments/assets/e347bd43-9e52-4036-9ec8-b939571ad42f" />

Hata Matrisi (Confusion Matrix)
Model, özellikle sağlık alanında hayati önem taşıyan Duyarlılık (Recall) değerinde yüksek başarı göstermiştir. Yani gerçek Parkinson vakalarını kaçırma oranı minimize edilmiştir.

Model Açıklanabilirliği (SHAP Analysis)
Modelin kararlarını anlamak için SHAP kullanılmıştır. Bu analiz, özellikle TQWT (Tunable Q-Wavelet Transform) tabanlı özelliklerin teşhis koymada en belirleyici faktörler olduğunu ortaya koymuştur.

💻 Kullanım (How to Run)
Gereksinimleri Yükleyin:

```Bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

🗂 Veri Seti Hakkında
Veri seti, 188 hastadan alınan 756 ses kaydı içermektedir. Veriler; temel ses özellikleri, MFCC'ler, Wavelet özellikleri ve TQWT parametrelerinden oluşmaktadır.

Bu çalışma, Bilgisayar Mühendisliği Yüksek Lisans düzeyi veri bilimi çalışmaları kapsamında hazırlanmıştır.
 < Begüm Yaren ÖZTÜRK >
