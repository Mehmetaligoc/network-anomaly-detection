# Çalıştırılan kontroller

25 Eylül 2026, Windows ve Python 3.12.14 üzerinde **23 test geçti**. Test süresi 26,38 saniye. TensorFlow 2.20.0, Keras 3.13.2, scikit-learn 1.6.1 ve Streamlit 1.45.1 kullanıldı.

- Eksik özellikler, geçersiz sayılar, NaN/sonsuz değerler ve yanlış etiketler reddedildi.
- Tek sınıflı veri için hata matrisi ve tanımsız ölçümler doğrulandı.
- Eğitimden sonra dönüşümün ölçekleyici parametrelerini değiştirmediği kontrol edildi.
- Eğitim, doğrulama ve kalibrasyon indekslerinin kesişmediği doğrulandı.
- Model paketi yeniden yüklendi; tek tek ve toplu tahmin skorları karşılaştırıldı.
- KDDTest+ dosyasının tamamındaki tahminlerden kayıtlı raporun hata matrisi ve ölçümleri yeniden üretildi.
- Streamlit AppTest ile başlangıç, 5.000 kayıtlık örnek CSV analizi ve tek kayıt analizi çalıştırıldı.
- Ham veri dosyalarının SHA-256 değerleri doğrulandı.

Testlerde 14 Matplotlib/Pyparsing kullanımdan kaldırma uyarısı vardı; başarısız test yoktu. Arayüz kontrolleri AppTest üzerinden yapıldı; gerçek tarayıcıda manuel kullanım veya yük altında performans testi yapılmadı. Eğitim tek tohumla çalıştırıldı; farklı tohumlar için ortalama/güven aralığı ölçülmedi.

Sunumun 10 slaydı görsel önizlemeler üzerinden incelendi; dosya yapısı, düzenlenebilir yeni tablolar ve eğitim grafiğinin veri bağlantısı kontrol edildi. PowerPoint uygulamasında açılarak doğrulama yapılmadı.
