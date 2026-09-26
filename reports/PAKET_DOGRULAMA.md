# GitHub paketi doğrulaması

26 Eylül 2026 tarihinde, teslim klasöründen ayrı bir test kopyasında çalıştırıldı. Python 3.12.14 ve requirements-dev.txt bağımlılıkları kullanıldı. Model yeniden eğitilmedi.

| Durum | Geçen | Atlanan | Süre |
|---|---:|---:|---:|
| Veri hazırlanmadan | 21 | 4 | 31,30 saniye |
| Veri hazırlandıktan sonra | 23 | 2 | 27,98 saniye |

İlk aşamada veri gerektiren kontroller ve paket dışında tutulan bölüm indeksleri kontrolü atlandı. İlk açılış ekranının veri hazırlama yönlendirmesi başarıyla kontrol edildi.

İkinci aşamada tam KDDTest+ sonuç karşılaştırması, modelin yeniden yüklenmesi, tekli/toplu skor tutarlılığı, 5.000 kayıtlık örnek CSV analizi ve tek kayıt analizi geçti. Atlanan iki kontrol: pakette olmayan bölüm indeksleri ve yalnızca veri yokken geçerli olan ilk açılış testi. Her iki aşamada da 14 Matplotlib/Pyparsing kullanımdan kaldırma uyarısı vardı; başarısız test yoktu.

## Paket için ek kontroller

- Yeni hazırlık betiği, mevcut SHA-256 doğrulanmış KDDTest+ dosyasından çalıştırıldı. Bu kontrol sırasında yeni ağ indirmesi yapılmadı.
- Üretilen üç CSV'nin SHA-256 değerleri önceki deneyin örnekleriyle birebir eşleşti.
- Hatalı kaynak hash'i örnek dosya oluşturulmadan reddedildi.
- Model ve ön işleyicinin hash'leri mevcut manifestle eşleşti.
- Kaynak dosyaların Python sözdizimi kontrol edildi.
- Git'in ham veri, örnek CSV, .env, Streamlit gizli ayarları, sanal ortam ve önbellekleri dışarıda bıraktığı kontrol edildi.
- Metin dosyalarında yapılan sınırlı kullanıcı yolu / anahtar örüntüsü taramasında eşleşme bulunmadı. Bu kontrol kapsamlı bir güvenlik denetimi değildir.

Streamlit kontrolleri AppTest ile yapıldı. Kullanıcının kendi bilgisayarında gerçek tarayıcıyla denemesi henüz tamamlanmadı. Kaydedilen doğruluk ve F1 değerleri önceki ae_v2 deneyine aittir; bu paketleme çalışması yeni bir başarı iddiası üretmez.
