# GitHub paketi hakkında

Bu klasör yayımlama için hazırlanmış çalışma kopyasıdır; GitHub'a yüklenmemiştir. Model ve temel analiz kodu `ae_v2` ile aynıdır. Paket değişikliği, veri dosyalarını dağıtmak yerine yerelde hazırlama adımı ekler.

## Dahil edilenler

- Kaynak kod, bağımlılık dosyaları ve otomatik testler.
- Eğitilmiş autoencoder, ön işleme nesnesi ve manifest.
- Kayıtlı sonuçlar, deney grafikleri, doğrulama notları ve PDF/metin raporu.
- Veri indirme ve örnek oluşturma betikleri.

Ham veri ve örnek CSV'ler pakette bulunmaz; `scripts/prepare_data.py` örnekleri SHA-256 kontrolünden geçen KDDTest+ verisinden oluşturur. Bu işlem eğitim yapmaz. Yerelde oluşturulan veri dosyaları `.gitignore` ile dışarıda tutulur. Önbellekler, sanal ortam, bölüm indeksleri ve eski teslim ZIP'i de pakete dahil edilmez.

Sunumdaki üçüncü taraf görsellerin paylaşım durumu doğrulanmadığı için PPTX bu pakete eklenmemiştir. Yerel proje klasöründeki sunum korunmuştur.

## Raporun kapsamı

`Proje_Raporu.pdf`, 26 Eylül 2026 tarihinde, önceki deney kayıtlarından hazırlanmıştır. Raporda sözü edilen ham veri içeren teslim ZIP'i, önceki `IDS_Project_Guncellenmis.zip` dosyasıdır; bu GitHub paketi ham veri içermez. 23 test sonucu önceki tam çalışma kopyasına aittir. Veri hazırlanmadan bazı entegrasyon kontrolleri atlanır; atlanan kontroller geçmiş test sonucu gibi sunulmamalıdır.

## Lisans ve kaynak

Projeye bir açık kaynak lisansı henüz atanmadı; geliştirici seçimi yapılana kadar LICENSE dosyası eklenmedi. Bağımlılıkların ve kaynak veri kümesinin kullanım koşulları ayrı değerlendirilir. Veri dosyalarını depoya eklememek, veri kaynağının kullanım koşullarını kontrol etme gereğini ortadan kaldırmaz.

## Önerilen depo bilgileri

- Ad: `network-anomaly-detection`
- Açıklama: `NSL-KDD üzerinde autoencoder tabanlı çevrimdışı ağ anomali tespiti; Streamlit arayüzü, doğrulama testleri ve deney raporu.`
- Konular: `python`, `cybersecurity`, `anomaly-detection`, `autoencoder`, `tensorflow`, `streamlit`, `nsl-kdd`

Yayımlarken 41 özellikli çevrimdışı prototip kapsamını ve kayıtlı yanlış alarm oranını koru. Kullanıcı ortamında gerçek tarayıcıyla çalıştırma kontrolü tamamlanınca sonucu ve kullanılan Python sürümünü README'ye ekle.
