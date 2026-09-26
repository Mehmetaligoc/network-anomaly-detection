# Yapılan değişiklikler ve nedenleri

1. **Tek veri sözleşmesi:** Model ile arayüz artık 41 ham özelliği birlikte kullanır. Önceki arayüz 6 özellikten hareketle diğer alanları sıfırlıyordu; bu, modelin öğrendiği trafik gösterimini bozuyordu.
2. **Eğitim / doğrulama / kalibrasyon ayrımı:** Ölçekleyici ve kategorik sözlük yalnızca eğitimde öğrenilir. Eşik, ayrı normal kalibrasyon verisinden belirlenir. Test normal etiketlerinden eşik türetilmez.
3. **Bir model paketi:** Model, ön işleme, özellik sırası ve eşik birlikte saklanır. Uygulama açılırken boyut ve dosya hash kontrolleri yapılır.
4. **Geçersiz veri için açık hata:** NaN karşılaştırmasının yanlışlıkla normal kararına dönüşmesi önlenir. Eksik sütunlar kabul edilmez.
5. **Ölçümlerde tutarlılık:** Confusion matrix her zaman iki sınıfı içerir. Sıfır paydalı ölçümler tanımsız gösterilir. Yüklenen dosyanın ölçümleri ile ana test raporu ayrılır.
6. **Tekrarlanabilir çalışma:** Tohum, bağımlılık sürümleri, veri hash'leri, bölüm indeksleri ve eğitim geçmişi kaydedilir. Farklı donanım/sürümlerde birebir aynı sayılar garanti edilmez.

## Bilerek korunan sınırlamalar

- Autoencoder normal trafikle eğitilir; görülmeyen kategoriler için uyarı vardır ancak öğrenilmiş temsilleri yoktur.
- %95 yüzdelik eşik, kalibrasyon verisinde yaklaşık %5 yanlış alarm hedefidir. Son testte %10,20 yanlış alarm gözlenmiştir.
- Eski 6 özellikli sentetik veri dosyaları yeni sürümün performans kanıtı olarak kullanılmaz.
- Test sonuçları görüldükten sonra bu sürümün eşiği yükseltilip düşürülmemiştir. Sonraki geliştirmeler için yalnızca doğrulama verisiyle karar verilmeli; yeni bir dış değerlendirme seti ayrılmalıdır.
- Arayüzde elle girilen özellikler mevcut NSL-KDD tanımlarını takip etmelidir. Bu proje ham PCAP'ten 41 özelliği hesaplamaz.

## Birlikte çalışırken öğrenme sırası

Önce `ids/schema.py` ve eski 6 özellik sorununu incele. Sonra `train.py` içindeki veri ayrımını ve normal trafikle öğrenimi takip et. `ids/metrics.py` ile yanlış alarm / kaçırılan saldırı ayrımını öğren. En son arayüzde örnek normal ve saldırı kayıtlarını çalıştırıp kararın hangi eşikten geldiğini açıkla.

## CV için doğrulanabilir proje metni

**Ağ Anomali Tespit Uygulaması — Python, TensorFlow/Keras, scikit-learn, Streamlit**

NSL-KDD kayıtları üzerinde normal trafikle eğitilen autoencoder tabanlı bir anomali tespit prototipi geliştirdim. Eğitim, doğrulama ve eşik kalibrasyonu bölümlerini ayırarak 22.544 kayıtlık ayrı test kümesinde %87,55 doğruluk ve %88,70 F1 ölçtüm. CSV analizi, girdi doğrulama ve sonuç dışa aktarımı sunan bir Streamlit arayüzü hazırladım.

Bu metni ancak değişiklikleri inceleyip mülakatta açıklayabildiğinde kullan. %10,20 yanlış alarm oranını ve projenin çevrimdışı kapsamını teknik sorularda açıkça anlat. Önceki %90 sentetik test sonucu bu ölçümle aynı deney değildir.
