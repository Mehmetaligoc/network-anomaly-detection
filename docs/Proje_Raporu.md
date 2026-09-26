TEKNİK PROJE RAPORU  /  ae_v2

Ağ Anomali TespitUygulaması

Mehmet Ali GöçMehmet Akif Ersoy ÜniversitesiBilişim Sistemleri Mühendisliği

Rapor tarihi: 26 Eylül 2026Deney ve doğrulama kayıtları: 25 Eylül 2026

### Özet

Bu proje, NSL-KDD biçimindeki ağ bağlantı kayıtlarında olağandışı davranışları belirlemek amacıyla geliştirilmiş çevrimdışı bir anomali tespit prototipidir. Normal trafikle eğitilen autoencoder, girdiyi yeniden oluşturmaya çalışır. Yeniden oluşturma hatası kalibrasyon eşiğini aşan kayıtlar anomali olarak işaretlenir.

Yeni sürümde eğitim ve arayüz 41 ham özellik için ortak bir veri sözleşmesi kullanır. Ön işleme yalnızca normal eğitim kayıtlarında öğrenilir; doğrulama, eşik kalibrasyonu ve son test ayrı tutulur. Streamlit arayüzü, CSV ve tek kayıt analizi ile sonuçların dışa aktarılmasını sağlar.

| Son test: 22.544 kayıt | Ölçülen değer |
| --- | --- |
| Doğruluk | %87,55 |
| Saldırı yakalama oranı | %85,86 |
| F1 | %88,70 |
| Normal trafikte yanlış alarm | %10,20 |

Bu oranlar ayrı KDDTest+ dosyasına aittir. Önceki altı özellikli sentetik veri üzerindeki yaklaşık %90 sonucu ile aynı deney değildir. Bulgular, güncel bir kurum ağında veya bilinmeyen saldırılarda doğrulanmış başarı olarak yorumlanmamalıdır.

Doğrulama durumu: Önceki çalışmada 23 otomatik test geçmiştir. Kullanıcının kendi ortamında gerçek tarayıcı üzerinden çalıştırma kontrolü henüz tamamlanmamıştır. Bu rapor hazırlanırken model yeniden eğitilmemiş; kayıtlı deney, kaynak kod ve doğrulama notları esas alınmıştır.

## 1. Veri ve deney tasarımı

KDDTrain+ dosyası 125.973 bağlantı kaydı içerir: 67.343 normal, 58.630 saldırı. Her kayıtta 41 özellik ile saldırı etiketi ve zorluk bilgisi bulunur. Model girdisine etiket ve zorluk alanları alınmaz. Normal etiket 0, diğer saldırı etiketleri 1 olarak değerlendirilir.

| Bölüm | Tüm kayıtlar | AE için normal | Kullanım |
| --- | --- | --- | --- |
| Eğitim | 100.778 | 53.874 | Ön işleme ve ağırlıklar |
| Doğrulama | 12.597 | 6.734 | Kayıp takibi |
| Kalibrasyon | 12.598 | 6.735 | Eşik belirleme |
| Ayrı son test | 22.544 | 9.711 | Nihai ölçüm |

Eğitim dosyası, sınıf oranları korunarak yaklaşık %80 / %10 / %10 bölünür. Sabit rastgelelik tohumu 42’dir. Bölüm indeksleri kaydedilmiştir; otomatik testler bu bölümlerin kesişmediğini doğrular. Autoencoder eğitim, doğrulama ve kalibrasyon aşamalarında yalnızca ilgili bölümün normal kayıtlarını kullanır.

### Eşik kalibrasyonu

Normal kalibrasyon kayıtlarının yeniden oluşturma hatalarının %95 yüzdeliği, NumPy “higher” yöntemiyle seçilmiştir. Eşik 0,01442244645 olarak kaydedilmiştir. Karar kuralı katıdır: skor eşikten büyükse anomali; eşitse veya küçükse eşik altında.

Kalibrasyon kayıtlarında gözlenen yanlış alarm oranı %4,99 iken ayrı test dosyasında %10,20’dir. Kalibrasyonda seçilen yüzdelik, dağılımı farklı bir veri kümesinde aynı yanlış alarm oranını garanti etmez.

### Önceki sürümden ayrılan noktalar

Önceki arayüzün altı özellikten hareketle diğer özellikleri sıfırla doldurma davranışı kaldırılmıştır. Yeni model 41 ham özelliğin tamamını ister. Kategorik sözlük oluşturmak için test verisi eğitim verisine eklenmez; eşik testteki normal etiketlere bakılarak belirlenmez.

Yorum sınırı: KDDTest+ önceki proje incelemesinde de görülmüştür. Yeni eğitim betiğinde test dosyası kararlar tamamlandıktan sonra açılır; ancak bu sonuçlar daha önce hiç incelenmemiş, tamamen yeni bir dış doğrulama deneyinin karşılığı değildir.

## 2. Ön işleme ve model

Girdi, 38 sayısal ve 3 kategorik özellikten oluşur. Sayısal alanlarda MinMaxScaler; protocol_type, service ve flag alanlarında one-hot kodlama kullanılır. Her iki dönüşüm yalnızca normal eğitim verisiyle öğrenilir. Bu deneyde kodlanmış girdi boyutu 77’dir; boyut eğitimde görülen kategorilere bağlıdır.

Eğitimde görülmeyen bir kategori, ilgili one-hot alanlarında sıfır vektörüyle temsil edilir ve kullanıcıya uyarı verilir. Testte 1.343 kaydın service, 2 kaydın flag değeri eğitimde görülmemiştir; bu gruplar çakışabileceğinden sayılar benzersiz kayıt toplamı olarak toplanmamalıdır.

| Ayar | Bu deneydeki değer |
| --- | --- |
| Katman genişlikleri | 77 - 64 - 32 - 16 - 32 - 64 - 77 |
| Aktivasyonlar | Gizli katmanlar ReLU; çıkış sigmoid |
| Düzenlileştirme | İki Dropout katmanı, oran 0,2 |
| Optimizasyon / kayıp | Adam / ortalama karesel hata (MSE) |
| Parametre / grup büyüklüğü | 15.261 / 256 |
| Eğitim | 20 dönem; en iyi doğrulama dönemi 20 |

Erken durdurma, doğrulama kaybını izler; sabır değeri 3’tür ve en iyi ağırlıklar geri yüklenir. Bu deney 20 dönemin tamamını çalıştırmıştır. Son eğitim kaybı yaklaşık 0,003190, doğrulama kaybı 0,002608’dir. Düşen yeniden oluşturma kaybı tek başına saldırı yakalama başarısını kanıtlamaz.

Şekil 1. Kayıtlı deneyin eğitim ve doğrulama MSE eğrileri. Kaynak: reports/ae_v2.json ve reports/loss.png.

Görsel kaynağı: reports/loss.png

## 3. Ayrı test kümesindeki sonuçlar

KDDTest+ içindeki 9.711 normal ve 12.833 saldırı kaydı, sabit eşik kullanılarak değerlendirilmiştir. Karşılaştırma için Random Forest aynı %80 eğitim bölümünün hem normal hem saldırı etiketleriyle, 100 ağaç ve tohum 42 kullanılarak eğitilmiştir. Ön işlemesi kendi eğitim bölümünde öğrenilir; hiperparametre araması yapılmamıştır.

| Ölçüm | Autoencoder | Random Forest |
| --- | --- | --- |
| Doğruluk | %87,55 | %77,38 |
| Kesinlik (precision) | %91,75 | %96,88 |
| Yakalama (recall) | %85,86 | %62,26 |
| F1 | %88,70 | %75,81 |
| Yanlış alarm (FPR) | %10,20 | %2,65 |
| ROC-AUC | 0,9324 | 0,9594 |
| Average precision | 0,9258 | 0,9612 |

| Gerçek sınıf / Karar | Eşik altında | Anomali |
| --- | --- | --- |
| Normal | 8.720 (TN) | 991 (FP) |
| Saldırı | 1.815 (FN) | 11.018 (TP) |

Yakalama oranı = TP / (TP + FN); yanlış alarm oranı = FP / (FP + TN). Kesinlik, anomali olarak işaretlenen kayıtların ne kadarının gerçek saldırı olduğunu gösterir. F1, kesinlik ve yakalamanın harmonik ortalamasıdır.

Bu karar eşiklerinde autoencoder daha fazla saldırı yakalar ve daha yüksek F1 elde eder. Random Forest daha az yanlış alarm üretir; ROC-AUC ve average precision değerleri daha yüksektir. Dolayısıyla autoencoder’ın her ölçütte üstün olduğu sonucu çıkarılamaz. Denetimli ve normal trafikle öğrenen yöntemlerin amaçları da farklıdır.

Random Forest ağırlıkları teslim paketinde saklanmaz; eğitim betiği karşılaştırmayı yeniden üretir. Önceki not defterindeki XGBoost sonuçları bu deneyde yeniden ölçülmediği için buraya taşınmamıştır.

### Operasyonel karşılığı

Bu testte 991 normal kayıt yanlış alarm üretmiş, 1.815 saldırı kaydı eşik altında kalmıştır. Bir kaydın eşik altında olması güvenli olduğunu kanıtlamaz. Anomali skoru da kalibre edilmiş saldırı olasılığı değildir.

## 4. Etikete göre ayrıntılı sonuçlar

Aşağıdaki oran saldırı satırlarında yakalama oranıdır; “normal” satırında yanlış alarm oranıdır. Genel skor, saldırı türleri arasındaki büyük farkları tek başına göstermez.

| NSL-KDD etiketi | Kayıt sayısı | Anomali işaretlenen | Oran |
| --- | --- | --- | --- |
| apache2 | 737 | 735 | %99,73 |
| back | 359 | 53 | %14,76 |
| buffer_overflow | 20 | 7 | %35,00 |
| ftp_write | 3 | 1 | %33,33 |
| guess_passwd | 1.231 | 1.047 | %85,05 |
| httptunnel | 133 | 114 | %85,71 |
| imap | 1 | 1 | %100,00 |
| ipsweep | 141 | 53 | %37,59 |
| land | 7 | 7 | %100,00 |
| loadmodule | 2 | 2 | %100,00 |
| mailbomb | 293 | 14 | %4,78 |
| mscan | 996 | 944 | %94,78 |
| multihop | 18 | 8 | %44,44 |
| named | 17 | 7 | %41,18 |
| neptune | 4.657 | 4.656 | %99,98 |
| nmap | 73 | 73 | %100,00 |
| normal | 9.711 | 991 | %10,20 |
| perl | 2 | 2 | %100,00 |
| phf | 2 | 0 | %0,00 |
| pod | 41 | 38 | %92,68 |
| portsweep | 157 | 157 | %100,00 |
| processtable | 685 | 641 | %93,58 |
| ps | 15 | 4 | %26,67 |
| rootkit | 13 | 3 | %23,08 |
| saint | 319 | 281 | %88,09 |
| satan | 735 | 643 | %87,48 |
| sendmail | 14 | 4 | %28,57 |
| smurf | 665 | 541 | %81,35 |
| snmpgetattack | 178 | 113 | %63,48 |
| snmpguess | 331 | 330 | %99,70 |
| sqlattack | 2 | 2 | %100,00 |
| teardrop | 12 | 12 | %100,00 |
| udpstorm | 2 | 2 | %100,00 |
| warezmaster | 944 | 505 | %53,50 |
| worm | 2 | 0 | %0,00 |
| xlock | 9 | 5 | %55,56 |
| xsnoop | 4 | 3 | %75,00 |
| xterm | 13 | 10 | %76,92 |

Örneğin mailbomb için 293 kayıttan yalnızca 14’ü; back için 359 kayıttan 53’ü işaretlenmiştir. Birkaç örnekli sınıflarda %0 veya %100 gibi oranlar güçlü genelleme kanıtı değildir.

## 5. Uygulama ve doğrulama

### Kullanıcı akışı

Streamlit uygulaması hazır ae_v2 modelini yükler. Kullanıcı 41 özellikli CSV yükleyebilir veya KDDTest+ içinden sabit tohumla seçilen 5.000 kayıtlık gösterim dosyasını analiz edebilir. Tek kayıt sekmesi tüm özellikleri gösterir ve düzenlemeye açar; düzenlenen kaydın gerçek etiketi bilinmediğinden burada başarı oranı hesaplanmaz.

CSV’de True_Class alanı varsa 0 normal / 1 saldırı etiketleriyle ölçüm yapılır. Çıktıda AnomalyScore, Prediction ve Durum alanları bulunur. Eksik sütunlar, NaN/sonsuz sayılar, geçersiz sayısal aralıklar ve hatalı etiketler reddedilir. Üst sınır 25 MB ve 50.000 kayıttır. Tek sınıflı veride paydası sıfır olan ölçümler “tanımsız” gösterilir.

### Modelin taşınması

Model, ön işleyici ve manifest tek sürüm klasöründe saklanır. Manifest özellik sırasını, girdi boyutunu, eşiği, sürümleri ve dosya hash’lerini içerir. Yüklemede boyut ve hash tutarlılığı denetlenir. Hash kontrolü kaynak güvenilirliğini kanıtlayan bir dijital imza değildir; model dosyaları güvenilir kaynaktan alınmalıdır.

### Yapılmış kontroller

25 Eylül 2026 kaydına göre 23 test geçmiştir. Testler hatalı girdileri, tek sınıflı ölçümleri, dondurulmuş ön işleme parametrelerini, kesişmeyen bölüm indekslerini, modelin yeniden yüklenmesini ve tekli/toplu skor tutarlılığını kapsar. Tam test dosyasından kayıtlı hata matrisi ve ölçümler yeniden üretilmiştir. Streamlit AppTest ile örnek CSV ve tek kayıt akışları çalıştırılmıştır.

Test süresi 26,38 saniyedir; bu süre tahmin hızı ölçümü değildir. 14 Matplotlib/Pyparsing kullanımdan kaldırma uyarısı kaydedilmiştir. Gerçek tarayıcıda kullanıcı kontrolü, yük testi ve farklı tohumlarda tekrar deneyleri tamamlanmamıştır.

### Tekrar çalıştırma

Python 3.12 sanal ortamında requirements.txt ile bağımlılıklar kurulur; uygulama “python -m streamlit run app.py” ile açılır. “python evaluate.py” kayıtlı modeli sabit eşikle ölçer. “python train.py --version ae_v3 --epochs 20” yeni bir sürüm üretir. Testler için requirements-dev.txt ve “python -m pytest -q” kullanılır. Ayrıntılı Windows komutları README’de yer alır.

Başlıca sürümler: Python 3.12.14; TensorFlow 2.20.0; Keras 3.13.2; scikit-learn 1.6.1; Streamlit 1.45.1. Farklı donanım veya sürümlerde birebir aynı sayılar garanti edilmez. Yeni eğitim sırasında genel grafik dosyaları yeniden yazıldığı için mevcut deney çıktıları önce arşivlenmelidir.

## 6. Sınırlamalar ve paylaşım değerlendirmesi

### Araştırma sınırları

Proje ham ağ paketlerini toplamaz ve PCAP dosyasından 41 özelliği çıkarmaz. Önceden hazırlanmış NSL-KDD bağlantı kayıtlarını değerlendirir. Güncel saldırıların, gerçek kurum trafiğinin veya zero-day tespitinin doğrulanmış bir çözümü olarak sunulmamalıdır. MinMax ölçekleyicinin eğitim aralığını aşan yeni sayısal değerler 0-1 dışına çıkabilir; sigmoid çıkış ve bu dağılım farkı anomali skorunu etkileyebilir.

Deney tek tohumla yürütülmüştür; ortalama, güven aralığı veya zamansal genelleme ölçülmemiştir. İleri çalışmalar; doğrulama verisi üzerinden yöntem karşılaştırması, yanlış alarm analizi, eğitimde görülmeyen kategorilerin ele alınması ve daha önce incelenmemiş bir dış veri kümesiyle değerlendirme üzerine kurulmalıdır. Test sonucuna bakarak tekrar tekrar eşik seçmekten kaçınılmalıdır.

### GitHub için mevcut durum

Değerlendirme: öğrenci ve portföy projesi olarak paylaşılabilir. Ayrılmış kaynak kod, kurulum açıklaması, testler, sürümlenmiş model ve ölçüm raporu projenin incelenmesini kolaylaştırır. Açıklamada “NSL-KDD üzerinde çevrimdışı autoencoder tabanlı anomali tespiti” ifadesi mevcut kapsamla uyumludur. Üretime hazır bir güvenlik ürünü olduğu iddiası desteklenmemektedir.

Yayımdan önce kendi ortamındaki çalıştırma sonucunu README’ye eklemek yerinde olur. Paylaşımda kaynak kod, bağımlılık dosyaları, raporlar ve gerekli model paketi kullanılabilir. Ham veri .gitignore ile hariç tutulur; mevcut teslim ZIP’i ise ham verileri de içerir. ZIP’in tamamını yüklemek yerine proje dosyalarını Git üzerinden seçerek eklemek bu ayrımı korur. Örnek CSV’ler de kaynak veriden türetildiği için bunların paylaşım koşulları ayrıca kontrol edilmelidir.

İncelenen metin dosyalarında yapılan temel anahtar/parola taramasında eşleşme görülmemiştir; bu tam kapsamlı bir sır veya bağımlılık denetimi değildir. Kod için lisans henüz seçilmemiştir. Veri kullanım koşulları ve sunumdaki üçüncü taraf görsellerin paylaşım izni bu çalışma kapsamında doğrulanmamıştır. Bu rapor GitHub’a yükleme veya yayın işlemi yapmaz.

### Kaynaklar ve izlenebilirlik

[1] reports/ae_v2.json: metrikler, eğitim geçmişi, tür bazında sonuçlar.[2] models/ae_v2/manifest.json: eşik, boyutlar, sürümler ve hash’ler.[3] train.py; ids/schema.py; ids/pipeline.py; ids/metrics.py: yöntem.[4] app.py; tests/: arayüz ve otomatik kontroller.[5] reports/DOGRULAMA.md; README.md: çalışma kayıtları ve kullanım.[6] Veri aynası: github.com/defcom17/NSL_KDD. Kaynak, orijinal not defteri ve indirme betiğindeki adrestir; resmî veri kaynağı/lisans doğrulaması anlamına gelmez.
