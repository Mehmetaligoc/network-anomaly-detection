"""Local, offline NSL-KDD anomaly scoring interface."""
from pathlib import Path
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import pandas as pd
import streamlit as st
from ids.schema import FEATURES, InputError, read_upload
from ids.pipeline import load_bundle, score_frame, classify, sha256
from ids.metrics import binary_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models/ae_v2'
st.set_page_config(page_title='Ağ Anomali Analizi', page_icon='🔎', layout='wide')
st.title('Ağ Anomali Analizi')
st.caption('NSL-KDD kayıtları için çevrimdışı autoencoder prototipi · Mehmet Ali Göç')
st.info('41 özellikli trafik kayıtlarını değerlendirir. Eşik altında kalan bir kayıt, '
        'trafiğin güvenli olduğunu kanıtlamaz. Canlı ağ dinleme veya saldırı engelleme içermez.')

example_files = ['tek_normal.csv', 'tek_saldiri.csv', 'test_ornegi_5000.csv']
if any(not (ROOT / 'data/examples' / name).is_file() for name in example_files):
    st.info('İlk kullanım için örnek verileri hazırlayın. Proje klasöründe '
            '`python scripts/prepare_data.py` komutunu çalıştırıp sayfayı yenileyin. '
            'Bu adım modeli yeniden eğitmez.')
    st.stop()


@st.cache_resource
def get_bundle(manifest_hash):
    return load_bundle(MODEL_DIR)


@st.cache_data(show_spinner=False, max_entries=8)
def get_scores(frame, manifest_hash, _bundle):
    return score_frame(_bundle, frame)


def show_result(frame, threshold, bundle, manifest_hash, prefix):
    try:
        with st.spinner('Kayıtlar değerlendiriliyor…'):
            scores, warnings = get_scores(frame, manifest_hash, bundle)
        for message in warnings:
            st.warning(message)
        predictions = classify(scores, threshold)
        columns = st.columns(3)
        columns[0].metric('İncelenen kayıt', len(frame))
        columns[1].metric('Anomali', int(predictions.sum()))
        columns[2].metric('Eşik altında', int((predictions == 0).sum()))
        if 'True_Class' in frame:
            metrics = binary_metrics(frame.True_Class, predictions, scores)
            columns = st.columns(5)
            for col, (label, key) in zip(columns, [('Doğruluk', 'accuracy'), ('Kesinlik', 'precision'),
                        ('Yakalama oranı', 'recall'), ('F1', 'f1'), ('Yanlış alarm oranı', 'fpr')]):
                value = metrics[key]
                col.metric(label, 'Tanımsız' if value is None else f'%{value*100:.2f}')
            st.caption('Ölçümler yalnızca yüklenen dosya ve seçili eşik için geçerlidir. '
                       'Etiketler: 0 normal, 1 saldırı. Tanımsız ölçümlerde ilgili payda sıfırdır.')
            st.dataframe(pd.DataFrame([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]],
                    index=['Gerçek normal', 'Gerçek saldırı'], columns=['Eşik altında', 'Anomali']))
        result = frame.copy()
        result['AnomalyScore'] = scores
        result['Prediction'] = predictions
        result['Durum'] = ['Anomali' if p else 'Eşik altında' for p in predictions]
        st.dataframe(result, use_container_width=True)
        st.download_button('Sonuçları CSV olarak indir', result.to_csv(index=False).encode('utf-8-sig'),
                           file_name='anomali_sonuclari.csv', mime='text/csv', key=f'{prefix}_download')
    except (InputError, ValueError) as exc:
        st.error(str(exc))


try:
    manifest_hash = sha256(MODEL_DIR / 'manifest.json')
    bundle = get_bundle(manifest_hash)
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f'Model paketi yüklenemedi: {exc}. README içindeki kurulum adımlarını izleyin.')
    st.stop()

manifest = bundle[2]
threshold = manifest['threshold']
with st.sidebar:
    st.subheader('Model bilgileri')
    st.write(f"Sürüm: {manifest['version']}")
    st.write('Girdi: 41 ham özellik')
    st.write(f"Kodlanmış boyut: {manifest['encoded_features']}")
    st.write(f'Kalibre edilmiş eşik: {threshold:.8f}')
    st.caption('Eşik, eğitim dosyasından ayrılmış normal kalibrasyon kayıtlarının %95 yüzdeliğidir.')
    if st.toggle('Deneysel eşik kullan', key='experimental'):
        threshold = st.number_input('Deneysel eşik', min_value=0.0, value=float(threshold),
                                    step=0.00001, format='%.8f')
        st.warning('Bu ayar keşif içindir. Buradaki sonuçlar kayıtlı model değerlendirmesini değiştirmez.')
    st.caption('Model: normal trafik ile eğitim. Son test: ayrı KDDTest+ dosyası.')

csv_tab, single_tab = st.tabs(['CSV Analizi', 'Tek Kayıt'])
with csv_tab:
    source = st.radio('Veri kaynağı', ['CSV yükle', 'Örnek test dosyası'], horizontal=True)
    st.download_button('41 özellikli örnek CSV indir',
        (ROOT / 'data/examples/tek_normal.csv').read_bytes(), file_name='ornek_41_ozellik.csv', mime='text/csv')
    content = None
    if source == 'Örnek test dosyası':
        st.caption('KDDTest+ içinden sabit tohumla seçilmiş 5.000 kayıt. '
                   'Bu bir gösterim alt kümesidir; ana rapor 22.544 kaydı kullanır.')
        content = (ROOT / 'data/examples/test_ornegi_5000.csv').read_bytes()
    else:
        uploaded = st.file_uploader('CSV dosyası seç', type=['csv'])
        if uploaded is not None:
            content = uploaded.getvalue()
    if content is not None:
        try:
            frame = read_upload(content)
            st.caption(f'{len(frame):,} kayıt · 41 özellik · En fazla 50.000 kayıt / 25 MB')
            st.dataframe(frame.head(5), use_container_width=True)
            if st.button('CSV dosyasını analiz et', type='primary', key='analyze_csv'):
                show_result(frame, threshold, bundle, manifest_hash, 'csv')
        except InputError as exc:
            st.error(str(exc))

with single_tab:
    st.write('Örnekteki 41 özelliği inceleyebilir ve değiştirebilirsin. '
             'Özellikler NSL-KDD biçiminde önceden çıkarılmış olmalıdır.')
    selected = st.selectbox('Başlangıç kaydı', ['Normal etiketli örnek', 'Saldırı etiketli örnek'])
    filename = 'tek_normal.csv' if selected.startswith('Normal') else 'tek_saldiri.csv'
    sample = pd.read_csv(ROOT / 'data/examples' / filename)[FEATURES]
    editor = pd.DataFrame({'Özellik': FEATURES, 'Değer': [str(sample.iloc[0][c]) for c in FEATURES]})
    edited = st.data_editor(editor, disabled=['Özellik'], hide_index=True,
                            use_container_width=True, height=440, key=f'editor_{filename}')
    st.caption('Düzenlenen kaydın gerçek etiketi bilinmediği için burada başarı oranı hesaplanmaz.')
    if st.button('Kaydı analiz et', key='analyze_single'):
        frame = pd.DataFrame([dict(zip(edited['Özellik'], edited['Değer']))])
        show_result(frame, threshold, bundle, manifest_hash, 'single')
