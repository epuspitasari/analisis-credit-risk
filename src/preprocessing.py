import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings

# Mengabaikan peringatan merah (UserWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def ohe_transform(dataset, subset, prefix, ohe):
    """
    Fungsi untuk mentransformasi kolom kategorik menjadi kolom One-Hot Encoded.
    Diambil dari notebook preprocessing.ipynb.
    """
    # 1. Validasi parameter
    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError("Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!")
    if not isinstance(ohe, OneHotEncoder):
        raise RuntimeError("Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!")
    if not isinstance(prefix, str):
        raise RuntimeError("Fungsi ohe_transform: parameter prefix harus bertipe str!")
    if not isinstance(subset, str):
        raise RuntimeError("Fungsi ohe_transform: parameter subset harus bertipe str!")
    
    try:
        list(dataset.columns).index(subset)
    except:
        raise RuntimeError("Fungsi ohe_transform: parameter subset string namun data tidak ditemukan...")

    print("Fungsi ohe_transform: parameter telah divalidasi.")
    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}")
    
    dataset = dataset.copy()
    
    # 2. Proses Transformasi
    col_names = [f"{prefix}_{col_name}" for col_name in ohe.categories_[0].tolist()]
    
    encoded_array = ohe.transform(dataset[[subset]]).toarray()
    encoded = pd.DataFrame(encoded_array, columns=col_names, index=dataset.index)
    
    # 3. Gabung dan hapus kolom lama
    dataset = pd.concat([dataset, encoded], axis=1)
    dataset.drop(columns=[subset], inplace=True)
    
    return dataset