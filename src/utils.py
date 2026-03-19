import pandas as pd
import numpy as np
import joblib
import os
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# --- FUNGSI SERIALISASI (SANGAT PENTING) ---

def serialize_data(data, path):
    """Menyimpan objek ke file .pkl"""
    folder = os.path.dirname(path)
    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(data, path)
    print(f"Data berhasil diserialisasi ke: {path}")

def deserialize_data(path):
    """Memuat objek dari file .pkl"""
    return joblib.load(path)

# --- FUNGSI PREPROCESSING (SESUAI SOAL MENTORING 2) ---

def drop_duplicate_data(X, y):
    if not isinstance(X, pd.DataFrame):
        raise RuntimeError("Fungsi drop_duplicate_data: parameter X haruslah bertipe DataFrame!")
    if not isinstance(y, pd.Series):
        raise RuntimeError("Fungsi drop_duplicate_data: parameter y haruslah bertipe Series!")
    
    print("Fungsi drop_duplicate_data: parameter telah divalidasi.")
    X = X.copy()
    y = y.copy()
    
    X_duplicate = X[X.duplicated()]
    X_clean_shape = (X.shape[0] - X_duplicate.shape[0], X.shape[1])
    
    print(f"Fungsi drop_duplicate_data: shape dataset sebelum dropping duplicate adalah {X.shape}.")
    print(f"Fungsi drop_duplicate_data: shape dari data yang duplicate adalah {X_duplicate.shape}.")
    print(f"Fungsi drop_duplicate_data: shape dataset setelah drop duplicate seharusnya adalah {X_clean_shape}.")
    
    X.drop_duplicates(inplace=True)
    y = y.loc[X.index]
    
    print(f"Fungsi drop_duplicate_data: shape dataset setelah dropping duplicate adalah {X.shape}.")
    return X, y

def median_imputation(data, subset_data, fit):
    if not isinstance(data, pd.DataFrame):
        raise RuntimeError("Fungsi median_imputation: parameter data haruslah bertipe DataFrame!")
    
    if fit == True:
        if not isinstance(subset_data, list):
            raise RuntimeError("Fungsi median_imputation: untuk nilai parameter fit = True, subset_data harus bertipe list dan berisi daftar nama kolom yang ingin dicari nilai mediannya guna menjadi data imputasi pada kolom tersebut.")
        
        print("Fungsi median_imputation: parameter telah divalidasi.")
        data = data.copy()
        imputation_data = {}
        for subset in subset_data:
            imputation_data[subset] = data[subset].median()
            
        print(f"Fungsi median_imputation: proses fitting telah selesai, berikut hasilnya {imputation_data}.")
        return imputation_data
    
    elif fit == False:
        if not isinstance(subset_data, dict):
            raise RuntimeError("Fungsi median_imputation: untuk nilai parameter fit = False, subset_data harus bertipe dict dan berisi key yang merupakan nama kolom beserta value yang merupakan nilai median dari kolom tersebut.")
        
        print("Fungsi median_imputation: parameter telah divalidasi.")
        data = data.copy()
        print("Fungsi median_imputation: informasi count na sebelum dilakukan imputasi:")
        print(data.isna().sum())
        
        data.fillna(subset_data, inplace=True)
        
        print("\nFungsi median_imputation: informasi count na setelah dilakukan imputasi:")
        print(data.isna().sum())
        return data
    else:
        raise RuntimeError("Fungsi median_imputation: parameter fit haruslah bertipe boolean, bernilai True atau False.")

def create_onehot_encoder(categories, path):
    if not isinstance(categories, list):
        raise RuntimeError("Fungsi create_onehot_encoder: parameter categories haruslah bertipe list, berisi kategori yang akan dibuat encodernya.")
    if not isinstance(path, str):
        raise RuntimeError("Fungsi create_onehot_encoder: parameter path haruslah bertipe string, berisi lokasi pada disk komputer dimana encoder akan disimpan.")
    
    ohe = OneHotEncoder()
    cat_array = np.array(categories).reshape(-1, 1)
    ohe.fit(cat_array)
    
    serialize_data(ohe, path) # Memanggil fungsi serialisasi internal
    
    learned_cats = ohe.categories_[0].tolist()
    print(f"Kategori yang telah dipelajari adalah {learned_cats}")
    return ohe

def ohe_transform(dataset, subset, prefix, ohe):
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
        raise RuntimeError("Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset.")
    
    print("Fungsi ohe_transform: parameter telah divalidasi.")
    dataset = dataset.copy()
    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}")
    
    col_names = [f"{prefix}_{col_name}" for col_name in ohe.categories_[0].tolist()]
    encoded_array = ohe.transform(dataset[[subset]]).toarray()
    encoded = pd.DataFrame(encoded_array, columns=col_names, index=dataset.index)
    
    dataset = pd.concat([dataset, encoded], axis=1)
    dataset.drop(columns=[subset], inplace=True)
    
    print(f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}")
    return dataset