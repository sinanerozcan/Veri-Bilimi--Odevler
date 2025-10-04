#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('dava_sonuclari.csv')
data.head()


# ## VERİ SETİ İNCELEME : 
# Case Type: Davanın türü (Criminal, Civil, Commercial)  
# Case Duration (Days): Davanın süresi (gün olarak)  
# Judge Experience (Years): Hakimin deneyim yılı  
# Number of Witnesses: Tanık sayısı  
# Legal Fees (USD): Hukuk masrafları (USD olarak)  
# Plaintiff's Reputation: Davacının itibarı (1: Düşük, 2: Orta, 3: Yüksek)  
# Defendant's Wealth (USD): Davalının serveti  
# Number of Evidence Items: Delil sayısı  
# Number of Legal Precedents: İlgili hukuki emsal sayısı  
# Settlement Offered (USD): Teklif edilen uzlaşma miktarı  
# Severity: Davanın ciddiyet derecesi (1: Düşük, 2: Orta, 3: Yüksek)  
# Outcome: Davanın sonucu (0: Kaybetmek, 1: Kazanmak)  

# ## Görevler
# 
# ### Veri Ön İşleme:
# * Veri setini inceleyin ve eksik veya aykırı değerler olup olmadığını kontrol edin.  
# * Gerektiğinde eksik verileri doldurun veya çıkarın.  
# * Özelliklerin ölçeklendirilmesi gibi gerekli veri dönüşümlerini uygulayın. 
# 
# ### Veri Setini Ayırma:
# * Veri setini eğitim ve test setleri olarak ayırın (örn. %80 eğitim, %20 test).  
# 
# ### Model Kurulumu:
# * Karar ağacı modelini oluşturun ve eğitim verileri üzerinde eğitin.
# 
# ### Modeli Değerlendirme:
# * Test verilerini kullanarak modelin doğruluğunu değerlendirin.
# * Doğruluk, precision, recall ve F1-score gibi performans metriklerini hesaplayın.
# 
# ### Sonuçları Görselleştirme:
# * Karar ağacının yapısını görselleştirin.
# * Karar ağacının nasıl çalıştığını ve hangi özelliklerin davanın sonucunu belirlemede en etkili olduğunu açıklayın.

# In[ ]:
import numpy as np
import pandas as pd

data = pd.read_csv("dava_sonuclari.csv")

print(data.head())
print(data.info())
print(data.describe())

print(data["Outcome"].value_counts())

random_data = data.copy()

np.random.seed(42)
win_indicies = np.random.choice(random_data.index, size=len(random_data)//2, replace=False)
random_data.loc[win_indicies, "Outcome"] = 1

print(random_data["Outcome"].value_counts())

from sklearn.preprocessing import LabelEncoder

encoded_data = random_data.copy()

le = LabelEncoder()
encoded_data['Case Type'] = le.fit_transform(encoded_data['Case Type'])

from sklearn.model_selection import train_test_split

X = encoded_data[[
    'Case Type',
    'Case Duration (Days)',
    'Judge Experience (Years)',
    'Number of Witnesses',
    'Legal Fees (USD)',
    "Plaintiff's Reputation",
    "Defendant's Wealth (USD)",
    'Number of Evidence Items',
    'Number of Legal Precedents',
    'Settlement Offered (USD)',
    'Severity',

]]
Y = encoded_data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size= 0.2, random_state= 42, stratify=Y
)

print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clsfy = DecisionTreeClassifier(random_state=42, max_depth=4)

clsfy.fit(X_train,Y_train)

Y_pred = clsfy.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)

print("\nDesicion Tree Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(Y_test,Y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test,Y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(30,15))
plot_tree(
    clsfy,
    feature_names=list(X.columns),
    class_names=['0','1'],
    filled=True,
    rounded=True,
    fontsize= 14,
)

plt.show()

# Get feature importances from the trained model
importances = clsfy.feature_importances_

# Create a DataFrame to view the importances
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)



