#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


data = pd.read_csv('dava.csv')
data


# ## Veri Seti inceleme
# Veri Seti Özellikleri:  
# Case Duration (Gün): Davanın tamamlanması için geçen süre (gün cinsinden).  
# Number of Witnesses (Tanık Sayısı): Dava boyunca dinlenen tanık sayısı.  
# Legal Fees (Hukuk Maliyetleri): Dava süresince oluşan toplam hukuk maliyetleri (USD cinsinden).  
# Number of Evidence Items (Delil Sayısı): Davada kullanılan delil sayısı.  
# Severity (Ciddiyet Düzeyi): Davanın ciddiyet düzeyi (1: Düşük, 2: Orta, 3: Yüksek).  
# Outcome (Sonuç): Davanın sonucu (0: Aleyhte, 1: Lehinde).  

# ## GÖREV: 
# Özellik Seçimi: Hangi özelliklerin kümeleme için kullanılacağına karar verin.  
# Küme Sayısını Belirleme: Elbow yöntemi gibi tekniklerle optimal küme sayısını belirleyin.  
# Kümeleme İşlemi: K-Means algoritmasını kullanarak verileri kümeleyin.  
# Sonuçları Görselleştirme: Kümeleme sonuçlarını uygun grafiklerle görselleştirin ve yorumlayın.  

# In[ ]:
import numpy as np
import pandas as pd

import os
os.environ["OMP_NUM_THREADS"] = "1"

data = pd.read_csv("dava.csv")

print("Satır x Sütun:", data.shape)
print("\nSütun İsimleri:",list(data.columns))

print("\nEksik değerler:")
print(data.isnull().sum())

print("\nİlk 10 Satır:")
print(data.head(10))

print("\nOutcomes: (0: Kayıp, 1: Galibiyet)")
print(data['Outcome'].value_counts())

print("\nTemek İstatistikler:")
print(data.describe(include='all'))

print("\nDF info:")
data.info()

import seaborn as sns
import matplotlib.pyplot as plt

Ozellikler= [
    'Case Duration (Days)',
    'Number of Witnesses',
    'Legal Fees (USD)',
    'Number of Evidence Items',
    'Severity',
    'Outcome',
]

Korelasyon = data [Ozellikler + ["Outcome"]].corr()

plt.figure(figsize= (6,8))
sns.heatmap(Korelasyon, annot= True, cmap= "coolwarm", fmt= ".2f")
plt.title("Özellikler ve Outcome Korelasyon Matrisi")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = data[Ozellikler]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range= range(1,11)

for k in K_range:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Küme Sayısı')
plt.ylabel('Inertia')
plt.title('Elbow Yöntemi')
plt.show()

kmeans = KMeans(n_clusters = 3, random_state = 42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters

from sklearn.decomposition import PCA

pca = PCA(n_components= 2)
X_pca = pca.fit_transform(X_scaled)

centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize= (6,8))
plt.scatter(X_pca[:,0], X_pca[:,1], c = clusters, cmap= 'viridis', alpha=0.6)
plt.scatter(centers_pca[:,0], centers_pca[:,1], s=200, c='blue', marker='X', label='Merkezler')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('K-Means Kümeleme Sonucu')
plt.legend()
plt.show()

print("\nKümeler İçin Ortalama Özellik Değerleri:")
print(data.groupby("Cluster")[Ozellikler].mean())

# Kodu buraya yazınız.







