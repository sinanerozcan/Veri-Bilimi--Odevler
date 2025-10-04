#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[3]:

import numpy as np
import pandas as pd

data = pd.read_csv("country.csv")

data['Country'] = data['Country'].str.strip()

colums_clean = [
    'Population', 'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
    'Net migration', 'Infant mortality (per 1000 births)', 'Literacy (%)',
    'Phones (per 1000)', 'Arable (%)', 'Crops (%)',
    'Other (%)', 'Birthrate', 'Deathrate',
    'Agriculture', 'Industry', 'Service'
]

for col in colums_clean:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',','.', regex = False), errors='coerce')

cleaned_data = data.dropna(subset=['Population'])

# ##  Country.csv dosyasının özelliği
# Bu tablo, çeşitli ülkelerle ilgili bir dizi demografik, ekonomik ve coğrafi veriyi içermektedir. Tabloda her bir satır bir ülkeyi temsil ederken, sütunlar bu ülkelerle ilgili farklı özellikleri gösterir. İşte sütunların anlamları:
# 
# Country: Ülkenin adı.  
# Region: Ülkenin bulunduğu bölge (örneğin, Asya, Doğu Avrupa).  
# Population: Ülkenin toplam nüfusu.  
# Area (sq. mi.): Ülkenin yüzölçümü (mil kare olarak).  
# Pop. Density (per sq. mi.): Nüfus yoğunluğu (mil kare başına düşen kişi sayısı).  
# Coastline (coast/area ratio): Sahil uzunluğunun, ülkenin toplam alanına oranı.  
# Net migration: Net göç oranı (göçmenlerin ülkeye giren veya ülkeden çıkan kişi sayısına göre oranı).  
# Infant mortality (per 1000 births): Bebek ölüm oranı (1000 doğum başına).  
# GDP ($ per capita): Kişi başına düşen Gayri Safi Yurtiçi Hasıla (GSYİH).  
# Literacy (%): Okur-yazarlık oranı.  
# Phones (per 1000): Her 1000 kişi başına düşen telefon sayısı.  
# Arable (%): Tarıma elverişli arazi yüzdesi.  
# Crops (%): Ekilebilir ürünlerin yüzdesi.  
# Other (%): Diğer arazi kullanımı yüzdesi.  
# Climate: Ülkenin iklim kategorisi (numerik bir değer olarak gösterilmiş).  
# Birthrate: Doğum oranı.  
# Deathrate: Ölüm oranı.  
# Agriculture: Tarım sektörünün ekonomideki payı.  
# Industry: Sanayi sektörünün ekonomideki payı.  
# Service: Hizmet sektörünün ekonomideki payı.  
# 

# ## Bu Dosyada Yapacağınız görevleri alt taraftan bakabilirsiniz.

# ## 1. Görev : Nüfusa Göre Azalan Sırada Sıralama:

# In[4]:

# Nüfusa Göre Azalan Sırada Sıralama kodunu buraya yazınız
azalan_nufus = cleaned_data.sort_values(by= 'Population', ascending=False)
print("\n 1. Görev: Nüfusa Göre Azalan Sırada Tüm Ülkeler")
print(azalan_nufus[['Country', 'Region', 'Population']].to_markdown(index=False))


# ## 2. Görev: GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla).

# In[5]:

# GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla). kodunu buradan yazınız.
artan_gdp = cleaned_data.sort_values(by= 'GDP ($ per capita)', ascending=True)
print("\n 2. Görev: GDP ($ per capita) Sütununa Göre Artan Sırada Tüm Ülkeler")
print(artan_gdp[['Country', 'Region', 'GDP ($ per capita)']].to_markdown(index=False))


# ## 3. Görev: Population sütunu 10 milyonun üzerinde olan ülkeleri seçmek.

# In[6]:

# Kodunu buraya yazınız.
populasyon_buyukluk = cleaned_data[cleaned_data['Population'] >= 10_000_000]
print(f"\n 3. Görev: Population >= 10 Milyon Olan Tüm Ülkeler (Toplam: {len(populasyon_buyukluk)})")
print(populasyon_buyukluk[['Country', 'Population']].to_markdown(index=False))


# ## 4. Görev: Literacy (%) sütununa göre ülkeleri sıralayıp, en yüksek okur-yazarlık oranına sahip ilk 5 ülkeyi seçmek.

# In[7]:

# Kodunu buraya yazınız.
okur_yazar = cleaned_data.sort_values(by= 'Literacy (%)', ascending=False).head(5)
print("\n 4. Görev: En Yüksek Okur-Yazarlık Oranına Sahip İlk 5 Ülke")
print(okur_yazar[['Country', 'Region', 'Literacy (%)']].to_markdown(index=False))


# ## 5. Görev:  Kişi Başı GSYİH 10.000'in Üzerinde Olan Ülkeleri Filtreleme: GDP ( per capita) sütunu 10.000'in üzerinde olan ülkeleri seçmek.

# In[8]:

# Kodunu buraya yazınız.
gdp_10milyon = cleaned_data[cleaned_data['GDP ($ per capita)'] > 10_000_000]
print(f"\n 3. Görev: GDP ($ per capita) > 10 Milyon Olan Tüm Ülkeler (Toplam: {len(gdp_10milyon)})")
print(gdp_10milyon[['Country', 'GDP ($ per capita)']].to_markdown(index=False))

# ## Görev 6 : En Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülkeyi Seçme:
# Pop. Density (per sq. mi.) sütununa göre ülkeleri sıralayıp, en yüksek nüfus yoğunluğuna sahip ilk 10 ülkeyi seçmek.

# In[ ]:
nufus_yogun = cleaned_data.sort_values(by= 'Pop. Density (per sq. mi.)', ascending=False).head(10)
print("\n 6. Görev: En Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülke")
print(nufus_yogun[['Country', 'Region', 'Pop. Density (per sq. mi.)']].to_markdown(index=False))



