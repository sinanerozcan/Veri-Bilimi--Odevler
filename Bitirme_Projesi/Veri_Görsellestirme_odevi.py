#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[9]:


data = pd.read_csv('50_Startups.csv')


# In[10]:


data.head()
print("\n" + "="*50 + "\n")

# ## Bu veri seti, 50 farklı startup şirketinin çeşitli harcamalarını ve kârlılıklarını içermektedir.  
# 
# R&D Spend (Ar-Ge Harcaması): Şirketin araştırma ve geliştirme (Ar-Ge) için harcadığı tutar.  
# Administration (Yönetim Harcaması): Şirketin yönetim giderleri için harcadığı tutar.  
# Marketing Spend (Pazarlama Harcaması): Şirketin pazarlama ve reklam faaliyetleri için harcadığı tutar. 
# State (Eyalet): Şirketin faaliyet gösterdiği eyalet (örneğin, New York, California, Florida).  
# Profit (Kâr): Şirketin elde ettiği toplam kâr.  
# Bu veri seti, startup'ların çeşitli harcama kalemleri ile kârlılıkları arasındaki ilişkileri analiz   etmek için kullanılabilir. Örneğin, Ar-Ge veya pazarlama harcamalarının kârlılık üzerindeki etkisini   incelemek için uygun bir veri setidir.  

# ## 1.GÖREV : R&D Harcaması ve Kâr Arasındaki İlişki (Scatter Plot): Ar-Ge harcamaları ile kâr arasındaki ilişkiyi gösteren bir dağılım grafiği.

# In[11]:


# Kodu buraya yazınız 
print("1. GÖREV: R&D Harcaması ve Kâr Arasındaki İlişki (Scatter Plot)")
plt.figure(figsize=(10, 6))
plt.scatter(data['R&D Spend'], data['Profit'], color='blue', alpha=0.7)
plt.title('R&D Harcaması ve Kar İlişkisi')
plt.xlabel('R&D Spend')
plt.ylabel('Kar')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n" + "="*50 + "\n")

# ## 2.GÖREV: Yönetim Harcamaları ve Kâr Arasındaki İlişki (Scatter Plot): Yönetim harcamaları ile kâr arasındaki ilişkiyi gösteren bir dağılım grafiği.

# In[12]:


# Kodu buraya yazınız 
print("2.GÖREV: Yönetim Harcamaları ve Kâr Arasındaki İlişki (Scatter Plot):")
plt.figure(figsize=(10, 6))
plt.scatter(data['Administration'], data['Profit'], color='blue', alpha=0.7)
plt.title('Administration ve Kar İlişkisi')
plt.xlabel('Administration')
plt.ylabel('Kar')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n" + "="*50 + "\n")

# ## 3. GÖREV: Eyaletlere Göre Ortalama Kâr (Bar Chart): Farklı eyaletlerdeki startup'ların ortalama kârlarını karşılaştıran bir çubuk grafik.

# In[13]:


# Kodu buraya yazınız 
print("3. GÖREV: Eyaletlere Göre Ortalama Kâr (Bar Chart):")
avg_profit = data.groupby('State')['Profit'].mean().sort_values(ascending= False)

plt.figure(figsize=(10,6))
avg_profit.plot(kind= 'bar', color= ['blue', 'green', 'red'])
plt.title('Eyaletlere Göre Ortalama Kar')
plt.xlabel('State')
plt.ylabel('avg_profit')
plt.xticks(rotation= 0)
plt.grid(axis='y', linestyle= '--', alpha= 0.7)
plt.show()

print("\n" + "="*50 + "\n")

# ## 4. GÖREV: Harcama Türlerinin Karşılaştırması (Boxplot): R&D, yönetim ve pazarlama harcamalarının dağılımını karşılaştıran bir kutu grafiği.

# In[14]:


# Kodu buraya yazınız 
print("4. GÖREV: Harcama Türlerinin Karşılaştırması (Boxplot):")
spending_data = data [['R&D Spend', 'Administration', 'Marketing Spend']]

plt.figure(figsize=(10,6))
plt.boxplot(spending_data.values, labels=spending_data.columns, patch_artist= True)
plt.title('Harcama Türlerinin Karşılaştırması')
plt.xlabel('Harcama Türü')
plt.ylabel('Harcanan Miktar')
plt.grid(axis='y', linestyle= '--', alpha= 0.7)
plt.show()

# In[ ]:




