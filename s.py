from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('path/to/your/train.csv')  


numeric_features = df.select_dtypes(include=[np.number]).columns.drop(['Id', 'SalePrice'])


X = df[numeric_features]


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Calcular el VIF para cada variable
vif_data = pd.DataFrame()
vif_data["feature"] = numeric_features
vif_data["VIF"] = [variance_inflation_factor(X_imputed, i) for i in range(X_imputed.shape[1])]

# Filtrar las variables con VIF alto
vif_threshold = 10  
high_vif_features = vif_data[vif_data['VIF'] > vif_threshold]

print(high_vif_features)
