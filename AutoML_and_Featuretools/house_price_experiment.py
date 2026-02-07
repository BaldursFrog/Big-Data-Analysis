import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import featuretools as ft
import tpot
from tpot import TPOTRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_excel('california_housing.xlsx')
    data = df.iloc[:, 0].str.split(',', expand=True)
    data.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Price']
    data = data.astype(float)
    X = data.drop('Price', axis=1)
    y = data['Price']
    return X, y, data

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def approach1_manual_features(X, y):
    X_manual = X.copy()
    X_manual['Rooms_per_person'] = X_manual['AveRooms'] / X_manual['AveOccup']
    X_manual['Bedrooms_ratio'] = X_manual['AveBedrms'] / X_manual['AveRooms']
    X_manual['Population_density'] = X_manual['Population'] / (X_manual['AveOccup'] * X_manual['AveRooms'])
    X_manual['Location_score'] = np.sqrt((X_manual['Latitude'] - 35)**2 + (X_manual['Longitude'] + 119)**2)
    X_manual['Age_income_interaction'] = X_manual['HouseAge'] * X_manual['MedInc']
    
    X_train, X_test, y_train, y_test = train_test_split(X_manual, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred)

def approach2_featuretools(X, y, data):
    es = ft.EntitySet(id='housing_data')
    es = es.entity_from_dataframe(entity_id='houses', dataframe=data, index='index')
    
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='houses', 
                                         max_depth=2, verbose=0)
    
    feature_matrix = feature_matrix.dropna(axis=1)
    X_ft = feature_matrix.drop('Price', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_ft, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred)

def approach3_automl(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=0, random_state=42)
    tpot.fit(X_train, y_train)
    y_pred = tpot.predict(X_test)
    
    return evaluate_model(y_test, y_pred)

def main():
    X, y, data = load_and_preprocess_data()
    
    print("House Price Prediction Experiment")
    print("=================================")
    
    print("\nApproach 1: Original + Manual Feature Engineering")
    results1 = approach1_manual_features(X, y)
    print(f"RMSE: {results1['RMSE']:.4f}")
    print(f"MAE: {results1['MAE']:.4f}")
    print(f"R²: {results1['R2']:.4f}")
    
    print("\nApproach 2: Original + FeatureTools Features")
    results2 = approach2_featuretools(X, y, data)
    print(f"RMSE: {results2['RMSE']:.4f}")
    print(f"MAE: {results2['MAE']:.4f}")
    print(f"R²: {results2['R2']:.4f}")
    
    print("\nApproach 3: Original + AutoML (TPOT)")
    results3 = approach3_automl(X, y)
    print(f"RMSE: {results3['RMSE']:.4f}")
    print(f"MAE: {results3['MAE']:.4f}")
    print(f"R²: {results3['R2']:.4f}")
    
    print("\nComparison Summary")
    print("==================")
    comparison = pd.DataFrame({
        'Manual Features': results1,
        'FeatureTools': results2,
        'AutoML': results3
    })
    print(comparison)
    
    print("\nBest Approach by Metric:")
    print(f"Lowest RMSE: {comparison.loc['RMSE'].idxmin()} ({comparison.loc['RMSE'].min():.4f})")
    print(f"Lowest MAE: {comparison.loc['MAE'].idxmin()} ({comparison.loc['MAE'].min():.4f})")
    print(f"Highest R²: {comparison.loc['R2'].idxmin()} ({comparison.loc['R2'].max():.4f})")

if __name__ == "__main__":
    main()