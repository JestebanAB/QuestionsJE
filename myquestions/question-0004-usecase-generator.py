import numpy as np
import random
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso_pipeline_regresion():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función pipeline_regresion.
    """

    # 1. Configuración aleatoria
    n_samples   = random.randint(200, 600)
    n_features  = random.randint(4, 12)
    n_informative = random.randint(2, min(n_features, 6))
    noise       = random.uniform(10, 100)

    # 2. Generar datos sintéticos con make_regression
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random.randint(0, 9999)
    )

    # 3. Introducir algunos NaNs aleatorios (~5% de los datos)
    mask = np.random.choice([True, False], size=X.shape, p=[0.05, 0.95])
    X[mask] = np.nan

    # 4. Construir INPUT
    input_data = {'X': X.copy(), 'y': y.copy()}

    # 5. Calcular OUTPUT esperado replicando la lógica de pipeline_regresion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline de preprocesamiento
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    # Comparar modelos con validación cruzada
    modelos = {
        'Ridge':              Ridge(alpha=1.0),
        'GradientBoosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    resultados_cv = {}
    for nombre, modelo in modelos.items():
        pipeline_cv = Pipeline([
            ('preprocessor', SimpleImputer(strategy='median')),
            ('scaler',        StandardScaler()),
            ('modelo',        modelo),
        ])
        scores = cross_val_score(
            pipeline_cv, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error'
        )
        resultados_cv[nombre] = np.sqrt(-scores.mean())

    # Seleccionar mejor modelo
    mejor_nombre = min(resultados_cv, key=resultados_cv.get)
    rmse_cv      = round(resultados_cv[mejor_nombre], 4)

    # Entrenar pipeline completo con el mejor modelo
    mejor_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('modelo',  modelos[mejor_nombre]),
    ])
    mejor_pipeline.fit(X_train, y_train)

    # Evaluar sobre test
    y_pred   = mejor_pipeline.predict(X_test)
    rmse_test = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

    output_data = {
        'mejor_modelo':  mejor_nombre,
        'rmse_cv':       rmse_cv,
        'rmse_test':     rmse_test,
        'pipeline':      mejor_pipeline,
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_pipeline_regresion()

    print("=== INPUT ===")
    print(f"Shape de X: {entrada['X'].shape}")
    print(f"Shape de y: {entrada['y'].shape}")
    print(f"NaNs en X:  {np.isnan(entrada['X']).sum()}")

    print("\n=== OUTPUT ESPERADO ===")
    print(f"Mejor modelo: {salida_esperada['mejor_modelo']}")
    print(f"RMSE CV:      {salida_esperada['rmse_cv']}")
    print(f"RMSE Test:    {salida_esperada['rmse_test']}")
