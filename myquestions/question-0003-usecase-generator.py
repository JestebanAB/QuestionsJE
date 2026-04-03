import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def generar_caso_de_uso_entrenar_clasificador():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función entrenar_clasificador.
    """

    # 1. Configuración aleatoria
    n_samples  = random.randint(200, 600)
    n_features = random.randint(4, 12)
    n_classes  = random.randint(2, 4)
    n_informative = random.randint(2, min(n_features, 6))
    n_redundant  = random.randint(0, max(0, n_features - n_informative - 1))

    # 2. Generar datos sintéticos con make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random.randint(0, 9999)
    )

    # 3. Construir INPUT
    input_data = {'X': X.copy(), 'y': y.copy()}

    # 4. Calcular OUTPUT esperado replicando la lógica de entrenar_clasificador
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    output_data = {
        'accuracy':              round(accuracy_score(y_test, y_pred), 4),
        'reporte':               classification_report(y_test, y_pred),
        'importancia_features':  modelo.feature_importances_,
        'modelo':                modelo,
        'scaler':                scaler,
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_entrenar_clasificador()

    print("=== INPUT ===")
    print(f"Shape de X: {entrada['X'].shape}")
    print(f"Shape de y: {entrada['y'].shape}")
    print(f"Clases únicas: {np.unique(entrada['y'])}")

    print("\n=== OUTPUT ESPERADO ===")
    print(f"Accuracy:              {salida_esperada['accuracy']}")
    print(f"Importancia features:  {salida_esperada['importancia_features'].round(4)}")
    print(f"Reporte:\n{salida_esperada['reporte']}")
