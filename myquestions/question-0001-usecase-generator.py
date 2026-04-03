import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_limpiar_y_resumir_ventas():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función limpiar_y_resumir_ventas.
    """

    # 1. Configuración aleatoria
    n_rows = random.randint(8, 20)
    categorias = random.sample(['Electrónica', 'Accesorios', 'Ropa', 'Hogar', 'Deportes'], k=random.randint(2, 4))
    productos_por_categoria = {
        'Electrónica': ['Laptop', 'Tablet', 'Monitor'],
        'Accesorios':  ['Mouse', 'Teclado', 'Audífonos'],
        'Ropa':        ['Camiseta', 'Pantalón', 'Chaqueta'],
        'Hogar':       ['Lámpara', 'Cojín', 'Cortina'],
        'Deportes':    ['Balón', 'Raqueta', 'Guantes'],
    }
    precios_por_categoria = {
        'Electrónica': (500, 2000),
        'Accesorios':  (10, 100),
        'Ropa':        (20, 150),
        'Hogar':       (15, 200),
        'Deportes':    (10, 300),
    }

    # 2. Generar filas aleatorias
    fechas, productos, cats, precios, cantidades, descuentos = [], [], [], [], [], []

    for _ in range(n_rows):
        cat = random.choice(categorias)
        low, high = precios_por_categoria[cat]
        fechas.append(f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        productos.append(random.choice(productos_por_categoria[cat]))
        cats.append(cat)
        precios.append(round(random.uniform(low, high), 2))
        cantidades.append(random.randint(1, 10))
        descuentos.append(round(random.choice([0.0, 0.05, 0.1, 0.15, 0.2]), 2))

    df = pd.DataFrame({
        'fecha': fechas,
        'producto': productos,
        'categoria': cats,
        'precio': precios,
        'cantidad': cantidades,
        'descuento': descuentos,
    })

    # 3. Introducir nulos y duplicados aleatorios
    for col, prob in [('precio', 0.08), ('cantidad', 0.08), ('descuento', 0.08),
                      ('producto', 0.05), ('fecha', 0.05)]:
        mask = np.random.choice([True, False], size=len(df), p=[prob, 1 - prob])
        df.loc[mask, col] = np.nan

    # Duplicar alguna fila aleatoriamente
    if len(df) > 3:
        dup_idx = random.randint(0, len(df) - 1)
        df = pd.concat([df, df.iloc[[dup_idx]]], ignore_index=True)

    # 4. Construir INPUT
    input_data = {'df': df.copy()}

    # 5. Calcular OUTPUT esperado replicando la lógica de limpiar_y_resumir_ventas
    result = df.copy()

    # Eliminar duplicados
    result = result.drop_duplicates(subset=['fecha', 'producto', 'precio'])

    # Eliminar filas donde producto o fecha son nulos
    result = result.dropna(subset=['producto', 'fecha'])

    # Rellenar precio con mediana por categoria
    result['precio'] = result.groupby('categoria')['precio'].transform(
        lambda x: x.fillna(x.median())
    )

    # Rellenar cantidad y descuento
    result['cantidad'] = result['cantidad'].fillna(1)
    result['descuento'] = result['descuento'].fillna(0.0)

    # Crear columna total_venta
    result['total_venta'] = result['precio'] * result['cantidad'] * (1 - result['descuento'])

    # Convertir fecha a datetime
    result['fecha'] = pd.to_datetime(result['fecha'])

    # Ordenar y resetear índice
    result = result.sort_values('fecha').reset_index(drop=True)

    output_data = result

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_limpiar_y_resumir_ventas()

    print("=== INPUT ===")
    print(entrada['df'].to_string())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada.to_string())
