import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_segmentar_clientes():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función segmentar_clientes.
    """

    # 1. Configuración aleatoria
    n_rows = random.randint(30, 100)
    ciudades_disponibles = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 'Bucaramanga']
    ciudades = random.sample(ciudades_disponibles, k=random.randint(2, 5))

    # 2. Generar datos aleatorios
    df = pd.DataFrame({
        'cliente_id': range(1, n_rows + 1),
        'edad': np.random.randint(18, 70, n_rows),
        'gasto_total': np.random.uniform(50, 5000, n_rows),
        'num_compras': np.random.randint(1, 50, n_rows),
        'dias_desde_ultima_compra': np.random.randint(1, 365, n_rows),
        'ciudad': np.random.choice(ciudades, n_rows),
    })

    # 3. Construir INPUT
    input_data = {'df': df.copy()}

    # 4. Calcular OUTPUT esperado replicando la lógica de segmentar_clientes
    result = df.copy()

    # Gasto promedio por compra
    result['gasto_promedio_por_compra'] = result['gasto_total'] / result['num_compras']

    # Segmento según percentiles del gasto_total
    p33 = result['gasto_total'].quantile(0.33)
    p66 = result['gasto_total'].quantile(0.66)

    def asignar_segmento(gasto):
        if gasto < p33:
            return 'Bronce'
        elif gasto <= p66:
            return 'Plata'
        else:
            return 'Oro'

    result['segmento'] = result['gasto_total'].apply(asignar_segmento)

    # Cliente reciente
    result['cliente_reciente'] = result['dias_desde_ultima_compra'] <= 30

    # DataFrame resumen agrupado
    resumen = result.groupby(['ciudad', 'segmento']).agg(
        total_clientes=('cliente_id', 'count'),
        gasto_promedio=('gasto_total', 'mean'),
        pct_recientes=('cliente_reciente', lambda x: round(x.sum() / len(x) * 100, 2))
    ).reset_index()

    resumen = resumen.sort_values(['ciudad', 'segmento']).reset_index(drop=True)

    output_data = resumen

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_segmentar_clientes()

    print("=== INPUT ===")
    print(entrada['df'].head(10).to_string())
    print(f"\nTotal filas: {len(entrada['df'])}")

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada.to_string())
