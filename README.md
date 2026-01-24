# Prediccion_ventas_Equipo10
Alondra García y Ana Paredes


## Objetivo

El objetivo del proyecto es generar un modelo de Machine Learning que prediga la demanda mensual a nivel tienda-ítem, utilizando los datos otorgados por la empresa. Se espera que el modelo reduzca el error de predicción de RMSE ~ 11 unidades a menos de 5 unidades para alcanzar el margen operativo del 8.5%

## Datos
Se utilizaron datos históricos de ventas diarias con indicadores para las tiendas y artículos obtenidos de  [Kaggle - Predict Future Sales Competition](https://kaggle.com/competitions/competitive-data-science-predict-future-sales) 

### Datasets disponibles

| Archivo | Descripción |
|---------|-------------|
| `sales_train.csv` | Datos históricos de ventas diarias (2013-2015) |
| `test.csv` | Combinaciones producto-tienda para predicción |
| `items.csv` | Información de productos y categorías |
| `shops.csv` | Información de tiendas |
| `item_categories.csv` | Categorías de productos |

### Métrica de evaluación

**Root Mean Squared Error (RMSE)**

---

## Metodología
1. Limpieza y preparación de los datos.
2. Construcción de una base completa de combinaciones tienda–producto–mes.
3. Generación de variables rezagadas (lags) de 1, 3, 6 y 12 meses.
4. División temporal de los datos en entrenamiento y validación.
5. Entrenamiento de distintos modelos (baseline, Ridge y Random Forest).
6. Selección del modelo con mejor desempeño.

Nota: Solamente se creó un notebook pero se incluyeron secciones y comentarios para poder seguir el flujo 

### Modelo final 
El modelo final seleccionado fue un Random Forest entrenado con lags temporales de 1,3,6 y 12 meses. 

## Evaluación final 
l desempeño del modelo se evaluó mediante la métrica RMSE, obteniendo un valor de 0.7049 en el conjunto de validación y un score de 1.02 privado en kaggle (1.03 score público).

## Contenido requerido
- `Tarea1_FINAL.ipynb`: Notebook principal con todo el análisis y modelado.
  `Resumen ejecutivo.pdf`: Resumen ejecutivo con los puntos claves del proyecto.
- `README.md`: Descripción general del proyecto.
