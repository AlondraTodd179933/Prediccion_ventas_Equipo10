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
