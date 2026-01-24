# Tarea 01: Predicción de Demanda en Retail con Machine Learning

## Objetivo

Desarrollar un modelo de Machine Learning end-to-end que prediga ventas futuras en retail, aplicando el flujo completo de ciencia de datos desde la exploración hasta la comunicación de resultados a stakeholders de negocio.

---

## Contexto del Negocio

Este caso de estudio está basado en datos reales provistos por **1C Company**, una de las firmas de software más grandes de Rusia, que opera una cadena de tiendas retail distribuidas en múltiples ciudades.

La compañía enfrenta el desafío clásico del retail moderno: balancear inventarios para más de **22,000 productos** distintos a través de **60 ubicaciones** diferentes, mientras mantiene costos operativos bajo control y maximiza la satisfacción del cliente.

Con tres años de datos transaccionales históricos que capturan millones de ventas diarias, la empresa se encuentra en el punto de inflexión perfecto para implementar soluciones de machine learning que transformen sus operaciones de supply chain.

El Chief Operations Officer (COO) y el Chief Innovation Officer (CIO) han preparado las siguientes comunicaciones para explicar la urgencia y visión estratégica detrás de este proyecto de Data Science:

### Del Chief Operations Officer (COO)

> Enfrentamos un problema crítico de inventario que está destruyendo valor. El **23%** de nuestro inventario está en sobrestock, generando altos costos de almacenamiento y obligándonos a liquidar con descuentos del 35%. Al mismo tiempo, tenemos quiebres de stock en productos clave el **18%** del tiempo, perdiendo **$6.8M USD** en ventas y provocando que nuestro Net Promoter Score cayera 12 puntos.
>
> Nuestros planificadores usan métodos tradicionales—promedios móviles y ajustes manuales—que no pueden manejar la complejidad de 60 tiendas y 22,170 productos con patrones que varían por ubicación y estacionalidad. Ajustamos inventarios cada 14 días mientras la competencia lo hace en 48 horas.
>
> **Necesitamos reducir nuestro error de predicción de RMSE ~11 unidades a menos de 5 unidades** para alcanzar nuestro margen operativo objetivo del 8.5% y mejorar nuestro inventory turnover de 6.2x a 9x anual.

### Del Chief Innovation Officer (CIO)

> Tenemos una oportunidad transformacional. Tres años de datos transaccionales históricos (2.9M registros) nos permiten implementar machine learning para predecir demanda con precisión granular a nivel producto-tienda-mes, anticipando comportamientos futuros en lugar de solo reaccionar al pasado.
>
> Mi visión es empoderar a nuestros demand analysts con herramientas de Data Science e IA que automaticen el 70% de predicciones rutinarias, liberándolos para estrategias de alto valor. Los modelos modernos pueden capturar patrones complejos que los métodos tradicionales ignoran: efectos de promociones, estacionalidad regional, y eventos externos.
>
> Con ML en producción, actualizaremos predicciones diariamente con intervalos de confianza para gestión de riesgo informada. Este proyecto es el primer paso hacia una organización data-driven donde inventario, pricing, y staffing se respalden con modelos predictivos. **El ROI está en construir capacidades que nos mantengan competitivos la próxima década.**

---

## Datos

**Fuente:** [Kaggle - Predict Future Sales Competition](https://kaggle.com/competitions/competitive-data-science-predict-future-sales)

> Alexander Guschin, Dmitry Ulyanov, inversion, Mikhail Trofimov, utility, and Μαριος Μιχαηλιδης KazAnova. *Predict Future Sales*. Kaggle, 2018.

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

## Instrucciones

### 1. Jupyter Notebooks - Flujo completo de Data Science

Desarrollar notebooks que documenten el proceso end-to-end de modelado, incluyendo:

- [ ] Exploración y entendimiento profundo de los datos
- [ ] Transformación y preparación de features
- [ ] Entrenamiento de modelos de Machine Learning
- [ ] Evaluación rigurosa de resultados con análisis crítico
- [ ] Generación de predicciones finales para el test set

**Requisitos técnicos:**

- Guardar el(los) modelo(s) final(es) como `.pkl` o `.joblib` (se utilizarán en clases futuras)

### 2. Executive Summary

Elaborar un documento de **2-3 páginas** en formato Markdown, HTML o PDF dirigido a stakeholders no técnicos (COO, CIO, VP of Operations) que comunique:

1. **Resumen** del problema y solución propuesta

2. **Evaluación detallada del modelo:**
   - Métricas de performance con comparación entre modelo(s)
   - Análisis crítico de fortalezas, debilidades y limitaciones
   - Identificación de escenarios donde el modelo funciona bien y dónde falla

3. **Visualizaciones efectivas** que comuniquen resultados de negocio siguiendo mejores prácticas

4. **Recomendaciones accionables** para operaciones y próximos pasos

> **Importante:** Todos los argumentos deben estar respaldados con datos cuantitativos. Evitar jerga técnica innecesaria.

### 3. Kaggle Submission

- Subir predicciones a la competencia de Kaggle
- Incluir screenshot del leaderboard mostrando tu RMSE y timestamp como anexo del documento ejecutivo

> **Importante:** No se espera que alcancen el top del leaderboard, pero sí que demuestren un enfoque riguroso y crítico en su modelado. (Reality-check)
---

## Criterios de Evaluación

| Criterio | Descripción |
|----------|-------------|
| **Resumen ejecutivo** | Claridad narrativa, argumentación basada en evidencia, utilidad para stakeholders, evaluación honesta de resultados, identificación de limitaciones, análisis de errores |
| **Código en notebooks** | Que refleje todos los pasos del modelado. |
| **Visualizaciones ejecutivas** | Estrategia para mostrar resultados, efectividad comunicativa, aplicación de mejores prácticas de data visualization |
| **Capacidad de síntesis** | Comunicación clara y concisa de ideas complejas a audiencias no técnicas |
| **Kaggle Submission** | RMSE logrado en leaderboard público es el único Anexo que pueden incluir) |
---

## Importante

> **No quiero que se enfoquen en traducir el Ruso:** los datos ya están en inglés. Busquen en las discusiones de Kaggle, por ejemplo este [link](https://www.kaggle.com/datasets/remisharoon/predict-future-sales-translated-dataset).

> **Enfoque en calidad sobre cantidad:** No se espera que logren el mejor RMSE del leaderboard. Se evaluará la rigurosidad del enfoque, la calidad del análisis y la capacidad de comunicar resultados de manera efectiva.
---

## Formato de Entrega

📤 **Vía:** CANVAS
- **Repositorio Github**

- **Contenido requerido:**

  - Notebooks de Jupyter
  - Executive Summary (PDF, HTML o Markdown)
  - README

---

## Fecha de Entrega

📅 **Sábado 24, antes de la clase**

