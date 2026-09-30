# Despliegue en Databricks (Free Edition)

Todo se define en `databricks.yml` (Asset Bundle): esquema, dos Volumes, el job del pipeline y la app.

| Pieza | Recurso |
|---|---|
| Datos crudos | Volume `workspace.cluster3.raw` (los tres Excel + `insumos/` con la clasificación híbrida ya calculada) |
| Pipeline | Job `cluster3_pipeline` → notebook `notebooks/databricks/pipeline_job.py` (cómputo serverless) |
| Tablas | Delta en `workspace.cluster3`: `bronce_*` (como llegó), `plata_*` (limpio, PII enmascarada), `oro_*` (scores, JSON por llamada, accionables, lift, benchmark, evaluación de agentes) |
| Modelos | MLflow + Unity Catalog: `workspace.cluster3.modelo_intencion` y `modelo_churn`, alias `champion` |
| App | Databricks Apps `cluster3-churn` (`app.yaml` + `requirements.txt`) |
| Secretos | Scope `cluster3`: `OPENROUTER_API_KEY` y `TYPESAFE_API_KEY`; la app los recibe como recursos (`valueFrom`) |

## 1. CLI y autenticación

```bash
winget install Databricks.DatabricksCLI          # Windows (o brew / script oficial)
databricks auth login --host https://<tu-workspace>.cloud.databricks.com --profile DEFAULT
```

## 2. Secretos (una vez)

```bash
databricks secrets create-scope cluster3
databricks secrets put-secret cluster3 OPENROUTER_API_KEY   # pega la clave cuando la pida
databricks secrets put-secret cluster3 TYPESAFE_API_KEY
```

## 3. Desplegar recursos

```bash
databricks bundle validate
databricks bundle deploy -t dev
```

## 4. Subir datos al Volume raw

```bash
databricks fs cp data/raw/Clientes_Cluster_3.xlsx dbfs:/Volumes/workspace/cluster3/raw/
databricks fs cp data/raw/Llamadas.xlsx dbfs:/Volumes/workspace/cluster3/raw/
databricks fs cp data/raw/Diccionario_Datos_Cluster3.xlsx dbfs:/Volumes/workspace/cluster3/raw/
# Clasificación híbrida y benchmark ya calculados (evitan reclasificar 500 llamadas con el LLM)
databricks fs mkdir dbfs:/Volumes/workspace/cluster3/raw/insumos/data/processed
databricks fs mkdir dbfs:/Volumes/workspace/cluster3/raw/insumos/outputs/tables
databricks fs cp data/processed/llamadas_hibrido.parquet dbfs:/Volumes/workspace/cluster3/raw/insumos/data/processed/
databricks fs cp outputs/tables/nlp_benchmark.csv dbfs:/Volumes/workspace/cluster3/raw/insumos/outputs/tables/
databricks fs cp outputs/tables/agent_evals.csv dbfs:/Volumes/workspace/cluster3/raw/insumos/outputs/tables/
```

## 5. Correr el pipeline

```bash
databricks bundle run cluster3_pipeline -t dev
```

Para reclasificar las llamadas con Jev y el LLM: `usar_llm` / `usar_jev` en `true` (lee las claves del scope).

## 6. App

```bash
databricks bundle run cluster3_app -t dev
```

La app descarga los artefactos del Volume `artefactos` al arrancar. Los permisos (leer el Volume y los dos
secretos) se declaran como recursos de la app en `databricks.yml`: no hace falta dar GRANT a mano.

## Al cerrar el proceso

```bash
databricks bundle destroy -t dev
databricks secrets delete-scope cluster3
```
