# Despliegue en Databricks (Free Edition)

Todo se define en `databricks.yml` (Asset Bundle): esquema, dos Volumes, el job del pipeline y la app.

| Pieza | Recurso |
|---|---|
| Datos crudos | Volume `workspace.cluster3.raw` (los tres Excel, subidos a mano) |
| Pipeline | Job `cluster3_pipeline` → notebook `notebooks/databricks/pipeline_job.py` (cómputo serverless) |
| Tablas | Delta en `workspace.cluster3.*` |
| Modelos | MLflow + Unity Catalog: `workspace.cluster3.modelo_intencion`, `workspace.cluster3.modelo_churn` |
| App | Databricks Apps `cluster3-churn` (`app.yaml` + `requirements.txt`) |
| LLM | Foundation Model APIs (`LLM_PROVIDER=databricks`) |

## 1. CLI y autenticación

```bash
brew install databricks/tap/databricks   # o: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks auth login --host https://<tu-workspace>.cloud.databricks.com
```

## 2. Desplegar recursos

```bash
databricks bundle validate
databricks bundle deploy -t dev
```

## 3. Subir los Excel al Volume raw

```bash
databricks fs cp data/raw/Clientes_Cluster_3.xlsx dbfs:/Volumes/workspace/cluster3/raw/
databricks fs cp data/raw/Llamadas.xlsx dbfs:/Volumes/workspace/cluster3/raw/
databricks fs cp data/raw/Diccionario_Datos_Cluster3.xlsx dbfs:/Volumes/workspace/cluster3/raw/
```

## 4. Correr el pipeline

```bash
databricks bundle run cluster3_pipeline -t dev
```

Con LLM o Jev: editar `usar_llm` / `usar_jev` en `databricks.yml` o al lanzar el job desde la UI.
Para Jev, guardar la clave como secreto (`databricks secrets put-secret cluster3 TYPESAFE_API_KEY`) y
leerla en el notebook con `dbutils.secrets.get("cluster3", "TYPESAFE_API_KEY")`.

## 5. App

```bash
databricks bundle run cluster3_app -t dev
```

La app descarga los artefactos del Volume `artefactos` al arrancar. Su service principal necesita permiso de lectura
(una vez, en SQL):

```sql
GRANT USE CATALOG ON CATALOG workspace TO `<service-principal-de-la-app>`;
GRANT USE SCHEMA ON SCHEMA workspace.cluster3 TO `<service-principal-de-la-app>`;
GRANT READ VOLUME ON VOLUME workspace.cluster3.artefactos TO `<service-principal-de-la-app>`;
```

El nombre del service principal aparece en la pestaña de la app (Authorization). Para el LLM, agregar en la app el
recurso *Serving endpoint* con permiso **Can query** sobre el endpoint de `LLM_MODEL` (`app.yaml`).

## Límites de Free Edition a tener en cuenta

- Solo cómputo serverless y cuotas diarias; el pipeline completo tarda ~2 minutos en local.
- Apps con límite de instancias y apagado automático: encender la app antes de la sustentación.
- Endpoints de Foundation Model con cuota por minuto; si se agota, la app cae a modo offline.
Verificar los límites vigentes en la documentación de Databricks Free Edition antes de la entrega.

## Cierre del proceso (datos internos de Claro)

```bash
databricks bundle destroy -t dev
```

Borra job, app, Volumes y esquema (si el esquema aún tiene tablas: `DROP SCHEMA workspace.cluster3 CASCADE`). Revisar también el experimento MLflow en `/Users/<usuario>/cluster3_churn`.
