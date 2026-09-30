# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster 3 · pipeline en Databricks
# MAGIC Corre el mismo paquete `cluster3` que en local y deja, en `<catalog>.<schema>`:
# MAGIC - **Bronce**: clientes, llamadas y diccionario tal como llegaron (Delta).
# MAGIC - **Plata**: clientes limpios con decisiones por variable y llamadas con PII enmascarada y roles corregidos.
# MAGIC - **Oro**: scores por cliente, JSON por llamada (método híbrido), segmentos, accionables, lift y benchmark.
# MAGIC - Modelos de intención y churn en **MLflow + Unity Catalog** con alias `champion`.
# MAGIC - Artefactos en el Volume `artefactos` para la app de Databricks Apps.
# MAGIC
# MAGIC Requisitos en el Volume `raw`: los tres Excel y, en `raw/insumos/`, la clasificación híbrida ya calculada
# MAGIC (`data/processed/llamadas_hibrido.parquet`) y su benchmark (`outputs/tables/nlp_benchmark.csv`), para no
# MAGIC volver a clasificar las 500 llamadas con el LLM. Para reclasificar: `usar_llm = true` y `usar_jev = true`
# MAGIC (claves en el scope de secretos `cluster3`).

# COMMAND ----------

# MAGIC %pip install -q lightgbm shap duckdb openpyxl python-dotenv pydantic langgraph langchain-core langchain-openai typesafe-sdk mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("repo_root", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cluster3")
dbutils.widgets.dropdown("usar_llm", "false", ["false", "true"])
dbutils.widgets.dropdown("usar_jev", "false", ["false", "true"])

REPO = dbutils.widgets.get("repo_root")
CAT, SCH = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
USAR_LLM = dbutils.widgets.get("usar_llm") == "true"
USAR_JEV = dbutils.widgets.get("usar_jev") == "true"
VOL_RAW = f"/Volumes/{CAT}/{SCH}/raw"
VOL_ART = f"/Volumes/{CAT}/{SCH}/artefactos"
LOCAL = "/tmp/cluster3"  # disco local del cómputo: DuckDB y LightGBM necesitan escritura aleatoria
SCOPE = "cluster3"

# COMMAND ----------

import os
import shutil
import sys
from pathlib import Path

os.environ["CLUSTER3_ROOT"] = LOCAL
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
user = spark.sql("SELECT current_user()").first()[0]
os.environ["MLFLOW_EXPERIMENT_NAME"] = f"/Users/{user}/cluster3_churn"
# Claves solo si se va a reclasificar con LLM o Jev: se leen del scope de secretos, nunca del repo
if USAR_LLM:
    os.environ["LLM_PROVIDER"] = "openrouter"
    os.environ["OPENROUTER_API_KEY"] = dbutils.secrets.get(SCOPE, "OPENROUTER_API_KEY")
if USAR_JEV:
    os.environ["TYPESAFE_API_KEY"] = dbutils.secrets.get(SCOPE, "TYPESAFE_API_KEY")

# Datos crudos: Volume → disco local
raw = Path(LOCAL) / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)
for f in ["Clientes_Cluster_3.xlsx", "Llamadas.xlsx", "Diccionario_Datos_Cluster3.xlsx"]:
    shutil.copy(f"{VOL_RAW}/{f}", raw / f)
# Etiquetas humanas versionadas en el repo (solo ids y etiquetas)
lab = Path(REPO) / "data" / "labels"
if lab.exists():
    shutil.copytree(lab, Path(LOCAL) / "data" / "labels", dirs_exist_ok=True)
# Clasificación híbrida ya calculada: el pipeline la reutiliza en lugar de volver a llamar al LLM
insumos = Path(VOL_RAW) / "insumos"
if insumos.exists():
    shutil.copytree(insumos, LOCAL, dirs_exist_ok=True)
    print("Insumos reutilizados:", [str(p.relative_to(insumos)) for p in insumos.rglob("*") if p.is_file()])

sys.path.insert(0, f"{REPO}/src")

# COMMAND ----------

from cluster3.pipeline import main

if USAR_LLM or USAR_JEV:  # reclasificar las llamadas con el método híbrido antes del resto del pipeline
    from cluster3.nlp.run import run as run_nlp

    run_nlp(use_llm=USAR_LLM, use_jev=USAR_JEV)
main([])

# COMMAND ----------

# MAGIC %md ## Tablas Delta: bronce, plata y oro

# COMMAND ----------

import json
from datetime import datetime, timezone

import pandas as pd

from cluster3 import config
from cluster3.data.load import load_clientes, load_diccionario, load_llamadas

INGESTA = datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_delta(pdf: pd.DataFrame, name: str, comentario: str, bronce: bool = False) -> None:
    pdf = pdf.copy()
    for c in pdf.columns:
        if pdf[c].dtype == object:
            if pdf[c].map(lambda v: isinstance(v, (list, dict))).any():  # columnas anidadas → JSON
                pdf[c] = pdf[c].map(lambda v: json.dumps(v, ensure_ascii=False, default=str))
            elif bronce:  # bronce conserva el dato como llegó: tipos mixtos ('NR' en ESTRATO) como texto
                pdf[c] = pdf[c].astype(str)
    if bronce:
        pdf["_fuente"], pdf["_ingesta_utc"] = name.replace("bronce_", ""), INGESTA
    tabla = f"{CAT}.{SCH}.{name}"
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabla))
    spark.sql(f"COMMENT ON TABLE {tabla} IS '{comentario.replace(chr(39), '')}'")
    print(f"{tabla}: {len(pdf)} filas")


# Bronce: tal como llegó
to_delta(load_clientes(raw / "Clientes_Cluster_3.xlsx"), "bronce_clientes", "Bronce · clientes del Cluster 3 tal como llegaron", True)
to_delta(load_llamadas(raw / "Llamadas.xlsx"), "bronce_llamadas", "Bronce · transcripciones originales (acceso restringido)", True)
to_delta(load_diccionario(raw / "Diccionario_Datos_Cluster3.xlsx"), "bronce_diccionario", "Bronce · diccionario de datos", True)
# Plata: limpio y con decisiones documentadas
to_delta(pd.read_parquet(config.T_CLIENTES_LIMPIO), "plata_clientes", "Plata · clientes limpios (score reescalado, rangos corregidos)")
to_delta(pd.read_parquet(config.T_LLAMADAS_LIMPIAS), "plata_llamadas", "Plata · llamadas con PII enmascarada y roles corregidos")
to_delta(pd.read_csv(config.OUT_TAB / "registro_decisiones_variables.csv"), "plata_registro_decisiones",
         "Plata · decisión y motivo por variable (usar, transformar, excluir)")
# Oro: listo para consumo
to_delta(pd.read_parquet(config.T_SCORES), "oro_scores", "Oro · probabilidades de intención y churn por cliente (fuera de muestra)")
to_delta(pd.read_parquet(config.T_LLAMADAS_ANALISIS), "oro_llamadas_analisis", "Oro · JSON por llamada, método híbrido Jev + LLM")
for t, desc in [("segmentos_cluster3", "churn e intención por segmento"), ("accionables", "accionables e impacto anual"),
                ("lift_intencion", "lift por decil, intención"), ("lift_churn", "lift por decil, churn"),
                ("nlp_benchmark", "reglas vs LLM vs Jev vs híbrido contra 50 etiquetas humanas"),
                ("agent_evals", "preguntas doradas del sistema multiagente")]:
    f = config.OUT_TAB / f"{t}.csv"
    if f.exists():
        to_delta(pd.read_csv(f), f"oro_{t}", f"Oro · {desc}")

# COMMAND ----------

# MAGIC %md ## Modelos en Unity Catalog (alias `champion`)

# COMMAND ----------

import pickle

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
cliente_mlflow = MlflowClient()
metricas = json.loads((config.OUT_JSON / "metricas_modelo.json").read_text(encoding="utf-8"))
scores = pd.read_parquet(config.T_SCORES)
for key in ["intencion", "churn"]:
    with open(config.MODELS_DIR / f"modelo_{key}.pkl", "rb") as f:
        b = pickle.load(f)
    nombre = f"{CAT}.{SCH}.modelo_{key}"
    with mlflow.start_run(run_name=f"registro_{key}"):
        mlflow.log_metrics({k: float(metricas[key][k]) for k in ["auc_cv_media", "pr_auc", "lift_10"]})
        info = mlflow.sklearn.log_model(b["model"], name=f"modelo_{key}", input_example=scores[b["features"]].head(5),
                                        registered_model_name=nombre)
    version = info.registered_model_version
    cliente_mlflow.set_registered_model_alias(nombre, "champion", version)
    print(f"{nombre} v{version} → champion")

# COMMAND ----------

# MAGIC %md ## Artefactos para la app y limpieza

# COMMAND ----------

from cluster3.deploy.volume_sync import push

print("Archivos en el Volume:", push(VOL_ART, LOCAL))
shutil.rmtree(raw)  # los Excel originales solo viven en el Volume raw
