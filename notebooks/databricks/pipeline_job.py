# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster 3 · pipeline en Databricks
# MAGIC Corre el mismo paquete `cluster3` que en local y deja:
# MAGIC - tablas Delta en `<catalog>.<schema>` (clientes con scores, llamadas analizadas, segmentos, accionables)
# MAGIC - experimentos en MLflow y modelos registrados en Unity Catalog
# MAGIC - artefactos en el Volume `artefactos` para la app de Databricks Apps
# MAGIC
# MAGIC Requisito: subir los tres Excel al Volume `/Volumes/<catalog>/<schema>/raw/`.

# COMMAND ----------

# MAGIC %pip install -q lightgbm shap duckdb openpyxl python-dotenv pydantic langgraph langchain-core databricks-langchain mlflow

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

# COMMAND ----------

import os
import shutil
import sys
from pathlib import Path

os.environ["CLUSTER3_ROOT"] = LOCAL
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
user = spark.sql("SELECT current_user()").first()[0]
os.environ["MLFLOW_EXPERIMENT_NAME"] = f"/Users/{user}/cluster3_churn"
if USAR_LLM:
    os.environ.setdefault("LLM_PROVIDER", "databricks")

# Datos crudos: Volume → disco local
raw = Path(LOCAL) / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)
for f in ["Clientes_Cluster_3.xlsx", "Llamadas.xlsx", "Diccionario_Datos_Cluster3.xlsx"]:
    shutil.copy(f"{VOL_RAW}/{f}", raw / f)
# Etiquetas humanas versionadas en el repo (solo ids y etiquetas)
lab = Path(REPO) / "data" / "labels"
if lab.exists():
    shutil.copytree(lab, Path(LOCAL) / "data" / "labels", dirs_exist_ok=True)

sys.path.insert(0, f"{REPO}/src")

# COMMAND ----------

from cluster3.pipeline import main

args = (["--llm"] if USAR_LLM else []) + (["--jev"] if USAR_JEV else [])
main(args)

# COMMAND ----------

# MAGIC %md ## Tablas Delta

# COMMAND ----------

import json

import pandas as pd

from cluster3 import config


def to_delta(pdf: pd.DataFrame, name: str) -> None:
    pdf = pdf.copy()
    for c in pdf.columns:  # columnas anidadas (listas, dicts) → JSON
        if pdf[c].dtype == object and pdf[c].map(lambda v: isinstance(v, (list, dict))).any():
            pdf[c] = pdf[c].map(lambda v: json.dumps(v, ensure_ascii=False, default=str))
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{CAT}.{SCH}.{name}"))
    print(f"{CAT}.{SCH}.{name}: {len(pdf)} filas")


to_delta(pd.read_parquet(config.T_SCORES), "clientes_c3_scores")
to_delta(pd.read_parquet(config.T_LLAMADAS_ANALISIS), "llamadas_analisis")
for t in ["segmentos_cluster3", "accionables", "lift_intencion", "lift_churn", "balance_importancia_targets",
          "registro_decisiones_variables"]:
    to_delta(pd.read_csv(config.OUT_TAB / f"{t}.csv"), t)

# COMMAND ----------

# MAGIC %md ## Modelos en Unity Catalog

# COMMAND ----------

import pickle

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
scores = pd.read_parquet(config.T_SCORES)
for key in ["intencion", "churn"]:
    with open(config.MODELS_DIR / f"modelo_{key}.pkl", "rb") as f:
        b = pickle.load(f)
    X = scores[b["features"]].head(5)
    with mlflow.start_run(run_name=f"registro_{key}"):
        mlflow.sklearn.log_model(b["model"], name=f"modelo_{key}", input_example=X,
                                 registered_model_name=f"{CAT}.{SCH}.modelo_{key}")

# COMMAND ----------

# MAGIC %md ## Artefactos para la app y limpieza

# COMMAND ----------

from cluster3.deploy.volume_sync import push

print("Archivos en el Volume:", push(VOL_ART, LOCAL))
shutil.rmtree(raw)  # los Excel originales solo viven en el Volume raw
