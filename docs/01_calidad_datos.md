# 01 · Calidad, consistencia y representatividad de los datos

Dataset: 20.000 filas × 129 variables · duplicados: 0.

## Hallazgos y decisiones

| hallazgo | decisión |
|---|---|
| ESTADO_FUENTE_C es idéntica a BAN_CHURN (estado de la cuenta después de la baja) | Excluir ESTADO_FUENTE_* de los modelos |
| El 100% de los reincidentes a 30 días tiene intención = 1: son un subconjunto del target | No usar reincidencias ni CANTIDAD_INTENCIONES para predecir intención |
| MOTIVO_LLAM_CANCELA = 1 en 2220 clientes, pero solo 132 tienen intención = 1 | Documentar; se trata como variable de contacto, no como etiqueta |
| CANTIDAD_INTENCIONES > 0 en 19327 clientes vs 3981 con intención en el mes | Supuesto: BAN_INTENCION es del mes; CANTIDAD_INTENCIONES es histórica |
| VAL_SCORE_CREDITICIO va de 400 a 124.377.815: el separador decimal se perdió | Reescalar a 0–1000 (SCORE_CREDITICIO_FIX) |
| VAL_DOWNTIME tiene 8 valores negativos (mínimo -1.060) | negativos a nulo |
| VAL_DIAS_PAGO tiene 1.272 valores negativos (mínimo -1.158) | recorte p1–p99 |
| VAL_SALDO_ACTUAL tiene 3.273 valores negativos (mínimo -2.887.532) | recorte p1–p99 |
| VAL_SALDO_ACTUAL = 0 en 104 de 104 clientes con churn, vs 33% con saldo en el resto | La cuenta se liquida al cancelar: se excluye del modelo de churn |
| TIPO_TV_DIGITAL_PI/BI = 1 en 103 clientes, todos con cuenta no activa (ESTADO_FUENTE_A = 0 en 103 de 103); son 103 de las 159 cuentas no activas y 66 de las 104 bajas | El sufijo I codifica el estado de la cuenta: se excluye de ambos modelos (validar con Claro) |
| PERIODO único = [202508] | Sin validación temporal posible; se reporta como limitación |

## Variables sin información

- Constantes: CLUSTER_ID, PERIODO, BAN_DESPOSICIONADO_COMPETENCIA, BAN_DESPOSICIONADO_COMERCIAL, BAN_DESPOSICIONADO_TENOLOGICO, BAN_SUSPENCION_COBRANZA, BAN_MIGRACION_ZONA, VAL_SS_BCART, VAL_LLAMADAS_MES_CALLCENTER, VAL_PENETRACION_NODO, VAL_CHURN_ZONA, BAN_DESPOSICIONADO_NODO, TIPO_TV_CLAROTV_B, TIPO_TV_CLAROTV_PI, TIPO_TV_SATELITAL_S, TIPO_TV_BASICOTV.
- 100 % vacías: VAL_PENETRACION_NODO, VAL_CHURN_ZONA.
- `BAN_DESPOSICIONADO_*` (pedidas en el caso) son constantes en 0: no se pueden interpretar.

## Fuga de información (AUC univariado contra churn)

| variable | auc_univariado | media_positivos | media_negativos | identica_al_target |
|---|---|---|---|---|
| ESTADO_FUENTE_C | 1 | 1 | 0 | True |
| ESTADO_FUENTE_A | 0.999 | 0 | 0.997 | False |
| BAN_RETENCION_ACTIVA | 0.885 | 0.154 | 0.923 | False |
| TIPO_TV_DIGITAL_PI | 0.783 | 0.567 | 0.00166 | False |
| VAL_EQUIP_ADIC | 0.767 | 1.32 | 0.0775 | False |
| VAL_RECLAMOS_MES | 0.745 | 2.25 | 0.979 | False |
| VAL_LLAM_ADTIVAS_NEGATIVAS | 0.695 | 1.37 | 0.317 | False |
| VAL_UW | 0.688 | 0.779 | 0.037 | False |

## Diccionario vs dataset

{'solo_diccionario': 137, 'ambos': 85, 'solo_dataset': 44} — detalle en `outputs/tables/diccionario_reconciliado.csv`.

## Llamadas

- llamadas_por_cluster: {'1': 35, '2': 20, '3': 309, '6': 84, '7': 52}
- largo_caracteres: {'count': 500.0, 'mean': 6991.0, 'std': 4156.0, 'min': 81.0, '25%': 4080.0, '50%': 6280.0, '75%': 9534.0, 'max': 21688.0}
- roles_invertidos: 133 llamadas con guion del agente en turnos CLIENT
- pii_residual: 32 llamadas con posibles nombres sin anonimizar
- llamadas_cortas: 19 llamadas con menos de 1.000 caracteres
- sin_llave: Las llamadas no tienen CUENTA ni DOCUMENTO: no se pueden unir individualmente al dataset

## Registro de decisiones por variable

130 variables con decisión documentada: usar = 90, excluir_modelo = 19, eliminar = 15, transformar = 4, target = 2. Detalle en `outputs/tables/registro_decisiones_variables.csv`.