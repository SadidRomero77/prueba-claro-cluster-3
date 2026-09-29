# Supuestos y decisiones

Registro de las decisiones que cambian los resultados. Cada una está implementada en código
(`src/cluster3/config.py` concentra las listas) y el detalle por variable está en
`outputs/tables/registro_decisiones_variables.csv`.

## Datos

| # | Decisión | Por qué | Dónde |
|---|---|---|---|
| D1 | Se analiza solo el Cluster 3 (20.000 clientes, periodo 202508) | Es el clúster crítico definido en la prueba | `data/load.py` |
| D2 | Sin validación temporal | Hay un solo periodo; se usa validación cruzada estratificada y se declara como limitación | `models/churn.py` |
| D3 | `VAL_SCORE_CREDITICIO` se reescala a 0–1000 (`SCORE_CREDITICIO_FIX`) | Va de 400 a 124.377.815: el separador decimal se perdió | `data/quality.py` |
| D4 | `VAL_DOWNTIME` negativo → nulo; `VAL_DIAS_PAGO` y `VAL_SALDO_ACTUAL` recortados p1–p99 | Valores imposibles o extremos | `data/clean.py` |
| D5 | Se eliminan 16 variables constantes (2 de ellas 100 % vacías) | No aportan información; `BAN_DESPOSICIONADO_*` es constante en 0 | `data/clean.py` |
| D6 | `BAN_INTENCION_CANCELACION` es del mes; `CANTIDAD_INTENCIONES` es histórica | 19.327 clientes con intenciones históricas vs 3.981 con intención en el mes | supuesto a validar |

## Fuga de información

| # | Variables excluidas | Evidencia | Modelo |
|---|---|---|---|
| F1 | `ESTADO_FUENTE_A/C/MAS/MENOS` | `ESTADO_FUENTE_C` es idéntica a `BAN_CHURN` | ambos |
| F2 | `TIPO_TV_DIGITAL_PI`, `TIPO_TV_DIGITAL_BI` | Los 103 clientes con sufijo I tienen cuenta no activa (`ESTADO_FUENTE_A = 0`); son 103 de las 159 cuentas no activas y 66 de las 104 bajas. El sufijo codifica el estado, no el plan | ambos |
| F3 | `CANTIDAD_INTENCIONES`, `BAN_REINCIDENTE_30/60`, `MOTIVO_LLAM_CANCELA` | El 100 % de los reincidentes a 30 días tiene intención = 1 | intención (en churn sí se usan: ocurren antes de la baja) |
| F4 | `BAN_RETENCION_ACTIVA`, `BAN_CAMPANA_*` (incluida `BAN_CAMPANA_VENTA`), `VAL_CAMPANA*`, `VAL_OFER_ANTERIORES` | Son la reacción de la compañía a la intención. `BAN_CAMPANA_VENTA` describe el tipo de la misma campaña del mes; con un solo corte no se sabe si fue antes o después de la llamada (intención 46,4 % con campaña de venta vs 15,4 % sin ella) | ambos |
| F5 | `VAL_SALDO_ACTUAL` | Es 0 en los 104 clientes con churn vs 33 % con saldo en el resto: la cuenta se liquida al cancelar | churn |

Con estas variables ambos modelos llegan a AUC 1,0 (`outputs/tables/comparacion_fuga.csv`): predicen el resultado
porque ya lo contienen, no porque lo anticipen.

**Corrección posterior al primer análisis.** En la primera versión, el plan TV Digital PI aparecía como el hallazgo
principal (92 clientes con 64 % de churn y un accionable propio). Al cruzarlo con el estado de la cuenta se vio que todos
esos clientes están inactivos, así que se reclasificó como fuga (F2), se retiró el accionable y se reentrenó el modelo
de churn (AUC 0,991 → 0,970).

## Señales a validar con Claro

| Variable | Observación | Tratamiento |
|---|---|---|
| `VAL_EQUIP_ADIC`, `VAL_UW` | Más altos en cuentas no activas (0,93 y 0,55 vs 0,08 y 0,04 en activas) | Se mantienen; sin ellas el churn da AUC 0,951 (sensibilidad en `metricas_modelo.json`) |
| `MOTIVO_LLAM_CANCELA` | = 1 en 2.220 clientes pero solo 132 tienen intención | Variable de contacto, no etiqueta |
| Sufijo I de los planes de TV | Hipótesis: I = inactivo | Confirmar el significado antes de usarlo en cualquier regla |

## Modelado

| # | Decisión | Por qué |
|---|---|---|
| M1 | Dos modelos: intención (alerta temprana) y churn (baja efectiva) | 19,9 % llama a cancelar y solo 0,52 % se va; son problemas distintos |
| M2 | LightGBM con `scale_pos_weight`, sin SMOTE | Conserva la distribución real; la calibración posterior corrige las probabilidades |
| M3 | Validación cruzada estratificada repetida 3×5 e IC 95 % por bootstrap | 104 positivos en churn: una sola partición es inestable |
| M4 | Calibración isotónica (intención) y sigmoide (churn) | Pocos positivos en churn: la isotónica sobreajusta |
| M5 | Deciles y listas se calculan con probabilidades fuera de muestra | Con las de entrenamiento el decil 1 capturaba 104 de 104 bajas (optimista) |
| M6 | Umbral por valor esperado: 12 meses de renta, 30 % de éxito, $15.000 por contacto | Supuestos de negocio a validar con Claro |
| M7 | SHAP por fold con estabilidad y agregado por categoría del diccionario | Importancia robusta y legible para negocio |

## NLP de llamadas

| # | Decisión | Por qué |
|---|---|---|
| N1 | Segunda pasada de anonimización antes de cualquier LLM | Quedaban nombres después de "mi nombre es" y números largos (81 reemplazos) |
| N2 | Corrección de roles por guion del agente | 158 llamadas con AGENT/CLIENT invertidos |
| N3 | Si la diarización falla, se usa el texto completo y se marca `diarizacion_ok = False` | 78 llamadas; descartarlas sesgaba la muestra |
| N4 | Taxonomía v1.0 con 8 motivos y submotivos | Fija y versionada: los tres clasificadores (reglas, LLM, Jev) responden igual |
| N5 | La evidencia del LLM debe ser cita literal; si no, un reintento | Evita motivos sin sustento |
| N6 | Evaluación contra 50 llamadas etiquetadas a mano (40 del Cluster 3 + 10 de otros) | Es la única forma de elegir método con datos |
| N7 | Llamadas y dataset se cruzan en agregado | No hay llave común a nivel cliente |

## Negocio

| # | Supuesto | Valor | Cómo se valida |
|---|---|---|---|
| B1 | ARPU = `VAL_RENTA_ACTUAL` | Mediana $94.841 | Confirmar con finanzas |
| B2 | Horizonte de renta salvada | 12 meses | Supervivencia real de retenidos |
| B3 | Tasa de éxito de retención | 15 % / 30 % / 45 % | Grupo de control en el piloto |
| B4 | Costo de contacto proactivo | $15.000 | Costo real del canal |
| B5 | Contactos reactivos sin costo incremental | El cliente ya llamó | — |
| B6 | En accionables proactivos solo se contacta a quien tiene valor esperado positivo | p_churn fuera de muestra × 30 % × ARPU × 12 > $15.000 | Grupo de control en el piloto |

Contactar a todo un segmento incluía clientes cuyo contacto cuesta más de lo que salva: la lista del decil 1 y la
alerta de subida de factura daban impacto negativo. Con la regla B6 ambas son positivas en el escenario base; en el
conservador (15 % de éxito) siguen negativas, por eso se proponen como piloto con grupo de control.
