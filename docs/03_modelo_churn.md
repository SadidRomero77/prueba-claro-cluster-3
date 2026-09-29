# 03 · Modelo de propensión en dos etapas

Etapa 1 predice la **intención** (alerta temprana). Etapa 2 predice el **churn** (baja efectiva) e incluye las señales de intención, que ocurren antes de la baja.

## Métricas (validación cruzada estratificada, fuera de muestra)

| modelo | positivos | tasa base | AUC (CV 3×5) | IC 95 % AUC | PR-AUC | LIFT@5 % | LIFT@10 % | LIFT@20 % |
|---|---|---|---|---|---|---|---|---|
| intencion | 3981 | 19,91 % | 0.797 ± 0.009 | 0.789–0.805 | 0.566 | 4.1 | 3.4 | 2.6 |
| churn | 104 | 0,52 % | 0.974 ± 0.012 | 0.967–0.984 | 0.588 | 16.9 | 9.2 | 5.0 |

## Por qué se excluyen variables: con fuga vs sin fuga

| modelo | variables | n_variables | auc | pr_auc | lift_10 |
|---|---|---|---|---|---|
| intencion | con fuga | 111 | 1 | 1 | 5.02 |
| intencion | sin fuga (modelo final) | 93 | 0.797 | 0.567 | 3.38 |
| churn | con fuga | 112 | 1 | 1 | 10 |
| churn | sin fuga (modelo final) | 97 | 0.976 | 0.598 | 9.23 |

Los planes de TV con sufijo I (`TIPO_TV_DIGITAL_PI`, `TIPO_TV_DIGITAL_BI`) se excluyen: el 100 % son cuentas no activas, así que codifican el estado de la cuenta igual que `ESTADO_FUENTE_*`.

Sensibilidad: el churn sin equipos adicionales ni UltraWiFi (`VAL_EQUIP_ADIC`, `VAL_UW`, más altos en cuentas no activas) da AUC 0.951 y LIFT@10 8.8. Ambas variables quedan como señales a validar con Claro.

## Matriz de confusión · intencion

| Umbral | VP | FP | FN | VN | Precisión | Recall |
|---|---|---|---|---|---|---|
| Top 10 % (0.713) | 1344 | 656 | 2637 | 15363 | 67,2 % | 33,8 % |
| Negocio (0.379) | 3296 | 6704 | 685 | 9315 | 33,0 % | 82,8 % |

Umbral de negocio: maximiza ARPU × 12 meses × 30 % de éxito − $15.000 por contacto.

## Matriz de confusión · churn

| Umbral | VP | FP | FN | VN | Precisión | Recall |
|---|---|---|---|---|---|---|
| Top 10 % (0.033) | 96 | 1904 | 8 | 17992 | 4,8 % | 92,3 % |
| Negocio (0.526) | 68 | 200 | 36 | 19696 | 25,4 % | 65,4 % |

Umbral de negocio: maximiza ARPU × 12 meses × 30 % de éxito − $15.000 por contacto.

## Balance de importancia entre targets (SHAP)

Estructural = importante en ambos modelos · Negociación = solo intención · Salida = solo churn.

| variable | rank_intencion | rank_churn | tipo_palanca |
|---|---|---|---|
| SCORE_CREDITICIO_FIX | 2 | 14 | Estructural (ambos) |
| VAL_VAR_RENTA | 3 | 10 | Estructural (ambos) |
| ANTIGUEDAD_MESES | 4 | 13 | Estructural (ambos) |
| VAL_FRECUENCIA_COMPRA | 6 | 17 | Estructural (ambos) |
| VAL_RENTA_BSC_IVA | 9 | 7 | Estructural (ambos) |
| VAL_LLAM_ADTIVAS_NEGATIVAS | 10 | 5 | Estructural (ambos) |
| VELOCIDAD_INTERNET_MBPS | 11 | 2 | Estructural (ambos) |
| VAL_RECLAMOS_MES | 12 | 1 | Estructural (ambos) |
| VAL_RENTA_ADIC_IVA | 15 | 15 | Estructural (ambos) |
| VAL_RENTA_ACTUAL | 18 | 4 | Estructural (ambos) |
| VAL_LLAM_ADTIVAS_NEUTRAS | 1 | 21 | Negociación (solo intención) |
| BAN_CAZA_OFERTA | 5 | 34 | Negociación (solo intención) |
| VAL_RENTA_M6 | 7 | 24 | Negociación (solo intención) |
| TIPO_TV_DIGITAL_P | 8 | 55 | Negociación (solo intención) |
| VAL_DECO_BAS | 13 | 33 | Negociación (solo intención) |
| MOTIVO_LLAM_INFO_GRAL | 14 | 28 | Negociación (solo intención) |
| VAL_EQUIP_BAS | 16 | 22 | Negociación (solo intención) |
| VAL_SUM_VAL_USO_REDES_SOCIALES | 17 | 29 | Negociación (solo intención) |
| BAN_HBO | 19 | 32 | Negociación (solo intención) |
| VAL_SUM_VAL_SMS_ENVIADOS | 20 | 25 | Negociación (solo intención) |
| VAL_SUM_VAL_MINUTOS_VOZ | 21 | 6 | Salida (solo churn) |
| VAL_DIAS_PAGO | 23 | 20 | Salida (solo churn) |
| VAL_DENSIDAD_ZONA | 25 | 16 | Salida (solo churn) |
| VAL_SUM_VAL_USO_CAT_APPS_PREFERIDAS | 29 | 18 | Salida (solo churn) |
| VAL_DOWNTIME | 33 | 12 | Salida (solo churn) |
| VAL_EQUIP_ADIC | 63 | 9 | Salida (solo churn) |
| BAN_INTENCION_CANCELACION |  | 3 | Salida (solo churn) |
| BAN_REINCIDENTE_30 |  | 19 | Salida (solo churn) |
| BAN_REINCIDENTE_60 |  | 11 | Salida (solo churn) |
| CANTIDAD_INTENCIONES |  | 8 | Salida (solo churn) |

## Importancia por categoría del diccionario (% del SHAP total)

Intención:

| categoria | pct_importancia |
|---|---|
| Facturación & Cartera | 33.9 |
| Reclamos, Llamadas & Soporte | 25.9 |
| Servicios & Productos | 8.8 |
| Identificación & Demografía | 8.1 |
| Servicio Móvil | 7.2 |
| Segmentación | 4.8 |
| Equipos & Dispositivos (Hogar) | 4.6 |
| Campañas & Retención (NBO/NBA) | 4.4 |
| Zona, Nodo & Calidad de Red | 2.1 |
| Intención de Cancelación & Churn | 0.1 |
| Scores & Modelos Analíticos | 0 |

Churn:

| categoria | pct_importancia |
|---|---|
| Reclamos, Llamadas & Soporte | 21.2 |
| Facturación & Cartera | 20.9 |
| Intención de Cancelación & Churn | 17.8 |
| Servicios & Productos | 12.1 |
| Servicio Móvil | 10.5 |
| Equipos & Dispositivos (Hogar) | 6 |
| Zona, Nodo & Calidad de Red | 5.4 |
| Identificación & Demografía | 3.6 |
| Segmentación | 1.8 |
| Campañas & Retención (NBO/NBA) | 0.7 |
| Scores & Modelos Analíticos | 0 |

## Desbalance

- Pesos de clase (`scale_pos_weight`) en lugar de SMOTE: con 104 positivos SMOTE fabrica clientes casi idénticos.
- Métricas que no se inflan con el desbalance: PR-AUC y LIFT por decil.
- Probabilidades calibradas (isotónica en intención, sigmoide en churn) para poder calcular impacto en pesos.

Figuras: `outputs/figures/ganancia_*.png`, `outputs/figures/shap_*.png`.