# 03 · Modelo de propensión en dos etapas

Etapa 1 predice la **intención** (alerta temprana). Etapa 2 predice el **churn** (baja efectiva) e incluye las señales de intención, que ocurren antes de la baja.

## Métricas (validación cruzada estratificada, fuera de muestra)

| modelo | positivos | tasa base | AUC (CV 3×5) | IC 95 % AUC | PR-AUC | LIFT@5 % | LIFT@10 % | LIFT@20 % |
|---|---|---|---|---|---|---|---|---|
| intencion | 3981 | 19,91 % | 0.813 ± 0.009 | 0.806–0.821 | 0.625 | 4.6 | 3.7 | 2.7 |
| churn | 104 | 0,52 % | 0.970 ± 0.016 | 0.960–0.983 | 0.596 | 17.1 | 8.9 | 4.9 |

## Por qué se excluyen variables: con fuga vs sin fuga

| modelo | variables | n_variables | auc | pr_auc | lift_10 |
|---|---|---|---|---|---|
| intencion | con fuga | 111 | 1 | 1 | 5.02 |
| intencion | sin fuga (modelo final) | 94 | 0.814 | 0.625 | 3.72 |
| churn | con fuga | 112 | 1 | 1 | 10 |
| churn | sin fuga (modelo final) | 98 | 0.972 | 0.621 | 8.94 |

Los planes de TV con sufijo I (`TIPO_TV_DIGITAL_PI`, `TIPO_TV_DIGITAL_BI`) se excluyen: el 100 % son cuentas no activas, así que codifican el estado de la cuenta igual que `ESTADO_FUENTE_*`.

Sensibilidad: el churn sin equipos adicionales ni UltraWiFi (`VAL_EQUIP_ADIC`, `VAL_UW`, más altos en cuentas no activas) da AUC 0.964 y LIFT@10 9.0. Ambas variables quedan como señales a validar con Claro.

## Matriz de confusión · intencion

| Umbral | VP | FP | FN | VN | Precisión | Recall |
|---|---|---|---|---|---|---|
| Top 10 % (0.732) | 1481 | 519 | 2500 | 15500 | 74,1 % | 37,2 % |
| Negocio (0.352) | 3333 | 6667 | 648 | 9352 | 33,3 % | 83,7 % |

Umbral de negocio: maximiza ARPU × 12 meses × 30 % de éxito − $15.000 por contacto.

## Matriz de confusión · churn

| Umbral | VP | FP | FN | VN | Precisión | Recall |
|---|---|---|---|---|---|---|
| Top 10 % (0.036) | 93 | 1907 | 11 | 17989 | 4,7 % | 89,4 % |
| Negocio (0.502) | 73 | 195 | 31 | 19701 | 27,2 % | 70,2 % |

Umbral de negocio: maximiza ARPU × 12 meses × 30 % de éxito − $15.000 por contacto.

## Balance de importancia entre targets (SHAP)

Estructural = importante en ambos modelos · Negociación = solo intención · Salida = solo churn.

| variable | rank_intencion | rank_churn | tipo_palanca |
|---|---|---|---|
| VAL_VAR_RENTA | 3 | 11 | Estructural (ambos) |
| SCORE_CREDITICIO_FIX | 4 | 17 | Estructural (ambos) |
| VAL_FRECUENCIA_COMPRA | 6 | 19 | Estructural (ambos) |
| VAL_RECLAMOS_MES | 7 | 1 | Estructural (ambos) |
| ANTIGUEDAD_MESES | 8 | 14 | Estructural (ambos) |
| VAL_RENTA_BSC_IVA | 10 | 6 | Estructural (ambos) |
| VAL_LLAM_ADTIVAS_NEGATIVAS | 11 | 8 | Estructural (ambos) |
| VELOCIDAD_INTERNET_MBPS | 14 | 2 | Estructural (ambos) |
| VAL_SUM_VAL_USO_REDES_SOCIALES | 16 | 18 | Estructural (ambos) |
| VAL_RENTA_ADIC_IVA | 18 | 15 | Estructural (ambos) |
| VAL_RENTA_ACTUAL | 19 | 4 | Estructural (ambos) |
| VAL_EQUIP_BAS | 20 | 20 | Estructural (ambos) |
| VAL_LLAM_ADTIVAS_NEUTRAS | 1 | 21 | Negociación (solo intención) |
| BAN_CAMPANA_VENTA | 2 | 45 | Negociación (solo intención) |
| BAN_CAZA_OFERTA | 5 | 36 | Negociación (solo intención) |
| TIPO_TV_DIGITAL_P | 9 | 52 | Negociación (solo intención) |
| VAL_DECO_BAS | 12 | 35 | Negociación (solo intención) |
| VAL_RENTA_M6 | 13 | 25 | Negociación (solo intención) |
| BAN_HBO | 15 | 34 | Negociación (solo intención) |
| MOTIVO_LLAM_INFO_GRAL | 17 | 28 | Negociación (solo intención) |
| VAL_SUM_VAL_MINUTOS_VOZ | 24 | 7 | Salida (solo churn) |
| VAL_DENSIDAD_ZONA | 26 | 12 | Salida (solo churn) |
| VAL_DOWNTIME | 28 | 13 | Salida (solo churn) |
| VAL_EQUIP_ADIC | 67 | 9 | Salida (solo churn) |
| BAN_INTENCION_CANCELACION |  | 3 | Salida (solo churn) |
| BAN_REINCIDENTE_30 |  | 16 | Salida (solo churn) |
| BAN_REINCIDENTE_60 |  | 10 | Salida (solo churn) |
| CANTIDAD_INTENCIONES |  | 5 | Salida (solo churn) |

## Importancia por categoría del diccionario (% del SHAP total)

Intención:

| categoria | pct_importancia |
|---|---|
| Facturación & Cartera | 30.2 |
| Reclamos, Llamadas & Soporte | 24.8 |
| Campañas & Retención (NBO/NBA) | 14.9 |
| Servicios & Productos | 8.7 |
| Servicio Móvil | 6.6 |
| Segmentación | 4.5 |
| Equipos & Dispositivos (Hogar) | 4.1 |
| Identificación & Demografía | 3.9 |
| Zona, Nodo & Calidad de Red | 2.2 |
| Intención de Cancelación & Churn | 0.1 |
| Scores & Modelos Analíticos | 0 |

Churn:

| categoria | pct_importancia |
|---|---|
| Facturación & Cartera | 20.9 |
| Reclamos, Llamadas & Soporte | 19.9 |
| Intención de Cancelación & Churn | 19.3 |
| Servicios & Productos | 11.2 |
| Servicio Móvil | 10.7 |
| Equipos & Dispositivos (Hogar) | 6 |
| Zona, Nodo & Calidad de Red | 5.7 |
| Identificación & Demografía | 3.6 |
| Segmentación | 1.9 |
| Campañas & Retención (NBO/NBA) | 0.8 |
| Scores & Modelos Analíticos | 0 |

## Desbalance

- Pesos de clase (`scale_pos_weight`) en lugar de SMOTE: con 104 positivos SMOTE fabrica clientes casi idénticos.
- Métricas que no se inflan con el desbalance: PR-AUC y LIFT por decil.
- Probabilidades calibradas (isotónica en intención, sigmoide en churn) para poder calcular impacto en pesos.

Figuras: `outputs/figures/ganancia_*.png`, `outputs/figures/shap_*.png`.