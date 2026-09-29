# 07 · Accionables priorizados e impacto económico

| id | accionable | tipo | prioridad | evidencia | metrica |
|---|---|---|---|---|---|
| A1 | Lista semanal del decil de mayor riesgo de churn para contacto proactivo | Proactivo | 1 · Quick win | En validación cruzada, el decil 1 del modelo concentra 96 de 104 bajas (92 %) contactando al 10 % de la base; se contacta a los 1.756 con valor esperado positivo, que concentran 92 bajas | Churn del decil contactado vs grupo de control |
| A2 | Alerta de subida de factura: contactar antes de que llegue el incremento | Proactivo | 2 · Estratégico | Con renta +5 % vs 6 meses el churn es 0,87 % vs 0,40 %; 5.178 clientes con subida | Churn a 60 días en clientes con incremento; tasa de aceptación del ajuste |
| A3 | Ajustar el plan a lo que el cliente usa (retirar TV, decos o adicionales) en lugar de descontar | Reactivo | 2 · Estratégico | 2.473 clientes con intención tienen renta menor que hace 6 meses (−$62,4 M/mes); con equipos adicionales el churn es 7,8 % | Renta retenida neta de descuentos; % retenidos con ajuste vs con descuento |
| A4 | Oferta de retención por motivo (Next Best Offer) y sin descuento adicional a cazadores de ofertas | Reactivo | 3 · Complementario | 967 cazadores de ofertas: churn 0,0 %, intención 11,9 %; negocian pero no se van. Descuento promedio implícito: $25.226/mes | Tasa de retención por motivo; costo de retención por cliente salvado |
| A5 | Coaching de agentes de retención con la trayectoria de sentimiento de sus llamadas | Reactivo | 3 · Complementario | En el 27 % de las llamadas del Cluster 3 el sentimiento empeora hacia el final | % de llamadas con sentimiento que mejora; retención por agente |
| A6 | Resolver cobros no reconocidos en el primer contacto, con autonomía del agente para reversar | Reactivo | 4 · Evaluar | Con 2+ reclamos en el mes el churn es 1,25 % vs 0,31 %; precio y facturación es el motivo del 24 % de las llamadas del Cluster 3 | Resolución en primer contacto; reclamos repetidos a 30 días |

## Impacto anual estimado (COP)

| id | impacto_anual_conservador_cop | impacto_anual_base_cop | impacto_anual_optimista_cop |
|---|---|---|---|
| A1 | $-9,0 M | $8,4 M | $25,8 M |
| A2 | $-1,5 M | $7,0 M | $15,4 M |
| A3 | $74,9 M | $149,7 M | $224,6 M |
| A4 | $5,2 M | $10,4 M | $15,7 M |
| A5 |  |  |  |
| A6 | $10,1 M | $20,1 M | $30,2 M |

Supuestos: tasa de éxito por escenario {'conservador': 0.15, 'base': 0.3, 'optimista': 0.45}, horizonte 12 meses, costo de contacto proactivo $15.000. Los reactivos no tienen costo de contacto (el cliente ya llamó). El impacto real se mide con grupo de control.

Matriz impacto/esfuerzo: `outputs/figures/matriz_impacto_esfuerzo.png`.