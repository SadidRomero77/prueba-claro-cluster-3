"""Reglas de calidad y exclusión de variables. Usan datos sintéticos: no dependen de data/raw."""
import numpy as np
import pandas as pd

from cluster3 import config
from cluster3.data.clean import model_features
from cluster3.data.quality import fix_score_crediticio, univariate_auc


def test_univariate_auc_detecta_fuga():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    df = pd.DataFrame({"target": y, "copia": y, "ruido": rng.normal(size=500)})
    res = univariate_auc(df, "target").set_index("variable")
    assert res.loc["copia", "auc_univariado"] == 1.0
    assert bool(res.loc["copia", "identica_al_target"])
    assert res.loc["ruido", "auc_univariado"] < 0.6


def test_score_crediticio_vuelve_a_escala_0_1000():
    s = pd.Series([8763653, 400, 999.9, 124377815, np.nan])
    out = fix_score_crediticio(s)
    assert out.dropna().between(0, 1000).all()
    assert out.iloc[1] == 400
    assert round(out.iloc[0], 3) == 876.365
    assert np.isnan(out.iloc[4])


def test_model_features_excluye_fuga_y_post_evento():
    cols = (config.ID_COLS + [config.TARGET_CHURN, config.TARGET_INTENCION] + config.LEAK_ESTADO
            + config.LEAK_ESTADO_TV + config.LEAK_INTENCION + config.POST_EVENTO + ["VAL_RECLAMOS_MES", "VAL_VAR_RENTA"])
    df = pd.DataFrame({c: [0, 1] for c in cols})
    feats = model_features(df)
    assert set(feats) == {"VAL_RECLAMOS_MES", "VAL_VAR_RENTA"}


def test_listas_de_fuga_no_se_solapan_con_targets():
    fuga = set(config.LEAK_ESTADO + config.LEAK_ESTADO_TV + config.LEAK_INTENCION + config.POST_EVENTO
               + config.POST_EVENTO_CHURN)
    assert config.TARGET_CHURN not in fuga and config.TARGET_INTENCION not in fuga
    assert "TIPO_TV_DIGITAL_PI" in fuga and "VAL_SALDO_ACTUAL" in fuga
