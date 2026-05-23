from src.domain.entities import (
    AlertEngine,
    BacSi,
    CanhBao,
    HoSoLamSang,
    KetQuaDuDoan,
    LogisticRegressionModel,
    MoHinhDuDoan,
    NguoiDung,
    QuanTriVien,
    RandomForestModel,
    XGBoostModel,
)


def clinical_rec():
    return HoSoLamSang(
        ma_ho_so="hs1",
        bmi=25.0,
        sp=120,
        bp=80,
        hba1c=7.1,
        fps=100,
        pps=130,
        family_history=0,
        onset_age=40,
        dia_life=2,
        smoking=0,
        phy_act=1,
        med_use=1,
        med_adh=2,
        age=45,
        sex=1,
    )


def test_bac_si_is_nguoi_dung():
    rec = BacSi("u1", "Doctor", "doctor@test.com", "bac_si", "bs1", "noi_tiet")
    assert isinstance(rec, NguoiDung)


def test_quan_tri_vien_is_nguoi_dung():
    rec = QuanTriVien("u2", "Admin", "admin@test.com", "admin", "ad1")
    assert isinstance(rec, NguoiDung)


def test_logistic_regression_is_model():
    mdl = LogisticRegressionModel("logistic_regression", "v1")
    assert isinstance(mdl, MoHinhDuDoan)


def test_random_forest_is_model():
    mdl = RandomForestModel("random_forest", "v1")
    assert isinstance(mdl, MoHinhDuDoan)


def test_xgboost_is_model():
    mdl = XGBoostModel("xgboost", "v1")
    assert isinstance(mdl, MoHinhDuDoan)


def test_ho_so_to_feature_dict_has_15_features():
    ft = clinical_rec().to_feature_dict()
    assert len(ft) == 15
    assert set(ft) == {
        "age",
        "sex",
        "bmi",
        "sp",
        "bp",
        "hba1c",
        "fps",
        "pps",
        "family_h",
        "onset_age",
        "dia_life",
        "smoking",
        "phy_act",
        "med_use",
        "med_adh",
    }


def test_ket_qua_to_dict_has_5_scores():
    rec = KetQuaDuDoan("kq1", 0.1, 0.2, 0.3, 0.4, 0.5, "2026-05-14")
    data = rec.to_dict()
    scores = [k for k in data if k.endswith("_score")]
    assert len(scores) == 5


def test_canh_bao_to_dict_runs():
    rec = CanhBao("cb1", "high", "risk", "2026-05-14")
    data = rec.to_dict()
    assert data["ma_canh_bao"] == "cb1"


def test_alert_engine_has_threshold():
    rec = AlertEngine(0.5)
    assert rec.threshold == 0.5
