from dataclasses import dataclass


@dataclass
class NguoiDung:
    id: str
    ho_ten: str
    email: str
    vai_tro: str

    def dang_nhap(self):
        return f"{self.ho_ten} dang_nhap"

    def dang_xuat(self):
        return f"{self.ho_ten} dang_xuat"


@dataclass
class BacSi(NguoiDung):
    ma_bac_si: str
    khoa: str

    def xem_danh_sach_benh_nhan(self):
        return []

    def xem_chi_tiet_benh_nhan(self):
        return {}

    def xem_canh_bao(self):
        return []

    def xem_lich_su_du_doan(self):
        return []


@dataclass
class QuanTriVien(NguoiDung):
    ma_admin: str

    def xem_monitoring(self):
        return {"status": "ok"}

    def kich_hoat_retrain(self):
        return "retrain_requested"

    def quan_ly_model_version(self):
        return "model_version"


@dataclass
class BenhNhan:
    ma_benh_nhan: str
    ten: str
    tuoi: int
    gioi_tinh: int

    def cap_nhat_ho_so(self):
        return "updated"

    def lay_thong_tin_co_ban(self):
        return {
            "ma_benh_nhan": self.ma_benh_nhan,
            "ten": self.ten,
            "tuoi": self.tuoi,
            "gioi_tinh": self.gioi_tinh,
        }


@dataclass
class HoSoLamSang:
    ma_ho_so: str
    bmi: float
    sp: float
    bp: float
    hba1c: float
    fps: float
    pps: float
    family_history: float
    onset_age: float
    dia_life: float
    smoking: float
    phy_act: float
    med_use: float
    med_adh: float
    age: float | None = None
    sex: float | None = None

    def kiem_tra_hop_le(self):
        return self.ma_ho_so != ""

    def to_feature_dict(self):
        return {
            "age": self.age,
            "sex": self.sex,
            "bmi": self.bmi,
            "sp": self.sp,
            "bp": self.bp,
            "hba1c": self.hba1c,
            "fps": self.fps,
            "pps": self.pps,
            "family_h": self.family_history,
            "onset_age": self.onset_age,
            "dia_life": self.dia_life,
            "smoking": self.smoking,
            "phy_act": self.phy_act,
            "med_use": self.med_use,
            "med_adh": self.med_adh,
        }


@dataclass
class FeatureVector:
    ma_feature: str
    features: dict

    def lay_feature(self):
        return self.features


@dataclass
class MoHinhDuDoan:
    model_name: str
    model_version: str

    def train(self):
        return f"{self.model_name} train"

    def predict(self):
        return {}

    def evaluate(self):
        return {}

    def save_model(self):
        return f"{self.model_name}:{self.model_version}"


@dataclass
class LogisticRegressionModel(MoHinhDuDoan):
    pass


@dataclass
class RandomForestModel(MoHinhDuDoan):
    pass


@dataclass
class XGBoostModel(MoHinhDuDoan):
    pass


@dataclass
class KetQuaDuDoan:
    ma_ket_qua: str
    nep_score: float
    neu_score: float
    ret_score: float
    cv_score: float
    per_vas_score: float
    thoi_gian: str

    def lay_risk_level(self):
        score = max(self.nep_score, self.neu_score, self.ret_score, self.cv_score, self.per_vas_score)
        if score >= 0.8:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def to_dict(self):
        return {
            "ma_ket_qua": self.ma_ket_qua,
            "nep_score": self.nep_score,
            "neu_score": self.neu_score,
            "ret_score": self.ret_score,
            "cv_score": self.cv_score,
            "per_vas_score": self.per_vas_score,
            "thoi_gian": self.thoi_gian,
        }


@dataclass
class CanhBao:
    ma_canh_bao: str
    muc_canh_bao: str
    noi_dung: str
    thoi_gian: str

    def tao_canh_bao(self):
        return self.noi_dung

    def to_dict(self):
        return {
            "ma_canh_bao": self.ma_canh_bao,
            "muc_canh_bao": self.muc_canh_bao,
            "noi_dung": self.noi_dung,
            "thoi_gian": self.thoi_gian,
        }


@dataclass
class AlertEngine:
    threshold: float

    def kiem_tra_nguy_co(self, score):
        return score >= self.threshold

    def tao_thong_diep(self):
        return f"risk_threshold {self.threshold}"


@dataclass
class MonitoringService:
    service_name: str

    def kiem_tra_drift(self):
        return {"drift": False}

    def kiem_tra_performance(self):
        return {"performance": "ok"}


@dataclass
class RetrainingPipeline:
    pipeline_id: str

    def chay_retrain(self):
        return "retrain_started"

    def dang_ky_model_moi(self):
        return "model_registered"

    def promote_production(self):
        return "production_promoted"
