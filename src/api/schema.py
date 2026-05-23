from typing import Optional

from pydantic import BaseModel


class PredictInput(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = None
    age: int
    sex: int
    bmi: float
    sp: float
    bp: float
    hba1c: float
    fps: float
    pps: float
    family_h: int
    onset_age: int
    dia_life: float
    smoking: int
    phy_act: int
    med_use: int
    med_adh: int


class PredictOutput(BaseModel):
    patient_id: str
    prediction_id: str
    risk_scores: dict
    risk_labels: dict
    risk_levels: dict
    warning_level: str
    warning_message: str
    model_version: str
    timestamp: str
