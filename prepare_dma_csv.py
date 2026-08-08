"""
prepare_dma_csv.py  (v2 — RTU Inlet-Only Leakage & Burst Detection Engine)
---------------------------------------------------------------------------
ตามสเปก "RTU INLET-ONLY LEAKAGE & BURST DETECTION SYSTEM ARCHITECTURE"

ทำทั้งหมดฝั่ง Python:
  1. โหลด config รายตู้ RTU จาก rtu_configs.csv (มี fallback ค่า default อัตโนมัติ)
  2. เตรียมข้อมูล 15 นาที (Q, P) — clean + rolling 30-day baseline ต่อช่วงเวลา (time-of-day)
  3. คำนวณ MNF รายวัน + Mann-Kendall trend test + CUSUM (ตามพารามิเตอร์ต่อ RTU)
  4. จำแนกเหตุการณ์ 3 กรณี: 🔴 Pipe Burst / 🟠 Developing Leak / 🟡 High Usage
  5. คำนวณ hydraulic metrics เสริม: R_eff, Leak Location Proximity, FAVAD estimate
  6. Export CSV 4 ไฟล์ให้ dashboard โหลดไปแสดงผลอย่างเดียว (ไม่คำนวณเอง)

ติดตั้ง dependency:
    pip install pandas numpy pymannkendall pyarrow

⚠️ หมายเหตุความไม่แน่นอนของสเปกที่ต้องตัดสินใจเอง (โปร่งใสไว้ตรงนี้ ไม่ใช่เดาเงียบๆ):
  - "P_inlet(t)" ในสูตร R_eff/Loss_Factor: ระบบมีจุดวัดจุดเดียว (RTU ต้นทาง) จึงตีความ
    P_inlet(t) = P_avg(t) (ค่าคาดหวังทางสถิติจาก baseline) ไม่ใช่จุดวัดที่ 2 จริง
  - FAVAD "C" ในสูตร Q_leak = C·P^N1 ไม่มีค่าคาลิเบรตมาให้ในสเปก (ต่างจาก Hazen-Williams C
    ที่เป็นคนละตัวแปรกัน) — จึงประมาณปริมาณรั่วรวมจากส่วนเกิน MNF ที่วัดได้จริง แล้วใช้ P(t)^N1
    เป็น "น้ำหนักกระจายรายชั่วโมง" เท่านั้น ไม่ใช่ค่าสัมบูรณ์จากฟิสิกส์ล้วน ๆ — ควรคาลิเบรต C
    จริงจากเหตุการณ์ท่อแตกที่ยืนยันแล้วในอดีต ถ้าต้องการความแม่นระดับ production
  - Case A ต้องมีเงื่อนไข sustained-flow เกิดร่วมกับ (CUSUM step-up เร็ว หรือ pressure drop)
    ตีความว่าใช้ AND ระหว่าง sustained-flow กับอย่างน้อยหนึ่งใน (CUSUM/Pressure) เพื่อกันสัญญาณเท็จ
"""

import numpy as np
import pandas as pd
import os
import json
import time

from datetime import datetime, timedelta

try:
    import pyarrow  # noqa: F401  — จำเป็นสำหรับ parquet caching (STEP 1) และ predicate pushdown ตอนอ่าน
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

try:
    from tqdm import tqdm
    tqdm.pandas()  # เปิดใช้ .progress_apply() บน pandas Series/DataFrame
    _HAS_TQDM = True
except ImportError:
    # เผื่อเครื่องไม่มี tqdm ติดตั้งไว้ (หรือไม่มีเน็ตให้ pip install ตอนนั้น) — ทำงานต่อได้ปกติ
    # แค่ไม่มี progress bar สวยๆ ให้ดู (แนะนำ: pip install tqdm)
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(f"{desc}... (ติดตั้ง 'pip install tqdm' เพื่อดู progress bar)")
        return iterable
    _HAS_TQDM = False



# ============================================================
# CONFIG
# ============================================================
N_DAYS = 30
INTERVALS_PER_DAY = 96          # ทุก 15 นาที
INTERVAL_MIN = 15
OUTPUT_FLOW_CSV = "flow_log.csv"
OUTPUT_SERIES_CSV = "dma_daily_series.csv"
OUTPUT_SUMMARY_CSV = "dma_status_summary.csv"
RTU_CONFIG_PATH = "rtu_configs.csv"
USE_DEMO_DATA = False
FLOW_LOG_ONLY_FLAGGED = True     # true = flow_log.csv เขียนเฉพาะ DMA ที่ status เป็น watch/alert (สเกลใหญ่ๆ ไฟล์เล็กลงมาก)
                                  # false = เขียนทุก DMA เหมือนเดิม (เหมาะกับ RTU น้อยๆ ที่ยังอยากดูกราฟรายวันของ DMA ปกติได้ทุกตัว)

# --- ปรับความละเอียดของ flow_log.csv (ใช้แค่วาดกราฟใน dashboard เท่านั้น ไม่กระทบการคำนวณ
#     MNF/CUSUM/Mann-Kendall จริงใน compute_metrics() ซึ่งยังใช้ข้อมูลเต็มความละเอียดเสมอ) ---
FLOW_LOG_DAYS_KEPT = 16           # เก็บเฉพาะ N วันล่าสุด/DMA ใน flow_log.csv (dashboard ใช้จริงแค่ 14 วันแรกเป็น
                                  # baseline band + วันล่าสุด 1 วันเป็นเส้นเทียบ = 15 วัน, 16 คือมีกันชน 1 วัน)
                                  # ตั้งเป็น None หรือ 0 = ไม่ตัดวัน เก็บครบ N_DAYS เหมือนเดิม
FLOW_LOG_EXPORT_INTERVAL_MIN = 30  # ความถี่ที่ export เข้า flow_log.csv (นาที) ต้องหารด้วย INTERVAL_MIN ลงตัว
                                    # 15 = ความถี่เต็มเหมือนเดิม (ไม่ลด), 30/60 = ลดขนาดไฟล์ลงอีก 2x/4x
                                    # ช่องที่หายไปฝั่ง dashboard ถูกเติมด้วย fillGaps() ที่มีอยู่แล้ว ไม่ต้องแก้ JS

# --- ปรับความไวของตัวจับ spike ใน flow (ดู fix_spikes_boundary_safe) ---
SPIKE_THRESHOLD_RATIO = 2.0   # ค่าจริงต้องสูงกว่า median ย้อนหลังกี่เท่าถึงจะถือว่าเป็น spike ที่ต้องแก้
SPIKE_WINDOW_SIZE = 5         # จำนวนจุดย้อนหลัง (ไม่รวมจุดปัจจุบัน) ที่ใช้คำนวณ median baseline

RECOVERY_LOOKBACK_DAYS = 10    # จำนวนวันย้อนหลังที่ใช้หา "พีค MNF ล่าสุด" เพื่อเทียบว่าตอนนี้ลงมาจากพีคแค่ไหนแล้ว
RECOVERY_DROP_RATIO = 0.7      # MNF วันล่าสุดต้องต่ำกว่ากี่เท่าของพีคล่าสุดถึงจะถือว่า "กำลังฟื้น/เหตุการณ์จบแล้ว"
                                # (กันเคส flow ขึ้นแล้วลงมาแล้ว เช่นซ่อมเสร็จแล้ว ไม่ให้ CUSUM ที่มี "ความจำ"
                                # ค้างขึ้นเตือนต่อ — ใช้สัดส่วนจากพีค ไม่ใช่แค่เทียบ threshold เฉยๆ เพราะ MNF
                                # อาจยังอยู่ระหว่างลงจากพีค ยังไม่ทันต่ำกว่า threshold แต่ก็เห็นชัดว่ากำลังฟื้นแล้ว)

RTU_IGNORE_CSV = "rtu_ignore.csv"   # ไฟล์เดียวกับที่ dma_nrw_dashboard.html ใช้ซ่อนการแสดงผล (คอลัมน์ rtu_id)
                                     # ตอนนี้ pipeline ฝั่ง Python ก็อ่านไฟล์นี้ด้วย เพื่อข้าม RTU ที่คุณภาพไม่ดี
                                     # (ตรวจแยกไว้แล้วโดยกระบวนการอื่น ไม่ได้ตรวจในไฟล์นี้) ตั้งแต่ต้น pipeline เลย
                                     # ไม่ต้องเสีย compute clean_data()/compute_metrics() ให้ RTU ที่รู้อยู่แล้วว่าไม่ดี

DEFAULT_RTU_CONFIG = {
    "user_type": "RESIDENTIAL",
    "pipe_material": "PVC",
    "favad_n1": 1.15,
    # mnf_start/mnf_end ตรงนี้เป็นแค่ค่า fallback ตอนยังไม่เคยรัน detect_mnf_window.py
    # ปกติค่าจริงต่อ RTU จะถูกคำนวณอัตโนมัติ (จาก median profile 28 วัน + sliding window)
    # แล้วเขียนทับกลับเข้า rtu_configs.csv โดย job แยกต่างหาก detect_mnf_window.py (รันสัปดาห์ละครั้ง)
    "mnf_start": "02:00",
    "mnf_end": "04:00",
    "cusum_k_factor": 0.5,
    "cusum_h_factor": 4.0,
    "burst_intervals": 8,
    "q_std_mult": 2.0,
    "p_drop_bar": 0.25,
    "hard_flow_threshold": 90.0,  # m3/hr — ถ้า flow ณ จุดใดๆ เกินค่านี้ เตือนทันที ไม่ต้องรอสถิติ/สะสม
    "is_active": True,
}

DMA_DEFS = [
    {"code": "DMA-01", "name": "ราชดำริ",   "base": 78,  "seed": 11, "anomaly": None},
    {"code": "DMA-02", "name": "พระราม 4",  "base": 95,  "seed": 23, "anomaly": None},
    {"code": "DMA-03", "name": "เจริญกรุง", "base": 52,  "seed": 37, "anomaly": {"type": "gradual", "start_day": 15, "max_extra": 13}},
    {"code": "DMA-04", "name": "สุขุมวิท",  "base": 110, "seed": 41, "anomaly": None},
    {"code": "DMA-05", "name": "สีลม",      "base": 64,  "seed": 53, "anomaly": None},
    {"code": "DMA-06", "name": "ลาดพร้าว",  "base": 88,  "seed": 59, "anomaly": {"type": "step", "start_day": 27, "extra": 30}},
    {"code": "DMA-07", "name": "บางนา",     "base": 70,  "seed": 61, "anomaly": None},
    {"code": "DMA-08", "name": "ปทุมวัน",   "base": 58,  "seed": 67, "anomaly": {"type": "spike", "day": 18, "extra": 34}},
    {"code": "DMA-09", "name": "อ่อนนุช",   "base": 66,  "seed": 71, "anomaly": {"type": "step", "start_day": 25, "extra": 11}},
    {"code": "DMA-10", "name": "ทองหล่อ",   "base": 82,  "seed": 79, "anomaly": None},
]

# ตัวอย่าง override รายตู้ (สาธิตความสามารถ per-RTU customization ตามสเปกข้อ 5)
CONFIG_OVERRIDES = {
    "DMA-04": {"user_type": "INDUSTRIAL", "q_std_mult": 2.5, "burst_intervals": 12},
    "DMA-08": {"user_type": "COMMERCIAL", "q_std_mult": 2.2},
    "DMA-06": {"pipe_material": "DI", "favad_n1": 0.5},  # ท่อเหล็กเหนียว N1 ต่างจาก PVC
}

DIURNAL_24 = np.array([.55, .45, .35, .32, .34, .45, .65, .95, 1.15, 1.05, .9, .85,
                        .9, .85, .8, .8, .85, .95, 1.1, 1.2, 1.15, .95, .8, .65])
# ขยายจาก 24 จุด (รายชั่วโมง) เป็น 96 จุด (ราย 15 นาที) ด้วย interpolation
_hour_centers = np.arange(24) + 0.5
_interval_centers = np.arange(INTERVALS_PER_DAY) / 4.0
DIURNAL_96 = np.interp(_interval_centers, _hour_centers, DIURNAL_24,
                        period=24)  # wrap รอบเที่ยงคืน


def hhmm_to_interval(s: str) -> int:
    h, m = str(s).split(":")
    return (int(h) * 60 + int(m)) // INTERVAL_MIN


_step_start_time = [None]
_step_count = [0]
_step_total = 6

def step(msg: str):
    """แสดงความคืบหน้าระดับขั้นตอนหลักของ pipeline พร้อมเวลาที่ใช้ในขั้นก่อนหน้า"""
    now = time.time()
    if _step_start_time[0] is not None:
        elapsed = now - _step_start_time[0]
        print(f"      ↳ เสร็จใน {elapsed:.1f} วินาที")
    _step_count[0] += 1
    _step_start_time[0] = now
    print(f"[{_step_count[0]}/{_step_total}] {msg}")


# ============================================================
# STEP 0: RTU CONFIG — โหลดพร้อม fallback ค่า default อัตโนมัติ
# ============================================================
def load_rtu_configs(path: str, dma_defs: list) -> pd.DataFrame:
    """dma_defs: ส่ง [] (list ว่าง) เมื่อใช้ข้อมูลจริง (USE_DEMO_DATA=False) เพื่อไม่ให้ไป
    เติมแถว DMA-01..10 (รหัสตัวอย่างสำหรับโหมดจำลอง) ปนเข้าไปใน config ของ RTU จริง"""
    known_ids = [d["code"] for d in dma_defs]
    known_names = {d["code"]: d["name"] for d in dma_defs}

    if not os.path.exists(path):
        # ยังไม่มีไฟล์ config -> สร้างไฟล์ template ให้ (ทุกช่องเป็น default ยกเว้นตัวอย่าง override)
        rows = []
        for code in known_ids:
            row = {"rtu_id": code, "dma_name": known_names[code], **DEFAULT_RTU_CONFIG}
            row.update(CONFIG_OVERRIDES.get(code, {}))
            rows.append(row)
        cfg = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rtu_id", "dma_name", *DEFAULT_RTU_CONFIG.keys()])
        cfg.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"ไม่พบ {path} — สร้างไฟล์ config เริ่มต้นให้แล้ว (แก้ผ่าน Excel ได้เลยรอบถัดไป)")
        return cfg.set_index("rtu_id")

    cfg = pd.read_csv(path, encoding="utf-8-sig")
    cfg = cfg.loc[:, ~cfg.columns.duplicated()]  # กันคอลัมน์ชื่อซ้ำ (เช่น เปิด/บันทึกผ่าน Excel แล้วเผลอสร้างซ้ำ)
    cfg = cfg.dropna(how="all")               # ตัดแถวว่างล้วน (เช่น แถวท้ายไฟล์ที่ Excel เผลอเซฟติดมา)
    cfg = cfg[cfg["rtu_id"].notna()]          # ตัดแถวที่ rtu_id ว่าง (กัน index เป็น NaN)
    cfg = cfg.set_index("rtu_id")
    n_before = len(cfg)
    dup_ids = cfg.index[cfg.index.duplicated()].unique().tolist()
    if dup_ids:
        print(f"⚠️  พบ rtu_id ซ้ำใน {path}: {dup_ids} — ใช้แถวล่าสุดของแต่ละ rtu_id เท่านั้น กรุณาตรวจสอบไฟล์ config")
        cfg = cfg[~cfg.index.duplicated(keep="last")]
    # เติมแถวที่ config ไม่มี (เฉพาะตอน demo — ของจริงไม่ควรมี DMA-01..10 ปนเข้ามา)
    for code in known_ids:
        if code not in cfg.index:
            row = {"dma_name": known_names[code], **DEFAULT_RTU_CONFIG}
            cfg.loc[code] = row
    # Fallback: ช่องไหน Null/NaN เติม default อัตโนมัติ (กัน crash ตามสเปกข้อ 5.2)
    for key, default_val in DEFAULT_RTU_CONFIG.items():
        if key not in cfg.columns:
            cfg[key] = default_val
        cfg[key] = cfg[key].where(cfg[key].notna(), default_val)
    cfg["is_active"] = cfg["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])

    # Self-heal: เขียนไฟล์ config ที่ทำความสะอาดแล้ว (ตัดซ้ำ/แถวว่างออก) กลับลงดิสก์
    # ป้องกันไม่ให้คำเตือนเรื่องแถวซ้ำขึ้นซ้ำทุกรอบที่รัน ถ้าไม่ได้ไปแก้ไฟล์เองก่อน
    if len(cfg) != n_before or dup_ids:
        cfg.reset_index().to_csv(path, index=False, encoding="utf-8-sig")
        print(f"      ↳ บันทึก {path} เวอร์ชันที่ทำความสะอาดแล้วกลับลงดิสก์ ({n_before} → {len(cfg)} แถว)")
    return cfg


def sync_rtu_configs_with_data(cfg: pd.DataFrame, dma_codes_seen, path: str) -> pd.DataFrame:
    """เพิ่ม RTU ที่เจอในข้อมูลจริง (จาก rtu_raw_export.csv) แต่ยังไม่มีอยู่ใน rtu_configs.csv ให้อัตโนมัติ
    (self-heal อีกแบบ ต่อยอดจาก load_rtu_configs) — ตั้งค่าเริ่มต้นด้วย DEFAULT_RTU_CONFIG แล้วเขียนไฟล์
    กลับลงดิสก์ทันที เหตุผลที่ต้องทำ:
      - จะได้แก้ mnf_start/mnf_end, favad_n1, user_type ฯลฯ ของ RTU ใหม่ผ่าน Excel ได้ตั้งแต่รอบถัดไป
      - detect_mnf_window.py จะเขียนช่วง MNF ที่ตรวจจับได้กลับเข้าไปให้ RTU ตัวนี้ได้ (เดิมถ้าไม่มีแถว
        ใน config จะถูกข้ามไปเฉยๆ ไม่มีวันได้ช่วง MNF ที่ปรับตามข้อมูลจริงเลย)

    หมายเหตุ: เคยลองให้คำนวณ hard_flow_threshold อัตโนมัติจากพีค flow กลางวัน (x multiplier) แต่ตอนนี้เลิกทำแล้ว
    เพราะ hard_flow_threshold เปลี่ยนความหมายเป็น "ส่วนต่างของ MNF เทียบ baseline" (ดู classify_case) ซึ่งเป็น
    ปริมาณรั่วที่มีความหมายใกล้เคียงกันข้าม DMA อยู่แล้วโดยธรรมชาติ ไม่ต้อง calibrate ตามสเกล peak รายวันอีกต่อไป
    ไม่กระทบ RTU ที่มีอยู่แล้วเลย แก้ไฟล์เฉพาะตอนมี RTU ใหม่จริงๆ เท่านั้น"""
    new_codes = sorted(set(dma_codes_seen) - set(cfg.index))
    if not new_codes:
        return cfg

    for code in new_codes:
        cfg.loc[code] = {"dma_name": code, **DEFAULT_RTU_CONFIG}

    cfg.reset_index().to_csv(path, index=False, encoding="utf-8-sig")
    print(f"      ↳ พบ RTU ใหม่ {len(new_codes)} ตัวที่ยังไม่มีใน {path} — เพิ่มแถว default ให้อัตโนมัติแล้ว: {new_codes}")
    return cfg


# ==========================================================
# STEP 1: ดึงข้อมูลดิบจากแหล่งจริง — Two-layer parquet caching
# ==========================================================
# STEP 1: ดึงข้อมูลดิบจากแหล่งจริง
#   แหล่งข้อมูลตอนนี้คือ rtu_good_quality_data.parquet — output จาก evaluate_and_split_rtu_data.py
#   (สคริปต์แยกที่กรองคุณภาพ RTU มาให้แล้ว: Stage-1 filter ตาม F_STATUS, Stage-2 gap repair,
#   frozen/stuck sensor detection ฯลฯ ตามที่คุยกันไว้) schema คอลัมน์เหมือน rtu_raw_export.csv เดิม
#   (METER_CODE/LOG_DT/F/P/F_STATUS/P_STATUS) แต่เป็น parquet ที่ type ถูกต้องแล้ว (LOG_DT เป็น datetime
#   จริงอยู่แล้ว ไม่ใช่ string ที่ต้อง parse)
#
#   ไม่มี two-layer caching เหมือนตอนอ่านจาก CSV เดิมอีกต่อไป เพราะไม่มีขั้นตอน parse ที่แพง (แปลงวันที่/dtype)
#   ให้ต้อง cache แล้ว — parquet อ่านเร็วและมี type ถูกต้องอยู่แล้วในตัว อ่านตรงพร้อม predicate pushdown
#   (กรองช่วง start-end ตั้งแต่ตอนอ่านเลย ไม่โหลดทั้งไฟล์เข้า RAM) ก็เพียงพอ
# ==========================================================
from pathlib import Path

RAW_SOURCE_PARQUET = "rtu_good_quality_data.parquet"
RAW_REQUIRED_COLS = ["dma_code", "timestamp", "flow_m3hr"]   # ยืนยันจาก schema จริงของไฟล์ (ดู error message
                                                                # ตอนเปิดไฟล์ครั้งแรก): dma_code, dma_name,
                                                                # timestamp[us], flow_m3hr, pressure_bar
                                                                # — ไม่มี METER_CODE/LOG_DT/F/P/F_STATUS/P_STATUS
                                                                # เหมือนที่เดาไว้ตอนแรก เพราะ evaluate_and_split_
                                                                # rtu_data.py แปลง schema ให้เรียบร้อยแล้วตั้งแต่ต้น


def _parse_raw_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """ตรวจสอบ + ทำความสะอาดเบื้องต้นให้ DataFrame จาก rtu_good_quality_data.parquet ซึ่งมี schema
    มาตรฐาน (dma_code/dma_name/timestamp/flow_m3hr/pressure_bar) อยู่แล้วจาก evaluate_and_split_rtu_data.py
    (ไม่มี F_STATUS/P_STATUS เหลือให้ต้องกรองอีก เพราะสคริปต์นั้นจัดการไปแล้วตั้งแต่ต้นทาง) ฟังก์ชันนี้แค่
    เช็คคอลัมน์ที่จำเป็นและ cast type ให้ชัวร์เท่านั้น"""
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"ไฟล์ {RAW_SOURCE_PARQUET} ขาดคอลัมน์ที่จำเป็น: {missing} "
                        f"(คาดคอลัมน์: dma_code, dma_name, timestamp, flow_m3hr, pressure_bar)")

    out = pd.DataFrame()
    out['dma_code'] = df['dma_code'].astype(str)
    out['dma_name'] = df['dma_name'].astype(str) if 'dma_name' in df.columns else out['dma_code']
    out['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # no-op ถ้าเป็น datetime อยู่แล้ว
    out['flow_m3hr'] = pd.to_numeric(df['flow_m3hr'], errors='coerce')
    out['pressure_bar'] = pd.to_numeric(df['pressure_bar'], errors='coerce') if 'pressure_bar' in df.columns else np.nan

    return out


def fetch_raw_from_rtu(start: datetime, end: datetime) -> pd.DataFrame:
    """
    อ่านข้อมูลจาก RAW_SOURCE_PARQUET (rtu_good_quality_data.parquet) โดยตรง กรองช่วง start-end แบบ
    predicate pushdown ตั้งแต่ตอนอ่าน (pyarrow ข้าม row-group ที่ไม่เข้าเกณฑ์ได้เลย ไม่โหลดทั้งไฟล์เข้า RAM)
    """
    if not _HAS_PYARROW:
        raise ImportError("ต้องมี pyarrow ถึงจะอ่าน parquet ได้ — รัน: pip install pyarrow")
    if not os.path.exists(RAW_SOURCE_PARQUET):
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล: {RAW_SOURCE_PARQUET}")

    filters = []
    if start:
        filters.append(("timestamp", ">=", pd.to_datetime(start)))
    if end:
        filters.append(("timestamp", "<=", pd.to_datetime(end)))

    raw = pd.read_parquet(RAW_SOURCE_PARQUET, engine="pyarrow", filters=filters or None)
    print(f"      ↳ อ่านจาก {RAW_SOURCE_PARQUET} (predicate pushdown) — ได้ {len(raw):,} แถว ในช่วง {start} ถึง {end}")

    result_df = _parse_raw_chunk(raw)
    result_df = result_df.sort_values(["dma_code", "timestamp"]).reset_index(drop=True)
    result_df['timestamp'] = result_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return result_df


# ============================================================
# STEP 1b: ข้อมูลจำลอง (Q + P ราย 15 นาที, 96 จุด/วัน)
# ============================================================
def generate_demo_data(n_days: int = N_DAYS) -> pd.DataFrame:
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days - 1)

    rows = []
    for dma in tqdm(DMA_DEFS, desc="สร้างข้อมูลจำลอง (demo)", unit="DMA"):
        rng = np.random.default_rng(dma["seed"])
        base_pressure = 3.0 + (rng.random() - 0.5) * 0.6   # bar, สมมติฐานต้นทาง ~2.7-3.3 bar

        for d in range(n_days):
            day_date = start_date + timedelta(days=d)
            dow = day_date.weekday()
            weekend_factor = 0.9 if dow in (5, 6) else 1.0

            for t in range(INTERVALS_PER_DAY):
                extra_q = 0.0
                a = dma["anomaly"]
                if a:
                    if a["type"] == "step" and d >= a["start_day"]:
                        extra_q = a["extra"]
                    elif a["type"] == "gradual" and d >= a["start_day"]:
                        prog = min(1.0, (d - a["start_day"]) / max(1, (n_days - a["start_day"])))
                        extra_q = a["max_extra"] * prog
                    elif a["type"] == "spike" and d == a["day"]:
                        extra_q = a["extra"]

                noise_q = rng.normal(0, dma["base"] * 0.05)
                flow = max(2.0, dma["base"] * DIURNAL_96[t] * weekend_factor + extra_q + noise_q)

                # ความดัน: ลดลงตามความต้องการใช้น้ำปกติ (friction loss) + ลดเพิ่มเมื่อมี extra_q (รั่ว/burst ดึงแรงดัน)
                demand_drop = (DIURNAL_96[t] - 1.0) * 0.25          # peak ชม. ดึงแรงดันลงเล็กน้อย
                leak_drop = min(0.9, extra_q / max(dma["base"], 1) * 1.4)  # extra flow มาก -> pressure ตกมาก
                noise_p = rng.normal(0, 0.03)
                pressure = max(0.3, base_pressure - demand_drop - leak_drop + noise_p)

                rows.append({
                    "dma_code": dma["code"], "dma_name": dma["name"],
                    "timestamp": day_date + timedelta(minutes=t * INTERVAL_MIN),
                    "flow_m3hr": round(flow, 2), "pressure_bar": round(pressure, 3),
                })

    return pd.DataFrame(rows)


# ============================================================
# STEP 2: Clean + validate
# ============================================================
def _mask_isolated_zero(s: pd.Series) -> pd.Series:
    """หา flow=0 ที่เกิดขึ้นแค่ 1 จุดเดียว โดยมีค่าปกติ (ไม่ใช่ 0/NaN) คั่นอยู่ทั้งสองฝั่ง — สงสัยว่าเป็น
    sensor glitch/dropout ชั่วขณะ ไม่ใช่ช่วงที่ flow หยุดจริง (ถ้า 0 ติดกันตั้งแต่ 2 จุดขึ้นไป หรืออยู่ริมสุด
    ของช่วงข้อมูล จะไม่แตะเลย เพราะอาจเป็นเหตุการณ์จริง เช่น ปิดวาล์ว/หยุดจ่ายน้ำ) คืนค่า Series เดิมที่
    แทนที่จุดต้องสงสัยด้วย NaN ให้ interpolate() ต่อได้เหมือนกรณีค่าติดลบ"""
    is_zero = s == 0
    prev_ok = s.shift(1).notna() & (s.shift(1) != 0)
    next_ok = s.shift(-1).notna() & (s.shift(-1) != 0)
    isolated = is_zero & prev_ok & next_ok
    return s.mask(isolated)


def fix_spikes_boundary_safe(s: pd.Series, threshold_ratio: float = SPIKE_THRESHOLD_RATIO,
                              window_size: int = SPIKE_WINDOW_SIZE) -> pd.Series:
    """ตรวจจับและแก้ไข spike (ค่าพุ่งขึ้นผิดปกติแบบฉับพลัน) โดยอิง baseline ย้อนหลังอย่างเดียว (shift(1) +
    rolling median) — ไม่อ้างอิงค่าถัดไปเลย จึงไม่มีปัญหาขอบขวาสุดของข้อมูล (จุดล่าสุดที่ยังไม่มี "อนาคต"
    ให้เทียบ เช่นตอนรันแบบ near-real-time ที่ยังไม่รู้ค่าของ interval ถัดไป)
    - threshold_ratio: ค่าจริงต้องสูงกว่า baseline กี่เท่าถึงจะถือว่าเป็น spike (ปรับได้ที่ SPIKE_THRESHOLD_RATIO)
    - window_size: จำนวนจุดย้อนหลังที่ใช้หา median baseline (ปรับได้ที่ SPIKE_WINDOW_SIZE)
    หมายเหตุ: รันหลัง interpolate() เสมอ เพื่อให้ rolling median คำนวณจากข้อมูลที่ไม่มีช่องว่าง (NaN) ปนอยู่แล้ว
    """
    baseline = s.shift(1).rolling(window=window_size, min_periods=1).median()
    baseline = baseline.fillna(s)  # จุดแรกสุดของช่วงข้อมูลที่ไม่มีอดีตให้อ้างอิง ใช้ค่าตัวเองไปก่อน

    is_spike = (s >= threshold_ratio * baseline) & (baseline > 0)

    s_cleaned = s.copy()
    s_cleaned[is_spike] = baseline[is_spike]
    return s_cleaned


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # เดิม dropna(subset=[...,"flow_m3hr"]) ตัดทิ้งทั้งแถวถ้า flow เป็น NaN ตั้งแต่ตอน parse
    # (เช่นแถวที่ F_STATUS='E') ก่อนจะได้ interpolate เลย ทำให้ E-flag กับค่าติดลบถูกจัดการไม่เหมือนกัน
    # ตัด "flow_m3hr" ออกจาก subset นี้ ให้เหลือแค่ทิ้งแถวที่ไม่มี dma_code/timestamp จริงๆ (ใช้อะไรไม่ได้แน่ๆ)
    # ส่วน flow_m3hr ที่เป็น NaN (ไม่ว่าจะมาจาก E-flag หรือ parse ไม่ได้) จะถูก interpolate ด้านล่างเหมือนกรณีติดลบ
    df = df.dropna(subset=["dma_code", "timestamp"])
    df = df.sort_values(["dma_code", "timestamp"]).drop_duplicates(subset=["dma_code", "timestamp"], keep="last")

    if "flow_m3hr" in df.columns:
        df.loc[df["flow_m3hr"] < 0, "flow_m3hr"] = np.nan
        df["flow_m3hr"] = df.groupby("dma_code")["flow_m3hr"].transform(_mask_isolated_zero)
        df["flow_m3hr"] = df.groupby("dma_code")["flow_m3hr"].transform(lambda s: s.interpolate(limit_direction="both"))
        # spike-fix รันหลัง interpolate เสมอ — ให้ rolling median baseline คำนวณจากข้อมูลที่เติมช่องว่างแล้ว
        # (ไม่งั้นถ้ายังมี NaN ปนอยู่ตอนคำนวณ baseline อาจทำให้ threshold เพี้ยนได้)
        df["flow_m3hr"] = df.groupby("dma_code")["flow_m3hr"].transform(fix_spikes_boundary_safe)

    if "pressure_bar" in df.columns:
        df.loc[df["pressure_bar"] < 0, "pressure_bar"] = np.nan
        df["pressure_bar"] = df.groupby("dma_code")["pressure_bar"].transform(lambda s: s.interpolate(limit_direction="both"))

    return df


def downsample_for_flow_log(df: pd.DataFrame, days_kept, export_interval_min: int) -> pd.DataFrame:
    """ลดขนาดข้อมูลเฉพาะก่อน export เข้า flow_log.csv (ใช้แค่วาดกราฟใน dashboard)
    ไม่กระทบ df ต้นฉบับที่ compute_metrics() ใช้คำนวณ MNF/CUSUM/Mann-Kendall — ฟังก์ชันนี้ทำงานบนสำเนา
    แยกต่างหากที่เตรียมไว้ export เท่านั้น (main pipeline เรียกหลังคำนวณสถิติเสร็จแล้ว)

    - days_kept: เก็บเฉพาะ N วันปฏิทินล่าสุดต่อ DMA (นับจากวันข้อมูลล่าสุดของ DMA นั้น) — None/0 = ไม่ตัด
    - export_interval_min: เก็บเฉพาะจุดที่ตกในทุกๆ N นาที ต้องหารด้วย INTERVAL_MIN ลงตัว
      (15 = ไม่ลดความถี่เลย) ช่องที่หายไปฝั่ง dashboard ถูกเติมด้วย fillGaps() ที่มีอยู่แล้วในตัว JS
    """
    if df.empty:
        return df

    out = df

    if days_kept:
        day = out["timestamp"].dt.normalize()
        cutoff = out.groupby("dma_code")["timestamp"].transform("max").dt.normalize() - pd.Timedelta(days=days_kept - 1)
        out = out[day >= cutoff]

    if export_interval_min and export_interval_min != INTERVAL_MIN:
        if export_interval_min % INTERVAL_MIN != 0:
            raise ValueError(f"FLOW_LOG_EXPORT_INTERVAL_MIN ({export_interval_min}) ต้องหารด้วย INTERVAL_MIN ({INTERVAL_MIN}) ลงตัว")
        step = export_interval_min // INTERVAL_MIN
        interval_idx = (out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute) // INTERVAL_MIN
        out = out[interval_idx % step == 0]

    return out


def export_long_csv(df: pd.DataFrame, path: str):
    out = df[["dma_code", "dma_name", "timestamp", "flow_m3hr", "pressure_bar"]].copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"บันทึกไฟล์แล้ว: {path}  ({len(out):,} แถว)")


# ============================================================
# STEP 3: Mann-Kendall + CUSUM (เหมือนเดิม แต่รับ k/h factor จาก config ต่อ RTU)
# ============================================================
def manual_mann_kendall(x: np.ndarray):
    n = len(x)
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    if var_s <= 0:
        return s, 0.0
    z = (s - 1) / np.sqrt(var_s) if s > 0 else ((s + 1) / np.sqrt(var_s) if s < 0 else 0.0)
    return s, z


def mann_kendall(series: np.ndarray):
    if len(series) < 4:
        return {"S": 0, "Z": 0.0, "trend": "ข้อมูลน้อยเกินไป", "significant": False}
    try:
        import pymannkendall as pmk
        r = pmk.original_test(series)
        S, Z = r.s, r.z
    except ImportError:
        S, Z = manual_mann_kendall(np.asarray(series, dtype=float))
    significant = abs(Z) >= 1.645
    if significant:
        trend = "เพิ่มขึ้นต่อเนื่อง (มีนัยสำคัญ)" if Z > 0 else "ลดลงต่อเนื่อง (มีนัยสำคัญ)"
    else:
        trend = "มีแนวโน้มเพิ่มขึ้นเล็กน้อย (ยังไม่ชัดเจนพอ)" if Z > 0 else "ไม่มีแนวโน้มชัดเจน"
    return {"S": S, "Z": float(Z), "trend": trend, "significant": significant}


def cusum(series: np.ndarray, target: float, sd: float, k_factor: float, h_factor: float):
    k_abs, h_abs = k_factor * sd, h_factor * sd
    c_plus = 0.0
    c_plus_series = []
    breach_index = -1
    for i, v in enumerate(series):
        c_plus = max(0.0, c_plus + (v - target - k_abs))
        c_plus_series.append(c_plus)
        if breach_index == -1 and c_plus > h_abs:
            breach_index = i

    # "ความเร็วของการขยับฐาน" = จำนวนวันจากจุดที่ค่าสะสมเป็น 0 ครั้งล่าสุดก่อนตัดเกณฑ์ ถึงจุดที่ตัดเกณฑ์
    # (แยก "burst ฉับพลัน" ออกจาก "รั่วสะสมค่อยๆ ไต่" โดยไม่ขึ้นกับว่าเหตุการณ์เกิดขึ้นเมื่อไหร่เทียบกับวันนี้)
    rise_days = None
    if breach_index != -1:
        j = breach_index
        while j > 0 and c_plus_series[j - 1] > 0:
            j -= 1
        rise_days = breach_index - j + 1

    return {
        "c_plus_series": c_plus_series, "h_abs": h_abs, "breach_index": breach_index,
        "breached": breach_index != -1, "rise_days": rise_days,
    }


# ============================================================
# STEP 4: Rolling time-of-day baseline (Q_avg(t), Q_std(t), P_avg(t), P_std(t))
# ============================================================
def rolling_time_of_day_baseline(dma_df: pd.DataFrame, exclude_recent_days: int = 1):
    """dma_df ต้องเรียงตามเวลาแล้ว มีคอลัมน์ interval_idx (0..95) และ day_idx
    exclude_recent_days: จำนวนวันล่าสุดที่ไม่นำมารวมใน baseline (กันข้อมูลที่อาจปนเปื้อนจาก
    เหตุการณ์ผิดปกติที่เพิ่งเริ่มเข้าไปทำให้ baseline สูงขึ้นตามไปด้วย)"""
    piv_q = dma_df.pivot(index="day_idx", columns="interval_idx", values="flow_m3hr")
    piv_p = dma_df.pivot(index="day_idx", columns="interval_idx", values="pressure_bar")
    cutoff = max(1, min(exclude_recent_days, len(piv_q) - 3))
    q_avg, q_std = piv_q.iloc[:-cutoff].mean(), piv_q.iloc[:-cutoff].std().fillna(0)
    p_avg, p_std = piv_p.iloc[:-cutoff].mean(), piv_p.iloc[:-cutoff].std().fillna(0)
    return q_avg, q_std, p_avg, p_std, piv_q.iloc[-1], piv_p.iloc[-1]


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def load_ignore_list(path: str = RTU_IGNORE_CSV) -> set:
    """อ่านรายชื่อ RTU ที่ต้องข้ามจาก rtu_ignore.csv (ไฟล์เดียวกับที่ dma_nrw_dashboard.html ใช้ซ่อนการแสดงผล
    ใช้คอลัมน์ rtu_id เหมือนกัน) — ไฟล์นี้เป็น optional โดยเจตนา ไม่มีไฟล์/อ่านไม่ได้/ว่างเปล่า/ไม่มีคอลัมน์
    rtu_id = ไม่ข้ามอะไรเลย ไม่ error ทั้งกระบวนการ"""
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"      ⚠️  อ่าน {path} ไม่ได้ ({e}) — ถือว่าไม่มี RTU ที่ต้องข้ามรอบนี้")
        return set()
    if "rtu_id" not in df.columns:
        return set()
    codes = df["rtu_id"].astype(str).str.strip()
    return set(codes[codes != ""])


# ============================================================
# STEP 5: Hydraulic metrics — R_eff, Loss Factor (proximity), FAVAD estimate
# ============================================================
def hydraulic_metrics_for_day(q_today, p_today, q_avg, p_avg, favad_n1, daily_excess_volume):
    """
    หมายเหตุ: P_inlet(t) ตีความเป็น p_avg(t) (baseline คาดหวัง) เพราะมีจุดวัดเดียว — ดูคำอธิบายบนสุดไฟล์
    """
    delta_p = (p_avg - p_today).clip(lower=0.001)      # แรงดันตกจาก baseline
    delta_q = (q_today - q_avg).clip(lower=0.1)       # flow ส่วนเกินจาก baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        r_eff = delta_p / (q_today.replace(0, np.nan) ** 2)
        loss_factor = delta_p / (delta_q.replace(0, np.nan) ** 2)
    r_eff_median = float(np.nanmedian(r_eff.replace([np.inf, -np.inf], np.nan)))
    loss_factor_median = float(np.nanmedian(loss_factor.replace([np.inf, -np.inf], np.nan)))
    if np.isnan(loss_factor_median):
        loss_factor_median = None

    # FAVAD: กระจาย daily_excess_volume (ที่วัดได้จริงจาก MNF) ตามน้ำหนัก P(t)^N1
    # แทนการใช้สัมประสิทธิ์ C ที่ไม่ได้คาลิเบรต (ดูหมายเหตุบนสุดไฟล์)
    weights = p_today.clip(lower=0.05) ** favad_n1
    weights_norm = weights / weights.sum() if weights.sum() > 0 else weights
    favad_hourly = weights_norm * daily_excess_volume

    return {
        "r_eff_median": None if np.isnan(r_eff_median) else round(r_eff_median, 5),
        "loss_factor_median": None if loss_factor_median is None else round(loss_factor_median, 5),
        "favad_volume_total": round(float(favad_hourly.sum()), 2),
    }


# ============================================================
# STEP 6: Classification — Case A (Burst) / B (Developing Leak) / C (High Usage)
# ============================================================
def classify_case(cfg_row, mnf_series, mk, cs, q_today, p_today, q_avg, q_std, p_avg,
                   still_elevated: bool, current_mnf: float, watch_threshold: float, peak_recent: float,
                   baseline_mean: float):
    burst_intervals = int(cfg_row["burst_intervals"])
    q_std_mult = float(cfg_row["q_std_mult"])
    p_drop_bar = float(cfg_row["p_drop_bar"])
    hard_flow_threshold = float(cfg_row.get("hard_flow_threshold", DEFAULT_RTU_CONFIG["hard_flow_threshold"]))

    high_flow_mask = (q_today > (q_avg + q_std_mult * q_std)).to_numpy()
    run_len = longest_run(high_flow_mask)
    sustained_flow_breach = run_len >= burst_intervals

    pressure_drop_mask = (p_today < (p_avg - p_drop_bar)).to_numpy()
    pressure_drop_confirmed = bool(np.any(high_flow_mask & pressure_drop_mask))

    post_baseline_len = len(cs["c_plus_series"])
    cusum_step_rapid = cs["breached"] and cs["rise_days"] is not None and cs["rise_days"] <= 3
    cusum_recent = cs["breached"] and (post_baseline_len - cs["breach_index"]) <= 7

    # เกณฑ์ Hard threshold — เดิมเทียบ flow ดิบทั้งวันกับเลขคงที่ (ผิด เพราะ flow กลางวันสูงกว่ากลางคืนเป็น
    # ปกติอยู่แล้ว ไม่เกี่ยวกับ leak) แก้เป็นเทียบ "ส่วนต่างของ MNF วันนี้กับค่าเฉลี่ยฐาน" แทน — ถ้า MNF (ซึ่งควร
    # นิ่งเพราะเป็นช่วงกลางคืน) พุ่งเกินฐานไปมากกว่าค่านี้ ถือเป็นสัญญาณ burst ที่น่าเชื่อถือ เตือนได้ทันทีโดย
    # ไม่ต้องรอสถิติ/สะสม — ถ้า MNF วันนี้ต่ำกว่าหรือใกล้ฐาน (ส่วนต่างติดลบ/น้อย) ถือเป็นเคสปกติ ไม่เตือน
    mnf_excess = current_mnf - baseline_mean
    hard_threshold_breached = mnf_excess > hard_flow_threshold

    case = "C"
    reason = "ไม่พบสัญญาณผิดปกติเชิงสถิติ (ใช้น้ำตามพฤติกรรมปกติ)"

    # เส้นทาง sustained_flow_breach อ้างอิง q_today (flow จริงวันนี้) อยู่แล้ว — ถ้าเหตุการณ์จบไปแล้ว/ซ่อมเสร็จแล้ว
    # q_today จะกลับมาปกติเอง จึงไม่ต้องกัน still_elevated ซ้ำ
    if hard_threshold_breached:
        case = "A"
        reason = (f"MNF วันนี้ ({current_mnf:.1f} m³/hr) สูงกว่าค่าเฉลี่ยฐาน ({baseline_mean:.1f} m³/hr) "
                  f"อยู่ {mnf_excess:.1f} m³/hr ซึ่งเกิน Hard Threshold ที่ตั้งไว้ ({hard_flow_threshold:.0f} m³/hr) "
                  f"เตือนทันทีโดยไม่รอผลสถิติ")
    elif sustained_flow_breach and cusum_step_rapid:
        case = "A"
        reason = f"Flow สูงกว่าเกณฑ์ต่อเนื่อง {run_len} intervals + CUSUM ขยับฐานฉับพลันใน 1-2 วันล่าสุด" + \
                 (" + แรงดันตกยืนยัน" if pressure_drop_confirmed else "")
    elif sustained_flow_breach and pressure_drop_confirmed:
        case = "B"
        reason = (f"Flow สูงกว่าเกณฑ์ต่อเนื่อง {run_len} intervals พร้อมแรงดันตก แต่ไม่ใช่การเปลี่ยนแปลงฉับพลันใน 1-2 วัน "
                   f"(น่าจะเป็นรั่วที่ดำเนินมาระยะหนึ่งแล้วและขยายจนคงที่) — ควรตรวจสอบโดยเร็วแต่ไม่ใช่เหตุฉุกเฉินแบบท่อแตกใหม่")
    elif mk["significant"] or (cs["breached"] and cs["c_plus_series"][-1] > 0):
        # เส้นทางนี้เส้นเดียวที่ไม่ได้อิง q_today เลย (มองแค่ trend/ผลสะสมย้อนหลัง) — CUSUM มี "ความจำ" คือ
        # c_plus ลดช้ากว่าที่ขึ้น ต่อให้ MNF กำลังลงจากพีคแล้วจริงๆ c_plus อาจยังค้างเป็นบวกอยู่อีกหลายวัน
        # จึงต้องเช็คเพิ่มว่า MNF ล่าสุดยังค้างใกล้พีค (ไม่ได้ลงมาต่ำกว่า RECOVERY_DROP_RATIO ของพีค) จริงไหม
        # ก่อนขึ้นเตือน — ถ้าลงมาแล้ว (เช่นซ่อมเสร็จแล้ว/เหตุการณ์จบไปแล้ว) ให้ถือว่าเป็น case ปกติแทน
        if still_elevated:
            case = "B"
            reason = "Mann-Kendall ยืนยันแนวโน้มขึ้นต่อเนื่อง" if mk["significant"] else "CUSUM สะสมเป็นบวกต่อเนื่อง (ยังไม่ Step-up ฉับพลัน)"
        else:
            reason = (f"เคยมีสัญญาณผิดปกติ (Mann-Kendall/CUSUM สะสม) แต่ MNF ล่าสุด ({current_mnf:.1f}) ลดลงมาต่ำกว่า "
                      f"{RECOVERY_DROP_RATIO*100:.0f}% ของพีคล่าสุด ({peak_recent:.1f}) แล้ว — น่าจะเป็นเหตุการณ์ที่จบไปแล้ว/"
                      f"ซ่อมเสร็จแล้ว จึงไม่ขึ้นเตือน")

    return {
        "case": case, "reason": reason, "run_len_intervals": run_len,
        "sustained_flow_breach": sustained_flow_breach,
        "pressure_drop_confirmed": pressure_drop_confirmed,
        "cusum_step_rapid": cusum_step_rapid, "cusum_recent": cusum_recent,
        "hard_threshold_breached": hard_threshold_breached,
        "still_elevated": still_elevated,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def compute_metrics(df: pd.DataFrame, rtu_cfg: pd.DataFrame):
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["interval_idx"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // INTERVAL_MIN

    daily_rows, summary_rows = [], []

    dma_codes = df["dma_code"].unique()
    for dma_code, g in tqdm(df.groupby("dma_code"), total=len(dma_codes),
                             desc="คำนวณสถิติต่อ DMA (MNF/Mann-Kendall/CUSUM/hydraulic)", unit="DMA"):
        cfg_row = rtu_cfg.loc[dma_code] if dma_code in rtu_cfg.index else {**DEFAULT_RTU_CONFIG, "dma_name": dma_code}
        if isinstance(cfg_row, pd.DataFrame):
            # เผื่อหลุดมาถึงตรงนี้ (rtu_id ซ้ำ) — กันไม่ให้ crash โดยใช้แถวแรกไปก่อน
            cfg_row = cfg_row.iloc[0]
        if not bool(cfg_row.get("is_active", True)):
            continue  # RTU ปิดการเตือนตาม config (is_active = FALSE)

        g = g.sort_values("timestamp").reset_index(drop=True)
        day_map = {d: i for i, d in enumerate(sorted(g["date"].unique()))}
        g["day_idx"] = g["date"].map(day_map)
        n_days = len(day_map)
        if n_days < 4:
            continue

        dma_name = g["dma_name"].iloc[0]
        mnf_start_t = hhmm_to_interval(cfg_row["mnf_start"])
        mnf_end_t = hhmm_to_interval(cfg_row["mnf_end"])
        if mnf_end_t > mnf_start_t:
            # ช่วงปกติ ไม่ข้ามเที่ยงคืน (เช่น 02:00-04:00)
            night = g[(g["interval_idx"] >= mnf_start_t) & (g["interval_idx"] < mnf_end_t)]
        else:
            # ช่วงข้ามเที่ยงคืน (เช่น 22:00-00:00, 23:00-01:00) — เดิมใช้ AND ธรรมดา ซึ่ง interval_idx จะ
            # >= mnf_start_t และ < mnf_end_t พร้อมกันไม่ได้เลยถ้า mnf_end_t <= mnf_start_t (ได้ 0 แถวเสมอ
            # ไม่ว่าข้อมูลจริงจะมีแค่ไหน) แก้เป็น OR แทน: เอาช่วงท้ายวัน (>= start) รวมกับช่วงต้นวันถัดไป (< end)
            # หมายเหตุ: ส่วนที่ "ข้าม" ไปวันถัดไปจะถูกนับเข้า day_idx ของวันที่ปฏิทินที่มันตกอยู่จริง (วันถัดไป)
            # ไม่ได้ย้อนกลับไปรวมกับ night ของวันก่อนหน้า — เป็นข้อจำกัดที่ยอมรับได้ตอนนี้ (ดีกว่าไม่ได้ข้อมูลเลย)
            night = g[(g["interval_idx"] >= mnf_start_t) | (g["interval_idx"] < mnf_end_t)]
        mnf_by_day = night.groupby("day_idx")["flow_m3hr"].mean().reindex(range(n_days))
        total_by_day = g.groupby("day_idx")["flow_m3hr"].sum().reindex(range(n_days)) * (INTERVAL_MIN / 60)

        mnf_series = mnf_by_day.to_numpy()
        base_win_len = max(3, min(14, n_days - 3))
        baseline = mnf_series[:base_win_len]

        # ถ้าไม่มีข้อมูล MNF เลยสักวันในช่วง baseline (เช่น ไม่มีแถวไหนตกในช่วง mnf_start-mnf_end ของวันเหล่านั้น
        # เลย) np.nanmean/np.nanstd จะเตือน "All-NaN slice"/"Degrees of freedom <= 0" แล้วคืน NaN มาเงียบๆ
        # เช็คดักไว้ก่อน ข้าม DMA นี้ไปพร้อม log ชัดๆ แทนที่จะปล่อยให้ NaN ไหลต่อเข้า threshold/CUSUM/output
        if np.all(np.isnan(baseline)):
            print(f"      ⚠️  {dma_code}: ไม่มีข้อมูล MNF ในช่วง baseline ({base_win_len} วันแรก) เลยสักวัน "
                  f"(อาจเพราะไม่มีแถวไหนตกในช่วง MNF window {cfg_row['mnf_start']}-{cfg_row['mnf_end']} "
                  f"ของวันเหล่านั้น) — ข้าม DMA นี้ไปก่อนรอบนี้")
            continue

        mean = float(np.nanmean(baseline))
        sd = float(np.nanstd(baseline)) or mean * 0.05
        threshold = mean + 2 * sd
        watch_threshold = mean + 1 * sd

        last5 = mnf_series[-5:]
        days_over_alert = int(np.sum(last5 > threshold))

        # เช็คว่า MNF "ยังค้างใกล้พีคอยู่จริง" ไหม (ไม่ใช่แค่เทียบ threshold เฉยๆ) — ใช้กันเคส flow ขึ้นแล้ว
        # ลงมาแล้ว (ซ่อมเสร็จแล้ว/เหตุการณ์จบไปแล้ว) ไม่ให้ CUSUM ที่มี "ความจำ" (c_plus ลดช้ากว่าที่ขึ้น) ค้าง
        # ขึ้นเตือนต่อ ทั้งที่ MNF กำลังลงจากพีคแล้วจริงๆ แม้จะยังไม่ทันต่ำกว่า threshold ก็ตาม
        recent_window = mnf_series[-min(n_days, RECOVERY_LOOKBACK_DAYS):]
        peak_recent = float(np.nanmax(recent_window))
        current_mnf = float(mnf_series[-1])
        declining_from_peak = peak_recent > 0 and current_mnf < peak_recent * RECOVERY_DROP_RATIO
        still_elevated = (current_mnf > watch_threshold) and not declining_from_peak

        mk = mann_kendall(mnf_series[-min(10, n_days):])
        post_baseline = mnf_series[base_win_len:]
        cs = cusum(post_baseline, mean, sd, float(cfg_row["cusum_k_factor"]), float(cfg_row["cusum_h_factor"]))
        cusum_recent_breach = cs["breached"] and (len(post_baseline) - cs["breach_index"]) <= 7

        # rolling time-of-day baseline + วันล่าสุดราย interval (สำหรับ burst detection + hydraulic metrics)
        # กันข้อมูล baseline ปนเปื้อนจากเหตุการณ์ที่เพิ่งเริ่ม โดยตัดวันล่าสุดออกเท่ากับช่วง post-baseline ที่มี
        exclude_recent = max(1, n_days - base_win_len)
        q_avg, q_std, p_avg, p_std, q_today, p_today = rolling_time_of_day_baseline(g, exclude_recent_days=exclude_recent)
        q_today = q_today.reindex(range(INTERVALS_PER_DAY)).interpolate(limit_direction="both")
        p_today = p_today.reindex(range(INTERVALS_PER_DAY)).interpolate(limit_direction="both")
        q_avg = q_avg.reindex(range(INTERVALS_PER_DAY)).interpolate(limit_direction="both")
        q_std = q_std.reindex(range(INTERVALS_PER_DAY)).fillna(0)
        p_avg = p_avg.reindex(range(INTERVALS_PER_DAY)).interpolate(limit_direction="both")

        classification = classify_case(cfg_row, mnf_series, mk, cs, q_today, p_today, q_avg, q_std, p_avg,
                                        still_elevated, current_mnf, watch_threshold, peak_recent, mean)

        excess_per_hour = max(0.0, float(np.mean(mnf_series[-3:])) - mean)
        est_loss_per_day = excess_per_hour * 24
        hyd = hydraulic_metrics_for_day(q_today, p_today, q_avg, p_avg, float(cfg_row["favad_n1"]), est_loss_per_day)

        status = {"A": "alert", "B": "watch", "C": "ok"}[classification["case"]]

        for i in range(n_days):
            cplus = cs["c_plus_series"][i - base_win_len] if i >= base_win_len else None
            daily_rows.append({
                "dma_code": dma_code,
                "date": sorted(day_map, key=day_map.get)[i],
                "mnf": None if np.isnan(mnf_series[i]) else round(float(mnf_series[i]), 3),
                "daily_total": None if np.isnan(total_by_day.iloc[i]) else round(float(total_by_day.iloc[i]), 2),
                "cusum_cplus": round(cplus, 3) if cplus is not None else "",
            })

        summary_rows.append({
            "dma_code": dma_code, "dma_name": dma_name,
            "case": classification["case"], "status": status, "case_reason": classification["reason"],
            "mnf_start": cfg_row["mnf_start"], "mnf_end": cfg_row["mnf_end"],
            "current_mnf": round(float(mnf_series[-1]), 3), "baseline_mean": round(mean, 3),
            "baseline_sd": round(sd, 3), "threshold": round(threshold, 3), "watch_threshold": round(watch_threshold, 3),
            "mk_z": round(mk["Z"], 3), "mk_trend": mk["trend"], "mk_significant": mk["significant"],
            "cusum_h": round(cs["h_abs"], 3), "cusum_breached": cs["breached"],
            "cusum_recent_breach": cusum_recent_breach, "cusum_step_rapid": classification["cusum_step_rapid"],
            "sustained_flow_breach": classification["sustained_flow_breach"],
            "sustained_run_intervals": classification["run_len_intervals"],
            "pressure_drop_confirmed": classification["pressure_drop_confirmed"],
            "hard_threshold_breached": classification["hard_threshold_breached"],
            "excess_per_hour": round(excess_per_hour, 3), "est_loss_per_day": round(est_loss_per_day, 2),
            "base_win_len": base_win_len,
            "r_eff_median": hyd["r_eff_median"], "loss_factor_median": hyd["loss_factor_median"],
            "proximity_estimate": None, "favad_volume_total": hyd["favad_volume_total"],
            "user_type": cfg_row.get("user_type", "RESIDENTIAL"), "pipe_material": cfg_row.get("pipe_material", "PVC"),
        })

    daily_df = pd.DataFrame(daily_rows)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print(f"      ⚠️  ทุก DMA ({len(dma_codes)} ตัว) ในรอบนี้ถูกข้ามหมด (ข้อมูลน้อยเกินไป/baseline คำนวณไม่ได้) "
              f"— dma_status_summary.csv รอบนี้จะว่างเปล่า")
    summary_df = classify_proximity_relative(summary_df)
    return daily_df, summary_df


def classify_proximity_relative(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    จัดประเภท "ใกล้/ไกล RTU" แบบเทียบสัมพัทธ์ระหว่าง DMA ที่เกิดเหตุ (Case A/B)
    """
    if summary_df.empty or "case" not in summary_df.columns:
        # เผื่อทุก DMA ในรอบนี้โดนข้ามหมด (เช่น n_days<4 หรือ baseline เป็น NaN ทั้งชุด) — pd.DataFrame([])
        # จะไม่มีคอลัมน์เลยสักตัว เข้าถึง summary_df["case"] ตรงๆ จะ KeyError แทนที่จะได้ DataFrame ว่างเปล่ากลับไป
        return summary_df
    active = summary_df[summary_df["case"].isin(["A", "B"]) & summary_df["loss_factor_median"].notna()]
    if len(active) < 2:
        return summary_df
    median_lf = active["loss_factor_median"].median()

    def label(row):
        if row["case"] not in ("A", "B") or pd.isna(row["loss_factor_median"]):
            return None
        
        # 🟢 แก้ไข: Loss Factor สูง = แรงดันตกมาก = อยู่ไกล (ปลายสาย)
        #          Loss Factor ต่ำ = แรงดันตกน้อย = อยู่ใกล้ RTU (ต้นสาย)
        return ("ไกลจาก RTU สัมพัทธ์ / ปลายสาย (> 2 กม. โดยประมาณ)" if row["loss_factor_median"] > median_lf
                else "ใกล้ RTU สัมพัทธ์ (< 2 กม. โดยประมาณ)")

    summary_df["proximity_estimate"] = summary_df.apply(label, axis=1)
    return summary_df


if __name__ == "__main__":
    _pipeline_start = time.time()
    end = datetime.now()
    start = end - timedelta(days=N_DAYS)

    step("โหลด config รายตู้ RTU (rtu_configs.csv)")
    rtu_cfg = load_rtu_configs(RTU_CONFIG_PATH, DMA_DEFS if USE_DEMO_DATA else [])

    step("ดึงข้อมูลดิบ (flow_log.csv จริง หรือข้อมูลจำลอง)")
    if USE_DEMO_DATA:
        raw = generate_demo_data(N_DAYS)
    else:
        raw = fetch_raw_from_rtu(start, end)
    print(f"      ↳ อ่านมาได้ {len(raw):,} แถว")

    ignore_set = load_ignore_list(RTU_IGNORE_CSV)
    if ignore_set:
        matched = ignore_set & set(raw["dma_code"].unique())
        if matched:
            n_before = len(raw)
            raw = raw[~raw["dma_code"].isin(ignore_set)]
            print(f"      ↳ ข้าม {len(matched)} RTU ตาม {RTU_IGNORE_CSV} (คุณภาพไม่ดี ตรวจแยกไว้แล้ว): "
                  f"{sorted(matched)} — ตัดออก {n_before - len(raw):,} แถว ก่อนเริ่มประมวลผลใดๆ")

    step("ทำความสะอาดข้อมูล (clean + interpolate ค่าที่ขาดหาย)")
    cleaned = clean_data(raw)

    rtu_cfg = sync_rtu_configs_with_data(rtu_cfg, cleaned["dma_code"].unique(), RTU_CONFIG_PATH)

    step("คำนวณสถิติต่อ DMA (นี่คือขั้นตอนหลัก อาจใช้เวลาถ้ามีหลาย RTU)")
    daily_series, status_summary = compute_metrics(cleaned, rtu_cfg)

    step("บันทึกไฟล์ดิบ (flow_log.csv)")
    if FLOW_LOG_ONLY_FLAGGED:
        flagged_codes = set(status_summary.loc[status_summary["status"].isin(["watch", "alert"]), "dma_code"])
        flow_export_df = cleaned[cleaned["dma_code"].isin(flagged_codes)]
        n_total = cleaned["dma_code"].nunique()
        print(f"      ↳ FLOW_LOG_ONLY_FLAGGED=True: ส่งออกเฉพาะ {len(flagged_codes)}/{n_total} DMA ที่สถานะ watch/alert "
              f"(DMA ที่เหลือยังดู MNF/CUSUM ได้ปกติจาก dma_daily_series.csv แค่กราฟรายวัน 15 นาทีจะไม่มีข้อมูล)")
    else:
        flow_export_df = cleaned
    n_before = len(flow_export_df)
    flow_export_df = downsample_for_flow_log(flow_export_df, FLOW_LOG_DAYS_KEPT, FLOW_LOG_EXPORT_INTERVAL_MIN)
    if len(flow_export_df) != n_before:
        print(f"      ↳ downsample_for_flow_log: {n_before:,} → {len(flow_export_df):,} แถว "
              f"(days_kept={FLOW_LOG_DAYS_KEPT or 'ไม่ตัด'}, export_interval_min={FLOW_LOG_EXPORT_INTERVAL_MIN})")
    export_long_csv(flow_export_df, OUTPUT_FLOW_CSV)

    step("บันทึกไฟล์ผลลัพธ์ (dma_daily_series.csv, dma_status_summary.csv)")
    daily_series.to_csv(OUTPUT_SERIES_CSV, index=False, encoding="utf-8-sig")
    status_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"      ↳ เสร็จใน {time.time() - _step_start_time[0]:.1f} วินาที")

    print(f"บันทึกไฟล์แล้ว: {OUTPUT_SERIES_CSV}   ({len(daily_series):,} แถว)")
    print(f"บันทึกไฟล์แล้ว: {OUTPUT_SUMMARY_CSV} ({len(status_summary):,} แถว)")

    for case, label in [("A", "🔴 PIPE BURST"), ("B", "🟠 DEVELOPING LEAK")]:
        hits = status_summary[status_summary.case == case]["dma_code"].tolist()
        if hits:
            print(f"{label}: {', '.join(hits)}")

    total_elapsed = time.time() - _pipeline_start
    print(f"\n✅ รันเสร็จทั้งหมดใน {total_elapsed:.1f} วินาที ({total_elapsed/60:.1f} นาที)")