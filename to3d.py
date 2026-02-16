"""
python to3d.py --input ./raw_data --output ./processed/3d_input
"""

import os
import shutil
import re
import argparse
import pydicom
import dicom2nifti
import dicom2nifti.convert_dicom as convert_dicom
import numpy as np
import logging
from pathlib import Path
import pandas as pd

# 불필요한 경고 메시지 숨김
logging.getLogger('dicom2nifti').setLevel(logging.CRITICAL)

# ============================================================
# [Logic 1] 데이터 클리닝 (Cleaner)
# 역할: 파일명/폴더명의 특수문자를 제거하고, DICOM 헤더 기반으로 폴더 재정렬
# ============================================================

def safe_name(name: str) -> str:
    """파일명 안전 변환 (공백 -> _, 특수문자 제거)"""
    name = re.sub(r'\s+', '_', str(name).strip())
    name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)
    return name

def organize_dicom_folder(src_dir: Path, temp_base: Path) -> Path:
    """
    [핵심] 원본 폴더(SA..., 301, 501 등)를 읽어 
    실제 촬영 명칭(SeriesDescription)으로 된 임시 폴더로 정리합니다.
    """
    # 임시 정리 폴더: temp/환자ID
    patient_id = src_dir.name
    dest_parent = temp_base / patient_id
    dest_parent.mkdir(parents=True, exist_ok=True)

    print(f"   Note: 정리 중... {src_dir.name}")
    
    # 재귀적으로 모든 파일 탐색 (.dcm 확장자가 없어도 읽어봄)
    for f in src_dir.rglob("*"):
        if not f.is_file(): continue
        
        try:
            # 픽셀 데이터 제외하고 헤더만 빠르게 읽기
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            
            # 시리즈 명칭 추출 (없으면 Unknown)
            series_desc = ds.get("SeriesDescription", "UnknownSeries")
            safe_desc = safe_name(series_desc)
            
            # 목표 폴더: temp/환자ID/T1_Axial 등
            target_dir = dest_parent / safe_desc
            target_dir.mkdir(exist_ok=True)
            
            # 파일 복사
            shutil.copy2(str(f), str(target_dir / f.name))
            
        except Exception:
            continue # DICOM 아닌 파일은 무시

    return dest_parent

# ============================================================
# [Logic 2] 구조대 (Rescuer)
# 역할: 일반 변환 실패 시, 슬라이스 위치를 분석해 가장 긴 연속 구간을 살려냄
# ============================================================

def attempt_rescue_conversion(series_folder_path, temp_output_dir):
    """
    [핵심] dicom2nifti가 포기한 데이터를 살려내는 함수
    Z축(높이) 위치를 분석하여 끊기지 않고 연속된 슬라이스 뭉치를 찾아냅니다.
    """
    print("      -> 🚑 구조 모드(Rescue Mode) 진입...")
    dicom_slices = []
    
    # 1. 폴더 내 파일들의 위치 정보 수집
    for filename in os.listdir(series_folder_path):
        filepath = os.path.join(series_folder_path, filename)
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            if 'ImagePositionPatient' in dcm:
                dicom_slices.append({
                    'path': filepath,
                    'pos': dcm.ImagePositionPatient, # [x, y, z]
                    'inst': dcm.InstanceNumber
                })
        except:
            continue

    if len(dicom_slices) < 10: 
        print("      -> ❌ 슬라이스 개수가 너무 적어 구조 불가.")
        return None

    # 2. Z축 기준 정렬 (머리->다리 순서)
    dicom_slices.sort(key=lambda s: s['pos'][2])
    
    # 3. 연속성 검사 (가장 긴 덩어리 찾기)
    longest_group = []
    current_group = [dicom_slices[0]]
    
    # 첫 번째와 두 번째 슬라이스 간격을 기준점으로 잡음
    if len(dicom_slices) > 1:
        base_dist = np.linalg.norm(np.array(dicom_slices[1]['pos']) - np.array(dicom_slices[0]['pos']))
    else:
        return None

    for i in range(len(dicom_slices)-1):
        # 현재 간격 계산
        dist = np.linalg.norm(np.array(dicom_slices[i+1]['pos']) - np.array(dicom_slices[i]['pos']))
        
        # 간격이 일정하면(오차범위 내) 같은 그룹으로 인정
        if np.isclose(dist, base_dist, atol=0.1): 
            current_group.append(dicom_slices[i+1])
        else:
            # 간격이 달라지면 끊김 발생. 현재 그룹 저장하고 초기화
            if len(current_group) > len(longest_group):
                longest_group = current_group
            current_group = [dicom_slices[i+1]]
            # 끊긴 지점부터 새로운 간격을 기준으로 삼음 (다음 루프 위해)
            if i + 2 < len(dicom_slices):
                base_dist = np.linalg.norm(np.array(dicom_slices[i+2]['pos']) - np.array(dicom_slices[i+1]['pos']))

    if len(current_group) > len(longest_group):
        longest_group = current_group

    # 4. 구조된 데이터로 강제 변환
    if len(longest_group) > 10:
        print(f"      -> ✅ 연속된 {len(longest_group)}개 슬라이스 구조 성공! 변환 시도.")
        try:
            dicom_objects = [pydicom.dcmread(s['path']) for s in longest_group]
            temp_nii_name = "rescued_temp.nii.gz"
            temp_nii_path = os.path.join(temp_output_dir, temp_nii_name)
            
            # 로우레벨 변환 함수 호출
            convert_dicom.dicom_array_to_nifti(dicom_objects, temp_nii_path, reorient=True)
            return temp_nii_path
        except Exception as e:
            print(f"      -> ❌ 구조 중 에러 발생: {e}")
            return None
    else:
        print("      -> ❌ 유효한 연속 구간이 없습니다.")
        return None

# ============================================================
# [Main] 실행 파이프라인
# ============================================================

def process_to_nifti(input_root, output_root):
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== [QC] CSV 초기화 ==========
    qc_csv_path = output_path.parent / "qc_report.csv"
    if qc_csv_path.exists():
        qc_df = pd.read_csv(qc_csv_path)
    else:
        qc_df = pd.DataFrame(columns=["case_id", "nifti_conversion", "defacing_target", "defacing_done", "error_files"])
    # =====================================
    
    # 임시 작업 공간 (정리된 DICOM용)
    temp_workspace = output_path / "_temp_organized"
    temp_workspace.mkdir(exist_ok=True)

    print(f"🚀 [Start] DICOM to NIfTI Conversion")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")

    # 1. 환자 폴더 순회
    # 예: SA00013..., SA00031... 폴더들을 찾음
    patient_folders = sorted([p for p in input_path.iterdir() if p.is_dir()])
    
    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        print(f"\n🔹 Processing Patient: {patient_id}")

        # ========== [QC] 카운터 초기화 ==========
        series_total = 0
        convert_success = 0
        # =======================================
        
        # [Step 1] 복잡한 폴더 구조(301, 501...)를 깔끔하게(T1, FLAIR...) 정리
        organized_patient_dir = organize_dicom_folder(patient_dir, temp_workspace)
        
        # [Step 2] 정리된 폴더별로 NIfTI 변환 수행
        for series_dir in organized_patient_dir.iterdir():
            if not series_dir.is_dir(): continue
            
            series_name = series_dir.name  # 예: T1_Axial
            save_name = f"{patient_id}_{series_name}.nii.gz"
            final_path = output_path / patient_id / save_name
            
            # 이미 변환된 파일 있으면 스킵
            if final_path.exists():
                print(f"   - Skip: {save_name} (이미 존재함)")
                series_total += 1      # [QC]
                convert_success += 1   # [QC]
                continue
            series_total += 1  # [QC]

            # 환자별 결과 폴더 생성
            final_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"   - Converting: {series_name} ... ", end="")
            
            try:
                # 1차 시도: 표준 변환 (dicom2nifti)
                # 임시로 저장할 곳
                dicom2nifti.convert_directory(str(series_dir), str(final_path.parent), 
                                            compression=True, reorient=True)
                
                # dicom2nifti는 랜덤한 이름(예: 4_series.nii.gz)으로 저장하므로
                # 방금 생성된 파일을 찾아 내가 원하는 이름으로 변경해야 함
                generated_files = sorted(list(final_path.parent.glob("*.nii.gz")), 
                                       key=os.path.getmtime, reverse=True)
                
                found = False
                for gf in generated_files:
                    # 파일명이 내가 지정한 save_name과 다르고, 환자ID가 포함 안 된(랜덤생성된) 파일 찾기
                    if gf.name != save_name and patient_id not in gf.name:
                        gf.rename(final_path)
                        found = True
                        break
                
                # 남은 랜덤 생성 파일들 삭제 (중복 방지)
                for gf in final_path.parent.glob("*.nii.gz"):
                    if patient_id not in gf.name:
                        gf.unlink()  # 삭제
                
                if found:
                    print("✅ Success")
                    convert_success += 1  # [QC]
                else:
                    raise Exception("변환된 파일을 찾을 수 없음")

            except Exception:
                # 2차 시도: 실패 시 구조대 호출
                rescued_file = attempt_rescue_conversion(str(series_dir), str(final_path.parent))
                if rescued_file:
                    os.rename(rescued_file, final_path)
                    print("✅ Success (Rescued)")
                    convert_success += 1  # [QC]
                else:
                    print("❌ Failed")

        # ========== [QC] CSV 업데이트 (환자 하나 완료 시마다) ==========
        nifti_conversion = f"{convert_success}/{series_total}"
        
        if patient_id in qc_df["case_id"].values:
            qc_df.loc[qc_df["case_id"] == patient_id, "nifti_conversion"] = nifti_conversion
        else:
            new_row = pd.DataFrame([{
                "case_id": patient_id,
                "nifti_conversion": nifti_conversion,
                "defacing_target": "",
                "defacing_done": "",
                "error_files": ""
            }])
            qc_df = pd.concat([qc_df, new_row], ignore_index=True)
        
        qc_df.to_csv(qc_csv_path, index=False)
        print(f"   📊 [QC] {patient_id}: {nifti_conversion} → CSV 업데이트")
        # =============================================================

    # [Cleanup] 임시 폴더 삭제
    try:
        shutil.rmtree(temp_workspace)
        print("\n🧹 임시 파일 정리 완료")
    except:
        pass

    print(f"\n🎉 모든 변환 작업 완료! 저장 위치: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM to NIfTI Converter with Rescue Mode")
    parser.add_argument("--input", required=True, help="Raw Data 폴더 경로 (예: raw_data)")
    parser.add_argument("--output", required=True, help="결과 NIfTI 저장 경로 (예: nifti_output)")
    
    args = parser.parse_args()
    
    process_to_nifti(args.input, args.output)