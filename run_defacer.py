"""
python run_defacer.py --input ./processed/3d_input --output ./processed/defaced_output
"""

import argparse
from pathlib import Path
import pandas as pd
import tempfile
import numpy as np
import nibabel as nib
import nibabel.processing
from defacer import Defacer


def make_canonical(input_path, temp_path):
    orig_img = nib.load(str(input_path))
    cano_img = nib.as_closest_canonical(orig_img)
    nib.save(cano_img, str(temp_path))
    return orig_img, cano_img


def restore_original_orientation(defaced_path, orig_img, cano_img, final_path):
    defaced_img = nib.load(str(defaced_path))
    orig_ornt = nib.io_orientation(orig_img.affine)
    cano_ornt = nib.io_orientation(cano_img.affine)
    transform = nib.orientations.ornt_transform(cano_ornt, orig_ornt)
    restored_data = nib.orientations.apply_orientation(defaced_img.get_fdata(), transform)

    orig_dtype = np.asanyarray(orig_img.dataobj).dtype
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        restored_data = np.rint(restored_data)
        restored_data = np.clip(restored_data, info.min, info.max)
    restored_data = restored_data.astype(orig_dtype)

    restored_img = nib.Nifti1Image(restored_data, orig_img.affine, orig_img.header)
    nib.save(restored_img, str(final_path))


def apply_mask_to_other_sequence(other_file, mask_img, output_path):
    target_img = nib.load(str(other_file))
    target_data = target_img.get_fdata()

    # 마스크를 타겟 영상의 좌표/해상도에 맞춰 nearest-neighbor 보간
    resampled_mask_img = nib.processing.resample_from_to(mask_img, target_img, order=0)
    resampled_mask_data = resampled_mask_img.get_fdata() > 0.5

    target_data[resampled_mask_data] = 0

    target_dtype = np.asanyarray(target_img.dataobj).dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        target_data = np.rint(target_data)
        target_data = np.clip(target_data, info.min, info.max)
    target_data = target_data.astype(target_dtype)

    final_img = nib.Nifti1Image(target_data, target_img.affine, target_img.header)
    nib.save(final_img, str(output_path))


def list_nifti_files(directory: Path):
    files = []
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            files.append(p)
    return sorted(files)


def discover_patient_groups(input_path: Path):
    groups = {}
    direct_files = []

    for child in sorted(input_path.iterdir()):
        if child.is_dir():
            nii_files = list_nifti_files(child)
            if nii_files:
                groups[child.name] = nii_files
        elif child.is_file():
            name = child.name.lower()
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                direct_files.append(child)

    # 환자 폴더 없이 파일이 직접 들어있는 경우도 지원
    if direct_files:
        groups["_root"] = sorted(direct_files)

    # 하위 구조가 더 깊고, 루트 바로 아래 폴더엔 파일이 없는 경우를 대비한 fallback
    if not groups:
        all_files = list_nifti_files(input_path)
        if all_files:
            groups["_root"] = all_files

    return groups


def choose_reference_t1(nifti_files):
    candidates = []
    for f in nifti_files:
        name_upper = f.name.upper()
        if "T1" in name_upper and "SAG" in name_upper:
            candidates.append((0, len(name_upper), f))
        elif "T1" in name_upper:
            candidates.append((1, len(name_upper), f))
        else:
            candidates.append((2, len(name_upper), f))

    candidates.sort(key=lambda x: (x[0], x[1], x[2].name))
    return candidates[0][2]


def run_dl_deface(defacer, input_file: Path, output_file: Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input = Path(temp_dir) / input_file.name
        orig_img, cano_img = make_canonical(input_file, temp_input)

        result = defacer.Deidentification_image_nii(
            where=(1, 1, 1, 1),
            nfti_path=str(temp_input),
            dest_path=temp_dir,
            prefix="defaced",
        )
        if not result["success"]:
            raise RuntimeError(result["msg"])

        defaced_temp_file = Path(result["path"])
        restore_original_orientation(defaced_temp_file, orig_img, cano_img, output_file)

    return nib.load(str(output_file)), nib.load(str(input_file))


def build_mask_from_reference(orig_img, defaced_img):
    # 숫자 오차에 민감하지 않게 threshold 기반 차이 마스크 생성
    orig_data = orig_img.get_fdata()
    defaced_data = defaced_img.get_fdata()
    mask_data = (np.abs(orig_data - defaced_data) > 1e-6).astype(np.float32)
    return nib.Nifti1Image(mask_data, orig_img.affine)


def main(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    qc_csv_path = output_path.parent / "qc_report.csv"
    if qc_csv_path.exists():
        qc_df = pd.read_csv(qc_csv_path)
        for col in ["defacing_target", "defacing_done"]:
            if col in qc_df.columns:
                qc_df[col] = qc_df[col].fillna(0).astype(int)
    else:
        qc_df = pd.DataFrame(columns=["case_id", "nifti_conversion", "defacing_target", "defacing_done", "error_files"])

    print("🚀 Defacing Start")
    print("   ⏳ Loading DL Model...")
    defacer = Defacer()

    patient_groups = discover_patient_groups(input_path)
    if not patient_groups:
        print("❌ No NIfTI files found in input.")
        return

    success_count = 0
    total_files = sum(len(v) for v in patient_groups.values())

    for patient_id, nifti_files in patient_groups.items():
        out_case_id = patient_id if patient_id != "_root" else "root"
        patient_out_dir = output_path / out_case_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔹 Processing: {patient_id} ({len(nifti_files)} files)")

        reference_t1 = choose_reference_t1(nifti_files)
        print(f"   🎯 Reference selected: {reference_t1.name}")

        mask_img = None
        patient_errors = []
        patient_done = 0

        # 기준 시퀀스 DL 수행
        try:
            final_t1_path = patient_out_dir / f"defaced_{reference_t1.name}"
            defaced_t1_img, orig_t1_img = run_dl_deface(defacer, reference_t1, final_t1_path)
            mask_img = build_mask_from_reference(orig_t1_img, defaced_t1_img)
            print("   ✅ Reference defaced and mask extracted")
            success_count += 1
            patient_done += 1
        except Exception as e:
            print(f"   ❌ Reference DL Error: {e}")
            patient_errors.append(reference_t1.name)

        # 마스크 적용 or fallback DL
        for nii_file in nifti_files:
            if nii_file == reference_t1:
                continue

            final_path = patient_out_dir / f"defaced_{nii_file.name}"
            try:
                if mask_img is not None:
                    apply_mask_to_other_sequence(nii_file, mask_img, final_path)
                    print(f"   ⚡ Mask Applied: {nii_file.name}")
                else:
                    # 기준 생성 실패 시, 파일별 DL로 fallback
                    run_dl_deface(defacer, nii_file, final_path)
                    print(f"   🧠 Fallback DL: {nii_file.name}")

                success_count += 1
                patient_done += 1
            except Exception as e:
                print(f"   ❌ Defacing Error ({nii_file.name}): {e}")
                patient_errors.append(nii_file.name)

        error_str = "; ".join(patient_errors)
        if "case_id" in qc_df.columns and patient_id in qc_df["case_id"].values:
            qc_df.loc[qc_df["case_id"] == patient_id, "defacing_target"] = len(nifti_files)
            qc_df.loc[qc_df["case_id"] == patient_id, "defacing_done"] = patient_done
            qc_df.loc[qc_df["case_id"] == patient_id, "error_files"] = error_str
        else:
            new_row = pd.DataFrame([
                {
                    "case_id": patient_id,
                    "nifti_conversion": "",
                    "defacing_target": len(nifti_files),
                    "defacing_done": patient_done,
                    "error_files": error_str,
                }
            ])
            qc_df = pd.concat([qc_df, new_row], ignore_index=True)

        qc_df.to_csv(qc_csv_path, index=False)

    print(f"\n📋 QC report saved: {qc_csv_path}")
    print(f"🎉 Completed: {success_count}/{total_files} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to NIfTI files")
    parser.add_argument("--output", required=True, help="Path to save defaced files")
    args = parser.parse_args()
    main(args.input, args.output)
