"""
python run_defacer.py --input ./processed/3d_input --output ./processed/defaced_output
"""

import os
import argparse
import glob
from pathlib import Path
from defacer import Defacer 

def main(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 결과 폴더 생성
    output_path.mkdir(parents=True, exist_ok=True)
    # verif_path = output_path / "verification"
    # verif_path.mkdir(exist_ok=True)

    print(f"🚀 Defacing Start")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")

    # Defacer 모델 로딩 (시간이 좀 걸림)
    print("   ⏳ Loading Model... (Wait)")
    defacer = Defacer()
    
    # NIfTI 파일 탐색 (하위 폴더 포함)
    nifti_files = list(input_path.rglob("*.nii.gz"))
    
    if not nifti_files:
        print("❌ No .nii.gz files found!")
        return

    print(f"   -> Found {len(nifti_files)} files.")

    success_count = 0
    
    for nii_file in nifti_files:
        print(f"\n🔹 Processing: {nii_file.name}")
        
        # 환자별 결과 폴더 유지 (선택사항)
        # 예: output/SA00013/defaced_file.nii.gz
        patient_id = nii_file.parent.name
        patient_out_dir = output_path / patient_id
        patient_out_dir.mkdir(exist_ok=True)

        try:
            # Defacing 실행
            # where=(1,1,1,1) -> 눈, 코, 귀, 입 모두 지움
            result = defacer.Deidentification_image_nii(
                where=(1, 1, 1, 1),
                nfti_path=str(nii_file),
                dest_path=str(patient_out_dir),
                # verif_path=str(verif_path),
                prefix="defaced"
            )
            
            if result['success']:
                print(f"   ✅ Saved: {result['path']}")
                success_count += 1
            else:
                print(f"   ❌ Failed: {result['msg']}")
                
        except Exception as e:
            print(f"   ❌ Critical Error: {e}")

    print(f"\n🎉 완료! {len(nifti_files)}개 중 {success_count}개 성공")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to NIfTI files (e.g., ./processed/3d_input)")
    parser.add_argument("--output", required=True, help="Path to save defaced files")
    args = parser.parse_args()
    
    main(args.input, args.output)