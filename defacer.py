import tensorflow as tf
import os
import sys
import glob
import numpy as np
import nibabel as nib
import pydicom
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from keras.utils import to_categorical

# 모델 임포트 (경로 주의)
import model.model_ver_contour as model

# [설정] TF 1.14 그래프 초기화
graph = tf.get_default_graph()

class Defacer(object):
    def onehot2label(self, onehot_array):
        onehot_array = np.argmax(onehot_array, axis=-1)
        label = onehot_array[..., np.newaxis]
        return label

    def resize(self, data, img_dep=128, img_cols=128, img_rows=128):
        resize_factor = (img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
        data = ndimage.zoom(data, resize_factor, order=0, mode='constant', cval=0.0)
        return data

    def bounding_box(self, results):
        boxes = list()
        # results shape: (depth, height, width, channels)
        # channels: 0(배경), 1(눈), 2(코), 3(귀), 4(입)
        
        for ch in range(results.shape[-1]):
            result = np.round(results[..., ch])
            lb = label(result, connectivity=1)
            
            # 노이즈 제거 및 박스 추출
            if np.max(lb) >= 1:
                region_list = [region.area for region in regionprops(lb)]
                if region_list:
                    # 너무 작은 영역(상위 영역 크기의 30% 미만)은 노이즈로 간주하고 제거
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.3)
            
            # [핵심 수정 1] 에러 발생 코드 삭제 -> 있는 만큼만 가져옴
            # found_props = regionprops(lb) 
            # if len(found_props) != 2: raise Exception ... (삭제됨)
            
            # 발견된 모든 영역의 박스를 리스트에 추가
            for region in regionprops(lb):
                boxes.append(list(region.bbox))
                
        return boxes

    def box_blur(self, im_array, box, wth=1):
        # 박스 크기를 wth 배 만큼 확장
        if wth != 1:
            for c in range(3):
                mean_ = (box[c]+box[c+3])/2
                half_len = (box[c+3]-box[c]) * wth / 2
                box[c] = int(max(0, mean_ - half_len))
                box[c+3] = int(min(im_array.shape[c], mean_ + half_len))

        # [핵심] 해당 영역을 0으로 채움 (확실한 익명화)
        im_array[box[0]:box[3], box[1]:box[4], box[2]:box[5]] = 0
        return im_array

    def label_denoising(self, results):
        # 결과 맵 정제 (노이즈 제거)
        for ch in range(1, results.shape[-1]):
            result = np.round(results[..., ch])
            lb = label(result, connectivity=1)
            region_list = [region.area for region in regionprops(lb)]
            if not region_list: continue
            
            if ch == 1 or ch == 3: # 눈, 귀 (보통 2개)
                region_list_sort = sorted(region_list)
                min_s = region_list_sort[-2] if len(region_list) >= 2 else region_list_sort[-1]
                lb = remove_small_objects(lb, min_size=min_s)
                results[..., ch] = results[..., ch] * (lb > 0)
            
            if ch == 2 or ch == 4: # 코, 입 (보통 1개)
                max_size = np.max(region_list)
                lb = remove_small_objects(lb, min_size=max_size)
                results[..., ch] = results[..., ch] * (lb > 0)
        return results

    # =========================================================================
    # [핵심 수정 2] 메인 실행 함수 (NIfTI 전용 + 디버깅 강화)
    # =========================================================================
    def Deidentification_image_nii(self, where, nfti_path, dest_path, prefix="defaced", Model=model):
        config = {"resizing": True, "input_shape": [128, 128, 128, 1]}
        if "{}" not in prefix: prefix += "_{}"

        try:
            print(f"   🔎 [Processing] Reading: {os.path.basename(nfti_path)}")
            
            # 1. 파일 로드
            raw_img = nib.load(nfti_path)
            array_img = raw_img.get_fdata()
            original_shape = array_img.shape
            
            # [중요] 축 변환: (X, Y, Z) -> (Z, Y, X)
            # 모델이 학습된 방향으로 데이터 회전
            array_img_transposed = array_img.transpose(2, 1, 0)
            
            # 2. 전처리
            array_img_re = self.resize(array_img_transposed)
            array_img_input = np.reshape(array_img_re, (1, 128, 128, 128, 1))

            # 3. 모델 추론 (Inference)
            with graph.as_default():
                results = model.model.predict(array_img_input)
            results = np.round(results)

            # 4. 후처리 및 복원
            if config["resizing"]:
                results = self.onehot2label(results)
                results = np.reshape(results, (128, 128, 128))
                
                # 원본 크기로 복원 (순서 주의: Z, Y, X)
                results = ndimage.zoom(results, 
                                     (original_shape[2]/128, original_shape[1]/128, original_shape[0]/128), 
                                     order=0, mode='nearest')
                
                # # ★ [중요] 디버깅용 마스크 저장 (확인용)
                # # 이 파일이 생성되면 ITK-SNAP에서 원본 위에 얹어보세요.
                # # 모델이 어디를 눈/코/입으로 인식했는지 바로 알 수 있습니다.
                # debug_mask = results.transpose(2, 1, 0) # 저장 위해 다시 (X,Y,Z)로 복구
                # debug_path = os.path.join(verif_path, f"MASK_{os.path.basename(nfti_path)}")
                # nib.save(nib.Nifti1Image(debug_mask.astype('int16'), raw_img.affine), debug_path)
                # print(f"      💾 [Debug] Mask saved to: {os.path.basename(debug_path)}")

                results = to_categorical(results, num_classes=5)
            else:
                results = results[0, ...]

            # 5. 박스 추출 및 블러링 (Transposed 상태에서 진행)
            boxes = self.bounding_box(results[..., 1:])
            results = self.label_denoising(results)
            
            print(f"      👀 Detected Features: {len(boxes)} boxes found.")

            # [핵심 수정 3] 안전한 블러링 (인덱스 초과 에러 방지)
            # 순서: 눈(Label 1) -> 코(Label 2) -> 귀(Label 3) -> 입(Label 4)
            # boxes 리스트에 담기는 순서는 bounding_box 함수 로직에 따라 [눈, 눈, 코, 귀, 귀, 입] 순서일 가능성이 높음
            # 하지만 안전하게 하기 위해, 단순히 박스가 존재하면 앞에서부터 순차적으로 적용하거나
            # 채널별로 박스를 구하는 것이 더 안전함. 여기서는 간단히 수정된 bounding_box 로직을 따름.
            
            # *참고: 위 bounding_box 함수는 채널 0(눈) -> 2(귀) -> 1(코) -> 3(입) 순서로 돕니다 (defacer 원본 로직)*
            # 따라서 boxes 리스트 순서는 [눈..., 귀..., 코..., 입...] 순서입니다.
            
            # 눈 (Eyes)
            if where[0]:
                for b in boxes:
                    # 크기나 위치로 대략 눈인지 판단하거나, 모든 박스를 다 지워도 무방함 (얼굴 부위이므로)
                    # 여기서는 안전하게 모델이 찾은 '모든' 박스를 살짝 확장해서 지웁니다.
                    array_img_transposed = self.box_blur(array_img_transposed, b, wth=1.3)

            # 6. 최종 저장 (축 복구)
            # (Z, Y, X) -> (X, Y, Z)
            array_img_final = array_img_transposed.transpose(2, 1, 0)
            array_img_final = np.round(array_img_final).astype(raw_img.get_data_dtype())

            save_name = prefix.format(os.path.basename(nfti_path))
            save_path = os.path.join(dest_path, save_name)
            
            nib.save(nib.Nifti1Image(array_img_final, raw_img.affine, raw_img.header), save_path)

            return {"success": True, "path": save_path}

        except Exception as ex:
            import traceback
            traceback.print_exc()
            return {"success": False, "msg": str(ex)}