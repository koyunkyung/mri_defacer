
---

- [사전 준비사항](#-사전-준비사항)
  - [1단계: 프로젝트 다운로드](#1단계-프로젝트-다운로드)
  - [2단계: Conda 설치](#2단계-conda-설치)
  - [3단계: 환경 설정](#3단계-환경-설정)
- [실행 방법](#-실행-방법)
  - [Step 1: DICOM → NIfTI 변환](#step-1-dicom--nifti-변환-to3dpy)
  - [Step 2: Defacing 실행](#step-2-defacing-실행-run_defacerpy)
- [결과물 및 폴더 구조](#-결과물-및-폴더-구조)

---

## ☝🏻 사전 준비사항

### 1단계: 프로젝트 다운로드

GitHub에서 프로젝트를 다운로드합니다.

#### Windows 사용자

```cmd
# 1. 원하는 위치로 이동 (예: 문서 폴더)
cd %USERPROFILE%\Documents

# 2. 프로젝트 다운로드
git clone https://github.com/koyunkyung/mri_reface.git

# 3. 다운로드된 폴더로 이동
cd mri_reface
```

> **Git이 설치되어 있지 않다면?**  
> GitHub 페이지에서 **Code → Download ZIP** 클릭 후 압축 해제

<details>
<summary><b>Mac/Linux 사용자</b></summary>

```bash
# 1. 원하는 위치로 이동 (예: 홈 디렉토리)
cd ~

# 2. 프로젝트 다운로드
git clone https://github.com/koyunkyung/mri_reface.git

# 3. 다운로드된 폴더로 이동
cd mri_reface
```

</details>

**다운로드 후 폴더 구조:**
```
mri_reface/
├── env.yaml           # Conda 환경 설정 파일
├── to3d.py            # DICOM → NIfTI 변환 스크립트
├── defacer.py         # Defacing 모델 코드
├── run_defacer.py     # Defacing 실행 스크립트
└── model/             # 학습된 모델 파일
```

---

### 2단계: Conda 설치

Conda는 Python 환경을 관리해주는 도구입니다. 이 프로젝트는 **Python 3.7**과 특정 버전의 TensorFlow가 필요하므로 Conda 환경 사용을 권장합니다.

#### Windows 사용자

1. [Miniconda 다운로드 페이지](https://docs.conda.io/en/latest/miniconda.html)에서 **Windows 64-bit** 설치 파일 다운로드
2. 설치 파일 실행
3. **"Add Miniconda3 to my PATH environment variable"** 체크박스 선택 (권장)
4. 설치 완료 후 **명령 프롬프트(cmd)** 또는 **Anaconda Prompt** 실행

**설치 확인:**
```cmd
conda --version
```

<details>
<summary><b>Mac 사용자</b></summary>

1. [Miniconda 다운로드 페이지](https://docs.conda.io/en/latest/miniconda.html) 접속
2. Mac 칩에 맞는 버전 선택:
   - **Apple Silicon (M1/M2/M3/M4)**: `Miniconda3 macOS Apple M1 64-bit pkg`
   - **Intel 칩**: `Miniconda3 macOS Intel x86 64-bit pkg`
3. 설치 파일 실행 후 안내에 따라 진행
4. **터미널** 재시작

**설치 확인:**
```bash
conda --version
```

</details>

<details>
<summary><b>Linux 사용자</b></summary>

```bash
# Miniconda 설치 스크립트 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 실행
bash Miniconda3-latest-Linux-x86_64.sh

# 설치 중 라이선스 동의: yes 입력
# 설치 경로: Enter (기본값 사용)
# conda init 실행 여부: yes 입력

# 터미널 재시작 또는 다음 명령 실행
source ~/.bashrc
```

**설치 확인:**
```bash
conda --version
```

</details>

---

### 3단계: 환경 설정

`mri_reface` 폴더 안에서 Conda 환경을 생성합니다.

#### 환경 생성 및 활성화

**Windows 사용자:**
```cmd
# 1. mri_reface 폴더에 있는지 확인 (위 2단계에서 이미 이동했다면 생략)
cd %USERPROFILE%\Documents\mri_reface

# 2. Conda 환경 생성 (최초 1회만 실행, 5-10분 소요)
conda env create -f env.yaml

# 3. 환경 활성화 (매번 작업 전 실행)
conda activate deface
```

<details>
<summary><b>Mac/Linux 사용자</b></summary>

```bash
# 1. mri_reface 폴더에 있는지 확인 (위 2단계에서 이미 이동했다면 생략)
cd ~/mri_reface

# 2. Conda 환경 생성 (최초 1회만 실행, 5-10분 소요)
conda env create -f env.yaml

# 3. 환경 활성화 (매번 작업 전 실행)
conda activate deface
```

</details>

> **참고**: 환경이 활성화되면 터미널 프롬프트 앞에 `(deface)`가 표시됩니다.

**환경 설정 확인:**
```bash
python --version
# 출력: Python 3.7.x
```

---

## ✌🏻 실행 방법

> **중요**: 아래 모든 명령어는 `mri_reface` 폴더 안에서 실행해야 합니다.  
> 터미널에서 현재 위치 확인: `pwd` (Mac/Linux) 또는 `cd` (Windows)

### Step 1: DICOM → NIfTI 변환 (`to3d.py`)

DICOM 파일들을 3D NIfTI 형식(`.nii.gz`)으로 변환합니다.

#### 사전 준비: 원본 데이터 배치

변환할 DICOM 파일들을 `mri_reface/raw_data` 폴더에 넣어주세요:

```
mri_reface/
└── raw_data/                    # 이 폴더에 원본 DICOM 넣기
    ├── Patient_001/
    │   ├── 301/
    │   ├── 501/
    │   └── ...
    └── Patient_002/
        └── ...
```

#### Windows 사용자

```cmd
# mri_reface 폴더로 이동 (이미 이동했다면 생략)
cd %USERPROFILE%\Documents\mri_reface

# 환경 활성화 확인 (프롬프트에 (deface) 표시되어야 함)
conda activate deface

# 변환 실행
python to3d.py --input ./raw_data --output ./processed/3d_input
```

<details>
<summary><b>Mac/Linux 사용자</b></summary>

```bash
# mri_reface 폴더로 이동 (이미 이동했다면 생략)
cd ~/mri_reface

# 환경 활성화 확인 (프롬프트에 (deface) 표시되어야 함)
conda activate deface

# 변환 실행
python to3d.py --input ./raw_data --output ./processed/3d_input
```

</details>

#### 📝 옵션 설명

| 옵션 | 필수 여부 | 설명 |
|------|----------|------|
| `--input` | ✅ 필수 | 원본 DICOM 파일이 있는 폴더 경로 |
| `--output` | ✅ 필수 | 변환된 NIfTI 파일을 저장할 폴더 경로 |

#### 예상 실행 시간
- 환자 1명당 약 1-3분 소요 (파일 수에 따라 다름)

---

### Step 2: Defacing 실행 (`run_defacer.py`)

NIfTI 파일에서 얼굴 부위(눈, 코, 귀, 입)를 제거하여 익명화합니다.

#### Windows 사용자

```cmd
# mri_reface 폴더에서 실행 (환경 활성화 상태에서)
python run_defacer.py --input ./processed/3d_input --output ./processed/defaced_output
```

<details>
<summary><b>Mac/Linux 사용자</b></summary>

```bash
# mri_reface 폴더에서 실행 (환경 활성화 상태에서)
python run_defacer.py --input ./processed/3d_input --output ./processed/defaced_output
```

</details>

#### 📝 옵션 설명

| 옵션 | 필수 여부 | 설명 |
|------|----------|------|
| `--input` | ✅ 필수 | Step 1에서 생성된 NIfTI 파일 폴더 |
| `--output` | ✅ 필수 | Defacing 결과를 저장할 폴더 |


---

## 👌🏻 결과물 및 폴더 구조

프로그램이 성공적으로 실행되면 다음과 같은 구조로 결과물이 생성됩니다:

```
mri_reface/
├── raw_data/                          # 원본 DICOM 파일 (직접 넣어야 함)
│   ├── Patient_001/
│   │   ├── 301/
│   │   ├── 501/
│   │   └── ...
│   └── Patient_002/
│
├── processed/                         # 자동 생성되는 결과 폴더
│   ├── 3d_input/                      # Step 1 결과: NIfTI 변환 파일
│   │   ├── Patient_001/
│   │   │   ├── Patient_001_T1_MPRAGE.nii.gz
│   │   │   ├── Patient_001_T2_FLAIR.nii.gz
│   │   │   └── ...
│   │   └── Patient_002/
│   │       └── ...
│   │
│   └── defaced_output/                # Step 2 결과: Defacing 완료 파일
│       ├── Patient_001/
│       │   ├── defaced_Patient_001_T1_MPRAGE.nii.gz
│       │   ├── defaced_Patient_001_T2_FLAIR.nii.gz
│       │   └── ...
│       └── Patient_002/
│           └── ...
│
├── env.yaml                           # Conda 환경 설정 파일
├── to3d.py                            # DICOM → NIfTI 변환 스크립트
├── defacer.py                         # Defacing 모델 코드
├── run_defacer.py                     # Defacing 실행 스크립트
└── model/                             # 학습된 모델 파일
```


## 💡 추가 팁

### NIfTI 파일 확인 방법

#### VSCode에서 확인 (권장)
1. VSCode에서 **Extensions** 열기 (`Ctrl+Shift+X`)
2. **"NiiVue"** 검색 후 설치
3. `.nii.gz` 파일 클릭하면 3D 뷰어로 바로 확인 가능

#### 전문 뷰어 사용
- [ITK-SNAP](http://www.itksnap.org/) - 무료, 크로스플랫폼
- [3D Slicer](https://www.slicer.org/) - 무료, 고급 기능
