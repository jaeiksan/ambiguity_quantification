# Ambiguity Quantification 실험

이 폴더는 전체 쿼리 대상 ambiguity scoring 실험을 재현하는 데 필요한 코드, 입력 데이터, 결과 파일만 따로 정리한 전달용 폴더입니다.

## 구성

- `ambiguity_score/`
  - ambiguity scoring을 수행하는 핵심 패키지
- `scripts/run_full_query_experiment.sh`
  - 전체 쿼리 실험 재현용 실행 스크립트
- `data/benchmarks/ambigqa.validation.goldfull.jsonl`
  - gold disambiguated query가 있는 AmbigQA validation 전체 입력 파일
- `results/scores_qwen14b_goldfull_strict.jsonl`
  - 전체 쿼리 실험 결과 파일
- `.env.example`
  - 환경 변수 예시 파일
- `requirements.txt`
  - pip 기반 최소 의존성 목록
- `environment.yml`
  - conda 환경 재현용 파일

## 실험 설정

- 모델: `Qwen/Qwen2.5-14B-Instruct`
- 백엔드: `hf`
- 사용 지표:
  - `infogain`
  - `sample_rep`
  - `semantic_entropy`
  - `token_entropy`
- `k_samples=8`
- semantic clustering 설정:
  - `threshold=0.93`
  - `linkage=complete`

## 실행 방법

이 폴더에서 아래 순서로 실행하면 됩니다.

```bash
conda activate apa
cd /path/to/full_query_experiment
cp .env.example .env
# 필요한 값으로 .env 수정
bash scripts/run_full_query_experiment.sh
```

새 환경에서 바로 재현하려면 아래 둘 중 하나를 사용하면 됩니다.

```bash
pip install -r requirements.txt
```

또는

```bash
conda env create -f environment.yml
conda activate ambiguity-full-query
```

결과 파일은 아래 경로에 저장됩니다.

```bash
results/scores_qwen14b_goldfull_strict.jsonl
```

## 코드 설명

- `ambiguity_score/run.py`
  - 메인 CLI 진입점입니다. 입력 JSONL을 읽고 선택한 지표를 계산합니다.
- `ambiguity_score/llm_client.py`
  - Hugging Face / OpenAI 호환 / fake 백엔드를 다루는 클라이언트입니다.
- `ambiguity_score/measures/infogain.py`
  - 원 query와 disambiguated query의 entropy 차이를 계산합니다.
- `ambiguity_score/measures/sample_rep.py`
  - 동일 query에 대해 여러 번 생성한 답변들의 반복 불확실성을 계산합니다.
- `ambiguity_score/measures/semantic_entropy.py`
  - 여러 답변을 semantic clustering한 뒤 entropy를 계산합니다.
- `ambiguity_score/measures/token_entropy.py`
  - query 자체의 mean token entropy를 계산합니다.

## 참고 사항

- 이 전달 폴더에는 탐색용 스크립트, 100개 샘플 실험 결과, 중간 산출물은 포함하지 않았습니다.
- `semantic_entropy`는 이전 실험에서 `0.0`으로 과도하게 몰리던 문제를 완화하기 위해 stricter setting을 사용했습니다.
