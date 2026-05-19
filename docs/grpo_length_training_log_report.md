# LENGTH GRPO Training Log Report

## 분석 대상
- 로그 파일: `outputs/grpo_length_real_9rollout/training_log.txt`
- 분석 범위: step **1~1000** (완주)
- 실행 요약 레코드 존재: `train_runtime`, `train_loss` 확인됨

## 실행 완료 여부
- 진행 상태: `1000/1000` 완료
- 최종 요약:
  - `train_runtime`: **28737.0665s** (약 7시간 58분 57초)
  - `train_steps_per_second`: **0.035**
  - `train_samples_per_second`: **0.104**
  - `train_loss`: **0.015024**
  - 최종 `epoch`: **0.43**

## 오류/경고 점검
- `ERROR`: 0
- `Traceback`: 0
- `OOM`: 0
- `NaN`: 0
- `WARNING`: 7 (초기 vLLM/NCCL 관련 경고 포함)

해석:
- 치명 실패 없이 런은 정상 종료됨.
- 종료 시 NCCL teardown 경고가 보이더라도 결과 산출 자체를 깨는 형태는 아님.

## 핵심 지표 (전체 1000 step 평균)
- `reward`: **0.690326** (min 0.140741, max 1.0)
- `rewards/answer_accuracy_reward`: **0.634481**
- `rewards/answer_format_reward`: **0.913704**
- `loss`: **0.015023** (max 5.5571)
- `kl`: **0.374385** (max 138.0)
- `grad_norm`: **1.520884** (max 606.951477)
- `completion_length`: **23.947333**
- `reward_std`: **0.281317**

## 구간 추세 (100-step bin)
- 1~100: reward 0.5421 / acc 0.4793 / format 0.7937
- 101~200: reward 0.6939 / acc 0.6348 / format 0.9304
- 201~700: reward 약 0.695~0.709, format 약 0.903~0.934
- 701~800: reward 0.7292 (상대적 고점)
- 801~900: reward 0.7090, **kl 2.0227 / loss 0.0814** (불안정 구간)
- 901~1000: reward 0.7247 / acc 0.6711 / format 0.9389 (회복)

해석:
- 전반적으로 정확도/포맷 보상은 초반 대비 개선.
- 다만 800~900 구간에 큰 스파이크가 집중됨.

## 전략별 분포/정확도/인사이트 (LENGTH: direct/cot/long_cot)

### 1) 정책 로그 기준 분포 (step-level)
- `strategy/reward_best_tie_or_unclear_rate`: **0.956667**
- `strategy/tie_break_applied_rate`: **0.956667**
- `strategy/tie_break_to_direct_rate`: **0.956667**
- `strategy/effective_best_direct_rate`: **0.977000**
- `strategy/effective_best_cot_rate`: **0.014667**
- `strategy/effective_best_long_cot_rate`: **0.008333**
- `strategy/strategy_bonus_applied_rate`: **0.043333**

핵심:
- best 판정이 tie/unclear로 끝나는 비율이 매우 높고, tie-break가 direct로 고정되어 최종 분포가 direct로 수렴.

### 2) 전략별 평균 성능 (training_log의 strategy mean reward)
- `mean_reward_direct`: **0.735044**
- `mean_reward_cot`: **0.702756**
- `mean_reward_long_cot`: **0.633178**

gap(평균):
- `direct - cot`: **+0.032289**
- `direct - long_cot`: **+0.101867**
- `cot - long_cot`: **+0.069578**

해석:
- direct가 일관적으로 우세하며, 특히 long_cot 대비 격차가 큼.
- cot는 direct와의 격차가 작아, tie-break 정책 영향이 더 크게 반영되는 구조.

### 3) strategy_debug.jsonl 기반 전략별 실측 지표 (slot-level, 27,000 샘플)
전략별 샘플 수(강제 rollout):
- `direct`: 9,000 / `cot`: 9,000 / `long_cot`: 9,000

전략별 포맷/전략 준수율:
- `format_ok`  
  - direct: **0.321556**
  - cot: **0.312444**
  - long_cot: **0.281333**
- `parsed_strategy == forced_strategy` (전략 준수율)  
  - direct: **0.998556**
  - cot: **0.085778**
  - long_cot: **0.093000**

전략별 보상/정확도 proxy:
- `base_reward` 평균  
  - direct: **0.735044**
  - cot: **0.702756**
  - long_cot: **0.633178**
- `base_reward >= 1.0` 비율 (정답+형식 동시 충족 proxy)  
  - direct: **0.678222**
  - cot: **0.644556**
  - long_cot: **0.580667**

해석:
- cot/long_cot의 핵심 병목은 정답 자체보다도 **전략 태그/형식 준수율**(전략 준수율 ~9%)에서 크게 발생.
- direct는 구조적으로 태그 요구가 약해 전략 준수율이 사실상 100%에 가깝고, 그 이점이 누적됨.

### 4) 시간축 인사이트 (early/mid/late)
구간별(0~199 / 400~599 / 800~999 step) 변화:
- direct `base_reward`: 0.662 -> 0.764 -> 0.753
- cot `base_reward`: 0.595 -> 0.728 -> 0.749
- long_cot `base_reward`: 0.596 -> 0.618 -> 0.648

전략 준수율(`parsed_strategy==forced`) 변화:
- cot: **0.0156 -> 0.0711 -> 0.2272**
- long_cot: **0.0094 -> 0.0733 -> 0.2267**

해석:
- cot/long_cot은 후반으로 갈수록 전략 준수율이 개선되지만, 절대치가 아직 낮아 direct 우세를 뒤집기엔 부족.
- 즉, 이 실험은 "전략 학습이 아예 안 되는 상태"는 아니고, **느리게 따라오는 상태**에 가깝다.

## 스파이크/이상치 상세
- KL 상위:
  - step 840: **138.0**
  - step 819: **35.75**
  - step 703: **6.09375**
- Loss 상위:
  - step 840: **5.5571**
  - step 819: **1.437**
  - step 703: **0.2449**
- Grad norm 상위:
  - step 840: **606.9515**
  - step 819: **207.5960**
  - step 788: **28.2760**

임계치 빈도:
- `kl >= 1`: 35회, `kl >= 2`: 12회, `kl >= 10`: 2회
- `loss >= 0.1`: 7회, `loss >= 1`: 2회
- `grad_norm >= 10`: 7회, `>=100`: 2회

해석:
- 대부분 step은 안정 구간이지만, 소수 step에서 매우 큰 outlier가 존재.
- outlier는 주로 700~900 구간에 집중.

## 종합 결론
1. LENGTH 실험은 설정된 1000 step을 정상 완주했다.
2. 포맷/정답 보상은 전반적으로 개선되며 학습은 진행됐다.
3. 전략 선택 관점에서는 tie 비율이 높고 tie-break 규칙 영향으로 direct 편향이 강하게 나타난다.
4. 후반 일부 구간에서 KL/loss/grad 급등 스파이크가 관측되어, 재현 실험 시 해당 구간(특히 819, 840 주변) 모니터링이 필요하다.
