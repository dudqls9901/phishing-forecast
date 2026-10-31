# 상태 적응형 사회공학 디지털 트윈 — 본선 시제품

## 실행
```
pip install -r requirements.txt streamlit altair
python fixed_layer.py      # 고정층 학습 (약 13초, fixed_layer_model.json 생성. 이미 있으면 생략 가능)
streamlit run app.py       # 대시보드
```
맥락 판단에 LLM을 쓰려면 `ANTHROPIC_API_KEY` 환경 변수를 설정. 없으면 키워드 규칙으로 자동 대체.

## 파일
| 파일 | 역할 |
|---|---|
| params.py | 근거 파라미터 (출처·근거 강도·확인 필요 표시) |
| fixed_layer.py | 고정층: Spamley 공개 통계 기반 합성 → 로지스틱 회귀 → 검증 |
| scenarios.py | 공격 시나리오 구조(대본 없음), 맥락 매칭 규칙 |
| context_llm.py | 맥락층: LLM 맞물림 판단 + 규칙 대체 |
| simulate.py | 1,000회 반복 시뮬레이션, 방어 단계 결정, 시연 프리셋 |
| bayes_update.py | 현실과의 동기화: 반응 기록으로 개인 효과 갱신 |
| sensitivity.py | 민감도 분석 |
| app.py | Streamlit 대시보드 (5개 탭) |

## 단독 실행 점검
```
python simulate.py         # 프리셋 3종 + 시간 압박 반대 연구 시나리오
python bayes_update.py     # 개인화 수렴 확인
python sensitivity.py      # 민감도 분석
```
시제품은 가상 사용자·가상 일정·가상 반응 기록만 사용한다.
