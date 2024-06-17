# Day 1 - Question 1

## CoT + Prompt Compression + Benchmark Evaluation

1. OpenAI API 실행 준비 (각자 부여된 API 키 확인)
https://platform.openai.com/docs/quickstart

2. LLMlingua2 설치 후 실행
https://github.com/microsoft/LLMLingua

3. CoT prompt 를 활용하여 GPT 3.5 모델을 GSM8k 데이터셋에 대해 평가한 코드 실행
https://github.com/FranxYao/chain-of-thought-hub/blob/main/gsm8k/gpt3.5turbo_gsm8k_complex.ipynb

4. 3 에서 쓰인 CoT prompt 를 LLMlinguage2 를 사용하여 300 token target 으로 compress 후 compress 하지 않은 경우와 결과 비교


## 작업 결과
- 50개씩 결과 테스트 실험

### `gpt3.5turbo_gsm8k_complex.py`
- Not compressed: 60%

### `gpt3.5turbo_gsm8k_complex_with_llmlingua.py`
|NUM_COMPRESSION_TOKEN|ACCURACY(%)|
|--|--|
|Not compressed|60.0|
|100|66.0|
|200|70.0|
|300|70.0|
|400|68.0|
|500|62.0|
|600|66.0|
|700|64.0|