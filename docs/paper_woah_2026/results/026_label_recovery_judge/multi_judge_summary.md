# Cross-judge label/target recoverability (run 026 multi-judge)

4 independent judges. Headline = I-CEM significantly beats the token-matched window on label fidelity AND target recovery.

## Hate-label fidelity (%) by judge

| Judge | imp-only | win r=2 | win r=3 (matched) | I-CEM | I-CEM vs matched (p) |
|  --- | --- | --- | --- | --- | --- |
| qwen/qwen3.5-9b (plain-completion) | 76.9 | 84.9 | 87.7 | 89.7 | 0.0395* |
| gemma-4-26b-a4b-it (tool-call) | 53.1 | 73.3 | 79.6 | 82.1 | 0.2451 |
| openai/gpt-oss-20b (tool-call) | 65.7 | 76.8 | 81.9 | 84.4 | 0.237 |
| qwen/qwen3.5-9b (tool-call) | 61.2 | 76.8 | 77.8 | 83.0 | 0.0099* |

## Target recovery (%) by judge

| Judge | imp-only | win r=2 | win r=3 (matched) | I-CEM | I-CEM vs matched (p) |
|  --- | --- | --- | --- | --- | --- |
| qwen/qwen3.5-9b (plain-completion) | 35.5 | 52.1 | 58.2 | 67.3 | 0.0* |
| gemma-4-26b-a4b-it (tool-call) | 58.9 | 69.9 | 73.4 | 75.3 | 0.4296 |
| openai/gpt-oss-20b (tool-call) | 56.6 | 66.1 | 69.5 | 75.9 | 0.0154* |
| qwen/qwen3.5-9b (tool-call) | 50.3 | 63.4 | 67.5 | 73.2 | 0.0222* |

## Non-hate misread as hate (NO->YES %) by judge

| Judge | imp-only | win r=2 | win r=3 (matched) | I-CEM |
|  --- | --- | --- | --- | --- |
| qwen/qwen3.5-9b (plain-completion) | 53.0 | 34.5 | 28.0 | 24.5 |
| gemma-4-26b-a4b-it (tool-call) | 19.5 | 16.8 | 13.3 | 13.3 |
| openai/gpt-oss-20b (tool-call) | 31.5 | 21.3 | 25.8 | 28.1 |
| qwen/qwen3.5-9b (tool-call) | 33.9 | 28.4 | 27.5 | 23.9 |

## Headline replication verdict

- **qwen/qwen3.5-9b (plain-completion)**: label-fidelity win YES (p=0.0395); target-recovery win YES (p=0.0).
- **gemma-4-26b-a4b-it (tool-call)**: label-fidelity win directional (p=0.2451); target-recovery win directional (p=0.4296).
- **openai/gpt-oss-20b (tool-call)**: label-fidelity win directional (p=0.237); target-recovery win YES (p=0.0154).
- **qwen/qwen3.5-9b (tool-call)**: label-fidelity win YES (p=0.0099); target-recovery win YES (p=0.0222).
