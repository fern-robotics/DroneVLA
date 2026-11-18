.PHONY: validate-tasks sample-tasks build-tasks b0-grpo
validate-tasks:
	python scripts/validate_uavflow_pool.py --train-dir $${TRAIN:-/workspace/uavflow_train_tasks} --test-dir $${TEST:-/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons}
sample-tasks:
	python scripts/sample_uavflow_tasks.py --train-dir $${TRAIN:-/workspace/uavflow_train_tasks} --test-dir $${TEST:-/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons} --out $${RUN:-/workspace/runs/b0_grpo_full}/manifest.json --count 100 --seed 7
b0-grpo:
	bash scripts/reproduce_b0_grpo.sh
