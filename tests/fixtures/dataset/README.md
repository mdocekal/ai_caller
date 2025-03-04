---
configs:
- config_name: default
  data_files:
  - split: train
    path: "short/train.jsonl"
  - split: test
    path: "short/test.jsonl"
- config_name: short
  data_files:
  - split: train
    path: "short/train.jsonl"
  - split: test
    path: "short/test.jsonl"
- config_name: long
  data_files:
  - split: train
    path: "long/train.jsonl"
  - split: test
    path: "long/test.jsonl"
---
