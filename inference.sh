#!/bin/bash

cd "$(dirname "$0")"

echo "====================================="
echo "BẮT ĐẦU CHẠY ĐÁNH GIÁ (INFERENCE)"
echo "====================================="

python scripts/inference.py
