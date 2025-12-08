#!/bin/bash
# 파일 정리 스크립트

echo "파일 정리 시작..."

# 문서 파일들을 05_docs로 이동
mv -f API_DATABASE_SPEC.md 05_docs/ 2>/dev/null || true
mv -f DATA_TABLES.md 05_docs/ 2>/dev/null || true
mv -f DASHBOARD_GUIDE.md 05_docs/ 2>/dev/null || true
mv -f FILTERING_STRATEGIES.md 05_docs/ 2>/dev/null || true
mv -f GPU_QUICK_START.md 05_docs/ 2>/dev/null || true
mv -f MODEL_COMPARISON.md 05_docs/ 2>/dev/null || true
mv -f PIPELINE_GUIDE.md 05_docs/ 2>/dev/null || true

# 앱 파일들을 새 폴더로
mkdir -p 06_apps
mv -f app_dashboard.py 06_apps/ 2>/dev/null || true
mv -f requirements_dashboard.txt 06_apps/ 2>/dev/null || true

# 쉘 스크립트들을 scripts 폴더로
mkdir -p 07_scripts
mv -f cleanup.sh 07_scripts/ 2>/dev/null || true
mv -f cleanup_env.sh 07_scripts/ 2>/dev/null || true
mv -f run_enhanced_pipeline.sh 07_scripts/ 2>/dev/null || true
mv -f run_fast_pipeline.sh 07_scripts/ 2>/dev/null || true
mv -f run_full_pipeline.sh 07_scripts/ 2>/dev/null || true
mv -f run_gpu.sh 07_scripts/ 2>/dev/null || true
mv -f run_sequence_pipeline.sh 07_scripts/ 2>/dev/null || true

# 개인 데이터 파일
mkdir -p 08_personal_data
mv -f 2024-12-03~2025-12-03.csv 08_personal_data/ 2>/dev/null || true

echo "정리 완료!"
echo ""
echo "최상위 폴더 파일:"
ls -1 *.md 2>/dev/null || echo "  (문서 없음)"
