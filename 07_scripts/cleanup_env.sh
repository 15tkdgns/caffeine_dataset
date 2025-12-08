#!/bin/bash
# 안쓰는 라이브러리 정리 스크립트

echo "=============================================================="
echo "conda 환경 정리 (gemini_gpu)"
echo "=============================================================="
echo ""

# 현재 설치된 패키지 수
echo "[1/4] 현재 상태 확인..."
source /root/miniconda3/bin/activate gemini_gpu
BEFORE_COUNT=$(conda list | wc -l)
echo "  설치된 패키지: $BEFORE_COUNT개"
echo ""

# 불필요한 패키지 정리
echo "[2/4] 불필요한 패키지 제거..."

# 사용하지 않는 패키지들
UNUSED_PACKAGES=(
    "matplotlib"
    "seaborn"
    "jupyter"
    "notebook"
    "ipython"
    "ipykernel"
    "lightgbm"  # CPU 버전만 사용
)

for pkg in "${UNUSED_PACKAGES[@]}"; do
    if conda list | grep -q "^$pkg "; then
        echo "  - $pkg 제거 중..."
        conda remove -y "$pkg" >/dev/null 2>&1 || pip uninstall -y "$pkg" >/dev/null 2>&1
    fi
done

echo "  ✅ 불필요한 패키지 제거 완료"
echo ""

# conda clean
echo "[3/4] conda 캐시 정리..."
conda clean -a -y >/dev/null 2>&1
echo "  ✅ 캐시 정리 완료"
echo ""

# pip cache
echo "[4/4] pip 캐시 정리..."
pip cache purge >/dev/null 2>&1
echo "  ✅ pip 캐시 정리 완료"
echo ""

# 정리 후 상태
AFTER_COUNT=$(conda list | wc -l)
echo "=============================================================="
echo "정리 완료"
echo "=============================================================="
echo "  패키지: $BEFORE_COUNT → $AFTER_COUNT개"
echo "  절감: $((BEFORE_COUNT - AFTER_COUNT))개"
echo ""

# 디스크 사용량
df -h / | tail -1

echo ""
echo "✅ 환경 정리 완료!"
