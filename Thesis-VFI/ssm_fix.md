# 修復 mamba_ssm `libstdc++` 版本不符問題

> 常見於共享伺服器或 HPC 叢集環境中，系統預設的 C++ 標準函式庫版本過舊。

## 症狀

```
ImportError: .../selective_scan_cuda.cpython-311-x86_64-linux-gnu.so:
undefined symbol: ... CXXABI_1.3.15
```

`mamba_ssm` 的 CUDA extension 使用較新的 C++ 編譯器編譯，而系統的 `/lib64/libstdc++.so.6` 版本過舊，缺少 `CXXABI_1.3.15`。

## 修復步驟

### Step 1：在 Conda 環境中安裝新版 libstdc++

```bash
conda activate thesis
conda install -c conda-forge libstdcxx-ng
```

### Step 2：設定動態連結器路徑

讓 Conda 環境的 `lib/` 優先於系統目錄：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Step 3：驗證

```bash
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
```

### Step 4（建議）：設為永久生效

```bash
# 建立 activate/deactivate 腳本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Activate 時自動設定
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

# Deactivate 時還原
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
EOF
```

測試：`conda deactivate && conda activate thesis`，然後重新執行 Step 3 驗證。
