This is a classic library mismatch error, which is very common when working on shared servers or HPC clusters (like `cbi-lgn01` appears to be).

### What's Happening?

The `mamba_ssm` library uses a custom CUDA extension (`selective_scan_cuda`). This extension was compiled using a newer C++ compiler than the default one installed on your system's OS. When Python tries to load the extension, it looks at the system's default C++ library (`/lib64/libstdc++.so.6`), realizes it's too old to contain the required `CXXABI_1.3.15`, and crashes.

Since you are using a Conda environment (`thesis`), the cleanest way to fix this is to install a newer C++ standard library directly inside your environment and tell your system to use that one instead.

Here is how to fix it:

### Step 1: Install the updated C++ library in your Conda env

Make sure your `thesis` environment is activated, then install the `libstdcxx-ng` package from the `conda-forge` channel. This will provide an up-to-date version of `libstdc++.so.6`.

```bash
conda install -c conda-forge libstdcxx-ng

```

### Step 2: Link the Conda library path

Sometimes installing it isn't enough because Linux still defaults to searching the `/lib64/` system directory first. You need to tell the dynamic linker to look in your Conda environment's `lib` folder before looking at the system libraries.

Run this command to temporarily set the path for your current session:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

```

### Step 3: Verify the fix

Run your test command again to see if the import is now successful:

```bash
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"

```

---

### Step 4 (Optional but Recommended): Make it permanent

If Step 3 worked, you won't want to run the `export` command every single time you log in. You can configure your Conda environment to automatically set this variable whenever you activate it.

Run these commands while your `thesis` environment is activated:

```bash
# Create the activation and deactivation directories
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Write the export command to the activation script
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Write the restore command to the deactivation script
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

```

To test this, simply deactivate and reactivate your environment (`conda deactivate` then `conda activate thesis`).
