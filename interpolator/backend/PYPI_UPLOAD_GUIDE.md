# Guide: Uploading pydis_nn to PyPI

This guide walks you through uploading your `pydis_nn` package to PyPI (Python Package Index).

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **TestPyPI Account** (recommended for testing): https://test.pypi.org/account/register/
3. **Build tools**: Install build tools
   ```bash
   pip install --upgrade build twine
   ```

## Important: Check Package Name Availability

⚠️ **CRITICAL**: The name `pydis_nn` might already be taken on PyPI. Check availability first:

```bash
pip search pydis_nn  # Won't work, but you can check at pypi.org
```

Or visit: https://pypi.org/project/pydis_nn/

**If the name is taken**, you'll need to use a different name. Options:
- Add your username prefix: `harvey-pydis-nn`
- Use a more descriptive name: `pydis-nn-5d-interpolator`
- Use a hyphenated version: `pydis-nn-interpolator`

**If you need to change the name**, update `pyproject.toml`:
```toml
[project]
name = "your-chosen-name"  # Must be unique on PyPI
```

## Step-by-Step Upload Process

### Step 1: Prepare Your Package

1. **Ensure everything is committed** (PyPI doesn't use .git, but good practice):
   ```bash
   cd backend
   git status  # Check for uncommitted changes
   ```

2. **Clean previous builds**:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ```

3. **Verify your pyproject.toml** is correct:
   - ✅ Package name
   - ✅ Version (currently 1.0.0)
   - ✅ Description
   - ✅ Dependencies
   - ✅ Author information

### Step 2: Build the Package

Build source distribution (sdist) and wheel:

```bash
cd backend
python -m build
```

This creates:
- `dist/pydis_nn-1.0.0.tar.gz` (source distribution)
- `dist/pydis_nn-1.0.0-py3-none-any.whl` (wheel)

### Step 3: Test on TestPyPI (Recommended)

**Always test on TestPyPI first!**

1. **Upload to TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. **Enter credentials** when prompted:
   - Username: `__token__`
   - Password: Your TestPyPI API token (create at https://test.pypi.org/manage/account/token/)

3. **Test installation** from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pydis_nn
   ```

   Note: `--extra-index-url` is needed because dependencies (like numpy, tensorflow) come from regular PyPI.

4. **Verify it works**:
   ```bash
   python -c "import pydis_nn; print(pydis_nn.__version__)"
   ```

### Step 4: Upload to Production PyPI

Once tested on TestPyPI, upload to production:

1. **Build again** (to ensure fresh build):
   ```bash
   rm -rf build/ dist/ *.egg-info/
   python -m build
   ```

2. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

3. **Enter credentials**:
   - Username: `__token__`
   - Password: Your PyPI API token (create at https://pypi.org/manage/account/token/)

### Step 5: Verify Upload

1. **Check your package page**:
   https://pypi.org/project/pydis_nn/ (or your chosen name)

2. **Test installation**:
   ```bash
   pip install pydis_nn
   ```

3. **Verify it works**:
   ```bash
   python -c "from pydis_nn import data, neuralnetwork, utils; print('Success!')"
   ```

## Creating API Tokens

### TestPyPI Token

1. Go to: https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Name it (e.g., "pydis_nn-upload")
4. Scope: "Entire account" or "Project: pydis_nn" (if available)
5. Copy the token (starts with `pypi-`)

### PyPI Token

1. Go to: https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name it (e.g., "pydis_nn-production")
4. Scope: "Entire account" or "Project: pydis_nn"
5. Copy the token (starts with `pypi-`)

⚠️ **Security**: Never commit API tokens to git! Store them securely.

## Troubleshooting

### Error: "File already exists"

If you get an error that the version already exists:
- **Increment the version** in `pyproject.toml`:
  ```toml
  version = "1.0.1"  # or 1.1.0, 2.0.0, etc.
  ```
- **Rebuild and re-upload**

### Error: "Package name already taken"

The package name `pydis_nn` is already registered. You must:
1. Choose a different name
2. Update `pyproject.toml` with the new name
3. Rebuild and upload

### Error: Missing files

Ensure all necessary files are included. Check:
- `pydis_nn/__init__.py` exists
- All Python modules are in `pydis_nn/` directory
- `pyproject.toml` has correct package discovery:
  ```toml
  [tool.setuptools]
  packages = {find = {}}
  ```

### Dependencies not found during install

If users get dependency errors:
- Ensure all dependencies are listed in `pyproject.toml` under `[project.dependencies]`
- Check that version constraints are correct
- Test in a clean virtual environment

## Updating the Package

When you want to release a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "1.0.1"  # or 1.1.0, 2.0.0, etc.
   ```

2. **Update version** in `docs/source/conf.py` (if you want docs to match):
   ```python
   version = '1.0.1'
   release = '1.0.1'
   ```

3. **Rebuild and upload**:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   python -m build
   python -m twine upload dist/*
   ```

## Alternative: Using .pypirc for Credentials

Instead of entering credentials each time, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_ACTUAL_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Then upload with:
```bash
twine upload --repository pypi dist/*  # Production
twine upload --repository testpypi dist/*  # Test
```

⚠️ **Security**: Make sure `~/.pypirc` has correct permissions:
```bash
chmod 600 ~/.pypirc
```

## Quick Reference Commands

```bash
# 1. Install build tools
pip install --upgrade build twine

# 2. Clean previous builds
rm -rf build/ dist/ *.egg-info/

# 3. Build package
cd backend
python -m build

# 4. Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# 5. Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pydis_nn

# 6. Upload to production PyPI
python -m twine upload dist/*

# 7. Install from production PyPI
pip install pydis_nn
```

## Package Metadata Checklist

Before uploading, verify in `pyproject.toml`:

- [x] Package name is unique and available
- [x] Version is correct (currently 1.0.0)
- [x] Description is clear and informative
- [x] Author name and information
- [x] License specified (MIT)
- [x] All dependencies listed with correct versions
- [x] Python version requirements (>=3.10)
- [x] Keywords for discoverability
- [x] Classifiers for categorization
- [x] README file referenced (if you want it on PyPI page)

## Notes

- **Package name on PyPI**: PyPI uses case-insensitive matching, but the name in `pyproject.toml` should match exactly (hyphens vs underscores matter for the package name, but PyPI normalizes them)
- **Version numbers**: Follow semantic versioning (major.minor.patch)
- **First upload**: The first upload creates the package on PyPI
- **Subsequent uploads**: Must have a higher version number
- **Documentation**: PyPI will display your README.md if referenced in pyproject.toml (you have `readme = "README.md"`)

Good luck with your PyPI upload! 🚀

