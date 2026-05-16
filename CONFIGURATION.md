# GitaWise Configuration System

## Overview

A centralized configuration system has been implemented for the GitaWise project. All configuration values, file paths, API credentials, and settings are managed through a single `config.py` file located in the project root.

## Table of Contents
1. [What Was Done](#what-was-done)
2. [Quick Start](#quick-start)
3. [Configuration Categories](#configuration-categories)
4. [How to Use](#how-to-use)
5. [Making Changes](#making-changes)
6. [Adding New Configurations](#adding-new-configurations)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## What Was Done

### Files Created
- **config.py** - Central configuration file with 46 variables
- **CONFIGURATION.md** - This comprehensive guide

### Files Modified
1. **backend/gita_vector_store/create_chunks.py** - Now uses config values
2. **backend/gita_vector_store/generate_embeddings.py** - Now uses config values
3. **backend/gita_vector_store/upload_to_qdrant.py** - Now uses config values

### Key Improvements
- ✅ **Single Source of Truth** - All config in one place
- ✅ **No Code Duplication** - Settings defined once, used everywhere
- ✅ **Easy Maintenance** - Change once, update everywhere
- ✅ **Consistency** - All modules guaranteed to use same values
- ✅ **Professional Standard** - Industry-standard practice

---

## Quick Start

### Basic Usage
```python
# In any Python file:
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLEAN_GITA_CSV, EMBEDDING_MODEL_NAME

# Use the values
df = pd.read_csv(CLEAN_GITA_CSV)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
```

### Change Any Setting
Edit `config.py`:
```python
# Change one value
EMBEDDING_BATCH_SIZE = 64  # Changed from 32

# Save - all modules automatically use the new value!
```

### Run Scripts
All updated scripts automatically use config values:
```bash
python backend/gita_vector_store/create_chunks.py
python backend/gita_vector_store/generate_embeddings.py
python backend/gita_vector_store/upload_to_qdrant.py
```

---

## Configuration Categories

### 1. Project Paths (6 variables)
Central directory references for your project.

```python
PROJECT_ROOT          # /c/Users/ritam/Desktop/GitaWise
BACKEND_DIR          # Project backend directory
FRONTEND_DIR         # Project frontend directory
DATASETS_DIR         # Data files directory
NOTEBOOKS_DIR        # Jupyter notebooks directory
SCRIPTS_DIR          # Scripts directory
```

### 2. Data File Paths (7 variables)
File paths for datasets and embeddings.

```python
# Raw and processed data
CLEAN_GITA_CSV              # datasets/clean_gita.csv
ENRICHED_GITA_CSV           # datasets/enriched_gita.csv
BHAGWAD_GITA_CSV            # datasets/Bhagwad_Gita.csv
BHAGWAD_GITA_JSON           # datasets/Bhagwad_Gita.json

# Vector store and embeddings
GITA_CHUNKS_CSV             # datasets/gita_chunks.csv
GITA_EMBEDDINGS_NPY         # datasets/gita_embeddings.npy
GITA_METADATA_PKL           # datasets/gita_metadata.pkl
```

### 3. Embedding Model Configuration (3 variables)
Settings for generating embeddings.

```python
EMBEDDING_MODEL_NAME        # "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE        # 32 (chunks processed per batch)
EMBEDDING_DIMENSION         # 1024 (embedding vector size)
```

### 4. Qdrant Vector Store Configuration (4 variables)
Settings for the Qdrant vector database.

```python
QDRANT_COLLECTION_NAME      # "gita_verses"
QDRANT_UPLOAD_BATCH_SIZE    # 64 (vectors uploaded per batch)
QDRANT_DISTANCE_METRIC      # "COSINE" (similarity metric)
QDRANT_TIMEOUT              # 60 (seconds)
```

### 5. API Credentials (5 variables)
API keys and endpoints loaded from `.env` file.

```python
QDRANT_API_KEY              # From environment
QDRANT_ENDPOINT             # From environment
GEMINI_API_KEY              # From environment
GROQ_API_KEY                # From environment
SARVAM_API_KEY              # From environment
```

### 6. Data Schema (3 variables)
Required columns for data processing.

```python
REQUIRED_SOURCE_COLUMNS     # List of columns in source CSV
REQUIRED_EMBEDDING_COLUMNS  # List of columns for embeddings
METADATA_COLUMNS            # List of metadata columns for Qdrant
```

### 7. Other Settings (7 variables)
Miscellaneous configuration.

```python
CSV_ENCODING                # "utf-8-sig" (UTF-8 with BOM)
RETRIEVAL_TEXT_TEMPLATE     # Template for retrieval text
API_HOST                    # "0.0.0.0"
API_PORT                    # 8000
API_DEBUG                   # True
API_RELOAD                  # True
LOG_LEVEL                   # "INFO"
```

---

## How to Use

### Import Configuration Values
```python
from config import (
    CLEAN_GITA_CSV,
    GITA_CHUNKS_CSV,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)

# Use in your code
df = pd.read_csv(CLEAN_GITA_CSV)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
```

### Files Currently Using Config
1. **create_chunks.py**
   - Imports: `CLEAN_GITA_CSV`, `GITA_CHUNKS_CSV`, `REQUIRED_SOURCE_COLUMNS`, `CSV_ENCODING`
   - Processes: Clean dataset → Chunks CSV

2. **generate_embeddings.py**
   - Imports: `GITA_CHUNKS_CSV`, `GITA_EMBEDDINGS_NPY`, `GITA_METADATA_PKL`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_BATCH_SIZE`
   - Generates: Embeddings for all chunks

3. **upload_to_qdrant.py**
   - Imports: `GITA_EMBEDDINGS_NPY`, `GITA_METADATA_PKL`, `QDRANT_COLLECTION_NAME`, `QDRANT_UPLOAD_BATCH_SIZE`, `QDRANT_API_KEY`, `QDRANT_ENDPOINT`
   - Uploads: Embeddings to Qdrant

---

## Making Changes

### Change a File Path
Edit in `config.py`:
```python
# Before
GITA_CHUNKS_CSV = DATASETS_DIR / "gita_chunks.csv"

# After (example)
GITA_CHUNKS_CSV = DATASETS_DIR / "v2_gita_chunks.csv"
```
All modules automatically use the new path.

### Change the Embedding Model
```python
# Before
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# After (example)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
```

### Change Batch Sizes
```python
EMBEDDING_BATCH_SIZE = 64  # Changed from 32
QDRANT_UPLOAD_BATCH_SIZE = 128  # Changed from 64
```

### Change Qdrant Settings
```python
QDRANT_COLLECTION_NAME = "gita_verses_v2"
QDRANT_UPLOAD_BATCH_SIZE = 128
```

---

## Adding New Configurations

### Step 1: Add to config.py
```python
# In the appropriate section of config.py
MAX_WORKERS = 4
CACHE_TTL_SECONDS = 3600
```

### Step 2: Import in Your Module
```python
from config import MAX_WORKERS, CACHE_TTL_SECONDS

# Use it
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Process with max workers
    pass
```

### Step 3: Document
Add a comment in config.py explaining what it does.

---

## Before and After: Code Comparison

### Before (Hardcoded - ❌ Not Recommended)

**File: create_chunks.py**
```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
INPUT_PATH = DATASET_DIR / "clean_gita.csv"          # ❌ Hardcoded
OUTPUT_PATH = DATASET_DIR / "gita_chunks.csv"        # ❌ Hardcoded

REQUIRED_COLUMNS = ["ID", "Chapter", "Verse", ...]   # ❌ Hardcoded
```

**File: generate_embeddings.py**
```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
INPUT_PATH = DATASET_DIR / "gita_chunks.csv"         # ❌ Hardcoded (duplicate)

MODEL_NAME = "BAAI/bge-large-en-v1.5"                # ❌ Hardcoded
BATCH_SIZE = 32                                       # ❌ Hardcoded
```

**File: upload_to_qdrant.py**
```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
EMBEDDINGS_PATH = DATASET_DIR / "gita_embeddings.npy" # ❌ Hardcoded (duplicate)

COLLECTION_NAME = "gita_verses"                       # ❌ Hardcoded
UPLOAD_BATCH_SIZE = 64                                # ❌ Hardcoded
```

**Issues**: Hardcoded values repeated in 3 files, no single place to change settings, high risk of inconsistency.

### After (Centralized - ✅ Best Practice)

**File: config.py (Central Hub)**
```python
# ✅ All paths in ONE place
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ✅ All data files in ONE place
CLEAN_GITA_CSV = DATASETS_DIR / "clean_gita.csv"
GITA_CHUNKS_CSV = DATASETS_DIR / "gita_chunks.csv"
GITA_EMBEDDINGS_NPY = DATASETS_DIR / "gita_embeddings.npy"

# ✅ All model settings in ONE place
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE = 32

# ✅ All Qdrant settings in ONE place
QDRANT_COLLECTION_NAME = "gita_verses"
QDRANT_UPLOAD_BATCH_SIZE = 64

# ✅ All required columns in ONE place
REQUIRED_SOURCE_COLUMNS = ["ID", "Chapter", "Verse", ...]
```

**File: create_chunks.py (Clean and Simple)**
```python
from config import CLEAN_GITA_CSV, GITA_CHUNKS_CSV, REQUIRED_SOURCE_COLUMNS

df = pd.read_csv(CLEAN_GITA_CSV)  # ✅ Uses config
# ... process ...
df.to_csv(GITA_CHUNKS_CSV)        # ✅ Uses config
```

**File: generate_embeddings.py (Clean and Simple)**
```python
from config import GITA_CHUNKS_CSV, EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

df = load_chunks(GITA_CHUNKS_CSV)  # ✅ Uses config
model = SentenceTransformer(EMBEDDING_MODEL_NAME)  # ✅ Uses config
embeddings = generate_embeddings(texts, model, EMBEDDING_BATCH_SIZE)  # ✅ Uses config
```

**File: upload_to_qdrant.py (Clean and Simple)**
```python
from config import GITA_EMBEDDINGS_NPY, QDRANT_COLLECTION_NAME, QDRANT_UPLOAD_BATCH_SIZE

embeddings = load_embeddings(GITA_EMBEDDINGS_NPY)  # ✅ Uses config
client.recreate_collection(QDRANT_COLLECTION_NAME, ...)  # ✅ Uses config
```

**Benefits**: No duplication, single change point, consistent across all modules, cleaner code.

---

## Real-World Examples

### Example 1: Change Embedding Model
**Task**: Switch from large to small embedding model for faster processing

**Old Way (❌ Edit 3 files)**:
```
1. Open generate_embeddings.py
2. Find: MODEL_NAME = "BAAI/bge-large-en-v1.5"
3. Change: MODEL_NAME = "BAAI/bge-small-en-v1.5"
4. Risk: Might miss other files using this value
```

**New Way (✅ Edit 1 file)**:
```python
# config.py
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Change once
# Save - all 3 modules automatically use new model!
```

### Example 2: Switch to Different Dataset
**Task**: Process a new Gita dataset

**Old Way (❌ Edit multiple files)**:
```
1. Edit create_chunks.py - change INPUT_PATH
2. Edit other files that reference the path
3. Risk of inconsistency
```

**New Way (✅ Edit 1 file)**:
```python
# config.py
CLEAN_GITA_CSV = DATASETS_DIR / "new_dataset.csv"  # Change once
# All modules automatically use new dataset!
```

### Example 3: Optimize Batch Sizes
**Task**: Increase batch sizes for better performance

**Old Way (❌ Edit multiple files)**:
```
1. Edit generate_embeddings.py - change BATCH_SIZE
2. Edit upload_to_qdrant.py - change UPLOAD_BATCH_SIZE
3. Need to remember both locations
```

**New Way (✅ Edit 1 file)**:
```python
# config.py
EMBEDDING_BATCH_SIZE = 64          # Changed from 32
QDRANT_UPLOAD_BATCH_SIZE = 128     # Changed from 64
# All modules automatically use new batch sizes!
```

---

## Best Practices

### ✅ Do's
1. **Always import from config** - Never hardcode paths or values
2. **Keep related configs together** - Group similar settings in config.py
3. **Use descriptive names** - Make it clear what each config does
4. **Document changes** - Update this file when adding new configs
5. **Use environment variables for secrets** - API keys go in .env
6. **Validate on startup** - Use `validate_configuration()` in main modules
7. **Keep config.py at root** - Makes imports simpler from any subdirectory

### ❌ Don'ts
1. **Don't hardcode paths** - Always use config values
2. **Don't duplicate settings** - Define once, use everywhere
3. **Don't put secrets in config.py** - Use .env for API keys
4. **Don't scatter configuration** - Keep all config in one file
5. **Don't forget to import** - Add necessary imports at top of files

---

## Utility Functions

### validate_configuration()
Validates that critical configuration values are set:
```python
from config import validate_configuration

validate_configuration()  # Raises error if QDRANT_API_KEY or QDRANT_ENDPOINT not set
```

### get_config_summary()
Returns a summary of active configuration:
```python
from config import get_config_summary

config = get_config_summary()
print(config)  # Shows JSON with all key settings
```

---

## Troubleshooting

### Q: Import error when running scripts?
**A**: Add this to the top of your script:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import YOUR_VARIABLE
```

### Q: Want to verify config is working?
**A**: Run this from project root:
```bash
python -c "import config; print(config.get_config_summary())"
```

### Q: How do I add API keys?
**A**: 
1. Add to `.env` file in project root:
```
QDRANT_API_KEY=your_api_key_here
QDRANT_ENDPOINT=https://your-endpoint.us-east-2-0.aws.cloud.qdrant.io
GEMINI_API_KEY=your_gemini_key_here
```
2. config.py automatically loads them with `os.getenv()`

### Q: Can I have different configs for dev/prod?
**A**: Yes! Add conditional logic in config.py:
```python
import os
ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    EMBEDDING_BATCH_SIZE = 128
else:
    EMBEDDING_BATCH_SIZE = 32
```

### Q: ModuleNotFoundError: No module named 'config'?
**A**: Ensure you have the path setup:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import VARIABLE_NAME
```

### Q: What if config.py doesn't exist?
**A**: It should be in the project root: `c:\Users\ritam\Desktop\GitaWise\config.py`
If missing, check the SETUP_COMPLETE.md or recreate it with the documented structure.

---

## File Structure

```
GitaWise/
├── 🔧 config.py                    ← Main configuration file
├── 📚 CONFIGURATION.md             ← This guide (replaces 5 files)
│
├── backend/
│   ├── main.py
│   └── gita_vector_store/
│       ├── create_chunks.py        ✅ Uses config
│       ├── generate_embeddings.py  ✅ Uses config
│       └── upload_to_qdrant.py     ✅ Uses config
│
├── frontend/                       (No changes needed)
├── datasets/                       (Data files referenced by config)
├── notebooks/                      (No changes needed)
├── scripts/                        (No changes needed)
│
├── requirements.txt
├── .env                            (API credentials)
└── README.md
```

---

## Environment Variables

Ensure your `.env` file contains:
```
QDRANT_API_KEY=your_api_key_here
QDRANT_ENDPOINT=https://your-endpoint.us-east-2-0.aws.cloud.qdrant.io
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
SARVAM_API_KEY=your_sarvam_key_here
```

These are loaded automatically by config.py using `load_dotenv()`.

---

## Key Takeaways

1. ✅ All configuration is in `config.py` (one file)
2. ✅ Change any setting in `config.py` only
3. ✅ All modules automatically use updated values
4. ✅ No need to edit individual files for configuration changes
5. ✅ Professional, scalable, maintainable approach
6. ✅ Easy to collaborate with team on same settings
7. ✅ Simple to add new configurations as project grows

---

## Next Steps

1. **Review** the configuration file: `config.py`
2. **Understand** the sections: Read through Configuration Categories above
3. **Customize** as needed: Edit `config.py` values
4. **Test** by running scripts: Use updated modules
5. **Extend** when needed: Add new configurations following the pattern

---

## Quick Reference Table

| Task | How To |
|------|--------|
| View all config | `python config.py` |
| Change a setting | Edit `config.py` and save |
| Add new setting | Add to `config.py`, import in module |
| Validate config | `validate_configuration()` |
| Get config summary | `get_config_summary()` |
| Import in module | `from config import VARIABLE_NAME` |
| Use in code | `df = pd.read_csv(CLEAN_GITA_CSV)` |
| Check API keys | Look in `.env` file |

---

## Summary

Your GitaWise project now has a **professional, enterprise-grade configuration management system**. 

**What you get:**
- ✅ Single source of truth for all configuration
- ✅ No code duplication or scattered hardcoded values
- ✅ Easy to make changes across entire project
- ✅ Consistent configuration across all modules
- ✅ Professional, industry-standard approach
- ✅ Scalable for future growth
- ✅ Better code organization and maintainability

**Start using it:**
1. Open `config.py` and review available settings
2. Import config values in your modules
3. Run any script to verify it uses the configuration
4. Make changes by editing `config.py` only

---


