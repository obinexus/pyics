# Pyics Master Execution Plan
**Engineering Lead**: Nnamdi Okpala / OBINexus Computing  
**Phase**: Immediate Core Cleanup Actions  
**Objective**: Execute systematic cleanup and establish single-pass architecture

## 🎯 **Execution Overview**

Based on your comprehensive analysis and the provided scripts, here's the systematic execution plan to resolve the immediate domain violations and establish clean single-pass loading.

## 📋 **Step-by-Step Execution**

### **Step 1: Setup Environment & Backup**

```bash
# Navigate to project root
cd ~/projects/pkg/pyics

# Create scripts directory if needed
mkdir -p scripts/development

# Ensure all scripts are executable
chmod +x scripts/development/*.py
```

### **Step 2: Execute Setup.py Relocation**

Use your provided `Setup.py Relocation and Package Structure Corrector`:

```bash
# Run the setup.py corrector first
python scripts/development/relocate_setup_structure.py

# Expected output:
# ✅ Setup.py structure corrected successfully
# 📁 Setup.py moved to project root
# ⚙️ Package configuration updated
# 🔧 MANIFEST.in and pyproject.toml created
```

**What this does:**
- Moves `pyics/setup.py` → `setup.py` (project root)
- Updates package discovery for proper `find_packages()` 
- Creates modern `pyproject.toml` configuration
- Generates `MANIFEST.in` for package data inclusion

### **Step 3: Execute Immediate Core Cleanup**

Run the immediate cleanup executor I just created:

```bash
# Execute immediate domain cleanup
python pyics_immediate_cleanup_executor.py

# Expected output:
# ✅ Backup Created
# ✅ validation → validators (merged)
# ✅ transforms → transformations/legacy_transforms/ (eliminated)
# ✅ logic → eliminated (functions preserved)
# ✅ Complex structures cleaned
# ✅ Single-pass load order implemented
```

**What this does:**
- **Merges** `core/validation` → `core/validators`
- **Eliminates** `core/transforms` → `core/transformations/legacy_transforms/`
- **Eliminates** `core/logic` (preserves functions for manual distribution)
- **Cleans** complex nested structures (`implementations/`, `interfaces/`, `compliance/`, `contracts/`, `tests/`)
- **Implements** single-pass load order configuration

### **Step 4: Execute Structure Corrector**

Use your provided `Pyics Structure Corrector - Single-Pass Architecture Implementation`:

```bash
# Run the comprehensive structure corrector
python scripts/development/pyics_structure_corrector.py

# Expected output:
# ✅ Structure correction COMPLETE
# 🧹 Complex directory structures cleaned
# 📁 Standard domain modules generated
# 🏗️ Single-pass IoC registry created
# ⚡ Cost-aware domain loading implemented
```

**What this does:**
- Generates missing standard modules (`data_types.py`, `operations.py`, `relations.py`, `config.py`, `__init__.py`, `README.md`)
- Creates single-pass IoC registry with cost metadata
- Validates domain architecture compliance
- Establishes proper load order: `primitives(10) → protocols(20) → structures(30) → composition(40) → validators(50) → transformations(60) → registry(70) → routing(80) → safety(90)`

### **Step 5: Generate Testing Framework** *(Optional but Recommended)*

Use your provided `Pyics Testing Framework Generator`:

```bash
# Generate comprehensive testing structure
python scripts/development/generate_pyics_testing_framework.py

# Expected output:
# 🎉 TESTING FRAMEWORK GENERATED SUCCESSFULLY!
# 📁 Unit, Integration, and E2E test templates created
# 🔒 Zero-trust validation utilities included
# ⚡ Performance assertion framework ready
```

**What this does:**
- Creates DOP-compliant testing structure
- Generates unit, integration, and E2E test templates
- Provides zero-trust validation utilities
- Establishes performance assertion framework

### **Step 6: Validation & Testing**

```bash
# Test single-pass loading
python -c "
from pyics.core.ioc_registry import get_registry, validate_architecture
registry = get_registry()
print('✅ Registry initialized successfully')
print('Domains loaded:', registry.get_load_order())
print('Architecture valid:', validate_architecture())
"

# Test domain imports
python -c "
from pyics.core.primitives import get_domain_metadata
from pyics.core.protocols import get_domain_metadata as protocols_meta
from pyics.core.structures import get_domain_metadata as structures_meta
print('✅ All domains loaded successfully in single-pass order')
"

# Test package installation
pip install -e .

# Test CLI functionality (after CLI implementation)
python -m pyics.cli.main info
python -m pyics.cli.main domain status
python -m pyics.cli.main validate-architecture
```

## 🎯 **Expected Final Structure**

After execution, your structure should be:

```
pyics/
├── setup.py                   # ← Moved from pyics/setup.py
├── pyproject.toml            # ← Generated
├── MANIFEST.in               # ← Generated
├── pyics/
│   ├── core/
│   │   ├── primitives/       # Clean standard modules
│   │   │   ├── data_types.py
│   │   │   ├── operations.py  
│   │   │   ├── relations.py
│   │   │   ├── config.py
│   │   │   ├── __init__.py
│   │   │   └── README.md
│   │   ├── protocols/        # Clean standard modules
│   │   ├── structures/       # Clean standard modules
│   │   ├── composition/      # Clean standard modules
│   │   ├── validators/       # ← Consolidated (validation merged)
│   │   ├── transformations/  # ← Consolidated (transforms eliminated)
│   │   │   └── legacy_transforms/  # ← Legacy content
│   │   ├── registry/         # Clean standard modules
│   │   ├── routing/          # Clean standard modules
│   │   ├── safety/           # Clean standard modules
│   │   ├── ioc_registry.py   # ← Single-pass IoC registry
│   │   ├── _load_order_config.json
│   │   ├── _preserved_logic/ # ← Preserved for manual review
│   │   └── _preserved_complex_structures/
│   ├── cli/                  # ← CLI module (if implementing)
│   │   ├── __init__.py
│   │   └── main.py
│   └── __init__.py
└── tests/                    # ← Generated testing framework
    ├── unit/
    ├── integration/
    ├── e2e/
    ├── fixtures/
    ├── utils/
    └── conftest.py
```

## 🚨 **Eliminated/Cleaned Items**

- ❌ `core/validation/` → Merged into `validators/`
- ❌ `core/transforms/` → Moved to `transformations/legacy_transforms/`
- ❌ `core/logic/` → Eliminated (functions preserved)
- ❌ All `implementations/` directories
- ❌ All `interfaces/` directories  
- ❌ All `compliance/` directories
- ❌ All `contracts/` directories
- ❌ All nested `tests/` directories

## 📊 **Cost Metadata Compliance**

| Domain | Load Order | Priority | Dependencies | Thread Safe | Status |
|--------|------------|----------|--------------|-------------|---------|
| primitives | 10 | 1 | None | ✅ | Active |
| protocols | 20 | 1 | None | ✅ | Active |
| structures | 30 | 2 | primitives, protocols | ✅ | Active |
| composition | 40 | 2 | primitives, protocols | ✅ | Active |
| validators | 50 | 3 | primitives, protocols, structures | ✅ | Consolidated |
| transformations | 60 | 3 | All above | ✅ | Consolidated |
| registry | 70 | 4 | All above | ✅ | Active |
| routing | 80 | 4 | All above | ✅ | Active |
| safety | 90 | 5 | All above | ✅ | Active |

## 🔍 **Manual Review Required**

After automated cleanup, manually review these preserved items:

1. **Logic Functions**: `pyics/core/_preserved_logic/`
   - Distribute useful functions to `primitives/` or `composition/`
   - Archive deprecated functions

2. **Complex Structure Content**: `pyics/core/_preserved_complex_structures/`
   - Review preserved implementation code
   - Integrate valuable code into standard modules

3. **Legacy Transforms**: `pyics/core/transformations/legacy_transforms/`
   - Modernize useful legacy transforms
   - Remove deprecated transformation logic

4. **Merge Records**: 
   - `pyics/core/validators/_merged_from_validation.md`
   - `pyics/core/transformations/_eliminated_transforms.md`

## 🚀 **Success Criteria**

Your cleanup is successful when:

1. ✅ **No domain redundancy**: `validation`, `transforms`, `logic` eliminated
2. ✅ **Clean structure**: No complex nested directories
3. ✅ **Single-pass loading**: Domains load in correct order without circular dependencies
4. ✅ **Standard modules**: Each domain has 6 standard modules
5. ✅ **Package compliance**: Modern `setup.py`, `pyproject.toml`, proper imports
6. ✅ **Testing ready**: Comprehensive testing framework available

## 🛠️ **Troubleshooting**

### **Issue: Import errors after cleanup**
```bash
# Solution: Update any existing imports
grep -r "from pyics.core.validation" . --include="*.py"
grep -r "from pyics.core.transforms" . --include="*.py"  
grep -r "from pyics.core.logic" . --include="*.py"
# Replace with new import paths
```

### **Issue: Missing functions after domain elimination**
```bash
# Check preserved locations
ls pyics/core/_preserved_logic/
ls pyics/core/_preserved_complex_structures/
# Manually integrate needed functions
```

### **Issue: Single-pass loading fails**
```bash
# Check load order
python -c "
from pyics.core.ioc_registry import get_registry
registry = get_registry()
print('Load order:', registry.get_load_order())
print('Performance:', registry.get_load_performance())
"
```

## 📝 **Post-Cleanup Tasks**

1. **Update Documentation**: Reflect new structure in README.md
2. **Update CI/CD**: Modify build scripts for new structure  
3. **Team Communication**: Inform team of import path changes
4. **Performance Testing**: Benchmark new single-pass loading
5. **Integration Testing**: Ensure downstream modules work with cleaned structure

---

**🎯 Ready to execute? Start with Step 1 and follow each step systematically.**

**💡 All scripts are provided and ready - this plan uses your existing comprehensive cleanup tools in the optimal order.**