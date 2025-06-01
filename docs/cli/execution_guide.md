# Pyics Single-Pass Architecture Implementation Guide

**Engineering Lead**: Nnamdi Okpala / OBINexus Computing  
**Phase**: 3.1.6.3 - Single-Pass Modular Architecture Implementation  
**Objective**: Correct existing violations and implement clean single-pass loading

## 🎯 **Current Issues Identified**

Based on your structure analysis, the following violations need correction:

1. **Complex Nested Structures**: `implementations/`, `interfaces/`, `compliance/`, `contracts/`, `tests/`
2. **Missing Core Domain**: No `structures/` domain (foundational requirement)
3. **Scattered Files**: Multiple Python files outside standard pattern
4. **Potential Circular Dependencies**: Complex structure enables dependency loops

## 🚀 **Step-by-Step Execution Plan**

### **Phase 1: Prepare Environment**

```bash
# 1. Navigate to your project root
cd ~/projects/pkg/pyics

# 2. Create scripts directory if needed
mkdir -p scripts/development

# 3. Save the correction scripts
# Copy the two artifacts I created:
# - pyics_modular_architecture_validator.py → scripts/development/
# - pyics_structure_corrector.py → scripts/development/
```

### **Phase 2: Create CLI Structure**

```bash
# 1. Create CLI directory
mkdir -p pyics/cli

# 2. Create CLI __init__.py
cat > pyics/cli/__init__.py << 'EOF'
"""Pyics CLI Module"""
__version__ = "1.0.0"
EOF

# 3. Create main CLI entry point
# Copy the pyics_cli_main.py artifact → pyics/cli/main.py
```

### **Phase 3: Execute Structure Correction**

```bash
# 1. Run the structure corrector (this will fix violations)
python scripts/development/pyics_structure_corrector.py

# Expected output:
# ✅ Structure correction COMPLETE: Single-pass architecture implemented
# 🧹 Cleaned up X complex directory structures
# 📁 Consolidated Y scattered files
# 🏗️ Generated Z missing modules
```

### **Phase 4: Validate Architecture**

```bash
# 1. Test CLI functionality
python -m pyics.cli.main info

# 2. Validate domain architecture
python -m pyics.cli.main domain validate

# 3. Check load order
python -m pyics.cli.main domain load-order

# 4. Run comprehensive validation
python -m pyics.cli.main validate-architecture
```

### **Phase 5: Test Domain Imports**

```python
# Test single-pass loading in Python
python3 -c "
# Test foundational domains (load_order: 10, 20)
from pyics.core.primitives import get_domain_metadata, create_entity
from pyics.core.protocols import DomainInterface, ValidationProtocol

# Test dependent domain (load_order: 30)
from pyics.core.structures import EventStructure, create_event_structure

print('✅ All domains loaded successfully in single-pass order')
print('🎯 Architecture compliance verified')
"
```

## 📁 **Expected Final Structure**

After correction, your structure should look like this:

```
pyics/
├── core/
│   ├── primitives/
│   │   ├── data_types.py      # Atomic data containers
│   │   ├── operations.py      # Pure atomic functions
│   │   ├── relations.py       # Minimal atomic relations
│   │   ├── config.py          # Domain configuration
│   │   ├── __init__.py        # Public interface
│   │   └── README.md          # Documentation
│   ├── protocols/
│   │   ├── data_types.py      # Interface definitions
│   │   ├── operations.py      # Protocol operations
│   │   ├── relations.py       # Interface relations
│   │   ├── config.py          # Domain configuration
│   │   ├── __init__.py        # Public interface
│   │   └── README.md          # Documentation
│   ├── structures/
│   │   ├── data_types.py      # Immutable calendar structures
│   │   ├── operations.py      # Structure manipulation
│   │   ├── relations.py       # Temporal relationships
│   │   ├── config.py          # Domain configuration
│   │   ├── __init__.py        # Public interface
│   │   └── README.md          # Documentation
│   └── ioc_registry.py        # Single-pass IoC registry
├── cli/
│   ├── __init__.py            # CLI module init
│   └── main.py                # Main CLI entry point
└── __init__.py                # Package init
```

## 🔍 **Validation Commands**

### **Domain Status Checks**

```bash
# Show all domains
pyics domain status

# Show specific domain
pyics domain status primitives
pyics domain status protocols  
pyics domain status structures

# Show detailed metadata
pyics domain metadata primitives --format json
```

### **Architecture Validation**

```bash
# Validate single-pass compliance
pyics domain validate

# Show load order and dependencies
pyics domain load-order

# Comprehensive architecture check
pyics validate-architecture
```

### **Domain-Specific Operations**

```bash
# Test primitives
pyics primitives status

# Test protocols  
pyics protocols status

# Test structures
pyics structures status

# Create a test event
pyics structures create-event \
  --event-id "test_001" \
  --title "Team Meeting" \
  --start "2024-01-15 09:00" \
  --end "2024-01-15 10:00"
```

## 🛠️ **Troubleshooting Guide**

### **Issue: ImportError during validation**

```bash
# Solution: Run structure correction
pyics fix-structure

# Then retry validation
pyics domain validate
```

### **Issue: Circular dependency detected**

```bash
# Check load order
pyics domain load-order

# Verify dependencies
pyics domain metadata [domain_name]

# Look for violations in output
```

### **Issue: Complex nested directories remain**

```bash
# Manually remove if corrector didn't catch them
find pyics/core -name "implementations" -type d -exec rm -rf {} +
find pyics/core -name "interfaces" -type d -exec rm -rf {} +
find pyics/core -name "compliance" -type d -exec rm -rf {} +
find pyics/core -name "contracts" -type d -exec rm -rf {} +
find pyics/core -name "tests" -type d -exec rm -rf {} +

# Then re-run correction
pyics fix-structure
```

## 🎯 **Success Criteria**

Your implementation is successful when:

1. **✅ All domains load in correct order**: `primitives(10) → protocols(20) → structures(30)`
2. **✅ No circular dependencies**: Each domain only depends on lower load_order domains
3. **✅ Standard module pattern**: Each domain has the 6 required modules
4. **✅ CLI functionality**: All commands work without import errors
5. **✅ Architecture validation**: `pyics validate-architecture` passes

## 📊 **Cost Metadata Compliance**

| Domain | Load Order | Priority | Dependencies | Thread Safe | Compute Weight |
|--------|------------|----------|--------------|-------------|----------------|
| primitives | 10 | 1 | None | ✅ | 0.1 |
| protocols | 20 | 1 | None | ✅ | 0.05 |
| structures | 30 | 2 | primitives, protocols | ✅ | 0.2 |

## 🔗 **Integration Examples**

### **Basic Domain Usage**

```python
# Single-pass loading example
from pyics.core.primitives import create_entity, AtomicValue
from pyics.core.protocols import ValidationProtocol  
from pyics.core.structures import create_event_structure, EventStructure

# Create atomic value
atomic_val = AtomicValue(value="test", value_type="string")

# Create domain entity
entity = create_entity("test_001", "Test Entity")

# Create calendar event
from datetime import datetime
event = create_event_structure(
    event_id="meeting_001",
    title="Daily Standup", 
    start_time=datetime(2024, 1, 15, 9, 0),
    end_time=datetime(2024, 1, 15, 9, 30)
)

print("✅ Single-pass architecture working correctly!")
```

### **CLI Integration**

```bash
# Add to your PATH or create alias
alias pyics='python -m pyics.cli.main'

# Now use anywhere
pyics info
pyics domain status
pyics validate-architecture
```

## 🚨 **Emergency Recovery**

If something goes wrong during correction:

```bash
# 1. Restore from backup (automatically created)
ls structure_backup/  # Find latest backup
cp -r structure_backup/structure_backup_YYYYMMDD_HHMMSS/* pyics/core/

# 2. Start over with manual cleanup
rm -rf pyics/core/*/implementations
rm -rf pyics/core/*/interfaces  
rm -rf pyics/core/*/compliance
rm -rf pyics/core/*/contracts
rm -rf pyics/core/*/tests

# 3. Re-run correction
python scripts/development/pyics_structure_corrector.py
```

## 🎉 **Next Steps After Success**

1. **Set up CI/CD validation**: Add architecture checks to your pipeline
2. **Implement domain-specific business logic**: Build on the clean foundation
3. **Add more domains**: Follow the same pattern for additional domains
4. **Performance optimization**: Use cost metadata for optimization decisions
5. **Documentation**: Enhance domain README files with usage examples

---

**🎯 Ready to execute? Start with Phase 1 and work through each step systematically.**

**💡 Need help? Run `pyics info` for quick reference or `pyics --help` for full command list.**