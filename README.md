# Pyics - Data-Oriented Calendar Automation System

[![Development Status](https://img.shields.io/badge/status-under%20development-orange)](https://github.com/obinexus/pyics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-modular%20versioned-green.svg)](https://github.com/obinexus/pyics/tree/dev/docs)
[![OBINexus](https://img.shields.io/badge/maintainer-OBINexus-red.svg)](https://github.com/obinexus)

**Maintainer**: [OBINexus Computing](https://github.com/obinexus)  
**Status**: Under Active Development 🚧  
**Branch**: `dev`  
**License**: MIT  

## 🚀 Overview

**Pyics** is a modular Python package designed for **automated calendar generation**, milestone tracking, and secure distribution of `.ics` files across distributed systems. Built with a **data-oriented programming (DOP)** architecture, Pyics decouples logic and state, enabling pure transformations, policy-driven hooks, and scalable integration into enterprise-grade workflows.

### Core Philosophy

Pyics implements **data-oriented programming** principles through **lambda calculus function composition**, creating a system where:
- **Pure functions** handle calendar transformations without side effects
- **Immutable state** flows through composable pipelines
- **Policy hooks** provide pre/post processing with zero overhead
- **Version isolation** ensures backward compatibility and future extensibility

## 🔧 Problem Statement

Current `.ics` solutions suffer from:
- Rigid, monolithic architectures tightly coupled to specific UIs
- Lack of scalable event handling for enterprise business operations
- Insufficient automation for compliance and milestone tracking
- Poor integration capabilities for distributed systems
- Absence of policy-driven processing pipelines

**Pyics solves this by offering:**
- ✅ Policy-wrapped calendar logic with hooks and escalation chains
- ✅ Version-isolated architecture with CLI discovery mechanisms
- ✅ Authenticated .ics sharing with email domain validation and API tokens
- ✅ Developer-ready CLI and REPL tooling for rapid iteration
- ✅ Integration-ready API contracts for internal and external systems

## 🧱 Architecture Summary

### Version Isolation Strategy
```
pyics/
├── v1/           # Stable implementation (LTS)
├── v2/           # Enhanced features with telemetry
├── v3-preview/   # Experimental AI/blockchain features
├── audit/        # Business logic (version-agnostic)
├── calendar/     # Core calendar factories
├── notify/       # Notification engines
└── cli/          # Unified command interface
```

### Key Architectural Components

- **📁 Version Isolation**: `v1/`, `v2/`, and `v3-preview/` directories for seamless upgrades
- **🧩 Modular CLI System**: `cli/**/main.py` mapped to domain logic in `*/py/`
- **🔁 Hookable Pipeline**: Pure function pre/post logic with policy enforcement
- **🔐 Authentication Interface**: Token-based and custom auth provider support
- **📊 Audit & Telemetry**: Configurable backends with compliance reporting
- **🧪 Comprehensive Testing**: Unit, integration, E2E, and migration test suites

### Data-Oriented Programming Implementation

```python
# Function composition example
from pyics.v2.core.py import compose, transform, pipeline

# Pure transformation pipeline
calendar_pipeline = compose(
    transform.validate_events,
    transform.apply_timezone,
    transform.generate_ics,
    pipeline.apply_hooks
)

# Immutable state flow
result = calendar_pipeline(input_data)
```

## 🧑‍💻 Use Cases

### Primary Applications
- **Civil/Legal Milestone Tracking**: Automated compliance deadline management
- **Enterprise Operations**: Team-wide milestone and deadline automation
- **Research Workflows**: Academic project timeline calendarization
- **Business-Critical Distribution**: Policy-compliant `.ics` sharing across networks
- **Developer Tooling**: CLI-driven calendar creation with REPL control

### Integration Scenarios
- **Email Networks**: Automated distribution to business stakeholders
- **Enterprise Systems**: API integration with existing workflow platforms
- **Distributed Teams**: Cross-node calendar synchronization
- **Compliance Systems**: Audit trail generation for regulatory requirements

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or poetry for dependency management
- Git for version control

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/obinexus/pyics.git
cd pyics

# Install with basic dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install enterprise features
pip install -e ".[enterprise,distribute]"
```

### Initialize System

```bash
# Initialize CLI registry and IoC container
pyics setup

# Verify installation
pyics version

# Start interactive REPL
pyics-repl

# Generate sample calendar
pyics generate --template milestone --output sample.ics
```

### Basic Usage Example

```python
from pyics.v2.core.py import CalendarEngine
from pyics.v2.distribution.py import SMTPDistributor

# Initialize calendar engine
engine = CalendarEngine(version="v2", config="production")

# Create milestone events
events = engine.create_milestone_series(
    start_date="2024-12-30",
    interval_days=14,
    penalties=[
        "Initial Violation (£1M)",
        "Continued Breach (£1M)",
        "Systemic Neglect (£1M)"
    ]
)

# Generate ICS file
ics_content = engine.generate_ics(events)

# Distribute via authenticated email
distributor = SMTPDistributor(auth_config="smtp_config.json")
distributor.send_calendar(
    ics_content=ics_content,
    recipients=["legal@company.com", "compliance@company.com"],
    subject="Civil Compliance Milestone Tracker"
)
```

## 📁 Project Structure

```
pyics/
├── v1/                     # Stable v1 implementation
│   ├── core/              # Calendar engine, event builders
│   ├── hooks/             # Pre/post processing hooks
│   ├── distribution/      # SMTP, file distribution
│   ├── auth/              # Basic authentication
│   └── cli/               # V1 CLI commands
├── v2/                     # Enhanced v2 features
│   ├── core/              # Enhanced engines with workers
│   ├── hooks/             # Priority and conditional hooks
│   ├── distribution/      # Multi-channel distribution
│   ├── auth/              # Enterprise authentication
│   ├── telemetry/         # Metrics and audit trails
│   └── cli/               # Advanced CLI interface
├── v3-preview/            # Experimental features
│   ├── experimental/      # AI integration, blockchain
│   └── migration/         # Version migration tools
├── audit/                 # Business audit logic
├── calendar/              # Core calendar factories
├── notify/                # Notification engines
├── tests/                 # Comprehensive test suites
├── docs/                  # Version-specific documentation
├── config/                # Environment configurations
└── scripts/               # Development and deployment tools
```

## 🔧 Development

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install

# Run test suite
pytest tests/

# Run integration tests
pytest tests/integration/

# Generate documentation
cd docs && make html
```

### CLI Command Structure

Pyics uses an IoC (Inversion of Control) container for command registration:

```bash
# Core commands
pyics generate      # Generate calendar files
pyics audit         # Audit existing calendars
pyics distribute    # Distribute calendar files
pyics notify        # Send notifications
pyics verify        # Verify calendar integrity

# Version-specific commands
pyics-v1 <command>   # Use v1 implementation
pyics-v2 <command>   # Use v2 implementation

# Interactive mode
pyics-repl          # Start REPL environment
```

### Testing Strategy

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Version compatibility tests
pytest tests/migration/

# Performance benchmarks
python scripts/testing/benchmark.py
```

## 🔜 Roadmap

### Version 2.1 (Current)
- ✅ Enhanced CLI with IoC container
- ✅ Multi-channel distribution system
- ✅ Enterprise authentication integration
- 🚧 Telemetry and audit trail implementation

### Version 2.2 (Q2 2025)
- 📋 WebSocket real-time synchronization
- 📋 Advanced policy engine with custom rules
- 📋 REST API for external integrations
- 📋 Performance optimization for large datasets

### Version 3.0 Preview (Q3 2025)
- 🔬 AI-driven smart scheduling
- 🔬 Blockchain-based audit trails
- 🔬 Machine learning prediction models
- 🔬 Advanced migration tooling

## 🤝 Contributing

We welcome contributions from the developer community. Please follow our contribution guidelines:

### Contribution Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Follow** the modular structure: add new modules under `*/py/` folders
4. **Expose** functionality in `cli/**/main.py` 
5. **Register** commands with `cli/ioc.py`
6. **Add** comprehensive tests
7. **Update** documentation
8. **Submit** a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Maintain 90%+ test coverage
- Use type hints throughout
- Document public APIs with docstrings
- Follow data-oriented programming principles

### Development Guidelines
- Use pure functions where possible
- Implement proper error handling
- Add telemetry points for monitoring
- Ensure version compatibility
- Update migration scripts when needed

## 📄 License

```
MIT License

Copyright (c) 2024 OBINexus Computing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🔗 Links

- **Documentation**: [https://pyics.readthedocs.io](https://pyics.readthedocs.io)
- **API Reference**: [https://pyics.readthedocs.io/en/latest/api/](https://pyics.readthedocs.io/en/latest/api/)
- **Issue Tracker**: [https://github.com/obinexus/pyics/issues](https://github.com/obinexus/pyics/issues)
- **Discussions**: [https://github.com/obinexus/pyics/discussions](https://github.com/obinexus/pyics/discussions)
- **Changelog**: [https://github.com/obinexus/pyics/blob/dev/CHANGELOG.md](https://github.com/obinexus/pyics/blob/dev/CHANGELOG.md)

---

**Built with ❤️ by the OBINexus Engineering Team**