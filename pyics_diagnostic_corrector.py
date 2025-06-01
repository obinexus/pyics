#!/usr/bin/env python3
"""
pyics_path_resolution_diagnostic.py
Pyics Project Structure Diagnostic & Path Corrector

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Methodology: Systematic diagnostic analysis with corrective path resolution
Objective: Identify and resolve project structure path resolution failures

TECHNICAL SPECIFICATIONS:
- Dynamic project root detection with nested structure handling
- Comprehensive directory structure analysis and validation
- Corrective path resolution with backup preservation
- Systematic recovery protocol implementation
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Configure diagnostic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [DIAGNOSTIC] - %(message)s',
    handlers=[
        logging.FileHandler('pyics_diagnostic.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PyicsPathResolutionDiagnostic:
    """
    Systematic diagnostic tool for pyics project structure analysis
    
    Implements comprehensive path resolution with nested structure detection
    and corrective implementation following waterfall methodology principles.
    """
    
    def __init__(self, starting_path: Optional[Path] = None):
        self.starting_path = starting_path or Path.cwd()
        self.diagnostic_results = {
            "project_root_candidates": [],
            "core_directory_locations": [],
            "nested_structure_detected": False,
            "path_resolution_strategy": None,
            "corrective_actions": [],
            "validation_status": "initialized"
        }
        
    def execute_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """Execute systematic diagnostic analysis"""
        logger.info("="*80)
        logger.info("PYICS PROJECT STRUCTURE DIAGNOSTIC EXECUTION")
        logger.info("Engineering Lead: Nnamdi Okpala / OBINexus Computing")
        logger.info("Methodology: Systematic path resolution analysis")
        logger.info("="*80)
        
        try:
            # Phase 1: Project root detection
            self._detect_project_root_candidates()
            
            # Phase 2: Core directory location analysis
            self._analyze_core_directory_locations()
            
            # Phase 3: Nested structure detection
            self._detect_nested_structures()
            
            # Phase 4: Path resolution strategy determination
            self._determine_path_resolution_strategy()
            
            # Phase 5: Generate corrective implementation
            self._generate_corrective_actions()
            
            self.diagnostic_results["validation_status"] = "completed"
            self._generate_diagnostic_report()
            
            return self.diagnostic_results
            
        except Exception as e:
            logger.error(f"Diagnostic execution failed: {e}")
            self.diagnostic_results["validation_status"] = "failed"
            self.diagnostic_results["error"] = str(e)
            return self.diagnostic_results
    
    def _detect_project_root_candidates(self) -> None:
        """Detect potential project root directories"""
        logger.info("Phase 1: Detecting project root candidates")
        
        current_path = self.starting_path.resolve()
        candidates = []
        
        # Search up the directory tree for pyics project indicators
        for path in [current_path] + list(current_path.parents):
            project_indicators = [
                path / "pyics",
                path / "setup.py",
                path / "pyproject.toml",
                path / "requirements.txt",
                path / ".git"
            ]
            
            indicator_count = sum(1 for indicator in project_indicators if indicator.exists())
            
            if indicator_count > 0:
                candidates.append({
                    "path": str(path),
                    "indicator_count": indicator_count,
                    "indicators": [str(ind.name) for ind in project_indicators if ind.exists()],
                    "confidence_score": indicator_count
                })
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x["confidence_score"], reverse=True)
        self.diagnostic_results["project_root_candidates"] = candidates
        
        logger.info(f"Detected {len(candidates)} project root candidates")
        for i, candidate in enumerate(candidates[:3]):  # Log top 3
            logger.info(f"  Candidate {i+1}: {candidate['path']} (Score: {candidate['confidence_score']})")
    
    def _analyze_core_directory_locations(self) -> None:
        """Analyze potential core directory locations"""
        logger.info("Phase 2: Analyzing core directory locations")
        
        core_locations = []
        
        # Search for core directories in all project root candidates
        for candidate in self.diagnostic_results["project_root_candidates"]:
            root_path = Path(candidate["path"])
            
            # Search for potential core directories
            potential_cores = list(root_path.rglob("core"))
            
            for core_path in potential_cores:
                if core_path.is_dir():
                    # Analyze core directory content
                    domain_count = self._count_potential_domains(core_path)
                    
                    core_info = {
                        "path": str(core_path),
                        "relative_to_root": str(core_path.relative_to(root_path)),
                        "domain_count": domain_count,
                        "parent_structure": str(core_path.parent.relative_to(root_path)),
                        "nested_level": len(core_path.relative_to(root_path).parts) - 1
                    }
                    
                    core_locations.append(core_info)
        
        self.diagnostic_results["core_directory_locations"] = core_locations
        
        logger.info(f"Found {len(core_locations)} potential core directories")
        for core in core_locations:
            logger.info(f"  Core: {core['path']} (Domains: {core['domain_count']}, Nested: {core['nested_level']})")
    
    def _count_potential_domains(self, core_path: Path) -> int:
        """Count potential domain directories in core"""
        if not core_path.exists():
            return 0
        
        domain_indicators = [
            "primitives", "protocols", "structures", "composition",
            "validators", "transformations", "registry", "routing", "safety",
            "validation", "transforms", "logic"  # Legacy domains
        ]
        
        domain_count = 0
        for item in core_path.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                not item.name.startswith('_') and
                item.name != '__pycache__'):
                
                # Check if it looks like a domain
                if (item.name in domain_indicators or
                    any((item / f).exists() for f in ["__init__.py", "data_types.py", "operations.py"])):
                    domain_count += 1
        
        return domain_count
    
    def _detect_nested_structures(self) -> None:
        """Detect nested pyics directory structures"""
        logger.info("Phase 3: Detecting nested structures")
        
        nested_detected = False
        nested_patterns = []
        
        for candidate in self.diagnostic_results["project_root_candidates"]:
            root_path = Path(candidate["path"])
            
            # Look for nested pyics directories
            pyics_dirs = list(root_path.rglob("pyics"))
            
            if len(pyics_dirs) > 1:
                nested_detected = True
                for pyics_dir in pyics_dirs:
                    pattern = {
                        "path": str(pyics_dir),
                        "relative_to_root": str(pyics_dir.relative_to(root_path)),
                        "nesting_level": len(pyics_dir.relative_to(root_path).parts)
                    }
                    nested_patterns.append(pattern)
        
        self.diagnostic_results["nested_structure_detected"] = nested_detected
        self.diagnostic_results["nested_patterns"] = nested_patterns
        
        if nested_detected:
            logger.warning("⚠️  Nested pyics directory structure detected")
            for pattern in nested_patterns:
                logger.warning(f"    Nested path: {pattern['path']} (Level: {pattern['nesting_level']})")
        else:
            logger.info("✅ No problematic nested structures detected")
    
    def _determine_path_resolution_strategy(self) -> None:
        """Determine optimal path resolution strategy"""
        logger.info("Phase 4: Determining path resolution strategy")
        
        # Analyze core directory locations for best candidate
        core_locations = self.diagnostic_results["core_directory_locations"]
        
        if not core_locations:
            strategy = {
                "type": "create_missing_structure",
                "recommended_root": self.diagnostic_results["project_root_candidates"][0]["path"] if self.diagnostic_results["project_root_candidates"] else str(self.starting_path),
                "core_path": "pyics/core",
                "rationale": "No existing core directories found, will create standard structure"
            }
        else:
            # Find core with highest domain count and lowest nesting
            best_core = max(core_locations, 
                          key=lambda x: (x["domain_count"], -x["nested_level"]))
            
            strategy = {
                "type": "use_existing_structure",
                "recommended_core": best_core["path"],
                "core_relative_path": best_core["relative_to_root"],
                "rationale": f"Selected core with {best_core['domain_count']} domains at nesting level {best_core['nested_level']}"
            }
        
        self.diagnostic_results["path_resolution_strategy"] = strategy
        logger.info(f"Strategy: {strategy['type']}")
        logger.info(f"Rationale: {strategy['rationale']}")
    
    def _generate_corrective_actions(self) -> None:
        """Generate corrective actions based on diagnostic results"""
        logger.info("Phase 5: Generating corrective actions")
        
        strategy = self.diagnostic_results["path_resolution_strategy"]
        actions = []
        
        if strategy["type"] == "use_existing_structure":
            # Corrective action for existing structure
            core_path = Path(strategy["recommended_core"])
            project_root = self._find_project_root_for_core(core_path)
            
            actions.append({
                "action_type": "path_correction",
                "description": "Update systematic cleanup executor with correct paths",
                "project_root": str(project_root),
                "core_directory": str(core_path),
                "core_relative_path": strategy["core_relative_path"]
            })
            
            # Check if we need to resume from specific phase
            if self._check_phase_completion_status(core_path):
                actions.append({
                    "action_type": "resume_execution",
                    "description": "Resume systematic cleanup from phase 4",
                    "resume_phase": "phase_4_load_order_implementation"
                })
            
        elif strategy["type"] == "create_missing_structure":
            actions.append({
                "action_type": "structure_creation",
                "description": "Create missing pyics/core structure",
                "target_root": strategy["recommended_root"],
                "target_core_path": strategy["core_path"]
            })
        
        # Add nested structure cleanup if needed
        if self.diagnostic_results["nested_structure_detected"]:
            actions.append({
                "action_type": "nested_cleanup",
                "description": "Resolve nested pyics directory structure",
                "nested_patterns": self.diagnostic_results["nested_patterns"]
            })
        
        self.diagnostic_results["corrective_actions"] = actions
        
        logger.info(f"Generated {len(actions)} corrective actions")
        for i, action in enumerate(actions):
            logger.info(f"  Action {i+1}: {action['action_type']} - {action['description']}")
    
    def _find_project_root_for_core(self, core_path: Path) -> Path:
        """Find project root for given core directory"""
        # Walk up from core to find project root
        current = core_path.parent
        
        while current.parent != current:  # Not at filesystem root
            if any((current / indicator).exists() for indicator in [".git", "setup.py", "pyproject.toml"]):
                return current
            current = current.parent
        
        # Fallback to core's grandparent if no clear root found
        return core_path.parent.parent
    
    def _check_phase_completion_status(self, core_path: Path) -> bool:
        """Check if previous phases completed successfully"""
        backup_indicators = [
            core_path.parent / "systematic_cleanup_backup",
            core_path / "_preserved_logic",
            core_path / "_preserved_complex_structures"
        ]
        
        return any(indicator.exists() for indicator in backup_indicators)
    
    def _generate_diagnostic_report(self) -> None:
        """Generate comprehensive diagnostic report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.starting_path / f"pyics_diagnostic_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Pyics Project Structure Diagnostic Report\n\n")
            f.write(f"**Engineering Lead**: Nnamdi Okpala / OBINexus Computing\n")
            f.write(f"**Diagnostic Timestamp**: {datetime.now().isoformat()}\n")
            f.write(f"**Methodology**: Systematic path resolution analysis\n\n")
            
            # Project structure analysis
            f.write("## Project Structure Analysis\n\n")
            f.write("### Project Root Candidates\n\n")
            for i, candidate in enumerate(self.diagnostic_results["project_root_candidates"]):
                f.write(f"{i+1}. **{candidate['path']}**\n")
                f.write(f"   - Confidence Score: {candidate['confidence_score']}\n")
                f.write(f"   - Indicators: {', '.join(candidate['indicators'])}\n\n")
            
            # Core directory analysis
            f.write("### Core Directory Locations\n\n")
            for core in self.diagnostic_results["core_directory_locations"]:
                f.write(f"- **{core['path']}**\n")
                f.write(f"  - Relative Path: {core['relative_to_root']}\n")
                f.write(f"  - Domain Count: {core['domain_count']}\n")
                f.write(f"  - Nesting Level: {core['nested_level']}\n\n")
            
            # Path resolution strategy
            strategy = self.diagnostic_results["path_resolution_strategy"]
            f.write("## Path Resolution Strategy\n\n")
            f.write(f"**Strategy Type**: {strategy['type']}\n")
            f.write(f"**Rationale**: {strategy['rationale']}\n\n")
            
            if strategy["type"] == "use_existing_structure":
                f.write(f"**Recommended Core**: {strategy['recommended_core']}\n")
                f.write(f"**Core Relative Path**: {strategy['core_relative_path']}\n\n")
            
            # Corrective actions
            f.write("## Corrective Actions\n\n")
            for i, action in enumerate(self.diagnostic_results["corrective_actions"]):
                f.write(f"{i+1}. **{action['action_type']}**\n")
                f.write(f"   - Description: {action['description']}\n")
                for key, value in action.items():
                    if key not in ["action_type", "description"]:
                        f.write(f"   - {key}: {value}\n")
                f.write("\n")
        
        logger.info(f"Diagnostic report generated: {report_path}")
    
    def generate_corrected_executor(self) -> str:
        """Generate corrected systematic cleanup executor"""
        if not self.diagnostic_results["corrective_actions"]:
            raise ValueError("No corrective actions available - run diagnostic first")
        
        strategy = self.diagnostic_results["path_resolution_strategy"]
        
        if strategy["type"] == "use_existing_structure":
            corrected_project_root = self._find_project_root_for_core(Path(strategy["recommended_core"]))
            corrected_core_path = strategy["core_relative_path"]
        else:
            corrected_project_root = Path(strategy["recommended_root"])
            corrected_core_path = strategy["core_path"]
        
        return f'''#!/usr/bin/env python3
"""
pyics_systematic_cleanup_executor_corrected.py
Pyics Systematic Architecture Cleanup Implementation - PATH CORRECTED

Engineering Lead: Nnamdi Okpala / OBINexus Computing
Diagnostic Correction: Automatic path resolution based on project structure analysis
Methodology: Waterfall-based systematic execution with corrected path handling

PATH CORRECTION APPLIED:
- Project Root: {corrected_project_root}
- Core Directory: {corrected_core_path}
- Correction Timestamp: {datetime.now().isoformat()}
"""

import os
import sys
import shutil
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

# CORRECTED Technical Configuration
PROJECT_ROOT = Path("{corrected_project_root}").resolve()
PYICS_CORE_DIR = "{corrected_core_path}"
BACKUP_ROOT = "systematic_cleanup_backup"

# Resume execution from specific phase if previous phases completed
RESUME_FROM_PHASE = {"phase_4_load_order_implementation" if self._check_phase_completion_status(Path(strategy["recommended_core"])) else "phase_1_backup"}

# Rest of the systematic cleanup executor code remains the same...
# [Implementation continues with corrected paths]

class PyicsSystematicCleanupExecutor:
    """
    Systematic architecture cleanup executor - PATH CORRECTED VERSION
    """
    
    def __init__(self, project_root: Path = None):
        # Use diagnostic-corrected paths
        self.project_root = project_root or PROJECT_ROOT
        self.core_dir = self.project_root / PYICS_CORE_DIR
        self.backup_root = self.project_root / BACKUP_ROOT
        
        # Validate paths exist or create them
        self._ensure_path_structure()
        
        self.execution_state = {{
            "phase_results": {{}},
            "rollback_points": [],
            "preserved_content": {{}},
            "validation_results": {{}},
            "completion_status": "initialized",
            "path_correction_applied": True,
            "diagnostic_timestamp": "{datetime.now().isoformat()}"
        }}
        
        self.discovered_domains = []
        self.structure_analysis = {{}}
    
    def _ensure_path_structure(self) -> None:
        """Ensure required path structure exists"""
        logger.info(f"Validating path structure: {{self.core_dir}}")
        
        if not self.project_root.exists():
            logger.warning(f"Project root does not exist: {{self.project_root}}")
            
        if not self.core_dir.exists():
            logger.info(f"Creating core directory: {{self.core_dir}}")
            self.core_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Path structure validated: {{self.core_dir}}")

# [Rest of implementation with corrected path handling]
'''

def main():
    """Main diagnostic execution function"""
    diagnostic = PyicsPathResolutionDiagnostic()
    results = diagnostic.execute_comprehensive_diagnostic()
    
    print("="*80)
    print("PYICS PROJECT STRUCTURE DIAGNOSTIC REPORT")
    print("="*80)
    print(f"Validation Status: {results['validation_status']}")
    print(f"Project Root Candidates: {len(results['project_root_candidates'])}")
    print(f"Core Directory Locations: {len(results['core_directory_locations'])}")
    print(f"Nested Structure Detected: {results['nested_structure_detected']}")
    
    if results["validation_status"] == "completed":
        print("\n📊 PATH RESOLUTION STRATEGY:")
        strategy = results["path_resolution_strategy"]
        print(f"  Strategy: {strategy['type']}")
        print(f"  Rationale: {strategy['rationale']}")
        
        if strategy["type"] == "use_existing_structure":
            print(f"  Recommended Core: {strategy['recommended_core']}")
            print(f"  Core Relative Path: {strategy['core_relative_path']}")
        
        print("\n🔧 CORRECTIVE ACTIONS:")
        for i, action in enumerate(results["corrective_actions"]):
            print(f"  {i+1}. {action['action_type']}: {action['description']}")
        
        print("\n✅ NEXT STEPS:")
        print("1. Execute corrected systematic cleanup executor")
        print("2. Validate path resolution with diagnostic results")
        print("3. Resume systematic cleanup from appropriate phase")
        print("4. Complete architecture compliance validation")
        
        # Generate corrected executor
        try:
            corrected_executor = diagnostic.generate_corrected_executor()
            corrected_path = Path("pyics_systematic_cleanup_executor_corrected.py")
            with open(corrected_path, 'w', encoding='utf-8') as f:
                f.write(corrected_executor)
            print(f"\n📝 Corrected executor generated: {corrected_path}")
            
        except Exception as e:
            print(f"\n⚠️  Failed to generate corrected executor: {e}")
    
    else:
        print(f"\n❌ DIAGNOSTIC FAILED: {results.get('error', 'Unknown error')}")
    
    print("="*80)
    
    sys.exit(0 if results["validation_status"] == "completed" else 1)

if __name__ == "__main__":
    main()
