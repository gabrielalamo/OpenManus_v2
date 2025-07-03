"""
Code Analyzer Tool
Provides advanced code analysis, quality assessment, and improvement recommendations
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class CodeAnalyzerTool(BaseTool):
    """
    Advanced code analysis tool for evaluating code quality, identifying issues,
    and providing improvement recommendations.
    """

    name: str = "code_analyzer"
    description: str = """
    Analyze code for quality, maintainability, security issues, and performance concerns.
    Provides detailed reports with actionable recommendations for improvement.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "analyze_code", "review_file", "check_security", "assess_complexity",
                    "suggest_improvements", "check_style", "detect_bugs", "analyze_dependencies",
                    "evaluate_test_coverage", "generate_metrics"
                ],
                "description": "The type of code analysis to perform"
            },
            "code": {
                "type": "string",
                "description": "Code snippet to analyze (for analyze_code action)"
            },
            "file_path": {
                "type": "string",
                "description": "Path to the file to analyze (for review_file action)"
            },
            "language": {
                "type": "string",
                "enum": ["python", "javascript", "typescript", "java", "csharp", "go", "rust", "php", "ruby", "other"],
                "description": "Programming language of the code"
            },
            "analysis_depth": {
                "type": "string",
                "enum": ["basic", "standard", "deep", "comprehensive"],
                "description": "Depth of analysis to perform"
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific areas to focus the analysis on (e.g., 'security', 'performance', 'maintainability')"
            },
            "ignore_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patterns to ignore during analysis"
            }
        },
        "required": ["action"]
    }

    # Analysis data storage
    analysis_history: List[Dict[str, Any]] = Field(default_factory=list)
    code_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    known_patterns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_analysis_patterns()

    def _initialize_analysis_patterns(self):
        """Initialize patterns for code analysis."""
        
        # Initialize known code patterns for different languages
        self.known_patterns = {
            "python": {
                "security_issues": [
                    {"pattern": r"eval\(", "description": "Potentially unsafe eval() function"},
                    {"pattern": r"exec\(", "description": "Potentially unsafe exec() function"},
                    {"pattern": r"os\.system\(", "description": "Potentially unsafe system command execution"},
                    {"pattern": r"subprocess\.call\(", "description": "Subprocess call without proper input validation"},
                    {"pattern": r"__import__\(", "description": "Dynamic import may be unsafe"},
                    {"pattern": r"pickle\.load", "description": "Unsafe deserialization with pickle"},
                    {"pattern": r"yaml\.load\([^,]", "description": "Unsafe YAML loading without safe_load"},
                    {"pattern": r"request\.form\[", "description": "Form data used without validation"},
                ],
                "performance_issues": [
                    {"pattern": r"for\s+\w+\s+in\s+range\(len\((\w+)\)\):", "description": "Inefficient for loop using range(len())"},
                    {"pattern": r"\.append\(.*\)\s+for\s+", "description": "List building in loop, consider list comprehension"},
                    {"pattern": r"\+\s*=.*\s+in\s+", "description": "String concatenation in loop, consider join()"},
                    {"pattern": r"if\s+\w+\s+in\s+\[", "description": "Membership test in list, consider using set"},
                ],
                "style_issues": [
                    {"pattern": r"^[^#\n]*[^:]\s*$", "description": "Missing trailing colon"},
                    {"pattern": r"^\s*def\s+\w+\s*\([^)]*\)\s*:", "description": "Function definition without docstring"},
                    {"pattern": r"^\s*class\s+\w+[^:]*:", "description": "Class definition without docstring"},
                    {"pattern": r"\s{2,}$", "description": "Trailing whitespace"},
                    {"pattern": r"\t", "description": "Tab character used instead of spaces"},
                ],
                "maintainability_issues": [
                    {"pattern": r"def\s+\w+\s*\([^)]{120,}\)", "description": "Function with too many parameters"},
                    {"pattern": r"def\s+\w+\s*\(.*\):\s*\n\s{4}[^\n]{200,}", "description": "Function with very long body"},
                    {"pattern": r"if.*if.*if.*if", "description": "Deeply nested conditionals"},
                    {"pattern": r"for.*for.*for.*for", "description": "Deeply nested loops"},
                    {"pattern": r"except\s*:", "description": "Bare except clause"},
                ],
            },
            "javascript": {
                "security_issues": [
                    {"pattern": r"eval\(", "description": "Potentially unsafe eval() function"},
                    {"pattern": r"document\.write\(", "description": "Potentially unsafe document.write()"},
                    {"pattern": r"innerHTML\s*=", "description": "Potentially unsafe innerHTML assignment"},
                    {"pattern": r"new\s+Function\(", "description": "Potentially unsafe Function constructor"},
                    {"pattern": r"localStorage\.", "description": "Storing sensitive data in localStorage"},
                ],
                "performance_issues": [
                    {"pattern": r"for\s*\(\s*var\s+\w+\s*=\s*0", "description": "Consider using for...of or forEach"},
                    {"pattern": r"\+\s*=.*\s+in\s+", "description": "String concatenation in loop, consider template literals"},
                    {"pattern": r"\.forEach\(.*\{\s*return", "description": "forEach with return, consider map()"},
                ],
                "style_issues": [
                    {"pattern": r"var\s+", "description": "Using var instead of let/const"},
                    {"pattern": r"==(?!=)", "description": "Using == instead of ==="},
                    {"pattern": r"!=(?!=)", "description": "Using != instead of !=="},
                    {"pattern": r";\s*$", "description": "Unnecessary semicolon"},
                ],
                "maintainability_issues": [
                    {"pattern": r"function\s+\w+\s*\([^)]{120,}\)", "description": "Function with too many parameters"},
                    {"pattern": r"function\s+\w+\s*\(.*\)\s*\{[^}]{200,}", "description": "Function with very long body"},
                    {"pattern": r"if.*if.*if.*if", "description": "Deeply nested conditionals"},
                    {"pattern": r"for.*for.*for.*for", "description": "Deeply nested loops"},
                    {"pattern": r"catch\s*\(\s*\)", "description": "Empty catch block"},
                ],
            },
        }

    async def execute(
        self,
        action: str,
        code: Optional[str] = None,
        file_path: Optional[str] = None,
        language: str = "python",
        analysis_depth: str = "standard",
        focus_areas: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the code analysis action."""
        
        try:
            if action == "analyze_code":
                return await self._analyze_code(code, language, analysis_depth, focus_areas, ignore_patterns)
            elif action == "review_file":
                return await self._review_file(file_path, language, analysis_depth, focus_areas)
            elif action == "check_security":
                return await self._check_security(code, file_path, language)
            elif action == "assess_complexity":
                return await self._assess_complexity(code, file_path, language)
            elif action == "suggest_improvements":
                return await self._suggest_improvements(code, file_path, language, focus_areas)
            elif action == "check_style":
                return await self._check_style(code, file_path, language)
            elif action == "detect_bugs":
                return await self._detect_bugs(code, file_path, language)
            elif action == "analyze_dependencies":
                return await self._analyze_dependencies(file_path)
            elif action == "evaluate_test_coverage":
                return await self._evaluate_test_coverage(file_path)
            elif action == "generate_metrics":
                return await self._generate_metrics(code, file_path, language)
            else:
                return ToolResult(error=f"Unknown code analysis action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Code analysis error: {str(e)}")

    async def _analyze_code(
        self, 
        code: Optional[str], 
        language: str,
        analysis_depth: str,
        focus_areas: Optional[List[str]],
        ignore_patterns: Optional[List[str]]
    ) -> ToolResult:
        """Perform comprehensive code analysis."""
        
        if not code:
            return ToolResult(error="Code is required for analysis")

        analysis_result = {
            "language": language,
            "analysis_depth": analysis_depth,
            "timestamp": datetime.now().isoformat(),
            "code_metrics": {},
            "issues": {
                "security": [],
                "performance": [],
                "style": [],
                "maintainability": [],
                "bugs": []
            },
            "recommendations": []
        }

        # Calculate basic metrics
        analysis_result["code_metrics"] = self._calculate_code_metrics(code, language)
        
        # Identify issues based on patterns
        if language in self.known_patterns:
            patterns = self.known_patterns[language]
            
            # Filter focus areas if specified
            if focus_areas:
                filtered_patterns = {}
                for area in focus_areas:
                    area_key = f"{area}_issues"
                    if area_key in patterns:
                        filtered_patterns[area_key] = patterns[area_key]
                patterns = filtered_patterns if filtered_patterns else patterns
            
            # Apply ignore patterns
            if ignore_patterns:
                for category, category_patterns in patterns.items():
                    patterns[category] = [
                        p for p in category_patterns 
                        if not any(re.search(ignore, p["pattern"]) for ignore in ignore_patterns)
                    ]
            
            # Check for issues
            for category, category_patterns in patterns.items():
                category_name = category.replace("_issues", "")
                for pattern_info in category_patterns:
                    matches = re.finditer(pattern_info["pattern"], code)
                    for match in matches:
                        line_number = code[:match.start()].count('\n') + 1
                        issue = {
                            "line": line_number,
                            "description": pattern_info["description"],
                            "severity": "medium",  # Default severity
                            "code_snippet": code.splitlines()[line_number-1] if line_number <= len(code.splitlines()) else ""
                        }
                        analysis_result["issues"][category_name].append(issue)
        
        # Generate recommendations based on issues
        analysis_result["recommendations"] = self._generate_recommendations(analysis_result["issues"], language)
        
        # Store analysis in history
        self.analysis_history.append(analysis_result)
        
        return ToolResult(output=self._format_analysis_result(analysis_result))

    async def _review_file(
        self,
        file_path: str,
        language: str,
        analysis_depth: str,
        focus_areas: Optional[List[str]]
    ) -> ToolResult:
        """Review a file for code quality issues."""
        
        if not file_path:
            return ToolResult(error="File path is required for file review")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Determine language from file extension if not specified
            if language == "other":
                extension = file_path.split('.')[-1].lower()
                language_map = {
                    'py': 'python',
                    'js': 'javascript',
                    'ts': 'typescript',
                    'java': 'java',
                    'cs': 'csharp',
                    'go': 'go',
                    'rs': 'rust',
                    'php': 'php',
                    'rb': 'ruby'
                }
                language = language_map.get(extension, 'other')
            
            # Analyze the code
            return await self._analyze_code(code, language, analysis_depth, focus_areas, None)
            
        except FileNotFoundError:
            return ToolResult(error=f"File not found: {file_path}")
        except Exception as e:
            return ToolResult(error=f"Error reviewing file: {str(e)}")

    async def _check_security(
        self,
        code: Optional[str],
        file_path: Optional[str],
        language: str
    ) -> ToolResult:
        """Check code for security vulnerabilities."""
        
        # Get code from file if not provided directly
        if not code and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return ToolResult(error=f"File not found: {file_path}")
            except Exception as e:
                return ToolResult(error=f"Error reading file: {str(e)}")
        
        if not code:
            return ToolResult(error="Code or file path is required for security check")
        
        security_result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "vulnerabilities": [],
            "security_score": 0.0,
            "critical_issues": [],
            "recommendations": []
        }
        
        # Check for security vulnerabilities
        if language in self.known_patterns and "security_issues" in self.known_patterns[language]:
            security_patterns = self.known_patterns[language]["security_issues"]
            
            for pattern_info in security_patterns:
                matches = re.finditer(pattern_info["pattern"], code)
                for match in matches:
                    line_number = code[:match.start()].count('\n') + 1
                    vulnerability = {
                        "line": line_number,
                        "description": pattern_info["description"],
                        "severity": "high",  # Security issues are typically high severity
                        "code_snippet": code.splitlines()[line_number-1] if line_number <= len(code.splitlines()) else ""
                    }
                    security_result["vulnerabilities"].append(vulnerability)
        
        # Calculate security score (inverse of vulnerability density)
        lines_of_code = len(code.splitlines())
        vulnerability_count = len(security_result["vulnerabilities"])
        
        if lines_of_code > 0:
            vulnerability_density = vulnerability_count / lines_of_code
            security_result["security_score"] = max(0.0, 1.0 - (vulnerability_density * 100))
        else:
            security_result["security_score"] = 1.0 if vulnerability_count == 0 else 0.0
        
        # Identify critical issues
        security_result["critical_issues"] = [
            v for v in security_result["vulnerabilities"] 
            if v["severity"] == "high"
        ]
        
        # Generate security recommendations
        security_result["recommendations"] = self._generate_security_recommendations(
            security_result["vulnerabilities"], language
        )
        
        return ToolResult(output=self._format_security_result(security_result))

    async def _assess_complexity(
        self,
        code: Optional[str],
        file_path: Optional[str],
        language: str
    ) -> ToolResult:
        """Assess code complexity metrics."""
        
        # Get code from file if not provided directly
        if not code and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return ToolResult(error=f"File not found: {file_path}")
            except Exception as e:
                return ToolResult(error=f"Error reading file: {str(e)}")
        
        if not code:
            return ToolResult(error="Code or file path is required for complexity assessment")
        
        complexity_result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "complexity_metrics": {},
            "hotspots": [],
            "maintainability_index": 0.0,
            "recommendations": []
        }
        
        # Calculate complexity metrics
        complexity_result["complexity_metrics"] = self._calculate_complexity_metrics(code, language)
        
        # Identify complexity hotspots
        complexity_result["hotspots"] = self._identify_complexity_hotspots(code, language)
        
        # Calculate maintainability index
        complexity_result["maintainability_index"] = self._calculate_maintainability_index(
            complexity_result["complexity_metrics"]
        )
        
        # Generate recommendations for reducing complexity
        complexity_result["recommendations"] = self._generate_complexity_recommendations(
            complexity_result["hotspots"], language
        )
        
        return ToolResult(output=self._format_complexity_result(complexity_result))

    async def _suggest_improvements(
        self,
        code: Optional[str],
        file_path: Optional[str],
        language: str,
        focus_areas: Optional[List[str]]
    ) -> ToolResult:
        """Suggest code improvements based on best practices."""
        
        # Get code from file if not provided directly
        if not code and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return ToolResult(error=f"File not found: {file_path}")
            except Exception as e:
                return ToolResult(error=f"Error reading file: {str(e)}")
        
        if not code:
            return ToolResult(error="Code or file path is required for suggesting improvements")
        
        improvements_result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "focus_areas": focus_areas or ["style", "performance", "maintainability", "security"],
            "suggestions": [],
            "code_samples": {},
            "priority_improvements": []
        }
        
        # Analyze code for different improvement areas
        for area in improvements_result["focus_areas"]:
            area_suggestions = self._analyze_improvement_area(code, language, area)
            improvements_result["suggestions"].extend(area_suggestions)
        
        # Generate code samples for top improvements
        top_suggestions = improvements_result["suggestions"][:3]
        for i, suggestion in enumerate(top_suggestions):
            improvements_result["code_samples"][f"sample_{i+1}"] = self._generate_improved_code_sample(
                code, suggestion, language
            )
        
        # Identify priority improvements
        improvements_result["priority_improvements"] = [
            s for s in improvements_result["suggestions"] 
            if s.get("priority", "medium") == "high"
        ]
        
        return ToolResult(output=self._format_improvements_result(improvements_result))

    async def _check_style(
        self,
        code: Optional[str],
        file_path: Optional[str],
        language: str
    ) -> ToolResult:
        """Check code style against language conventions."""
        
        # Get code from file if not provided directly
        if not code and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return ToolResult(error=f"File not found: {file_path}")
            except Exception as e:
                return ToolResult(error=f"Error reading file: {str(e)}")
        
        if not code:
            return ToolResult(error="Code or file path is required for style check")
        
        style_result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "style_issues": [],
            "style_score": 0.0,
            "convention_violations": {},
            "auto_fixable": []
        }
        
        # Check for style issues
        if language in self.known_patterns and "style_issues" in self.known_patterns[language]:
            style_patterns = self.known_patterns[language]["style_issues"]
            
            for pattern_info in style_patterns:
                matches = re.finditer(pattern_info["pattern"], code)
                for match in matches:
                    line_number = code[:match.start()].count('\n') + 1
                    issue = {
                        "line": line_number,
                        "description": pattern_info["description"],
                        "severity": "low",  # Style issues are typically low severity
                        "code_snippet": code.splitlines()[line_number-1] if line_number <= len(code.splitlines()) else "",
                        "auto_fixable": pattern_info.get("auto_fixable", False)
                    }
                    style_result["style_issues"].append(issue)
                    
                    # Track auto-fixable issues
                    if issue["auto_fixable"]:
                        style_result["auto_fixable"].append(issue)
        
        # Calculate style score
        lines_of_code = len(code.splitlines())
        style_issue_count = len(style_result["style_issues"])
        
        if lines_of_code > 0:
            style_issue_density = style_issue_count / lines_of_code
            style_result["style_score"] = max(0.0, 1.0 - (style_issue_density * 10))
        else:
            style_result["style_score"] = 1.0 if style_issue_count == 0 else 0.0
        
        # Group violations by convention
        for issue in style_result["style_issues"]:
            convention = self._determine_style_convention(issue["description"], language)
            if convention not in style_result["convention_violations"]:
                style_result["convention_violations"][convention] = 0
            style_result["convention_violations"][convention] += 1
        
        return ToolResult(output=self._format_style_result(style_result))

    async def _detect_bugs(
        self,
        code: Optional[str],
        file_path: Optional[str],
        language: str
    ) -> ToolResult:
        """Detect potential bugs and logical errors in code."""
        
        # Get code from file if not provided directly
        if not code and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return ToolResult(error=f"File not found: {file_path}")
            except Exception as e:
                return ToolResult(error=f"Error reading file: {str(e)}")
        
        if not code:
            return ToolResult(error="Code or file path is required for bug detection")
        
        bugs_result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "potential_bugs": [],
            "logical_errors": [],
            "risk_assessment": {},
            "fix_suggestions": []
        }
        
        # Detect potential bugs
        bugs_result["potential_bugs"] = self._detect_potential_bugs(code, language)
        
        # Identify logical errors
        bugs_result["logical_errors"] = self._identify_logical_errors(code, language)
        
        # Assess risk level for each bug
        bugs_result["risk_assessment"] = self._assess_bug_risks(
            bugs_result["potential_bugs"] + bugs_result["logical_errors"]
        )
        
        # Generate fix suggestions
        bugs_result["fix_suggestions"] = self._generate_bug_fix_suggestions(
            bugs_result["potential_bugs"] + bugs_result["logical_errors"], language
        )
        
        return ToolResult(output=self._format_bugs_result(bugs_result))

    async def _analyze_dependencies(self, file_path: str) -> ToolResult:
        """Analyze project dependencies for issues and updates."""
        
        if not file_path:
            return ToolResult(error="File path is required for dependency analysis")
        
        # Determine dependency file type
        if file_path.endswith('requirements.txt'):
            return await self._analyze_python_dependencies(file_path)
        elif file_path.endswith('package.json'):
            return await self._analyze_node_dependencies(file_path)
        elif file_path.endswith('pom.xml'):
            return await self._analyze_java_dependencies(file_path)
        elif file_path.endswith('Gemfile'):
            return await self._analyze_ruby_dependencies(file_path)
        else:
            return ToolResult(error=f"Unsupported dependency file format: {file_path}")

    async def _evaluate_test_coverage(self, file_path: str) -> ToolResult:
        """Evaluate test coverage for a project or specific file."""
        
        if not file_path:
            return ToolResult(error="File path is required for test coverage evaluation")
        
        coverage_result = {
            "timestamp": datetime.now().isoformat(),
            "target": file_path,
            "coverage_metrics": {},
            "uncovered_areas": [],
            "test_quality_assessment": {},
            "recommendations": []
        }
        
        # This would implement actual test coverage analysis
        # For now, return a placeholder result
        coverage_result["coverage_metrics"] = {
            "line_coverage": 0.75,
            "branch_coverage": 0.65,
            "function_coverage": 0.80,
            "overall_coverage": 0.73
        }
        
        coverage_result["uncovered_areas"] = [
            {"file": "example.py", "lines": "45-52", "description": "Error handling branch"},
            {"file": "example.py", "lines": "78-85", "description": "Edge case processing"}
        ]
        
        coverage_result["test_quality_assessment"] = {
            "test_count": 24,
            "assertion_count": 67,
            "test_to_code_ratio": 0.8,
            "quality_score": 0.7
        }
        
        coverage_result["recommendations"] = [
            "Add tests for error handling in example.py (lines 45-52)",
            "Improve branch coverage with additional test cases",
            "Consider property-based testing for edge cases"
        ]
        
        return ToolResult(output=self._format_coverage_result(coverage_result))

    async def _generate_metrics(
        self,
        code: Optional[str],
        file_path: Optional[str],
        language: str
    ) -> ToolResult:
        """Generate comprehensive code metrics."""
        
        # Get code from file if not provided directly
        if not code and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return ToolResult(error=f"File not found: {file_path}")
            except Exception as e:
                return ToolResult(error=f"Error reading file: {str(e)}")
        
        if not code:
            return ToolResult(error="Code or file path is required for metrics generation")
        
        metrics_result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "size_metrics": {},
            "complexity_metrics": {},
            "maintainability_metrics": {},
            "quality_score": 0.0,
            "benchmark_comparison": {}
        }
        
        # Calculate size metrics
        metrics_result["size_metrics"] = self._calculate_size_metrics(code)
        
        # Calculate complexity metrics
        metrics_result["complexity_metrics"] = self._calculate_complexity_metrics(code, language)
        
        # Calculate maintainability metrics
        metrics_result["maintainability_metrics"] = self._calculate_maintainability_metrics(
            code, metrics_result["complexity_metrics"]
        )
        
        # Calculate overall quality score
        metrics_result["quality_score"] = self._calculate_quality_score(
            metrics_result["complexity_metrics"],
            metrics_result["maintainability_metrics"]
        )
        
        # Compare with benchmarks
        metrics_result["benchmark_comparison"] = self._compare_with_benchmarks(
            metrics_result, language
        )
        
        return ToolResult(output=self._format_metrics_result(metrics_result))

    # Helper methods for analysis

    def _calculate_code_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate basic code metrics."""
        lines = code.splitlines()
        
        # Count different types of lines
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        comment_markers = {
            "python": ["#"],
            "javascript": ["//", "/*", "*/"],
            "typescript": ["//", "/*", "*/"],
            "java": ["//", "/*", "*/"],
            "csharp": ["//", "/*", "*/"],
            "go": ["//", "/*", "*/"],
            "rust": ["//", "/*", "*/"],
            "php": ["//", "#", "/*", "*/"],
            "ruby": ["#"],
        }
        
        markers = comment_markers.get(language, ["#", "//"])
        in_block_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for blank lines
            if not stripped:
                blank_lines += 1
                continue
            
            # Check for block comments
            if language in ["javascript", "typescript", "java", "csharp", "go", "rust", "php"]:
                if "/*" in stripped and "*/" in stripped:
                    comment_lines += 1
                    continue
                elif "/*" in stripped:
                    in_block_comment = True
                    comment_lines += 1
                    continue
                elif "*/" in stripped:
                    in_block_comment = False
                    comment_lines += 1
                    continue
                elif in_block_comment:
                    comment_lines += 1
                    continue
            
            # Check for line comments
            is_comment = False
            for marker in markers:
                if stripped.startswith(marker):
                    comment_lines += 1
                    is_comment = True
                    break
            
            if not is_comment:
                code_lines += 1
        
        # Calculate metrics
        total_lines = len(lines)
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "comment_ratio": comment_ratio,
            "average_line_length": sum(len(line) for line in lines) / total_lines if total_lines > 0 else 0
        }

    def _generate_recommendations(self, issues: Dict[str, List[Dict[str, Any]]], language: str) -> List[str]:
        """Generate recommendations based on identified issues."""
        recommendations = []
        
        # Security recommendations
        if issues["security"]:
            recommendations.append("Address security vulnerabilities as highest priority")
            for issue in issues["security"][:3]:  # Top 3 security issues
                recommendations.append(f"Fix security issue: {issue['description']} (line {issue['line']})")
        
        # Performance recommendations
        if issues["performance"]:
            recommendations.append("Improve code performance by addressing inefficient patterns")
            for issue in issues["performance"][:2]:  # Top 2 performance issues
                recommendations.append(f"Optimize: {issue['description']} (line {issue['line']})")
        
        # Style recommendations
        if issues["style"]:
            style_count = len(issues["style"])
            if style_count > 10:
                recommendations.append(f"Apply consistent code style (found {style_count} style issues)")
            elif style_count > 0:
                recommendations.append("Address style inconsistencies for better readability")
        
        # Maintainability recommendations
        if issues["maintainability"]:
            recommendations.append("Improve code maintainability")
            for issue in issues["maintainability"][:2]:  # Top 2 maintainability issues
                recommendations.append(f"Refactor: {issue['description']} (line {issue['line']})")
        
        # General recommendations based on language
        if language == "python":
            recommendations.append("Consider using type hints for better code documentation")
            recommendations.append("Ensure proper docstrings for functions and classes")
        elif language in ["javascript", "typescript"]:
            recommendations.append("Consider using ESLint to enforce code quality")
            recommendations.append("Use modern ES6+ features for cleaner code")
        
        return recommendations

    def _calculate_complexity_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        # This is a simplified implementation
        # In a real tool, this would use language-specific parsers
        
        lines = code.splitlines()
        
        # Count control structures as a simple proxy for cyclomatic complexity
        control_keywords = {
            "python": ["if", "elif", "else", "for", "while", "try", "except", "with"],
            "javascript": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
            "typescript": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
            "java": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
            "csharp": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
            "go": ["if", "else", "for", "switch", "case", "select"],
            "rust": ["if", "else", "for", "while", "match", "loop"],
            "php": ["if", "elseif", "else", "for", "foreach", "while", "switch", "case", "try", "catch"],
            "ruby": ["if", "elsif", "else", "for", "while", "until", "case", "when", "begin", "rescue"],
        }
        
        keywords = control_keywords.get(language, control_keywords["python"])
        
        # Count control structures
        control_count = 0
        for line in lines:
            stripped = line.strip()
            for keyword in keywords:
                # Match keyword as a whole word
                if re.search(r'\b' + keyword + r'\b', stripped):
                    control_count += 1
                    break
        
        # Count function/method definitions
        function_patterns = {
            "python": r'def\s+\w+\s*\(',
            "javascript": r'function\s+\w+\s*\(|const\s+\w+\s*=\s*\([^)]*\)\s*=>',
            "typescript": r'function\s+\w+\s*\(|const\s+\w+\s*=\s*\([^)]*\)\s*=>',
            "java": r'(public|private|protected)?\s+\w+\s+\w+\s*\(',
            "csharp": r'(public|private|protected)?\s+\w+\s+\w+\s*\(',
            "go": r'func\s+\w+\s*\(',
            "rust": r'fn\s+\w+\s*\(',
            "php": r'function\s+\w+\s*\(',
            "ruby": r'def\s+\w+\s*(\(|$)',
        }
        
        function_pattern = function_patterns.get(language, function_patterns["python"])
        function_count = len(re.findall(function_pattern, code))
        
        # Count class definitions
        class_patterns = {
            "python": r'class\s+\w+',
            "javascript": r'class\s+\w+',
            "typescript": r'class\s+\w+',
            "java": r'class\s+\w+',
            "csharp": r'class\s+\w+',
            "go": r'type\s+\w+\s+struct',
            "rust": r'struct\s+\w+|enum\s+\w+',
            "php": r'class\s+\w+',
            "ruby": r'class\s+\w+',
        }
        
        class_pattern = class_patterns.get(language, class_patterns["python"])
        class_count = len(re.findall(class_pattern, code))
        
        # Calculate nesting depth
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Increment nesting for opening structures
            if any(stripped.endswith(c) for c in [':', '{']) and any(k in stripped for k in keywords):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            
            # Decrement nesting for closing structures
            if stripped == '}' or (language == "python" and re.match(r'^\s*\w+', stripped) and current_nesting > 0):
                current_nesting = max(0, current_nesting - 1)
        
        # Calculate cyclomatic complexity (simplified)
        cyclomatic_complexity = control_count + 1
        
        return {
            "cyclomatic_complexity": cyclomatic_complexity,
            "function_count": function_count,
            "class_count": class_count,
            "control_structure_count": control_count,
            "max_nesting_depth": max_nesting,
            "complexity_per_function": cyclomatic_complexity / function_count if function_count > 0 else cyclomatic_complexity
        }

    def _identify_complexity_hotspots(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Identify complexity hotspots in the code."""
        lines = code.splitlines()
        hotspots = []
        
        # Function to extract function/method definitions with their body
        def extract_functions(code, language):
            functions = []
            
            if language == "python":
                # Match Python function definitions
                pattern = r'def\s+(\w+)\s*\([^)]*\):\s*(?:\n\s+[^\n]+)*'
                matches = re.finditer(pattern, code)
                for match in matches:
                    func_start = code[:match.start()].count('\n')
                    func_end = func_start + code[match.start():match.end()].count('\n')
                    functions.append({
                        "name": match.group(1),
                        "start_line": func_start + 1,
                        "end_line": func_end + 1,
                        "body": match.group(0)
                    })
            
            elif language in ["javascript", "typescript"]:
                # Match JS/TS function definitions (simplified)
                patterns = [
                    r'function\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\}',  # function declaration
                    r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*\}',  # arrow function
                    r'const\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*\{[^}]*\}'  # function expression
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, code)
                    for match in matches:
                        func_start = code[:match.start()].count('\n')
                        func_end = func_start + code[match.start():match.end()].count('\n')
                        functions.append({
                            "name": match.group(1),
                            "start_line": func_start + 1,
                            "end_line": func_end + 1,
                            "body": match.group(0)
                        })
            
            return functions
        
        functions = extract_functions(code, language)
        
        # Analyze each function for complexity
        for func in functions:
            # Count control structures in function body
            control_keywords = {
                "python": ["if", "elif", "else", "for", "while", "try", "except", "with"],
                "javascript": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
                "typescript": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
            }
            
            keywords = control_keywords.get(language, control_keywords["python"])
            control_count = 0
            
            for line in func["body"].splitlines():
                stripped = line.strip()
                for keyword in keywords:
                    if re.search(r'\b' + keyword + r'\b', stripped):
                        control_count += 1
                        break
            
            # Calculate function complexity
            complexity = control_count + 1
            
            # Check if function is a complexity hotspot
            if complexity > 10:  # High complexity threshold
                hotspots.append({
                    "type": "function",
                    "name": func["name"],
                    "start_line": func["start_line"],
                    "end_line": func["end_line"],
                    "complexity": complexity,
                    "severity": "high" if complexity > 15 else "medium"
                })
        
        return hotspots

    def _calculate_maintainability_index(self, complexity_metrics: Dict[str, Any]) -> float:
        """Calculate maintainability index based on complexity metrics."""
        # Simplified maintainability index calculation
        # In a real implementation, this would use more sophisticated formulas
        
        cyclomatic_complexity = complexity_metrics.get("cyclomatic_complexity", 1)
        max_nesting = complexity_metrics.get("max_nesting_depth", 1)
        
        # Higher is better (0-100 scale)
        maintainability = 100 - (cyclomatic_complexity * 0.8 + max_nesting * 5)
        
        # Clamp to 0-100 range
        return max(0, min(100, maintainability)) / 100

    def _generate_complexity_recommendations(self, hotspots: List[Dict[str, Any]], language: str) -> List[str]:
        """Generate recommendations for reducing complexity."""
        recommendations = []
        
        if not hotspots:
            recommendations.append("Code complexity is within acceptable limits")
            return recommendations
        
        # General recommendations
        recommendations.append("Consider refactoring complex functions into smaller, more focused units")
        recommendations.append("Reduce nesting depth by extracting logic into helper functions")
        
        # Specific recommendations for hotspots
        for hotspot in hotspots[:3]:  # Top 3 hotspots
            if hotspot["type"] == "function":
                recommendations.append(
                    f"Refactor function '{hotspot['name']}' (lines {hotspot['start_line']}-{hotspot['end_line']}) "
                    f"with complexity {hotspot['complexity']}"
                )
        
        # Language-specific recommendations
        if language == "python":
            recommendations.append("Use list comprehensions instead of complex loops where appropriate")
            recommendations.append("Consider using generator expressions for memory efficiency")
        elif language in ["javascript", "typescript"]:
            recommendations.append("Use array methods (map, filter, reduce) instead of complex loops")
            recommendations.append("Consider breaking complex components into smaller ones")
        
        return recommendations

    def _analyze_improvement_area(self, code: str, language: str, area: str) -> List[Dict[str, Any]]:
        """Analyze code for a specific improvement area."""
        suggestions = []
        
        if area == "style":
            # Style improvement suggestions
            style_issues = []
            if language in self.known_patterns and "style_issues" in self.known_patterns[language]:
                style_patterns = self.known_patterns[language]["style_issues"]
                
                for pattern_info in style_patterns:
                    matches = re.finditer(pattern_info["pattern"], code)
                    for match in matches:
                        line_number = code[:match.start()].count('\n') + 1
                        style_issues.append({
                            "line": line_number,
                            "description": pattern_info["description"],
                            "priority": "low"
                        })
            
            for issue in style_issues:
                suggestions.append({
                    "area": "style",
                    "description": f"Improve style: {issue['description']} (line {issue['line']})",
                    "priority": issue["priority"],
                    "effort": "low"
                })
        
        elif area == "performance":
            # Performance improvement suggestions
            performance_issues = []
            if language in self.known_patterns and "performance_issues" in self.known_patterns[language]:
                perf_patterns = self.known_patterns[language]["performance_issues"]
                
                for pattern_info in perf_patterns:
                    matches = re.finditer(pattern_info["pattern"], code)
                    for match in matches:
                        line_number = code[:match.start()].count('\n') + 1
                        performance_issues.append({
                            "line": line_number,
                            "description": pattern_info["description"],
                            "priority": "medium"
                        })
            
            for issue in performance_issues:
                suggestions.append({
                    "area": "performance",
                    "description": f"Optimize performance: {issue['description']} (line {issue['line']})",
                    "priority": issue["priority"],
                    "effort": "medium"
                })
        
        elif area == "maintainability":
            # Maintainability improvement suggestions
            maintainability_issues = []
            if language in self.known_patterns and "maintainability_issues" in self.known_patterns[language]:
                maint_patterns = self.known_patterns[language]["maintainability_issues"]
                
                for pattern_info in maint_patterns:
                    matches = re.finditer(pattern_info["pattern"], code)
                    for match in matches:
                        line_number = code[:match.start()].count('\n') + 1
                        maintainability_issues.append({
                            "line": line_number,
                            "description": pattern_info["description"],
                            "priority": "medium"
                        })
            
            for issue in maintainability_issues:
                suggestions.append({
                    "area": "maintainability",
                    "description": f"Improve maintainability: {issue['description']} (line {issue['line']})",
                    "priority": issue["priority"],
                    "effort": "high"
                })
        
        elif area == "security":
            # Security improvement suggestions
            security_issues = []
            if language in self.known_patterns and "security_issues" in self.known_patterns[language]:
                sec_patterns = self.known_patterns[language]["security_issues"]
                
                for pattern_info in sec_patterns:
                    matches = re.finditer(pattern_info["pattern"], code)
                    for match in matches:
                        line_number = code[:match.start()].count('\n') + 1
                        security_issues.append({
                            "line": line_number,
                            "description": pattern_info["description"],
                            "priority": "high"
                        })
            
            for issue in security_issues:
                suggestions.append({
                    "area": "security",
                    "description": f"Fix security issue: {issue['description']} (line {issue['line']})",
                    "priority": issue["priority"],
                    "effort": "high"
                })
        
        return suggestions

    def _generate_improved_code_sample(self, code: str, suggestion: Dict[str, Any], language: str) -> str:
        """Generate an improved code sample based on a suggestion."""
        # This is a placeholder implementation
        # In a real tool, this would generate actual improved code
        
        return f"# Improved code sample for: {suggestion['description']}\n# This would show the actual implementation"

    def _detect_potential_bugs(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect potential bugs in the code."""
        bugs = []
        
        # Define bug patterns for different languages
        bug_patterns = {
            "python": [
                {"pattern": r'if\s+\w+\s*=\s*\w+', "description": "Assignment in if condition (should be ==)"},
                {"pattern": r'except\s*:', "description": "Bare except clause catches all exceptions"},
                {"pattern": r'return\s+\w+\s*\+\s*\w+\s*\+', "description": "Potential type error in addition"},
                {"pattern": r'for\s+\w+\s+in\s+\w+\s*:\s*\n\s*pass', "description": "Empty loop body"},
                {"pattern": r'def\s+\w+\s*\([^)]*\)\s*:\s*\n\s*pass', "description": "Empty function body"},
            ],
            "javascript": [
                {"pattern": r'if\s*\(\s*\w+\s*=\s*\w+\s*\)', "description": "Assignment in if condition (should be ==)"},
                {"pattern": r'==\s*null', "description": "Use === null for strict equality"},
                {"pattern": r'for\s*\(\s*var\s+\w+\s*=\s*0;\s*\w+\s*<\s*\w+\.length;\s*\w+\+\+\s*\)\s*\{\s*\}', "description": "Empty loop body"},
                {"pattern": r'console\.log\(', "description": "Debug code left in production"},
                {"pattern": r'setTimeout\(\s*function\s*\(\s*\)\s*\{\s*\},\s*0\s*\)', "description": "Zero timeout is an anti-pattern"},
            ],
        }
        
        patterns = bug_patterns.get(language, [])
        
        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], code)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                bugs.append({
                    "line": line_number,
                    "description": pattern_info["description"],
                    "severity": "medium",
                    "code_snippet": code.splitlines()[line_number-1] if line_number <= len(code.splitlines()) else ""
                })
        
        return bugs

    def _identify_logical_errors(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Identify potential logical errors in the code."""
        errors = []
        
        # Define logical error patterns for different languages
        logical_error_patterns = {
            "python": [
                {"pattern": r'if\s+\w+\s*>\s*\d+\s*:\s*\n\s*.*\n\s*elif\s+\w+\s*>\s*\d+\s*:', "description": "Overlapping if-elif conditions"},
                {"pattern": r'if\s+\w+\s*>\s*\d+\s*:\s*\n\s*.*\n\s*if\s+\w+\s*<\s*\d+\s*:', "description": "Potentially redundant nested if conditions"},
                {"pattern": r'return\s+\w+\s*\n\s*\w+', "description": "Unreachable code after return statement"},
                {"pattern": r'while\s+True\s*:\s*\n\s*[^b\n]*\n\s*[^b\n]*\n\s*[^b\n]*\n\s*[^b\n]*\n', "description": "Infinite loop without break condition"},
            ],
            "javascript": [
                {"pattern": r'if\s*\(\s*\w+\s*>\s*\d+\s*\)\s*\{\s*.*\s*\}\s*else\s+if\s*\(\s*\w+\s*>\s*\d+\s*\)', "description": "Overlapping if-else if conditions"},
                {"pattern": r'return\s+\w+\s*;\s*\n\s*\w+', "description": "Unreachable code after return statement"},
                {"pattern": r'while\s*\(\s*true\s*\)\s*\{\s*[^b\n]*\n\s*[^b\n]*\n\s*[^b\n]*\n\s*[^b\n]*\n', "description": "Infinite loop without break condition"},
                {"pattern": r'switch\s*\(\s*\w+\s*\)\s*\{\s*case[^:]*:[^b\n]*\n\s*[^b\n]*\n\s*[^b\n]*\n\s*[^c\n]*\n', "description": "Switch case without break statement"},
            ],
        }
        
        patterns = logical_error_patterns.get(language, [])
        
        for pattern_info in patterns:
            matches = re.finditer(pattern_info["pattern"], code)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                errors.append({
                    "line": line_number,
                    "description": pattern_info["description"],
                    "severity": "high",
                    "code_snippet": code.splitlines()[line_number-1] if line_number <= len(code.splitlines()) else ""
                })
        
        return errors

    def _assess_bug_risks(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risks associated with identified bugs."""
        risk_levels = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for bug in bugs:
            severity = bug.get("severity", "medium")
            risk_levels[severity] = risk_levels.get(severity, 0) + 1
        
        overall_risk = "low"
        if risk_levels["critical"] > 0:
            overall_risk = "critical"
        elif risk_levels["high"] > 0:
            overall_risk = "high"
        elif risk_levels["medium"] > 2:
            overall_risk = "medium"
        
        return {
            "risk_levels": risk_levels,
            "overall_risk": overall_risk,
            "total_bugs": sum(risk_levels.values())
        }

    def _generate_bug_fix_suggestions(self, bugs: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Generate suggestions for fixing identified bugs."""
        fix_suggestions = []
        
        for bug in bugs:
            suggestion = {
                "bug_description": bug["description"],
                "line": bug["line"],
                "fix_description": "",
                "priority": bug.get("severity", "medium")
            }
            
            # Generate fix description based on bug type
            if "Assignment in if condition" in bug["description"]:
                suggestion["fix_description"] = "Change '=' (assignment) to '==' (comparison)"
            elif "Bare except clause" in bug["description"]:
                suggestion["fix_description"] = "Specify exception types to catch (e.g., 'except ValueError:')"
            elif "Potential type error" in bug["description"]:
                suggestion["fix_description"] = "Ensure all operands are of compatible types"
            elif "Empty loop body" in bug["description"]:
                suggestion["fix_description"] = "Add meaningful code to the loop body or consider removing the loop"
            elif "Empty function body" in bug["description"]:
                suggestion["fix_description"] = "Implement the function or use 'raise NotImplementedError()'"
            elif "Debug code left" in bug["description"]:
                suggestion["fix_description"] = "Remove debug statements before deployment"
            elif "Infinite loop" in bug["description"]:
                suggestion["fix_description"] = "Add a break condition or ensure the loop terminates"
            elif "Unreachable code" in bug["description"]:
                suggestion["fix_description"] = "Remove or fix the unreachable code"
            elif "Switch case without break" in bug["description"]:
                suggestion["fix_description"] = "Add break statement to prevent fall-through behavior"
            else:
                suggestion["fix_description"] = "Review and fix the issue according to best practices"
            
            fix_suggestions.append(suggestion)
        
        return fix_suggestions

    async def _analyze_python_dependencies(self, file_path: str) -> ToolResult:
        """Analyze Python dependencies from requirements.txt."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            dependencies = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse dependency specification
                    parts = re.split(r'[=<>~!]', line, 1)
                    package = parts[0].strip()
                    version_spec = line[len(package):].strip() if len(parts) > 1 else ""
                    
                    dependencies.append({
                        "package": package,
                        "version_spec": version_spec,
                        "is_pinned": "==" in version_spec,
                        "is_range": any(op in version_spec for op in [">=", "<=", ">", "<"])
                    })
            
            # Analyze dependencies
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "dependency_count": len(dependencies),
                "pinned_dependencies": sum(1 for d in dependencies if d["is_pinned"]),
                "range_dependencies": sum(1 for d in dependencies if d["is_range"]),
                "unpinned_dependencies": sum(1 for d in dependencies if not d["is_pinned"] and not d["is_range"]),
                "recommendations": []
            }
            
            # Generate recommendations
            if analysis_result["unpinned_dependencies"] > 0:
                analysis_result["recommendations"].append(
                    f"Pin {analysis_result['unpinned_dependencies']} dependencies to specific versions for reproducibility"
                )
            
            if analysis_result["dependency_count"] > 20:
                analysis_result["recommendations"].append(
                    "Consider reviewing dependencies to remove unnecessary ones"
                )
            
            return ToolResult(output=self._format_dependency_analysis(analysis_result))
            
        except FileNotFoundError:
            return ToolResult(error=f"File not found: {file_path}")
        except Exception as e:
            return ToolResult(error=f"Error analyzing dependencies: {str(e)}")

    async def _analyze_node_dependencies(self, file_path: str) -> ToolResult:
        """Analyze Node.js dependencies from package.json."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse package.json
            package_data = json.loads(content)
            
            # Extract dependencies
            dependencies = []
            for dep_type in ["dependencies", "devDependencies"]:
                if dep_type in package_data:
                    for package, version in package_data[dep_type].items():
                        dependencies.append({
                            "package": package,
                            "version_spec": version,
                            "type": dep_type,
                            "is_pinned": version.startswith("^") or version.startswith("~"),
                            "is_exact": not (version.startswith("^") or version.startswith("~") or version.startswith(">") or version.startswith("<"))
                        })
            
            # Analyze dependencies
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "dependency_count": len(dependencies),
                "runtime_dependencies": sum(1 for d in dependencies if d["type"] == "dependencies"),
                "dev_dependencies": sum(1 for d in dependencies if d["type"] == "devDependencies"),
                "pinned_dependencies": sum(1 for d in dependencies if d["is_pinned"]),
                "exact_dependencies": sum(1 for d in dependencies if d["is_exact"]),
                "recommendations": []
            }
            
            # Generate recommendations
            if analysis_result["pinned_dependencies"] > analysis_result["exact_dependencies"]:
                analysis_result["recommendations"].append(
                    "Consider using exact versions (without ^ or ~) for critical dependencies"
                )
            
            if "scripts" not in package_data:
                analysis_result["recommendations"].append(
                    "Add npm scripts for common operations"
                )
            
            if analysis_result["dependency_count"] > 30:
                analysis_result["recommendations"].append(
                    "Review dependencies to identify and remove unnecessary ones"
                )
            
            return ToolResult(output=self._format_dependency_analysis(analysis_result))
            
        except FileNotFoundError:
            return ToolResult(error=f"File not found: {file_path}")
        except json.JSONDecodeError:
            return ToolResult(error=f"Invalid JSON in package.json")
        except Exception as e:
            return ToolResult(error=f"Error analyzing dependencies: {str(e)}")

    async def _analyze_java_dependencies(self, file_path: str) -> ToolResult:
        """Analyze Java dependencies from pom.xml."""
        # Placeholder implementation
        return ToolResult(output="Java dependency analysis not fully implemented yet")

    async def _analyze_ruby_dependencies(self, file_path: str) -> ToolResult:
        """Analyze Ruby dependencies from Gemfile."""
        # Placeholder implementation
        return ToolResult(output="Ruby dependency analysis not fully implemented yet")

    def _calculate_size_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate code size metrics."""
        lines = code.splitlines()
        
        return {
            "total_lines": len(lines),
            "source_lines": len([l for l in lines if l.strip() and not l.strip().startswith(('#', '//', '/*'))]),
            "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "max_line_length": max(len(line) for line in lines) if lines else 0,
            "character_count": len(code)
        }

    def _calculate_maintainability_metrics(self, code: str, complexity_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maintainability metrics."""
        # This is a simplified implementation
        
        # Calculate Halstead metrics (placeholder values)
        halstead_metrics = {
            "program_length": len(code),
            "vocabulary_size": 0,
            "program_volume": 0,
            "difficulty": 0,
            "effort": 0
        }
        
        # Calculate maintainability index
        # MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        # where V is volume, G is cyclomatic complexity, LOC is lines of code
        
        loc = len(code.splitlines())
        volume = halstead_metrics["program_volume"] or loc * 10  # Simplified
        complexity = complexity_metrics.get("cyclomatic_complexity", 1)
        
        import math
        maintainability_index = 171 - 5.2 * math.log(volume + 1) - 0.23 * complexity - 16.2 * math.log(loc + 1)
        maintainability_index = max(0, min(100, maintainability_index)) / 100
        
        return {
            "maintainability_index": maintainability_index,
            "comment_ratio": len([l for l in code.splitlines() if l.strip().startswith(('#', '//', '/*'))]) / loc if loc > 0 else 0,
            "average_function_length": loc / complexity_metrics.get("function_count", 1) if complexity_metrics.get("function_count", 0) > 0 else loc,
            "halstead_metrics": halstead_metrics
        }

    def _calculate_quality_score(
        self, 
        complexity_metrics: Dict[str, Any],
        maintainability_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall code quality score."""
        # This is a simplified implementation
        
        # Factors and their weights
        factors = {
            "cyclomatic_complexity": complexity_metrics.get("cyclomatic_complexity", 1),
            "max_nesting_depth": complexity_metrics.get("max_nesting_depth", 1),
            "maintainability_index": maintainability_metrics.get("maintainability_index", 0.5),
            "comment_ratio": maintainability_metrics.get("comment_ratio", 0)
        }
        
        weights = {
            "cyclomatic_complexity": -0.3,  # Higher complexity reduces score
            "max_nesting_depth": -0.2,      # Higher nesting reduces score
            "maintainability_index": 0.4,   # Higher maintainability increases score
            "comment_ratio": 0.1            # Higher comment ratio increases score
        }
        
        # Normalize factors
        normalized_factors = {
            "cyclomatic_complexity": 1 / (1 + factors["cyclomatic_complexity"] / 10),
            "max_nesting_depth": 1 / (1 + factors["max_nesting_depth"] / 5),
            "maintainability_index": factors["maintainability_index"],
            "comment_ratio": min(factors["comment_ratio"] * 5, 1)  # Cap at 1.0
        }
        
        # Calculate weighted score
        score = sum(normalized_factors[factor] * weights[factor] for factor in normalized_factors)
        
        # Normalize to 0-1 range
        score = (score + 1) / 2
        
        # Clamp to 0-1 range
        return max(0, min(1, score))

    def _compare_with_benchmarks(self, metrics: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Compare metrics with language-specific benchmarks."""
        # This is a placeholder implementation
        # In a real tool, this would compare with actual benchmarks
        
        benchmarks = {
            "python": {
                "cyclomatic_complexity": 10,
                "max_nesting_depth": 4,
                "maintainability_index": 0.7,
                "comment_ratio": 0.2
            },
            "javascript": {
                "cyclomatic_complexity": 12,
                "max_nesting_depth": 3,
                "maintainability_index": 0.65,
                "comment_ratio": 0.15
            }
        }
        
        language_benchmarks = benchmarks.get(language, benchmarks["python"])
        
        comparison = {}
        
        # Compare complexity metrics
        complexity = metrics["complexity_metrics"].get("cyclomatic_complexity", 0)
        benchmark_complexity = language_benchmarks["cyclomatic_complexity"]
        comparison["complexity"] = {
            "value": complexity,
            "benchmark": benchmark_complexity,
            "status": "good" if complexity <= benchmark_complexity else "needs_improvement"
        }
        
        # Compare nesting depth
        nesting = metrics["complexity_metrics"].get("max_nesting_depth", 0)
        benchmark_nesting = language_benchmarks["max_nesting_depth"]
        comparison["nesting"] = {
            "value": nesting,
            "benchmark": benchmark_nesting,
            "status": "good" if nesting <= benchmark_nesting else "needs_improvement"
        }
        
        # Compare maintainability
        maintainability = metrics["maintainability_metrics"].get("maintainability_index", 0)
        benchmark_maintainability = language_benchmarks["maintainability_index"]
        comparison["maintainability"] = {
            "value": maintainability,
            "benchmark": benchmark_maintainability,
            "status": "good" if maintainability >= benchmark_maintainability else "needs_improvement"
        }
        
        # Overall comparison
        comparison["overall"] = {
            "status": "good" if comparison["complexity"]["status"] == "good" and 
                              comparison["maintainability"]["status"] == "good" else "needs_improvement"
        }
        
        return comparison

    def _generate_security_recommendations(self, vulnerabilities: List[Dict[str, Any]], language: str) -> List[str]:
        """Generate security recommendations based on identified vulnerabilities."""
        recommendations = []
        
        if not vulnerabilities:
            recommendations.append("No security vulnerabilities detected")
            return recommendations
        
        # General recommendations
        recommendations.append("Address all identified security vulnerabilities as a priority")
        
        # Specific recommendations for vulnerabilities
        for vulnerability in vulnerabilities[:3]:  # Top 3 vulnerabilities
            recommendations.append(
                f"Fix security issue: {vulnerability['description']} (line {vulnerability['line']})"
            )
        
        # Language-specific security recommendations
        if language == "python":
            if any("eval" in v["description"] for v in vulnerabilities):
                recommendations.append("Avoid using eval() - use safer alternatives like ast.literal_eval()")
            if any("subprocess" in v["description"] for v in vulnerabilities):
                recommendations.append("Use subprocess.run() with shell=False and input validation")
            if any("pickle" in v["description"] for v in vulnerabilities):
                recommendations.append("Avoid pickle for untrusted data - use JSON or other secure serialization")
        
        elif language in ["javascript", "typescript"]:
            if any("eval" in v["description"] for v in vulnerabilities):
                recommendations.append("Avoid using eval() - use safer alternatives")
            if any("innerHTML" in v["description"] for v in vulnerabilities):
                recommendations.append("Use textContent instead of innerHTML when possible to prevent XSS")
            if any("localStorage" in v["description"] for v in vulnerabilities):
                recommendations.append("Don't store sensitive data in localStorage - use secure cookies or server storage")
        
        return recommendations

    def _determine_style_convention(self, description: str, language: str) -> str:
        """Determine which style convention is violated."""
        if language == "python":
            if "whitespace" in description.lower() or "indentation" in description.lower():
                return "PEP 8 Whitespace"
            elif "naming" in description.lower():
                return "PEP 8 Naming"
            elif "import" in description.lower():
                return "PEP 8 Imports"
            else:
                return "PEP 8 General"
        
        elif language in ["javascript", "typescript"]:
            if "var" in description.lower():
                return "ES6 Variables"
            elif "==" in description.lower():
                return "Strict Equality"
            elif "semicolon" in description.lower():
                return "Semicolon Usage"
            else:
                return "ESLint General"
        
        return "General Style Convention"

    # Formatting methods for output

    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format analysis result for output."""
        output = ["Code Analysis Results"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        output.append(f"Analysis Depth: {result['analysis_depth']}")
        
        if result["code_metrics"]:
            output.append("\nCode Metrics:")
            for metric, value in result["code_metrics"].items():
                if isinstance(value, float):
                    output.append(f"  {metric}: {value:.2f}")
                else:
                    output.append(f"  {metric}: {value}")
        
        # Count issues by category
        issue_counts = {category: len(issues) for category, issues in result["issues"].items()}
        
        output.append("\nIssues Summary:")
        for category, count in issue_counts.items():
            output.append(f"  {category.title()}: {count} issues")
        
        # Show top issues from each category
        for category, issues in result["issues"].items():
            if issues:
                output.append(f"\nTop {category.title()} Issues:")
                for issue in issues[:3]:  # Show top 3 issues per category
                    output.append(f"  Line {issue['line']}: {issue['description']}")
        
        if result["recommendations"]:
            output.append("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                output.append(f"  {i}. {rec}")
        
        return "\n".join(output)

    def _format_security_result(self, result: Dict[str, Any]) -> str:
        """Format security analysis result for output."""
        output = ["Security Analysis Results"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        output.append(f"Security Score: {result['security_score']:.2f}/1.00")
        
        if result["vulnerabilities"]:
            output.append(f"\nVulnerabilities Found: {len(result['vulnerabilities'])}")
            output.append("\nDetails:")
            for vuln in result["vulnerabilities"]:
                output.append(f"  Line {vuln['line']} ({vuln['severity']}): {vuln['description']}")
                if vuln.get("code_snippet"):
                    output.append(f"    Code: {vuln['code_snippet']}")
        else:
            output.append("\nNo vulnerabilities detected!")
        
        if result["critical_issues"]:
            output.append(f"\nCritical Issues: {len(result['critical_issues'])}")
            for issue in result["critical_issues"]:
                output.append(f"  Line {issue['line']}: {issue['description']}")
        
        if result["recommendations"]:
            output.append("\nSecurity Recommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                output.append(f"  {i}. {rec}")
        
        return "\n".join(output)

    def _format_complexity_result(self, result: Dict[str, Any]) -> str:
        """Format complexity analysis result for output."""
        output = ["Code Complexity Analysis"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        output.append(f"Maintainability Index: {result['maintainability_index']:.2f}/1.00")
        
        if result["complexity_metrics"]:
            output.append("\nComplexity Metrics:")
            for metric, value in result["complexity_metrics"].items():
                if isinstance(value, float):
                    output.append(f"  {metric}: {value:.2f}")
                else:
                    output.append(f"  {metric}: {value}")
        
        if result["hotspots"]:
            output.append(f"\nComplexity Hotspots: {len(result['hotspots'])}")
            for hotspot in result["hotspots"]:
                output.append(f"  {hotspot['type'].title()} '{hotspot['name']}' (lines {hotspot['start_line']}-{hotspot['end_line']})")
                output.append(f"    Complexity: {hotspot['complexity']} ({hotspot['severity']} severity)")
        else:
            output.append("\nNo complexity hotspots detected!")
        
        if result["recommendations"]:
            output.append("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                output.append(f"  {i}. {rec}")
        
        return "\n".join(output)

    def _format_improvements_result(self, result: Dict[str, Any]) -> str:
        """Format improvement suggestions for output."""
        output = ["Code Improvement Suggestions"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        output.append(f"Focus Areas: {', '.join(result['focus_areas'])}")
        
        if result["suggestions"]:
            output.append(f"\nTotal Suggestions: {len(result['suggestions'])}")
            
            # Group suggestions by area
            by_area = {}
            for suggestion in result["suggestions"]:
                area = suggestion["area"]
                if area not in by_area:
                    by_area[area] = []
                by_area[area].append(suggestion)
            
            # Show suggestions by area
            for area, suggestions in by_area.items():
                output.append(f"\n{area.title()} Improvements:")
                for i, suggestion in enumerate(suggestions[:3], 1):  # Top 3 per area
                    priority = suggestion.get("priority", "medium").upper()
                    output.append(f"  {i}. [{priority}] {suggestion['description']}")
        else:
            output.append("\nNo improvement suggestions found!")
        
        if result["priority_improvements"]:
            output.append("\nPriority Improvements:")
            for i, improvement in enumerate(result["priority_improvements"], 1):
                output.append(f"  {i}. {improvement['description']}")
        
        if result["code_samples"]:
            output.append("\nSample Improvements:")
            for sample_name, sample_code in result["code_samples"].items():
                output.append(f"\n  {sample_name.replace('_', ' ').title()}:")
                output.append(f"    {sample_code}")
        
        return "\n".join(output)

    def _format_style_result(self, result: Dict[str, Any]) -> str:
        """Format style check result for output."""
        output = ["Code Style Analysis"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        output.append(f"Style Score: {result['style_score']:.2f}/1.00")
        
        if result["style_issues"]:
            output.append(f"\nStyle Issues: {len(result['style_issues'])}")
            
            # Group issues by convention
            if result["convention_violations"]:
                output.append("\nConvention Violations:")
                for convention, count in result["convention_violations"].items():
                    output.append(f"  {convention}: {count} issues")
            
            output.append("\nIssue Details:")
            for i, issue in enumerate(result["style_issues"][:5], 1):  # Top 5 issues
                output.append(f"  {i}. Line {issue['line']}: {issue['description']}")
                if issue.get("code_snippet"):
                    output.append(f"     Code: {issue['code_snippet']}")
        else:
            output.append("\nNo style issues detected!")
        
        if result["auto_fixable"]:
            output.append(f"\nAuto-fixable Issues: {len(result['auto_fixable'])}")
            output.append("  These issues can be automatically fixed with a linter or formatter")
        
        return "\n".join(output)

    def _format_bugs_result(self, result: Dict[str, Any]) -> str:
        """Format bug detection result for output."""
        output = ["Bug Detection Results"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        
        # Show risk assessment
        if result["risk_assessment"]:
            output.append(f"\nRisk Assessment:")
            output.append(f"  Overall Risk: {result['risk_assessment'].get('overall_risk', 'unknown').upper()}")
            output.append(f"  Total Bugs: {result['risk_assessment'].get('total_bugs', 0)}")
            
            if "risk_levels" in result["risk_assessment"]:
                output.append("  Risk Breakdown:")
                for level, count in result["risk_assessment"]["risk_levels"].items():
                    if count > 0:
                        output.append(f"    {level.title()}: {count}")
        
        # Show potential bugs
        if result["potential_bugs"]:
            output.append(f"\nPotential Bugs: {len(result['potential_bugs'])}")
            for i, bug in enumerate(result["potential_bugs"], 1):
                output.append(f"  {i}. Line {bug['line']} ({bug.get('severity', 'medium')}): {bug['description']}")
                if bug.get("code_snippet"):
                    output.append(f"     Code: {bug['code_snippet']}")
        
        # Show logical errors
        if result["logical_errors"]:
            output.append(f"\nLogical Errors: {len(result['logical_errors'])}")
            for i, error in enumerate(result["logical_errors"], 1):
                output.append(f"  {i}. Line {error['line']} ({error.get('severity', 'medium')}): {error['description']}")
                if error.get("code_snippet"):
                    output.append(f"     Code: {error['code_snippet']}")
        
        # Show fix suggestions
        if result["fix_suggestions"]:
            output.append("\nFix Suggestions:")
            for i, suggestion in enumerate(result["fix_suggestions"], 1):
                output.append(f"  {i}. Line {suggestion['line']} - {suggestion['bug_description']}")
                output.append(f"     Fix: {suggestion['fix_description']}")
        
        return "\n".join(output)

    def _format_dependency_analysis(self, result: Dict[str, Any]) -> str:
        """Format dependency analysis result for output."""
        output = ["Dependency Analysis Results"]
        output.append("=" * 40)
        
        output.append(f"\nFile: {result['file_path']}")
        output.append(f"Timestamp: {result['timestamp']}")
        
        output.append(f"\nDependency Count: {result['dependency_count']}")
        
        # Show dependency breakdown
        if "runtime_dependencies" in result:
            output.append("Dependency Breakdown:")
            output.append(f"  Runtime Dependencies: {result['runtime_dependencies']}")
            output.append(f"  Dev Dependencies: {result['dev_dependencies']}")
        
        if "pinned_dependencies" in result:
            output.append("\nVersion Specifications:")
            output.append(f"  Pinned Dependencies: {result['pinned_dependencies']}")
            if "range_dependencies" in result:
                output.append(f"  Range Dependencies: {result['range_dependencies']}")
            if "unpinned_dependencies" in result:
                output.append(f"  Unpinned Dependencies: {result['unpinned_dependencies']}")
            if "exact_dependencies" in result:
                output.append(f"  Exact Dependencies: {result['exact_dependencies']}")
        
        # Show recommendations
        if result["recommendations"]:
            output.append("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                output.append(f"  {i}. {rec}")
        
        return "\n".join(output)

    def _format_coverage_result(self, result: Dict[str, Any]) -> str:
        """Format test coverage result for output."""
        output = ["Test Coverage Analysis"]
        output.append("=" * 40)
        
        output.append(f"\nTarget: {result['target']}")
        output.append(f"Timestamp: {result['timestamp']}")
        
        if result["coverage_metrics"]:
            output.append("\nCoverage Metrics:")
            for metric, value in result["coverage_metrics"].items():
                output.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
        
        if result["uncovered_areas"]:
            output.append("\nUncovered Areas:")
            for area in result["uncovered_areas"]:
                output.append(f"  {area['file']} (lines {area['lines']}): {area['description']}")
        
        if result["test_quality_assessment"]:
            output.append("\nTest Quality Assessment:")
            for metric, value in result["test_quality_assessment"].items():
                if isinstance(value, float):
                    output.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                else:
                    output.append(f"  {metric.replace('_', ' ').title()}: {value}")
        
        if result["recommendations"]:
            output.append("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                output.append(f"  {i}. {rec}")
        
        return "\n".join(output)

    def _format_metrics_result(self, result: Dict[str, Any]) -> str:
        """Format code metrics result for output."""
        output = ["Code Metrics Analysis"]
        output.append("=" * 40)
        
        output.append(f"\nLanguage: {result['language']}")
        output.append(f"Timestamp: {result['timestamp']}")
        output.append(f"Quality Score: {result['quality_score']:.2f}/1.00")
        
        if result["size_metrics"]:
            output.append("\nSize Metrics:")
            for metric, value in result["size_metrics"].items():
                if isinstance(value, float):
                    output.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                else:
                    output.append(f"  {metric.replace('_', ' ').title()}: {value}")
        
        if result["complexity_metrics"]:
            output.append("\nComplexity Metrics:")
            for metric, value in result["complexity_metrics"].items():
                if isinstance(value, float):
                    output.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                else:
                    output.append(f"  {metric.replace('_', ' ').title()}: {value}")
        
        if result["maintainability_metrics"]:
            output.append("\nMaintainability Metrics:")
            for metric, value in result["maintainability_metrics"].items():
                if metric == "halstead_metrics":
                    continue  # Skip nested metrics
                if isinstance(value, float):
                    output.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                else:
                    output.append(f"  {metric.replace('_', ' ').title()}: {value}")
        
        if result["benchmark_comparison"]:
            output.append("\nBenchmark Comparison:")
            for metric, comparison in result["benchmark_comparison"].items():
                if metric == "overall":
                    output.append(f"  Overall Status: {comparison['status'].replace('_', ' ').title()}")
                elif isinstance(comparison, dict):
                    status_emoji = "✅" if comparison.get("status") == "good" else "⚠️"
                    output.append(f"  {metric.title()}: {comparison.get('value')} vs benchmark {comparison.get('benchmark')} {status_emoji}")
        
        return "\n".join(output)