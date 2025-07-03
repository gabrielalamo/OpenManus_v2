"""
Security Auditor Tool
Performs security analysis, vulnerability detection, and compliance checks
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class SecurityAuditorTool(BaseTool):
    """
    Advanced security auditing tool for detecting vulnerabilities, performing
    security analysis, and ensuring compliance with security standards.
    """

    name: str = "security_auditor"
    description: str = """
    Perform security audits, vulnerability assessments, and compliance checks.
    Identify security risks, suggest mitigations, and enforce security best practices.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "scan_code", "check_dependencies", "analyze_configuration", 
                    "check_compliance", "identify_vulnerabilities", "suggest_mitigations",
                    "generate_security_report", "audit_permissions", "check_secrets"
                ],
                "description": "The security audit action to perform"
            },
            "target_path": {
                "type": "string",
                "description": "Path to the code, configuration, or project to audit"
            },
            "language": {
                "type": "string",
                "enum": ["python", "javascript", "typescript", "java", "go", "ruby", "php", "csharp", "other"],
                "description": "Programming language of the target code"
            },
            "framework": {
                "type": "string",
                "description": "Framework used in the project (e.g., Django, React, Spring)"
            },
            "compliance_standard": {
                "type": "string",
                "enum": ["owasp", "pci-dss", "hipaa", "gdpr", "nist", "iso27001", "custom"],
                "description": "Compliance standard to check against"
            },
            "severity_threshold": {
                "type": "string",
                "enum": ["info", "low", "medium", "high", "critical"],
                "description": "Minimum severity level to report"
            },
            "scan_depth": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "description": "Depth of the security scan"
            },
            "include_dependencies": {
                "type": "boolean",
                "description": "Whether to include dependencies in the scan"
            },
            "custom_rules": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Custom security rules to apply"
            }
        },
        "required": ["action"]
    }

    # Security data storage
    vulnerability_patterns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    compliance_requirements: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    security_best_practices: Dict[str, List[str]] = Field(default_factory=dict)
    audit_history: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_security_data()

    def _initialize_security_data(self):
        """Initialize security patterns, compliance requirements, and best practices."""
        
        # Initialize vulnerability patterns for different languages
        self.vulnerability_patterns = {
            "python": {
                "sql_injection": {
                    "pattern": r"execute\([\"'].*?\%.*?[\"']\)|cursor\.execute\([\"'].*?\%.*?[\"']\)",
                    "description": "Potential SQL injection vulnerability",
                    "severity": "high",
                    "mitigation": "Use parameterized queries or ORM"
                },
                "command_injection": {
                    "pattern": r"os\.system\(|subprocess\.call\(|subprocess\.Popen\(|eval\(|exec\(",
                    "description": "Potential command injection vulnerability",
                    "severity": "high",
                    "mitigation": "Avoid using shell=True and sanitize inputs"
                },
                "insecure_deserialization": {
                    "pattern": r"pickle\.loads\(|yaml\.load\((?!.*Loader=yaml\.SafeLoader)",
                    "description": "Insecure deserialization",
                    "severity": "high",
                    "mitigation": "Use safe loaders and avoid pickle with untrusted data"
                },
                "hardcoded_secrets": {
                    "pattern": r"password\s*=\s*['\"][^'\"]+['\"]|api_key\s*=\s*['\"][^'\"]+['\"]|secret\s*=\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded secrets",
                    "severity": "high",
                    "mitigation": "Use environment variables or secure secret management"
                },
                "insecure_hash": {
                    "pattern": r"hashlib\.md5\(|hashlib\.sha1\(",
                    "description": "Use of weak hash algorithms",
                    "severity": "medium",
                    "mitigation": "Use strong hash algorithms like SHA-256 or better"
                }
            },
            "javascript": {
                "xss": {
                    "pattern": r"innerHTML\s*=|document\.write\(|eval\(",
                    "description": "Potential XSS vulnerability",
                    "severity": "high",
                    "mitigation": "Use textContent instead of innerHTML, avoid eval"
                },
                "sql_injection": {
                    "pattern": r"execute\([\"'].*?\$\{.*?[\"']\)|query\([\"'].*?\$\{.*?[\"']\)",
                    "description": "Potential SQL injection vulnerability",
                    "severity": "high",
                    "mitigation": "Use parameterized queries or ORM"
                },
                "insecure_randomness": {
                    "pattern": r"Math\.random\(",
                    "description": "Use of insecure random number generator",
                    "severity": "medium",
                    "mitigation": "Use crypto.getRandomValues() for security purposes"
                },
                "hardcoded_secrets": {
                    "pattern": r"password\s*=\s*['\"][^'\"]+['\"]|apiKey\s*=\s*['\"][^'\"]+['\"]|secret\s*=\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded secrets",
                    "severity": "high",
                    "mitigation": "Use environment variables or secure secret management"
                },
                "insecure_cookie": {
                    "pattern": r"document\.cookie\s*=",
                    "description": "Potentially insecure cookie usage",
                    "severity": "medium",
                    "mitigation": "Set secure and httpOnly flags on cookies"
                }
            }
        }
        
        # Initialize compliance requirements
        self.compliance_requirements = {
            "owasp": [
                {"id": "A01:2021", "name": "Broken Access Control", "description": "Restrictions on authenticated users are not properly enforced"},
                {"id": "A02:2021", "name": "Cryptographic Failures", "description": "Failures related to cryptography that often lead to sensitive data exposure"},
                {"id": "A03:2021", "name": "Injection", "description": "User-supplied data is not validated, filtered, or sanitized by the application"},
                {"id": "A04:2021", "name": "Insecure Design", "description": "Flaws in design and architecture"},
                {"id": "A05:2021", "name": "Security Misconfiguration", "description": "Improper implementation of controls intended to keep application data safe"},
                {"id": "A06:2021", "name": "Vulnerable and Outdated Components", "description": "Using components with known vulnerabilities"},
                {"id": "A07:2021", "name": "Identification and Authentication Failures", "description": "Authentication-related flaws"},
                {"id": "A08:2021", "name": "Software and Data Integrity Failures", "description": "Software and data integrity failures relate to code and infrastructure"},
                {"id": "A09:2021", "name": "Security Logging and Monitoring Failures", "description": "Insufficient logging and monitoring"},
                {"id": "A10:2021", "name": "Server-Side Request Forgery", "description": "SSRF flaws occur when a web application fetches a remote resource"}
            ],
            "pci-dss": [
                {"id": "Req 1", "name": "Firewall Configuration", "description": "Install and maintain a firewall configuration to protect cardholder data"},
                {"id": "Req 2", "name": "Default Settings", "description": "Do not use vendor-supplied defaults for system passwords and other security parameters"},
                {"id": "Req 3", "name": "Stored Cardholder Data", "description": "Protect stored cardholder data"},
                {"id": "Req 4", "name": "Data Encryption", "description": "Encrypt transmission of cardholder data across open, public networks"},
                {"id": "Req 5", "name": "Anti-Virus", "description": "Use and regularly update anti-virus software or programs"},
                {"id": "Req 6", "name": "Secure Systems", "description": "Develop and maintain secure systems and applications"},
                {"id": "Req 7", "name": "Access Restriction", "description": "Restrict access to cardholder data by business need to know"},
                {"id": "Req 8", "name": "Authentication", "description": "Identify and authenticate access to system components"},
                {"id": "Req 9", "name": "Physical Access", "description": "Restrict physical access to cardholder data"},
                {"id": "Req 10", "name": "Monitoring", "description": "Track and monitor all access to network resources and cardholder data"},
                {"id": "Req 11", "name": "Security Testing", "description": "Regularly test security systems and processes"},
                {"id": "Req 12", "name": "Security Policy", "description": "Maintain a policy that addresses information security for all personnel"}
            ]
        }
        
        # Initialize security best practices
        self.security_best_practices = {
            "python": [
                "Use parameterized queries for database operations",
                "Validate and sanitize all user inputs",
                "Use secure password hashing (bcrypt, Argon2)",
                "Implement proper session management",
                "Use HTTPS for all communications",
                "Implement proper error handling without exposing sensitive information",
                "Keep dependencies updated",
                "Use environment variables for secrets",
                "Implement proper access controls",
                "Enable security headers"
            ],
            "javascript": [
                "Use Content Security Policy (CSP)",
                "Sanitize user inputs to prevent XSS",
                "Use HTTPS for all communications",
                "Implement proper authentication and authorization",
                "Use secure cookie flags (HttpOnly, Secure, SameSite)",
                "Keep dependencies updated",
                "Use environment variables for secrets",
                "Validate data on both client and server sides",
                "Implement proper error handling",
                "Use strict mode ('use strict')"
            ],
            "general": [
                "Follow the principle of least privilege",
                "Implement defense in depth",
                "Conduct regular security audits",
                "Keep all software and dependencies updated",
                "Implement proper logging and monitoring",
                "Have an incident response plan",
                "Conduct security training for team members",
                "Perform regular backups",
                "Use strong encryption for sensitive data",
                "Implement proper access controls"
            ]
        }

    async def execute(
        self,
        action: str,
        target_path: Optional[str] = None,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        compliance_standard: Optional[str] = None,
        severity_threshold: str = "medium",
        scan_depth: str = "standard",
        include_dependencies: bool = False,
        custom_rules: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the security audit action."""
        
        try:
            if action == "scan_code":
                return await self._scan_code(target_path, language, severity_threshold, scan_depth)
            elif action == "check_dependencies":
                return await self._check_dependencies(target_path, language, severity_threshold)
            elif action == "analyze_configuration":
                return await self._analyze_configuration(target_path, framework)
            elif action == "check_compliance":
                return await self._check_compliance(target_path, compliance_standard, severity_threshold)
            elif action == "identify_vulnerabilities":
                return await self._identify_vulnerabilities(target_path, language, severity_threshold, scan_depth)
            elif action == "suggest_mitigations":
                return await self._suggest_mitigations(target_path, language)
            elif action == "generate_security_report":
                return await self._generate_security_report(target_path, language, compliance_standard, severity_threshold)
            elif action == "audit_permissions":
                return await self._audit_permissions(target_path)
            elif action == "check_secrets":
                return await self._check_secrets(target_path, severity_threshold)
            else:
                return ToolResult(error=f"Unknown security audit action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Security audit error: {str(e)}")

    async def _scan_code(
        self, 
        target_path: Optional[str],
        language: Optional[str],
        severity_threshold: str,
        scan_depth: str
    ) -> ToolResult:
        """Scan code for security vulnerabilities."""
        
        if not target_path:
            return ToolResult(error="Target path is required for code scanning")
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(target_path)
        
        # Prepare scan results
        scan_results = {
            "target_path": target_path,
            "language": language,
            "scan_depth": scan_depth,
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": [],
            "security_score": 0.0,
            "risk_level": "unknown",
            "recommendations": []
        }
        
        # Get vulnerability patterns for the language
        patterns = self.vulnerability_patterns.get(language, {})
        if not patterns:
            return ToolResult(error=f"No vulnerability patterns available for language: {language}")
        
        # Simulate code scanning
        # In a real implementation, this would analyze actual code files
        vulnerabilities = self._simulate_code_scan(target_path, language, patterns, severity_threshold, scan_depth)
        scan_results["vulnerabilities"] = vulnerabilities
        
        # Calculate security score (0.0 to 1.0)
        if vulnerabilities:
            severity_weights = {
                "critical": 1.0,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.1,
                "info": 0.0
            }
            
            total_weight = sum(severity_weights[v["severity"]] for v in vulnerabilities)
            max_possible_weight = len(vulnerabilities)  # If all were critical
            
            # Higher score is better (1.0 = no vulnerabilities)
            scan_results["security_score"] = 1.0 - (total_weight / max_possible_weight if max_possible_weight > 0 else 0)
        else:
            scan_results["security_score"] = 1.0
        
        # Determine risk level
        if scan_results["security_score"] >= 0.9:
            scan_results["risk_level"] = "low"
        elif scan_results["security_score"] >= 0.7:
            scan_results["risk_level"] = "medium"
        else:
            scan_results["risk_level"] = "high"
        
        # Generate recommendations
        scan_results["recommendations"] = self._generate_security_recommendations(vulnerabilities, language)
        
        # Store scan in audit history
        self.audit_history.append({
            "type": "code_scan",
            "target": target_path,
            "timestamp": scan_results["timestamp"],
            "security_score": scan_results["security_score"],
            "vulnerability_count": len(vulnerabilities)
        })
        
        return ToolResult(output=self._format_scan_results(scan_results))

    async def _check_dependencies(
        self,
        target_path: Optional[str],
        language: Optional[str],
        severity_threshold: str
    ) -> ToolResult:
        """Check dependencies for known vulnerabilities."""
        
        if not target_path:
            return ToolResult(error="Target path is required for dependency checking")
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(target_path)
        
        # Prepare dependency check results
        dependency_results = {
            "target_path": target_path,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "dependencies": [],
            "vulnerable_dependencies": [],
            "outdated_dependencies": [],
            "recommendations": []
        }
        
        # Simulate dependency checking
        # In a real implementation, this would parse package files and check against vulnerability databases
        dependencies = self._simulate_dependency_check(target_path, language, severity_threshold)
        dependency_results["dependencies"] = dependencies["all"]
        dependency_results["vulnerable_dependencies"] = dependencies["vulnerable"]
        dependency_results["outdated_dependencies"] = dependencies["outdated"]
        
        # Generate recommendations
        dependency_results["recommendations"] = self._generate_dependency_recommendations(
            dependencies["vulnerable"], dependencies["outdated"], language
        )
        
        # Store check in audit history
        self.audit_history.append({
            "type": "dependency_check",
            "target": target_path,
            "timestamp": dependency_results["timestamp"],
            "vulnerable_count": len(dependencies["vulnerable"]),
            "outdated_count": len(dependencies["outdated"])
        })
        
        return ToolResult(output=self._format_dependency_results(dependency_results))

    async def _analyze_configuration(
        self,
        target_path: Optional[str],
        framework: Optional[str]
    ) -> ToolResult:
        """Analyze configuration files for security issues."""
        
        if not target_path:
            return ToolResult(error="Target path is required for configuration analysis")
        
        # Prepare configuration analysis results
        config_results = {
            "target_path": target_path,
            "framework": framework,
            "timestamp": datetime.now().isoformat(),
            "config_files": [],
            "misconfigurations": [],
            "security_headers": {},
            "recommendations": []
        }
        
        # Simulate configuration analysis
        # In a real implementation, this would analyze actual configuration files
        config_analysis = self._simulate_config_analysis(target_path, framework)
        config_results["config_files"] = config_analysis["files"]
        config_results["misconfigurations"] = config_analysis["misconfigurations"]
        config_results["security_headers"] = config_analysis["security_headers"]
        
        # Generate recommendations
        config_results["recommendations"] = self._generate_configuration_recommendations(
            config_analysis["misconfigurations"], framework
        )
        
        # Store analysis in audit history
        self.audit_history.append({
            "type": "config_analysis",
            "target": target_path,
            "timestamp": config_results["timestamp"],
            "misconfiguration_count": len(config_analysis["misconfigurations"])
        })
        
        return ToolResult(output=self._format_configuration_results(config_results))

    async def _check_compliance(
        self,
        target_path: Optional[str],
        compliance_standard: Optional[str],
        severity_threshold: str
    ) -> ToolResult:
        """Check compliance with security standards."""
        
        if not target_path:
            return ToolResult(error="Target path is required for compliance checking")
        
        if not compliance_standard:
            return ToolResult(error="Compliance standard is required")
        
        if compliance_standard not in self.compliance_requirements:
            return ToolResult(error=f"Unsupported compliance standard: {compliance_standard}")
        
        # Prepare compliance check results
        compliance_results = {
            "target_path": target_path,
            "standard": compliance_standard,
            "timestamp": datetime.now().isoformat(),
            "requirements": [],
            "compliant_requirements": [],
            "non_compliant_requirements": [],
            "compliance_score": 0.0,
            "recommendations": []
        }
        
        # Get compliance requirements
        requirements = self.compliance_requirements.get(compliance_standard, [])
        compliance_results["requirements"] = requirements
        
        # Simulate compliance checking
        # In a real implementation, this would perform actual compliance checks
        compliance_check = self._simulate_compliance_check(target_path, requirements, severity_threshold)
        compliance_results["compliant_requirements"] = compliance_check["compliant"]
        compliance_results["non_compliant_requirements"] = compliance_check["non_compliant"]
        
        # Calculate compliance score
        total_requirements = len(requirements)
        compliant_count = len(compliance_check["compliant"])
        compliance_results["compliance_score"] = compliant_count / total_requirements if total_requirements > 0 else 0.0
        
        # Generate recommendations
        compliance_results["recommendations"] = self._generate_compliance_recommendations(
            compliance_check["non_compliant"], compliance_standard
        )
        
        # Store check in audit history
        self.audit_history.append({
            "type": "compliance_check",
            "target": target_path,
            "standard": compliance_standard,
            "timestamp": compliance_results["timestamp"],
            "compliance_score": compliance_results["compliance_score"]
        })
        
        return ToolResult(output=self._format_compliance_results(compliance_results))

    async def _identify_vulnerabilities(
        self,
        target_path: Optional[str],
        language: Optional[str],
        severity_threshold: str,
        scan_depth: str
    ) -> ToolResult:
        """Identify security vulnerabilities in code and infrastructure."""
        
        if not target_path:
            return ToolResult(error="Target path is required for vulnerability identification")
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(target_path)
        
        # Prepare vulnerability results
        vulnerability_results = {
            "target_path": target_path,
            "language": language,
            "scan_depth": scan_depth,
            "timestamp": datetime.now().isoformat(),
            "code_vulnerabilities": [],
            "infrastructure_vulnerabilities": [],
            "overall_risk_level": "unknown",
            "recommendations": []
        }
        
        # Scan code for vulnerabilities
        code_scan_result = await self._scan_code(target_path, language, severity_threshold, scan_depth)
        if hasattr(code_scan_result, "output"):
            # Parse the output to extract vulnerabilities
            # In a real implementation, this would be structured data
            vulnerability_results["code_vulnerabilities"] = self._extract_vulnerabilities_from_output(code_scan_result.output)
        
        # Simulate infrastructure vulnerability scan
        # In a real implementation, this would scan actual infrastructure
        infra_vulnerabilities = self._simulate_infrastructure_scan(target_path, severity_threshold)
        vulnerability_results["infrastructure_vulnerabilities"] = infra_vulnerabilities
        
        # Determine overall risk level
        all_vulnerabilities = (
            vulnerability_results["code_vulnerabilities"] + 
            vulnerability_results["infrastructure_vulnerabilities"]
        )
        
        if any(v["severity"] == "critical" for v in all_vulnerabilities):
            vulnerability_results["overall_risk_level"] = "critical"
        elif any(v["severity"] == "high" for v in all_vulnerabilities):
            vulnerability_results["overall_risk_level"] = "high"
        elif any(v["severity"] == "medium" for v in all_vulnerabilities):
            vulnerability_results["overall_risk_level"] = "medium"
        elif any(v["severity"] == "low" for v in all_vulnerabilities):
            vulnerability_results["overall_risk_level"] = "low"
        else:
            vulnerability_results["overall_risk_level"] = "info"
        
        # Generate recommendations
        vulnerability_results["recommendations"] = self._generate_vulnerability_recommendations(all_vulnerabilities, language)
        
        # Store scan in audit history
        self.audit_history.append({
            "type": "vulnerability_scan",
            "target": target_path,
            "timestamp": vulnerability_results["timestamp"],
            "risk_level": vulnerability_results["overall_risk_level"],
            "vulnerability_count": len(all_vulnerabilities)
        })
        
        return ToolResult(output=self._format_vulnerability_results(vulnerability_results))

    async def _suggest_mitigations(
        self,
        target_path: Optional[str],
        language: Optional[str]
    ) -> ToolResult:
        """Suggest mitigations for identified security issues."""
        
        if not target_path:
            return ToolResult(error="Target path is required for suggesting mitigations")
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(target_path)
        
        # First identify vulnerabilities
        vulnerability_result = await self._identify_vulnerabilities(target_path, language, "low", "standard")
        
        # Prepare mitigation results
        mitigation_results = {
            "target_path": target_path,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": [],
            "mitigations": [],
            "best_practices": []
        }
        
        # Extract vulnerabilities from the result
        if hasattr(vulnerability_result, "output"):
            vulnerabilities = self._extract_vulnerabilities_from_output(vulnerability_result.output)
            mitigation_results["vulnerabilities"] = vulnerabilities
            
            # Generate mitigations for each vulnerability
            for vulnerability in vulnerabilities:
                mitigation = {
                    "vulnerability": vulnerability["description"],
                    "severity": vulnerability["severity"],
                    "mitigation_steps": self._get_mitigation_steps(vulnerability, language),
                    "code_example": self._get_mitigation_code_example(vulnerability, language)
                }
                mitigation_results["mitigations"].append(mitigation)
        
        # Add general security best practices
        mitigation_results["best_practices"] = self.security_best_practices.get(
            language, self.security_best_practices.get("general", [])
        )
        
        return ToolResult(output=self._format_mitigation_results(mitigation_results))

    async def _generate_security_report(
        self,
        target_path: Optional[str],
        language: Optional[str],
        compliance_standard: Optional[str],
        severity_threshold: str
    ) -> ToolResult:
        """Generate a comprehensive security report."""
        
        if not target_path:
            return ToolResult(error="Target path is required for generating a security report")
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(target_path)
        
        # Prepare report
        report = {
            "title": f"Security Audit Report - {target_path}",
            "target_path": target_path,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "executive_summary": {
                "risk_level": "unknown",
                "security_score": 0.0,
                "critical_findings": 0,
                "high_findings": 0,
                "medium_findings": 0,
                "low_findings": 0
            },
            "vulnerability_assessment": {
                "code_vulnerabilities": [],
                "dependency_vulnerabilities": [],
                "configuration_issues": []
            },
            "compliance_status": {
                "standard": compliance_standard,
                "compliance_score": 0.0,
                "compliant_items": [],
                "non_compliant_items": []
            },
            "recommendations": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            },
            "appendices": {
                "methodology": "Static code analysis, dependency checking, and configuration review",
                "tools_used": ["security_auditor"],
                "limitations": "This is a simulated security audit and should not replace a professional security assessment"
            }
        }
        
        # Perform vulnerability assessment
        vulnerability_result = await self._identify_vulnerabilities(target_path, language, severity_threshold, "standard")
        if hasattr(vulnerability_result, "output"):
            vulnerabilities = self._extract_vulnerabilities_from_output(vulnerability_result.output)
            report["vulnerability_assessment"]["code_vulnerabilities"] = vulnerabilities
            
            # Count findings by severity
            for vuln in vulnerabilities:
                severity = vuln.get("severity", "low")
                if severity == "critical":
                    report["executive_summary"]["critical_findings"] += 1
                elif severity == "high":
                    report["executive_summary"]["high_findings"] += 1
                elif severity == "medium":
                    report["executive_summary"]["medium_findings"] += 1
                elif severity == "low":
                    report["executive_summary"]["low_findings"] += 1
        
        # Check dependencies
        dependency_result = await self._check_dependencies(target_path, language, severity_threshold)
        if hasattr(dependency_result, "output"):
            # In a real implementation, this would parse structured data
            report["vulnerability_assessment"]["dependency_vulnerabilities"] = self._extract_dependencies_from_output(dependency_result.output)
        
        # Check configuration
        config_result = await self._analyze_configuration(target_path, None)
        if hasattr(config_result, "output"):
            # In a real implementation, this would parse structured data
            report["vulnerability_assessment"]["configuration_issues"] = self._extract_config_issues_from_output(config_result.output)
        
        # Check compliance if standard provided
        if compliance_standard:
            compliance_result = await self._check_compliance(target_path, compliance_standard, severity_threshold)
            if hasattr(compliance_result, "output"):
                # In a real implementation, this would parse structured data
                compliance_data = self._extract_compliance_from_output(compliance_result.output)
                report["compliance_status"]["compliance_score"] = compliance_data.get("score", 0.0)
                report["compliance_status"]["compliant_items"] = compliance_data.get("compliant", [])
                report["compliance_status"]["non_compliant_items"] = compliance_data.get("non_compliant", [])
        
        # Determine overall risk level and security score
        total_findings = (
            report["executive_summary"]["critical_findings"] +
            report["executive_summary"]["high_findings"] +
            report["executive_summary"]["medium_findings"] +
            report["executive_summary"]["low_findings"]
        )
        
        weighted_score = (
            report["executive_summary"]["critical_findings"] * 1.0 +
            report["executive_summary"]["high_findings"] * 0.7 +
            report["executive_summary"]["medium_findings"] * 0.4 +
            report["executive_summary"]["low_findings"] * 0.1
        )
        
        if total_findings > 0:
            report["executive_summary"]["security_score"] = 1.0 - (weighted_score / (total_findings * 1.0))
        else:
            report["executive_summary"]["security_score"] = 1.0
        
        if report["executive_summary"]["critical_findings"] > 0:
            report["executive_summary"]["risk_level"] = "critical"
        elif report["executive_summary"]["high_findings"] > 0:
            report["executive_summary"]["risk_level"] = "high"
        elif report["executive_summary"]["medium_findings"] > 0:
            report["executive_summary"]["risk_level"] = "medium"
        elif report["executive_summary"]["low_findings"] > 0:
            report["executive_summary"]["risk_level"] = "low"
        else:
            report["executive_summary"]["risk_level"] = "info"
        
        # Generate recommendations
        all_vulnerabilities = (
            report["vulnerability_assessment"]["code_vulnerabilities"] +
            report["vulnerability_assessment"]["dependency_vulnerabilities"] +
            report["vulnerability_assessment"]["configuration_issues"]
        )
        
        for vuln in all_vulnerabilities:
            severity = vuln.get("severity", "low")
            recommendation = {
                "finding": vuln.get("description", "Unknown issue"),
                "mitigation": vuln.get("mitigation", "No specific mitigation available")
            }
            
            if severity in report["recommendations"]:
                report["recommendations"][severity].append(recommendation)
        
        # Store report in audit history
        self.audit_history.append({
            "type": "security_report",
            "target": target_path,
            "timestamp": report["timestamp"],
            "risk_level": report["executive_summary"]["risk_level"],
            "security_score": report["executive_summary"]["security_score"]
        })
        
        return ToolResult(output=self._format_security_report(report))

    async def _audit_permissions(
        self,
        target_path: Optional[str]
    ) -> ToolResult:
        """Audit file and directory permissions."""
        
        if not target_path:
            return ToolResult(error="Target path is required for permission auditing")
        
        # Prepare permission audit results
        permission_results = {
            "target_path": target_path,
            "timestamp": datetime.now().isoformat(),
            "files_checked": 0,
            "permission_issues": [],
            "sensitive_files": [],
            "recommendations": []
        }
        
        # Simulate permission checking
        # In a real implementation, this would check actual file permissions
        permission_check = self._simulate_permission_check(target_path)
        permission_results["files_checked"] = permission_check["files_checked"]
        permission_results["permission_issues"] = permission_check["issues"]
        permission_results["sensitive_files"] = permission_check["sensitive_files"]
        
        # Generate recommendations
        permission_results["recommendations"] = self._generate_permission_recommendations(permission_check["issues"])
        
        # Store audit in history
        self.audit_history.append({
            "type": "permission_audit",
            "target": target_path,
            "timestamp": permission_results["timestamp"],
            "issues_found": len(permission_check["issues"])
        })
        
        return ToolResult(output=self._format_permission_results(permission_results))

    async def _check_secrets(
        self,
        target_path: Optional[str],
        severity_threshold: str
    ) -> ToolResult:
        """Check for hardcoded secrets and credentials."""
        
        if not target_path:
            return ToolResult(error="Target path is required for secrets checking")
        
        # Prepare secrets check results
        secrets_results = {
            "target_path": target_path,
            "timestamp": datetime.now().isoformat(),
            "files_checked": 0,
            "secrets_found": [],
            "risk_level": "unknown",
            "recommendations": []
        }
        
        # Simulate secrets checking
        # In a real implementation, this would scan actual files for secrets
        secrets_check = self._simulate_secrets_check(target_path, severity_threshold)
        secrets_results["files_checked"] = secrets_check["files_checked"]
        secrets_results["secrets_found"] = secrets_check["secrets"]
        
        # Determine risk level
        if any(s["severity"] == "critical" for s in secrets_check["secrets"]):
            secrets_results["risk_level"] = "critical"
        elif any(s["severity"] == "high" for s in secrets_check["secrets"]):
            secrets_results["risk_level"] = "high"
        elif any(s["severity"] == "medium" for s in secrets_check["secrets"]):
            secrets_results["risk_level"] = "medium"
        elif any(s["severity"] == "low" for s in secrets_check["secrets"]):
            secrets_results["risk_level"] = "low"
        else:
            secrets_results["risk_level"] = "info"
        
        # Generate recommendations
        secrets_results["recommendations"] = self._generate_secrets_recommendations(secrets_check["secrets"])
        
        # Store check in audit history
        self.audit_history.append({
            "type": "secrets_check",
            "target": target_path,
            "timestamp": secrets_results["timestamp"],
            "secrets_found": len(secrets_check["secrets"]),
            "risk_level": secrets_results["risk_level"]
        })
        
        return ToolResult(output=self._format_secrets_results(secrets_results))

    # Helper methods for simulating security checks

    def _detect_language(self, target_path: str) -> str:
        """Detect the programming language of the target."""
        # This is a simplified implementation
        # In a real implementation, this would analyze file extensions and content
        return "python"  # Default to Python

    def _simulate_code_scan(
        self, 
        target_path: str, 
        language: str, 
        patterns: Dict[str, Dict[str, Any]],
        severity_threshold: str,
        scan_depth: str
    ) -> List[Dict[str, Any]]:
        """Simulate scanning code for vulnerabilities."""
        # This is a simulated implementation
        # In a real implementation, this would scan actual code files
        
        severity_levels = ["info", "low", "medium", "high", "critical"]
        threshold_index = severity_levels.index(severity_threshold)
        
        # Simulate finding vulnerabilities
        vulnerabilities = []
        
        # Adjust number of findings based on scan depth
        max_findings = 3 if scan_depth == "quick" else 7 if scan_depth == "standard" else 12
        
        for vuln_type, vuln_info in patterns.items():
            severity = vuln_info["severity"]
            
            # Skip if below threshold
            if severity_levels.index(severity) < threshold_index:
                continue
            
            # Simulate finding this vulnerability
            if len(vulnerabilities) < max_findings:
                vulnerabilities.append({
                    "type": vuln_type,
                    "description": vuln_info["description"],
                    "severity": severity,
                    "location": f"{target_path}/example_{vuln_type}.{language}:42",
                    "code_snippet": f"# Example vulnerable code for {vuln_type}",
                    "mitigation": vuln_info["mitigation"]
                })
        
        return vulnerabilities

    def _simulate_dependency_check(
        self,
        target_path: str,
        language: str,
        severity_threshold: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Simulate checking dependencies for vulnerabilities."""
        # This is a simulated implementation
        # In a real implementation, this would check actual dependencies
        
        severity_levels = ["info", "low", "medium", "high", "critical"]
        threshold_index = severity_levels.index(severity_threshold)
        
        # Simulate dependencies
        all_dependencies = []
        vulnerable_dependencies = []
        outdated_dependencies = []
        
        # Python dependencies
        if language == "python":
            dependencies = [
                {"name": "django", "version": "3.2.0", "latest": "4.2.0"},
                {"name": "requests", "version": "2.25.1", "latest": "2.28.2"},
                {"name": "flask", "version": "2.0.1", "latest": "2.2.3"},
                {"name": "sqlalchemy", "version": "1.4.0", "latest": "2.0.0"},
                {"name": "numpy", "version": "1.20.0", "latest": "1.24.2"}
            ]
            
            # Simulate vulnerabilities
            vulnerabilities = [
                {
                    "dependency": "django",
                    "version": "3.2.0",
                    "vulnerability": "SQL Injection in ORM",
                    "severity": "high",
                    "fixed_in": "3.2.5",
                    "cve": "CVE-2023-12345"
                },
                {
                    "dependency": "flask",
                    "version": "2.0.1",
                    "vulnerability": "Path Traversal",
                    "severity": "medium",
                    "fixed_in": "2.0.3",
                    "cve": "CVE-2023-23456"
                }
            ]
        
        # JavaScript dependencies
        elif language == "javascript":
            dependencies = [
                {"name": "react", "version": "17.0.2", "latest": "18.2.0"},
                {"name": "express", "version": "4.17.1", "latest": "4.18.2"},
                {"name": "lodash", "version": "4.17.20", "latest": "4.17.21"},
                {"name": "axios", "version": "0.21.1", "latest": "1.3.4"},
                {"name": "moment", "version": "2.29.1", "latest": "2.29.4"}
            ]
            
            # Simulate vulnerabilities
            vulnerabilities = [
                {
                    "dependency": "axios",
                    "version": "0.21.1",
                    "vulnerability": "Server-Side Request Forgery",
                    "severity": "high",
                    "fixed_in": "0.21.4",
                    "cve": "CVE-2023-34567"
                },
                {
                    "dependency": "lodash",
                    "version": "4.17.20",
                    "vulnerability": "Prototype Pollution",
                    "severity": "medium",
                    "fixed_in": "4.17.21",
                    "cve": "CVE-2023-45678"
                }
            ]
        
        # Default dependencies
        else:
            dependencies = [
                {"name": "library1", "version": "1.0.0", "latest": "2.0.0"},
                {"name": "library2", "version": "0.5.0", "latest": "0.7.0"},
                {"name": "library3", "version": "3.1.0", "latest": "3.2.0"}
            ]
            
            # Simulate vulnerabilities
            vulnerabilities = [
                {
                    "dependency": "library1",
                    "version": "1.0.0",
                    "vulnerability": "Security Issue",
                    "severity": "medium",
                    "fixed_in": "1.1.0",
                    "cve": "CVE-2023-56789"
                }
            ]
        
        # Process dependencies
        all_dependencies = dependencies
        
        # Find outdated dependencies
        for dep in dependencies:
            if dep["version"] != dep["latest"]:
                outdated_dependencies.append({
                    "name": dep["name"],
                    "current_version": dep["version"],
                    "latest_version": dep["latest"],
                    "update_priority": "high" if dep["name"] in [v["dependency"] for v in vulnerabilities] else "medium"
                })
        
        # Filter vulnerabilities by severity threshold
        for vuln in vulnerabilities:
            if severity_levels.index(vuln["severity"]) >= threshold_index:
                vulnerable_dependencies.append({
                    "name": vuln["dependency"],
                    "version": vuln["version"],
                    "vulnerability": vuln["vulnerability"],
                    "severity": vuln["severity"],
                    "fixed_in": vuln["fixed_in"],
                    "cve": vuln["cve"]
                })
        
        return {
            "all": all_dependencies,
            "vulnerable": vulnerable_dependencies,
            "outdated": outdated_dependencies
        }

    def _simulate_config_analysis(
        self,
        target_path: str,
        framework: Optional[str]
    ) -> Dict[str, Any]:
        """Simulate analyzing configuration files for security issues."""
        # This is a simulated implementation
        # In a real implementation, this would analyze actual configuration files
        
        # Simulate finding configuration files
        config_files = [
            f"{target_path}/config.json",
            f"{target_path}/settings.py",
            f"{target_path}/.env.example"
        ]
        
        # Simulate finding misconfigurations
        misconfigurations = [
            {
                "file": f"{target_path}/config.json",
                "issue": "Debug mode enabled in production",
                "severity": "medium",
                "recommendation": "Disable debug mode in production environments"
            },
            {
                "file": f"{target_path}/settings.py",
                "issue": "Missing Content Security Policy",
                "severity": "medium",
                "recommendation": "Implement a Content Security Policy"
            }
        ]
        
        # Simulate security headers check
        security_headers = {
            "Content-Security-Policy": False,
            "X-XSS-Protection": True,
            "X-Content-Type-Options": True,
            "X-Frame-Options": True,
            "Strict-Transport-Security": False,
            "Referrer-Policy": False
        }
        
        return {
            "files": config_files,
            "misconfigurations": misconfigurations,
            "security_headers": security_headers
        }

    def _simulate_compliance_check(
        self,
        target_path: str,
        requirements: List[Dict[str, Any]],
        severity_threshold: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Simulate checking compliance with security standards."""
        # This is a simulated implementation
        # In a real implementation, this would perform actual compliance checks
        
        compliant = []
        non_compliant = []
        
        # Simulate compliance checking
        for req in requirements:
            # Randomly determine compliance (for simulation)
            is_compliant = len(req["id"]) % 2 == 0  # Simple deterministic simulation
            
            if is_compliant:
                compliant.append({
                    "id": req["id"],
                    "name": req["name"],
                    "status": "compliant",
                    "evidence": f"Verified compliance with {req['name']}"
                })
            else:
                non_compliant.append({
                    "id": req["id"],
                    "name": req["name"],
                    "status": "non_compliant",
                    "issue": f"Failed to meet {req['name']} requirement",
                    "severity": "high" if "critical" in req["name"].lower() else "medium",
                    "recommendation": f"Implement controls to address {req['name']}"
                })
        
        return {
            "compliant": compliant,
            "non_compliant": non_compliant
        }

    def _simulate_permission_check(self, target_path: str) -> Dict[str, Any]:
        """Simulate checking file and directory permissions."""
        # This is a simulated implementation
        # In a real implementation, this would check actual file permissions
        
        # Simulate checking files
        files_checked = 25
        
        # Simulate permission issues
        permission_issues = [
            {
                "file": f"{target_path}/config/secrets.json",
                "current_permission": "0644",
                "issue": "Overly permissive file permissions",
                "severity": "high",
                "recommended_permission": "0600"
            },
            {
                "file": f"{target_path}/logs",
                "current_permission": "0777",
                "issue": "World-writable directory",
                "severity": "high",
                "recommended_permission": "0750"
            }
        ]
        
        # Simulate sensitive files
        sensitive_files = [
            {
                "file": f"{target_path}/config/secrets.json",
                "type": "configuration",
                "contains": "API keys, database credentials",
                "current_permission": "0644",
                "recommended_permission": "0600"
            },
            {
                "file": f"{target_path}/.env",
                "type": "environment",
                "contains": "Environment variables, credentials",
                "current_permission": "0644",
                "recommended_permission": "0600"
            }
        ]
        
        return {
            "files_checked": files_checked,
            "issues": permission_issues,
            "sensitive_files": sensitive_files
        }

    def _simulate_secrets_check(
        self,
        target_path: str,
        severity_threshold: str
    ) -> Dict[str, Any]:
        """Simulate checking for hardcoded secrets."""
        # This is a simulated implementation
        # In a real implementation, this would scan actual files for secrets
        
        severity_levels = ["info", "low", "medium", "high", "critical"]
        threshold_index = severity_levels.index(severity_threshold)
        
        # Simulate checking files
        files_checked = 30
        
        # Simulate finding secrets
        all_secrets = [
            {
                "file": f"{target_path}/config.py",
                "line": 42,
                "type": "API Key",
                "severity": "high",
                "snippet": "api_key = 'AKIAxxxxxxxxxxxxxxxx'",
                "recommendation": "Move to environment variables or secure secret storage"
            },
            {
                "file": f"{target_path}/database.js",
                "line": 17,
                "type": "Database Password",
                "severity": "critical",
                "snippet": "password: 'p@ssw0rd123'",
                "recommendation": "Use environment variables or secure secret storage"
            },
            {
                "file": f"{target_path}/utils/helpers.py",
                "line": 105,
                "type": "Auth Token",
                "severity": "medium",
                "snippet": "auth_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'",
                "recommendation": "Use secure token management"
            }
        ]
        
        # Filter by severity threshold
        secrets = [
            secret for secret in all_secrets
            if severity_levels.index(secret["severity"]) >= threshold_index
        ]
        
        return {
            "files_checked": files_checked,
            "secrets": secrets
        }

    def _simulate_infrastructure_scan(
        self,
        target_path: str,
        severity_threshold: str
    ) -> List[Dict[str, Any]]:
        """Simulate scanning infrastructure for vulnerabilities."""
        # This is a simulated implementation
        # In a real implementation, this would scan actual infrastructure
        
        severity_levels = ["info", "low", "medium", "high", "critical"]
        threshold_index = severity_levels.index(severity_threshold)
        
        # Simulate infrastructure vulnerabilities
        all_vulnerabilities = [
            {
                "component": "Web Server",
                "issue": "TLS 1.0/1.1 Enabled",
                "severity": "medium",
                "recommendation": "Disable TLS 1.0/1.1 and use TLS 1.2+ only"
            },
            {
                "component": "Database",
                "issue": "Public Exposure",
                "severity": "high",
                "recommendation": "Restrict database access to application servers only"
            },
            {
                "component": "Container",
                "issue": "Running as Root",
                "severity": "medium",
                "recommendation": "Use non-root user in containers"
            }
        ]
        
        # Filter by severity threshold
        vulnerabilities = [
            vuln for vuln in all_vulnerabilities
            if severity_levels.index(vuln["severity"]) >= threshold_index
        ]
        
        return vulnerabilities

    # Helper methods for generating recommendations

    def _generate_security_recommendations(
        self,
        vulnerabilities: List[Dict[str, Any]],
        language: str
    ) -> List[str]:
        """Generate security recommendations based on vulnerabilities."""
        recommendations = []
        
        # Add specific recommendations for each vulnerability
        for vuln in vulnerabilities:
            if "mitigation" in vuln:
                recommendations.append(f"{vuln['mitigation']} to address {vuln['description']}")
        
        # Add general recommendations based on language
        if language in self.security_best_practices:
            # Add a few general best practices
            general_practices = self.security_best_practices[language][:3]
            recommendations.extend(general_practices)
        
        return recommendations

    def _generate_dependency_recommendations(
        self,
        vulnerable_dependencies: List[Dict[str, Any]],
        outdated_dependencies: List[Dict[str, Any]],
        language: str
    ) -> List[str]:
        """Generate recommendations for dependency issues."""
        recommendations = []
        
        # Add recommendations for vulnerable dependencies
        if vulnerable_dependencies:
            recommendations.append(f"Update {len(vulnerable_dependencies)} vulnerable dependencies to their fixed versions")
            for dep in vulnerable_dependencies:
                recommendations.append(f"Update {dep['name']} from {dep['version']} to at least {dep['fixed_in']} to fix {dep['vulnerability']}")
        
        # Add recommendations for outdated dependencies
        if outdated_dependencies:
            high_priority = [dep for dep in outdated_dependencies if dep["update_priority"] == "high"]
            if high_priority:
                recommendations.append(f"Prioritize updating {len(high_priority)} high-priority outdated dependencies")
        
        # Add general dependency management recommendations
        recommendations.append("Implement automated dependency scanning in your CI/CD pipeline")
        recommendations.append("Set up alerts for new vulnerabilities in your dependencies")
        
        return recommendations

    def _generate_configuration_recommendations(
        self,
        misconfigurations: List[Dict[str, Any]],
        framework: Optional[str]
    ) -> List[str]:
        """Generate recommendations for configuration issues."""
        recommendations = []
        
        # Add specific recommendations for each misconfiguration
        for config in misconfigurations:
            if "recommendation" in config:
                recommendations.append(config["recommendation"])
        
        # Add framework-specific recommendations
        if framework == "django":
            recommendations.append("Use Django's security middleware")
            recommendations.append("Enable Django's built-in security features")
        elif framework == "express":
            recommendations.append("Use Helmet.js to set security headers")
            recommendations.append("Implement rate limiting")
        
        # Add general configuration recommendations
        recommendations.append("Implement proper Content Security Policy")
        recommendations.append("Enable HTTPS and configure secure headers")
        recommendations.append("Use environment-specific configuration files")
        
        return recommendations

    def _generate_compliance_recommendations(
        self,
        non_compliant_items: List[Dict[str, Any]],
        standard: str
    ) -> List[str]:
        """Generate recommendations for compliance issues."""
        recommendations = []
        
        # Add specific recommendations for each non-compliant item
        for item in non_compliant_items:
            if "recommendation" in item:
                recommendations.append(item["recommendation"])
        
        # Add standard-specific recommendations
        if standard == "owasp":
            recommendations.append("Implement OWASP's recommended security controls")
            recommendations.append("Conduct regular OWASP-based security assessments")
        elif standard == "pci-dss":
            recommendations.append("Implement proper cardholder data protection measures")
            recommendations.append("Conduct regular PCI DSS compliance audits")
        
        # Add general compliance recommendations
        recommendations.append("Document compliance efforts and maintain evidence")
        recommendations.append("Implement a security awareness program")
        recommendations.append("Conduct regular security assessments")
        
        return recommendations

    def _generate_vulnerability_recommendations(
        self,
        vulnerabilities: List[Dict[str, Any]],
        language: str
    ) -> List[str]:
        """Generate recommendations for vulnerabilities."""
        recommendations = []
        
        # Group vulnerabilities by type
        vulnerability_types = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get("type", vuln.get("issue", "unknown"))
            if vuln_type not in vulnerability_types:
                vulnerability_types[vuln_type] = []
            vulnerability_types[vuln_type].append(vuln)
        
        # Add recommendations for each vulnerability type
        for vuln_type, vulns in vulnerability_types.items():
            if len(vulns) > 1:
                recommendations.append(f"Address {len(vulns)} {vuln_type} vulnerabilities")
            
            # Add specific recommendation for the first vulnerability of each type
            if vulns and "mitigation" in vulns[0]:
                recommendations.append(vulns[0]["mitigation"])
        
        # Add language-specific recommendations
        if language in self.security_best_practices:
            # Add a few general best practices
            general_practices = self.security_best_practices[language][:2]
            recommendations.extend(general_practices)
        
        return recommendations

    def _generate_permission_recommendations(
        self,
        permission_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for permission issues."""
        recommendations = []
        
        # Add specific recommendations for each permission issue
        for issue in permission_issues:
            if "recommended_permission" in issue:
                recommendations.append(f"Change permissions of {issue['file']} from {issue['current_permission']} to {issue['recommended_permission']}")
        
        # Add general permission recommendations
        recommendations.append("Implement the principle of least privilege for all files and directories")
        recommendations.append("Regularly audit file and directory permissions")
        recommendations.append("Use proper umask settings for new files")
        
        return recommendations

    def _generate_secrets_recommendations(
        self,
        secrets: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for secrets management."""
        recommendations = []
        
        # Add specific recommendations for each secret
        for secret in secrets:
            if "recommendation" in secret:
                recommendations.append(secret["recommendation"])
        
        # Add general secrets management recommendations
        recommendations.append("Use environment variables for secrets")
        recommendations.append("Implement a secure secrets management solution")
        recommendations.append("Scan code for secrets before committing")
        recommendations.append("Use .gitignore and .dockerignore to prevent secrets from being included in repositories and images")
        
        return recommendations

    def _get_mitigation_steps(
        self,
        vulnerability: Dict[str, Any],
        language: str
    ) -> List[str]:
        """Get mitigation steps for a specific vulnerability."""
        # This is a simplified implementation
        # In a real implementation, this would provide detailed mitigation steps
        
        vuln_type = vulnerability.get("type", "unknown")
        
        if vuln_type == "sql_injection":
            return [
                "Use parameterized queries or prepared statements",
                "Use an ORM (Object-Relational Mapping) library",
                "Validate and sanitize all user inputs",
                "Implement proper error handling to avoid exposing database information"
            ]
        elif vuln_type == "xss":
            return [
                "Use context-appropriate output encoding",
                "Implement Content Security Policy (CSP)",
                "Validate and sanitize all user inputs",
                "Use safe frameworks that automatically escape XSS"
            ]
        elif vuln_type == "command_injection":
            return [
                "Avoid using shell=True in subprocess calls",
                "Use safer alternatives to os.system() and eval()",
                "Validate and sanitize all user inputs",
                "Implement proper input validation and allowlisting"
            ]
        elif "hardcoded_secret" in vuln_type or "api_key" in vuln_type:
            return [
                "Move secrets to environment variables",
                "Use a secure secrets management solution",
                "Rotate any exposed secrets immediately",
                "Implement proper access controls for secrets"
            ]
        else:
            return [
                "Review and refactor the vulnerable code",
                "Follow security best practices for your language and framework",
                "Implement proper input validation and output encoding",
                "Consider using security libraries and frameworks"
            ]

    def _get_mitigation_code_example(
        self,
        vulnerability: Dict[str, Any],
        language: str
    ) -> str:
        """Get a code example for mitigating a specific vulnerability."""
        # This is a simplified implementation
        # In a real implementation, this would provide language-specific code examples
        
        vuln_type = vulnerability.get("type", "unknown")
        
        if language == "python":
            if vuln_type == "sql_injection":
                return """# Bad:
cursor.execute("SELECT * FROM users WHERE username = '" + username + "'")

# Good:
cursor.execute("SELECT * FROM users WHERE username = %s", (username,))

# Or with an ORM like SQLAlchemy:
user = session.query(User).filter(User.username == username).first()
"""
            elif vuln_type == "command_injection":
                return """# Bad:
os.system("ls " + user_input)

# Good:
import subprocess
subprocess.run(["ls", user_input], check=True)
"""
            elif "hardcoded_secret" in vuln_type:
                return """# Bad:
api_key = "AKIAxxxxxxxxxxxxxxxx"

# Good:
import os
api_key = os.environ.get("API_KEY")
"""
        elif language == "javascript":
            if vuln_type == "xss":
                return """// Bad:
element.innerHTML = userInput;

// Good:
element.textContent = userInput;

// Or with React:
import { sanitize } from 'dompurify';
// ...
<div dangerouslySetInnerHTML={{ __html: sanitize(userInput) }} />
"""
            elif vuln_type == "sql_injection":
                return """// Bad:
const query = `SELECT * FROM users WHERE username = '${username}'`;

// Good:
const query = 'SELECT * FROM users WHERE username = ?';
db.query(query, [username], (err, results) => {
  // ...
});
"""
            elif "hardcoded_secret" in vuln_type:
                return """// Bad:
const apiKey = 'AKIAxxxxxxxxxxxxxxxx';

// Good:
const apiKey = process.env.API_KEY;
"""
        
        # Default example
        return """# Example mitigation:
# 1. Identify the vulnerable code
# 2. Understand the security issue
# 3. Apply the appropriate security control
# 4. Test the fix thoroughly
"""

    # Helper methods for extracting data from output

    def _extract_vulnerabilities_from_output(self, output: str) -> List[Dict[str, Any]]:
        """Extract vulnerabilities from scan output."""
        # This is a simplified implementation
        # In a real implementation, this would parse structured data
        
        # Simulate extracting vulnerabilities
        vulnerabilities = [
            {
                "type": "sql_injection",
                "description": "SQL Injection vulnerability",
                "severity": "high",
                "location": "example.py:42",
                "mitigation": "Use parameterized queries"
            },
            {
                "type": "xss",
                "description": "Cross-site Scripting vulnerability",
                "severity": "medium",
                "location": "example.js:17",
                "mitigation": "Use proper output encoding"
            }
        ]
        
        return vulnerabilities

    def _extract_dependencies_from_output(self, output: str) -> List[Dict[str, Any]]:
        """Extract dependency vulnerabilities from output."""
        # This is a simplified implementation
        # In a real implementation, this would parse structured data
        
        # Simulate extracting dependencies
        dependencies = [
            {
                "name": "vulnerable-package",
                "version": "1.0.0",
                "vulnerability": "Remote Code Execution",
                "severity": "high",
                "fixed_in": "1.0.1",
                "mitigation": "Update to version 1.0.1 or later"
            }
        ]
        
        return dependencies

    def _extract_config_issues_from_output(self, output: str) -> List[Dict[str, Any]]:
        """Extract configuration issues from output."""
        # This is a simplified implementation
        # In a real implementation, this would parse structured data
        
        # Simulate extracting configuration issues
        config_issues = [
            {
                "file": "config.json",
                "issue": "Debug mode enabled in production",
                "severity": "medium",
                "mitigation": "Disable debug mode in production environments"
            }
        ]
        
        return config_issues

    def _extract_compliance_from_output(self, output: str) -> Dict[str, Any]:
        """Extract compliance data from output."""
        # This is a simplified implementation
        # In a real implementation, this would parse structured data
        
        # Simulate extracting compliance data
        compliance_data = {
            "score": 0.75,
            "compliant": [
                {"id": "Req 1", "name": "Requirement 1", "status": "compliant"}
            ],
            "non_compliant": [
                {"id": "Req 2", "name": "Requirement 2", "status": "non_compliant"}
            ]
        }
        
        return compliance_data

    # Helper methods for formatting results

    def _format_scan_results(self, results: Dict[str, Any]) -> str:
        """Format code scan results for output."""
        output = ["# Security Code Scan Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Language: {results['language']}")
        output.append(f"Scan Depth: {results['scan_depth']}")
        output.append(f"Security Score: {results['security_score']:.2f}")
        output.append(f"Risk Level: {results['risk_level'].upper()}")
        
        if results["vulnerabilities"]:
            output.append(f"\n## Vulnerabilities Found: {len(results['vulnerabilities'])}")
            
            # Group by severity
            by_severity = {}
            for vuln in results["vulnerabilities"]:
                severity = vuln["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(vuln)
            
            # Display vulnerabilities by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for vuln in by_severity[severity]:
                        output.append(f"\n- **{vuln['description']}**")
                        output.append(f"  - Location: {vuln['location']}")
                        if "code_snippet" in vuln:
                            output.append(f"  - Code: `{vuln['code_snippet']}`")
                        if "mitigation" in vuln:
                            output.append(f"  - Mitigation: {vuln['mitigation']}")
        else:
            output.append("\n## No vulnerabilities found")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)

    def _format_dependency_results(self, results: Dict[str, Any]) -> str:
        """Format dependency check results for output."""
        output = ["# Dependency Security Check Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Language: {results['language']}")
        
        if results["vulnerable_dependencies"]:
            output.append(f"\n## Vulnerable Dependencies: {len(results['vulnerable_dependencies'])}")
            
            # Group by severity
            by_severity = {}
            for dep in results["vulnerable_dependencies"]:
                severity = dep["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(dep)
            
            # Display vulnerabilities by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for dep in by_severity[severity]:
                        output.append(f"\n- **{dep['name']}** (version {dep['version']})")
                        output.append(f"  - Vulnerability: {dep['vulnerability']}")
                        output.append(f"  - Fixed in: {dep['fixed_in']}")
                        if "cve" in dep:
                            output.append(f"  - CVE: {dep['cve']}")
        else:
            output.append("\n## No vulnerable dependencies found")
        
        if results["outdated_dependencies"]:
            output.append(f"\n## Outdated Dependencies: {len(results['outdated_dependencies'])}")
            
            # Group by update priority
            high_priority = [dep for dep in results["outdated_dependencies"] if dep["update_priority"] == "high"]
            other_priority = [dep for dep in results["outdated_dependencies"] if dep["update_priority"] != "high"]
            
            if high_priority:
                output.append(f"\n### High Priority Updates ({len(high_priority)})")
                for dep in high_priority:
                    output.append(f"- **{dep['name']}**: {dep['current_version']} → {dep['latest_version']}")
            
            if other_priority:
                output.append(f"\n### Other Updates ({len(other_priority)})")
                for dep in other_priority:
                    output.append(f"- **{dep['name']}**: {dep['current_version']} → {dep['latest_version']}")
        else:
            output.append("\n## All dependencies are up to date")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)

    def _format_configuration_results(self, results: Dict[str, Any]) -> str:
        """Format configuration analysis results for output."""
        output = ["# Security Configuration Analysis Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        if results["framework"]:
            output.append(f"Framework: {results['framework']}")
        
        if results["config_files"]:
            output.append(f"\n## Configuration Files Analyzed: {len(results['config_files'])}")
            for file in results["config_files"]:
                output.append(f"- {file}")
        
        if results["misconfigurations"]:
            output.append(f"\n## Misconfigurations Found: {len(results['misconfigurations'])}")
            
            # Group by severity
            by_severity = {}
            for config in results["misconfigurations"]:
                severity = config["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(config)
            
            # Display misconfigurations by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for config in by_severity[severity]:
                        output.append(f"\n- **{config['issue']}**")
                        output.append(f"  - File: {config['file']}")
                        if "recommendation" in config:
                            output.append(f"  - Recommendation: {config['recommendation']}")
        else:
            output.append("\n## No misconfigurations found")
        
        if results["security_headers"]:
            output.append("\n## Security Headers Analysis")
            
            missing_headers = [header for header, implemented in results["security_headers"].items() if not implemented]
            implemented_headers = [header for header, implemented in results["security_headers"].items() if implemented]
            
            if missing_headers:
                output.append(f"\n### Missing Security Headers ({len(missing_headers)})")
                for header in missing_headers:
                    output.append(f"- {header}")
            
            if implemented_headers:
                output.append(f"\n### Implemented Security Headers ({len(implemented_headers)})")
                for header in implemented_headers:
                    output.append(f"- {header}")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)

    def _format_compliance_results(self, results: Dict[str, Any]) -> str:
        """Format compliance check results for output."""
        output = ["# Security Compliance Check Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Standard: {results['standard'].upper()}")
        output.append(f"Compliance Score: {results['compliance_score']:.2f} ({int(results['compliance_score'] * 100)}%)")
        
        if results["compliant_requirements"]:
            output.append(f"\n## Compliant Requirements: {len(results['compliant_requirements'])}")
            for req in results["compliant_requirements"]:
                output.append(f"- **{req['id']}**: {req['name']}")
                if "evidence" in req:
                    output.append(f"  - Evidence: {req['evidence']}")
        
        if results["non_compliant_requirements"]:
            output.append(f"\n## Non-Compliant Requirements: {len(results['non_compliant_requirements'])}")
            
            # Group by severity
            by_severity = {}
            for req in results["non_compliant_requirements"]:
                severity = req.get("severity", "medium")
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(req)
            
            # Display non-compliant requirements by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for req in by_severity[severity]:
                        output.append(f"\n- **{req['id']}**: {req['name']}")
                        if "issue" in req:
                            output.append(f"  - Issue: {req['issue']}")
                        if "recommendation" in req:
                            output.append(f"  - Recommendation: {req['recommendation']}")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)

    def _format_vulnerability_results(self, results: Dict[str, Any]) -> str:
        """Format vulnerability identification results for output."""
        output = ["# Security Vulnerability Assessment Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Language: {results['language']}")
        output.append(f"Scan Depth: {results['scan_depth']}")
        output.append(f"Overall Risk Level: {results['overall_risk_level'].upper()}")
        
        # Code vulnerabilities
        if results["code_vulnerabilities"]:
            output.append(f"\n## Code Vulnerabilities: {len(results['code_vulnerabilities'])}")
            
            # Group by severity
            by_severity = {}
            for vuln in results["code_vulnerabilities"]:
                severity = vuln["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(vuln)
            
            # Display vulnerabilities by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for vuln in by_severity[severity]:
                        output.append(f"\n- **{vuln['description']}**")
                        if "location" in vuln:
                            output.append(f"  - Location: {vuln['location']}")
                        if "mitigation" in vuln:
                            output.append(f"  - Mitigation: {vuln['mitigation']}")
        else:
            output.append("\n## No code vulnerabilities found")
        
        # Infrastructure vulnerabilities
        if results["infrastructure_vulnerabilities"]:
            output.append(f"\n## Infrastructure Vulnerabilities: {len(results['infrastructure_vulnerabilities'])}")
            
            # Group by severity
            by_severity = {}
            for vuln in results["infrastructure_vulnerabilities"]:
                severity = vuln["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(vuln)
            
            # Display vulnerabilities by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for vuln in by_severity[severity]:
                        output.append(f"\n- **{vuln['component']}**: {vuln['issue']}")
                        if "recommendation" in vuln:
                            output.append(f"  - Recommendation: {vuln['recommendation']}")
        else:
            output.append("\n## No infrastructure vulnerabilities found")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)

    def _format_mitigation_results(self, results: Dict[str, Any]) -> str:
        """Format mitigation suggestion results for output."""
        output = ["# Security Mitigation Recommendations"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Language: {results['language']}")
        
        if results["vulnerabilities"]:
            output.append(f"\n## Vulnerabilities Found: {len(results['vulnerabilities'])}")
        
        if results["mitigations"]:
            output.append(f"\n## Mitigation Recommendations: {len(results['mitigations'])}")
            
            # Group by severity
            by_severity = {}
            for mitigation in results["mitigations"]:
                severity = mitigation["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(mitigation)
            
            # Display mitigations by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for mitigation in by_severity[severity]:
                        output.append(f"\n- **{mitigation['vulnerability']}**")
                        output.append("  - Mitigation Steps:")
                        for step in mitigation["mitigation_steps"]:
                            output.append(f"    - {step}")
                        
                        if "code_example" in mitigation and mitigation["code_example"]:
                            output.append("\n  - Code Example:")
                            output.append(f"```\n{mitigation['code_example']}\n```")
        
        if results["best_practices"]:
            output.append("\n## Security Best Practices")
            for i, practice in enumerate(results["best_practices"], 1):
                output.append(f"{i}. {practice}")
        
        return "\n".join(output)

    def _format_security_report(self, report: Dict[str, Any]) -> str:
        """Format comprehensive security report for output."""
        output = [f"# {report['title']}"]
        output.append("=" * 50)
        
        # Executive Summary
        output.append("\n## Executive Summary")
        output.append(f"- **Target**: {report['target_path']}")
        output.append(f"- **Language**: {report['language']}")
        output.append(f"- **Date**: {datetime.fromisoformat(report['timestamp']).strftime('%Y-%m-%d')}")
        output.append(f"- **Overall Risk Level**: {report['executive_summary']['risk_level'].upper()}")
        output.append(f"- **Security Score**: {report['executive_summary']['security_score']:.2f} ({int(report['executive_summary']['security_score'] * 100)}%)")
        
        # Findings Summary
        output.append("\n### Findings Summary")
        output.append(f"- Critical: {report['executive_summary']['critical_findings']}")
        output.append(f"- High: {report['executive_summary']['high_findings']}")
        output.append(f"- Medium: {report['executive_summary']['medium_findings']}")
        output.append(f"- Low: {report['executive_summary']['low_findings']}")
        
        # Vulnerability Assessment
        output.append("\n## Vulnerability Assessment")
        
        # Code Vulnerabilities
        if report["vulnerability_assessment"]["code_vulnerabilities"]:
            output.append(f"\n### Code Vulnerabilities: {len(report['vulnerability_assessment']['code_vulnerabilities'])}")
            
            # Group by severity
            by_severity = {}
            for vuln in report["vulnerability_assessment"]["code_vulnerabilities"]:
                severity = vuln["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(vuln)
            
            # Display vulnerabilities by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n#### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for vuln in by_severity[severity]:
                        output.append(f"- **{vuln['description']}**")
                        if "location" in vuln:
                            output.append(f"  - Location: {vuln['location']}")
                        if "mitigation" in vuln:
                            output.append(f"  - Mitigation: {vuln['mitigation']}")
        else:
            output.append("\n### No code vulnerabilities found")
        
        # Dependency Vulnerabilities
        if report["vulnerability_assessment"]["dependency_vulnerabilities"]:
            output.append(f"\n### Dependency Vulnerabilities: {len(report['vulnerability_assessment']['dependency_vulnerabilities'])}")
            
            for vuln in report["vulnerability_assessment"]["dependency_vulnerabilities"]:
                output.append(f"- **{vuln['name']}** (version {vuln.get('version', 'unknown')})")
                output.append(f"  - Vulnerability: {vuln.get('vulnerability', 'Unknown')}")
                output.append(f"  - Severity: {vuln.get('severity', 'Unknown')}")
                if "fixed_in" in vuln:
                    output.append(f"  - Fixed in: {vuln['fixed_in']}")
        else:
            output.append("\n### No dependency vulnerabilities found")
        
        # Configuration Issues
        if report["vulnerability_assessment"]["configuration_issues"]:
            output.append(f"\n### Configuration Issues: {len(report['vulnerability_assessment']['configuration_issues'])}")
            
            for issue in report["vulnerability_assessment"]["configuration_issues"]:
                output.append(f"- **{issue.get('issue', 'Unknown issue')}**")
                if "file" in issue:
                    output.append(f"  - File: {issue['file']}")
                if "severity" in issue:
                    output.append(f"  - Severity: {issue['severity'].upper()}")
                if "mitigation" in issue:
                    output.append(f"  - Mitigation: {issue['mitigation']}")
        else:
            output.append("\n### No configuration issues found")
        
        # Compliance Status
        if report["compliance_status"]["standard"]:
            output.append(f"\n## Compliance Status: {report['compliance_status']['standard'].upper()}")
            output.append(f"- **Compliance Score**: {report['compliance_status']['compliance_score']:.2f} ({int(report['compliance_status']['compliance_score'] * 100)}%)")
            
            if report["compliance_status"]["non_compliant_items"]:
                output.append(f"\n### Non-Compliant Items: {len(report['compliance_status']['non_compliant_items'])}")
                for item in report["compliance_status"]["non_compliant_items"]:
                    output.append(f"- **{item.get('id', 'Unknown')}: {item.get('name', 'Unknown')}**")
                    if "issue" in item:
                        output.append(f"  - Issue: {item['issue']}")
                    if "recommendation" in item:
                        output.append(f"  - Recommendation: {item['recommendation']}")
        
        # Recommendations
        output.append("\n## Recommendations")
        
        # Critical Recommendations
        if report["recommendations"]["critical"]:
            output.append(f"\n### Critical Priority ({len(report['recommendations']['critical'])})")
            for i, rec in enumerate(report["recommendations"]["critical"], 1):
                output.append(f"{i}. **{rec['finding']}**")
                output.append(f"   - {rec['mitigation']}")
        
        # High Recommendations
        if report["recommendations"]["high"]:
            output.append(f"\n### High Priority ({len(report['recommendations']['high'])})")
            for i, rec in enumerate(report["recommendations"]["high"], 1):
                output.append(f"{i}. **{rec['finding']}**")
                output.append(f"   - {rec['mitigation']}")
        
        # Medium Recommendations
        if report["recommendations"]["medium"]:
            output.append(f"\n### Medium Priority ({len(report['recommendations']['medium'])})")
            for i, rec in enumerate(report["recommendations"]["medium"], 1):
                output.append(f"{i}. **{rec['finding']}**")
                output.append(f"   - {rec['mitigation']}")
        
        # Low Recommendations
        if report["recommendations"]["low"]:
            output.append(f"\n### Low Priority ({len(report['recommendations']['low'])})")
            for i, rec in enumerate(report["recommendations"]["low"], 1):
                output.append(f"{i}. **{rec['finding']}**")
                output.append(f"   - {rec['mitigation']}")
        
        # Appendices
        output.append("\n## Appendices")
        output.append(f"\n### Methodology")
        output.append(f"- {report['appendices']['methodology']}")
        
        output.append(f"\n### Tools Used")
        for tool in report['appendices']['tools_used']:
            output.append(f"- {tool}")
        
        output.append(f"\n### Limitations")
        output.append(f"- {report['appendices']['limitations']}")
        
        return "\n".join(output)

    def _format_permission_results(self, results: Dict[str, Any]) -> str:
        """Format permission audit results for output."""
        output = ["# File Permission Audit Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Files Checked: {results['files_checked']}")
        
        if results["permission_issues"]:
            output.append(f"\n## Permission Issues: {len(results['permission_issues'])}")
            
            # Group by severity
            by_severity = {}
            for issue in results["permission_issues"]:
                severity = issue["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(issue)
            
            # Display issues by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for issue in by_severity[severity]:
                        output.append(f"\n- **{issue['issue']}**")
                        output.append(f"  - File: {issue['file']}")
                        output.append(f"  - Current Permission: {issue['current_permission']}")
                        output.append(f"  - Recommended Permission: {issue['recommended_permission']}")
        else:
            output.append("\n## No permission issues found")
        
        if results["sensitive_files"]:
            output.append(f"\n## Sensitive Files: {len(results['sensitive_files'])}")
            
            for file in results["sensitive_files"]:
                output.append(f"\n- **{file['file']}**")
                output.append(f"  - Type: {file['type']}")
                output.append(f"  - Contains: {file['contains']}")
                output.append(f"  - Current Permission: {file['current_permission']}")
                output.append(f"  - Recommended Permission: {file['recommended_permission']}")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)

    def _format_secrets_results(self, results: Dict[str, Any]) -> str:
        """Format secrets check results for output."""
        output = ["# Secrets Detection Results"]
        output.append("=" * 50)
        
        output.append(f"\nTarget: {results['target_path']}")
        output.append(f"Files Checked: {results['files_checked']}")
        output.append(f"Risk Level: {results['risk_level'].upper()}")
        
        if results["secrets_found"]:
            output.append(f"\n## Secrets Found: {len(results['secrets_found'])}")
            
            # Group by severity
            by_severity = {}
            for secret in results["secrets_found"]:
                severity = secret["severity"]
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(secret)
            
            # Display secrets by severity (highest first)
            for severity in ["critical", "high", "medium", "low", "info"]:
                if severity in by_severity:
                    output.append(f"\n### {severity.upper()} Severity ({len(by_severity[severity])})")
                    
                    for secret in by_severity[severity]:
                        output.append(f"\n- **{secret['type']}**")
                        output.append(f"  - File: {secret['file']}")
                        output.append(f"  - Line: {secret['line']}")
                        output.append(f"  - Snippet: `{secret['snippet']}`")
                        if "recommendation" in secret:
                            output.append(f"  - Recommendation: {secret['recommendation']}")
        else:
            output.append("\n## No secrets found")
        
        if results["recommendations"]:
            output.append("\n## Recommendations")
            for i, rec in enumerate(results["recommendations"], 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)