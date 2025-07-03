"""
Quality Assurance Tool
Validates outputs, checks completeness, and ensures quality standards
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class QualityAssuranceTool(BaseTool):
    """
    Advanced quality assurance tool for validating outputs, checking completeness,
    and ensuring adherence to quality standards.
    """

    name: str = "quality_assurance"
    description: str = """
    Perform comprehensive quality assurance checks on outputs, validate completeness,
    check accuracy, and ensure adherence to quality standards and requirements.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "validate_output", "check_completeness", "verify_accuracy",
                    "assess_clarity", "check_requirements", "evaluate_quality",
                    "generate_checklist", "perform_audit"
                ],
                "description": "The type of quality assurance check to perform"
            },
            "content": {
                "type": "string",
                "description": "Content to be checked for quality"
            },
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of requirements to check against"
            },
            "quality_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific quality criteria to evaluate"
            },
            "content_type": {
                "type": "string",
                "enum": ["text", "code", "documentation", "analysis", "plan", "report"],
                "description": "Type of content being checked"
            },
            "target_audience": {
                "type": "string",
                "enum": ["technical", "business", "general", "academic"],
                "description": "Target audience for the content"
            },
            "strictness_level": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Level of strictness for quality checks"
            }
        },
        "required": ["action"]
    }

    # Quality standards and checklists
    quality_standards: Dict[str, List[str]] = Field(default_factory=dict)
    quality_history: List[Dict[str, Any]] = Field(default_factory=list)
    common_issues: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_quality_standards()

    def _initialize_quality_standards(self):
        """Initialize quality standards for different content types."""
        
        self.quality_standards = {
            "text": [
                "Clear and coherent structure",
                "Proper grammar and spelling",
                "Appropriate tone and style",
                "Logical flow of ideas",
                "Complete sentences and paragraphs"
            ],
            "code": [
                "Syntactically correct",
                "Follows coding standards",
                "Proper error handling",
                "Adequate comments and documentation",
                "Efficient and readable implementation"
            ],
            "documentation": [
                "Complete coverage of topics",
                "Clear explanations and examples",
                "Proper formatting and structure",
                "Accurate and up-to-date information",
                "Appropriate level of detail"
            ],
            "analysis": [
                "Comprehensive data coverage",
                "Logical reasoning and conclusions",
                "Supporting evidence provided",
                "Clear methodology explained",
                "Actionable insights included"
            ],
            "plan": [
                "Clear objectives and goals",
                "Realistic timelines and milestones",
                "Resource requirements identified",
                "Risk assessment included",
                "Success criteria defined"
            ],
            "report": [
                "Executive summary provided",
                "Key findings highlighted",
                "Data properly presented",
                "Conclusions supported by evidence",
                "Recommendations are actionable"
            ]
        }

    async def execute(
        self,
        action: str,
        content: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        quality_criteria: Optional[List[str]] = None,
        content_type: str = "text",
        target_audience: str = "general",
        strictness_level: str = "medium",
        **kwargs
    ) -> ToolResult:
        """Execute the quality assurance action."""
        
        try:
            if action == "validate_output":
                return await self._validate_output(content, content_type, strictness_level)
            elif action == "check_completeness":
                return await self._check_completeness(content, requirements, content_type)
            elif action == "verify_accuracy":
                return await self._verify_accuracy(content, content_type)
            elif action == "assess_clarity":
                return await self._assess_clarity(content, target_audience)
            elif action == "check_requirements":
                return await self._check_requirements(content, requirements)
            elif action == "evaluate_quality":
                return await self._evaluate_quality(content, quality_criteria, content_type)
            elif action == "generate_checklist":
                return await self._generate_checklist(content_type, target_audience)
            elif action == "perform_audit":
                return await self._perform_audit(content, content_type, requirements)
            else:
                return ToolResult(error=f"Unknown QA action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Quality assurance error: {str(e)}")

    async def _validate_output(self, content: str, content_type: str, strictness_level: str) -> ToolResult:
        """Perform comprehensive validation of output."""
        
        if not content:
            return ToolResult(error="Content is required for validation")

        validation_result = {
            "content_type": content_type,
            "strictness_level": strictness_level,
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0.0,
            "passed_checks": [],
            "failed_checks": [],
            "warnings": [],
            "recommendations": []
        }

        # Get applicable quality standards
        standards = self.quality_standards.get(content_type, self.quality_standards["text"])
        
        # Perform validation checks
        total_checks = 0
        passed_checks = 0

        for standard in standards:
            total_checks += 1
            check_result = self._perform_quality_check(content, standard, content_type)
            
            if check_result["passed"]:
                passed_checks += 1
                validation_result["passed_checks"].append(standard)
            else:
                validation_result["failed_checks"].append({
                    "standard": standard,
                    "issue": check_result["issue"],
                    "severity": check_result["severity"]
                })

        # Calculate overall score
        validation_result["overall_score"] = passed_checks / total_checks if total_checks > 0 else 0.0

        # Add specific checks based on content type
        if content_type == "code":
            code_checks = self._perform_code_specific_checks(content)
            validation_result.update(code_checks)
        elif content_type == "documentation":
            doc_checks = self._perform_documentation_checks(content)
            validation_result.update(doc_checks)

        # Generate recommendations based on failed checks
        validation_result["recommendations"] = self._generate_validation_recommendations(
            validation_result["failed_checks"], strictness_level
        )

        # Store validation history
        self.quality_history.append(validation_result)

        return ToolResult(output=self._format_validation_result(validation_result))

    async def _check_completeness(self, content: str, requirements: Optional[List[str]], content_type: str) -> ToolResult:
        """Check if content meets completeness requirements."""
        
        if not content:
            return ToolResult(error="Content is required for completeness check")

        completeness_result = {
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "completeness_score": 0.0,
            "missing_elements": [],
            "present_elements": [],
            "suggestions": []
        }

        # Use provided requirements or generate based on content type
        if not requirements:
            requirements = self._get_default_requirements(content_type)

        # Check each requirement
        total_requirements = len(requirements)
        met_requirements = 0

        for requirement in requirements:
            if self._requirement_met(content, requirement, content_type):
                met_requirements += 1
                completeness_result["present_elements"].append(requirement)
            else:
                completeness_result["missing_elements"].append(requirement)

        # Calculate completeness score
        completeness_result["completeness_score"] = met_requirements / total_requirements if total_requirements > 0 else 0.0

        # Generate suggestions for missing elements
        completeness_result["suggestions"] = self._generate_completeness_suggestions(
            completeness_result["missing_elements"], content_type
        )

        return ToolResult(output=self._format_completeness_result(completeness_result))

    async def _verify_accuracy(self, content: str, content_type: str) -> ToolResult:
        """Verify accuracy of content."""
        
        if not content:
            return ToolResult(error="Content is required for accuracy verification")

        accuracy_result = {
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "accuracy_score": 0.0,
            "verified_facts": [],
            "questionable_claims": [],
            "verification_notes": []
        }

        # Perform accuracy checks based on content type
        if content_type == "code":
            accuracy_result.update(self._verify_code_accuracy(content))
        elif content_type == "analysis":
            accuracy_result.update(self._verify_analysis_accuracy(content))
        elif content_type == "documentation":
            accuracy_result.update(self._verify_documentation_accuracy(content))
        else:
            accuracy_result.update(self._verify_general_accuracy(content))

        return ToolResult(output=self._format_accuracy_result(accuracy_result))

    async def _assess_clarity(self, content: str, target_audience: str) -> ToolResult:
        """Assess clarity and readability of content."""
        
        if not content:
            return ToolResult(error="Content is required for clarity assessment")

        clarity_result = {
            "target_audience": target_audience,
            "timestamp": datetime.now().isoformat(),
            "clarity_score": 0.0,
            "readability_metrics": {},
            "clarity_issues": [],
            "improvement_suggestions": []
        }

        # Calculate readability metrics
        clarity_result["readability_metrics"] = self._calculate_readability_metrics(content)
        
        # Assess clarity based on target audience
        clarity_issues = self._identify_clarity_issues(content, target_audience)
        clarity_result["clarity_issues"] = clarity_issues

        # Calculate overall clarity score
        clarity_result["clarity_score"] = self._calculate_clarity_score(
            clarity_result["readability_metrics"], clarity_issues, target_audience
        )

        # Generate improvement suggestions
        clarity_result["improvement_suggestions"] = self._generate_clarity_suggestions(
            clarity_issues, target_audience
        )

        return ToolResult(output=self._format_clarity_result(clarity_result))

    async def _check_requirements(self, content: str, requirements: List[str]) -> ToolResult:
        """Check if content meets specific requirements."""
        
        if not content or not requirements:
            return ToolResult(error="Content and requirements are required")

        requirements_result = {
            "timestamp": datetime.now().isoformat(),
            "total_requirements": len(requirements),
            "met_requirements": 0,
            "compliance_score": 0.0,
            "requirement_status": [],
            "non_compliance_issues": []
        }

        # Check each requirement
        for requirement in requirements:
            status = self._check_single_requirement(content, requirement)
            requirements_result["requirement_status"].append(status)
            
            if status["met"]:
                requirements_result["met_requirements"] += 1
            else:
                requirements_result["non_compliance_issues"].append({
                    "requirement": requirement,
                    "issue": status["issue"]
                })

        # Calculate compliance score
        requirements_result["compliance_score"] = (
            requirements_result["met_requirements"] / requirements_result["total_requirements"]
        )

        return ToolResult(output=self._format_requirements_result(requirements_result))

    async def _evaluate_quality(self, content: str, quality_criteria: Optional[List[str]], content_type: str) -> ToolResult:
        """Evaluate overall quality against specific criteria."""
        
        if not content:
            return ToolResult(error="Content is required for quality evaluation")

        # Use provided criteria or default for content type
        if not quality_criteria:
            quality_criteria = self.quality_standards.get(content_type, self.quality_standards["text"])

        quality_result = {
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "overall_quality_score": 0.0,
            "criteria_scores": {},
            "strengths": [],
            "weaknesses": [],
            "quality_recommendations": []
        }

        # Evaluate each criterion
        total_score = 0.0
        for criterion in quality_criteria:
            score = self._evaluate_criterion(content, criterion, content_type)
            quality_result["criteria_scores"][criterion] = score
            total_score += score

        # Calculate overall quality score
        quality_result["overall_quality_score"] = total_score / len(quality_criteria) if quality_criteria else 0.0

        # Identify strengths and weaknesses
        quality_result["strengths"] = [
            criterion for criterion, score in quality_result["criteria_scores"].items() if score >= 0.8
        ]
        quality_result["weaknesses"] = [
            criterion for criterion, score in quality_result["criteria_scores"].items() if score < 0.6
        ]

        # Generate quality recommendations
        quality_result["quality_recommendations"] = self._generate_quality_recommendations(
            quality_result["weaknesses"], content_type
        )

        return ToolResult(output=self._format_quality_result(quality_result))

    async def _generate_checklist(self, content_type: str, target_audience: str) -> ToolResult:
        """Generate a quality checklist for the specified content type and audience."""
        
        checklist = {
            "content_type": content_type,
            "target_audience": target_audience,
            "timestamp": datetime.now().isoformat(),
            "checklist_items": [],
            "priority_items": [],
            "optional_items": []
        }

        # Get base checklist for content type
        base_items = self.quality_standards.get(content_type, self.quality_standards["text"])
        
        # Customize for target audience
        customized_items = self._customize_checklist_for_audience(base_items, target_audience)
        
        # Categorize items by priority
        for item in customized_items:
            priority = self._determine_item_priority(item, content_type, target_audience)
            
            checklist["checklist_items"].append({
                "item": item,
                "priority": priority,
                "description": self._get_item_description(item, content_type)
            })
            
            if priority == "high":
                checklist["priority_items"].append(item)
            elif priority == "low":
                checklist["optional_items"].append(item)

        return ToolResult(output=self._format_checklist(checklist))

    async def _perform_audit(self, content: str, content_type: str, requirements: Optional[List[str]]) -> ToolResult:
        """Perform comprehensive quality audit."""
        
        if not content:
            return ToolResult(error="Content is required for audit")

        audit_result = {
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "audit_score": 0.0,
            "validation_results": {},
            "completeness_results": {},
            "accuracy_results": {},
            "clarity_results": {},
            "overall_assessment": "",
            "critical_issues": [],
            "recommendations": []
        }

        # Perform all quality checks
        validation = await self._validate_output(content, content_type, "high")
        completeness = await self._check_completeness(content, requirements, content_type)
        accuracy = await self._verify_accuracy(content, content_type)
        clarity = await self._assess_clarity(content, "general")

        # Extract results (simplified - would parse actual ToolResult outputs)
        audit_result["validation_results"] = {"score": 0.8}  # Mock data
        audit_result["completeness_results"] = {"score": 0.7}
        audit_result["accuracy_results"] = {"score": 0.9}
        audit_result["clarity_results"] = {"score": 0.75}

        # Calculate overall audit score
        scores = [
            audit_result["validation_results"]["score"],
            audit_result["completeness_results"]["score"],
            audit_result["accuracy_results"]["score"],
            audit_result["clarity_results"]["score"]
        ]
        audit_result["audit_score"] = sum(scores) / len(scores)

        # Generate overall assessment
        audit_result["overall_assessment"] = self._generate_overall_assessment(audit_result["audit_score"])

        # Identify critical issues
        audit_result["critical_issues"] = self._identify_critical_issues(audit_result)

        # Generate comprehensive recommendations
        audit_result["recommendations"] = self._generate_audit_recommendations(audit_result)

        return ToolResult(output=self._format_audit_result(audit_result))

    # Helper methods for quality checks

    def _perform_quality_check(self, content: str, standard: str, content_type: str) -> Dict[str, Any]:
        """Perform a specific quality check."""
        
        result = {"passed": False, "issue": "", "severity": "medium"}
        
        # Basic checks based on standard
        if "grammar" in standard.lower():
            result = self._check_grammar(content)
        elif "structure" in standard.lower():
            result = self._check_structure(content, content_type)
        elif "clarity" in standard.lower():
            result = self._check_basic_clarity(content)
        elif "completeness" in standard.lower():
            result = self._check_basic_completeness(content, content_type)
        else:
            # Default check
            result["passed"] = len(content) > 10  # Basic length check
            if not result["passed"]:
                result["issue"] = "Content too short"
        
        return result

    def _check_grammar(self, content: str) -> Dict[str, Any]:
        """Basic grammar check."""
        # Simplified grammar check
        issues = []
        
        # Check for basic punctuation
        if not re.search(r'[.!?]$', content.strip()):
            issues.append("Missing ending punctuation")
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                issues.append("Sentence doesn't start with capital letter")
                break
        
        return {
            "passed": len(issues) == 0,
            "issue": "; ".join(issues) if issues else "",
            "severity": "low" if len(issues) <= 1 else "medium"
        }

    def _check_structure(self, content: str, content_type: str) -> Dict[str, Any]:
        """Check content structure."""
        issues = []
        
        if content_type == "documentation":
            # Check for headings
            if not re.search(r'^#+ ', content, re.MULTILINE):
                issues.append("No headings found")
        
        # Check paragraph structure
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2 and len(content) > 200:
            issues.append("Content lacks paragraph breaks")
        
        return {
            "passed": len(issues) == 0,
            "issue": "; ".join(issues) if issues else "",
            "severity": "medium"
        }

    def _check_basic_clarity(self, content: str) -> Dict[str, Any]:
        """Basic clarity check."""
        issues = []
        
        # Check sentence length
        sentences = re.split(r'[.!?]+', content)
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        
        if len(long_sentences) > len(sentences) * 0.3:
            issues.append("Too many long sentences")
        
        # Check for jargon (simplified)
        jargon_words = ['utilize', 'facilitate', 'implement', 'leverage']
        jargon_count = sum(content.lower().count(word) for word in jargon_words)
        
        if jargon_count > 5:
            issues.append("Excessive use of jargon")
        
        return {
            "passed": len(issues) == 0,
            "issue": "; ".join(issues) if issues else "",
            "severity": "low"
        }

    def _check_basic_completeness(self, content: str, content_type: str) -> Dict[str, Any]:
        """Basic completeness check."""
        issues = []
        
        # Minimum length requirements
        min_lengths = {
            "text": 50,
            "documentation": 200,
            "analysis": 300,
            "report": 500,
            "plan": 150
        }
        
        min_length = min_lengths.get(content_type, 50)
        if len(content) < min_length:
            issues.append(f"Content too short (minimum {min_length} characters)")
        
        return {
            "passed": len(issues) == 0,
            "issue": "; ".join(issues) if issues else "",
            "severity": "high" if issues else "low"
        }

    # Additional helper methods for formatting results

    def _format_validation_result(self, result: Dict[str, Any]) -> str:
        """Format validation result for output."""
        output = ["Quality Validation Results"]
        output.append("=" * 40)
        
        output.append(f"\nOverall Score: {result['overall_score']:.2f}")
        output.append(f"Content Type: {result['content_type']}")
        output.append(f"Strictness Level: {result['strictness_level']}")
        
        if result["passed_checks"]:
            output.append(f"\nPassed Checks ({len(result['passed_checks'])}):")
            for check in result["passed_checks"]:
                output.append(f"  ✓ {check}")
        
        if result["failed_checks"]:
            output.append(f"\nFailed Checks ({len(result['failed_checks'])}):")
            for check in result["failed_checks"]:
                output.append(f"  ✗ {check['standard']}: {check['issue']}")
        
        if result["recommendations"]:
            output.append("\nRecommendations:")
            for rec in result["recommendations"]:
                output.append(f"  → {rec}")
        
        return "\n".join(output)

    def _format_completeness_result(self, result: Dict[str, Any]) -> str:
        """Format completeness result for output."""
        output = ["Completeness Check Results"]
        output.append("=" * 40)
        
        output.append(f"\nCompleteness Score: {result['completeness_score']:.2f}")
        
        if result["present_elements"]:
            output.append(f"\nPresent Elements ({len(result['present_elements'])}):")
            for element in result["present_elements"]:
                output.append(f"  ✓ {element}")
        
        if result["missing_elements"]:
            output.append(f"\nMissing Elements ({len(result['missing_elements'])}):")
            for element in result["missing_elements"]:
                output.append(f"  ✗ {element}")
        
        if result["suggestions"]:
            output.append("\nSuggestions:")
            for suggestion in result["suggestions"]:
                output.append(f"  💡 {suggestion}")
        
        return "\n".join(output)

    # Additional helper methods would be implemented for:
    # - _get_default_requirements
    # - _requirement_met
    # - _generate_completeness_suggestions
    # - _verify_code_accuracy
    # - _verify_analysis_accuracy
    # - _calculate_readability_metrics
    # - _identify_clarity_issues
    # - etc.

    def _get_default_requirements(self, content_type: str) -> List[str]:
        """Get default requirements for content type."""
        requirements_map = {
            "text": ["Has clear introduction", "Contains main content", "Has conclusion"],
            "code": ["Syntactically correct", "Has comments", "Handles errors"],
            "documentation": ["Has overview", "Contains examples", "Explains usage"],
            "analysis": ["States methodology", "Presents findings", "Draws conclusions"],
            "plan": ["Defines objectives", "Lists steps", "Includes timeline"],
            "report": ["Has executive summary", "Presents data", "Makes recommendations"]
        }
        return requirements_map.get(content_type, requirements_map["text"])

    def _requirement_met(self, content: str, requirement: str, content_type: str) -> bool:
        """Check if a specific requirement is met."""
        # Simplified requirement checking
        requirement_lower = requirement.lower()
        content_lower = content.lower()
        
        if "introduction" in requirement_lower:
            return len(content) > 100 and any(word in content_lower for word in ["introduction", "overview", "begin"])
        elif "conclusion" in requirement_lower:
            return any(word in content_lower for word in ["conclusion", "summary", "finally", "in conclusion"])
        elif "example" in requirement_lower:
            return "example" in content_lower or "for instance" in content_lower
        else:
            # Default: check if requirement keywords appear in content
            keywords = requirement_lower.split()
            return any(keyword in content_lower for keyword in keywords)

    def _generate_completeness_suggestions(self, missing_elements: List[str], content_type: str) -> List[str]:
        """Generate suggestions for missing elements."""
        suggestions = []
        
        for element in missing_elements:
            if "introduction" in element.lower():
                suggestions.append("Add an introduction section to provide context")
            elif "conclusion" in element.lower():
                suggestions.append("Include a conclusion to summarize key points")
            elif "example" in element.lower():
                suggestions.append("Add examples to illustrate concepts")
            else:
                suggestions.append(f"Consider adding content related to: {element}")
        
        return suggestions