"""
Context Analyzer Tool
Analyzes context, understands nuanced requirements, and provides contextual insights
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class ContextAnalyzerTool(BaseTool):
    """
    Advanced context analysis tool for understanding nuanced requirements,
    analyzing implicit needs, and providing contextual insights.
    """

    name: str = "context_analyzer"
    description: str = """
    Analyze context, understand implicit requirements, identify nuanced needs,
    and provide insights about the situational context of requests and tasks.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "analyze_request", "identify_implicit_needs", "assess_complexity",
                    "understand_intent", "extract_constraints", "identify_stakeholders",
                    "analyze_domain", "suggest_clarifications", "map_dependencies"
                ],
                "description": "The type of context analysis to perform"
            },
            "text": {
                "type": "string",
                "description": "Text to analyze for context"
            },
            "domain": {
                "type": "string",
                "enum": ["technical", "business", "academic", "creative", "general"],
                "description": "Domain context for the analysis"
            },
            "analysis_depth": {
                "type": "string",
                "enum": ["surface", "moderate", "deep", "comprehensive"],
                "description": "Depth of analysis to perform"
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific areas to focus the analysis on"
            },
            "previous_context": {
                "type": "string",
                "description": "Previous context or conversation history"
            }
        },
        "required": ["action", "text"]
    }

    # Context analysis data
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    domain_patterns: Dict[str, List[str]] = Field(default_factory=dict)
    intent_patterns: Dict[str, List[str]] = Field(default_factory=dict)
    complexity_indicators: Dict[str, List[str]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_analysis_patterns()

    def _initialize_analysis_patterns(self):
        """Initialize patterns for context analysis."""
        
        # Domain-specific patterns
        self.domain_patterns = {
            "technical": [
                "implement", "develop", "code", "system", "architecture", "API",
                "database", "algorithm", "performance", "scalability", "security"
            ],
            "business": [
                "strategy", "revenue", "market", "customer", "ROI", "stakeholder",
                "process", "efficiency", "cost", "profit", "competitive", "growth"
            ],
            "academic": [
                "research", "study", "analysis", "methodology", "hypothesis",
                "literature", "peer-review", "citation", "theory", "empirical"
            ],
            "creative": [
                "design", "creative", "artistic", "visual", "aesthetic", "brand",
                "concept", "inspiration", "innovative", "original", "style"
            ]
        }

        # Intent patterns
        self.intent_patterns = {
            "information_seeking": [
                "what is", "how does", "explain", "describe", "tell me about",
                "I need to know", "help me understand", "clarify"
            ],
            "problem_solving": [
                "solve", "fix", "resolve", "troubleshoot", "debug", "issue",
                "problem", "error", "not working", "broken"
            ],
            "creation": [
                "create", "build", "make", "develop", "generate", "produce",
                "design", "construct", "write", "compose"
            ],
            "analysis": [
                "analyze", "evaluate", "assess", "compare", "review", "examine",
                "investigate", "study", "research"
            ],
            "planning": [
                "plan", "strategy", "roadmap", "schedule", "timeline", "organize",
                "prepare", "outline", "structure"
            ]
        }

        # Complexity indicators
        self.complexity_indicators = {
            "high": [
                "integrate", "multiple systems", "complex", "enterprise", "scalable",
                "distributed", "real-time", "high-performance", "mission-critical"
            ],
            "medium": [
                "several", "various", "different", "multiple", "coordinate",
                "manage", "organize", "comprehensive"
            ],
            "low": [
                "simple", "basic", "straightforward", "quick", "easy", "single"
            ]
        }

    async def execute(
        self,
        action: str,
        text: str,
        domain: str = "general",
        analysis_depth: str = "moderate",
        focus_areas: Optional[List[str]] = None,
        previous_context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the context analysis action."""
        
        try:
            if action == "analyze_request":
                return await self._analyze_request(text, domain, analysis_depth, previous_context)
            elif action == "identify_implicit_needs":
                return await self._identify_implicit_needs(text, domain)
            elif action == "assess_complexity":
                return await self._assess_complexity(text, domain)
            elif action == "understand_intent":
                return await self._understand_intent(text, previous_context)
            elif action == "extract_constraints":
                return await self._extract_constraints(text)
            elif action == "identify_stakeholders":
                return await self._identify_stakeholders(text, domain)
            elif action == "analyze_domain":
                return await self._analyze_domain(text)
            elif action == "suggest_clarifications":
                return await self._suggest_clarifications(text, focus_areas)
            elif action == "map_dependencies":
                return await self._map_dependencies(text, domain)
            else:
                return ToolResult(error=f"Unknown context analysis action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Context analyzer error: {str(e)}")

    async def _analyze_request(
        self, 
        text: str, 
        domain: str, 
        analysis_depth: str,
        previous_context: Optional[str]
    ) -> ToolResult:
        """Perform comprehensive request analysis."""
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "analysis_depth": analysis_depth,
            "request_text": text,
            "analysis_results": {}
        }

        # Core analysis components
        analysis["analysis_results"] = {
            "intent": self._analyze_intent(text),
            "complexity": self._analyze_complexity_level(text),
            "domain_classification": self._classify_domain(text),
            "key_entities": self._extract_key_entities(text),
            "requirements": self._extract_requirements(text),
            "constraints": self._identify_constraints(text),
            "success_criteria": self._infer_success_criteria(text),
            "potential_challenges": self._identify_potential_challenges(text),
            "recommended_approach": self._recommend_approach(text, domain)
        }

        # Add contextual analysis if previous context provided
        if previous_context:
            analysis["analysis_results"]["contextual_continuity"] = self._analyze_contextual_continuity(
                text, previous_context
            )

        # Deep analysis if requested
        if analysis_depth in ["deep", "comprehensive"]:
            analysis["analysis_results"].update({
                "implicit_assumptions": self._identify_implicit_assumptions(text),
                "stakeholder_analysis": self._analyze_stakeholders(text, domain),
                "risk_assessment": self._assess_risks(text),
                "resource_implications": self._analyze_resource_implications(text)
            })

        # Store analysis
        self.context_history.append(analysis)

        return ToolResult(output=self._format_request_analysis(analysis))

    async def _identify_implicit_needs(self, text: str, domain: str) -> ToolResult:
        """Identify implicit needs and unstated requirements."""
        
        implicit_analysis = {
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "explicit_needs": [],
            "implicit_needs": [],
            "unstated_assumptions": [],
            "hidden_requirements": [],
            "contextual_needs": []
        }

        # Extract explicit needs
        implicit_analysis["explicit_needs"] = self._extract_explicit_needs(text)

        # Identify implicit needs based on domain and context
        implicit_analysis["implicit_needs"] = self._infer_implicit_needs(text, domain)

        # Identify unstated assumptions
        implicit_analysis["unstated_assumptions"] = self._identify_assumptions(text)

        # Find hidden requirements
        implicit_analysis["hidden_requirements"] = self._discover_hidden_requirements(text, domain)

        # Analyze contextual needs
        implicit_analysis["contextual_needs"] = self._analyze_contextual_needs(text, domain)

        return ToolResult(output=self._format_implicit_needs_analysis(implicit_analysis))

    async def _assess_complexity(self, text: str, domain: str) -> ToolResult:
        """Assess the complexity of the request."""
        
        complexity_assessment = {
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "overall_complexity": "medium",
            "complexity_factors": {},
            "complexity_score": 0.0,
            "complexity_breakdown": {},
            "simplification_opportunities": [],
            "complexity_management_strategies": []
        }

        # Analyze different complexity dimensions
        complexity_factors = {
            "technical_complexity": self._assess_technical_complexity(text),
            "scope_complexity": self._assess_scope_complexity(text),
            "integration_complexity": self._assess_integration_complexity(text),
            "time_complexity": self._assess_time_complexity(text),
            "resource_complexity": self._assess_resource_complexity(text)
        }

        complexity_assessment["complexity_factors"] = complexity_factors

        # Calculate overall complexity score
        complexity_assessment["complexity_score"] = sum(complexity_factors.values()) / len(complexity_factors)

        # Determine overall complexity level
        if complexity_assessment["complexity_score"] >= 0.8:
            complexity_assessment["overall_complexity"] = "very_high"
        elif complexity_assessment["complexity_score"] >= 0.6:
            complexity_assessment["overall_complexity"] = "high"
        elif complexity_assessment["complexity_score"] >= 0.4:
            complexity_assessment["overall_complexity"] = "medium"
        else:
            complexity_assessment["overall_complexity"] = "low"

        # Provide complexity breakdown
        complexity_assessment["complexity_breakdown"] = self._break_down_complexity(text, complexity_factors)

        # Suggest simplification opportunities
        complexity_assessment["simplification_opportunities"] = self._identify_simplification_opportunities(
            text, complexity_factors
        )

        # Recommend complexity management strategies
        complexity_assessment["complexity_management_strategies"] = self._recommend_complexity_strategies(
            complexity_assessment["overall_complexity"], domain
        )

        return ToolResult(output=self._format_complexity_assessment(complexity_assessment))

    async def _understand_intent(self, text: str, previous_context: Optional[str]) -> ToolResult:
        """Understand the intent behind the request."""
        
        intent_analysis = {
            "timestamp": datetime.now().isoformat(),
            "primary_intent": "",
            "secondary_intents": [],
            "intent_confidence": 0.0,
            "intent_indicators": [],
            "goal_hierarchy": [],
            "success_indicators": [],
            "intent_evolution": []
        }

        # Identify primary intent
        intent_analysis["primary_intent"] = self._identify_primary_intent(text)

        # Identify secondary intents
        intent_analysis["secondary_intents"] = self._identify_secondary_intents(text)

        # Calculate intent confidence
        intent_analysis["intent_confidence"] = self._calculate_intent_confidence(text, intent_analysis["primary_intent"])

        # Extract intent indicators
        intent_analysis["intent_indicators"] = self._extract_intent_indicators(text)

        # Build goal hierarchy
        intent_analysis["goal_hierarchy"] = self._build_goal_hierarchy(text)

        # Define success indicators
        intent_analysis["success_indicators"] = self._define_success_indicators(text, intent_analysis["primary_intent"])

        # Analyze intent evolution if previous context available
        if previous_context:
            intent_analysis["intent_evolution"] = self._analyze_intent_evolution(text, previous_context)

        return ToolResult(output=self._format_intent_analysis(intent_analysis))

    async def _extract_constraints(self, text: str) -> ToolResult:
        """Extract constraints and limitations from the request."""
        
        constraints_analysis = {
            "timestamp": datetime.now().isoformat(),
            "explicit_constraints": [],
            "implicit_constraints": [],
            "constraint_types": {},
            "constraint_priorities": {},
            "constraint_conflicts": [],
            "constraint_relaxation_options": []
        }

        # Extract explicit constraints
        constraints_analysis["explicit_constraints"] = self._extract_explicit_constraints(text)

        # Infer implicit constraints
        constraints_analysis["implicit_constraints"] = self._infer_implicit_constraints(text)

        # Categorize constraints by type
        all_constraints = constraints_analysis["explicit_constraints"] + constraints_analysis["implicit_constraints"]
        constraints_analysis["constraint_types"] = self._categorize_constraints(all_constraints)

        # Prioritize constraints
        constraints_analysis["constraint_priorities"] = self._prioritize_constraints(all_constraints, text)

        # Identify constraint conflicts
        constraints_analysis["constraint_conflicts"] = self._identify_constraint_conflicts(all_constraints)

        # Suggest constraint relaxation options
        constraints_analysis["constraint_relaxation_options"] = self._suggest_constraint_relaxations(
            all_constraints, constraints_analysis["constraint_conflicts"]
        )

        return ToolResult(output=self._format_constraints_analysis(constraints_analysis))

    async def _identify_stakeholders(self, text: str, domain: str) -> ToolResult:
        """Identify stakeholders and their interests."""
        
        stakeholder_analysis = {
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "primary_stakeholders": [],
            "secondary_stakeholders": [],
            "stakeholder_interests": {},
            "stakeholder_influence": {},
            "potential_conflicts": [],
            "engagement_strategies": []
        }

        # Identify stakeholders based on text and domain
        all_stakeholders = self._extract_stakeholders(text, domain)
        
        # Categorize stakeholders
        stakeholder_analysis["primary_stakeholders"] = [s for s in all_stakeholders if s.get("importance") == "high"]
        stakeholder_analysis["secondary_stakeholders"] = [s for s in all_stakeholders if s.get("importance") != "high"]

        # Analyze stakeholder interests
        for stakeholder in all_stakeholders:
            stakeholder_analysis["stakeholder_interests"][stakeholder["name"]] = stakeholder.get("interests", [])
            stakeholder_analysis["stakeholder_influence"][stakeholder["name"]] = stakeholder.get("influence", "medium")

        # Identify potential conflicts
        stakeholder_analysis["potential_conflicts"] = self._identify_stakeholder_conflicts(all_stakeholders)

        # Suggest engagement strategies
        stakeholder_analysis["engagement_strategies"] = self._suggest_engagement_strategies(all_stakeholders, domain)

        return ToolResult(output=self._format_stakeholder_analysis(stakeholder_analysis))

    async def _analyze_domain(self, text: str) -> ToolResult:
        """Analyze the domain context of the request."""
        
        domain_analysis = {
            "timestamp": datetime.now().isoformat(),
            "detected_domains": [],
            "primary_domain": "",
            "domain_confidence": 0.0,
            "domain_indicators": [],
            "cross_domain_elements": [],
            "domain_specific_considerations": [],
            "recommended_expertise": []
        }

        # Detect all possible domains
        domain_analysis["detected_domains"] = self._detect_domains(text)

        # Identify primary domain
        if domain_analysis["detected_domains"]:
            domain_analysis["primary_domain"] = domain_analysis["detected_domains"][0]["domain"]
            domain_analysis["domain_confidence"] = domain_analysis["detected_domains"][0]["confidence"]

        # Extract domain indicators
        domain_analysis["domain_indicators"] = self._extract_domain_indicators(text)

        # Identify cross-domain elements
        domain_analysis["cross_domain_elements"] = self._identify_cross_domain_elements(
            domain_analysis["detected_domains"]
        )

        # Provide domain-specific considerations
        domain_analysis["domain_specific_considerations"] = self._get_domain_considerations(
            domain_analysis["primary_domain"]
        )

        # Recommend required expertise
        domain_analysis["recommended_expertise"] = self._recommend_expertise(
            domain_analysis["detected_domains"], text
        )

        return ToolResult(output=self._format_domain_analysis(domain_analysis))

    async def _suggest_clarifications(self, text: str, focus_areas: Optional[List[str]]) -> ToolResult:
        """Suggest clarifying questions to better understand the request."""
        
        clarification_analysis = {
            "timestamp": datetime.now().isoformat(),
            "focus_areas": focus_areas or [],
            "ambiguous_elements": [],
            "missing_information": [],
            "clarifying_questions": [],
            "assumption_validations": [],
            "scope_clarifications": []
        }

        # Identify ambiguous elements
        clarification_analysis["ambiguous_elements"] = self._identify_ambiguous_elements(text)

        # Identify missing information
        clarification_analysis["missing_information"] = self._identify_missing_information(text, focus_areas)

        # Generate clarifying questions
        clarification_analysis["clarifying_questions"] = self._generate_clarifying_questions(
            text, clarification_analysis["ambiguous_elements"], clarification_analysis["missing_information"]
        )

        # Suggest assumption validations
        clarification_analysis["assumption_validations"] = self._suggest_assumption_validations(text)

        # Suggest scope clarifications
        clarification_analysis["scope_clarifications"] = self._suggest_scope_clarifications(text)

        return ToolResult(output=self._format_clarification_suggestions(clarification_analysis))

    async def _map_dependencies(self, text: str, domain: str) -> ToolResult:
        """Map dependencies and relationships in the request."""
        
        dependency_analysis = {
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "internal_dependencies": [],
            "external_dependencies": [],
            "dependency_types": {},
            "critical_path": [],
            "dependency_risks": [],
            "mitigation_strategies": []
        }

        # Identify internal dependencies
        dependency_analysis["internal_dependencies"] = self._identify_internal_dependencies(text)

        # Identify external dependencies
        dependency_analysis["external_dependencies"] = self._identify_external_dependencies(text, domain)

        # Categorize dependencies by type
        all_dependencies = dependency_analysis["internal_dependencies"] + dependency_analysis["external_dependencies"]
        dependency_analysis["dependency_types"] = self._categorize_dependencies(all_dependencies)

        # Identify critical path
        dependency_analysis["critical_path"] = self._identify_critical_path(all_dependencies)

        # Assess dependency risks
        dependency_analysis["dependency_risks"] = self._assess_dependency_risks(all_dependencies)

        # Suggest mitigation strategies
        dependency_analysis["mitigation_strategies"] = self._suggest_dependency_mitigations(
            dependency_analysis["dependency_risks"]
        )

        return ToolResult(output=self._format_dependency_analysis(dependency_analysis))

    # Helper methods for analysis

    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze the intent of the text."""
        intent_scores = {}
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                intent_scores[intent] = score / len(patterns)
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1]) if intent_scores else ("unknown", 0.0)
        
        return {
            "primary": primary_intent[0],
            "confidence": primary_intent[1],
            "all_scores": intent_scores
        }

    def _analyze_complexity_level(self, text: str) -> str:
        """Analyze the complexity level of the request."""
        text_lower = text.lower()
        
        complexity_scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            complexity_scores[level] = score
        
        if complexity_scores["high"] > 0:
            return "high"
        elif complexity_scores["medium"] > complexity_scores["low"]:
            return "medium"
        else:
            return "low"

    def _classify_domain(self, text: str) -> Dict[str, float]:
        """Classify the domain of the text."""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                domain_scores[domain] = score / len(patterns)
        
        return domain_scores

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from the text."""
        # Simplified entity extraction
        entities = []
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized_words)
        
        # Extract quoted strings
        quoted_strings = re.findall(r'"([^"]*)"', text)
        entities.extend(quoted_strings)
        
        # Extract technical terms (words with underscores or camelCase)
        technical_terms = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z]+\b', text)
        entities.extend(technical_terms)
        
        return list(set(entities))

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from the text."""
        requirements = []
        
        # Look for requirement indicators
        requirement_patterns = [
            r'(?:must|should|need to|required to|have to)\s+([^.!?]+)',
            r'(?:requirement|spec|specification):\s*([^.!?]+)',
            r'(?:I need|we need|it needs)\s+([^.!?]+)'
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            requirements.extend(matches)
        
        return [req.strip() for req in requirements if req.strip()]

    def _identify_constraints(self, text: str) -> List[str]:
        """Identify constraints in the text."""
        constraints = []
        
        # Look for constraint indicators
        constraint_patterns = [
            r'(?:cannot|can\'t|unable to|not allowed|forbidden)\s+([^.!?]+)',
            r'(?:within|under|less than|maximum|minimum)\s+([^.!?]+)',
            r'(?:budget|time|resource)\s+(?:constraint|limit|restriction):\s*([^.!?]+)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            constraints.extend(matches)
        
        return [constraint.strip() for constraint in constraints if constraint.strip()]

    def _infer_success_criteria(self, text: str) -> List[str]:
        """Infer success criteria from the text."""
        criteria = []
        
        # Look for success indicators
        success_patterns = [
            r'(?:success|successful|complete|finished|done)\s+(?:when|if)\s+([^.!?]+)',
            r'(?:goal|objective|target):\s*([^.!?]+)',
            r'(?:expect|expecting|should result in)\s+([^.!?]+)'
        ]
        
        for pattern in success_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend(matches)
        
        # Add default criteria based on intent
        intent_analysis = self._analyze_intent(text)
        if intent_analysis["primary"] == "creation":
            criteria.append("Successfully created the requested item")
        elif intent_analysis["primary"] == "problem_solving":
            criteria.append("Problem is resolved and functioning correctly")
        
        return [criterion.strip() for criterion in criteria if criterion.strip()]

    # Formatting methods

    def _format_request_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format request analysis for output."""
        output = ["Request Analysis"]
        output.append("=" * 30)
        
        results = analysis["analysis_results"]
        
        output.append(f"\nDomain: {analysis['domain']}")
        output.append(f"Analysis Depth: {analysis['analysis_depth']}")
        
        if results.get("intent"):
            output.append(f"\nPrimary Intent: {results['intent']['primary']} (confidence: {results['intent']['confidence']:.2f})")
        
        if results.get("complexity"):
            output.append(f"Complexity Level: {results['complexity']}")
        
        if results.get("key_entities"):
            output.append(f"\nKey Entities: {', '.join(results['key_entities'][:5])}")
        
        if results.get("requirements"):
            output.append("\nRequirements:")
            for req in results["requirements"][:3]:
                output.append(f"  • {req}")
        
        if results.get("constraints"):
            output.append("\nConstraints:")
            for constraint in results["constraints"][:3]:
                output.append(f"  • {constraint}")
        
        if results.get("recommended_approach"):
            output.append(f"\nRecommended Approach: {results['recommended_approach']}")
        
        return "\n".join(output)

    def _format_implicit_needs_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format implicit needs analysis for output."""
        output = ["Implicit Needs Analysis"]
        output.append("=" * 35)
        
        if analysis["explicit_needs"]:
            output.append("\nExplicit Needs:")
            for need in analysis["explicit_needs"]:
                output.append(f"  ✓ {need}")
        
        if analysis["implicit_needs"]:
            output.append("\nImplicit Needs:")
            for need in analysis["implicit_needs"]:
                output.append(f"  💡 {need}")
        
        if analysis["unstated_assumptions"]:
            output.append("\nUnstated Assumptions:")
            for assumption in analysis["unstated_assumptions"]:
                output.append(f"  🤔 {assumption}")
        
        if analysis["hidden_requirements"]:
            output.append("\nHidden Requirements:")
            for req in analysis["hidden_requirements"]:
                output.append(f"  🔍 {req}")
        
        return "\n".join(output)

    def _format_complexity_assessment(self, assessment: Dict[str, Any]) -> str:
        """Format complexity assessment for output."""
        output = ["Complexity Assessment"]
        output.append("=" * 30)
        
        output.append(f"\nOverall Complexity: {assessment['overall_complexity']}")
        output.append(f"Complexity Score: {assessment['complexity_score']:.2f}")
        
        if assessment["complexity_factors"]:
            output.append("\nComplexity Factors:")
            for factor, score in assessment["complexity_factors"].items():
                output.append(f"  {factor}: {score:.2f}")
        
        if assessment["simplification_opportunities"]:
            output.append("\nSimplification Opportunities:")
            for opportunity in assessment["simplification_opportunities"]:
                output.append(f"  💡 {opportunity}")
        
        if assessment["complexity_management_strategies"]:
            output.append("\nManagement Strategies:")
            for strategy in assessment["complexity_management_strategies"]:
                output.append(f"  📋 {strategy}")
        
        return "\n".join(output)

    # Additional helper methods would be implemented for:
    # - _assess_technical_complexity
    # - _assess_scope_complexity
    # - _identify_primary_intent
    # - _extract_explicit_constraints
    # - _extract_stakeholders
    # - _detect_domains
    # - etc.

    def _assess_technical_complexity(self, text: str) -> float:
        """Assess technical complexity of the request."""
        technical_indicators = [
            "API", "database", "algorithm", "architecture", "integration",
            "performance", "scalability", "security", "distributed", "real-time"
        ]
        
        text_lower = text.lower()
        complexity_score = sum(1 for indicator in technical_indicators if indicator in text_lower)
        return min(complexity_score / len(technical_indicators), 1.0)

    def _assess_scope_complexity(self, text: str) -> float:
        """Assess scope complexity of the request."""
        scope_indicators = [
            "multiple", "various", "different", "several", "many", "all",
            "comprehensive", "complete", "entire", "full"
        ]
        
        text_lower = text.lower()
        complexity_score = sum(1 for indicator in scope_indicators if indicator in text_lower)
        return min(complexity_score / len(scope_indicators), 1.0)