"""
Reflection Tool
Enables self-reflection, performance analysis, and continuous improvement
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class ReflectionTool(BaseTool):
    """
    Advanced reflection tool for self-analysis, performance evaluation,
    and continuous improvement of agent behavior.
    """

    name: str = "reflection_tool"
    description: str = """
    Perform self-reflection and analysis to improve performance. Analyze past actions,
    identify patterns, evaluate outcomes, and generate insights for better decision-making.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "analyze_performance", "identify_patterns", "evaluate_decision",
                    "generate_insights", "review_errors", "assess_efficiency",
                    "compare_approaches", "set_improvement_goals"
                ],
                "description": "The type of reflection to perform"
            },
            "time_period": {
                "type": "string",
                "enum": ["current_session", "last_hour", "last_day", "last_week", "all_time"],
                "description": "Time period to analyze"
            },
            "focus_area": {
                "type": "string",
                "enum": ["decision_making", "tool_usage", "problem_solving", "communication", "efficiency"],
                "description": "Specific area to focus reflection on"
            },
            "decision_context": {
                "type": "string",
                "description": "Context of a specific decision to evaluate"
            },
            "outcome_description": {
                "type": "string",
                "description": "Description of the outcome to analyze"
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Criteria for measuring success"
            }
        },
        "required": ["action"]
    }

    # Reflection data storage
    reflection_history: List[Dict[str, Any]] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    identified_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_goals: List[Dict[str, Any]] = Field(default_factory=list)
    decision_log: List[Dict[str, Any]] = Field(default_factory=list)

    async def execute(
        self,
        action: str,
        time_period: str = "current_session",
        focus_area: Optional[str] = None,
        decision_context: Optional[str] = None,
        outcome_description: Optional[str] = None,
        success_criteria: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the reflection action."""
        
        try:
            if action == "analyze_performance":
                return await self._analyze_performance(time_period, focus_area)
            elif action == "identify_patterns":
                return await self._identify_patterns(time_period, focus_area)
            elif action == "evaluate_decision":
                return await self._evaluate_decision(decision_context, outcome_description, success_criteria)
            elif action == "generate_insights":
                return await self._generate_insights(focus_area)
            elif action == "review_errors":
                return await self._review_errors(time_period)
            elif action == "assess_efficiency":
                return await self._assess_efficiency(time_period)
            elif action == "compare_approaches":
                return await self._compare_approaches(focus_area)
            elif action == "set_improvement_goals":
                return await self._set_improvement_goals(focus_area)
            else:
                return ToolResult(error=f"Unknown reflection action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Reflection tool error: {str(e)}")

    async def _analyze_performance(self, time_period: str, focus_area: Optional[str]) -> ToolResult:
        """Analyze overall performance for the specified period."""
        
        # Get performance data for the time period
        performance_data = self._get_performance_data(time_period)
        
        analysis = {
            "time_period": time_period,
            "focus_area": focus_area,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Calculate key metrics
        if performance_data:
            analysis["metrics"] = {
                "total_actions": len(performance_data),
                "success_rate": self._calculate_success_rate(performance_data),
                "average_response_time": self._calculate_avg_response_time(performance_data),
                "tool_usage_efficiency": self._calculate_tool_efficiency(performance_data),
                "error_rate": self._calculate_error_rate(performance_data)
            }
            
            # Identify strengths and weaknesses
            analysis["strengths"] = self._identify_strengths(analysis["metrics"])
            analysis["weaknesses"] = self._identify_weaknesses(analysis["metrics"])
            analysis["recommendations"] = self._generate_recommendations(analysis["metrics"])
        
        # Store reflection
        self.reflection_history.append(analysis)
        
        return ToolResult(output=self._format_performance_analysis(analysis))

    async def _identify_patterns(self, time_period: str, focus_area: Optional[str]) -> ToolResult:
        """Identify patterns in behavior and decision-making."""
        
        performance_data = self._get_performance_data(time_period)
        
        patterns = {
            "time_period": time_period,
            "focus_area": focus_area,
            "timestamp": datetime.now().isoformat(),
            "behavioral_patterns": [],
            "decision_patterns": [],
            "tool_usage_patterns": [],
            "error_patterns": []
        }
        
        if performance_data:
            # Analyze behavioral patterns
            patterns["behavioral_patterns"] = self._analyze_behavioral_patterns(performance_data)
            
            # Analyze decision patterns
            patterns["decision_patterns"] = self._analyze_decision_patterns(performance_data)
            
            # Analyze tool usage patterns
            patterns["tool_usage_patterns"] = self._analyze_tool_usage_patterns(performance_data)
            
            # Analyze error patterns
            patterns["error_patterns"] = self._analyze_error_patterns(performance_data)
        
        # Store identified patterns
        self.identified_patterns.append(patterns)
        
        return ToolResult(output=self._format_pattern_analysis(patterns))

    async def _evaluate_decision(
        self, 
        decision_context: Optional[str], 
        outcome_description: Optional[str],
        success_criteria: Optional[List[str]]
    ) -> ToolResult:
        """Evaluate a specific decision and its outcome."""
        
        if not decision_context:
            return ToolResult(error="Decision context is required for evaluation")
        
        evaluation = {
            "decision_context": decision_context,
            "outcome_description": outcome_description,
            "success_criteria": success_criteria or [],
            "timestamp": datetime.now().isoformat(),
            "evaluation_score": 0.0,
            "what_worked": [],
            "what_didnt_work": [],
            "lessons_learned": [],
            "alternative_approaches": []
        }
        
        # Evaluate against success criteria
        if success_criteria and outcome_description:
            met_criteria = 0
            for criterion in success_criteria:
                if self._criterion_met(criterion, outcome_description):
                    met_criteria += 1
                    evaluation["what_worked"].append(f"Met criterion: {criterion}")
                else:
                    evaluation["what_didnt_work"].append(f"Did not meet criterion: {criterion}")
            
            evaluation["evaluation_score"] = met_criteria / len(success_criteria)
        
        # Generate lessons learned
        evaluation["lessons_learned"] = self._extract_lessons(decision_context, outcome_description)
        
        # Suggest alternative approaches
        evaluation["alternative_approaches"] = self._suggest_alternatives(decision_context)
        
        # Store decision evaluation
        self.decision_log.append(evaluation)
        
        return ToolResult(output=self._format_decision_evaluation(evaluation))

    async def _generate_insights(self, focus_area: Optional[str]) -> ToolResult:
        """Generate insights from accumulated reflection data."""
        
        insights = {
            "focus_area": focus_area,
            "timestamp": datetime.now().isoformat(),
            "key_insights": [],
            "recurring_themes": [],
            "improvement_opportunities": [],
            "success_factors": []
        }
        
        # Analyze reflection history for insights
        if self.reflection_history:
            insights["key_insights"] = self._extract_key_insights()
            insights["recurring_themes"] = self._identify_recurring_themes()
            insights["improvement_opportunities"] = self._identify_improvement_opportunities()
            insights["success_factors"] = self._identify_success_factors()
        
        return ToolResult(output=self._format_insights(insights))

    async def _review_errors(self, time_period: str) -> ToolResult:
        """Review and analyze errors from the specified time period."""
        
        performance_data = self._get_performance_data(time_period)
        errors = [item for item in performance_data if item.get("error") or item.get("success") == False]
        
        error_review = {
            "time_period": time_period,
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(errors),
            "error_categories": {},
            "common_causes": [],
            "prevention_strategies": [],
            "recovery_patterns": []
        }
        
        if errors:
            # Categorize errors
            error_review["error_categories"] = self._categorize_errors(errors)
            
            # Identify common causes
            error_review["common_causes"] = self._identify_error_causes(errors)
            
            # Suggest prevention strategies
            error_review["prevention_strategies"] = self._suggest_prevention_strategies(errors)
            
            # Analyze recovery patterns
            error_review["recovery_patterns"] = self._analyze_recovery_patterns(errors)
        
        return ToolResult(output=self._format_error_review(error_review))

    async def _assess_efficiency(self, time_period: str) -> ToolResult:
        """Assess efficiency of actions and decisions."""
        
        performance_data = self._get_performance_data(time_period)
        
        efficiency_assessment = {
            "time_period": time_period,
            "timestamp": datetime.now().isoformat(),
            "efficiency_score": 0.0,
            "time_utilization": {},
            "resource_usage": {},
            "optimization_opportunities": [],
            "efficiency_trends": []
        }
        
        if performance_data:
            # Calculate efficiency metrics
            efficiency_assessment["efficiency_score"] = self._calculate_efficiency_score(performance_data)
            efficiency_assessment["time_utilization"] = self._analyze_time_utilization(performance_data)
            efficiency_assessment["resource_usage"] = self._analyze_resource_usage(performance_data)
            efficiency_assessment["optimization_opportunities"] = self._identify_optimization_opportunities(performance_data)
            efficiency_assessment["efficiency_trends"] = self._analyze_efficiency_trends(performance_data)
        
        return ToolResult(output=self._format_efficiency_assessment(efficiency_assessment))

    async def _compare_approaches(self, focus_area: Optional[str]) -> ToolResult:
        """Compare different approaches used for similar tasks."""
        
        comparison = {
            "focus_area": focus_area,
            "timestamp": datetime.now().isoformat(),
            "approaches_compared": [],
            "effectiveness_ranking": [],
            "context_dependencies": [],
            "recommendations": []
        }
        
        # Find similar tasks with different approaches
        similar_tasks = self._find_similar_tasks(focus_area)
        
        if similar_tasks:
            comparison["approaches_compared"] = self._extract_approaches(similar_tasks)
            comparison["effectiveness_ranking"] = self._rank_approaches(similar_tasks)
            comparison["context_dependencies"] = self._analyze_context_dependencies(similar_tasks)
            comparison["recommendations"] = self._generate_approach_recommendations(similar_tasks)
        
        return ToolResult(output=self._format_approach_comparison(comparison))

    async def _set_improvement_goals(self, focus_area: Optional[str]) -> ToolResult:
        """Set specific improvement goals based on reflection insights."""
        
        # Analyze current performance to identify improvement areas
        current_performance = self._get_current_performance_summary()
        
        goals = {
            "focus_area": focus_area,
            "timestamp": datetime.now().isoformat(),
            "improvement_goals": [],
            "success_metrics": [],
            "action_plans": [],
            "timeline": "1 week"
        }
        
        # Generate improvement goals based on weaknesses and opportunities
        goals["improvement_goals"] = self._generate_improvement_goals(current_performance, focus_area)
        goals["success_metrics"] = self._define_success_metrics(goals["improvement_goals"])
        goals["action_plans"] = self._create_action_plans(goals["improvement_goals"])
        
        # Store improvement goals
        self.improvement_goals.append(goals)
        
        return ToolResult(output=self._format_improvement_goals(goals))

    # Helper methods for data analysis and formatting

    def _get_performance_data(self, time_period: str) -> List[Dict[str, Any]]:
        """Get performance data for the specified time period."""
        # This would integrate with the agent's execution history
        # For now, return mock data structure
        return []

    def _calculate_success_rate(self, data: List[Dict[str, Any]]) -> float:
        """Calculate success rate from performance data."""
        if not data:
            return 0.0
        successful = sum(1 for item in data if item.get("success", False))
        return successful / len(data)

    def _calculate_avg_response_time(self, data: List[Dict[str, Any]]) -> float:
        """Calculate average response time."""
        if not data:
            return 0.0
        times = [item.get("duration", 0) for item in data if item.get("duration")]
        return sum(times) / len(times) if times else 0.0

    def _calculate_tool_efficiency(self, data: List[Dict[str, Any]]) -> float:
        """Calculate tool usage efficiency."""
        # Mock calculation - would analyze tool selection appropriateness
        return 0.75

    def _calculate_error_rate(self, data: List[Dict[str, Any]]) -> float:
        """Calculate error rate."""
        if not data:
            return 0.0
        errors = sum(1 for item in data if item.get("error") or item.get("success") == False)
        return errors / len(data)

    def _identify_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify strengths based on metrics."""
        strengths = []
        
        if metrics.get("success_rate", 0) > 0.8:
            strengths.append("High success rate in task completion")
        
        if metrics.get("error_rate", 1) < 0.1:
            strengths.append("Low error rate")
        
        if metrics.get("tool_usage_efficiency", 0) > 0.7:
            strengths.append("Efficient tool selection and usage")
        
        return strengths

    def _identify_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify weaknesses based on metrics."""
        weaknesses = []
        
        if metrics.get("success_rate", 1) < 0.6:
            weaknesses.append("Low success rate needs improvement")
        
        if metrics.get("error_rate", 0) > 0.2:
            weaknesses.append("High error rate requires attention")
        
        if metrics.get("average_response_time", 0) > 60:
            weaknesses.append("Response time could be improved")
        
        return weaknesses

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if metrics.get("error_rate", 0) > 0.15:
            recommendations.append("Implement more thorough validation before executing actions")
        
        if metrics.get("tool_usage_efficiency", 1) < 0.6:
            recommendations.append("Review tool selection criteria and improve decision-making")
        
        if metrics.get("success_rate", 1) < 0.7:
            recommendations.append("Break down complex tasks into smaller, more manageable steps")
        
        return recommendations

    def _criterion_met(self, criterion: str, outcome: str) -> bool:
        """Check if a success criterion was met based on outcome description."""
        # Simple keyword matching - would be more sophisticated in practice
        return any(word in outcome.lower() for word in criterion.lower().split())

    def _extract_lessons(self, context: str, outcome: str) -> List[str]:
        """Extract lessons learned from decision context and outcome."""
        lessons = []
        
        if outcome and "error" in outcome.lower():
            lessons.append("Need to validate inputs more thoroughly before proceeding")
        
        if context and "complex" in context.lower():
            lessons.append("Complex tasks benefit from step-by-step planning")
        
        return lessons

    def _suggest_alternatives(self, context: str) -> List[str]:
        """Suggest alternative approaches for the given context."""
        alternatives = []
        
        if "research" in context.lower():
            alternatives.extend([
                "Use multiple search sources for comprehensive coverage",
                "Validate information from multiple independent sources"
            ])
        
        if "code" in context.lower():
            alternatives.extend([
                "Implement incremental testing during development",
                "Use modular approach for complex implementations"
            ])
        
        return alternatives

    # Formatting methods

    def _format_performance_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis for output."""
        output = [f"Performance Analysis - {analysis['time_period']}"]
        output.append("=" * 50)
        
        if analysis["metrics"]:
            output.append("\nKey Metrics:")
            for metric, value in analysis["metrics"].items():
                if isinstance(value, float):
                    output.append(f"  {metric}: {value:.2f}")
                else:
                    output.append(f"  {metric}: {value}")
        
        if analysis["strengths"]:
            output.append("\nStrengths:")
            for strength in analysis["strengths"]:
                output.append(f"  + {strength}")
        
        if analysis["weaknesses"]:
            output.append("\nAreas for Improvement:")
            for weakness in analysis["weaknesses"]:
                output.append(f"  - {weakness}")
        
        if analysis["recommendations"]:
            output.append("\nRecommendations:")
            for rec in analysis["recommendations"]:
                output.append(f"  → {rec}")
        
        return "\n".join(output)

    def _format_pattern_analysis(self, patterns: Dict[str, Any]) -> str:
        """Format pattern analysis for output."""
        output = [f"Pattern Analysis - {patterns['time_period']}"]
        output.append("=" * 50)
        
        for pattern_type, pattern_list in patterns.items():
            if pattern_type in ["time_period", "focus_area", "timestamp"]:
                continue
            
            if pattern_list:
                output.append(f"\n{pattern_type.replace('_', ' ').title()}:")
                for pattern in pattern_list:
                    output.append(f"  • {pattern}")
        
        return "\n".join(output)

    def _format_decision_evaluation(self, evaluation: Dict[str, Any]) -> str:
        """Format decision evaluation for output."""
        output = ["Decision Evaluation"]
        output.append("=" * 30)
        
        output.append(f"\nContext: {evaluation['decision_context']}")
        if evaluation['outcome_description']:
            output.append(f"Outcome: {evaluation['outcome_description']}")
        
        output.append(f"\nEvaluation Score: {evaluation['evaluation_score']:.2f}")
        
        if evaluation["what_worked"]:
            output.append("\nWhat Worked:")
            for item in evaluation["what_worked"]:
                output.append(f"  ✓ {item}")
        
        if evaluation["what_didnt_work"]:
            output.append("\nWhat Didn't Work:")
            for item in evaluation["what_didnt_work"]:
                output.append(f"  ✗ {item}")
        
        if evaluation["lessons_learned"]:
            output.append("\nLessons Learned:")
            for lesson in evaluation["lessons_learned"]:
                output.append(f"  📚 {lesson}")
        
        return "\n".join(output)

    def _format_insights(self, insights: Dict[str, Any]) -> str:
        """Format insights for output."""
        output = ["Generated Insights"]
        output.append("=" * 30)
        
        for insight_type, insight_list in insights.items():
            if insight_type in ["focus_area", "timestamp"]:
                continue
            
            if insight_list:
                output.append(f"\n{insight_type.replace('_', ' ').title()}:")
                for insight in insight_list:
                    output.append(f"  💡 {insight}")
        
        return "\n".join(output)

    def _format_error_review(self, review: Dict[str, Any]) -> str:
        """Format error review for output."""
        output = [f"Error Review - {review['time_period']}"]
        output.append("=" * 40)
        
        output.append(f"\nTotal Errors: {review['total_errors']}")
        
        if review["error_categories"]:
            output.append("\nError Categories:")
            for category, count in review["error_categories"].items():
                output.append(f"  {category}: {count}")
        
        if review["common_causes"]:
            output.append("\nCommon Causes:")
            for cause in review["common_causes"]:
                output.append(f"  • {cause}")
        
        if review["prevention_strategies"]:
            output.append("\nPrevention Strategies:")
            for strategy in review["prevention_strategies"]:
                output.append(f"  🛡️ {strategy}")
        
        return "\n".join(output)

    def _format_efficiency_assessment(self, assessment: Dict[str, Any]) -> str:
        """Format efficiency assessment for output."""
        output = [f"Efficiency Assessment - {assessment['time_period']}"]
        output.append("=" * 50)
        
        output.append(f"\nEfficiency Score: {assessment['efficiency_score']:.2f}")
        
        if assessment["optimization_opportunities"]:
            output.append("\nOptimization Opportunities:")
            for opportunity in assessment["optimization_opportunities"]:
                output.append(f"  ⚡ {opportunity}")
        
        return "\n".join(output)

    def _format_approach_comparison(self, comparison: Dict[str, Any]) -> str:
        """Format approach comparison for output."""
        output = ["Approach Comparison"]
        output.append("=" * 30)
        
        if comparison["effectiveness_ranking"]:
            output.append("\nEffectiveness Ranking:")
            for i, approach in enumerate(comparison["effectiveness_ranking"], 1):
                output.append(f"  {i}. {approach}")
        
        if comparison["recommendations"]:
            output.append("\nRecommendations:")
            for rec in comparison["recommendations"]:
                output.append(f"  → {rec}")
        
        return "\n".join(output)

    def _format_improvement_goals(self, goals: Dict[str, Any]) -> str:
        """Format improvement goals for output."""
        output = ["Improvement Goals"]
        output.append("=" * 30)
        
        if goals["improvement_goals"]:
            output.append("\nGoals:")
            for goal in goals["improvement_goals"]:
                output.append(f"  🎯 {goal}")
        
        if goals["success_metrics"]:
            output.append("\nSuccess Metrics:")
            for metric in goals["success_metrics"]:
                output.append(f"  📊 {metric}")
        
        if goals["action_plans"]:
            output.append("\nAction Plans:")
            for plan in goals["action_plans"]:
                output.append(f"  📋 {plan}")
        
        return "\n".join(output)

    # Additional helper methods would be implemented here for:
    # - _analyze_behavioral_patterns
    # - _analyze_decision_patterns  
    # - _analyze_tool_usage_patterns
    # - _analyze_error_patterns
    # - _extract_key_insights
    # - _identify_recurring_themes
    # - etc.