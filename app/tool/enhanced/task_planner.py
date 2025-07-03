"""
Advanced Task Planning Tool
Breaks down complex tasks into manageable steps with dependencies and success criteria
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class TaskPlannerTool(BaseTool):
    """
    Advanced task planning tool that can break down complex tasks into manageable steps,
    identify dependencies, estimate effort, and track progress.
    """

    name: str = "task_planner"
    description: str = """
    Plan and manage complex tasks by breaking them down into structured, manageable steps.
    This tool helps with task decomposition, dependency mapping, effort estimation, and progress tracking.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_plan", "update_plan", "get_plan", "add_step", "complete_step", "analyze_task"],
                "description": "The planning action to perform"
            },
            "task_description": {
                "type": "string",
                "description": "Description of the task to plan (required for create_plan and analyze_task)"
            },
            "plan_id": {
                "type": "string", 
                "description": "ID of the plan to work with (required for update_plan, get_plan, add_step, complete_step)"
            },
            "step_description": {
                "type": "string",
                "description": "Description of the step to add (required for add_step)"
            },
            "step_id": {
                "type": "string",
                "description": "ID of the step to complete (required for complete_step)"
            },
            "complexity_level": {
                "type": "string",
                "enum": ["low", "medium", "high", "very_high"],
                "description": "Complexity level of the task (optional, will be auto-detected if not provided)"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
                "description": "Priority level of the task"
            },
            "estimated_duration": {
                "type": "integer",
                "description": "Estimated duration in minutes"
            }
        },
        "required": ["action"]
    }

    # Store plans in memory (in production, this would be persisted)
    plans: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    async def execute(
        self,
        action: str,
        task_description: Optional[str] = None,
        plan_id: Optional[str] = None,
        step_description: Optional[str] = None,
        step_id: Optional[str] = None,
        complexity_level: Optional[str] = None,
        priority: str = "medium",
        estimated_duration: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the task planning action."""
        
        try:
            if action == "create_plan":
                return await self._create_plan(
                    task_description, complexity_level, priority, estimated_duration
                )
            elif action == "update_plan":
                return await self._update_plan(plan_id, **kwargs)
            elif action == "get_plan":
                return await self._get_plan(plan_id)
            elif action == "add_step":
                return await self._add_step(plan_id, step_description)
            elif action == "complete_step":
                return await self._complete_step(plan_id, step_id)
            elif action == "analyze_task":
                return await self._analyze_task(task_description)
            else:
                return ToolResult(error=f"Unknown action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Task planner error: {str(e)}")

    async def _create_plan(
        self, 
        task_description: str, 
        complexity_level: Optional[str] = None,
        priority: str = "medium",
        estimated_duration: Optional[int] = None
    ) -> ToolResult:
        """Create a new task plan."""
        
        if not task_description:
            return ToolResult(error="Task description is required for creating a plan")

        # Generate plan ID
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze task complexity if not provided
        if not complexity_level:
            complexity_level = self._analyze_complexity(task_description)
        
        # Decompose task into steps
        steps = self._decompose_task(task_description, complexity_level)
        
        # Estimate duration if not provided
        if not estimated_duration:
            estimated_duration = self._estimate_duration(steps, complexity_level)
        
        # Create plan structure
        plan = {
            "id": plan_id,
            "title": task_description[:100] + "..." if len(task_description) > 100 else task_description,
            "description": task_description,
            "complexity": complexity_level,
            "priority": priority,
            "estimated_duration": estimated_duration,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "progress": 0,
            "steps": steps,
            "dependencies": self._identify_dependencies(steps),
            "success_criteria": self._define_success_criteria(task_description),
            "risks": self._identify_risks(task_description, complexity_level),
            "resources_needed": self._identify_resources(task_description)
        }
        
        self.plans[plan_id] = plan
        
        return ToolResult(
            output=f"Created plan '{plan_id}' with {len(steps)} steps.\n\n{self._format_plan(plan)}"
        )

    async def _analyze_task(self, task_description: str) -> ToolResult:
        """Analyze a task without creating a full plan."""
        
        complexity = self._analyze_complexity(task_description)
        estimated_steps = len(self._decompose_task(task_description, complexity))
        estimated_duration = self._estimate_duration([], complexity) * estimated_steps
        
        analysis = {
            "task": task_description,
            "complexity": complexity,
            "estimated_steps": estimated_steps,
            "estimated_duration": estimated_duration,
            "key_challenges": self._identify_challenges(task_description),
            "recommended_approach": self._recommend_approach(task_description, complexity),
            "required_tools": self._identify_required_tools(task_description)
        }
        
        return ToolResult(
            output=f"Task Analysis:\n{json.dumps(analysis, indent=2)}"
        )

    def _analyze_complexity(self, task_description: str) -> str:
        """Analyze task complexity based on description."""
        
        # Simple heuristics for complexity analysis
        word_count = len(task_description.split())
        
        # Keywords that indicate complexity
        complex_keywords = [
            "integrate", "analyze", "research", "develop", "design", "optimize",
            "multiple", "various", "complex", "advanced", "comprehensive"
        ]
        
        simple_keywords = [
            "create", "write", "list", "find", "search", "copy", "simple", "basic"
        ]
        
        complexity_score = 0
        
        # Word count factor
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1
        
        # Keyword analysis
        for keyword in complex_keywords:
            if keyword in task_description.lower():
                complexity_score += 1
        
        for keyword in simple_keywords:
            if keyword in task_description.lower():
                complexity_score -= 1
        
        # Determine complexity level
        if complexity_score >= 4:
            return "very_high"
        elif complexity_score >= 2:
            return "high"
        elif complexity_score >= 0:
            return "medium"
        else:
            return "low"

    def _decompose_task(self, task_description: str, complexity: str) -> List[Dict[str, Any]]:
        """Decompose task into manageable steps."""
        
        steps = []
        task_lower = task_description.lower()
        
        # Common task patterns and their decomposition
        if any(keyword in task_lower for keyword in ["research", "analyze", "study"]):
            steps.extend([
                {
                    "id": "research_1",
                    "title": "Define research scope and objectives",
                    "description": "Clearly define what needs to be researched and the expected outcomes",
                    "estimated_duration": 15,
                    "status": "pending",
                    "dependencies": []
                },
                {
                    "id": "research_2", 
                    "title": "Gather information from multiple sources",
                    "description": "Search and collect relevant information from various sources",
                    "estimated_duration": 30,
                    "status": "pending",
                    "dependencies": ["research_1"]
                },
                {
                    "id": "research_3",
                    "title": "Analyze and synthesize findings",
                    "description": "Process the gathered information and identify key insights",
                    "estimated_duration": 25,
                    "status": "pending", 
                    "dependencies": ["research_2"]
                }
            ])
        
        if any(keyword in task_lower for keyword in ["code", "program", "develop", "build"]):
            steps.extend([
                {
                    "id": "dev_1",
                    "title": "Plan architecture and design",
                    "description": "Design the overall structure and approach",
                    "estimated_duration": 20,
                    "status": "pending",
                    "dependencies": []
                },
                {
                    "id": "dev_2",
                    "title": "Implement core functionality", 
                    "description": "Write the main code implementation",
                    "estimated_duration": 45,
                    "status": "pending",
                    "dependencies": ["dev_1"]
                },
                {
                    "id": "dev_3",
                    "title": "Test and debug",
                    "description": "Test the implementation and fix any issues",
                    "estimated_duration": 20,
                    "status": "pending",
                    "dependencies": ["dev_2"]
                }
            ])
        
        if any(keyword in task_lower for keyword in ["write", "create", "document"]):
            steps.extend([
                {
                    "id": "write_1",
                    "title": "Create outline and structure",
                    "description": "Plan the structure and main points",
                    "estimated_duration": 15,
                    "status": "pending",
                    "dependencies": []
                },
                {
                    "id": "write_2",
                    "title": "Write first draft",
                    "description": "Create the initial version of the content",
                    "estimated_duration": 30,
                    "status": "pending", 
                    "dependencies": ["write_1"]
                },
                {
                    "id": "write_3",
                    "title": "Review and refine",
                    "description": "Edit and improve the content",
                    "estimated_duration": 15,
                    "status": "pending",
                    "dependencies": ["write_2"]
                }
            ])
        
        # Default steps if no specific patterns match
        if not steps:
            steps = [
                {
                    "id": "step_1",
                    "title": "Analyze requirements",
                    "description": "Understand what needs to be done",
                    "estimated_duration": 10,
                    "status": "pending",
                    "dependencies": []
                },
                {
                    "id": "step_2", 
                    "title": "Execute main task",
                    "description": "Perform the primary work",
                    "estimated_duration": 30,
                    "status": "pending",
                    "dependencies": ["step_1"]
                },
                {
                    "id": "step_3",
                    "title": "Verify and finalize",
                    "description": "Check results and complete the task",
                    "estimated_duration": 10,
                    "status": "pending",
                    "dependencies": ["step_2"]
                }
            ]
        
        # Adjust complexity based on level
        if complexity in ["high", "very_high"]:
            # Add additional steps for complex tasks
            steps.append({
                "id": "complex_review",
                "title": "Comprehensive review and optimization",
                "description": "Thorough review and optimization of the work",
                "estimated_duration": 20,
                "status": "pending",
                "dependencies": [step["id"] for step in steps]
            })
        
        return steps

    def _estimate_duration(self, steps: List[Dict[str, Any]], complexity: str) -> int:
        """Estimate total duration for the task."""
        
        if steps:
            return sum(step.get("estimated_duration", 15) for step in steps)
        
        # Base estimates by complexity
        base_estimates = {
            "low": 30,
            "medium": 60,
            "high": 120,
            "very_high": 240
        }
        
        return base_estimates.get(complexity, 60)

    def _identify_dependencies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify dependencies between steps."""
        
        dependencies = []
        for step in steps:
            if step.get("dependencies"):
                for dep in step["dependencies"]:
                    dependencies.append({
                        "step": step["id"],
                        "depends_on": dep,
                        "type": "sequential"
                    })
        
        return dependencies

    def _define_success_criteria(self, task_description: str) -> List[str]:
        """Define success criteria for the task."""
        
        criteria = ["Task completed as described"]
        
        task_lower = task_description.lower()
        
        if "research" in task_lower:
            criteria.extend([
                "Comprehensive information gathered",
                "Key insights identified",
                "Sources properly documented"
            ])
        
        if any(keyword in task_lower for keyword in ["code", "program", "develop"]):
            criteria.extend([
                "Code functions correctly",
                "Requirements met",
                "Code is well-documented"
            ])
        
        if "write" in task_lower:
            criteria.extend([
                "Content is clear and well-structured",
                "All key points covered",
                "Grammar and style are correct"
            ])
        
        return criteria

    def _identify_risks(self, task_description: str, complexity: str) -> List[Dict[str, str]]:
        """Identify potential risks and mitigation strategies."""
        
        risks = []
        
        if complexity in ["high", "very_high"]:
            risks.append({
                "risk": "Task complexity may lead to scope creep",
                "mitigation": "Break down into smaller, well-defined steps"
            })
        
        if "research" in task_description.lower():
            risks.append({
                "risk": "Information may be outdated or unreliable",
                "mitigation": "Verify sources and cross-reference information"
            })
        
        if any(keyword in task_description.lower() for keyword in ["code", "program"]):
            risks.append({
                "risk": "Technical issues or bugs may arise",
                "mitigation": "Implement thorough testing and debugging"
            })
        
        return risks

    def _identify_resources(self, task_description: str) -> List[str]:
        """Identify resources needed for the task."""
        
        resources = []
        task_lower = task_description.lower()
        
        if "research" in task_lower:
            resources.extend(["Web search capability", "Access to reliable sources"])
        
        if any(keyword in task_lower for keyword in ["code", "program", "develop"]):
            resources.extend(["Code execution environment", "Development tools"])
        
        if "write" in task_lower:
            resources.extend(["Text editing capability", "Grammar checking"])
        
        if "browser" in task_lower or "web" in task_lower:
            resources.append("Web browser automation")
        
        return resources

    def _identify_challenges(self, task_description: str) -> List[str]:
        """Identify potential challenges."""
        
        challenges = []
        task_lower = task_description.lower()
        
        if len(task_description.split()) > 50:
            challenges.append("Complex requirements may need clarification")
        
        if "integrate" in task_lower:
            challenges.append("Integration complexity between different components")
        
        if "optimize" in task_lower:
            challenges.append("Balancing multiple optimization criteria")
        
        return challenges

    def _recommend_approach(self, task_description: str, complexity: str) -> str:
        """Recommend an approach for the task."""
        
        if complexity == "very_high":
            return "Break into multiple phases, validate each phase before proceeding"
        elif complexity == "high":
            return "Use iterative approach with regular checkpoints"
        elif complexity == "medium":
            return "Follow structured plan with clear milestones"
        else:
            return "Direct implementation with basic validation"

    def _identify_required_tools(self, task_description: str) -> List[str]:
        """Identify tools likely needed for the task."""
        
        tools = []
        task_lower = task_description.lower()
        
        if "research" in task_lower or "search" in task_lower:
            tools.append("web_search")
        
        if any(keyword in task_lower for keyword in ["code", "program", "script"]):
            tools.append("python_execute")
        
        if "file" in task_lower or "edit" in task_lower:
            tools.append("str_replace_editor")
        
        if "browser" in task_lower or "website" in task_lower:
            tools.append("browser_use")
        
        return tools

    async def _get_plan(self, plan_id: str) -> ToolResult:
        """Get details of a specific plan."""
        
        if not plan_id:
            return ToolResult(error="Plan ID is required")
        
        if plan_id not in self.plans:
            return ToolResult(error=f"Plan {plan_id} not found")
        
        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    async def _add_step(self, plan_id: str, step_description: str) -> ToolResult:
        """Add a new step to an existing plan."""
        
        if not plan_id or not step_description:
            return ToolResult(error="Plan ID and step description are required")
        
        if plan_id not in self.plans:
            return ToolResult(error=f"Plan {plan_id} not found")
        
        plan = self.plans[plan_id]
        step_id = f"step_{len(plan['steps']) + 1}"
        
        new_step = {
            "id": step_id,
            "title": step_description[:50] + "..." if len(step_description) > 50 else step_description,
            "description": step_description,
            "estimated_duration": 15,
            "status": "pending",
            "dependencies": []
        }
        
        plan["steps"].append(new_step)
        plan["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(output=f"Added step '{step_id}' to plan {plan_id}")

    async def _complete_step(self, plan_id: str, step_id: str) -> ToolResult:
        """Mark a step as completed."""
        
        if not plan_id or not step_id:
            return ToolResult(error="Plan ID and step ID are required")
        
        if plan_id not in self.plans:
            return ToolResult(error=f"Plan {plan_id} not found")
        
        plan = self.plans[plan_id]
        
        # Find and update the step
        step_found = False
        for step in plan["steps"]:
            if step["id"] == step_id:
                step["status"] = "completed"
                step["completed_at"] = datetime.now().isoformat()
                step_found = True
                break
        
        if not step_found:
            return ToolResult(error=f"Step {step_id} not found in plan {plan_id}")
        
        # Update plan progress
        completed_steps = sum(1 for step in plan["steps"] if step["status"] == "completed")
        plan["progress"] = (completed_steps / len(plan["steps"])) * 100
        plan["updated_at"] = datetime.now().isoformat()
        
        # Check if plan is complete
        if plan["progress"] == 100:
            plan["status"] = "completed"
            plan["completed_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Completed step '{step_id}' in plan {plan_id}. Progress: {plan['progress']:.1f}%"
        )

    async def _update_plan(self, plan_id: str, **updates) -> ToolResult:
        """Update an existing plan."""
        
        if not plan_id:
            return ToolResult(error="Plan ID is required")
        
        if plan_id not in self.plans:
            return ToolResult(error=f"Plan {plan_id} not found")
        
        plan = self.plans[plan_id]
        
        # Update allowed fields
        allowed_updates = ["priority", "estimated_duration", "status"]
        for key, value in updates.items():
            if key in allowed_updates:
                plan[key] = value
        
        plan["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(output=f"Updated plan {plan_id}")

    def _format_plan(self, plan: Dict[str, Any]) -> str:
        """Format a plan for display."""
        
        output = []
        output.append(f"Plan: {plan['title']}")
        output.append(f"ID: {plan['id']}")
        output.append(f"Status: {plan['status']} ({plan['progress']:.1f}% complete)")
        output.append(f"Complexity: {plan['complexity']}")
        output.append(f"Priority: {plan['priority']}")
        output.append(f"Estimated Duration: {plan['estimated_duration']} minutes")
        output.append("")
        
        output.append("Steps:")
        for i, step in enumerate(plan['steps'], 1):
            status_icon = "✓" if step['status'] == 'completed' else "○"
            output.append(f"  {i}. {status_icon} {step['title']} ({step['estimated_duration']}min)")
            if step.get('dependencies'):
                output.append(f"     Dependencies: {', '.join(step['dependencies'])}")
        
        if plan.get('success_criteria'):
            output.append("")
            output.append("Success Criteria:")
            for criterion in plan['success_criteria']:
                output.append(f"  - {criterion}")
        
        if plan.get('risks'):
            output.append("")
            output.append("Risks & Mitigations:")
            for risk in plan['risks']:
                output.append(f"  - Risk: {risk['risk']}")
                output.append(f"    Mitigation: {risk['mitigation']}")
        
        return "\n".join(output)