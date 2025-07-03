"""
Enhanced Manus Agent with advanced prompting and tool capabilities
Based on OpenManus architecture with improvements from agent prompting best practices
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.web_search import WebSearch
from app.tool.mcp import MCPClients, MCPClientTool

# Import enhanced tools
from app.tool.enhanced import (
    TaskPlannerTool,
    MemoryManagerTool,
    ReflectionTool,
    QualityAssuranceTool,
    ContextAnalyzerTool,
    ProgressTrackerTool
)

class EnhancedManus(ToolCallAgent):
    """
    Enhanced Manus Agent with advanced reasoning, planning, and execution capabilities.
    
    Features:
    - Advanced task planning and decomposition
    - Memory management and context awareness
    - Self-reflection and quality assurance
    - Progress tracking and adaptive execution
    - Enhanced error handling and recovery
    """

    name: str = "EnhancedManus"
    description: str = """
    An advanced AI agent capable of complex reasoning, planning, and execution.
    I can break down complex tasks, maintain context across interactions, 
    reflect on my performance, and adapt my approach based on results.
    """

    # Enhanced system prompt with advanced reasoning capabilities
    system_prompt: str = """
You are EnhancedManus, an advanced AI agent with sophisticated reasoning and execution capabilities.

## Core Principles:
1. **Think Before Acting**: Always analyze the task thoroughly before taking action
2. **Plan Systematically**: Break complex tasks into manageable steps
3. **Maintain Context**: Keep track of progress, decisions, and learnings
4. **Reflect and Adapt**: Continuously evaluate performance and adjust approach
5. **Quality First**: Ensure accuracy and completeness in all outputs

## Reasoning Framework:
When approaching any task, follow this structured thinking process:

1. **UNDERSTAND**: Analyze the request thoroughly
   - What is the user really asking for?
   - What are the explicit and implicit requirements?
   - What context or constraints should I consider?

2. **PLAN**: Develop a systematic approach
   - Break the task into logical steps
   - Identify required tools and resources
   - Consider potential challenges and alternatives

3. **EXECUTE**: Implement the plan methodically
   - Follow the planned steps in order
   - Monitor progress and results
   - Adapt if unexpected issues arise

4. **REFLECT**: Evaluate the outcome
   - Did I achieve the intended goal?
   - What worked well and what could be improved?
   - What did I learn for future similar tasks?

## Tool Usage Guidelines:
- Use task_planner for complex multi-step tasks
- Use memory_manager to maintain context across interactions
- Use reflection_tool to evaluate performance and learn
- Use quality_assurance to verify outputs before completion
- Use context_analyzer to understand nuanced requirements
- Use progress_tracker to monitor task completion

## Communication Style:
- Be clear and concise in explanations
- Show your reasoning process when helpful
- Acknowledge uncertainties and limitations
- Ask clarifying questions when needed
- Provide structured, actionable outputs

Remember: Your goal is not just to complete tasks, but to do so thoughtfully, 
efficiently, and with high quality results.
"""

    next_step_prompt: str = """
Based on the current context and conversation history, determine the most appropriate next action.

Consider:
1. What has been accomplished so far?
2. What remains to be done?
3. Are there any issues or blockers to address?
4. What would be the most valuable next step?

If this is a complex task, consider using the task planner to break it down systematically.
If you need to maintain context across multiple interactions, use the memory manager.
If you're unsure about requirements, use the context analyzer or ask for clarification.

Always reflect on your approach and be prepared to adapt based on results.
"""

    max_observe: int = 15000
    max_steps: int = 30

    # Enhanced tool collection with advanced capabilities
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # Core tools
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            WebSearch(),
            AskHuman(),
            
            # Enhanced tools
            TaskPlannerTool(),
            MemoryManagerTool(),
            ReflectionTool(),
            QualityAssuranceTool(),
            ContextAnalyzerTool(),
            ProgressTrackerTool(),
            
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # Enhanced state management
    task_context: Dict[str, Any] = Field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_plan: Optional[Dict[str, Any]] = None
    reflection_notes: List[str] = Field(default_factory=list)

    # MCP integration
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    connected_servers: Dict[str, str] = Field(default_factory=dict)
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_enhanced_features(self) -> "EnhancedManus":
        """Initialize enhanced features and context."""
        self.task_context = {
            "session_id": datetime.now().isoformat(),
            "start_time": datetime.now(),
            "total_tasks": 0,
            "completed_tasks": 0,
            "current_task": None,
            "context_memory": {},
            "learned_patterns": [],
            "performance_metrics": {
                "success_rate": 0.0,
                "avg_steps_per_task": 0.0,
                "common_errors": [],
                "improvement_areas": []
            }
        }
        return self

    @classmethod
    async def create(cls, **kwargs) -> "EnhancedManus":
        """Factory method to create and properly initialize an Enhanced Manus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(f"Connected to MCP server {server_id} at {server_config.url}")
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(f"Connected to MCP server {server_id} using command {server_config.command}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(server_url, stdio_args or [], server_id)
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with new tools from this server
        new_tools = [tool for tool in self.mcp_clients.tools if tool.server_id == server_id]
        self.available_tools.add_tools(*new_tools)

    async def think(self) -> bool:
        """Enhanced thinking process with structured reasoning."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        # Update task context
        self.task_context["current_step"] = self.current_step
        self.task_context["last_action_time"] = datetime.now()

        # Enhanced next step prompt with context
        enhanced_prompt = self._build_enhanced_prompt()
        
        # Store original prompt and use enhanced version
        original_prompt = self.next_step_prompt
        self.next_step_prompt = enhanced_prompt

        try:
            result = await super().think()
            
            # Log the thinking process
            self._log_thinking_process()
            
            return result
        finally:
            # Restore original prompt
            self.next_step_prompt = original_prompt

    def _build_enhanced_prompt(self) -> str:
        """Build an enhanced prompt with current context and reasoning framework."""
        context_summary = self._get_context_summary()
        
        enhanced_prompt = f"""
## Current Context:
{context_summary}

## Reasoning Framework:
Please follow this structured approach:

1. **ANALYZE**: What is the current situation?
   - What has been accomplished?
   - What are the current goals?
   - What challenges or opportunities exist?

2. **PLAN**: What should be done next?
   - What is the most logical next step?
   - What tools or resources are needed?
   - What are potential risks or alternatives?

3. **DECIDE**: Choose the best course of action
   - Select the most appropriate tool or approach
   - Consider efficiency and effectiveness
   - Ensure alignment with overall goals

## Available Enhanced Tools:
- task_planner: For breaking down complex tasks
- memory_manager: For maintaining context and learning
- reflection_tool: For evaluating performance
- quality_assurance: For verifying outputs
- context_analyzer: For understanding nuanced requirements
- progress_tracker: For monitoring completion

{self.next_step_prompt}

Remember to think systematically and explain your reasoning when helpful.
"""
        return enhanced_prompt

    def _get_context_summary(self) -> str:
        """Generate a summary of current context and progress."""
        summary_parts = []
        
        # Session info
        summary_parts.append(f"Session: {self.task_context.get('session_id', 'Unknown')}")
        summary_parts.append(f"Step: {self.current_step}/{self.max_steps}")
        
        # Current task info
        if self.current_plan:
            summary_parts.append(f"Current Plan: {self.current_plan.get('title', 'Unnamed')}")
            summary_parts.append(f"Plan Progress: {self.current_plan.get('progress', 'Unknown')}")
        
        # Recent actions
        if self.execution_history:
            recent_actions = self.execution_history[-3:]
            summary_parts.append("Recent Actions:")
            for action in recent_actions:
                summary_parts.append(f"  - {action.get('action', 'Unknown')}: {action.get('result', 'No result')[:100]}")
        
        # Memory context
        if self.task_context.get("context_memory"):
            summary_parts.append("Key Context:")
            for key, value in list(self.task_context["context_memory"].items())[-3:]:
                summary_parts.append(f"  - {key}: {str(value)[:100]}")
        
        return "\n".join(summary_parts)

    def _log_thinking_process(self) -> None:
        """Log the current thinking process for analysis and improvement."""
        thinking_log = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "context": self.task_context,
            "recent_messages": len(self.messages),
            "available_tools": len(self.available_tools.tools),
            "current_plan": self.current_plan
        }
        
        self.execution_history.append(thinking_log)
        
        # Keep only recent history to manage memory
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-30:]

    async def execute_enhanced_action(self, action_type: str, **kwargs) -> Any:
        """Execute enhanced actions with proper logging and error handling."""
        try:
            start_time = datetime.now()
            
            # Log action start
            action_log = {
                "action": action_type,
                "start_time": start_time,
                "parameters": kwargs,
                "step": self.current_step
            }
            
            # Execute the action
            if action_type == "plan_task":
                result = await self._plan_complex_task(**kwargs)
            elif action_type == "update_memory":
                result = await self._update_context_memory(**kwargs)
            elif action_type == "reflect_performance":
                result = await self._reflect_on_performance(**kwargs)
            elif action_type == "quality_check":
                result = await self._perform_quality_check(**kwargs)
            else:
                # Fallback to standard tool execution
                result = await super().execute_tool(**kwargs)
            
            # Log action completion
            action_log.update({
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "result": str(result)[:500],  # Truncate long results
                "success": True
            })
            
            self.execution_history.append(action_log)
            return result
            
        except Exception as e:
            # Log action failure
            action_log.update({
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "error": str(e),
                "success": False
            })
            
            self.execution_history.append(action_log)
            logger.error(f"Enhanced action failed: {action_type} - {str(e)}")
            raise

    async def _plan_complex_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """Plan a complex task using enhanced planning capabilities."""
        plan = {
            "title": task_description,
            "created_at": datetime.now().isoformat(),
            "steps": [],
            "estimated_duration": None,
            "complexity": "medium",
            "dependencies": [],
            "success_criteria": [],
            "progress": 0
        }
        
        # Analyze task complexity
        if len(task_description.split()) > 50 or "complex" in task_description.lower():
            plan["complexity"] = "high"
        elif len(task_description.split()) < 10:
            plan["complexity"] = "low"
        
        # Break down into steps (simplified logic - would be enhanced with LLM)
        steps = self._decompose_task(task_description)
        plan["steps"] = steps
        plan["estimated_duration"] = len(steps) * 2  # 2 minutes per step estimate
        
        self.current_plan = plan
        return plan

    def _decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """Decompose a task into manageable steps."""
        # This is a simplified version - in practice, this would use LLM reasoning
        steps = []
        
        # Common task patterns
        if "research" in task_description.lower():
            steps.extend([
                {"action": "web_search", "description": "Search for relevant information"},
                {"action": "analyze_results", "description": "Analyze search results"},
                {"action": "synthesize", "description": "Synthesize findings"}
            ])
        
        if "code" in task_description.lower() or "program" in task_description.lower():
            steps.extend([
                {"action": "plan_architecture", "description": "Plan code structure"},
                {"action": "implement", "description": "Write code implementation"},
                {"action": "test", "description": "Test the implementation"}
            ])
        
        if "write" in task_description.lower() or "create" in task_description.lower():
            steps.extend([
                {"action": "outline", "description": "Create outline or structure"},
                {"action": "draft", "description": "Create initial draft"},
                {"action": "review", "description": "Review and refine"}
            ])
        
        # Default steps if no patterns match
        if not steps:
            steps = [
                {"action": "analyze", "description": "Analyze requirements"},
                {"action": "execute", "description": "Execute main task"},
                {"action": "verify", "description": "Verify completion"}
            ]
        
        return steps

    async def _update_context_memory(self, key: str, value: Any, **kwargs) -> str:
        """Update context memory with new information."""
        if "context_memory" not in self.task_context:
            self.task_context["context_memory"] = {}
        
        self.task_context["context_memory"][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step
        }
        
        return f"Updated context memory: {key} = {value}"

    async def _reflect_on_performance(self, **kwargs) -> Dict[str, Any]:
        """Reflect on recent performance and identify improvements."""
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "steps_completed": self.current_step,
            "recent_actions": self.execution_history[-5:] if self.execution_history else [],
            "success_rate": 0.0,
            "identified_patterns": [],
            "improvement_suggestions": []
        }
        
        # Calculate success rate
        if self.execution_history:
            successful_actions = sum(1 for action in self.execution_history if action.get("success", False))
            reflection["success_rate"] = successful_actions / len(self.execution_history)
        
        # Identify patterns (simplified)
        if reflection["success_rate"] < 0.7:
            reflection["improvement_suggestions"].append("Consider breaking down tasks into smaller steps")
        
        if self.current_step > self.max_steps * 0.8:
            reflection["improvement_suggestions"].append("Focus on efficiency and direct approaches")
        
        self.reflection_notes.append(reflection)
        return reflection

    async def _perform_quality_check(self, output: str, criteria: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Perform quality assurance check on output."""
        if criteria is None:
            criteria = ["completeness", "accuracy", "clarity", "relevance"]
        
        quality_check = {
            "timestamp": datetime.now().isoformat(),
            "output_length": len(output),
            "criteria_checked": criteria,
            "passed_checks": [],
            "failed_checks": [],
            "overall_score": 0.0,
            "recommendations": []
        }
        
        # Simple quality checks (would be enhanced with LLM evaluation)
        for criterion in criteria:
            if criterion == "completeness" and len(output) > 100:
                quality_check["passed_checks"].append(criterion)
            elif criterion == "clarity" and len(output.split('.')) > 2:
                quality_check["passed_checks"].append(criterion)
            elif criterion == "relevance":
                quality_check["passed_checks"].append(criterion)  # Assume relevant for now
            else:
                quality_check["failed_checks"].append(criterion)
        
        quality_check["overall_score"] = len(quality_check["passed_checks"]) / len(criteria)
        
        if quality_check["overall_score"] < 0.7:
            quality_check["recommendations"].append("Consider revising output for better quality")
        
        return quality_check

    async def cleanup(self):
        """Enhanced cleanup with performance summary."""
        try:
            # Generate session summary
            session_summary = {
                "session_id": self.task_context.get("session_id"),
                "duration": (datetime.now() - self.task_context.get("start_time", datetime.now())).total_seconds(),
                "total_steps": self.current_step,
                "total_actions": len(self.execution_history),
                "success_rate": self._calculate_session_success_rate(),
                "key_learnings": self.reflection_notes[-3:] if self.reflection_notes else []
            }
            
            logger.info(f"Enhanced Manus session completed: {session_summary}")
            
            # Cleanup MCP connections
            if self._initialized:
                await self.disconnect_mcp_server()
                self._initialized = False
            
            # Call parent cleanup
            await super().cleanup()
            
        except Exception as e:
            logger.error(f"Error during enhanced cleanup: {e}")

    def _calculate_session_success_rate(self) -> float:
        """Calculate the success rate for the current session."""
        if not self.execution_history:
            return 0.0
        
        successful_actions = sum(1 for action in self.execution_history if action.get("success", False))
        return successful_actions / len(self.execution_history)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from MCP servers."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without disconnected server's tools
        base_tools = [
            tool for tool in self.available_tools.tools 
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)