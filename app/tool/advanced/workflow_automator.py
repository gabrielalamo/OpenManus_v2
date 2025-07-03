"""
Workflow Automator Tool
Automates complex workflows and processes
"""

import json
import os
import re
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class WorkflowAutomatorTool(BaseTool):
    """
    Advanced workflow automation tool for creating, managing, and executing
    complex workflows and business processes.
    """

    name: str = "workflow_automator"
    description: str = """
    Create, manage, and execute automated workflows. Define process steps,
    conditions, actions, and triggers to automate complex business processes.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_workflow", "execute_workflow", "list_workflows", 
                    "get_workflow", "update_workflow", "delete_workflow",
                    "add_step", "update_step", "delete_step", "get_execution_history"
                ],
                "description": "The workflow automation action to perform"
            },
            "workflow_id": {
                "type": "string",
                "description": "ID of the workflow to work with"
            },
            "workflow_name": {
                "type": "string",
                "description": "Name of the workflow (for create_workflow)"
            },
            "workflow_definition": {
                "type": "string",
                "description": "JSON string containing workflow definition"
            },
            "step_id": {
                "type": "string",
                "description": "ID of the workflow step to work with"
            },
            "step_definition": {
                "type": "string",
                "description": "JSON string containing step definition"
            },
            "input_data": {
                "type": "string",
                "description": "JSON string containing input data for workflow execution"
            },
            "execution_id": {
                "type": "string",
                "description": "ID of a specific workflow execution"
            }
        },
        "required": ["action"]
    }

    # Workflow storage
    workflows: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    execution_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    active_executions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    async def execute(
        self,
        action: str,
        workflow_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        workflow_definition: Optional[str] = None,
        step_id: Optional[str] = None,
        step_definition: Optional[str] = None,
        input_data: Optional[str] = None,
        execution_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the workflow automation action."""
        
        try:
            if action == "create_workflow":
                return await self._create_workflow(workflow_name, workflow_definition)
            elif action == "execute_workflow":
                return await self._execute_workflow(workflow_id, input_data)
            elif action == "list_workflows":
                return await self._list_workflows()
            elif action == "get_workflow":
                return await self._get_workflow(workflow_id)
            elif action == "update_workflow":
                return await self._update_workflow(workflow_id, workflow_definition)
            elif action == "delete_workflow":
                return await self._delete_workflow(workflow_id)
            elif action == "add_step":
                return await self._add_step(workflow_id, step_definition)
            elif action == "update_step":
                return await self._update_step(workflow_id, step_id, step_definition)
            elif action == "delete_step":
                return await self._delete_step(workflow_id, step_id)
            elif action == "get_execution_history":
                return await self._get_execution_history(workflow_id, execution_id)
            else:
                return ToolResult(error=f"Unknown workflow automation action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Workflow automation error: {str(e)}")

    async def _create_workflow(self, workflow_name: Optional[str], workflow_definition: Optional[str]) -> ToolResult:
        """Create a new workflow."""
        if not workflow_name:
            return ToolResult(error="Workflow name is required")
            
        # Generate workflow ID
        workflow_id = f"wf_{int(time.time())}_{workflow_name.lower().replace(' ', '_')}"
        
        # Parse workflow definition if provided
        workflow_data = {}
        if workflow_definition:
            try:
                workflow_data = json.loads(workflow_definition)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid workflow definition JSON")
        
        # Create workflow structure
        workflow = {
            "id": workflow_id,
            "name": workflow_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "steps": workflow_data.get("steps", []),
            "triggers": workflow_data.get("triggers", []),
            "variables": workflow_data.get("variables", {}),
            "metadata": workflow_data.get("metadata", {})
        }
        
        # Validate workflow structure
        validation_result = self._validate_workflow(workflow)
        if not validation_result["valid"]:
            return ToolResult(error=f"Invalid workflow: {validation_result['errors']}")
        
        # Store workflow
        self.workflows[workflow_id] = workflow
        self.execution_history[workflow_id] = []
        
        return ToolResult(
            output=f"Workflow '{workflow_name}' created with ID: {workflow_id}\n\n{self._format_workflow(workflow)}"
        )

    async def _execute_workflow(self, workflow_id: Optional[str], input_data: Optional[str]) -> ToolResult:
        """Execute a workflow with the provided input data."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
        
        workflow = self.workflows[workflow_id]
        
        # Parse input data
        execution_data = {}
        if input_data:
            try:
                execution_data = json.loads(input_data)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid input data JSON")
        
        # Generate execution ID
        execution_id = f"exec_{workflow_id}_{int(time.time())}"
        
        # Initialize execution context
        context = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "current_step": 0,
            "variables": {**workflow.get("variables", {}), **execution_data},
            "results": {},
            "logs": []
        }
        
        # Store active execution
        self.active_executions[execution_id] = context
        
        # Start execution in background
        asyncio.create_task(self._run_workflow_execution(execution_id, workflow))
        
        return ToolResult(
            output=f"Workflow execution started with ID: {execution_id}\n"
                   f"Workflow: {workflow['name']}\n"
                   f"Started at: {context['start_time']}\n"
                   f"Status: {context['status']}"
        )

    async def _run_workflow_execution(self, execution_id: str, workflow: Dict[str, Any]) -> None:
        """Run workflow execution in background."""
        context = self.active_executions[execution_id]
        steps = workflow.get("steps", [])
        
        try:
            # Log execution start
            self._log_execution_event(context, "Workflow execution started")
            
            # Execute each step in sequence
            for i, step in enumerate(steps):
                context["current_step"] = i
                
                # Log step start
                self._log_execution_event(context, f"Starting step {i+1}/{len(steps)}: {step.get('name', 'Unnamed step')}")
                
                # Check conditions
                if not self._evaluate_conditions(step.get("conditions", []), context["variables"]):
                    self._log_execution_event(context, f"Step {i+1} skipped - conditions not met")
                    continue
                
                # Execute step
                try:
                    step_result = await self._execute_step(step, context["variables"])
                    context["results"][f"step_{i}"] = step_result
                    context["variables"].update(step_result.get("outputs", {}))
                    self._log_execution_event(context, f"Step {i+1} completed successfully")
                except Exception as e:
                    error_msg = f"Error executing step {i+1}: {str(e)}"
                    self._log_execution_event(context, error_msg, level="error")
                    context["status"] = "failed"
                    context["error"] = error_msg
                    break
            
            # Set final status if not already failed
            if context["status"] != "failed":
                context["status"] = "completed"
                self._log_execution_event(context, "Workflow execution completed successfully")
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error in workflow execution: {str(e)}"
            self._log_execution_event(context, error_msg, level="error")
            context["status"] = "failed"
            context["error"] = error_msg
            
        finally:
            # Record end time
            context["end_time"] = datetime.now().isoformat()
            
            # Move from active to history
            if execution_id in self.active_executions:
                execution_record = self.active_executions.pop(execution_id)
                
                # Add to history
                workflow_id = execution_record["workflow_id"]
                if workflow_id not in self.execution_history:
                    self.execution_history[workflow_id] = []
                self.execution_history[workflow_id].append(execution_record)

    def _log_execution_event(self, context: Dict[str, Any], message: str, level: str = "info") -> None:
        """Log an event in the execution context."""
        context["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        })

    def _evaluate_conditions(self, conditions: List[Dict[str, Any]], variables: Dict[str, Any]) -> bool:
        """Evaluate conditions against variables."""
        if not conditions:
            return True  # No conditions means always execute
            
        for condition in conditions:
            condition_type = condition.get("type", "equals")
            variable = condition.get("variable")
            value = condition.get("value")
            
            if not variable or variable not in variables:
                return False
                
            var_value = variables[variable]
            
            if condition_type == "equals" and var_value != value:
                return False
            elif condition_type == "not_equals" and var_value == value:
                return False
            elif condition_type == "greater_than" and not (var_value > value):
                return False
            elif condition_type == "less_than" and not (var_value < value):
                return False
            elif condition_type == "contains" and value not in var_value:
                return False
            elif condition_type == "not_contains" and value in var_value:
                return False
                
        return True

    async def _execute_step(self, step: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step."""
        step_type = step.get("type", "manual")
        
        # Replace variables in parameters
        parameters = self._replace_variables(step.get("parameters", {}), variables)
        
        if step_type == "manual":
            # Manual steps just pass through
            return {
                "status": "completed",
                "outputs": {}
            }
            
        elif step_type == "http_request":
            # Simulate HTTP request
            return {
                "status": "completed",
                "outputs": {
                    "response_code": 200,
                    "response_body": "Simulated HTTP response"
                }
            }
            
        elif step_type == "data_transformation":
            # Simulate data transformation
            return {
                "status": "completed",
                "outputs": {
                    "transformed_data": f"Transformed: {parameters.get('input_data', 'No data')}"
                }
            }
            
        elif step_type == "conditional":
            # Evaluate condition and execute appropriate branch
            condition_result = self._evaluate_conditions(step.get("conditions", []), variables)
            branch = "true_branch" if condition_result else "false_branch"
            
            branch_steps = step.get(branch, [])
            branch_results = {}
            
            for i, branch_step in enumerate(branch_steps):
                branch_step_result = await self._execute_step(branch_step, variables)
                branch_results[f"{branch}_{i}"] = branch_step_result
                variables.update(branch_step_result.get("outputs", {}))
                
            return {
                "status": "completed",
                "condition_result": condition_result,
                "executed_branch": branch,
                "outputs": branch_results
            }
            
        else:
            # Unknown step type
            return {
                "status": "skipped",
                "error": f"Unknown step type: {step_type}",
                "outputs": {}
            }

    def _replace_variables(self, obj: Any, variables: Dict[str, Any]) -> Any:
        """Replace variable placeholders in strings with their values."""
        if isinstance(obj, str):
            # Replace ${variable} with its value
            pattern = r'\${([^}]+)}'
            
            def replace_var(match):
                var_name = match.group(1)
                return str(variables.get(var_name, match.group(0)))
                
            return re.sub(pattern, replace_var, obj)
            
        elif isinstance(obj, dict):
            return {k: self._replace_variables(v, variables) for k, v in obj.items()}
            
        elif isinstance(obj, list):
            return [self._replace_variables(item, variables) for item in obj]
            
        else:
            return obj

    def _validate_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow structure and content."""
        errors = []
        
        # Check required fields
        for field in ["id", "name", "steps"]:
            if field not in workflow:
                errors.append(f"Missing required field: {field}")
                
        # Validate steps
        steps = workflow.get("steps", [])
        if not isinstance(steps, list):
            errors.append("Steps must be a list")
        else:
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be an object")
                elif "type" not in step:
                    errors.append(f"Step {i} missing required field: type")
                    
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def _list_workflows(self) -> ToolResult:
        """List all available workflows."""
        if not self.workflows:
            return ToolResult(output="No workflows found")
            
        output = ["Available Workflows:"]
        
        for workflow_id, workflow in self.workflows.items():
            executions = len(self.execution_history.get(workflow_id, []))
            status = workflow.get("status", "active")
            output.append(f"\n{workflow['name']} (ID: {workflow_id})")
            output.append(f"  Status: {status}")
            output.append(f"  Steps: {len(workflow.get('steps', []))}")
            output.append(f"  Executions: {executions}")
            output.append(f"  Created: {workflow.get('created_at', 'Unknown')}")
            
        return ToolResult(output="\n".join(output))

    async def _get_workflow(self, workflow_id: Optional[str]) -> ToolResult:
        """Get details of a specific workflow."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        workflow = self.workflows[workflow_id]
        return ToolResult(output=self._format_workflow(workflow))

    def _format_workflow(self, workflow: Dict[str, Any]) -> str:
        """Format workflow details for display."""
        output = [f"Workflow: {workflow['name']} (ID: {workflow['id']})"]
        output.append(f"Status: {workflow.get('status', 'active')}")
        output.append(f"Created: {workflow.get('created_at', 'Unknown')}")
        output.append(f"Updated: {workflow.get('updated_at', 'Unknown')}")
        
        # Add steps
        steps = workflow.get("steps", [])
        output.append(f"\nSteps ({len(steps)}):")
        for i, step in enumerate(steps):
            step_name = step.get("name", f"Step {i+1}")
            step_type = step.get("type", "unknown")
            output.append(f"  {i+1}. {step_name} (Type: {step_type})")
            
        # Add triggers
        triggers = workflow.get("triggers", [])
        if triggers:
            output.append(f"\nTriggers ({len(triggers)}):")
            for i, trigger in enumerate(triggers):
                trigger_type = trigger.get("type", "unknown")
                output.append(f"  {i+1}. Type: {trigger_type}")
                
        # Add variables
        variables = workflow.get("variables", {})
        if variables:
            output.append(f"\nVariables ({len(variables)}):")
            for name, value in variables.items():
                output.append(f"  {name}: {value}")
                
        return "\n".join(output)

    async def _update_workflow(self, workflow_id: Optional[str], workflow_definition: Optional[str]) -> ToolResult:
        """Update an existing workflow."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        if not workflow_definition:
            return ToolResult(error="Workflow definition is required")
            
        # Parse workflow definition
        try:
            workflow_data = json.loads(workflow_definition)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid workflow definition JSON")
            
        # Update workflow
        workflow = self.workflows[workflow_id]
        
        # Update fields
        for field in ["name", "steps", "triggers", "variables", "metadata"]:
            if field in workflow_data:
                workflow[field] = workflow_data[field]
                
        # Update timestamp
        workflow["updated_at"] = datetime.now().isoformat()
        
        # Validate updated workflow
        validation_result = self._validate_workflow(workflow)
        if not validation_result["valid"]:
            return ToolResult(error=f"Invalid workflow: {validation_result['errors']}")
            
        return ToolResult(
            output=f"Workflow '{workflow['name']}' updated successfully\n\n{self._format_workflow(workflow)}"
        )

    async def _delete_workflow(self, workflow_id: Optional[str]) -> ToolResult:
        """Delete a workflow."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        # Get workflow name before deletion
        workflow_name = self.workflows[workflow_id]["name"]
        
        # Delete workflow
        del self.workflows[workflow_id]
        
        # Delete execution history
        if workflow_id in self.execution_history:
            del self.execution_history[workflow_id]
            
        return ToolResult(output=f"Workflow '{workflow_name}' (ID: {workflow_id}) deleted successfully")

    async def _add_step(self, workflow_id: Optional[str], step_definition: Optional[str]) -> ToolResult:
        """Add a step to a workflow."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        if not step_definition:
            return ToolResult(error="Step definition is required")
            
        # Parse step definition
        try:
            step_data = json.loads(step_definition)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid step definition JSON")
            
        # Validate step
        if "type" not in step_data:
            return ToolResult(error="Step definition must include 'type' field")
            
        # Add step to workflow
        workflow = self.workflows[workflow_id]
        workflow["steps"].append(step_data)
        workflow["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Step added to workflow '{workflow['name']}'\n\n{self._format_workflow(workflow)}"
        )

    async def _update_step(self, workflow_id: Optional[str], step_id: Optional[str], step_definition: Optional[str]) -> ToolResult:
        """Update a workflow step."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        if not step_id:
            return ToolResult(error="Step ID is required")
            
        if not step_definition:
            return ToolResult(error="Step definition is required")
            
        # Parse step definition
        try:
            step_data = json.loads(step_definition)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid step definition JSON")
            
        # Find and update step
        workflow = self.workflows[workflow_id]
        steps = workflow.get("steps", [])
        
        try:
            step_index = int(step_id)
            if step_index < 0 or step_index >= len(steps):
                return ToolResult(error=f"Step index {step_index} out of range (0-{len(steps)-1})")
                
            # Update step
            steps[step_index] = step_data
            workflow["updated_at"] = datetime.now().isoformat()
            
            return ToolResult(
                output=f"Step {step_index} updated in workflow '{workflow['name']}'\n\n{self._format_workflow(workflow)}"
            )
        except ValueError:
            # Try to find step by ID
            for i, step in enumerate(steps):
                if step.get("id") == step_id:
                    steps[i] = step_data
                    workflow["updated_at"] = datetime.now().isoformat()
                    
                    return ToolResult(
                        output=f"Step '{step_id}' updated in workflow '{workflow['name']}'\n\n{self._format_workflow(workflow)}"
                    )
                    
            return ToolResult(error=f"Step with ID '{step_id}' not found in workflow")

    async def _delete_step(self, workflow_id: Optional[str], step_id: Optional[str]) -> ToolResult:
        """Delete a workflow step."""
        if not workflow_id:
            return ToolResult(error="Workflow ID is required")
            
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        if not step_id:
            return ToolResult(error="Step ID is required")
            
        # Find and delete step
        workflow = self.workflows[workflow_id]
        steps = workflow.get("steps", [])
        
        try:
            step_index = int(step_id)
            if step_index < 0 or step_index >= len(steps):
                return ToolResult(error=f"Step index {step_index} out of range (0-{len(steps)-1})")
                
            # Delete step
            deleted_step = steps.pop(step_index)
            workflow["updated_at"] = datetime.now().isoformat()
            
            return ToolResult(
                output=f"Step {step_index} deleted from workflow '{workflow['name']}'\n\n{self._format_workflow(workflow)}"
            )
        except ValueError:
            # Try to find step by ID
            for i, step in enumerate(steps):
                if step.get("id") == step_id:
                    deleted_step = steps.pop(i)
                    workflow["updated_at"] = datetime.now().isoformat()
                    
                    return ToolResult(
                        output=f"Step '{step_id}' deleted from workflow '{workflow['name']}'\n\n{self._format_workflow(workflow)}"
                    )
                    
            return ToolResult(error=f"Step with ID '{step_id}' not found in workflow")

    async def _get_execution_history(self, workflow_id: Optional[str], execution_id: Optional[str]) -> ToolResult:
        """Get execution history for a workflow or specific execution."""
        if not workflow_id and not execution_id:
            return ToolResult(error="Either workflow_id or execution_id is required")
            
        # Get history for specific execution
        if execution_id:
            # Check active executions
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                return ToolResult(output=self._format_execution(execution))
                
            # Check execution history
            for wf_id, executions in self.execution_history.items():
                for execution in executions:
                    if execution.get("execution_id") == execution_id:
                        return ToolResult(output=self._format_execution(execution))
                        
            return ToolResult(error=f"Execution with ID '{execution_id}' not found")
            
        # Get history for workflow
        if workflow_id not in self.workflows:
            return ToolResult(error=f"Workflow with ID '{workflow_id}' not found")
            
        # Get active executions for this workflow
        active_executions = [
            execution for execution in self.active_executions.values()
            if execution.get("workflow_id") == workflow_id
        ]
        
        # Get completed executions
        completed_executions = self.execution_history.get(workflow_id, [])
        
        if not active_executions and not completed_executions:
            return ToolResult(output=f"No execution history found for workflow '{workflow_id}'")
            
        # Format output
        output = [f"Execution History for Workflow: {self.workflows[workflow_id]['name']} (ID: {workflow_id})"]
        
        if active_executions:
            output.append(f"\nActive Executions ({len(active_executions)}):")
            for execution in active_executions:
                output.append(f"  ID: {execution.get('execution_id')}")
                output.append(f"  Started: {execution.get('start_time')}")
                output.append(f"  Status: {execution.get('status')}")
                output.append(f"  Current Step: {execution.get('current_step')}")
                output.append("")
                
        if completed_executions:
            output.append(f"\nCompleted Executions ({len(completed_executions)}):")
            for execution in completed_executions[-10:]:  # Show last 10
                output.append(f"  ID: {execution.get('execution_id')}")
                output.append(f"  Started: {execution.get('start_time')}")
                output.append(f"  Ended: {execution.get('end_time', 'Unknown')}")
                output.append(f"  Status: {execution.get('status')}")
                output.append("")
                
        return ToolResult(output="\n".join(output))

    def _format_execution(self, execution: Dict[str, Any]) -> str:
        """Format execution details for display."""
        output = [f"Execution: {execution.get('execution_id')}"]
        output.append(f"Workflow: {execution.get('workflow_id')}")
        output.append(f"Status: {execution.get('status')}")
        output.append(f"Started: {execution.get('start_time')}")
        
        if execution.get('end_time'):
            output.append(f"Ended: {execution.get('end_time')}")
            
        if execution.get('error'):
            output.append(f"Error: {execution.get('error')}")
            
        # Add results
        results = execution.get('results', {})
        if results:
            output.append("\nResults:")
            for step_id, result in results.items():
                output.append(f"  {step_id}:")
                output.append(f"    Status: {result.get('status')}")
                
                # Add outputs
                outputs = result.get('outputs', {})
                if outputs:
                    output.append("    Outputs:")
                    for key, value in outputs.items():
                        output.append(f"      {key}: {value}")
                        
        # Add logs
        logs = execution.get('logs', [])
        if logs:
            output.append("\nLogs:")
            for log in logs[-10:]:  # Show last 10 logs
                output.append(f"  [{log.get('timestamp')}] [{log.get('level')}] {log.get('message')}")
                
        return "\n".join(output)