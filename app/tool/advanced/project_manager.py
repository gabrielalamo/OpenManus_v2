"""
Project Manager Tool
Advanced project management and organization capabilities
"""

import json
import os
import re
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pydantic import Field

from app.tool.base import BaseTool, ToolResult
from app.config import config


class ProjectManagerTool(BaseTool):
    """
    Advanced project management tool for organizing, tracking, and managing
    software development projects and tasks.
    """

    name: str = "project_manager"
    description: str = """
    Manage software development projects, track tasks, organize resources,
    and monitor progress. Create project structures, define milestones, 
    assign tasks, and generate reports on project status.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_project", "update_project", "list_projects", "get_project", 
                    "delete_project", "add_milestone", "update_milestone", "add_task", 
                    "update_task", "complete_task", "generate_report", "create_structure"
                ],
                "description": "The project management action to perform"
            },
            "project_id": {
                "type": "string",
                "description": "ID of the project to work with"
            },
            "project_name": {
                "type": "string",
                "description": "Name of the project (for create_project)"
            },
            "project_description": {
                "type": "string",
                "description": "Description of the project"
            },
            "project_type": {
                "type": "string",
                "enum": ["software", "data", "research", "documentation", "general"],
                "description": "Type of project to create"
            },
            "milestone_id": {
                "type": "string",
                "description": "ID of the milestone to work with"
            },
            "milestone_name": {
                "type": "string",
                "description": "Name of the milestone"
            },
            "milestone_description": {
                "type": "string",
                "description": "Description of the milestone"
            },
            "milestone_due_date": {
                "type": "string",
                "description": "Due date for the milestone (ISO format)"
            },
            "task_id": {
                "type": "string",
                "description": "ID of the task to work with"
            },
            "task_name": {
                "type": "string",
                "description": "Name of the task"
            },
            "task_description": {
                "type": "string",
                "description": "Description of the task"
            },
            "task_priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Priority level of the task"
            },
            "task_status": {
                "type": "string",
                "enum": ["not_started", "in_progress", "blocked", "completed"],
                "description": "Status of the task"
            },
            "task_assignee": {
                "type": "string",
                "description": "Person assigned to the task"
            },
            "task_due_date": {
                "type": "string",
                "description": "Due date for the task (ISO format)"
            },
            "task_dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of task IDs that this task depends on"
            },
            "report_type": {
                "type": "string",
                "enum": ["status", "milestone", "task", "timeline", "summary"],
                "description": "Type of report to generate"
            },
            "structure_template": {
                "type": "string",
                "enum": ["basic", "web", "api", "data_science", "documentation", "custom"],
                "description": "Template for project structure creation"
            },
            "custom_structure": {
                "type": "string",
                "description": "JSON string defining custom project structure"
            },
            "target_directory": {
                "type": "string",
                "description": "Target directory for project structure creation"
            }
        },
        "required": ["action"]
    }

    # Project storage
    projects: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    active_project: Optional[str] = Field(default=None)
    
    # Project templates
    structure_templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize project structure templates."""
        self.structure_templates = {
            "basic": {
                "directories": [
                    "src",
                    "docs",
                    "tests",
                    "resources"
                ],
                "files": {
                    "README.md": "# ${project_name}\n\n${project_description}\n\n## Getting Started\n\n## Features\n\n## License",
                    ".gitignore": "*.log\n__pycache__/\n*.py[cod]\n*$py.class\n.env\n.venv\nenv/\nvenv/\nENV/\n.idea/\n.vscode/\n",
                    "requirements.txt": "# Project dependencies\n",
                    "src/__init__.py": "",
                    "tests/__init__.py": ""
                }
            },
            "web": {
                "directories": [
                    "src",
                    "public",
                    "src/components",
                    "src/pages",
                    "src/styles",
                    "src/utils",
                    "tests"
                ],
                "files": {
                    "README.md": "# ${project_name}\n\n${project_description}\n\n## Getting Started\n\n## Features\n\n## License",
                    ".gitignore": "node_modules/\n.env\n.env.local\ndist/\nbuild/\n.DS_Store\n*.log\n",
                    "package.json": "{\n  \"name\": \"${project_id}\",\n  \"version\": \"0.1.0\",\n  \"private\": true,\n  \"scripts\": {\n    \"start\": \"echo 'Add start script'\",\n    \"build\": \"echo 'Add build script'\",\n    \"test\": \"echo 'Add test script'\"\n  }\n}",
                    "src/index.js": "// Main entry point\n",
                    "public/index.html": "<!DOCTYPE html>\n<html>\n<head>\n  <title>${project_name}</title>\n</head>\n<body>\n  <div id=\"root\"></div>\n</body>\n</html>"
                }
            },
            "api": {
                "directories": [
                    "src",
                    "src/controllers",
                    "src/models",
                    "src/routes",
                    "src/middleware",
                    "src/utils",
                    "tests",
                    "config"
                ],
                "files": {
                    "README.md": "# ${project_name} API\n\n${project_description}\n\n## API Documentation\n\n## Getting Started\n\n## Endpoints\n\n## License",
                    ".gitignore": "node_modules/\n.env\ndist/\n*.log\ncoverage/\n",
                    "package.json": "{\n  \"name\": \"${project_id}\",\n  \"version\": \"0.1.0\",\n  \"private\": true,\n  \"scripts\": {\n    \"start\": \"echo 'Add start script'\",\n    \"dev\": \"echo 'Add dev script'\",\n    \"test\": \"echo 'Add test script'\"\n  }\n}",
                    "src/index.js": "// API entry point\n",
                    "config/default.json": "{\n  \"port\": 3000,\n  \"environment\": \"development\"\n}"
                }
            },
            "data_science": {
                "directories": [
                    "data",
                    "data/raw",
                    "data/processed",
                    "notebooks",
                    "src",
                    "src/data",
                    "src/features",
                    "src/models",
                    "src/visualization",
                    "reports",
                    "reports/figures"
                ],
                "files": {
                    "README.md": "# ${project_name}\n\n${project_description}\n\n## Project Organization\n\n## Data Sources\n\n## Models\n\n## Results\n\n## License",
                    ".gitignore": "*.csv\n*.parquet\n*.h5\n*.pkl\n*.model\n__pycache__/\n*.py[cod]\n*$py.class\n.ipynb_checkpoints\n.env\n.venv\nenv/\nvenv/\n",
                    "requirements.txt": "# Data science dependencies\nnumpy\npandas\nscikit-learn\nmatplotlib\nseaborn\njupyter\n",
                    "notebooks/exploration.ipynb": "{\n \"cells\": [\n  {\n   \"cell_type\": \"markdown\",\n   \"metadata\": {},\n   \"source\": [\"# ${project_name} - Data Exploration\\n\", \"${project_description}\"]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"execution_count\": null,\n   \"metadata\": {},\n   \"outputs\": [],\n   \"source\": [\"import numpy as np\\n\", \"import pandas as pd\\n\", \"import matplotlib.pyplot as plt\\n\", \"import seaborn as sns\\n\", \"\\n\", \"# Your code here\"]\n  }\n ],\n \"metadata\": {\n  \"kernelspec\": {\n   \"display_name\": \"Python 3\",\n   \"language\": \"python\",\n   \"name\": \"python3\"\n  }\n },\n \"nbformat\": 4,\n \"nbformat_minor\": 4\n}"
                }
            },
            "documentation": {
                "directories": [
                    "docs",
                    "docs/user",
                    "docs/developer",
                    "docs/api",
                    "docs/tutorials",
                    "docs/images",
                    "examples"
                ],
                "files": {
                    "README.md": "# ${project_name} Documentation\n\n${project_description}\n\n## Overview\n\n## User Documentation\n\n## Developer Documentation\n\n## API Reference\n\n## Tutorials\n\n## License",
                    "docs/index.md": "# ${project_name}\n\n${project_description}\n\n## Documentation Sections\n\n- [User Guide](user/index.md)\n- [Developer Guide](developer/index.md)\n- [API Reference](api/index.md)\n- [Tutorials](tutorials/index.md)\n",
                    "docs/user/index.md": "# User Guide\n\n## Introduction\n\n## Installation\n\n## Basic Usage\n\n## Advanced Features\n",
                    "docs/developer/index.md": "# Developer Guide\n\n## Architecture\n\n## Contributing\n\n## Building from Source\n\n## Testing\n",
                    "mkdocs.yml": "site_name: ${project_name}\nnav:\n  - Home: index.md\n  - User Guide: user/index.md\n  - Developer Guide: developer/index.md\n  - API Reference: api/index.md\n  - Tutorials: tutorials/index.md\n"
                }
            }
        }

    async def execute(
        self,
        action: str,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
        project_type: Optional[str] = None,
        milestone_id: Optional[str] = None,
        milestone_name: Optional[str] = None,
        milestone_description: Optional[str] = None,
        milestone_due_date: Optional[str] = None,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
        task_priority: Optional[str] = None,
        task_status: Optional[str] = None,
        task_assignee: Optional[str] = None,
        task_due_date: Optional[str] = None,
        task_dependencies: Optional[List[str]] = None,
        report_type: Optional[str] = None,
        structure_template: Optional[str] = None,
        custom_structure: Optional[str] = None,
        target_directory: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the project management action."""
        
        try:
            if action == "create_project":
                return await self._create_project(project_name, project_description, project_type)
            elif action == "update_project":
                return await self._update_project(project_id, project_name, project_description, project_type)
            elif action == "list_projects":
                return await self._list_projects()
            elif action == "get_project":
                return await self._get_project(project_id)
            elif action == "delete_project":
                return await self._delete_project(project_id)
            elif action == "add_milestone":
                return await self._add_milestone(project_id, milestone_name, milestone_description, milestone_due_date)
            elif action == "update_milestone":
                return await self._update_milestone(project_id, milestone_id, milestone_name, milestone_description, milestone_due_date)
            elif action == "add_task":
                return await self._add_task(
                    project_id, task_name, task_description, task_priority, 
                    task_status, task_assignee, task_due_date, task_dependencies, milestone_id
                )
            elif action == "update_task":
                return await self._update_task(
                    project_id, task_id, task_name, task_description, task_priority, 
                    task_status, task_assignee, task_due_date, task_dependencies, milestone_id
                )
            elif action == "complete_task":
                return await self._complete_task(project_id, task_id)
            elif action == "generate_report":
                return await self._generate_report(project_id, report_type)
            elif action == "create_structure":
                return await self._create_structure(
                    project_id, structure_template, custom_structure, target_directory
                )
            else:
                return ToolResult(error=f"Unknown project management action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Project management error: {str(e)}")

    async def _create_project(
        self, 
        project_name: Optional[str], 
        project_description: Optional[str], 
        project_type: Optional[str]
    ) -> ToolResult:
        """Create a new project."""
        if not project_name:
            return ToolResult(error="Project name is required")
            
        # Generate project ID
        project_id = f"proj_{int(time.time())}_{project_name.lower().replace(' ', '_')}"
        
        # Create project structure
        project = {
            "id": project_id,
            "name": project_name,
            "description": project_description or "",
            "type": project_type or "general",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "milestones": [],
            "tasks": [],
            "metadata": {}
        }
        
        # Store project
        self.projects[project_id] = project
        
        # Set as active project if none is active
        if not self.active_project:
            self.active_project = project_id
            
        return ToolResult(
            output=f"Project '{project_name}' created with ID: {project_id}\n\n{self._format_project(project)}"
        )

    async def _update_project(
        self, 
        project_id: Optional[str], 
        project_name: Optional[str], 
        project_description: Optional[str], 
        project_type: Optional[str]
    ) -> ToolResult:
        """Update an existing project."""
        if not project_id:
            return ToolResult(error="Project ID is required")
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        project = self.projects[project_id]
        
        # Update fields if provided
        if project_name:
            project["name"] = project_name
            
        if project_description is not None:
            project["description"] = project_description
            
        if project_type:
            project["type"] = project_type
            
        # Update timestamp
        project["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Project '{project['name']}' updated successfully\n\n{self._format_project(project)}"
        )

    async def _list_projects(self) -> ToolResult:
        """List all projects."""
        if not self.projects:
            return ToolResult(output="No projects found")
            
        output = ["Projects:"]
        
        for project_id, project in self.projects.items():
            active_marker = " (active)" if project_id == self.active_project else ""
            output.append(f"\n{project['name']}{active_marker} (ID: {project_id})")
            output.append(f"  Type: {project['type']}")
            output.append(f"  Status: {project['status']}")
            output.append(f"  Created: {project['created_at']}")
            output.append(f"  Milestones: {len(project['milestones'])}")
            output.append(f"  Tasks: {len(project['tasks'])}")
            
        return ToolResult(output="\n".join(output))

    async def _get_project(self, project_id: Optional[str]) -> ToolResult:
        """Get details of a specific project."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        project = self.projects[project_id]
        return ToolResult(output=self._format_project(project))

    def _format_project(self, project: Dict[str, Any]) -> str:
        """Format project details for display."""
        output = [f"Project: {project['name']} (ID: {project['id']})"]
        output.append(f"Type: {project['type']}")
        output.append(f"Status: {project['status']}")
        output.append(f"Created: {project['created_at']}")
        output.append(f"Updated: {project['updated_at']}")
        
        if project.get("description"):
            output.append(f"\nDescription:\n{project['description']}")
            
        # Add milestones
        milestones = project.get("milestones", [])
        if milestones:
            output.append(f"\nMilestones ({len(milestones)}):")
            for i, milestone in enumerate(milestones):
                output.append(f"  {i+1}. {milestone['name']}")
                if milestone.get("due_date"):
                    output.append(f"     Due: {milestone['due_date']}")
                    
                # Count tasks in this milestone
                milestone_tasks = [
                    task for task in project.get("tasks", [])
                    if task.get("milestone_id") == milestone["id"]
                ]
                output.append(f"     Tasks: {len(milestone_tasks)}")
                
        # Add tasks
        tasks = project.get("tasks", [])
        if tasks:
            # Group tasks by status
            tasks_by_status = {}
            for task in tasks:
                status = task.get("status", "not_started")
                if status not in tasks_by_status:
                    tasks_by_status[status] = []
                tasks_by_status[status].append(task)
                
            output.append(f"\nTasks ({len(tasks)}):")
            
            # Show tasks by status
            for status, status_tasks in tasks_by_status.items():
                output.append(f"  {status.replace('_', ' ').title()} ({len(status_tasks)}):")
                for task in status_tasks[:5]:  # Show up to 5 tasks per status
                    priority_marker = {
                        "low": "🔵",
                        "medium": "🟢",
                        "high": "🟠",
                        "critical": "🔴"
                    }.get(task.get("priority", "medium"), "⚪")
                    
                    output.append(f"    {priority_marker} {task['name']}")
                    
                if len(status_tasks) > 5:
                    output.append(f"    ... and {len(status_tasks) - 5} more")
                    
        return "\n".join(output)

    async def _delete_project(self, project_id: Optional[str]) -> ToolResult:
        """Delete a project."""
        if not project_id:
            return ToolResult(error="Project ID is required")
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        # Get project name before deletion
        project_name = self.projects[project_id]["name"]
        
        # Delete project
        del self.projects[project_id]
        
        # Update active project if needed
        if self.active_project == project_id:
            self.active_project = next(iter(self.projects)) if self.projects else None
            
        return ToolResult(output=f"Project '{project_name}' (ID: {project_id}) deleted successfully")

    async def _add_milestone(
        self, 
        project_id: Optional[str], 
        milestone_name: Optional[str], 
        milestone_description: Optional[str], 
        milestone_due_date: Optional[str]
    ) -> ToolResult:
        """Add a milestone to a project."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        if not milestone_name:
            return ToolResult(error="Milestone name is required")
            
        # Validate due date if provided
        if milestone_due_date:
            try:
                datetime.fromisoformat(milestone_due_date)
            except ValueError:
                return ToolResult(error="Invalid due date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                
        # Generate milestone ID
        milestone_id = f"ms_{int(time.time())}_{milestone_name.lower().replace(' ', '_')}"
        
        # Create milestone
        milestone = {
            "id": milestone_id,
            "name": milestone_name,
            "description": milestone_description or "",
            "due_date": milestone_due_date,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Add to project
        project = self.projects[project_id]
        project["milestones"].append(milestone)
        project["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Milestone '{milestone_name}' added to project '{project['name']}'\n\n{self._format_milestone(milestone)}"
        )

    def _format_milestone(self, milestone: Dict[str, Any]) -> str:
        """Format milestone details for display."""
        output = [f"Milestone: {milestone['name']} (ID: {milestone['id']})"]
        output.append(f"Status: {milestone['status']}")
        
        if milestone.get("due_date"):
            output.append(f"Due Date: {milestone['due_date']}")
            
        if milestone.get("description"):
            output.append(f"\nDescription:\n{milestone['description']}")
            
        return "\n".join(output)

    async def _update_milestone(
        self, 
        project_id: Optional[str], 
        milestone_id: Optional[str], 
        milestone_name: Optional[str], 
        milestone_description: Optional[str], 
        milestone_due_date: Optional[str]
    ) -> ToolResult:
        """Update a milestone."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        if not milestone_id:
            return ToolResult(error="Milestone ID is required")
            
        # Find milestone
        project = self.projects[project_id]
        milestone_index = None
        
        for i, milestone in enumerate(project["milestones"]):
            if milestone["id"] == milestone_id:
                milestone_index = i
                break
                
        if milestone_index is None:
            return ToolResult(error=f"Milestone with ID '{milestone_id}' not found in project")
            
        milestone = project["milestones"][milestone_index]
        
        # Update fields if provided
        if milestone_name:
            milestone["name"] = milestone_name
            
        if milestone_description is not None:
            milestone["description"] = milestone_description
            
        if milestone_due_date:
            try:
                datetime.fromisoformat(milestone_due_date)
                milestone["due_date"] = milestone_due_date
            except ValueError:
                return ToolResult(error="Invalid due date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                
        # Update timestamp
        milestone["updated_at"] = datetime.now().isoformat()
        project["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Milestone '{milestone['name']}' updated successfully\n\n{self._format_milestone(milestone)}"
        )

    async def _add_task(
        self, 
        project_id: Optional[str], 
        task_name: Optional[str], 
        task_description: Optional[str], 
        task_priority: Optional[str], 
        task_status: Optional[str], 
        task_assignee: Optional[str], 
        task_due_date: Optional[str], 
        task_dependencies: Optional[List[str]], 
        milestone_id: Optional[str]
    ) -> ToolResult:
        """Add a task to a project."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        if not task_name:
            return ToolResult(error="Task name is required")
            
        project = self.projects[project_id]
        
        # Validate milestone if provided
        if milestone_id:
            milestone_exists = any(m["id"] == milestone_id for m in project["milestones"])
            if not milestone_exists:
                return ToolResult(error=f"Milestone with ID '{milestone_id}' not found in project")
                
        # Validate due date if provided
        if task_due_date:
            try:
                datetime.fromisoformat(task_due_date)
            except ValueError:
                return ToolResult(error="Invalid due date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                
        # Validate dependencies if provided
        if task_dependencies:
            project_task_ids = [task["id"] for task in project["tasks"]]
            invalid_dependencies = [dep for dep in task_dependencies if dep not in project_task_ids]
            if invalid_dependencies:
                return ToolResult(error=f"Invalid task dependencies: {', '.join(invalid_dependencies)}")
                
        # Generate task ID
        task_id = f"task_{int(time.time())}_{task_name.lower().replace(' ', '_')}"
        
        # Create task
        task = {
            "id": task_id,
            "name": task_name,
            "description": task_description or "",
            "priority": task_priority or "medium",
            "status": task_status or "not_started",
            "assignee": task_assignee,
            "due_date": task_due_date,
            "dependencies": task_dependencies or [],
            "milestone_id": milestone_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        # Add to project
        project["tasks"].append(task)
        project["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Task '{task_name}' added to project '{project['name']}'\n\n{self._format_task(task)}"
        )

    def _format_task(self, task: Dict[str, Any]) -> str:
        """Format task details for display."""
        output = [f"Task: {task['name']} (ID: {task['id']})"]
        output.append(f"Status: {task['status']}")
        output.append(f"Priority: {task['priority']}")
        
        if task.get("assignee"):
            output.append(f"Assignee: {task['assignee']}")
            
        if task.get("due_date"):
            output.append(f"Due Date: {task['due_date']}")
            
        if task.get("milestone_id"):
            output.append(f"Milestone: {task['milestone_id']}")
            
        if task.get("dependencies"):
            output.append(f"Dependencies: {', '.join(task['dependencies'])}")
            
        if task.get("description"):
            output.append(f"\nDescription:\n{task['description']}")
            
        return "\n".join(output)

    async def _update_task(
        self, 
        project_id: Optional[str], 
        task_id: Optional[str], 
        task_name: Optional[str], 
        task_description: Optional[str], 
        task_priority: Optional[str], 
        task_status: Optional[str], 
        task_assignee: Optional[str], 
        task_due_date: Optional[str], 
        task_dependencies: Optional[List[str]], 
        milestone_id: Optional[str]
    ) -> ToolResult:
        """Update a task."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        if not task_id:
            return ToolResult(error="Task ID is required")
            
        # Find task
        project = self.projects[project_id]
        task_index = None
        
        for i, task in enumerate(project["tasks"]):
            if task["id"] == task_id:
                task_index = i
                break
                
        if task_index is None:
            return ToolResult(error=f"Task with ID '{task_id}' not found in project")
            
        task = project["tasks"][task_index]
        
        # Update fields if provided
        if task_name:
            task["name"] = task_name
            
        if task_description is not None:
            task["description"] = task_description
            
        if task_priority:
            task["priority"] = task_priority
            
        if task_status:
            old_status = task["status"]
            task["status"] = task_status
            
            # If completing the task, set completed_at
            if task_status == "completed" and old_status != "completed":
                task["completed_at"] = datetime.now().isoformat()
            elif task_status != "completed":
                task["completed_at"] = None
                
        if task_assignee is not None:
            task["assignee"] = task_assignee
            
        if task_due_date:
            try:
                datetime.fromisoformat(task_due_date)
                task["due_date"] = task_due_date
            except ValueError:
                return ToolResult(error="Invalid due date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                
        if task_dependencies is not None:
            # Validate dependencies
            if task_dependencies:
                project_task_ids = [t["id"] for t in project["tasks"]]
                invalid_dependencies = [dep for dep in task_dependencies if dep not in project_task_ids]
                if invalid_dependencies:
                    return ToolResult(error=f"Invalid task dependencies: {', '.join(invalid_dependencies)}")
                    
            task["dependencies"] = task_dependencies
            
        if milestone_id is not None:
            # Validate milestone if provided
            if milestone_id:
                milestone_exists = any(m["id"] == milestone_id for m in project["milestones"])
                if not milestone_exists:
                    return ToolResult(error=f"Milestone with ID '{milestone_id}' not found in project")
                    
            task["milestone_id"] = milestone_id
            
        # Update timestamp
        task["updated_at"] = datetime.now().isoformat()
        project["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Task '{task['name']}' updated successfully\n\n{self._format_task(task)}"
        )

    async def _complete_task(self, project_id: Optional[str], task_id: Optional[str]) -> ToolResult:
        """Mark a task as completed."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        if not task_id:
            return ToolResult(error="Task ID is required")
            
        # Find task
        project = self.projects[project_id]
        task_index = None
        
        for i, task in enumerate(project["tasks"]):
            if task["id"] == task_id:
                task_index = i
                break
                
        if task_index is None:
            return ToolResult(error=f"Task with ID '{task_id}' not found in project")
            
        task = project["tasks"][task_index]
        
        # Update task status
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        task["updated_at"] = datetime.now().isoformat()
        project["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Task '{task['name']}' marked as completed\n\n{self._format_task(task)}"
        )

    async def _generate_report(self, project_id: Optional[str], report_type: Optional[str]) -> ToolResult:
        """Generate a project report."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        if not report_type:
            report_type = "status"  # Default report type
            
        project = self.projects[project_id]
        
        if report_type == "status":
            return ToolResult(output=self._generate_status_report(project))
        elif report_type == "milestone":
            return ToolResult(output=self._generate_milestone_report(project))
        elif report_type == "task":
            return ToolResult(output=self._generate_task_report(project))
        elif report_type == "timeline":
            return ToolResult(output=self._generate_timeline_report(project))
        elif report_type == "summary":
            return ToolResult(output=self._generate_summary_report(project))
        else:
            return ToolResult(error=f"Unknown report type: {report_type}")

    def _generate_status_report(self, project: Dict[str, Any]) -> str:
        """Generate a status report for the project."""
        tasks = project.get("tasks", [])
        milestones = project.get("milestones", [])
        
        # Calculate task statistics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.get("status") == "completed")
        in_progress_tasks = sum(1 for task in tasks if task.get("status") == "in_progress")
        blocked_tasks = sum(1 for task in tasks if task.get("status") == "blocked")
        not_started_tasks = sum(1 for task in tasks if task.get("status") == "not_started")
        
        completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate milestone statistics
        total_milestones = len(milestones)
        completed_milestones = 0
        upcoming_milestones = []
        
        now = datetime.now()
        
        for milestone in milestones:
            # Check if all tasks for this milestone are completed
            milestone_tasks = [task for task in tasks if task.get("milestone_id") == milestone["id"]]
            if all(task.get("status") == "completed" for task in milestone_tasks):
                completed_milestones += 1
            
            # Check for upcoming milestones
            if milestone.get("due_date"):
                due_date = datetime.fromisoformat(milestone["due_date"])
                if due_date > now:
                    upcoming_milestones.append(milestone)
        
        # Sort upcoming milestones by due date
        upcoming_milestones.sort(key=lambda m: m.get("due_date", "9999-12-31"))
        
        # Generate report
        output = [f"# Status Report: {project['name']}"]
        output.append(f"Generated: {datetime.now().isoformat()}")
        output.append(f"Project Status: {project['status']}")
        output.append(f"Last Updated: {project['updated_at']}")
        
        output.append("\n## Progress Summary")
        output.append(f"Overall Completion: {completion_percentage:.1f}%")
        output.append(f"Tasks: {completed_tasks}/{total_tasks} completed")
        output.append(f"Milestones: {completed_milestones}/{total_milestones} completed")
        
        output.append("\n## Task Breakdown")
        output.append(f"- Completed: {completed_tasks} ({completed_tasks/total_tasks*100:.1f}% if total_tasks > 0 else 0)")
        output.append(f"- In Progress: {in_progress_tasks} ({in_progress_tasks/total_tasks*100:.1f}% if total_tasks > 0 else 0)")
        output.append(f"- Blocked: {blocked_tasks} ({blocked_tasks/total_tasks*100:.1f}% if total_tasks > 0 else 0)")
        output.append(f"- Not Started: {not_started_tasks} ({not_started_tasks/total_tasks*100:.1f}% if total_tasks > 0 else 0)")
        
        if upcoming_milestones:
            output.append("\n## Upcoming Milestones")
            for i, milestone in enumerate(upcoming_milestones[:3]):  # Show top 3
                due_date = datetime.fromisoformat(milestone["due_date"])
                days_remaining = (due_date - now).days
                output.append(f"{i+1}. {milestone['name']}")
                output.append(f"   Due: {milestone['due_date']} ({days_remaining} days remaining)")
                
                # Show tasks for this milestone
                milestone_tasks = [task for task in tasks if task.get("milestone_id") == milestone["id"]]
                completed_milestone_tasks = sum(1 for task in milestone_tasks if task.get("status") == "completed")
                output.append(f"   Tasks: {completed_milestone_tasks}/{len(milestone_tasks)} completed")
                
        # Show recent activity
        output.append("\n## Recent Activity")
        
        # Sort tasks by updated_at
        recent_tasks = sorted(
            tasks, 
            key=lambda t: t.get("updated_at", "1970-01-01"), 
            reverse=True
        )[:5]  # Get 5 most recently updated
        
        for task in recent_tasks:
            status = task.get("status", "unknown").replace("_", " ").title()
            output.append(f"- {task['name']} - {status} ({task['updated_at']})")
            
        return "\n".join(output)

    def _generate_milestone_report(self, project: Dict[str, Any]) -> str:
        """Generate a milestone report for the project."""
        milestones = project.get("milestones", [])
        tasks = project.get("tasks", [])
        
        if not milestones:
            return f"No milestones found for project '{project['name']}'"
            
        # Generate report
        output = [f"# Milestone Report: {project['name']}"]
        output.append(f"Generated: {datetime.now().isoformat()}")
        
        # Sort milestones by due date
        sorted_milestones = sorted(
            milestones, 
            key=lambda m: m.get("due_date", "9999-12-31")
        )
        
        for milestone in sorted_milestones:
            # Get tasks for this milestone
            milestone_tasks = [task for task in tasks if task.get("milestone_id") == milestone["id"]]
            total_tasks = len(milestone_tasks)
            completed_tasks = sum(1 for task in milestone_tasks if task.get("status") == "completed")
            completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Determine status
            status = "Not Started"
            if completion_percentage == 100:
                status = "Completed"
            elif completion_percentage > 0:
                status = "In Progress"
                
            # Format due date
            due_date_str = "No due date"
            days_remaining = None
            
            if milestone.get("due_date"):
                due_date = datetime.fromisoformat(milestone["due_date"])
                due_date_str = milestone["due_date"]
                days_remaining = (due_date - datetime.now()).days
                
            output.append(f"\n## {milestone['name']}")
            output.append(f"Status: {status}")
            output.append(f"Progress: {completion_percentage:.1f}% ({completed_tasks}/{total_tasks} tasks)")
            output.append(f"Due Date: {due_date_str}")
            
            if days_remaining is not None:
                if days_remaining < 0:
                    output.append(f"Overdue by {abs(days_remaining)} days")
                else:
                    output.append(f"{days_remaining} days remaining")
                    
            if milestone.get("description"):
                output.append(f"\nDescription:\n{milestone['description']}")
                
            # List tasks
            if milestone_tasks:
                output.append("\nTasks:")
                
                # Group tasks by status
                tasks_by_status = {}
                for task in milestone_tasks:
                    status = task.get("status", "not_started")
                    if status not in tasks_by_status:
                        tasks_by_status[status] = []
                    tasks_by_status[status].append(task)
                    
                for status, status_tasks in tasks_by_status.items():
                    status_display = status.replace("_", " ").title()
                    output.append(f"  {status_display} ({len(status_tasks)}):")
                    
                    for task in status_tasks:
                        priority_marker = {
                            "low": "🔵",
                            "medium": "🟢",
                            "high": "🟠",
                            "critical": "🔴"
                        }.get(task.get("priority", "medium"), "⚪")
                        
                        output.append(f"    {priority_marker} {task['name']}")
                        
        return "\n".join(output)

    def _generate_task_report(self, project: Dict[str, Any]) -> str:
        """Generate a task report for the project."""
        tasks = project.get("tasks", [])
        
        if not tasks:
            return f"No tasks found for project '{project['name']}'"
            
        # Generate report
        output = [f"# Task Report: {project['name']}"]
        output.append(f"Generated: {datetime.now().isoformat()}")
        
        # Group tasks by status
        tasks_by_status = {}
        for task in tasks:
            status = task.get("status", "not_started")
            if status not in tasks_by_status:
                tasks_by_status[status] = []
            tasks_by_status[status].append(task)
            
        # Define status order
        status_order = ["completed", "in_progress", "blocked", "not_started"]
        
        # Add tasks by status
        for status in status_order:
            if status not in tasks_by_status:
                continue
                
            status_tasks = tasks_by_status[status]
            status_display = status.replace("_", " ").title()
            output.append(f"\n## {status_display} ({len(status_tasks)})")
            
            # Sort tasks by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_tasks = sorted(
                status_tasks,
                key=lambda t: priority_order.get(t.get("priority", "medium"), 2)
            )
            
            for task in sorted_tasks:
                priority_display = task.get("priority", "medium").title()
                output.append(f"\n### {task['name']}")
                output.append(f"ID: {task['id']}")
                output.append(f"Priority: {priority_display}")
                
                if task.get("assignee"):
                    output.append(f"Assignee: {task['assignee']}")
                    
                if task.get("due_date"):
                    output.append(f"Due Date: {task['due_date']}")
                    
                if task.get("milestone_id"):
                    # Find milestone name
                    milestone_name = "Unknown"
                    for milestone in project.get("milestones", []):
                        if milestone["id"] == task["milestone_id"]:
                            milestone_name = milestone["name"]
                            break
                    output.append(f"Milestone: {milestone_name}")
                    
                if task.get("dependencies"):
                    # Find dependency names
                    dependency_names = []
                    for dep_id in task["dependencies"]:
                        for dep_task in tasks:
                            if dep_task["id"] == dep_id:
                                dependency_names.append(dep_task["name"])
                                break
                    output.append(f"Dependencies: {', '.join(dependency_names)}")
                    
                if task.get("description"):
                    output.append(f"\nDescription:\n{task['description']}")
                    
        return "\n".join(output)

    def _generate_timeline_report(self, project: Dict[str, Any]) -> str:
        """Generate a timeline report for the project."""
        milestones = project.get("milestones", [])
        tasks = project.get("tasks", [])
        
        # Filter items with due dates
        dated_milestones = [m for m in milestones if m.get("due_date")]
        dated_tasks = [t for t in tasks if t.get("due_date")]
        
        if not dated_milestones and not dated_tasks:
            return f"No timeline data found for project '{project['name']}'. Add due dates to milestones and tasks."
            
        # Combine and sort all items by due date
        timeline_items = []
        
        for milestone in dated_milestones:
            timeline_items.append({
                "type": "milestone",
                "id": milestone["id"],
                "name": milestone["name"],
                "due_date": milestone["due_date"],
                "status": "completed" if all(
                    task.get("status") == "completed" 
                    for task in tasks 
                    if task.get("milestone_id") == milestone["id"]
                ) else "active"
            })
            
        for task in dated_tasks:
            timeline_items.append({
                "type": "task",
                "id": task["id"],
                "name": task["name"],
                "due_date": task["due_date"],
                "status": task.get("status", "not_started"),
                "milestone_id": task.get("milestone_id")
            })
            
        # Sort by due date
        timeline_items.sort(key=lambda item: item["due_date"])
        
        # Generate report
        output = [f"# Timeline Report: {project['name']}"]
        output.append(f"Generated: {datetime.now().isoformat()}")
        
        # Group by month
        current_month = None
        
        for item in timeline_items:
            due_date = datetime.fromisoformat(item["due_date"])
            month_year = due_date.strftime("%B %Y")
            
            if month_year != current_month:
                current_month = month_year
                output.append(f"\n## {month_year}")
                
            # Format date
            date_str = due_date.strftime("%Y-%m-%d")
            
            # Format item
            if item["type"] == "milestone":
                status_marker = "✅" if item["status"] == "completed" else "🔶"
                output.append(f"{date_str} {status_marker} **Milestone**: {item['name']}")
            else:  # task
                status_marker = {
                    "completed": "✅",
                    "in_progress": "🔄",
                    "blocked": "🚫",
                    "not_started": "⏳"
                }.get(item["status"], "⏳")
                
                # Find milestone name if applicable
                milestone_info = ""
                if item.get("milestone_id"):
                    for milestone in milestones:
                        if milestone["id"] == item["milestone_id"]:
                            milestone_info = f" (Milestone: {milestone['name']})"
                            break
                            
                output.append(f"{date_str} {status_marker} {item['name']}{milestone_info}")
                
        return "\n".join(output)

    def _generate_summary_report(self, project: Dict[str, Any]) -> str:
        """Generate a summary report for the project."""
        tasks = project.get("tasks", [])
        milestones = project.get("milestones", [])
        
        # Calculate overall statistics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.get("status") == "completed")
        completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate days active
        created_date = datetime.fromisoformat(project["created_at"])
        days_active = (datetime.now() - created_date).days
        
        # Find upcoming due dates
        now = datetime.now()
        upcoming_due_dates = []
        
        for task in tasks:
            if task.get("due_date") and task.get("status") != "completed":
                due_date = datetime.fromisoformat(task["due_date"])
                if due_date > now:
                    upcoming_due_dates.append({
                        "type": "task",
                        "name": task["name"],
                        "due_date": due_date,
                        "days_remaining": (due_date - now).days
                    })
                    
        for milestone in milestones:
            if milestone.get("due_date"):
                due_date = datetime.fromisoformat(milestone["due_date"])
                if due_date > now:
                    # Check if milestone is completed
                    milestone_tasks = [task for task in tasks if task.get("milestone_id") == milestone["id"]]
                    if not all(task.get("status") == "completed" for task in milestone_tasks):
                        upcoming_due_dates.append({
                            "type": "milestone",
                            "name": milestone["name"],
                            "due_date": due_date,
                            "days_remaining": (due_date - now).days
                        })
                        
        # Sort by days remaining
        upcoming_due_dates.sort(key=lambda item: item["days_remaining"])
        
        # Generate report
        output = [f"# Project Summary: {project['name']}"]
        output.append(f"Generated: {datetime.now().isoformat()}")
        
        output.append("\n## Overview")
        output.append(f"Status: {project['status']}")
        output.append(f"Type: {project['type']}")
        output.append(f"Days Active: {days_active}")
        output.append(f"Last Updated: {project['updated_at']}")
        
        if project.get("description"):
            output.append(f"\nDescription:\n{project['description']}")
            
        output.append("\n## Progress")
        output.append(f"Overall Completion: {completion_percentage:.1f}%")
        output.append(f"Tasks: {completed_tasks}/{total_tasks} completed")
        output.append(f"Milestones: {len(milestones)}")
        
        # Task breakdown by priority
        output.append("\n## Task Breakdown by Priority")
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for task in tasks:
            priority = task.get("priority", "medium")
            priority_counts[priority] += 1
            
        for priority, count in priority_counts.items():
            if count > 0:
                output.append(f"- {priority.title()}: {count} tasks")
                
        # Upcoming due dates
        if upcoming_due_dates:
            output.append("\n## Upcoming Due Dates")
            
            for i, item in enumerate(upcoming_due_dates[:5]):  # Show top 5
                item_type = "Milestone" if item["type"] == "milestone" else "Task"
                output.append(f"{i+1}. {item['name']} ({item_type})")
                output.append(f"   Due: {item['due_date'].isoformat()} ({item['days_remaining']} days remaining)")
                
        # Blocked tasks
        blocked_tasks = [task for task in tasks if task.get("status") == "blocked"]
        if blocked_tasks:
            output.append("\n## Blocked Tasks")
            
            for task in blocked_tasks:
                output.append(f"- {task['name']}")
                
        return "\n".join(output)

    async def _create_structure(
        self, 
        project_id: Optional[str], 
        structure_template: Optional[str], 
        custom_structure: Optional[str], 
        target_directory: Optional[str]
    ) -> ToolResult:
        """Create a project directory structure."""
        if not project_id:
            # If no project_id is provided, use the active project
            if not self.active_project:
                return ToolResult(error="No active project. Please specify a project_id or set an active project.")
            project_id = self.active_project
            
        if project_id not in self.projects:
            return ToolResult(error=f"Project with ID '{project_id}' not found")
            
        project = self.projects[project_id]
        
        # Determine structure to use
        structure = None
        
        if custom_structure:
            try:
                structure = json.loads(custom_structure)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid custom structure JSON")
        elif structure_template:
            if structure_template not in self.structure_templates:
                return ToolResult(error=f"Unknown structure template: {structure_template}")
            structure = self.structure_templates[structure_template]
        else:
            # Default to basic template
            structure = self.structure_templates["basic"]
            
        # Determine target directory
        if not target_directory:
            target_directory = os.path.join(config.workspace_root, project_id)
            
        # Create directories
        created_dirs = []
        created_files = []
        
        try:
            # Create base directory
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
                created_dirs.append(target_directory)
                
            # Create subdirectories
            for directory in structure.get("directories", []):
                dir_path = os.path.join(target_directory, directory)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    created_dirs.append(dir_path)
                    
            # Create files
            for file_path, content in structure.get("files", {}).items():
                # Replace variables in content
                variables = {
                    "project_name": project["name"],
                    "project_id": project["id"],
                    "project_description": project["description"],
                    "created_at": project["created_at"],
                    "updated_at": project["updated_at"]
                }
                
                for var_name, var_value in variables.items():
                    content = content.replace(f"${{{var_name}}}", str(var_value))
                    
                # Write file
                full_path = os.path.join(target_directory, file_path)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, "w") as f:
                    f.write(content)
                    
                created_files.append(full_path)
                
            # Update project metadata
            if "metadata" not in project:
                project["metadata"] = {}
                
            project["metadata"]["structure"] = {
                "template": structure_template or "custom",
                "created_at": datetime.now().isoformat(),
                "target_directory": target_directory
            }
            
            project["updated_at"] = datetime.now().isoformat()
            
            return ToolResult(
                output=f"Project structure created successfully for '{project['name']}'\n\n"
                       f"Target Directory: {target_directory}\n"
                       f"Directories Created: {len(created_dirs)}\n"
                       f"Files Created: {len(created_files)}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error creating project structure: {str(e)}")