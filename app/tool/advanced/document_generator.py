"""
Document Generator Tool
Generates comprehensive documentation for code, projects, and APIs
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class DocumentGeneratorTool(BaseTool):
    """
    Advanced documentation generation tool for creating comprehensive
    documentation for code, projects, APIs, and user guides.
    """

    name: str = "document_generator"
    description: str = """
    Generate comprehensive documentation for code, projects, APIs, and user guides.
    Creates well-structured, professional documentation in various formats.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "generate_readme", "create_api_docs", "document_code", 
                    "create_user_guide", "generate_project_docs", "create_architecture_doc",
                    "generate_changelog", "create_contributing_guide"
                ],
                "description": "The type of documentation to generate"
            },
            "project_path": {
                "type": "string",
                "description": "Path to the project directory or file to document"
            },
            "output_path": {
                "type": "string",
                "description": "Path where the documentation should be saved"
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "html", "rst", "text"],
                "description": "Output format for the documentation"
            },
            "template": {
                "type": "string",
                "description": "Template name or path to use for documentation"
            },
            "project_info": {
                "type": "object",
                "description": "Additional project information for documentation"
            },
            "include_sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific sections to include in the documentation"
            },
            "exclude_sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Sections to exclude from the documentation"
            },
            "language": {
                "type": "string",
                "description": "Programming language of the code to document"
            }
        },
        "required": ["action"]
    }

    # Document templates and history
    document_templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    generation_history: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize document templates."""
        
        # README template
        self.document_templates["readme"] = {
            "sections": [
                {"name": "title", "heading": "# {project_name}", "required": True},
                {"name": "badges", "heading": "", "required": False},
                {"name": "description", "heading": "## Description", "required": True},
                {"name": "features", "heading": "## Features", "required": True},
                {"name": "installation", "heading": "## Installation", "required": True},
                {"name": "usage", "heading": "## Usage", "required": True},
                {"name": "examples", "heading": "## Examples", "required": False},
                {"name": "api", "heading": "## API", "required": False},
                {"name": "configuration", "heading": "## Configuration", "required": False},
                {"name": "contributing", "heading": "## Contributing", "required": False},
                {"name": "license", "heading": "## License", "required": True},
            ],
            "format": "markdown"
        }
        
        # API documentation template
        self.document_templates["api_docs"] = {
            "sections": [
                {"name": "title", "heading": "# API Documentation", "required": True},
                {"name": "overview", "heading": "## Overview", "required": True},
                {"name": "authentication", "heading": "## Authentication", "required": True},
                {"name": "endpoints", "heading": "## Endpoints", "required": True},
                {"name": "request_format", "heading": "## Request Format", "required": True},
                {"name": "response_format", "heading": "## Response Format", "required": True},
                {"name": "error_handling", "heading": "## Error Handling", "required": True},
                {"name": "rate_limits", "heading": "## Rate Limits", "required": False},
                {"name": "examples", "heading": "## Examples", "required": True},
                {"name": "changelog", "heading": "## Changelog", "required": False},
            ],
            "format": "markdown"
        }
        
        # User guide template
        self.document_templates["user_guide"] = {
            "sections": [
                {"name": "title", "heading": "# User Guide", "required": True},
                {"name": "introduction", "heading": "## Introduction", "required": True},
                {"name": "getting_started", "heading": "## Getting Started", "required": True},
                {"name": "installation", "heading": "## Installation", "required": True},
                {"name": "basic_usage", "heading": "## Basic Usage", "required": True},
                {"name": "advanced_usage", "heading": "## Advanced Usage", "required": False},
                {"name": "configuration", "heading": "## Configuration", "required": False},
                {"name": "troubleshooting", "heading": "## Troubleshooting", "required": True},
                {"name": "faq", "heading": "## FAQ", "required": False},
                {"name": "glossary", "heading": "## Glossary", "required": False},
            ],
            "format": "markdown"
        }
        
        # Architecture documentation template
        self.document_templates["architecture"] = {
            "sections": [
                {"name": "title", "heading": "# Architecture Documentation", "required": True},
                {"name": "overview", "heading": "## System Overview", "required": True},
                {"name": "components", "heading": "## Components", "required": True},
                {"name": "data_flow", "heading": "## Data Flow", "required": True},
                {"name": "interfaces", "heading": "## Interfaces", "required": True},
                {"name": "deployment", "heading": "## Deployment", "required": False},
                {"name": "security", "heading": "## Security Considerations", "required": False},
                {"name": "performance", "heading": "## Performance Considerations", "required": False},
                {"name": "scalability", "heading": "## Scalability", "required": False},
                {"name": "decisions", "heading": "## Design Decisions", "required": False},
            ],
            "format": "markdown"
        }

    async def execute(
        self,
        action: str,
        project_path: Optional[str] = None,
        output_path: Optional[str] = None,
        format: str = "markdown",
        template: Optional[str] = None,
        project_info: Optional[Dict[str, Any]] = None,
        include_sections: Optional[List[str]] = None,
        exclude_sections: Optional[List[str]] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the documentation generation action."""
        
        try:
            if action == "generate_readme":
                return await self._generate_readme(project_path, output_path, format, project_info)
            elif action == "create_api_docs":
                return await self._create_api_docs(project_path, output_path, format, project_info)
            elif action == "document_code":
                return await self._document_code(project_path, output_path, format, language)
            elif action == "create_user_guide":
                return await self._create_user_guide(project_path, output_path, format, project_info)
            elif action == "generate_project_docs":
                return await self._generate_project_docs(project_path, output_path, format, project_info)
            elif action == "create_architecture_doc":
                return await self._create_architecture_doc(project_path, output_path, format, project_info)
            elif action == "generate_changelog":
                return await self._generate_changelog(project_path, output_path, format)
            elif action == "create_contributing_guide":
                return await self._create_contributing_guide(project_path, output_path, format, project_info)
            else:
                return ToolResult(error=f"Unknown documentation action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Documentation generation error: {str(e)}")

    async def _generate_readme(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        project_info: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Generate a README file for a project."""
        
        if not project_path:
            return ToolResult(error="Project path is required for README generation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "README.md")
        
        # Get project information
        project_data = project_info or {}
        if not project_data.get("project_name"):
            project_data["project_name"] = os.path.basename(os.path.abspath(project_path))
        
        # Analyze project to gather information
        project_analysis = await self._analyze_project(project_path)
        
        # Merge analysis with provided info
        for key, value in project_analysis.items():
            if key not in project_data:
                project_data[key] = value
        
        # Generate README content
        readme_template = self.document_templates["readme"]
        content = await self._generate_document_from_template(readme_template, project_data, format)
        
        # Write to file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log generation
            self.generation_history.append({
                "type": "readme",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format
            })
            
            return ToolResult(output=f"README generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write README: {str(e)}")

    async def _create_api_docs(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        project_info: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Create API documentation."""
        
        if not project_path:
            return ToolResult(error="Project path is required for API documentation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "docs", "api.md")
        
        # Get project information
        project_data = project_info or {}
        
        # Analyze API to gather information
        api_analysis = await self._analyze_api(project_path)
        
        # Merge analysis with provided info
        for key, value in api_analysis.items():
            if key not in project_data:
                project_data[key] = value
        
        # Generate API documentation content
        api_template = self.document_templates["api_docs"]
        content = await self._generate_document_from_template(api_template, project_data, format)
        
        # Write to file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log generation
            self.generation_history.append({
                "type": "api_docs",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format
            })
            
            return ToolResult(output=f"API documentation generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write API documentation: {str(e)}")

    async def _document_code(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        language: Optional[str]
    ) -> ToolResult:
        """Generate code documentation."""
        
        if not project_path:
            return ToolResult(error="Project path is required for code documentation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "docs", "code")
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(project_path)
        
        # Analyze code to gather information
        code_analysis = await self._analyze_code_for_docs(project_path, language)
        
        # Generate documentation for each module/file
        documentation = {}
        
        for module_name, module_info in code_analysis["modules"].items():
            module_doc = await self._generate_module_documentation(module_name, module_info, format)
            documentation[module_name] = module_doc
        
        # Generate index file
        index_content = await self._generate_documentation_index(code_analysis, documentation.keys(), format)
        
        # Write files
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Write index file
            index_file = os.path.join(output_path, f"index.{self._get_extension(format)}")
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            # Write module documentation files
            for module_name, content in documentation.items():
                safe_name = module_name.replace("/", "_").replace("\\", "_")
                module_file = os.path.join(output_path, f"{safe_name}.{self._get_extension(format)}")
                with open(module_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Log generation
            self.generation_history.append({
                "type": "code_docs",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format,
                "language": language,
                "modules_documented": len(documentation)
            })
            
            return ToolResult(output=f"Code documentation generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write code documentation: {str(e)}")

    async def _create_user_guide(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        project_info: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Create a user guide."""
        
        if not project_path:
            return ToolResult(error="Project path is required for user guide generation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "docs", "user_guide.md")
        
        # Get project information
        project_data = project_info or {}
        if not project_data.get("project_name"):
            project_data["project_name"] = os.path.basename(os.path.abspath(project_path))
        
        # Analyze project to gather information
        project_analysis = await self._analyze_project(project_path)
        
        # Merge analysis with provided info
        for key, value in project_analysis.items():
            if key not in project_data:
                project_data[key] = value
        
        # Generate user guide content
        user_guide_template = self.document_templates["user_guide"]
        content = await self._generate_document_from_template(user_guide_template, project_data, format)
        
        # Write to file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log generation
            self.generation_history.append({
                "type": "user_guide",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format
            })
            
            return ToolResult(output=f"User guide generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write user guide: {str(e)}")

    async def _generate_project_docs(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        project_info: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Generate comprehensive project documentation."""
        
        if not project_path:
            return ToolResult(error="Project path is required for project documentation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "docs")
        
        # Create docs directory
        os.makedirs(output_path, exist_ok=True)
        
        # Generate all documentation types
        results = []
        
        # Generate README
        readme_result = await self._generate_readme(
            project_path, 
            os.path.join(project_path, "README.md"), 
            format, 
            project_info
        )
        results.append(("README", readme_result))
        
        # Generate API docs if applicable
        api_path = os.path.join(output_path, "api.md")
        api_result = await self._create_api_docs(
            project_path,
            api_path,
            format,
            project_info
        )
        results.append(("API Documentation", api_result))
        
        # Generate code documentation
        code_docs_path = os.path.join(output_path, "code")
        code_result = await self._document_code(
            project_path,
            code_docs_path,
            format,
            None  # Auto-detect language
        )
        results.append(("Code Documentation", code_result))
        
        # Generate user guide
        user_guide_path = os.path.join(output_path, "user_guide.md")
        user_guide_result = await self._create_user_guide(
            project_path,
            user_guide_path,
            format,
            project_info
        )
        results.append(("User Guide", user_guide_result))
        
        # Generate architecture documentation
        arch_path = os.path.join(output_path, "architecture.md")
        arch_result = await self._create_architecture_doc(
            project_path,
            arch_path,
            format,
            project_info
        )
        results.append(("Architecture Documentation", arch_result))
        
        # Generate index file
        index_content = await self._generate_docs_index(results, project_info, format)
        index_path = os.path.join(output_path, f"index.{self._get_extension(format)}")
        
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            # Log generation
            self.generation_history.append({
                "type": "project_docs",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format,
                "components": [r[0] for r in results]
            })
            
            return ToolResult(output=f"Project documentation generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write project documentation: {str(e)}")

    async def _create_architecture_doc(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        project_info: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Create architecture documentation."""
        
        if not project_path:
            return ToolResult(error="Project path is required for architecture documentation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "docs", "architecture.md")
        
        # Get project information
        project_data = project_info or {}
        if not project_data.get("project_name"):
            project_data["project_name"] = os.path.basename(os.path.abspath(project_path))
        
        # Analyze project architecture
        architecture_analysis = await self._analyze_architecture(project_path)
        
        # Merge analysis with provided info
        for key, value in architecture_analysis.items():
            if key not in project_data:
                project_data[key] = value
        
        # Generate architecture documentation content
        architecture_template = self.document_templates["architecture"]
        content = await self._generate_document_from_template(architecture_template, project_data, format)
        
        # Write to file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log generation
            self.generation_history.append({
                "type": "architecture_doc",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format
            })
            
            return ToolResult(output=f"Architecture documentation generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write architecture documentation: {str(e)}")

    async def _generate_changelog(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str
    ) -> ToolResult:
        """Generate a changelog from version control history."""
        
        if not project_path:
            return ToolResult(error="Project path is required for changelog generation")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "CHANGELOG.md")
        
        # This would normally analyze git history
        # For now, create a placeholder changelog
        
        changelog_content = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core functionality
- Basic documentation

### Changed
- Improved performance
- Updated dependencies

### Fixed
- Various bug fixes
"""
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(changelog_content)
            
            # Log generation
            self.generation_history.append({
                "type": "changelog",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format
            })
            
            return ToolResult(output=f"Changelog generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write changelog: {str(e)}")

    async def _create_contributing_guide(
        self, 
        project_path: Optional[str],
        output_path: Optional[str],
        format: str,
        project_info: Optional[Dict[str, Any]]
    ) -> ToolResult:
        """Create a contributing guide."""
        
        if not project_path:
            return ToolResult(error="Project path is required for contributing guide")
        
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.join(project_path, "CONTRIBUTING.md")
        
        # Get project information
        project_data = project_info or {}
        if not project_data.get("project_name"):
            project_data["project_name"] = os.path.basename(os.path.abspath(project_path))
        
        # Generate contributing guide content
        contributing_content = f"""# Contributing to {project_data.get('project_name')}

Thank you for considering contributing to {project_data.get('project_name')}!

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported
- Use the bug report template
- Include detailed steps to reproduce
- Describe the expected behavior
- Include screenshots if applicable

### Suggesting Features

- Check if the feature has already been suggested
- Use the feature request template
- Describe the feature in detail
- Explain why this feature would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/username/{project_data.get('project_name')}.git

# Install dependencies
npm install  # or pip install -r requirements.txt

# Run tests
npm test  # or pytest
```

## Coding Guidelines

- Follow the existing code style
- Write tests for new features
- Update documentation for changes
- Keep pull requests focused on a single topic

## License

By contributing, you agree that your contributions will be licensed under the project's license.
"""
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(contributing_content)
            
            # Log generation
            self.generation_history.append({
                "type": "contributing_guide",
                "timestamp": datetime.now().isoformat(),
                "project_path": project_path,
                "output_path": output_path,
                "format": format
            })
            
            return ToolResult(output=f"Contributing guide generated successfully at {output_path}")
        except Exception as e:
            return ToolResult(error=f"Failed to write contributing guide: {str(e)}")

    # Helper methods

    async def _analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze a project to gather information for documentation."""
        project_info = {
            "project_name": os.path.basename(os.path.abspath(project_path)),
            "description": "A comprehensive project for solving specific problems",
            "features": [
                "Feature 1: Core functionality",
                "Feature 2: Advanced capabilities",
                "Feature 3: Integration options"
            ],
            "installation": [
                "```bash",
                "git clone https://github.com/username/project.git",
                "cd project",
                "pip install -r requirements.txt  # or npm install",
                "```"
            ],
            "usage": [
                "```python",
                "import project",
                "",
                "# Initialize",
                "client = project.Client()",
                "",
                "# Use functionality",
                "result = client.process('data')",
                "print(result)",
                "```"
            ],
            "license": "MIT"
        }
        
        # Check for package.json
        package_json_path = os.path.join(project_path, "package.json")
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                
                project_info["project_name"] = package_data.get("name", project_info["project_name"])
                project_info["description"] = package_data.get("description", project_info["description"])
                project_info["license"] = package_data.get("license", project_info["license"])
                
                # Update installation instructions for npm
                project_info["installation"] = [
                    "```bash",
                    f"npm install {package_data.get('name', 'project')}",
                    "```"
                ]
                
                # Update usage example for JavaScript
                project_info["usage"] = [
                    "```javascript",
                    f"const {package_data.get('name', 'project')} = require('{package_data.get('name', 'project')}');",
                    "",
                    "// Use functionality",
                    f"const result = {package_data.get('name', 'project')}.process('data');",
                    "console.log(result);",
                    "```"
                ]
            except Exception:
                pass
        
        # Check for setup.py
        setup_py_path = os.path.join(project_path, "setup.py")
        if os.path.exists(setup_py_path):
            try:
                with open(setup_py_path, 'r', encoding='utf-8') as f:
                    setup_content = f.read()
                
                # Extract project name
                name_match = re.search(r'name=[\'"]([^\'"]+)[\'"]', setup_content)
                if name_match:
                    project_info["project_name"] = name_match.group(1)
                
                # Extract description
                desc_match = re.search(r'description=[\'"]([^\'"]+)[\'"]', setup_content)
                if desc_match:
                    project_info["description"] = desc_match.group(1)
                
                # Update installation instructions for pip
                project_info["installation"] = [
                    "```bash",
                    f"pip install {project_info['project_name']}",
                    "```"
                ]
            except Exception:
                pass
        
        return project_info

    async def _analyze_api(self, project_path: str) -> Dict[str, Any]:
        """Analyze API to gather information for documentation."""
        api_info = {
            "api_name": "Project API",
            "overview": "This API provides access to core functionality of the project.",
            "authentication": "API uses token-based authentication. Include the token in the Authorization header.",
            "endpoints": [
                {
                    "path": "/api/v1/resource",
                    "method": "GET",
                    "description": "Retrieve resources",
                    "parameters": [
                        {"name": "limit", "type": "integer", "description": "Maximum number of resources to return"}
                    ],
                    "responses": [
                        {"code": 200, "description": "Success", "example": '{"data": [...]}'},
                        {"code": 401, "description": "Unauthorized", "example": '{"error": "Invalid token"}'}
                    ]
                },
                {
                    "path": "/api/v1/resource",
                    "method": "POST",
                    "description": "Create a new resource",
                    "parameters": [
                        {"name": "name", "type": "string", "description": "Resource name"},
                        {"name": "value", "type": "string", "description": "Resource value"}
                    ],
                    "responses": [
                        {"code": 201, "description": "Created", "example": '{"id": "123", "name": "Resource"}'},
                        {"code": 400, "description": "Bad Request", "example": '{"error": "Invalid parameters"}'}
                    ]
                }
            ],
            "request_format": "All requests should be in JSON format.",
            "response_format": "All responses are in JSON format.",
            "error_handling": "Errors are returned with appropriate HTTP status codes and error messages in the response body.",
            "rate_limits": "API is rate limited to 100 requests per minute per API key."
        }
        
        # Look for API definition files
        api_files = []
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith((".yaml", ".yml", ".json")) and ("api" in file.lower() or "swagger" in file.lower() or "openapi" in file.lower()):
                    api_files.append(os.path.join(root, file))
        
        # If API definition files found, try to parse them
        if api_files:
            for api_file in api_files:
                try:
                    if api_file.endswith((".yaml", ".yml")):
                        # This would use a YAML parser in a real implementation
                        pass
                    elif api_file.endswith(".json"):
                        with open(api_file, 'r', encoding='utf-8') as f:
                            api_def = json.load(f)
                        
                        # Extract information from OpenAPI/Swagger JSON
                        if "info" in api_def:
                            api_info["api_name"] = api_def["info"].get("title", api_info["api_name"])
                            api_info["overview"] = api_def["info"].get("description", api_info["overview"])
                        
                        # Extract endpoints
                        if "paths" in api_def:
                            api_info["endpoints"] = []
                            for path, methods in api_def["paths"].items():
                                for method, details in methods.items():
                                    if method.lower() in ["get", "post", "put", "delete", "patch"]:
                                        endpoint = {
                                            "path": path,
                                            "method": method.upper(),
                                            "description": details.get("summary", ""),
                                            "parameters": [],
                                            "responses": []
                                        }
                                        
                                        # Extract parameters
                                        if "parameters" in details:
                                            for param in details["parameters"]:
                                                endpoint["parameters"].append({
                                                    "name": param.get("name", ""),
                                                    "type": param.get("schema", {}).get("type", "string"),
                                                    "description": param.get("description", "")
                                                })
                                        
                                        # Extract responses
                                        if "responses" in details:
                                            for code, response in details["responses"].items():
                                                endpoint["responses"].append({
                                                    "code": int(code),
                                                    "description": response.get("description", ""),
                                                    "example": "Example response"  # Would extract from examples in real implementation
                                                })
                                        
                                        api_info["endpoints"].append(endpoint)
                except Exception:
                    continue
        
        return api_info

    async def _analyze_code_for_docs(self, project_path: str, language: str) -> Dict[str, Any]:
        """Analyze code to gather information for documentation."""
        code_info = {
            "language": language,
            "modules": {}
        }
        
        # Define file extensions for different languages
        extensions = {
            "python": [".py"],
            "javascript": [".js"],
            "typescript": [".ts"],
            "java": [".java"],
            "csharp": [".cs"],
            "go": [".go"],
            "rust": [".rs"],
            "php": [".php"],
            "ruby": [".rb"]
        }
        
        lang_extensions = extensions.get(language, [".py"])
        
        # Find all code files
        code_files = []
        for root, _, files in os.walk(project_path):
            for file in files:
                if any(file.endswith(ext) for ext in lang_extensions):
                    code_files.append(os.path.join(root, file))
        
        # Process each file
        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get relative path for module name
                rel_path = os.path.relpath(file_path, project_path)
                module_name = rel_path.replace("\\", "/")
                
                # Extract module information
                module_info = await self._extract_module_info(content, language)
                
                code_info["modules"][module_name] = {
                    "path": file_path,
                    "classes": module_info.get("classes", []),
                    "functions": module_info.get("functions", []),
                    "description": module_info.get("description", ""),
                    "imports": module_info.get("imports", [])
                }
            except Exception:
                continue
        
        return code_info

    async def _analyze_architecture(self, project_path: str) -> Dict[str, Any]:
        """Analyze project architecture."""
        architecture_info = {
            "components": [],
            "data_flow": [],
            "interfaces": [],
            "deployment": {},
            "security": [],
            "performance": [],
            "scalability": [],
            "decisions": []
        }
        
        # This would be a more sophisticated analysis in a real implementation
        # For now, return placeholder data
        
        architecture_info["components"] = [
            {
                "name": "Frontend",
                "description": "User interface component",
                "technologies": ["React", "TypeScript", "CSS"],
                "responsibilities": ["User interaction", "Data display", "Input validation"]
            },
            {
                "name": "Backend API",
                "description": "Server-side application logic",
                "technologies": ["Node.js", "Express", "MongoDB"],
                "responsibilities": ["Business logic", "Data processing", "Authentication"]
            },
            {
                "name": "Database",
                "description": "Data storage",
                "technologies": ["MongoDB"],
                "responsibilities": ["Data persistence", "Data retrieval", "Data integrity"]
            }
        ]
        
        architecture_info["data_flow"] = [
            {
                "from": "Frontend",
                "to": "Backend API",
                "description": "HTTP/REST requests",
                "data": "User input, authentication tokens"
            },
            {
                "from": "Backend API",
                "to": "Database",
                "description": "Database queries",
                "data": "CRUD operations"
            },
            {
                "from": "Database",
                "to": "Backend API",
                "description": "Query results",
                "data": "Stored data, query results"
            },
            {
                "from": "Backend API",
                "to": "Frontend",
                "description": "HTTP/REST responses",
                "data": "Processed data, status codes"
            }
        ]
        
        architecture_info["interfaces"] = [
            {
                "name": "REST API",
                "description": "HTTP-based API for client-server communication",
                "endpoints": ["/api/v1/resource", "/api/v1/auth"]
            },
            {
                "name": "Database Interface",
                "description": "Interface for database operations",
                "methods": ["create", "read", "update", "delete"]
            }
        ]
        
        return architecture_info

    async def _extract_module_info(self, content: str, language: str) -> Dict[str, Any]:
        """Extract information from a code module."""
        module_info = {
            "classes": [],
            "functions": [],
            "description": "",
            "imports": []
        }
        
        if language == "python":
            # Extract module docstring
            module_docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if module_docstring_match:
                module_info["description"] = module_docstring_match.group(1).strip()
            
            # Extract imports
            import_matches = re.finditer(r'^\s*(?:import|from)\s+([^\s]+)', content, re.MULTILINE)
            for match in import_matches:
                module_info["imports"].append(match.group(1))
            
            # Extract classes
            class_matches = re.finditer(r'^\s*class\s+(\w+)(?:\(([^)]*)\))?:(.*?)(?=^\s*class|\Z)', content, re.DOTALL | re.MULTILINE)
            for match in class_matches:
                class_name = match.group(1)
                class_bases = match.group(2) or ""
                class_body = match.group(3)
                
                # Extract class docstring
                class_docstring = ""
                docstring_match = re.search(r'"""(.*?)"""', class_body, re.DOTALL)
                if docstring_match:
                    class_docstring = docstring_match.group(1).strip()
                
                # Extract methods
                methods = []
                method_matches = re.finditer(r'^\s{4}def\s+(\w+)\s*\(([^)]*)\):(.*?)(?=^\s{4}def|\Z)', class_body, re.DOTALL | re.MULTILINE)
                for method_match in method_matches:
                    method_name = method_match.group(1)
                    method_params = method_match.group(2)
                    method_body = method_match.group(3)
                    
                    # Extract method docstring
                    method_docstring = ""
                    method_doc_match = re.search(r'"""(.*?)"""', method_body, re.DOTALL)
                    if method_doc_match:
                        method_docstring = method_doc_match.group(1).strip()
                    
                    methods.append({
                        "name": method_name,
                        "parameters": method_params,
                        "description": method_docstring
                    })
                
                module_info["classes"].append({
                    "name": class_name,
                    "bases": class_bases,
                    "description": class_docstring,
                    "methods": methods
                })
            
            # Extract functions
            function_matches = re.finditer(r'^\s*def\s+(\w+)\s*\(([^)]*)\):(.*?)(?=^\s*def|^\s*class|\Z)', content, re.DOTALL | re.MULTILINE)
            for match in function_matches:
                function_name = match.group(1)
                function_params = match.group(2)
                function_body = match.group(3)
                
                # Extract function docstring
                function_docstring = ""
                docstring_match = re.search(r'"""(.*?)"""', function_body, re.DOTALL)
                if docstring_match:
                    function_docstring = docstring_match.group(1).strip()
                
                module_info["functions"].append({
                    "name": function_name,
                    "parameters": function_params,
                    "description": function_docstring
                })
        
        elif language in ["javascript", "typescript"]:
            # Extract file description from JSDoc
            file_doc_match = re.search(r'/\*\*(.*?)\*/', content, re.DOTALL)
            if file_doc_match:
                module_info["description"] = file_doc_match.group(1).strip()
            
            # Extract imports
            import_matches = re.finditer(r'^\s*(?:import|require)\s+([^\s;]+)', content, re.MULTILINE)
            for match in import_matches:
                module_info["imports"].append(match.group(1))
            
            # Extract classes
            class_matches = re.finditer(r'^\s*class\s+(\w+)(?:\s+extends\s+([^\s{]+))?', content, re.MULTILINE)
            for match in class_matches:
                class_name = match.group(1)
                class_extends = match.group(2) or ""
                
                # Find class body
                class_start = match.start()
                class_body_start = content.find("{", class_start)
                class_body_end = -1
                
                if class_body_start != -1:
                    # Find matching closing brace
                    brace_count = 1
                    pos = class_body_start + 1
                    while pos < len(content) and brace_count > 0:
                        if content[pos] == "{":
                            brace_count += 1
                        elif content[pos] == "}":
                            brace_count -= 1
                        pos += 1
                    
                    if brace_count == 0:
                        class_body_end = pos
                
                if class_body_end != -1:
                    class_body = content[class_body_start:class_body_end]
                    
                    # Extract class JSDoc
                    class_docstring = ""
                    jsdoc_match = re.search(r'/\*\*(.*?)\*/', content[:class_start], re.DOTALL)
                    if jsdoc_match:
                        class_docstring = jsdoc_match.group(1).strip()
                    
                    # Extract methods
                    methods = []
                    method_matches = re.finditer(r'(?:async\s+)?(\w+)\s*\(([^)]*)\)', class_body)
                    for method_match in method_matches:
                        method_name = method_match.group(1)
                        method_params = method_match.group(2)
                        
                        # Skip constructor
                        if method_name == "constructor":
                            continue
                        
                        # Find method JSDoc
                        method_start = class_body_start + method_match.start()
                        method_docstring = ""
                        jsdoc_match = re.search(r'/\*\*(.*?)\*/', content[:method_start], re.DOTALL)
                        if jsdoc_match:
                            method_docstring = jsdoc_match.group(1).strip()
                        
                        methods.append({
                            "name": method_name,
                            "parameters": method_params,
                            "description": method_docstring
                        })
                    
                    module_info["classes"].append({
                        "name": class_name,
                        "bases": class_extends,
                        "description": class_docstring,
                        "methods": methods
                    })
            
            # Extract functions
            function_matches = re.finditer(r'(?:function|const)\s+(\w+)\s*(?:=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)|\([^)]*\))', content, re.MULTILINE)
            for match in function_matches:
                function_name = match.group(1)
                
                # Find function JSDoc
                function_start = match.start()
                function_docstring = ""
                jsdoc_match = re.search(r'/\*\*(.*?)\*/', content[:function_start], re.DOTALL)
                if jsdoc_match:
                    function_docstring = jsdoc_match.group(1).strip()
                
                # Extract parameters
                params_match = re.search(r'\(([^)]*)\)', content[function_start:function_start + 100])
                function_params = params_match.group(1) if params_match else ""
                
                module_info["functions"].append({
                    "name": function_name,
                    "parameters": function_params,
                    "description": function_docstring
                })
        
        return module_info

    async def _generate_document_from_template(
        self, 
        template: Dict[str, Any],
        data: Dict[str, Any],
        format: str
    ) -> str:
        """Generate document content from a template."""
        content = []
        
        for section in template["sections"]:
            section_name = section["name"]
            section_heading = section["heading"]
            
            # Skip excluded sections
            if section_name in data.get("exclude_sections", []):
                continue
            
            # Skip non-required sections if not in include_sections
            if not section["required"] and section_name not in data.get("include_sections", []):
                continue
            
            # Format heading with data
            formatted_heading = section_heading.format(**data)
            content.append(formatted_heading)
            
            # Add section content if available
            if section_name in data:
                section_content = data[section_name]
                if isinstance(section_content, list):
                    content.append("\n".join(section_content))
                else:
                    content.append(str(section_content))
            
            # Add empty line after section
            content.append("")
        
        return "\n".join(content)

    async def _generate_module_documentation(
        self, 
        module_name: str,
        module_info: Dict[str, Any],
        format: str
    ) -> str:
        """Generate documentation for a code module."""
        content = []
        
        # Add module header
        content.append(f"# Module: {module_name}")
        content.append("")
        
        # Add module description
        if module_info.get("description"):
            content.append(module_info["description"])
            content.append("")
        
        # Add imports section
        if module_info.get("imports"):
            content.append("## Imports")
            content.append("")
            for imp in module_info["imports"]:
                content.append(f"- `{imp}`")
            content.append("")
        
        # Add classes section
        if module_info.get("classes"):
            content.append("## Classes")
            content.append("")
            
            for cls in module_info["classes"]:
                content.append(f"### {cls['name']}")
                
                if cls.get("bases"):
                    content.append(f"*Extends: {cls['bases']}*")
                
                if cls.get("description"):
                    content.append("")
                    content.append(cls["description"])
                
                if cls.get("methods"):
                    content.append("")
                    content.append("#### Methods")
                    content.append("")
                    
                    for method in cls["methods"]:
                        content.append(f"##### `{method['name']}({method['parameters']})`")
                        
                        if method.get("description"):
                            content.append("")
                            content.append(method["description"])
                        
                        content.append("")
                
                content.append("")
        
        # Add functions section
        if module_info.get("functions"):
            content.append("## Functions")
            content.append("")
            
            for func in module_info["functions"]:
                content.append(f"### `{func['name']}({func['parameters']})`")
                
                if func.get("description"):
                    content.append("")
                    content.append(func["description"])
                
                content.append("")
        
        return "\n".join(content)

    async def _generate_documentation_index(
        self, 
        code_info: Dict[str, Any],
        module_names: List[str],
        format: str
    ) -> str:
        """Generate an index file for code documentation."""
        content = []
        
        # Add header
        content.append("# Code Documentation")
        content.append("")
        
        # Add language info
        content.append(f"Language: {code_info.get('language', 'Unknown')}")
        content.append("")
        
        # Add modules section
        content.append("## Modules")
        content.append("")
        
        for module_name in sorted(module_names):
            safe_name = module_name.replace("/", "_").replace("\\", "_")
            content.append(f"- [{module_name}]({safe_name}.{self._get_extension(format)})")
        
        return "\n".join(content)

    async def _generate_docs_index(
        self, 
        results: List[tuple],
        project_info: Optional[Dict[str, Any]],
        format: str
    ) -> str:
        """Generate an index file for project documentation."""
        content = []
        
        # Get project name
        project_name = project_info.get("project_name", "Project") if project_info else "Project"
        
        # Add header
        content.append(f"# {project_name} Documentation")
        content.append("")
        
        # Add description
        if project_info and project_info.get("description"):
            content.append(project_info["description"])
            content.append("")
        
        # Add documentation sections
        content.append("## Documentation Sections")
        content.append("")
        
        for doc_type, result in results:
            if not hasattr(result, "error") or not result.error:
                # Extract filename from output path
                output_path = ""
                if hasattr(result, "output"):
                    output_match = re.search(r'at\s+(.+)$', result.output)
                    if output_match:
                        output_path = output_match.group(1)
                
                if output_path:
                    # Get relative path
                    rel_path = os.path.basename(output_path)
                    content.append(f"- [{doc_type}]({rel_path})")
        
        return "\n".join(content)

    def _detect_language(self, project_path: str) -> str:
        """Detect the primary programming language of a project."""
        # Count files by extension
        extension_counts = {}
        
        for root, _, files in os.walk(project_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext:
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        # Map extensions to languages
        extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby"
        }
        
        # Count by language
        language_counts = {}
        for ext, count in extension_counts.items():
            if ext in extension_to_language:
                lang = extension_to_language[ext]
                language_counts[lang] = language_counts.get(lang, 0) + count
        
        # Return the most common language
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]
        
        # Default to python if no language detected
        return "python"

    def _get_extension(self, format: str) -> str:
        """Get file extension for the specified format."""
        format_extensions = {
            "markdown": "md",
            "html": "html",
            "rst": "rst",
            "text": "txt"
        }
        
        return format_extensions.get(format, "md")