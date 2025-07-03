"""
Data Processor Tool
Advanced data processing, transformation, and analysis capabilities
"""

import json
import os
import re
import csv
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import Field

from app.tool.base import BaseTool, ToolResult
from app.config import config


class DataProcessorTool(BaseTool):
    """
    Advanced data processing tool for transforming, analyzing, and extracting insights
    from various data formats.
    """

    name: str = "data_processor"
    description: str = """
    Process, transform, analyze, and extract insights from data in various formats.
    Supports CSV, JSON, text, and structured data processing with advanced operations.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "load_data", "transform_data", "analyze_data", "extract_insights",
                    "filter_data", "aggregate_data", "join_data", "export_data",
                    "validate_data", "generate_report", "detect_anomalies"
                ],
                "description": "The data processing action to perform"
            },
            "data_path": {
                "type": "string",
                "description": "Path to the data file or directory"
            },
            "data_format": {
                "type": "string",
                "enum": ["csv", "json", "text", "auto"],
                "description": "Format of the data to process"
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the processed data or results"
            },
            "output_format": {
                "type": "string",
                "enum": ["csv", "json", "text", "html"],
                "description": "Format for the output data"
            },
            "transformation": {
                "type": "string",
                "description": "JSON string defining the transformation operations"
            },
            "filter_criteria": {
                "type": "string",
                "description": "JSON string defining filter criteria"
            },
            "aggregation_config": {
                "type": "string",
                "description": "JSON string defining aggregation configuration"
            },
            "join_config": {
                "type": "string",
                "description": "JSON string defining join configuration"
            },
            "analysis_type": {
                "type": "string",
                "enum": ["statistical", "correlation", "distribution", "trend", "custom"],
                "description": "Type of analysis to perform"
            },
            "custom_code": {
                "type": "string",
                "description": "Custom Python code for data processing (use with caution)"
            },
            "visualization": {
                "type": "boolean",
                "description": "Whether to include visualizations in the output"
            }
        },
        "required": ["action"]
    }

    # Data storage
    loaded_datasets: Dict[str, Any] = Field(default_factory=dict)
    processing_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Workspace directory
    workspace_dir: str = Field(default_factory=lambda: str(config.workspace_root))

    async def execute(
        self,
        action: str,
        data_path: Optional[str] = None,
        data_format: str = "auto",
        output_path: Optional[str] = None,
        output_format: str = "json",
        transformation: Optional[str] = None,
        filter_criteria: Optional[str] = None,
        aggregation_config: Optional[str] = None,
        join_config: Optional[str] = None,
        analysis_type: str = "statistical",
        custom_code: Optional[str] = None,
        visualization: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute the data processing action."""
        
        try:
            # Record operation start
            operation_start = datetime.now()
            operation_record = {
                "action": action,
                "start_time": operation_start.isoformat(),
                "parameters": {
                    "data_path": data_path,
                    "data_format": data_format,
                    "output_path": output_path,
                    "output_format": output_format,
                    "analysis_type": analysis_type,
                    "visualization": visualization
                }
            }
            
            # Execute requested action
            if action == "load_data":
                result = await self._load_data(data_path, data_format)
            elif action == "transform_data":
                result = await self._transform_data(data_path, transformation, output_path, output_format)
            elif action == "analyze_data":
                result = await self._analyze_data(data_path, analysis_type, visualization, output_path)
            elif action == "extract_insights":
                result = await self._extract_insights(data_path, analysis_type, output_path)
            elif action == "filter_data":
                result = await self._filter_data(data_path, filter_criteria, output_path, output_format)
            elif action == "aggregate_data":
                result = await self._aggregate_data(data_path, aggregation_config, output_path, output_format)
            elif action == "join_data":
                result = await self._join_data(data_path, join_config, output_path, output_format)
            elif action == "export_data":
                result = await self._export_data(data_path, output_path, output_format)
            elif action == "validate_data":
                result = await self._validate_data(data_path, data_format)
            elif action == "generate_report":
                result = await self._generate_report(data_path, analysis_type, visualization, output_path)
            elif action == "detect_anomalies":
                result = await self._detect_anomalies(data_path, output_path)
            else:
                return ToolResult(error=f"Unknown data processing action: {action}")
            
            # Record operation completion
            operation_record.update({
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - operation_start).total_seconds(),
                "status": "success" if not result.error else "error",
                "error": result.error
            })
            
            self.processing_history.append(operation_record)
            
            return result
                
        except Exception as e:
            # Record operation failure
            if 'operation_record' in locals():
                operation_record.update({
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": (datetime.now() - operation_start).total_seconds(),
                    "status": "error",
                    "error": str(e)
                })
                self.processing_history.append(operation_record)
                
            return ToolResult(error=f"Data processing error: {str(e)}")

    async def _load_data(self, data_path: Optional[str], data_format: str) -> ToolResult:
        """Load data from a file into memory."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        # Resolve path
        full_path = self._resolve_path(data_path)
        
        if not os.path.exists(full_path):
            return ToolResult(error=f"Data file not found: {full_path}")
            
        # Determine format if auto
        if data_format == "auto":
            data_format = self._detect_format(full_path)
            
        try:
            # Load data based on format
            if data_format == "csv":
                data = self._load_csv(full_path)
            elif data_format == "json":
                data = self._load_json(full_path)
            elif data_format == "text":
                data = self._load_text(full_path)
            else:
                return ToolResult(error=f"Unsupported data format: {data_format}")
                
            # Generate dataset ID and store
            dataset_id = f"ds_{os.path.basename(full_path)}_{int(datetime.now().timestamp())}"
            self.loaded_datasets[dataset_id] = {
                "id": dataset_id,
                "path": full_path,
                "format": data_format,
                "loaded_at": datetime.now().isoformat(),
                "data": data
            }
            
            # Generate summary
            summary = self._generate_data_summary(data, data_format)
            
            return ToolResult(
                output=f"Data loaded successfully from {full_path}\n"
                       f"Dataset ID: {dataset_id}\n"
                       f"Format: {data_format}\n\n"
                       f"Data Summary:\n{summary}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error loading data: {str(e)}")

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the workspace directory."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_dir, path)

    def _detect_format(self, file_path: str) -> str:
        """Detect the format of a data file based on extension and content."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.csv':
            return "csv"
        elif ext == '.json':
            return "json"
        elif ext in ['.txt', '.text', '.log']:
            return "text"
            
        # Try to detect by content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(1024)
            
        if sample.strip().startswith('{') or sample.strip().startswith('['):
            return "json"
        elif ',' in sample and '\n' in sample:
            return "csv"
            
        return "text"  # Default to text

    def _load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a CSV file."""
        data = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data

    def _load_json(self, file_path: str) -> Any:
        """Load data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_text(self, file_path: str) -> str:
        """Load data from a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _generate_data_summary(self, data: Any, data_format: str) -> str:
        """Generate a summary of the loaded data."""
        if data_format == "csv" or (data_format == "json" and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict)):
            # Tabular data
            record_count = len(data)
            
            if record_count == 0:
                return "Empty dataset (0 records)"
                
            # Get columns
            columns = list(data[0].keys())
            
            # Sample data
            sample_size = min(5, record_count)
            sample = data[:sample_size]
            
            summary = [f"Records: {record_count}"]
            summary.append(f"Columns: {', '.join(columns)}")
            summary.append(f"\nSample ({sample_size} records):")
            
            for i, record in enumerate(sample):
                summary.append(f"  Record {i+1}:")
                for col, val in record.items():
                    summary.append(f"    {col}: {val}")
                    
            return "\n".join(summary)
            
        elif data_format == "json":
            # Non-tabular JSON
            if isinstance(data, dict):
                keys = list(data.keys())
                summary = ["JSON Object"]
                summary.append(f"Top-level keys: {', '.join(keys[:10])}")
                if len(keys) > 10:
                    summary.append(f"...and {len(keys) - 10} more keys")
            elif isinstance(data, list):
                summary = [f"JSON Array with {len(data)} items"]
                if len(data) > 0:
                    summary.append(f"First item type: {type(data[0]).__name__}")
            else:
                summary = [f"JSON data of type: {type(data).__name__}"]
                
            return "\n".join(summary)
            
        elif data_format == "text":
            # Text data
            lines = data.count('\n') + 1
            words = len(data.split())
            chars = len(data)
            
            summary = [f"Text data:"]
            summary.append(f"Lines: {lines}")
            summary.append(f"Words: {words}")
            summary.append(f"Characters: {chars}")
            
            # Preview
            preview_lines = data.split('\n')[:5]
            if preview_lines:
                summary.append("\nPreview:")
                for i, line in enumerate(preview_lines):
                    if line.strip():
                        summary.append(f"  {i+1}: {line[:80]}")
                        
            return "\n".join(summary)
            
        else:
            return f"Data of type {type(data).__name__}"

    async def _transform_data(
        self, 
        data_path: Optional[str], 
        transformation: Optional[str],
        output_path: Optional[str],
        output_format: str
    ) -> ToolResult:
        """Transform data according to specified operations."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        if not transformation:
            return ToolResult(error="Transformation definition is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Parse transformation
        try:
            transform_config = json.loads(transformation)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid transformation JSON")
            
        # Apply transformations
        try:
            transformed_data = self._apply_transformations(dataset["data"], transform_config)
            
            # Save result if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                result_text = await self._save_data(transformed_data, full_output_path, output_format)
                
            # Generate summary
            summary = self._generate_data_summary(transformed_data, dataset["format"])
            
            return ToolResult(
                output=f"Data transformation completed successfully\n"
                       f"Original dataset: {dataset['id']}\n"
                       f"{result_text}\n"
                       f"Transformation Summary:\n{summary}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error transforming data: {str(e)}")

    def _get_dataset(self, data_path: str) -> Optional[Dict[str, Any]]:
        """Get a dataset by ID or path."""
        # Check if it's a dataset ID
        if data_path in self.loaded_datasets:
            return self.loaded_datasets[data_path]
            
        # Check if it's a file path that matches a loaded dataset
        full_path = self._resolve_path(data_path)
        for dataset in self.loaded_datasets.values():
            if dataset["path"] == full_path:
                return dataset
                
        return None

    def _apply_transformations(self, data: Any, transform_config: Dict[str, Any]) -> Any:
        """Apply transformations to data."""
        operations = transform_config.get("operations", [])
        result = data
        
        for operation in operations:
            op_type = operation.get("type")
            
            if op_type == "select_columns":
                columns = operation.get("columns", [])
                result = self._select_columns(result, columns)
                
            elif op_type == "rename_columns":
                mapping = operation.get("mapping", {})
                result = self._rename_columns(result, mapping)
                
            elif op_type == "filter_rows":
                condition = operation.get("condition", {})
                result = self._filter_rows(result, condition)
                
            elif op_type == "sort":
                key = operation.get("key")
                reverse = operation.get("reverse", False)
                result = self._sort_data(result, key, reverse)
                
            elif op_type == "group_by":
                key = operation.get("key")
                aggregations = operation.get("aggregations", {})
                result = self._group_by(result, key, aggregations)
                
            elif op_type == "calculate":
                formula = operation.get("formula")
                output_column = operation.get("output_column")
                result = self._calculate(result, formula, output_column)
                
            elif op_type == "flatten":
                path = operation.get("path")
                result = self._flatten(result, path)
                
        return result

    def _select_columns(self, data: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
        """Select only specified columns from tabular data."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records for select_columns operation")
            
        return [{col: row.get(col) for col in columns if col in row} for row in data]

    def _rename_columns(self, data: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Rename columns in tabular data."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records for rename_columns operation")
            
        result = []
        for row in data:
            new_row = {}
            for old_key, value in row.items():
                new_key = mapping.get(old_key, old_key)
                new_row[new_key] = value
            result.append(new_row)
            
        return result

    def _filter_rows(self, data: List[Dict[str, Any]], condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter rows based on condition."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records for filter_rows operation")
            
        field = condition.get("field")
        operator = condition.get("operator", "equals")
        value = condition.get("value")
        
        if not field:
            raise ValueError("Field is required for filter condition")
            
        result = []
        for row in data:
            if field not in row:
                continue
                
            row_value = row[field]
            
            if operator == "equals" and row_value == value:
                result.append(row)
            elif operator == "not_equals" and row_value != value:
                result.append(row)
            elif operator == "greater_than" and row_value > value:
                result.append(row)
            elif operator == "less_than" and row_value < value:
                result.append(row)
            elif operator == "contains" and value in row_value:
                result.append(row)
            elif operator == "starts_with" and str(row_value).startswith(str(value)):
                result.append(row)
            elif operator == "ends_with" and str(row_value).endswith(str(value)):
                result.append(row)
                
        return result

    def _sort_data(self, data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
        """Sort data by key."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records for sort operation")
            
        if not key:
            raise ValueError("Sort key is required")
            
        return sorted(data, key=lambda x: x.get(key, None), reverse=reverse)

    def _group_by(self, data: List[Dict[str, Any]], key: str, aggregations: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
        """Group data by key and apply aggregations."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records for group_by operation")
            
        if not key:
            raise ValueError("Group by key is required")
            
        # Group data
        groups = {}
        for row in data:
            group_key = row.get(key)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(row)
            
        # Apply aggregations
        result = []
        for group_key, group_data in groups.items():
            group_result = {key: group_key}
            
            for output_field, agg_config in aggregations.items():
                agg_type = agg_config.get("type")
                field = agg_config.get("field")
                
                if not field or not agg_type:
                    continue
                    
                values = [row.get(field) for row in group_data if field in row]
                
                # Skip if no values
                if not values:
                    group_result[output_field] = None
                    continue
                    
                # Apply aggregation
                if agg_type == "sum":
                    group_result[output_field] = sum(values)
                elif agg_type == "avg":
                    group_result[output_field] = sum(values) / len(values)
                elif agg_type == "min":
                    group_result[output_field] = min(values)
                elif agg_type == "max":
                    group_result[output_field] = max(values)
                elif agg_type == "count":
                    group_result[output_field] = len(values)
                elif agg_type == "list":
                    group_result[output_field] = values
                    
            result.append(group_result)
            
        return result

    def _calculate(self, data: List[Dict[str, Any]], formula: str, output_column: str) -> List[Dict[str, Any]]:
        """Calculate a new column based on a formula."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list of records for calculate operation")
            
        if not formula or not output_column:
            raise ValueError("Formula and output_column are required")
            
        # Simple formula evaluation (for demonstration)
        # In a real implementation, this would use a safer evaluation method
        result = []
        for row in data:
            new_row = row.copy()
            
            # Replace field references with values
            calc_formula = formula
            for field, value in row.items():
                calc_formula = calc_formula.replace(f"{{{field}}}", str(value))
                
            try:
                # Evaluate formula (simplified for demonstration)
                # WARNING: eval is unsafe for production use
                new_row[output_column] = eval(calc_formula)
            except Exception as e:
                new_row[output_column] = f"Error: {str(e)}"
                
            result.append(new_row)
            
        return result

    def _flatten(self, data: Any, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Flatten nested data structures."""
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Already flat list of dicts
            if not path:
                return data
                
            # Extract nested data
            result = []
            for item in data:
                nested = self._get_nested_value(item, path)
                if isinstance(nested, list):
                    for nested_item in nested:
                        if isinstance(nested_item, dict):
                            # Combine parent and nested
                            flat_item = {k: v for k, v in item.items() if k != path.split('.')[0]}
                            flat_item.update(nested_item)
                            result.append(flat_item)
                        else:
                            # Can't flatten non-dict items
                            result.append({**item, "value": nested_item})
                            
            return result
            
        elif isinstance(data, dict):
            # Single dict, convert to list
            return [data]
            
        else:
            raise ValueError(f"Cannot flatten data of type {type(data)}")

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a nested value from a dictionary using dot notation."""
        if not path:
            return data
            
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current

    async def _save_data(self, data: Any, output_path: str, output_format: str) -> str:
        """Save data to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_format == "csv":
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                raise ValueError("Data must be a list of dictionaries for CSV export")
                
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if not data:
                    writer = csv.writer(f)
                    writer.writerow([])
                else:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    
        elif output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        elif output_format == "text":
            with open(output_path, 'w', encoding='utf-8') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))
                    
        elif output_format == "html":
            html_content = self._generate_html_report(data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        return f"Data saved to {output_path} in {output_format} format"

    def _generate_html_report(self, data: Any) -> str:
        """Generate an HTML report from data."""
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<meta charset='UTF-8'>")
        html.append("<title>Data Processing Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append("<h1>Data Processing Report</h1>")
        html.append(f"<p>Generated: {datetime.now().isoformat()}</p>")
        
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Tabular data
            html.append("<h2>Data Table</h2>")
            html.append("<table>")
            
            # Headers
            if data:
                html.append("<tr>")
                for key in data[0].keys():
                    html.append(f"<th>{key}</th>")
                html.append("</tr>")
                
                # Rows
                for item in data:
                    html.append("<tr>")
                    for value in item.values():
                        html.append(f"<td>{value}</td>")
                    html.append("</tr>")
                    
            html.append("</table>")
            
        elif isinstance(data, dict):
            # Dictionary data
            html.append("<h2>Data Object</h2>")
            html.append("<table>")
            html.append("<tr><th>Key</th><th>Value</th></tr>")
            
            for key, value in data.items():
                html.append("<tr>")
                html.append(f"<td>{key}</td>")
                html.append(f"<td>{value}</td>")
                html.append("</tr>")
                
            html.append("</table>")
            
        else:
            # Other data
            html.append("<h2>Data</h2>")
            html.append(f"<pre>{data}</pre>")
            
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)

    async def _analyze_data(
        self, 
        data_path: Optional[str], 
        analysis_type: str,
        visualization: bool,
        output_path: Optional[str]
    ) -> ToolResult:
        """Analyze data and generate statistics."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Perform analysis
        try:
            analysis_result = self._perform_analysis(dataset["data"], analysis_type)
            
            # Save result if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                output_format = "html" if visualization else "json"
                result_text = await self._save_data(analysis_result, full_output_path, output_format)
                
            # Format output
            output = [f"Data Analysis ({analysis_type}) Completed"]
            output.append(f"Dataset: {dataset['id']}")
            output.append(result_text)
            
            # Add summary of results
            output.append("\nAnalysis Results Summary:")
            if isinstance(analysis_result, dict):
                for key, value in analysis_result.items():
                    if isinstance(value, dict):
                        output.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            output.append(f"    {subkey}: {subvalue}")
                    else:
                        output.append(f"  {key}: {value}")
            else:
                output.append(str(analysis_result))
                
            return ToolResult(output="\n".join(output))
            
        except Exception as e:
            return ToolResult(error=f"Error analyzing data: {str(e)}")

    def _perform_analysis(self, data: Any, analysis_type: str) -> Dict[str, Any]:
        """Perform analysis on data."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list for analysis")
            
        if not data:
            return {"error": "Empty dataset"}
            
        # For tabular data
        if all(isinstance(item, dict) for item in data):
            if analysis_type == "statistical":
                return self._statistical_analysis(data)
            elif analysis_type == "correlation":
                return self._correlation_analysis(data)
            elif analysis_type == "distribution":
                return self._distribution_analysis(data)
            elif analysis_type == "trend":
                return self._trend_analysis(data)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        # For non-tabular data
        return {"error": "Data format not supported for analysis"}

    def _statistical_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on tabular data."""
        result = {"record_count": len(data)}
        
        # Get numeric columns
        numeric_columns = self._identify_numeric_columns(data)
        
        # Calculate statistics for each numeric column
        column_stats = {}
        for column in numeric_columns:
            values = [float(row[column]) for row in data if column in row and row[column] is not None]
            
            if not values:
                continue
                
            stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "sum": sum(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }
            
            # Calculate standard deviation
            mean = stats["mean"]
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            stats["std_dev"] = variance ** 0.5
            
            column_stats[column] = stats
            
        result["column_statistics"] = column_stats
        
        # Get categorical columns
        categorical_columns = self._identify_categorical_columns(data)
        
        # Calculate frequency distributions for categorical columns
        category_stats = {}
        for column in categorical_columns:
            values = [row[column] for row in data if column in row and row[column] is not None]
            
            if not values:
                continue
                
            # Count frequencies
            frequencies = {}
            for value in values:
                frequencies[value] = frequencies.get(value, 0) + 1
                
            category_stats[column] = {
                "count": len(values),
                "unique_values": len(frequencies),
                "frequencies": frequencies
            }
            
        result["categorical_statistics"] = category_stats
        
        return result

    def _identify_numeric_columns(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify numeric columns in tabular data."""
        if not data:
            return []
            
        # Get all columns
        columns = list(data[0].keys())
        
        # Check each column
        numeric_columns = []
        for column in columns:
            # Check first 10 non-null values
            values = [row[column] for row in data[:min(100, len(data))] 
                     if column in row and row[column] is not None][:10]
            
            if not values:
                continue
                
            # Check if all values can be converted to float
            try:
                all(float(value) for value in values)
                numeric_columns.append(column)
            except (ValueError, TypeError):
                continue
                
        return numeric_columns

    def _identify_categorical_columns(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify categorical columns in tabular data."""
        if not data:
            return []
            
        # Get all columns
        columns = list(data[0].keys())
        
        # Get numeric columns
        numeric_columns = self._identify_numeric_columns(data)
        
        # Identify categorical columns (non-numeric with limited unique values)
        categorical_columns = []
        for column in columns:
            if column in numeric_columns:
                continue
                
            # Get unique values
            values = set(row[column] for row in data[:min(1000, len(data))] 
                        if column in row and row[column] is not None)
            
            # If number of unique values is reasonable, consider it categorical
            if len(values) <= min(20, len(data) // 5):
                categorical_columns.append(column)
                
        return categorical_columns

    def _correlation_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform correlation analysis on numeric columns."""
        # Identify numeric columns
        numeric_columns = self._identify_numeric_columns(data)
        
        if len(numeric_columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
            
        # Calculate correlations
        correlations = {}
        for i, col1 in enumerate(numeric_columns):
            correlations[col1] = {}
            
            for col2 in numeric_columns[i:]:
                # Get paired values
                pairs = [(float(row[col1]), float(row[col2])) 
                         for row in data 
                         if col1 in row and col2 in row 
                         and row[col1] is not None and row[col2] is not None]
                
                if not pairs:
                    correlations[col1][col2] = None
                    continue
                    
                # Calculate Pearson correlation
                x_values = [p[0] for p in pairs]
                y_values = [p[1] for p in pairs]
                
                n = len(pairs)
                sum_x = sum(x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in pairs)
                sum_x2 = sum(x * x for x in x_values)
                sum_y2 = sum(y * y for y in y_values)
                
                numerator = n * sum_xy - sum_x * sum_y
                denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
                
                if denominator == 0:
                    correlation = 0
                else:
                    correlation = numerator / denominator
                    
                correlations[col1][col2] = correlation
                
                # Add symmetric value
                if col1 != col2:
                    if col2 not in correlations:
                        correlations[col2] = {}
                    correlations[col2][col1] = correlation
                    
        return {"correlations": correlations}

    def _distribution_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of values in each column."""
        result = {}
        
        # Analyze numeric columns
        numeric_columns = self._identify_numeric_columns(data)
        numeric_distributions = {}
        
        for column in numeric_columns:
            values = [float(row[column]) for row in data if column in row and row[column] is not None]
            
            if not values:
                continue
                
            # Calculate basic statistics
            min_val = min(values)
            max_val = max(values)
            
            # Create bins
            bin_count = min(10, len(set(values)))
            bin_size = (max_val - min_val) / bin_count if max_val > min_val else 1
            
            bins = {}
            for i in range(bin_count):
                bin_min = min_val + i * bin_size
                bin_max = min_val + (i + 1) * bin_size
                bin_name = f"{bin_min:.2f}-{bin_max:.2f}"
                bins[bin_name] = 0
                
            # Count values in each bin
            for value in values:
                bin_index = min(bin_count - 1, int((value - min_val) / bin_size))
                bin_min = min_val + bin_index * bin_size
                bin_max = min_val + (bin_index + 1) * bin_size
                bin_name = f"{bin_min:.2f}-{bin_max:.2f}"
                bins[bin_name] += 1
                
            numeric_distributions[column] = {
                "min": min_val,
                "max": max_val,
                "distribution": bins
            }
            
        result["numeric_distributions"] = numeric_distributions
        
        # Analyze categorical columns
        categorical_columns = self._identify_categorical_columns(data)
        categorical_distributions = {}
        
        for column in categorical_columns:
            values = [row[column] for row in data if column in row and row[column] is not None]
            
            if not values:
                continue
                
            # Count frequencies
            frequencies = {}
            for value in values:
                frequencies[value] = frequencies.get(value, 0) + 1
                
            # Sort by frequency
            sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
            
            categorical_distributions[column] = {
                "unique_values": len(frequencies),
                "most_common": list(sorted_frequencies.keys())[:5],
                "distribution": sorted_frequencies
            }
            
        result["categorical_distributions"] = categorical_distributions
        
        return result

    def _trend_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        # Identify potential date/time columns
        date_columns = []
        for column in data[0].keys():
            # Check if column name suggests date
            if any(date_term in column.lower() for date_term in ["date", "time", "day", "month", "year"]):
                date_columns.append(column)
                
        if not date_columns:
            return {"error": "No date/time columns identified for trend analysis"}
            
        # Use first identified date column
        date_column = date_columns[0]
        
        # Identify numeric columns for trend analysis
        numeric_columns = self._identify_numeric_columns(data)
        
        if not numeric_columns:
            return {"error": "No numeric columns found for trend analysis"}
            
        # Sort data by date column (assuming string representation)
        sorted_data = sorted(data, key=lambda x: x.get(date_column, ""))
        
        # Analyze trends for each numeric column
        trends = {}
        for column in numeric_columns:
            # Extract date-value pairs
            series = [(row.get(date_column), float(row.get(column, 0))) 
                     for row in sorted_data 
                     if column in row and row[column] is not None]
            
            if len(series) < 2:
                continue
                
            # Calculate simple trend indicators
            values = [pair[1] for pair in series]
            first_value = values[0]
            last_value = values[-1]
            min_value = min(values)
            max_value = max(values)
            
            # Calculate change
            absolute_change = last_value - first_value
            percent_change = (absolute_change / first_value) * 100 if first_value != 0 else float('inf')
            
            # Determine trend direction
            if absolute_change > 0:
                direction = "increasing"
            elif absolute_change < 0:
                direction = "decreasing"
            else:
                direction = "stable"
                
            # Calculate simple moving average (last 3 points)
            moving_avg = sum(values[-3:]) / min(3, len(values))
            
            trends[column] = {
                "first_value": first_value,
                "last_value": last_value,
                "min_value": min_value,
                "max_value": max_value,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "direction": direction,
                "moving_average": moving_avg
            }
            
        return {
            "date_column": date_column,
            "data_points": len(sorted_data),
            "trends": trends
        }

    async def _extract_insights(
        self, 
        data_path: Optional[str], 
        analysis_type: str,
        output_path: Optional[str]
    ) -> ToolResult:
        """Extract insights from data analysis."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        # First perform analysis
        analysis_result = await self._analyze_data(data_path, analysis_type, False, None)
        if analysis_result.error:
            return analysis_result
            
        # Extract insights from analysis results
        try:
            # Parse analysis results
            analysis_output = analysis_result.output
            dataset_id = analysis_output.split("Dataset: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
            if not dataset:
                return ToolResult(error="Dataset not found")
                
            # Generate insights based on analysis type
            insights = self._generate_insights(dataset["data"], analysis_type)
            
            # Save insights if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                result_text = await self._save_data(insights, full_output_path, "html")
                
            # Format output
            output = ["Data Insights Extracted"]
            output.append(f"Dataset: {dataset['id']}")
            output.append(f"Analysis Type: {analysis_type}")
            output.append(result_text)
            
            # Add key insights
            output.append("\nKey Insights:")
            for category, category_insights in insights.items():
                output.append(f"\n{category}:")
                for insight in category_insights:
                    output.append(f"  • {insight}")
                    
            return ToolResult(output="\n".join(output))
            
        except Exception as e:
            return ToolResult(error=f"Error extracting insights: {str(e)}")

    def _generate_insights(self, data: Any, analysis_type: str) -> Dict[str, List[str]]:
        """Generate insights from analyzed data."""
        insights = {
            "key_findings": [],
            "patterns": [],
            "anomalies": [],
            "recommendations": []
        }
        
        if not isinstance(data, list) or not data:
            insights["key_findings"].append("Insufficient data for meaningful insights")
            return insights
            
        # For tabular data
        if all(isinstance(item, dict) for item in data):
            # Perform analysis if needed
            if analysis_type == "statistical":
                stats = self._statistical_analysis(data)
                
                # Extract insights from statistics
                column_stats = stats.get("column_statistics", {})
                for column, col_stats in column_stats.items():
                    # Check for extreme values
                    if col_stats.get("max") > col_stats.get("mean") * 2:
                        insights["anomalies"].append(
                            f"Column '{column}' has extreme high values (max: {col_stats['max']:.2f}, mean: {col_stats['mean']:.2f})"
                        )
                        
                    # Check for high variance
                    if col_stats.get("std_dev", 0) > col_stats.get("mean", 0):
                        insights["patterns"].append(
                            f"Column '{column}' shows high variability (std dev: {col_stats['std_dev']:.2f}, mean: {col_stats['mean']:.2f})"
                        )
                        
                # Check categorical distributions
                cat_stats = stats.get("categorical_statistics", {})
                for column, col_stats in cat_stats.items():
                    frequencies = col_stats.get("frequencies", {})
                    if frequencies:
                        # Find dominant category
                        top_category, top_count = max(frequencies.items(), key=lambda x: x[1])
                        dominance = top_count / col_stats.get("count", 1)
                        
                        if dominance > 0.7:
                            insights["key_findings"].append(
                                f"Column '{column}' is dominated by value '{top_category}' ({dominance:.1%} of records)"
                            )
                            
            elif analysis_type == "correlation":
                corr_analysis = self._correlation_analysis(data)
                correlations = corr_analysis.get("correlations", {})
                
                # Find strong correlations
                strong_correlations = []
                for col1, col_corrs in correlations.items():
                    for col2, corr_value in col_corrs.items():
                        if col1 != col2 and corr_value is not None:
                            if abs(corr_value) > 0.7:
                                strong_correlations.append((col1, col2, corr_value))
                                
                # Add insights for strong correlations
                for col1, col2, corr_value in strong_correlations:
                    direction = "positive" if corr_value > 0 else "negative"
                    insights["patterns"].append(
                        f"Strong {direction} correlation ({corr_value:.2f}) between '{col1}' and '{col2}'"
                    )
                    
                # Add recommendations based on correlations
                if strong_correlations:
                    insights["recommendations"].append(
                        f"Consider further investigation of relationships between correlated variables"
                    )
                    
            elif analysis_type == "trend":
                trend_analysis = self._trend_analysis(data)
                trends = trend_analysis.get("trends", {})
                
                # Add insights for significant trends
                for column, trend in trends.items():
                    percent_change = trend.get("percent_change")
                    direction = trend.get("direction")
                    
                    if abs(percent_change) > 20:
                        insights["key_findings"].append(
                            f"'{column}' shows significant {direction} trend ({percent_change:.1f}% change)"
                        )
                        
                    # Check for volatility
                    min_val = trend.get("min_value", 0)
                    max_val = trend.get("max_value", 0)
                    first_val = trend.get("first_value", 0)
                    
                    if first_val and (max_val - min_val) / first_val > 0.5:
                        insights["patterns"].append(
                            f"'{column}' shows high volatility (range: {min_val:.2f} to {max_val:.2f})"
                        )
                        
        # Add general insights
        insights["key_findings"].append(f"Dataset contains {len(data)} records")
        
        # Add recommendations if none yet
        if not insights["recommendations"]:
            if analysis_type == "statistical":
                insights["recommendations"].append("Consider correlation analysis to identify relationships between variables")
            elif analysis_type == "correlation":
                insights["recommendations"].append("Consider trend analysis to identify changes over time")
            elif analysis_type == "trend":
                insights["recommendations"].append("Consider forecasting future values based on identified trends")
                
        return insights

    async def _filter_data(
        self, 
        data_path: Optional[str], 
        filter_criteria: Optional[str],
        output_path: Optional[str],
        output_format: str
    ) -> ToolResult:
        """Filter data based on criteria."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        if not filter_criteria:
            return ToolResult(error="Filter criteria is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Parse filter criteria
        try:
            criteria = json.loads(filter_criteria)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid filter criteria JSON")
            
        # Apply filter
        try:
            filtered_data = self._apply_filter(dataset["data"], criteria)
            
            # Save result if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                result_text = await self._save_data(filtered_data, full_output_path, output_format)
                
            # Generate summary
            original_count = len(dataset["data"]) if isinstance(dataset["data"], list) else 1
            filtered_count = len(filtered_data) if isinstance(filtered_data, list) else 1
            
            return ToolResult(
                output=f"Data filtered successfully\n"
                       f"Original records: {original_count}\n"
                       f"Filtered records: {filtered_count}\n"
                       f"Filter rate: {(original_count - filtered_count) / original_count:.1%}\n"
                       f"{result_text}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error filtering data: {str(e)}")

    def _apply_filter(self, data: Any, criteria: Dict[str, Any]) -> Any:
        """Apply filter criteria to data."""
        if not isinstance(data, list):
            raise ValueError("Data must be a list for filtering")
            
        # Get filter conditions
        conditions = criteria.get("conditions", [])
        operator = criteria.get("operator", "and").lower()
        
        if not conditions:
            return data
            
        # Apply filters
        filtered_data = []
        for item in data:
            matches = []
            
            for condition in conditions:
                field = condition.get("field")
                op = condition.get("operator", "equals")
                value = condition.get("value")
                
                if field not in item:
                    matches.append(False)
                    continue
                    
                item_value = item[field]
                
                # Apply operator
                if op == "equals":
                    matches.append(item_value == value)
                elif op == "not_equals":
                    matches.append(item_value != value)
                elif op == "greater_than":
                    matches.append(item_value > value)
                elif op == "less_than":
                    matches.append(item_value < value)
                elif op == "contains":
                    matches.append(value in item_value)
                elif op == "starts_with":
                    matches.append(str(item_value).startswith(str(value)))
                elif op == "ends_with":
                    matches.append(str(item_value).endswith(str(value)))
                elif op == "in":
                    matches.append(item_value in value)
                elif op == "not_in":
                    matches.append(item_value not in value)
                else:
                    matches.append(False)
                    
            # Combine conditions
            if operator == "and" and all(matches):
                filtered_data.append(item)
            elif operator == "or" and any(matches):
                filtered_data.append(item)
                
        return filtered_data

    async def _aggregate_data(
        self, 
        data_path: Optional[str], 
        aggregation_config: Optional[str],
        output_path: Optional[str],
        output_format: str
    ) -> ToolResult:
        """Aggregate data based on configuration."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        if not aggregation_config:
            return ToolResult(error="Aggregation configuration is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Parse aggregation configuration
        try:
            config = json.loads(aggregation_config)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid aggregation configuration JSON")
            
        # Apply aggregation
        try:
            aggregated_data = self._apply_aggregation(dataset["data"], config)
            
            # Save result if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                result_text = await self._save_data(aggregated_data, full_output_path, output_format)
                
            return ToolResult(
                output=f"Data aggregation completed successfully\n"
                       f"Original dataset: {dataset['id']}\n"
                       f"Aggregated records: {len(aggregated_data)}\n"
                       f"{result_text}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error aggregating data: {str(e)}")

    def _apply_aggregation(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply aggregation to data."""
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Data must be a list of dictionaries for aggregation")
            
        # Get group by fields
        group_by = config.get("group_by", [])
        if not group_by:
            raise ValueError("group_by is required for aggregation")
            
        # Get aggregations
        aggregations = config.get("aggregations", {})
        if not aggregations:
            raise ValueError("aggregations is required")
            
        # Group data
        groups = {}
        for item in data:
            # Create group key
            key_parts = []
            for field in group_by:
                key_parts.append(str(item.get(field, "")))
                
            group_key = "|".join(key_parts)
            
            if group_key not in groups:
                groups[group_key] = {
                    "items": [],
                    "group_values": {field: item.get(field) for field in group_by}
                }
                
            groups[group_key]["items"].append(item)
            
        # Apply aggregations
        result = []
        for group_key, group_data in groups.items():
            group_result = group_data["group_values"].copy()
            
            for output_field, agg_config in aggregations.items():
                field = agg_config.get("field")
                agg_type = agg_config.get("type")
                
                if not field or not agg_type:
                    continue
                    
                # Get values
                values = [
                    float(item[field]) 
                    for item in group_data["items"] 
                    if field in item and item[field] is not None
                ]
                
                if not values:
                    group_result[output_field] = None
                    continue
                    
                # Apply aggregation
                if agg_type == "sum":
                    group_result[output_field] = sum(values)
                elif agg_type == "avg":
                    group_result[output_field] = sum(values) / len(values)
                elif agg_type == "min":
                    group_result[output_field] = min(values)
                elif agg_type == "max":
                    group_result[output_field] = max(values)
                elif agg_type == "count":
                    group_result[output_field] = len(values)
                elif agg_type == "count_distinct":
                    group_result[output_field] = len(set(values))
                    
            result.append(group_result)
            
        return result

    async def _join_data(
        self, 
        data_path: Optional[str], 
        join_config: Optional[str],
        output_path: Optional[str],
        output_format: str
    ) -> ToolResult:
        """Join two datasets based on configuration."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        if not join_config:
            return ToolResult(error="Join configuration is required")
            
        # Parse join configuration
        try:
            config = json.loads(join_config)
        except json.JSONDecodeError:
            return ToolResult(error="Invalid join configuration JSON")
            
        # Get required parameters
        right_data_path = config.get("right_data_path")
        if not right_data_path:
            return ToolResult(error="right_data_path is required in join configuration")
            
        join_type = config.get("join_type", "inner")
        left_key = config.get("left_key")
        right_key = config.get("right_key")
        
        if not left_key or not right_key:
            return ToolResult(error="left_key and right_key are required in join configuration")
            
        # Load left dataset
        left_dataset = self._get_dataset(data_path)
        if not left_dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            left_dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            left_dataset = self.loaded_datasets.get(left_dataset_id)
            
        # Load right dataset
        right_dataset = self._get_dataset(right_data_path)
        if not right_dataset:
            load_result = await self._load_data(right_data_path, "auto")
            if load_result.error:
                return load_result
                
            right_dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            right_dataset = self.loaded_datasets.get(right_dataset_id)
            
        # Perform join
        try:
            joined_data = self._perform_join(
                left_dataset["data"], 
                right_dataset["data"], 
                left_key, 
                right_key, 
                join_type
            )
            
            # Save result if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                result_text = await self._save_data(joined_data, full_output_path, output_format)
                
            return ToolResult(
                output=f"Data join completed successfully\n"
                       f"Left dataset: {left_dataset['id']} ({len(left_dataset['data'])} records)\n"
                       f"Right dataset: {right_dataset['id']} ({len(right_dataset['data'])} records)\n"
                       f"Join type: {join_type}\n"
                       f"Joined records: {len(joined_data)}\n"
                       f"{result_text}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error joining data: {str(e)}")

    def _perform_join(
        self, 
        left_data: List[Dict[str, Any]], 
        right_data: List[Dict[str, Any]],
        left_key: str,
        right_key: str,
        join_type: str
    ) -> List[Dict[str, Any]]:
        """Perform a join operation between two datasets."""
        if not isinstance(left_data, list) or not all(isinstance(item, dict) for item in left_data):
            raise ValueError("Left data must be a list of dictionaries")
            
        if not isinstance(right_data, list) or not all(isinstance(item, dict) for item in right_data):
            raise ValueError("Right data must be a list of dictionaries")
            
        # Create lookup for right data
        right_lookup = {}
        for item in right_data:
            if right_key in item:
                key_value = item[right_key]
                if key_value not in right_lookup:
                    right_lookup[key_value] = []
                right_lookup[key_value].append(item)
                
        # Perform join
        result = []
        
        # Process left data
        for left_item in left_data:
            if left_key not in left_item:
                if join_type == "left" or join_type == "full":
                    # Include left items without join key for left and full joins
                    result.append(left_item)
                continue
                
            left_key_value = left_item[left_key]
            matching_right_items = right_lookup.get(left_key_value, [])
            
            if matching_right_items:
                # Join with matching right items
                for right_item in matching_right_items:
                    # Create joined item
                    joined_item = left_item.copy()
                    
                    # Add right item fields, prefixing any duplicate keys
                    for key, value in right_item.items():
                        if key in joined_item and key != right_key:
                            joined_item[f"right_{key}"] = value
                        else:
                            joined_item[key] = value
                            
                    result.append(joined_item)
            elif join_type == "left" or join_type == "full":
                # Include left items without matches for left and full joins
                result.append(left_item)
                
        # For full and right joins, include right items without matches
        if join_type == "full" or join_type == "right":
            # Find right items without matches
            matched_right_keys = set()
            for left_item in left_data:
                if left_key in left_item:
                    matched_right_keys.add(left_item[left_key])
                    
            # Add unmatched right items
            for right_item in right_data:
                if right_key in right_item and right_item[right_key] not in matched_right_keys:
                    result.append(right_item)
                    
        return result

    async def _export_data(
        self, 
        data_path: Optional[str],
        output_path: Optional[str],
        output_format: str
    ) -> ToolResult:
        """Export data to a file in the specified format."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        if not output_path:
            return ToolResult(error="Output path is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Export data
        try:
            full_output_path = self._resolve_path(output_path)
            result_text = await self._save_data(dataset["data"], full_output_path, output_format)
            
            return ToolResult(
                output=f"Data exported successfully\n"
                       f"Source dataset: {dataset['id']}\n"
                       f"{result_text}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error exporting data: {str(e)}")

    async def _validate_data(
        self, 
        data_path: Optional[str],
        data_format: str
    ) -> ToolResult:
        """Validate data against schema or rules."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, data_format)
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Perform validation
        try:
            validation_results = self._validate_dataset(dataset["data"])
            
            return ToolResult(
                output=f"Data validation completed\n"
                       f"Dataset: {dataset['id']}\n"
                       f"Valid: {validation_results['valid']}\n"
                       f"Issues found: {len(validation_results['issues'])}\n\n"
                       f"Validation Summary:\n{self._format_validation_results(validation_results)}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error validating data: {str(e)}")

    def _validate_dataset(self, data: Any) -> Dict[str, Any]:
        """Validate dataset for common issues."""
        validation_results = {
            "valid": True,
            "issues": []
        }
        
        if not isinstance(data, list):
            validation_results["valid"] = False
            validation_results["issues"].append({
                "type": "format",
                "message": f"Data is not a list (type: {type(data).__name__})"
            })
            return validation_results
            
        if not data:
            validation_results["issues"].append({
                "type": "warning",
                "message": "Dataset is empty"
            })
            return validation_results
            
        # For tabular data
        if all(isinstance(item, dict) for item in data):
            # Check for consistent schema
            first_item_keys = set(data[0].keys())
            for i, item in enumerate(data[1:], 1):
                item_keys = set(item.keys())
                if item_keys != first_item_keys:
                    missing = first_item_keys - item_keys
                    extra = item_keys - first_item_keys
                    
                    validation_results["valid"] = False
                    validation_results["issues"].append({
                        "type": "schema",
                        "message": f"Inconsistent schema at record {i}",
                        "details": {
                            "missing_fields": list(missing),
                            "extra_fields": list(extra)
                        }
                    })
                    
            # Check for missing values in key fields
            for field in first_item_keys:
                missing_count = sum(1 for item in data if field not in item or item[field] is None)
                if missing_count > 0:
                    validation_results["issues"].append({
                        "type": "missing_values",
                        "message": f"Field '{field}' has {missing_count} missing values ({missing_count/len(data):.1%})"
                    })
                    
                    if missing_count / len(data) > 0.5:
                        validation_results["valid"] = False
                        
            # Check for duplicate records
            record_hashes = set()
            duplicates = 0
            
            for item in data:
                # Create a simple hash of the record
                item_hash = hash(frozenset(item.items()))
                if item_hash in record_hashes:
                    duplicates += 1
                else:
                    record_hashes.add(item_hash)
                    
            if duplicates > 0:
                validation_results["issues"].append({
                    "type": "duplicates",
                    "message": f"Found {duplicates} duplicate records ({duplicates/len(data):.1%})"
                })
                
                if duplicates / len(data) > 0.1:
                    validation_results["valid"] = False
                    
            # Check data types
            type_issues = self._check_data_types(data)
            if type_issues:
                validation_results["issues"].extend(type_issues)
                validation_results["valid"] = False
                
        return validation_results

    def _check_data_types(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for data type inconsistencies."""
        issues = []
        
        # Get all fields
        fields = set()
        for item in data:
            fields.update(item.keys())
            
        # Check each field
        for field in fields:
            # Get non-null values
            values = [item[field] for item in data if field in item and item[field] is not None]
            
            if not values:
                continue
                
            # Determine expected type from first value
            first_type = type(values[0])
            
            # Check for type inconsistencies
            type_counts = {}
            for value in values:
                value_type = type(value)
                type_counts[value_type] = type_counts.get(value_type, 0) + 1
                
            # If multiple types, report issue
            if len(type_counts) > 1:
                details = {str(t.__name__): count for t, count in type_counts.items()}
                
                issues.append({
                    "type": "inconsistent_types",
                    "message": f"Field '{field}' has inconsistent data types",
                    "details": details
                })
                
        return issues

    def _format_validation_results(self, results: Dict[str, Any]) -> str:
        """Format validation results for display."""
        output = []
        
        if results["valid"]:
            output.append("✅ Data is valid")
        else:
            output.append("❌ Data validation failed")
            
        if not results["issues"]:
            output.append("No issues found")
        else:
            output.append(f"Issues found: {len(results['issues'])}")
            
            for issue in results["issues"]:
                icon = "❌" if issue["type"] not in ["warning"] else "⚠️"
                output.append(f"\n{icon} {issue['message']}")
                
                if "details" in issue:
                    for key, value in issue["details"].items():
                        output.append(f"   - {key}: {value}")
                        
        return "\n".join(output)

    async def _generate_report(
        self, 
        data_path: Optional[str],
        analysis_type: str,
        visualization: bool,
        output_path: Optional[str]
    ) -> ToolResult:
        """Generate a comprehensive data report."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        if not output_path:
            return ToolResult(error="Output path is required")
            
        # First perform analysis
        analysis_result = await self._analyze_data(data_path, analysis_type, False, None)
        if analysis_result.error:
            return analysis_result
            
        # Then extract insights
        insights_result = await self._extract_insights(data_path, analysis_type, None)
        if insights_result.error:
            return insights_result
            
        # Generate report
        try:
            # Parse results
            analysis_output = analysis_result.output
            dataset_id = analysis_output.split("Dataset: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
            if not dataset:
                return ToolResult(error="Dataset not found")
                
            # Parse insights
            insights_output = insights_result.output
            insights_text = insights_output.split("Key Insights:")[1] if "Key Insights:" in insights_output else ""
            
            # Generate report
            report = self._generate_data_report(dataset, analysis_type, insights_text, visualization)
            
            # Save report
            full_output_path = self._resolve_path(output_path)
            await self._save_data(report, full_output_path, "html")
            
            return ToolResult(
                output=f"Data report generated successfully\n"
                       f"Dataset: {dataset['id']}\n"
                       f"Analysis Type: {analysis_type}\n"
                       f"Report saved to: {full_output_path}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error generating report: {str(e)}")

    def _generate_data_report(
        self, 
        dataset: Dict[str, Any],
        analysis_type: str,
        insights_text: str,
        visualization: bool
    ) -> str:
        """Generate a comprehensive HTML data report."""
        data = dataset["data"]
        
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<meta charset='UTF-8'>")
        html.append("<title>Data Analysis Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }")
        html.append("h1, h2, h3 { color: #333; }")
        html.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append(".summary { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }")
        html.append(".insights { background-color: #f0fff0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }")
        html.append(".warning { color: #ff4500; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append("<h1>Data Analysis Report</h1>")
        html.append(f"<p>Generated: {datetime.now().isoformat()}</p>")
        html.append(f"<p>Dataset: {dataset['id']}</p>")
        html.append(f"<p>Format: {dataset['format']}</p>")
        
        # Summary section
        html.append("<div class='summary'>")
        html.append("<h2>Dataset Summary</h2>")
        
        if isinstance(data, list):
            html.append(f"<p>Records: {len(data)}</p>")
            
            if data and isinstance(data[0], dict):
                html.append(f"<p>Fields: {len(data[0].keys())}</p>")
                html.append("<p>Field names: " + ", ".join(data[0].keys()) + "</p>")
                
        html.append("</div>")
        
        # Insights section
        if insights_text:
            html.append("<div class='insights'>")
            html.append("<h2>Key Insights</h2>")
            
            # Convert insights text to HTML
            insights_lines = insights_text.strip().split("\n")
            current_category = None
            
            for line in insights_lines:
                line = line.strip()
                if not line:
                    continue
                    
                if not line.startswith("  "):
                    # Category header
                    if current_category:
                        html.append("</ul>")
                    current_category = line.strip(":")
                    html.append(f"<h3>{current_category}</h3>")
                    html.append("<ul>")
                else:
                    # Insight item
                    insight = line.strip("• ").strip()
                    html.append(f"<li>{insight}</li>")
                    
            if current_category:
                html.append("</ul>")
                
            html.append("</div>")
            
        # Analysis section
        html.append("<h2>Analysis Results</h2>")
        
        if analysis_type == "statistical":
            html.append("<h3>Statistical Analysis</h3>")
            
            # Numeric columns
            numeric_columns = self._identify_numeric_columns(data)
            if numeric_columns:
                html.append("<h4>Numeric Fields</h4>")
                html.append("<table>")
                html.append("<tr><th>Field</th><th>Min</th><th>Max</th><th>Mean</th><th>Median</th><th>Std Dev</th></tr>")
                
                for column in numeric_columns:
                    values = [float(row[column]) for row in data if column in row and row[column] is not None]
                    
                    if values:
                        min_val = min(values)
                        max_val = max(values)
                        mean = sum(values) / len(values)
                        median = sorted(values)[len(values) // 2]
                        variance = sum((x - mean) ** 2 for x in values) / len(values)
                        std_dev = variance ** 0.5
                        
                        html.append("<tr>")
                        html.append(f"<td>{column}</td>")
                        html.append(f"<td>{min_val:.2f}</td>")
                        html.append(f"<td>{max_val:.2f}</td>")
                        html.append(f"<td>{mean:.2f}</td>")
                        html.append(f"<td>{median:.2f}</td>")
                        html.append(f"<td>{std_dev:.2f}</td>")
                        html.append("</tr>")
                        
                html.append("</table>")
                
            # Categorical columns
            categorical_columns = self._identify_categorical_columns(data)
            if categorical_columns:
                html.append("<h4>Categorical Fields</h4>")
                
                for column in categorical_columns:
                    values = [row[column] for row in data if column in row and row[column] is not None]
                    
                    if values:
                        # Count frequencies
                        frequencies = {}
                        for value in values:
                            frequencies[value] = frequencies.get(value, 0) + 1
                            
                        # Sort by frequency
                        sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
                        
                        html.append(f"<h5>{column}</h5>")
                        html.append("<table>")
                        html.append("<tr><th>Value</th><th>Count</th><th>Percentage</th></tr>")
                        
                        for value, count in list(sorted_frequencies.items())[:10]:  # Top 10
                            percentage = count / len(values) * 100
                            html.append("<tr>")
                            html.append(f"<td>{value}</td>")
                            html.append(f"<td>{count}</td>")
                            html.append(f"<td>{percentage:.1f}%</td>")
                            html.append("</tr>")
                            
                        html.append("</table>")
                        
        elif analysis_type == "correlation":
            html.append("<h3>Correlation Analysis</h3>")
            
            # Identify numeric columns
            numeric_columns = self._identify_numeric_columns(data)
            
            if len(numeric_columns) >= 2:
                # Calculate correlations
                correlations = {}
                for i, col1 in enumerate(numeric_columns):
                    correlations[col1] = {}
                    
                    for col2 in numeric_columns[i:]:
                        # Get paired values
                        pairs = [(float(row[col1]), float(row[col2])) 
                                for row in data 
                                if col1 in row and col2 in row 
                                and row[col1] is not None and row[col2] is not None]
                        
                        if not pairs:
                            correlations[col1][col2] = None
                            continue
                            
                        # Calculate Pearson correlation
                        x_values = [p[0] for p in pairs]
                        y_values = [p[1] for p in pairs]
                        
                        n = len(pairs)
                        sum_x = sum(x_values)
                        sum_y = sum(y_values)
                        sum_xy = sum(x * y for x, y in pairs)
                        sum_x2 = sum(x * x for x in x_values)
                        sum_y2 = sum(y * y for y in y_values)
                        
                        numerator = n * sum_xy - sum_x * sum_y
                        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
                        
                        if denominator == 0:
                            correlation = 0
                        else:
                            correlation = numerator / denominator
                            
                        correlations[col1][col2] = correlation
                        
                        # Add symmetric value
                        if col1 != col2:
                            if col2 not in correlations:
                                correlations[col2] = {}
                            correlations[col2][col1] = correlation
                            
                # Display correlation matrix
                html.append("<h4>Correlation Matrix</h4>")
                html.append("<table>")
                
                # Header row
                html.append("<tr><th></th>")
                for col in numeric_columns:
                    html.append(f"<th>{col}</th>")
                html.append("</tr>")
                
                # Data rows
                for row_col in numeric_columns:
                    html.append(f"<tr><th>{row_col}</th>")
                    
                    for col in numeric_columns:
                        corr = correlations.get(row_col, {}).get(col)
                        
                        if corr is None:
                            cell = "N/A"
                        else:
                            # Color-code based on correlation strength
                            if abs(corr) > 0.7:
                                color = "#ff9999" if corr < 0 else "#99ff99"
                            elif abs(corr) > 0.4:
                                color = "#ffcccc" if corr < 0 else "#ccffcc"
                            else:
                                color = "#ffffff"
                                
                            cell = f"<span style='background-color: {color}'>{corr:.2f}</span>"
                            
                        html.append(f"<td>{cell}</td>")
                        
                    html.append("</tr>")
                    
                html.append("</table>")
                
                # Strong correlations
                strong_correlations = []
                for col1 in numeric_columns:
                    for col2 in numeric_columns:
                        if col1 >= col2:
                            continue
                            
                        corr = correlations.get(col1, {}).get(col2)
                        if corr is not None and abs(corr) > 0.7:
                            strong_correlations.append((col1, col2, corr))
                            
                if strong_correlations:
                    html.append("<h4>Strong Correlations</h4>")
                    html.append("<ul>")
                    
                    for col1, col2, corr in strong_correlations:
                        direction = "positive" if corr > 0 else "negative"
                        html.append(f"<li><strong>{col1}</strong> and <strong>{col2}</strong>: {direction} correlation ({corr:.2f})</li>")
                        
                    html.append("</ul>")
                    
            else:
                html.append("<p class='warning'>Insufficient numeric columns for correlation analysis</p>")
                
        elif analysis_type == "trend":
            html.append("<h3>Trend Analysis</h3>")
            
            # Identify potential date/time columns
            date_columns = []
            for column in data[0].keys():
                # Check if column name suggests date
                if any(date_term in column.lower() for date_term in ["date", "time", "day", "month", "year"]):
                    date_columns.append(column)
                    
            if not date_columns:
                html.append("<p class='warning'>No date/time columns identified for trend analysis</p>")
            else:
                # Use first identified date column
                date_column = date_columns[0]
                html.append(f"<p>Time dimension: <strong>{date_column}</strong></p>")
                
                # Identify numeric columns
                numeric_columns = self._identify_numeric_columns(data)
                
                if not numeric_columns:
                    html.append("<p class='warning'>No numeric columns found for trend analysis</p>")
                else:
                    # Sort data by date column
                    sorted_data = sorted(data, key=lambda x: x.get(date_column, ""))
                    
                    # Analyze trends for each numeric column
                    for column in numeric_columns:
                        # Extract date-value pairs
                        series = [(row.get(date_column), float(row.get(column, 0))) 
                                for row in sorted_data 
                                if column in row and row[column] is not None]
                        
                        if len(series) < 2:
                            continue
                            
                        # Calculate trend indicators
                        values = [pair[1] for pair in series]
                        first_value = values[0]
                        last_value = values[-1]
                        
                        # Calculate change
                        absolute_change = last_value - first_value
                        percent_change = (absolute_change / first_value) * 100 if first_value != 0 else float('inf')
                        
                        # Determine trend direction
                        if absolute_change > 0:
                            direction = "increasing"
                            direction_class = "positive"
                        elif absolute_change < 0:
                            direction = "decreasing"
                            direction_class = "negative"
                        else:
                            direction = "stable"
                            direction_class = "neutral"
                            
                        html.append(f"<h4>{column}</h4>")
                        html.append("<table>")
                        html.append("<tr><th>Metric</th><th>Value</th></tr>")
                        html.append(f"<tr><td>First Value</td><td>{first_value:.2f}</td></tr>")
                        html.append(f"<tr><td>Last Value</td><td>{last_value:.2f}</td></tr>")
                        html.append(f"<tr><td>Absolute Change</td><td>{absolute_change:.2f}</td></tr>")
                        html.append(f"<tr><td>Percent Change</td><td>{percent_change:.1f}%</td></tr>")
                        html.append(f"<tr><td>Direction</td><td class='{direction_class}'>{direction}</td></tr>")
                        html.append("</table>")
                        
                        # Add data points table
                        html.append("<h5>Data Points</h5>")
                        html.append("<table>")
                        html.append(f"<tr><th>{date_column}</th><th>{column}</th></tr>")
                        
                        # Show at most 10 points
                        display_points = []
                        if len(series) <= 10:
                            display_points = series
                        else:
                            # Show first, last, and some middle points
                            display_points = series[:3] + series[len(series)//2-1:len(series)//2+2] + series[-3:]
                            
                        for date_val, value in display_points:
                            html.append(f"<tr><td>{date_val}</td><td>{value:.2f}</td></tr>")
                            
                        html.append("</table>")
                        
        # Data sample section
        html.append("<h2>Data Sample</h2>")
        
        if isinstance(data, list) and data:
            if all(isinstance(item, dict) for item in data):
                # Tabular data
                html.append("<table>")
                
                # Headers
                html.append("<tr>")
                for key in data[0].keys():
                    html.append(f"<th>{key}</th>")
                html.append("</tr>")
                
                # Rows (max 10)
                for item in data[:10]:
                    html.append("<tr>")
                    for value in item.values():
                        html.append(f"<td>{value}</td>")
                    html.append("</tr>")
                    
                html.append("</table>")
                
                if len(data) > 10:
                    html.append(f"<p>Showing 10 of {len(data)} records</p>")
            else:
                # Non-tabular list
                html.append("<pre>")
                html.append(str(data[:10]))
                html.append("</pre>")
                
                if len(data) > 10:
                    html.append(f"<p>Showing 10 of {len(data)} items</p>")
        else:
            # Non-list data
            html.append("<pre>")
            html.append(str(data))
            html.append("</pre>")
            
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)

    async def _detect_anomalies(
        self, 
        data_path: Optional[str],
        output_path: Optional[str]
    ) -> ToolResult:
        """Detect anomalies in data."""
        if not data_path:
            return ToolResult(error="Data path is required")
            
        # Load data if not already loaded
        dataset = self._get_dataset(data_path)
        if not dataset:
            load_result = await self._load_data(data_path, "auto")
            if load_result.error:
                return load_result
                
            # Get the newly loaded dataset
            dataset_id = load_result.output.split("Dataset ID: ")[1].split("\n")[0]
            dataset = self.loaded_datasets.get(dataset_id)
            
        # Detect anomalies
        try:
            anomalies = self._find_anomalies(dataset["data"])
            
            # Save result if output path provided
            result_text = ""
            if output_path:
                full_output_path = self._resolve_path(output_path)
                result_text = await self._save_data(anomalies, full_output_path, "html")
                
            return ToolResult(
                output=f"Anomaly detection completed\n"
                       f"Dataset: {dataset['id']}\n"
                       f"Anomalies found: {len(anomalies['records'])}\n"
                       f"{result_text}\n\n"
                       f"Anomaly Summary:\n{self._format_anomalies(anomalies)}"
            )
            
        except Exception as e:
            return ToolResult(error=f"Error detecting anomalies: {str(e)}")

    def _find_anomalies(self, data: Any) -> Dict[str, Any]:
        """Find anomalies in data."""
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Data must be a list of dictionaries for anomaly detection")
            
        anomalies = {
            "summary": {},
            "records": []
        }
        
        # Identify numeric columns
        numeric_columns = self._identify_numeric_columns(data)
        
        if not numeric_columns:
            anomalies["summary"]["message"] = "No numeric columns found for anomaly detection"
            return anomalies
            
        # Calculate statistics for each numeric column
        column_stats = {}
        for column in numeric_columns:
            values = [float(row[column]) for row in data if column in row and row[column] is not None]
            
            if not values:
                continue
                
            # Calculate mean and standard deviation
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            column_stats[column] = {
                "mean": mean,
                "std_dev": std_dev,
                "min": min(values),
                "max": max(values),
                "q1": sorted(values)[int(len(values) * 0.25)],
                "q3": sorted(values)[int(len(values) * 0.75)]
            }
            
            # Calculate IQR
            iqr = column_stats[column]["q3"] - column_stats[column]["q1"]
            column_stats[column]["lower_bound"] = column_stats[column]["q1"] - 1.5 * iqr
            column_stats[column]["upper_bound"] = column_stats[column]["q3"] + 1.5 * iqr
            
        # Find anomalies
        for i, row in enumerate(data):
            row_anomalies = []
            
            for column in numeric_columns:
                if column not in row or row[column] is None:
                    continue
                    
                value = float(row[column])
                stats = column_stats[column]
                
                # Check for outliers using Z-score
                z_score = (value - stats["mean"]) / stats["std_dev"] if stats["std_dev"] > 0 else 0
                
                if abs(z_score) > 3:
                    row_anomalies.append({
                        "column": column,
                        "value": value,
                        "z_score": z_score,
                        "type": "z_score",
                        "message": f"Z-score of {z_score:.2f} (more than 3 standard deviations from mean)"
                    })
                    
                # Check for outliers using IQR
                if value < stats["lower_bound"] or value > stats["upper_bound"]:
                    row_anomalies.append({
                        "column": column,
                        "value": value,
                        "type": "iqr",
                        "message": f"Outside IQR bounds ({stats['lower_bound']:.2f} - {stats['upper_bound']:.2f})"
                    })
                    
            if row_anomalies:
                anomalies["records"].append({
                    "index": i,
                    "record": row,
                    "anomalies": row_anomalies
                })
                
        # Add summary
        anomalies["summary"] = {
            "total_records": len(data),
            "anomalous_records": len(anomalies["records"]),
            "anomaly_rate": len(anomalies["records"]) / len(data) if data else 0,
            "column_stats": column_stats
        }
        
        return anomalies

    def _format_anomalies(self, anomalies: Dict[str, Any]) -> str:
        """Format anomalies for display."""
        output = []
        
        summary = anomalies["summary"]
        records = anomalies["records"]
        
        output.append(f"Total Records: {summary.get('total_records', 0)}")
        output.append(f"Anomalous Records: {summary.get('anomalous_records', 0)}")
        output.append(f"Anomaly Rate: {summary.get('anomaly_rate', 0):.1%}")
        
        if not records:
            output.append("\nNo anomalies detected")
            return "\n".join(output)
            
        # Group anomalies by column
        column_anomalies = {}
        for record in records:
            for anomaly in record["anomalies"]:
                column = anomaly["column"]
                if column not in column_anomalies:
                    column_anomalies[column] = []
                column_anomalies[column].append(anomaly)
                
        # Show anomalies by column
        output.append("\nAnomalies by Column:")
        for column, anomalies_list in column_anomalies.items():
            output.append(f"\n  {column}:")
            output.append(f"    Count: {len(anomalies_list)}")
            
            # Show example anomalies
            if anomalies_list:
                output.append("    Examples:")
                for anomaly in anomalies_list[:3]:
                    output.append(f"      - Value: {anomaly['value']}, {anomaly['message']}")
                    
                if len(anomalies_list) > 3:
                    output.append(f"      - ... and {len(anomalies_list) - 3} more")
                    
        # Show top anomalous records
        output.append("\nTop Anomalous Records:")
        for record in records[:5]:
            output.append(f"\n  Record #{record['index']}:")
            for anomaly in record["anomalies"]:
                output.append(f"    - {anomaly['column']}: {anomaly['value']} ({anomaly['message']})")
                
        if len(records) > 5:
            output.append(f"\n  ... and {len(records) - 5} more anomalous records")
            
        return "\n".join(output)