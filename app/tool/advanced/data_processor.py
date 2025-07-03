"""
Data Processor Tool
Handles data processing, transformation, and analysis
"""

import json
import os
import re
import csv
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class DataProcessorTool(BaseTool):
    """
    Advanced data processing tool for handling data transformation,
    cleaning, analysis, and visualization preparation.
    """

    name: str = "data_processor"
    description: str = """
    Process, transform, and analyze data from various sources.
    Clean data, perform transformations, generate statistics, and prepare data for visualization.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "analyze_data", "clean_data", "transform_data", "merge_datasets",
                    "extract_features", "generate_statistics", "detect_anomalies",
                    "prepare_visualization", "export_data", "generate_report"
                ],
                "description": "The data processing action to perform"
            },
            "data_source": {
                "type": "string",
                "description": "Path to data file or JSON string of data"
            },
            "data_format": {
                "type": "string",
                "enum": ["csv", "json", "xml", "excel", "text", "tabular"],
                "description": "Format of the input data"
            },
            "output_format": {
                "type": "string",
                "enum": ["csv", "json", "excel", "html", "markdown", "text"],
                "description": "Format for the output data"
            },
            "transformations": {
                "type": "string",
                "description": "JSON string of transformations to apply"
            },
            "filters": {
                "type": "string",
                "description": "JSON string of filters to apply"
            },
            "groupby": {
                "type": "string",
                "description": "Column or field to group data by"
            },
            "aggregations": {
                "type": "string",
                "description": "JSON string of aggregation operations"
            },
            "secondary_data_source": {
                "type": "string",
                "description": "Path to secondary data file for merge operations"
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the output data"
            },
            "visualization_type": {
                "type": "string",
                "enum": ["table", "bar", "line", "scatter", "pie", "heatmap", "histogram"],
                "description": "Type of visualization to prepare data for"
            },
            "analysis_depth": {
                "type": "string",
                "enum": ["basic", "intermediate", "advanced"],
                "description": "Depth of analysis to perform"
            }
        },
        "required": ["action", "data_source"]
    }

    # Data processing history for tracking and auditing
    processing_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Cache for processed datasets
    data_cache: Dict[str, Any] = Field(default_factory=dict)

    async def execute(
        self,
        action: str,
        data_source: str,
        data_format: str = "csv",
        output_format: str = "json",
        transformations: Optional[str] = None,
        filters: Optional[str] = None,
        groupby: Optional[str] = None,
        aggregations: Optional[str] = None,
        secondary_data_source: Optional[str] = None,
        output_path: Optional[str] = None,
        visualization_type: Optional[str] = None,
        analysis_depth: str = "basic",
        **kwargs
    ) -> ToolResult:
        """Execute the data processing action."""
        
        try:
            # Record operation start
            operation_start = datetime.now()
            operation_log = {
                "action": action,
                "data_source": data_source,
                "start_time": operation_start.isoformat(),
                "parameters": {
                    "data_format": data_format,
                    "output_format": output_format,
                    "transformations": transformations,
                    "filters": filters,
                    "groupby": groupby,
                    "aggregations": aggregations,
                    "secondary_data_source": secondary_data_source,
                    "output_path": output_path,
                    "visualization_type": visualization_type,
                    "analysis_depth": analysis_depth
                }
            }
            
            # Load data
            data = self._load_data(data_source, data_format)
            
            # Process based on action
            if action == "analyze_data":
                result = self._analyze_data(data, analysis_depth)
            elif action == "clean_data":
                result = self._clean_data(data, transformations)
            elif action == "transform_data":
                result = self._transform_data(data, transformations)
            elif action == "merge_datasets":
                if not secondary_data_source:
                    return ToolResult(error="Secondary data source is required for merge operation")
                secondary_data = self._load_data(secondary_data_source, data_format)
                result = self._merge_datasets(data, secondary_data, transformations)
            elif action == "extract_features":
                result = self._extract_features(data, transformations)
            elif action == "generate_statistics":
                result = self._generate_statistics(data, analysis_depth)
            elif action == "detect_anomalies":
                result = self._detect_anomalies(data, analysis_depth)
            elif action == "prepare_visualization":
                if not visualization_type:
                    return ToolResult(error="Visualization type is required for prepare_visualization action")
                result = self._prepare_visualization(data, visualization_type, transformations)
            elif action == "export_data":
                if not output_path:
                    return ToolResult(error="Output path is required for export_data action")
                result = self._export_data(data, output_path, output_format)
            elif action == "generate_report":
                result = self._generate_report(data, analysis_depth, output_format, output_path)
            else:
                return ToolResult(error=f"Unknown data processing action: {action}")
            
            # Save to output path if specified
            if output_path and action != "export_data" and action != "generate_report":
                self._save_data(result, output_path, output_format)
            
            # Update operation log
            operation_log.update({
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - operation_start).total_seconds(),
                "success": True,
                "result_summary": str(result)[:500] if isinstance(result, (dict, list)) else "Data processed successfully"
            })
            
            # Add to history
            self.processing_history.append(operation_log)
            
            # Format result for output
            formatted_result = self._format_result(result, action, output_format)
            
            return ToolResult(output=formatted_result)
            
        except Exception as e:
            # Log error
            operation_log.update({
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - operation_start).total_seconds(),
                "success": False,
                "error": str(e)
            })
            
            self.processing_history.append(operation_log)
            return ToolResult(error=f"Data processing error: {str(e)}")

    def _load_data(self, data_source: str, data_format: str) -> Any:
        """Load data from source with format detection."""
        # Check if data is a JSON string
        if data_source.startswith('{') or data_source.startswith('['):
            try:
                return json.loads(data_source)
            except json.JSONDecodeError:
                pass  # Not valid JSON, continue with file loading
        
        # Check if data source is a file path
        if os.path.exists(data_source):
            if data_format == "csv":
                return self._load_csv(data_source)
            elif data_format == "json":
                return self._load_json(data_source)
            elif data_format == "text":
                return self._load_text(data_source)
            elif data_format == "tabular":
                return self._load_tabular(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_format}")
        
        # If not a file or JSON string, treat as raw text data
        if data_format == "tabular":
            return self._parse_tabular_text(data_source)
        elif data_format == "csv":
            return self._parse_csv_text(data_source)
        
        # Default fallback
        return data_source

    def _load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                result.append(dict(row))
        return result

    def _load_json(self, file_path: str) -> Any:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_text(self, file_path: str) -> str:
        """Load data from text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_tabular(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from tabular text file (e.g., pipe or tab delimited)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._parse_tabular_text(content)

    def _parse_tabular_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse tabular text into structured data."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if not lines:
            return []
        
        # Detect delimiter
        first_line = lines[0]
        if '|' in first_line:
            delimiter = '|'
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = None  # Let csv.reader detect
        
        # Clean up lines if using pipe delimiter
        if delimiter == '|':
            lines = [line.strip('|').strip() for line in lines]
        
        # Parse header and data
        if delimiter:
            reader = csv.reader([lines[0]], delimiter=delimiter)
            headers = next(reader)
            headers = [h.strip() for h in headers]
            
            result = []
            for line in lines[1:]:
                reader = csv.reader([line], delimiter=delimiter)
                values = next(reader)
                values = [v.strip() for v in values]
                
                # Ensure headers and values have same length
                if len(headers) != len(values):
                    # Adjust by padding or truncating
                    if len(headers) > len(values):
                        values.extend([''] * (len(headers) - len(values)))
                    else:
                        values = values[:len(headers)]
                
                result.append(dict(zip(headers, values)))
            
            return result
        else:
            # Fallback to space-based splitting
            headers = re.split(r'\s{2,}', lines[0].strip())
            headers = [h.strip() for h in headers]
            
            result = []
            for line in lines[1:]:
                values = re.split(r'\s{2,}', line.strip())
                values = [v.strip() for v in values]
                
                # Ensure headers and values have same length
                if len(headers) != len(values):
                    if len(headers) > len(values):
                        values.extend([''] * (len(headers) - len(values)))
                    else:
                        values = values[:len(headers)]
                
                result.append(dict(zip(headers, values)))
            
            return result

    def _parse_csv_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse CSV text into structured data."""
        lines = text.strip().split('\n')
        reader = csv.DictReader(lines)
        return [dict(row) for row in reader]

    def _analyze_data(self, data: Any, analysis_depth: str) -> Dict[str, Any]:
        """Analyze data and generate insights."""
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "data_type": type(data).__name__,
            "summary": {},
            "structure": {},
            "statistics": {},
            "quality_issues": [],
            "recommendations": []
        }
        
        # Handle different data types
        if isinstance(data, list):
            analysis_result["summary"]["record_count"] = len(data)
            
            if data and isinstance(data[0], dict):
                # Tabular data (list of dicts)
                analysis_result["structure"]["fields"] = list(data[0].keys())
                analysis_result["structure"]["sample"] = data[0]
                
                # Basic field statistics
                field_stats = {}
                for field in analysis_result["structure"]["fields"]:
                    field_values = [item.get(field) for item in data if field in item]
                    field_stats[field] = self._analyze_field(field_values)
                
                analysis_result["statistics"]["fields"] = field_stats
                
                # Data quality checks
                analysis_result["quality_issues"] = self._check_data_quality(data)
                
                # Generate recommendations
                analysis_result["recommendations"] = self._generate_data_recommendations(data, field_stats)
            
        elif isinstance(data, dict):
            # Dictionary data
            analysis_result["structure"]["keys"] = list(data.keys())
            analysis_result["structure"]["nested_objects"] = [k for k, v in data.items() if isinstance(v, (dict, list))]
            
            # Analyze values
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    analysis_result["statistics"][key] = {
                        "type": "array",
                        "length": len(value),
                        "sample": value[0] if len(value) > 0 else None
                    }
        
        elif isinstance(data, str):
            # Text data
            analysis_result["summary"]["character_count"] = len(data)
            analysis_result["summary"]["word_count"] = len(data.split())
            analysis_result["summary"]["line_count"] = len(data.splitlines())
            
            # Text patterns
            analysis_result["statistics"]["patterns"] = {
                "urls": len(re.findall(r'https?://\S+', data)),
                "emails": len(re.findall(r'\S+@\S+\.\S+', data)),
                "numbers": len(re.findall(r'\b\d+\b', data)),
                "dates": len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', data))
            }
        
        # Advanced analysis for deeper analysis levels
        if analysis_depth in ["intermediate", "advanced"]:
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # Correlation analysis for numeric fields
                numeric_fields = {}
                for field, stats in analysis_result["statistics"]["fields"].items():
                    if stats.get("data_type") == "numeric":
                        values = [float(item.get(field, 0)) for item in data if field in item and item[field] and str(item[field]).replace('.', '', 1).isdigit()]
                        if values:
                            numeric_fields[field] = values
                
                if len(numeric_fields) >= 2:
                    analysis_result["statistics"]["correlations"] = self._calculate_correlations(numeric_fields)
            
            # Add data distribution analysis
            if isinstance(data, list) and len(data) > 0:
                analysis_result["statistics"]["distribution"] = self._analyze_distribution(data)
        
        # Even more advanced analysis
        if analysis_depth == "advanced":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # Outlier detection
                analysis_result["statistics"]["outliers"] = self._detect_outliers(data)
                
                # Pattern detection
                analysis_result["statistics"]["patterns"] = self._detect_patterns(data)
        
        return analysis_result

    def _analyze_field(self, values: List[Any]) -> Dict[str, Any]:
        """Analyze a single field/column of data."""
        if not values:
            return {"data_type": "unknown", "count": 0}
        
        # Count non-null values
        non_null_values = [v for v in values if v is not None and v != ""]
        non_null_count = len(non_null_values)
        
        # Determine data type
        numeric_count = sum(1 for v in non_null_values if str(v).replace('.', '', 1).isdigit())
        date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
        date_count = sum(1 for v in non_null_values if isinstance(v, str) and date_pattern.match(v))
        
        if numeric_count / max(1, non_null_count) > 0.8:
            data_type = "numeric"
            # Convert to float for statistics
            try:
                numeric_values = [float(v) for v in non_null_values if str(v).replace('.', '', 1).isdigit()]
                if numeric_values:
                    return {
                        "data_type": data_type,
                        "count": len(values),
                        "non_null_count": non_null_count,
                        "null_percentage": (len(values) - non_null_count) / max(1, len(values)),
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "mean": sum(numeric_values) / len(numeric_values),
                        "unique_count": len(set(str(v) for v in values))
                    }
            except (ValueError, TypeError):
                pass
        
        elif date_count / max(1, non_null_count) > 0.8:
            data_type = "date"
        else:
            data_type = "string"
            
        # Basic statistics for all types
        return {
            "data_type": data_type,
            "count": len(values),
            "non_null_count": non_null_count,
            "null_percentage": (len(values) - non_null_count) / max(1, len(values)),
            "unique_count": len(set(str(v) for v in values)),
            "unique_percentage": len(set(str(v) for v in values)) / max(1, non_null_count)
        }

    def _check_data_quality(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for data quality issues."""
        quality_issues = []
        
        if not data:
            return quality_issues
        
        # Check for missing values
        for field in data[0].keys():
            missing_count = sum(1 for item in data if field not in item or item[field] is None or item[field] == "")
            if missing_count > 0:
                missing_percentage = missing_count / len(data)
                if missing_percentage > 0.1:  # More than 10% missing
                    quality_issues.append({
                        "type": "missing_values",
                        "field": field,
                        "count": missing_count,
                        "percentage": missing_percentage,
                        "severity": "high" if missing_percentage > 0.5 else "medium"
                    })
        
        # Check for duplicate records
        if len(data) > 1:
            # Create a simple hash for each record
            record_hashes = {}
            for i, item in enumerate(data):
                item_hash = hash(frozenset(item.items()))
                if item_hash in record_hashes:
                    quality_issues.append({
                        "type": "duplicate_record",
                        "first_index": record_hashes[item_hash],
                        "second_index": i,
                        "severity": "medium"
                    })
                else:
                    record_hashes[item_hash] = i
        
        # Check for inconsistent data types
        for field in data[0].keys():
            field_types = set()
            for item in data:
                if field in item and item[field] is not None:
                    field_types.add(type(item[field]).__name__)
            
            if len(field_types) > 1:
                quality_issues.append({
                    "type": "inconsistent_types",
                    "field": field,
                    "types": list(field_types),
                    "severity": "high"
                })
        
        return quality_issues

    def _generate_data_recommendations(self, data: List[Dict[str, Any]], field_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on data analysis."""
        recommendations = []
        
        if not data:
            recommendations.append("No data available for analysis")
            return recommendations
        
        # Check for missing values
        fields_with_nulls = [field for field, stats in field_stats.items() 
                            if stats.get("null_percentage", 0) > 0]
        
        if fields_with_nulls:
            if any(field_stats[f].get("null_percentage", 0) > 0.5 for f in fields_with_nulls):
                recommendations.append(f"Consider removing or imputing fields with high null percentages: {', '.join(f for f in fields_with_nulls if field_stats[f].get('null_percentage', 0) > 0.5)}")
            else:
                recommendations.append(f"Consider strategies for handling missing values in: {', '.join(fields_with_nulls)}")
        
        # Check for highly correlated fields
        if "correlations" in field_stats:
            high_correlations = [(f1, f2, corr) for f1, f2, corr in field_stats["correlations"] if abs(corr) > 0.9]
            if high_correlations:
                recommendations.append(f"Found {len(high_correlations)} highly correlated field pairs. Consider feature selection or dimensionality reduction.")
        
        # Check for low cardinality fields
        low_cardinality_fields = [field for field, stats in field_stats.items() 
                                if stats.get("unique_count", 0) == 1]
        if low_cardinality_fields:
            recommendations.append(f"Fields with constant values detected: {', '.join(low_cardinality_fields)}. Consider removing these fields.")
        
        # Check for high cardinality fields
        high_cardinality_fields = [field for field, stats in field_stats.items() 
                                  if stats.get("data_type") == "string" and 
                                  stats.get("unique_percentage", 0) > 0.9 and
                                  stats.get("unique_count", 0) > 10]
        if high_cardinality_fields:
            recommendations.append(f"High cardinality string fields detected: {', '.join(high_cardinality_fields)}. These might be IDs or unique identifiers.")
        
        # Data size recommendations
        if len(data) < 10:
            recommendations.append("Small dataset detected. Results may not be statistically significant.")
        elif len(data) > 10000:
            recommendations.append("Large dataset detected. Consider sampling for exploratory analysis.")
        
        return recommendations

    def _calculate_correlations(self, numeric_fields: Dict[str, List[float]]) -> List[tuple]:
        """Calculate correlations between numeric fields."""
        correlations = []
        fields = list(numeric_fields.keys())
        
        for i in range(len(fields)):
            for j in range(i+1, len(fields)):
                field1 = fields[i]
                field2 = fields[j]
                values1 = numeric_fields[field1]
                values2 = numeric_fields[field2]
                
                # Ensure same length by using the shorter list
                min_length = min(len(values1), len(values2))
                values1 = values1[:min_length]
                values2 = values2[:min_length]
                
                # Simple correlation calculation
                if min_length < 2:
                    continue
                    
                mean1 = sum(values1) / min_length
                mean2 = sum(values2) / min_length
                
                variance1 = sum((x - mean1) ** 2 for x in values1) / min_length
                variance2 = sum((x - mean2) ** 2 for x in values2) / min_length
                
                if variance1 == 0 or variance2 == 0:
                    continue
                    
                covariance = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(min_length)) / min_length
                correlation = covariance / ((variance1 * variance2) ** 0.5)
                
                correlations.append((field1, field2, correlation))
        
        return correlations

    def _analyze_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data distribution for key fields."""
        if not data or not isinstance(data[0], dict):
            return {}
            
        distribution = {}
        sample_item = data[0]
        
        for field, value in sample_item.items():
            # Only analyze certain types
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                # For numeric fields, create histogram buckets
                try:
                    values = [float(item.get(field, 0)) for item in data if field in item and item[field] and str(item[field]).replace('.', '', 1).isdigit()]
                    if not values:
                        continue
                        
                    min_val = min(values)
                    max_val = max(values)
                    
                    # Create 5 buckets
                    bucket_size = (max_val - min_val) / 5 if max_val > min_val else 1
                    buckets = {}
                    
                    for i in range(5):
                        bucket_min = min_val + i * bucket_size
                        bucket_max = min_val + (i + 1) * bucket_size
                        bucket_name = f"{bucket_min:.2f}-{bucket_max:.2f}"
                        buckets[bucket_name] = sum(1 for v in values if bucket_min <= v < bucket_max or (i == 4 and v == max_val))
                    
                    distribution[field] = {
                        "type": "numeric",
                        "min": min_val,
                        "max": max_val,
                        "histogram": buckets
                    }
                except (ValueError, TypeError):
                    continue
            elif isinstance(value, str):
                # For string fields, count frequencies of top values
                value_counts = {}
                for item in data:
                    if field in item and item[field]:
                        val = str(item[field])
                        value_counts[val] = value_counts.get(val, 0) + 1
                
                # Get top 5 values
                top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                distribution[field] = {
                    "type": "categorical",
                    "unique_values": len(value_counts),
                    "top_values": dict(top_values)
                }
        
        return distribution

    def _detect_outliers(self, data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Detect outliers in numeric fields using IQR method."""
        if not data or not isinstance(data[0], dict):
            return {}
            
        outliers = {}
        sample_item = data[0]
        
        for field, value in sample_item.items():
            # Only analyze numeric fields
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                try:
                    values = [float(item.get(field, 0)) for item in data if field in item and item[field] and str(item[field]).replace('.', '', 1).isdigit()]
                    if len(values) < 4:  # Need enough data points
                        continue
                        
                    # Sort values
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    
                    # Calculate Q1 and Q3
                    q1_idx = n // 4
                    q3_idx = (3 * n) // 4
                    q1 = sorted_values[q1_idx]
                    q3 = sorted_values[q3_idx]
                    
                    # Calculate IQR and bounds
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    
                    # Find outliers
                    outlier_indices = [i for i, item in enumerate(data) 
                                      if field in item and item[field] and 
                                      str(item[field]).replace('.', '', 1).isdigit() and
                                      (float(item[field]) < lower_bound or float(item[field]) > upper_bound)]
                    
                    if outlier_indices:
                        outliers[field] = outlier_indices
                except (ValueError, TypeError):
                    continue
        
        return outliers

    def _detect_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in the data."""
        if not data:
            return {}
            
        patterns = {}
        
        # Detect sequential patterns in numeric fields
        for field in data[0].keys():
            try:
                values = [float(item.get(field, 0)) for item in data if field in item and item[field] and str(item[field]).replace('.', '', 1).isdigit()]
                if len(values) < 3:
                    continue
                
                # Check for arithmetic sequence
                diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_diff = sum(diffs) / len(diffs)
                is_arithmetic = all(abs(diff - avg_diff) < 0.001 for diff in diffs)
                
                if is_arithmetic:
                    patterns[field] = {
                        "type": "arithmetic_sequence",
                        "common_difference": avg_diff
                    }
                    continue
                
                # Check for geometric sequence
                if all(v > 0 for v in values):
                    ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
                    avg_ratio = sum(ratios) / len(ratios)
                    is_geometric = all(abs(ratio - avg_ratio) < 0.001 for ratio in ratios)
                    
                    if is_geometric:
                        patterns[field] = {
                            "type": "geometric_sequence",
                            "common_ratio": avg_ratio
                        }
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        
        return patterns

    def _clean_data(self, data: Any, transformations: Optional[str]) -> Any:
        """Clean data by removing nulls, duplicates, and applying basic transformations."""
        if not data:
            return data
            
        # Parse transformations if provided
        cleaning_rules = {}
        if transformations:
            try:
                cleaning_rules = json.loads(transformations)
            except json.JSONDecodeError:
                # If not valid JSON, treat as simple comma-separated list of fields to clean
                cleaning_rules = {"fields": transformations.split(",")}
        
        # Handle different data types
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries (tabular data)
            result = []
            
            # Get fields to process
            fields_to_clean = cleaning_rules.get("fields", list(data[0].keys()))
            
            # Apply cleaning to each record
            for item in data:
                clean_item = item.copy()
                
                # Remove or replace nulls
                for field in fields_to_clean:
                    if field in clean_item:
                        # Handle null values
                        if clean_item[field] is None or clean_item[field] == "":
                            if cleaning_rules.get("null_strategy", "remove") == "remove":
                                del clean_item[field]
                            else:
                                # Replace with default value
                                clean_item[field] = cleaning_rules.get("default_value", "")
                        
                        # Trim whitespace for strings
                        elif isinstance(clean_item[field], str):
                            clean_item[field] = clean_item[field].strip()
                
                result.append(clean_item)
            
            # Remove duplicates if specified
            if cleaning_rules.get("remove_duplicates", True):
                # Create a set of frozensets for deduplication
                seen = set()
                unique_result = []
                
                for item in result:
                    # Create a hashable representation
                    item_hash = frozenset(item.items())
                    if item_hash not in seen:
                        seen.add(item_hash)
                        unique_result.append(item)
                
                result = unique_result
            
            return result
            
        elif isinstance(data, dict):
            # Single dictionary
            result = data.copy()
            
            # Clean each field
            for key, value in list(result.items()):
                if value is None or value == "":
                    if cleaning_rules.get("null_strategy", "remove") == "remove":
                        del result[key]
                    else:
                        result[key] = cleaning_rules.get("default_value", "")
                elif isinstance(value, str):
                    result[key] = value.strip()
            
            return result
            
        elif isinstance(data, str):
            # Text data
            result = data
            
            # Apply text cleaning
            if cleaning_rules.get("trim_whitespace", True):
                result = result.strip()
            
            if cleaning_rules.get("normalize_whitespace", True):
                result = re.sub(r'\s+', ' ', result)
            
            if cleaning_rules.get("remove_special_chars", False):
                result = re.sub(r'[^\w\s]', '', result)
            
            return result
            
        # Default: return data as is
        return data

    def _transform_data(self, data: Any, transformations: Optional[str]) -> Any:
        """Apply transformations to data."""
        if not data or not transformations:
            return data
            
        # Parse transformations
        try:
            transform_rules = json.loads(transformations)
        except json.JSONDecodeError:
            raise ValueError("Invalid transformations JSON")
        
        # Handle different data types
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries (tabular data)
            result = []
            
            for item in data:
                transformed_item = item.copy()
                
                # Apply field transformations
                for transform in transform_rules.get("field_operations", []):
                    field = transform.get("field")
                    operation = transform.get("operation")
                    
                    if not field or not operation or field not in transformed_item:
                        continue
                    
                    value = transformed_item[field]
                    
                    # Apply operation
                    if operation == "uppercase" and isinstance(value, str):
                        transformed_item[field] = value.upper()
                    elif operation == "lowercase" and isinstance(value, str):
                        transformed_item[field] = value.lower()
                    elif operation == "capitalize" and isinstance(value, str):
                        transformed_item[field] = value.capitalize()
                    elif operation == "trim" and isinstance(value, str):
                        transformed_item[field] = value.strip()
                    elif operation == "replace" and isinstance(value, str):
                        old_val = transform.get("old_value", "")
                        new_val = transform.get("new_value", "")
                        transformed_item[field] = value.replace(old_val, new_val)
                    elif operation == "extract" and isinstance(value, str):
                        pattern = transform.get("pattern", "")
                        if pattern:
                            match = re.search(pattern, value)
                            transformed_item[field] = match.group(0) if match else ""
                    elif operation == "multiply" and (isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit())):
                        factor = transform.get("factor", 1)
                        transformed_item[field] = float(value) * factor
                    elif operation == "round" and (isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit())):
                        decimals = transform.get("decimals", 0)
                        transformed_item[field] = round(float(value), decimals)
                
                # Add computed fields
                for compute in transform_rules.get("computed_fields", []):
                    new_field = compute.get("name")
                    expression = compute.get("expression")
                    
                    if not new_field or not expression:
                        continue
                    
                    # Simple expression evaluation
                    try:
                        # Replace field references with values
                        eval_expr = expression
                        for field in transformed_item:
                            if isinstance(transformed_item[field], (int, float)) or (isinstance(transformed_item[field], str) and transformed_item[field].replace('.', '', 1).isdigit()):
                                eval_expr = eval_expr.replace(f"{{{field}}}", str(transformed_item[field]))
                        
                        # Evaluate expression
                        transformed_item[new_field] = eval(eval_expr)
                    except Exception:
                        transformed_item[new_field] = None
                
                # Rename fields
                for rename in transform_rules.get("rename_fields", []):
                    old_name = rename.get("old_name")
                    new_name = rename.get("new_name")
                    
                    if old_name and new_name and old_name in transformed_item:
                        transformed_item[new_name] = transformed_item.pop(old_name)
                
                # Filter fields
                if "keep_fields" in transform_rules:
                    keep_fields = transform_rules["keep_fields"]
                    transformed_item = {k: v for k, v in transformed_item.items() if k in keep_fields}
                elif "drop_fields" in transform_rules:
                    drop_fields = transform_rules["drop_fields"]
                    transformed_item = {k: v for k, v in transformed_item.items() if k not in drop_fields}
                
                result.append(transformed_item)
            
            # Apply filters
            if "filters" in transform_rules:
                for filter_rule in transform_rules["filters"]:
                    field = filter_rule.get("field")
                    operator = filter_rule.get("operator")
                    value = filter_rule.get("value")
                    
                    if not field or not operator:
                        continue
                    
                    result = [item for item in result if self._apply_filter(item, field, operator, value)]
            
            return result
            
        elif isinstance(data, dict):
            # Single dictionary
            result = data.copy()
            
            # Apply transformations
            for transform in transform_rules.get("field_operations", []):
                field = transform.get("field")
                operation = transform.get("operation")
                
                if not field or not operation or field not in result:
                    continue
                
                value = result[field]
                
                # Apply operation (same as above)
                if operation == "uppercase" and isinstance(value, str):
                    result[field] = value.upper()
                elif operation == "lowercase" and isinstance(value, str):
                    result[field] = value.lower()
                # ... other operations
            
            return result
            
        elif isinstance(data, str):
            # Text data
            result = data
            
            # Apply text transformations
            for transform in transform_rules.get("text_operations", []):
                operation = transform.get("operation")
                
                if operation == "uppercase":
                    result = result.upper()
                elif operation == "lowercase":
                    result = result.lower()
                elif operation == "capitalize":
                    result = result.capitalize()
                elif operation == "trim":
                    result = result.strip()
                elif operation == "replace":
                    old_val = transform.get("old_value", "")
                    new_val = transform.get("new_value", "")
                    result = result.replace(old_val, new_val)
                elif operation == "extract":
                    pattern = transform.get("pattern", "")
                    if pattern:
                        match = re.search(pattern, result)
                        result = match.group(0) if match else ""
            
            return result
            
        # Default: return data as is
        return data

    def _apply_filter(self, item: Dict[str, Any], field: str, operator: str, value: Any) -> bool:
        """Apply filter condition to an item."""
        if field not in item:
            return False
            
        item_value = item[field]
        
        # Handle numeric comparisons
        if operator in ["eq", "=", "=="]:
            return item_value == value
        elif operator in ["neq", "!=", "<>"]:
            return item_value != value
        elif operator in ["gt", ">"]:
            try:
                return float(item_value) > float(value)
            except (ValueError, TypeError):
                return False
        elif operator in ["lt", "<"]:
            try:
                return float(item_value) < float(value)
            except (ValueError, TypeError):
                return False
        elif operator in ["gte", ">="]:
            try:
                return float(item_value) >= float(value)
            except (ValueError, TypeError):
                return False
        elif operator in ["lte", "<="]:
            try:
                return float(item_value) <= float(value)
            except (ValueError, TypeError):
                return False
        
        # String operations
        elif operator == "contains" and isinstance(item_value, str):
            return value in item_value
        elif operator == "startswith" and isinstance(item_value, str):
            return item_value.startswith(value)
        elif operator == "endswith" and isinstance(item_value, str):
            return item_value.endswith(value)
        elif operator == "matches" and isinstance(item_value, str):
            try:
                return bool(re.search(value, item_value))
            except re.error:
                return False
        
        # Default
        return True

    def _merge_datasets(self, primary_data: Any, secondary_data: Any, merge_config: Optional[str]) -> Any:
        """Merge two datasets based on configuration."""
        if not primary_data or not secondary_data:
            return primary_data
            
        # Parse merge configuration
        merge_rules = {}
        if merge_config:
            try:
                merge_rules = json.loads(merge_config)
            except json.JSONDecodeError:
                raise ValueError("Invalid merge configuration JSON")
        
        # Handle different data types
        if isinstance(primary_data, list) and isinstance(secondary_data, list):
            # Both are lists
            if not primary_data or not secondary_data:
                return primary_data + secondary_data
                
            if isinstance(primary_data[0], dict) and isinstance(secondary_data[0], dict):
                # Both are lists of dictionaries (tabular data)
                merge_type = merge_rules.get("merge_type", "inner")
                join_field = merge_rules.get("join_field")
                
                if not join_field:
                    # Without join field, just concatenate
                    return primary_data + secondary_data
                
                # Perform join operation
                result = []
                
                # Create lookup for secondary data
                secondary_lookup = {}
                for item in secondary_data:
                    if join_field in item:
                        key = item[join_field]
                        if key not in secondary_lookup:
                            secondary_lookup[key] = []
                        secondary_lookup[key].append(item)
                
                # Perform join
                for primary_item in primary_data:
                    if join_field not in primary_item:
                        if merge_type == "left" or merge_type == "outer":
                            result.append(primary_item)
                        continue
                    
                    key = primary_item[join_field]
                    
                    if key in secondary_lookup:
                        # Match found
                        for secondary_item in secondary_lookup[key]:
                            # Merge items
                            merged_item = primary_item.copy()
                            
                            # Add fields from secondary item
                            for field, value in secondary_item.items():
                                if field != join_field or merge_rules.get("include_join_field", True):
                                    field_name = field
                                    
                                    # Handle field conflicts
                                    if field in merged_item and field != join_field:
                                        conflict_strategy = merge_rules.get("conflict_strategy", "suffix")
                                        if conflict_strategy == "suffix":
                                            field_name = f"{field}_secondary"
                                        elif conflict_strategy == "prefix":
                                            field_name = f"secondary_{field}"
                                        elif conflict_strategy == "overwrite":
                                            field_name = field
                                    
                                    merged_item[field_name] = value
                            
                            result.append(merged_item)
                    elif merge_type in ["left", "outer"]:
                        # No match but include in left or outer join
                        result.append(primary_item)
                
                # Add unmatched secondary items for outer join
                if merge_type == "outer":
                    primary_keys = {item.get(join_field) for item in primary_data if join_field in item}
                    
                    for key, items in secondary_lookup.items():
                        if key not in primary_keys:
                            for secondary_item in items:
                                result.append(secondary_item)
                
                return result
            else:
                # Simple lists, just concatenate
                return primary_data + secondary_data
        
        elif isinstance(primary_data, dict) and isinstance(secondary_data, dict):
            # Both are dictionaries, merge them
            result = primary_data.copy()
            
            # Add fields from secondary data
            for key, value in secondary_data.items():
                if key not in result:
                    result[key] = value
                else:
                    # Handle conflicts
                    conflict_strategy = merge_rules.get("conflict_strategy", "suffix")
                    if conflict_strategy == "suffix":
                        result[f"{key}_secondary"] = value
                    elif conflict_strategy == "prefix":
                        result[f"secondary_{key}"] = value
                    elif conflict_strategy == "overwrite":
                        result[key] = value
            
            return result
        
        # Default: return primary data
        return primary_data

    def _extract_features(self, data: Any, feature_config: Optional[str]) -> Any:
        """Extract features from data based on configuration."""
        if not data:
            return data
            
        # Parse feature configuration
        feature_rules = {}
        if feature_config:
            try:
                feature_rules = json.loads(feature_config)
            except json.JSONDecodeError:
                raise ValueError("Invalid feature configuration JSON")
        
        # Handle different data types
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries (tabular data)
            result = []
            
            for item in data:
                features = {}
                
                # Extract specified features
                for feature in feature_rules.get("features", []):
                    feature_name = feature.get("name")
                    source_field = feature.get("source_field")
                    feature_type = feature.get("type", "direct")
                    
                    if not feature_name or not source_field:
                        continue
                    
                    if source_field not in item:
                        continue
                    
                    source_value = item[source_field]
                    
                    # Apply feature extraction based on type
                    if feature_type == "direct":
                        features[feature_name] = source_value
                    elif feature_type == "binary" and feature.get("threshold") is not None:
                        threshold = float(feature.get("threshold"))
                        try:
                            features[feature_name] = 1 if float(source_value) >= threshold else 0
                        except (ValueError, TypeError):
                            features[feature_name] = 0
                    elif feature_type == "categorical" and isinstance(source_value, str):
                        # One-hot encoding
                        categories = feature.get("categories", [])
                        if not categories:
                            features[feature_name] = source_value
                        else:
                            for category in categories:
                                features[f"{feature_name}_{category}"] = 1 if source_value == category else 0
                    elif feature_type == "normalized" and (isinstance(source_value, (int, float)) or (isinstance(source_value, str) and source_value.replace('.', '', 1).isdigit())):
                        min_val = float(feature.get("min", 0))
                        max_val = float(feature.get("max", 1))
                        try:
                            value = float(source_value)
                            if max_val > min_val:
                                features[feature_name] = (value - min_val) / (max_val - min_val)
                            else:
                                features[feature_name] = value
                        except (ValueError, TypeError):
                            features[feature_name] = 0
                
                # Add computed features
                for compute in feature_rules.get("computed_features", []):
                    feature_name = compute.get("name")
                    expression = compute.get("expression")
                    
                    if not feature_name or not expression:
                        continue
                    
                    # Simple expression evaluation
                    try:
                        # Replace field references with values
                        eval_expr = expression
                        for field in item:
                            if isinstance(item[field], (int, float)) or (isinstance(item[field], str) and item[field].replace('.', '', 1).isdigit()):
                                eval_expr = eval_expr.replace(f"{{{field}}}", str(item[field]))
                        
                        # Evaluate expression
                        features[feature_name] = eval(eval_expr)
                    except Exception:
                        features[feature_name] = None
                
                # Include original fields if specified
                if feature_rules.get("include_original", False):
                    result.append({**item, **features})
                else:
                    result.append(features)
            
            return result
        
        # Default: return data as is
        return data

    def _generate_statistics(self, data: Any, analysis_depth: str) -> Dict[str, Any]:
        """Generate statistical summary of the data."""
        # This is similar to _analyze_data but focused on statistics
        stats = {
            "timestamp": datetime.now().isoformat(),
            "record_count": 0,
            "field_statistics": {},
            "summary_statistics": {}
        }
        
        # Handle different data types
        if isinstance(data, list):
            stats["record_count"] = len(data)
            
            if data and isinstance(data[0], dict):
                # Tabular data
                
                # Calculate field statistics
                for field in data[0].keys():
                    field_values = [item.get(field) for item in data if field in item]
                    stats["field_statistics"][field] = self._calculate_field_statistics(field_values, analysis_depth)
                
                # Calculate summary statistics
                stats["summary_statistics"] = self._calculate_summary_statistics(data, analysis_depth)
        
        return stats

    def _calculate_field_statistics(self, values: List[Any], analysis_depth: str) -> Dict[str, Any]:
        """Calculate statistics for a single field."""
        if not values:
            return {"count": 0}
            
        field_stats = {
            "count": len(values),
            "null_count": sum(1 for v in values if v is None or v == ""),
            "unique_count": len(set(str(v) for v in values if v is not None and v != ""))
        }
        
        # Calculate null percentage
        field_stats["null_percentage"] = field_stats["null_count"] / field_stats["count"]
        
        # Determine data type
        numeric_values = []
        for v in values:
            if v is not None and v != "":
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    pass
        
        if len(numeric_values) > 0.5 * len(values):
            # Mostly numeric
            field_stats["data_type"] = "numeric"
            field_stats["min"] = min(numeric_values)
            field_stats["max"] = max(numeric_values)
            field_stats["mean"] = sum(numeric_values) / len(numeric_values)
            
            # Calculate median
            sorted_values = sorted(numeric_values)
            n = len(sorted_values)
            if n % 2 == 0:
                field_stats["median"] = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
            else:
                field_stats["median"] = sorted_values[n//2]
            
            # Calculate standard deviation
            if len(numeric_values) > 1:
                mean = field_stats["mean"]
                variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                field_stats["std_dev"] = variance ** 0.5
            
            # Advanced statistics for deeper analysis
            if analysis_depth in ["intermediate", "advanced"]:
                # Calculate quartiles
                field_stats["q1"] = sorted_values[n//4]
                field_stats["q3"] = sorted_values[(3*n)//4]
                field_stats["iqr"] = field_stats["q3"] - field_stats["q1"]
                
                # Calculate skewness (simplified)
                if len(numeric_values) > 2:
                    mean = field_stats["mean"]
                    std_dev = field_stats["std_dev"]
                    if std_dev > 0:
                        skewness = sum(((x - mean) / std_dev) ** 3 for x in numeric_values) / len(numeric_values)
                        field_stats["skewness"] = skewness
        else:
            # Categorical
            field_stats["data_type"] = "categorical"
            
            # Calculate value frequencies
            value_counts = {}
            for v in values:
                if v is not None and v != "":
                    str_v = str(v)
                    value_counts[str_v] = value_counts.get(str_v, 0) + 1
            
            # Get top values
            top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            field_stats["top_values"] = dict(top_values)
            
            # Calculate mode
            if value_counts:
                mode_value, mode_count = max(value_counts.items(), key=lambda x: x[1])
                field_stats["mode"] = mode_value
                field_stats["mode_count"] = mode_count
                field_stats["mode_percentage"] = mode_count / field_stats["count"]
        
        return field_stats

    def _calculate_summary_statistics(self, data: List[Dict[str, Any]], analysis_depth: str) -> Dict[str, Any]:
        """Calculate summary statistics across the dataset."""
        if not data:
            return {}
            
        summary = {
            "record_count": len(data),
            "field_count": len(data[0]) if data else 0,
            "completeness": 0.0
        }
        
        # Calculate overall completeness
        if data and isinstance(data[0], dict):
            total_cells = len(data) * len(data[0])
            filled_cells = 0
            
            for item in data:
                for value in item.values():
                    if value is not None and value != "":
                        filled_cells += 1
            
            summary["completeness"] = filled_cells / total_cells if total_cells > 0 else 0
        
        # Advanced summary statistics
        if analysis_depth in ["intermediate", "advanced"]:
            # Identify potential key fields
            if data and isinstance(data[0], dict):
                potential_keys = []
                
                for field in data[0].keys():
                    values = [item.get(field) for item in data if field in item]
                    unique_values = set(str(v) for v in values if v is not None and v != "")
                    
                    if len(unique_values) == len(values) - sum(1 for v in values if v is None or v == ""):
                        # All non-null values are unique
                        potential_keys.append(field)
                
                summary["potential_key_fields"] = potential_keys
        
        return summary

    def _detect_anomalies(self, data: Any, analysis_depth: str) -> Dict[str, Any]:
        """Detect anomalies in the data."""
        anomalies = {
            "timestamp": datetime.now().isoformat(),
            "record_anomalies": [],
            "field_anomalies": {},
            "pattern_violations": []
        }
        
        if not data or not isinstance(data, list) or not data:
            return anomalies
            
        # Only process tabular data
        if not isinstance(data[0], dict):
            return anomalies
        
        # Detect field-level anomalies
        for field in data[0].keys():
            field_values = [item.get(field) for item in data if field in item]
            field_anomalies = self._detect_field_anomalies(field, field_values, analysis_depth)
            
            if field_anomalies:
                anomalies["field_anomalies"][field] = field_anomalies
        
        # Detect record-level anomalies
        for i, item in enumerate(data):
            record_anomalies = []
            
            # Check for unusually sparse records
            non_null_count = sum(1 for v in item.values() if v is not None and v != "")
            if non_null_count < 0.5 * len(item):
                record_anomalies.append({
                    "type": "sparse_record",
                    "non_null_percentage": non_null_count / len(item)
                })
            
            # Check for records with multiple anomalous fields
            anomalous_fields = []
            for field, field_anomalies in anomalies["field_anomalies"].items():
                if field in item:
                    value = item[field]
                    
                    # Check if this value is in the anomalies
                    for anomaly in field_anomalies:
                        if anomaly["type"] == "outlier" and i in anomaly.get("record_indices", []):
                            anomalous_fields.append(field)
                        elif anomaly["type"] == "unusual_value" and value == anomaly.get("value"):
                            anomalous_fields.append(field)
            
            if len(anomalous_fields) >= 2:
                record_anomalies.append({
                    "type": "multiple_anomalies",
                    "anomalous_fields": anomalous_fields
                })
            
            if record_anomalies:
                anomalies["record_anomalies"].append({
                    "record_index": i,
                    "anomalies": record_anomalies
                })
        
        # Advanced pattern detection for deeper analysis
        if analysis_depth == "advanced":
            # Check for pattern violations
            # This would implement more sophisticated pattern detection
            pass
        
        return anomalies

    def _detect_field_anomalies(self, field: str, values: List[Any], analysis_depth: str) -> List[Dict[str, Any]]:
        """Detect anomalies in a specific field."""
        if not values:
            return []
            
        anomalies = []
        
        # Determine data type
        numeric_values = []
        for v in values:
            if v is not None and v != "":
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    pass
        
        if len(numeric_values) > 0.5 * len(values):
            # Numeric field
            
            # Detect outliers using IQR method
            if len(numeric_values) >= 4:
                sorted_values = sorted(numeric_values)
                n = len(sorted_values)
                
                # Calculate quartiles
                q1 = sorted_values[n//4]
                q3 = sorted_values[(3*n)//4]
                iqr = q3 - q1
                
                # Define bounds
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                
                # Find outliers
                outlier_indices = []
                for i, v in enumerate(values):
                    if v is not None and v != "":
                        try:
                            float_v = float(v)
                            if float_v < lower_bound or float_v > upper_bound:
                                outlier_indices.append(i)
                        except (ValueError, TypeError):
                            pass
                
                if outlier_indices:
                    anomalies.append({
                        "type": "outlier",
                        "method": "iqr",
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "record_indices": outlier_indices,
                        "count": len(outlier_indices)
                    })
        else:
            # Categorical field
            
            # Detect unusual values
            value_counts = {}
            for v in values:
                if v is not None and v != "":
                    str_v = str(v)
                    value_counts[str_v] = value_counts.get(str_v, 0) + 1
            
            # Find rare values
            total_count = sum(value_counts.values())
            rare_values = []
            
            for value, count in value_counts.items():
                percentage = count / total_count
                if percentage < 0.01 and count == 1:  # Less than 1% and only appears once
                    rare_values.append({
                        "value": value,
                        "count": count,
                        "percentage": percentage
                    })
            
            if rare_values:
                anomalies.append({
                    "type": "unusual_value",
                    "rare_values": rare_values,
                    "count": len(rare_values)
                })
        
        return anomalies

    def _prepare_visualization(self, data: Any, visualization_type: str, config: Optional[str]) -> Dict[str, Any]:
        """Prepare data for visualization."""
        if not data:
            return {"error": "No data provided"}
            
        # Parse visualization configuration
        viz_config = {}
        if config:
            try:
                viz_config = json.loads(config)
            except json.JSONDecodeError:
                raise ValueError("Invalid visualization configuration JSON")
        
        # Prepare result structure
        result = {
            "visualization_type": visualization_type,
            "data": None,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "record_count": 0 if not isinstance(data, list) else len(data)
            }
        }
        
        # Handle different visualization types
        if visualization_type == "table":
            # Simple pass-through for tabular data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                result["data"] = data
                result["metadata"]["columns"] = list(data[0].keys())
            else:
                raise ValueError("Table visualization requires tabular data (list of dictionaries)")
        
        elif visualization_type in ["bar", "pie"]:
            # Requires category and value fields
            category_field = viz_config.get("category_field")
            value_field = viz_config.get("value_field")
            
            if not category_field or not value_field:
                raise ValueError(f"{visualization_type} visualization requires category_field and value_field")
                
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise ValueError(f"{visualization_type} visualization requires tabular data")
                
            # Aggregate data
            aggregated = {}
            for item in data:
                if category_field in item and value_field in item:
                    category = str(item[category_field])
                    try:
                        value = float(item[value_field])
                        aggregated[category] = aggregated.get(category, 0) + value
                    except (ValueError, TypeError):
                        pass
            
            # Convert to visualization format
            viz_data = [{"category": k, "value": v} for k, v in aggregated.items()]
            result["data"] = viz_data
            result["metadata"]["categories"] = len(viz_data)
        
        elif visualization_type == "line":
            # Requires x and y fields
            x_field = viz_config.get("x_field")
            y_field = viz_config.get("y_field")
            series_field = viz_config.get("series_field")  # Optional
            
            if not x_field or not y_field:
                raise ValueError("Line visualization requires x_field and y_field")
                
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise ValueError("Line visualization requires tabular data")
                
            # Prepare data
            if series_field:
                # Multiple series
                series_data = {}
                
                for item in data:
                    if x_field in item and y_field in item and series_field in item:
                        x = item[x_field]
                        try:
                            y = float(item[y_field])
                            series = str(item[series_field])
                            
                            if series not in series_data:
                                series_data[series] = []
                            
                            series_data[series].append({"x": x, "y": y})
                        except (ValueError, TypeError):
                            pass
                
                # Convert to visualization format
                viz_data = [{"series": k, "data": sorted(v, key=lambda point: point["x"])} for k, v in series_data.items()]
                result["data"] = viz_data
                result["metadata"]["series_count"] = len(viz_data)
            else:
                # Single series
                viz_data = []
                
                for item in data:
                    if x_field in item and y_field in item:
                        x = item[x_field]
                        try:
                            y = float(item[y_field])
                            viz_data.append({"x": x, "y": y})
                        except (ValueError, TypeError):
                            pass
                
                # Sort by x value
                viz_data.sort(key=lambda point: point["x"])
                result["data"] = viz_data
                result["metadata"]["point_count"] = len(viz_data)
        
        elif visualization_type == "scatter":
            # Requires x and y fields
            x_field = viz_config.get("x_field")
            y_field = viz_config.get("y_field")
            color_field = viz_config.get("color_field")  # Optional
            size_field = viz_config.get("size_field")    # Optional
            
            if not x_field or not y_field:
                raise ValueError("Scatter visualization requires x_field and y_field")
                
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise ValueError("Scatter visualization requires tabular data")
                
            # Prepare data
            viz_data = []
            
            for item in data:
                if x_field in item and y_field in item:
                    try:
                        point = {
                            "x": float(item[x_field]),
                            "y": float(item[y_field])
                        }
                        
                        # Add color if specified
                        if color_field and color_field in item:
                            point["color"] = item[color_field]
                        
                        # Add size if specified
                        if size_field and size_field in item:
                            try:
                                point["size"] = float(item[size_field])
                            except (ValueError, TypeError):
                                pass
                        
                        viz_data.append(point)
                    except (ValueError, TypeError):
                        pass
            
            result["data"] = viz_data
            result["metadata"]["point_count"] = len(viz_data)
        
        elif visualization_type == "heatmap":
            # Requires x, y, and value fields
            x_field = viz_config.get("x_field")
            y_field = viz_config.get("y_field")
            value_field = viz_config.get("value_field")
            
            if not x_field or not y_field or not value_field:
                raise ValueError("Heatmap visualization requires x_field, y_field, and value_field")
                
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise ValueError("Heatmap visualization requires tabular data")
                
            # Prepare data
            viz_data = []
            
            for item in data:
                if x_field in item and y_field in item and value_field in item:
                    try:
                        viz_data.append({
                            "x": item[x_field],
                            "y": item[y_field],
                            "value": float(item[value_field])
                        })
                    except (ValueError, TypeError):
                        pass
            
            result["data"] = viz_data
            result["metadata"]["cell_count"] = len(viz_data)
        
        elif visualization_type == "histogram":
            # Requires value field
            value_field = viz_config.get("value_field")
            bins = viz_config.get("bins", 10)
            
            if not value_field:
                raise ValueError("Histogram visualization requires value_field")
                
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise ValueError("Histogram visualization requires tabular data")
                
            # Extract values
            values = []
            for item in data:
                if value_field in item:
                    try:
                        values.append(float(item[value_field]))
                    except (ValueError, TypeError):
                        pass
            
            if not values:
                raise ValueError(f"No numeric values found for field: {value_field}")
                
            # Calculate histogram
            min_val = min(values)
            max_val = max(values)
            bin_width = (max_val - min_val) / bins if max_val > min_val else 1
            
            histogram = [0] * bins
            for value in values:
                bin_index = min(bins - 1, max(0, int((value - min_val) / bin_width)))
                histogram[bin_index] += 1
            
            # Create bin labels
            bin_labels = []
            for i in range(bins):
                bin_min = min_val + i * bin_width
                bin_max = min_val + (i + 1) * bin_width
                bin_labels.append(f"{bin_min:.2f}-{bin_max:.2f}")
            
            # Format for visualization
            viz_data = [{"bin": bin_labels[i], "count": histogram[i]} for i in range(bins)]
            result["data"] = viz_data
            result["metadata"]["bin_count"] = bins
            result["metadata"]["value_count"] = len(values)
        
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")
        
        return result

    def _export_data(self, data: Any, output_path: str, output_format: str) -> Dict[str, Any]:
        """Export data to a file in the specified format."""
        if not data:
            return {"error": "No data to export"}
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "output_path": output_path,
            "output_format": output_format,
            "success": False,
            "record_count": 0 if not isinstance(data, list) else len(data)
        }
        
        try:
            if output_format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                result["success"] = True
            
            elif output_format == "csv":
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        fieldnames = data[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
                    result["success"] = True
                else:
                    result["error"] = "CSV export requires tabular data (list of dictionaries)"
            
            elif output_format == "text":
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        f.write(str(data))
                result["success"] = True
            
            elif output_format == "markdown":
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        # Create markdown table
                        headers = list(data[0].keys())
                        f.write("| " + " | ".join(headers) + " |\n")
                        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                        
                        for item in data:
                            row = []
                            for header in headers:
                                value = item.get(header, "")
                                row.append(str(value).replace("|", "\\|"))
                            f.write("| " + " | ".join(row) + " |\n")
                    else:
                        # Just write as code block
                        f.write("```\n")
                        f.write(str(data))
                        f.write("\n```\n")
                result["success"] = True
            
            elif output_format == "html":
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        # Create HTML table
                        f.write("<table>\n<thead>\n<tr>\n")
                        headers = list(data[0].keys())
                        for header in headers:
                            f.write(f"<th>{header}</th>\n")
                        f.write("</tr>\n</thead>\n<tbody>\n")
                        
                        for item in data:
                            f.write("<tr>\n")
                            for header in headers:
                                value = item.get(header, "")
                                f.write(f"<td>{value}</td>\n")
                            f.write("</tr>\n")
                        
                        f.write("</tbody>\n</table>")
                    else:
                        # Just write as pre
                        f.write("<pre>\n")
                        f.write(str(data))
                        f.write("\n</pre>\n")
                result["success"] = True
            
            else:
                result["error"] = f"Unsupported output format: {output_format}"
        
        except Exception as e:
            result["error"] = str(e)
        
        return result

    def _generate_report(self, data: Any, analysis_depth: str, output_format: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Generate a comprehensive report from the data."""
        # First analyze the data
        analysis = self._analyze_data(data, analysis_depth)
        
        # Generate statistics
        statistics = self._generate_statistics(data, analysis_depth)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(data, analysis_depth)
        
        # Combine into a report
        report = {
            "title": "Data Analysis Report",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "data_type": analysis["data_type"],
                "record_count": statistics["record_count"],
                "field_count": statistics.get("field_count", 0),
                "completeness": statistics.get("summary_statistics", {}).get("completeness", 0),
                "quality_issues": len(analysis.get("quality_issues", [])),
                "anomalies_detected": sum(len(field_anomalies) for field_anomalies in anomalies.get("field_anomalies", {}).values())
            },
            "analysis": analysis,
            "statistics": statistics,
            "anomalies": anomalies,
            "recommendations": analysis.get("recommendations", [])
        }
        
        # Format and export the report if output_path is provided
        if output_path:
            self._export_report(report, output_path, output_format)
            report["output_path"] = output_path
        
        return report

    def _export_report(self, report: Dict[str, Any], output_path: str, output_format: str) -> None:
        """Export a report to a file in the specified format."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        elif output_format == "markdown":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {report['title']}\n\n")
                f.write(f"Generated: {report['timestamp']}\n\n")
                
                f.write("## Summary\n\n")
                for key, value in report["summary"].items():
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                
                f.write("\n## Key Findings\n\n")
                
                # Quality issues
                if report["analysis"].get("quality_issues"):
                    f.write("### Quality Issues\n\n")
                    for issue in report["analysis"]["quality_issues"]:
                        f.write(f"- {issue['type']} in field '{issue.get('field', 'unknown')}': {issue.get('count', 0)} instances\n")
                
                # Anomalies
                if report["anomalies"].get("field_anomalies"):
                    f.write("\n### Anomalies\n\n")
                    for field, anomalies in report["anomalies"]["field_anomalies"].items():
                        f.write(f"#### Field: {field}\n\n")
                        for anomaly in anomalies:
                            f.write(f"- {anomaly['type']}: {anomaly.get('count', 0)} instances\n")
                
                # Recommendations
                if report.get("recommendations"):
                    f.write("\n## Recommendations\n\n")
                    for i, rec in enumerate(report["recommendations"], 1):
                        f.write(f"{i}. {rec}\n")
                
                # Field Statistics
                if report["statistics"].get("field_statistics"):
                    f.write("\n## Field Statistics\n\n")
                    for field, stats in report["statistics"]["field_statistics"].items():
                        f.write(f"### {field}\n\n")
                        f.write(f"- **Type**: {stats.get('data_type', 'unknown')}\n")
                        f.write(f"- **Count**: {stats.get('count', 0)}\n")
                        f.write(f"- **Null Count**: {stats.get('null_count', 0)} ({stats.get('null_percentage', 0):.1%})\n")
                        f.write(f"- **Unique Count**: {stats.get('unique_count', 0)}\n")
                        
                        if stats.get('data_type') == 'numeric':
                            f.write(f"- **Min**: {stats.get('min', 0)}\n")
                            f.write(f"- **Max**: {stats.get('max', 0)}\n")
                            f.write(f"- **Mean**: {stats.get('mean', 0)}\n")
                            f.write(f"- **Median**: {stats.get('median', 0)}\n")
                            if 'std_dev' in stats:
                                f.write(f"- **Std Dev**: {stats['std_dev']}\n")
                        
                        if stats.get('data_type') == 'categorical' and 'top_values' in stats:
                            f.write("\n**Top Values**:\n\n")
                            for value, count in stats['top_values'].items():
                                f.write(f"- {value}: {count}\n")
                        
                        f.write("\n")
        
        elif output_format == "html":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{report['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .issues {{ background-color: #fff0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .recommendations {{ background-color: #f0fff0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{report['title']}</h1>
    <p>Generated: {report['timestamp']}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <ul>
""")
                
                for key, value in report["summary"].items():
                    f.write(f"            <li><strong>{key.replace('_', ' ').title()}</strong>: {value}</li>\n")
                
                f.write("""        </ul>
    </div>
    
    <h2>Key Findings</h2>
""")
                
                # Quality issues
                if report["analysis"].get("quality_issues"):
                    f.write("""    <div class="issues">
        <h3>Quality Issues</h3>
        <ul>
""")
                    for issue in report["analysis"]["quality_issues"]:
                        f.write(f"            <li>{issue['type']} in field '{issue.get('field', 'unknown')}': {issue.get('count', 0)} instances</li>\n")
                    f.write("""        </ul>
    </div>
""")
                
                # Anomalies
                if report["anomalies"].get("field_anomalies"):
                    f.write("""    <div class="issues">
        <h3>Anomalies</h3>
""")
                    for field, anomalies in report["anomalies"]["field_anomalies"].items():
                        f.write(f"        <h4>Field: {field}</h4>\n        <ul>\n")
                        for anomaly in anomalies:
                            f.write(f"            <li>{anomaly['type']}: {anomaly.get('count', 0)} instances</li>\n")
                        f.write("        </ul>\n")
                    f.write("    </div>\n")
                
                # Recommendations
                if report.get("recommendations"):
                    f.write("""    <div class="recommendations">
        <h3>Recommendations</h3>
        <ol>
""")
                    for rec in report["recommendations"]:
                        f.write(f"            <li>{rec}</li>\n")
                    f.write("""        </ol>
    </div>
""")
                
                # Field Statistics
                if report["statistics"].get("field_statistics"):
                    f.write("    <h2>Field Statistics</h2>\n")
                    
                    for field, stats in report["statistics"]["field_statistics"].items():
                        f.write(f"    <h3>{field}</h3>\n")
                        f.write("    <table>\n        <tr><th>Metric</th><th>Value</th></tr>\n")
                        f.write(f"        <tr><td>Type</td><td>{stats.get('data_type', 'unknown')}</td></tr>\n")
                        f.write(f"        <tr><td>Count</td><td>{stats.get('count', 0)}</td></tr>\n")
                        f.write(f"        <tr><td>Null Count</td><td>{stats.get('null_count', 0)} ({stats.get('null_percentage', 0):.1%})</td></tr>\n")
                        f.write(f"        <tr><td>Unique Count</td><td>{stats.get('unique_count', 0)}</td></tr>\n")
                        
                        if stats.get('data_type') == 'numeric':
                            f.write(f"        <tr><td>Min</td><td>{stats.get('min', 0)}</td></tr>\n")
                            f.write(f"        <tr><td>Max</td><td>{stats.get('max', 0)}</td></tr>\n")
                            f.write(f"        <tr><td>Mean</td><td>{stats.get('mean', 0)}</td></tr>\n")
                            f.write(f"        <tr><td>Median</td><td>{stats.get('median', 0)}</td></tr>\n")
                            if 'std_dev' in stats:
                                f.write(f"        <tr><td>Std Dev</td><td>{stats['std_dev']}</td></tr>\n")
                        
                        f.write("    </table>\n")
                        
                        if stats.get('data_type') == 'categorical' and 'top_values' in stats:
                            f.write("    <h4>Top Values</h4>\n")
                            f.write("    <table>\n        <tr><th>Value</th><th>Count</th></tr>\n")
                            for value, count in stats['top_values'].items():
                                f.write(f"        <tr><td>{value}</td><td>{count}</td></tr>\n")
                            f.write("    </table>\n")
                
                f.write("""</body>
</html>
""")
        
        elif output_format == "text":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{report['title']}\n")
                f.write(f"Generated: {report['timestamp']}\n\n")
                
                f.write("SUMMARY\n")
                f.write("=======\n")
                for key, value in report["summary"].items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\nKEY FINDINGS\n")
                f.write("============\n")
                
                # Quality issues
                if report["analysis"].get("quality_issues"):
                    f.write("\nQuality Issues:\n")
                    for issue in report["analysis"]["quality_issues"]:
                        f.write(f"- {issue['type']} in field '{issue.get('field', 'unknown')}': {issue.get('count', 0)} instances\n")
                
                # Anomalies
                if report["anomalies"].get("field_anomalies"):
                    f.write("\nAnomalies:\n")
                    for field, anomalies in report["anomalies"]["field_anomalies"].items():
                        f.write(f"\nField: {field}\n")
                        for anomaly in anomalies:
                            f.write(f"- {anomaly['type']}: {anomaly.get('count', 0)} instances\n")
                
                # Recommendations
                if report.get("recommendations"):
                    f.write("\nRECOMMENDATIONS\n")
                    f.write("===============\n")
                    for i, rec in enumerate(report["recommendations"], 1):
                        f.write(f"{i}. {rec}\n")
                
                # Field Statistics
                if report["statistics"].get("field_statistics"):
                    f.write("\nFIELD STATISTICS\n")
                    f.write("================\n")
                    
                    for field, stats in report["statistics"]["field_statistics"].items():
                        f.write(f"\n{field}\n")
                        f.write(f"{'-' * len(field)}\n")
                        f.write(f"Type: {stats.get('data_type', 'unknown')}\n")
                        f.write(f"Count: {stats.get('count', 0)}\n")
                        f.write(f"Null Count: {stats.get('null_count', 0)} ({stats.get('null_percentage', 0):.1%})\n")
                        f.write(f"Unique Count: {stats.get('unique_count', 0)}\n")
                        
                        if stats.get('data_type') == 'numeric':
                            f.write(f"Min: {stats.get('min', 0)}\n")
                            f.write(f"Max: {stats.get('max', 0)}\n")
                            f.write(f"Mean: {stats.get('mean', 0)}\n")
                            f.write(f"Median: {stats.get('median', 0)}\n")
                            if 'std_dev' in stats:
                                f.write(f"Std Dev: {stats['std_dev']}\n")
                        
                        if stats.get('data_type') == 'categorical' and 'top_values' in stats:
                            f.write("\nTop Values:\n")
                            for value, count in stats['top_values'].items():
                                f.write(f"- {value}: {count}\n")
        
        else:
            raise ValueError(f"Unsupported output format for report: {output_format}")

    def _save_data(self, data: Any, output_path: str, output_format: str) -> None:
        """Save data to a file in the specified format."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif output_format == "csv":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
            else:
                raise ValueError("CSV export requires tabular data (list of dictionaries)")
        
        elif output_format == "text":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _format_result(self, result: Any, action: str, output_format: str) -> str:
        """Format the result for output."""
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
            
        if action == "analyze_data":
            return self._format_analysis_result(result)
        elif action == "generate_statistics":
            return self._format_statistics_result(result)
        elif action == "detect_anomalies":
            return self._format_anomalies_result(result)
        elif action == "prepare_visualization":
            return self._format_visualization_result(result)
        elif action == "export_data" or action == "generate_report":
            if isinstance(result, dict) and "output_path" in result:
                return f"Data successfully exported to {result['output_path']} in {result.get('output_format', output_format)} format."
            return "Data export completed."
        else:
            # Default formatting
            if output_format == "json":
                return json.dumps(result, indent=2)
            else:
                return str(result)

    def _format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format analysis result for output."""
        output = ["Data Analysis Results"]
        output.append("=" * 50)
        
        # Summary
        output.append("\nSummary:")
        if "summary" in result:
            for key, value in result["summary"].items():
                output.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Structure
        if "structure" in result:
            output.append("\nStructure:")
            if "fields" in result["structure"]:
                output.append(f"  Fields: {', '.join(result['structure']['fields'])}")
            if "sample" in result["structure"]:
                output.append("  Sample Record:")
                for k, v in result["structure"]["sample"].items():
                    output.append(f"    {k}: {v}")
        
        # Quality issues
        if "quality_issues" in result and result["quality_issues"]:
            output.append("\nQuality Issues:")
            for issue in result["quality_issues"]:
                output.append(f"  - {issue['type']} in field '{issue.get('field', 'unknown')}': {issue.get('count', 0)} instances (Severity: {issue.get('severity', 'medium')})")
        
        # Recommendations
        if "recommendations" in result and result["recommendations"]:
            output.append("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                output.append(f"  {i}. {rec}")
        
        return "\n".join(output)

    def _format_statistics_result(self, result: Dict[str, Any]) -> str:
        """Format statistics result for output."""
        output = ["Data Statistics Results"]
        output.append("=" * 50)
        
        output.append(f"\nRecord Count: {result.get('record_count', 0)}")
        
        # Field statistics
        if "field_statistics" in result and result["field_statistics"]:
            output.append("\nField Statistics:")
            
            for field, stats in result["field_statistics"].items():
                output.append(f"\n  {field}:")
                output.append(f"    Type: {stats.get('data_type', 'unknown')}")
                output.append(f"    Count: {stats.get('count', 0)}")
                output.append(f"    Null: {stats.get('null_count', 0)} ({stats.get('null_percentage', 0):.1%})")
                output.append(f"    Unique: {stats.get('unique_count', 0)}")
                
                if stats.get('data_type') == 'numeric':
                    output.append(f"    Min: {stats.get('min', 0)}")
                    output.append(f"    Max: {stats.get('max', 0)}")
                    output.append(f"    Mean: {stats.get('mean', 0)}")
                    output.append(f"    Median: {stats.get('median', 0)}")
                    if 'std_dev' in stats:
                        output.append(f"    Std Dev: {stats['std_dev']}")
                
                if stats.get('data_type') == 'categorical' and 'top_values' in stats:
                    output.append("    Top Values:")
                    for value, count in stats['top_values'].items():
                        output.append(f"      - {value}: {count}")
        
        # Summary statistics
        if "summary_statistics" in result and result["summary_statistics"]:
            output.append("\nSummary Statistics:")
            for key, value in result["summary_statistics"].items():
                if key != "field_count" and key != "record_count":  # Already shown
                    output.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(output)

    def _format_anomalies_result(self, result: Dict[str, Any]) -> str:
        """Format anomalies result for output."""
        output = ["Data Anomalies Results"]
        output.append("=" * 50)
        
        # Field anomalies
        if "field_anomalies" in result and result["field_anomalies"]:
            output.append("\nField Anomalies:")
            
            for field, anomalies in result["field_anomalies"].items():
                output.append(f"\n  {field}:")
                for anomaly in anomalies:
                    if anomaly["type"] == "outlier":
                        output.append(f"    - Outliers: {anomaly.get('count', 0)} values outside range [{anomaly.get('lower_bound', 'N/A')}, {anomaly.get('upper_bound', 'N/A')}]")
                    elif anomaly["type"] == "unusual_value":
                        output.append(f"    - Unusual Values: {anomaly.get('count', 0)} rare values detected")
                        if "rare_values" in anomaly and len(anomaly["rare_values"]) <= 5:
                            for rare in anomaly["rare_values"]:
                                output.append(f"      - '{rare['value']}': {rare['count']} occurrences ({rare['percentage']:.1%})")
                    else:
                        output.append(f"    - {anomaly['type']}: {anomaly.get('count', 0)} instances")
        
        # Record anomalies
        if "record_anomalies" in result and result["record_anomalies"]:
            output.append("\nRecord Anomalies:")
            output.append(f"  {len(result['record_anomalies'])} records with anomalies detected")
            
            # Show details for up to 5 records
            for i, record in enumerate(result["record_anomalies"][:5]):
                output.append(f"\n  Record #{record['record_index']}:")
                for anomaly in record["anomalies"]:
                    if anomaly["type"] == "sparse_record":
                        output.append(f"    - Sparse Record: Only {anomaly['non_null_percentage']:.1%} of fields have values")
                    elif anomaly["type"] == "multiple_anomalies":
                        output.append(f"    - Multiple Anomalies: Issues in fields {', '.join(anomaly['anomalous_fields'])}")
                    else:
                        output.append(f"    - {anomaly['type']}")
            
            if len(result["record_anomalies"]) > 5:
                output.append(f"\n  ... and {len(result['record_anomalies']) - 5} more anomalous records")
        
        # Pattern violations
        if "pattern_violations" in result and result["pattern_violations"]:
            output.append("\nPattern Violations:")
            for violation in result["pattern_violations"]:
                output.append(f"  - {violation['type']}: {violation.get('description', 'No description')}")
        
        return "\n".join(output)

    def _format_visualization_result(self, result: Dict[str, Any]) -> str:
        """Format visualization preparation result for output."""
        output = [f"Data Prepared for {result['visualization_type'].title()} Visualization"]
        output.append("=" * 50)
        
        # Metadata
        if "metadata" in result:
            output.append("\nMetadata:")
            for key, value in result["metadata"].items():
                output.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Data preview
        if "data" in result and result["data"]:
            output.append("\nData Preview:")
            
            if isinstance(result["data"], list):
                # Show first 5 items
                for i, item in enumerate(result["data"][:5]):
                    output.append(f"\n  Item {i+1}:")
                    if isinstance(item, dict):
                        for k, v in item.items():
                            output.append(f"    {k}: {v}")
                    else:
                        output.append(f"    {item}")
                
                if len(result["data"]) > 5:
                    output.append(f"\n  ... and {len(result['data']) - 5} more items")
        
        return "\n".join(output)

    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get the data processing history."""
        return self.processing_history

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache.clear()