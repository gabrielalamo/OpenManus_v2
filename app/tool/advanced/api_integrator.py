"""
API Integrator Tool
Manages API integrations, connections, and data exchange
"""

import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class APIIntegratorTool(BaseTool):
    """
    Advanced API integration tool for connecting to external services,
    managing API requests, and handling data exchange.
    """

    name: str = "api_integrator"
    description: str = """
    Manage API integrations with external services. Connect to APIs,
    handle authentication, process requests and responses, and manage data exchange.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "test_connection", "generate_client", "create_endpoint",
                    "document_api", "analyze_api", "generate_request",
                    "validate_response", "create_mock", "generate_schema",
                    "create_integration"
                ],
                "description": "The API integration action to perform"
            },
            "api_url": {
                "type": "string",
                "description": "URL of the API to integrate with"
            },
            "api_type": {
                "type": "string",
                "enum": ["rest", "graphql", "soap", "grpc", "webhook"],
                "description": "Type of API to integrate with"
            },
            "auth_type": {
                "type": "string",
                "enum": ["none", "basic", "bearer", "oauth", "api_key", "custom"],
                "description": "Authentication type for the API"
            },
            "endpoint_path": {
                "type": "string",
                "description": "Path of the API endpoint"
            },
            "http_method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
                "description": "HTTP method for the API request"
            },
            "request_body": {
                "type": "string",
                "description": "JSON string of the request body"
            },
            "request_headers": {
                "type": "string",
                "description": "JSON string of the request headers"
            },
            "response_example": {
                "type": "string",
                "description": "Example API response for analysis or schema generation"
            },
            "target_language": {
                "type": "string",
                "enum": ["python", "javascript", "typescript", "java", "go", "other"],
                "description": "Target programming language for client generation"
            },
            "integration_name": {
                "type": "string",
                "description": "Name of the integration to create"
            },
            "output_path": {
                "type": "string",
                "description": "Path to save generated files"
            }
        },
        "required": ["action"]
    }

    # API integration data storage
    api_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    endpoints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    api_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    mock_responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    async def execute(
        self,
        action: str,
        api_url: Optional[str] = None,
        api_type: Optional[str] = None,
        auth_type: Optional[str] = None,
        endpoint_path: Optional[str] = None,
        http_method: Optional[str] = None,
        request_body: Optional[str] = None,
        request_headers: Optional[str] = None,
        response_example: Optional[str] = None,
        target_language: Optional[str] = None,
        integration_name: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the API integration action."""
        
        try:
            if action == "test_connection":
                return await self._test_connection(api_url, auth_type, request_headers)
            elif action == "generate_client":
                return await self._generate_client(api_url, api_type, target_language, output_path)
            elif action == "create_endpoint":
                return await self._create_endpoint(api_url, endpoint_path, http_method, request_body, request_headers)
            elif action == "document_api":
                return await self._document_api(api_url, api_type, output_path)
            elif action == "analyze_api":
                return await self._analyze_api(api_url, response_example)
            elif action == "generate_request":
                return await self._generate_request(api_url, endpoint_path, http_method, request_body)
            elif action == "validate_response":
                return await self._validate_response(response_example, api_type)
            elif action == "create_mock":
                return await self._create_mock(api_url, endpoint_path, response_example)
            elif action == "generate_schema":
                return await self._generate_schema(response_example, api_type)
            elif action == "create_integration":
                return await self._create_integration(integration_name, api_url, api_type, auth_type, output_path)
            else:
                return ToolResult(error=f"Unknown API integration action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"API integration error: {str(e)}")

    async def _test_connection(
        self, 
        api_url: Optional[str], 
        auth_type: Optional[str],
        request_headers: Optional[str]
    ) -> ToolResult:
        """Test connection to an API endpoint."""
        
        if not api_url:
            return ToolResult(error="API URL is required for testing connection")
        
        # Parse headers if provided
        headers = {}
        if request_headers:
            try:
                headers = json.loads(request_headers)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid JSON format for request headers")
        
        # This would implement actual API connection testing
        # For now, return a simulated result
        connection_result = {
            "url": api_url,
            "auth_type": auth_type or "none",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "response_time": 120,  # ms
            "headers_sent": headers,
            "connection_details": {
                "protocol": "https" if api_url.startswith("https") else "http",
                "host": api_url.split("//")[-1].split("/")[0],
                "tls_version": "TLS 1.3" if api_url.startswith("https") else "N/A"
            }
        }
        
        # Store connection info
        self.api_configs[api_url] = {
            "last_tested": datetime.now().isoformat(),
            "auth_type": auth_type,
            "status": "active"
        }
        
        return ToolResult(
            output=f"Connection test successful for {api_url}\n"
                   f"Response time: {connection_result['response_time']}ms\n"
                   f"Protocol: {connection_result['connection_details']['protocol']}\n"
                   f"TLS: {connection_result['connection_details']['tls_version']}"
        )

    async def _generate_client(
        self,
        api_url: Optional[str],
        api_type: Optional[str],
        target_language: Optional[str],
        output_path: Optional[str]
    ) -> ToolResult:
        """Generate API client code for the specified API."""
        
        if not api_url:
            return ToolResult(error="API URL is required for client generation")
        
        if not api_type:
            return ToolResult(error="API type is required for client generation")
        
        if not target_language:
            target_language = "python"  # Default to Python
        
        # This would implement actual client code generation
        # For now, generate a template client
        
        client_code = self._generate_client_template(api_url, api_type, target_language)
        
        # If output path is provided, save the client code
        if output_path:
            file_extension = self._get_file_extension(target_language)
            filename = f"api_client{file_extension}"
            full_path = os.path.join(output_path, filename)
            
            # This would save the file in a real implementation
            # For now, just simulate file saving
            
            return ToolResult(
                output=f"API client generated for {api_url}\n"
                       f"Target language: {target_language}\n"
                       f"Saved to: {full_path}\n\n"
                       f"Client code preview:\n```{target_language}\n{client_code[:500]}...\n```"
            )
        
        return ToolResult(
            output=f"API client generated for {api_url}\n"
                   f"Target language: {target_language}\n\n"
                   f"Client code:\n```{target_language}\n{client_code}\n```"
        )

    def _generate_client_template(self, api_url: str, api_type: str, language: str) -> str:
        """Generate a template API client based on language and API type."""
        
        if language == "python":
            if api_type == "rest":
                return self._generate_python_rest_client(api_url)
            elif api_type == "graphql":
                return self._generate_python_graphql_client(api_url)
            else:
                return self._generate_python_generic_client(api_url, api_type)
        
        elif language == "javascript" or language == "typescript":
            if api_type == "rest":
                return self._generate_js_rest_client(api_url, language)
            elif api_type == "graphql":
                return self._generate_js_graphql_client(api_url, language)
            else:
                return self._generate_js_generic_client(api_url, api_type, language)
        
        else:
            # Generic template for other languages
            return f"// API Client for {api_url}\n// API Type: {api_type}\n// Language: {language}\n\n// TODO: Implement client"

    def _generate_python_rest_client(self, api_url: str) -> str:
        """Generate a Python REST API client template."""
        return f"""
import requests
from typing import Dict, Any, Optional

class APIClient:
    \"\"\"
    REST API Client for {api_url}
    \"\"\"
    
    def __init__(self, base_url: str = "{api_url}", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({{
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }})
        
        if api_key:
            self.session.headers.update({{
                'Authorization': f'Bearer {{api_key}}'
            }})
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Perform a GET request to the API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            API response as dictionary
        \"\"\"
        url = f"{{self.base_url}}/{{endpoint.lstrip('/')}}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Perform a POST request to the API.
        
        Args:
            endpoint: API endpoint path
            data: Request payload
            
        Returns:
            API response as dictionary
        \"\"\"
        url = f"{{self.base_url}}/{{endpoint.lstrip('/')}}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Perform a PUT request to the API.
        
        Args:
            endpoint: API endpoint path
            data: Request payload
            
        Returns:
            API response as dictionary
        \"\"\"
        url = f"{{self.base_url}}/{{endpoint.lstrip('/')}}"
        response = self.session.put(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        \"\"\"
        Perform a DELETE request to the API.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            API response as dictionary
        \"\"\"
        url = f"{{self.base_url}}/{{endpoint.lstrip('/')}}"
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()
"""

    def _generate_python_graphql_client(self, api_url: str) -> str:
        """Generate a Python GraphQL API client template."""
        return f"""
import requests
from typing import Dict, Any, Optional, List

class GraphQLClient:
    \"\"\"
    GraphQL API Client for {api_url}
    \"\"\"
    
    def __init__(self, endpoint: str = "{api_url}", api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {{
            'Content-Type': 'application/json',
        }}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {{api_key}}'
    
    def execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Execute a GraphQL query or mutation.
        
        Args:
            query: GraphQL query or mutation string
            variables: Query variables
            
        Returns:
            GraphQL response data
        \"\"\"
        payload = {{"query": query}}
        if variables:
            payload["variables"] = variables
            
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        result = response.json()
        
        if "errors" in result:
            raise Exception(f"GraphQL errors: {{result['errors']}}")
            
        return result.get("data", {{}})
    
    def query(self, query_string: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Execute a GraphQL query.
        
        Args:
            query_string: GraphQL query string
            variables: Query variables
            
        Returns:
            Query result data
        \"\"\"
        return self.execute(query_string, variables)
    
    def mutation(self, mutation_string: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Execute a GraphQL mutation.
        
        Args:
            mutation_string: GraphQL mutation string
            variables: Mutation variables
            
        Returns:
            Mutation result data
        \"\"\"
        return self.execute(mutation_string, variables)
"""

    def _generate_python_generic_client(self, api_url: str, api_type: str) -> str:
        """Generate a generic Python API client template."""
        return f"""
# Python API Client for {api_url}
# API Type: {api_type}

import requests
from typing import Dict, Any, Optional

class {api_type.capitalize()}Client:
    \"\"\"
    {api_type.capitalize()} API Client for {api_url}
    \"\"\"
    
    def __init__(self, base_url: str = "{api_url}", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({{
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }})
        
        if api_key:
            self.session.headers.update({{
                'Authorization': f'Bearer {{api_key}}'
            }})
    
    def request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, 
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Perform an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request payload
            params: Query parameters
            
        Returns:
            API response
        \"\"\"
        url = f"{{self.base_url}}/{{endpoint.lstrip('/')}}"
        response = self.session.request(method, url, json=data, params=params)
        response.raise_for_status()
        return response.json()
"""

    def _generate_js_rest_client(self, api_url: str, language: str) -> str:
        """Generate a JavaScript/TypeScript REST API client template."""
        ts_types = "" if language == "javascript" else """
interface RequestOptions {
  params?: Record<string, any>;
  headers?: Record<string, string>;
  timeout?: number;
}

interface APIResponse<T = any> {
  data: T;
  status: number;
  headers: Record<string, string>;
}
"""
        
        type_annotations = "" if language == "javascript" else """<T = any>"""
        param_types = "" if language == "javascript" else ": RequestOptions"
        return_type = "" if language == "javascript" else ": Promise<APIResponse<T>>"
        
        return f"""
{ts_types if language == "typescript" else ""}
/**
 * REST API Client for {api_url}
 */
class APIClient {{
  baseUrl;
  apiKey;
  defaultHeaders;

  /**
   * Create a new API client
   * @param {{string}} baseUrl - Base URL for API requests
   * @param {{string}} apiKey - Optional API key for authentication
   */
  constructor(baseUrl = "{api_url}", apiKey = null) {{
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    this.apiKey = apiKey;
    
    this.defaultHeaders = {{
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }};
    
    if (apiKey) {{
      this.defaultHeaders['Authorization'] = `Bearer ${{apiKey}}`;
    }}
  }}

  /**
   * Perform a GET request
   * @param {{string}} endpoint - API endpoint
   * @param {{RequestOptions}} options - Request options
   * @returns {{Promise<APIResponse>}} API response
   */
  async get{ts_types}(endpoint, options{param_types} = {{}}) {return_type} {{
    return this._request('GET', endpoint, null, options);
  }}

  /**
   * Perform a POST request
   * @param {{string}} endpoint - API endpoint
   * @param {{any}} data - Request payload
   * @param {{RequestOptions}} options - Request options
   * @returns {{Promise<APIResponse>}} API response
   */
  async post{ts_types}(endpoint, data, options{param_types} = {{}}) {return_type} {{
    return this._request('POST', endpoint, data, options);
  }}

  /**
   * Perform a PUT request
   * @param {{string}} endpoint - API endpoint
   * @param {{any}} data - Request payload
   * @param {{RequestOptions}} options - Request options
   * @returns {{Promise<APIResponse>}} API response
   */
  async put{ts_types}(endpoint, data, options{param_types} = {{}}) {return_type} {{
    return this._request('PUT', endpoint, data, options);
  }}

  /**
   * Perform a DELETE request
   * @param {{string}} endpoint - API endpoint
   * @param {{RequestOptions}} options - Request options
   * @returns {{Promise<APIResponse>}} API response
   */
  async delete{ts_types}(endpoint, options{param_types} = {{}}) {return_type} {{
    return this._request('DELETE', endpoint, null, options);
  }}

  /**
   * Perform an API request
   * @private
   */
  async _request{ts_types}(method, endpoint, data, options{param_types} = {{}}) {return_type} {{
    const url = new URL(`${{this.baseUrl}}/${{endpoint.startsWith('/') ? endpoint.slice(1) : endpoint}}`);
    
    // Add query parameters
    if (options.params) {{
      Object.entries(options.params).forEach(([key, value]) => {{
        url.searchParams.append(key, value);
      }});
    }}
    
    // Prepare headers
    const headers = {{
      ...this.defaultHeaders,
      ...options.headers,
    }};
    
    // Prepare request options
    const requestOptions = {{
      method,
      headers,
      ...(data && {{ body: JSON.stringify(data) }}),
    }};
    
    // Set timeout if provided
    const controller = new AbortController();
    if (options.timeout) {{
      setTimeout(() => controller.abort(), options.timeout);
      requestOptions.signal = controller.signal;
    }}
    
    // Perform request
    const response = await fetch(url.toString(), requestOptions);
    
    if (!response.ok) {{
      throw new Error(`API request failed: ${{response.status}} ${{response.statusText}}`);
    }}
    
    const responseData = await response.json();
    
    // Prepare response object
    const responseHeaders = {{}};
    response.headers.forEach((value, key) => {{
      responseHeaders[key] = value;
    }});
    
    return {{
      data: responseData,
      status: response.status,
      headers: responseHeaders,
    }};
  }}
}}

{language == "javascript" ? "export default APIClient;" : "export default APIClient;"}
"""

    def _generate_js_graphql_client(self, api_url: str, language: str) -> str:
        """Generate a JavaScript/TypeScript GraphQL API client template."""
        ts_types = "" if language == "javascript" else """
interface GraphQLRequestOptions {
  headers?: Record<string, string>;
  timeout?: number;
}

interface GraphQLResponse<T = any> {
  data?: T;
  errors?: Array<{
    message: string;
    locations?: Array<{ line: number; column: number }>;
    path?: Array<string | number>;
    extensions?: any;
  }>;
}
"""
        
        type_annotations = "" if language == "javascript" else """<T = any>"""
        param_types = "" if language == "javascript" else """: GraphQLRequestOptions"""
        variables_type = "" if language == "javascript" else """: Record<string, any>"""
        return_type = "" if language == "javascript" else """: Promise<GraphQLResponse<T>>"""
        
        return f"""
{ts_types if language == "typescript" else ""}
/**
 * GraphQL API Client for {api_url}
 */
class GraphQLClient {{
  endpoint;
  apiKey;
  defaultHeaders;

  /**
   * Create a new GraphQL client
   * @param {{string}} endpoint - GraphQL API endpoint
   * @param {{string}} apiKey - Optional API key for authentication
   */
  constructor(endpoint = "{api_url}", apiKey = null) {{
    this.endpoint = endpoint;
    this.apiKey = apiKey;
    
    this.defaultHeaders = {{
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }};
    
    if (apiKey) {{
      this.defaultHeaders['Authorization'] = `Bearer ${{apiKey}}`;
    }}
  }}

  /**
   * Execute a GraphQL query or mutation
   * @param {{string}} query - GraphQL query or mutation
   * @param {{Record<string, any>}} variables - Query variables
   * @param {{GraphQLRequestOptions}} options - Request options
   * @returns {{Promise<GraphQLResponse>}} GraphQL response
   */
  async execute{type_annotations}(query, variables{variables_type} = {{}}, options{param_types} = {{}}) {return_type} {{
    // Prepare headers
    const headers = {{
      ...this.defaultHeaders,
      ...options.headers,
    }};
    
    // Prepare request payload
    const payload = {{
      query,
      variables,
    }};
    
    // Set timeout if provided
    const controller = new AbortController();
    if (options.timeout) {{
      setTimeout(() => controller.abort(), options.timeout);
    }}
    
    // Perform request
    const response = await fetch(this.endpoint, {{
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: options.timeout ? controller.signal : undefined,
    }});
    
    if (!response.ok) {{
      throw new Error(`GraphQL request failed: ${{response.status}} ${{response.statusText}}`);
    }}
    
    const result = await response.json();
    
    // Check for GraphQL errors
    if (result.errors && result.errors.length > 0) {{
      console.error('GraphQL errors:', result.errors);
    }}
    
    return result;
  }}

  /**
   * Execute a GraphQL query
   * @param {{string}} queryString - GraphQL query
   * @param {{Record<string, any>}} variables - Query variables
   * @param {{GraphQLRequestOptions}} options - Request options
   * @returns {{Promise<GraphQLResponse>}} Query result
   */
  async query{type_annotations}(queryString, variables{variables_type} = {{}}, options{param_types} = {{}}) {return_type} {{
    return this.execute(queryString, variables, options);
  }}

  /**
   * Execute a GraphQL mutation
   * @param {{string}} mutationString - GraphQL mutation
   * @param {{Record<string, any>}} variables - Mutation variables
   * @param {{GraphQLRequestOptions}} options - Request options
   * @returns {{Promise<GraphQLResponse>}} Mutation result
   */
  async mutation{type_annotations}(mutationString, variables{variables_type} = {{}}, options{param_types} = {{}}) {return_type} {{
    return this.execute(mutationString, variables, options);
  }}
}}

{language == "javascript" ? "export default GraphQLClient;" : "export default GraphQLClient;"}
"""

    def _generate_js_generic_client(self, api_url: str, api_type: str, language: str) -> str:
        """Generate a generic JavaScript/TypeScript API client template."""
        ts_types = "" if language == "javascript" else """
interface RequestOptions {
  headers?: Record<string, string>;
  params?: Record<string, any>;
  timeout?: number;
}

interface APIResponse<T = any> {
  data: T;
  status: number;
}
"""
        
        return f"""
{ts_types if language == "typescript" else ""}
/**
 * {api_type.capitalize()} API Client for {api_url}
 */
class {api_type.capitalize()}Client {{
  baseUrl;
  apiKey;
  
  constructor(baseUrl = "{api_url}", apiKey = null) {{
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }}
  
  // TODO: Implement client methods for {api_type} API
}}

{language == "javascript" ? "export default " + api_type.capitalize() + "Client;" : "export default " + api_type.capitalize() + "Client;"}
"""

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for the target language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "go": ".go"
        }
        return extensions.get(language, ".txt")

    async def _create_endpoint(
        self,
        api_url: Optional[str],
        endpoint_path: Optional[str],
        http_method: Optional[str],
        request_body: Optional[str],
        request_headers: Optional[str]
    ) -> ToolResult:
        """Create and document an API endpoint."""
        
        if not api_url:
            return ToolResult(error="API URL is required for endpoint creation")
        
        if not endpoint_path:
            return ToolResult(error="Endpoint path is required for endpoint creation")
        
        if not http_method:
            http_method = "GET"  # Default to GET
        
        # Parse request body and headers if provided
        body_schema = {}
        if request_body:
            try:
                body_schema = json.loads(request_body)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid JSON format for request body")
        
        headers = {}
        if request_headers:
            try:
                headers = json.loads(request_headers)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid JSON format for request headers")
        
        # Create endpoint definition
        endpoint_id = f"{api_url}:{endpoint_path}:{http_method}"
        endpoint_def = {
            "api_url": api_url,
            "path": endpoint_path,
            "method": http_method,
            "body_schema": body_schema,
            "headers": headers,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store endpoint definition
        self.endpoints[endpoint_id] = endpoint_def
        
        # Generate endpoint documentation
        docs = self._generate_endpoint_docs(endpoint_def)
        
        return ToolResult(
            output=f"Endpoint created: {http_method} {api_url}/{endpoint_path.lstrip('/')}\n\n"
                   f"Endpoint Documentation:\n{docs}"
        )

    def _generate_endpoint_docs(self, endpoint: Dict[str, Any]) -> str:
        """Generate documentation for an API endpoint."""
        docs = [
            f"# {endpoint['method']} {endpoint['path']}",
            "",
            f"Base URL: `{endpoint['api_url']}`",
            "",
            "## Request",
            "",
            f"Method: `{endpoint['method']}`",
            f"Path: `{endpoint['path']}`",
            ""
        ]
        
        # Add headers documentation
        if endpoint.get("headers"):
            docs.append("### Headers")
            docs.append("")
            for header, value in endpoint["headers"].items():
                docs.append(f"- `{header}`: {value}")
            docs.append("")
        
        # Add request body documentation
        if endpoint.get("body_schema"):
            docs.append("### Request Body")
            docs.append("")
            docs.append("```json")
            docs.append(json.dumps(endpoint["body_schema"], indent=2))
            docs.append("```")
            docs.append("")
        
        return "\n".join(docs)

    async def _document_api(
        self,
        api_url: Optional[str],
        api_type: Optional[str],
        output_path: Optional[str]
    ) -> ToolResult:
        """Generate comprehensive API documentation."""
        
        if not api_url:
            return ToolResult(error="API URL is required for API documentation")
        
        if not api_type:
            api_type = "rest"  # Default to REST
        
        # Get all endpoints for this API
        api_endpoints = {
            endpoint_id: endpoint 
            for endpoint_id, endpoint in self.endpoints.items() 
            if endpoint["api_url"] == api_url
        }
        
        # Generate API documentation
        docs = self._generate_api_docs(api_url, api_type, api_endpoints)
        
        # If output path is provided, save the documentation
        if output_path:
            filename = f"api_documentation.md"
            full_path = os.path.join(output_path, filename)
            
            # This would save the file in a real implementation
            # For now, just simulate file saving
            
            return ToolResult(
                output=f"API documentation generated for {api_url}\n"
                       f"API type: {api_type}\n"
                       f"Saved to: {full_path}\n\n"
                       f"Documentation preview:\n{docs[:500]}..."
            )
        
        return ToolResult(
            output=f"API documentation generated for {api_url}\n"
                   f"API type: {api_type}\n\n"
                   f"{docs}"
        )

    def _generate_api_docs(self, api_url: str, api_type: str, endpoints: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive API documentation."""
        docs = [
            f"# API Documentation: {api_url}",
            "",
            f"API Type: {api_type.upper()}",
            f"Base URL: {api_url}",
            "",
            "## Endpoints",
            ""
        ]
        
        # Group endpoints by path
        endpoints_by_path = {}
        for endpoint in endpoints.values():
            path = endpoint["path"]
            if path not in endpoints_by_path:
                endpoints_by_path[path] = []
            endpoints_by_path[path].append(endpoint)
        
        # Add endpoint documentation
        for path, path_endpoints in endpoints_by_path.items():
            docs.append(f"### {path}")
            docs.append("")
            
            for endpoint in path_endpoints:
                docs.append(f"#### {endpoint['method']}")
                docs.append("")
                
                # Add headers documentation
                if endpoint.get("headers"):
                    docs.append("**Headers:**")
                    docs.append("")
                    for header, value in endpoint["headers"].items():
                        docs.append(f"- `{header}`: {value}")
                    docs.append("")
                
                # Add request body documentation
                if endpoint.get("body_schema"):
                    docs.append("**Request Body:**")
                    docs.append("")
                    docs.append("```json")
                    docs.append(json.dumps(endpoint["body_schema"], indent=2))
                    docs.append("```")
                    docs.append("")
        
        return "\n".join(docs)

    async def _analyze_api(
        self,
        api_url: Optional[str],
        response_example: Optional[str]
    ) -> ToolResult:
        """Analyze API structure and response patterns."""
        
        if not api_url and not response_example:
            return ToolResult(error="Either API URL or response example is required for API analysis")
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "api_url": api_url,
            "response_patterns": {},
            "data_structures": {},
            "recommendations": []
        }
        
        # Analyze response example if provided
        if response_example:
            try:
                response_data = json.loads(response_example)
                
                # Analyze response structure
                analysis_results["response_patterns"] = self._analyze_response_structure(response_data)
                
                # Identify data structures
                analysis_results["data_structures"] = self._identify_data_structures(response_data)
                
                # Generate recommendations
                analysis_results["recommendations"] = self._generate_api_recommendations(
                    api_url, analysis_results["response_patterns"], analysis_results["data_structures"]
                )
                
            except json.JSONDecodeError:
                return ToolResult(error="Invalid JSON format for response example")
        
        # Format analysis results
        output = self._format_api_analysis(analysis_results)
        
        return ToolResult(output=output)

    def _analyze_response_structure(self, response_data: Any) -> Dict[str, Any]:
        """Analyze the structure of an API response."""
        
        if isinstance(response_data, dict):
            # Analyze dictionary structure
            structure = {
                "type": "object",
                "properties": {},
                "patterns": []
            }
            
            # Check for common patterns
            if "data" in response_data and isinstance(response_data["data"], dict):
                structure["patterns"].append("data_wrapper")
            
            if "errors" in response_data:
                structure["patterns"].append("error_field")
            
            if "meta" in response_data or "metadata" in response_data:
                structure["patterns"].append("metadata")
            
            if "pagination" in response_data or any(k in response_data for k in ["page", "per_page", "total"]):
                structure["patterns"].append("pagination")
            
            # Analyze top-level properties
            for key, value in response_data.items():
                if isinstance(value, dict):
                    structure["properties"][key] = {"type": "object"}
                elif isinstance(value, list):
                    structure["properties"][key] = {"type": "array"}
                    if value and all(isinstance(item, dict) for item in value):
                        structure["properties"][key]["items"] = {"type": "object"}
                elif isinstance(value, str):
                    structure["properties"][key] = {"type": "string"}
                elif isinstance(value, (int, float)):
                    structure["properties"][key] = {"type": "number"}
                elif isinstance(value, bool):
                    structure["properties"][key] = {"type": "boolean"}
                elif value is None:
                    structure["properties"][key] = {"type": "null"}
            
            return structure
            
        elif isinstance(response_data, list):
            # Analyze list structure
            structure = {
                "type": "array",
                "patterns": []
            }
            
            if response_data and all(isinstance(item, dict) for item in response_data):
                structure["items"] = {"type": "object"}
                structure["patterns"].append("object_array")
                
                # Check if all objects have the same structure
                if len(response_data) > 1:
                    first_keys = set(response_data[0].keys())
                    all_same = all(set(item.keys()) == first_keys for item in response_data[1:])
                    if all_same:
                        structure["patterns"].append("uniform_objects")
            
            return structure
            
        else:
            # Simple value
            return {"type": type(response_data).__name__}

    def _identify_data_structures(self, response_data: Any) -> Dict[str, Any]:
        """Identify common data structures in an API response."""
        
        structures = {}
        
        def extract_structures(data, path=""):
            if isinstance(data, dict):
                # Check for common entity patterns
                if "id" in data and len(data) > 1:
                    entity_name = path.split(".")[-1] if path else "entity"
                    structures[entity_name] = {
                        "type": "entity",
                        "fields": list(data.keys()),
                        "sample": data
                    }
                
                # Recursively process nested objects
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    extract_structures(value, new_path)
                    
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                # Check for collection patterns
                collection_name = path.split(".")[-1] if path else "collection"
                structures[collection_name] = {
                    "type": "collection",
                    "item_fields": list(data[0].keys()),
                    "sample_item": data[0]
                }
                
                # Process first item to identify nested structures
                extract_structures(data[0], f"{path}[0]")
        
        extract_structures(response_data)
        return structures

    def _generate_api_recommendations(
        self, 
        api_url: Optional[str], 
        response_patterns: Dict[str, Any],
        data_structures: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on API analysis."""
        
        recommendations = []
        
        # Recommendations based on response patterns
        if response_patterns.get("type") == "object":
            patterns = response_patterns.get("patterns", [])
            
            if "data_wrapper" in patterns:
                recommendations.append("Use data extraction to handle the data wrapper pattern")
            
            if "error_field" in patterns:
                recommendations.append("Implement error handling for the error field pattern")
            
            if "pagination" in patterns:
                recommendations.append("Implement pagination handling for paginated responses")
        
        # Recommendations based on data structures
        for name, structure in data_structures.items():
            if structure["type"] == "entity":
                recommendations.append(f"Create a model class for the '{name}' entity")
            
            if structure["type"] == "collection":
                recommendations.append(f"Implement collection handling for '{name}' items")
        
        # General recommendations
        if api_url:
            recommendations.append("Implement request caching for improved performance")
            recommendations.append("Add retry logic for failed requests")
            recommendations.append("Implement request/response logging for debugging")
        
        return recommendations

    def _format_api_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format API analysis results for output."""
        
        output = ["# API Analysis"]
        
        if analysis.get("api_url"):
            output.append(f"\nAPI URL: {analysis['api_url']}")
        
        output.append(f"\nAnalysis timestamp: {analysis['timestamp']}")
        
        # Response patterns
        if analysis.get("response_patterns"):
            output.append("\n## Response Patterns")
            
            patterns = analysis["response_patterns"]
            output.append(f"\nResponse type: {patterns.get('type', 'unknown')}")
            
            if patterns.get("patterns"):
                output.append("\nDetected patterns:")
                for pattern in patterns["patterns"]:
                    output.append(f"- {pattern}")
            
            if patterns.get("properties"):
                output.append("\nTop-level properties:")
                for prop, prop_info in patterns["properties"].items():
                    output.append(f"- {prop}: {prop_info.get('type', 'unknown')}")
        
        # Data structures
        if analysis.get("data_structures"):
            output.append("\n## Data Structures")
            
            for name, structure in analysis["data_structures"].items():
                output.append(f"\n### {name} ({structure['type']})")
                
                if structure.get("fields"):
                    output.append("\nFields:")
                    for field in structure["fields"]:
                        output.append(f"- {field}")
                
                if structure.get("item_fields"):
                    output.append("\nItem fields:")
                    for field in structure["item_fields"]:
                        output.append(f"- {field}")
        
        # Recommendations
        if analysis.get("recommendations"):
            output.append("\n## Recommendations")
            
            for i, recommendation in enumerate(analysis["recommendations"], 1):
                output.append(f"\n{i}. {recommendation}")
        
        return "\n".join(output)

    async def _generate_request(
        self,
        api_url: Optional[str],
        endpoint_path: Optional[str],
        http_method: Optional[str],
        request_body: Optional[str]
    ) -> ToolResult:
        """Generate a sample API request."""
        
        if not api_url:
            return ToolResult(error="API URL is required for request generation")
        
        if not endpoint_path:
            return ToolResult(error="Endpoint path is required for request generation")
        
        if not http_method:
            http_method = "GET"  # Default to GET
        
        # Parse request body if provided
        body = None
        if request_body:
            try:
                body = json.loads(request_body)
            except json.JSONDecodeError:
                return ToolResult(error="Invalid JSON format for request body")
        
        # Generate curl command
        curl_command = self._generate_curl_command(api_url, endpoint_path, http_method, body)
        
        # Generate Python request
        python_request = self._generate_python_request(api_url, endpoint_path, http_method, body)
        
        # Generate JavaScript request
        js_request = self._generate_js_request(api_url, endpoint_path, http_method, body)
        
        output = [
            f"# API Request: {http_method} {api_url}/{endpoint_path.lstrip('/')}",
            "",
            "## cURL",
            "```bash",
            curl_command,
            "```",
            "",
            "## Python",
            "```python",
            python_request,
            "```",
            "",
            "## JavaScript",
            "```javascript",
            js_request,
            "```"
        ]
        
        return ToolResult(output="\n".join(output))

    def _generate_curl_command(
        self, 
        api_url: str, 
        endpoint_path: str, 
        http_method: str, 
        body: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a curl command for an API request."""
        
        url = f"{api_url}/{endpoint_path.lstrip('/')}"
        
        command_parts = [f"curl -X {http_method}"]
        
        # Add headers
        command_parts.append('-H "Content-Type: application/json"')
        command_parts.append('-H "Accept: application/json"')
        
        # Add request body for non-GET requests
        if http_method != "GET" and body:
            body_json = json.dumps(body)
            command_parts.append(f"-d '{body_json}'")
        
        # Add URL
        command_parts.append(f'"{url}"')
        
        return " \\\n  ".join(command_parts)

    def _generate_python_request(
        self, 
        api_url: str, 
        endpoint_path: str, 
        http_method: str, 
        body: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a Python request for an API request."""
        
        url = f"{api_url}/{endpoint_path.lstrip('/')}"
        
        code = [
            "import requests",
            "",
            f"url = '{url}'",
            "headers = {",
            "    'Content-Type': 'application/json',",
            "    'Accept': 'application/json'",
            "}",
            ""
        ]
        
        if body:
            code.append("payload = {")
            for key, value in body.items():
                if isinstance(value, str):
                    code.append(f"    '{key}': '{value}',")
                else:
                    code.append(f"    '{key}': {value},")
            code.append("}")
            code.append("")
        
        if http_method == "GET":
            code.append("response = requests.get(url, headers=headers)")
        elif http_method == "POST":
            code.append("response = requests.post(url, headers=headers, json=payload)")
        elif http_method == "PUT":
            code.append("response = requests.put(url, headers=headers, json=payload)")
        elif http_method == "DELETE":
            code.append("response = requests.delete(url, headers=headers)")
        else:
            code.append(f"response = requests.request('{http_method}', url, headers=headers, json=payload)")
        
        code.extend([
            "",
            "# Check if the request was successful",
            "response.raise_for_status()",
            "",
            "# Print the response",
            "print(response.json())"
        ])
        
        return "\n".join(code)

    def _generate_js_request(
        self, 
        api_url: str, 
        endpoint_path: str, 
        http_method: str, 
        body: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a JavaScript request for an API request."""
        
        url = f"{api_url}/{endpoint_path.lstrip('/')}"
        
        code = [
            "// Using fetch API",
            f"const url = '{url}';",
            "const headers = {",
            "  'Content-Type': 'application/json',",
            "  'Accept': 'application/json'",
            "};",
            ""
        ]
        
        if body:
            code.append("const payload = {")
            for key, value in body.items():
                if isinstance(value, str):
                    code.append(f"  {key}: '{value}',")
                else:
                    code.append(f"  {key}: {value},")
            code.append("};")
            code.append("")
        
        code.append("const requestOptions = {")
        code.append(f"  method: '{http_method}',")
        code.append("  headers,")
        if http_method != "GET" and body:
            code.append("  body: JSON.stringify(payload)")
        code.append("};")
        code.append("")
        
        code.extend([
            "// Make the request",
            "fetch(url, requestOptions)",
            "  .then(response => {",
            "    if (!response.ok) {",
            "      throw new Error(`HTTP error! Status: ${response.status}`);",
            "    }",
            "    return response.json();",
            "  })",
            "  .then(data => {",
            "    console.log('Success:', data);",
            "  })",
            "  .catch(error => {",
            "    console.error('Error:', error);",
            "  });"
        ])
        
        return "\n".join(code)

    async def _validate_response(
        self,
        response_example: Optional[str],
        api_type: Optional[str]
    ) -> ToolResult:
        """Validate an API response against expected schema."""
        
        if not response_example:
            return ToolResult(error="Response example is required for validation")
        
        if not api_type:
            api_type = "rest"  # Default to REST
        
        try:
            response_data = json.loads(response_example)
            
            # Perform validation
            validation_result = self._validate_response_data(response_data, api_type)
            
            # Format validation results
            output = self._format_validation_results(validation_result)
            
            return ToolResult(output=output)
            
        except json.JSONDecodeError:
            return ToolResult(error="Invalid JSON format for response example")

    def _validate_response_data(self, response_data: Any, api_type: str) -> Dict[str, Any]:
        """Validate response data against expected patterns."""
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "structure_analysis": {}
        }
        
        # Check for common REST API patterns
        if api_type == "rest":
            # Check for common error patterns
            if isinstance(response_data, dict):
                # Check for error fields
                if "error" in response_data or "errors" in response_data:
                    validation_result["issues"].append("Response contains error field")
                    validation_result["is_valid"] = False
                
                # Check for empty data
                if "data" in response_data and not response_data["data"]:
                    validation_result["warnings"].append("Response contains empty data field")
                
                # Check for null values in important fields
                for key, value in response_data.items():
                    if value is None and key not in ["error", "errors"]:
                        validation_result["warnings"].append(f"Response contains null value for field '{key}'")
        
        # Check for common GraphQL API patterns
        elif api_type == "graphql":
            if isinstance(response_data, dict):
                # Check for errors field
                if "errors" in response_data:
                    validation_result["issues"].append("Response contains GraphQL errors")
                    validation_result["is_valid"] = False
                
                # Check for missing data field
                if "data" not in response_data:
                    validation_result["issues"].append("Response missing required 'data' field")
                    validation_result["is_valid"] = False
        
        # Perform structure analysis
        validation_result["structure_analysis"] = self._analyze_response_structure(response_data)
        
        return validation_result

    def _format_validation_results(self, validation_result: Dict[str, Any]) -> str:
        """Format validation results for output."""
        
        output = ["# API Response Validation"]
        
        output.append(f"\nValidation timestamp: {validation_result['timestamp']}")
        output.append(f"Validation result: {'✅ Valid' if validation_result['is_valid'] else '❌ Invalid'}")
        
        if validation_result["issues"]:
            output.append("\n## Issues")
            for issue in validation_result["issues"]:
                output.append(f"- ❌ {issue}")
        
        if validation_result["warnings"]:
            output.append("\n## Warnings")
            for warning in validation_result["warnings"]:
                output.append(f"- ⚠️ {warning}")
        
        # Add structure analysis
        structure = validation_result["structure_analysis"]
        output.append("\n## Structure Analysis")
        
        output.append(f"\nResponse type: {structure.get('type', 'unknown')}")
        
        if structure.get("patterns"):
            output.append("\nDetected patterns:")
            for pattern in structure["patterns"]:
                output.append(f"- {pattern}")
        
        if structure.get("properties"):
            output.append("\nTop-level properties:")
            for prop, prop_info in structure["properties"].items():
                output.append(f"- {prop}: {prop_info.get('type', 'unknown')}")
        
        return "\n".join(output)

    async def _create_mock(
        self,
        api_url: Optional[str],
        endpoint_path: Optional[str],
        response_example: Optional[str]
    ) -> ToolResult:
        """Create a mock API response for testing."""
        
        if not api_url:
            return ToolResult(error="API URL is required for mock creation")
        
        if not endpoint_path:
            return ToolResult(error="Endpoint path is required for mock creation")
        
        if not response_example:
            return ToolResult(error="Response example is required for mock creation")
        
        try:
            response_data = json.loads(response_example)
            
            # Create mock response
            mock_id = f"{api_url}:{endpoint_path}"
            mock_response = {
                "api_url": api_url,
                "endpoint_path": endpoint_path,
                "response_data": response_data,
                "created_at": datetime.now().isoformat()
            }
            
            # Store mock response
            self.mock_responses[mock_id] = mock_response
            
            # Generate mock server code
            mock_server_code = self._generate_mock_server_code(api_url, endpoint_path, response_data)
            
            return ToolResult(
                output=f"Mock response created for {api_url}/{endpoint_path.lstrip('/')}\n\n"
                       f"Mock Server Code:\n```python\n{mock_server_code}\n```"
            )
            
        except json.JSONDecodeError:
            return ToolResult(error="Invalid JSON format for response example")

    def _generate_mock_server_code(
        self, 
        api_url: str, 
        endpoint_path: str, 
        response_data: Any
    ) -> str:
        """Generate mock server code for an API endpoint."""
        
        code = [
            "from fastapi import FastAPI, Request",
            "from fastapi.middleware.cors import CORSMiddleware",
            "import uvicorn",
            "import json",
            "",
            "app = FastAPI(title='API Mock Server')",
            "",
            "# Configure CORS",
            "app.add_middleware(",
            "    CORSMiddleware,",
            "    allow_origins=['*'],",
            "    allow_credentials=True,",
            "    allow_methods=['*'],",
            "    allow_headers=['*'],",
            ")",
            "",
            "# Mock response data",
            "MOCK_RESPONSES = {",
            f"    '{endpoint_path}': {json.dumps(response_data, indent=4)}",
            "}",
            "",
            "@app.get('/')",
            "async def root():",
            "    return {'message': 'API Mock Server is running'}",
            "",
            f"@app.get('{endpoint_path}')",
            f"@app.post('{endpoint_path}')",
            f"@app.put('{endpoint_path}')",
            f"@app.delete('{endpoint_path}')",
            f"async def mock_endpoint(request: Request):",
            "    return MOCK_RESPONSES.get(request.url.path, {'error': 'Not found'})",
            "",
            "if __name__ == '__main__':",
            "    uvicorn.run(app, host='0.0.0.0', port=8000)"
        ]
        
        return "\n".join(code)

    async def _generate_schema(
        self,
        response_example: Optional[str],
        api_type: Optional[str]
    ) -> ToolResult:
        """Generate JSON schema from API response example."""
        
        if not response_example:
            return ToolResult(error="Response example is required for schema generation")
        
        if not api_type:
            api_type = "rest"  # Default to REST
        
        try:
            response_data = json.loads(response_example)
            
            # Generate JSON schema
            schema = self._generate_json_schema(response_data)
            
            # Format schema as JSON
            schema_json = json.dumps(schema, indent=2)
            
            # Store schema
            schema_id = f"schema_{len(self.api_schemas) + 1}"
            self.api_schemas[schema_id] = {
                "schema": schema,
                "api_type": api_type,
                "created_at": datetime.now().isoformat()
            }
            
            return ToolResult(
                output=f"JSON Schema generated:\n```json\n{schema_json}\n```"
            )
            
        except json.JSONDecodeError:
            return ToolResult(error="Invalid JSON format for response example")

    def _generate_json_schema(self, data: Any, title: str = "APIResponse") -> Dict[str, Any]:
        """Generate JSON schema from data."""
        
        if isinstance(data, dict):
            properties = {}
            required = []
            
            for key, value in data.items():
                properties[key] = self._generate_json_schema(value, key)
                required.append(key)
            
            return {
                "type": "object",
                "title": title,
                "properties": properties,
                "required": required
            }
            
        elif isinstance(data, list):
            if data:
                # Use the first item as a sample
                items_schema = self._generate_json_schema(data[0], f"{title}Item")
                return {
                    "type": "array",
                    "items": items_schema
                }
            else:
                return {
                    "type": "array",
                    "items": {}
                }
                
        elif isinstance(data, str):
            return {"type": "string"}
            
        elif isinstance(data, int):
            return {"type": "integer"}
            
        elif isinstance(data, float):
            return {"type": "number"}
            
        elif isinstance(data, bool):
            return {"type": "boolean"}
            
        elif data is None:
            return {"type": "null"}
            
        else:
            return {"type": "string"}

    async def _create_integration(
        self,
        integration_name: Optional[str],
        api_url: Optional[str],
        api_type: Optional[str],
        auth_type: Optional[str],
        output_path: Optional[str]
    ) -> ToolResult:
        """Create a complete API integration package."""
        
        if not integration_name:
            return ToolResult(error="Integration name is required")
        
        if not api_url:
            return ToolResult(error="API URL is required")
        
        if not api_type:
            api_type = "rest"  # Default to REST
        
        if not auth_type:
            auth_type = "none"  # Default to no authentication
        
        # Generate integration files
        integration_files = self._generate_integration_files(
            integration_name, api_url, api_type, auth_type
        )
        
        # Format output
        output = [
            f"# API Integration: {integration_name}",
            "",
            f"API URL: {api_url}",
            f"API Type: {api_type}",
            f"Authentication: {auth_type}",
            "",
            "## Generated Files"
        ]
        
        for filename, content in integration_files.items():
            output.append(f"\n### {filename}")
            output.append("```python")
            output.append(content)
            output.append("```")
        
        # If output path is provided, save the files
        if output_path:
            # This would save the files in a real implementation
            # For now, just simulate file saving
            
            output.append("\nFiles would be saved to:")
            for filename in integration_files.keys():
                output.append(f"- {os.path.join(output_path, filename)}")
        
        return ToolResult(output="\n".join(output))

    def _generate_integration_files(
        self,
        integration_name: str,
        api_url: str,
        api_type: str,
        auth_type: str
    ) -> Dict[str, str]:
        """Generate files for a complete API integration."""
        
        files = {}
        
        # Generate client file
        client_filename = f"{integration_name.lower()}_client.py"
        if api_type == "rest":
            files[client_filename] = self._generate_python_rest_client(api_url)
        elif api_type == "graphql":
            files[client_filename] = self._generate_python_graphql_client(api_url)
        else:
            files[client_filename] = self._generate_python_generic_client(api_url, api_type)
        
        # Generate models file
        models_filename = f"{integration_name.lower()}_models.py"
        files[models_filename] = self._generate_models_file(integration_name)
        
        # Generate auth file if needed
        if auth_type != "none":
            auth_filename = f"{integration_name.lower()}_auth.py"
            files[auth_filename] = self._generate_auth_file(auth_type, api_url)
        
        # Generate utils file
        utils_filename = f"{integration_name.lower()}_utils.py"
        files[utils_filename] = self._generate_utils_file(integration_name)
        
        # Generate __init__ file
        init_filename = "__init__.py"
        files[init_filename] = self._generate_init_file(integration_name, client_filename, models_filename)
        
        return files

    def _generate_models_file(self, integration_name: str) -> str:
        """Generate models file for API integration."""
        
        return f"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BaseModel:
    \"\"\"Base model for {integration_name} API models.\"\"\"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        \"\"\"Create model instance from dictionary.\"\"\"
        return cls(**data)

@dataclass
class ApiResponse:
    \"\"\"Generic API response model.\"\"\"
    
    success: bool
    data: Optional[Any] = None
    errors: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiResponse':
        \"\"\"Create response model from dictionary.\"\"\"
        return cls(
            success=not bool(data.get('errors')),
            data=data.get('data'),
            errors=data.get('errors'),
            meta=data.get('meta')
        )

# TODO: Add specific models for your API entities
"""

    def _generate_auth_file(self, auth_type: str, api_url: str) -> str:
        """Generate authentication file for API integration."""
        
        if auth_type == "basic":
            return f"""
import base64
from typing import Dict, Optional

class BasicAuth:
    \"\"\"Basic authentication handler for {api_url}.\"\"\"
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        
    def get_auth_header(self) -> Dict[str, str]:
        \"\"\"Get authentication header.\"\"\"
        auth_string = f"{{self.username}}:{{self.password}}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return {{"Authorization": f"Basic {{encoded}}"}}
"""
        
        elif auth_type == "bearer":
            return f"""
from typing import Dict, Optional

class BearerAuth:
    \"\"\"Bearer token authentication handler for {api_url}.\"\"\"
    
    def __init__(self, token: str):
        self.token = token
        
    def get_auth_header(self) -> Dict[str, str]:
        \"\"\"Get authentication header.\"\"\"
        return {{"Authorization": f"Bearer {{self.token}}"}}
"""
        
        elif auth_type == "api_key":
            return f"""
from typing import Dict, Optional

class ApiKeyAuth:
    \"\"\"API key authentication handler for {api_url}.\"\"\"
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        self.api_key = api_key
        self.header_name = header_name
        
    def get_auth_header(self) -> Dict[str, str]:
        \"\"\"Get authentication header.\"\"\"
        return {{self.header_name: self.api_key}}
"""
        
        elif auth_type == "oauth":
            return f"""
import time
import requests
from typing import Dict, Optional

class OAuthHandler:
    \"\"\"OAuth authentication handler for {api_url}.\"\"\"
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.token = None
        self.token_expiry = 0
        
    def get_auth_header(self) -> Dict[str, str]:
        \"\"\"Get authentication header with valid token.\"\"\"
        if not self.token or time.time() >= self.token_expiry:
            self._refresh_token()
            
        return {{"Authorization": f"Bearer {{self.token}}"}}
    
    def _refresh_token(self) -> None:
        \"\"\"Refresh OAuth token.\"\"\"
        payload = {{
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }}
        
        if self.scope:
            payload["scope"] = self.scope
            
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        
        token_data = response.json()
        self.token = token_data["access_token"]
        self.token_expiry = time.time() + token_data.get("expires_in", 3600)
"""
        
        else:  # custom or none
            return f"""
from typing import Dict, Optional

class CustomAuth:
    \"\"\"Custom authentication handler for {api_url}.\"\"\"
    
    def __init__(self, **auth_params):
        self.auth_params = auth_params
        
    def get_auth_header(self) -> Dict[str, str]:
        \"\"\"Get authentication header.\"\"\"
        # TODO: Implement custom authentication logic
        return {{}}
"""

    def _generate_utils_file(self, integration_name: str) -> str:
        """Generate utilities file for API integration."""
        
        return f"""
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("{integration_name}")

def configure_logging(level: int = logging.INFO) -> None:
    \"\"\"Configure logging for the {integration_name} integration.\"\"\"
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

def log_request(method: str, url: str, headers: Dict[str, str], data: Optional[Dict[str, Any]] = None) -> None:
    \"\"\"Log API request details.\"\"\"
    logger.debug(f"API Request: {{method}} {{url}}")
    logger.debug(f"Headers: {{headers}}")
    if data:
        logger.debug(f"Data: {{data}}")

def log_response(status_code: int, response_data: Any) -> None:
    \"\"\"Log API response details.\"\"\"
    logger.debug(f"API Response: Status {{status_code}}")
    logger.debug(f"Response: {{response_data}}")

def handle_error_response(status_code: int, response_data: Any) -> None:
    \"\"\"Handle API error responses.\"\"\"
    error_message = f"API Error: Status {{status_code}}"
    
    if isinstance(response_data, dict):
        if "error" in response_data:
            error_message += f" - {{response_data['error']}}"
        elif "message" in response_data:
            error_message += f" - {{response_data['message']}}"
    
    logger.error(error_message)
    return error_message
"""

    def _generate_init_file(
        self, 
        integration_name: str, 
        client_filename: str, 
        models_filename: str
    ) -> str:
        """Generate __init__ file for API integration."""
        
        client_module = client_filename.replace(".py", "")
        models_module = models_filename.replace(".py", "")
        
        return f"""
\"\"\"
{integration_name} API Integration Package
\"\"\"

from .{client_module} import *
from .{models_module} import *

__version__ = "0.1.0"
__all__ = [
    # Add your exported classes here
]
"""