"""
Memory Management Tool
Manages context, learning, and knowledge persistence across interactions
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool, ToolResult


class MemoryManagerTool(BaseTool):
    """
    Advanced memory management tool for maintaining context, storing learnings,
    and managing knowledge across interactions.
    """

    name: str = "memory_manager"
    description: str = """
    Manage memory, context, and learning across interactions. Store important information,
    retrieve relevant context, and maintain knowledge for improved performance.
    """
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "store", "retrieve", "update", "delete", "search", 
                    "get_context", "add_learning", "get_patterns", "clear_memory"
                ],
                "description": "The memory action to perform"
            },
            "key": {
                "type": "string",
                "description": "Key for storing/retrieving information"
            },
            "value": {
                "type": "string",
                "description": "Value to store (for store/update actions)"
            },
            "category": {
                "type": "string",
                "enum": ["context", "learning", "preference", "fact", "pattern", "error"],
                "description": "Category of information being stored"
            },
            "query": {
                "type": "string", 
                "description": "Search query for finding relevant information"
            },
            "importance": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Importance level of the information"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorizing and finding information"
            }
        },
        "required": ["action"]
    }

    # Memory storage (in production, this would be persisted)
    memory_store: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    learning_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    session_context: Dict[str, Any] = Field(default_factory=dict)

    async def execute(
        self,
        action: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        category: str = "context",
        query: Optional[str] = None,
        importance: str = "medium",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the memory management action."""
        
        try:
            if action == "store":
                return await self._store_information(key, value, category, importance, tags)
            elif action == "retrieve":
                return await self._retrieve_information(key)
            elif action == "update":
                return await self._update_information(key, value, category, importance, tags)
            elif action == "delete":
                return await self._delete_information(key)
            elif action == "search":
                return await self._search_memory(query, category)
            elif action == "get_context":
                return await self._get_current_context()
            elif action == "add_learning":
                return await self._add_learning(value, tags)
            elif action == "get_patterns":
                return await self._get_learned_patterns()
            elif action == "clear_memory":
                return await self._clear_memory(category)
            else:
                return ToolResult(error=f"Unknown action: {action}")
                
        except Exception as e:
            return ToolResult(error=f"Memory manager error: {str(e)}")

    async def _store_information(
        self, 
        key: str, 
        value: str, 
        category: str = "context",
        importance: str = "medium",
        tags: Optional[List[str]] = None
    ) -> ToolResult:
        """Store information in memory."""
        
        if not key or not value:
            return ToolResult(error="Key and value are required for storing information")

        memory_entry = {
            "key": key,
            "value": value,
            "category": category,
            "importance": importance,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }
        
        self.memory_store[key] = memory_entry
        
        # Update session context if it's context information
        if category == "context":
            self.session_context[key] = value
        
        return ToolResult(
            output=f"Stored information: {key} = {value[:100]}{'...' if len(value) > 100 else ''}"
        )

    async def _retrieve_information(self, key: str) -> ToolResult:
        """Retrieve information from memory."""
        
        if not key:
            return ToolResult(error="Key is required for retrieving information")
        
        if key not in self.memory_store:
            return ToolResult(error=f"No information found for key: {key}")
        
        entry = self.memory_store[key]
        
        # Update access statistics
        entry["access_count"] += 1
        entry["last_accessed"] = datetime.now().isoformat()
        
        return ToolResult(
            output=f"Retrieved: {key} = {entry['value']}\n"
                   f"Category: {entry['category']}, Importance: {entry['importance']}\n"
                   f"Tags: {', '.join(entry['tags'])}"
        )

    async def _update_information(
        self,
        key: str,
        value: str,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ToolResult:
        """Update existing information in memory."""
        
        if not key:
            return ToolResult(error="Key is required for updating information")
        
        if key not in self.memory_store:
            return ToolResult(error=f"No information found for key: {key}")
        
        entry = self.memory_store[key]
        
        # Update provided fields
        if value is not None:
            entry["value"] = value
        if category is not None:
            entry["category"] = category
        if importance is not None:
            entry["importance"] = importance
        if tags is not None:
            entry["tags"] = tags
        
        entry["updated_at"] = datetime.now().isoformat()
        
        return ToolResult(output=f"Updated information for key: {key}")

    async def _delete_information(self, key: str) -> ToolResult:
        """Delete information from memory."""
        
        if not key:
            return ToolResult(error="Key is required for deleting information")
        
        if key not in self.memory_store:
            return ToolResult(error=f"No information found for key: {key}")
        
        del self.memory_store[key]
        
        # Remove from session context if present
        if key in self.session_context:
            del self.session_context[key]
        
        return ToolResult(output=f"Deleted information for key: {key}")

    async def _search_memory(self, query: str, category: Optional[str] = None) -> ToolResult:
        """Search memory for relevant information."""
        
        if not query:
            return ToolResult(error="Query is required for searching memory")
        
        query_lower = query.lower()
        results = []
        
        for key, entry in self.memory_store.items():
            # Check if query matches key, value, or tags
            matches = (
                query_lower in key.lower() or
                query_lower in entry["value"].lower() or
                any(query_lower in tag.lower() for tag in entry["tags"])
            )
            
            # Filter by category if specified
            if category and entry["category"] != category:
                matches = False
            
            if matches:
                results.append({
                    "key": key,
                    "value": entry["value"][:200] + "..." if len(entry["value"]) > 200 else entry["value"],
                    "category": entry["category"],
                    "importance": entry["importance"],
                    "relevance_score": self._calculate_relevance(query_lower, entry)
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if not results:
            return ToolResult(output=f"No results found for query: {query}")
        
        output = [f"Found {len(results)} results for '{query}':"]
        for i, result in enumerate(results[:10], 1):  # Limit to top 10 results
            output.append(f"\n{i}. {result['key']} (relevance: {result['relevance_score']:.2f})")
            output.append(f"   Category: {result['category']}, Importance: {result['importance']}")
            output.append(f"   Value: {result['value']}")
        
        return ToolResult(output="\n".join(output))

    def _calculate_relevance(self, query: str, entry: Dict[str, Any]) -> float:
        """Calculate relevance score for search results."""
        
        score = 0.0
        
        # Exact matches in key get highest score
        if query in entry["key"].lower():
            score += 1.0
        
        # Partial matches in key
        if any(word in entry["key"].lower() for word in query.split()):
            score += 0.5
        
        # Matches in value
        value_lower = entry["value"].lower()
        if query in value_lower:
            score += 0.8
        
        # Partial matches in value
        if any(word in value_lower for word in query.split()):
            score += 0.3
        
        # Matches in tags
        for tag in entry["tags"]:
            if query in tag.lower():
                score += 0.6
        
        # Boost score based on importance
        importance_boost = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.0
        }
        score += importance_boost.get(entry["importance"], 0.0)
        
        # Boost score based on recent access
        if entry.get("last_accessed"):
            # More recent access gets higher score
            days_since_access = (datetime.now() - datetime.fromisoformat(entry["last_accessed"])).days
            if days_since_access < 1:
                score += 0.2
            elif days_since_access < 7:
                score += 0.1
        
        return score

    async def _get_current_context(self) -> ToolResult:
        """Get current session context."""
        
        if not self.session_context:
            return ToolResult(output="No current context available")
        
        output = ["Current Session Context:"]
        for key, value in self.session_context.items():
            output.append(f"  {key}: {value}")
        
        # Add recent context history
        if self.context_history:
            output.append("\nRecent Context History:")
            for i, context in enumerate(self.context_history[-5:], 1):
                output.append(f"  {i}. {context.get('summary', 'No summary')}")
        
        return ToolResult(output="\n".join(output))

    async def _add_learning(self, learning: str, tags: Optional[List[str]] = None) -> ToolResult:
        """Add a learning or insight to memory."""
        
        if not learning:
            return ToolResult(error="Learning content is required")
        
        learning_entry = {
            "content": learning,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "confidence": "medium",  # Could be determined by analysis
            "applications": []  # Where this learning has been applied
        }
        
        self.learning_patterns.append(learning_entry)
        
        # Also store in main memory with learning category
        learning_key = f"learning_{len(self.learning_patterns)}"
        await self._store_information(
            learning_key, learning, "learning", "high", tags
        )
        
        return ToolResult(output=f"Added learning: {learning[:100]}{'...' if len(learning) > 100 else ''}")

    async def _get_learned_patterns(self) -> ToolResult:
        """Get learned patterns and insights."""
        
        if not self.learning_patterns:
            return ToolResult(output="No learned patterns available")
        
        output = ["Learned Patterns and Insights:"]
        
        # Group learnings by tags
        tagged_learnings = {}
        untagged_learnings = []
        
        for learning in self.learning_patterns:
            if learning["tags"]:
                for tag in learning["tags"]:
                    if tag not in tagged_learnings:
                        tagged_learnings[tag] = []
                    tagged_learnings[tag].append(learning)
            else:
                untagged_learnings.append(learning)
        
        # Display tagged learnings
        for tag, learnings in tagged_learnings.items():
            output.append(f"\n{tag.upper()}:")
            for learning in learnings[-3:]:  # Show last 3 per tag
                output.append(f"  - {learning['content'][:150]}{'...' if len(learning['content']) > 150 else ''}")
        
        # Display untagged learnings
        if untagged_learnings:
            output.append("\nGENERAL:")
            for learning in untagged_learnings[-5:]:  # Show last 5 untagged
                output.append(f"  - {learning['content'][:150]}{'...' if len(learning['content']) > 150 else ''}")
        
        return ToolResult(output="\n".join(output))

    async def _clear_memory(self, category: Optional[str] = None) -> ToolResult:
        """Clear memory, optionally by category."""
        
        if category:
            # Clear specific category
            keys_to_delete = [
                key for key, entry in self.memory_store.items() 
                if entry["category"] == category
            ]
            
            for key in keys_to_delete:
                del self.memory_store[key]
            
            # Clear from session context if it's context category
            if category == "context":
                self.session_context.clear()
            
            # Clear learning patterns if it's learning category
            if category == "learning":
                self.learning_patterns.clear()
            
            return ToolResult(output=f"Cleared {len(keys_to_delete)} items from {category} category")
        
        else:
            # Clear all memory
            count = len(self.memory_store)
            self.memory_store.clear()
            self.session_context.clear()
            self.context_history.clear()
            self.learning_patterns.clear()
            
            return ToolResult(output=f"Cleared all memory ({count} items)")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        
        stats = {
            "total_items": len(self.memory_store),
            "categories": {},
            "importance_levels": {},
            "session_context_items": len(self.session_context),
            "learning_patterns": len(self.learning_patterns),
            "context_history": len(self.context_history)
        }
        
        # Count by category and importance
        for entry in self.memory_store.values():
            category = entry["category"]
            importance = entry["importance"]
            
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            stats["importance_levels"][importance] = stats["importance_levels"].get(importance, 0) + 1
        
        return stats