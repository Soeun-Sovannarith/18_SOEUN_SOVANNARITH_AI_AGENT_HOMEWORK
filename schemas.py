"""
Data schemas and models for the Library AI Agent application.
Defines Pydantic models for tool inputs, user contexts, permission checks,
chat messages, agent execution traces, and evaluation results.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for application-level RBAC permission control."""
    MEMBER = "member"
    LIBRARIAN = "librarian"
    CUSTOMER = "customer"
    ADMIN = "admin"

    @classmethod
    def normalize(cls, role_input: Union[str, "UserRole"]) -> "UserRole":
        """Normalize role string or enum to canonical role."""
        if isinstance(role_input, UserRole):
            r = role_input.value
        else:
            r = str(role_input).strip().lower()
        if r in ("librarian", "admin"):
            return cls.LIBRARIAN
        return cls.MEMBER


class RiskLevel(str, Enum):
    """Safety risk classification for tools."""
    GREEN = "green"    # Read-only, safe
    YELLOW = "yellow"  # State modification
    RED = "red"        # Destructive or sensitive action


class UserContext(BaseModel):
    """Session context representing the current active user and their role."""
    user_id: str = Field(default="user_001", description="Unique user identifier")
    username: str = Field(default="Patron User", description="Display name")
    role: UserRole = Field(default=UserRole.MEMBER, description="Role determining tool permissions")


class PermissionCheckResult(BaseModel):
    """Result returned by the application-level permission harness."""
    allowed: bool = Field(..., description="Whether the action is allowed to execute")
    reason: Optional[str] = Field(default=None, description="Explanation if denied or conditional")
    risk_level: RiskLevel = Field(default=RiskLevel.GREEN, description="Risk tier of the tool")


# Tool Input Schemas

class SearchBooksInput(BaseModel):
    """Schema for search_books tool."""
    query: str = Field(default="", description="Keywords to search library catalog (title, author, topic). Use empty string '' or 'all' to list all books in catalog.")
    category: Optional[str] = Field(default=None, description="Optional category filter")


class CheckAvailabilityInput(BaseModel):
    """Schema for check_availability tool."""
    book_id: int = Field(..., gt=0, description="Positive integer ID of the book")


class GetBookDetailsInput(BaseModel):
    """Schema for get_book_details tool."""
    book_id: int = Field(..., gt=0, description="Positive integer ID of the book to retrieve from the database")


class BorrowBookInput(BaseModel):
    """Schema for borrow_book tool."""
    book_id: int = Field(..., gt=0, description="Positive integer ID of the book to borrow")
    duration_days: int = Field(default=14, gt=0, le=30, description="Borrow duration in days (between 1 and 30 days)")


class ReturnBookInput(BaseModel):
    """Schema for return_book tool."""
    loan_id: str = Field(..., min_length=3, description="Active Loan identifier string (e.g. 'LOAN-1001')")


class GetLoanStatusInput(BaseModel):
    """Schema for get_loan_status tool."""
    loan_id: str = Field(..., min_length=3, description="Loan identifier string (e.g. 'LOAN-1001')")


class DeleteBookInput(BaseModel):
    """Schema for delete_book tool (Librarian only)."""
    book_id: int = Field(..., gt=0, description="Positive integer ID of the book to delete")


# Message and Tool Calling Schemas

class Role(str, Enum):
    """Supported roles for chat message turns."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """Representation of a function call proposed by the model."""
    name: str = Field(..., description="Name of the tool to invoke")
    arguments: Union[Dict[str, Any], str] = Field(
        default_factory=dict,
        description="Dictionary of keyword arguments or raw JSON string"
    )


class ToolCall(BaseModel):
    """Tool call payload matching OpenAI and Ollama specs."""
    id: str = Field(..., description="Unique tool call invocation ID")
    type: str = Field(default="function", description="Tool type")
    function: FunctionCall = Field(..., description="Function invocation details")


class ToolResult(BaseModel):
    """Result returned after tool execution through the safety harness."""
    tool_call_id: str = Field(..., description="ID matching the tool call")
    name: str = Field(..., description="Name of the tool")
    content: str = Field(..., description="Output observation or structured error message")
    success: bool = Field(default=True, description="Whether the tool succeeded")
    error_code: Optional[str] = Field(default=None, description="Standardized error code if failed")


class ChatMessage(BaseModel):
    """Unified message object."""
    role: Role = Field(..., description="Role of the sender")
    content: Optional[str] = Field(default=None, description="Textual message content")
    name: Optional[str] = Field(default=None, description="Author or tool name")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Tool calls proposed")
    tool_call_id: Optional[str] = Field(default=None, description="Matching ID if tool response")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to API-compatible dictionary."""
        payload: Dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            payload["content"] = self.content
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments if isinstance(tc.function.arguments, str)
                        else str(tc.function.arguments)
                    }
                }
                for tc in self.tool_calls
            ]
        return payload


class ToolDefinition(BaseModel):
    """Metadata schema defining tool specification for the LLM."""
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Functionality summary")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for inputs")
    risk_level: RiskLevel = Field(default=RiskLevel.GREEN, description="Risk tier")

    def to_openai_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI / Ollama function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"[{self.risk_level.value.upper()} RISK] {self.description}",
                "parameters": self.parameters,
            }
        }


# Agent Configuration and Traces

class AgentConfig(BaseModel):
    """Configuration options for the Agent runtime."""
    model_name: str = Field(default="llama3.2:latest", description="LLM model name")
    provider: str = Field(default="auto", description="Backend provider ('auto', 'ollama', 'openai')")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_steps: int = Field(default=6, ge=1, le=20, description="Max allowed reasoning loops")
    system_prompt: Optional[str] = Field(default=None)
    verbose: bool = Field(default=True, description="Print step-by-step traces")


class AgentStep(BaseModel):
    """Record of a single step in the agent loop."""
    step_number: int = Field(..., description="Step index")
    thought: Optional[str] = Field(default=None, description="Model reasoning")
    action: Optional[str] = Field(default=None, description="Tool name called")
    action_input: Optional[Dict[str, Any]] = Field(default=None, description="Input arguments")
    observation: Optional[str] = Field(default=None, description="Tool output observation")
    permission_allowed: Optional[bool] = Field(default=None, description="Harness permission decision")


class AgentResponse(BaseModel):
    """Final output object from an agent execution run."""
    query: str = Field(..., description="Original user prompt")
    final_answer: str = Field(..., description="Synthesized final response")
    steps: List[AgentStep] = Field(default_factory=list, description="All trace steps")
    total_steps: int = Field(default=0)
    tool_calls_count: int = Field(default=0)
    execution_time_seconds: float = Field(default=0.0)
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)


# Test Case Models

class TestCase(BaseModel):
    """Test case definition for harness benchmark."""
    name: str = Field(..., description="Test name")
    query: str = Field(..., description="User prompt")
    role: UserRole = Field(default=UserRole.MEMBER, description="Role context for this test")
    expected_tools: Optional[List[str]] = Field(default=None, description="Expected tools called")
    expected_permission_denied: bool = Field(default=False, description="Whether permission should be rejected")
    expected_substrings: Optional[List[str]] = Field(default=None, description="Keywords expected in answer")
    max_steps: int = Field(default=6)


class HarnessResult(BaseModel):
    """Result of evaluating a test case."""
    test_case: str = Field(...)
    role: str = Field(...)
    passed: bool = Field(...)
    details: str = Field(...)
    execution_time: float = Field(default=0.0)
    steps_taken: int = Field(default=0)
    tools_used: List[str] = Field(default_factory=list)
