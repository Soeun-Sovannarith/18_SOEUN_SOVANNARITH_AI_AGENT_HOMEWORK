"""
Safety and permission harness for the Library AI Agent.
Enforces application-level Role-Based Access Control (RBAC),
Pydantic input validation, loop limits, and risk classification.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from schemas import (
    PermissionCheckResult,
    RiskLevel,
    ToolCall,
    ToolResult,
    UserContext,
    UserRole,
)
from tools import ToolRegistry, registry as default_registry


# Role-Based Access Control matrix

ROLE_PERMISSIONS: Dict[UserRole, List[str]] = {
    UserRole.MEMBER: [
        "get_all_books",
        "get_book_details",
        "loan_book",
        "return_book",
    ],
    UserRole.LIBRARIAN: [
        "get_all_books",
        "get_book_details",
        "loan_book",
        "return_book",
    ],
}


class SafetyHarness:
    """Application-level safety guard to intercept tool calls before execution."""

    def __init__(
        self,
        user_context: Optional[UserContext] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_calls_per_run: int = 12,
        require_hitl_for_red_risk: bool = False,
    ) -> None:
        self.user_context = user_context or UserContext(role=UserRole.MEMBER)
        self.registry = tool_registry or default_registry
        self.max_tool_calls = max_tool_calls_per_run
        self.require_hitl = require_hitl_for_red_risk
        self.tool_calls_count = 0

    def reset(self) -> None:
        """Reset tool call counter for a new turn."""
        self.tool_calls_count = 0

    def check_permission(self, tool_name: str) -> PermissionCheckResult:
        """Verify if current role is authorized to execute requested tool."""
        role = self.user_context.role
        allowed_tools = ROLE_PERMISSIONS.get(role, [])
        tool_def = self.registry.get_definition(tool_name)
        risk = tool_def.risk_level if tool_def else RiskLevel.GREEN

        if tool_name not in allowed_tools:
            return PermissionCheckResult(
                allowed=False,
                reason=(
                    f"PERMISSION_DENIED: Role '{role.value.upper()}' is not authorized "
                    f"to execute action '{tool_name}'."
                ),
                risk_level=risk,
            )

        return PermissionCheckResult(
            allowed=True,
            reason=f"Action '{tool_name}' authorized for role '{role.value.upper()}'.",
            risk_level=risk,
        )

    def intercept_and_execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Intercept tool call and perform checks:
        1. Tool call limit check
        2. Application RBAC permission check
        3. Input validation against Pydantic schema
        4. Tool execution
        """
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments

        self.tool_calls_count += 1
        if self.tool_calls_count > self.max_tool_calls:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=f"SAFETY_LIMIT_EXCEEDED: Maximum tool execution limit ({self.max_tool_calls}) reached.",
                success=False,
                error_code="SAFETY_LIMIT_EXCEEDED",
            )

        perm = self.check_permission(tool_name)
        if not perm.allowed:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=perm.reason or "PERMISSION_DENIED",
                success=False,
                error_code="PERMISSION_DENIED",
            )

        validated_model, val_error = self.registry.validate_inputs(tool_name, raw_args)
        if val_error:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=f"INPUT_VALIDATION_ERROR: {val_error}",
                success=False,
                error_code="INPUT_VALIDATION_ERROR",
            )

        func = self.registry.get_tool(tool_name)
        if not func:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=f"TOOL_NOT_FOUND: Tool '{tool_name}' does not exist.",
                success=False,
                error_code="TOOL_NOT_FOUND",
            )

        try:
            kwargs = validated_model.model_dump() if validated_model else {}
            output = func(**kwargs)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=str(output),
                success=True,
                error_code=None,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=f"EXECUTION_ERROR: Internal tool failure ({type(e).__name__}: {str(e)})",
                success=False,
                error_code="EXECUTION_ERROR",
            )
