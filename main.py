"""
Main Application Entry Point for the Library AI Agent.
Starts the Library Agent application with interactive REPL,
role switching (Member vs Librarian), single-query CLI mode, and benchmark harness trigger.
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional

from agent import Agent
from harness import SafetyHarness, run_benchmark_suite
from schemas import AgentConfig, UserContext, UserRole
from tools import registry


BANNER = r"""
========================================================================
   SAFE LIBRARY AI AGENT (Topic 07: Autonomous Agents & Safety)
========================================================================
"""


def print_help() -> None:
    """Print available interactive commands."""
    print("""
Available Commands:
  /role <member|librarian>  - Switch active role (RBAC permission control)
  /tools                    - List available tools, schemas, and permissions
  /status                   - Show active user, role, and model configuration
  /reset                    - Clear agent conversation memory and reset loan state
  /test                     - Run automated safety and permission benchmark suite
  /help                     - Display this help message
  /exit or q                - Exit the application
""")


def print_tools_table(current_role: UserRole) -> None:
    """Display table of tools with permission and risk level."""
    from harness import ROLE_PERMISSIONS

    allowed_tools = ROLE_PERMISSIONS.get(current_role, [])
    print("\n" + "=" * 75)
    print(f" REGISTERED LIBRARY TOOLS (Current Role: {current_role.value.upper()})")
    print("=" * 75)
    print(f"{'Tool Name':<20} | {'Risk Tier':<10} | {'Permission':<12} | {'Description'}")
    print("-" * 75)

    for td in registry.get_definitions():
        is_allowed = "ALLOWED" if td.name in allowed_tools else "FORBIDDEN"
        print(f"{td.name:<20} | {td.risk_level.value.upper():<10} | {is_allowed:<12} | {td.description[:35]}")
    print("=" * 75 + "\n")


def interactive_repl(agent: Agent) -> None:
    """Run interactive terminal loop for agent interaction."""
    print(BANNER)
    print(f"Logged in as: {agent.user_context.username} | Role: {agent.user_context.role.value.upper()}")
    print(f"Model Backend: {agent.config.model_name} (Provider: {agent.config.provider})")
    print("Type your message or type '/help' for command options. Enter '/exit' to quit.\n")

    while True:
        try:
            prompt_prefix = f"[{agent.user_context.role.value.upper()}] You > "
            user_input = input(prompt_prefix).strip()

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "exit", "quit", "q"):
                print("Exiting Library Agent.")
                break

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.lower() == "/tools":
                print_tools_table(agent.user_context.role)
                continue

            if user_input.lower() == "/reset":
                from tools import reset_db
                reset_db()
                agent.reset()
                print("Conversation memory and database state reset.\n")
                continue

            if user_input.lower() == "/status":
                print(f"\nUser: {agent.user_context.username}")
                print(f"Role: {agent.user_context.role.value.upper()}")
                print(f"Model: {agent.config.model_name}")
                print(f"Max Steps: {agent.config.max_steps}\n")
                continue

            if user_input.lower() == "/test":
                run_benchmark_suite()
                continue

            if user_input.lower().startswith("/role"):
                parts = user_input.split()
                if len(parts) >= 2:
                    raw_role = parts[1].lower()
                    if raw_role in ("member", "librarian", "customer", "admin"):
                        canonical_role = UserRole.normalize(raw_role)
                        agent.set_role(canonical_role)
                        print(f"Switched active role to: {canonical_role.value.upper()}\n")
                    else:
                        print("Usage: /role member  OR  /role librarian\n")
                else:
                    print("Usage: /role member  OR  /role librarian\n")
                continue

            agent.run(user_input)

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as err:
            print(f"\nError during execution: {err}\n")


def main() -> None:
    """CLI Argument parser and entry point."""
    parser = argparse.ArgumentParser(
        description="Safe AI Library Agent with Application-level Permission Control"
    )
    parser.add_argument(
        "--role",
        type=str,
        default="member",
        help="Initial user role (member or librarian)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:latest",
        help="LLM model name (default: llama3.2:latest)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "ollama", "openai"],
        default="auto",
        help="LLM backend provider",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single-shot query and exit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the automated test and safety benchmark suite",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose step traces",
    )

    args = parser.parse_args()

    if args.test:
        run_benchmark_suite()
        return

    canonical_role = UserRole.normalize(args.role)
    user_ctx = UserContext(
        username=f"User_{canonical_role.value.capitalize()}",
        role=canonical_role
    )
    config = AgentConfig(
        model_name=args.model,
        provider=args.provider,
        verbose=not args.quiet,
    )
    harness = SafetyHarness(user_context=user_ctx)
    agent = Agent(config=config, user_context=user_ctx, safety_harness=harness)

    if args.query:
        agent.run(args.query)
    else:
        interactive_repl(agent)


if __name__ == "__main__":
    main()
