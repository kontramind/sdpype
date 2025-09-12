# sdpype/utils/console.py - Console utilities for purge
"""
Console utilities for CLI operations
"""

from rich.console import Console
from rich.panel import Panel

console = Console()

def print_warning(message: str, title: str = None):
    """Print a warning message"""
    if title:
        console.print(Panel.fit(message, title=f"⚠️ {title}", border_style="yellow"))
    else:
        console.print(f"⚠️ {message}", style="bold yellow")

def print_success(message: str):
    """Print a success message"""
    console.print(f"✅ {message}", style="bold green")

def confirm_dangerous_action(action_name: str, details: list) -> bool:
    """Ask user to confirm a dangerous action"""
    
    print_warning(f"You are about to perform: {action_name}")
    console.print("\nThis will affect:")
    for detail in details:
        console.print(f"  • {detail}")
    
    response = console.input("\n❓ Type 'yes' to confirm: ")
    return response.lower() == 'yes'

def print_completion_panel(title: str, next_steps: list, success_message: str = None):
    """Print a completion panel with next steps"""
    
    content = ""
    if success_message:
        content += f"{success_message}\n\n"
    
    content += "Next steps:\n"
    for step in next_steps:
        content += f"• {step}\n"
    
    console.print(Panel.fit(content.strip(), title=f"🎉 {title}"))
