"""CLI commands for configuration and secrets management."""

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from typing import Optional

app = typer.Typer(help="Configuration and secrets management")
console = Console()


@app.command("set-key")
def set_api_key(
    provider: str = typer.Argument(
        ...,
        help="Provider name (alphavantage, polygon, alpaca, fred, bloomberg, refinitiv)"
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="API key value (will prompt if not provided)"
    ),
    store: bool = typer.Option(
        False,
        "--store",
        help="Store directly without confirmation"
    )
):
    """
    Securely store an API key.
    
    Examples:
        quantterm config set-key alphavantage
        quantterm config set-key polygon --value sk-xxxxx
    """
    from quantterm.security import SecureSecret, SecretsManager, get_secrets_manager
    
    # Normalize provider name
    provider = provider.lower().strip()
    
    # Validate provider
    valid_providers = SecretsManager.PROVIDERS.keys()
    if provider not in valid_providers:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"Valid providers: {', '.join(valid_providers)}")
        raise typer.Exit(1)
    
    # Get value if not provided
    if value is None:
        value = Prompt.ask(
            f"Enter {provider} API key",
            password=True
        )
    
    if not value:
        console.print("[red]Error: API key cannot be empty[/red]")
        raise typer.Exit(1)
    
    # Store securely
    try:
        secret = SecureSecret(
            service_name="quantterm",
            key_name=provider
        )
        secret.set(value)
        
        console.print(f"[green]Successfully stored {provider} API key[/green]")
        console.print("[dim]The key has been stored in your system keychain.[/dim]")
        
        # Clear the value from memory
        value = None
        secret.clear_cache()
        
    except Exception as e:
        console.print(f"[red]Error storing key: {e}[/red]")
        raise typer.Exit(1)


@app.command("list-keys")
def list_keys():
    """
    List configured API keys.
    
    Shows which providers have keys configured (not the actual values).
    """
    from quantterm.security import get_secrets_manager
    
    manager = get_secrets_manager()
    providers = manager.list_providers()
    
    table = Table(title="Configured API Keys")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description", style="dim")
    
    for provider, is_set in providers.items():
        status = "[green]✓ Configured[/green]" if is_set else "[yellow]✗ Not set[/yellow]"
        description = SecretsManager.PROVIDERS.get(provider, "")
        
        table.add_row(provider, status, description)
    
    console.print(table)


@app.command("remove-key")
def remove_key(
    provider: str = typer.Argument(
        ...,
        help="Provider name to remove"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation"
    )
):
    """
    Remove an API key.
    
    Example:
        quantterm config remove-key alphavantage
    """
    from quantterm.security import SecureSecret, SecretsManager
    
    # Normalize provider name
    provider = provider.lower().strip()
    
    # Validate provider
    valid_providers = SecretsManager.PROVIDERS.keys()
    if provider not in valid_providers:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        raise typer.Exit(1)
    
    # Confirm
    if not force:
        if not Confirm.ask(f"Remove {provider} API key?"):
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    # Delete
    try:
        secret = SecureSecret(
            service_name="quantterm",
            key_name=provider
        )
        secret.delete()
        
        console.print(f"[green]Removed {provider} API key[/green]")
        
    except Exception as e:
        console.print(f"[red]Error removing key: {e}[/red]")
        raise typer.Exit(1)


@app.command("migrate")
def migrate_keys():
    """
    Migrate API keys from environment variables to secure storage.
    
    This command scans environment variables for QuantTerm API keys
    and prompts you to store them securely.
    """
    import os
    from quantterm.security import get_secrets_manager
    
    console.print("[bold]API Key Migration[/bold]\n")
    
    # Check for environment variables
    env_vars = {}
    for key in os.environ:
        if key.startswith("QUANTTERM_") or key.startswith("QUANTTERM_"):
            # Found a potential API key
            value = os.environ[key]
            provider = key.replace("QUANTTERM_", "").replace("FRED_", "").lower()
            
            # Only add if it looks like an API key (not empty, not "true"/"false")
            if value and value.lower() not in ("true", "false"):
                env_vars[key] = value
    
    if not env_vars:
        console.print("[green]No API keys found in environment variables.[/green]")
        return
    
    console.print(f"Found {len(env_vars)} potential API key(s) in environment:\n")
    
    for key, value in env_vars.items():
        console.print(f"  {key}")
    
    console.print()
    
    if not Confirm.ask("Migrate these keys to secure storage?"):
        console.print("[yellow]Migration cancelled[/yellow]")
        return
    
    # Migrate
    manager = get_secrets_manager()
    migrated = []
    
    for key in env_vars:
        try:
            # Extract provider name
            provider = key.replace("QUANTTERM_", "").replace("FRED_", "").lower()
            
            manager.set_api_key(provider, env_vars[key])
            migrated.append(provider)
            
        except Exception as e:
            console.print(f"[red]Failed to migrate {key}: {e}[/red]")
    
    console.print()
    if migrated:
        console.print(f"[green]Successfully migrated {len(migrated)} key(s)[/green]")
        console.print("[yellow]Warning: Remove old environment variables manually![/yellow]")


@app.command("backend")
def show_backend():
    """
    Show which secure storage backend is in use.
    """
    from quantterm.security import SecureSecret, KEYRING_AVAILABLE, CRYPTO_AVAILABLE
    
    console.print("[bold]Secure Storage Backend[/bold]\n")
    
    console.print(f"keyring available: {KEYRING_AVAILABLE}")
    console.print(f"cryptography available: {CRYPTO_AVAILABLE}")
    console.print()
    
    # Test actual backend
    test = SecureSecret("quantterm", "_test")
    backend = test.backend
    
    console.print(f"Current backend: [cyan]{backend}[/cyan]\n")
    
    if backend == "environment":
        console.print("[yellow]Warning: Using environment variables.[/yellow]")
        console.print("Install keyring for better security:")
        console.print("  pip install keyring")
    elif backend == "encrypted_file":
        console.print("[yellow]Warning: Using encrypted file storage.[/yellow]")
        console.print("Install keyring for system keychain integration:")
        console.print("  pip install keyring")


if __name__ == "__main__":
    app()
