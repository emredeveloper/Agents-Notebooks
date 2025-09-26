import os
import json
import sys

try:
    import requests
except ImportError:
    print("'requests' package is required: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("'rich' package is required: pip install rich", file=sys.stderr)
    sys.exit(1)


API_URL = "https://ollama.com/api/web_search"
API_KEY = "api_key"  # set via env var in production

if not API_KEY:
    print("Error: OLLAMA_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

payload = {"query": "what is ollama?"}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

console = Console()

try:
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
except requests.RequestException as e:
    console.print(f"Request error: [red]{e}[/red]", style="bold red")
    sys.exit(1)

try:
    data = resp.json()
except ValueError:
    console.print("Invalid JSON response received:", style="bold red")
    console.print(resp.text)
    sys.exit(1)

# Rich output: display results in a table
results = data.get("results")
if isinstance(results, list) and results:
    table = Table(title="Ollama Web Search Results", show_lines=True)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("URL", style="magenta")
    table.add_column("Snippet", style="white")

    for idx, item in enumerate(results, start=1):
        title = str(item.get("title") or "(no title)")
        url = str(item.get("url") or "")
        content = str(item.get("content") or "")
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + " …"

        table.add_row(str(idx), title, url, snippet)

    console.print(table)
else:
    console.print(Panel.fit("Unexpected response format. Showing raw JSON.", style="yellow"))
    console.print_json(data=data)