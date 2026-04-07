# Axioma VS Code Extension

The Axioma extension adds editor support for the Axioma scientific computing language, with syntax highlighting, language-server integration, an interactive compute panel, and bundled MCP configuration for AI coding tools.

## Features

- Syntax highlighting for statements, assumptions, properties, built-in functions, Greek identifiers, operators, pattern variables, and indexed tensor variance markers.
- Language support for comment toggling, bracket matching, auto-closing pairs, indentation, and folding markers.
- LSP client support for diagnostics, hover, completion, and code actions via `axioma-lsp`.
- Compute panel for evaluating Axioma expressions and viewing plain-text and LaTeX output.
- Workflow browser that exposes available Axioma MCP workflows from the compute server.
- Bundled MCP server configuration in [`mcp-config.json`](/Users/manavmadanrawal/Dev/axioma/editors/vscode/mcp-config.json) for AI tooling discovery.
- Packaged light and dark extension icons in [`icons/axioma-light.png`](/Users/manavmadanrawal/Dev/axioma/editors/vscode/icons/axioma-light.png) and [`icons/axioma-dark.png`](/Users/manavmadanrawal/Dev/axioma/editors/vscode/icons/axioma-dark.png).

## Installation

### From a VSIX

1. Build the extension:

```bash
cd editors/vscode
npm install
npm run compile
npm run package
```

2. In VS Code, open `Extensions: Install from VSIX...` and choose the generated `.vsix` file.

### From source

```bash
cd editors/vscode
npm install
npm run compile
```

Then open this folder in VS Code and press `F5` to launch an Extension Development Host.

## Usage

- Open any `.ax` or `.axioma` file to activate the extension.
- Run `Axioma: Evaluate Selection` with `Shift+Enter`.
- Run `Axioma: Evaluate Current Line` with `Ctrl+Enter` or `Cmd+Enter` on macOS.
- Run `Axioma: Evaluate File` from the editor title bar.
- Run `Axioma: Show Compute Panel` to open the compute panel beside the editor.
- Run `Axioma: Show Available Workflows` to browse MCP-backed workflow templates.

## Configuration

The extension contributes these settings:

- `axioma.lspPath`: path to the `axioma-lsp` binary.
- `axioma.mcpPath`: path to the `axioma-mcp` binary.
- `axioma.mcpTimeout`: timeout in seconds for MCP tool calls.
- `axioma.renderMode`: choose `latex`, `unicode`, or `both` in the compute panel.
- `axioma.autoEvaluate`: evaluate the active Axioma file automatically on save.

## Keybindings

- `Shift+Enter`: evaluate the current selection in an Axioma editor.
- `Ctrl+Enter` / `Cmd+Enter`: evaluate the current line in an Axioma editor.

## Manual Testing

Use the Extension Development Host and verify these scenarios:

- Open a `.ax` file and confirm syntax highlighting appears.
- Type `property R riemann_symmetry;` and confirm both `property` and `riemann_symmetry` are highlighted correctly.
- Press `Ctrl+Enter` on `1 + 1;` and confirm the compute panel shows `2`.
- Press `Shift+Enter` on a selection and confirm the compute panel shows the result.
- Run `Axioma: Show Available Workflows` and confirm a quick pick appears with workflow names.
- Confirm parse errors surface as diagnostics.
- Confirm hover appears on function names once the language server provides hover content.

## Screenshot

Screenshot placeholder:

[`docs/screenshot-placeholder.png`](/Users/manavmadanrawal/Dev/axioma/editors/vscode/docs/screenshot-placeholder.png)

## Icons

The extension includes generated placeholder PNG icons for light and dark themes.
