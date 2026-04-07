import * as vscode from 'vscode';
import { McpManager } from './mcp';

type ResultCallback = (uri: string, line: number, result: string) => void;
type ResultEntry = {
    code: string;
    latex: string;
    unicode: string;
    status: string;
    timestamp: number;
};

export class ComputePanel implements vscode.Disposable, vscode.WebviewViewProvider {
    private panel: vscode.WebviewPanel | undefined;
    private sidebarView: vscode.WebviewView | undefined;
    private readonly results: ResultEntry[] = [];
    private readonly resultCallbacks: ResultCallback[] = [];

    constructor(
        private readonly context: vscode.ExtensionContext,
        private readonly mcp: McpManager
    ) {}

    onResult(callback: ResultCallback): void {
        this.resultCallbacks.push(callback);
    }

    resolveWebviewView(webviewView: vscode.WebviewView): void {
        this.sidebarView = webviewView;
        this.configureWebview(webviewView.webview);
        this.updateContent();
        webviewView.onDidDispose(() => {
            if (this.sidebarView === webviewView) {
                this.sidebarView = undefined;
            }
        });
    }

    show(): void {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.Beside);
            this.updateContent();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'axiomaCompute',
            'Axioma Compute',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        this.configureWebview(this.panel.webview);

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });

        this.updateContent();
    }

    async evaluate(code: string, uri: string): Promise<void> {
        this.show();

        const statements = code
            .split(';')
            .map((statement) => statement.trim())
            .filter((statement) => statement.length > 0);

        for (const statement of statements) {
            try {
                const result = await this.mcp.evaluate(`${statement};`);
                const entry: ResultEntry = {
                    code: statement,
                    latex: stringifyField(result?.latex),
                    unicode: stringifyField(result?.unicode ?? result?.message ?? result),
                    status: stringifyField(result?.status || 'ok'),
                    timestamp: Date.now()
                };
                this.results.push(entry);

                if (uri) {
                    this.emitInlineResult(uri, statement, entry.unicode);
                }
            } catch (error) {
                this.results.push({
                    code: statement,
                    latex: '',
                    unicode: `Error: ${error instanceof Error ? error.message : String(error)}`,
                    status: 'error',
                    timestamp: Date.now()
                });
            }
        }

        this.updateContent();
    }

    clear(): void {
        this.results.length = 0;
        this.updateContent();
    }

    dispose(): void {
        this.panel?.dispose();
        this.sidebarView = undefined;
    }

    private configureWebview(webview: vscode.Webview): void {
        webview.options = {
            enableScripts: true
        };

        webview.onDidReceiveMessage(async (message: { type: string; code?: string }) => {
            if (message.type === 'evaluate' && typeof message.code === 'string') {
                await this.evaluate(message.code, '');
                return;
            }

            if (message.type === 'clear') {
                this.clear();
            }
        });
    }

    private emitInlineResult(uri: string, statement: string, result: string): void {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.uri.toString() !== uri) {
            return;
        }

        const lines = editor.document.getText().split('\n');
        const prefix = statement.slice(0, 20);
        const lineNumber = lines.findIndex((line) => line.trim().startsWith(prefix));
        if (lineNumber < 0) {
            return;
        }

        for (const callback of this.resultCallbacks) {
            callback(uri, lineNumber, result);
        }
    }

    private updateContent(): void {
        const html = this.renderHtml();
        if (this.panel) {
            this.panel.webview.html = html;
        }
        if (this.sidebarView) {
            this.sidebarView.webview.html = html;
        }
    }

    private renderHtml(): string {
        const renderMode = vscode.workspace
            .getConfiguration('axioma')
            .get<string>('renderMode', 'both');

        const resultsHtml = this.results
            .map((result, index) => {
                const statusClass =
                    result.status === 'error'
                        ? 'error'
                        : result.status === 'unchanged'
                          ? 'unchanged'
                          : 'ok';
                const latexBlock =
                    (renderMode === 'latex' || renderMode === 'both') && result.latex
                        ? `<div class="latex" id="latex-${index}"></div>`
                        : '';
                const unicodeBlock =
                    (renderMode === 'unicode' || renderMode === 'both') && result.unicode
                        ? `<div class="unicode">${escapeHtml(result.unicode)}</div>`
                        : '';

                return `
                    <div class="result ${statusClass}">
                        <div class="input"><span class="prompt">In[${index + 1}]:</span> <code>${escapeHtml(result.code)}</code></div>
                        <div class="output"><span class="prompt">Out[${index + 1}]:</span></div>
                        ${latexBlock}
                        ${unicodeBlock}
                    </div>
                `;
            })
            .join('');

        const katexCalls = this.results
            .map((result, index) => {
                if (!result.latex || (renderMode !== 'latex' && renderMode !== 'both')) {
                    return '';
                }

                const latex = escapeJsString(result.latex);
                return `
                    try {
                        const target = document.getElementById('latex-${index}');
                        if (target) {
                            katex.render('${latex}', target, { throwOnError: false, displayMode: true });
                        }
                    } catch (error) {
                        const fallback = document.getElementById('latex-${index}');
                        if (fallback) {
                            fallback.textContent = '${latex}';
                        }
                    }
                `;
            })
            .join('\n');

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            padding: 16px;
            font-size: 14px;
        }
        .toolbar {
            margin-bottom: 16px;
            display: flex;
            gap: 8px;
        }
        .input-area {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        .result {
            margin-bottom: 16px;
            padding: 12px;
            border-left: 3px solid var(--vscode-textLink-foreground);
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 4px;
        }
        .result.error {
            border-left-color: var(--vscode-errorForeground);
        }
        .result.unchanged {
            border-left-color: var(--vscode-editorWarning-foreground);
        }
        .input {
            margin-bottom: 8px;
        }
        .prompt {
            color: var(--vscode-textLink-foreground);
            font-weight: 600;
        }
        code {
            font-family: var(--vscode-editor-font-family);
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 4px;
            border-radius: 2px;
        }
        .latex {
            margin: 8px 0;
            padding: 8px;
            text-align: center;
            overflow-x: auto;
        }
        .unicode {
            font-family: var(--vscode-editor-font-family);
            white-space: pre-wrap;
            word-break: break-word;
        }
        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 6px 12px;
            cursor: pointer;
            border-radius: 2px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        input[type="text"] {
            flex: 1;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 6px 8px;
            font-family: var(--vscode-editor-font-family);
            font-size: 14px;
        }
        .empty-state {
            margin-top: 40px;
            text-align: center;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <button onclick="clearAll()">Clear All</button>
    </div>
    <div class="input-area">
        <input type="text" id="codeInput" placeholder="Enter Axioma expression..." onkeydown="if (event.key === 'Enter') evalInput()">
        <button onclick="evalInput()">Evaluate</button>
    </div>
    ${
        resultsHtml ||
        '<div class="empty-state">Press Shift+Enter to evaluate a selection, or type an expression above.</div>'
    }
    <script>
        const vscode = acquireVsCodeApi();
        function evalInput() {
            const input = document.getElementById('codeInput');
            const code = input.value.trim();
            if (code) {
                vscode.postMessage({ type: 'evaluate', code });
                input.value = '';
            }
        }
        function clearAll() {
            vscode.postMessage({ type: 'clear' });
        }
        ${katexCalls}
    </script>
</body>
</html>`;
    }
}

function stringifyField(value: unknown): string {
    if (typeof value === 'string') {
        return value;
    }
    if (value == null) {
        return '';
    }
    return JSON.stringify(value);
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function escapeJsString(text: string): string {
    return text
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/\r/g, '\\r')
        .replace(/\n/g, '\\n');
}
