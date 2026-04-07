import * as vscode from 'vscode';
import { ComputePanel } from './compute';
import { activateLsp, deactivateLsp } from './lsp';
import { McpManager } from './mcp';

let computePanel: ComputePanel | undefined;
let mcpManager: McpManager | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    await activateLsp(context);

    mcpManager = new McpManager(context);
    await mcpManager.start();

    computePanel = new ComputePanel(context, mcpManager);

    context.subscriptions.push(
        computePanel,
        mcpManager,
        vscode.window.registerWebviewViewProvider('axioma.stateView', computePanel),
        vscode.commands.registerCommand('axioma.evaluate', () => evaluateFile()),
        vscode.commands.registerCommand('axioma.evaluateSelection', () => evaluateSelection()),
        vscode.commands.registerCommand('axioma.evaluateLine', () => evaluateLine()),
        vscode.commands.registerCommand('axioma.showPanel', () => computePanel?.show()),
        vscode.commands.registerCommand('axioma.clearPanel', () => computePanel?.clear()),
        vscode.commands.registerCommand('axioma.restartLsp', () => restartLsp(context)),
        vscode.commands.registerCommand('axioma.restartMcp', () => mcpManager?.restart()),
        vscode.commands.registerCommand('axioma.showWorkflows', () => showWorkflows())
    );

    if (vscode.workspace.getConfiguration('axioma').get<boolean>('autoEvaluate', false)) {
        context.subscriptions.push(
            vscode.workspace.onDidSaveTextDocument((doc) => {
                if (doc.languageId === 'axioma') {
                    void evaluateFile();
                }
            })
        );
    }

    registerInlineDecorations(context);
}

export async function deactivate(): Promise<void> {
    computePanel?.dispose();
    computePanel = undefined;
    await mcpManager?.stop();
    mcpManager = undefined;
    await deactivateLsp();
}

async function evaluateFile(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'axioma') {
        return;
    }

    const code = editor.document.getText();
    await computePanel?.evaluate(code, editor.document.uri.toString());
}

async function evaluateSelection(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'axioma') {
        return;
    }

    const selection = editor.selection;
    const code = editor.document.getText(selection.isEmpty ? undefined : selection).trim();
    if (!code) {
        return;
    }

    await computePanel?.evaluate(code, editor.document.uri.toString());
}

async function evaluateLine(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'axioma') {
        return;
    }

    const line = editor.document.lineAt(editor.selection.active.line);
    const code = line.text.trim();
    if (code && !code.startsWith('//')) {
        await computePanel?.evaluate(code, editor.document.uri.toString());
    }
}

async function restartLsp(context: vscode.ExtensionContext): Promise<void> {
    await deactivateLsp();
    await activateLsp(context);
    void vscode.window.showInformationMessage('Axioma language server restarted.');
}

async function showWorkflows(): Promise<void> {
    if (!mcpManager) {
        return;
    }

    const result = await callFirstTool(
        [
            ['axioma_list_workflows', {}],
            ['list_workflows', {}]
        ],
        'Unable to query available workflows from the Axioma MCP server.'
    );

    const available = Array.isArray(result?.available) ? result.available : [];
    if (available.length === 0) {
        void vscode.window.showInformationMessage('No Axioma workflows are available.');
        return;
    }

    const items: vscode.QuickPickItem[] = available.map(
        (workflow: { goal: string; description?: string }) => ({
        label: workflow.goal,
        description: workflow.description
    })
    );

    const picked = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select a workflow to see its steps'
    });
    if (!picked) {
        return;
    }

    const detail = await callFirstTool(
        [
            ['axioma_workflow', { goal: picked.label }],
            ['workflow', { goal: picked.label }]
        ],
        `Unable to load workflow details for '${picked.label}'.`
    );

    if (Array.isArray(detail?.steps)) {
        const steps = detail.steps
            .map((step: { tool: string; description: string }, index: number) =>
                `${index + 1}. ${step.tool}: ${step.description}`
            )
            .join('\n');
        await vscode.window.showInformationMessage(`Workflow: ${picked.label}\n\n${steps}`, {
            modal: true
        });
    }
}

async function callFirstTool(
    attempts: Array<[string, Record<string, unknown>]>,
    failureMessage: string
): Promise<any> {
    if (!mcpManager) {
        return undefined;
    }

    let lastError: unknown;
    for (const [name, args] of attempts) {
        try {
            return await mcpManager.callTool(name, args);
        } catch (error) {
            lastError = error;
        }
    }

    void vscode.window.showWarningMessage(
        `${failureMessage} ${lastError instanceof Error ? lastError.message : String(lastError)}`
    );
    return undefined;
}

function registerInlineDecorations(context: vscode.ExtensionContext): void {
    const resultDecorationType = vscode.window.createTextEditorDecorationType({
        after: {
            color: new vscode.ThemeColor('editorCodeLens.foreground'),
            fontStyle: 'italic',
            margin: '0 0 0 2em'
        }
    });

    context.subscriptions.push(resultDecorationType);

    computePanel?.onResult((uri: string, line: number, result: string) => {
        const editor = vscode.window.visibleTextEditors.find(
            (visibleEditor) => visibleEditor.document.uri.toString() === uri
        );
        if (!editor || line < 0 || line >= editor.document.lineCount) {
            return;
        }

        const documentLine = editor.document.lineAt(line);
        const range = new vscode.Range(line, 0, line, documentLine.text.length);
        editor.setDecorations(resultDecorationType, [
            {
                range,
                renderOptions: {
                    after: { contentText: `  -> ${result}` }
                }
            }
        ]);
    });
}
