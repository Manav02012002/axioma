import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export async function activateLsp(context: vscode.ExtensionContext): Promise<void> {
    const config = vscode.workspace.getConfiguration('axioma');
    const lspPath = config.get<string>('lspPath', 'axioma-lsp');

    const serverOptions: ServerOptions = {
        command: lspPath,
        transport: TransportKind.stdio
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'axioma' },
            { scheme: 'untitled', language: 'axioma' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{ax,axioma}')
        },
        outputChannelName: 'Axioma Language Server'
    };

    client = new LanguageClient(
        'axioma',
        'Axioma Language Server',
        serverOptions,
        clientOptions
    );

    try {
        await client.start();
        context.subscriptions.push({
            dispose: () => {
                void deactivateLsp();
            }
        });
    } catch (error) {
        void vscode.window.showWarningMessage(
            `Failed to start Axioma language server at '${lspPath}'. ` +
                'Make sure axioma-lsp is installed and in your PATH, or set axioma.lspPath in settings. ' +
                `Error: ${error instanceof Error ? error.message : String(error)}`
        );
    }
}

export async function deactivateLsp(): Promise<void> {
    if (client) {
        await client.stop();
        client = undefined;
    }
}
