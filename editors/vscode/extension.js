const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

let client;

function activate(context) {
    const serverOptions = {
        command: 'axioma-lsp',
        transport: TransportKind.stdio
    };
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'axioma' }]
    };
    client = new LanguageClient('axioma', 'Axioma Language Server', serverOptions, clientOptions);
    client.start();
}

function deactivate() {
    if (client) return client.stop();
}

module.exports = { activate, deactivate };
