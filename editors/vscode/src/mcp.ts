import * as cp from 'child_process';
import * as readline from 'readline';
import * as vscode from 'vscode';

type PendingRequest = {
    resolve: (value: unknown) => void;
    reject: (reason?: unknown) => void;
};

export class McpManager implements vscode.Disposable {
    private process: cp.ChildProcessWithoutNullStreams | undefined;
    private rl: readline.Interface | undefined;
    private pendingRequests = new Map<number, PendingRequest>();
    private nextId = 1;
    private readonly outputChannel: vscode.OutputChannel;

    constructor(private readonly context: vscode.ExtensionContext) {
        this.outputChannel = vscode.window.createOutputChannel('Axioma Compute Server');
        this.context.subscriptions.push(this.outputChannel);
    }

    async start(): Promise<void> {
        if (this.process) {
            return;
        }

        const config = vscode.workspace.getConfiguration('axioma');
        const mcpPath = config.get<string>('mcpPath', 'axioma-mcp');
        const timeout = config.get<number>('mcpTimeout', 60);

        try {
            this.process = cp.spawn(mcpPath, ['--timeout', timeout.toString()], {
                stdio: 'pipe'
            });

            this.process.stderr.on('data', (data: Buffer) => {
                this.outputChannel.appendLine(`[stderr] ${data.toString().trim()}`);
            });

            this.process.on('exit', (code, signal) => {
                this.outputChannel.appendLine(
                    `MCP server exited with code ${code ?? 'null'}${signal ? ` signal ${signal}` : ''}`
                );
                this.rejectAllPending(new Error('MCP server exited'));
                this.process = undefined;
            });

            this.process.on('error', (error) => {
                this.outputChannel.appendLine(`Failed to start MCP server: ${error.message}`);
                this.rejectAllPending(error);
                void vscode.window.showWarningMessage(
                    `Failed to start Axioma compute server at '${mcpPath}'. ` +
                        'Make sure axioma-mcp is installed and in your PATH, or set axioma.mcpPath in settings. ' +
                        `Error: ${error.message}`
                );
            });

            this.rl = readline.createInterface({ input: this.process.stdout });
            this.rl.on('line', (line: string) => this.handleLine(line));

            await this.sendRequest('initialize', {});
            this.outputChannel.appendLine('MCP server initialized.');
        } catch (error) {
            this.outputChannel.appendLine(
                `Failed to initialize MCP server: ${error instanceof Error ? error.message : String(error)}`
            );
        }
    }

    async stop(): Promise<void> {
        this.rl?.close();
        this.rl = undefined;

        if (this.process) {
            this.process.kill();
            this.process = undefined;
        }

        this.rejectAllPending(new Error('MCP server stopped'));
    }

    async restart(): Promise<void> {
        await this.stop();
        await this.start();
        void vscode.window.showInformationMessage('Axioma compute server restarted.');
    }

    async sendRequest(method: string, params: Record<string, unknown>): Promise<any> {
        if (!this.process?.stdin) {
            throw new Error('MCP server not running');
        }

        const id = this.nextId++;
        const request = JSON.stringify({
            jsonrpc: '2.0',
            id,
            method,
            params
        });

        return await new Promise((resolve, reject) => {
            const timeoutSeconds = vscode.workspace
                .getConfiguration('axioma')
                .get<number>('mcpTimeout', 60);
            const timer = setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error(`MCP request timed out after ${timeoutSeconds}s`));
                }
            }, timeoutSeconds * 1000);

            this.pendingRequests.set(id, {
                resolve: (value) => {
                    clearTimeout(timer);
                    resolve(value);
                },
                reject: (reason) => {
                    clearTimeout(timer);
                    reject(reason);
                }
            });

            this.process?.stdin.write(`${request}\n`);
        });
    }

    async callTool(name: string, args: Record<string, unknown>): Promise<any> {
        return await this.sendRequest('tools/call', { name, arguments: args });
    }

    async evaluate(code: string): Promise<any> {
        return await this.callTool('axioma_eval', { code });
    }

    dispose(): void {
        void this.stop();
        this.outputChannel.dispose();
    }

    private handleLine(line: string): void {
        try {
            const response = JSON.parse(line) as {
                id?: number;
                error?: { message?: string };
                result?: unknown;
            };

            if (response.id == null) {
                return;
            }

            const pending = this.pendingRequests.get(response.id);
            if (!pending) {
                return;
            }

            this.pendingRequests.delete(response.id);
            if (response.error) {
                pending.reject(
                    new Error(response.error.message ?? JSON.stringify(response.error))
                );
                return;
            }

            pending.resolve(response.result);
        } catch {
            this.outputChannel.appendLine(`Failed to parse MCP response: ${line}`);
        }
    }

    private rejectAllPending(error: Error): void {
        for (const [, pending] of this.pendingRequests) {
            pending.reject(error);
        }
        this.pendingRequests.clear();
    }
}
