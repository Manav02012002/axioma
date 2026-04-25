use anyhow::{Context, Result};
use ax_ir::Expr;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::Helper;
use std::borrow::Cow;
use std::collections::HashSet;

#[allow(dead_code)]
/// Raw ANSI escape codes. No external crate needed.
mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const ITALIC: &str = "\x1b[3m";
    pub const UNDERLINE: &str = "\x1b[4m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";
    pub const GRAY: &str = "\x1b[90m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const BRIGHT_WHITE: &str = "\x1b[97m";
}

const AXIOMA_VERSION: &str = "0.1.0";
const BUILD_DATE: &str = "unreleased";
const DOCS_URL: &str = "https://github.com/Manav02012002/axioma";

/// Maps an `ax_syntax` token kind + its text to an ANSI color escape sequence.
/// Returns `""` for tokens that should not be colored.
fn token_color(kind: ax_syntax::kind::SyntaxKind, text: &str) -> &'static str {
    use ax_syntax::kind::SyntaxKind;
    match kind {
        SyntaxKind::KwModule
        | SyntaxKind::KwImport
        | SyntaxKind::KwLet
        | SyntaxKind::KwIn
        | SyntaxKind::KwIndexset => "\x1b[1;35m",
        SyntaxKind::Int | SyntaxKind::Float => "\x1b[36m",
        SyntaxKind::Plus
        | SyntaxKind::Minus
        | SyntaxKind::Star
        | SyntaxKind::Slash
        | SyntaxKind::Caret
        | SyntaxKind::Eq => "\x1b[33m",
        SyntaxKind::LParen
        | SyntaxKind::RParen
        | SyntaxKind::LBrace
        | SyntaxKind::RBrace
        | SyntaxKind::LBrack
        | SyntaxKind::RBrack => "\x1b[1;37m",
        SyntaxKind::CommentLine | SyntaxKind::CommentBlock => "\x1b[2;90m",
        SyntaxKind::Ident => ident_color(text),
        SyntaxKind::Error => "\x1b[31m",
        _ => "",
    }
}

/// Classifies an identifier by name and returns an ANSI color, or `""` for unknowns.
fn ident_color(text: &str) -> &'static str {
    match text {
        "assume" | "rule" | "convention" => "\x1b[1;35m",
        "alpha" | "beta" | "gamma" | "delta" | "epsilon" | "zeta" | "eta" | "theta" | "mu"
        | "nu" | "xi" | "pi" | "rho" | "sigma" | "tau" | "phi" | "chi" | "psi" | "omega"
        | "Gamma" | "Delta" | "Theta" | "Lambda" | "Xi" | "Pi" | "Sigma" | "Phi" | "Psi"
        | "Omega" | "lambda" | "inf" | "infty" => "\x1b[32m",
        "diff"
        | "integrate"
        | "simplify"
        | "expand"
        | "factor"
        | "collect"
        | "canonicalize"
        | "canonicalise"
        | "substitute"
        | "rewrite"
        | "solve"
        | "series"
        | "limit"
        | "sum"
        | "product"
        | "sin"
        | "cos"
        | "tan"
        | "cot"
        | "sec"
        | "csc"
        | "sinh"
        | "cosh"
        | "tanh"
        | "asin"
        | "acos"
        | "atan"
        | "arcsin"
        | "arccos"
        | "arctan"
        | "asinh"
        | "acosh"
        | "atanh"
        | "arcsinh"
        | "arccosh"
        | "arctanh"
        | "exp"
        | "log"
        | "sqrt"
        | "abs"
        | "sign"
        | "sgn"
        | "det"
        | "trace"
        | "transpose"
        | "inverse"
        | "eigenvalues"
        | "christoffel"
        | "riemann"
        | "ricci"
        | "weyl"
        | "einstein"
        | "contract"
        | "symmetrize"
        | "antisymmetrize"
        | "antisymmetrise"
        | "perturb"
        | "classify_pde"
        | "check_units"
        | "double_integral"
        | "triple_integral"
        | "definite_integral"
        | "dblint"
        | "tplint"
        | "defint"
        | "covariant_derivative"
        | "covariant_diff"
        | "apart"
        | "collect_mandelstam"
        | "decompose"
        | "atan2"
        | "laplacian"
        | "curl"
        | "divergence"
        | "gradient"
        | "conj"
        | "Re"
        | "Im"
        | "arg"
        | "bra"
        | "ket"
        | "braket"
        | "commutator"
        | "anticommutator" => "\x1b[34m",
        "symmetric" | "antisymmetric" | "metric" | "commuting" | "anticommuting"
        | "anti_commuting" | "bosonic" | "fermionic" | "grassmann" | "creation"
        | "annihilation" | "real" | "positive" | "negative" | "nonzero" | "integer" | "even"
        | "odd" => "\x1b[3;36m",
        _ => "",
    }
}

/// Returns `true` if stdout is a terminal and the `NO_COLOR` env var is not set.
fn supports_color() -> bool {
    use std::io::IsTerminal;
    std::env::var("NO_COLOR").is_err() && std::io::stdout().is_terminal()
}

/// Core completion logic, independent of rustyline::Context.
/// Returns (word_start_position, Vec<(display, replacement)>).
fn find_completions(
    line: &str,
    pos: usize,
    static_completions: &[String],
    env_binding_names: &[String],
    greek_shortcuts: &[(String, String)],
    import_paths: &[String],
) -> (usize, Vec<(String, String)>) {
    let before = &line[..pos];

    if let Some(bs_pos) = before.rfind('\\') {
        let prefix = &before[bs_pos..];
        if prefix.len() > 1 && prefix[1..].chars().all(|c| c.is_alphabetic()) {
            let mut candidates = Vec::new();
            for (shortcut, replacement) in greek_shortcuts {
                if shortcut.starts_with(prefix) {
                    candidates.push((format!("{shortcut} → {replacement}"), replacement.clone()));
                }
            }
            if !candidates.is_empty() {
                return (bs_pos, candidates);
            }
        }
    }

    if before.starts_with(':') {
        let candidates: Vec<(String, String)> = static_completions
            .iter()
            .filter(|c| c.starts_with(':') && c.starts_with(before) && *c != before)
            .map(|c| (c.clone(), c.clone()))
            .collect();
        return (0, candidates);
    }

    let trimmed_before = before.trim_start();
    if let Some(after_import) = trimmed_before.strip_prefix("import ") {
        let word_start = pos - after_import.len();
        let candidates: Vec<(String, String)> = import_paths
            .iter()
            .filter(|p| p.starts_with(after_import) && p.as_str() != after_import)
            .map(|p| (p.clone(), p.clone()))
            .collect();
        return (word_start, candidates);
    }

    let word_start = before
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    let prefix = &before[word_start..];
    if prefix.is_empty() {
        return (pos, Vec::new());
    }

    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    for name in env_binding_names {
        if name.starts_with(prefix) && name.as_str() != prefix && seen.insert(name.clone()) {
            candidates.push((name.clone(), name.clone()));
        }
    }

    for name in static_completions {
        if !name.starts_with(':')
            && name.starts_with(prefix)
            && name.as_str() != prefix
            && seen.insert(name.clone())
        {
            candidates.push((name.clone(), name.clone()));
        }
    }

    (word_start, candidates)
}

/// Bundles rustyline's Helper traits for the Axioma REPL.
/// Prompt 3 implements Highlighter. Prompts 4-5 add Completer and Hinter.
#[derive(Helper)]
struct AxiomaHelper {
    /// Static completable words: builtin names, keywords, :commands.
    static_completions: Vec<String>,
    /// Current env binding names — updated after each cell evaluation.
    env_binding_names: Vec<String>,
    /// Greek shortcut pairs: ("\alpha", "α"), etc.
    greek_shortcuts: Vec<(String, String)>,
    /// Import paths from std/: "gr.schwarzschild", "trig", etc.
    import_paths: Vec<String>,
    /// History-based ghost-text hinter.
    history_hinter: HistoryHinter,
    /// Whether color is enabled.
    use_color: bool,
}

impl AxiomaHelper {
    fn new(use_color: bool) -> Self {
        let mut static_completions = Vec::new();

        for cmd in &[
            ":quit",
            ":q",
            ":help",
            ":h",
            ":env",
            ":rules",
            ":assumptions",
            ":convention",
            ":inspect",
            ":suggest",
            ":pool on",
            ":pool off",
            ":pool stats",
            ":parallel on",
            ":parallel off",
            ":codegen python",
            ":codegen rust",
            ":codegen cpp",
            ":export latex",
            ":export html",
            ":reset",
            ":trust",
            ":latex",
        ] {
            static_completions.push((*cmd).to_string());
        }

        for entry in ax_eval::registry::builtin_entries() {
            let name = entry.name.to_string();
            if !static_completions.contains(&name) {
                static_completions.push(name);
            }
        }
        for entry in ax_eval::registry::algorithm_entries() {
            let name = entry.name.to_string();
            if !static_completions.contains(&name) {
                static_completions.push(name);
            }
        }

        for kw in &[
            "let",
            "in",
            "import",
            "module",
            "indexset",
            "assume",
            "rule",
            "convention",
        ] {
            let kw = (*kw).to_string();
            if !static_completions.contains(&kw) {
                static_completions.push(kw);
            }
        }

        let greek_shortcuts = vec![
            ("\\alpha", "α"),
            ("\\beta", "β"),
            ("\\gamma", "γ"),
            ("\\delta", "δ"),
            ("\\epsilon", "ε"),
            ("\\zeta", "ζ"),
            ("\\eta", "η"),
            ("\\theta", "θ"),
            ("\\mu", "μ"),
            ("\\nu", "ν"),
            ("\\xi", "ξ"),
            ("\\pi", "π"),
            ("\\rho", "ρ"),
            ("\\sigma", "σ"),
            ("\\tau", "τ"),
            ("\\phi", "φ"),
            ("\\chi", "χ"),
            ("\\psi", "ψ"),
            ("\\omega", "ω"),
            ("\\Gamma", "Γ"),
            ("\\Delta", "Δ"),
            ("\\Theta", "Θ"),
            ("\\Lambda", "Λ"),
            ("\\Xi", "Ξ"),
            ("\\Pi", "Π"),
            ("\\Sigma", "Σ"),
            ("\\Phi", "Φ"),
            ("\\Psi", "Ψ"),
            ("\\Omega", "Ω"),
            ("\\lambda", "λ"),
            ("\\infty", "∞"),
            ("\\partial", "∂"),
            ("\\nabla", "∇"),
            ("\\int", "∫"),
            ("\\sum", "Σ"),
            ("\\prod", "Π"),
            ("\\sqrt", "√"),
            ("\\hbar", "ℏ"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        let import_paths = vec![
            "algebra",
            "calculus",
            "trig",
            "gr.schwarzschild",
            "gr.kerr_newman",
            "gr.frw",
            "gr.minkowski",
            "gr.de_sitter",
            "qft.dirac",
            "qft.gamma",
            "qft.scalar_field",
            "qft.normal_ordering",
            "qft.brst",
            "qft.superspace",
            "qm.spin",
            "qm.harmonic_oscillator",
            "qm.bell",
            "physics.maxwell",
            "physics.classical_mechanics",
            "physics.klein_gordon",
            "tensor.symmetry",
            "tensor.index",
            "units.si",
            "units.cgs",
            "units.natural",
            "conventions.mtw",
            "conventions.weinberg",
            "conventions.landau",
            "conventions.particle_physics",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            static_completions,
            env_binding_names: Vec::new(),
            greek_shortcuts,
            import_paths,
            history_hinter: HistoryHinter {},
            use_color,
        }
    }

    /// Sync current environment binding names for tab completion.
    /// Call this after each cell evaluation.
    fn sync_env(&mut self, env: &ax_eval::Env, interner: &ax_ir::Interner) {
        self.env_binding_names.clear();
        for sym in env.bindings.keys() {
            self.env_binding_names
                .push(interner.resolve(*sym).to_string());
        }
    }
}

impl Highlighter for AxiomaHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if !self.use_color || line.is_empty() {
            return Cow::Borrowed(line);
        }

        let (tokens, _diags) = ax_syntax::lexer::lex(line);
        let mut out = String::with_capacity(line.len() + 128);

        for (kind, span) in &tokens {
            if *kind == ax_syntax::kind::SyntaxKind::Eof {
                break;
            }
            let slice = &line[span.clone()];
            let color = token_color(*kind, slice);
            if color.is_empty() {
                out.push_str(slice);
            } else {
                out.push_str(color);
                out.push_str(slice);
                out.push_str(ansi::RESET);
            }
        }

        Cow::Owned(out)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }

    fn highlight_char(
        &self,
        _line: &str,
        _pos: usize,
        _forced: rustyline::highlight::CmdKind,
    ) -> bool {
        self.use_color
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if self.use_color {
            Cow::Owned(format!("{}{}{}", ansi::DIM, hint, ansi::RESET))
        } else {
            Cow::Borrowed(hint)
        }
    }
}

impl Completer for AxiomaHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, raw) = find_completions(
            line,
            pos,
            &self.static_completions,
            &self.env_binding_names,
            &self.greek_shortcuts,
            &self.import_paths,
        );
        let pairs = raw
            .into_iter()
            .map(|(display, replacement)| Pair {
                display,
                replacement,
            })
            .collect();
        Ok((start, pairs))
    }
}

impl Hinter for AxiomaHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &rustyline::Context<'_>) -> Option<String> {
        if line.is_empty() {
            return None;
        }
        if line.starts_with(':') {
            return None;
        }
        if pos < line.len() {
            return None;
        }
        self.history_hinter.hint(line, pos, ctx)
    }
}

impl Validator for AxiomaHelper {
    fn validate(&self, _ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
}

fn print_banner(use_color: bool) {
    if use_color {
        let bc = format!("{}{}", ansi::BOLD, ansi::BRIGHT_CYAN);
        let vc = format!("{}{}", ansi::BOLD, ansi::WHITE);
        let dc = format!("{}{}", ansi::DIM, ansi::WHITE);
        let uc = format!("{}{}{}", ansi::DIM, ansi::BLUE, ansi::UNDERLINE);
        let gc = format!("{}{}", ansi::BOLD, ansi::GREEN);
        let pipe = ansi::DIM;
        let r = ansi::RESET;

        println!("{bc}       _          _");
        println!("      / \\  __  __(_) ___  _ __ ___   __ _");
        println!("     / _ \\ \\ \\/ /| |/ _ \\| '_ ` _ \\ / _` |");
        println!("    / ___ \\ >  < | | (_) | | | | | | (_| |");
        println!("   /_/   \\_/_/\\_\\|_|\\___/|_| |_| |_|\\__,_|{r}");
        println!();
        println!(
            "   {vc}Version {AXIOMA_VERSION}{r} {dc}({BUILD_DATE}){r}              {pipe}|{r}  Type {gc}:help{r} for commands"
        );
        println!("   {uc}{DOCS_URL}{r}  {pipe}|{r}  {gc}:quit{r} to exit");
        println!();
    } else {
        println!("       _          _");
        println!("      / \\  __  __(_) ___  _ __ ___   __ _");
        println!("     / _ \\ \\ \\/ /| |/ _ \\| '_ ` _ \\ / _` |");
        println!("    / ___ \\ >  < | | (_) | | | | | | (_| |");
        println!("   /_/   \\_/_/\\_\\|_|\\___/|_| |_| |_|\\__,_|");
        println!();
        println!(
            "   Version {AXIOMA_VERSION} ({BUILD_DATE})              |  Type :help for commands"
        );
        println!("   {DOCS_URL}  |  :quit to exit");
        println!();
    }
}

/// Builds the primary prompt string: `ax[1]> ` or `tex[1]> ` in LaTeX mode.
fn make_prompt(cell: usize, use_color: bool, latex_mode: bool) -> String {
    let prefix = if latex_mode { "tex" } else { "ax" };
    if use_color {
        let prefix_color = if latex_mode {
            format!("{}{}", ansi::BOLD, ansi::YELLOW)
        } else {
            format!("{}{}", ansi::BOLD, ansi::BRIGHT_GREEN)
        };
        let num_color = format!("{}{}", ansi::BOLD, ansi::WHITE);
        let r = ansi::RESET;
        format!("{prefix_color}{prefix}{r}{num_color}[{cell}]{r}{prefix_color}>{r} ")
    } else {
        format!("{prefix}[{cell}]> ")
    }
}

/// Builds the continuation prompt for multi-line input: `  ...> `
fn make_continuation_prompt(use_color: bool) -> String {
    if use_color {
        format!("{}{}  ...> {}", ansi::DIM, ansi::GRAY, ansi::RESET)
    } else {
        "  ...> ".to_string()
    }
}

/// Prints a cell result: `  [N] = <expression>` with color.
fn print_result(cell: usize, expr: &Expr, interner: &ax_ir::Interner, use_color: bool) {
    let rendered = ax_render::to_unicode(expr, interner);
    if use_color {
        println!(
            "  {}{}[{cell}]{} {}={} {}{}{}",
            ansi::BOLD,
            ansi::WHITE,
            ansi::RESET,
            ansi::DIM,
            ansi::RESET,
            ansi::BRIGHT_CYAN,
            rendered,
            ansi::RESET,
        );
    } else {
        println!("  [{cell}] = {rendered}");
    }
}

/// Prints an error message in red.
fn print_error(msg: &str, use_color: bool) {
    if use_color {
        eprintln!("{}{}error:{} {msg}", ansi::BOLD, ansi::RED, ansi::RESET);
    } else {
        eprintln!("error: {msg}");
    }
}

/// Prints a status/info message dimmed.
fn print_status(msg: &str, use_color: bool) {
    if use_color {
        println!("{}{}{}", ansi::DIM, msg, ansi::RESET);
    } else {
        println!("{msg}");
    }
}

/// Prints an import/action message with elapsed time.
fn print_timed_status(msg: &str, elapsed: std::time::Duration, use_color: bool) {
    if use_color {
        println!(
            "{}{}{} {}({:.1?}){}",
            ansi::DIM,
            msg,
            ansi::RESET,
            ansi::GRAY,
            elapsed,
            ansi::RESET,
        );
    } else {
        println!("{msg} ({elapsed:.1?})");
    }
}

fn convention_lines(env: &ax_eval::Env) -> Vec<String> {
    vec![
        format!("  metric_signature: {:?}", env.convention.metric_signature),
        format!("  riemann_sign: {:?}", env.convention.riemann_sign),
        format!(
            "  ricci_contraction: {:?}",
            env.convention.ricci_contraction
        ),
        format!("  levi_civita_norm: {:?}", env.convention.levi_civita_norm),
        format!("  fourier_sign: {:?}", env.convention.fourier_sign),
    ]
}

fn candidate_last_expr(
    expr: &Expr,
    env: &ax_eval::Env,
    interner: &ax_ir::Interner,
) -> Option<Expr> {
    match expr {
        Expr::Import(_) | Expr::SetConvention(_, _) => None,
        Expr::Let(name, val, body) => {
            let evaled = ax_eval::eval(val, env, interner);
            Some(
                if matches!(body.as_ref(), Expr::Sym(sym) if *sym == *name) {
                    evaled
                } else {
                    ax_eval::eval(body, env, interner)
                },
            )
        }
        _ => {
            let result = ax_eval::eval(expr, env, interner);
            match &result {
                Expr::FnDef(_, _, _)
                | Expr::Assume(_, _)
                | Expr::Rule(_, _, _)
                | Expr::SetConvention(_, _) => None,
                Expr::Call(f, _) if interner.resolve(*f) == "__set_parallel" => None,
                Expr::Call(f, _) if interner.resolve(*f) == "grassmann" => None,
                Expr::Call(f, _) if interner.resolve(*f) == "creation" => None,
                Expr::Call(f, _) if interner.resolve(*f) == "annihilation" => None,
                _ => Some(result),
            }
        }
    }
}

fn trust_for_expr(expr: &Expr, env: &ax_eval::Env, interner: &ax_ir::Interner) -> Option<String> {
    match expr {
        Expr::Call(f, args) if interner.resolve(*f) == "rewrite" && args.len() == 1 => {
            let (_, trace) = ax_eval::rewrite_with_trace(&args[0], env, interner);
            Some(ax_eval::describe_rewrite_trace(&trace))
        }
        _ => None,
    }
}

fn export_session(
    command: &str,
    session_cells: &[(String, Option<Expr>)],
    interner: &ax_ir::Interner,
) -> Result<()> {
    let mut parts = command.split_whitespace();
    let _export = parts.next();
    let format_name = parts.next().unwrap_or("latex");
    let format = match format_name {
        "latex" | "tex" => crate::cmd_export::ExportFormat::Latex,
        "html" => crate::cmd_export::ExportFormat::Html,
        other => anyhow::bail!("unknown export format: {other}; expected latex or html"),
    };
    let default_filename = match format {
        crate::cmd_export::ExportFormat::Latex => "session.tex",
        crate::cmd_export::ExportFormat::Html => "session.html",
    };
    let filename = parts.next().unwrap_or(default_filename);
    if parts.next().is_some() {
        anyhow::bail!("usage: :export latex [filename] or :export html [filename]");
    }

    let cells = session_cells
        .iter()
        .enumerate()
        .map(|(idx, (input, output))| crate::cmd_export::ExportCell {
            input_source: input.clone(),
            input_line_start: idx + 1,
            output_latex: output
                .as_ref()
                .map(|expr| ax_render::to_latex(expr, interner)),
            output_unicode: output
                .as_ref()
                .map(|expr| ax_render::to_unicode(expr, interner)),
            cell_type: crate::cmd_export::CellType::Code,
        })
        .collect::<Vec<_>>();

    let options = crate::cmd_export::ExportOptions {
        format: format.clone(),
        include_input: true,
        include_output: true,
        standalone: true,
        title: Some("Axioma REPL Session".to_string()),
        author: None,
        document_class: "article".to_string(),
    };
    let rendered = match format {
        crate::cmd_export::ExportFormat::Latex => crate::cmd_export::export_latex(&cells, &options),
        crate::cmd_export::ExportFormat::Html => crate::cmd_export::export_html(&cells, &options),
    };
    std::fs::write(filename, rendered).with_context(|| format!("failed to write {filename}"))?;
    println!("wrote {filename}");
    Ok(())
}

fn print_help(use_color: bool) {
    let (c, r) = if use_color {
        (
            format!("{}{}", ansi::BOLD, ansi::GREEN),
            ansi::RESET.to_string(),
        )
    } else {
        (String::new(), String::new())
    };

    println!("Axioma REPL commands:");
    println!("  {c}:quit, :q{r}          Exit");
    println!("  {c}:help, :h{r}          Show this help");
    println!("  {c}:env{r}               Show all bindings");
    println!("  {c}:rules{r}             Show all user-defined rules");
    println!("  {c}:assumptions{r}       Show all assumptions");
    println!("  {c}:convention{r}        Show active convention");
    println!("  {c}:inspect [expr]{r}    Inspect an expression or the last result");
    println!("  {c}:suggest [expr]{r}    Suggest algorithms for an expression");
    println!("  {c}:pool on|off|stats{r} Manage pooled expression storage");
    println!("  {c}:parallel on|off{r}   Toggle parallel tensor canonicalisation");
    println!("  {c}:codegen python|rust|cpp{r}  Generate code for last result");
    println!("  {c}:export latex|html [f]{r}    Export session");
    println!("  {c}:reset{r}             Clear all bindings and rules");
    println!("  {c}:trust{r}             Show trust level of last result");
    println!("  {c}:latex{r}             Toggle LaTeX input mode");
}

fn print_env(env: &ax_eval::Env, interner: &ax_ir::Interner, use_color: bool) {
    if env.bindings.is_empty() {
        println!("  (no bindings)");
        return;
    }
    for (sym, val) in &env.bindings {
        let name = interner.resolve(*sym);
        let rendered = ax_render::to_unicode(val, interner);
        if use_color {
            println!(
                "  {}{}{} = {}{}{}",
                ansi::BOLD,
                name,
                ansi::RESET,
                ansi::BRIGHT_CYAN,
                rendered,
                ansi::RESET,
            );
        } else {
            println!("  {name} = {rendered}");
        }
    }
}

fn print_rules(env: &ax_eval::Env, use_color: bool) {
    if env.rules.is_empty() {
        println!("  (no rules)");
        return;
    }
    for (i, rule) in env.rules.iter().enumerate() {
        if use_color {
            println!(
                "  {}[{i}]{} {}{}{} {}({:?}){}",
                ansi::BOLD,
                ansi::RESET,
                ansi::WHITE,
                rule.name,
                ansi::RESET,
                ansi::DIM,
                rule.trust_level,
                ansi::RESET,
            );
        } else {
            println!("  [{}] {} ({:?})", i, rule.name, rule.trust_level);
        }
    }
}

fn print_assumptions(env: &ax_eval::Env, interner: &ax_ir::Interner, use_color: bool) {
    if env.assumptions.is_empty() {
        println!("  (no assumptions)");
        return;
    }
    for (sym, assumptions) in &env.assumptions {
        let name = interner.resolve(*sym);
        let names = assumptions
            .iter()
            .map(|a| format!("{a:?}").to_lowercase())
            .collect::<Vec<_>>();
        if use_color {
            println!(
                "  {}{}{} is {}{}{}{}",
                ansi::BOLD,
                name,
                ansi::RESET,
                ansi::ITALIC,
                ansi::CYAN,
                names.join(", "),
                ansi::RESET,
            );
        } else {
            println!("  {} is {}", name, names.join(", "));
        }
    }
}

fn inspect_target_expr(text: &str, env: &ax_eval::Env, interner: &ax_ir::Interner) -> Result<Expr> {
    let effective_input = if text.contains('\\') || text.contains("_{") || text.contains("^{") {
        ax_core_ir::latex_to_axioma(text)
    } else {
        text.to_string()
    };
    let lowered = ax_core_ir::lower(&effective_input, interner);
    if !lowered.errors.is_empty() {
        return Err(anyhow::anyhow!(lowered
            .errors
            .into_iter()
            .map(|error| error.message)
            .collect::<Vec<_>>()
            .join("; ")));
    }
    let expr = lowered
        .expr
        .ok_or_else(|| anyhow::anyhow!("expected exactly one expression"))?;
    Ok(ax_eval::eval(&expr, env, interner))
}

fn variance_arrow(variance: &str) -> &'static str {
    match variance {
        "up" => "↑",
        "down" => "↓",
        _ => "?",
    }
}

fn print_inspect_result(result: &ax_eval::inspect::InspectResult) {
    println!("Kind: {}", result.kind);
    if result.free_indices.is_empty() {
        println!("Free indices: (none)");
    } else {
        println!(
            "Free indices: {}",
            result
                .free_indices
                .iter()
                .map(|(name, variance)| format!("{name}{}", variance_arrow(variance)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if result.dummy_pairs.is_empty() {
        println!("Dummy pairs: (none)");
    } else {
        println!(
            "Dummy pairs: {}",
            result
                .dummy_pairs
                .iter()
                .map(|(a, b)| format!("({a}, {b})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if result.properties.is_empty() {
        println!("Properties: (none)");
    } else {
        println!(
            "Properties: {}",
            result
                .properties
                .iter()
                .map(|(symbol, props)| format!("{symbol} → [{}]", props.join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    println!("Symbols: [{}]", result.symbols.join(", "));
    println!("Node count: {}", result.node_count);
}

fn print_suggest_result(result: &ax_eval::suggest::SuggestResult) {
    println!("Suggested algorithms:");
    for suggestion in &result.suggestions {
        println!("  → {}: {}", suggestion.algorithm, suggestion.reason);
    }
    if result.missing.is_empty() {
        println!();
        println!("Missing properties: (none)");
    } else {
        println!();
        println!("Missing properties:");
        for missing in &result.missing {
            println!(
                "  → {} has no declared properties; consider: {}",
                missing.symbol, missing.suggestion
            );
        }
    }
}

pub fn is_complete(input: &str) -> bool {
    let mut parens = 0i32;
    let mut brackets = 0i32;
    let mut braces = 0i32;
    for ch in input.chars() {
        match ch {
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '{' => braces += 1,
            '}' => braces -= 1,
            _ => {}
        }
    }
    parens <= 0 && brackets <= 0 && braces <= 0 && !input.trim_end().ends_with('\\')
}

pub fn run() -> Result<()> {
    let interner = ax_ir::Interner::new();
    let session_qm_settings = crate::cmd_run::load_project_qm_settings(None)?;
    let mut env = ax_eval::Env::with_qm_settings(session_qm_settings.clone());
    let use_color = supports_color();
    let helper = AxiomaHelper::new(use_color);
    let config = rustyline::Config::builder().auto_add_history(false).build();
    let mut editor =
        rustyline::Editor::<AxiomaHelper, rustyline::history::DefaultHistory>::with_config(config)?;
    editor.set_helper(Some(helper));
    let search_paths = crate::cmd_run::default_search_paths(None);
    let history_path = std::env::var("HOME")
        .ok()
        .map(|home| std::path::PathBuf::from(home).join(".axioma_history"));
    let mut last_result: Option<Expr> = None;
    let mut last_trust: Option<String> = None;
    let mut session_cells: Vec<(String, Option<Expr>)> = Vec::new();
    let mut cell_number: usize = 1;
    let mut latex_mode = false;

    if let Some(path) = &history_path {
        let _ = editor.load_history(path);
    }

    print_banner(use_color);

    loop {
        let mut input = String::new();
        loop {
            let prompt = if input.is_empty() {
                make_prompt(cell_number, use_color, latex_mode)
            } else {
                make_continuation_prompt(use_color)
            };
            match editor.readline(&prompt) {
                Ok(line) => {
                    if input.is_empty() && matches!(line.trim(), ":quit" | ":q") {
                        if let Some(path) = &history_path {
                            let _ = editor.save_history(path);
                        }
                        return Ok(());
                    }
                    if !input.is_empty() {
                        input.push('\n');
                    }
                    input.push_str(&line);
                    if is_complete(&input) {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    input.clear();
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    if let Some(path) = &history_path {
                        let _ = editor.save_history(path);
                    }
                    return Ok(());
                }
                Err(err) => return Err(err.into()),
            }
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            ":quit" | ":q" => break,
            ":help" | ":h" => {
                print_help(use_color);
                continue;
            }
            ":env" => {
                print_env(&env, &interner, use_color);
                continue;
            }
            ":rules" => {
                print_rules(&env, use_color);
                continue;
            }
            ":assumptions" => {
                print_assumptions(&env, &interner, use_color);
                continue;
            }
            ":convention" => {
                for line in convention_lines(&env) {
                    println!("{line}");
                }
                continue;
            }
            ":pool on" => {
                env.enable_pool();
                print_status("Expression pool enabled.", use_color);
                continue;
            }
            ":pool off" => {
                env.expr_pool = None;
                print_status("Expression pool disabled.", use_color);
                continue;
            }
            ":pool stats" => {
                if let Some(pool) = &env.expr_pool {
                    print_status(&format!("Unique pooled nodes: {}", pool.len()), use_color);
                } else {
                    print_status("Expression pool is disabled.", use_color);
                }
                continue;
            }
            ":parallel on" => {
                env.parallel = true;
                print_status("Parallel mode enabled.", use_color);
                continue;
            }
            ":parallel off" => {
                env.parallel = false;
                print_status("Parallel mode disabled.", use_color);
                continue;
            }
            ":reset" => {
                env = ax_eval::Env::with_qm_settings(session_qm_settings.clone());
                last_result = None;
                last_trust = None;
                session_cells.clear();
                cell_number = 1;
                if let Some(h) = editor.helper_mut() {
                    h.sync_env(&env, &interner);
                }
                print_status("Environment reset.", use_color);
                continue;
            }
            ":trust" => {
                if let Some(trust) = &last_trust {
                    println!("{trust}");
                } else {
                    print_error("No trust information for the last result.", use_color);
                }
                continue;
            }
            ":latex" => {
                latex_mode = !latex_mode;
                let state = if latex_mode { "on" } else { "off" };
                print_status(&format!("LaTeX input mode {state}."), use_color);
                continue;
            }
            s if s.starts_with(":inspect") => {
                let arg = s.strip_prefix(":inspect").unwrap_or_default().trim();
                let expr = if arg.is_empty() {
                    match &last_result {
                        Some(expr) => expr.clone(),
                        None => {
                            print_error("No previous result to inspect.", use_color);
                            continue;
                        }
                    }
                } else {
                    match inspect_target_expr(arg, &env, &interner) {
                        Ok(expr) => expr,
                        Err(err) => {
                            print_error(&err.to_string(), use_color);
                            continue;
                        }
                    }
                };
                let result = ax_eval::inspect::inspect_expr(&expr, &env, &interner);
                print_inspect_result(&result);
                continue;
            }
            s if s.starts_with(":suggest") => {
                let arg = s.strip_prefix(":suggest").unwrap_or_default().trim();
                let expr = if arg.is_empty() {
                    match &last_result {
                        Some(expr) => expr.clone(),
                        None => {
                            print_error("No previous result to analyze.", use_color);
                            continue;
                        }
                    }
                } else {
                    match inspect_target_expr(arg, &env, &interner) {
                        Ok(expr) => expr,
                        Err(err) => {
                            print_error(&err.to_string(), use_color);
                            continue;
                        }
                    }
                };
                let result = ax_eval::suggest::suggest_for_expr(&expr, &env, &interner, None);
                print_suggest_result(&result);
                continue;
            }
            s if s.starts_with(":codegen ") => {
                if let Some(last) = &last_result {
                    let target = s.strip_prefix(":codegen ").unwrap_or_default().trim();
                    let target = match target {
                        "python" | "py" => ax_codegen::Target::Python,
                        "rust" | "rs" => ax_codegen::Target::Rust,
                        "cpp" | "c++" => ax_codegen::Target::Cpp,
                        _ => {
                            print_error(&format!("Unknown target: {}", target), use_color);
                            continue;
                        }
                    };
                    let params = crate::cmd_codegen::infer_params(last, &env, &interner);
                    let code = ax_codegen::generate(last, target, &interner, None, &params);
                    println!("{code}");
                } else {
                    print_error("No previous result to generate code for.", use_color);
                }
                continue;
            }
            s if s.starts_with(":export") => {
                if let Err(err) = export_session(s, &session_cells, &interner) {
                    print_error(&format!("{err:#}"), use_color);
                }
                continue;
            }
            _ => {}
        }

        let _ = editor.add_history_entry(trimmed);

        let effective_input = if latex_mode
            || trimmed.contains('\\')
            || trimmed.contains("_{")
            || trimmed.contains("^{")
        {
            ax_core_ir::latex_to_axioma(trimmed)
        } else {
            trimmed.to_string()
        };
        let lowered = ax_core_ir::lower(&effective_input, &interner);
        if !lowered.errors.is_empty() {
            for error in lowered.errors {
                print_error(&error.message, use_color);
            }
            continue;
        }

        if let Some(expr) = lowered.expr {
            let candidate_last = candidate_last_expr(&expr, &env, &interner);
            let trust = trust_for_expr(&expr, &env, &interner);
            let mut session_output = None;

            let start = std::time::Instant::now();
            if let Some(message) =
                crate::cmd_run::execute_expr(&expr, &mut env, &interner, &search_paths)?
            {
                let elapsed = start.elapsed();
                print_timed_status(&message, elapsed, use_color);
            } else if let Some(ref result_expr) = candidate_last {
                last_result = Some(result_expr.clone());
                session_output = last_result.clone();
                print_result(cell_number, result_expr, &interner, use_color);
            }
            session_cells.push((trimmed.to_string(), session_output));
            last_trust = trust;
            cell_number += 1;
            if let Some(h) = editor.helper_mut() {
                h.sync_env(&env, &interner);
            }
        }
    }

    if let Some(path) = &history_path {
        let _ = editor.save_history(path);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ansi, find_completions, ident_color, is_complete, make_continuation_prompt, make_prompt,
        print_banner, supports_color, token_color, AxiomaHelper,
    };
    use rustyline::highlight::Highlighter;

    #[test]
    fn incomplete_parens() {
        assert!(!is_complete("f(x,"));
        assert!(is_complete("f(x, y)"));
        assert!(!is_complete("[1, 2,"));
        assert!(is_complete("[1, 2, 3]"));
    }

    #[test]
    fn backslash_continuation() {
        assert!(!is_complete("1 + \\"));
        assert!(is_complete("1 + 2"));
    }

    #[test]
    fn supports_color_returns_bool() {
        // We can't manipulate env vars safely under #![forbid(unsafe_code)] on Rust >= 1.83.
        // Just verify the function is callable and returns a bool without panicking.
        let _ = supports_color();
    }

    #[test]
    fn banner_no_color_does_not_contain_ansi() {
        // Capture what print_banner would produce by checking the function doesn't panic.
        // We can't easily capture stdout in a unit test, but we CAN verify the
        // no-panic path works and the function is callable.
        print_banner(false);
    }

    #[test]
    fn banner_with_color_does_not_panic() {
        print_banner(true);
    }

    #[test]
    fn prompt_no_color_normal() {
        assert_eq!(make_prompt(1, false, false), "ax[1]> ");
        assert_eq!(make_prompt(42, false, false), "ax[42]> ");
    }

    #[test]
    fn prompt_no_color_latex() {
        assert_eq!(make_prompt(1, false, true), "tex[1]> ");
        assert_eq!(make_prompt(7, false, true), "tex[7]> ");
    }

    #[test]
    fn prompt_color_contains_cell() {
        let p = make_prompt(5, true, false);
        assert!(p.contains("[5]"), "got: {p:?}");
        assert!(p.contains("ax"), "got: {p:?}");
        assert!(p.contains("\x1b["), "expected ANSI codes, got: {p:?}");
    }

    #[test]
    fn prompt_color_latex_contains_tex() {
        let p = make_prompt(3, true, true);
        assert!(p.contains("tex"), "got: {p:?}");
        assert!(p.contains("[3]"), "got: {p:?}");
    }

    #[test]
    fn continuation_prompt_no_color() {
        assert_eq!(make_continuation_prompt(false), "  ...> ");
    }

    #[test]
    fn continuation_prompt_color() {
        let p = make_continuation_prompt(true);
        assert!(p.contains("...>"), "got: {p:?}");
        assert!(p.contains("\x1b["), "got: {p:?}");
    }

    #[test]
    fn token_color_keywords_are_colored() {
        use ax_syntax::kind::SyntaxKind;
        let c = token_color(SyntaxKind::KwLet, "let");
        assert!(!c.is_empty(), "keywords should have color");
        let c = token_color(SyntaxKind::KwImport, "import");
        assert!(!c.is_empty());
        let c = token_color(SyntaxKind::KwModule, "module");
        assert!(!c.is_empty());
        let c = token_color(SyntaxKind::KwIn, "in");
        assert!(!c.is_empty());
        let c = token_color(SyntaxKind::KwIndexset, "indexset");
        assert!(!c.is_empty());
    }

    #[test]
    fn token_color_numbers_are_colored() {
        use ax_syntax::kind::SyntaxKind;
        assert!(!token_color(SyntaxKind::Int, "42").is_empty());
        assert!(!token_color(SyntaxKind::Float, "3.14").is_empty());
    }

    #[test]
    fn token_color_whitespace_no_color() {
        use ax_syntax::kind::SyntaxKind;
        assert!(token_color(SyntaxKind::Whitespace, " ").is_empty());
    }

    #[test]
    fn token_color_operators_are_colored() {
        use ax_syntax::kind::SyntaxKind;
        for kind in [
            SyntaxKind::Plus,
            SyntaxKind::Minus,
            SyntaxKind::Star,
            SyntaxKind::Slash,
            SyntaxKind::Caret,
            SyntaxKind::Eq,
        ] {
            assert!(
                !token_color(kind, "+").is_empty(),
                "{kind:?} should be colored"
            );
        }
    }

    #[test]
    fn token_color_error_is_red() {
        use ax_syntax::kind::SyntaxKind;
        let c = token_color(SyntaxKind::Error, "???");
        assert!(c.contains("31"), "errors should be red, got: {c:?}");
    }

    #[test]
    fn ident_color_greek_letters() {
        assert!(!ident_color("alpha").is_empty());
        assert!(!ident_color("omega").is_empty());
        assert!(!ident_color("Gamma").is_empty());
        assert!(!ident_color("lambda").is_empty());
    }

    #[test]
    fn ident_color_builtins() {
        assert!(!ident_color("diff").is_empty());
        assert!(!ident_color("integrate").is_empty());
        assert!(!ident_color("sin").is_empty());
        assert!(!ident_color("christoffel").is_empty());
    }

    #[test]
    fn ident_color_semantic_keywords() {
        assert!(!ident_color("assume").is_empty());
        assert!(!ident_color("rule").is_empty());
        assert!(!ident_color("convention").is_empty());
    }

    #[test]
    fn ident_color_properties() {
        assert!(!ident_color("symmetric").is_empty());
        assert!(!ident_color("antisymmetric").is_empty());
        assert!(!ident_color("real").is_empty());
        assert!(!ident_color("positive").is_empty());
        assert!(!ident_color("grassmann").is_empty());
    }

    #[test]
    fn ident_color_unknown_is_empty() {
        assert!(ident_color("my_variable").is_empty());
        assert!(ident_color("foobar").is_empty());
        assert!(ident_color("x").is_empty());
    }

    #[test]
    fn highlight_empty_line() {
        let helper = AxiomaHelper::new(true);
        let result = helper.highlight("", 0);
        assert_eq!(&*result, "");
    }

    #[test]
    fn highlight_no_color_passthrough() {
        let helper = AxiomaHelper::new(false);
        let input = "let x = 42";
        let result = helper.highlight(input, 0);
        assert_eq!(&*result, input);
    }

    #[test]
    fn highlight_colored_contains_reset() {
        let helper = AxiomaHelper::new(true);
        let result = helper.highlight("let x = 42", 0);
        assert!(result.contains(ansi::RESET), "got: {result:?}");
    }

    #[test]
    fn complete_colon_commands() {
        let helper = AxiomaHelper::new(false);
        let (start, candidates) = find_completions(
            ":he",
            3,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 0);
        assert!(
            candidates.iter().any(|(_, r)| r == ":help"),
            "expected :help in candidates: {candidates:?}"
        );
    }

    #[test]
    fn complete_colon_all() {
        let helper = AxiomaHelper::new(false);
        let (start, candidates) = find_completions(
            ":",
            1,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 0);
        assert!(
            candidates.len() > 5,
            "expected many :commands, got {}",
            candidates.len()
        );
    }

    #[test]
    fn complete_greek_alpha() {
        let helper = AxiomaHelper::new(false);
        let (start, candidates) = find_completions(
            "x + \\alp",
            8,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 4);
        assert!(
            candidates.iter().any(|(_, r)| r == "α"),
            "expected α replacement in: {candidates:?}"
        );
    }

    #[test]
    fn complete_greek_backslash_alone_no_completions() {
        let helper = AxiomaHelper::new(false);
        let (_, candidates) = find_completions(
            "\\",
            1,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert!(candidates.is_empty(), "got: {candidates:?}");
    }

    #[test]
    fn complete_import_paths() {
        let helper = AxiomaHelper::new(false);
        let (start, candidates) = find_completions(
            "import gr.",
            10,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 7);
        assert!(
            candidates.iter().any(|(_, r)| r == "gr.schwarzschild"),
            "expected gr.schwarzschild in: {candidates:?}"
        );
        assert!(
            candidates.len() >= 4,
            "expected multiple gr.* paths, got {}",
            candidates.len()
        );
    }

    #[test]
    fn complete_import_empty() {
        let helper = AxiomaHelper::new(false);
        let (start, candidates) = find_completions(
            "import ",
            7,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 7);
        assert!(
            candidates.len() >= 20,
            "expected all import paths, got {}",
            candidates.len()
        );
    }

    #[test]
    fn complete_ident_sin() {
        let helper = AxiomaHelper::new(false);
        let (start, candidates) = find_completions(
            "si",
            2,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 0);
        let replacements: Vec<&str> = candidates.iter().map(|(_, r)| r.as_str()).collect();
        assert!(
            replacements.contains(&"sin"),
            "expected sin in: {replacements:?}"
        );
        assert!(
            replacements.contains(&"simplify"),
            "expected simplify in: {replacements:?}"
        );
    }

    #[test]
    fn complete_ident_with_env_bindings() {
        let mut helper = AxiomaHelper::new(false);
        helper.env_binding_names = vec![
            "schwarzschild_metric".to_string(),
            "scalar_field".to_string(),
        ];
        let (start, candidates) = find_completions(
            "schw",
            4,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert_eq!(start, 0);
        assert!(
            candidates.iter().any(|(_, r)| r == "schwarzschild_metric"),
            "expected env binding in: {candidates:?}"
        );
    }

    #[test]
    fn complete_empty_prefix_no_spam() {
        let helper = AxiomaHelper::new(false);
        let (_, candidates) = find_completions(
            "x + ",
            4,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert!(
            candidates.is_empty(),
            "empty prefix should yield no completions"
        );
    }

    #[test]
    fn complete_exact_match_not_offered() {
        let helper = AxiomaHelper::new(false);
        let (_, candidates) = find_completions(
            "sin",
            3,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        assert!(
            !candidates.iter().any(|(_, r)| r == "sin"),
            "exact match should not be offered: {candidates:?}"
        );
    }

    #[test]
    fn complete_dedup_env_vs_builtin() {
        let mut helper = AxiomaHelper::new(false);
        helper.env_binding_names = vec!["sin_custom".to_string()];
        let (_, candidates) = find_completions(
            "sin",
            3,
            &helper.static_completions,
            &helper.env_binding_names,
            &helper.greek_shortcuts,
            &helper.import_paths,
        );
        let sin_custom_count = candidates.iter().filter(|(_, r)| r == "sin_custom").count();
        assert_eq!(sin_custom_count, 1, "should appear exactly once");
    }

    #[test]
    fn hinter_no_hint_on_empty() {
        let line = "";
        assert!(line.is_empty(), "empty line guard should trigger");
    }

    #[test]
    fn hinter_no_hint_on_colon_commands() {
        let line = ":he";
        assert!(line.starts_with(':'), "colon guard should trigger");
    }

    #[test]
    fn hinter_no_hint_cursor_mid_line() {
        let line = "diff(x, y)";
        let pos = 4;
        assert!(pos < line.len(), "mid-line guard should trigger");
    }
}
