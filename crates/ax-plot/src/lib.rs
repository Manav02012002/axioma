#![forbid(unsafe_code)]

use ax_ir::Expr;
use num_traits::ToPrimitive;
use std::collections::HashMap;

const WIDTH: f64 = 800.0;
const HEIGHT: f64 = 500.0;
const PLOT_LEFT: f64 = 60.0;
const PLOT_RIGHT: f64 = 780.0;
const PLOT_TOP: f64 = 30.0;
const PLOT_BOTTOM: f64 = 460.0;
const SAMPLE_COUNT: usize = 500;

fn eval_numeric(
    expr: &Expr,
    bindings: &HashMap<lasso::Spur, f64>,
    interner: &ax_ir::Interner,
) -> Option<f64> {
    match expr {
        Expr::Int(n) => n.to_f64(),
        Expr::Rational(r) => Some(r.numer().to_f64()? / r.denom().to_f64()?),
        Expr::Float(f) => Some(*f),
        Expr::Sym(sym) => bindings.get(sym).copied(),
        Expr::Add(terms) => {
            let mut acc = 0.0;
            for term in terms {
                acc += eval_numeric(term, bindings, interner)?;
            }
            Some(acc)
        }
        Expr::Mul(factors) => {
            let mut acc = 1.0;
            for factor in factors {
                acc *= eval_numeric(factor, bindings, interner)?;
            }
            Some(acc)
        }
        Expr::Pow(base, exp) => Some(
            eval_numeric(base, bindings, interner)?.powf(eval_numeric(exp, bindings, interner)?),
        ),
        Expr::Neg(inner) => Some(-eval_numeric(inner, bindings, interner)?),
        Expr::Call(f, args) if args.len() == 1 => {
            let arg = eval_numeric(&args[0], bindings, interner)?;
            match interner.resolve(*f) {
                "exp" => Some(arg.exp()),
                "log" | "ln" => Some(arg.ln()),
                "sin" => Some(arg.sin()),
                "cos" => Some(arg.cos()),
                "tan" => Some(arg.tan()),
                "sqrt" => Some(arg.sqrt()),
                "abs" => Some(arg.abs()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn format_tick(value: f64) -> String {
    if value.abs() < 1e-9 {
        "0".to_string()
    } else if value.abs() >= 1_000.0 || (value.abs() < 0.01 && value != 0.0) {
        format!("{value:.2e}")
    } else {
        let mut s = format!("{value:.3}");
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    }
}

fn to_svg_coords(
    x: f64,
    y: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> (f64, f64) {
    let sx = PLOT_LEFT + (x - x_min) / (x_max - x_min) * (PLOT_RIGHT - PLOT_LEFT);
    let sy = PLOT_BOTTOM - (y - y_min) / (y_max - y_min) * (PLOT_BOTTOM - PLOT_TOP);
    (sx, sy)
}

fn sample_function(
    expr: &Expr,
    var: lasso::Spur,
    min: f64,
    max: f64,
    interner: &ax_ir::Interner,
) -> Vec<(f64, f64)> {
    let mut bindings = HashMap::new();
    let mut points = Vec::with_capacity(SAMPLE_COUNT);
    for i in 0..SAMPLE_COUNT {
        let t = if SAMPLE_COUNT <= 1 {
            0.0
        } else {
            i as f64 / (SAMPLE_COUNT - 1) as f64
        };
        let x = min + t * (max - min);
        bindings.insert(var, x);
        if let Some(y) = eval_numeric(expr, &bindings, interner) {
            if y.is_finite() {
                points.push((x, y));
            }
        }
    }
    points
}

fn sample_parametric(
    x_expr: &Expr,
    y_expr: &Expr,
    t_var: lasso::Spur,
    t_min: f64,
    t_max: f64,
    interner: &ax_ir::Interner,
) -> Vec<(f64, f64)> {
    let mut bindings = HashMap::new();
    let mut points = Vec::with_capacity(SAMPLE_COUNT);
    for i in 0..SAMPLE_COUNT {
        let t = if SAMPLE_COUNT <= 1 {
            0.0
        } else {
            i as f64 / (SAMPLE_COUNT - 1) as f64
        };
        let value = t_min + t * (t_max - t_min);
        bindings.insert(t_var, value);
        let Some(x) = eval_numeric(x_expr, &bindings, interner) else {
            continue;
        };
        let Some(y) = eval_numeric(y_expr, &bindings, interner) else {
            continue;
        };
        if x.is_finite() && y.is_finite() {
            points.push((x, y));
        }
    }
    points
}

fn compute_bounds(data_sets: &[Vec<(f64, f64)>]) -> (f64, f64, f64, f64) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for data in data_sets {
        for (x, y) in data {
            x_min = x_min.min(*x);
            x_max = x_max.max(*x);
            y_min = y_min.min(*y);
            y_max = y_max.max(*y);
        }
    }

    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = -1.0;
        x_max = 1.0;
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    }
    if (x_max - x_min).abs() < 1e-9 {
        x_min -= 1.0;
        x_max += 1.0;
    }
    if (y_max - y_min).abs() < 1e-9 {
        y_min -= 1.0;
        y_max += 1.0;
    }

    let y_pad = (y_max - y_min).abs() * 0.1;
    let x_pad = (x_max - x_min).abs() * 0.05;
    (
        x_min - x_pad,
        x_max + x_pad,
        y_min - y_pad,
        y_max + y_pad,
    )
}

fn polyline_segments(
    points: &[(f64, f64)],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Vec<Vec<(f64, f64)>> {
    if points.is_empty() {
        return Vec::new();
    }

    let y_range = (y_max - y_min).abs().max(1e-9);
    let threshold = y_range * 0.5;
    let mut segments = Vec::new();
    let mut current = Vec::new();
    let mut prev_y: Option<f64> = None;

    for &(x, y) in points {
        if let Some(prev) = prev_y {
            if (y - prev).abs() > threshold && !current.is_empty() {
                if current.len() >= 2 {
                    segments.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
        current.push(to_svg_coords(x, y, x_min, x_max, y_min, y_max));
        prev_y = Some(y);
    }

    if current.len() >= 2 {
        segments.push(current);
    }

    segments
}

fn append_axes(svg: &mut String, x_min: f64, x_max: f64, y_min: f64, y_max: f64) {
    svg.push_str(&format!(
        r#"<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>"#
    ));

    for i in 0..=5 {
        let t = i as f64 / 5.0;
        let x = x_min + t * (x_max - x_min);
        let y = y_min + t * (y_max - y_min);
        let (sx, _) = to_svg_coords(x, y_min, x_min, x_max, y_min, y_max);
        let (_, sy) = to_svg_coords(x_min, y, x_min, x_max, y_min, y_max);

        svg.push_str(&format!(
            r##"<line x1="{sx:.2}" y1="{PLOT_TOP:.2}" x2="{sx:.2}" y2="{PLOT_BOTTOM:.2}" stroke="#e6e6e6" stroke-width="1"/>"##
        ));
        svg.push_str(&format!(
            r##"<line x1="{PLOT_LEFT:.2}" y1="{sy:.2}" x2="{PLOT_RIGHT:.2}" y2="{sy:.2}" stroke="#e6e6e6" stroke-width="1"/>"##
        ));
        svg.push_str(&format!(
            r##"<text x="{sx:.2}" y="478" text-anchor="middle" font-size="12" fill="#555">{}</text>"##,
            format_tick(x)
        ));
        svg.push_str(&format!(
            r##"<text x="54" y="{:.2}" text-anchor="end" dominant-baseline="middle" font-size="12" fill="#555">{}</text>"##,
            sy,
            format_tick(y)
        ));
    }

    if y_min <= 0.0 && 0.0 <= y_max {
        let (_, y0) = to_svg_coords(x_min, 0.0, x_min, x_max, y_min, y_max);
        svg.push_str(&format!(
            r##"<line x1="{PLOT_LEFT:.2}" y1="{y0:.2}" x2="{PLOT_RIGHT:.2}" y2="{y0:.2}" stroke="#888" stroke-width="1.5"/>"##
        ));
    }
    if x_min <= 0.0 && 0.0 <= x_max {
        let (x0, _) = to_svg_coords(0.0, y_min, x_min, x_max, y_min, y_max);
        svg.push_str(&format!(
            r##"<line x1="{x0:.2}" y1="{PLOT_TOP:.2}" x2="{x0:.2}" y2="{PLOT_BOTTOM:.2}" stroke="#888" stroke-width="1.5"/>"##
        ));
    }
}

fn render_plot(
    title: &str,
    series: &[(&[(f64, f64)], &str, &str)],
    explicit_bounds: Option<(f64, f64, f64, f64)>,
) -> String {
    let all_data = series.iter().map(|(points, _, _)| points.to_vec()).collect::<Vec<_>>();
    let (x_min, x_max, y_min, y_max) = explicit_bounds.unwrap_or_else(|| compute_bounds(&all_data));

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">"#
    );
    append_axes(&mut svg, x_min, x_max, y_min, y_max);
    svg.push_str(&format!(
        r##"<text x="400" y="20" text-anchor="middle" font-size="16" font-family="monospace" fill="#222">{}</text>"##,
        escape_xml(title)
    ));

    for (points, color, _) in series {
        for segment in polyline_segments(points, x_min, x_max, y_min, y_max) {
            let polyline = segment
                .iter()
                .map(|(x, y)| format!("{x:.2},{y:.2}"))
                .collect::<Vec<_>>()
                .join(" ");
            svg.push_str(&format!(
                r#"<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>"#
            ));
        }
    }

    if series.len() > 1 {
        let legend_x = 620.0;
        let legend_y = 40.0;
        let legend_h = 24.0 + series.len() as f64 * 18.0;
        svg.push_str(&format!(
            r##"<rect x="{legend_x:.2}" y="{legend_y:.2}" width="145" height="{legend_h:.2}" fill="#ffffffcc" stroke="#ccc"/>"##
        ));
        for (idx, (_, color, label)) in series.iter().enumerate() {
            let y = legend_y + 18.0 + idx as f64 * 18.0;
            svg.push_str(&format!(
                r#"<line x1="{:.2}" y1="{y:.2}" x2="{:.2}" y2="{y:.2}" stroke="{color}" stroke-width="3"/>"#,
                legend_x + 10.0,
                legend_x + 34.0
            ));
            svg.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-size="12" dominant-baseline="middle" fill="#222">{}</text>"##,
                legend_x + 42.0,
                y,
                escape_xml(label)
            ));
        }
    }

    svg.push_str("</svg>");
    svg
}

pub fn plot_2d(
    expr: &Expr,
    var: lasso::Spur,
    x_min: f64,
    x_max: f64,
    interner: &ax_ir::Interner,
) -> String {
    let points = sample_function(expr, var, x_min, x_max, interner);
    let title = ax_ir::pretty_print(expr, interner);
    render_plot(&title, &[(&points, "#2563eb", &title)], None)
}

pub fn plot_multi(
    exprs: &[Expr],
    var: lasso::Spur,
    x_min: f64,
    x_max: f64,
    interner: &ax_ir::Interner,
) -> String {
    let colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed"];
    let titles = exprs
        .iter()
        .map(|expr| ax_ir::pretty_print(expr, interner))
        .collect::<Vec<_>>();
    let points = exprs
        .iter()
        .map(|expr| sample_function(expr, var, x_min, x_max, interner))
        .collect::<Vec<_>>();
    let series = points
        .iter()
        .zip(titles.iter())
        .enumerate()
        .map(|(idx, (pts, title))| (pts.as_slice(), colors[idx % colors.len()], title.as_str()))
        .collect::<Vec<_>>();
    render_plot("Multiple Functions", &series, None)
}

pub fn plot_parametric(
    x_expr: &Expr,
    y_expr: &Expr,
    t_var: lasso::Spur,
    t_min: f64,
    t_max: f64,
    interner: &ax_ir::Interner,
) -> String {
    let points = sample_parametric(x_expr, y_expr, t_var, t_min, t_max, interner);
    let title = format!(
        "({}, {})",
        ax_ir::pretty_print(x_expr, interner),
        ax_ir::pretty_print(y_expr, interner)
    );
    render_plot(&title, &[(&points, "#2563eb", &title)], None)
}

pub fn plot_data(points: &[(f64, f64)], title: &str) -> String {
    let plot_points = points.to_vec();
    let (x_min, x_max, y_min, y_max) = compute_bounds(std::slice::from_ref(&plot_points));
    let mut svg = render_plot(title, &[(&plot_points, "#2563eb", title)], Some((x_min, x_max, y_min, y_max)));
    let insert_at = svg.rfind("</svg>").unwrap_or(svg.len());
    let mut overlays = String::new();
    for &(x, y) in points {
        let (sx, sy) = to_svg_coords(x, y, x_min, x_max, y_min, y_max);
        overlays.push_str(&format!(
            r##"<circle cx="{sx:.2}" cy="{sy:.2}" r="3" fill="#dc2626"/>"##
        ));
    }
    svg.insert_str(insert_at, &overlays);
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_generates_valid_svg() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = Expr::pow(Expr::Sym(x), Expr::Int(2.into()));
        let svg = plot_2d(&expr, x, -5.0, 5.0, &interner);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("polyline"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn plot_handles_discontinuity() {
        let interner = ax_ir::Interner::new();
        let x = interner.get_or_intern("x");
        let expr = Expr::pow(Expr::Sym(x), Expr::Int((-1).into()));
        let svg = plot_2d(&expr, x, -5.0, 5.0, &interner);
        assert!(svg.contains("<svg"));
    }
}
