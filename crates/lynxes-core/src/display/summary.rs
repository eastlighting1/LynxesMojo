use arrow_array::Array;

pub(crate) fn ellipsis_right(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= width {
        return value.to_owned();
    }
    if width == 1 {
        return "…".to_owned();
    }
    chars[..width - 1].iter().collect::<String>() + "…"
}

pub(crate) fn ellipsis_middle(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= width {
        return value.to_owned();
    }
    if width <= 2 {
        return ellipsis_right(value, width);
    }
    let left = (width - 1) / 2;
    let right = width - 1 - left;
    let mut out = String::new();
    out.extend(chars[..left].iter());
    out.push('…');
    out.extend(chars[chars.len() - right..].iter());
    out
}

pub(crate) fn format_cell_value(array: &dyn Array, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    let rendered = arrow::util::display::array_value_to_string(array, row)
        .unwrap_or_else(|_| "<invalid>".to_owned());
    Some(rendered.replace('\n', "\\n"))
}

pub(crate) fn join_attr_tokens(tokens: Vec<String>, width: Option<usize>) -> String {
    if tokens.is_empty() {
        return "-".to_owned();
    }

    let full = tokens.join(", ");
    let Some(limit) = width else {
        return full;
    };
    if full.chars().count() <= limit {
        return full;
    }
    if limit <= 12 {
        return ellipsis_right(&full, limit);
    }

    let mut out = String::new();
    for token in tokens {
        let candidate = if out.is_empty() {
            token
        } else {
            format!("{out}, {token}")
        };
        if candidate.chars().count() > limit.saturating_sub(1) {
            break;
        }
        out = candidate;
    }
    if out.is_empty() {
        ellipsis_right(&full, limit)
    } else {
        out.push('…');
        out
    }
}
