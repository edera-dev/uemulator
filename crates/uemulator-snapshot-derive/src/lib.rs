//! `#[derive(Snapshot)]`: saves and restores a struct's or enum's fields in declaration order
//! through `crate::snapshot` (`crates/uemulator/src/snapshot.rs`, `docs/plans/29-snapshots.md`).
//! Only the `uemulator` crate uses it, so the generated code names `crate::snapshot` directly.
//!
//! Field attributes:
//! - `#[snapshot(skip)]`: not saved; a restore leaves the field as the freshly built machine has
//!   it. For host state (handles, callbacks, the JIT) and caches the restore empties anyway.
//! - `#[snapshot(fresh)]`: restored with `Snapshot::load_new` instead of in place, for data whose
//!   shape the snapshot decides (a queue whose length differs from the new machine's).
//!
//! Container attribute `#[snapshot(new)]` also derives `load_new`, so the type can be built from
//! a snapshot (needed inside a `Vec` that grows, an `Option` that is `None` in the new machine, or
//! a `fresh` field); its skipped fields are then `Default::default()`. Enums always have it: a
//! restore builds the saved variant.
//!
//! Every saved struct and enum writes a 32-bit tag first, a hash of its name and its saved fields'
//! (or variants') names, so a snapshot from a build whose layout differs fails at the first type
//! that changed instead of misreading everything after it. A saved field under `#[cfg]` is
//! refused: the snapshot would differ between targets, and the native and WebAssembly engines
//! must read each other's.
//!
//! The parser handles what the library writes: named, tuple and unit structs, and enums with
//! unit, tuple and named variants, optionally generic. It does not know Rust's grammar beyond
//! that, which is why it has no dependencies.

use proc_macro::{Delimiter, Group, Spacing, TokenStream, TokenTree};

#[proc_macro_derive(Snapshot, attributes(snapshot))]
pub fn derive_snapshot(input: TokenStream) -> TokenStream {
    match derive(input) {
        Ok(output) => output,
        Err(message) => compile_error(&message),
    }
}

fn compile_error(message: &str) -> TokenStream {
    format!("compile_error!({message:?});").parse().unwrap()
}

#[derive(Default)]
struct Options {
    skip: bool,
    fresh: bool,
    new: bool,
}

struct Field {
    /// The member: a name, or an index for a tuple struct or variant.
    member: String,
    options: Options,
    /// `#[cfg(...)]` attributes, as tokens.
    cfgs: Vec<TokenStream>,
}

enum Fields {
    Named(Vec<Field>),
    Tuple(Vec<Field>),
    Unit,
}

struct Variant {
    name: String,
    fields: Fields,
    cfgs: Vec<TokenStream>,
}

enum Body {
    Struct(Fields),
    Enum(Vec<Variant>),
}

struct Item {
    name: String,
    options: Options,
    /// The generic parameters as declared (with bounds), and their names alone.
    params: Vec<TokenStream>,
    param_names: Vec<String>,
    type_params: Vec<String>,
    where_clause: Vec<TokenTree>,
    body: Body,
}

/// Reads `#[...]` attributes: `snapshot(...)` options and `cfg(...)`; doc comments and the rest are
/// ignored.
fn attributes(tokens: &[TokenTree], at: &mut usize) -> Result<(Options, Vec<TokenStream>), String> {
    let mut options = Options::default();
    let mut cfgs = Vec::new();
    while let Some(TokenTree::Punct(punct)) = tokens.get(*at) {
        if punct.as_char() != '#' {
            break;
        }
        let Some(TokenTree::Group(group)) = tokens.get(*at + 1) else {
            return Err("expected an attribute after `#`".into());
        };
        *at += 2;
        let inner: Vec<TokenTree> = group.stream().into_iter().collect();
        match inner.first() {
            Some(TokenTree::Ident(ident)) if ident.to_string() == "snapshot" => {
                let Some(TokenTree::Group(arguments)) = inner.get(1) else {
                    return Err("expected #[snapshot(...)]".into());
                };
                for token in arguments.stream() {
                    match token {
                        TokenTree::Ident(word) => match word.to_string().as_str() {
                            "skip" => options.skip = true,
                            "fresh" => options.fresh = true,
                            "new" => options.new = true,
                            other => return Err(format!("unknown snapshot option `{other}`")),
                        },
                        TokenTree::Punct(comma) if comma.as_char() == ',' => {}
                        other => return Err(format!("unexpected `{other}` in #[snapshot]")),
                    }
                }
            }
            Some(TokenTree::Ident(ident)) if ident.to_string() == "cfg" => {
                cfgs.push(group.stream());
            }
            _ => {}
        }
    }
    Ok((options, cfgs))
}

/// Skips `pub`, `pub(crate)`, `pub(super)`, `pub(in path)`.
fn visibility(tokens: &[TokenTree], at: &mut usize) {
    if let Some(TokenTree::Ident(ident)) = tokens.get(*at)
        && ident.to_string() == "pub"
    {
        *at += 1;
        if let Some(TokenTree::Group(group)) = tokens.get(*at)
            && group.delimiter() == Delimiter::Parenthesis
        {
            *at += 1;
        }
    }
}

/// Splits on commas outside angle brackets. Groups are single tokens already; `<` and `>` are
/// not, so generic arguments (`BTreeMap<u32, u64>`) must not split. The `>` of `->` is not a
/// closing bracket.
fn split_commas(tokens: &[TokenTree]) -> Vec<Vec<TokenTree>> {
    let mut parts = vec![Vec::new()];
    let mut depth = 0i32;
    let mut after_dash = false;
    for token in tokens {
        let mut dash = false;
        if let TokenTree::Punct(punct) = token {
            match punct.as_char() {
                '<' => depth += 1,
                '>' if !after_dash => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(Vec::new());
                    continue;
                }
                '-' => dash = punct.spacing() == Spacing::Joint,
                _ => {}
            }
        }
        after_dash = dash;
        parts.last_mut().unwrap().push(token.clone());
    }
    parts.retain(|part| !part.is_empty());
    parts
}

fn fields(group: &Group) -> Result<Fields, String> {
    let tokens: Vec<TokenTree> = group.stream().into_iter().collect();
    let named = group.delimiter() == Delimiter::Brace;
    let mut fields = Vec::new();
    for (index, part) in split_commas(&tokens).into_iter().enumerate() {
        let mut at = 0;
        let (options, cfgs) = attributes(&part, &mut at)?;
        visibility(&part, &mut at);
        let member = if named {
            match part.get(at) {
                Some(TokenTree::Ident(ident)) => ident.to_string(),
                _ => return Err("expected a field name".into()),
            }
        } else {
            index.to_string()
        };
        if options.skip && options.fresh {
            return Err(format!(
                "field `{member}`: skip and fresh exclude each other"
            ));
        }
        if !options.skip && !cfgs.is_empty() {
            return Err(format!(
                "field `{member}` is saved but exists only under #[cfg]: snapshots would differ \
                 between targets; skip it or save it everywhere"
            ));
        }
        fields.push(Field {
            member,
            options,
            cfgs,
        });
    }
    Ok(if named {
        Fields::Named(fields)
    } else {
        Fields::Tuple(fields)
    })
}

fn parse(input: TokenStream) -> Result<Item, String> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let mut at = 0;
    let (options, _) = attributes(&tokens, &mut at)?;
    visibility(&tokens, &mut at);
    let kind = match tokens.get(at) {
        Some(TokenTree::Ident(ident)) => ident.to_string(),
        _ => return Err("expected struct or enum".into()),
    };
    at += 1;
    let name = match tokens.get(at) {
        Some(TokenTree::Ident(ident)) => ident.to_string(),
        _ => return Err("expected a name".into()),
    };
    at += 1;
    let mut params = Vec::new();
    let mut param_names = Vec::new();
    let mut type_params = Vec::new();
    if let Some(TokenTree::Punct(punct)) = tokens.get(at)
        && punct.as_char() == '<'
    {
        let start = at + 1;
        let mut depth = 1;
        at += 1;
        while depth > 0 {
            match tokens.get(at) {
                Some(TokenTree::Punct(p)) if p.as_char() == '<' => depth += 1,
                Some(TokenTree::Punct(p)) if p.as_char() == '>' => depth -= 1,
                Some(_) => {}
                None => return Err("unclosed generics".into()),
            }
            at += 1;
        }
        for param in split_commas(&tokens[start..at - 1]) {
            let (param_name, is_type) = match (&param[0], param.get(1)) {
                (TokenTree::Punct(tick), Some(TokenTree::Ident(lifetime)))
                    if tick.as_char() == '\'' =>
                {
                    (format!("'{lifetime}"), false)
                }
                (TokenTree::Ident(word), Some(TokenTree::Ident(constant)))
                    if word.to_string() == "const" =>
                {
                    (constant.to_string(), false)
                }
                (TokenTree::Ident(ident), _) => (ident.to_string(), true),
                _ => return Err("unsupported generic parameter".into()),
            };
            if is_type {
                type_params.push(param_name.clone());
            }
            param_names.push(param_name);
            params.push(param.into_iter().collect());
        }
    }
    let mut where_clause = Vec::new();
    if let Some(TokenTree::Ident(ident)) = tokens.get(at)
        && ident.to_string() == "where"
    {
        at += 1;
        while let Some(token) = tokens.get(at) {
            if matches!(token, TokenTree::Group(group) if group.delimiter() == Delimiter::Brace) {
                break;
            }
            if matches!(token, TokenTree::Punct(p) if p.as_char() == ';') {
                break;
            }
            where_clause.push(token.clone());
            at += 1;
        }
    }
    let body = match (kind.as_str(), tokens.get(at)) {
        ("struct", Some(TokenTree::Group(group))) => Body::Struct(fields(group)?),
        ("struct", _) => Body::Struct(Fields::Unit),
        ("enum", Some(TokenTree::Group(group))) => {
            let inner: Vec<TokenTree> = group.stream().into_iter().collect();
            let mut variants = Vec::new();
            for part in split_commas(&inner) {
                let mut at = 0;
                let (variant_options, cfgs) = attributes(&part, &mut at)?;
                if variant_options.skip || variant_options.fresh || variant_options.new {
                    return Err("snapshot options go on fields, not variants".into());
                }
                let Some(TokenTree::Ident(ident)) = part.get(at) else {
                    return Err("expected a variant name".into());
                };
                let fields = match part.get(at + 1) {
                    Some(TokenTree::Group(group)) => fields(group)?,
                    _ => Fields::Unit,
                };
                variants.push(Variant {
                    name: ident.to_string(),
                    fields,
                    cfgs,
                });
            }
            Body::Enum(variants)
        }
        _ => return Err(format!("cannot derive Snapshot for `{kind}`")),
    };
    Ok(Item {
        name,
        options,
        params,
        param_names,
        type_params,
        where_clause,
        body,
    })
}

/// FNV-1a, 32 bits: stable across builds and hosts, unlike `DefaultHasher`.
fn tag(text: &str) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in text.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn saved(fields: &Fields) -> Vec<&Field> {
    match fields {
        Fields::Named(list) | Fields::Tuple(list) => {
            list.iter().filter(|field| !field.options.skip).collect()
        }
        Fields::Unit => Vec::new(),
    }
}

fn cfg_prefix(cfgs: &[TokenStream]) -> String {
    cfgs.iter().map(|cfg| format!("#[{cfg}] ")).collect()
}

const RESULT: &str = "::core::result::Result<(), crate::snapshot::SnapshotError>";

fn derive(input: TokenStream) -> Result<TokenStream, String> {
    let item = parse(input)?;
    let impl_generics = if item.params.is_empty() {
        String::new()
    } else {
        let params: Vec<String> = item.params.iter().map(|p| p.to_string()).collect();
        format!("<{}>", params.join(", "))
    };
    let type_generics = if item.param_names.is_empty() {
        String::new()
    } else {
        format!("<{}>", item.param_names.join(", "))
    };
    let mut bounds: Vec<String> = item
        .type_params
        .iter()
        .map(|param| format!("{param}: crate::snapshot::Snapshot"))
        .collect();
    let existing: TokenStream = item.where_clause.iter().cloned().collect();
    let existing = existing.to_string();
    if !existing.trim().is_empty() {
        bounds.insert(0, existing.trim().trim_end_matches(',').to_string());
    }
    let where_clause = if bounds.is_empty() {
        String::new()
    } else {
        format!("where {}", bounds.join(", "))
    };
    let name = &item.name;
    let methods = match &item.body {
        Body::Struct(fields) => struct_methods(name, fields, item.options.new),
        Body::Enum(variants) => enum_methods(name, variants),
    };
    let code = format!(
        "impl{impl_generics} crate::snapshot::Snapshot for {name}{type_generics} {where_clause} {{ {methods} }}"
    );
    code.parse()
        .map_err(|error| format!("derive(Snapshot) generated bad code: {error:?}"))
}

fn struct_methods(name: &str, fields: &Fields, new: bool) -> String {
    let saved_fields = saved(fields);
    let names: Vec<&str> = saved_fields.iter().map(|f| f.member.as_str()).collect();
    let tag = tag(&format!("{name}{{{}}}", names.join(",")));
    let mut save = String::new();
    let mut load = String::new();
    for field in &saved_fields {
        let member = &field.member;
        save += &format!("crate::snapshot::Snapshot::save(&self.{member}, w)?;");
        if field.options.fresh {
            load += &format!("self.{member} = crate::snapshot::Snapshot::load_new(r)?;");
        } else {
            load += &format!("crate::snapshot::Snapshot::load(&mut self.{member}, r)?;");
        }
    }
    let mut out = format!(
        "fn save(&self, w: &mut crate::snapshot::Writer<'_>) -> {RESULT} {{ w.tag({tag})?; {save} Ok(()) }}
         fn load(&mut self, r: &mut crate::snapshot::Reader<'_>) -> {RESULT} {{ r.tag({tag}, {name:?})?; {load} Ok(()) }}"
    );
    if new {
        let construct = construct(fields);
        out += &format!(
            "fn load_new(r: &mut crate::snapshot::Reader<'_>) -> ::core::result::Result<Self, crate::snapshot::SnapshotError> {{
                 r.tag({tag}, {name:?})?; Ok(Self{construct}) }}"
        );
    }
    out
}

/// A constructor body for `fields`, reading saved ones in order (struct and variant expressions
/// evaluate their fields in the order written, which is the order they were saved).
fn construct(fields: &Fields) -> String {
    match fields {
        Fields::Unit => String::new(),
        Fields::Named(list) => {
            let parts: Vec<String> = list
                .iter()
                .map(|field| {
                    let value = if field.options.skip {
                        "::core::default::Default::default()"
                    } else {
                        "crate::snapshot::Snapshot::load_new(r)?"
                    };
                    format!("{}{}: {value}", cfg_prefix(&field.cfgs), field.member)
                })
                .collect();
            format!(" {{ {} }}", parts.join(", "))
        }
        Fields::Tuple(list) => {
            let parts: Vec<String> = list
                .iter()
                .map(|field| {
                    if field.options.skip {
                        "::core::default::Default::default()".to_string()
                    } else {
                        "crate::snapshot::Snapshot::load_new(r)?".to_string()
                    }
                })
                .collect();
            format!("({})", parts.join(", "))
        }
    }
}

fn enum_methods(name: &str, variants: &[Variant]) -> String {
    let mut signature = String::from(name);
    for variant in variants {
        let names: Vec<&str> = saved(&variant.fields)
            .iter()
            .map(|f| f.member.as_str())
            .collect();
        signature += &format!("|{}({})", variant.name, names.join(","));
    }
    let tag = tag(&signature);
    let mut save_arms = String::new();
    let mut load_arms = String::new();
    // Indexes count every variant, `#[cfg]`'d ones included, so a variant that exists on one
    // target only does not renumber the others.
    for (index, variant) in variants.iter().enumerate() {
        let cfg = cfg_prefix(&variant.cfgs);
        let variant_name = &variant.name;
        let (pattern, saves) = match &variant.fields {
            Fields::Unit => (String::new(), String::new()),
            Fields::Named(list) => {
                let binds: Vec<String> = list
                    .iter()
                    .map(|field| {
                        if field.options.skip {
                            format!("{}: _", field.member)
                        } else {
                            format!("{0}: field_{0}", field.member)
                        }
                    })
                    .collect();
                let saves: String = list
                    .iter()
                    .filter(|field| !field.options.skip)
                    .map(|field| {
                        format!(
                            "crate::snapshot::Snapshot::save(field_{}, w)?;",
                            field.member
                        )
                    })
                    .collect();
                (format!(" {{ {} }}", binds.join(", ")), saves)
            }
            Fields::Tuple(list) => {
                let binds: Vec<String> = list
                    .iter()
                    .map(|field| {
                        if field.options.skip {
                            "_".to_string()
                        } else {
                            format!("field_{}", field.member)
                        }
                    })
                    .collect();
                let saves: String = list
                    .iter()
                    .filter(|field| !field.options.skip)
                    .map(|field| {
                        format!(
                            "crate::snapshot::Snapshot::save(field_{}, w)?;",
                            field.member
                        )
                    })
                    .collect();
                (format!("({})", binds.join(", ")), saves)
            }
        };
        save_arms +=
            &format!("{cfg}Self::{variant_name}{pattern} => {{ w.u32({index})?; {saves} }}");
        load_arms += &format!(
            "{cfg}{index} => Self::{variant_name}{},",
            construct(&variant.fields)
        );
    }
    format!(
        "fn save(&self, w: &mut crate::snapshot::Writer<'_>) -> {RESULT} {{
             w.tag({tag})?; match self {{ {save_arms} }} Ok(()) }}
         fn load(&mut self, r: &mut crate::snapshot::Reader<'_>) -> {RESULT} {{
             *self = <Self as crate::snapshot::Snapshot>::load_new(r)?; Ok(()) }}
         fn load_new(r: &mut crate::snapshot::Reader<'_>) -> ::core::result::Result<Self, crate::snapshot::SnapshotError> {{
             r.tag({tag}, {name:?})?;
             let variant = r.u32()?;
             Ok(match variant {{ {load_arms} _ => return Err(r.bad_variant({name:?}, variant)) }}) }}"
    )
}
