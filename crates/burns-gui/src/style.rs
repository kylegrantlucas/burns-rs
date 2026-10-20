//! The workbench look: a warm darkroom graphite with one safelight-amber accent. The preview
//! itself always sits on a pure black matte so the frame edges read exactly.

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Vec2};

/// Panel and window background.
pub const PANEL: Color32 = Color32::from_rgb(0x2a, 0x27, 0x24);
/// Inputs, sliders, and inactive buttons.
pub const INPUT: Color32 = Color32::from_rgb(0x38, 0x34, 0x30);
/// Hovered inputs.
pub const INPUT_HOVER: Color32 = Color32::from_rgb(0x44, 0x3f, 0x3a);
/// Hairlines between regions.
pub const BORDER: Color32 = Color32::from_rgb(0x48, 0x42, 0x3c);
/// Primary text: warm photographic paper.
pub const TEXT: Color32 = Color32::from_rgb(0xec, 0xe6, 0xdc);
/// Labels and secondary text.
pub const MUTED: Color32 = Color32::from_rgb(0xa8, 0x9f, 0x93);
/// The safelight: selection, the playhead, and primary actions.
pub const ACCENT: Color32 = Color32::from_rgb(0xe8, 0xa3, 0x3d);
/// Accent fill behind selected text and active toggles.
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x6b, 0x4d, 0x22);
pub const ERROR: Color32 = Color32::from_rgb(0xf0, 0x74, 0x66);
pub const OK: Color32 = Color32::from_rgb(0x9c, 0xc9, 0x86);
/// The preview matte.
pub const MATTE: Color32 = Color32::BLACK;

pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (
                TextStyle::Heading,
                FontId::new(15.0, FontFamily::Proportional),
            ),
            (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
            (
                TextStyle::Button,
                FontId::new(13.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Small,
                FontId::new(11.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(12.0, FontFamily::Monospace),
            ),
        ]
        .into();

        let spacing = &mut style.spacing;
        spacing.item_spacing = Vec2::new(8.0, 6.0);
        spacing.button_padding = Vec2::new(10.0, 4.0);
        spacing.interact_size = Vec2::new(40.0, 24.0);
        spacing.slider_width = 150.0;
        spacing.combo_width = 170.0;

        let visuals = &mut style.visuals;
        *visuals = egui::Visuals::dark();
        visuals.panel_fill = PANEL;
        visuals.window_fill = PANEL;
        visuals.extreme_bg_color = Color32::from_rgb(0x1f, 0x1d, 0x1b);
        visuals.faint_bg_color = INPUT;
        visuals.code_bg_color = INPUT;
        visuals.hyperlink_color = ACCENT;
        visuals.warn_fg_color = ACCENT;
        visuals.error_fg_color = ERROR;
        visuals.override_text_color = None;
        visuals.selection.bg_fill = ACCENT_DIM;
        visuals.selection.stroke = Stroke::new(1.0, ACCENT);
        visuals.slider_trailing_fill = true;
        visuals.window_stroke = Stroke::new(1.0, BORDER);

        let radius = CornerRadius::same(4);
        let widgets = &mut visuals.widgets;
        widgets.noninteractive.bg_fill = PANEL;
        widgets.noninteractive.weak_bg_fill = PANEL;
        widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        widgets.noninteractive.corner_radius = radius;
        for (state, fill) in [
            (&mut widgets.inactive, INPUT),
            (&mut widgets.hovered, INPUT_HOVER),
            (&mut widgets.active, INPUT_HOVER),
            (&mut widgets.open, INPUT),
        ] {
            state.bg_fill = fill;
            state.weak_bg_fill = fill;
            state.corner_radius = radius;
            state.fg_stroke = Stroke::new(1.0, TEXT);
            state.bg_stroke = Stroke::NONE;
        }
        widgets.hovered.bg_stroke = Stroke::new(1.0, BORDER);
        widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
        widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    });
}

/// A filled primary action, used once per region.
pub fn primary(text: &str) -> egui::Button<'_> {
    egui::Button::new(egui::RichText::new(text).color(Color32::from_rgb(0x22, 0x1a, 0x10)))
        .fill(ACCENT)
}

/// Group title in an inspector.
pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(title)
            .text_style(TextStyle::Heading)
            .color(TEXT),
    );
    ui.add_space(2.0);
}

pub fn muted(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text.into()).color(MUTED)
}

pub fn small(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text.into()).small().color(MUTED)
}

pub fn error(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text.into()).color(ERROR)
}

pub fn ok(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text.into()).color(OK)
}
