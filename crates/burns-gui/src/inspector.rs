//! Motion settings drawn entirely from the core descriptor registry: labels, help, ranges,
//! steps, choices, and visibility all come from [`burns::setting_groups`]. Edits go through
//! [`MotionProject::set_value`], which validates the whole project and leaves it unchanged on
//! any error.

use burns::{MotionProject, SettingDescriptor, SettingGroup, SettingKind, SettingValue};
use eframe::egui::{self, DragValue, Slider};

use crate::style;

/// The last rejected edit. It is shown under its setting only while the project is still in
/// the state the edit was rejected from; any later change, from anywhere, retires it.
pub struct FieldError {
    id: String,
    message: String,
    project: MotionProject,
}

/// Largest integer span drawn as a slider; wider ranges (the seed) use a number field.
const INT_SLIDER_SPAN: i64 = 64;

/// Draws one group's visible settings.
pub fn group(
    ui: &mut egui::Ui,
    project: &mut MotionProject,
    error: &mut Option<FieldError>,
    group: &SettingGroup,
) {
    style::section(ui, &group.label);
    egui::Grid::new(("settings", group.id.as_str()))
        .num_columns(2)
        .spacing([12.0, 7.0])
        .min_col_width(92.0)
        .show(ui, |ui| {
            for descriptor in &group.settings {
                // An earlier edit in this pass (a family change) may have hidden it.
                if !descriptor.is_visible(project) {
                    continue;
                }
                let label = ui
                    .label(style::muted(&descriptor.label))
                    .on_hover_ui(|ui| {
                        ui.label(&descriptor.help);
                        if !descriptor.unit.is_empty() && descriptor.unit != "×" {
                            ui.horizontal(|ui| {
                                ui.label("Unit:");
                                ui.label(&descriptor.unit);
                            });
                        }
                    })
                    .id;
                let edit = match project.get_by_id(&descriptor.id) {
                    Ok(current) => widget(ui, descriptor, current, label),
                    Err(err) => {
                        ui.label(style::error(err.to_string()));
                        None
                    }
                };
                ui.end_row();
                if let Some(value) = edit {
                    match project.set_value(&descriptor.id, value) {
                        Ok(()) => *error = None,
                        Err(err) => {
                            *error = Some(FieldError {
                                id: descriptor.id.clone(),
                                message: err.to_string(),
                                project: project.clone(),
                            });
                        }
                    }
                }
                if let Some(err) = error
                    .as_ref()
                    .filter(|err| err.id == descriptor.id && err.project == *project)
                {
                    ui.label("");
                    ui.add(egui::Label::new(style::error(&err.message)).wrap());
                    ui.end_row();
                }
            }
        });
}

/// The control for one setting; `Some` with the new value when the user changed it. `label`
/// names the control for assistive technology.
fn widget(
    ui: &mut egui::Ui,
    descriptor: &SettingDescriptor,
    current: SettingValue,
    label: egui::Id,
) -> Option<SettingValue> {
    match (&descriptor.kind, current) {
        (&SettingKind::Float { min, max, step }, SettingValue::Float(mut value)) => {
            let decimals = decimals(step);
            let mut slider = Slider::new(&mut value, min..=max)
                .step_by(f64::from(step))
                .min_decimals(decimals)
                .max_decimals(decimals);
            if descriptor.unit == "×" {
                slider = slider.suffix("×");
            }
            ui.add(slider)
                .labelled_by(label)
                .on_hover_text(descriptor.help.as_str())
                .changed()
                .then_some(SettingValue::Float(value))
        }
        (&SettingKind::Int { min, max }, SettingValue::Int(mut value)) => {
            let response = if max - min <= INT_SLIDER_SPAN {
                ui.add(Slider::new(&mut value, min..=max).integer())
            } else {
                ui.add(
                    DragValue::new(&mut value)
                        .range(min..=max)
                        .speed(1.0)
                        .update_while_editing(false),
                )
            };
            response
                .labelled_by(label)
                .on_hover_text(descriptor.help.as_str())
                .changed()
                .then_some(SettingValue::Int(value))
        }
        (SettingKind::Bool, SettingValue::Bool(mut on)) => ui
            .checkbox(&mut on, "")
            .labelled_by(label)
            .on_hover_text(descriptor.help.as_str())
            .changed()
            .then_some(SettingValue::Bool(on)),
        (SettingKind::Choice { options }, SettingValue::Choice(index)) => {
            let mut chosen = None;
            let selected = options
                .get(index as usize)
                .map_or("", |option| option.label.as_str());
            egui::ComboBox::from_id_salt(descriptor.id.as_str())
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (i, option) in options.iter().enumerate() {
                        let i = i as u32;
                        if ui
                            .selectable_label(i == index, option.label.as_str())
                            .clicked()
                            && i != index
                        {
                            chosen = Some(SettingValue::Choice(i));
                        }
                    }
                })
                .response
                .labelled_by(label)
                .on_hover_text(descriptor.help.as_str());
            chosen
        }
        _ => {
            ui.label(style::error(format!(
                "`{}` has a value that does not match its descriptor",
                descriptor.id
            )));
            None
        }
    }
}

/// Decimal places needed to show multiples of `step`.
fn decimals(step: f32) -> usize {
    let mut decimals = 0;
    let mut scaled = f64::from(step);
    while decimals < 6 && (scaled - scaled.round()).abs() > 1e-6 {
        scaled *= 10.0;
        decimals += 1;
    }
    decimals
}
