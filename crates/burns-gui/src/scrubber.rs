//! The timeline scrubber: a strip of frame ticks with a safelight playhead. Clicking or dragging
//! selects the nearest frame in `0..N`.

use eframe::egui::accesskit::{Action, ActionData};
use eframe::egui::{
    self, Align2, FontId, Key, Modifiers, Pos2, Rect, Sense, Stroke, Vec2, WidgetInfo,
};

use crate::style;

/// Minimum horizontal gap between ticks, in points.
const TICK_GAP: f32 = 7.0;

/// Draws the scrubber across the available width. `changed()` reports a new frame.
pub fn scrubber(ui: &mut egui::Ui, frame: &mut u32, frames: u32) -> egui::Response {
    let height = 34.0;
    let (rect, mut response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), height),
        Sense::click_and_drag(),
    );
    let last = frames.saturating_sub(1);
    let track = Rect::from_min_max(
        Pos2::new(rect.left() + 6.0, rect.top() + 12.0),
        Pos2::new(rect.right() - 6.0, rect.bottom() - 6.0),
    );
    let x_of = |index: u32| {
        if last == 0 {
            track.left()
        } else {
            track.left() + track.width() * (f64::from(index) / f64::from(last)) as f32
        }
    };

    if last > 0
        && (response.dragged() || response.clicked())
        && let Some(pointer) = response.interact_pointer_pos()
    {
        let t = ((pointer.x - track.left()) / track.width()).clamp(0.0, 1.0);
        let index = (f64::from(t) * f64::from(last)).round() as u32;
        if index != *frame {
            *frame = index.min(last);
            response.mark_changed();
        }
    }

    // Keyboard (when focused) and assistive technology: step, or jump to a frame.
    if last > 0 && ui.is_enabled() {
        let id = response.id;
        let focused = response.has_focus();
        let (steps, set) = ui.input_mut(|input| {
            let mut steps = input.num_accesskit_action_requests(id, Action::Increment) as i64
                - input.num_accesskit_action_requests(id, Action::Decrement) as i64;
            if focused {
                steps += input.count_and_consume_key(Modifiers::NONE, Key::ArrowRight) as i64
                    - input.count_and_consume_key(Modifiers::NONE, Key::ArrowLeft) as i64;
            }
            let set = input
                .accesskit_action_requests(id, Action::SetValue)
                .filter_map(|request| match request.data {
                    Some(ActionData::NumericValue(value)) => Some(value),
                    _ => None,
                })
                .last();
            (steps, set)
        });
        let mut target = i64::from(*frame) + steps;
        if let Some(value) = set.filter(|value| value.is_finite()) {
            target = value.round().clamp(0.0, f64::from(last)) as i64;
        }
        let target = target.clamp(0, i64::from(last)) as u32;
        if target != *frame {
            *frame = target;
            response.mark_changed();
        }
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(track, 3.0, style::INPUT);
    let head = x_of((*frame).min(last));
    painter.rect_filled(
        Rect::from_min_max(track.min, Pos2::new(head, track.max.y)),
        3.0,
        style::ACCENT_DIM,
    );

    // Ticks: every `step` frames, so they never crowd; every fifth drawn taller.
    if last > 0 {
        let step = ((f64::from(TICK_GAP) * f64::from(last) / f64::from(track.width().max(1.0)))
            .ceil() as u32)
            .max(1);
        let mut index = 0u32;
        let mut count = 0u32;
        while index <= last {
            let x = x_of(index);
            let tall = count.is_multiple_of(5);
            let top = if tall {
                track.top() + 2.0
            } else {
                track.center().y
            };
            painter.line_segment(
                [Pos2::new(x, top), Pos2::new(x, track.bottom() - 2.0)],
                Stroke::new(1.0, style::BORDER),
            );
            count += 1;
            match index.checked_add(step) {
                Some(next) => index = next,
                None => break,
            }
        }
    }

    // Playhead with its frame number riding above the track.
    painter.line_segment(
        [
            Pos2::new(head, rect.top() + 10.0),
            Pos2::new(head, rect.bottom() - 2.0),
        ],
        Stroke::new(2.0, style::ACCENT),
    );
    painter.text(
        Pos2::new(
            head.clamp(rect.left() + 14.0, rect.right() - 14.0),
            rect.top(),
        ),
        Align2::CENTER_TOP,
        frame.to_string(),
        FontId::proportional(10.5),
        style::ACCENT,
    );

    let value = f64::from(*frame);
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::slider(enabled, value, "Frame"));
    ui.ctx().accesskit_node_builder(response.id, |node| {
        node.set_min_numeric_value(0.0);
        node.set_max_numeric_value(f64::from(last));
        node.set_numeric_value_step(1.0);
        if enabled && last > 0 {
            node.add_action(Action::SetValue);
            node.add_action(Action::Increment);
            node.add_action(Action::Decrement);
        }
    });
    response
}
