/**
 * Inspector controls built from the Rust settings registry.
 *
 * Widgets, ranges, steps, labels, choices and visibility all come from
 * `schemaJson()`. Every edit goes through `setSetting(id, raw)`; the widget
 * then shows the canonical value Rust reports. Failed edits show Rust's
 * validation message next to the control and never clamp.
 */
import {
  durationSeconds,
  errorMessage,
  type Engine,
  type SettingDescriptor,
  type SettingGroup,
  type SettingValue,
} from './engine';

export interface ControlHost {
  readonly engine: Engine;
  /** A setting edit succeeded; re-render the current frame. */
  settingChanged(): void;
}

interface Row {
  descriptor: SettingDescriptor;
  error: HTMLParagraphElement;
  /** Shows `value`; leaves `except` alone so a dragged slider keeps its pointer position. */
  show(value: SettingValue, except?: HTMLElement): void;
  invalid(on: boolean): void;
}

/** Longest integer range still edited with a slider as well as a number field. */
const MAX_INT_SLIDER_SPAN = 64;

const elementId = (id: string, suffix = '') => `set-${id.replaceAll('.', '-')}${suffix}`;

function create<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<HTMLElementTagNameMap[K]> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const element = Object.assign(document.createElement(tag), props);
  element.append(...children);
  return element;
}

function labelFor(descriptor: SettingDescriptor, target: string): HTMLLabelElement {
  const label = create('label', { htmlFor: target }, [descriptor.label]);
  if (descriptor.unit) label.append(' ', create('span', { className: 'unit' }, [descriptor.unit]));
  return label;
}

export class SettingsPanel {
  readonly #root: HTMLElement;
  readonly #host: ControlHost;
  readonly #lock: HTMLFieldSetElement;

  constructor(root: HTMLElement, host: ControlHost) {
    this.#root = root;
    this.#host = host;
    this.#lock = create('fieldset', { className: 'settings-lock' });
    this.#root.replaceChildren(this.#lock);
    this.rebuild();
  }

  set locked(locked: boolean) {
    this.#lock.disabled = locked;
  }

  /**
   * Recreates the active controls (after a variant switch, preset or import),
   * keeping keyboard focus on the same setting when it still exists.
   */
  rebuild(): void {
    const active = document.activeElement;
    const focusId = active instanceof HTMLElement && this.#root.contains(active) ? active.id : '';
    const sections = this.#host.engine.schema.groups.map((group) => this.#group(group));
    this.#lock.replaceChildren(...sections);
    if (focusId) document.getElementById(focusId)?.focus();
  }

  #group(group: SettingGroup): HTMLElement {
    const headingId = `group-${group.id}`;
    const section = create('section', { className: 'panel' });
    section.setAttribute('aria-labelledby', headingId);
    section.append(
      create('h2', { className: 'section-heading', id: headingId }, [group.label]),
      create('p', { className: 'help group-help' }, [group.help]),
    );
    const engine = this.#host.engine;
    for (const descriptor of group.settings) {
      if (!engine.isVisible(descriptor)) continue;
      section.append(this.#field(descriptor));
    }
    return section;
  }

  #field(descriptor: SettingDescriptor): HTMLElement {
    const id = descriptor.id;
    const field = create('div', { className: 'field' });
    const help = create('p', { className: 'help', id: elementId(id, '-help') }, [descriptor.help]);
    const error = create('p', { className: 'error', id: elementId(id, '-error'), hidden: true });
    error.setAttribute('role', 'alert');
    const describedBy = `${help.id} ${error.id}`;
    const kind = descriptor.kind;
    let row: Row;

    if (kind.type === 'bool') {
      const input = create('input', { type: 'checkbox', id: elementId(id) });
      input.setAttribute('aria-describedby', describedBy);
      input.addEventListener('change', () => this.#commit(row, input.checked ? 'true' : 'false', input, false));
      const label = create('label', { className: 'check-field', htmlFor: input.id }, [input, descriptor.label]);
      field.append(label);
      row = {
        descriptor,
        error,
        show: (value) => {
          if (value.type === 'bool') input.checked = value.value;
        },
        invalid: (on) => input.setAttribute('aria-invalid', String(on)),
      };
    } else if (kind.type === 'choice') {
      const select = create('select', { id: elementId(id) });
      select.setAttribute('aria-describedby', describedBy);
      kind.options.forEach((option, index) => select.append(create('option', { value: String(index) }, [option.label])));
      // The option index is accepted by setSetting exactly like the stable value.
      select.addEventListener('change', () => this.#commit(row, select.value, select, false));
      field.append(labelFor(descriptor, select.id), select);
      row = {
        descriptor,
        error,
        show: (value) => {
          if (value.type === 'choice') select.value = String(value.value);
        },
        invalid: (on) => select.setAttribute('aria-invalid', String(on)),
      };
    } else {
      const number = create('input', {
        type: 'number',
        id: elementId(id),
        min: String(kind.min),
        max: String(kind.max),
        step: kind.type === 'float' ? String(kind.step) : '1',
      });
      number.inputMode = kind.type === 'float' ? 'decimal' : 'numeric';
      number.setAttribute('aria-describedby', describedBy);
      number.addEventListener('change', () => this.#commit(row, number.value.trim(), number, false));
      const inputs: HTMLInputElement[] = [number];
      const withSlider = kind.type === 'float' || kind.max - kind.min <= MAX_INT_SLIDER_SPAN;
      field.append(labelFor(descriptor, number.id));
      if (withSlider) {
        const slider = create('input', {
          type: 'range',
          id: elementId(id, '-slider'),
          min: String(kind.min),
          max: String(kind.max),
          step: number.step,
        });
        slider.setAttribute('aria-label', `${descriptor.label} slider`);
        slider.setAttribute('aria-describedby', describedBy);
        slider.addEventListener('input', () => this.#commit(row, slider.value, slider, true));
        slider.addEventListener('change', () => this.#commit(row, slider.value, slider, false));
        inputs.unshift(slider);
        field.append(create('div', { className: 'slider-pair' }, [slider, number]));
      } else {
        field.append(number);
      }
      row = {
        descriptor,
        error,
        show: (value, except) => {
          if (value.type !== 'float' && value.type !== 'int') return;
          for (const input of inputs) if (input !== except) input.value = String(value.value);
        },
        invalid: (on) => inputs.forEach((input) => input.setAttribute('aria-invalid', String(on))),
      };
    }

    field.append(help, error);
    row.show(this.#host.engine.getSetting(id));
    return field;
  }

  /**
   * Applies one edit. `live` slider drags keep the pointer position on
   * failure; committed edits snap back to the canonical value.
   */
  #commit(row: Row, raw: string, source: HTMLElement, live: boolean): void {
    const engine = this.#host.engine;
    const id = row.descriptor.id;
    try {
      engine.setSetting(id, raw);
    } catch (error) {
      row.error.textContent = errorMessage(error);
      row.error.hidden = false;
      row.invalid(true);
      if (!live) row.show(engine.getSetting(id));
      return;
    }
    row.error.hidden = true;
    row.error.textContent = '';
    row.invalid(false);
    if (row.descriptor.kind.type === 'choice') {
      // A variant switch installs that variant's defaults and changes which settings are active.
      this.rebuild();
    } else {
      row.show(engine.getSetting(id), live ? source : undefined);
    }
    this.#host.settingChanged();
  }
}

export interface TimingHost {
  readonly engine: Engine;
  /** Output size changed: resize the preview target, then render. */
  outputChanged(): void;
  /** Frame rate or frame count changed. */
  timingChanged(): void;
}

function wholeNumber(text: string, what: string): number {
  const value = Number(text);
  if (text.trim() === '' || !Number.isInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new Error(`${what} must be a whole number from 1 to 4294967295`);
  }
  return value;
}

/** Parses `NUM` or `NUM/DEN` (whole numbers, as on the command line). */
function frameRate(text: string): { num: number; den: number } {
  const match = /^\s*(\d+)\s*(?:\/\s*(\d+)\s*)?$/.exec(text);
  if (!match) throw new Error('Frame rate must be a whole number or a ratio such as 30000/1001');
  return { num: wholeNumber(match[1], 'Frame rate numerator'), den: wholeNumber(match[2] ?? '1', 'Frame rate denominator') };
}

/** Output size and rational timing; the Rust setters validate everything. */
export class TimingPanel {
  readonly #host: TimingHost;
  readonly #width = document.querySelector<HTMLInputElement>('#output-width')!;
  readonly #height = document.querySelector<HTMLInputElement>('#output-height')!;
  readonly #fps = document.querySelector<HTMLInputElement>('#fps')!;
  readonly #duration = document.querySelector<HTMLInputElement>('#duration')!;
  readonly #summary = document.querySelector<HTMLElement>('#timing-summary')!;
  readonly #outputError = document.querySelector<HTMLElement>('#output-error')!;
  readonly #timingError = document.querySelector<HTMLElement>('#timing-error')!;

  constructor(host: TimingHost) {
    this.#host = host;
    for (const input of [this.#width, this.#height]) {
      input.addEventListener('change', () => this.#commitOutput(input));
    }
    this.#fps.addEventListener('change', () => this.#commitFps());
    this.#duration.addEventListener('change', () => this.#commitDuration());
    this.sync();
  }

  set locked(locked: boolean) {
    for (const input of [this.#width, this.#height, this.#fps, this.#duration]) {
      input.disabled = locked;
    }
  }

  /** Shows the canonical project values. */
  sync(): void {
    const project = this.#host.engine.project;
    const { fps, frames } = project.timeline;
    this.#width.value = String(project.output.width);
    this.#height.value = String(project.output.height);
    this.#fps.value = `${fps.num}/${fps.den}`;
    const seconds = durationSeconds(project);
    this.#duration.value = String(Number(seconds.toFixed(6)));
    this.#summary.textContent = `${frames} ${frames === 1 ? 'frame' : 'frames'} (0 to ${frames - 1}) at ${fps.num}/${fps.den} fps: ${seconds.toFixed(3)} s exactly.`;
  }

  #fail(element: HTMLElement, inputs: HTMLInputElement[], error: unknown): void {
    element.textContent = errorMessage(error);
    element.hidden = false;
    inputs.forEach((input) => input.setAttribute('aria-invalid', 'true'));
    this.sync();
  }

  #clear(element: HTMLElement, inputs: HTMLInputElement[]): void {
    element.hidden = true;
    element.textContent = '';
    inputs.forEach((input) => input.removeAttribute('aria-invalid'));
  }

  #commitOutput(source: HTMLInputElement): void {
    const engine = this.#host.engine;
    try {
      engine.setOutputSize(wholeNumber(this.#width.value, 'Width'), wholeNumber(this.#height.value, 'Height'));
    } catch (error) {
      this.#fail(this.#outputError, [source], error);
      return;
    }
    this.#clear(this.#outputError, [this.#width, this.#height]);
    this.sync();
    this.#host.outputChanged();
  }

  /** FPS edits keep the current clip duration; Rust converts with the final rate. */
  #commitFps(): void {
    const engine = this.#host.engine;
    try {
      const { num, den } = frameRate(this.#fps.value);
      engine.setTiming(num, den, durationSeconds(engine.project));
    } catch (error) {
      this.#fail(this.#timingError, [this.#fps], error);
      return;
    }
    this.#clear(this.#timingError, [this.#fps, this.#duration]);
    this.sync();
    this.#host.timingChanged();
  }

  #commitDuration(): void {
    const engine = this.#host.engine;
    const { fps } = engine.project.timeline;
    try {
      const seconds = Number(this.#duration.value);
      if (this.#duration.value.trim() === '' || !Number.isFinite(seconds)) {
        throw new Error('Duration must be a number of seconds');
      }
      engine.setTiming(fps.num, fps.den, seconds);
    } catch (error) {
      this.#fail(this.#timingError, [this.#duration], error);
      return;
    }
    this.#clear(this.#timingError, [this.#fps, this.#duration]);
    this.sync();
    this.#host.timingChanged();
  }

  clearErrors(): void {
    this.#clear(this.#outputError, [this.#width, this.#height]);
    this.#clear(this.#timingError, [this.#fps, this.#duration]);
  }
}
