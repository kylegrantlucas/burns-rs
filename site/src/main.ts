import '@fontsource/b612/400.css';
import '@fontsource/b612/700.css';
import './styles/tokens.css';
import './styles/app.css';

import { SettingsPanel, TimingPanel } from './controls';
import { DepthGenerator, type DepthProgress } from './depth';
import { cachedDepth, storeDepth } from './depth-cache';
import {
  Engine,
  cameraUsesDepth,
  depthCacheKey,
  durationSeconds,
  errorMessage,
  frameSeconds,
  loadWasm,
  readModelManifest,
  type RenderView,
  type Size,
} from './engine';
import { Mp4Export, avcUnsupportedReason, type Mp4Settings } from './export';

function $<T extends HTMLElement = HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`missing element ${selector}`);
  return element;
}

const ui = {
  sourceInfo: $('#source-info'),
  gpuInfo: $('#gpu-info'),
  stage: $('#stage'),
  canvas: $<HTMLCanvasElement>('#preview'),
  stageEmpty: $('#stage-empty'),
  stageGpuError: $('#stage-gpu-error'),
  stageMessage: $('#stage-message'),
  stageMessageText: $('#stage-message-text'),
  stageNotice: $('#stage-notice'),
  stageLegend: $('#stage-legend'),
  dropHint: $('#drop-hint'),
  scrub: $<HTMLInputElement>('#scrub'),
  firstFrame: $<HTMLButtonElement>('#first-frame'),
  prevFrame: $<HTMLButtonElement>('#prev-frame'),
  play: $<HTMLButtonElement>('#play'),
  playLabel: $('#play-label'),
  nextFrame: $<HTMLButtonElement>('#next-frame'),
  lastFrame: $<HTMLButtonElement>('#last-frame'),
  repeat: $<HTMLButtonElement>('#repeat'),
  readoutFrame: $('#readout-frame'),
  readoutTime: $('#readout-time'),
  readoutFps: $('#readout-fps'),
  timelineTicks: Array.from(document.querySelectorAll<HTMLElement>('[data-timeline-tick]')),
  viewRadios: Array.from(document.querySelectorAll<HTMLInputElement>('input[name="view"]')),
  exportMp4: $<HTMLButtonElement>('#export-mp4'),
  cancelExport: $<HTMLButtonElement>('#cancel-export'),
  savePng: $<HTMLButtonElement>('#save-png'),
  downloadProject: $<HTMLButtonElement>('#download-project'),
  importProject: $<HTMLButtonElement>('#import-project'),
  downloadDepth: $<HTMLButtonElement>('#download-depth'),
  exportProgress: $('#export-progress'),
  exportProgressBar: $<HTMLProgressElement>('#export-progress-bar'),
  exportProgressText: $('#export-progress-text'),
  exportStatus: $('#export-status'),
  exportResult: $('#export-result'),
  exportError: $('#export-error'),
  depthDownloadHelp: $('#depth-download-help'),
  cliCommand: $('#cli-command'),
  copyCli: $<HTMLButtonElement>('#copy-cli'),
  cliStatus: $('#cli-status'),
  openImage: $<HTMLButtonElement>('#open-image'),
  openImageStage: $<HTMLButtonElement>('#open-image-stage'),
  imageError: $('#image-error'),
  depthState: $('#depth-state'),
  loadDepth: $<HTMLButtonElement>('#load-depth'),
  generateDepth: $<HTMLButtonElement>('#generate-depth'),
  cancelDepth: $<HTMLButtonElement>('#cancel-depth'),
  depthProgress: $('#depth-progress'),
  depthProgressBar: $<HTMLProgressElement>('#depth-progress-bar'),
  depthProgressText: $('#depth-progress-text'),
  depthError: $('#depth-error'),
  depthWarning: $('#depth-warning'),
  modelNote: $('#model-note'),
  modelLicense: $('#model-license'),
  presetSelect: $<HTMLSelectElement>('#preset-select'),
  applyPreset: $<HTMLButtonElement>('#apply-preset'),
  presetError: $('#preset-error'),
  settings: $('#settings'),
  depthSettings: $('#depth-settings'),
  imageFile: $<HTMLInputElement>('#image-file'),
  depthFile: $<HTMLInputElement>('#depth-file'),
  projectFile: $<HTMLInputElement>('#project-file'),
};

function showMessage(element: HTMLElement, text: string): void {
  element.textContent = text;
  element.hidden = text === '';
}

/** Presentation-only navigation stays usable independently of GPU initialization. */
function setupInspector(): void {
  const inspector = $('.inspector');
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('.inspector-tabs [role="tab"]'));
  const panels = tabs.map((tab) => $(`#${tab.getAttribute('aria-controls')}`));
  const activate = (index: number, focus: boolean): void => {
    tabs.forEach((tab, current) => {
      const selected = current === index;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[current].hidden = !selected;
    });
    $('.inspector-content').scrollTop = 0;
    if (focus) tabs[index].focus();
  };
  tabs.forEach((tab, index) => tab.addEventListener('click', () => activate(index, false)));
  $('.inspector-tabs').addEventListener('keydown', (event) => {
    const index = tabs.indexOf(document.activeElement as HTMLButtonElement);
    if (index < 0) return;
    let next: number;
    switch (event.key) {
      case 'ArrowRight':
        next = (index + 1) % tabs.length;
        break;
      case 'ArrowLeft':
        next = (index + tabs.length - 1) % tabs.length;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = tabs.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    activate(next, true);
  });
  const guidance = $<HTMLButtonElement>('#toggle-guidance');
  guidance.addEventListener('click', () => {
    const show = guidance.getAttribute('aria-pressed') !== 'true';
    guidance.setAttribute('aria-pressed', String(show));
    guidance.textContent = show ? 'Hide help' : 'Show help';
    inspector.dataset.guidance = String(show);
  });
  const menu = $<HTMLDetailsElement>('#project-menu');
  menu.querySelectorAll('button').forEach((button) => {
    button.addEventListener('click', () => { menu.open = false; });
  });
  document.addEventListener('click', (event) => {
    if (!(event.target instanceof Node) || !menu.contains(event.target)) menu.open = false;
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !menu.open) return;
    menu.open = false;
    menu.querySelector('summary')?.focus();
    event.preventDefault();
  });
}

setupInspector();

try {
  await loadWasm();
  start();
} catch (error) {
  const message = `The burns engine failed to load: ${errorMessage(error)} Reload the page; if this persists, the build is incomplete.`;
  ui.gpuInfo.textContent = 'Engine unavailable';
  ui.gpuInfo.classList.add('is-error');
  ui.stage.classList.add('is-unavailable');
  showMessage(ui.stageGpuError, message);
  ui.openImageStage.disabled = true;
  ui.openImage.disabled = true;
}

function start(): void {
  type DepthOrigin = 'imported' | 'generated' | 'cached';
  type Busy = 'png' | 'mp4';

  interface SourceState {
    name: string;
    bytes: Uint8Array;
    size: Size;
    key: string | null;
  }

  interface DepthResult {
    png: Uint8Array;
    generation: number;
    epoch: number;
    origin: DepthOrigin;
    label: string;
  }

  const engine = new Engine();
  const manifest = readModelManifest();
  const depthGenerator = new DepthGenerator();
  const state = {
    frame: 0,
    playing: false,
    repeat: true,
    view: 'motion' as RenderView,
    busy: null as Busy | null,
    gpu: 'pending' as 'pending' | 'ready' | 'failed',
    gpuError: '',
    source: null as SourceState | null,
    /** Incremented whenever the source image is replaced. */
    generation: 0,
    /** Incremented whenever a depth map is imported by hand; imports win over late results. */
    depthEpoch: 0,
    depth: null as { label: string; size: Size } | null,
    /** A depth result that arrived while a capture or export held the GPU. */
    pendingDepth: null as DepthResult | null,
    renderError: '',
    depthNotice: '',
    generating: false,
    avc: { key: '', reason: null as string | null, checking: false },
  };
  let exportJob: Mp4Export | null = null;

  // ---------------------------------------------------------------- render

  /** Renders the current integer frame; while a capture or export holds the GPU, unlock renders. */
  function render(): void {
    if (state.busy) return;
    state.depthNotice = '';
    const hasDepth = engine.raw.hasDepth();
    if (state.view === 'depth' && !hasDepth) state.view = 'motion';
    if (state.gpu === 'ready' && engine.raw.hasSource()) {
      let view = state.view;
      if (view === 'motion' && cameraUsesDepth(engine.project) && !hasDepth) {
        // Never fake parallax: show the unchanged source and say why.
        view = 'source';
        state.depthNotice = state.generating
          ? 'Depth motion needs a depth map. Showing the unchanged source while the map generates.'
          : 'Depth motion needs a depth map. Showing the unchanged source until you load a map or generate one.';
      }
      try {
        engine.raw.renderFrame(state.frame, view);
        state.renderError = '';
      } catch (error) {
        state.renderError = errorMessage(error);
      }
    }
    updateStage();
  }

  function resizePreview(): void {
    const { width, height } = engine.project.output;
    try {
      engine.raw.resizeTarget(width, height, false);
      state.renderError = '';
    } catch (error) {
      state.renderError = errorMessage(error);
    }
  }

  function updateStage(): void {
    const hasSource = engine.raw.hasSource();
    ui.stage.classList.toggle('has-source', hasSource);
    ui.stage.classList.toggle('is-unavailable', state.gpu === 'failed');
    ui.stageEmpty.hidden = hasSource;
    ui.canvas.hidden = !(hasSource && state.gpu === 'ready');
    showMessage(ui.stageGpuError, state.gpu === 'failed' ? gpuFailureText() : '');
    let message = '';
    if (hasSource && state.gpu === 'pending') message = 'Starting WebGPU…';
    if (hasSource && state.gpu === 'failed') message = gpuFailureText();
    ui.stageMessage.hidden = message === '';
    ui.stageMessageText.textContent = message;
    const showing = hasSource && state.gpu === 'ready';
    showMessage(ui.stageNotice, showing ? state.renderError || state.depthNotice : '');
    let legend = '';
    if (showing && state.view === 'depth' && state.depth && state.source && !state.renderError) {
      const { width, height } = state.depth.size;
      legend = `Effective depth: white is near, black is far.${
        engine.project.depth.invert ? ' Inverted by the project; the raw map is unchanged.' : ''
      } ${width}×${height} map over the full ${state.source.size.width}×${state.source.size.height} source.`;
    }
    showMessage(ui.stageLegend, state.busy ? '' : legend);
  }

  function gpuFailureText(): string {
    return `WebGPU is unavailable: ${state.gpuError} Preview and browser export need WebGPU in a secure context (HTTPS or localhost) on a browser that supports it. Presets, settings, project files and the command line still work.`;
  }

  // ------------------------------------------------------------- playback

  let animation = 0;
  let clockStart = 0;
  let clockFrame = 0;

  const canPlay = () => state.gpu === 'ready' && engine.raw.hasSource() && !state.busy;

  function restartClock(): void {
    clockStart = performance.now();
    clockFrame = state.frame;
  }

  function play(): void {
    if (!canPlay() || state.playing) return;
    const { frames } = engine.project.timeline;
    if (!state.repeat && state.frame >= frames - 1) state.frame = 0;
    state.playing = true;
    restartClock();
    animation = requestAnimationFrame(tick);
    syncTransport();
  }

  function pause(): void {
    state.playing = false;
    cancelAnimationFrame(animation);
    syncTransport();
  }

  /** The playback clock selects the nearest frame index; the renderer only sees integers. */
  function tick(now: number): void {
    if (!state.playing) return;
    const { fps, frames } = engine.project.timeline;
    let index = clockFrame + Math.round((((now - clockStart) / 1000) * fps.num) / fps.den);
    let finished = false;
    if (index >= frames) {
      if (state.repeat) {
        index %= frames;
      } else {
        index = frames - 1;
        finished = true;
      }
    }
    if (index !== state.frame) {
      state.frame = index;
      render();
      syncTransport();
    }
    if (finished) pause();
    else animation = requestAnimationFrame(tick);
  }

  function seek(frame: number): void {
    const { frames } = engine.project.timeline;
    state.frame = Math.min(Math.max(0, Math.round(frame)), frames - 1);
    if (state.playing) restartClock();
    render();
    syncTransport();
  }

  function clampFrame(): void {
    state.frame = Math.min(state.frame, engine.project.timeline.frames - 1);
    if (state.playing) restartClock();
  }

  function syncTransport(): void {
    const project = engine.project;
    const { fps, frames } = project.timeline;
    ui.scrub.max = String(frames - 1);
    ui.scrub.value = String(state.frame);
    ui.scrub.setAttribute('aria-valuetext', `Frame ${state.frame} of ${frames - 1}`);
    ui.scrub.style.setProperty('--scrub-progress', `${frames > 1 ? (state.frame / (frames - 1)) * 100 : 0}%`);
    ui.timelineTicks.forEach((tick, index) => {
      const last = ui.timelineTicks.length - 1;
      tick.hidden = (frames < 5 && index > 0 && index < last) || (frames === 1 && index === last);
      const label = String(Math.round(((frames - 1) * index) / last));
      if (tick.textContent !== label) tick.textContent = label;
    });
    ui.readoutFrame.textContent = `${state.frame} / ${frames - 1}`;
    ui.readoutTime.textContent = `${frameSeconds(project, state.frame).toFixed(3)} s / ${durationSeconds(project).toFixed(3)} s`;
    ui.readoutFps.textContent = `${fps.num}/${fps.den} fps`;
    ui.play.classList.toggle('is-playing', state.playing);
    ui.playLabel.textContent = state.playing ? 'Pause' : 'Play';
    ui.repeat.setAttribute('aria-pressed', String(state.repeat));
    for (const radio of ui.viewRadios) radio.checked = radio.value === state.view;
  }

  // ------------------------------------------------------------ UI state

  function needsMissingDepth(): boolean {
    return cameraUsesDepth(engine.project) && !engine.raw.hasDepth();
  }

  function updateUi(): void {
    const hasSource = engine.raw.hasSource();
    const hasDepth = engine.raw.hasDepth();
    const busy = state.busy !== null;
    const playable = canPlay();
    for (const button of [ui.firstFrame, ui.prevFrame, ui.play, ui.nextFrame, ui.lastFrame]) button.disabled = !playable;
    ui.scrub.disabled = !playable;
    for (const radio of ui.viewRadios) {
      radio.disabled = busy || !hasSource || state.gpu !== 'ready' || (radio.value === 'depth' && !hasDepth);
    }

    ui.openImage.disabled = busy;
    ui.openImageStage.disabled = busy;
    ui.loadDepth.disabled = busy || !hasSource;
    ui.generateDepth.disabled = busy || !hasSource || state.generating;
    ui.cancelDepth.hidden = !state.generating;
    ui.depthProgress.hidden = !state.generating;
    ui.presetSelect.disabled = busy;
    ui.applyPreset.disabled = busy;
    timing.locked = busy;
    settings.locked = busy;
    ui.importProject.disabled = busy;
    ui.downloadDepth.disabled = !hasDepth;
    ui.depthDownloadHelp.hidden = !hasDepth;

    const captureReady = hasSource && state.gpu === 'ready' && !busy && !needsMissingDepth();
    ui.savePng.disabled = !captureReady;
    ui.exportMp4.disabled = !captureReady || state.avc.checking || state.avc.reason !== null;
    ui.exportMp4.hidden = state.busy === 'mp4';
    ui.cancelExport.hidden = state.busy !== 'mp4';
    ui.exportStatus.textContent = exportReadiness(hasSource);
    ui.exportStatus.classList.toggle('is-error', state.avc.reason !== null);

    updateDepthPanel();
    updateStage();
    updateCli();
  }

  function exportReadiness(hasSource: boolean): string {
    const project = engine.project;
    const { fps, frames } = project.timeline;
    if (state.busy === 'mp4') return '';
    if (!hasSource) return 'Open an image to export frames or an MP4.';
    if (state.gpu === 'failed') return 'Browser export needs WebGPU. Use the command line below instead.';
    if (state.gpu === 'pending') return 'Waiting for WebGPU…';
    if (needsMissingDepth()) return 'This motion needs a depth map before export. Load or generate one, or pick a 2D preset.';
    if (state.avc.checking) return 'Checking H.264 encoder support…';
    if (state.avc.reason) {
      return `${state.avc.reason} MP4 export is off; frame PNG, project and depth downloads still work, and the command line below renders the MP4 with FFmpeg.`;
    }
    return `MP4: H.264 at ${project.output.width}×${project.output.height}, ${fps.num}/${fps.den} fps, ${frames} frames, browser encoder quality "high", no audio.`;
  }

  function updateDepthPanel(): void {
    const source = state.source;
    let text: string;
    if (!source) {
      text = 'Open an image first.';
    } else if (state.depth) {
      const map = state.depth.size;
      const image = source.size;
      text = `${state.depth.label}: ${map.width}×${map.height} map, stretched over the full ${image.width}×${image.height} oriented source. White is near.`;
      if (Math.abs(map.width / map.height - image.width / image.height) > 0.01 * (image.width / image.height)) {
        text += ' Its aspect ratio differs from the image; check alignment in the Depth view.';
      }
    } else {
      text = 'No depth map. Load a grayscale PNG (white is near) or generate one here. Until then depth motion shows the unchanged source; 2D motion works normally.';
    }
    ui.depthState.textContent = text;
  }

  function updateCli(): void {
    const quote = (text: string) => `'${text.replaceAll("'", `'\\''`)}'`;
    const name = state.source?.name;
    const image = name ? quote(name) : 'IMAGE.png';
    const clip = name ? quote(`${stem()}.mp4`) : 'CLIP.mp4';
    let depth = '';
    if (engine.raw.hasDepth()) depth = ' --depth depth.png';
    else if (cameraUsesDepth(engine.project)) depth = ' --auto-depth';
    const command = `burns render -i ${image} -o ${clip} --config project.json${depth}`;
    if (ui.cliCommand.textContent !== command) ui.cliStatus.textContent = '';
    ui.cliCommand.textContent = command;
  }

  /** Re-checks H.264 support for the exact output size and rate when they change. */
  function checkAvc(): void {
    const project = engine.project;
    const settings: Mp4Settings = {
      width: project.output.width,
      height: project.output.height,
      fps: project.timeline.fps,
      frames: project.timeline.frames,
    };
    const key = `${settings.width}x${settings.height}@${settings.fps.num}/${settings.fps.den}`;
    if (key === state.avc.key) return;
    state.avc = { key, reason: null, checking: true };
    updateUi();
    avcUnsupportedReason(settings)
      .catch((error: unknown) => `H.264 support check failed: ${errorMessage(error)}`)
      .then((reason) => {
        if (state.avc.key !== key) return;
        state.avc = { key, reason, checking: false };
        updateUi();
      });
  }

  // --------------------------------------------------------------- locking

  function lock(kind: Busy): void {
    pause();
    state.busy = kind;
    updateUi();
  }

  function unlock(): void {
    state.busy = null;
    const pending = state.pendingDepth;
    state.pendingDepth = null;
    render();
    if (pending) deliverDepth(pending);
    updateUi();
    syncTransport();
  }

  // ------------------------------------------------------------- downloads

  function download(blob: Blob, name: string): void {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }

  const stem = () => state.source?.name.replace(/\.[^.]+$/, '') || 'burns';

  function exportFailed(text: string): void {
    ui.exportResult.textContent = '';
    showMessage(ui.exportError, text);
  }

  function exportDone(text: string): void {
    showMessage(ui.exportError, '');
    ui.exportResult.textContent = text;
  }

  // --------------------------------------------------------------- images

  const VIDEO_EXTENSION = /\.(mp4|m4v|mov|webm|mkv|avi|gif|apng)$/i;

  async function openImage(files: FileList | File[]): Promise<void> {
    showMessage(ui.imageError, '');
    if (state.busy) {
      showMessage(ui.imageError, 'Wait for the capture or export to finish before opening another image.');
      return;
    }
    if (files.length !== 1) {
      showMessage(ui.imageError, `Open exactly one still image; ${files.length} files were given.`);
      return;
    }
    const file = files[0];
    if (file.type.startsWith('video/') || VIDEO_EXTENSION.test(file.name)) {
      showMessage(ui.imageError, `${file.name} is a video or animation. burns animates one still PNG, JPEG or WebP image.`);
      return;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (state.busy) return;
    try {
      engine.loadImage(bytes);
    } catch (error) {
      showMessage(ui.imageError, `Could not open ${file.name}: ${errorMessage(error)}`);
      return;
    }
    depthGenerator.cancel();
    state.generating = false;
    state.generation += 1;
    state.depth = null;
    state.pendingDepth = null;
    state.source = { name: file.name, bytes, size: engine.sourceSize(), key: null };
    showMessage(ui.depthError, '');
    showMessage(ui.depthWarning, '');
    ui.sourceInfo.textContent = `${file.name}, ${state.source.size.width}×${state.source.size.height} px`;
    if (state.view === 'depth') state.view = 'motion';
    clampFrame();
    render();
    updateUi();
    syncTransport();
    const generation = state.generation;
    const epoch = state.depthEpoch;
    // Hashing decodes the image again in Rust; let the first frame paint first.
    setTimeout(() => void findCachedDepth(generation, epoch), 0);
  }

  function sourceKey(source: SourceState): string {
    source.key ??= depthCacheKey(source.bytes);
    return source.key;
  }

  async function findCachedDepth(generation: number, epoch: number): Promise<void> {
    const source = state.source;
    if (!source || generation !== state.generation) return;
    let png: Uint8Array | null;
    try {
      png = await cachedDepth(sourceKey(source));
    } catch (error) {
      if (generation === state.generation) {
        showMessage(ui.depthWarning, `Browser storage is unavailable, so generated depth maps are not reused: ${errorMessage(error)}`);
      }
      return;
    }
    if (png) {
      deliverDepth({ png, generation, epoch, origin: 'cached', label: `Generated earlier with ${manifest.name} (browser storage)` });
    }
  }

  /** Installs a cached or generated map unless the image changed or a map was imported meanwhile. */
  function deliverDepth(result: DepthResult): void {
    if (result.generation !== state.generation || result.epoch !== state.depthEpoch) return;
    if (result.origin === 'cached' && engine.raw.hasDepth()) return;
    if (state.busy) {
      state.pendingDepth = result;
      return;
    }
    try {
      engine.loadDepth(result.png);
    } catch (error) {
      showMessage(ui.depthError, `The ${result.origin} depth map could not be loaded: ${errorMessage(error)}`);
      return;
    }
    state.depth = { label: result.label, size: engine.depthSize() };
    render();
    updateUi();
  }

  async function loadDepthFile(files: FileList): Promise<void> {
    showMessage(ui.depthError, '');
    if (state.busy || !state.source || files.length !== 1) return;
    const file = files[0];
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (state.busy) return;
    try {
      engine.loadDepth(bytes);
    } catch (error) {
      showMessage(ui.depthError, `Could not load ${file.name}: ${errorMessage(error)}`);
      return;
    }
    // A hand-picked map wins over any generation or cache lookup still in flight.
    state.depthEpoch += 1;
    state.pendingDepth = null;
    if (state.generating) {
      depthGenerator.cancel();
      state.generating = false;
    }
    state.depth = { label: `Imported ${file.name}`, size: engine.depthSize() };
    render();
    updateUi();
  }

  function describeProgress(progress: DepthProgress): { text: string; value?: number } {
    const megabytes = (bytes: number) => (bytes / 1e6).toFixed(1);
    switch (progress.stage) {
      case 'model-cache':
        return { text: 'Looking for the model in the browser cache…' };
      case 'download': {
        const loaded = progress.loaded ?? 0;
        const total = progress.total ?? manifest.bytes;
        return { text: `Downloading ${manifest.name}: ${megabytes(loaded)} of ${megabytes(total)} MB`, value: loaded / total };
      }
      case 'verify':
        return { text: 'Verifying the model SHA-256…' };
      case 'runtime':
        return { text: 'Starting ONNX Runtime (CPU)…' };
      case 'prepare':
        return { text: 'Preparing the image…' };
      case 'inference':
        return { text: 'Estimating depth on the CPU; this can take a while…' };
      case 'postprocess':
        return { text: 'Smoothing and encoding the depth map…' };
    }
  }

  function generateDepth(): void {
    const source = state.source;
    if (!source || state.busy || state.generating) return;
    const generation = state.generation;
    const epoch = state.depthEpoch;
    state.generating = true;
    showMessage(ui.depthError, '');
    showMessage(ui.depthWarning, '');
    ui.depthProgressText.textContent = 'Starting the depth worker…';
    ui.depthProgressBar.removeAttribute('value');
    depthGenerator.start(source.bytes, {
      progress: (progress) => {
        const { text, value } = describeProgress(progress);
        ui.depthProgressText.textContent = text;
        if (value === undefined) ui.depthProgressBar.removeAttribute('value');
        else ui.depthProgressBar.value = value;
      },
      result: (png, modelNote) => {
        state.generating = false;
        if (generation === state.generation) ui.modelNote.textContent = `${modelNote} ${modelNoteText()}`;
        deliverDepth({ png, generation, epoch, origin: 'generated', label: `Generated with ${manifest.name}` });
        void (async () => {
          try {
            await storeDepth(sourceKey(source), png);
          } catch (error) {
            if (generation !== state.generation) return;
            showMessage(
              ui.depthWarning,
              `The generated depth map is loaded, but browser storage refused to keep it, so it will be generated again next time: ${errorMessage(error)}`,
            );
          }
        })();
        render();
        updateUi();
      },
      error: (message) => {
        state.generating = false;
        if (generation === state.generation) {
          showMessage(ui.depthError, `Depth generation failed: ${message} The image still works in 2D; load a depth map or try again.`);
        }
        render();
        updateUi();
      },
    });
    render();
    updateUi();
  }

  function cancelDepth(): void {
    depthGenerator.cancel();
    state.generating = false;
    showMessage(ui.depthWarning, 'Depth generation cancelled.');
    render();
    updateUi();
  }

  const modelNoteText = () =>
    `Generate downloads ${manifest.name} (${(manifest.bytes / 1e6).toFixed(1)} MB, ${manifest.license}) from ${new URL(manifest.url).host} the first time, checks its SHA-256 and runs it on this computer's CPU. Your image is never uploaded.`;

  // ------------------------------------------------------- capture/export

  async function savePng(): Promise<void> {
    if (ui.savePng.disabled) return;
    const frame = state.frame;
    lock('png');
    exportDone(`Capturing frame ${frame}…`);
    try {
      const png = await engine.raw.framePng(frame);
      const name = `${stem()}-frame-${String(frame).padStart(4, '0')}.png`;
      download(new Blob([png as Uint8Array<ArrayBuffer>], { type: 'image/png' }), name);
      const { width, height } = engine.project.output;
      exportDone(`Saved ${name}: Motion frame ${frame} at ${width}×${height}.`);
    } catch (error) {
      exportFailed(`Frame capture failed: ${errorMessage(error)} No file was saved.`);
    } finally {
      unlock();
    }
  }

  async function exportMp4(): Promise<void> {
    if (ui.exportMp4.disabled) return;
    // Freeze the project and resident image/map before the first asynchronous
    // capability check; late depth results are held until unlock.
    lock('mp4');
    const project = engine.project;
    const settings: Mp4Settings = {
      width: project.output.width,
      height: project.output.height,
      fps: project.timeline.fps,
      frames: project.timeline.frames,
    };
    const saved = { target: engine.target(), frame: state.frame, view: state.view };
    const name = `${stem()}.mp4`;
    showMessage(ui.exportError, '');
    ui.exportProgress.hidden = false;
    ui.exportProgressBar.value = 0;
    ui.exportProgressText.textContent = 'Checking H.264 encoder support…';
    ui.cancelExport.disabled = true;
    ui.exportResult.textContent = '';
    let blob: Blob | null = null;
    let failure = '';
    try {
      const reason = await avcUnsupportedReason(settings);
      if (reason) {
        state.avc = { key: state.avc.key, reason, checking: false };
        throw new Error(reason);
      }
      engine.raw.resizeTarget(settings.width, settings.height, true);
      exportJob = new Mp4Export(
        ui.canvas,
        settings,
        (index) => engine.raw.renderFrame(index, 'motion'),
        (done, total) => {
          ui.exportProgressBar.value = done / total;
          ui.exportProgressText.textContent = `Encoded frame ${done} of ${total}`;
        },
      );
      ui.cancelExport.disabled = false;
      blob = await exportJob.run();
    } catch (error) {
      failure = errorMessage(error);
    } finally {
      exportJob = null;
      try {
        engine.raw.resizeTarget(saved.target.width, saved.target.height, saved.target.final);
      } catch (error) {
        failure ||= `restoring the preview failed: ${errorMessage(error)}`;
      }
      state.frame = Math.min(saved.frame, engine.project.timeline.frames - 1);
      state.view = saved.view;
      ui.exportProgress.hidden = true;
      unlock();
    }
    if (failure) {
      exportFailed(`MP4 export failed: ${failure} No file was saved.`);
    } else if (!blob) {
      exportDone('Export cancelled. Nothing was saved.');
    } else {
      download(blob, name);
      exportDone(`Saved ${name}: ${settings.frames} frames, ${(blob.size / 1e6).toFixed(2)} MB.`);
    }
  }

  // ------------------------------------------------------- project edits

  function projectReplaced(): void {
    settings.rebuild();
    timing.sync();
    timing.clearErrors();
    resizePreview();
    clampFrame();
    render();
    syncTransport();
    checkAvc();
    updateUi();
  }

  async function importProject(files: FileList): Promise<void> {
    showMessage(ui.exportError, '');
    if (state.busy || files.length !== 1) return;
    const file = files[0];
    const text = await file.text();
    if (state.busy) return;
    try {
      engine.setProjectJson(text);
    } catch (error) {
      exportFailed(`Project not imported: ${errorMessage(error)} The current project is unchanged.`);
      return;
    }
    projectReplaced();
    exportDone(`Opened ${file.name}: output, timing, seed, depth inversion and motion restored.`);
  }

  function applyPreset(): void {
    showMessage(ui.presetError, '');
    if (state.busy) return;
    try {
      engine.applyPreset(ui.presetSelect.value);
    } catch (error) {
      showMessage(ui.presetError, errorMessage(error));
      return;
    }
    settings.rebuild();
    render();
    updateUi();
  }

  // ------------------------------------------------------------- panels

  const settings = new SettingsPanel(ui.settings, ui.depthSettings, {
    engine,
    settingChanged: () => {
      render();
      updateUi();
    },
  });

  const timing = new TimingPanel({
    engine,
    outputChanged: () => {
      resizePreview();
      render();
      checkAvc();
      updateUi();
    },
    timingChanged: () => {
      clampFrame();
      render();
      syncTransport();
      checkAvc();
      updateUi();
    },
  });

  for (const preset of engine.presets) {
    const label = preset.requires_depth ? `${preset.label} (needs depth map)` : preset.label;
    ui.presetSelect.append(new Option(label, preset.name));
  }

  ui.modelNote.textContent = modelNoteText();
  const modelLink = document.createElement('a');
  modelLink.href = `https://huggingface.co/${manifest.id}/tree/${manifest.revision}`;
  modelLink.textContent = manifest.name;
  ui.modelLicense.append(
    modelLink,
    ` depth model, ${manifest.license}. The weights are not bundled; they download from Hugging Face only when you choose Generate.`,
  );

  // ---------------------------------------------------------------- events

  ui.openImage.addEventListener('click', () => ui.imageFile.click());
  ui.openImageStage.addEventListener('click', () => ui.imageFile.click());
  ui.imageFile.addEventListener('change', () => {
    const files = ui.imageFile.files;
    if (files) void openImage(Array.from(files));
    ui.imageFile.value = '';
  });
  ui.loadDepth.addEventListener('click', () => ui.depthFile.click());
  ui.depthFile.addEventListener('change', () => {
    const files = ui.depthFile.files;
    if (files) void loadDepthFile(files);
    ui.depthFile.value = '';
  });
  ui.generateDepth.addEventListener('click', generateDepth);
  ui.cancelDepth.addEventListener('click', cancelDepth);
  ui.applyPreset.addEventListener('click', applyPreset);

  ui.importProject.addEventListener('click', () => ui.projectFile.click());
  ui.projectFile.addEventListener('change', () => {
    const files = ui.projectFile.files;
    if (files) void importProject(files);
    ui.projectFile.value = '';
  });
  ui.downloadProject.addEventListener('click', () => {
    download(new Blob([engine.projectJson], { type: 'application/json' }), 'project.json');
    exportDone('Saved project.json.');
  });
  ui.downloadDepth.addEventListener('click', () => {
    try {
      const png = engine.raw.depthPng();
      download(new Blob([png as Uint8Array<ArrayBuffer>], { type: 'image/png' }), 'depth.png');
      exportDone('Saved depth.png (raw 16-bit map; the project carries inversion).');
    } catch (error) {
      exportFailed(`Depth map download failed: ${errorMessage(error)}`);
    }
  });
  ui.savePng.addEventListener('click', () => void savePng());
  ui.exportMp4.addEventListener('click', () => void exportMp4());
  ui.cancelExport.addEventListener('click', () => {
    ui.cancelExport.disabled = true;
    ui.exportProgressText.textContent = 'Cancelling…';
    void exportJob?.cancel().catch((error: unknown) => {
      exportFailed(`Export cancellation failed: ${errorMessage(error)}`);
    });
  });
  ui.copyCli.addEventListener('click', () => {
    const text = ui.cliCommand.textContent ?? '';
    navigator.clipboard.writeText(text).then(
      () => (ui.cliStatus.textContent = 'Copied.'),
      () => {
        getSelection()?.selectAllChildren(ui.cliCommand);
        ui.cliStatus.textContent = 'Copying was blocked; the command is selected so you can copy it.';
      },
    );
  });

  ui.play.addEventListener('click', () => (state.playing ? pause() : play()));
  ui.repeat.addEventListener('click', () => {
    state.repeat = !state.repeat;
    syncTransport();
  });
  ui.firstFrame.addEventListener('click', () => {
    pause();
    seek(0);
  });
  ui.prevFrame.addEventListener('click', () => {
    pause();
    seek(state.frame - 1);
  });
  ui.nextFrame.addEventListener('click', () => {
    pause();
    seek(state.frame + 1);
  });
  ui.lastFrame.addEventListener('click', () => {
    pause();
    seek(engine.project.timeline.frames - 1);
  });
  ui.scrub.addEventListener('input', () => seek(Number(ui.scrub.value)));
  for (const radio of ui.viewRadios) {
    radio.addEventListener('change', () => {
      if (!radio.checked) return;
      state.view = radio.value as RenderView;
      render();
      updateUi();
    });
  }

  /** Elements where Space has its own meaning (typing, activation, toggling). */
  function ownsSpace(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    if (target instanceof HTMLInputElement) return target.type !== 'range';
    return target.closest('button, select, textarea, summary, a[href]') !== null;
  }

  document.addEventListener('keydown', (event) => {
    if (event.key !== ' ' || event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
    if (ownsSpace(event.target)) return;
    event.preventDefault();
    if (state.playing) pause();
    else play();
  });

  let dragDepth = 0;
  const carriesFiles = (event: DragEvent) => event.dataTransfer?.types.includes('Files') ?? false;
  document.addEventListener('dragenter', (event) => {
    if (!carriesFiles(event)) return;
    dragDepth += 1;
    ui.dropHint.hidden = state.busy !== null;
  });
  document.addEventListener('dragleave', (event) => {
    if (!carriesFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) ui.dropHint.hidden = true;
  });
  document.addEventListener('dragover', (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = state.busy ? 'none' : 'copy';
  });
  document.addEventListener('drop', (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    ui.dropHint.hidden = true;
    const files = event.dataTransfer?.files;
    if (files) void openImage(Array.from(files));
  });

  // ---------------------------------------------------------------- start

  resizePreview();
  syncTransport();
  checkAvc();
  updateUi();

  ui.gpuInfo.textContent = 'Starting WebGPU…';
  engine.raw.initCanvas(ui.canvas).then(
    () => {
      state.gpu = 'ready';
      ui.gpuInfo.textContent = 'WebGPU ready';
      render();
      updateUi();
    },
    (error: unknown) => {
      state.gpu = 'failed';
      state.gpuError = errorMessage(error);
      ui.gpuInfo.textContent = 'WebGPU unavailable';
      ui.gpuInfo.classList.add('is-error');
      pause();
      render();
      updateUi();
    },
  );
}
