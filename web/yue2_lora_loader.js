import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ── YUE2 Universal Adapter Loader — frontend extension ──────────────────────
//
// Each LoRA row renders two independent strength sliders:
//   M (model / diffusion transformer)  → `strength`    in serializeValue
//   C (clip / text encoder)            → `strengthTwo` in serializeValue
//
// serializeValue() returns the rgthree-style dict so _parse_lora_stack_kwargs
// in nodes.py handles model and clip strengths separately.
// ────────────────────────────────────────────────────────────────────────────

const YUE2_EXT_NAME = "comfyui_yue2_adapter_loader.power_lora_loader";
const YUE2_LORA_API = "/yue2_adapter_loader/loras";
const ACESTEP_LORA_API = "/acestep_adapter_loader/loras";

const ROW_HEIGHT = 24;
const HORIZ_MARGIN = 15;
const WEIGHT_STEP = 0.05;
const WEIGHT_DRAG_PIXELS_PER_STEP = 6;
// Width reserved for both M and C sliders on the right of each row.
// Each half gets SINGLE_WEIGHT_WIDTH px.
const WEIGHT_AREA_WIDTH = 158;
const SINGLE_WEIGHT_WIDTH = 79;

function roundToStep(v, step = WEIGHT_STEP) {
  const n = Number.isFinite(+v) ? +v : 0;
  return Number((Math.round(n / step) * step).toFixed(10));
}

// ── LoRA list cache ──────────────────────────────────────────────────────────

let _cachedLoras = null;
let _cachedLorasPromise = null;

function invalidateLorasCache() {
  _cachedLoras = null;
  _cachedLorasPromise = null;
}

async function fetchLoras() {
  if (_cachedLoras) return _cachedLoras;
  if (_cachedLorasPromise) return _cachedLorasPromise;

  _cachedLorasPromise = (async () => {
    try {
      let loras = null;
      for (const url of [YUE2_LORA_API, ACESTEP_LORA_API]) {
        try {
          const r = await fetch(url, { cache: "no-store" });
          if (r.ok) {
            const j = await r.json();
            if (Array.isArray(j)) { loras = j; break; }
          }
        } catch (_) {}
      }
      if (!Array.isArray(loras)) {
        try {
          const r = await fetch("/object_info", { cache: "no-store" });
          if (r.ok) {
            const json = await r.json();
            const spec = json?.LoraLoader?.input?.required?.lora_name;
            const values = Array.isArray(spec) ? spec[0] : null;
            if (Array.isArray(values)) loras = values.slice();
          }
        } catch (_) {}
      }
      const list = Array.isArray(loras) ? loras : [];
      _cachedLoras = ["None", ...list.filter((x) => x && x !== "None" && x !== "NONE")];
      return _cachedLoras;
    } catch (e) {
      console.warn(`[${YUE2_EXT_NAME}] Failed to fetch LoRAs`, e);
      return ["None"];
    } finally {
      _cachedLorasPromise = null;
    }
  })();
  return _cachedLorasPromise;
}

// ── Theme / drawing helpers ──────────────────────────────────────────────────

function getThemeColors() {
  const lg = globalThis.LiteGraph || {};
  return {
    bg: lg.WIDGET_BGCOLOR || "#222",
    text: lg.WIDGET_TEXT_COLOR || "#DDD",
    secondary: lg.WIDGET_SECONDARY_TEXT_COLOR || "#999",
    outline: lg.WIDGET_OUTLINE_COLOR || "#666",
  };
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath(); ctx.roundRect(x, y, w, h, [r]); return;
  }
  const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function fitText(ctx, text, maxWidth) {
  const s = String(text ?? "");
  if (maxWidth <= 8 || !s) return "";
  if (ctx.measureText(s).width <= maxWidth) return s;
  let out = s;
  while (out.length > 3 && ctx.measureText(`${out}…`).width > maxWidth) out = out.slice(0, -1);
  return out.length ? `${out}…` : "";
}

function setCanvasTooltip(text) {
  const canvas = app?.canvas?.canvas;
  if (canvas) canvas.title = text || "";
}

function syncNodeHeightToContent(node) {
  if (!node || typeof node.computeSize !== "function") return;
  const size = node.computeSize();
  if (!Array.isArray(size) || size.length < 2) return;
  const minSize = Array.isArray(node.min_size) ? node.min_size : [];
  node.size = Array.isArray(node.size) ? node.size : [0, 0];
  node.size[0] = Math.max(Number(node.size[0]) || 0, Number(minSize[0]) || 0);
  node.size[1] = Math.max(size[1], Number(minSize[1]) || 0);
}

function removeAllWidgets(node) {
  node.widgets = node.widgets || [];
  node.widgets.length = 0;
}

// ── State helpers ────────────────────────────────────────────────────────────

function makeRowDefaults() {
  return { enabled: true, name: "None", strengthModel: 1.0, strengthClip: 1.0 };
}

function sanitizeRow(r) {
  const sm = Number.isFinite(+r.strengthModel) ? +r.strengthModel : 1.0;
  return {
    enabled: !!r.enabled,
    name: typeof r.name === "string" ? r.name : "None",
    strengthModel: sm,
    strengthClip: Number.isFinite(+r.strengthClip) ? +r.strengthClip : sm,
  };
}

function defaultState() {
  return {
    rows: [makeRowDefaults()],
    globals: {
      stack_enabled: true,
      verbose: false,
      log_unloaded_keys: false,
      lora_subfolder: "",
      layer_scale_enabled: false,
      self_attn_scale: 1.0,
      cross_attn_scale: 1.0,
      ffn_scale: 1.0,
      other_scale: 1.0,
    },
  };
}

function sanitizeState(st) {
  const s = st && typeof st === "object" ? st : defaultState();
  const rowsIn = Array.isArray(s.rows) ? s.rows : [];
  const g = s.globals && typeof s.globals === "object" ? s.globals : {};
  return {
    rows: rowsIn.length ? rowsIn.map(sanitizeRow) : [makeRowDefaults()],
    globals: {
      stack_enabled:       g.stack_enabled !== undefined ? !!g.stack_enabled : true,
      verbose:             g.verbose !== undefined ? !!g.verbose : false,
      log_unloaded_keys:   g.log_unloaded_keys !== undefined ? !!g.log_unloaded_keys : false,
      lora_subfolder:      typeof g.lora_subfolder === "string" ? g.lora_subfolder : "",
      layer_scale_enabled: g.layer_scale_enabled !== undefined ? !!g.layer_scale_enabled : false,
      self_attn_scale:     Number.isFinite(+g.self_attn_scale)  ? +g.self_attn_scale  : 1.0,
      cross_attn_scale:    Number.isFinite(+g.cross_attn_scale) ? +g.cross_attn_scale : 1.0,
      ffn_scale:           Number.isFinite(+g.ffn_scale)        ? +g.ffn_scale        : 1.0,
      other_scale:         Number.isFinite(+g.other_scale)      ? +g.other_scale      : 1.0,
    },
  };
}

function getState(node) {
  node.properties = node.properties || {};
  return sanitizeState(node.properties.yue2_lora_state);
}

function setState(node, st) {
  node.properties = node.properties || {};
  node.properties.yue2_lora_state = sanitizeState(st);
}

function persistNodeState(node) {
  setState(node, { rows: node._yue2Rows || [], globals: node._yue2Globals || {} });
  node.setDirtyCanvas?.(true, true);
}

function setButtonCallback(widget, cb) {
  if (!widget) return;
  widget.callback = cb;
  widget.options = widget.options || {};
  widget.options.callback = cb;
}

// ── Active lora picker ───────────────────────────────────────────────────────

function closeActiveLoraPicker() {
  const p = globalThis.__yue2ActiveLoraPicker;
  if (p && typeof p.close === "function") p.close();
  globalThis.__yue2ActiveLoraPicker = null;
}

// ── Row widget — M and C strength sliders ────────────────────────────────────

class YUE2LoraRowWidget {
  constructor(name, parentNode, row) {
    this.name = name;
    this.type = "custom";
    this.parentNode = parentNode;
    this.row = row;
    this.last_y = 0;
    this.nameTooltip = "";
    this._drag = null;
    this.hitAreas = {
      toggle: [0, 0], name: [0, 0],
      mDec: [0, 0], mVal: [0, 0], mInc: [0, 0],
      cDec: [0, 0], cVal: [0, 0], cInc: [0, 0],
    };
  }

  computeSize(width) { return [width, ROW_HEIGHT]; }

  // Rgthree-style dict — _parse_lora_stack_kwargs in Python reads strength/strengthTwo.
  serializeValue() {
    const sm = Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0;
    const sc = Number.isFinite(+this.row.strengthClip)  ? +this.row.strengthClip  : sm;
    return { on: !!this.row.enabled, lora: this.row.name ?? "None", strength: sm, strengthTwo: sc };
  }

  draw(ctx, node, width, posY, height) {
    const colors = getThemeColors();
    const x = HORIZ_MARGIN;
    const widgetWidth = width - HORIZ_MARGIN * 2;
    const midY = posY + height / 2;
    this.last_y = posY;

    const globalOn = node.widgets?.find((w) => w.name === "stack_enabled")?.value !== false;
    const effectiveOn = !!this.row.enabled && globalOn;

    ctx.save();

    // Background
    ctx.fillStyle = colors.bg;
    ctx.strokeStyle = colors.outline;
    drawRoundedRect(ctx, x, posY + 2, widgetWidth, height - 4, 4);
    ctx.fill(); ctx.stroke();

    // Toggle
    const toggleW = 20, toggleH = 12;
    const toggleX = x + 8, toggleY = midY - toggleH / 2;
    ctx.fillStyle = colors.outline;
    drawRoundedRect(ctx, toggleX, toggleY, toggleW, toggleH, toggleH / 2);
    ctx.fill();
    ctx.fillStyle = globalOn ? colors.text : colors.secondary;
    const knobX = this.row.enabled ? toggleX + toggleW - 6 : toggleX + 6;
    ctx.beginPath(); ctx.arc(knobX, midY, 4, 0, Math.PI * 2); ctx.fill();
    this.hitAreas.toggle = [toggleX, toggleX + toggleW];

    // LoRA name
    const nameX = toggleX + toggleW + 10;
    const nameWidth = widgetWidth - (nameX - x) - WEIGHT_AREA_WIDTH - 4;
    ctx.fillStyle = effectiveOn ? colors.text : colors.secondary;
    ctx.font = "12px Arial"; ctx.textAlign = "left";
    const fullLabel = String(this.row.name || "None");
    const displayLabel = fitText(ctx, fullLabel, nameWidth);
    this.nameTooltip = displayLabel !== fullLabel ? fullLabel : "";
    ctx.fillText(displayLabel, nameX, midY + 4);
    this.hitAreas.name = [nameX, nameX + nameWidth];

    // Two strength sliders
    const sliderX = x + widgetWidth - WEIGHT_AREA_WIDTH;
    this._drawSlider(ctx, sliderX,                    midY, SINGLE_WEIGHT_WIDTH, "M", this.row.strengthModel, effectiveOn, colors, "mDec", "mVal", "mInc");
    this._drawSlider(ctx, sliderX + SINGLE_WEIGHT_WIDTH, midY, SINGLE_WEIGHT_WIDTH, "C", this.row.strengthClip,  effectiveOn, colors, "cDec", "cVal", "cInc");

    ctx.restore();
  }

  _drawSlider(ctx, startX, midY, blockW, label, value, effectiveOn, colors, decKey, valKey, incKey) {
    const val = Number.isFinite(+value) ? +value : 1.0;
    const ARROW_W = 14, LABEL_W = 14;
    const valW = blockW - ARROW_W * 2 - LABEL_W;
    const labelX = startX + 2, decX = labelX + LABEL_W, valX = decX + ARROW_W, incX = valX + valW;

    ctx.fillStyle = effectiveOn ? colors.secondary : colors.outline;
    ctx.font = "10px Arial"; ctx.textAlign = "left";
    ctx.fillText(label, labelX, midY + 3);

    ctx.fillStyle = effectiveOn ? colors.text : colors.secondary;
    ctx.font = "11px Arial"; ctx.textAlign = "center";
    ctx.fillText("<", decX + ARROW_W / 2, midY + 4);
    this.hitAreas[decKey] = [decX, decX + ARROW_W];
    ctx.fillText(val.toFixed(2), valX + valW / 2, midY + 4);
    this.hitAreas[valKey] = [valX, valX + valW];
    ctx.fillText(">", incX + ARROW_W / 2, midY + 4);
    this.hitAreas[incKey] = [incX, incX + ARROW_W];
  }

  // ── Weight editing ───────────────────────────────────────────────────────

  updateWeight(field, next) {
    const value = roundToStep(Number.isFinite(+next) ? +next : 1.0);
    if (field === "model") this.row.strengthModel = value;
    else                   this.row.strengthClip  = value;
    persistNodeState(this.parentNode);
  }

  startWeightDrag(event, field) {
    if (this._drag) this.finishWeightDrag(event);
    const startValue = field === "model"
      ? (Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0)
      : (Number.isFinite(+this.row.strengthClip)  ? +this.row.strengthClip  : 1.0);
    const onPointerMove = (e) => this.handleWeightDrag(e);
    const onPointerUp   = (e) => this.finishWeightDrag(e);
    this._drag = { field, startClientX: Number.isFinite(event?.clientX) ? event.clientX : 0,
                   startValue, moved: false, lastSteps: 0, onPointerMove, onPointerUp };
    window.addEventListener("pointermove", onPointerMove, true);
    window.addEventListener("pointerup",   onPointerUp,   true);
    window.addEventListener("pointercancel", onPointerUp, true);
  }

  handleWeightDrag(event) {
    if (!this._drag) return false;
    const dx = (Number.isFinite(event?.clientX) ? event.clientX : this._drag.startClientX) - this._drag.startClientX;
    const steps = Math.trunc(dx / WEIGHT_DRAG_PIXELS_PER_STEP);
    if (Math.abs(dx) >= 2) this._drag.moved = true;
    if (steps !== this._drag.lastSteps) {
      this._drag.lastSteps = steps;
      this.updateWeight(this._drag.field, this._drag.startValue + steps * WEIGHT_STEP);
      this.parentNode.setDirtyCanvas(true, true);
    }
    event.preventDefault?.(); return true;
  }

  finishWeightDrag(event) {
    if (!this._drag) return false;
    const drag = this._drag; this._drag = null;
    window.removeEventListener("pointermove",   drag.onPointerMove, true);
    window.removeEventListener("pointerup",     drag.onPointerUp,   true);
    window.removeEventListener("pointercancel", drag.onPointerUp,   true);
    if (!drag.moved) this.openNumericWeightPrompt(event, drag.field);
    this.parentNode.setDirtyCanvas(true, true);
    event.preventDefault?.(); return true;
  }

  openNumericWeightPrompt(event, field) {
    const current = field === "model"
      ? (Number.isFinite(+this.row.strengthModel) ? +this.row.strengthModel : 1.0)
      : (Number.isFinite(+this.row.strengthClip)  ? +this.row.strengthClip  : 1.0);
    app.canvas.prompt(field === "model" ? "Model Strength" : "Clip Strength", current, (v) => {
      const parsed = parseFloat(v);
      if (!Number.isNaN(parsed)) { this.updateWeight(field, parsed); this.parentNode.setDirtyCanvas(true, true); }
    }, event);
  }

  // ── LoRA picker ──────────────────────────────────────────────────────────

  async openLoraPicker(event) {
    const allLoras = await fetchLoras();
    closeActiveLoraPicker();

    let subfolder = String(this.parentNode?._yue2Globals?.lora_subfolder ?? "").trim().replace(/\\/g, "/");
    const lorasIdx = subfolder.toLowerCase().lastIndexOf("/loras/");
    if (lorasIdx !== -1) subfolder = subfolder.slice(lorasIdx + 7);
    subfolder = subfolder.replace(/^\/+|\/+$/g, "");
    const subLower = subfolder.toLowerCase();

    const loras = subLower
      ? allLoras.filter((n) => {
          if (n === "None") return true;
          const norm = String(n).replace(/\\/g, "/").toLowerCase();
          return norm.startsWith(subLower + "/") || norm.startsWith(subLower) ||
                 norm.includes("/" + subLower + "/") || norm.includes("/" + subLower);
        })
      : allLoras;

    const picker = document.createElement("div");
    picker.style.cssText = `position:fixed;left:${Math.max(8,Math.floor(event.clientX||0)-12)}px;top:${Math.max(8,Math.floor(event.clientY||0)-12)}px;width:520px;max-width:min(520px,calc(100vw - 16px));max-height:min(70vh,560px);background:#18181b;border:1px solid #3f3f46;border-radius:10px;box-shadow:0 12px 30px rgba(0,0,0,0.45);padding:10px;z-index:100000;display:flex;flex-direction:column;gap:8px;`;
    const input = document.createElement("input");
    input.type = "text"; input.placeholder = "Filter list";
    input.style.cssText = `width:100%;box-sizing:border-box;background:#09090b;color:#e4e4e7;border:1px solid #52525b;border-radius:8px;padding:8px 10px;outline:none;`;
    const list = document.createElement("div");
    list.style.cssText = `overflow-y:auto;max-height:min(52vh,420px);display:flex;flex-direction:column;gap:4px;padding-right:2px;box-sizing:border-box;`;
    picker.append(input, list);
    document.body.appendChild(picker);

    let filtered = [], activeIndex = Math.max(0, loras.indexOf(this.row.name ?? "None"));

    const selectValue = (value) => { this.row.name = value || "None"; persistNodeState(this.parentNode); this.parentNode.setDirtyCanvas(true, true); close(); };
    const render = () => {
      const q = input.value.trim().toLowerCase();
      filtered = loras.filter((n) => !q || String(n).toLowerCase().includes(q));
      activeIndex = Math.max(0, Math.min(activeIndex, filtered.length - 1));
      list.innerHTML = "";
      if (!filtered.length) { list.innerHTML = `<div style="padding:8px 10px;color:#a1a1aa;">No matches</div>`; return; }
      filtered.forEach((name, idx) => {
        const item = document.createElement("div");
        item.textContent = name;
        item.style.cssText = `display:flex;align-items:center;width:100%;box-sizing:border-box;min-height:32px;height:32px;padding:0 10px;background:${idx===activeIndex?"#27272a":"#111215"};color:#e4e4e7;border:1px solid #3f3f46;border-radius:8px;font-size:13px;font-family:Arial,sans-serif;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;user-select:none;`;
        item.onmouseenter = () => { activeIndex = idx; render(); };
        item.onpointerdown = (e) => { e.preventDefault(); e.stopPropagation(); selectValue(name); };
        list.appendChild(item);
      });
    };

    const onOutside = (e) => { if (!picker.contains(e.target)) close(); };
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); close(); }
      else if (e.key === "ArrowDown") { e.preventDefault(); if (filtered.length) { activeIndex = Math.min(filtered.length-1,activeIndex+1); render(); } }
      else if (e.key === "ArrowUp")   { e.preventDefault(); if (filtered.length) { activeIndex = Math.max(0,activeIndex-1); render(); } }
      else if (e.key === "Enter")     { e.preventDefault(); if (filtered.length) selectValue(filtered[activeIndex]); }
    };
    const close = () => {
      document.removeEventListener("pointerdown", onOutside, true);
      input.removeEventListener("keydown", onKey);
      picker.remove();
      if (globalThis.__yue2ActiveLoraPicker?.close === close) globalThis.__yue2ActiveLoraPicker = null;
    };
    globalThis.__yue2ActiveLoraPicker = { close };
    document.addEventListener("pointerdown", onOutside, true);
    input.addEventListener("keydown", onKey);
    input.addEventListener("input", () => { activeIndex = 0; render(); });
    render();
    queueMicrotask(() => { input.focus(); input.select(); });
  }

  // ── Right-click context menu ─────────────────────────────────────────────

  openRowMenu(event, node) {
    const CM = globalThis.LiteGraph?.ContextMenu;
    if (!CM) return false;
    const rows = node._yue2Rows || [];
    const ri = rows.indexOf(this.row);
    if (ri < 0) return false;
    const rebuild_ = () => { persistNodeState(node); buildUI(node, getState(node), _cachedLoras || ["None"]); node.setDirtyCanvas(true, true); };
    new CM([
      { content: this.row.enabled ? "Toggle Off" : "Toggle On",
        callback: () => { this.row.enabled = !this.row.enabled; persistNodeState(node); node.setDirtyCanvas(true,true); } },
      ri > 0 ? { content: "Move Up",
        callback: () => { [rows[ri-1],rows[ri]]=[rows[ri],rows[ri-1]]; rebuild_(); } } : null,
      ri < rows.length-1 ? { content: "Move Down",
        callback: () => { [rows[ri],rows[ri+1]]=[rows[ri+1],rows[ri]]; rebuild_(); } } : null,
      { content: "Set M = C",
        callback: () => { this.row.strengthModel = this.row.strengthClip; persistNodeState(node); node.setDirtyCanvas(true,true); } },
      { content: "Set C = M",
        callback: () => { this.row.strengthClip = this.row.strengthModel; persistNodeState(node); node.setDirtyCanvas(true,true); } },
      { content: "Remove",
        callback: () => { rows.splice(ri,1); if(!rows.length) rows.push(makeRowDefaults()); rebuild_(); } },
    ].filter(Boolean), { event });
    return true;
  }

  // ── Mouse ────────────────────────────────────────────────────────────────

  mouse(event, pos, node) {
    if (event.type === "pointermove" || event.type === "mousemove") {
      const dragging = this.handleWeightDrag(event);
      if (!this._drag && this.nameTooltip) {
        const wx = pos[0];
        setCanvasTooltip(wx >= this.hitAreas.name[0] && wx <= this.hitAreas.name[1] ? this.nameTooltip : "");
      }
      return dragging;
    }
    if (event.type === "pointerleave" || event.type === "mouseout") { setCanvasTooltip(""); return false; }
    if (event.type === "pointerup" || event.type === "pointercancel" || event.type === "mouseup") return this.finishWeightDrag(event);
    if (event.type !== "pointerdown") return false;
    if (event.button === 2) { event.preventDefault?.(); event.stopPropagation?.(); return this.openRowMenu(event, node); }

    const globalOn = node.widgets?.find((w) => w.name === "stack_enabled")?.value !== false;
    if (!globalOn) return false;

    const wx = pos[0];
    const H = this.hitAreas;
    if      (wx >= H.toggle[0] && wx <= H.toggle[1]) { this.row.enabled = !this.row.enabled; persistNodeState(this.parentNode); }
    else if (wx >= H.name[0]   && wx <= H.name[1])   { this.openLoraPicker(event); }
    else if (wx >= H.mDec[0]   && wx <= H.mDec[1])   { this.updateWeight("model", (+this.row.strengthModel||1.0) - WEIGHT_STEP); }
    else if (wx >= H.mInc[0]   && wx <= H.mInc[1])   { this.updateWeight("model", (+this.row.strengthModel||1.0) + WEIGHT_STEP); }
    else if (wx >= H.mVal[0]   && wx <= H.mVal[1])   { event.preventDefault?.(); event.stopPropagation?.(); this.startWeightDrag(event, "model"); }
    else if (wx >= H.cDec[0]   && wx <= H.cDec[1])   { this.updateWeight("clip", (+this.row.strengthClip||1.0) - WEIGHT_STEP); }
    else if (wx >= H.cInc[0]   && wx <= H.cInc[1])   { this.updateWeight("clip", (+this.row.strengthClip||1.0) + WEIGHT_STEP); }
    else if (wx >= H.cVal[0]   && wx <= H.cVal[1])   { event.preventDefault?.(); event.stopPropagation?.(); this.startWeightDrag(event, "clip"); }
    else return false;

    this.parentNode.setDirtyCanvas(true, true);
    return true;
  }
}

// ── Build UI ─────────────────────────────────────────────────────────────────

function buildUI(node, state, loraValues) {
  const st = sanitizeState(state);
  setCanvasTooltip("");
  setState(node, st);
  removeAllWidgets(node);

  node._yue2Rows    = st.rows;
  node._yue2Globals = st.globals;
  node.resizable    = true;

  // ── Refresh + subfolder filter ────────────────────────────────────────────
  const wRefresh = node.addWidget("button", "refresh_loras", "Refresh LoRAs");
  setButtonCallback(wRefresh, () => { invalidateLorasCache(); rebuild(node); });
  wRefresh.serialize = false;

  const wSubfolder = node.addWidget(
    "text", "lora_subfolder",
    node._yue2Globals.lora_subfolder ?? "",
    (v) => { node._yue2Globals.lora_subfolder = String(v ?? "").trim(); persistNodeState(node); }
  );
  wSubfolder.label = "LoRA subfolder filter";
  wSubfolder.tooltip = "Type a subfolder name to restrict the picker. Leave empty for all LoRAs.";

  // ── LoRA row widgets ──────────────────────────────────────────────────────
  node._yue2Rows.forEach((row, i) => {
    const widget = new YUE2LoraRowWidget(`LORA_${i + 1}`, node, row);
    if (typeof node.addCustomWidget === "function") node.addCustomWidget(widget);
    else node.widgets.push(widget);
  });

  // ── Add LoRA ──────────────────────────────────────────────────────────────
  const wAdd = node.addWidget("button", "add_lora", "Add LoRA");
  setButtonCallback(wAdd, () => {
    node._yue2Rows.push(makeRowDefaults());
    persistNodeState(node);
    buildUI(node, getState(node), _cachedLoras || ["None"]);
    node.setDirtyCanvas(true, true);
  });
  wAdd.serialize = false;

  // ── Stack Enabled ─────────────────────────────────────────────────────────
  const wStack = node.addWidget("toggle", "stack_enabled", !!node._yue2Globals.stack_enabled, (v) => {
    node._yue2Globals.stack_enabled = !!v;
    persistNodeState(node);
    node.setDirtyCanvas(true, true);
  });
  wStack.label = "Stack Enabled";

  // ── Verbose / log ─────────────────────────────────────────────────────────
  const wVerbose = node.addWidget("toggle", "verbose", !!node._yue2Globals.verbose, (v) => {
    node._yue2Globals.verbose = !!v;
    persistNodeState(node);
  });
  wVerbose.label = "Verbose";

  const wLog = node.addWidget("toggle", "log_unloaded_keys", !!node._yue2Globals.log_unloaded_keys, (v) => {
    node._yue2Globals.log_unloaded_keys = !!v;
    persistNodeState(node);
  });
  wLog.label = "Log Unloaded Keys";

  // ── Layer Category Scaling ────────────────────────────────────────────────
  const wLayerScale = node.addWidget("toggle", "layer_scale_enabled", !!node._yue2Globals.layer_scale_enabled, (v) => {
    node._yue2Globals.layer_scale_enabled = !!v;
    persistNodeState(node);
  });
  wLayerScale.label = "Enable Layer Category Scaling";

  const wSA = node.addWidget("number", "self_attn_scale",  Number.isFinite(+node._yue2Globals.self_attn_scale)  ? +node._yue2Globals.self_attn_scale  : 1.0, (v) => { node._yue2Globals.self_attn_scale  = Math.max(0, +v||0); persistNodeState(node); }, { min: 0.0, max: 3.0, step: 0.05 });
  wSA.label = "self_attn_scale";
  const wCA = node.addWidget("number", "cross_attn_scale", Number.isFinite(+node._yue2Globals.cross_attn_scale) ? +node._yue2Globals.cross_attn_scale : 1.0, (v) => { node._yue2Globals.cross_attn_scale = Math.max(0, +v||0); persistNodeState(node); }, { min: 0.0, max: 3.0, step: 0.05 });
  wCA.label = "cross_attn_scale";
  const wFF = node.addWidget("number", "ffn_scale",        Number.isFinite(+node._yue2Globals.ffn_scale)        ? +node._yue2Globals.ffn_scale        : 1.0, (v) => { node._yue2Globals.ffn_scale        = Math.max(0, +v||0); persistNodeState(node); }, { min: 0.0, max: 3.0, step: 0.05 });
  wFF.label = "ffn_scale";
  const wOt = node.addWidget("number", "other_scale",      Number.isFinite(+node._yue2Globals.other_scale)      ? +node._yue2Globals.other_scale      : 1.0, (v) => { node._yue2Globals.other_scale      = Math.max(0, +v||0); persistNodeState(node); }, { min: 0.0, max: 3.0, step: 0.05 });
  wOt.label = "other_scale";

  // ── Size ──────────────────────────────────────────────────────────────────
  const size = node.computeSize();
  const minWidth = 400;
  const minSize = Array.isArray(node.min_size) ? node.min_size.slice(0, 2) : [0, 0];
  minSize[0] = Math.max(Number(minSize[0]) || 0, minWidth);
  minSize[1] = Number(minSize[1]) || 0;
  node.min_size = minSize;
  node.size[0] = Math.max(node.size[0], size[0], minWidth);
  syncNodeHeightToContent(node);
}

function rebuild(node) {
  const st = getState(node);
  buildUI(node, st, _cachedLoras || ["None"]);
  fetchLoras().then((loras) => {
    buildUI(node, getState(node), loras);
    node.setDirtyCanvas(true, true);
  });
}

// ── Register extension ───────────────────────────────────────────────────────

app.registerExtension({
  name: YUE2_EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const nodeName   = nodeData?.name ?? "";
    const displayName = nodeData?.display_name ?? "";
    const comfyClass = nodeType?.comfyClass ?? "";

    const isTarget =
      nodeName.includes("YUE2 Universal Adapter Loader") ||
      displayName.includes("YUE2 Universal Adapter Loader") ||
      comfyClass.includes("YUE2 Universal Adapter Loader");

    if (!isTarget) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated?.apply(this, arguments);
      this.resizable = true;
      rebuild(this);
      return r;
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const r = origConfigure?.apply(this, arguments);
      const saved = info?.properties?.yue2_lora_state;
      if (saved) setState(this, saved);
      rebuild(this);
      return r;
    };

    const origOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      const r = origOnSerialize?.apply(this, arguments);
      persistNodeState(this);
      o.properties = o.properties || {};
      o.properties.yue2_lora_state = getState(this);
      // widgets_values: 4 per row [on, name, strengthModel, strengthClip], then globals
      const rows = this._yue2Rows || [];
      const g    = this._yue2Globals || {};
      const vals = [];
      for (const row of rows) {
        vals.push(!!row.enabled);
        vals.push(row.name ?? "None");
        vals.push(Number.isFinite(+row.strengthModel) ? +row.strengthModel : 1.0);
        vals.push(Number.isFinite(+row.strengthClip)  ? +row.strengthClip  : 1.0);
      }
      vals.push(!!g.stack_enabled);
      vals.push(!!g.verbose);
      vals.push(!!g.log_unloaded_keys);
      vals.push(!!g.layer_scale_enabled);
      vals.push(Number.isFinite(+g.self_attn_scale)  ? +g.self_attn_scale  : 1.0);
      vals.push(Number.isFinite(+g.cross_attn_scale) ? +g.cross_attn_scale : 1.0);
      vals.push(Number.isFinite(+g.ffn_scale)        ? +g.ffn_scale        : 1.0);
      vals.push(Number.isFinite(+g.other_scale)      ? +g.other_scale      : 1.0);
      o.widgets_values = vals;
      return r;
    };
  },
});
