import { app } from "../../scripts/app.js";

function buildAudioUrl(file) {
    const params = new URLSearchParams({
        filename: file.filename,
        type: file.type,
        subfolder: file.subfolder || "",
    });
    return `/view?${params.toString()}`;
}

const SLOT_LABELS = ["A", "B", "C", "D", "E", "F"];

app.registerExtension({
    name: "PreviewAudioMultiCompare.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PreviewAudioMultiCompare") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            // Collect available tracks from the message
            const tracks = [];
            for (const label of SLOT_LABELS) {
                const key = `${label.toLowerCase()}_audio`;
                if (message && message[key] && message[key].length > 0) {
                    tracks.push({
                        label: label,
                        url: buildAudioUrl(message[key][0])
                    });
                }
            }

            if (tracks.length === 0) return;

            // Create the widget container once
            if (!this._multiCompareContainer) {
                const container = document.createElement("div");
                container.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:8px;";

                // Button row
                const btnRow = document.createElement("div");
                btnRow.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;justify-content:center;";

                // Audio element
                const audio = document.createElement("audio");
                audio.controls = true;
                audio.style.cssText = "width:100%;";

                // Label
                const labelEl = document.createElement("div");
                labelEl.style.cssText = "text-align:center;font-size:11px;color:#aaa;";
                
                this._mcAudioEl = audio;
                this._mcLabelEl = labelEl;
                this._mcBtnRow = btnRow;
                this._mcContainer = container;

                container.appendChild(btnRow);
                container.appendChild(audio);
                container.appendChild(labelEl);

                // Create a DOM widget
                const widget = this.addDOMWidget("audio_multi_compare", "custom", container, {
                    serialize: false,
                    hideOnZoom: false,
                });
                widget.computeSize = () => [Math.max(300, this.size[0]), 120];
                this._multiCompareContainer = true;
            }

            // Update UI for the new tracks
            this._mcBtnRow.innerHTML = "";
            this._mcBtns = {};

            tracks.forEach((track) => {
                const btn = document.createElement("button");
                btn.style.cssText =
                    "padding:6px 16px;font-weight:bold;font-size:14px;cursor:pointer;" +
                    "border:2px solid #666;border-radius:6px;background:#333;color:#fff;" +
                    "flex:1;min-width:30px;user-select:none;";
                btn.textContent = track.label;
                
                this._mcBtns[track.label] = btn;
                this._mcBtnRow.appendChild(btn);

                btn.addEventListener("click", () => {
                    const currentTime = this._mcAudioEl.currentTime;
                    const wasPlaying = !this._mcAudioEl.paused;

                    // Update State
                    this._mcAudioEl.src = track.url;
                    
                    // Update Visuals
                    Object.keys(this._mcBtns).forEach(lbl => {
                        this._mcBtns[lbl].style.borderColor = (lbl === track.label) ? "#3080e0" : "#666";
                        this._mcBtns[lbl].style.color = (lbl === track.label) ? "#3080e0" : "#fff";
                    });
                    this._mcLabelEl.textContent = "Playing: " + track.label;

                    // Resume Playback
                    this._mcAudioEl.currentTime = currentTime;
                    if (wasPlaying) {
                        this._mcAudioEl.play();
                    }
                });
            });

            // Reset state to the first available track
            const firstTrack = tracks[0];            
            Object.keys(this._mcBtns).forEach(lbl => {
                this._mcBtns[lbl].style.borderColor = (lbl === firstTrack.label) ? "#3080e0" : "#666";
                this._mcBtns[lbl].style.color = (lbl === firstTrack.label) ? "#3080e0" : "#fff";
            });
            this._mcLabelEl.textContent = "Playing: " + firstTrack.label;
            this._mcAudioEl.src = firstTrack.url;
        };
    },
});
