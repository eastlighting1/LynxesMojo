import init, { inspect_gfb_bytes, wasm_version } from "./pkg/lynxes_wasm.js";

const versionEl = document.getElementById("version");
const fileEl = document.getElementById("file");
const summaryEl = document.getElementById("summary");
const rawEl = document.getElementById("raw");

function renderSummary(info) {
  summaryEl.innerHTML = "";
  const fields = [
    ["Version", `${info.version[0]}.${info.version[1]}`],
    ["Nodes", info.node_count],
    ["Edges", info.edge_count],
    ["Compression", info.compression],
    ["Schema", info.has_schema ? "yes" : "no"],
  ];

  for (const [label, value] of fields) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`;
    summaryEl.appendChild(card);
  }
}

async function main() {
  await init();
  versionEl.textContent = `WASM ready: LynxesMojo ${wasm_version()}`;

  fileEl.addEventListener("change", async (event) => {
    const [file] = event.target.files ?? [];
    if (!file) {
      return;
    }

    const bytes = new Uint8Array(await file.arrayBuffer());
    const info = inspect_gfb_bytes(bytes);
    renderSummary(info);
    rawEl.textContent = JSON.stringify(info, null, 2);
  });
}

main().catch((error) => {
  versionEl.textContent = "Failed to initialize wasm module.";
  rawEl.textContent = String(error);
});
