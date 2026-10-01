// Run the dashboard's inline script against the stub DOM and report what each
// section produced. Catches runtime throws that `node --check` cannot see.
require(require("path").join(__dirname, "domstub.js"));

const fs = require("fs");
const path = require("path").join(__dirname, "..", "dashboard_artifact.html");
const html = fs.readFileSync(path, "utf8");

const payloadRe = /<script type="application\/json" id="payload">([\s\S]*?)<\/script>/;
const payload = html.match(payloadRe);
if (!payload) throw new Error("payload block not found");

const after = html.slice(payload.index + payload[0].length);
const main = after.match(/<script>([\s\S]*?)<\/script>/);
if (!main) throw new Error("main script not found");

// Extra shims this page touches that the base stub does not carry.
global.document.createDocumentFragment = () => {
  const frag = { children: [], appendChild(c) { this.children.push(c); return c; } };
  return frag;
};
global.document.createRange = () => ({ selectNodeContents() {} });
global.window.getSelection = () => ({ removeAllRanges() {}, addRange() {} });
global.window.innerWidth = 1280;
global.window.innerHeight = 900;

const reg = global.__registry;
reg.payload = { textContent: payload[1] };

const t0 = Date.now();
eval(main[1]);
const ms = Date.now() - t0;

const count = (id) => {
  const n = reg[id];
  return n && n.children ? n.children.length : 0;
};
const text = (id) => {
  const n = reg[id];
  return n ? String(n.textContent || "").slice(0, 190) : "(missing)";
};

const ids = ["judgeseg", "measureseg", "chiprow", "summary", "scoreboard",
             "grid", "ramp", "runlist", "foot"];
console.log("rendered in " + ms + " ms, " + global.__made.length + " nodes\n");
for (const id of ids) {
  const n = count(id);
  console.log((n ? "  " : "! ") + id.padEnd(16) + String(n).padStart(4) + " children");
}
console.log("\nprov:      " + text("prov"));
console.log("gridnote:  " + text("gridnote"));
console.log("detailctx: " + text("detailctx"));
