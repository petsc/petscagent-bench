// Dump the human-visible text of each card so the content can be judged for
// redundancy without opening a browser.
require(require("path").join(__dirname, "domstub.js"));
const fs = require("fs");
const html = fs.readFileSync(require("path").join(__dirname, "..", "dashboard_artifact.html"), "utf8");
const payload = html.match(/<script type="application\/json" id="payload">([\s\S]*?)<\/script>/);
const main = html.slice(payload.index + payload[0].length).match(/<script>([\s\S]*?)<\/script>/);
global.document.createDocumentFragment = () => ({ children: [], appendChild(c){this.children.push(c);return c;} });
global.document.createRange = () => ({ selectNodeContents(){} });
global.window.getSelection = () => ({ removeAllRanges(){}, addRange(){} });
global.window.innerWidth = 1280; global.window.innerHeight = 900;
const reg = global.__registry;
reg.payload = { textContent: payload[1] };
eval(main[1]);
function txt(n, d) {
  d = d || 0;
  if (!n) return "";
  let s = n.textContent != null && (!n.children || !n.children.length) ? String(n.textContent) : "";
  const kids = (n.children || []).map((c) => txt(c, d + 1)).filter(Boolean);
  return [s].concat(kids).filter(Boolean).join(" | ");
}
for (const id of process.argv.slice(2)) {
  console.log("\n===== " + id + " =====");
  console.log(txt(reg[id]).slice(0, 2600));
}
