// Drive the dashboard's controls and assert every section survives each state
// change. The static render check only proves the first frame; the bugs live in
// the re-render, where a filter can leave a section empty or throw.
require(require("path").join(__dirname, "domstub.js"));

const fs = require("fs");
const html = fs.readFileSync(
  require("path").join(__dirname, "..", "dashboard_artifact.html"), "utf8");
const payload = html.match(
  /<script type="application\/json" id="payload">([\s\S]*?)<\/script>/);
const main = html.slice(payload.index + payload[0].length).match(/<script>([\s\S]*?)<\/script>/);

// A real fragment splices its children into the parent on append; the plain
// stub would push the fragment itself and make the line count read as 1.
global.document.createDocumentFragment = () => ({
  __frag: true, children: [], appendChild(c) { this.children.push(c); return c; }
});
const basePush = Object.getPrototypeOf({});
global.__spliceFrags = true;
global.document.createRange = () => ({ selectNodeContents() {} });
global.window.getSelection = () => ({ removeAllRanges() {}, addRange() {} });
global.window.innerWidth = 1280;
global.window.innerHeight = 900;
global.navigator = { clipboard: { writeText: () => Promise.resolve() } };

const reg = global.__registry;
reg.payload = { textContent: payload[1] };
const PAYLOAD = JSON.parse(payload[1].replace(/<\\\//g, "</"));
// Node formats the throw against the eval'd source, which is one enormous line,
// so the echoed line is the whole payload and tells us nothing. Print the stack
// ourselves instead.
try { eval(main[1]); }
catch (e) { console.log("THREW on first render: " + e.message + "\n" + e.stack); process.exit(1); }

const SECTIONS = ["summary", "scoreboard", "grid", "ramp", "runlist", "chiprow"];
let failures = 0;
function check(what, allowEmpty) {
  const bad = SECTIONS.filter((id) => {
    const n = reg[id];
    const c = n && n.children ? n.children.length : 0;
    return c === 0 && !(allowEmpty || []).includes(id);
  });
  const sizes = SECTIONS.map((id) => (reg[id].children || []).length).join(",");
  if (bad.length) { failures++; console.log("FAIL " + what + " -> empty: " + bad.join(", ")); }
  else console.log("ok   " + what.padEnd(34) + "[" + sizes + "]");
}

function click(id, i, ev) { reg[id].children[i].dispatch("click", ev); }

check("initial");

// Judge filter. The segment is one button per judge plus "both", so its length
// follows the data: a single-judge set has two buttons, not three. Walking it
// rather than clicking fixed indices keeps the driver working on whatever is
// currently in output/.
const judgeseg = reg.judgeseg.children;
for (let i = 1; i < judgeseg.length; i++) { click("judgeseg", i); check("judge = " + i); }
click("judgeseg", 0); check("judge = all");

// Grid measure: score, ran, and one entry per scoring category. Every one of
// them has to redraw the grid without emptying it.
const measureseg = reg.measureseg.children;
for (let i = 0; i < measureseg.length; i++) { click("measureseg", i); check("measure = " + i); }
click("measureseg", 0);
check("measure = score");

// Variant chips: hide them one at a time down to the last, then restore.
// The last chip is "show all", not a variant.
const nChips = reg.chiprow.children.length - 1;
for (let i = 0; i < nChips; i++) {
  click("chiprow", i);
  check("hid variant " + i);
}
click("chiprow", nChips);   // "show all"
check("restored all variants");

// Table view.
reg.tabletoggle.dispatch("click");
const cols = (reg.datatable.children[0].children[0].children || []).length;
const rows = (reg.datatable.children[1].children || []).length;
console.log("ok   table view                       " + cols + " cols, " + rows + " rows");

// Grid selection drives the detail panel, and the source panel must fill on
// open without throwing. The first cells in the flow are the corner and the
// column headers, which carry no listener, so a click there is inert.
click("grid", PAYLOAD.problems.length + 2);
check("selected a grid cell");

// Source capture is per run, not per set, so one cell proves nothing: a set can
// carry source for a single run and the hard-coded cell land on a different
// problem entirely. Walk the cells until a run body yields a <details>, and
// only call it a failure if the payload says some run has source and no cell
// ever shows one.
function sourcePanels() {
  const out = [];
  for (const run of reg.runlist.children || []) {
    const body = run.children && run.children[1];
    for (const n of (body && body.children) || []) {
      if (n.tagName === "DETAILS") out.push(n);
    }
  }
  return out;
}
let src = sourcePanels();
for (let i = 0; i < reg.grid.children.length && !src.length; i++) {
  const cell = reg.grid.children[i];
  if (!cell.dispatch) continue;
  click("grid", i);
  src = sourcePanels();
}
if (!reg.runlist.children.length) {
  console.log("skip source panel                    run list is empty");
} else if (src.length) {
  src[0].open = true;
  src[0].dispatch("toggle");
  const lines = (src[0].children[1].children || []).length;
  console.log("ok   source panel                    " + lines + " lines on open");
  src[0].children[0].children[3].dispatch("click");   // copy
  console.log("ok   copy button                     no throw");
} else if (!PAYLOAD.codeRuns) {
  console.log("skip source panel                    no runs carry source in this set");
} else {
  failures++;
  console.log("FAIL source panel                    payload claims " + PAYLOAD.codeRuns +
              " runs with source, no cell renders one");
}

// The metric strip decomposes a run's composite. Its contributions must add to
// the total it prints, because that row exists to be checked.
function strips(node, out) {
  out = out || [];
  if (node && String(node.className || "").split(" ").includes("metrics")) out.push(node);
  for (const c of (node && node.children) || []) strips(c, out);
  return out;
}
let checked = 0, bad = 0;
for (let i = 0; i < reg.grid.children.length; i++) {
  if (!reg.grid.children[i].dispatch) continue;
  click("grid", i);
  for (const m of strips(reg.runlist)) {
    const rows = (m.children || []).filter(
      (r) => !String(r.className || "").split(" ").some((c) => c === "mhead"));
    const total = rows.pop();
    const sum = rows.reduce((a, r) => a + parseFloat(r.children[4].textContent), 0);
    const got = parseFloat(total.children[4].textContent);
    checked++;
    if (Math.abs(sum - got) > 0.05) {
      bad++;
      if (bad < 4) console.log("     sum " + sum.toFixed(1) + " vs printed " + got.toFixed(1));
    }
  }
}
if (!checked) console.log("skip metric strips                   no scored run in any cell");
else if (bad) { failures++; console.log("FAIL metric strips                   " + bad + " of " + checked + " do not add up"); }
else console.log("ok   metric strips                   " + checked + " decompositions add up");

console.log(failures ? "\n" + failures + " FAILURES" : "\nall states clean");
process.exit(failures ? 1 : 0);
