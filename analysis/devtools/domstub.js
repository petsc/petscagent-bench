// Minimal DOM shim: enough to run the dashboard's render functions and report
// what they produced. Catches runtime throws that `node --check` cannot.
const made = [];
function mk(tag) {
  const n = {
    tagName: String(tag).toUpperCase(), children: [], _text: "", style: new Proxy({}, {get:(t,k)=>t[k]||"", set:(t,k,v)=>{t[k]=v;return true;}}),
    classList: { add(){}, remove(){}, toggle(){} }, dataset: {}, hidden: false,
    appendChild(c){ if (c && c.__frag) { this.children.push(...c.children); return c; } this.children.push(c); return c; },
    append(...c){ c.forEach(x=>this.children.push(x)); },
    replaceChildren(...c){ this.children = c.slice(); },
    setAttribute(k,v){ this[k]=v; }, getAttribute(k){ return this[k]; },
    addEventListener(t,f){ (this.__on || (this.__on = {}))[t] = ((this.__on||{})[t]||[]).concat([f]); },
    dispatch(t, ev){ ((this.__on||{})[t]||[]).forEach(f=>f(ev||{preventDefault(){},stopPropagation(){}})); },
    removeEventListener(){}, remove(){},
    querySelector(){ return mk("div"); }, querySelectorAll(){ return []; },
    getBoundingClientRect(){ return {x:0,y:0,width:520,height:330,top:0,left:0,right:520,bottom:330}; },
    focus(){}, contains(){ return false; }, closest(){ return null; },
    get textContent(){ return this._text + this.children.map(c=>c.textContent||"").join(""); },
    set textContent(v){ this._text = String(v); this.children = []; },
    get firstChild(){ return this.children[0] || null; },
    insertBefore(c){ this.children.unshift(c); return c; },
  };
  made.push(n); return n;
}
const registry = {};
global.document = {
  createElement: mk, createElementNS: (ns,t)=>mk(t),
  createTextNode: (t)=>({ nodeType:3, textContent:String(t), children:[] }),
  getElementById(id){ return registry[id] || (registry[id] = mk("div")); },
  querySelector(){ return mk("div"); }, querySelectorAll(){ return []; },
  addEventListener(){}, documentElement: mk("html"), body: mk("body"),
};
global.window = { matchMedia:()=>({matches:false, addEventListener(){}}), addEventListener(){}, getComputedStyle:()=>({ getPropertyValue:(p)=>({
  "--status-good":"#0ca30c","--status-warning":"#fab219","--status-serious":"#ec835a",
  "--status-critical":"#d03b3b","--status-neutral":"#c3c2b7","--ink-muted":"#898781",
}[p] || "#2a78d6") }) };
global.localStorage = { getItem(){return null;}, setItem(){}, removeItem(){} };
global.requestAnimationFrame = (f)=>f();
global.__registry = registry; global.__made = made;
global.getComputedStyle = global.window.getComputedStyle;
