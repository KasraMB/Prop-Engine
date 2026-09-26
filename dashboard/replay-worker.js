/* One isolated Python runtime: uploads never leave this worker. */
let execute;
async function boot() {
  const base = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  postMessage({ type: "status", message: "Loading Python runtime…" });
  importScripts(base + "pyodide.js");
  const py = await loadPyodide({ indexURL: base });
  postMessage({ type: "status", message: "Loading NumPy and timezone data…" });
  await py.loadPackage(["numpy", "micropip"]);
  await py.runPythonAsync(
    'import micropip\nawait micropip.install("tzdata==2025.2")',
  );
  const response = await fetch("py/manifest_replay.json", {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Cannot load engine manifest");
  const manifest = await response.json();
  postMessage({ type: "status", message: "Verifying engine bundle…" });
  await Promise.all(
    manifest.files.map(async ({ path, sha256 }) => {
      const r = await fetch("py/" + path, { cache: "no-store" });
      if (!r.ok) throw new Error("Missing engine module: " + path);
      const bytes = await r.arrayBuffer();
      const digest = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      )
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      if (digest !== sha256)
        throw new Error(
          "Engine bundle changed during loading; reload the page (" +
            path +
            ")",
        );
      const dest = "/pkg/" + path;
      py.FS.mkdirTree(dest.substring(0, dest.lastIndexOf("/")));
      py.FS.writeFile(dest, new Uint8Array(bytes));
    }),
  );
  py.runPython(`import sys, json
sys.path.insert(0, "/pkg")
from replay import run
def run_json(payload):
    return json.dumps(run(json.loads(payload)), allow_nan=False)
`);
  execute = py.globals.get("run_json");
  postMessage({ type: "ready", version: manifest.bundle_sha256.slice(0, 12) });
}
self.onmessage = ({ data }) => {
  if (data.type !== "run") return;
  try {
    if (!execute) throw new Error("Engine is not ready");
    postMessage({
      type: "result",
      result: JSON.parse(execute(JSON.stringify(data.request))),
    });
  } catch (error) {
    postMessage({ type: "error", message: String(error) });
  }
};
boot().catch((error) =>
  postMessage({ type: "boot-error", message: String(error) }),
);
