// 在瀏覽器裡執行 sphereinside：照片不會離開這台裝置。
// 用 Pyodide 直接跑專案的 Python 程式，所以合成的結果與命令列版相同。
"use strict";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const SOURCES = ["__init__.py", "__main__.py", "camera.py", "coverage.py", "export.py", "paint.py", "polyhedron.py", "synthetic.py", "unfold.py"];

importScripts(PYODIDE + "pyodide.js");

const say = (text) => self.postMessage({ type: "log", text });
let ready = null;

async function setup() {
  say("載入執行環境（第一次需要下載約 30 MB）…");
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  pyodide.setStdout({ batched: say });
  pyodide.setStderr({ batched: say });
  await pyodide.loadPackage(["numpy", "scipy", "pillow", "micropip"]);
  await pyodide.runPythonAsync('import micropip\nawait micropip.install("reportlab==5.0.1")');

  pyodide.FS.mkdirTree("/pkg/sphereinside");
  const base = new URL("../sphereinside/", self.location.href);
  for (const name of SOURCES) {
    const response = await fetch(new URL(name, base));
    if (!response.ok) throw new Error(`無法載入 ${name}（${response.status}）`);
    pyodide.FS.writeFile("/pkg/sphereinside/" + name, new Uint8Array(await response.arrayBuffer()));
  }
  pyodide.runPython('import sys\nsys.path.insert(0, "/pkg")');
  return pyodide;
}

// 把照片與對照檔放進 /work；ZIP 只取出其中的照片與 meta.json
const PREPARE = `
import io, os, shutil, zipfile

def prepare(files, archive):
    shutil.rmtree("/work", ignore_errors=True)
    os.makedirs("/work")
    if archive is not None:
        with zipfile.ZipFile(io.BytesIO(bytes(archive))) as z:
            for info in z.infolist():
                name = os.path.basename(info.filename)
                if not info.is_dir() and (name == "meta.json" or name.lower().endswith((".jpg", ".jpeg", ".png"))):
                    with open("/work/" + name, "wb") as f:
                        f.write(z.read(info))
    for name, data in files:
        with open("/work/" + os.path.basename(name), "wb") as f:
            f.write(bytes(data))
    if not os.path.exists("/work/meta.json"):
        raise ValueError("找不到 meta.json")
`;

async function build(job) {
  ready = ready || setup().catch((error) => { ready = null; throw error; });
  const pyodide = await ready;
  pyodide.runPython(PREPARE);
  const prepare = pyodide.globals.get("prepare");
  prepare(pyodide.toPy(job.files.map((f) => [f.name, f.data])), job.zip ? pyodide.toPy(job.zip) : undefined);
  prepare.destroy();

  say("開始合成…");
  const args = ["build", "/work/meta.json", "-o", "/work/out.pdf", ...job.args];
  pyodide.globals.set("build_args", pyodide.toPy(args));
  const code = pyodide.runPython("from sphereinside.__main__ import main\nmain(list(build_args))");
  if (code !== 0) return self.postMessage({ type: "failed", code });
  const pdf = pyodide.FS.readFile("/work/out.pdf");
  self.postMessage({ type: "done", pdf }, [pdf.buffer]);
}

self.onmessage = (event) => {
  build(event.data).catch((error) => {
    say(String(error && error.message ? error.message : error));
    self.postMessage({ type: "failed", code: -1 });
  });
};
