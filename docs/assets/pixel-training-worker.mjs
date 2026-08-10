const RUNTIME_BASE = new URL("./runtime/", self.location.href);
const MANIFEST_URL = new URL("manifest.json", RUNTIME_BASE);
const INPUT_MOUNT = "/pixel-training-input";
const PACKAGES = ["numpy","scipy","opencv-python","scikit-image","Pillow","pyyaml","micropip"];
let runtimePromise = null;
let manifest = null;
let sourcePath = null;
let sourceFile = null;
let sourceAudit = null;

function status(message, phase) { self.postMessage({type:"status",message,phase}); }
function fail(error) { self.postMessage({type:"error",message:error instanceof Error?error.message:String(error)}); }
async function digest(buffer) { const hash=await crypto.subtle.digest("SHA-256",buffer); return [...new Uint8Array(hash)].map(v=>v.toString(16).padStart(2,"0")).join(""); }
async function verifiedWheel(entry) { const response=await fetch(new URL(entry.file,RUNTIME_BASE),{cache:"no-cache"}); if(!response.ok)throw new Error(`无法下载 ${entry.file}`); const buffer=await response.arrayBuffer(); if(await digest(buffer)!==entry.sha256)throw new Error(`${entry.file} SHA-256 校验失败`); return {name:entry.file,bytes:new Uint8Array(buffer)}; }
async function loadManifest(){if(manifest)return manifest;const response=await fetch(MANIFEST_URL,{cache:"no-cache"});if(!response.ok)throw new Error("无法读取浏览器运行清单");manifest=await response.json();return manifest;}
async function runtime() {
  if (runtimePromise) return runtimePromise;
  runtimePromise=(async()=>{
    await loadManifest();
    status("正在载入隔离 Python 科研运行时…","runtime");
    const indexURL=`https://cdn.jsdelivr.net/pyodide/v${manifest.pyodide_version}/full/`;
    const {loadPyodide}=await import(`${indexURL}pyodide.mjs`); const pyodide=await loadPyodide({indexURL});
    await pyodide.loadPackage(PACKAGES); const [project,tiff]=await Promise.all([verifiedWheel(manifest.package_wheel),verifiedWheel(manifest.tifffile_wheel)]);
    pyodide.FS.mkdirTree("/pixel-runtime"); for(const wheel of [project,tiff])pyodide.FS.writeFile(`/pixel-runtime/${wheel.name}`,wheel.bytes);
    pyodide.globals.set("wheel_names",JSON.stringify([project.name,tiff.name]));
    await pyodide.runPythonAsync(`
import json, micropip
for wheel in json.loads(wheel_names):
    await micropip.install(f"emfs:/pixel-runtime/{wheel}", deps=False)
`); pyodide.globals.delete("wheel_names");
    status("像素训练运行时已就绪。","ready"); return pyodide;
  })().catch(error=>{runtimePromise=null;throw error;}); return runtimePromise;
}
function resetMount(pyodide){try{pyodide.FS.unmount(INPUT_MOUNT)}catch{} try{pyodide.FS.rmdir(INPUT_MOUNT)}catch{}}
function mount(pyodide,file){const workerfs=pyodide.FS?.filesystems?.WORKERFS;if(!workerfs)throw new Error("浏览器缺少只读 File 挂载能力，请使用本机工作台");resetMount(pyodide);pyodide.FS.mkdirTree(INPUT_MOUNT);pyodide.FS.mount(workerfs,{files:[file]},INPUT_MOUNT);return `${INPUT_MOUNT}/${file.name}`;}
function ext(name){const match=String(name).toLowerCase().match(/\.(png|jpe?g|bmp|tiff?|btf|bigtif|bigtiff)$/);if(!match)throw new Error("仅支持 PNG/JPG/BMP/TIFF/BigTIFF");return match[0];}
function checkFile(file){if(!file||typeof file.slice!=="function"||!file.size)throw new Error("请选择非空原始图像");const extension=ext(file.name),isTiff=/\.(tif|tiff|btf|bigtif|bigtiff)$/.test(extension),tier=isTiff?manifest.limits.tiff_bounded:manifest.limits.raster_full_array,max=Math.min(Number(manifest.limits.selection.max_file_bytes),Number(tier.max_file_bytes));if(file.size>max)throw new Error(`该格式训练原图超过网页 ${(max/1048576).toFixed(0)} MiB 安全上限，请使用本机工作台`);return {is_tiff:isTiff,limits:tier};}
function setJson(pyodide,name,value){pyodide.globals.set(name,JSON.stringify(value));}
function takeFile(pyodide,path,mime){const bytes=pyodide.FS.readFile(path).slice();return {artifact:{buffer:bytes.buffer,mime},transfer:[bytes.buffer]};}

async function openSource(message){await loadManifest();const tier=checkFile(message.file);const pyodide=await runtime();sourceFile=message.file;sourcePath=mount(pyodide,sourceFile);setJson(pyodide,"open_values",{...(message.values||{}),browser_tier:tier});pyodide.globals.set("open_path",sourcePath);status("正在读取全局归一化窗口与低分辨率预览…","preview");const raw=await pyodide.runPythonAsync(`
import hashlib, importlib.resources, json
from pathlib import Path
import cv2, yaml
from sic_wafer_counter.image_io import load_image
from sic_wafer_counter.pixel_classifier import build_training_project, training_project_sha256
values=json.loads(open_values); path=Path(open_path)
cfg=yaml.safe_load(importlib.resources.files("sic_wafer_counter").joinpath("resources/default.yaml").read_text(encoding="utf-8"))
if path.suffix.lower() in {".tif",".tiff",".btf",".bigtif",".bigtiff"}:
    cfg["io"].update({"prefer_bounded_tiff_regions":True,"allow_tiff_memmap":False,"allow_tiff_full_decode":False})
digest=hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
with load_image(path,cfg) as data:
    preview=data.preview.copy(); meta=data.metadata.to_dict(); shape=data.shape
limits=values["browser_tier"]["limits"];pixels=int(shape[0])*int(shape[1])
if pixels>int(limits["max_pixels"]) or max(shape)>int(limits["max_dimension_px"]):
    raise ValueError(f"源图 {shape[1]}x{shape[0]} px 超过浏览器训练安全层级")
if values["browser_tier"]["is_tiff"] and (meta.get("source_region_read_bounded") is not True or meta.get("decoded_full_source_resident") is True):
    raise ValueError("TIFF 未确认使用有界随机访问后端，网页训练已停止")
ok,encoded=cv2.imencode(".png",preview)
if not ok: raise RuntimeError("无法编码原图预览")
Path("/pixel-preview.png").write_bytes(encoded.tobytes())
project=build_training_project(image_sha256=digest.hexdigest(),image_name=path.name,source_shape=shape,wafer_id=str(values.get("wafer_id") or digest.hexdigest()[:16]),split=str(values.get("split","calibration")),reviewer_id=str(values.get("reviewer_id","browser_expert")),annotations=[],operation_history=[{"action":"browser_project_created"}])
project["source_image"].update({"source_dtype":meta.get("dtype"),"analysis_dtype":meta.get("analysis_dtype"),"normalization_low_value":meta.get("normalization_low_value"),"normalization_high_value":meta.get("normalization_high_value"),"white_is_zero":meta.get("white_is_zero"),"preview_shape_yx":list(preview.shape)})
project["project_sha256"]=training_project_sha256(project)
json.dumps({"project":project,"metadata":meta})
`);pyodide.globals.delete("open_values");pyodide.globals.delete("open_path");const data=JSON.parse(raw);sourceAudit=data;const file=takeFile(pyodide,"/pixel-preview.png","image/png");self.postMessage({type:"opened",...data,preview:file.artifact},file.transfer);}

async function roi(message){if(!sourcePath)throw new Error("请先载入原图");const pyodide=await runtime();setJson(pyodide,"roi_values",message.roi);pyodide.globals.set("roi_path",sourcePath);const raw=await pyodide.runPythonAsync(`
import importlib.resources, json
from pathlib import Path
import cv2, numpy as np, yaml
from sic_wafer_counter.image_io import load_image
roi=json.loads(roi_values); x,y,w,h=[int(roi[k]) for k in ("x","y","width","height")]
if min(w,h)<32 or max(w,h)>1024: raise ValueError("浏览器 ROI 边长必须为 32–1024 px")
path=Path(roi_path);cfg=yaml.safe_load(importlib.resources.files("sic_wafer_counter").joinpath("resources/default.yaml").read_text(encoding="utf-8"))
if path.suffix.lower() in {".tif",".tiff",".btf",".bigtif",".bigtiff"}:cfg["io"].update({"prefer_bounded_tiff_regions":True,"allow_tiff_memmap":False,"allow_tiff_full_decode":False})
with load_image(path,cfg) as data:
    if x<0 or y<0 or x+w>data.shape[1] or y+h>data.shape[0]:raise ValueError("ROI 超出原图")
    image=data.source.read_region(x,y,w,h,normalize=True)
gray=np.rint(np.clip(image,0,1)*255).astype(np.uint8);ok,encoded=cv2.imencode(".png",gray)
if not ok:raise RuntimeError("无法编码 ROI")
Path("/pixel-roi.png").write_bytes(encoded.tobytes());json.dumps({"roi":[x,y,w,h]})
`);pyodide.globals.delete("roi_values");pyodide.globals.delete("roi_path");const file=takeFile(pyodide,"/pixel-roi.png","image/png");self.postMessage({type:"roi",...JSON.parse(raw),image:file.artifact},file.transfer);}

const PY_PROJECT_HELPER=`
import hashlib, importlib.resources, json
from pathlib import Path
import cv2, numpy as np, yaml
from sic_wafer_counter.image_io import load_image
from sic_wafer_counter.pixel_classifier import (PixelTrainingSample, build_training_project, decode_label_mask_rle, encode_label_mask_rle, evaluate_samples, predict_pixel_probability, probability_to_mask, train_pixel_classifier, training_project_sha256, validate_pixel_model, validate_training_project)
values=json.loads(action_values); path=Path(action_path)
cfg=yaml.safe_load(importlib.resources.files("sic_wafer_counter").joinpath("resources/default.yaml").read_text(encoding="utf-8"))
if path.suffix.lower() in {".tif",".tiff",".btf",".bigtif",".bigtiff"}:cfg["io"].update({"prefer_bounded_tiff_regions":True,"allow_tiff_memmap":False,"allow_tiff_full_decode":False})
def rebuild(model=None, history_action="browser_project_saved"):
    source=values["project"]["source_image"]
    project=build_training_project(image_sha256=source["sha256"],image_name=source["file_name"],source_shape=tuple(source["shape_yx"]),wafer_id=values["project"]["wafer_id"],split=values["project"]["split"],reviewer_id=values["project"].get("reviewer_id","browser_expert"),annotations=values["project"].get("annotations",[]),model=model,operation_history=[*values["project"].get("operation_history",[]),{"action":history_action}])
    for key,val in source.items():
        if key not in project["source_image"]: project["source_image"][key]=val
    project["project_sha256"]=training_project_sha256(project)
    return project
`;
async function actionPython(message,mode){if(!sourcePath)throw new Error("请先载入原图");const pyodide=await runtime();setJson(pyodide,"action_values",message);pyodide.globals.set("action_path",sourcePath);status(mode==="train"?"正在提取多尺度特征并训练类平衡随机森林…":"正在应用像素模型…",mode);const raw=await pyodide.runPythonAsync(PY_PROJECT_HELPER+`
with load_image(path,cfg) as data:
    samples=[]
    for annotation in values["project"].get("annotations",[]):
        x,y,w,h=[int(v) for v in annotation["roi_xywh"]]
        if min(w,h)<32 or max(w,h)>1024: raise ValueError("浏览器训练只接受 32–1024 px ROI")
        image=data.source.read_region(x,y,w,h,normalize=True)
        labels=decode_label_mask_rle(annotation["labels"])
        samples.append(PixelTrainingSample(image=np.asarray(image,np.float32),labels=labels,image_sha256=values["project"]["source_image"]["sha256"],image_name=values["project"]["source_image"]["file_name"],wafer_id=values["project"]["wafer_id"],split=values["project"]["split"],roi_xywh=(x,y,w,h)))
    if not samples: raise ValueError("至少需要一个已保存 ROI 标签")
    settings=values.get("settings",{})
    if values["mode"]=="train":
        model=train_pixel_classifier(samples,n_trees=int(settings.get("n_trees",32)),random_seed=int(settings.get("random_seed",1729)),probability_threshold=float(settings.get("probability_threshold",.5)),minimum_object_area_px=int(settings.get("minimum_object_area_px",5)))
    else:model=validate_pixel_model(values["model"])
    current=samples[int(values.get("annotation_index",len(samples)-1))]
    probability=predict_pixel_probability(current.image,model);threshold=float(settings.get("probability_threshold",model["probability_threshold"]));minimum=int(settings.get("minimum_object_area_px",model["minimum_object_area_px"]));prediction=probability_to_mask(probability,threshold=threshold,minimum_object_area_px=minimum)
    metrics=evaluate_samples([current],model,probability_threshold=threshold,minimum_object_area_px=minimum)
    p8=np.rint(np.clip(probability,0,1)*255).astype(np.uint8);prob=cv2.applyColorMap(p8,cv2.COLORMAP_VIRIDIS);mask=np.zeros((*prediction.shape,3),np.uint8);mask[prediction]=(50,210,255)
    gray=np.rint(np.clip(current.image,0,1)*255).astype(np.uint8);overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);boundary=cv2.morphologyEx(prediction.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);overlay[boundary]=(255,70,210);overlay[current.labels==1]=(0,0,255);overlay[current.labels==2]=(0,190,0);overlay[current.labels==3]=(0,210,235)
    for name,image in (("probability",prob),("segmentation",mask),("overlay",overlay)):
        ok,encoded=cv2.imencode(".png",image)
        if not ok:raise RuntimeError("无法编码训练预览")
        Path(f"/pixel-{name}.png").write_bytes(encoded.tobytes())
project=rebuild(model,"pixel_model_trained" if values["mode"]=="train" else "pixel_model_applied")
json.dumps({"model":model,"project":project,"metrics":metrics,"threshold":threshold,"minimum_object_area_px":minimum})
`);pyodide.globals.delete("action_values");pyodide.globals.delete("action_path");const result=JSON.parse(raw);const names=["probability","segmentation","overlay"],artifacts={},transfer=[];for(const name of names){const file=takeFile(pyodide,`/pixel-${name}.png`,"image/png");artifacts[name]=file.artifact;transfer.push(...file.transfer);}self.postMessage({type:"prediction",...result,artifacts},transfer);}

async function saveProject(message){if(!sourcePath)throw new Error("请先载入原图");const pyodide=await runtime();setJson(pyodide,"action_values",message);pyodide.globals.set("action_path",sourcePath);const raw=await pyodide.runPythonAsync(PY_PROJECT_HELPER+`\njson.dumps(rebuild(values.get("model"),"browser_project_saved"))`);pyodide.globals.delete("action_values");pyodide.globals.delete("action_path");self.postMessage({type:"project",project:JSON.parse(raw)});}
async function importProject(message){if(!sourcePath)throw new Error("导入项目前请先选择对应原图");const pyodide=await runtime();setJson(pyodide,"import_project",message.project);pyodide.globals.set("expected_hash",sourceAudit.project.source_image.sha256);const raw=await pyodide.runPythonAsync(`
import json
from sic_wafer_counter.pixel_classifier import validate_training_project
project=validate_training_project(json.loads(import_project))
if project["source_image"]["sha256"]!=expected_hash:raise ValueError("训练项目与当前原图 SHA-256 不一致")
json.dumps(project)
`);pyodide.globals.delete("import_project");pyodide.globals.delete("expected_hash");self.postMessage({type:"project",project:JSON.parse(raw)});}
async function importModel(message){const pyodide=await runtime();setJson(pyodide,"import_model",message.model);const raw=await pyodide.runPythonAsync(`
import json
from sic_wafer_counter.pixel_classifier import validate_pixel_model
json.dumps(validate_pixel_model(json.loads(import_model)))
`);pyodide.globals.delete("import_model");self.postMessage({type:"model",model:JSON.parse(raw)});}
self.addEventListener("message",async event=>{try{const m=event.data||{};if(m.type==="open")await openSource(m);else if(m.type==="roi")await roi(m);else if(m.type==="train"||m.type==="predict")await actionPython(m,m.type);else if(m.type==="save-project")await saveProject(m);else if(m.type==="import-project")await importProject(m);else if(m.type==="import-model")await importModel(m);}catch(error){fail(error);}});
