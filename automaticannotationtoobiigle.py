# Enables postponed type annotations and pulls in all the libraries you need to: read env/config, 
#call the BIIGLE API, load images, run YOLO, and (optionally) download model weights from Hugging Face.                                       
from __future__ import annotations                                          

import io, os, sys, time, json, requests                                 
from typing import Dict, List, Optional                                  
from requests.auth import HTTPBasicAuth                                      
from PIL import Image                                                       
from ultralytics import YOLO                                                 
from huggingface_hub import hf_hub_download                               

# Reads environment variable, then returns its value with whitespace removed or a default if its missing,
def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:      
    v = os.getenv(name, default)                                             
    return v.strip() if isinstance(v, str) else v                            

#loads config from environment variables, trims the base urls and parses types (ints/floats/bool) for the rest. 
BASE_URL   = (_getenv("BIIGLE_BASE_URL", "https://biigle.de") or "").rstrip("/")  
VOLUME_ID  = int(_getenv("BIIGLE_VOLUME_ID", "26581") or "26581")             
USERNAME   = _getenv("BIIGLE_USERNAME")                                    
API_TOKEN  = _getenv("BIIGLE_API_TOKEN")                                     
CONF_THR   = float(_getenv("CONF_THRESH", "0.25") or "0.25")                 
DRY_RUN    = (_getenv("DRY_RUN", "0") == "1")                                
MAX_IMAGES = int(_getenv("MAX_IMAGES", "0") or "0")                       


#verifies that biigle username and api token exists if either is missing it prints an error and terminate the script. 
if not USERNAME or not API_TOKEN:                                            
    print("ERROR: set BIIGLE_USERNAME and BIIGLE_API_TOKEN env vars (no leading/trailing spaces).") 
    sys.exit(1)                                                               

"tries to read weights oath form the enviromnemtn if missing fetches train weights bestpt forom the hugginface repo adn returns the local path"
WEIGHTS_PATH = _getenv("WEIGHTS_PATH") or hf_hub_download(                  
    repo_id="akridge/yolo11-fish-detector-grayscale",                        
    filename="train/weights/best.pt"                                      
)

#Biigles id. Detections can be created with the correct label.
LABEL_MAP: Dict[str, int] = {                                                 
    "fish":        475638,
    "red algae":   475635,
    "green algae": 475636,
    "brown algae": 475637,
    "jellyfish":   475639,
    "hydroid":     475640,
    "hydroids":    475640,
    "barnacle":    475641,
    "barnacles":   475641,
    "bivalve":     475642,
    "bivalves":    475642,
}


#Creates a reusable HTTP client that authenticates every reques with basic auth and asks the servert ot retuns JSON.
session = requests.Session()                                                 
session.auth = HTTPBasicAuth(USERNAME, API_TOKEN)                           
session.headers.update({"Accept": "application/json"})                        

#Calls api v1 shapes, scans the lists for an entry named "rectangle", and returns its id; otherwise returns none. 
def get_shape_id(name: str = "rectangle") -> Optional[int]:                  
    """Discover numeric shape_id for 'rectangle'."""                         
    url = f"{BASE_URL}/api/v1/shapes"                                      
    try:
        r = session.get(url, timeout=30)                                  
        if r.status_code == 404:                                              
            return None                                                     
        r.raise_for_status()                                                
        data = r.json()                                                     
        if isinstance(data, list):                                            
            for sh in data:                                               
                if (sh.get("name", "") or "").lower() == name:              
                    return int(sh.get("id"))                                
    except Exception:                                                         
        pass
    return None                                                               


#Looks up the numeric ID for the rectangle shape; if not found and you are not in dry-run it prints a helpful hint and exits.
RECT_SHAPE_ID = get_shape_id("rectangle")                                     
if RECT_SHAPE_ID is None and not DRY_RUN:                                
    print("ERROR: Server requires 'shape_id' (per your 422). Could not discover it from /api/v1/shapes.")  
    print("Run:  curl -u \"$BIIGLE_USERNAME:$BIIGLE_API_TOKEN\" \"$BIIGLE_BASE_URL/api/v1/shapes\"")      
    sys.exit(2)                                                               


#calls Get api v1 volumes id files, tolerates a couple of JSON shapes, extracts and id for each item and returns a list of ints. 
def list_volume_file_ids(volume_id: int) -> List[int]:                        
    """GET /volumes/{id}/files → list of image/file IDs."""               
    url = f"{BASE_URL}/api/v1/volumes/{volume_id}/files"                     
    r = session.get(url, timeout=60)                                          
    r.raise_for_status()                                                      
    ct = (r.headers.get("content-type") or "").lower()                        
    if "json" not in ct:                                                     
        raise RuntimeError(f"Expected JSON from {url}, got {ct}\n{r.text[:300]}")  
    data = r.json()                                                          
    out: List[int] = []                                                       
    if isinstance(data, list):                                                
        for item in data:                                                  
            if isinstance(item, int):                                       
                out.append(item)
            elif isinstance(item, dict):                                     
                for k in ("id", "image_id", "file_id"):                  
                    if k in item:
                        out.append(int(item[k]))                          
                        break
    elif isinstance(data, dict) and isinstance(data.get("data"), list):       
        out = [int(x) for x in data["data"]]                                 
    else:
        raise RuntimeError(f"Unexpected shape from {url}: {type(data)}")     
    return out                                                                 


#Downloads the raw bytes for an image and returns them so they can be opened with PIL. 
def download_image_bytes(image_id: int) -> bytes:                              
    url = f"{BASE_URL}/api/v1/images/{image_id}/file"                         
    r = session.get(url, stream=True, timeout=60)                           
    r.raise_for_status()                                                       
    return r.content                                                         


# Builds an annotation payload for rectangle (shape_id, points, label_id, confidence) and POSTs it to Biigle. 
#If the server rejects one point format, it tries a few alternative encodings until one succeeds
def post_labeled_rectangle(image_id: int, x1: int, y1: int, x2: int, y2: int,  
                           label_id: int, conf: float) -> int:
    """
    Create a labeled rectangle in ONE request with required fields:
    shape_id, points, label_id, confidence.
    Tries several point encodings in case the server wants 2-point or 4-corner.
    """  # docstring explaining robust payload fallbacks
    if DRY_RUN:                                                               
        return -1                                                           

    url = f"{BASE_URL}/api/v1/images/{image_id}/annotations"                  
    conf = float(max(0.0, min(1.0, conf)))                                     
    w = max(0, x2 - x1)                                                      
    h = max(0, y2 - y1)                                                      

    candidates = [                                                            
        {"shape_id": RECT_SHAPE_ID, "points": [x1, y1, x2, y2], "label_id": label_id, "confidence": conf},
        {"shape_id": RECT_SHAPE_ID, "points": [x1, y1, x2, y1, x2, y2, x1, y2], "label_id": label_id, "confidence": conf},
        {"shape_id": RECT_SHAPE_ID, "points": [[x1, y1], [x2, y2]], "label_id": label_id, "confidence": conf},
        {"shape_id": RECT_SHAPE_ID, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "label_id": label_id, "confidence": conf},
        {"shape_id": RECT_SHAPE_ID, "x": x1, "y": y1, "width": w, "height": h, "label_id": label_id, "confidence": conf},
    ]

    last_resp = None                                                          
    for payload in candidates:                                                
        r = session.post(url, json=payload, timeout=60)                        
        if r.status_code in (200, 201):                                     
            try:
                data = r.json()                                            
                return int(data["id"])                                       
            except Exception:
                try:
                    return int(r.text.strip())                               
                except Exception:
                    pass
        last_resp = r                                                          
        if r.status_code not in (400, 401, 403, 404, 409, 422):                
            break                                                              

    if last_resp is not None:                                                  
        try:
            print(f"[ERR] {url} → {last_resp.status_code} {last_resp.reason}") 
            ct = (last_resp.headers.get("content-type") or "").lower()         
            if "json" in ct:
                print(json.dumps(last_resp.json(), indent=2)[:1200])           
            else:
                print(last_resp.text[:1200])                                   
        except Exception:
            pass
        last_resp.raise_for_status()                                           
    raise RuntimeError("Failed to create labeled rectangle (no response).")    


#Takes a string and normalizes it by converting it to lowercase, replacing underscores with spaces, stripping whitespace, if the input is noen threats it liek an empty string
def _norm(s: str) -> str:                                                      
    return (s or "").lower().replace("_", " ").strip()                         


#Normalizes a namme and returns a tiny set of singular/plural variants (adds and removes trailing s-es) so barnacle match barnacles and so on.
def _variants(s: str) -> List[str]:                                           
    s = _norm(s)                                                               
    out = {s}                                                                  
    if s.endswith("es"):                                                      
        out.update({s[:-2], s[:-1]})                                           
    elif s.endswith("s"):                                                  
        out.add(s[:-1])                                                       
    else:
        out.add(s + "s")                                                       
    return list(out)                                                           


#Normalizes the biigle label names, generates plural singular variants for each model class name and builds a dict for any class that matches. 
def build_model2label(model_names: Dict[int, str], label_map: Dict[str, int]) -> Dict[int, int]:
    norm_label_map = {_norm(k): v for k, v in label_map.items()}               
    m2l: Dict[int, int] = {}                                                   
    for idx, raw in model_names.items():                                     
        for c in _variants(raw):                                              
            if c in norm_label_map:                                         
                m2l[idx] = norm_label_map[c]                                  
                break                                                          
    return m2l                                                                


#Initializes YOLO with your weights, grabs the models class index name. then convertsw those names to biigle labels ids using label_map. 
model = YOLO(WEIGHTS_PATH)                                                  
MODEL_NAMES = dict(model.names)                                              
MODEL2LABEL = build_model2label(MODEL_NAMES, LABEL_MAP)                      


#Prints which weights were loaded, what the model’s classes are, and the mapping from class index BIIGLE label_id; exits with code 3 if the mapping is empty.
print("Loaded weights:", WEIGHTS_PATH)                                       
print("Model classes :", MODEL_NAMES)                                          
print("Mapped (idx->label_id):", MODEL2LABEL)                                 
if not MODEL2LABEL:                                                            
    print("No model classes matched LABEL_MAP. Adjust LABEL_MAP or weights.")  
    sys.exit(3)                                                                


#Downloads an image from BIIGLE, runs yolo on it, clamps box to image bound, maps the model class to biigle label id and unless dry'_run post each rectangle as an annotation. 
def clamp(v: float, lo: int, hi: int) -> int:                                  
    return int(max(lo, min(hi, v)))                                            
def predict_and_annotate(image_id: int) -> int:                               
    try:
        img_bytes = download_image_bytes(image_id)                              
    except requests.HTTPError as e:                                            
        print(f"[WARN] download failed for image {image_id}: {e}")            
        return 0                                                               

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")                     
    w, h = img.size                                                            

    results = model.predict(img, conf=CONF_THR, verbose=False)                 
    if not results:                                                            
        return 0                                                               

    created = 0                                                                
    for r in results:                                                          
        if not getattr(r, "boxes", None):                                      
            continue                                                        
        for b in r.boxes:                                                      
            cls_id = int(b.cls.item())                                        
            label_id = MODEL2LABEL.get(cls_id)                                 
            if not label_id:                                                   
                continue                                                      

            # confidence from detector
            conf = float(getattr(b, "conf", [CONF_THR])[0])                   
            conf = float(max(0.0, min(1.0, conf)))                             

            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())                  
            x1 = clamp(x1, 0, w - 1)                                           
            y1 = clamp(y1, 0, h - 1)                                          
            x2 = clamp(x2, 0, w - 1)                                           
            y2 = clamp(y2, 0, h - 1)                                           
            if x2 <= x1 or y2 <= y1:                                           
                continue

            if DRY_RUN:                                                        
                print(f"[DRY] image {image_id}: box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                      f"label_id={label_id} conf={conf:.2f}")                  
                created += 1                                                   
                continue                                                      

            try:
                _ = post_labeled_rectangle(image_id, int(x1), int(y1), int(x2), int(y2), 
                                           int(label_id), conf)
                created += 1                                                   
            except requests.HTTPError as e:                                    
                print(f"[WARN] post failed for image {image_id}: {e}")        
    return created                                                            


#Gets all image IDs for your BIIGLE volume, runs YOLO + posting for each, 
#retries once on “Too Many Requests” (429), logs every 25 images, then prints how many annotations were created.
def main():                                                                   
    try:
        ids = list_volume_file_ids(VOLUME_ID)                                  
    except Exception as e:                                                  
        print(f"[ERR] Could not list files for volume {VOLUME_ID}: {e}")       
        sys.exit(4)                                                           

    if MAX_IMAGES > 0:                                                     
        ids = ids[:MAX_IMAGES]                                                 

    print(f"Found {len(ids)} images in volume {VOLUME_ID} (DRY_RUN={'ON' if DRY_RUN else 'OFF'})") 
    total = 0                                                                   
    for i, img_id in enumerate(ids, 1):                                        
        try:
            n = predict_and_annotate(img_id)                                   
            total += n                                                         
        except requests.HTTPError as e:                                        
            if e.response is not None and e.response.status_code == 429:       
                time.sleep(2.0)                                                
                total += predict_and_annotate(img_id)                        
            else:
                print(f"[WARN] error on image {img_id}: {e}")                  
        if i % 25 == 0:                                                        
            print(f"{i}/{len(ids)} processed… (created {total} annotations)")  
    print(f"Done. Created {total} annotations.")                               


#Required for terminal commands to work and run the script from terminal.
if __name__ == "__main__":                                                    
    main()                                                                     

