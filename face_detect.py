import insightface
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

    
_app = FaceAnalysis(name="buffalo_l", providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
_app.prepare(ctx_id=0, det_size=(640, 640))  

_last_conf = 1e-6 
    
def get_face(image, threshold=0.9):
    global _last_conf
    data = []
    faces = _app.get(image)
    
    for face in faces:
        conf = face.det_score
        if conf < 0.9 and _last_conf > 0.9:
            conf = _last_conf * 0.9 + conf * 0.1
            
        if conf >= threshold:
            bbox = face.bbox
            kps = face.kps
            data.append((conf, bbox, kps))
            
        _last_conf = conf
        
    if len(data) > 0:
        data.sort(key=lambda x: -x[0])
        conf, bbox, kps = data[0]
        aligned = norm_crop(image, kps, image_size=112)
        return aligned, conf, bbox, kps
    
    return None
       
    