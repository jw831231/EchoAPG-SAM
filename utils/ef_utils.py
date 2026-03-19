import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import pandas as pd

def line_segment_intersection(p1, p2, p3, p4):

def find_intersection_points(line_p1, line_p2, contour_points):
    intersections = []
    n = len(contour_points)
    for i in range(n):
        edge_p1 = tuple(contour_points[i])
        edge_p2 = tuple(contour_points[(i + 1) % n])
        inter = line_segment_intersection(line_p1, line_p2, edge_p1, edge_p2)
        if inter:
            intersections.append(inter)
    if len(intersections) < 2:
        return None
    intersections = np.unique(np.array(intersections), axis=0)
    if len(intersections) < 2:
        return None
    dist_matrix = np.linalg.norm(intersections[:, np.newaxis] - intersections[np.newaxis, :], axis=2)
    idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    p1 = tuple(intersections[idx1])
    p2 = tuple(intersections[idx2])
    return p1, p2

def calculate_volume_from_mask(mask, visualize=False, spacing=0.66):

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    

    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    if not props:
        return 0.0, None
    largest_region = max(props, key=lambda x: x.area)
    mask = (labeled_mask == largest_region.label).astype(np.uint8)
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
    contour = max(contours, key=cv2.contourArea)
    

    hull = cv2.convexHull(contour)
    hull_points = hull.squeeze()
    if hull_points.ndim == 1:
        return 0.0, None
    apex_idx = np.argmin(hull_points[:, 1])
    apex = tuple(hull_points[apex_idx].astype(float))
    
    rect = cv2.minAreaRect(contour)  
    box = cv2.boxPoints(rect) 
    box = box.astype(float)
    

    bottom_two_idxs = np.argsort(box[:, 1])[-2:]
    base = tuple(np.mean(box[bottom_two_idxs], axis=0))
    

    dx = base[0] - apex[0]
    dy = base[1] - apex[1]
    L = np.sqrt(dx**2 + dy**2)
    if L < 20: 
        return 0.0, None
    
    angle_rad = np.arctan2(dy, dx)
    perp_cos = -np.sin(angle_rad)
    perp_sin = np.cos(angle_rad)
    
    L_mm = L * spacing
    n_slices = 20
    volume_mm3 = 0.0
    slice_lines = []
    line_length = max(mask.shape) * 2
    contour_points = contour.squeeze().astype(float)
    
    for i in range(1, n_slices + 1):
        t = i / n_slices
        px = apex[0] + t * dx
        py = apex[1] + t * dy
        
        x1 = px - line_length * perp_cos
        y1 = py - line_length * perp_sin
        x2 = px + line_length * perp_cos
        y2 = py + line_length * perp_sin
        
        inter_points = find_intersection_points((x1, y1), (x2, y2), contour_points)
        if inter_points is None:
            d_i = 0
            slice_line = None
        else:
            p1, p2 = inter_points
            d_i = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            slice_line = (p1, p2)
        
        slice_lines.append(slice_line)
        d_i_mm = d_i * spacing
        volume_mm3 += np.pi * (d_i_mm / 2)**2
    
    volume_mm3 *= (L_mm / n_slices)
    volume = volume_mm3 / 1000   # ml
    
    geometry = None
    if visualize:
        geometry = {
            'apex': apex,
            'base': base,
            'slice_lines': [sl for sl in slice_lines if sl is not None],
            'left_mitral': tuple(box[bottom_two_idxs[0]]),
            'right_mitral': tuple(box[bottom_two_idxs[1]]), 
        }
    
    return volume, geometry

def visualize_volume_geometry(mask, geometry, filename, phase, save_dir):
    vis_image = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    apex = (int(geometry['apex'][0]), int(geometry['apex'][1]))
    cv2.circle(vis_image, apex, 3, (0, 0, 255), -1)
    base = (int(geometry['base'][0]), int(geometry['base'][1]))
    cv2.circle(vis_image, base, 3, (255, 0, 0), -1)
    left_mitral = (int(geometry['left_mitral'][0]), int(geometry['left_mitral'][1]))
    cv2.circle(vis_image, left_mitral, 3, (0, 255, 0), -1)
    right_mitral = (int(geometry['right_mitral'][0]), int(geometry['right_mitral'][1]))
    cv2.circle(vis_image, right_mitral, 3, (0, 255, 0), -1)
    cv2.line(vis_image, apex, base, (0, 255, 255), 2)
    for line in geometry['slice_lines']:
        if line:
            p1 = (int(line[0][0]), int(line[0][1]))
            p2 = (int(line[1][0]), int(line[1][1]))
            cv2.line(vis_image, p1, p2, (255, 0, 255), 1)
    save_path = os.path.join(save_dir, f"{phase}_{filename}_volume_geometry.png")
    cv2.imwrite(save_path, vis_image)
    print(f"Volume geometry visualization saved for {phase} {filename} at {save_path}")

def calculate_s_old(filename, volume_tracings_df, filelist_df):
    original_row = filelist_df[filelist_df['FileName'] == filename]
    if original_row.empty:
        print(f"{filename} not found in FileList.csv")
        return None, None
    edv_ml = original_row['EDV'].values[0]
    esv_ml = original_row['ESV'].values[0]
 
    tracings = volume_tracings_df[volume_tracings_df['FileName'] == filename]
    if tracings.empty:
        print(f"No tracings for {filename}")
        return None, None
 
    ed_frame = tracings['Frame'].min()
    es_frame = tracings['Frame'].max()
 
    ed_tracings = tracings[tracings['Frame'] == ed_frame]
    if len(ed_tracings) != 21:
        print(f"Invalid tracings count for ED in {filename}")
        return None, None
    L_ed = np.sqrt((ed_tracings.iloc[0]['X2'] - ed_tracings.iloc[0]['X1'])**2 +
                   (ed_tracings.iloc[0]['Y2'] - ed_tracings.iloc[0]['Y1'])**2)
    d_i_ed = [np.sqrt((row['X2'] - row['X1'])**2 + (row['Y2'] - row['Y1'])**2)
              for _, row in ed_tracings.iloc[1:].iterrows()]
    V_pixels_ed = (np.pi * L_ed / 80) * sum(d**2 for d in d_i_ed)
    if V_pixels_ed == 0:
        print(f"Zero V_pixels_ed for {filename}")
        return None, None
    s_old_ed = np.float64((edv_ml * 1000) / V_pixels_ed)**(1/3)
 
    es_tracings = tracings[tracings['Frame'] == es_frame]
    if len(es_tracings) != 21:
        print(f"Invalid tracings count for ES in {filename}")
        return None, None
    L_es = np.sqrt((es_tracings.iloc[0]['X2'] - es_tracings.iloc[0]['X1'])**2 +
                   (es_tracings.iloc[0]['Y2'] - es_tracings.iloc[0]['Y1'])**2)
    d_i_es = [np.sqrt((row['X2'] - row['X1'])**2 + (row['Y2'] - row['Y1'])**2)
              for _, row in es_tracings.iloc[1:].iterrows()]
    V_pixels_es = (np.pi * L_es / 80) * sum(d**2 for d in d_i_es)
    if V_pixels_es == 0:
        print(f"Zero V_pixels_es for {filename}")
        return None, None
    s_old_es = np.float64((esv_ml * 1000) / V_pixels_es)**(1/3)
 
    print(f"{filename}: EDV_ml={edv_ml:.3f}, ESV_ml={esv_ml:.3f}, V_pixels_ed={V_pixels_ed:.3f}, V_pixels_es={V_pixels_es:.3f}")
    print(f"{filename}: s_old_ed={s_old_ed:.3f}, s_old_es={s_old_es:.3f}")
    return s_old_ed, s_old_es  
