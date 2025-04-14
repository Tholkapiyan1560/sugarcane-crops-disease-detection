import os
import cv2
import numpy as np
from PIL import Image
import piexif
from math import radians, cos, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, make_response

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-strong-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ------------------- Helper Functions for User Input -------------------
def get_int_input(prompt):
    # Not used in the web version.
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

# ------------------- Image Processing Functions -------------------
def split_image_horizontally(image, num_parts):
    """Extracts the upper part of the image based on the number of parts."""
    height = image.shape[0]
    part_height = height // num_parts
    return image[:part_height, :] if part_height > 0 else image

def stitch_images_vertically(images):
    """Stitches images vertically."""
    return cv2.vconcat(images) if images else None

def stitch_images_horizontally(images):
    """Stitches images horizontally."""
    return cv2.hconcat(images) if images else None

def split_image_vertically(image, num_parts):
    """Splits an image into vertical parts."""
    width = image.shape[1]
    part_width = width // num_parts
    return [image[:, i * part_width:(i + 1) * part_width] for i in range(num_parts)]

def save_image(image, path):
    """Saves an image to the specified path."""
    if image is not None:
        cv2.imwrite(path, image)
        print(f"Saved: {path}")

# ------------------- GPS Functions -------------------
def extract_gps(image_path):
    try:
        image = Image.open(image_path)
        exif_data = piexif.load(image.info.get('exif', b''))
        gps_info = exif_data.get("GPS", {})
        if not gps_info:
            return None, None
        def convert_to_decimal(gps_value):
            degrees = gps_value[0][0] / gps_value[0][1]
            minutes = gps_value[1][0] / gps_value[1][1]
            seconds = gps_value[2][0] / gps_value[2][1]
            return degrees + (minutes / 60) + (seconds / 3600)
        latitude = convert_to_decimal(gps_info.get(piexif.GPSIFD.GPSLatitude))
        longitude = convert_to_decimal(gps_info.get(piexif.GPSIFD.GPSLongitude))
        if gps_info.get(piexif.GPSIFD.GPSLatitudeRef) == b'S':
            latitude = -latitude
        if gps_info.get(piexif.GPSIFD.GPSLongitudeRef) == b'W':
            longitude = -longitude
        return latitude, longitude
    except Exception as e:
        print(f"GPS extraction error: {e}")
        return None, None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters"""
    R = 6371000  # Earth radius in meters
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = (np.sin(dLat / 2) ** 2 +
         np.cos(radians(lat1)) * np.cos(radians(lat2)) *
         np.sin(dLon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

# ------------------- Detection Functions -------------------
def detect_sugarcane(img):
    """Detect sugarcane crops using color and texture analysis"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]

def detect_pokkah_boeng(img, sugarcane_boxes):
    """Detect Pokkah Boeng disease within sugarcane regions based on symptoms."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    color_mask = cv2.bitwise_or(mask_yellow, mask_red)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = cv2.convertScaleAbs(laplacian)
    _, texture_mask = cv2.threshold(abs_lap, 50, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(color_mask, texture_mask)
    v_channel = hsv[:, :, 2]
    brightness_mask = cv2.inRange(v_channel, 0, 220)
    combined_mask = cv2.bitwise_and(combined_mask, brightness_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]

# ------------------- Edge Detection for Planted Area -------------------
def detect_planted_area(img):
    """
    Detects the planted crop area using edge detection.
    Returns:
      - planted_area: total number of pixels in the filled mask.
      - mask: the filled mask.
      - marked_img: the image with edge contours drawn in blue.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = np.zeros_like(gray)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    planted_area = cv2.countNonZero(mask)
    marked_img = img.copy()
    cv2.drawContours(marked_img, contours, -1, (255, 0, 0), 2)
    return planted_area, mask, marked_img

# ------------------- Original Yield Analysis Function -------------------
def calculate_yield(sugarcane_boxes, disease_boxes, img_area):
    """
    Calculate yield percentage based on detected crop areas.
    """
    total_crop_area = sum(w * h for (x, y, w, h) in sugarcane_boxes)
    disease_area = sum(w * h for (x, y, w, h) in disease_boxes)
    healthy_area = max(0, total_crop_area - disease_area)
    yield_percent = (healthy_area / total_crop_area) * 100 if total_crop_area > 0 else 0
    crop_coverage = (total_crop_area / img_area) * 100 if img_area > 0 else 0
    return yield_percent, crop_coverage

# ------------------- Coordinate Mapping -------------------
def get_box_coordinates(box, height_map, gps_data):
    x, y, w, h = box
    center_y = y + h // 2
    for idx, (h_start, h_end) in enumerate(height_map):
        if h_start <= center_y < h_end:
            lat, lon = gps_data[::-1][idx]
            if lat and lon:
                return (lat, lon, x, center_y)
    return None

# ------------------- Clustering Functions -------------------
def cluster_disease_regions(points, max_distance=5):
    clusters = []
    for lat, lon, x, y in points:
        added = False
        for cluster in clusters:
            clat, clon, cx, cy, count = cluster
            distance = haversine_distance(lat, lon, clat, clon)
            if distance <= max_distance:
                total = count + 1
                cluster[0] = (clat * count + lat) / total
                cluster[1] = (clon * count + lon) / total
                cluster[2] = (cx * count + x) / total
                cluster[3] = (cy * count + y) / total
                cluster[4] = total
                added = True
                break
        if not added:
            clusters.append([lat, lon, x, y, 1])
    return clusters

# ------------------- Visualization Functions -------------------
def draw_detections(img, boxes, color=(0, 0, 255), thickness=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img

def draw_clusters(img, clusters, color=(0, 255, 0), thickness=2):
    for lat, lon, x, y, count in clusters:
        cv2.circle(img, (int(x), int(y)), 20, color, thickness)
        cv2.putText(img, f"{count}", (int(x) - 10, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img

# ------------------- Gap Analysis Function (Method 1) -------------------
def perform_gap_analysis(img, gap_output_folder, analysis_title="Gap Analysis", height_map=None, gps_data=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 25, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    gap_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gap_boxes.append((x, y, w, h))
    gap_image_path = os.path.join(gap_output_folder, "gap_analysis_output.jpg")
    save_image(result_img, gap_image_path)
    report_file = os.path.join(gap_output_folder, "gap_analysis_report.txt")
    with open(report_file, "w") as f:
        f.write(f"{analysis_title}\n")
        f.write("=============================\n")
        f.write(f"Number of Gap Areas Detected: {len(gap_boxes)}\n\n")
        f.write("GPS details not applied for gap analysis in this version.\n")
    print(f"Gap analysis report saved at: {report_file}")
    return result_img, gap_boxes

# ------------------- Process Region for Web -------------------
def process_region_web(base_folder, region_num, start_img, end_img, num_horizontal_parts,
                       vertical_preview, num_vsplits,
                       vertical_split, num_vertical_parts, selected_parts):
    """Process a single region using web form inputs and return a dictionary of results."""
    region_folder = os.path.join(app.config['OUTPUT_FOLDER'], f"Region_{region_num}")
    os.makedirs(region_folder, exist_ok=True)
    images = []
    gps_data = []
    height_map = []
    current_height = 0
    # Load images
    for i in range(start_img, end_img + 1):
        image_path = os.path.join(base_folder, f"DJI_{i:04d}.JPG")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image at path: {image_path}")
            continue
        lat, lon = extract_gps(image_path)
        if i == start_img:
            cropped = img
        else:
            h = img.shape[0] // num_horizontal_parts
            cropped = img[:h, :] if h > 0 else img
        gps_data.append((lat, lon))
        height_map.append((current_height, current_height + cropped.shape[0]))
        current_height += cropped.shape[0]
        images.append(cropped)
    stitched = stitch_images_vertically(images[::-1])
    stitched_path = os.path.join(region_folder, f"region{region_num}_stitched.jpg")
    save_image(stitched, stitched_path)
    # Vertical Preview Feature
    if vertical_preview.lower() == 'yes' and num_vsplits:
        try:
            vsplits = int(num_vsplits)
            preview_img = stitched.copy()
            h, w = preview_img.shape[:2]
            part_width = w // vsplits
            for i in range(1, vsplits):
                x_coord = i * part_width
                cv2.line(preview_img, (x_coord, 0), (x_coord, h), (255, 255, 255), 3)
            vprev_folder = os.path.join(region_folder, "Vertical_Preview")
            os.makedirs(vprev_folder, exist_ok=True)
            preview_path = os.path.join(vprev_folder, f"region{region_num}_vertical_preview.jpg")
            save_image(preview_img, preview_path)
        except Exception as e:
            print(f"Vertical preview failed: {e}")
    # Gap Analysis
    gap_folder = os.path.join(region_folder, "gap_analysis")
    os.makedirs(gap_folder, exist_ok=True)
    gap_img, _ = perform_gap_analysis(stitched, gap_folder, analysis_title=f"Region {region_num} Gap Analysis")
    # Disease Detection
    sugarcane_boxes = detect_sugarcane(stitched)
    disease_boxes = detect_pokkah_boeng(stitched, sugarcane_boxes)
    img_area = stitched.shape[0] * stitched.shape[1]
    # Edge Detection (region-level)
    _, _, edge_marked = detect_planted_area(stitched)
    edge_output_path = os.path.join(region_folder, "edge_detection.jpg")
    save_image(edge_marked, edge_output_path)
    # Yield Analysis
    yield_percent, crop_coverage = calculate_yield(sugarcane_boxes, disease_boxes, img_area)
    disease_img = draw_detections(stitched.copy(), disease_boxes)
    save_image(disease_img, os.path.join(region_folder, "disease_detections.jpg"))
    disease_points = []
    for box in disease_boxes:
        coords = get_box_coordinates(box, height_map, gps_data)
        if coords:
            disease_points.append(coords)
    clusters = cluster_disease_regions(disease_points)
    cluster_img = draw_clusters(stitched.copy(), clusters)
    save_image(cluster_img, os.path.join(region_folder, "disease_clusters.jpg"))
    # Vertical Split for Focus
    final_stitched = stitched
    if vertical_split.lower() == 'yes' and num_vertical_parts and selected_parts:
        try:
            n_parts = int(num_vertical_parts)
            vertical_parts = split_image_vertically(stitched, n_parts)
            if '-' in selected_parts:
                sp, ep = map(int, selected_parts.split('-'))
                sel_indices = list(range(sp - 1, ep))
            else:
                sel_indices = [int(selected_parts) - 1]
            selected_images = [vertical_parts[i] for i in sel_indices if 0 <= i < len(vertical_parts)]
            final_stitched = stitch_images_horizontally(selected_images)
            final_path = os.path.join(region_folder, f"region{region_num}_final.jpg")
            save_image(final_stitched, final_path)
        except Exception as e:
            print(f"Vertical splitting failed: {e}")
    return {
        'stitched': final_stitched,
        'height_map': height_map,
        'gps_data': gps_data,
        'region_width': final_stitched.shape[1] if final_stitched is not None else 0,
        'yield_percent': yield_percent,
        'crop_coverage': crop_coverage
    }

# ------------------- Yield Gap Analysis Visualization -------------------
def plot_yield_gap_analysis():
    actual_yield = np.array([10, 12, 15, 14, 13])
    potential_yield = np.array([15, 15, 15, 15, 15])
    yield_gap = potential_yield - actual_yield
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(actual_yield)), yield_gap, color='orange', label='Yield Gap')
    plt.plot(range(len(actual_yield)), actual_yield, marker='o', label='Actual Yield', color='blue')
    plt.plot(range(len(actual_yield)), potential_yield, marker='x', label='Potential Yield', color='green')
    plt.xlabel('Field Index')
    plt.ylabel('Yield (tons per hectare)')
    plt.title('Yield Gap Analysis')
    plt.legend()
    gap_plot_path = os.path.join(app.config['OUTPUT_FOLDER'], "yield_gap_analysis.jpg")
    plt.savefig(gap_plot_path)
    plt.close()
    return gap_plot_path

# ------------------- Download Routes -------------------
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download/<filetype>')
def download_file(filetype):
    output_folder = app.config['OUTPUT_FOLDER']
    if filetype == 'marked':
        return send_from_directory(output_folder, "FINAL_PANORAMA_MARKED.jpg", as_attachment=True)
    elif filetype == 'gap':
        return send_from_directory(output_folder, "FINAL_PANORAMA_GAP.jpg", as_attachment=True)
    elif filetype == 'edge':
        return send_from_directory(output_folder, "edge_detection.jpg", as_attachment=True)
    elif filetype == 'report':
        return send_from_directory(output_folder, "final_report.txt", as_attachment=True)
    else:
        flash("Invalid file type requested.")
        return redirect(url_for('index'))

# ------------------- Flask Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            region_count = int(request.form.get('region_count', 1))
        except:
            region_count = 1
        overall_yield_sum = 0.0
        overall_coverage_sum = 0.0
        region_order = []
        base_folder = app.config['UPLOAD_FOLDER']
        # Process each region using form inputs
        for i in range(1, region_count + 1):
            try:
                region_start = int(request.form.get(f'region_{i}_start', 1))
                region_end = int(request.form.get(f'region_{i}_end', 1))
                region_horizontal = int(request.form.get(f'region_{i}_horizontal', 1))
            except Exception as e:
                flash(f"Invalid input for region {i}: {e}")
                return redirect(request.url)
            vertical_preview = request.form.get(f'region_{i}_vertical_preview', 'no')
            num_vsplits = request.form.get(f'region_{i}_num_vsplits', '')
            vertical_split = request.form.get(f'region_{i}_vertical_split', 'no')
            num_vertical_parts = request.form.get(f'region_{i}_num_vertical_parts', '')
            selected_parts = request.form.get(f'region_{i}_selected_parts', '')
            region_data = process_region_web(base_folder, i, region_start, region_end, region_horizontal,
                                             vertical_preview, num_vsplits,
                                             vertical_split, num_vertical_parts, selected_parts)
            overall_yield_sum += region_data.get('yield_percent', 0)
            overall_coverage_sum += region_data.get('crop_coverage', 0)
            if i == 1:
                region_order.append(region_data)
            else:
                order = request.form.get(f'region_{i}_order', 'r').lower()
                if order == 'l':
                    region_order.insert(0, region_data)
                else:
                    region_order.append(region_data)
        # Create final panorama from region order
        if region_order:
            final_images = [r['stitched'] for r in region_order if r.get('stitched') is not None]
            if len(final_images) > 1:
                min_height = min(img.shape[0] for img in final_images)
                resized = [cv2.resize(img, (img.shape[1], min_height)) for img in final_images]
                panorama = stitch_images_horizontally(resized)
            else:
                panorama = final_images[0] if final_images else None
            if panorama is not None:
                output_folder = app.config['OUTPUT_FOLDER']
                panorama_path = os.path.join(output_folder, "FINAL_PANORAMA.jpg")
                save_image(panorama, panorama_path)
                # Gap analysis on final panorama
                full_gap_folder = os.path.join(output_folder, "gap_analysis")
                os.makedirs(full_gap_folder, exist_ok=True)
                gap_panorama, _ = perform_gap_analysis(panorama, full_gap_folder, analysis_title="Full Land Gap Analysis")
                gap_panorama_path = os.path.join(output_folder, "FINAL_PANORAMA_GAP.jpg")
                save_image(gap_panorama, gap_panorama_path)
                # Perform edge detection on final panorama
                _, _, final_edge_marked = detect_planted_area(panorama)
                edge_output_path = os.path.join(output_folder, "edge_detection.jpg")
                save_image(final_edge_marked, edge_output_path)
                stitched_area = panorama.shape[0] * panorama.shape[1]
                sugarcane_boxes = detect_sugarcane(panorama)
                disease_boxes = detect_pokkah_boeng(panorama, sugarcane_boxes)
                pan_yield, pan_coverage = calculate_yield(sugarcane_boxes, disease_boxes, stitched_area)
                panorama_gps = []
                current_x = 0
                for region in region_order:
                    region_width = region.get('region_width', 0)
                    for lat, lon in region.get('gps_data', []):
                        if lat and lon:
                            panorama_gps.append((lat, lon, current_x))
                    current_x += region_width
                disease_points = []
                for (x, y, w, h) in disease_boxes:
                    center_x = x + w // 2
                    for lat, lon, region_x in panorama_gps:
                        if region_x <= center_x < region_x + 2000:
                            disease_points.append((lat, lon, x - region_x, y))
                            break
                clusters = cluster_disease_regions(disease_points)
                marked_panorama = draw_detections(panorama.copy(), disease_boxes)
                marked_panorama = draw_clusters(marked_panorama, clusters)
                marked_path = os.path.join(output_folder, "FINAL_PANORAMA_MARKED.jpg")
                save_image(marked_panorama, marked_path)
                # Create final report
                report_path = os.path.join(output_folder, "final_report.txt")
                with open(report_path, "w") as f:
                    f.write("Final Combined Disease and Yield Report\n")
                    f.write("=========================================\n")
                    f.write(f"Panorama Crop Coverage: {pan_coverage:.2f}%\n")
                    f.write(f"Panorama Estimated Yield: {pan_yield:.2f}%\n")
                    avg_yield = overall_yield_sum / region_count if region_count else 0
                    avg_coverage = overall_coverage_sum / region_count if region_count else 0
                    f.write(f"Average Region Yield: {avg_yield:.2f}%\n")
                    f.write(f"Average Region Crop Coverage: {avg_coverage:.2f}%\n")
                    f.write("\nDisease Clusters in Panorama:\n")
                    for i, (lat, lon, x, y, count) in enumerate(clusters, 1):
                        f.write(f"Cluster {i}:\n")
                        f.write(f"GPS: {lat:.6f}, {lon:.6f}\n")
                        f.write(f"Image Coordinates: ({x}, {y})\n")
                        f.write(f"Detection Count: {count}\n\n")
                final_marked_url = url_for('output_file', filename="FINAL_PANORAMA_MARKED.jpg")
                gap_url = url_for('output_file', filename="FINAL_PANORAMA_GAP.jpg")
                edge_url = url_for('output_file', filename="edge_detection.jpg")
                report_url = url_for('output_file', filename="final_report.txt")
            else:
                final_marked_url = gap_url = edge_url = report_url = None
        else:
            final_marked_url = gap_url = edge_url = report_url = None
        # Generate yield gap analysis plot
        gap_plot_path = os.path.join(app.config['OUTPUT_FOLDER'], "yield_gap_analysis.jpg")
        actual_yield = np.array([10, 12, 15, 14, 13])
        potential_yield = np.array([15, 15, 15, 15, 15])
        yield_gap = potential_yield - actual_yield
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(actual_yield)), yield_gap, color='orange', label='Yield Gap')
        plt.plot(range(len(actual_yield)), actual_yield, marker='o', label='Actual Yield', color='blue')
        plt.plot(range(len(actual_yield)), potential_yield, marker='x', label='Potential Yield', color='green')
        plt.xlabel('Field Index')
        plt.ylabel('Yield (tons per hectare)')
        plt.title('Yield Gap Analysis')
        plt.legend()
        plt.savefig(gap_plot_path)
        plt.close()
        gap_plot_url = url_for('output_file', filename="yield_gap_analysis.jpg")
        return render_template('result.html',
                               marked_url=final_marked_url,
                               gap_url=gap_url,
                               edge_url=edge_url,
                               report_url=report_url,
                               yield_percent=pan_yield if panorama is not None else 0,
                               crop_coverage=pan_coverage if panorama is not None else 0,
                               gap_plot_url=gap_plot_url)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
