
import sys
import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QFileDialog,
                             QFrame, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from skimage.feature import graycomatrix, graycoprops

class ImageProcessor:
    def __init__(self):
        # Konfigurasi Folder Laporan
        self.report_dir = "laporan_analisis"
        self.img_dir = os.path.join(self.report_dir, "temp_img")
        self.html_log = []
        
        # Folder Database Logo (Harus diisi user dengan gambar .jpg/.png logo)
        self.db_dir = "database_logo"
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            print(f"INFO: Folder '{self.db_dir}' dibuat. Silakan isi dengan logo referensi.")
        
        # Reset Folder Laporan setiap kali jalan
        if os.path.exists(self.report_dir):
            shutil.rmtree(self.report_dir)
        os.makedirs(self.img_dir)
        
        # Inisialisasi ORB Detector (Feature Matching)
        self.orb = cv2.ORB_create(nfeatures=2500)

    def save_temp_image(self, img, name):
        """Menyimpan gambar temp dan mengembalikan path relatif untuk HTML"""
        filename = f"{name}.jpg"
        full_path = os.path.join(self.img_dir, filename)
        cv2.imwrite(full_path, img)
        return os.path.join("temp_img", filename)

    def save_plot_histogram(self, hist, name):
        """Menyimpan plot histogram matplotlib"""
        plt.figure(figsize=(6, 4))
        plt.plot(hist, color='orange', linewidth=2)
        plt.fill_between(range(len(hist)), hist.flatten(), color='orange', alpha=0.3)
        plt.title("Histogram Hue (Warna Dominan)")
        plt.xlabel("Nilai Hue (0-179)")
        plt.ylabel("Jumlah Pixel")
        plt.grid(True, alpha=0.3)
        
        filename = f"{name}.png"
        full_path = os.path.join(self.img_dir, filename)
        plt.savefig(full_path)
        plt.close()
        return os.path.join("temp_img", filename)

    def add_to_log(self, step_num, title, description, img_path=None, data_dict=None, extra_html=None):
        """Menambahkan langkah ke log HTML"""
        html_segment = f"""
        <div class="step">
            <div class="step-header">
                <span class="step-num">{step_num}</span>
                <h3>{title}</h3>
            </div>
            <p>{description}</p>
        """
        if img_path:
            html_segment += f'<div class="img-container"><img src="{img_path}" alt="{title}" class="process-img"></div>'
        
        if data_dict:
            html_segment += '<div class="data-box"><ul>'
            for k, v in data_dict.items():
                html_segment += f'<li><strong>{k}:</strong> {v}</li>'
            html_segment += '</ul></div>'
            
        if extra_html:
            html_segment += f'<div style="margin-top:15px;">{extra_html}</div>'
            
        html_segment += "</div>"
        self.html_log.append(html_segment)

    def generate_html_report(self, result_text, brand_text):
        """Membuat file HTML akhir"""
        css = """
        body { font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; color: #333; margin:0; padding: 20px; }
        .container { max-width: 950px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        h1 { text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; color: #2c3e50; }
        .step { background: #fff; border: 1px solid #eee; margin-bottom: 30px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }
        .step-header { display: flex; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #f0f0f0; padding-bottom: 10px; }
        .step-num { background: #3498db; color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px; }
        h3 { margin: 0; color: #2980b9; }
        .img-container { text-align: center; background: #fafafa; padding: 10px; border-radius: 5px; }
        .process-img { max-width: 100%; max-height: 500px; border: 1px solid #ddd; }
        .data-box { background: #e8f6f3; padding: 15px; margin-top: 15px; border-radius: 5px; border-left: 4px solid #1abc9c; }
        ul { margin: 0; padding-left: 20px; }
        li { margin-bottom: 5px; }
        
        .result-panel { display: flex; gap: 20px; margin-top: 40px; }
        .res-box { flex: 1; padding: 20px; text-align: center; border-radius: 8px; color: white; font-weight: bold; font-size: 1.2em; }
        .res-flavor { background: #27ae60; }
        .res-brand { background: #e67e22; }
        
        /* Table Style */
        .comp-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
        .comp-table th { background-color: #34495e; color: white; padding: 12px; text-align: left; }
        .comp-table td { border-bottom: 1px solid #ddd; padding: 10px; }
        .score-high { color: #27ae60; font-weight: bold; background-color: #e9f7ef; }
        .score-low { color: #c0392b; }
        """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Laporan Lengkap Analisis Minuman</title><style>{css}</style></head>
        <body>
            <div class="container">
                <h1>Laporan Analisis Citra Digital</h1>
                <p style="text-align:center; color:#7f8c8d;">Laporan ini digenerate secara otomatis oleh sistem Python OpenCV</p>
                <hr style="border:0; border-top:1px solid #eee; margin: 30px 0;">
                
                {''.join(self.html_log)}
                
                <div class="result-panel">
                    <div class="res-box res-flavor">
                        HASIL RASA<br><br>{result_text}
                    </div>
                    <div class="res-box res-brand">
                        HASIL MERK<br><br>{brand_text}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        with open(os.path.join(self.report_dir, "index.html"), "w") as f:
            f.write(html_content)
        return os.path.abspath(os.path.join(self.report_dir, "index.html"))

    # =========================================================================
    # CORE LOGIC: FEATURE MATCHING (BRAND)
    # =========================================================================
    def match_features_detailed(self, full_image):
        ref_images = [f for f in os.listdir(self.db_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not ref_images:
            return "DATABASE KOSONG", None, "Folder database_logo kosong", []

        # Resize Input untuk matching
        h, w = full_image.shape[:2]
        target_w = 1000
        scale = target_w / w
        target_h = int(h * scale)
        img_input = cv2.resize(full_image, (target_w, target_h))
        
        kp1, des1 = self.orb.detectAndCompute(img_input, None)
        if des1 is None:
            return "Gagal Fitur", None, "Gambar input terlalu polos", []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        best_brand = "TIDAK TERDETEKSI"
        highest_inliers = 0
        best_vis = None
        all_stats = []

        print(f"--- Matching dengan {len(ref_images)} logo ---")

        for filename in ref_images:
            path = os.path.join(self.db_dir, filename)
            img_ref = cv2.imread(path)
            if img_ref is None: continue
            
            kp2, des2 = self.orb.detectAndCompute(img_ref, None)
            stat = {'name': filename, 'raw': 0, 'inliers': 0, 'status': 'No Match'}

            if des2 is not None and len(kp2) > 5:
                # KNN Match
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Lowe's Ratio Test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                stat['raw'] = len(good_matches)

                if len(good_matches) >= 10:
                    # Geometric Verification (RANSAC)
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    try:
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if mask is not None:
                            inliers_count = np.sum(mask.ravel().tolist())
                            stat['inliers'] = inliers_count
                            
                            if inliers_count > 5:
                                stat['status'] = 'Valid Match'
                                # Update Best Candidate
                                if inliers_count > highest_inliers:
                                    highest_inliers = inliers_count
                                    best_brand = os.path.splitext(filename)[0].upper()
                                    
                                    # Draw visual only for the best/valid ones
                                    matchesMask = mask.ravel().tolist()
                                    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
                                    best_vis = cv2.drawMatches(img_input, kp1, img_ref, kp2, good_matches, None, **draw_params)
                            else:
                                stat['status'] = 'Gagal Geometri'
                    except:
                        pass
            
            all_stats.append(stat)

        # Sorting hasil
        all_stats.sort(key=lambda x: x['inliers'], reverse=True)
        
        info = f"Inliers Tertinggi: {highest_inliers}"
        if highest_inliers < 5:
            return "TIDAK DIKENAL", best_vis, "Skor terlalu rendah (<5)", all_stats

        return best_brand, best_vis, info, all_stats

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    def process_image(self, image_path):
        self.html_log = []
        
        # 1. LOAD IMAGE
        original = cv2.imread(image_path)
        if original is None: return None, None, "Gagal Load", "", ""
        
        path_orig = self.save_temp_image(original, "1_original")
        self.add_to_log(1, "Input Gambar Asli",
                        "Memuat gambar input dalam format RGB.",
                        path_orig, {"Dimensi": f"{original.shape[1]}x{original.shape[0]} px"})

        # 2. DETEKSI MERK (FEATURE MATCHING)
        detected_brand, match_img, match_info, stats = self.match_features_detailed(original)
        
        # Generate Tabel HTML untuk Merk
        table_html = """
        <table class="comp-table">
            <thead><tr><th>Logo Database</th><th>Raw Match</th><th>Inliers (Final)</th><th>Status</th></tr></thead><tbody>
        """
        for s in stats:
            cls = "score-high" if s['inliers'] > 5 else "score-low"
            table_html += f"<tr><td>{s['name']}</td><td>{s['raw']}</td><td class='{cls}'>{s['inliers']}</td><td>{s['status']}</td></tr>"
        table_html += "</tbody></table>"
        
        path_match = None
        if match_img is not None:
            path_match = self.save_temp_image(match_img, "2_brand_matching")
            
        self.add_to_log(2, "Deteksi Merk (Feature Matching)",
                        "Mencocokkan fitur unik pada gambar asli dengan database logo menggunakan ORB + RANSAC.",
                        path_match, {"Merk Terdeteksi": detected_brand, "Info": match_info}, table_html)

        # 3. PREPROCESSING (GRAYSCALE)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        path_gray = self.save_temp_image(gray, "3_grayscale")
        self.add_to_log(3, "Konversi Grayscale", "Mengubah citra ke derajat keabuan.", path_gray)

        # 4. PREPROCESSING (OTSU THRESHOLD)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        path_thresh = self.save_temp_image(thresh, "4_otsu")
        self.add_to_log(4, "Thresholding Otsu", f"Binarisasi otomatis (Nilai ambang: {ret}).", path_thresh)

        # 5. MORFOLOGI
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Tutup lubang
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)   # Hapus noise
        path_morph = self.save_temp_image(morph, "5_morfologi")
        self.add_to_log(5, "Operasi Morfologi", "Closing & Opening untuk menyempurnakan bentuk objek.", path_morph)

        # 6. SEGMENTASI KONTUR
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Inisialisasi Default
#        size_label = "Unknown"
        flavor = "Unknown"
        color_swatch_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_contour = original.copy()

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Gambar kotak deteksi
            cv2.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 0), 3)
            path_cont = self.save_temp_image(img_contour, "6_contour_box")
            self.add_to_log(6, "Deteksi Kontur Utama", "Menemukan objek terbesar sebagai kemasan.", path_cont,
                            {"Posisi X": x, "Posisi Y": y, "Lebar": w, "Tinggi": h})

            # 7. CROP ROI
            roi_color = original[y:y+h, x:x+w]
            path_roi = self.save_temp_image(roi_color, "7_roi_crop")
            self.add_to_log(7, "Cropping ROI", "Memotong area kemasan untuk analisis fokus.", path_roi)

#            # 8. ANALISIS UKURAN (GEOMETRI)
#            aspect_ratio = float(h) / w
#            if aspect_ratio > 1.4: size_label = "Tinggi (Tall)"
#            elif aspect_ratio >= 0.95: size_label = "Sedang (Medium)"
#            else: size_label = "Pendek (Short)"
#
#            self.add_to_log(8, "Analisis Ukuran", "Menghitung Aspect Ratio (Tinggi/Lebar).", None,
#                            {"Tinggi": f"{h} px", "Lebar": f"{w} px", "Ratio": f"{aspect_ratio:.2f}", "Kategori": size_label})

            # 9. MASKING WARNA (CLEANING)
            hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            lower_mask = np.array([0, 25, 40])
            upper_mask = np.array([179, 255, 255])
            mask_vibrant = cv2.inRange(hsv_roi, lower_mask, upper_mask)
            path_mask = self.save_temp_image(mask_vibrant, "9_color_mask")
            self.add_to_log(9, "Masking Warna", "Memisahkan warna background dari teks putih/bayangan hitam (Putih = Dihitung).", path_mask)

            # 10. HISTOGRAM WARNA
            hist_hue = cv2.calcHist([hsv_roi], [0], mask_vibrant, [180], [0, 180])
            path_hist = self.save_plot_histogram(hist_hue, "10_histogram")
            dominant_hue = np.argmax(hist_hue)
            mean_val = cv2.mean(hsv_roi, mask=mask_vibrant)
            
            self.add_to_log(10, "Histogram Hue", "Grafik distribusi warna Hue pada area masking.", path_hist,
                            {"Puncak Hue": f"{dominant_hue}", "Avg Brightness": f"{mean_val[2]:.2f}"})

            # 11. KLASIFIKASI RASA
            dh = dominant_hue
            color_name = "Undef"
            if (dh >= 0 and dh <= 7) or (dh >= 174 and dh <= 180):
                color_name, flavor = "Merah", "Stroberi / Apel"
            elif dh >= 8 and dh <= 20:
                color_name, flavor = "Oranye", "Karamel / Jeruk / Mangga"
            elif dh >= 21 and dh <= 48:
                color_name, flavor = "Kuning", "Coklat / Pisang / Lemon"
            elif dh >= 49 and dh <= 85:
                color_name, flavor = "Hijau", "Melon / Matcha"
            elif dh >= 86 and dh <= 128:
                color_name, flavor = "Biru", "Full Cream /Plain Milk / Vanilla"
            elif dh >= 129 and dh <= 165:
                color_name, flavor = "Ungu", "Anggur / Taro"
            elif dh >= 166 and dh <= 173:
                color_name, flavor = "Pink", "Sroberi / Jambu"
                
            if color_name in ["Oranye", "Merah", "Kuning"] and mean_val[2] < 90:
                 color_name, flavor = "Coklat", "Mocha / Coklat / Kopi"

            color_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
            color_swatch[:] = (dominant_hue, 255, 230)
            color_swatch_bgr = cv2.cvtColor(color_swatch, cv2.COLOR_HSV2BGR)
            path_swatch = self.save_temp_image(color_swatch_bgr, "11_final_color")
            
            self.add_to_log(11, "Hasil Analisis Warna", "Menentukan rasa berdasarkan Hue Dominan.", path_swatch,
                            {"Warna Terdeteksi": color_name, "Estimasi Rasa": flavor})

            # 12. GLCM (TEKSTUR)
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            glcm = graycomatrix(roi_gray, [1], [0], 256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            self.add_to_log(12, "Analisis Tekstur (GLCM)", "Statistik permukaan citra.", None,
                            {"Contrast": f"{contrast:.4f}", "Energy": f"{energy:.4f}", "Homogeneity": f"{homogeneity:.4f}"})

        else:
            self.add_to_log(6, "Error", "Tidak menemukan kontur objek pada gambar.")

        final_result = f"Rasa {flavor}"
        report_path = self.generate_html_report(final_result, detected_brand)
        
        return img_contour, color_swatch_bgr, final_result, detected_brand, report_path

class ResizableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(self.sizePolicy().Ignored, self.sizePolicy().Ignored)
        self._pixmap = None
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Area Gambar")

    def set_image(self, cv_img):
        if cv_img is None: return
        height, width, channel = cv_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self._pixmap = QPixmap.fromImage(qImg)
        self.update_view()
        self.setStyleSheet("border: 2px solid #3498db; background-color: #000;")

    def update_view(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_view()
        super().resizeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Analisis Minuman (Complete Report)")
        self.resize(1100, 750)
        self.processor = ImageProcessor()
        self.report_path = ""

        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Analisis Citra Minuman Kemasan")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50; margin: 10px 0;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Image Layout
        img_layout = QHBoxLayout()
        self.lbl_left = ResizableLabel()
        self.lbl_right = ResizableLabel()
        img_layout.addWidget(self.lbl_left, 2)
        img_layout.addWidget(self.lbl_right, 1)
        main_layout.addLayout(img_layout, stretch=1)

        # Labels Hasil
        control_layout = QVBoxLayout()
        
        self.lbl_result = QLabel("Rasa: -")
        self.lbl_result.setStyleSheet("font-size: 18px; font-weight: bold; color: white; background-color: #27ae60; padding: 10px; border-radius: 5px; margin-top: 10px;")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        
        self.lbl_brand = QLabel("Merk: -")
        self.lbl_brand.setStyleSheet("font-size: 20px; font-weight: bold; color: white; background-color: #e67e22; padding: 10px; border-radius: 5px;")
        self.lbl_brand.setAlignment(Qt.AlignCenter)
        
        control_layout.addWidget(self.lbl_result)
        control_layout.addWidget(self.lbl_brand)

        # Tombol
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Pilih Gambar")
        self.btn_load.setStyleSheet("font-size: 14px; padding: 10px; font-weight: bold;")
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_report = QPushButton("Buka Laporan Lengkap")
        self.btn_report.setStyleSheet("font-size: 14px; padding: 10px; font-weight: bold;")
        self.btn_report.setEnabled(False)
        self.btn_report.clicked.connect(self.open_report)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_report)
        control_layout.addLayout(btn_layout)
        main_layout.addLayout(control_layout)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Buka Gambar', '.', "Image Files (*.png *.jpg *.jpeg)")
        if fname:
            self.lbl_result.setText("Sedang Memproses...")
            self.lbl_brand.setText("Analisis...")
            QApplication.processEvents()
            
            img_res, img_col, txt_res, txt_brand, r_path = self.processor.process_image(fname)
            
            if img_res is not None:
                self.lbl_left.set_image(img_res)
                self.lbl_right.set_image(img_col)
                self.lbl_result.setText(txt_res)
                self.lbl_brand.setText(txt_brand)
                self.report_path = r_path
                self.btn_report.setEnabled(True)
                # Auto open report (Optional)
                # self.open_report()
            else:
                QMessageBox.warning(self, "Error", txt_res)

    def open_report(self):
        if self.report_path:
            import webbrowser
            webbrowser.open(f'file:///{self.report_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
