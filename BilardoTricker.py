import cv2
import numpy as np
import time

# Videoyu başlat
cap = cv2.VideoCapture('vid_1.avi')

# Çıktı video dosyası için yazıcıyı başlat
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (output_width, output_height))

# Başlangıç zamanı ve hareket süresini tutmak için değişkenler
start_time = time.time()
total_motion_time = 0
total_motion_distance = 0

# Önceki çerçeve ve hareketli olup olmadığını kontrol etmek için değişkenler
prev_frame = None
prev_cx, prev_cy = None, None
stop_threshold = 1  # Saniye cinsinden durma eşiği

# Son hesaplama zamanı
last_calculation_time = time.time()

# Özel işaretin çapı ve rengi
sign_radius = 10
big_sign_radius = 10
sign_color = (0, 0, 255)  # Kırmızı

# Hareket izini saklamak için boş bir liste oluştur
movement_trace = []

# Hareket zamanını hesaplama fonksiyonu
def detect_red_ball_motion(frame, prev_cx, prev_cy, stop_threshold):
    global total_motion_time, total_motion_distance
    
    # Kırmızı rengin HSV renk uzayındaki aralığını belirleme
    lower_red = np.array([160, 150, 150])
    upper_red = np.array([200, 255, 255])

    # Kırmızı renkleri içeren maske oluşturma
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask=cv2.dilate(mask,(5,5),1)
    # Kırmızı renklerin konturunu bulma
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("mask",mask)
    # Kırmızı top hareket ettiği zamanı kontrol etme ve hareket mesafesini hesaplama
    for contour in contours:
        # Kırmızı topun çevresini çizme
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Kırmızı noktanın merkezini bulma
        cx = x + w // 2
        cy = y + h // 2

        # Başlangıç koordinatları belirlenmemişse, mevcut koordinatları kullanarak başlangıç koordinatlarını belirle
        if prev_cx is None or prev_cy is None:
            prev_cx, prev_cy = cx, cy
            continue

        # Hareket mesafesini hesaplama
        motion_distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

        # Hareketli olup olmadığını kontrol etme ve hareket süresini hesaplama
        current_time = time.time()
        elapsed_time = current_time - last_calculation_time
        speed = calculate_speed(prev_cx, prev_cy, cx, cy, elapsed_time)

        # Hareket sonrası değerleri döndürme
        return frame, prev_cx, prev_cy, speed, (prev_cx, prev_cy), (cx, cy)

    # Eğer kırmızı top tespit edilemediyse, None döndür
    return None, prev_cx, prev_cy, 0, None, None

# Hızı hesaplama fonksiyonu
def calculate_speed(prev_cx, prev_cy, cx, cy, elapsed_time):
    return np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) / elapsed_time

while True:
    # Frame yakalama
    ret, frame = cap.read()
    if not ret:
        break
    
    # Kırmızı topun hareketini tespit etme
    result = detect_red_ball_motion(frame, prev_cx, prev_cy, stop_threshold)
    if result is not None:
        frame, prev_cx, prev_cy, speed, start_point, end_point = result
    else:
        speed = 0
        start_point, end_point = None, None

    # Noktaların silinmemesi için kopya kare oluşturma
    if frame is not None:
        frame_with_dots = frame.copy()

        # Hareketin izini oluşturma
        if end_point is not None:
            movement_trace.append(end_point)

            # İz çizgisi çizme
            for i in range(1, len(movement_trace)):
                cv2.line(frame_with_dots, movement_trace[i-1], movement_trace[i], (0, 255, 0), 2)

        # Kırmızı noktayı ekrana yazdırma
        if start_point is not None and end_point is not None:
            cv2.circle(frame_with_dots, start_point, big_sign_radius, sign_color, 1)
            cv2.circle(frame_with_dots, end_point, big_sign_radius, sign_color, 1)

            # İki büyük kırmızı nokta arasındaki mesafeyi hesapla
            distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
            # Mesafeyi terminale yazdır
            print(f"İki büyük kırmızı nokta arasındaki mesafe: {distance:.2f} piksel")

            # Yön oku çizme
            cv2.arrowedLine(frame_with_dots, start_point, end_point, (255, 0, 0), 2)

        # Eğer kırmızı top hareket etmiyorsa ve hareket öncesi koordinatlar mevcutsa, durduğu yere kırmızı bir nokta yerleştir
        if speed == 0 and prev_cx is not None and prev_cy is not None:
            cv2.circle(frame_with_dots, (prev_cx, prev_cy), big_sign_radius, sign_color, 1)

        # Her 2 saniyede bir kırmızı nesnenin hızını ekrana yazdırma
        if (time.time() - last_calculation_time >= 2) and speed < 50:
            print(f"Kırmızı nesnenin hızı: {speed:.2f} piksel/sn")
            last_calculation_time = time.time()

        # Kırmızı noktanın zaman bilgisini ekrana yazdırma
        current_time_str = time.strftime("%H:%M:%S", time.localtime(time.time()))
        cv2.putText(frame_with_dots, f"Aktüel Zaman: {current_time_str}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Sonuçları gösterme
        cv2.putText(frame_with_dots, f"Topun Hizi: {speed:.2f} piksel/sn", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame_with_dots, f"Topun gittiği yol: {speed:.2f} piksel/sn", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3)

        # Videoyu gösterme
        cv2.imshow('Video', frame_with_dots)
        
        # Çıktı videoya çerçeveyi yazma
        out.write(frame_with_dots)

    # Çıkış için 'q' tuşuna basılmasını bekleyin
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Son çıktı karesini kaydetme
cv2.imwrite('final_output_frame.jpg', frame_with_dots)

# Videoyu serbest bırak
cap.release()
out.release()

cv2.destroyAllWindows()
