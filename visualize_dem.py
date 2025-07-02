# Install lib: pip install rasterio matplotlib

import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Đọc dữ liệu DEM
dem_path = "data/CamLe_DN_n15_e108_1arc_v3.tif"
with rasterio.open(dem_path) as src:
    dem = src.read(1)
    extent = rasterio.plot.plotting_extent(src)

# Đọc kết quả dự đoán mực nước
pred_df = pd.read_csv("outputs/predicted_water_level.csv")
latest_level = round(pred_df['Muc_nuoc_du_doan'].iloc[0], 2)  # giá trị đầu tiên

# Xác định toạ độ trung tâm bản đồ DEM
center_x = (extent[0] + extent[1]) / 2
center_y = (extent[2] + extent[3]) / 2

# Vẽ bản đồ
plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap='terrain', extent=extent)
plt.colorbar(label='Độ cao (m)')
plt.title("🗺️ DEM + Dự đoán mực nước tại trạm Cẩm Lệ")

# Overlay điểm đo mực nước
plt.scatter(center_x, center_y, color='blue', s=100, marker='o', label='Trạm đo Cẩm Lệ')
plt.text(center_x + 0.01, center_y + 0.01, f"Mực nước: {latest_level} m", fontsize=12, color='blue')

plt.xlabel("Kinh độ")
plt.ylabel("Vĩ độ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
