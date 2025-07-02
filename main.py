# File: main.py

import argparse
from src.train import train_model
from src.predict import forecast_future
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n🔧 Starting training process...")
    model, scalers = train_model(config)

    print("\n📈 Forecasting future water levels (24h)...")
    forecast_future(config, model, scalers)


if __name__ == "__main__":
    main()

# Cài đặt thư viện cần thiết:
## pip install pandas numpy scikit-learn tensorflow pyyaml openpyxl
## pip install matplotlib

# Chạy pipeline bằng 1 dòng lệnh:
## python main.py --config config/camle_mtslstm.yaml

