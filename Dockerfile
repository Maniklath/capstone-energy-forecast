FROM python:3.10-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash","-lc","python src/data/generate_synthetic.py --out data/sample/sample_load.csv --hours 168 && python src/train.py --input data/sample/sample_load.csv --epochs 1 && python src/infer.py --input data/sample/sample_load.csv --output out/predictions.csv"]
