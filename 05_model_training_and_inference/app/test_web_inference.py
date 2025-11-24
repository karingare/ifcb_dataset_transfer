import time
from web_inference import predict_all

test_image_path = "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/testing_data_png/2024/D20240208/D20240208T090401_IFCB134/D20240208T090401_IFCB134_00050.png"

t0 = time.time()

with open(test_image_path, "rb") as f:
    image_bytes = f.read()

t1 = time.time()
print(f"Loaded image bytes in {t1 - t0:.4f} seconds")

results = predict_all(image_bytes, topk=3)

t2 = time.time()
print(f"Prediction took {t2 - t1:.4f} seconds")
print(f"Total time: {t2 - t0:.4f} seconds\n")

for model_name, preds in results.items():
    print("Model:", model_name)
    for cls, prob in preds:
        print(f"  {cls:30s} {prob:.3f}")
