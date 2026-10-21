import os

dataset_path = "Freshness Fish Dataset"

print("Checking fish dataset folders...\n")

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        images = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ]

        print(f"{folder}: {len(images)} images")

print("\nDataset folder check completed.")