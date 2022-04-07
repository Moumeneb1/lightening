import torch

import flash
from flash.audio import AudioClassificationData
from flash.core.data.utils import download_data
from flash.image import ImageClassifier

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/urban8k_images.zip", "./data")

datamodule = AudioClassificationData.from_folders(
    train_folder="data/urban8k_images/train",
    val_folder="data/urban8k_images/val",
    transform_kwargs=dict(spectrogram_size=(64, 64)),
    batch_size=4,
)

# 2. Build the model.
model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule)

# 4. Predict what's on few images! air_conditioner, children_playing, siren etc.
datamodule = AudioClassificationData.from_files(
    predict_files=[
        "data/urban8k_images/test/air_conditioner/13230-0-0-5.wav.jpg",
        "data/urban8k_images/test/children_playing/9223-2-0-15.wav.jpg",
        "data/urban8k_images/test/jackhammer/22883-7-10-0.wav.jpg",
    ],
    batch_size=3,
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("audio_classification_model.pt")