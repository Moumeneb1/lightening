import torch

import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData
from flash.core.data.utils import download_data
from pytorch_lightning import Trainer, seed_everything
from flash.audio.speech_recognition.output_transform import SpeechRecognitionOutputTransform

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/timit_data.zip", "./data")

datamodule = SpeechRecognitionData.from_json(
    "file",
    "text",
    train_file="data/timit/train.json",
    test_file="data/timit/test.json",
    batch_size=16,
)

# 2. Build the task
model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")


# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=100, gpus=2,strategy='ddp_sharded')
# trainer.fit(model, datamodule=datamodule)
trainer.fit(model, datamodule=datamodule)

# datamodule = SpeechRecognitionData.from_files(predict_files=["data/timit/example.wav","data/timit/example.wav"], batch_size=4)
# predictions = trainer.predict(model, datamodule=datamodule)

# predictor  = SpeechRecognitionOutputTransform(backbone="facebook/wav2vec2-base-960h",)
# print(predictor(predictions[0]))


# # # 4. Predict on audio files!
# )
# predictions = trainer.predict(model, datamodule=datamodule)
# print(predictions)

# # # 5. Save the model!
# trainer.save_checkpoint("speech_recognition_model.pt")