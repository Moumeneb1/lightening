import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier


download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')
datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=4,
)

model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")

trainer = flash.Trainer(max_epochs=1,strategy="deepspeed")

trainer.finetune(model, datamodule=datamodule, strategy="freeze")