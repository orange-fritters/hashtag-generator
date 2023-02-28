from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from model_lightning import HarrisonModel


def main():
    logger = TensorBoardLogger('tb_logs', name='harrison')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='harrison-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = Trainer(
        accelerator='cpu',
        max_epochs=100,
        logger=logger,
        callbacks=[lr_monitor, early_stopping, checkpoint_callback],
    )

    model = HarrisonModel(
        data_path='data.csv',
        prefix_length=5,
        dim_clip=512,
        dim_embedding=512,
        num_layers=4,
        dim_feedforward=512,
        nhead=4,
        dropout=0.1,
        lr=1e-4,
        batch_size=16,
        num_workers=4,
        normalize_prefix=False,
    )
    trainer.fit(model)
