import math
import tensorflow as tf

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20,
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def _to_float_dict(score: list) -> dict:
        return {"loss": float(score[0]), "accuracy": float(score[1])}

    def evaluate(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        self._valid_generator()

        validation_steps = max(
            1, math.ceil(self.valid_generator.samples / self.valid_generator.batch_size)
        )
        self.score = self.model.evaluate(self.valid_generator, steps=validation_steps)
        return self.score

    def save_score(self):
        scores = self._to_float_dict(self.score)
        save_json(path=self.config.score_file_path, data=scores)
