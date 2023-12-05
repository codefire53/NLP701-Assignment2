from transformers import Trainer

class CRFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        emissions = outputs[0]
        mask = inputs["attention_mask"]

        # Calculate CRF loss
        crf_loss = -model.crf(emissions, labels, mask=mask)

        return crf_loss

    def training_step(self, model, inputs):
        # Override the training step to return additional information
        loss = self.compute_loss(model, inputs)
        print("hello")
        return {"loss": loss, "inputs": inputs}