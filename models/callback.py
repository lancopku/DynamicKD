from transformers.integrations import TensorBoardCallback


class SupervisionWeightWriterCallback(TensorBoardCallback):
    def __init__(self, model):
        super(SupervisionWeightWriterCallback, self).__init__()
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        self.tb_writer.add_scalar('kl-weight', self.model.ds_weight, state.global_step)
        self.tb_writer.add_scalar('pt-weight', self.model.pt_weight, state.global_step)
