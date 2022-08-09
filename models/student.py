import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification


def get_sorted_indices(selection_strategy, logits, labels, selection_ratio):
    bsz = logits.size(0)
    device = logits.device
    if selection_strategy == 'none':
        indices = torch.arange(logits.size(0), device=device)
    elif selection_strategy == 'entropy':
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        _, indices = torch.sort(entropy, descending=True)
    elif selection_strategy == 'entropy-r':
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)  # select most certain data
        _, indices = torch.sort(entropy, descending=False)
    elif selection_strategy == 'kl' or selection_strategy == 'kl-fix':  # kl between logits & labels
        probs = F.log_softmax(logits, dim=-1)
        kl_distance = F.kl_div(probs, labels, reduction='none').sum(dim=-1)  # bsz,
        _, indices = torch.sort(kl_distance, descending=True)
    elif selection_strategy == 'kl-b':  # balanced difficulty & easy
        probs = F.log_softmax(logits, dim=-1)
        kl_distance = F.kl_div(probs, labels, reduction='none').sum(dim=-1)  # bsz,
        _, indices_d2e = torch.sort(kl_distance, descending=True)  # difficult to easy indices
        _, indices_e2d = torch.sort(kl_distance, descending=False)  # easy to difficult indices
        interleave = torch.stack((indices_d2e, indices_e2d), dim=1)
        indices = interleave.view(-1, 1).squeeze()[:bsz]  # one difficult one easy ...
    elif selection_strategy == 'kl-r':  # kl between logits & labels
        probs = F.log_softmax(logits, dim=-1)
        kl_distance = F.kl_div(probs, labels, reduction='none').sum(dim=-1)  # bsz,
        _, indices = torch.sort(kl_distance, descending=False)  # select most easy data?
    elif selection_strategy == "random":  # use random selected mixup samples for training
        indices = torch.randperm(bsz, device=device)
    elif selection_strategy == "confidence":
        s_probs = F.softmax(logits, dim=-1)
        s_conf, _ = torch.max(s_probs, dim=-1)
        _, indices = torch.sort(s_conf, descending=False)  # lower confidence indicate challenging input
    elif selection_strategy == "confidence-r":
        s_probs = F.softmax(logits, dim=-1)
        s_conf, _ = torch.max(s_probs, dim=-1)
        _, indices = torch.sort(s_conf, descending=True)  # lower confidence indicate challenging input
    elif selection_strategy == "margin":
        s_probs = F.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(s_probs, dim=-1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # top-1 prob - top-2 prob
        _, indices = torch.sort(margin, descending=False)  # lower margin indicate more uncertain examples
    else:
        raise ValueError("Unsupported uncertainty strategy")
    if selection_strategy != 'none':
        indices = indices[: int(bsz * selection_ratio)]
    # print(indices.size())
    return indices


class DynamicTeacherKDForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config,
                 kd_alpha=1.0, ce_alpha=1.0,
                 teacher_large=None,
                 teacher_small=None,
                 temperature=5.0,
                 small_teacher_alpha=0.0,
                 large_teacher_alpha=0.0,
                 kl_kd=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher_large = teacher_large
        self.teacher_small = teacher_small
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.small_teacher_alpha = small_teacher_alpha
        self.large_teacher_alpha = large_teacher_alpha

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):

        large_logits, small_logits = None, None

        if self.training:
            assert self.teacher_small is not None and self.teacher_large is not None, "student hold a None teacher reference"
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)
            with torch.no_grad():
                if self.small_teacher_alpha > 0:
                    small_teacher_outputs = self.teacher_small(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    small_logits = small_teacher_outputs[0]
                else:
                    small_logits = None
                if self.large_teacher_alpha > 0:
                    large_teacher_outputs = self.teacher_large(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    large_logits = large_teacher_outputs[0]
                else:
                    large_logits = None

        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))

            if small_logits is not None:
                if not self.kl_kd:
                    loss += self.kd_alpha * self.small_teacher_alpha * self.mse_loss(student_logits.view(-1),
                                                                                     small_logits.view(-1))
                else:
                    loss += self.kd_alpha * self.small_teacher_alpha * \
                            self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                         F.softmax(small_logits / self.temperature, dim=-1)) * self.temperature ** 2

            if large_logits is not None:
                if not self.kl_kd:
                    loss += self.kd_alpha * self.large_teacher_alpha * self.mse_loss(student_logits.view(-1),
                                                                                     large_logits.view(-1))
                else:
                    loss += self.kd_alpha * self.large_teacher_alpha * \
                            self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                         F.softmax(large_logits / self.temperature , dim=-1)) * self.temperature ** 2

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output


class UncertaintyTeacherKDForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config,
                 kd_alpha=1.0, ce_alpha=1.0,
                 teacher_large=None,
                 teacher_small=None,
                 temperature=5.0,
                 small_teacher_alpha=0.0,
                 large_teacher_alpha=0.0,
                 kl_kd=False,
                 uncertainty_mode='hard'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher_large = teacher_large
        self.teacher_small = teacher_small
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.small_teacher_alpha = small_teacher_alpha
        self.large_teacher_alpha = large_teacher_alpha
        self.uncertainty_mode = uncertainty_mode

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):

        large_logits, small_logits = None, None

        if self.training:
            assert self.teacher_small is not None and self.teacher_large is not None, "student hold a None teacher reference"
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            with torch.no_grad():
                small_teacher_outputs = self.teacher_small(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                small_logits = small_teacher_outputs[0]

                large_teacher_outputs = self.teacher_large(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                large_logits = large_teacher_outputs[0]

            if self.uncertainty_mode == 'hard':
                # student uncertainty
                probs = F.softmax(student_logits, dim=-1)
                entropy = - torch.sum(probs * torch.log(probs), dim=1)
                _, indices = torch.sort(entropy, descending=True)
                bsz = student_logits.size(0)
                large_kd_indices = indices[bsz // 2:]
                small_kd_indices = indices[: bsz // 2]
                if self.kl_kd:
                    large_kd_loss = self.kl_loss(
                        F.log_softmax(student_logits[large_kd_indices] / self.temperature, dim=1),
                        F.softmax(large_logits[large_kd_indices] / self.temperature , dim=-1)) * self.temperature ** 2
                    small_kd_loss = self.kl_loss(
                        F.log_softmax(student_logits[small_kd_indices] / self.temperature, dim=1),
                        F.softmax(small_logits[small_kd_indices] / self.temperature , dim=-1)) * self.temperature ** 2
                else:
                    large_kd_loss = self.mse_loss(student_logits[large_kd_indices], large_logits[large_kd_indices])
                    small_kd_loss = self.mse_loss(student_logits[small_kd_indices], small_logits[small_kd_indices])
            elif self.uncertainty_mode == 'soft':
                if self.kl_kd:
                    large_kd_loss = self.kl_loss(
                        F.log_softmax(student_logits / self.temperature, dim=1),
                        F.softmax(large_logits / self.temperature, dim=-1)) * self.temperature ** 2
                    small_kd_loss = self.kl_loss(
                        F.log_softmax(student_logits / self.temperature, dim=1),
                        F.softmax(small_logits / self.temperature, dim=-1)) * self.temperature ** 2
                else:
                    large_kd_loss = self.mse_loss(student_logits, large_logits)
                    small_kd_loss = self.mse_loss(student_logits, small_logits)
                # student uncertainty
                probs = F.softmax(student_logits, dim=-1)
                entropy = - torch.sum(probs * torch.log(probs), dim=1)
                mean_entropy = torch.mean(entropy)
                avg_prob = 1 / self.num_labels * torch.ones((1, self.num_labels))
                # normalize the entropy to a confidence score ranges from 0 to 1
                confidence = - mean_entropy / torch.sum(avg_prob * torch.log(avg_prob))
                # 0 indicate the current student is very confident ( entropy -> 0 )
                # 1 indicate the current student is very confusing ( entropy -> max)
                self.small_teacher_alpha = confidence  #
                self.large_teacher_alpha = 1 - confidence  #
            elif self.uncertainty_mode == 'soft-instance':  # instance-level adjustment
                if self.kl_kd:
                    large_kd_loss = self.temperature ** 2 * F.kl_div(
                        F.log_softmax(student_logits / self.temperature, dim=1),
                        F.softmax(large_logits / self.temperature, dim=-1), reduction='none').sum(-1)  # bsz
                    small_kd_loss = self.temperature ** 2 * F.kl_div(
                        F.log_softmax(student_logits / self.temperature, dim=1),
                        F.softmax(small_logits / self.temperature, dim=-1), reduction='none').sum(-1)  # bsz
                else:
                    large_kd_loss = F.mse_loss(student_logits, large_logits, reduction='none').mean(dim=-1)  # bsz
                    small_kd_loss = F.mse_loss(student_logits, small_logits, reduction='none').mean(dim=-1)  # bsz
                # student uncertainty
                probs = F.softmax(student_logits, dim=-1)  # bsz, num_labels
                entropy = - torch.sum(probs * torch.log(probs), dim=1)  # bsz
                avg_prob = 1 / self.num_labels * torch.ones((1, self.num_labels))  # (1)
                # normalize the entropy to a confidence score ranges from 0 to 1
                confidence = - entropy / torch.sum(avg_prob * torch.log(avg_prob))  # bsz
                # 0 indicate the current student is very confident ( entropy -> 0 )
                # 1 indicate the current student is very confusing ( entropy -> max)
                self.small_teacher_alpha = confidence  # bsz
                self.large_teacher_alpha = 1 - confidence  # bsz
            else:
                raise ValueError("Unsupported mode")

        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))

            if small_logits is not None:
                if self.uncertainty_mode == 'soft-instance':
                    loss += self.kd_alpha * torch.mean(self.small_teacher_alpha * small_kd_loss)
                else:
                    loss += self.kd_alpha * self.small_teacher_alpha * small_kd_loss

            if large_logits is not None:
                if self.uncertainty_mode == 'soft-instance':
                    loss += self.kd_alpha * torch.mean(self.large_teacher_alpha * large_kd_loss)
                else:
                    loss += self.kd_alpha * self.large_teacher_alpha * large_kd_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output


class DynamicDataKDForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config,
                 kd_alpha=1.0,
                 ce_alpha=1.0,
                 teacher=None,
                 temperature=5.0,
                 kl_kd=False,
                 selection_strategy='none',
                 selection_ratio=1.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher = teacher
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.selection_strategy = selection_strategy
        self.selection_ratio = selection_ratio

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):

        kd_loss = None

        if self.training:
            assert self.teacher is not None, "student hold a None teacher reference"
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            indices = get_sorted_indices(self.selection_strategy, student_logits, labels, self.selection_ratio)

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids[indices],
                    attention_mask=attention_mask[indices] if attention_mask is not None else None,
                    token_type_ids=token_type_ids[indices] if token_type_ids is not None else None,
                    position_ids=position_ids[indices] if position_ids is not None else None,
                    head_mask=head_mask[indices] if head_mask is not None else None,
                    inputs_embeds=inputs_embeds[indices] if inputs_embeds is not None else None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                teacher_logits = teacher_outputs[0]

            student_logits_for_kd = student_logits[indices]
            if self.kl_kd:
                kd_loss = self.kl_loss(F.log_softmax(student_logits_for_kd / self.temperature, dim=1),
                                       F.softmax(teacher_logits / self.temperature, dim=1)) * self.temperature ** 2
            else:
                kd_loss = self.mse_loss(student_logits_for_kd, teacher_logits)

        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))

            if kd_loss is not None:
                loss += self.kd_alpha * kd_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output


class DynamicObjectiveKDForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config,
                 kd_alpha=1.0, ce_alpha=1.0,
                 teacher=None,
                 temperature=5.0,
                 strategy='none',
                 kl_kd=False,
                 kd_kl_alpha=1.0,
                 kd_rep_alpha=1.0,
                 ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher = teacher
        self.kd_kl_alpha = kd_kl_alpha
        self.kd_rep_alpha = kd_rep_alpha
        self.ce_alpha = ce_alpha
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.teacher_time_alpha = 1.0
        self.strategy = strategy
        self.ds_weight = 1.0
        self.pt_weight = 1.0

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):

        teacher_logits = None

        if self.training:
            assert self.teacher is not None, "student hold a None teacher reference"
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
                teacher_logits = teacher_outputs[0]

            # compute hidden alignment objective if alpha_hidden > 0:
            teacher_reps, student_reps = teacher_outputs["hidden_states"], student_outputs["hidden_states"]

            # 2 for 12-L teacher and 6L-student, check bert-pkd for implementation details
            layers_per_block = self.teacher.config.num_hidden_layers // self.config.num_hidden_layers
            new_teacher_reps = [teacher_reps[i * layers_per_block]
                                for i in range(self.config.num_hidden_layers + 1)]  # 0, 2, 4, 6, 8, 10, 12
            new_student_reps = student_reps
            rep_loss = 0.0
            weight = None
            if self.kd_rep_alpha > 0 and self.strategy == 'none':
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=1), F.normalize(teacher_rep, p=2, dim=1))
                    rep_loss += tmp_loss
            elif self.kd_rep_alpha > 0 and "uncertainty" in self.strategy:
                new_student_reps = torch.stack(new_student_reps, dim=1)  # bsz, num_layer, seq_len,  hidden_size
                new_teacher_reps = torch.stack(new_teacher_reps, dim=1)  # bsz, num_layer, seq_len,hidden_size
                # (bsz )
                rep_loss = F.mse_loss(F.normalize(new_student_reps, p=2, dim=-1),
                                      F.normalize(new_teacher_reps, p=2, dim=-1), reduction='none').mean(dim=-1).mean(
                    dim=-1).sum(dim=1)
                probs = F.softmax(student_logits, dim=-1)
                entropy = torch.sum(probs * torch.log(probs), dim=1)  # bsz
                avg_prob = 1 / self.num_labels * torch.ones((1, self.num_labels))
                # normalize the entropy to  0 to 1
                weight = entropy / torch.sum(avg_prob * torch.log(avg_prob))  # bsz
                #print(weight) 
                # print(torch.mean(weight))
        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = self.ce_alpha * loss_fct(student_logits.view(-1, self.num_labels),
                                                labels.view(-1))
            if teacher_logits is not None:
                if weight is not None and self.strategy == 'uncertainty':
                    if not self.kl_kd:
                        kd_loss = F.mse_loss(student_logits,
                                             teacher_logits, reduction='none').mean(dim=-1)  # bsz
                    else:
                        # original reduction is batchmean,  sum over the logit dim, and mean over the batch level
                        kd_loss = self.temperature ** 2 * \
                                  F.kl_div(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits / self.temperature, dim=-1), reduction='none').sum(dim=1)  # bsz
                    # weight -> 1, model is most uncertain, thus need attention to the hidden loss
                    loss += self.kd_kl_alpha * torch.mean((1 - weight) * kd_loss,
                                                          dim=0) + self.kd_rep_alpha * torch.mean(weight * rep_loss,
                                                                                                  dim=0)
                    self.ds_weight = torch.mean(1 - weight).item()
                    self.pt_weight = torch.mean(weight).item()

                elif weight is not None and self.strategy == 'uncertainty-r':
                    if not self.kl_kd:
                        kd_loss = F.mse_loss(student_logits,
                                             teacher_logits, reduction='none').mean(dim=-1)  # bsz
                    else:
                        kd_loss = self.temperature ** 2 * \
                                  F.kl_div(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits / self.temperature, dim=-1), reduction='none').sum(dim=1)  # bsz
                    # weight -> 1, model is most uncertain, thus need attention to the hidden loss
                    loss += self.kd_kl_alpha * torch.mean(weight * kd_loss, dim=0) + self.kd_rep_alpha * torch.mean(
                        (1 - weight) * rep_loss, dim=0)

                else:
                    if not self.kl_kd:
                        kd_loss = self.mse_loss(student_logits.view(-1),
                                                teacher_logits.view(-1))
                    else:
                        kd_loss = self.kl_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                               F.softmax(teacher_logits / self.temperature, dim=-1)) * self.temperature ** 2
                    loss += self.kd_kl_alpha * kd_loss
                    loss += self.kd_rep_alpha * rep_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
