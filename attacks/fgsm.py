import torch


class FGSM(object):
    def __init__(self, model, device, epsilon):
        super().__init__(model, device)
        self.epsilon = torch.FloatTensor(epsilon).to(self.device)

    # FGSM attack code
    def _fgsm_attack(self, image, data_grad, epsilon):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # debug
        # Return the perturbed image
        return perturbed_image

    def attack(self, data, labels):
        for epsilon in self.epsilon:
            # Clean grad
            if data.grad is not None:
                data.grad.zero_()
            # Forward pass the data through the model
            output = self.model(data)

            # Calculate the loss
            loss = F.nll_loss(output, labels)

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            data = self._fgsm_attack(data, data_grad, epsilon).data
            data.requires_grad = True

            # Re-classify the perturbed image
            output = self.model(data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == labels.item():
                self.failed_attack_list.append((data, labels.item(), final_pred.item()))
            else:
                self.success_attack_list.append((data, labels.item(), final_pred.item()))
        return data


class iFGSM(FGSM):