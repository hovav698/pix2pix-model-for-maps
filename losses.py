from torch import nn
import torch
from params import LAMBDA

loss_object = nn.BCELoss()


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
    # Mean absolute error
    l1_loss = torch.mean(torch.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))

    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
