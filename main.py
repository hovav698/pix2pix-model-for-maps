import torch
import numpy as np
import matplotlib.pyplot as plt
import params
from models import generator, discriminator
import losses
from utils import get_data,load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# the training loop
def train():
    gen_model.train()
    disc_model.train()

    gen_losses = []
    disc_losses = []

    for epoch in range(params.epochs):
        epoch_losses = []
        batch_count = 0

        for inputs, targets in train_dl:

            gen_model.zero_grad()
            disc_model.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)

            gen_output = gen_model(inputs)
            # print(gen_output.shape)
            # print(target.shape)

            disc_real_output = disc_model(inputs, targets)
            disc_generated_output = disc_model(inputs, gen_output)
            disc_loss = losses.discriminator_loss(disc_real_output, disc_generated_output)
            gen_total_loss, gen_gan_loss, gen_l1_loss = losses.generator_loss(disc_generated_output, gen_output,
                                                                              targets)

            disc_loss.backward(retain_graph=True)
            gen_total_loss.backward()

            disc_optimizer.step()
            gen_optimizer.step()

            gen_losses.append(gen_total_loss.item())
            disc_losses.append(disc_loss.item())

            #plot the generated image every 200 batches
            if batch_count % 200 == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                title = 'Generated image and target for epoch ' + str(epoch) + ' batch ' + str(batch_count)
                fig.suptitle(title)
                ax1.imshow(np.transpose(inputs[0].cpu().detach().numpy(), (1, 2, 0)))
                ax2.imshow(np.transpose(gen_output[0].cpu().detach().numpy(), (1, 2, 0)))

                plt.show()

                print("epoch:", epoch, "batch:", batch_count, "gen loss:", gen_total_loss.item())
                print("epoch:", epoch, "batch:", batch_count, "disc loss:", disc_loss.item())

            batch_count += 1

    return gen_losses, disc_losses


#validate on the test data
def validate():
    gen_model.eval()
    disc_model.eval()

    count = 0
    for inputs, targets in test_dl:

        inputs = inputs.to(device)
        targets = targets.to(device)

        gen_output = gen_model(inputs)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.transpose(inputs[0].cpu().detach().numpy(), (1, 2, 0)))
        ax[0].set_title('Input Image')
        ax[1].imshow(np.transpose(targets[0].cpu().detach().numpy(), (1, 2, 0)))
        ax[1].set_title('Target Image')

        plt.show()

        plt.imshow(np.transpose(gen_output[0].cpu().detach().numpy(), (1, 2, 0)))
        plt.title('Predicted Image')

        plt.show()

        count += 1
        if count == 10:
            break


if __name__ == '__main__':
    get_data()

    train_dl = load_data('maps/train')
    test_dl = load_data('maps/val')

    gen_model = generator.Generator().to(device)
    disc_model = discriminator.Discriminator().to(device)

    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=params.lr)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=params.lr)

    gen_losses, disc_losses = train()

    validate()
