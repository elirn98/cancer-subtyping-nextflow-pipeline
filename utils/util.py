import matplotlib.pyplot as plt

def plot_loss(train_loss, eval_loss):
    # Plotting the loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.plot(range(len(eval_loss)), eval_loss, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Evaluation Loss vs Epoch')
    plt.savefig('../plots/loss_plot.png')