import torch
from sklearn.metrics import *
import os
import numpy as np
import matplotlib.pyplot as plt


def train(epochs, model, name, criterion, optimizer, train_dataloader, test_dataloader, datasets):

    # Create model directory
    if not os.path.exists(name):
        os.makedirs(name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    losses = {'train': [], 'test': []}
    accuracy_register = {'train': [], 'test': []}

    best_metrics_test = {"accuracy": 0,
                         "recall": 0, "precision": 0, "f1_score": 0}
    early_stopping_counter = 0
    for epoch in range(epochs):
        print('*'*200)
        for mode in ["train", "test"]:
            if mode == "train":
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = test_dataloader
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            epoch_outputs = np.array([])
            epoch_labels = np.array([])
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                epoch_outputs = np.concatenate(
                    (epoch_outputs, outputs.argmax(dim=1).cpu().detach().numpy()), axis=None)
                epoch_labels = np.concatenate(
                    (epoch_labels, labels.cpu().detach().numpy()), axis=None)

                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()*images.size(0)
                running_corrects += torch.sum(outputs.argmax(dim=1)
                                              == labels.data)

            # Calculate metrics
            f1 = f1_score(epoch_outputs, epoch_labels, average='macro')
            recall = recall_score(epoch_outputs, epoch_labels, average='macro')
            acc = accuracy_score(epoch_outputs, epoch_labels)
            prec = precision_score(
                epoch_outputs, epoch_labels, average='macro')

            # Save best model for test set
            if mode == "test":
                if acc > best_metrics_test["accuracy"]:
                    early_stopping_counter = 0
                    best_metrics_test["accuracy"] = acc
                    best_metrics_test["recall"] = recall
                    best_metrics_test["precision"] = prec
                    best_metrics_test["f1_score"] = f1
                    torch.save(model.state_dict(), os.path.join(
                        name, "best_model_test.pth"))

                    # Save classification report for test set
                    with open(os.path.join(name, "classification_report_test.txt"), "w") as f:
                        f.write(classification_report(
                            epoch_outputs, epoch_labels))

                    # Display confusion matrix for test set
                    conf = confusion_matrix(epoch_labels, epoch_outputs)
                    disp = ConfusionMatrixDisplay(
                        conf, display_labels=np.arange(0, 10))
                    # Save confusion matrix for test set
                    disp.plot()
                    plt.savefig(os.path.join(
                        name, "confusion_matrix_test.png"))
                    plt.clf()
                else:
                    early_stopping_counter += 1
                    print(">>No improvement detecte during the last " +
                          str(early_stopping_counter)+" epochs")

            # Saving accuracy and loss
            accuracy_register[mode].append(acc)
            losses[mode].append(running_loss/len(datasets[mode]))

            # Print metrics
            print("Epoch: {}/{} - Mode: {} - Loss: {:.4f} - Accuracy: {:.4f} - F1 Score: {:.4f} - Recall: {:.4f} - Precision: {:.4f}".format(epoch+1, epochs,
                  mode, running_loss/len(datasets[mode]), running_corrects.double()/len(datasets[mode]), f1, recall, prec))

        if early_stopping_counter == 5:
            break

    # Generating the plots
    plt.title("Loss Curves")
    plt.plot(losses['train'], label='Train')
    plt.plot(losses['test'], label='Test')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(name, "loss_curves.png"))

    plt.clf()

    plt.title("Accuracy Curves")
    plt.plot(accuracy_register['train'], label='Train')
    plt.plot(accuracy_register['test'], label='Test')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(name, "accuracy_curves.png"))

    # Write the metrics in a file
    with open(os.path.join(name, "metrics_test.txt"), "w") as f:
        f.write("Best metrics: \n")
        f.write("Accuracy: {}\n".format(best_metrics_test["accuracy"]))
        f.write("Recall: {}\n".format(best_metrics_test["recall"]))
        f.write("Precision: {}\n".format(best_metrics_test["precision"]))
        f.write("F1 Score: {}\n".format(best_metrics_test["f1_score"]))
