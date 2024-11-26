import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from torchvision import models
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import os

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.training_completed = False

    def is_trained(self):
        return self.training_completed


    def fit(self, X, y, device):

        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass

    def ConfusionMatrix(self, predictions,labels):
        self.labels = labels.cpu().numpy()
        self.predictions = predictions.cpu().numpy()
        conf_matrix = confusion_matrix(self.labels, self.predictions)
        label = sorted(set(self.labels))
        return conf_matrix
    def predict_confidence(self, X):
        pass

class TabularClassifier(Model):
    def __init__(self, model_name):
        super(TabularClassifier, self).__init__(model_name)
        self.model = model_name
        self.labels = None
        self.predictions = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        # Return probability estimates for the test data
        return self.model.predict_proba(X)

    def predict(self, X):
        # Return predicted labels for the test data
        return self.model.predict(X)

    def ConfusionMatrix(self, predictions, labels):
        self.labels = labels
        self.predictions = predictions
        conf_matrix = confusion_matrix(self.labels, self.predictions)
        label = sorted(set(self.labels))
        return conf_matrix

    def predict_confidence(self, X):
        # Example: Return predicted labels along with confidence scores
        confidence_scores = self.model.predict_proba(X)
        predicted_labels = self.model.predict(X)
        return predicted_labels, confidence_scores


class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes, max_epochs= 10, learning_rate=1e-3, pretrained = True,checkpoint_path=None, reset_weights = False):
        super(ImageClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.max_epochs = max_epochs
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.reset_weights = reset_weights
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        # self.trainer =  pl.Trainer(max_epochs=self.max_epochs,callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=3,verbose=True)])

        # Load the model dynamically
        self.model = self.load_model(self.model_name, self.num_classes, self.pretrained)
        self.learning_rate = learning_rate

    def load_model(self, model_name, num_classes, pretrained):
        # model_class = getattr(models, model_name)
        model_class = getattr(models, model_name.lower())  # Always use lowercase function name
        weights_enum = getattr(models, f"{model_name}_Weights", None)
        weights = weights_enum.DEFAULT if pretrained and weights_enum else None

        # Load model with or without weights

        if weights is not None:
            print(f"Pretrained weights for {model_name} loaded successfully")
            model = model_class(weights=weights)
        else:
            print(f"No pretrained weights available for {model_name}. Initializing from scratch.")
            model = model_class()



        # Modify the classifier or final layer
        if hasattr(model, 'classifier'):  # For AlexNet, VGG, etc.
            if isinstance(model.classifier, nn.Sequential):
                num_ftrs = model.classifier[-1].in_features
                model.classifier = nn.Sequential(
                    *list(model.classifier.children())[:-1],  # Keep all but the last layer
                    nn.Dropout(p=0.2),                       # Add Dropout
                    nn.Linear(num_ftrs, num_classes)         # Replace the last layer
                )
            elif isinstance(model.classifier, nn.Linear):
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),                       # Add Dropout before the new layer
                    nn.Linear(num_ftrs, num_classes)
                )
            else:
                raise ValueError(f"Unsupported classifier type: {type(model.classifier)}")
        elif hasattr(model, 'fc'):  # For ResNet, Inception, etc.
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.2),                           # Add Dropout
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"Model {model_name} does not have a recognized classifier layer")


        # Reset weights if specified
        if self.reset_weights:
            def reset_layer_weights(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    m.reset_parameters()
            model.apply(reset_layer_weights)

        # Load checkpoint if provided
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])

        return model



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Extract logits from InceptionOutputs
        if isinstance(y_hat, tuple):  # For models that return tuples
            logits = y_hat[0]
        elif hasattr(y_hat, "logits"):  # For InceptionOutputs
            logits = y_hat.logits
        else:
            logits = y_hat  # Assume it's already a tensor

        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = self.alpha * self.criterion(y_hat, y)
        loss = self.criterion(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def get_logits_from_model_output(self,outputs):
        """
        Extracts logits from model output, handling different output types.
        """
        if isinstance(outputs, tuple):
            # Some models might return a tuple (e.g., ResNet)
            return outputs[0]  # Usually the first element is the logits

        elif hasattr(outputs, 'logits'):
            # For models like Inception V3
            return outputs.logits

        elif isinstance(outputs, torch.Tensor):
            # If the output is directly a tensor
            return outputs

        else:
            raise ValueError("Unsupported model output type: {}".format(type(outputs)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]

    # Predict step to return predictions and labels
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch  # Extract inputs and labels
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self(inputs)
        return outputs, labels  # Return both predictions and true labels

    # New fit method
    def fit(self, train_dataloader, val_dataloader=None):
        """
        Trains the model using the provided dataloaders and PyTorch Lightning's Trainer.

        Args:
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data (optional).
            max_epochs: Number of epochs to train (default: 10).
        """
        # Directory and Model Path
        trained_model_dir = "/home/fahadk/Project/SPROUT/debug/trained_models"
        trained_model_path = os.path.join(trained_model_dir, f"{self.model_name}.pt")
        os.makedirs("/home/fahadk/Project/SPROUT/debug/trained_models", exist_ok=True)  # Create directory if it doesn't exist

        files = os.listdir(trained_model_dir)
        files = [os.path.splitext(f)[0] for f in files if os.path.isfile(os.path.join(trained_model_dir, f))]

        # Check if Trained Model Exists
        if os.path.exists(trained_model_path) and self.model_name in files:
            print(f"Found pre-trained model at: {trained_model_path}")

            # Load the Pre-trained Model
            self.load_state_dict(torch.load(trained_model_path))
            self.eval()  # Set the model to evaluation mode
            print("Model loaded and ready for predictions.")
            return

        # If No Pre-trained Model, Proceed with Training
        print("No pre-trained model found. Training the model...")
        # Callbacks for Early Stopping and Model Checkpointing
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3,min_delta = 0.001, verbose=True)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="./checkpoints",
            filename=self.model_name+"-{epoch:02d}-{val_loss:.2f}-{train_acc:.2f}-{val_acc:.2f}-" + str(self.learning_rate)
        )

        # Create the PyTorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=self.max_epochs,
                             callbacks=[early_stopping, checkpoint_callback],
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1
                             )
        # trainer = pl.Trainer(max_epochs=self.max_epochs)
        # Fit the model using the trainer and dataloaders
        if val_dataloader is not None:
            trainer.fit(self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self, train_dataloaders=train_dataloader)


        # Save the Best Model Path
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")

        # Save the Trained Model Explicitly
        torch.save(self.state_dict(), trained_model_path)
        print(f"Trained model saved at: {trained_model_path}")

        torch.cuda.empty_cache()

    def predict(self, dataloader):
        """Makes predictions on the provided dataloader."""
        trainer = pl.Trainer()
        predictions = trainer.predict(self, dataloaders=dataloader)

        # Initialize a list to collect all predictions
        all_predictions = []

        for batch_predictions in predictions:
            # If the predictions are a tuple, get the first element
            if isinstance(batch_predictions, tuple):
                batch_predictions = batch_predictions[0]  # or another index depending on your model output

            # Ensure predictions are 2D
            if batch_predictions.dim() == 1:
                batch_predictions = batch_predictions.unsqueeze(1)

            all_predictions.append(batch_predictions)

        # Convert to a single tensor
        all_predictions = torch.cat(all_predictions, dim=0)

        # Get class labels by taking argmax
        class_labels = all_predictions.argmax(dim=1)

        return class_labels

    def predict_proba(self, dataloader):
        """Makes probability predictions on the provided dataloader."""
        # Initialize the PyTorch Lightning Trainer
        trainer = pl.Trainer()
        predictions = trainer.predict(self, dataloaders=dataloader)

        # Initialize a list to collect all predictions
        all_predictions = []

        for batch_predictions in predictions:
            # If the predictions are a tuple, get the first element
            if isinstance(batch_predictions, tuple):
                batch_predictions = batch_predictions[0]  # or another index depending on your model output
            # Ensure predictions are 2D
            if batch_predictions.dim() == 1:
                batch_predictions = batch_predictions.unsqueeze(1)

            all_predictions.append(batch_predictions)

        # Concatenate all batch predictions into a single tensor
        all_predictions = torch.cat(all_predictions, dim=0)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(all_predictions, dim=1)

        # Convert the tensor to a NumPy array
        return probabilities.cpu().detach().numpy()

class ConvAutoEncoder(pl.LightningModule):
    def __init__(self,max_epochs=1):
        super(ConvAutoEncoder, self).__init__()
        self.max_epochs = max_epochs
        self.channel = 3
        self.individual_losses = []  # List to store training losses

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channel, 16, kernel_size=3, stride=2, padding=1),  # 3x299x299 -> 16x150x150
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16x150x150 -> 32x75x75
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x75x75 -> 64x38x38
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x38x38 -> 128x19x19
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128x19x19 -> 256x10x10
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 256x10x10 -> 512x5x5
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 512x5x5 -> 256x10x10
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x10x10 -> 128x19x19
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x19x19 -> 64x38x38
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x38x38 -> 32x75x75
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x75x75 -> 16x150x150
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.channel, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x150x150 -> 3x299x299
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Forward pass
        x_hat = self(x)

        # Calculate loss for each input in the batch (no reduction)
        losses = nn.functional.mse_loss(x_hat, x, reduction='none')

        # Sum the losses across dimensions (if multi-dimensional input/output)
        individual_losses = losses.view(losses.size(0), -1).mean(dim=1)

        # Store or process individual losses as needed
        self.individual_losses.extend(individual_losses.detach().cpu().numpy())

        # Log average loss for the batch
        average_loss = individual_losses.mean()
        self.log('train_loss', average_loss)

        return average_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def return_training_loss(self):
        """Return the training losses as a NumPy array."""
        return np.array(self.individual_losses)

    def fit(self, train_dataloader, val_dataloader=None):
        """
        Trains the model using the provided dataloaders and PyTorch Lightning's Trainer.

        Args:
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data (optional).
            max_epochs: Number of epochs to train (default: 10).
        """
        self.individual_losses = []
        # Create the PyTorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=self.max_epochs,callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=3)])

        # Fit the model using the trainer and dataloaders
        if val_dataloader is not None:
            trainer.fit(self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self, train_dataloaders=train_dataloader)
        self.max_epochs = 1
        torch.cuda.empty_cache()
