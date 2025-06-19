import numpy as np 
from src.models.base import BaseModel
import torch.nn as nn
from src.models.utils import *
import torch.optim as optim
from typing import List 

from src.config import(
    training_mlp_parameters,
    MLP_BATCHES, 
)


def InitializeDenseLayers(layer_dims: List[int], dropout: float=0.0):
    """
    Initialize a list of fully connected (dense) layers for a neural network.
    Parameters
    ----------
    layer_dims : List[int]
        A list of integers specifying the number of neurons in each layer. For example, [64, 128, 32, 1]. 
    dropout : float, optional (default=0.0)
        Dropout probability to apply after each hidden layer. Set to 0.0 to disable.

    Returns
    -------
    layers : List[nn.Module]
        A list of PyTorch layers (to be used with `nn.Sequential`).
    """
    layers = []
    nb_layers = len(layer_dims)
    for i in range(nb_layers -1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if(i < nb_layers - 2):
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
          
    return layers
 

def get_mlp_dataloader(XS,YS): 
    """
    Create PyTorch DataLoaders for MLP training from given data and label sets.

    Parameters
    ----------
    XS : dict of pd.DataFrame
        Dictionary of input data for each phase (e.g., 'train', 'validation', 'test').

    YS : dict of pd.Series
        Dictionary of target labels corresponding to each phase.

    Returns
    -------
    dataloader : dict of DataLoader
        Dictionary containing a DataLoader for each phase.
    """
    
    dataloader = {}
    
    # create tensors :
    for set_name, d in XS.items(): 
        y = torch.tensor(YS[set_name].values).float()
        xs = d.astype('float64')  
        x = torch.tensor(xs.values).float()
        X_dataset = TensorDataset(x, y)
        dataloader[set_name] = DataLoader(X_dataset, batch_size=MLP_BATCHES[set_name], shuffle=True)
        
    return dataloader 
    
def get_MLP_criterio_weights(Y_Train):
    """
    Compute class imbalance weight for binary classification with MLP Model 

    The weight is calculated as the ratio of negative to positive samples
    and is typically used to set `pos_weight` in `BCEWithLogitsLoss`.

    Parameters
    ----------
    Y_train : pd.Series
        Target labels for the training set (binary classification: 0 and 1).

    Returns
    -------
    float
        Class weight to handle imbalance: (count of class 0) / (count of class 1)
    """
    counts = Y_Train.value_counts()
    return counts[0]/counts[1]


class MLP_Classifier(nn.Module):
    def __init__(self, layer_dims: List[int], dropout: float = 0.0):
        super(MLP_Classifier, self).__init__()
        classif_layers = InitializeDenseLayers(layer_dims, dropout)
        self.classifier = nn.Sequential(*classif_layers)
        
    def forward(self, x):
        """
        Defines the forward pass through the MLP.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor. 

        Returns
        -------
        torch.Tensor
            Output tensor, with predictions (logits) 
        """
        x = self.classifier(x)
        return x.squeeze(1)



class MLPModel(BaseModel):
    """PyTorch MLP model implementation that follows the BaseModel interface."""
        
    def __init__(self, logger, model_params=None):
        super().__init__(name="MLP",model_params=model_params,logger=logger)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = training_mlp_parameters["device"]
            
        if model_params is not None: 
            self.model_params = model_params
            self.initialize_model()

        else : 
            self.model = None 
            self.model_params = {}
            
    def initialize_model(self): 
        """
        Initializes the PyTorch MLP classifier, optimizer, loss function, and device.
        Attributes Set
         --------------
        self.model : MLP_Classifier : The initialized neural network model.
    
        self.optimizer : torch.optim.Optimizer
        The optimizer used to update model weights during training.
    
        self.criterion : torch.nn.Module
        The loss function used to compute the error (Binary Cross Entropy with logits).
    
        self.device : str
        The device on which training will take place.
        """
        
        self.model = MLP_Classifier(
            layer_dims=self.model_params["layer_dimensions"],
            dropout=self.model_params["dropout_probability"]
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.model_params["learning_rate"]
        )

    def predict(self, X):
        """
        Make label predictions using the trained MLP model.
    
        Args:
            X (np.ndarray or pd.DataFrame)
        
        Returns:
            np.ndarray: Predicted labels  
        """
        self.model.to(self.device)
        self.model.eval()
        
        # Convert input to tensor
        xs = X.astype('float64')  
        x = torch.tensor(xs.values).float()
        x = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x).to(self.device)
            outputs_proba = torch.sigmoid(outputs) 
            lab_pred = (outputs_proba >= 0.5).int()

        return pd.DataFrame(lab_pred, columns=['predicted_label']) 
    
    def predict_classification_proba(self, X):
        """
        Make Probabilitie predictions with the trained MLP model.
        """
        self.model.to(self.device)
        self.model.eval()
        
        # Convert input to tensor
        xs = X.astype('float64')  
        x = torch.tensor(xs.values).float()
        x = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x).to(self.device)
            outputs_proba = torch.sigmoid(outputs) 

        return  pd.DataFrame(outputs_proba, columns=['predicted_proba']) 
        
        
    
    def train(self, XS, YS):
        """
        Trains the MLP model using the training and validation sets.

        Parameters
        ----------
        XS : dict
        Dictionary containing data for each phase ("train", "validation", etc.)
        YS : dict
        Dictionary containing labels for each phase ("train", "validation", etc.)

        Returns
        -------
        self : object
        The trained model object.

        history : dict
        Dictionary containing training history (loss and AUC per epoch).
        """
        
        # create dataloaders from the data 
        dataloader = get_mlp_dataloader(XS,YS) 
        # calculate imbalance ratio of the classe 
        weight = get_MLP_criterio_weights(YS["train"])
        weight = torch.tensor([weight], device=self.device)
        print(f'le poid de bce :{weight}')
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        
        history = {}
        history['train_loss'] = []
        history['val_loss'] = []
        history['train_qual'] = []
        history['val_qual'] = []
        threshold = 0.5

        #itérer sur un nombre d'epoch :
        for epoch in range(1, training_mlp_parameters["num_epochs"]+1):

            # car chaque epoch traitera plusieur batch:
            loss_train_tmp, loss_val_tmp = [], []
            qual_val_tmp, qual_train_tmp = [], []

            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                for idx, (inputs, labels) in enumerate(dataloader[phase]):
                    inputs = inputs.to( training_mlp_parameters["device"])
                    labels = labels.to( training_mlp_parameters["device"])
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs).to(training_mlp_parameters["device"])
                        loss = self.criterion(outputs,labels)

                        with torch.no_grad():
                            outputs_prob = torch.sigmoid(outputs) 
                            preds = (outputs_prob >= threshold).int()
                            qual = roc_auc_score(labels.data, preds)
                           
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            loss_train_tmp.append(loss.item())
                            qual_train_tmp.append(qual)

                        if phase == 'validation':
                            loss_val_tmp.append(loss.item())
                            qual_val_tmp.append(qual)

            # calcul des performance sur tout les batch de l'epoch en cour
            lossEpoch = np.average(loss_train_tmp)
            val_lossEpoch = np.average(loss_val_tmp)
            qualTrain = np.average(qual_train_tmp)
            qualVal = np.average(qual_val_tmp)

            # Sauvegarde dans l'historique :
            history['train_loss'].append(lossEpoch)
            history['val_loss'].append(val_lossEpoch)
            history['train_qual'].append(qualTrain)
            history['val_qual'].append(qualVal)

            if epoch % 1 == 0:
                self.logger.info(" \t Epoch {}/{}, train loss: {:.3f}, val loss: {:.3f}, train qualite:{:.3f}, val qualite: {:.3f}  ".format(
                epoch,training_mlp_parameters["num_epochs"]+1,lossEpoch, val_lossEpoch,qualTrain, qualVal ))
     

        avg_loss  = sum(history['train_loss']) / len(history['train_loss'])
        avg_val_loss =  sum(history['val_loss']) / len(history['val_loss'])
        avg_qual = sum(history['train_qual']) / len(history['train_qual'])
        avg_val_qual = sum(history['val_qual']) / len(history['val_qual'])

        self.logger.info("\n\n train_Loss {}, val_loss: {:.3f},  train qualite: {:.3f} , val qualite: {:.3f}  ".format( avg_loss,avg_val_loss,avg_qual,avg_val_qual))

        return self, history 
    
     
    
    
  
  
  
  
  
  
  
  
  
  
  

    
   
