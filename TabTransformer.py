# TabTransfromer
import torch
from torch import nn
import numpy as np
from einops import rearrange
from inspect import isfunction
from torch.nn import functional as F
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from tqdm import tqdm
from math import pow


class BaseModel:
    def __init__(self, params, arguments):
        self.params = params
        self.arguments = arguments
        self.device = self.get_device()
        self.gpus = arguments.gpu_ids if arguments.use_gpu and torch.cuda.is_available() and arguments.data_parallel else None

        # Model definition has to be implemented by the concrete model
        self.model = None
    
    def get_device(self):
        device = 'cpu'

        if self.arguments.use_gpu and torch.cuda.is_available():
            device = 'cuda'
        
        return torch.device(device)
    
    def fit(self, X, y, X_val=None, y_val=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'])

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()
        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.arguments.objective == 'regression':
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.arguments.objective == 'classification':
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.arguments.batch_size, shuffle=True, num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.arguments.val_batch_size, shuffle=True)

        min_val_loss, min_val_loss_idx = float('inf'), 0

        loss_history, val_loss_history = [], []

        for epoch in tqdm(range(1, self.arguments.epochs + 1), desc='Training'):
            for batch_x, batch_y in enumerate(train_loader):

                out = self.model(batch_x.to(self.device))

                if self.arguments.objective in ['regression', 'binary']:
                    out.squeeze_()
                
                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            val_loss, val_cnt = .0, 0

            for batch_val_x, batch_val_y in enumerate(val_loader):
                
                out = self.model(batch_val_x.to(self.device))

                if self.arguments.objective in ['regression', 'binary']:
                    out.squeeze_()

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_cnt += 1
            
            val_loss /= val_cnt
            val_loss_history.append(val_loss.item())

            if val_loss < min_val_loss:
                min_val_loss, min_val_loss_idx = val_loss, epoch

                self.save_model(extension='best', directory='tmp')
            
            if min_val_loss_idx + self.arguments.early_stopping_rounds < epoch:
                print("Early stopping applies.")
                break
        
        self.load_model(extension="best", directory="tmp")
        return loss_history, val_loss_history
    
    def predict(self, X):
        if self.arguments.objective == 'regression':
            self.predictions = self._predict(X)
        else:
            prediction_probabilities = self.predict_proba(X)
            self.predictions = np.argmax(prediction_probabilities, axis=1)
        return self.predictions
    
    def _predict(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.arguments.val_batch_size, shuffle=False, num_workers=2)
        predictions = []

        with torch.no_grad():
            for batch_x in test_loader:
                preds = self.model(batch_x[0].to(self.device))

                if self.arguments.objective == 'binary':
                    torch.sigmoid_(preds)
                
                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)
    
    def predict_proba(self, X):
        probas = self._predict(X)

        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        return probas
    
    def get_model_size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def load_model(self, extension='', directory='models'):
        filename = self.get_output_path(filename='m', directory=directory, extension=extension, file_type='pt')
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)
    
    def save_model(self, extension='', directory='models'):
        filename = self.get_output_path(filename='m', directory=directory, extension=extension, file_type='pt')
        torch.save(self.model.state_dict(), filename)

    def get_output_path(self, filename, file_type, directory=None, extension=None):
        output_dir = "output/"
        dir_path = output_dir + self.arguments.model_name + "/" + self.arguments.dataset

        if directory:
            # For example: .../models
            dir_path = dir_path + "/" + directory

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = dir_path + "/" + filename
        if extension is not None:
            file_path += "_" + str(extension)
        file_path += "." + file_type

        return file_path
    
    def clone(self):
        return self.__class__(self.params, self.arguments)


def launch(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def default(val, d):
    return val if val is not None else d


class CustomResidual(nn.Module):
    def __init__(self, fn):
        super(CustomResidual, self).__init__()

        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim_in, fn):
        super(PreNorm, self).__init__()

        self.fn = fn
        self.norm_ = nn.GroupNorm(1, dim_in)
    
    def forward(self, x):
        return self.fn(self.norm_(x))


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = 1 / float(np.sqrt(dim_head))

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda z: rearrange(z, 'b n (heads dim) -> b heads n dim', heads=self.heads), (q, k, v))
        
        similarity = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention = self.dropout(similarity.softmax(dim=-1))

        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        return self.out(out)


class Lin_n_Gelu(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FF(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            Lin_n_Gelu(),
            nn.Dropout(),
            nn.Linear(dim * mult, dim)
        )
    
    def forward(self, x, **kwargs):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, dimensions, activation=None) -> None:
        super().__init__()
        dims_pairs = list(zip(dimensions[:-1], dimensions[1:]))
        layers = []

        for idx, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = idx == len(dims_pairs) - 1
            layers.append(nn.Linear(dim_in, dim_out))

            if is_last:
                self.dim_out = dim_out
                continue

            layers.append(default(activation, nn.ReLU()))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        o = self.mlp(x)
        if self.dim_out > 1:
            o = torch.softmax(o, dim=1)
        return o


class Transformer(nn.Module):
    def __init__(self, num_tokens, depth, dim, heads, dim_head, attn_dropout, ff_dropout) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_tokens, dim) # for categorical features
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CustomResidual(PreNorm(dim, Attention(dim, heads, dim_head, attn_dropout))),
                CustomResidual(PreNorm(dim, FF(dim=dim, dropout=ff_dropout)))
            ]))
    
    def forward(self, x):
        x = self.embeddings(x)

        for attn, feedforward in self.layers:
            x = feedforward(attn(x))
        
        return x


class TabModel(nn.Module):
    def __init__(
            self,
            categories,
            num_continuous,
            dimension,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_activation=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.0,
            ff_dropout=0.0
    ):
        super(TabModel, self).__init__()
        assert all([cat > 0 for cat in categories])

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total = self.num_unique_categories + num_special_tokens

        cat_displacement = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        cat_displacement = cat_displacement.cumsum(dim=-1)[:-1]
        self.register_buffer('cat_displacement', cat_displacement)

        if continuous_mean_std is not None:
            assert continuous_mean_std.shape == (num_continuous, 2)
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        self.transformer = Transformer(
            num_tokens=total,
            dim=dimension,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        input_size = dimension * self.num_categories + num_continuous
        scale = input_size // 8
        hidden_dimensions = map(lambda x: scale * x, mlp_hidden_mults)
        dims = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(dimensions=dims, activation=mlp_activation)
    
    def forward(self, cat, continuous):
        if cat is not None:
            assert cat.shape[-1] == self.num_categories

            cat += self.cat_displacement
            x = self.transformer(cat)
            flattened_categ = x.flatten(1)
        
        assert continuous.shape == self.num_continuous

        if self.continuous_mean_std is not None:
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            continuous = (continuous - mean) / std
        
        continuous = self.norm(continuous)
        x = continuous
        if cat is not None:
            x = torch.cat((flattened_categ, continuous), dim=-1)
        
        return self.mlp(x)


class arguments:
    use_gpu = True
    data_parallel = True
    gpu_ids = []
    objective = 'regression'
    batch_size = 32
    epochs = 10
    early_stopping_rounds = 5
    val_batch_size = 32
    model_name = ''
    dataset = ''
    num_features = 100
    num_classes = 1
    cat_idx = None
    cat_dims = ()


params = {
    'learning_rate': -1,
    'dim': 16,
    'depth': 4,
    'heads': 4,
    'dropout': 0.1,
    'weight_decay': -1
}


class TabTransformer(BaseModel):
    def __init__(self, params, arguments):
        super(TabTransformer, self).__init__(params, arguments)
        
        if arguments.cat_idx:
            self.num_idx = list(set(range(arguments.num_features)).difference(arguments.cat_idx))
            num_continuous = arguments.num_features - len(arguments.cat_idx)
            categories_unique = arguments.cat_dims
        else:
            self.num_idx = list(set(range(arguments.num_features)))
            num_continuous = arguments.num_features
            categories_unique = ()
        
        dim = self.params['dim'] if arguments.num_features < 50 else 8
        self.batch_size = arguments.batch_size if arguments.num_features < 50 else 64

        self.model = TabModel(
            categories=categories_unique,
            num_continuous=num_continuous,
            dim_out=arguments.num_classes,
            mlp_activation=nn.ReLU(),
            dimension=dim,
            depth=self.params['depth'],
            heads=self.params['heads'],
            attn_dropout=self.params['dropout'],
            ff_dropout=self.params['dropout'],
            mlp_hidden_mults=(4, 2)
        )
        self.to_device()
    
    def fit(self, X, y, X_val=None, y_val=None):
        learning_rate = pow(10, self.params['learning_rate'])
        weight_decay = pow(10, self.params['weight_decay'])
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        X = torch.tensor(np.array(X)).float()
        X_val = torch.tensor(np.array(X_val)).float()
        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.arguments.objective == 'regression':
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.arguments.objective == 'classification':
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.arguments.val_batch_size, shuffle=True)

        min_val_loss, idx = float('inf'), 0
        loss_history, val_loss_history = [], []

        for epoch in tqdm(range(1, self.arguments.epochs + 1)):
            for batch_x, batch_y in train_loader:
                
                x_cat = None
                if self.arguments.cat_idx:
                    x_cat = batch_x[:, self.arguments.cat_idx].int().to(self.device)
                
                x_cont = batch_x[:, self.num_idx].to(self.device)
                
                out = self.model(x_cat, x_cont)
                if self.arguments.objective in ('regression', 'binary'):
                    out.squeeze_()

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Early stopping
            val_loss, val_dim = .0, 0
            for batch_val_x, batch_val_y in val_loader:

                x_cat = None
                if self.arguments.cat_idx:
                    x_cat = batch_val_x[:, self.arguments.cat_idx].int().to(self.device)
                
                x_cont = batch_val_x[:, self.num_idx].to(self.device)
                
                out = self.model(x_cat, x_cont)
                if self.arguments.objective in ('regression', 'binary'):
                    out.squeeze_()
                
                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1
            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            if val_loss < min_val_loss:
                min_val_loss, idx = val_loss, epoch

                self.save_model(extension='best', directory='tmp')
            
            if idx + self.arguments.early_stopping_rounds < epoch:
                print('Early stopping applies')
                break
        self.load_model(extension='best', directory='tmp')
        return loss_history, val_loss_history
    
    @classmethod
    def define_trial_optuna_parameters(cls, trial, arguments):
        params = {
            'dim': trial.suggest_categorical('dim', [32, 64, 128, 256]),
            'depth': trial.suggest_categorical('depth', [1, 2, 3, 4, 6, 12]),
            'heads': trial.suggest_categorical('heads', [2, 4, 8]),
            'weight_decay': trial.suggest_int('weight_decay', -6, -1),
            'learning_rate': trial.suggest_int('learning_rate', -6, -3),
            'dropout': trial.suggest_categorical('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        }
        return params

    def attribute(self, X, y, strategy=''):
        X = torch.tensor(np.array(X)).float()

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.arguments.val_batch_size, shuffle=False, num_workers=2)

        attentions_list = []
        with torch.no_grad():
            for batch_x in test_loader:
                x_cat = None
                if self.arguments.cat_idx:
                    x_cat = batch_x[0][:, self.cat_idx].int().to(self.device)

                if x_cat is None:
                    raise ValueError
                else:
                    x_cat += self.model.cat_displacement
                    x = self.model.transformer.embeddings(x_cat)
                    x = self.model.transformer.layers[0][0].fn.norm(x)
                    active = self.model.transformer.layers[0][0].fn.fn
                    h = active.heads
                    q, k, v = active.qkv(x).chunk(3, dim=-1)
                    q, k, v = map(lambda z: rearrange(z, 'b n (h d) -> b h n d', h=h), (q, k, v))
                    sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * active.scale
                    attn = sim.softmax(dim=-1)
                    if strategy == 'diag':
                        attentions_list.append(attn.diagonal(0, 2, 3))
                    else:
                        attentions_list.append(attn.sum(dim=1))

        return torch.cat(attentions_list).sum(dim=1).numpy()

    def _predict(self, X):
        self.model.eval()
        X = torch.tensor(np.array(X)).float()

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.arguments.val_batch_size, shuffle=False, num_workers=2)
        predictions = []

        with torch.no_grad():
            for batch_x in test_loader:
                x_cat = None
                if self.arguments.cat_idx:
                    x_cat = batch_x[0][:, self.arguments.cat_idx].int().to(self.device)
                x_cont = batch_x[0][:, self.num_idx].to(self.device)

                preds = self.model(x_cat, x_cont)

                if self.arguments.objective == 'binary':
                    preds = torch.sigmoid(preds)
                
                predictions.append(preds.cpu())
        return np.concatenate(predictions)
