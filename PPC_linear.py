# this file defines a PPC linear layer with Rank Inference Gate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
import math
import numpy as np

class PPCLinear(nn.Module):
    def __init__(self, in_features, out_features, maximal_rank, bias = True, eps = 1e-8, gate_act_prob = 0.0, in_eval = False):
        super(PPCLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.maximal_rank = maximal_rank
        self.eps = eps
        self.gate_act_prob = torch.tensor(gate_act_prob).to(torch.float32)  
        self.scale_coeff = (self.in_features * self.out_features) ** (7/24)
        self.in_eval = in_eval

        self.weight_in = Parameter(torch.Tensor(in_features, maximal_rank))
        self.weight_out = Parameter(torch.Tensor(maximal_rank, out_features))
        self.weight_singular = Parameter(torch.Tensor(maximal_rank))
        self.register_parameter('weight_in', self.weight_in)
        self.register_parameter('weight_out', self.weight_out)
        self.register_parameter('weight_singular', self.weight_singular)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()  
    
    def forward(self, input):
        if self.weight_in is None or self.weight_out is None or self.weight_singular is None:
            # make sure broadcasting works  
            shape = list(input.shape[:-1]) + [self.out_features]
            output = torch.zeros(shape).to(input.device)
            if self.bias is not None:
                output = output + self.bias
            return output
        
        gate_act = torch.rand(1).item()
        if gate_act < self.gate_act_prob or self.in_eval:
            # Sort the proxy singular values in descending order and get the indices
            sorted_eigenvalues, indices = torch.sort(self.weight_singular ** 2, descending=True)
            # Get the gate values
            numerator = torch.cat([sorted_eigenvalues[1:self.maximal_rank], torch.tensor([0.0]).to(sorted_eigenvalues.device)])            
            pre_gate_value = self.scale_coeff * (1 -  numerator / (sorted_eigenvalues + self.eps))
            pre_gate_value = torch.clamp(pre_gate_value, min=0, max=1)
            # multiply each entry of pre_gate_value by all priors to get the gate value
            gate_value = torch.ones_like(pre_gate_value)
            for i in range(len(pre_gate_value)):
                gate_value[i] = torch.min(pre_gate_value[:i+1])
            # rearrange the gate values to match the original order
            _, indices = torch.sort(indices)
            # Apply the gate values to the weight matrices, with the indices
            new_weight_singular = self.weight_singular * gate_value[indices]
            output = input.matmul(self.weight_in).matmul(torch.diag(new_weight_singular)).matmul(self.weight_out)
        else: 
            output = input.matmul(self.weight_in).matmul(torch.diag(self.weight_singular)).matmul(self.weight_out)
        if self.bias is not None:
            output = output + self.bias
        return output
        
    def reset_parameters(self):
        # Initialize the weights and bias
        """
        init.xavier_uniform_(self.weight_in)
        init.xavier_uniform_(self.weight_out)
        """
        init.xavier_normal_(self.weight_in)
        init.xavier_normal_(self.weight_out)

        with torch.no_grad():
            self.weight_singular = nn.Parameter(((1 + np.sqrt(self.in_features / self.out_features)) / ((self.in_features * self.out_features) ** (1/4))) * torch.ones(self.maximal_rank).to(self.weight_in.device) )
            # orthogonalization
            weight_in_Q, _ = torch.linalg.qr(self.weight_in, mode='reduced')
            weight_out_Q, _ = torch.linalg.qr(self.weight_out.t(), mode='reduced')
            # get the initial weight matrix
            self.weight_in =  nn.Parameter(weight_in_Q)
            self.weight_out = nn.Parameter(weight_out_Q.t())
        if self.bias is not None:
            init.zeros_(self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, maximal_rank={}, bias={}'.format(
            self.in_features, self.out_features, self.maximal_rank, self.bias is not None
        )
    
    def update_gate_act_prob(self, new_prob):
        self.gate_act_prob = new_prob

    def get_rank_inference(self):
        if self.weight_in is None or self.weight_out is None or self.weight_singular is None:
            return 0, 0
        with torch.no_grad():
            # Sort the proxy singular values in descending order and get the indices
            sorted_eigenvalues, indices = torch.sort(self.weight_singular ** 2, descending=True)
            # Get the gate values
            numerator = torch.cat([sorted_eigenvalues[1:self.maximal_rank], torch.tensor([0.0]).to(sorted_eigenvalues.device)])            
            pre_gate_value = self.scale_coeff * (1 -  numerator / (sorted_eigenvalues + self.eps))
            pre_gate_value = torch.clamp(pre_gate_value, min=0, max=1)
            # multiply each entry of pre_gate_value by all priors to get the gate value
            gate_value = torch.ones_like(pre_gate_value)
            for i in range(len(pre_gate_value)):
                gate_value[i] = torch.min(pre_gate_value[:i+1])
            rank_inference = torch.sum(gate_value == 1).item()
        return rank_inference, self.maximal_rank
    
    def truncate_PPC(self, maximal_truncate = None):
        if maximal_truncate is None:
            maximal_truncate = self.maximal_rank
        with torch.no_grad():
            rank_inference = self.get_rank_inference()[0]
            # Sort the proxy singular values in descending order and get the indices
            _, indices = torch.sort(self.weight_singular ** 2, descending=True)
            rank_inference = max(rank_inference, self.maximal_rank - maximal_truncate)
            if rank_inference == 0:
                # cut the whole weight matrix
                self.weight_in = None
                self.weight_out = None
                self.weight_singular = None
                self.maximal_rank = 0
                # set bias to 0
                if self.bias is not None:
                    self.bias = nn.Parameter(torch.zeros(self.out_features).to(self.bias.device))
                return self.maximal_rank
            else:
                # cut the indexes with gate_value < 1
                indices = indices[:rank_inference]
                self.weight_in = nn.Parameter(self.weight_in[:, indices])
                self.weight_out = nn.Parameter(self.weight_out[indices, :])
                self.weight_singular = nn.Parameter(self.weight_singular[indices])
                self.maximal_rank = rank_inference
                return self.maximal_rank
            
    def request_PPC(self, maximal_request, gap_scale = 0.33):
        with torch.no_grad():
            new_rank = min(self.maximal_rank + maximal_request, self.in_features, self.out_features)
            requested_rank = new_rank - self.maximal_rank
            if requested_rank <= 0:
                return
            # generate requested_singular_values from geometric progression
            # with the first value as the minimum singular value * ratio
            ratio = 1 - ((1 + gap_scale) / self.scale_coeff)
            requested_singular_values = torch.zeros(requested_rank).to(self.weight_in.device)
            for i in range(requested_rank):
                requested_singular_values[i] = self.weight_singular.min() * (ratio ** (i + 1))
            # generate requested weight_in and weight_out, new weights should be orthogonal to the old weights
            requested_weight_in = torch.randn(self.in_features, requested_rank).to(self.weight_in.device)
            requested_weight_out = torch.randn(requested_rank, self.out_features).to(self.weight_in.device)
            # Project the new matrix onto the null space of the old matrix
            in_projection = self.weight_in @ self.weight_in.t() @ requested_weight_in
            out_projection = requested_weight_out @ self.weight_out.t() @ self.weight_out
            in_perp = requested_weight_in - in_projection
            out_perp = requested_weight_out - out_projection
            # Orthogonalize the new weights
            in_Q, _ = torch.linalg.qr(in_perp, mode='reduced')
            out_Q, _ = torch.linalg.qr(out_perp.t(), mode='reduced')
            # Concatenate the new weights with the old weights
            self.weight_in = nn.Parameter(torch.cat([self.weight_in, in_Q], dim=1))
            self.weight_out = nn.Parameter(torch.cat([self.weight_out, out_Q.t()], dim=0))
            self.weight_singular = nn.Parameter(torch.cat([self.weight_singular, requested_singular_values], dim=0))
            self.maximal_rank = new_rank
            return self.maximal_rank

    def freeze_PPC(self):
        if self.weight_in is None or self.weight_out is None or self.weight_singular is None:
            return
        # Freeze the PPC weights
        self.weight_in.requires_grad = False
        self.weight_out.requires_grad = False
        self.weight_singular.requires_grad = False

    def unfreeze_PPC(self):
        if self.weight_in is None or self.weight_out is None or self.weight_singular is None:
            return
        # Unfreeze the PPC weights
        self.weight_in.requires_grad = True
        self.weight_out.requires_grad = True
        self.weight_singular.requires_grad = True

    def get_orthogonality_loss(self):
        if self.weight_in is None or self.weight_out is None:
            return 0
        orthgonality_loss = torch.linalg.matrix_norm(self.weight_in.t() @ self.weight_in - torch.eye(self.maximal_rank).to(self.weight_in.device), ord = 'fro') ** 2 / (self.maximal_rank ** 2)
        orthgonality_loss += torch.linalg.matrix_norm(self.weight_out @ self.weight_out.t() - torch.eye(self.maximal_rank).to(self.weight_out.device), ord = 'fro') ** 2 / (self.maximal_rank ** 2)
        return orthgonality_loss
    
    def unorthogonality_decay(self, decay_rate = 0.1):
        loss = decay_rate * self.get_orthogonality_loss()
        loss.backward()

def replace_linear_with_PPC(model, maximal_rank_scale = 0.5, gate_act_prob=0.0):
    replaced_layers = 0
    layers_to_replace = []
    # Iterate through all modules in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers_to_replace.append((name, module))

    for name, module in layers_to_replace:
        if isinstance(module, nn.Linear):
            # Get the input and output features
            in_features = module.in_features
            out_features = module.out_features
            maximal_rank = max(int(maximal_rank_scale * min(in_features, out_features)), 1)
            # Create a new PPCLinear layer
            ppc_linear_layer = PPCLinear(in_features, out_features, maximal_rank, bias=module.bias is not None, gate_act_prob=gate_act_prob)
            # Replace the original linear layer with the new PPCLinear layer
            parent_module = model
            *parent_path, child_name = name.split('.')
            for part in parent_path:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, ppc_linear_layer)
            replaced_layers += 1
    print(f"Replaced {replaced_layers} linear layers with PPCLinear layers.")
    return model


def update_all_gate_act_prob(model, new_prob):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            module.update_gate_act_prob(new_prob)

def get_all_gate_act_prob(model):
    gate_act_prob_list = []
    for module in model.modules():
        if isinstance(module, PPCLinear):
            gate_act_prob = module.gate_act_prob
            gate_act_prob_list.append(gate_act_prob)
    return gate_act_prob_list

def freeze_all_PPC(model):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            module.freeze_PPC()

def unfreeze_all_PPC(model):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            module.unfreeze_PPC()

def get_all_rank_inference(model):
    rank_inference_list = []
    for module in model.modules():
        if isinstance(module, PPCLinear):
            rank_inference = module.get_rank_inference()
            rank_inference_list.append(rank_inference)
    return rank_inference_list

def get_all_orthogonality_loss(model):
    orthogonality_loss_list = []
    for module in model.modules():
        if isinstance(module, PPCLinear):
            orthogonality_loss = module.get_orthogonality_loss()
            orthogonality_loss_list.append(orthogonality_loss)
    return orthogonality_loss_list

def unorthogonality_decay_all(model, decay_rate = 0.1):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            module.unorthogonality_decay(decay_rate)

def truncate_all_PPC(model, maximal_truncate = None):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            maximal_rank = module.truncate_PPC(maximal_truncate)

def request_all_PPC(model, maximal_request, gap_scale = 0.33):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            maximal_rank = module.request_PPC(maximal_request, gap_scale)

def PPC_adjust_all_rank(model, maximal_truncate = 5, maximal_request = 5, rank_relax=(0, 5), gap_scale = 0.33):
    for module in model.modules():
        if isinstance(module, PPCLinear):
            rank_inference, maximal_rank = module.get_rank_inference()
            if (maximal_rank - rank_inference) <= rank_relax[0]:
                module.request_PPC(maximal_request, gap_scale)
            elif (maximal_rank - rank_inference) >= rank_relax[1]:
                module.truncate_PPC(min(maximal_truncate, maximal_rank - rank_inference - rank_relax[1]))

def PPC_init_from_pretrained(pretrained_linear_layer, rank_relax=(0, 5)):
    """
    Initialize the PPCLinear layer from a pretrained linear layer.
    """
    # Get the input and output features
    in_features = pretrained_linear_layer.in_features
    out_features = pretrained_linear_layer.out_features
    bias = pretrained_linear_layer.bias is not None
    # Get the maximal rank
    maximal_rank = max(1, int(min(in_features, out_features)))
    # Create a new PPCLinear layer
    ppc_linear_layer = PPCLinear(in_features, out_features, maximal_rank, bias=bias, gate_act_prob=0.0)

    # Do SVD on the pretrained linear layer's weight
    weight = pretrained_linear_layer.weight.data
    weight = weight.t()
    u, s, v = torch.linalg.svd(weight, full_matrices=False)

    # replace the weights and bias from the pretrained linear layer
    ppc_linear_layer.weight_in.data = u
    ppc_linear_layer.weight_out.data = v
    ppc_linear_layer.weight_singular.data = s

    # get rank inference
    rank_inference, maximal_rank = ppc_linear_layer.get_rank_inference()

    # adjust the rank
    PPC_adjust_all_rank(ppc_linear_layer, maximal_truncate=maximal_rank, maximal_request=rank_relax[1], rank_relax=rank_relax)

    return ppc_linear_layer


# test
if __name__ == "__main__":
    test_PPC_linear = True
    if test_PPC_linear:
        # Create a random input tensor
        input_tensor = torch.randn(5, 20)
        # Create a PPCLinear layer
        ppc_linear_layer = PPCLinear(20, 15, 10, bias=True, gate_act_prob=0)
        print("weight_in shape:", ppc_linear_layer.weight_in.shape)
        print("weight_out shape:", ppc_linear_layer.weight_out.shape)
        # check orthogonality
        print("orthogonality check:", ppc_linear_layer.weight_in.t() @ ppc_linear_layer.weight_in)
        # Forward pass
        output_tensor = ppc_linear_layer(input_tensor)
        print("Output tensor shape:", output_tensor.shape)
        # check the rank inference
        for m in ppc_linear_layer.modules():
            if isinstance(m, PPCLinear):
                print(m.get_rank_inference())
        # check the orthogonality loss
        print("orthogonality loss:", ppc_linear_layer.get_orthogonality_loss())
        # check unorthgonality_decay
        ppc_linear_layer.unorthogonality_decay()
        # check the truncation
        maximal_rank = ppc_linear_layer.truncate_PPC(7)
        print("maximal rank:", maximal_rank)
        output_tensor = ppc_linear_layer(input_tensor)
        print("Output tensor after truncation:", output_tensor)
        # check request_PPC
        ppc_linear_layer.request_PPC(5)
        print("weight_in shape after request:", ppc_linear_layer.weight_in.shape)
        print("weight_out shape after request:", ppc_linear_layer.weight_out.shape)
        # check adjust rank
        ppc_linear_layer = PPCLinear(20, 15, 10, bias=True, gate_act_prob=0)
        PPC_adjust_all_rank(ppc_linear_layer, maximal_truncate=5, maximal_request=5, rank_relax=(0, 5), gap_scale=0.33)
        print("weight_in shape after adjust:", ppc_linear_layer.weight_in.shape)
        print("weight_out shape after adjust:", ppc_linear_layer.weight_out.shape)

    # test init from pretrained
    test_init_from_pretrained = False
    if test_init_from_pretrained:
        # Create a random input tensor
        input_tensor = torch.randn(5, 20)
        # Create a pretrained linear layer
        pretrained_linear_layer = nn.Linear(20, 15, bias=True)
        # Initialize the PPCLinear layer from the pretrained linear layer
        ppc_linear_layer = PPC_init_from_pretrained(pretrained_linear_layer, rank_relax=(0, 5))
        print("weight_in shape after init:", ppc_linear_layer.weight_in.shape)
        print("weight_out shape after init:", ppc_linear_layer.weight_out.shape)
        # Forward pass
        output_tensor = ppc_linear_layer(input_tensor)
        print("Output tensor shape after init:", output_tensor.shape)

    




    


        

